import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from sklearn.feature_extraction import image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json

# PDF processing libraries
import fitz  # PyMuPDF
from PIL import Image
import io

class CADDefectDetector:
    def __init__(self, template_folder="data/good", output_folder="results"):
        """
        CAD Defect Detection System
        - Matches problematic images to good templates
        - Identifies missing elements and their locations
        - NOW SUPPORTS PDF FILES!
        """
        self.template_folder = template_folder
        self.output_folder = output_folder
        self.templates = []
        self.template_features = []
        
        # Create output directory
        os.makedirs(output_folder, exist_ok=True)
        
        # Load and process templates
        self.load_templates()
        
    def load_templates(self):
        """Load all good template images and PDFs"""
        print("Loading template images and PDFs...")
        
        # Support both images and PDFs
        template_files = [f for f in os.listdir(self.template_folder) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf'))]
        
        for template_file in template_files:
            template_path = os.path.join(self.template_folder, template_file)
            
            # Check if it's a PDF or image
            if template_file.lower().endswith('.pdf'):
                # Process PDF
                images = self.pdf_to_images(template_path)
                for i, img_array in enumerate(images):
                    template_data = {
                        'filename': f"{template_file}_page_{i+1}",
                        'original_file': template_file,
                        'page_number': i+1,
                        'image': img_array,
                        'gray': cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY),
                        'features': self.extract_features(cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY))
                    }
                    self.templates.append(template_data)
            else:
                # Process regular image
                template = cv2.imread(template_path)
                if template is not None:
                    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                    
                    template_data = {
                        'filename': template_file,
                        'original_file': template_file,
                        'page_number': None,
                        'image': template,
                        'gray': template_gray,
                        'features': self.extract_features(template_gray)
                    }
                    self.templates.append(template_data)
            
        print(f"Loaded {len(self.templates)} template images (including PDF pages)")
    
    def pdf_to_images(self, pdf_path, dpi=200):
        """Convert PDF pages to images"""
        print(f"Converting PDF: {pdf_path}")
        images = []
        
        try:
            # Open PDF
            pdf_document = fitz.open(pdf_path)
            
            for page_num in range(len(pdf_document)):
                # Get page
                page = pdf_document.load_page(page_num)
                
                # Convert to image
                mat = fitz.Matrix(dpi/72, dpi/72)  # 200 DPI conversion
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image
                img_data = pix.tobytes("ppm")
                pil_image = Image.open(io.BytesIO(img_data))
                
                # Convert to OpenCV format (BGR)
                cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                
                images.append(cv_image)
                print(f"  Processed page {page_num + 1}")
            
            pdf_document.close()
            
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {str(e)}")
            return []
        
        return images
    
    def extract_features(self, image_gray):
        """Extract features for template matching"""
        # Use ORB detector for robust feature matching
        orb = cv2.ORB_create(nfeatures=1000)
        keypoints, descriptors = orb.detectAndCompute(image_gray, None)
        
        return {
            'keypoints': keypoints,
            'descriptors': descriptors,
            'shape': image_gray.shape
        }
    
    def find_best_template(self, problematic_image):
        """Find the best matching template for the problematic image"""
        prob_gray = cv2.cvtColor(problematic_image, cv2.COLOR_BGR2GRAY)
        prob_features = self.extract_features(prob_gray)
        
        best_match = None
        best_score = 0
        
        # Create matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        for template in self.templates:
            if (template['features']['descriptors'] is not None and 
                prob_features['descriptors'] is not None):
                
                # Match features
                matches = bf.match(template['features']['descriptors'], 
                                 prob_features['descriptors'])
                
                # Sort by distance
                matches = sorted(matches, key=lambda x: x.distance)
                
                # Calculate match score
                if len(matches) > 10:  # Need minimum matches
                    good_matches = [m for m in matches if m.distance < 50]
                    score = len(good_matches) / len(matches)
                    
                    if score > best_score:
                        best_score = score
                        best_match = template
        
        return best_match, best_score
    
    def detect_differences(self, problematic_image, template_data):
        """Detect differences between problematic image and template"""
        # Resize images to same size
        template = template_data['image']
        prob_resized = cv2.resize(problematic_image, 
                                (template.shape[1], template.shape[0]))
        
        # Convert to grayscale
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        prob_gray = cv2.cvtColor(prob_resized, cv2.COLOR_BGR2GRAY)
        
        # Calculate structural similarity
        ssim_score, diff_image = ssim(template_gray, prob_gray, full=True)
        
        # Convert difference image to uint8
        diff_image = (diff_image * 255).astype(np.uint8)
        
        # Find contours of differences
        _, thresh = cv2.threshold(255 - diff_image, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter significant differences
        significant_diffs = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                significant_diffs.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'center': (x + w//2, y + h//2)
                })
        
        return {
            'ssim_score': ssim_score,
            'diff_image': diff_image,
            'thresh_image': thresh,
            'differences': significant_diffs,
            'resized_problematic': prob_resized
        }
    
    def analyze_missing_elements(self, differences, template_data):
        """Analyze what specific elements are missing"""
        missing_elements = []
        
        for diff in differences['differences']:
            x, y, w, h = diff['bbox']
            center_x, center_y = diff['center']
            
            # Determine what type of element is missing based on location and size
            element_type = self.classify_missing_element(diff, template_data)
            
            missing_elements.append({
                'type': element_type,
                'location': f"({center_x}, {center_y})",
                'bbox': (x, y, w, h),
                'area': diff['area'],
                'description': f"Missing {element_type} at position ({center_x}, {center_y})"
            })
        
        return missing_elements
    
    def classify_missing_element(self, diff, template_data):
        """Classify what type of element is missing based on location and characteristics"""
        x, y, w, h = diff['bbox']
        center_x, center_y = diff['center']
        area = diff['area']
        
        # Get image dimensions
        img_height, img_width = template_data['gray'].shape
        
        # Classify based on location and size
        if (img_width * 0.4 < center_x < img_width * 0.6 and 
            img_height * 0.4 < center_y < img_height * 0.6):
            if area < 500:
                return "Letter/Character"
            else:
                return "Central Logo Element"
        
        elif area < 200:
            return "Small Feature/Hole"
        elif w > h * 2 or h > w * 2:
            return "Linear Feature/Line"
        elif area > 1000:
            return "Large Structural Element"
        else:
            return "Geometric Feature"
    
    def create_visual_report(self, problematic_image, template_data, differences, missing_elements, output_name):
        """Create visual report showing missing elements"""
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
        fig.suptitle('CAD Defect Detection Report (PDF Support)', fontsize=18, fontweight='bold', y=0.98)
        
        # Original template
        axes[0, 0].imshow(cv2.cvtColor(template_data['image'], cv2.COLOR_BGR2RGB))
        if template_data['page_number']:
            axes[0, 0].set_title(f'Good Template\n{template_data["original_file"]} (Page {template_data["page_number"]})', 
                               fontsize=12, fontweight='bold')
        else:
            axes[0, 0].set_title(f'Good Template\n{template_data["filename"]}', 
                               fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Problematic image
        axes[0, 1].imshow(cv2.cvtColor(differences['resized_problematic'], cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('Problematic Image', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Difference map
        axes[0, 2].imshow(differences['diff_image'], cmap='hot')
        axes[0, 2].set_title('Difference Map', fontsize=14, fontweight='bold')
        axes[0, 2].axis('off')
        
        # Threshold differences
        axes[1, 0].imshow(differences['thresh_image'], cmap='gray')
        axes[1, 0].set_title('Detected Differences', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Highlighted missing elements
        highlighted = differences['resized_problematic'].copy()
        for i, element in enumerate(missing_elements):
            x, y, w, h = element['bbox']
            # Use different colors for different elements
            colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0), (255, 0, 255)]
            color = colors[i % len(colors)]
            
            cv2.rectangle(highlighted, (x, y), (x+w, y+h), color, 3)
            cv2.putText(highlighted, f"{i+1}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        axes[1, 1].imshow(cv2.cvtColor(highlighted, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title('Missing Elements Highlighted', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        
        # Text report layout
        axes[1, 2].clear()
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')
        
        # Title
        axes[1, 2].text(0.1, 0.92, 'DETECTION RESULTS', 
                       fontsize=14, fontweight='bold', 
                       transform=axes[1, 2].transAxes)
        
        # Summary info
        template_name = template_data['filename'] if not template_data['page_number'] else f"{template_data['original_file']} (Page {template_data['page_number']})"
        axes[1, 2].text(0.1, 0.82, f'Template: {template_name}', 
                       fontsize=9, fontweight='bold', 
                       transform=axes[1, 2].transAxes)
        
        axes[1, 2].text(0.1, 0.76, f'Similarity: {differences["ssim_score"]:.1%}', 
                       fontsize=10, 
                       transform=axes[1, 2].transAxes)
        
        axes[1, 2].text(0.1, 0.70, f'Missing Elements: {len(missing_elements)}', 
                       fontsize=10, fontweight='bold', color='red',
                       transform=axes[1, 2].transAxes)
        
        # Divider line
        axes[1, 2].plot([0.1, 0.9], [0.65, 0.65], 'k-', linewidth=1, 
                       transform=axes[1, 2].transAxes)
        
        # Missing elements list
        axes[1, 2].text(0.1, 0.60, 'MISSING ELEMENTS:', 
                       fontsize=11, fontweight='bold', 
                       transform=axes[1, 2].transAxes)
        
        # List each missing element with generous spacing
        y_start = 0.52
        line_height = 0.08
        
        max_items = 5
        items_to_show = min(len(missing_elements), max_items)
        
        for i in range(items_to_show):
            element = missing_elements[i]
            y_pos = y_start - (i * line_height)
            
            # Number with colored background
            axes[1, 2].text(0.1, y_pos, f'{i+1}', 
                           fontsize=11, fontweight='bold', color='white',
                           bbox=dict(boxstyle="circle,pad=0.2", facecolor='darkred'),
                           transform=axes[1, 2].transAxes)
            
            # Element type
            axes[1, 2].text(0.2, y_pos, f'{element["type"]}', 
                           fontsize=10, fontweight='bold', 
                           transform=axes[1, 2].transAxes)
            
            # Location
            axes[1, 2].text(0.2, y_pos - 0.03, f'{element["location"]}', 
                           fontsize=8, color='gray',
                           transform=axes[1, 2].transAxes)
        
        # Show remaining count if there are more
        if len(missing_elements) > max_items:
            remaining = len(missing_elements) - max_items
            y_pos = y_start - (max_items * line_height)
            axes[1, 2].text(0.1, y_pos, f'+ {remaining} more elements...', 
                           fontsize=9, style='italic', color='gray',
                           transform=axes[1, 2].transAxes)
        
        # Add clean border
        axes[1, 2].add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, 
                                          fill=False, edgecolor='black', linewidth=1.5,
                                          transform=axes[1, 2].transAxes))
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.94, hspace=0.15, wspace=0.1)
        
        # Save report
        report_path = os.path.join(self.output_folder, f"{output_name}_defect_report.png")
        plt.savefig(report_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return report_path
    
    def process_file(self, file_path, output_name=None):
        """Main function to process a file (image or PDF)"""
        
        if output_name is None:
            output_name = Path(file_path).stem
        
        print(f"\n=== Processing: {file_path} ===")
        
        # Check file type
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.pdf':
            # Process PDF - convert to images first
            print("PDF detected - converting to images...")
            pdf_images = self.pdf_to_images(file_path)
            
            if not pdf_images:
                print("Error: Could not process PDF")
                return None
            
            # Process each page
            all_results = []
            for i, page_image in enumerate(pdf_images):
                print(f"\nProcessing PDF page {i+1}/{len(pdf_images)}...")
                result = self._process_single_image(page_image, f"{output_name}_page_{i+1}")
                if result:
                    result['source_file'] = file_path
                    result['page_number'] = i+1
                    all_results.append(result)
            
            return all_results
            
        elif file_extension in ['.png', '.jpg', '.jpeg']:
            # Process regular image
            problematic_image = cv2.imread(file_path)
            if problematic_image is None:
                print(f"Error: Could not load image {file_path}")
                return None
            
            result = self._process_single_image(problematic_image, output_name)
            if result:
                result['source_file'] = file_path
                result['page_number'] = None
            return result
        
        else:
            print(f"Error: Unsupported file format {file_extension}")
            return None
    
    def _process_single_image(self, problematic_image, output_name):
        """Process a single image (internal method)"""
        
        # Step 1: Find best matching template
        print("1. Finding best matching template...")
        best_template, match_score = self.find_best_template(problematic_image)
        
        if best_template is None:
            print("Error: No matching template found")
            return None
        
        print(f"   Best match: {best_template['filename']} (score: {match_score:.3f})")
        
        # Step 2: Detect differences
        print("2. Detecting differences...")
        differences = self.detect_differences(problematic_image, best_template)
        print(f"   Similarity score: {differences['ssim_score']:.3f}")
        print(f"   Differences found: {len(differences['differences'])}")
        
        # Step 3: Analyze missing elements
        print("3. Analyzing missing elements...")
        missing_elements = self.analyze_missing_elements(differences, best_template)
        
        # Step 4: Create visual report
        print("4. Creating visual report...")
        report_path = self.create_visual_report(
            problematic_image, best_template, differences, missing_elements, output_name
        )
        
        # Step 5: Generate text report
        text_report = {
            'template_used': best_template['filename'],
            'match_score': float(match_score),
            'similarity_score': float(differences['ssim_score']),
            'missing_elements': missing_elements,
            'total_missing': len(missing_elements)
        }
        
        # Save TXT report with UTF-8 encoding
        txt_path = os.path.join(self.output_folder, f"{output_name}_report.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("       CAD DEFECT DETECTION REPORT (PDF SUPPORT)\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"File Analyzed: {output_name}\n")
            template_name = best_template['filename'] if not best_template['page_number'] else f"{best_template['original_file']} (Page {best_template['page_number']})"
            f.write(f"Template Used: {template_name}\n")
            f.write(f"Match Score: {match_score:.3f}\n")
            f.write(f"Similarity Score: {differences['ssim_score']:.1%}\n")
            f.write(f"Total Missing Elements: {len(missing_elements)}\n\n")
            
            f.write("-"*60 + "\n")
            f.write("                    MISSING ELEMENTS\n")
            f.write("-"*60 + "\n\n")
            
            if missing_elements:
                for i, element in enumerate(missing_elements, 1):
                    f.write(f"{i:2d}. ELEMENT TYPE: {element['type']}\n")
                    f.write(f"    Location: {element['location']}\n")
                    f.write(f"    Area: {element['area']:.1f} pixels\n")
                    f.write(f"    Bounding Box: {element['bbox']}\n")
                    f.write("\n")
            else:
                f.write("    No missing elements detected.\n\n")
            
            f.write("-"*60 + "\n")
            f.write("                      SUMMARY\n")
            f.write("-"*60 + "\n\n")
            
            if len(missing_elements) == 0:
                f.write("PASS: No defects detected. Image matches template.\n")
            else:
                f.write("FAIL: Defects detected. Review missing elements above.\n")
                f.write(f"  Primary concerns: {len([e for e in missing_elements if 'Central' in e['type'] or 'Letter' in e['type']])} critical elements missing\n")
            
            f.write(f"\nReport generated: {output_name}_defect_report.png\n")
            f.write("="*60 + "\n")
        
        print(f"\nREPORT SUMMARY:")
        print(f"Template matched: {best_template['filename']}")
        print(f"Elements missing: {len(missing_elements)}")
        for element in missing_elements:
            print(f"  - {element['description']}")
        print(f"Visual report saved: {report_path}")
        print(f"Text report saved: {txt_path}")
        
        return text_report
        
    # Backward compatibility
    def process_problematic_image(self, image_path, output_name=None):
        """Backward compatibility method"""
        return self.process_file(image_path, output_name)

# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = CADDefectDetector()
    
    # Process a file (image or PDF)
    # detector.process_file("path/to/file.pdf")  # For PDF
    # detector.process_file("path/to/image.jpg")  # For image
    
    print("CAD Defect Detection System Ready! (Now with PDF support)")
    print("Usage:")
    print("  detector.process_file('path/to/file.pdf')   # For PDF files")
    print("  detector.process_file('path/to/image.jpg')  # For image files")