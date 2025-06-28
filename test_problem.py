from defect_detector import CADDefectDetector

# Initialize detector
detector = CADDefectDetector()

# Process the problematic image (now supports both images and PDFs)
print("Testing defect detection on problematic image/PDF")
result = detector.process_file("data/clearance_problems/problem_1.png")

if result:
    # Handle both single image results and multi-page PDF results
    if isinstance(result, list):
        # PDF with multiple pages
        print(f"\n🎯 SUCCESS! PDF defect detection completed.")
        print(f"Total pages processed: {len(result)}")
        
        total_missing = 0
        for i, page_result in enumerate(result, 1):
            page_missing = page_result['total_missing']
            total_missing += page_missing
            print(f"  Page {i}: {page_missing} missing elements")
        
        print(f"Total missing elements across all pages: {total_missing}")
        
        # Show detailed results for pages with defects
        defective_pages = [r for r in result if r['total_missing'] > 0]
        if defective_pages:
            print(f"\n🔍 Pages with defects: {len(defective_pages)}")
            for page_result in defective_pages:
                page_num = page_result.get('page_number', 'Unknown')
                print(f"\n  Page {page_num} Issues:")
                for element in page_result['missing_elements']:
                    print(f"    - {element['description']}")
    else:
        # Single image result
        print(f"\n🎯 SUCCESS! Defect detection completed.")
        print(f"Missing elements detected: {result['total_missing']}")
        
        if result['total_missing'] > 0:
            print(f"\n🔍 Detected Issues:")
            for element in result['missing_elements']:
                print(f"  - {element['description']}")
        else:
            print(f"\n✅ No defects found - image matches template perfectly!")
            
else:
    print("\n❌ Detection failed.")

print("\n" + "="*50)
print("🚀 SYSTEM CAPABILITIES:")
print("✅ Images: .png, .jpg, .jpeg")
print("✅ PDFs: .pdf (multi-page support)")
print("✅ Templates: Both images and PDFs supported")
print("✅ Batch processing: Multiple pages automatically")
print("="*50)