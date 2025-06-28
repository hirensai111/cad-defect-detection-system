# CAD Image Classification Model
# This script creates a CNN model to classify CAD images into 4 categories

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class CADImageClassifier:
    def __init__(self, img_height=224, img_width=224):
        self.img_height = img_height
        self.img_width = img_width
        self.num_classes = 4
        self.class_names = ['Good', 'Interference', 'Dimension_Issues', 'Clearance_Problems']
        self.model = None
        self.label_encoder = LabelEncoder()
        
    def create_model(self):
        """Create a CNN model for CAD image classification"""
        model = keras.Sequential([
            # Input layer
            layers.Rescaling(1./255, input_shape=(self.img_height, self.img_width, 3)),
            
            # First convolutional block
            layers.Conv2D(32, 3, activation='relu'),
            layers.MaxPooling2D(),
            
            # Second convolutional block
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(),
            
            # Third convolutional block
            layers.Conv2D(128, 3, activation='relu'),
            layers.MaxPooling2D(),
            
            # Fourth convolutional block
            layers.Conv2D(256, 3, activation='relu'),
            layers.MaxPooling2D(),
            
            # Dropout for regularization
            layers.Dropout(0.2),
            
            # Flatten and dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def preprocess_image(self, image_path):
        """Preprocess a single image for prediction"""
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize image
            img = cv2.resize(img, (self.img_width, self.img_height))
            
            # Convert to float and normalize
            img = img.astype(np.float32) / 255.0
            
            # Add batch dimension
            img = np.expand_dims(img, axis=0)
            
            return img
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {e}")
            return None
    
    def load_dataset(self, data_dir):
        """Load images from directory structure"""
        images = []
        labels = []
        
        print("Loading dataset...")
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(data_dir, class_name.lower())
            
            if not os.path.exists(class_dir):
                print(f"Warning: Directory {class_dir} does not exist")
                continue
            
            print(f"Loading {class_name} images...")
            
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(class_dir, filename)
                    
                    # Preprocess image
                    img = self.preprocess_image(image_path)
                    if img is not None:
                        images.append(img[0])  # Remove batch dimension
                        labels.append(class_idx)
        
        if len(images) == 0:
            print("No images found! Please check your directory structure.")
            return None, None
        
        images = np.array(images)
        labels = np.array(labels)
        
        print(f"Loaded {len(images)} images total")
        print(f"Image shape: {images.shape}")
        
        return images, labels
    
    def train_model(self, data_dir, epochs=20, validation_split=0.2):
        """Train the model on CAD images"""
        # Load dataset
        images, labels = self.load_dataset(data_dir)
        
        if images is None:
            print("Failed to load dataset")
            return None
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            images, labels, test_size=validation_split, random_state=42, stratify=labels
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        # Create model if not exists
        if self.model is None:
            self.create_model()
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            validation_data=(X_val, y_val),
            batch_size=32,
            verbose=1
        )
        
        return history
    
    def predict_image(self, image_path):
        """Predict the class of a single image"""
        if self.model is None:
            print("Model not trained yet!")
            return None
        
        # Preprocess image
        img = self.preprocess_image(image_path)
        if img is None:
            return None
        
        # Make prediction
        predictions = self.model.predict(img)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        return {
            'class': self.class_names[predicted_class],
            'confidence': float(confidence),
            'all_predictions': {
                self.class_names[i]: float(predictions[0][i]) 
                for i in range(len(self.class_names))
            }
        }
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.model is not None:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")

# Example usage and testing
if __name__ == "__main__":
    # Initialize classifier
    classifier = CADImageClassifier()
    
    # Create model architecture
    model = classifier.create_model()
    
    print("Model architecture:")
    model.summary()
    
    print("\nCAD Image Classifier ready!")
    print("Next steps:")
    print("1. Create data folders: good, interference, dimension_issues, clearance_problems")
    print("2. Add your CAD images to respective folders")
    print("3. Run: classifier.train_model('path_to_data_folder')")
    print("4. Use: classifier.predict_image('path_to_test_image')")