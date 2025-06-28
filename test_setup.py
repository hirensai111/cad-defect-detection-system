# Test Setup Script
# This script tests if everything is installed correctly

import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def test_installations():
    """Test if all required packages are working"""
    print("Testing package installations...")
    
    # Test TensorFlow
    try:
        print(f"✓ TensorFlow version: {tf.__version__}")
        # Test if GPU is available (optional)
        if tf.config.list_physical_devices('GPU'):
            print("✓ GPU support available")
        else:
            print("✓ CPU-only TensorFlow (this is fine for development)")
    except Exception as e:
        print(f"✗ TensorFlow error: {e}")
    
    # Test OpenCV
    try:
        print(f"✓ OpenCV version: {cv2.__version__}")
    except Exception as e:
        print(f"✗ OpenCV error: {e}")
    
    # Test other packages
    try:
        print(f"✓ NumPy version: {np.__version__}")
        print(f"✓ PIL/Pillow available")
        print(f"✓ Matplotlib available")
    except Exception as e:
        print(f"✗ Package error: {e}")
    
    print("\nPackage test complete!")

def create_folder_structure():
    """Create the basic folder structure for the project"""
    folders = [
        "data",
        "data/good",
        "data/interference", 
        "data/dimension_issues",
        "data/clearance_problems",
        "models",
        "test_images"
    ]
    
    print("Creating folder structure...")
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"✓ Created folder: {folder}")
        else:
            print(f"✓ Folder exists: {folder}")
    
    print("\nFolder structure ready!")

def create_sample_test():
    """Create a simple test to verify the model structure"""
    print("Testing model creation...")
    
    try:
        # Import our CAD classifier
        from main import CADImageClassifier
        
        # Create classifier instance
        classifier = CADImageClassifier()
        
        # Create model
        model = classifier.create_model()
        
        print("✓ Model created successfully")
        print(f"✓ Model expects input shape: {model.input_shape}")
        print(f"✓ Model outputs {model.output_shape[1]} classes")
        
        # Test with dummy data
        dummy_input = np.random.random((1, 224, 224, 3))
        dummy_prediction = model.predict(dummy_input, verbose=0)
        
        print(f"✓ Model prediction works, output shape: {dummy_prediction.shape}")
        print("✓ All tests passed!")
        
    except Exception as e:
        print(f"✗ Model test error: {e}")

if __name__ == "__main__":
    print("=== CAD Image Classifier Setup Test ===\n")
    
    # Test installations
    test_installations()
    print("\n" + "="*50 + "\n")
    
    # Create folders
    create_folder_structure()
    print("\n" + "="*50 + "\n")
    
    # Test model (only if main.py exists)
    if os.path.exists("main.py"):
        create_sample_test()
    else:
        print("Create main.py first to test the model!")