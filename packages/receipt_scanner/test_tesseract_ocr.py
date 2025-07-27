#!/usr/bin/env python3
"""
Test script for the Tesseract OCR implementation
"""

import cv2
import numpy as np
import os
from tesseract_ocr import TesseractOCRProcessor, create_tesseract_processor


def test_tesseract_ocr_with_ideal_test_jpeg():
    """Test specifically with ideal_test.jpeg image"""
    print("\nTesting Tesseract OCR with ideal_test.jpeg Image")
    print("=" * 50)
    
    # Use ideal_test.jpeg image
    test_image_path = "packages/receipt_scanner/test_images/ideal_test.jpeg"
    
    if not os.path.exists(test_image_path):
        print(f"Test image not found: {test_image_path}")
        return None, None, 0.0
    
    print(f"Using test image: {test_image_path}")
    
    try:
        # Load the test image
        test_image = cv2.imread(test_image_path)
        if test_image is None:
            print(f"Could not load image: {test_image_path}")
            return None, None, 0.0
        
        # Initialize Tesseract OCR processor
        print("\nInitializing Tesseract OCR processor...")
        ocr_processor = create_tesseract_processor()
        
        # Test preprocessing
        print("Testing preprocessing...")
        preprocessed = ocr_processor.preprocess_image(test_image_path, target_dpi=300, add_border=True)
        
        if preprocessed is not None:
            print("✓ Preprocessing successful")
            
            # Test simple OCR
            print("\nTesting simple OCR...")
            extracted_text = ocr_processor.perform_ocr(test_image_path)
            print(f"Extracted text:\n{extracted_text}")
            
            # Test boundary box display
            print("\nTesting boundary box display...")
            ocr_processor.perform_ocr(test_image_path, boundary_box_display=True)
            
            # Test process_image method
            print("\nTesting process_image method...")
            regions = ocr_processor.process_image(test_image)
            
            # Display results
            print(f"\nDetected {len(regions)} text regions:")
            for i, region in enumerate(regions[:5]):  # Show first 5
                print(f"  {i+1}. '{region.text}' (confidence: {region.confidence:.2f}, type: {region.region_type})")
            
            # Extract formatted text
            extracted_text = ocr_processor.extract_text_from_regions(regions)
            print(f"\nExtracted text:\n{extracted_text}")
            
            # Get confidence score
            confidence = ocr_processor.get_confidence_score(regions)
            print(f"\nOverall confidence: {confidence:.2f}")
            
            return regions, extracted_text, confidence
        else:
            print("✗ Preprocessing failed")
            return None, None, 0.0
        
    except Exception as e:
        print(f"Error testing with ideal_test.jpeg: {e}")
        return None, None, 0.0


def test_tesseract_ocr_with_real_image():
    """Test specifically with real_test.jpg image"""
    print("\nTesting Tesseract OCR with real_test.jpg Image")
    print("=" * 50)
    
    # Use real_test.jpeg image
    test_image_path = "packages/receipt_scanner/test_images/real_test.jpg"
    
    if not os.path.exists(test_image_path):
        print(f"Test image not found: {test_image_path}")
        return None, None, 0.0
    
    print(f"Using test image: {test_image_path}")
    
    try:
        # Load the test image
        test_image = cv2.imread(test_image_path)
        if test_image is None:
            print(f"Could not load image: {test_image_path}")
            return None, None, 0.0
        
        # Initialize Tesseract OCR processor
        print("\nInitializing Tesseract OCR processor...")
        ocr_processor = create_tesseract_processor()
        
        # Test preprocessing
        print("Testing preprocessing...")
        preprocessed = ocr_processor.preprocess_image(test_image_path, target_dpi=300, add_border=True)
        
        if preprocessed is not None:
            print("✓ Preprocessing successful")
            
            # Test simple OCR
            print("\nTesting simple OCR...")
            extracted_text = ocr_processor.perform_ocr(test_image_path)
            print(f"Extracted text:\n{extracted_text}")
            
            # Test boundary box display
            print("\nTesting boundary box display...")
            ocr_processor.perform_ocr(test_image_path, boundary_box_display=True)
            
            # Test process_image method
            print("\nTesting process_image method...")
            regions = ocr_processor.process_image(test_image)
            
            # Display results
            print(f"\nDetected {len(regions)} text regions:")
            for i, region in enumerate(regions[:5]):  # Show first 5
                print(f"  {i+1}. '{region.text}' (confidence: {region.confidence:.2f}, type: {region.region_type})")
            
            # Extract formatted text
            extracted_text = ocr_processor.extract_text_from_regions(regions)
            print(f"\nExtracted text:\n{extracted_text}")
            
            # Get confidence score
            confidence = ocr_processor.get_confidence_score(regions)
            print(f"\nOverall confidence: {confidence:.2f}")
            
            return regions, extracted_text, confidence
        else:
            print("✗ Preprocessing failed")
            return None, None, 0.0
        
    except Exception as e:
        print(f"Error testing with ideal_test.jpeg: {e}")
        return None, None, 0.0

def test_boundaries_to_items(regions: List[TextRegion], text: str, confidence: float):
    ocr_processor = create_tesseract_processor()
    ocr_processor.create_boundaries(regions, text, confidence)

    




def list_test_images():
    """List available test images"""
    test_images_dir = "packages/receipt_scanner/test_images"
    if os.path.exists(test_images_dir):
        print(f"\nAvailable test images in {test_images_dir}:")
        for file in os.listdir(test_images_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                print(f"  - {file}")
    else:
        print(f"\nTest images directory not found: {test_images_dir}")


def main():
    """Main test function"""
    print("Tesseract OCR Receipt Scanner Test")
    print("=" * 50)
    
    # List available test images
    list_test_images()
    
    # Test 1: Tesseract OCR with existing image
    regions, text, confidence = test_tesseract_ocr_with_real_image()
    
    # Test 2: Tesseract OCR with ideal_test.jpeg specifically
    test_regions, test_text, test_confidence = test_tesseract_ocr_with_ideal_test_jpeg()
    
    
    print("\n=== Tesseract OCR Test Complete ===")
    print("Check the generated tesseract_ocr_results/ directory for visualization files.")


if __name__ == "__main__":
    main() 