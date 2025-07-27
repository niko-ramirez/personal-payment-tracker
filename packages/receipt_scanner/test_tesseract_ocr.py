#!/usr/bin/env python3
"""
Test script for the Tesseract OCR implementation
"""

import cv2
import numpy as np
import os
from tesseract_ocr import TesseractOCRProcessor, create_tesseract_processor
from typing import List
from termcolor import colored




def test_tesseract_ocr(test_image_path: str):
    """Test specifically with test image"""
    print(colored(f"\nTesting Tesseract OCR with {test_image_path} Image", 'green'))
    print(colored("=" * 50, 'green'))
    
    if not os.path.exists(test_image_path):
        print(colored(f"Test image not found: {test_image_path}", 'red'))
        return None, None, 0.0
    
    print(colored(f"Using test image: {test_image_path}", 'green'))
    
    try:
        # Load the test image
        test_image = cv2.imread(test_image_path)
        if test_image is None:
            print(colored(f"Could not load image: {test_image_path}", 'red'))
            return None, None, 0.0
        
        # Initialize Tesseract OCR processor
        print(colored("\nInitializing Tesseract OCR processor...", 'green'))
        ocr_processor = create_tesseract_processor()
        
        # Test preprocessing
        print(colored("Testing preprocessing...", 'green'))
        preprocessed = ocr_processor.preprocess_image(test_image_path, target_dpi=300, add_border=True)
        
        # Test simple OCR
        print(colored("\nTesting simple OCR...", 'green'))
        extracted_text = ocr_processor.perform_text_ocr(test_image_path)

        output_dir = "tesseract_ocr_results"
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(test_image_path))[0]
        output_image_name = f"{base_name}_extracted_text.txt"
        with open(os.path.join(output_dir, output_image_name), "w") as f:
            f.write(extracted_text)
        
        # Test boundary box display
        print(colored("\nTesting boundary box display...", 'green'))
        regions = ocr_processor.perform_region_ocr(test_image_path, boundary_box_display=True)
        
        # Display results
        print(colored(f"\nDetected {len(regions)} text regions:", 'green'))
        for i, region in enumerate(regions[:5]):  # Show first 5
            print(colored(f"  {i+1}. '{region.text}' (confidence: {region.confidence:.2f}, type: {region.region_type})", 'green'))
        
        # Extract formatted text
        # extracted_text = ocr_processor.extract_text_from_regions(regions)
        # print(f"\nExtracted text:\n{extracted_text}")
        
        # Get confidence score
        confidence = ocr_processor.get_confidence_score(regions)
        print(colored(f"\nOverall confidence: {confidence:.2f}", 'green'))
        
        return regions, extracted_text, confidence
      
        
    except Exception as e:
        print(colored(f"Error testing with {test_image_path}: {e}", 'red'))
        return None, None, 0.0


def main():
    """Main test function"""
    print(colored("Tesseract OCR Receipt Scanner Test", 'green'))
    print(colored("=" * 50, 'green'))
    
    # Test 1: Tesseract OCR with existing image
    regions, text, confidence = test_tesseract_ocr("packages/receipt_scanner/test_images/real_test.jpg")
    
    # Test 2: Tesseract OCR with ideal_test.jpeg specifically
    test_regions, test_text, test_confidence = test_tesseract_ocr("packages/receipt_scanner/test_images/ideal_test.jpeg")
    
    
    print(colored("\n=== Tesseract OCR Test Complete ===", 'green'))
    print(colored("Check the generated tesseract_ocr_results/ directory for visualization files.", 'green'))


if __name__ == "__main__":
    main() 