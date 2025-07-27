#!/usr/bin/env python3
"""
Test script for the two-stage OCR implementation
"""

import cv2
import numpy as np
import os
from receipt_scanner import ReceiptScanner
from ocr_processor import TwoStageOCRProcessor, create_ocr_processor


def test_with_existing_image():
    """Test the OCR processor with the existing test image"""
    print("Testing Two-Stage OCR Processor with Existing Test Image")
    print("=" * 60)
    
    # Use existing test image
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
        
        # Initialize OCR processor
        print("\nInitializing OCR processor...")
        ocr_processor = create_ocr_processor()
        
        # Process with two-stage OCR (includes preprocessing)
        print("Processing image with two-stage OCR...")
        regions = ocr_processor.process_image(test_image)
        
        # Display results
        print(f"\nDetected {len(regions)} text regions:")
        for i, region in enumerate(regions):
            print(f"  {i+1}. '{region.text}' (confidence: {region.confidence:.2f}, type: {region.region_type})")
        
        # Extract formatted text
        extracted_text = ocr_processor.extract_text_from_regions(regions)
        print(f"\nExtracted text:\n{extracted_text}")
        
        # Get confidence score
        confidence = ocr_processor.get_confidence_score(regions)
        print(f"\nOverall confidence: {confidence:.2f}")
        
        return regions, extracted_text, confidence
        
    except Exception as e:
        print(f"Error testing OCR processor: {e}")
        return None, None, 0.0


def test_receipt_scanner_with_existing_image():
    """Test the complete receipt scanner with existing test image"""
    print("\nTesting Complete Receipt Scanner with Existing Test Image")
    print("=" * 60)
    
    # Use existing test image
    test_image_path = "packages/receipt_scanner/test_images/real_test.jpg"
    
    if not os.path.exists(test_image_path):
        print(f"Test image not found: {test_image_path}")
        return None
    
    print(f"Using test image: {test_image_path}")
    
    try:
        # Create scanner
        scanner = ReceiptScanner()
        
        # Scan receipt
        receipt = scanner.scan_receipt(test_image_path)
        
        # Display results
        print(f"Receipt scan results:")
        print(f"  Merchant: {receipt.merchant}")
        print(f"  Total: ${receipt.total:.2f}")
        print(f"  Tax: ${receipt.tax:.2f}")
        print(f"  Tip: ${receipt.tip:.2f}" if receipt.tip else "  Tip: None")
        print(f"  Items: {len(receipt.items)}")
        print(f"  Confidence: {receipt.confidence_score:.2f}")
        
        if receipt.items:
            print("\n  Items:")
            for item in receipt.items:
                print(f"    - {item.name}: ${item.price:.2f} x{item.quantity}")
        
        # Validate totals
        print(f"\n  Total validation: {'✓ PASS' if receipt.validate_totals() else '✗ FAIL'}")
        
        return receipt
        
    except Exception as e:
        print(f"Error testing receipt scanner: {e}")
        return None


def test_with_real_image(image_path: str):
    """Test with a real receipt image"""
    print(f"\nTesting with real image: {image_path}")
    print("=" * 40)
    
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        return None
    
    try:
        # Create scanner
        scanner = ReceiptScanner()
        
        # Scan receipt
        receipt = scanner.scan_receipt(image_path)
        
        # Display results
        print(f"Receipt scan results:")
        print(receipt)
        
        if receipt.items:
            print("\nItems:")
            for item in receipt.items:
                print(f"  - {item.name}: ${item.price:.2f} x{item.quantity}")
        
        return receipt
        
    except Exception as e:
        print(f"Error scanning receipt: {e}")
        return None


def test_with_ideal_test_jpeg():
    """Test specifically with ideal_test.jpeg image"""
    print("\nTesting with ideal_test.jpeg Image")
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
        
        # Initialize OCR processor
        print("\nInitializing OCR processor...")
        ocr_processor = create_ocr_processor()
        
        # Test preprocessing
        print("Testing preprocessing...")
        preprocessed = ocr_processor.preprocess_image_for_tesseract(test_image_path, target_dpi=300, add_border=True)
        
        if preprocessed is not None:
            print("✓ Preprocessing successful")
            
            # Test Tesseract OCR
            print("\nTesting Tesseract OCR...")
            tesseract_results = ocr_processor.perform_ocr_tesseract(preprocessed)
            print(f"✓ Tesseract detected {len(tesseract_results)} regions")
            
            # Test hybrid OCR
            print("\nTesting hybrid OCR...")
            hybrid_results = ocr_processor.perform_hybrid_ocr(test_image_path, preprocessed, ocr_processor.easyocr_reader)
            
            # Test boundary box display
            print("\nTesting boundary box display...")
            ocr_processor.perform_hybrid_ocr(test_image_path, preprocessed, ocr_processor.easyocr_reader, boundary_box_display=True)
            
            # Test complete receipt scanner
            print("\nTesting complete receipt scanner...")
            scanner = ReceiptScanner()
            receipt = scanner.scan_receipt(test_image_path)
            
            # Display results
            print(f"\nReceipt scan results:")
            print(f"  Merchant: {receipt.merchant}")
            print(f"  Total: ${receipt.total:.2f}")
            print(f"  Tax: ${receipt.tax:.2f}")
            print(f"  Tip: ${receipt.tip:.2f}" if receipt.tip else "  Tip: None")
            print(f"  Items: {len(receipt.items)}")
            print(f"  Confidence: {receipt.confidence_score:.2f}")
            
            if receipt.items:
                print("\n  Items:")
                for item in receipt.items:
                    print(f"    - {item.name}: ${item.price:.2f} x{item.quantity}")
            
            # Validate totals
            print(f"\n  Total validation: {'✓ PASS' if receipt.validate_totals() else '✗ FAIL'}")
            
            return receipt, tesseract_results, receipt.confidence_score
        else:
            print("✗ Preprocessing failed")
            return None, None, 0.0
        
    except Exception as e:
        print(f"Error testing with ideal_test.jpeg: {e}")
        return None, None, 0.0


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
    print("Two-Stage OCR Receipt Scanner Test")
    print("=" * 50)
    
    # List available test images
    list_test_images()
    
    # # Test 1: OCR Processor with existing image
    # regions, text, confidence = test_with_existing_image()
    
    # # Test 2: Complete Receipt Scanner with existing image
    # receipt = test_receipt_scanner_with_existing_image()
    
    # Test 3: Test with ideal_test.jpeg specifically
    test_receipt, test_tesseract_results, test_confidence = test_with_ideal_test_jpeg()
    
    # Test 4: Real image (if provided)
    import sys
    if len(sys.argv) > 1:
        real_image_path = sys.argv[1]
        test_with_real_image(real_image_path)


if __name__ == "__main__":
    main() 