#!/usr/bin/env python3
"""
Test script for text-to-items conversion logic
"""

import sys
import os
from receipt_scanner import ReceiptScanner, ReceiptItem
from termcolor import colored
from tesseract_ocr import TesseractOCRProcessor


def test_full_receipt_parsing(image_path: str):
    """Test parsing complete receipt text"""
    print(colored(f"\nTesting {image_path} Receipt Parsing", 'green'))
    print(colored("=" * 50, 'green'))

    
    try:
        ocr_processor = TesseractOCRProcessor()
        
        # Create receipt scanner with a dummy image path
        scanner = ReceiptScanner(ocr_processor, image_path)

        print(colored(f"Scanning {image_path}...", 'green'))
        
        # Scan the receipt
        receipt = scanner.scan()
        
        # Display results
        print(colored(f"\nScan Results:", 'green'))
        print(colored(f"Merchant: {receipt.merchant}", 'green'))
        print(colored(f"Total: ${receipt.total:.2f}", 'green'))
        print(colored(f"Tax: ${receipt.tax:.2f}", 'green'))
        print(colored(f"Tip: {f'${receipt.tip:.2f}' if receipt.tip else 'None'}", 'green'))
        print(colored(f"Items: {len(receipt.items)}", 'green'))
        print(colored(f"Confidence: {receipt.confidence_score:.2f}", 'green'))
        
        if receipt.items:
            print(colored(f"\nItems:", 'green'))
            for i, item in enumerate(receipt.items, 1):
                print(colored(f"  {i}. {item.name} x{item.quantity} @ ${item.price:.2f} (conf: {item.confidence:.2f})", 'green'))
        
        # Validate totals
        print(colored(f"\nValidation:", 'green'))
        subtotal = receipt.calculate_subtotal()
        print(colored(f"Calculated subtotal: ${subtotal:.2f}", 'green'))
        print(colored(f"Receipt total: ${receipt.total:.2f}", 'green'))
        print(colored(f"Total validation: {'✓ PASS' if receipt.validate_totals() else '✗ FAIL'}", 'green'))
        
        # Save results
        print(colored(f"\nSaving results...", 'green'))
        scanner.save_results()
        
        return receipt
            
    except Exception as e:
        print(colored(f"✗ ERROR: {e}", 'red'))


def list_test_images():
    """List available test images"""
    images = []
    test_images_dir = "packages/receipt_scanner/test_images"
    if os.path.exists(test_images_dir):
        for file in os.listdir(test_images_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                images.append(os.path.join(test_images_dir, file))
    return images



def main():
    """Main test function"""
    
    images = list_test_images()
    for img in images:
        test_full_receipt_parsing(img)
    
    print(colored("\n=== Text-to-Items Test Complete ===", 'green'))


if __name__ == "__main__":
    main() 