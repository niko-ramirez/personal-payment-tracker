#!/usr/bin/env python3
"""
Test script for the clean receipt scanner design
"""

import os
from receipt_scanner import ReceiptScanner, ReceiptItem
from tesseract_ocr import TesseractOCRProcessor
from termcolor import colored


def test_clean_receipt_scanner():
    """Test the new clean receipt scanner design"""
    print(colored("\nTesting Clean Receipt Scanner Design", 'green'))
    print(colored("=" * 50, 'green'))
    
    # Test image path
    test_image_path = "packages/receipt_scanner/test_images/real_test.jpg"
    
    if not os.path.exists(test_image_path):
        print(colored(f"Test image not found: {test_image_path}", 'red'))
        return None
    
    try:
        # Create OCR processor
        print(colored("Creating Tesseract OCR processor...", 'green'))
        ocr_processor = TesseractOCRProcessor()
        
        # Create receipt scanner
        print(colored("Creating receipt scanner...", 'green'))
        scanner = ReceiptScanner(ocr_processor, test_image_path)
        
        # Scan receipt
        print(colored("Scanning receipt...", 'green'))
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
        print(colored(f"Error testing clean receipt scanner: {e}", 'red'))
        return None


def test_receipt_item_creation():
    """Test ReceiptItem creation and manipulation"""
    print(colored("\nTesting ReceiptItem Creation", 'green'))
    print(colored("=" * 40, 'green'))
    
    # Create test items
    items = [
        ReceiptItem("Burger", 12.99, 1, 0.95),
        ReceiptItem("Fries", 8.50, 2, 0.88),
        ReceiptItem("Soda", 4.25, 1, 0.92),
    ]
    
    print(colored("Created items:", 'green'))
    for i, item in enumerate(items, 1):
        print(colored(f"  {i}. {item.name} x{item.quantity} @ ${item.price:.2f} (conf: {item.confidence:.2f})", 'green'))
    
    return items


def test_receipt_creation():
    """Test Receipt creation and manipulation"""
    print(colored("\nTesting Receipt Creation", 'green'))
    print(colored("=" * 40, 'green'))
    
    # Create receipt
    receipt = Receipt(
        merchant="Test Restaurant",
        total=25.74,
        tax=2.15,
        tip=5.00,
        image_path="test_image.jpg"
    )
    
    # Add items
    items = test_receipt_item_creation()
    if items:
        for item in items:
            receipt.add_item(item)
    
    print(colored(f"\nReceipt created:", 'green'))
    print(colored(f"Merchant: {receipt.merchant}", 'green'))
    print(colored(f"Total: ${receipt.total:.2f}", 'green'))
    print(colored(f"Tax: ${receipt.tax:.2f}", 'green'))
    print(colored(f"Tip: ${receipt.tip:.2f}", 'green'))
    print(colored(f"Items: {len(receipt.items)}", 'green'))
    
    # Test validation
    print(colored(f"\nValidation:", 'green'))
    subtotal = receipt.calculate_subtotal()
    print(colored(f"Subtotal: ${subtotal:.2f}", 'green'))
    print(colored(f"Total validation: {'✓ PASS' if receipt.validate_totals() else '✗ FAIL'}", 'green'))
    
    return receipt


def main():
    """Main test function"""
    print(colored("Clean Receipt Scanner Test", 'green'))
    print(colored("=" * 50, 'green'))
    
    # Test receipt item creation
    test_receipt_item_creation()
    
    # Test receipt creation
    test_receipt_creation()
    
    # Test full scanner
    receipt = test_clean_receipt_scanner()
    
    print(colored("\n=== Clean Receipt Scanner Test Complete ===", 'green'))
    
    if receipt:
        print(colored("✓ All tests completed successfully!", 'green'))
    else:
        print(colored("✗ Some tests failed", 'red'))


if __name__ == "__main__":
    main() 