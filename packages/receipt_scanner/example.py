#!/usr/bin/env python3
"""
Example usage of the ReceiptScanner class
"""

from receipt_scanner import ReceiptScanner, Receipt


def example_usage():
    """Demonstrate how to use the ReceiptScanner"""
    
    # Create scanner instance
    scanner = ReceiptScanner()
    
    # Example: Scan a receipt (replace with actual image path)
    image_path = "sample_receipt.jpg"  # You would provide a real image path
    
    try:
        # Scan the receipt
        receipt = scanner.scan_receipt(image_path)
        
        # Print basic information
        print("=== Receipt Scan Results ===")
        print(f"Merchant: {receipt.merchant}")
        print(f"Date: {receipt.date}")
        print(f"Total: ${receipt.total:.2f}")
        print(f"Tax: ${receipt.tax:.2f}")
        print(f"Tip: ${receipt.tip:.2f}" if receipt.tip else "Tip: None")
        print(f"Confidence: {receipt.confidence_score:.2f}")
        
        # Print items
        print(f"\nItems ({len(receipt.items)}):")
        for item in receipt.items:
            print(f"  - {item.name}: ${item.price:.2f} x{item.quantity}")
        
        # Validate totals
        print(f"\nTotal validation: {'✓ PASS' if receipt.validate_totals() else '✗ FAIL'}")
        
        # Calculate subtotal
        subtotal = receipt.calculate_subtotal()
        print(f"Calculated subtotal: ${subtotal:.2f}")
        
        return receipt
        
    except FileNotFoundError:
        print(f"Image file not found: {image_path}")
        print("Please provide a valid image path")
        return None
    except Exception as e:
        print(f"Error scanning receipt: {e}")
        return None


def create_mock_receipt():
    """Create a mock receipt for testing"""
    from receipt_scanner import ReceiptItem
    
    # Create items
    items = [
        ReceiptItem(name="Burger", price=15.99, quantity=1),
        ReceiptItem(name="Fries", price=4.99, quantity=1),
        ReceiptItem(name="Drink", price=2.99, quantity=1),
    ]
    
    # Create receipt
    receipt = Receipt(
        merchant="Sample Restaurant",
        total=30.69,
        tax=1.92,
        tip=4.80,
        date=None,
        items=items,
        confidence_score=0.85
    )
    
    return receipt


if __name__ == "__main__":
    print("Receipt Scanner Example")
    print("=" * 30)
    
    # Try to scan a real image (will fail if no image provided)
    print("\n1. Attempting to scan real image...")
    receipt = example_usage()
    
    # Create and display mock receipt
    print("\n2. Creating mock receipt...")
    mock_receipt = create_mock_receipt()
    print(mock_receipt)
    
    print("\n3. Mock receipt validation:")
    print(f"Total validation: {'✓ PASS' if mock_receipt.validate_totals() else '✗ FAIL'}")
    print(f"Subtotal: ${mock_receipt.calculate_subtotal():.2f}")
    
    print("\nUsage:")
    print("python receipt_scanner.py path/to/receipt.jpg")
    print("python receipt_scanner.py path/to/receipt.jpg --verbose") 