import cv2
import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import argparse
import os
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ReceiptItem:
    """Represents a single item on a receipt"""
    name: str
    price: float
    quantity: int = 1
    confidence: float = 0.0
    bbox: Optional[tuple] = None


@dataclass
class Receipt:
    """Represents a complete receipt with all extracted information"""
    merchant: str = "Unknown"
    total: float = 0.0
    tax: float = 0.0
    tip: Optional[float] = None
    date: Optional[datetime] = None
    items: List[ReceiptItem] = None
    image_path: Optional[str] = None
    confidence_score: float = 0.0
    raw_text: str = ""
    
    def __post_init__(self):
        if self.items is None:
            self.items = []
    
    def add_item(self, item: ReceiptItem):
        """Add an item to the receipt"""
        self.items.append(item)
    
    def calculate_subtotal(self) -> float:
        """Calculate subtotal from items"""
        return sum(item.price * item.quantity for item in self.items)
    
    def validate_totals(self) -> bool:
        """Validate that items + tax + tip = total"""
        subtotal = self.calculate_subtotal()
        calculated_total = subtotal + self.tax
        if self.tip:
            calculated_total += self.tip
        
        # Allow for small rounding differences
        return abs(calculated_total - self.total) < 0.01
    
    def __str__(self):
        return f"""
Receipt from {self.merchant}
Date: {self.date or 'Unknown'}
Total: ${self.total:.2f}
Tax: ${self.tax:.2f}
Tip: {f'${self.tip:.2f}' if self.tip else 'None'}
Items: {len(self.items)}
Confidence: {self.confidence_score:.2f}
"""


class ReceiptScanner:
    """Clean wrapper around OCR processors for receipt scanning"""
    
    def __init__(self, ocr_processor, image_path: str):
        """
        Initialize receipt scanner with OCR processor and image
        
        Args:
            ocr_processor: OCR processor instance (e.g., TesseractOCRProcessor)
            image_path: Path to receipt image
        """
        self.ocr_processor = ocr_processor
        self.image_path = image_path
        self.receipt = Receipt(image_path=image_path)
        
    
    def scan(self) -> Receipt:
        """
        Scan receipt and extract all information
        
        Returns:
            Receipt object with extracted information
        """
        logger.info(f"Scanning receipt: {self.image_path}")
        
        # Extract text using OCR
        raw_text = self.ocr_processor.perform_text_ocr(self.image_path)
        
        # Extract OCR regions data
        regions_data = self.ocr_processor.perform_region_ocr(self.image_path)
        
        # Use simplified parser
        from receipt_parser import ReceiptParser
        parser = ReceiptParser()
        self.receipt = parser.parse_receipt(raw_text)
        self.receipt.image_path = self.image_path
        
        logger.info(f"Scan complete. Found {len(self.receipt.items)} items.")
        return self.receipt
    
    def _parse_receipt_data(self, raw_text: str, items_data: List[Dict[str, Any]]):
        """Parse raw text and items data into receipt structure"""
        lines = raw_text.strip().split('\n')
        
        # Extract merchant (first non-empty line without numbers)
        for line in lines:
            line = line.strip()
            if line and not any(char.isdigit() for char in line) and len(line) > 2:
                self.receipt.merchant = line
                break
        
        # Convert items data to ReceiptItem objects
        for item_data in items_data:
            item = ReceiptItem(
                name=item_data['name'],
                price=item_data['price'],
                quantity=item_data['quantity'],
                confidence=item_data['confidence'],
                bbox=item_data.get('bbox')
            )
            self.receipt.add_item(item)
        
        # Extract totals from raw text
        self._extract_totals(raw_text)
        
        # Calculate confidence score
        if self.receipt.items:
            self.receipt.confidence_score = sum(item.confidence for item in self.receipt.items) / len(self.receipt.items)
    
    def _extract_totals(self, text: str):
        """Extract total, tax, and tip from text"""
        lines = text.lower().split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Extract total
            if 'total:' in line:
                total_match = self._extract_price(line)
                if total_match:
                    self.receipt.total = total_match
            
            # Extract tax
            elif 'tax:' in line:
                tax_match = self._extract_price(line)
                if tax_match:
                    self.receipt.tax = tax_match
            
            # Extract tip
            elif 'tip:' in line:
                tip_match = self._extract_price(line)
                if tip_match:
                    self.receipt.tip = tip_match
    
    def _extract_price(self, text: str) -> Optional[float]:
        """Extract price from text containing dollar amounts"""
        price_pattern = r'\$(\d+\.\d{2})'
        match = re.search(price_pattern, text)
        if match:
            return float(match.group(1))
        return None
    
    def save_results(self, output_dir: str = "receipt_results"):
        """Save scan results to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save raw text
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        text_file = os.path.join(output_dir, f"{base_name}_extracted_text.txt")
        with open(text_file, 'w') as f:
            f.write(self.receipt.raw_text)
        
        # Save structured data
        data_file = os.path.join(output_dir, f"{base_name}_receipt_data.txt")
        with open(data_file, 'w') as f:
            f.write(f"Merchant: {self.receipt.merchant}\n")
            f.write(f"Total: ${self.receipt.total:.2f}\n")
            f.write(f"Tax: ${self.receipt.tax:.2f}\n")
            f.write(f"Tip: ${self.receipt.tip:.2f}\n" if self.receipt.tip else "Tip: None\n")
            f.write(f"Items: {len(self.receipt.items)}\n")
            f.write(f"Confidence: {self.receipt.confidence_score:.2f}\n\n")
            
            for i, item in enumerate(self.receipt.items, 1):
                f.write(f"{i}. {item.name} x{item.quantity} @ ${item.price:.2f}\n")
        
        logger.info(f"Results saved to {output_dir}/")


def main():
    """Command line interface for receipt scanning"""
    parser = argparse.ArgumentParser(description='Scan and parse receipt images')
    parser.add_argument('image_path', help='Path to the receipt image file')
    parser.add_argument('--tesseract-path', help='Path to tesseract executable')
    parser.add_argument('--save-results', action='store_true', help='Save results to files')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    try:
        # Import OCR processor
        from tesseract_ocr import TesseractOCRProcessor
        
        # Create OCR processor
        ocr_processor = TesseractOCRProcessor(args.tesseract_path)
        
        # Create and run scanner
        scanner = ReceiptScanner(ocr_processor, args.image_path)
        receipt = scanner.scan()
        
        # Print results
        print(f"Successfully scanned receipt from: {receipt.merchant}")
        print(receipt)
        
        if args.verbose:
            print("\nDetailed Information:")
            print(f"Image path: {receipt.image_path}")
            print(f"Confidence score: {receipt.confidence_score:.2f}")
            print(f"Total validation: {'✓' if receipt.validate_totals() else '✗'}")
            
            if receipt.items:
                print("\nItems:")
                for item in receipt.items:
                    print(f"  - {item.name}: ${item.price:.2f} x{item.quantity}")
        
        # Save results if requested
        if args.save_results:
            scanner.save_results()
    
    except Exception as e:
        print(f"Error scanning receipt: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 