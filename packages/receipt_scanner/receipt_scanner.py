import cv2
import numpy as np
from PIL import Image
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import argparse
import os
import logging

# Import the OCR processor
from ocr_processor import TwoStageOCRProcessor, create_ocr_processor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ReceiptItem:
    """Represents a single item on a receipt"""
    name: str
    price: float
    quantity: int = 1
    category: Optional[str] = None


@dataclass
class Receipt:
    """Represents a complete receipt with all extracted information"""
    merchant: str
    total: float
    tax: float
    tip: Optional[float] = None
    date: Optional[datetime] = None
    items: List[ReceiptItem] = None
    image_path: Optional[str] = None
    confidence_score: float = 0.0
    
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
    """Main class for scanning and processing receipts"""
    
    def __init__(self, tesseract_path: Optional[str] = None):
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        # Initialize OCR processor
        try:
            self.ocr_processor = create_ocr_processor(tesseract_path)
            logger.info("OCR processor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OCR processor: {e}")
            raise
    
    def scan_receipt(self, image_path: str) -> Receipt:
        """
        Main method to scan a receipt image and extract information
        
        Args:
            image_path: Path to the receipt image file
            
        Returns:
            Receipt object with extracted information
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Validate file format
        file_ext = os.path.splitext(image_path)[1].lower()
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        # Load image
        image = self._load_image(image_path)
        
        # Extract text using two-stage OCR
        extracted_text = self._extract_text(image)
        
        # Parse receipt data
        receipt_data = self._parse_receipt_text(extracted_text)
        
        # Create Receipt object
        receipt = Receipt(
            merchant=receipt_data.get('merchant', 'Unknown'),
            total=receipt_data.get('total', 0.0),
            tax=receipt_data.get('tax', 0.0),
            tip=receipt_data.get('tip'),
            date=receipt_data.get('date'),
            items=receipt_data.get('items', []),
            image_path=image_path,
            confidence_score=receipt_data.get('confidence', 0.0)
        )
        
        return receipt
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load image from file path"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        return image
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR results
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        return thresh
    
    def _extract_text(self, image: np.ndarray) -> str:
        """
        Extract text using two-stage OCR process
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Extracted text as string
        """
        try:
            # Process image with two-stage OCR (includes preprocessing)
            regions = self.ocr_processor.process_image(image)
            
            # Extract formatted text from regions
            extracted_text = self.ocr_processor.extract_text_from_regions(regions)
            
            # Get overall confidence score
            confidence = self.ocr_processor.get_confidence_score(regions)
            
            logger.info(f"OCR extracted text with confidence: {confidence:.2f}")
            logger.debug(f"Extracted text:\n{extracted_text}")
            
            return extracted_text
            
        except Exception as e:
            logger.error(f"Error in OCR text extraction: {e}")
            # Fallback to mock text if OCR fails
            return ""
    
    
    def _parse_receipt_text(self, text: str) -> Dict[str, Any]:
        """
        Parse extracted text into structured receipt data
        
        Args:
            text: Raw text from OCR
            
        Returns:
            Dictionary with parsed receipt data
        """
        lines = text.strip().split('\n')
        receipt_data = {
            'merchant': 'Unknown',
            'total': 0.0,
            'tax': 0.0,
            'tip': None,
            'date': None,
            'items': [],
            'confidence': 0.8
        }
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Extract merchant name (first non-empty line that doesn't contain numbers)
            if receipt_data['merchant'] == 'Unknown' and not any(char.isdigit() for char in line):
                receipt_data['merchant'] = line
            
            # Extract date
            elif 'date:' in line.lower():
                date_str = line.split(':')[1].strip()
                try:
                    receipt_data['date'] = datetime.strptime(date_str, '%Y-%m-%d')
                except ValueError:
                    pass
            
            # Extract items (lines with prices)
            elif '$' in line and any(char.isdigit() for char in line):
                item = self._parse_item_line(line)
                if item:
                    receipt_data['items'].append(item)
            
            # Extract totals
            elif 'total:' in line.lower():
                total_match = self._extract_price(line)
                if total_match:
                    receipt_data['total'] = total_match
            
            elif 'tax:' in line.lower():
                tax_match = self._extract_price(line)
                if tax_match:
                    receipt_data['tax'] = tax_match
            
            elif 'tip:' in line.lower():
                tip_match = self._extract_price(line)
                if tip_match:
                    receipt_data['tip'] = tip_match
        
        return receipt_data
    
    def _parse_item_line(self, line: str) -> Optional[ReceiptItem]:
        """Parse a line containing item information"""
        # Extract price
        price_match = self._extract_price(line)
        if not price_match:
            return None
        
        # Extract item name (everything before the price)
        name_part = line.split('$')[0].strip()
        
        # Simple quantity detection (assume 1 if not specified)
        quantity = 1
        if name_part and name_part[0].isdigit():
            try:
                quantity = int(name_part[0])
                name_part = name_part[1:].strip()
            except ValueError:
                pass
        
        return ReceiptItem(
            name=name_part or 'Unknown Item',
            price=price_match,
            quantity=quantity
        )
    
    def _extract_price(self, text: str) -> Optional[float]:
        """Extract price from text containing dollar amounts"""
        import re
        price_pattern = r'\$(\d+\.\d{2})'
        match = re.search(price_pattern, text)
        if match:
            return float(match.group(1))
        return None


def main():
    """Command line interface for receipt scanning"""
    parser = argparse.ArgumentParser(description='Scan and parse receipt images')
    parser.add_argument('image_path', help='Path to the receipt image file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--tesseract-path', help='Path to tesseract executable')
    
    args = parser.parse_args()
    
    try:
        # Create scanner instance
        scanner = ReceiptScanner(tesseract_path=args.tesseract_path)
        
        # Scan receipt
        receipt = scanner.scan_receipt(args.image_path)
        
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
    
    except Exception as e:
        print(f"Error scanning receipt: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 