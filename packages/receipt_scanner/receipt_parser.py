#!/usr/bin/env python3
"""
Advanced receipt parser using spatial analysis and pattern recognition
"""

import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from receipt_scanner import ReceiptItem, Receipt
from thefuzz import fuzz, process






class ReceiptParser:
    """Advanced receipt parser using spatial analysis and pattern recognition"""
    
    def __init__(self):
        # Common receipt patterns
        self.price_patterns = [
            r'\$(\d+\.\d{2})',  # $12.99
            r'(\d+\.\d{2})',    # 12.99
            r'\$(\d+)',          # $12
            r'(\d+)',            # 12
        ]
        
        self.total_keywords = [
            'total', 'amount', 'sum', 'due', 'balance', 'grand total',
            'subtotal', 'tax', 'tip', 'gratuity', 'service charge'
        ]
        
        self.merchant_indicators = [
            'restaurant', 'cafe', 'store', 'shop', 'market', 'deli',
            'pizza', 'burger', 'coffee', 'bar', 'grill', 'repair', 'inc'
        ]
        
        # Keywords that indicate non-item lines
        self.non_item_keywords = [
            'bill to', 'ship to', 'receipt #', 'receipt date', 'due date',
            'p.o.#', 'po#', 'payment', 'routing', 'bank', 'transfer',
            'paypal', 'email', 'terms', 'conditions', 'make checks',
            'payable to', 'address', 'street', 'lane', 'drive', 'court',
            'square', 'new york', 'cambridge', 'ma', 'ny', 'zip', 'postal',
            'phone', 'fax', 'website', 'www', 'http', 'com', 'org',
            'qty', 'description', 'unit price', 'amount', 'subtotal',
            'sales tax', 'tax rate', '%', 'percent'
        ]
    
    def parse_receipt(self, raw_text: str) -> Receipt:
        """
        Parse receipt using simple text-based approach
        
        Args:
            raw_text: Full extracted text
            
        Returns:
            Receipt object with extracted information
        """
        # Analyze receipt structure
        receipt = Receipt()
        receipt.raw_text = raw_text

        print(f"Raw text: {self.clean_text(raw_text)}")
        
        # Extract merchant name
        receipt.merchant = self._extract_merchant_simple(raw_text)
        
        # Extract items using simple line-by-line approach
        receipt.items = self._extract_items_from_text(raw_text)
        
        # Extract totals using pattern matching
        totals = self._extract_totals_from_text(raw_text)
        receipt.total = totals.get('total', 0.0)
        receipt.tax = totals.get('tax', 0.0)
        receipt.tip = totals.get('tip')
        
        # Calculate confidence
        if receipt.items:
            receipt.confidence_score = sum(item.confidence for item in receipt.items) / len(receipt.items)
        
        return receipt
    

    
    def _extract_merchant_simple(self, raw_text: str) -> str:
        """Extract merchant name using simple text-based approach"""
        lines = raw_text.split('\n')
        
        # Look for merchant in first 10 lines
        for line in lines[:10]:
            line = line.strip()
            if (len(line) > 3 and 
                not any(char.isdigit() for char in line) and
                not any(keyword in line.lower() for keyword in self.total_keywords) and
                not any(keyword in line.lower() for keyword in self.non_item_keywords)):
                return line
        
        # Fallback: Look for common merchant indicators
        for line in lines[:10]:
            line = line.strip()
            if any(indicator in line.lower() for indicator in self.merchant_indicators):
                return line
        
        return "Unknown"
    
    def _extract_items_from_text(self, raw_text: str) -> List[ReceiptItem]:
        """Extract items using simple text-based approach"""
        items = []
        
        # Process each line
        lines = raw_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for price patterns
            price_match = self._find_price(line)
            if price_match:
                # Extract item information
                item = self._parse_item_line(line, price_match)
                if item:
                    items.append(item)
        
        return items
    

    
    def _find_price(self, text: str) -> Optional[Tuple[float, int]]:
        """Find price in text, return (price, start_position)"""
        for pattern in self.price_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    price = float(match.group(1))
                    # Validate that this looks like a reasonable price
                    if self._is_reasonable_price(price, text, match.start()):
                        return (price, match.start())
                except ValueError:
                    continue
        return None
    
    def _is_reasonable_price(self, price: float, text: str, price_pos: int) -> bool:
        """Check if a price looks reasonable for a receipt item"""
        # Skip very large numbers (likely not prices)
        if price > 10000:
            return False
        
        # Skip very small numbers that might be dates or other data
        if price < 0.01:
            return False
        
        # Check if the price is at the end of the line (typical for receipt items)
        text_after_price = text[price_pos:].strip()
        if len(text_after_price) > 20:  # Price should be near the end
            return False
        
        # Skip if it looks like a date, phone number, or other non-price data
        if re.search(r'\d{1,2}/\d{1,2}/\d{4}', text):  # Date pattern
            return False
        if re.search(r'\d{3}-\d{3}-\d{4}', text):  # Phone pattern
            return False
        
        return True
    
    def _parse_item_line(self, line_text: str, price_info: Tuple[float, int]) -> Optional[ReceiptItem]:
        """Parse a line containing an item"""
        price, price_pos = price_info
        
        # Extract item name (everything before the price)
        name_part = line_text[:price_pos].strip()
        
        # Skip if it's a total/tax/tip line
        line_lower = name_part.lower()
        if any(keyword in line_lower for keyword in self.total_keywords):
            return None
        
        # Skip if it contains non-item keywords
        if any(keyword in line_lower for keyword in self.non_item_keywords):
            return None
        
        # Skip if it looks like an address, date, or other non-item text
        if self._is_non_item_text(name_part):
            return None
        
        # Extract quantity
        quantity = self._extract_quantity(name_part)
        if quantity > 1:
            # Remove quantity from name
            name_part = self._remove_quantity_from_name(name_part, quantity)
        
        # Clean up item name
        name_part = self._clean_item_name(name_part)
        
        # Only add if we have a reasonable item name
        if len(name_part) >= 2:
            return ReceiptItem(
                name=name_part,
                price=price,
                quantity=quantity,
                confidence=0.8  # Default confidence for simple parser
            )
        
        return None
    
    def _is_non_item_text(self, text: str) -> bool:
        """Check if text looks like a non-item (address, date, etc.)"""
        text_lower = text.lower()
        
        # Check for address patterns
        address_patterns = [
            r'\d+\s+[a-z]+\s+(street|st|avenue|ave|road|rd|lane|drive|dr|court|square)',
            r'[a-z]+\s+[a-z]{2}\s+\d{5}',  # City State ZIP
            r'\d{1,2}/\d{1,2}/\d{4}',  # Date pattern
            r'\d{1,2}-\d{1,2}-\d{4}',  # Date pattern
        ]
        
        for pattern in address_patterns:
            if re.search(pattern, text_lower):
                return True
        
        # Check for common non-item patterns
        non_item_patterns = [
            r'^[a-z]+\s+#\s*[a-z0-9-]+',  # Receipt numbers
            r'^p\.?o\.?\s*#\s*\d+',  # PO numbers
            r'^routing\s*\([a-z]+\)',  # Routing info
            r'^\d{9,}$',  # Long numbers (likely not prices)
        ]
        
        for pattern in non_item_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    def _extract_quantity(self, name_part: str) -> int:
        """Extract quantity from item name"""
        quantity_patterns = [
            r'^(\d+)\s*[xX]\s*',  # "2x", "2 x"
            r'^(\d+)\s*-\s*',      # "2-"
            r'^(\d+)\s*@\s*',      # "2@"
            r'^(\d+)\s*',           # "2 " (just a number)
        ]
        
        for pattern in quantity_patterns:
            match = re.match(pattern, name_part)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    continue
        
        return 1
    
    def _remove_quantity_from_name(self, name_part: str, quantity: int) -> str:
        """Remove quantity indicator from item name"""
        quantity_str = str(quantity)
        
        # Remove common quantity patterns
        patterns = [
            rf'^{quantity_str}\s*[xX]\s*',
            rf'^{quantity_str}\s*-\s*',
            rf'^{quantity_str}\s*@\s*',
            rf'^{quantity_str}\s*',
        ]
        
        for pattern in patterns:
            name_part = re.sub(pattern, '', name_part)
        
        return name_part.strip()
    
    def _clean_item_name(self, name: str) -> str:
        """Clean up item name"""
        # Remove common artifacts
        artifacts = ['*', '•', '·', '|', ':', ';', '-', '_']
        for artifact in artifacts:
            name = name.replace(artifact, ' ')
        
        # Remove extra whitespace
        name = ' '.join(name.split())
        
        return name
    
    def _extract_totals_from_text(self, raw_text: str) -> Dict[str, float]:
        """Extract totals using simple text-based approach"""
        totals = {'total': 0.0, 'tax': 0.0, 'tip': None}
        
        # Process each line
        lines = raw_text.split('\n')
        
        # Look for totals in last 15 lines
        for line in lines[-15:]:
            line = line.strip()
            if not line:
                continue
                
            line_lower = line.lower()
            
            # Look for total patterns
            if any(keyword in line_lower for keyword in ['receipt total', 'total', 'amount', 'sum', 'due']):
                price_match = self._find_price(line)
                if price_match and price_match[0] > 0:
                    totals['total'] = price_match[0]
            
            # Look for tax patterns
            elif any(keyword in line_lower for keyword in ['sales tax', 'tax', 'vat']):
                price_match = self._find_price(line)
                if price_match and price_match[0] > 0:
                    totals['tax'] = price_match[0]
            
            # Look for tip patterns
            elif any(keyword in line_lower for keyword in ['tip', 'gratuity']):
                price_match = self._find_price(line)
                if price_match and price_match[0] > 0:
                    totals['tip'] = price_match[0]
        
        return totals

    def clean_text(self, text):
        """
        Performs advanced cleaning on extracted text with context awareness.
        - Converts to lowercase.
        - Removes extra whitespace.
        - Corrects common OCR misinterpretations.
        - Fixes numbers in words and comma placement.
        """
        if not isinstance(text, str):
            return ""
        
        # Split into lines for processing
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue

            cleaned_line = ""
            for word in line.split():
                cleaned_word = self._apply_ocr_corrections(word)
                cleaned_line += cleaned_word + " "
            
            cleaned_lines.append(cleaned_line)
        
        return '\n'.join(cleaned_lines)
    
    def _apply_ocr_corrections(self, word):
        """Apply OCR-specific corrections to text with heuristics"""
        # Split text into words and numbers for context-aware processing
        
        if len(word) == 1:
            return self.clean_single_letter(word)

        # Check if this word looks like a number or price
        if self._looks_like_number(word):
            # Apply aggressive OCR corrections for numbers
            return self._correct_number(word)
            
        return self._correct_word(word)
        
    def _looks_like_number(self, word):
        """Check if a word looks like a number or price"""
        # Remove currency symbols for checking
        clean_word = word.replace('$', '')
        
        # Check if it contains mostly digits and decimal points
        digit_count = sum(1 for c in clean_word if c.isdigit())
        total_chars = len(clean_word)
        
        # If more than 50% are digits, treat as number
        if total_chars > 0 and digit_count / total_chars > 0.5:
            return True
        
        # Check for common price patterns
        price_patterns = [
            r'^\$?\d+\.\d{2}$',  # $12.99 or 12.99
            r'^\$?\d+$',          # $12 or 12
            r'^\d+\.\d{1,2}$',    # 12.9 or 12.99
        ]
        
        for pattern in price_patterns:
            if re.match(pattern, clean_word):
                return True
        
        return False

    def clean_single_letter(self, word):
        """Apply aggressive OCR corrections for single letters"""
        # Common OCR errors for single letters
        number_corrections = {
            '?': '2',  # '?' often misread as '2'
            's': '5',  # 's' often misread as '$'
            '[': '1',
            ']': '1',
            '{': '1',
            '}': '1',
        }
        return self.replace_with_corrections(word, number_corrections)
    
    
    def _correct_number(self, word):
        """Apply aggressive OCR corrections for numbers"""
        
        # Common OCR errors for numbers
        number_corrections = {
            'o': '0',  # 'O' often misread as '0'
            'l': '1',  # 'l' (lowercase L) often misread as '1'
            'i': '1',  # 'i' often misread as '1'
            '|': '1',  # '|' often misread as '1'
            '?': '2',  # '?' often misread as '2'
            'z': '2',  # 'z' often misread as '2'
            'b': '8',  # 'b' often misread as '8'
            'g': '9',  # 'g' often misread as '9'
            's': '5',  # 's' often misread as '5'
            'a': '4',  # 'a' often misread as '4'
            'e': '3',  # 'e' often misread as '3'
            '€': '',
            '£': '',
            '¥': '',
        }
        corrected = self._fix_malformed_commas(word)
        return self.replace_with_corrections(corrected, number_corrections)
    
    def _correct_word(self, word):
        """Apply conservative OCR corrections for words"""
        # Only apply corrections that are very likely to be OCR errors
        word_corrections = {
            '€': 'e',
            '£': 'e',
            '¥': 'e',
            '[': 'l',
            ']': 'l',
            '{': 'l',
            '}': 'l',
            '0': 'o',  # '0' often misread as 'o'
            '1': 'l',  # '1' often misread as 'l'
            '2': 'z',  # '2' often misread as 'z'
            '3': 'e',  # '3' often misread as 'e'
            '4': 'a',  # '4' often misread as 'a'
            '5': 's',  # '5' often misread as 's'
            '6': 'g',  # '6' often misread as 'g'
            '7': 't',  # '7' often misread as 't'
            '8': 'b',  # '8' often misread as 'b'
            '9': 'g',  # '9' often misread as 'g'
            "!": "l", # "!" often misread as "l"
        }
        return self.replace_with_corrections(word, word_corrections)

    def replace_with_corrections(self, word, corrections):
        """Replace a string with a correction"""
        corrected = word
        for old, new in corrections.items():
            corrected = corrected.replace(old, new)
        return corrected
    
    def _fix_malformed_commas(self, word):
        """Fix malformed comma patterns by replacing comma with period"""
        import re
        
        # Pattern for malformed commas (like 12,9 or 12,99)
        malformed_pattern = r'^(\d+),(\d{1,2})$'
        
        def replace_malformed_comma(match):
            before_comma = match.group(1)
            after_comma = match.group(2)
            return f"{before_comma}.{after_comma}"
        
        return re.sub(malformed_pattern, replace_malformed_comma, word)
    
