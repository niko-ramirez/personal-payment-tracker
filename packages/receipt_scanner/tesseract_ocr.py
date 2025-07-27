import cv2
import numpy as np
import pytesseract
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from PIL import Image
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# IMPORTANT: Ensure Tesseract OCR engine is installed on your system.
# You might need to specify the path to your tesseract executable if it's not in your PATH.
# Example for Windows:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Example for Linux/macOS (often automatically found if installed via package manager):
# pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'


@dataclass
class TextRegion:
    """Represents a detected text region with bounding box and confidence"""
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    text: str
    confidence: float
    region_type: str = "unknown"  # "merchant", "item", "total", "date", etc.


class TesseractOCRProcessor:
    """
    Pure Tesseract OCR processor for receipt scanning
    """
    
    def __init__(self, image_path: str, tesseract_path: Optional[str] = None):
        """
        Initialize the Tesseract OCR processor
        
        Args:
            tesseract_path: Path to tesseract executable (if not in PATH)
        """
        self.tesseract_path = tesseract_path
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        logger.info("Tesseract OCR processor initialized successfully")

    
    def preprocess_image(self, image_path, target_dpi=300, add_border=False):
        """
        Applies a series of image preprocessing techniques to optimize images for Tesseract OCR.

        Args:
            image_path (str): Path to the input image file.
            target_dpi (int): Desired DPI for the image. Tesseract performs best at 300 DPI.
                              If the original image DPI is unknown or lower, it will be scaled.
            add_border (bool): Whether to add a white border around the image. This can help
                               if text is too tightly cropped.

        Returns:
            numpy.ndarray: The preprocessed image, ready for Tesseract.
        """
        # 1. Load the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image from {image_path}. Please check the path.")
            return None

        # 2. Convert to Grayscale
        # Grayscale conversion removes color variations that might confuse the OCR process.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 3. Resizing/DPI Adjustment
        # Tesseract performs optimally on images with a resolution of at least 300 DPI.
        if target_dpi > 0:
            current_height, current_width = gray.shape
            scale_factor = target_dpi / 96.0 # Assuming a base DPI of 96 for calculation
            if scale_factor > 1.0:
                gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
                print(f"Resized image to {gray.shape[1]}x{gray.shape} (scaled by {scale_factor:.2f} for {target_dpi} DPI).")
            else:
                print(f"Image resolution is sufficient or higher than {target_dpi} DPI. No upscaling applied.")

        # 4. Noise Reduction (Median Blur)
        # Applying denoising filters helps remove unwanted artifacts.
        denoised = cv2.medianBlur(gray, 5)

        # 5. Sharpening
        # Enhances character edges for blurry/low-resolution images.
        kernel_sharpening = np.array([[0, -1, 0],
                                      [-1, 5, -1],
                                      [0, -1, 0]])
        sharpened = cv2.filter2D(denoised, -1, kernel_sharpening)

        # 6. Binarization (Otsu's Thresholding)
        # Transforms the image into a binary (black and white) format.
        _, binarized = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 7. Dilation and Erosion (Optional, for character thickness/noise removal)
        kernel_dilate_erode = np.ones((1, 1), np.uint8)
        dilated = cv2.dilate(binarized, kernel_dilate_erode, iterations=1)
        eroded = cv2.erode(dilated, kernel_dilate_erode, iterations=1)
        
        final_image = eroded

        # 8. Add White Border (if text is too tightly cropped)
        if add_border:
            border_size = 10 # pixels
            final_image = cv2.copyMakeBorder(final_image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=255)

        return final_image

    def perform_region_ocr(self, image_path: str, boundary_box_display=False):
        """
        Performs OCR using a hybrid approach: Tesseract for bounding box detection
        and EasyOCR for text recognition within those boxes. [5]

        Args:
            image_path (str): Path to the original (unpreprocessed) image.
                                       Used to load the original image for cropping.

        Returns:
            list: A list of dictionaries, each containing 'box' (bounding box coordinates)
                  and 'text' (text recognized by EasyOCR).
        """
        preprocessed_img = self.preprocess_image(image_path, target_dpi=300, add_border=True)

        output_dict = pytesseract.image_to_data(preprocessed_img, output_type=pytesseract.Output.DICT, lang='eng', config='--psm 6')
        if boundary_box_display:
            self.boundary_box_display(image_path, preprocessed_img, output_dict)
        return output_dict

    def perform_text_ocr(self, image_path: str):
        """
        Performs OCR using Tesseract for text recognition.
        
        Args:
            image_path (str): Path to the original (unpreprocessed) image.
                                       Used to load the original image for cropping.

        Returns:
            str: The extracted text from the image.
        """
        preprocessed_img = self.preprocess_image(image_path, target_dpi=300, add_border=True)

        return pytesseract.image_to_string(preprocessed_img, lang='eng', config='--psm 6')

    def boundary_box_display(self, image_path, preprocessed_img, d):
        # Parse the filename from the path and create a better output name
        base_name = os.path.splitext(os.path.basename(image_path))[0]  # Remove extension
        output_image_name = f"{base_name}_boundaries.png"
        
        n_boxes = len(d['text'])
        for i in range(n_boxes):
            if int(d['conf'][i]) > 60:
                (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                original_img_display = cv2.rectangle(preprocessed_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        output_dir = "tesseract_ocr_results"
        os.makedirs(output_dir, exist_ok=True)
        output_image_path_viz = os.path.join(output_dir, output_image_name)
        cv2.imwrite(output_image_path_viz, preprocessed_img)
        print(f"\nVisualized bounding boxes saved to {output_image_path_viz}.")
        return

    def _categorize_regions(self, regions: List[TextRegion]) -> List[TextRegion]:
        """
        Categorize text regions based on content and position
        
        Args:
            regions: List of TextRegion objects
            
        Returns:
            List of categorized TextRegion objects
        """
        for region in regions:
            text = region.text.lower()
            
            # Categorize based on content patterns
            if any(keyword in text for keyword in ['total', 'amount', 'sum']):
                region.region_type = "total"
            elif any(keyword in text for keyword in ['tax', 'vat']):
                region.region_type = "tax"
            elif any(keyword in text for keyword in ['tip', 'gratuity']):
                region.region_type = "tip"
            elif any(keyword in text for keyword in ['date', 'time']):
                region.region_type = "date"
            elif any(keyword in text for keyword in ['$', 'price', 'cost']):
                region.region_type = "item"
            elif len(text) > 3 and not any(char.isdigit() for char in text):
                region.region_type = "merchant"
            else:
                region.region_type = "unknown"
        
        return regions
    
    def extract_text_from_regions(self, regions: List[TextRegion]) -> str:
        """
        Extract and format text from regions for receipt parsing
        
        Args:
            regions: List of TextRegion objects
            
        Returns:
            Formatted text string
        """
        # Sort regions by vertical position (top to bottom)
        sorted_regions = sorted(regions, key=lambda r: r.bbox[1])
        
        lines = []
        current_line = []
        current_y = None
        
        for region in sorted_regions:
            y = region.bbox[1]
            
            # Start new line if y position changes significantly
            if current_y is None or abs(y - current_y) > 10:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = []
                current_y = y
            
            current_line.append(region.text)
        
        # Add the last line
        if current_line:
            lines.append(' '.join(current_line))
        
        return '\n'.join(lines)

    def extract_items_from_regions(self, regions: List[TextRegion]):
        """
        Extract items from regions
        """
        items = []

        return items

    
    
    def get_confidence_score(self, regions: List[TextRegion]) -> float:
        """
        Calculate overall confidence score for the OCR results
        
        Args:
            regions: List of TextRegion objects
            
        Returns:
            Average confidence score
        """
        if not regions:
            return 0.0
        
        confidences = [region.confidence for region in regions]
        return np.mean(confidences)


def create_tesseract_processor(tesseract_path: Optional[str] = None) -> TesseractOCRProcessor:
    """
    Factory function to create Tesseract OCR processor with error handling
    
    Args:
        tesseract_path: Path to tesseract executable
        
    Returns:
        Initialized TesseractOCRProcessor
    """
    try:
        processor = TesseractOCRProcessor(tesseract_path)
        return processor
    except Exception as e:
        logger.error(f"Failed to initialize Tesseract OCR processor: {e}")
        raise 