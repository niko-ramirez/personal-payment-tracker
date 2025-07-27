import cv2
import numpy as np
import pytesseract
import easyocr
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TextRegion:
    """Represents a detected text region with bounding box and confidence"""
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    text: str
    confidence: float
    region_type: str = "unknown"  # "merchant", "item", "total", "date", etc.


class TwoStageOCRProcessor:
    """
    Two-stage OCR processor using Tesseract for text detection and EasyOCR for recognition
    """
    
    def __init__(self, tesseract_path: Optional[str] = None):
        """
        Initialize the OCR processor
        
        Args:
            tesseract_path: Path to tesseract executable (if not in PATH)
        """
        self.tesseract_path = tesseract_path
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Initialize EasyOCR reader
        logger.info("Initializing EasyOCR...")
        self.easyocr_reader = easyocr.Reader(['en'], gpu=False)
        logger.info("EasyOCR initialized successfully")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for OCR
        """
        preprocessed_img = cv2.resize(image, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)

        preprocessed_img = cv2.cvtColor(preprocessed_img, cv2.COLOR_BGR2GRAY)


        kernel = np.ones((1, 1), np.uint8)
        preprocessed_img = cv2.dilate(preprocessed_img, kernel, iterations=1)
        preprocessed_img = cv2.erode(preprocessed_img, kernel, iterations=1)


        preprocessed_img = cv2.threshold(cv2.GaussianBlur(preprocessed_img, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        preprocessed_img = cv2.threshold(cv2.bilateralFilter(preprocessed_img, 5, 75, 75), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        preprocessed_img = cv2.threshold(cv2.medianBlur(preprocessed_img, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        preprocessed_img = cv2.adaptiveThreshold(cv2.GaussianBlur(preprocessed_img, (5, 5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

        preprocessed_img = cv2.adaptiveThreshold(cv2.bilateralFilter(preprocessed_img, 9, 75, 75), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

        preprocessed_img = cv2.adaptiveThreshold(cv2.medianBlur(preprocessed_img, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

        return preprocessed_img

    def preprocess_image_robust(self, image: np.ndarray) -> np.ndarray:
        """
        Robust preprocessing function that handles both color and grayscale images
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        # Check if image is already grayscale
        if len(image.shape) == 3:
            # Color image - convert to grayscale
            preprocessed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            # Already grayscale
            preprocessed_img = image.copy()
        
        # Resize image for better OCR
        preprocessed_img = cv2.resize(preprocessed_img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)

        # Apply morphological operations
        kernel = np.ones((1, 1), np.uint8)
        preprocessed_img = cv2.dilate(preprocessed_img, kernel, iterations=1)
        preprocessed_img = cv2.erode(preprocessed_img, kernel, iterations=1)

        # Apply multiple thresholding techniques
        # Otsu's thresholding
        _, preprocessed_img = cv2.threshold(cv2.GaussianBlur(preprocessed_img, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Bilateral filter + Otsu
        preprocessed_img = cv2.bilateralFilter(preprocessed_img, 5, 75, 75)
        _, preprocessed_img = cv2.threshold(preprocessed_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Median blur + Otsu
        preprocessed_img = cv2.medianBlur(preprocessed_img, 3)
        _, preprocessed_img = cv2.threshold(preprocessed_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Adaptive thresholding
        preprocessed_img = cv2.adaptiveThreshold(cv2.GaussianBlur(preprocessed_img, (5, 5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

        # Final bilateral filter + adaptive threshold
        preprocessed_img = cv2.bilateralFilter(preprocessed_img, 9, 75, 75)
        preprocessed_img = cv2.adaptiveThreshold(preprocessed_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

        # Final median blur + adaptive threshold
        preprocessed_img = cv2.medianBlur(preprocessed_img, 3)
        preprocessed_img = cv2.adaptiveThreshold(preprocessed_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

        return preprocessed_img

    def process_image(self, image: np.ndarray) -> List[TextRegion]:
        """
        Process image using two-stage OCR
        
        Args:
            image: Preprocessed image as numpy array
            
        Returns:
            List of detected text regions
        """

        test_image = self.preprocess_image_for_tesseract("packages/receipt_scanner/test_images/real_test.jpg")
        tesseract_ocr_text = self.perform_ocr_tesseract(test_image)
        print(tesseract_ocr_text)

        #Stage 0: Preprocess image using robust preprocessing
        # img = self.preprocess_image_robust(image)

        # # Stage 1: Tesseract for text detection and bounding boxes
        # logger.info("Stage 1: Tesseract text detection")
        # tesseract_regions = self._detect_text_regions_tesseract(img)
        
        # # Stage 2: EasyOCR for character recognition within detected regions
        # logger.info("Stage 2: EasyOCR character recognition")
        # final_regions = self._recognize_text_easyocr(img, tesseract_regions)
        
        # Categorize regions
        categorized_regions = self._categorize_regions(final_regions)
        
        return categorized_regions

    def preprocess_image_for_tesseract(self, image_path, target_dpi=300, add_border=False):
        """
        Applies a series of image preprocessing techniques to optimize images for Tesseract OCR.

        Args:
            image_path (str): Path to the input image file.
            target_dpi (int): Desired DPI for the image. Tesseract performs best at 300 DPI. [2, 3]
                            If the original image DPI is unknown or lower, it will be scaled.
            add_border (bool): Whether to add a white border around the image. This can help
                            if text is too tightly cropped. [2]

        Returns:
            numpy.ndarray: The preprocessed image, ready for Tesseract.
        """
        # 1. Load the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image from {image_path}. Please check the path.")
            return None

        # 2. Convert to Grayscale
        # Grayscale conversion removes color variations that might confuse the OCR process. [4, 3]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 3. Resizing/DPI Adjustment
        # Tesseract performs optimally on images with a resolution of at least 300 DPI. [2, 3]
        # If the image's effective DPI is lower, scaling it up can significantly improve accuracy.
        # This is a heuristic; for precise DPI handling, you'd read image metadata.
        if target_dpi > 0:
            # A common assumption for web/screen images is ~72-96 DPI.
            # We calculate a scaling factor to bring it closer to the target_dpi.
            # If the image is already high-resolution, this step might be skipped or adjusted.
            current_height, current_width = gray.shape
            # Assuming a base DPI of 96 for calculation. Adjust if your source images have a different typical DPI.
            scale_factor = target_dpi / 96.0
            if scale_factor > 1.0: # Only upscale if the target DPI is higher
                gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
                print(f"Resized image to {gray.shape[1]}x{gray.shape} (scaled by {scale_factor:.2f} for {target_dpi} DPI).")
            else:
                print(f"Image resolution is sufficient or higher than {target_dpi} DPI. No upscaling applied.")

        # 4. Noise Reduction (Median Blur)
        # Applying denoising filters helps remove unwanted artifacts (e.g., specks, smudges)
        # leading to cleaner character recognition. [4, 3]
        denoised = cv2.medianBlur(gray, 5) # Kernel size 5 is a common choice.

        # 5. Sharpening
        # For blurry or low-resolution images, sharpening can enhance character edges,
        # making them more distinct and easier for the OCR engine to recognize. [4, 3]
        kernel_sharpening = np.array([[0, -1, 0],
                                    [-1, 5, -1],
                                    [0, -1, 0]])
        sharpened = cv2.filter2D(denoised, -1, kernel_sharpening)

        # 6. Binarization (Otsu's Thresholding)
        # This process transforms the image into a binary (black and white) format,
        # making the text stand out clearly against the background. Otsu's method
        # automatically finds the optimal threshold. [2, 3]
        _, binarized = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 7. Dilation and Erosion (Optional, for character thickness/noise removal)
        # These operations can help with bold or thin characters, or to remove very small noise.
        # The kernel size and iterations can be tuned based on your specific receipt characteristics. [2, 3]
        kernel_dilate_erode = np.ones((1, 1), np.uint8) # Small kernel for subtle effect
        dilated = cv2.dilate(binarized, kernel_dilate_erode, iterations=1)
        eroded = cv2.erode(dilated, kernel_dilate_erode, iterations=1)
        
        final_image = eroded

        # 8. Add White Border (if text is too tightly cropped)
        # Tesseract can sometimes struggle with text areas that are too tightly cropped. [2]
        if add_border:
            border_size = 10 # pixels
            final_image = cv2.copyMakeBorder(final_image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=255)
            print(f"Added a {border_size}-pixel white border.")

        # Note on Deskewing:
        # Deskewing (correcting image rotation/tilt) is a crucial step for Tesseract performance,
        # especially for mobile-captured receipts which are often skewed. [5, 3]
        # While OpenCV can be used for this (e.g., detecting text line angles with Hough Transform
        # and then rotating), it's a more complex implementation. Specialized libraries like `jdeskew`
        # (used by `deepdoctection`) can also be integrated for this purpose. [6]
        # For a general preprocessing function, a full deskewing implementation is often
        # handled separately or as an advanced step due to its complexity and variability.

        return final_image

    def perform_ocr_tesseract(self, image, lang='eng', psm=6) -> List[Dict[str, Any]]:
        """
        Performs OCR on the given image using Tesseract.

        Args:
            image (numpy.ndarray): The image (preprocessed or original) to OCR.
            lang (str): Language for Tesseract (e.g., 'eng' for English).
            psm (int): Page Segmentation Mode. This is critical for Tesseract's performance
                    on different document layouts. For receipts, common modes include:
                    - 6: Assume a single uniform block of text (often good for receipts).
                    - 11: Sparse text. Find as much text as possible in no particular order.
                    - 12: Sparse text with OSD (Orientation and Script Detection). [2]

        Returns:
            str: Extracted text.
        """
        # Convert the OpenCV image (numpy array) to a PIL Image, which pytesseract expects
        pil_image = Image.fromarray(image)

        # Define Tesseract configuration string
        # --oem 3: Use both LSTM (neural network) and legacy engine (default for Tesseract 4.00+)
        # --psm X: Set the Page Segmentation Mode
        config = f'--oem 3 --psm {psm}'

        # text = pytesseract.image_to_string(pil_image, lang=lang, config=config)
        data = pytesseract.image_to_data(pil_image, lang=lang, config=config)

        regions = []
        n_boxes = len(data['level'])
        
        for i in range(n_boxes):
            # Filter out low-confidence detections
            if int(data['conf'][i]) > 30:  # Confidence threshold
                x, y, w, h = (
                    data['left'][i],
                    data['top'][i],
                    data['width'][i],
                    data['height'][i]
                )
                
                # Filter out very small regions (likely noise)
                if w > 10 and h > 10:
                    regions.append({
                        'bbox': (x, y, w, h),
                        'text': data['text'][i],
                        'confidence': int(data['conf'][i]) / 100.0,
                        'level': data['level'][i]
                    })
        
        logger.info(f"Tesseract detected {len(regions)} text regions")
        return regions
    
    # def _detect_text_regions_tesseract(self, image: np.ndarray) -> List[Dict[str, Any]]:
    #     """
    #     Use Tesseract to detect text regions and get bounding boxes
        
    #     Args:
    #         image: Preprocessed image
            
    #     Returns:
    #         List of dictionaries with bounding box information
    #     """
    #     try:
    #         # Get detailed OCR data from Tesseract
    #         data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
    #         regions = []
    #         n_boxes = len(data['level'])
            
    #         for i in range(n_boxes):
    #             # Filter out low-confidence detections
    #             if int(data['conf'][i]) > 30:  # Confidence threshold
    #                 x, y, w, h = (
    #                     data['left'][i],
    #                     data['top'][i],
    #                     data['width'][i],
    #                     data['height'][i]
    #                 )
                    
    #                 # Filter out very small regions (likely noise)
    #                 if w > 10 and h > 10:
    #                     regions.append({
    #                         'bbox': (x, y, w, h),
    #                         'text': data['text'][i],
    #                         'confidence': int(data['conf'][i]) / 100.0,
    #                         'level': data['level'][i]
    #                     })
            
    #         logger.info(f"Tesseract detected {len(regions)} text regions")
    #         return regions
            
    #     except Exception as e:
    #         logger.error(f"Error in Tesseract detection: {e}")
    #         return []
    
    # def _recognize_text_easyocr(self, image: np.ndarray, regions: List[Dict[str, Any]]) -> List[TextRegion]:
    #     """
    #     Use EasyOCR to recognize text within detected regions
        
    #     Args:
    #         image: Original image
    #         regions: List of regions detected by Tesseract
            
    #     Returns:
    #         List of TextRegion objects with recognized text
    #     """
    #     final_regions = []
        
    #     for region in regions:
    #         x, y, w, h = region['bbox']
            
    #         # Extract region from image
    #         region_image = image[y:y+h, x:x+w]
            
    #         if region_image.size == 0:
    #             continue
            
    #         try:
    #             # Use EasyOCR to recognize text in this region
    #             results = self.easyocr_reader.readtext(region_image)
                
    #             if results:
    #                 # Combine all text found in this region
    #                 combined_text = ' '.join([result[1] for result in results])
    #                 avg_confidence = np.mean([result[2] for result in results])
                    
    #                 # Create TextRegion object
    #                 text_region = TextRegion(
    #                     bbox=region['bbox'],
    #                     text=combined_text.strip(),
    #                     confidence=avg_confidence
    #                 )
                    
    #                 final_regions.append(text_region)
                    
    #         except Exception as e:
    #             logger.warning(f"Error processing region {region['bbox']}: {e}")
    #             continue
        
    #     logger.info(f"EasyOCR processed {len(final_regions)} regions")
    #     return final_regions
    
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


def create_ocr_processor(tesseract_path: Optional[str] = None) -> TwoStageOCRProcessor:
    """
    Factory function to create OCR processor with error handling
    
    Args:
        tesseract_path: Path to tesseract executable
        
    Returns:
        Initialized TwoStageOCRProcessor
    """
    try:
        processor = TwoStageOCRProcessor(tesseract_path)
        return processor
    except Exception as e:
        logger.error(f"Failed to initialize OCR processor: {e}")
        raise 