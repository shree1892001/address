import io
import statistics
import arabic_reshaper
from bidi.algorithm import get_display
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import fitz  # PyMuPDF
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional

from app.Logger.ocr_logger import get_standard_logger
from app.Exceptions.custom_exceptions import (
    ArabicTextProcessingException, PDFOpenException, PDFCorruptedException
)
from app.Exceptions.custom_exceptions import (
    handle_ocr_processing, handle_text_extraction, handle_pdf_processing,
    log_method_entry_exit, ExceptionSeverity
)


class OCRService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OCRService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.logger = get_standard_logger("OCRService")
            self._initialized = True

    @log_method_entry_exit
    @handle_text_extraction(severity=ExceptionSeverity.LOW)
    def clean_text(self, text):
        """Clean and process text, especially Arabic text"""
        if not text:
            return ""
        text = " ".join(str(text).split())
        try:
            reshaped_text = arabic_reshaper.reshape(text)
            bidi_text = get_display(reshaped_text)
            return bidi_text
        except Exception as e:
            self.logger.warning(f"Arabic text processing failed: {e}")
            return text  # Return original text instead of raising exception

    @log_method_entry_exit
    @handle_pdf_processing(severity=ExceptionSeverity.HIGH)
    def open_pdf(self, pdf_path):
        """Open and validate PDF document"""
        try:
            doc = fitz.open(pdf_path)
            self.logger.info(f"Opened PDF: {pdf_path}")
            return doc
        except Exception as e:
            if "corrupted" in str(e).lower() or "invalid" in str(e).lower():
                raise PDFCorruptedException(pdf_path, details={"error": str(e)})
            else:
                raise PDFOpenException(pdf_path, details={"error": str(e)})

    @log_method_entry_exit
    @handle_text_extraction(severity=ExceptionSeverity.MEDIUM)
    def extract_page_spans(self, page):
        """Extract text spans from PDF page using original logic with dynamic fallback"""
        spans = []
        try:
            # Use original native text extraction first
            d = page.get_text(
                "dict",
                flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_SPANS
            )
            if not d["blocks"]:
                # Force OCR for scanned images
                pix = page.get_pixmap(dpi=300)
                img = Image.open(io.BytesIO(pix.tobytes()))
                ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config="--psm 6")
                
                self.logger.info(f"OCR found {len(ocr_data.get('text', []))} text elements")
                for i in range(len(ocr_data["text"])):
                    t = ocr_data["text"][i]
                    if t and str(t).strip():  # Accept any non-empty text
                        x0 = float(ocr_data["left"][i])
                        y0 = float(ocr_data["top"][i])
                        x1 = x0 + float(ocr_data["width"][i])
                        y1 = y0 + float(ocr_data["height"][i])
                        spans.append({
                            "x0": x0, "x1": x1, "y0": y0, "y1": y1,
                            "y_center": (y0 + y1) / 2.0, "text": str(t).strip()
                        })
                self.logger.info(f"Extracted {len(spans)} spans from OCR")
                return spans

            for block in d.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = self.clean_text(span.get("text", ""))
                        if not text:
                            continue
                        x0, y0, x1, y1 = span["bbox"]
                        spans.append({
                            "x0": float(x0),
                            "x1": float(x1),
                            "y0": float(y0),
                            "y1": float(y1),
                            "y_center": (y0 + y1) / 2.0,
                            "text": text
                        })
            return spans
        except Exception as e:
            self.logger.warning(f"Native text extraction failed, falling back to basic OCR: {e}")
            # Direct OCR fallback
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes()))
            ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            
            for i in range(len(ocr_data["text"])):
                t = ocr_data["text"][i].strip()
                if t:
                    x0 = float(ocr_data["left"][i])
                    y0 = float(ocr_data["top"][i])
                    x1 = x0 + float(ocr_data["width"][i])
                    y1 = y0 + float(ocr_data["height"][i])
                    spans.append({
                        "x0": x0, "x1": x1, "y0": y0, "y1": y1,
                        "y_center": (y0 + y1) / 2.0, "text": t
                    })
            return spans

    @log_method_entry_exit
    @handle_ocr_processing(severity=ExceptionSeverity.HIGH)
    def ocr_page(self, page):
        """Perform OCR on PDF page"""
        spans = []
        try:
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes()))
            ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, lang="ara+eng")
            
            for i in range(len(ocr_data["level"])):
                t = ocr_data["text"][i].strip()
                if not t:
                    continue
                text = self.clean_text(t)
                x0 = float(ocr_data["left"][i])
                y0 = float(ocr_data["top"][i])
                x1 = x0 + float(ocr_data["width"][i])
                y1 = y0 + float(ocr_data["height"][i])
                spans.append({
                    "x0": x0,
                    "x1": x1,
                    "y0": y0,
                    "y1": y1,
                    "y_center": (y0 + y1) / 2.0,
                    "text": text
                })
            return spans
        except Exception as e:
            self.logger.error(f"OCR processing failed: {e}")
            raise ArabicTextProcessingException(details={"error": str(e), "page_number": page.number})

    @log_method_entry_exit
    @handle_ocr_processing(severity=ExceptionSeverity.MEDIUM)
    def detect_image_quality(self, image: Image.Image) -> Dict[str, float]:
        """Detect image quality metrics for optimal processing"""
        try:
            # Convert to numpy array for analysis
            img_array = np.array(image.convert('L'))  # Convert to grayscale
            
            # Calculate quality metrics
            quality_metrics = {
                'blur_score': self._calculate_blur_score(img_array),
                'contrast_score': self._calculate_contrast_score(img_array),
                'brightness_score': self._calculate_brightness_score(img_array),
                'noise_score': self._calculate_noise_score(img_array),
                'resolution_score': self._calculate_resolution_score(img_array)
            }
            
            # Overall quality score (0-1, higher is better)
            quality_metrics['overall_quality'] = np.mean([
                quality_metrics['blur_score'],
                quality_metrics['contrast_score'],
                quality_metrics['brightness_score'],
                1 - quality_metrics['noise_score'],  # Invert noise (lower is better)
                quality_metrics['resolution_score']
            ])
            
            return quality_metrics
            
        except Exception as e:
            self.logger.warning(f"Quality detection failed: {e}")
            return {'overall_quality': 0.5}  # Default medium quality

    def _calculate_blur_score(self, img_array: np.ndarray) -> float:
        """Calculate blur score using Laplacian variance"""
        try:
            laplacian_var = cv2.Laplacian(img_array, cv2.CV_64F).var()
            # Normalize to 0-1 range (higher variance = less blur)
            return min(laplacian_var / 1000, 1.0)
        except:
            return 0.5

    def _calculate_contrast_score(self, img_array: np.ndarray) -> float:
        """Calculate contrast score using standard deviation"""
        try:
            contrast = np.std(img_array)
            # Normalize to 0-1 range
            return min(contrast / 100, 1.0)
        except:
            return 0.5

    def _calculate_brightness_score(self, img_array: np.ndarray) -> float:
        """Calculate brightness score (optimal range: 100-155)"""
        try:
            brightness = np.mean(img_array)
            # Optimal brightness is around 127 (middle of 0-255)
            # Score decreases as we move away from optimal
            optimal_brightness = 127
            brightness_score = 1 - abs(brightness - optimal_brightness) / 127
            return max(0, brightness_score)
        except:
            return 0.5

    def _calculate_noise_score(self, img_array: np.ndarray) -> float:
        """Calculate noise level (higher = more noise)"""
        try:
            # Use high-pass filter to detect noise
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            noise = cv2.filter2D(img_array, -1, kernel)
            noise_level = np.std(noise)
            return min(noise_level / 50, 1.0)  # Normalize
        except:
            return 0.3

    def _calculate_resolution_score(self, img_array: np.ndarray) -> float:
        """Calculate resolution adequacy score"""
        try:
            height, width = img_array.shape
            total_pixels = height * width
            
            # Score based on total pixels (higher resolution = better score)
            if total_pixels >= 2000000:  # 2MP+
                return 1.0
            elif total_pixels >= 1000000:  # 1MP+
                return 0.8
            elif total_pixels >= 500000:  # 500KP+
                return 0.6
            else:
                return 0.4
        except:
            return 0.5

    @log_method_entry_exit
    @handle_ocr_processing(severity=ExceptionSeverity.MEDIUM)
    def preprocess_image(self, image: Image.Image, quality_metrics: Dict[str, float]) -> Image.Image:
        """Apply image preprocessing based on quality metrics"""
        try:
            processed_image = image.copy()
            
            # Apply preprocessing based on quality issues
            if quality_metrics['blur_score'] < 0.3:
                # Apply sharpening for blurry images
                processed_image = self._apply_sharpening(processed_image)
            
            if quality_metrics['contrast_score'] < 0.4:
                # Enhance contrast
                processed_image = self._enhance_contrast(processed_image)
            
            if quality_metrics['brightness_score'] < 0.3:
                # Adjust brightness
                processed_image = self._adjust_brightness(processed_image)
            
            if quality_metrics['noise_score'] > 0.6:
                # Reduce noise
                processed_image = self._reduce_noise(processed_image)
            
            # Always apply basic enhancement
            processed_image = self._apply_basic_enhancement(processed_image)
            
            return processed_image
            
        except Exception as e:
            self.logger.warning(f"Image preprocessing failed: {e}")
            return image  # Return original if preprocessing fails

    def _apply_sharpening(self, image: Image.Image) -> Image.Image:
        """Apply sharpening filter"""
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Apply unsharp mask
            gaussian = cv2.GaussianBlur(img_array, (0, 0), 2.0)
            sharpened = cv2.addWeighted(img_array, 1.5, gaussian, -0.5, 0)
            
            return Image.fromarray(sharpened)
        except:
            return image

    def _enhance_contrast(self, image: Image.Image) -> Image.Image:
        """Enhance image contrast"""
        try:
            enhancer = ImageEnhance.Contrast(image)
            return enhancer.enhance(1.5)
        except:
            return image

    def _adjust_brightness(self, image: Image.Image) -> Image.Image:
        """Adjust image brightness"""
        try:
            enhancer = ImageEnhance.Brightness(image)
            return enhancer.enhance(1.2)
        except:
            return image

    def _reduce_noise(self, image: Image.Image) -> Image.Image:
        """Reduce image noise"""
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Apply bilateral filter for noise reduction
            if len(img_array.shape) == 3:  # Color image
                denoised = cv2.bilateralFilter(img_array, 9, 75, 75)
            else:  # Grayscale
                denoised = cv2.bilateralFilter(img_array, 9, 75, 75)
            
            return Image.fromarray(denoised)
        except:
            return image

    def _apply_basic_enhancement(self, image: Image.Image) -> Image.Image:
        """Apply basic image enhancement"""
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply slight sharpening
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)
            
            # Apply slight contrast enhancement
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.05)
            
            return image
        except:
            return image

    @log_method_entry_exit
    @handle_ocr_processing(severity=ExceptionSeverity.HIGH)
    def enhanced_ocr_page(self, page) -> List[Dict]:
        """Enhanced OCR processing with image preprocessing"""
        spans = []
        try:
            # Get page as image with adaptive DPI
            dpi = self._determine_optimal_dpi(page)
            pix = page.get_pixmap(dpi=dpi)
            img = Image.open(io.BytesIO(pix.tobytes()))
            
            # Detect image quality
            quality_metrics = self.detect_image_quality(img)
            self.logger.info(f"Image quality score: {quality_metrics['overall_quality']:.2f}")
            
            # Preprocess image if quality is poor
            if quality_metrics['overall_quality'] < 0.6:
                self.logger.info("Applying image preprocessing for better OCR")
                img = self.preprocess_image(img, quality_metrics)
            
            # Perform OCR with multiple configurations
            ocr_configs = self._get_ocr_configurations(quality_metrics)
            
            best_result = None
            best_confidence = 0
            
            for config in ocr_configs:
                try:
                    ocr_data = pytesseract.image_to_data(
                        img, 
                        output_type=pytesseract.Output.DICT, 
                        config=config
                    )
                    
                    # Calculate confidence score
                    confidence = self._calculate_ocr_confidence(ocr_data)
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_result = ocr_data
                        
                except Exception as e:
                    self.logger.warning(f"OCR config failed: {config}, error: {e}")
                    continue
            
            if best_result is None or len([t for t in best_result.get("text", []) if t.strip()]) == 0:
                self.logger.warning("No text found, trying aggressive fallback")
                try:
                    # Convert to grayscale and increase contrast
                    gray_img = img.convert('L')
                    enhancer = ImageEnhance.Contrast(gray_img)
                    contrast_img = enhancer.enhance(2.0)
                    
                    best_result = pytesseract.image_to_data(
                        contrast_img, output_type=pytesseract.Output.DICT, config="--psm 6"
                    )
                except Exception as e:
                    self.logger.error(f"Aggressive fallback failed: {e}")
                    best_result = {"text": [], "left": [], "top": [], "width": [], "height": []}
            
            # Process OCR results
            if "text" in best_result:
                for i in range(len(best_result["text"])):
                    t = best_result["text"][i].strip()
                    if t:
                        try:
                            text = self.clean_text(t)
                        except:
                            text = t
                            
                        x0 = float(best_result["left"][i])
                        y0 = float(best_result["top"][i])
                        x1 = x0 + float(best_result["width"][i])
                        y1 = y0 + float(best_result["height"][i])
                        
                        spans.append({
                            "x0": x0,
                            "x1": x1,
                            "y0": y0,
                            "y1": y1,
                            "y_center": (y0 + y1) / 2.0,
                            "text": text
                        })
            
            self.logger.info(f"Enhanced OCR extracted {len(spans)} text spans with confidence {best_confidence:.2f}")
            return spans
            
        except Exception as e:
            self.logger.error(f"Enhanced OCR processing failed: {e}")
            return self.ocr_page(page)

    def _determine_optimal_dpi(self, page) -> int:
        """Determine optimal DPI for OCR based on page characteristics"""
        try:
            # Get page dimensions
            rect = page.rect
            width, height = rect.width, rect.height
            
            # Calculate current DPI (assuming standard page size)
            if width > 0 and height > 0:
                # Estimate DPI based on page size
                if width > 800:  # Large page
                    return 300
                elif width > 600:  # Medium page
                    return 250
                else:  # Small page
                    return 200
            
            return 300  # Default high DPI
        except:
            return 300

    def _get_ocr_configurations(self, quality_metrics: Dict[str, float]) -> List[str]:
        """Get OCR configurations optimized for table structure"""
        return [
            "--psm 6",  # Uniform text block - best for tables
            "--psm 4",  # Single column
            "--psm 3"   # Fully automatic
        ]

    def _calculate_ocr_confidence(self, ocr_data: Dict) -> float:
        """Calculate overall confidence score for OCR results"""
        try:
            confidences = [int(conf) for conf in ocr_data["conf"] if int(conf) > 0]
            if confidences:
                return np.mean(confidences) / 100.0  # Normalize to 0-1
            return 0.0
        except:
            return 0.0
