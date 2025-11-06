import io
import statistics
import numpy as np
from typing import List, Dict, Tuple, Optional
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import fitz  # PyMuPDF
import cv2

from app.Logger.ocr_logger import get_standard_logger
from app.Exceptions.custom_exceptions import (
    handle_ocr_processing, handle_text_extraction, handle_pdf_processing,
    log_method_entry_exit, ExceptionSeverity
)


class DynamicOCRService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DynamicOCRService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.logger = get_standard_logger("DynamicOCRService")
            self._initialized = True

    @log_method_entry_exit
    @handle_ocr_processing(severity=ExceptionSeverity.HIGH)
    def determine_ocr_requirement(self, page) -> Dict:
        """Dynamically determine if OCR is required and what type"""
        try:
            # First, try native text extraction
            native_spans = self._extract_native_text(page)
            
            # Analyze native text quality
            native_analysis = self._analyze_text_quality(native_spans)
            
            # Determine if OCR is needed
            ocr_decision = self._make_ocr_decision(native_analysis, page)
            
            self.logger.info(f"OCR Decision: {ocr_decision['required']} - {ocr_decision['reason']}")
            return ocr_decision
            
        except Exception as e:
            self.logger.error(f"OCR requirement determination failed: {e}")
            return {
                "required": True,
                "reason": "Error in analysis, defaulting to OCR",
                "confidence": 0.5,
                "ocr_type": "enhanced",
                "native_quality": 0.0
            }

    def _extract_native_text(self, page) -> List[Dict]:
        """Extract native text from PDF page"""
        try:
            spans = []
            d = page.get_text(
                "dict",
                flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_SPANS
            )
            
            if not d.get("blocks"):
                return []
            
            for block in d.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if text:
                            x0, y0, x1, y1 = span["bbox"]
                            spans.append({
                                "x0": float(x0),
                                "x1": float(x1),
                                "y0": float(y0),
                                "y1": float(y1),
                                "y_center": (y0 + y1) / 2.0,
                                "text": text,
                                "font_size": span.get("size", 0),
                                "font_flags": span.get("flags", 0)
                            })
            return spans
            
        except Exception as e:
            self.logger.warning(f"Native text extraction failed: {e}")
            return []

    def _analyze_text_quality(self, spans: List[Dict]) -> Dict:
        """Analyze quality of native text extraction"""
        try:
            if not spans:
                return {
                    "text_density": 0.0,
                    "font_consistency": 0.0,
                    "spatial_regularity": 0.0,
                    "content_richness": 0.0,
                    "overall_quality": 0.0
                }
            
            # Calculate text density
            text_density = self._calculate_text_density(spans)
            
            # Calculate font consistency
            font_consistency = self._calculate_font_consistency(spans)
            
            # Calculate spatial regularity
            spatial_regularity = self._calculate_spatial_regularity(spans)
            
            # Calculate content richness
            content_richness = self._calculate_content_richness(spans)
            
            # Overall quality score
            overall_quality = np.mean([
                text_density,
                font_consistency,
                spatial_regularity,
                content_richness
            ])
            
            return {
                "text_density": text_density,
                "font_consistency": font_consistency,
                "spatial_regularity": spatial_regularity,
                "content_richness": content_richness,
                "overall_quality": overall_quality,
                "span_count": len(spans)
            }
            
        except Exception as e:
            self.logger.warning(f"Text quality analysis failed: {e}")
            return {"overall_quality": 0.0}

    def _calculate_text_density(self, spans: List[Dict]) -> float:
        """Calculate text density in the document"""
        try:
            if not spans:
                return 0.0
            
            # Calculate total text area
            total_text_area = sum((s["x1"] - s["x0"]) * (s["y1"] - s["y0"]) for s in spans)
            
            # Calculate document bounds
            min_x = min(s["x0"] for s in spans)
            max_x = max(s["x1"] for s in spans)
            min_y = min(s["y0"] for s in spans)
            max_y = max(s["y1"] for s in spans)
            
            document_area = (max_x - min_x) * (max_y - min_y)
            
            if document_area == 0:
                return 0.0
            
            density = total_text_area / document_area
            return min(density, 1.0)
            
        except:
            return 0.0

    def _calculate_font_consistency(self, spans: List[Dict]) -> float:
        """Calculate font consistency across spans"""
        try:
            if not spans:
                return 0.0
            
            # Extract font sizes
            font_sizes = [s.get("font_size", 0) for s in spans if s.get("font_size", 0) > 0]
            
            if not font_sizes:
                return 0.0
            
            # Calculate coefficient of variation (lower = more consistent)
            mean_size = np.mean(font_sizes)
            std_size = np.std(font_sizes)
            
            if mean_size == 0:
                return 0.0
            
            cv = std_size / mean_size
            # Convert to consistency score (lower CV = higher consistency)
            consistency = max(0, 1 - cv)
            return min(consistency, 1.0)
            
        except:
            return 0.0

    def _calculate_spatial_regularity(self, spans: List[Dict]) -> float:
        """Calculate spatial regularity of text layout"""
        try:
            if len(spans) < 2:
                return 0.0
            
            # Calculate horizontal and vertical gaps
            spans_sorted_x = sorted(spans, key=lambda s: s["x0"])
            spans_sorted_y = sorted(spans, key=lambda s: s["y0"])
            
            x_gaps = []
            y_gaps = []
            
            for i in range(len(spans_sorted_x) - 1):
                gap = spans_sorted_x[i+1]["x0"] - spans_sorted_x[i]["x1"]
                if gap > 0:
                    x_gaps.append(gap)
            
            for i in range(len(spans_sorted_y) - 1):
                gap = spans_sorted_y[i+1]["y0"] - spans_sorted_y[i]["y1"]
                if gap > 0:
                    y_gaps.append(gap)
            
            # Calculate regularity based on gap consistency
            x_regularity = self._calculate_gap_regularity(x_gaps)
            y_regularity = self._calculate_gap_regularity(y_gaps)
            
            return (x_regularity + y_regularity) / 2.0
            
        except:
            return 0.0

    def _calculate_gap_regularity(self, gaps: List[float]) -> float:
        """Calculate regularity of gaps"""
        try:
            if len(gaps) < 2:
                return 0.0
            
            # Calculate coefficient of variation
            mean_gap = np.mean(gaps)
            std_gap = np.std(gaps)
            
            if mean_gap == 0:
                return 0.0
            
            cv = std_gap / mean_gap
            # Lower CV = more regular
            regularity = max(0, 1 - cv)
            return min(regularity, 1.0)
            
        except:
            return 0.0

    def _calculate_content_richness(self, spans: List[Dict]) -> float:
        """Calculate content richness (diversity of text)"""
        try:
            if not spans:
                return 0.0
            
            # Calculate character diversity
            all_text = " ".join(s["text"] for s in spans)
            unique_chars = len(set(all_text.lower()))
            total_chars = len(all_text)
            
            if total_chars == 0:
                return 0.0
            
            char_diversity = unique_chars / total_chars
            
            # Calculate word diversity
            words = all_text.split()
            unique_words = len(set(word.lower() for word in words))
            total_words = len(words)
            
            if total_words == 0:
                return char_diversity
            
            word_diversity = unique_words / total_words
            
            # Calculate average word length (indicator of content complexity)
            avg_word_length = np.mean([len(word) for word in words]) if words else 0
            word_length_score = min(avg_word_length / 10, 1.0)  # Normalize to 0-1
            
            # Combine metrics
            richness = (char_diversity + word_diversity + word_length_score) / 3.0
            return min(richness, 1.0)
            
        except:
            return 0.0

    def _make_ocr_decision(self, native_analysis: Dict, page) -> Dict:
        """Make intelligent decision about OCR requirement"""
        try:
            overall_quality = native_analysis.get("overall_quality", 0.0)
            span_count = native_analysis.get("span_count", 0)
            
            # Dynamic thresholds based on document characteristics
            page_rect = page.rect
            page_area = page_rect.width * page_rect.height
            
            # Adjust thresholds based on page size
            if page_area > 1000000:  # Large page
                quality_threshold = 0.3
                min_spans = 5
            elif page_area > 500000:  # Medium page
                quality_threshold = 0.4
                min_spans = 3
            else:  # Small page
                quality_threshold = 0.5
                min_spans = 2
            
            # Decision logic
            if span_count < min_spans:
                return {
                    "required": True,
                    "reason": f"Insufficient native text ({span_count} spans)",
                    "confidence": 0.9,
                    "ocr_type": "enhanced",
                    "native_quality": overall_quality
                }
            
            if overall_quality < quality_threshold:
                return {
                    "required": True,
                    "reason": f"Low native text quality ({overall_quality:.2f})",
                    "confidence": 0.8,
                    "ocr_type": "enhanced",
                    "native_quality": overall_quality
                }
            
            # Check for specific quality issues
            if native_analysis.get("text_density", 0) < 0.1:
                return {
                    "required": True,
                    "reason": "Very low text density",
                    "confidence": 0.7,
                    "ocr_type": "enhanced",
                    "native_quality": overall_quality
                }
            
            if native_analysis.get("font_consistency", 0) < 0.2:
                return {
                    "required": True,
                    "reason": "Inconsistent font rendering",
                    "confidence": 0.6,
                    "ocr_type": "standard",
                    "native_quality": overall_quality
                }
            
            # Native text is sufficient
            return {
                "required": False,
                "reason": f"High quality native text ({overall_quality:.2f})",
                "confidence": 0.9,
                "ocr_type": "none",
                "native_quality": overall_quality
            }
            
        except Exception as e:
            self.logger.warning(f"OCR decision making failed: {e}")
            return {
                "required": True,
                "reason": "Error in decision making, defaulting to OCR",
                "confidence": 0.5,
                "ocr_type": "enhanced",
                "native_quality": 0.0
            }

    @log_method_entry_exit
    @handle_ocr_processing(severity=ExceptionSeverity.HIGH)
    def adaptive_ocr_processing(self, page, ocr_decision: Dict) -> List[Dict]:
        """Perform adaptive OCR processing based on decision"""
        try:
            if not ocr_decision.get("required", True):
                # Use native text
                return self._extract_native_text(page)
            
            ocr_type = ocr_decision.get("ocr_type", "enhanced")
            
            if ocr_type == "standard":
                return self._standard_ocr_processing(page)
            elif ocr_type == "enhanced":
                return self._enhanced_ocr_processing(page)
            else:
                return self._fallback_ocr_processing(page)
                
        except Exception as e:
            self.logger.error(f"Adaptive OCR processing failed: {e}")
            return self._fallback_ocr_processing(page)

    def _standard_ocr_processing(self, page) -> List[Dict]:
        """Standard OCR processing"""
        try:
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes()))
            
            # Use default Tesseract configuration
            ocr_data = pytesseract.image_to_data(
                img, 
                output_type=pytesseract.Output.DICT,
                lang="ara+eng"
            )
            
            return self._process_ocr_results(ocr_data)
            
        except Exception as e:
            self.logger.warning(f"Standard OCR failed: {e}")
            return []

    def _enhanced_ocr_processing(self, page) -> List[Dict]:
        """Enhanced OCR processing with image preprocessing"""
        try:
            # Get page as image
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes()))
            
            # Analyze image quality
            quality_metrics = self._analyze_image_quality(img)
            
            # Apply preprocessing if needed
            if quality_metrics.get("overall_quality", 0.5) < 0.6:
                img = self._preprocess_image(img, quality_metrics)
            
            # Try multiple OCR configurations
            best_result = self._try_multiple_ocr_configs(img)
            
            return self._process_ocr_results(best_result)
            
        except Exception as e:
            self.logger.warning(f"Enhanced OCR failed: {e}")
            return self._standard_ocr_processing(page)

    def _fallback_ocr_processing(self, page) -> List[Dict]:
        """Fallback OCR processing"""
        try:
            pix = page.get_pixmap(dpi=200)  # Lower DPI for speed
            img = Image.open(io.BytesIO(pix.tobytes()))
            
            ocr_data = pytesseract.image_to_data(
                img, 
                output_type=pytesseract.Output.DICT
            )
            
            return self._process_ocr_results(ocr_data)
            
        except Exception as e:
            self.logger.error(f"Fallback OCR failed: {e}")
            return []

    def _analyze_image_quality(self, img: Image.Image) -> Dict:
        """Analyze image quality for preprocessing decisions"""
        try:
            img_array = np.array(img.convert('L'))
            
            # Calculate quality metrics
            blur_score = self._calculate_blur_score(img_array)
            contrast_score = self._calculate_contrast_score(img_array)
            brightness_score = self._calculate_brightness_score(img_array)
            noise_score = self._calculate_noise_score(img_array)
            
            overall_quality = np.mean([blur_score, contrast_score, brightness_score, 1 - noise_score])
            
            return {
                "blur_score": blur_score,
                "contrast_score": contrast_score,
                "brightness_score": brightness_score,
                "noise_score": noise_score,
                "overall_quality": overall_quality
            }
            
        except Exception as e:
            self.logger.warning(f"Image quality analysis failed: {e}")
            return {"overall_quality": 0.5}

    def _calculate_blur_score(self, img_array: np.ndarray) -> float:
        """Calculate blur score using Laplacian variance"""
        try:
            laplacian_var = cv2.Laplacian(img_array, cv2.CV_64F).var()
            return min(laplacian_var / 1000, 1.0)
        except:
            return 0.5

    def _calculate_contrast_score(self, img_array: np.ndarray) -> float:
        """Calculate contrast score using standard deviation"""
        try:
            contrast = np.std(img_array)
            return min(contrast / 100, 1.0)
        except:
            return 0.5

    def _calculate_brightness_score(self, img_array: np.ndarray) -> float:
        """Calculate brightness score"""
        try:
            brightness = np.mean(img_array)
            optimal_brightness = 127
            brightness_score = 1 - abs(brightness - optimal_brightness) / 127
            return max(0, brightness_score)
        except:
            return 0.5

    def _calculate_noise_score(self, img_array: np.ndarray) -> float:
        """Calculate noise level"""
        try:
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            noise = cv2.filter2D(img_array, -1, kernel)
            noise_level = np.std(noise)
            return min(noise_level / 50, 1.0)
        except:
            return 0.3

    def _preprocess_image(self, img: Image.Image, quality_metrics: Dict) -> Image.Image:
        """Apply image preprocessing based on quality metrics"""
        try:
            processed_img = img.copy()
            
            # Apply preprocessing based on quality issues
            if quality_metrics.get("blur_score", 0.5) < 0.3:
                processed_img = self._apply_sharpening(processed_img)
            
            if quality_metrics.get("contrast_score", 0.5) < 0.4:
                enhancer = ImageEnhance.Contrast(processed_img)
                processed_img = enhancer.enhance(1.5)
            
            if quality_metrics.get("brightness_score", 0.5) < 0.3:
                enhancer = ImageEnhance.Brightness(processed_img)
                processed_img = enhancer.enhance(1.2)
            
            if quality_metrics.get("noise_score", 0.3) > 0.6:
                processed_img = self._reduce_noise(processed_img)
            
            return processed_img
            
        except Exception as e:
            self.logger.warning(f"Image preprocessing failed: {e}")
            return img

    def _apply_sharpening(self, img: Image.Image) -> Image.Image:
        """Apply sharpening filter"""
        try:
            img_array = np.array(img)
            gaussian = cv2.GaussianBlur(img_array, (0, 0), 2.0)
            sharpened = cv2.addWeighted(img_array, 1.5, gaussian, -0.5, 0)
            return Image.fromarray(sharpened)
        except:
            return img

    def _reduce_noise(self, img: Image.Image) -> Image.Image:
        """Reduce image noise"""
        try:
            img_array = np.array(img)
            if len(img_array.shape) == 3:
                denoised = cv2.bilateralFilter(img_array, 9, 75, 75)
            else:
                denoised = cv2.bilateralFilter(img_array, 9, 75, 75)
            return Image.fromarray(denoised)
        except:
            return img

    def _try_multiple_ocr_configs(self, img: Image.Image) -> Dict:
        """Try multiple OCR configurations and return the best result"""
        try:
            # Dynamic configuration generation based on image characteristics
            configs = self._generate_ocr_configs(img)
            
            best_result = None
            best_confidence = 0
            
            for config in configs:
                try:
                    ocr_data = pytesseract.image_to_data(
                        img, 
                        output_type=pytesseract.Output.DICT, 
                        config=config
                    )
                    
                    confidence = self._calculate_ocr_confidence(ocr_data)
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_result = ocr_data
                        
                except Exception as e:
                    self.logger.warning(f"OCR config failed: {config}, error: {e}")
                    continue
            
            return best_result or pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            
        except Exception as e:
            self.logger.warning(f"Multiple OCR configs failed: {e}")
            return pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    def _generate_ocr_configs(self, img: Image.Image) -> List[str]:
        """Generate OCR configurations dynamically based on image characteristics"""
        try:
            configs = []
            
            # Analyze image to determine optimal configurations
            img_array = np.array(img.convert('L'))
            height, width = img_array.shape
            
            # Base configuration
            base_config = "--psm 6"
            
            # Add language support dynamically
            # This could be enhanced to detect language from image content
            lang_config = "ara+eng"
            
            # Determine PSM mode based on image characteristics
            if height > width * 1.5:  # Tall image
                psm_modes = ["6", "4", "8"]
            elif width > height * 1.5:  # Wide image
                psm_modes = ["6", "3", "4"]
            else:  # Square-ish image
                psm_modes = ["6", "3", "4", "8"]
            
            # Generate configurations
            for psm in psm_modes:
                configs.append(f"{base_config} --psm {psm} -l {lang_config}")
            
            return configs
            
        except Exception as e:
            self.logger.warning(f"OCR config generation failed: {e}")
            return ["--psm 6 -l ara+eng"]

    def _calculate_ocr_confidence(self, ocr_data: Dict) -> float:
        """Calculate overall confidence score for OCR results"""
        try:
            confidences = [int(conf) for conf in ocr_data.get("conf", []) if int(conf) > 0]
            if confidences:
                return np.mean(confidences) / 100.0
            return 0.0
        except:
            return 0.0

    def _process_ocr_results(self, ocr_data: Dict) -> List[Dict]:
        """Process OCR results into span format"""
        try:
            spans = []
            
            for i in range(len(ocr_data.get("level", []))):
                text = ocr_data.get("text", [])[i].strip()
                if not text:
                    continue
                
                x0 = float(ocr_data.get("left", [])[i])
                y0 = float(ocr_data.get("top", [])[i])
                x1 = x0 + float(ocr_data.get("width", [])[i])
                y1 = y0 + float(ocr_data.get("height", [])[i])
                
                spans.append({
                    "x0": x0,
                    "x1": x1,
                    "y0": y0,
                    "y1": y1,
                    "y_center": (y0 + y1) / 2.0,
                    "text": text,
                    "confidence": float(ocr_data.get("conf", [])[i]) if "conf" in ocr_data else 0.0
                })
            
            return spans
            
        except Exception as e:
            self.logger.warning(f"OCR result processing failed: {e}")
            return []
