import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from app.Services.Extractors.BaseDocumentExtractor import BaseDocumentExtractor


class ImageExtractor(BaseDocumentExtractor):
    """Image document extractor (JPEG, PNG, JPG) using enhanced OCR with preprocessing"""
    
    def _preprocess_image(self, img: Image.Image, method: str = 'adaptive') -> Image.Image:
        """Preprocess image for better OCR accuracy.
        
        Args:
            img: PIL Image object
            method: Preprocessing method ('adaptive', 'otsu', 'morphology', 'simple')
            
        Returns:
            Preprocessed PIL Image
        """
        try:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Convert PIL to numpy array for OpenCV processing
            img_array = np.array(img)
            
            # Convert to grayscale for better OCR
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            if method == 'adaptive':
                # Method 1: Adaptive thresholding (good for varying lighting)
                # Apply denoising first
                denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
                
                # Enhance contrast using CLAHE
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(denoised)
                
                # Apply adaptive thresholding
                thresh = cv2.adaptiveThreshold(
                    enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 11, 2
                )
                
            elif method == 'otsu':
                # Method 2: Otsu's thresholding (good for bimodal histograms)
                denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
                _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
            elif method == 'morphology':
                # Method 3: Morphological operations (good for noisy images)
                denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(denoised)
                
                # Apply morphological opening to remove noise
                kernel = np.ones((2, 2), np.uint8)
                opened = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel)
                
                # Apply thresholding
                _, thresh = cv2.threshold(opened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
            else:  # simple
                # Method 4: Simple grayscale with high contrast
                denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
                clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
                thresh = clahe.apply(denoised)
            
            # Convert back to PIL Image
            processed_img = Image.fromarray(thresh)
            
            return processed_img
            
        except Exception as e:
            self.logger.warning(f"Image preprocessing failed ({method}), using fallback: {e}")
            # Fallback: basic enhancement
            try:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                # Convert to grayscale
                gray = img.convert('L')
                # Enhance contrast
                enhancer = ImageEnhance.Contrast(gray)
                enhanced = enhancer.enhance(2.5)
                return enhanced
            except:
                return img
    
    def _perform_ocr_with_configs_fast(self, img: Image.Image) -> tuple:
        """Perform OCR with optimized configurations for speed and accuracy.
        
        Args:
            img: PIL Image object
            
        Returns:
            Tuple of (best_text, ocr_data_dict) where ocr_data_dict contains spatial information
        """
        import pytesseract
        
        # Optimized: Only try best PSM configs (reduced from 5 to 2-3)
        # PSM 6 is best for structured documents like passports
        # PSM 11 is good for sparse/scattered text
        psm_configs = [
            '--psm 6',  # Uniform block of text (BEST for structured documents)
            '--psm 11',  # Sparse text (good fallback)
        ]
        
        best_text = ""
        best_ocr_data = None
        best_length = 0
        
        for psm_config in psm_configs:
            try:
                # Try with whitelist first (faster and more accurate for structured docs)
                config = f"{psm_config} -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz<>/ "
                
                # Get structured OCR data with coordinates (single call for both)
                ocr_data = pytesseract.image_to_data(img, lang='eng', config=config, output_type=pytesseract.Output.DICT)
                
                # Extract text from data (faster than separate call)
                ocr_text = ' '.join([word for word in ocr_data.get('text', []) if word.strip()])
                
                # If result is poor, try without whitelist
                if len(ocr_text.strip()) < 50:
                    ocr_data_no_whitelist = pytesseract.image_to_data(img, lang='eng', config=psm_config, output_type=pytesseract.Output.DICT)
                    ocr_text_no_whitelist = ' '.join([word for word in ocr_data_no_whitelist.get('text', []) if word.strip()])
                    if len(ocr_text_no_whitelist.strip()) > len(ocr_text.strip()):
                        ocr_text = ocr_text_no_whitelist
                        ocr_data = ocr_data_no_whitelist
                
                # Use the longest result (usually most complete)
                if len(ocr_text.strip()) > best_length:
                    best_length = len(ocr_text.strip())
                    best_text = ocr_text
                    best_ocr_data = ocr_data
                    
                    # Early exit if we get a good result (>200 chars)
                    if best_length > 200:
                        self.logger.info(f"Good result found with {psm_config}, early exit")
                        break
                    
            except Exception as e:
                self.logger.debug(f"OCR config {psm_config} failed: {e}")
                continue
        
        return best_text, best_ocr_data
    
    def _sort_text_by_position(self, ocr_data: dict, min_confidence: int = 30) -> str:
        """Sort OCR text by spatial position (top-to-bottom, left-to-right) - generic approach.
        
        Args:
            ocr_data: OCR data dictionary from pytesseract
            min_confidence: Minimum confidence threshold (0-100) to filter out low-quality detections
            
        Returns:
            Text sorted by reading order
        """
        if not ocr_data or 'text' not in ocr_data:
            return ""
        
        # Extract text elements with their positions and filter by confidence
        text_elements = []
        for i in range(len(ocr_data['text'])):
            text = ocr_data['text'][i].strip()
            conf = int(ocr_data['conf'][i]) if ocr_data['conf'][i] != -1 else 0
            
            # Only include words with sufficient confidence and filter noise
            if (text and 
                int(ocr_data['level'][i]) == 5 and 
                conf >= min_confidence and
                not self._is_likely_noise(text)):
                text_elements.append({
                    'text': text,
                    'left': int(ocr_data['left'][i]),
                    'top': int(ocr_data['top'][i]),
                    'width': int(ocr_data['width'][i]),
                    'height': int(ocr_data['height'][i]),
                    'conf': conf
                })
        
        if not text_elements:
            return ""
        
        # Calculate dynamic line height for grouping (generic approach)
        heights = [elem['height'] for elem in text_elements if elem['height'] > 0]
        if not heights:
            return ""
        
        # Use median for more robust line height calculation
        import statistics
        median_height = statistics.median(heights) if len(heights) > 1 else heights[0]
        line_tolerance = median_height * 0.7  # Words within 70% of median height are on same line
        
        # Group words into lines (top-to-bottom, left-to-right)
        lines = []
        current_line = []
        current_y = text_elements[0]['top']
        
        # Sort by position first
        sorted_elements = sorted(text_elements, key=lambda x: (x['top'], x['left']))
        
        for elem in sorted_elements:
            # Check if element is on the same line (within tolerance)
            if abs(elem['top'] - current_y) <= line_tolerance:
                current_line.append(elem)
            else:
                # Finalize current line: sort by x position
                if current_line:
                    current_line.sort(key=lambda x: x['left'])
                    lines.append(current_line)
                current_line = [elem]
                current_y = elem['top']
        
        # Add last line
        if current_line:
            current_line.sort(key=lambda x: x['left'])
            lines.append(current_line)
        
        # Build text from sorted lines - intelligently join words
        result_lines = []
        for line in lines:
            # Join words on same line with smart spacing
            words = [elem['text'] for elem in line]
            
            # Smart joining: if words are close together, join without space (might be one word split)
            # If far apart, add space
            line_text_parts = []
            for i, word in enumerate(words):
                if i == 0:
                    line_text_parts.append(word)
                else:
                    # Check distance from previous word
                    prev_elem = line[i-1]
                    curr_elem = line[i]
                    gap = curr_elem['left'] - (prev_elem['left'] + prev_elem['width'])
                    
                    # If gap is small (< 2x average char width), likely same word split
                    avg_char_width = prev_elem['width'] / max(len(prev_elem['text']), 1)
                    if gap < avg_char_width * 2:
                        line_text_parts.append(word)  # No space
                    else:
                        line_text_parts.append(' ' + word)  # Add space
            
            line_text = ''.join(line_text_parts)
            result_lines.append(line_text)
        
        return '\n'.join(result_lines)
    
    def _is_likely_noise(self, text: str) -> bool:
        """Generic check if text is likely noise/garbage.
        
        Args:
            text: Text to check
            
        Returns:
            True if likely noise
        """
        import re
        
        if not text or len(text.strip()) < 2:
            return True
        
        # Check if mostly special characters
        alphanumeric_ratio = len(re.sub(r'[^A-Za-z0-9]', '', text)) / len(text) if text else 0
        if alphanumeric_ratio < 0.3:
            return True
        
        # Check for common OCR garbage patterns
        garbage_patterns = [
            r'^[^A-Za-z0-9\s]{3,}$',  # Only special chars
            r'^[a-z]{1,2}$',  # Very short lowercase (likely noise)
            r'^\d{1,2}$',  # Single or double digits alone
        ]
        
        for pattern in garbage_patterns:
            if re.match(pattern, text.strip()):
                return True
        
        return False
    
    def _classify_text_line(self, line: str, context: list = None) -> str:
        """Generic classification of text line based on content characteristics.
        
        Args:
            line: Text line to classify
            context: Previous lines for context
            
        Returns:
            Classification: 'date', 'identifier', 'name', 'location', 'other', or None if noise
        """
        import re
        
        line = line.strip()
        if not line or self._is_likely_noise(line):
            return None
        
        # Date patterns (generic)
        if re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', line):
            return 'date'
        
        # Identifier patterns (passport numbers, IDs, etc.) - generic
        # Contains mix of letters and numbers in specific patterns
        if re.search(r'[A-Z]{1,3}\s*\d{4,}', line) or re.search(r'\d{4,}\s*[A-Z]{1,3}', line):
            # Check if it's a long alphanumeric string (likely ID)
            alphanum = re.sub(r'[^A-Za-z0-9]', '', line)
            if len(alphanum) >= 8 and re.search(r'[A-Z]', alphanum) and re.search(r'\d', alphanum):
                return 'identifier'
        
        # MRZ-like patterns (machine readable zone)
        if re.search(r'[<]{2,}', line) or (len(re.sub(r'[^<]', '', line)) >= 2 and len(line) > 20):
            return 'identifier'
        
        # Name patterns (generic heuristics)
        # Multiple capitalized words, typically 2-4 words
        words = line.split()
        if 2 <= len(words) <= 5:
            capitalized_words = sum(1 for w in words if w and w[0].isupper())
            if capitalized_words >= len(words) * 0.7:  # Most words capitalized
                # Check if it looks like a name (not all caps, has lowercase)
                has_lowercase = any(c.islower() for c in line)
                if has_lowercase or len(words) == 2:  # Names often have 2-3 words
                    return 'name'
        
        # Location patterns (generic)
        # All caps, 1-3 words, might have commas
        if line.isupper() and 1 <= len(words) <= 3:
            # Check if it's a known location pattern (not too short, not all numbers)
            if len(line) >= 4 and not line.replace(' ', '').isdigit():
                return 'location'
        
        # Default: other meaningful text
        return 'other'
    
    def _structure_text_generic(self, text: str) -> str:
        """Structure text generically based on content characteristics (not template-specific).
        
        Args:
            text: Raw OCR text
            
        Returns:
            Structured and formatted text
        """
        import re
        
        # Split into lines and clean
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        if not lines:
            return text
        
        # Classify and group lines
        structured_lines = []
        current_section = []
        prev_classification = None
        
        for i, line in enumerate(lines):
            classification = self._classify_text_line(line, lines[:i])
            
            if classification is None:  # Noise, skip
                continue
            
            # Group similar classifications together
            if classification == prev_classification and current_section:
                current_section.append(line)
            else:
                # Start new section
                if current_section:
                    structured_lines.append(' '.join(current_section))
                current_section = [line]
                prev_classification = classification
        
        # Add last section
        if current_section:
            structured_lines.append(' '.join(current_section))
        
        # Return structured text (grouped by similarity, but not hardcoded labels)
        return '\n'.join(structured_lines) if structured_lines else text
    
    def _post_process_ocr_text(self, text: str) -> str:
        """Post-process OCR text to fix common errors and structure it generically.
        
        Args:
            text: Raw OCR text
            
        Returns:
            Cleaned, corrected, and structured text
        """
        import re
        
        # First, structure text generically (not template-specific)
        text = self._structure_text_generic(text)
        
        # Fix common OCR errors (generic, not template-specific)
        corrections = {
            # Common character misreads (be careful with these)
            r'\bvv\b': 'w',  # vv -> w (whole word only)
            # Fix spacing issues
            r'([a-z])([A-Z])': r'\1 \2',  # Add space between lower and uppercase
            r'([A-Za-z])(\d)': r'\1 \2',  # Add space between letter and number
            r'(\d)([A-Za-z])': r'\1 \2',  # Add space between number and letter
        }
        
        # Apply corrections carefully
        for pattern, replacement in corrections.items():
            text = re.sub(pattern, replacement, text)
        
        # Remove excessive whitespace but preserve line breaks
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple newlines to double
        
        # Generic date pattern fixes (not passport-specific)
        # Fix common OCR date errors
        text = re.sub(r'(\d{2})/1[^0-9]/(\d{4})', r'\1/10/\2', text)  # Fix month OCR errors
        
        # Clean up common formatting issues (generic)
        text = re.sub(r'<<+', '<<', text)  # Multiple < to double
        text = re.sub(r'\s+<', '<', text)  # Remove space before <
        text = re.sub(r'>\s+', '>', text)  # Remove space after >
        
        return text.strip()
    
    async def extract_text(self, file_path: str) -> str:
        """Extract text from an image file (JPEG, PNG, JPG) using enhanced OCR.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Extracted text as a string
        """
        try:
            from PIL import Image
            import pytesseract
            
            # Open image
            original_img = Image.open(file_path)
            
            # Optimized: Try best preprocessing method first, only try others if needed
            self.logger.info("Preprocessing image for better OCR accuracy...")
            processed_img = self._preprocess_image(original_img, method='adaptive')
            
            # Optimized: Try only best PSM configs first (reduce from 5 to 2-3)
            self.logger.info("Performing OCR with optimized configurations...")
            ocr_text, ocr_data = self._perform_ocr_with_configs_fast(processed_img)
            
            # If we have spatial data, sort by position for proper reading order
            if ocr_data:
                self.logger.info("Sorting text by spatial position for proper reading order...")
                sorted_text = self._sort_text_by_position(ocr_data, min_confidence=30)
                if sorted_text.strip():
                    ocr_text = sorted_text
            
            # Only try alternative preprocessing if result is poor
            if len(ocr_text.strip()) < 100:
                self.logger.info("Result quality low, trying alternative preprocessing...")
                # Try Otsu method (usually second best)
                try:
                    processed_img_alt = self._preprocess_image(original_img, method='otsu')
                    alt_text, alt_ocr_data = self._perform_ocr_with_configs_fast(processed_img_alt)
                    
                    if alt_ocr_data:
                        sorted_alt_text = self._sort_text_by_position(alt_ocr_data, min_confidence=30)
                        if sorted_alt_text.strip():
                            alt_text = sorted_alt_text
                    
                    if len(alt_text.strip()) > len(ocr_text.strip()):
                        ocr_text = alt_text
                        ocr_data = alt_ocr_data
                        self.logger.info("Alternative preprocessing provided better results")
                except Exception as e:
                    self.logger.debug(f"Alternative preprocessing failed: {e}")
            
            # Also try using OCRService's enhanced OCR as fallback
            if len(ocr_text.strip()) < 100:
                self.logger.info("Trying OCRService enhanced OCR as additional fallback...")
                try:
                    import fitz
                    
                    # Create a temporary PDF from the image
                    doc = fitz.open()
                    page = doc.new_page(width=original_img.width, height=original_img.height)
                    rect = fitz.Rect(0, 0, original_img.width, original_img.height)
                    page.insert_image(rect, filename=file_path)
                    
                    # Use OCRService's enhanced OCR
                    spans = self.ocr_service.extract_page_spans(page)
                    doc.close()
                    
                    if spans:
                        ocr_service_text = "\n".join([
                            span.get("text", "").strip() 
                            for span in spans 
                            if span.get("text", "").strip()
                        ])
                        if len(ocr_service_text.strip()) > len(ocr_text.strip()):
                            ocr_text = ocr_service_text
                            self.logger.info("OCRService provided better results")
                
                except Exception as fallback_error:
                    self.logger.debug(f"OCRService fallback failed: {fallback_error}")
            
            if ocr_text.strip():
                # Post-process to fix common OCR errors
                self.logger.info("Post-processing OCR text to fix common errors...")
                ocr_text = self._post_process_ocr_text(ocr_text)
                
                self.logger.info(f"OCR extracted {len(ocr_text)} characters from image: {file_path}")
                return self.clean_text(ocr_text)
            else:
                self.logger.warning(f"No text found in image: {file_path}")
                return "No text content found in image"
                
        except ImportError:
            self.logger.error("Required libraries not available. Install with: pip install Pillow pytesseract opencv-python")
            return "Error: Required OCR libraries not installed"
        except Exception as e:
            self.logger.error(f"Error performing OCR on image {file_path}: {e}", exc_info=True)
            return f"Error performing OCR on image: {str(e)}"

