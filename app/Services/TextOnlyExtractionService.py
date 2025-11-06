from app.Logger.ocr_logger import get_standard_logger
from app.Exceptions.custom_exceptions import (
    handle_general_operations, log_method_entry_exit, ExceptionSeverity
)
from app.Services.OCRService import OCRService


class TextOnlyExtractionService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TextOnlyExtractionService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.logger = get_standard_logger("TextOnlyExtractionService")
            self.ocr_service = OCRService()
            self._initialized = True

    @log_method_entry_exit
    @handle_general_operations(severity=ExceptionSeverity.MEDIUM)
    async def extract_text_from_pdf(self, pdf_path):
        """Extract all text from entire PDF document - comprehensive extraction"""
        doc = None
        try:
            doc = self.ocr_service.open_pdf(pdf_path)
            total_pages = len(doc)
            self.logger.info(f"Processing {total_pages} pages for complete text extraction")
            all_pages_text = []
            
            for page_idx, page in enumerate(doc):
                self.logger.info(f"Extracting text from page {page_idx + 1}/{total_pages}")
                
                # Extract using multiple methods to ensure comprehensive coverage
                # Method 1: Native text extraction (fastest, works for text-based PDFs)
                native_text = page.get_text()
                
                # Method 2: Text extraction with blocks (more detailed)
                try:
                    text_dict = page.get_text("dict", flags=0)
                    block_text = ""
                    for block in text_dict.get("blocks", []):
                        if block.get("type") == 0:  # Text block
                            for line in block.get("lines", []):
                                line_text = " ".join([span.get("text", "") for span in line.get("spans", [])])
                                if line_text.strip():
                                    block_text += line_text + "\n"
                    block_text = block_text.strip()
                except Exception as e:
                    self.logger.warning(f"Block text extraction failed for page {page_idx + 1}: {e}")
                    block_text = ""
                
                # Method 3: Extract spans (catches OCR and all text elements)
                spans = self.ocr_service.extract_page_spans(page)
                spans_text = self._format_spans_as_text(spans)
                
                # Combine all methods - use the longest/most comprehensive result
                all_texts = [native_text, block_text, spans_text]
                all_texts = [t for t in all_texts if t and t.strip()]
                
                if not all_texts:
                    self.logger.warning(f"Page {page_idx + 1}: No text found with any method")
                    continue
                
                # Always combine ALL text sources to ensure nothing is missed
                # Start with the longest as base
                combined_text = max(all_texts, key=len)
                
                # Append any other text that has unique content
                for other_text in all_texts:
                    if other_text != combined_text:
                        # If other text is significantly different (more than 10% difference), add it
                        if len(other_text) > len(combined_text) * 1.1:
                            combined_text = f"{combined_text}\n{other_text}"
                        elif abs(len(other_text) - len(combined_text)) > 50:
                            # Significant difference in length, likely has unique content
                            combined_text = f"{combined_text}\n{other_text}"
                
                if combined_text.strip():
                    all_pages_text.append(combined_text.strip())
                    self.logger.info(f"Page {page_idx + 1}: Extracted {len(combined_text)} characters")
                else:
                    self.logger.warning(f"Page {page_idx + 1}: No text extracted")
            
            # Combine all pages into single text output
            full_document_text = "\n\n".join(all_pages_text)
            self.logger.info(f"Total extraction: {len(full_document_text)} characters from {total_pages} pages")
            return {"full_text": full_document_text}
            
        finally:
            if doc:
                doc.close()

    def _format_spans_as_text(self, spans):
        """Format OCR spans into readable text - comprehensive formatting"""
        if not spans:
            return ""
        
        # Sort spans by position (top to bottom, left to right)
        spans.sort(key=lambda s: (s.get("y0", 0), s.get("x0", 0)))
        
        # Group into lines with adaptive tolerance
        lines = []
        current_line = []
        current_y = spans[0].get("y0", 0) if spans else 0
        
        # Calculate dynamic line tolerance based on span heights
        if spans:
            heights = [abs(s.get("y1", 0) - s.get("y0", 0)) for s in spans if s.get("y1", 0) > s.get("y0", 0)]
            if heights:
                avg_height = sum(heights) / len(heights)
                line_tolerance = max(15, avg_height * 1.5)  # Adaptive tolerance
            else:
                line_tolerance = 15
        else:
            line_tolerance = 15
        
        for span in spans:
            span_y = span.get("y0", 0)
            span_text = span.get("text", "").strip()
            
            if not span_text:
                continue
                
            # Check if span is on the same line
            if abs(span_y - current_y) < line_tolerance:
                current_line.append(span_text)
            else:
                # Finalize current line
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [span_text]
                current_y = span_y
        
        # Add the last line
        if current_line:
            lines.append(" ".join(current_line))
        
        return "\n".join(lines)
    
    def _merge_texts_comprehensively(self, native_text, spans_text):
        """Merge native text and spans text, removing duplicates while preserving all content"""
        if not native_text and not spans_text:
            return ""
        
        if not native_text:
            return spans_text
        
        if not spans_text:
            return native_text
        
        # If texts are very similar, use the longer one (likely more complete)
        if abs(len(native_text) - len(spans_text)) < len(native_text) * 0.1:
            # Texts are similar length - combine unique parts
            native_words = set(native_text.split())
            spans_words = set(spans_text.split())
            
            # If there's significant overlap, use the longer text
            overlap = len(native_words & spans_words)
            if overlap > len(native_words) * 0.7:
                return native_text if len(native_text) >= len(spans_text) else spans_text
            else:
                # Combine unique content
                return f"{native_text}\n{spans_text}"
        else:
            # Use the longer text (more comprehensive)
            return native_text if len(native_text) >= len(spans_text) else spans_text
