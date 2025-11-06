import re
from typing import List, Dict, Optional
from app.Logger.ocr_logger import get_standard_logger
from app.Exceptions.custom_exceptions import (
    handle_general_operations, log_method_entry_exit, ExceptionSeverity
)


class TextExtractionService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TextExtractionService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.logger = get_standard_logger("TextExtractionService")
            self._initialized = True

    @log_method_entry_exit
    @handle_general_operations(severity=ExceptionSeverity.MEDIUM)
    def extract_text_from_regions(self, text_regions: List[Dict]) -> List[Dict]:
        """Extract and organize text from text regions"""
        try:
            if not text_regions:
                return []
            
            extracted_texts = []
            
            for region in text_regions:
                text_content = self._extract_text_from_region(region)
                if text_content:
                    extracted_texts.append(text_content)
            
            self.logger.info(f"Extracted text from {len(extracted_texts)} regions")
            return extracted_texts
            
        except Exception as e:
            self.logger.error(f"Text extraction from regions failed: {e}")
            return []

    def _extract_text_from_region(self, region: Dict) -> Optional[Dict]:
        """Extract text from a single region"""
        try:
            spans = region.get("spans", [])
            if not spans:
                return None
            
            # Sort spans by reading order (top to bottom, left to right)
            sorted_spans = self._sort_spans_by_reading_order(spans)
            
            # Extract and clean text
            raw_text = " ".join(span.get("text", "") for span in sorted_spans)
            cleaned_text = self._clean_extracted_text(raw_text)
            
            # Analyze text structure
            text_structure = self._analyze_text_structure(sorted_spans)
            
            # Determine text type
            text_type = self._determine_text_type(cleaned_text, text_structure)
            
            return {
                "raw_text": raw_text,
                "cleaned_text": cleaned_text,
                "text_type": text_type,
                "structure": text_structure,
                "bounds": region.get("bounds", {}),
                "confidence": region.get("confidence", 0.0),
                "spans": sorted_spans
            }
            
        except Exception as e:
            self.logger.warning(f"Text extraction from region failed: {e}")
            return None

    def _sort_spans_by_reading_order(self, spans: List[Dict]) -> List[Dict]:
        """Sort spans by reading order (top to bottom, left to right)"""
        try:
            # First sort by y-coordinate (top to bottom)
            # Then by x-coordinate (left to right) for spans on the same line
            return sorted(spans, key=lambda s: (s.get("y0", 0), s.get("x0", 0)))
            
        except Exception as e:
            self.logger.warning(f"Span sorting failed: {e}")
            return spans

    def _clean_extracted_text(self, text: str) -> str:
        """Clean and normalize extracted text using dynamic processing"""
        try:
            if not text:
                return ""
            
            # Use dynamic text processing service
            from app.Services.DynamicTextProcessingService import DynamicTextProcessingService
            dynamic_processor = DynamicTextProcessingService()
            
            # Process text dynamically
            result = dynamic_processor.process_text_dynamically(text)
            
            return result.get("processed_text", text)
            
        except Exception as e:
            self.logger.warning(f"Dynamic text processing failed: {e}")
            # Fallback to basic cleaning
            return " ".join(text.split())

    def _fix_common_ocr_errors(self, text: str) -> str:
        """Fix common OCR recognition errors dynamically"""
        try:
            # Use dynamic text processing service for OCR error correction
            from app.Services.DynamicTextProcessingService import DynamicTextProcessingService
            dynamic_processor = DynamicTextProcessingService()
            
            # Process text to fix OCR errors
            result = dynamic_processor.process_text_dynamically(text)
            
            return result.get("processed_text", text)
            
        except Exception as e:
            self.logger.warning(f"Dynamic OCR error fixing failed: {e}")
            return text

    def _normalize_punctuation(self, text: str) -> str:
        """Normalize punctuation marks dynamically"""
        try:
            # Use dynamic text processing service for punctuation normalization
            from app.Services.DynamicTextProcessingService import DynamicTextProcessingService
            dynamic_processor = DynamicTextProcessingService()
            
            # Process text to normalize punctuation
            result = dynamic_processor.process_text_dynamically(text)
            
            return result.get("processed_text", text)
            
        except Exception as e:
            self.logger.warning(f"Dynamic punctuation normalization failed: {e}")
            return text

    def _analyze_text_structure(self, spans: List[Dict]) -> Dict:
        """Analyze the structure of the text"""
        try:
            if not spans:
                return {}
            
            # Calculate line breaks
            lines = self._group_spans_into_lines(spans)
            
            # Analyze paragraph structure
            paragraphs = self._group_lines_into_paragraphs(lines)
            
            # Calculate text metrics
            total_text = " ".join(span.get("text", "") for span in spans)
            
            return {
                "line_count": len(lines),
                "paragraph_count": len(paragraphs),
                "word_count": len(total_text.split()),
                "character_count": len(total_text),
                "avg_line_length": self._calculate_avg_line_length(lines),
                "has_paragraphs": len(paragraphs) > 1,
                "lines": lines,
                "paragraphs": paragraphs
            }
            
        except Exception as e:
            self.logger.warning(f"Text structure analysis failed: {e}")
            return {}

    def _group_spans_into_lines(self, spans: List[Dict]) -> List[List[Dict]]:
        """Group spans into lines based on vertical position"""
        try:
            if not spans:
                return []
            
            # Sort spans by y-coordinate
            sorted_spans = sorted(spans, key=lambda s: s.get("y0", 0))
            
            lines = []
            current_line = [sorted_spans[0]]
            current_y = sorted_spans[0].get("y0", 0)
            
            for span in sorted_spans[1:]:
                span_y = span.get("y0", 0)
                
                # If span is on the same line (within dynamic tolerance)
                tolerance = self._calculate_dynamic_line_tolerance(spans)
                if abs(span_y - current_y) < tolerance:
                    current_line.append(span)
                else:
                    # Sort current line by x-coordinate
                    current_line.sort(key=lambda s: s.get("x0", 0))
                    lines.append(current_line)
                    current_line = [span]
                    current_y = span_y
            
            # Add the last line
            if current_line:
                current_line.sort(key=lambda s: s.get("x0", 0))
                lines.append(current_line)
            
            return lines
            
        except Exception as e:
            self.logger.warning(f"Line grouping failed: {e}")
            return []

    def _group_lines_into_paragraphs(self, lines: List[List[Dict]]) -> List[List[List[Dict]]]:
        """Group lines into paragraphs based on spacing"""
        try:
            if not lines:
                return []
            
            paragraphs = []
            current_paragraph = [lines[0]]
            
            for i in range(1, len(lines)):
                current_line = lines[i]
                previous_line = lines[i-1]
                
                # Calculate vertical gap between lines
                current_y = min(span.get("y0", 0) for span in current_line)
                previous_y = max(span.get("y1", 0) for span in previous_line)
                gap = current_y - previous_y
                
                # If gap is large, start new paragraph (dynamic threshold)
                threshold = self._calculate_dynamic_paragraph_threshold(lines)
                if gap > threshold:
                    paragraphs.append(current_paragraph)
                    current_paragraph = [current_line]
                else:
                    current_paragraph.append(current_line)
            
            # Add the last paragraph
            if current_paragraph:
                paragraphs.append(current_paragraph)
            
            return paragraphs
            
        except Exception as e:
            self.logger.warning(f"Paragraph grouping failed: {e}")
            return []

    def _calculate_avg_line_length(self, lines: List[List[Dict]]) -> float:
        """Calculate average line length in characters"""
        try:
            if not lines:
                return 0.0
            
            total_length = 0
            for line in lines:
                line_text = " ".join(span.get("text", "") for span in line)
                total_length += len(line_text)
            
            return total_length / len(lines)
            
        except:
            return 0.0

    def _determine_text_type(self, text: str, structure: Dict) -> str:
        """Determine the type of text content dynamically"""
        try:
            if not text:
                return "empty"
            
            # Use dynamic text processing service for type determination
            from app.Services.DynamicTextProcessingService import DynamicTextProcessingService
            dynamic_processor = DynamicTextProcessingService()
            
            # Analyze text characteristics
            characteristics = dynamic_processor._analyze_text_characteristics(text)
            
            # Determine text type dynamically
            text_type = dynamic_processor._determine_text_type_dynamically(characteristics)
            
            # Adjust based on structure dynamically
            paragraph_count = structure.get("paragraph_count", 0)
            line_count = structure.get("line_count", 0)
            
            # Calculate dynamic thresholds based on document characteristics
            if paragraph_count > self._calculate_dynamic_paragraph_threshold_for_type(structure):
                return "paragraph_text"
            elif line_count > self._calculate_dynamic_line_threshold_for_type(structure):
                return "multi_line_text"
            else:
                return text_type
                
        except Exception as e:
            self.logger.warning(f"Dynamic text type determination failed: {e}")
            return "unknown"

    @log_method_entry_exit
    @handle_general_operations(severity=ExceptionSeverity.MEDIUM)
    def format_extracted_text(self, extracted_texts: List[Dict]) -> Dict:
        """Format extracted text into a structured document"""
        try:
            if not extracted_texts:
                return {
                    "formatted_text": "",
                    "sections": [],
                    "total_words": 0,
                    "total_characters": 0
                }
            
            # Sort texts by position (top to bottom)
            sorted_texts = sorted(extracted_texts, 
                                key=lambda x: (x.get("bounds", {}).get("min_y", 0), 
                                             x.get("bounds", {}).get("min_x", 0)))
            
            # Format each section
            sections = []
            total_words = 0
            total_characters = 0
            
            for i, text_content in enumerate(sorted_texts):
                section = {
                    "section_id": i + 1,
                    "text_type": text_content.get("text_type", "unknown"),
                    "content": text_content.get("cleaned_text", ""),
                    "structure": text_content.get("structure", {}),
                    "bounds": text_content.get("bounds", {}),
                    "confidence": text_content.get("confidence", 0.0)
                }
                
                sections.append(section)
                
                # Update totals
                total_words += section["structure"].get("word_count", 0)
                total_characters += section["structure"].get("character_count", 0)
            
            # Create formatted text
            formatted_text = self._create_formatted_text(sections)
            
            result = {
                "formatted_text": formatted_text,
                "sections": sections,
                "total_words": total_words,
                "total_characters": total_characters,
                "section_count": len(sections)
            }
            
            self.logger.info(f"Formatted text with {len(sections)} sections, {total_words} words")
            return result
            
        except Exception as e:
            self.logger.error(f"Text formatting failed: {e}")
            return {
                "formatted_text": "",
                "sections": [],
                "total_words": 0,
                "total_characters": 0
            }

    def _create_formatted_text(self, sections: List[Dict]) -> str:
        """Create formatted text from sections"""
        try:
            formatted_lines = []
            
            for section in sections:
                content = section.get("content", "")
                text_type = section.get("text_type", "")
                
                if not content:
                    continue
                
                # Add section based on type
                if text_type == "document_section":
                    formatted_lines.append(f"\n{content.upper()}\n")
                elif text_type == "formal_text":
                    formatted_lines.append(f"{content}\n")
                elif text_type == "paragraph_text":
                    formatted_lines.append(f"{content}\n\n")
                else:
                    formatted_lines.append(f"{content}\n")
            
            return "\n".join(formatted_lines)
            
        except Exception as e:
            self.logger.warning(f"Formatted text creation failed: {e}")
            return ""

    def _calculate_dynamic_line_tolerance(self, spans: List[Dict]) -> float:
        """Calculate dynamic line tolerance based on span characteristics"""
        try:
            if not spans:
                return 10.0  # Default fallback
            
            # Calculate average span height
            heights = [s.get("y1", 0) - s.get("y0", 0) for s in spans if s.get("y1", 0) > s.get("y0", 0)]
            if not heights:
                return 10.0
            
            avg_height = sum(heights) / len(heights)
            
            # Dynamic tolerance based on average height
            tolerance = max(5.0, avg_height * 0.3)  # 30% of average height, minimum 5px
            
            return min(tolerance, 25.0)  # Cap at 25px
            
        except Exception as e:
            self.logger.warning(f"Dynamic line tolerance calculation failed: {e}")
            return 10.0

    def _calculate_dynamic_paragraph_threshold(self, lines: List[List[Dict]]) -> float:
        """Calculate dynamic paragraph threshold based on line characteristics"""
        try:
            if not lines or len(lines) < 2:
                return 20.0  # Default fallback
            
            # Calculate gaps between consecutive lines
            gaps = []
            for i in range(1, len(lines)):
                current_line = lines[i]
                previous_line = lines[i-1]
                
                current_y = min(span.get("y0", 0) for span in current_line)
                previous_y = max(span.get("y1", 0) for span in previous_line)
                gap = current_y - previous_y
                
                if gap > 0:
                    gaps.append(gap)
            
            if not gaps:
                return 20.0
            
            # Calculate dynamic threshold based on gap statistics
            avg_gap = sum(gaps) / len(gaps)
            gap_std = (sum((g - avg_gap) ** 2 for g in gaps) / len(gaps)) ** 0.5
            
            # Threshold is average gap plus one standard deviation
            threshold = avg_gap + gap_std
            
            return max(threshold, 15.0)  # Minimum 15px
            
        except Exception as e:
            self.logger.warning(f"Dynamic paragraph threshold calculation failed: {e}")
            return 20.0

    def _calculate_dynamic_paragraph_threshold_for_type(self, structure: Dict) -> int:
        """Calculate dynamic paragraph threshold for text type determination"""
        try:
            # Base threshold
            base_threshold = 3
            
            # Adjust based on document characteristics
            word_count = structure.get("word_count", 0)
            character_count = structure.get("character_count", 0)
            
            # If document is very long, be more lenient
            if word_count > 1000 or character_count > 5000:
                return base_threshold - 1
            elif word_count < 100 or character_count < 500:
                return base_threshold + 1
            else:
                return base_threshold
                
        except Exception as e:
            self.logger.warning(f"Dynamic paragraph threshold for type calculation failed: {e}")
            return 3

    def _calculate_dynamic_line_threshold_for_type(self, structure: Dict) -> int:
        """Calculate dynamic line threshold for text type determination"""
        try:
            # Base threshold
            base_threshold = 5
            
            # Adjust based on document characteristics
            word_count = structure.get("word_count", 0)
            avg_line_length = structure.get("avg_line_length", 0)
            
            # If lines are very long, be more lenient
            if avg_line_length > 100:
                return base_threshold - 1
            elif avg_line_length < 20:
                return base_threshold + 1
            elif word_count < 50:
                return base_threshold + 2
            else:
                return base_threshold
                
        except Exception as e:
            self.logger.warning(f"Dynamic line threshold for type calculation failed: {e}")
            return 5
