import pandas as pd
import subprocess
import os
from datetime import datetime

try:
    import ocrmypdf
except ImportError:
    ocrmypdf = None
from app.Logger.ocr_logger import get_standard_logger
from app.Exceptions.custom_exceptions import NoTableFoundException
from app.Exceptions.custom_exceptions import (
    handle_general_operations, log_method_entry_exit, monitor_performance, ExceptionSeverity
)
from app.Services.FileValidationService import FileValidationService
from app.Common.Constants import (
    CLASS_TEXT_CONF_MIN,
    CLASS_TOP_TABLE_CONF_MAX,
    TABLE_MIN_QUALITY,
    SINGLE_COL_MIN_ROWS,
    SINGLE_COL_MIN_QUALITY,
    SCANNED_TEXT_MIN_CHARS,
    SPATIAL_SAME_LINE_TOL_PX,
    WORD_TOKEN_MAX_LEN,
    SHORT_WORD_RATIO_MIN,
    TABLE_MIN_NUMERIC_CHAR_RATIO,
    ALPHA_CHAR_RATIO_MIN,
    PARAGRAPH_LIKE_MIN_COLS,
    AVG_TOKEN_LEN_MIN,
    AVG_TOKEN_LEN_MAX,
)
from app.Services.FileUploadService import FileUploadService
from app.Services.OCRService import OCRService
from app.Services.TableProcessingService import TableProcessingService
from app.Services.FileSaveService import FileSaveService
from app.Services.DocumentClassificationService import DocumentClassificationService
from app.Services.TextExtractionService import TextExtractionService
from app.Services.TextOnlyExtractionService import TextOnlyExtractionService


class TableExtractionService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TableExtractionService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.logger = get_standard_logger("TableExtractionService")
            self.file_validation_service = FileValidationService()
            self.file_upload_service = FileUploadService()
            self.ocr_service = OCRService()
            self.table_processing_service = TableProcessingService()
            self.file_save_service = FileSaveService()
            self.document_classification_service = DocumentClassificationService()
            self.text_extraction_service = TextExtractionService()
            self.text_only_extraction_service = TextOnlyExtractionService()
            self._initialized = True

    @log_method_entry_exit
    @monitor_performance
    @handle_general_operations(severity=ExceptionSeverity.HIGH)
    async def extract_complete_text_from_pdf(self, file):
        """Extract complete text content from entire PDF document - comprehensive extraction"""
        temp_files = []
        
        try:
            self.logger.info(f"=== EXTRACT COMPLETE TEXT - Starting for file: {file.filename} ===")
            
            # Validate file type
            self.file_validation_service.validate_pdf_file(file)
            
            # Save uploaded file
            file_paths = await self.file_upload_service.save_uploaded_file(file)
            temp_files.append(file_paths["pdf_path"])
            
            # Use multiple extraction methods to ensure comprehensive coverage
            doc = self.ocr_service.open_pdf(file_paths["pdf_path"])
            all_text_methods = []
            
            try:
                # Method 1: Direct text extraction
                direct_text = ""
                for page_idx, page in enumerate(doc):
                    page_text = page.get_text()
                    if page_text.strip():
                        direct_text += page_text + "\n\n"
                if direct_text.strip():
                    all_text_methods.append(("direct", direct_text.strip()))
                
                # Method 2: Dictionary-based extraction
                dict_text = ""
                for page_idx, page in enumerate(doc):
                    text_dict = page.get_text("dict")
                    page_text = ""
                    for block in text_dict.get("blocks", []):
                        if block.get("type") == 0:  # Text block
                            for line in block.get("lines", []):
                                line_text = ""
                                for span in line.get("spans", []):
                                    line_text += span.get("text", "") + " "
                                if line_text.strip():
                                    page_text += line_text.strip() + "\n"
                    if page_text.strip():
                        dict_text += page_text + "\n"
                if dict_text.strip():
                    all_text_methods.append(("dictionary", dict_text.strip()))
                
                # Method 3: Spans-based extraction
                spans_text = ""
                for page_idx, page in enumerate(doc):
                    spans = self.ocr_service.extract_page_spans(page)
                    if spans:
                        # Sort spans by position
                        spans.sort(key=lambda s: (s.get("y0", 0), s.get("x0", 0)))
                        page_text = ""
                        current_y = None
                        line_text = ""
                        
                        for span in spans:
                            span_y = span.get("y0", 0)
                            span_text = span.get("text", "").strip()
                            
                            if not span_text:
                                continue
                            
                            # Check if we're on a new line
                            if current_y is None or abs(span_y - current_y) > 5:
                                if line_text.strip():
                                    page_text += line_text.strip() + "\n"
                                line_text = span_text
                                current_y = span_y
                            else:
                                line_text += " " + span_text
                        
                        # Add the last line
                        if line_text.strip():
                            page_text += line_text.strip() + "\n"
                        
                        if page_text.strip():
                            spans_text += page_text + "\n"
                
                if spans_text.strip():
                    all_text_methods.append(("spans", spans_text.strip()))
                
            finally:
                if doc:
                    doc.close()
            
            # Combine all methods - use the most comprehensive result
            if not all_text_methods:
                self.logger.error("No text content found with any extraction method")
                raise NoTableFoundException(file_paths["pdf_path"], details={"reason": "No text content found"})
            
            # Find the method that extracted the most text
            best_method, full_text = max(all_text_methods, key=lambda x: len(x[1]))
            
            # If multiple methods have similar lengths, combine unique content
            for method_name, method_text in all_text_methods:
                if method_name != best_method and len(method_text) > len(full_text) * 0.8:
                    # Check for unique content
                    best_words = set(full_text.lower().split())
                    method_words = set(method_text.lower().split())
                    unique_words = method_words - best_words
                    
                    # If this method has significant unique content, append it
                    if len(unique_words) > len(method_words) * 0.1:
                        full_text += "\n\n" + method_text
            
            self.logger.info(f"Extracted {len(full_text)} characters using method: {best_method}")
            
            # Clean up the text
            full_text = self._clean_extracted_text(full_text)
            
            # Return complete response structure
            response = {
                "success": True,
                "message": f"Successfully extracted content from {file.filename}",
                "document_type": "text_document",
                "tables": [],
                "text_content": {
                    "formatted_text": full_text,
                    "sections": self._create_comprehensive_sections(full_text),
                    "total_words": len(full_text.split()),
                    "total_characters": len(full_text),
                    "section_count": len(full_text.split('\n\n'))
                },
                "content_priority": [{
                    "type": "text",
                    "priority": 1,
                    "confidence": 0.95
                }],
                "confidence": 0.95,
                "excel_file": None,
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"=== EXTRACT COMPLETE TEXT - Success: {len(full_text)} chars ===")
            return response
            
        except Exception as e:
            self.logger.error(f"EXTRACT COMPLETE TEXT FAILED: {e}", exc_info=True)
            # Cleanup temp files on error
            self.file_upload_service.cleanup_temp_files(temp_files)
            raise
    
    def _clean_extracted_text(self, text):
        """Clean and format extracted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                # Remove excessive spaces
                line = ' '.join(line.split())
                cleaned_lines.append(line)
            elif cleaned_lines and cleaned_lines[-1]:  # Add empty line only if previous line wasn't empty
                cleaned_lines.append('')
        
        return '\n'.join(cleaned_lines).strip()
    
    def _create_comprehensive_sections(self, full_text):
        """Create comprehensive sections structure for text content"""
        try:
            # Split text into meaningful sections
            lines = [line.strip() for line in full_text.split('\n') if line.strip()]
            
            sections = []
            current_section = []
            section_id = 1
            
            for line in lines:
                current_section.append(line)
                
                # Create a new section every 5-10 lines or at natural breaks
                if (len(current_section) >= 5 and 
                    (line.endswith('.') or line.endswith(':') or 
                     any(keyword in line.upper() for keyword in ['PROJECTS', 'EXPERIENCE', 'EDUCATION', 'SKILLS', 'CONTACT']))):
                    
                    section_content = '\n'.join(current_section)
                    sections.append({
                        "section_id": section_id,
                        "text_type": "text_heavy",
                        "content": section_content,
                        "structure": {
                            "line_count": len(current_section),
                            "paragraph_count": 1,
                            "word_count": len(section_content.split()),
                            "character_count": len(section_content),
                            "avg_line_length": len(section_content) / len(current_section),
                            "has_paragraphs": False
                        },
                        "confidence": 0.9
                    })
                    
                    current_section = []
                    section_id += 1
                    
                    if len(sections) >= 10:  # Limit to 10 sections
                        break
            
            # Add remaining content as final section
            if current_section and len(sections) < 10:
                section_content = '\n'.join(current_section)
                sections.append({
                    "section_id": section_id,
                    "text_type": "text_heavy",
                    "content": section_content,
                    "structure": {
                        "line_count": len(current_section),
                        "paragraph_count": 1,
                        "word_count": len(section_content.split()),
                        "character_count": len(section_content),
                        "avg_line_length": len(section_content) / len(current_section),
                        "has_paragraphs": False
                    },
                    "confidence": 0.9
                })
            
            return sections
        except Exception as e:
            self.logger.warning(f"Failed to create comprehensive sections: {e}")
            return []

    # Add placeholder methods for compatibility
    async def extract_table_from_pdf(self, file):
        """Placeholder method"""
        return {"success": False, "message": "Method not implemented"}
    
    async def extract_table_simple(self, file):
        """Placeholder method"""
        return {"success": False, "message": "Method not implemented"}