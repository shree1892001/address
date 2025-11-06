from datetime import datetime
from app.Logger.ocr_logger import get_standard_logger
from app.Exceptions.custom_exceptions import NoTableFoundException
from app.Exceptions.custom_exceptions import (
    handle_general_operations, log_method_entry_exit, monitor_performance, ExceptionSeverity
)
from app.Services.FileValidationService import FileValidationService
from app.Services.FileUploadService import FileUploadService
from app.Services.OCRService import OCRService


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
            self._initialized = True

    @log_method_entry_exit
    @monitor_performance
    @handle_general_operations(severity=ExceptionSeverity.HIGH)
    async def extract_complete_text_from_pdf(self, file):
        """Extract complete text content from entire PDF document"""
        temp_files = []
        
        try:
            self.logger.info(f"Starting complete text extraction for: {file.filename}")
            
            # Validate and save file
            self.file_validation_service.validate_pdf_file(file)
            file_paths = await self.file_upload_service.save_uploaded_file(file)
            temp_files.append(file_paths["pdf_path"])
            
            # Extract text using multiple methods
            doc = self.ocr_service.open_pdf(file_paths["pdf_path"])
            all_text = ""
            
            try:
                for page_idx, page in enumerate(doc):
                    # Method 1: Direct extraction
                    page_text = page.get_text()
                    
                    # Method 2: Dictionary extraction for better formatting
                    text_dict = page.get_text("dict")
                    dict_text = ""
                    for block in text_dict.get("blocks", []):
                        if block.get("type") == 0:
                            for line in block.get("lines", []):
                                line_text = ""
                                for span in line.get("spans", []):
                                    line_text += span.get("text", "") + " "
                                if line_text.strip():
                                    dict_text += line_text.strip() + "\n"
                    
                    # Use the longer text
                    final_page_text = dict_text if len(dict_text) > len(page_text) else page_text
                    
                    if final_page_text.strip():
                        all_text += final_page_text + "\n\n"
                        
            finally:
                if doc:
                    doc.close()
            
            if not all_text.strip():
                raise NoTableFoundException(file_paths["pdf_path"], details={"reason": "No text found"})
            
            # Clean text
            all_text = self._clean_text(all_text)
            
            self.logger.info(f"Extracted {len(all_text)} characters")
            
            return {
                "success": True,
                "message": f"Successfully extracted content from {file.filename}",
                "document_type": "text_document",
                "tables": [],
                "text_content": {
                    "formatted_text": all_text,
                    "total_words": len(all_text.split()),
                    "total_characters": len(all_text)
                },
                "confidence": 0.95,
                "excel_file": None,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Text extraction failed: {e}", exc_info=True)
            self.file_upload_service.cleanup_temp_files(temp_files)
            raise
    
    def _clean_text(self, text):
        """Clean extracted text"""
        lines = text.split('\n')
        cleaned = []
        
        for line in lines:
            line = line.strip()
            if line:
                cleaned.append(' '.join(line.split()))
            elif cleaned and cleaned[-1]:
                cleaned.append('')
        
        return '\n'.join(cleaned).strip()