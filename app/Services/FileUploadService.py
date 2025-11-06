import os
import uuid
from app.Logger.ocr_logger import get_standard_logger
from app.Exceptions.custom_exceptions import FileSaveException
from app.Exceptions.custom_exceptions import (
    handle_file_operations, log_method_entry_exit, ExceptionSeverity
)


class FileUploadService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FileUploadService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.logger = get_standard_logger("FileUploadService")
            self._initialized = True

    @log_method_entry_exit
    @handle_file_operations(severity=ExceptionSeverity.MEDIUM)
    async def save_uploaded_file(self, file, temp_directory="temp_uploads"):
        """Save uploaded file to temporary directory and return file paths"""
        try:
            # Generate unique filename
            file_id = str(uuid.uuid4())
            temp_pdf_path = f"{temp_directory}/{file_id}.pdf"
            
            # Ensure temp directory exists
            os.makedirs(temp_directory, exist_ok=True)
            
            # Save uploaded file
            with open(temp_pdf_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            self.logger.info(f"Successfully saved uploaded file: {file.filename} to {temp_pdf_path}")
            
            return {
                "file_id": file_id,
                "pdf_path": temp_pdf_path,
                "json_path": f"{temp_directory}/{file_id}.json",
                "csv_path": f"{temp_directory}/{file_id}.csv",
                "excel_path": f"{temp_directory}/{file_id}.xlsx"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to save uploaded file: {e}")
            raise FileSaveException(temp_pdf_path, details={"error": str(e), "filename": file.filename})

    @log_method_entry_exit
    @handle_file_operations(severity=ExceptionSeverity.LOW)
    def cleanup_temp_files(self, file_paths):
        """Clean up temporary files"""
        cleaned_files = []
        for path in file_paths:
            if os.path.exists(path):
                try:
                    os.remove(path)
                    cleaned_files.append(path)
                    self.logger.info(f"Cleaned up temp file: {path}")
                except Exception as e:
                    self.logger.warning(f"Failed to clean up temp file {path}: {e}")
        return cleaned_files

