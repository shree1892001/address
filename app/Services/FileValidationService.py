import os
from app.Logger.ocr_logger import get_standard_logger
from app.Exceptions.custom_exceptions import InvalidFileTypeException, FileNotFoundException
from app.Exceptions.custom_exceptions import (
    handle_general_operations, log_method_entry_exit, ExceptionSeverity
)


class FileValidationService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FileValidationService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.logger = get_standard_logger("FileValidationService")
            self._initialized = True

    @log_method_entry_exit
    @handle_general_operations(severity=ExceptionSeverity.MEDIUM)
    def validate_pdf_file(self, file):
        """Validate that the uploaded file is a PDF"""
        if not file.filename.lower().endswith(".pdf"):
            self.logger.warning(f"Invalid file type uploaded: {file.filename}")
            raise InvalidFileTypeException(
                file_type=file.filename.split('.')[-1] if '.' in file.filename else "unknown",
                supported_types=["pdf"]
            )
        return True

    @log_method_entry_exit
    @handle_general_operations(severity=ExceptionSeverity.LOW)
    def validate_file_exists(self, file_path):
        """Validate that a file exists at the given path"""
        if not os.path.exists(file_path):
            self.logger.warning(f"File not found: {file_path}")
            raise FileNotFoundException(file_path)
        return True

    @log_method_entry_exit
    @handle_general_operations(severity=ExceptionSeverity.LOW)
    def ensure_directory_exists(self, directory_path):
        """Ensure that a directory exists, create if it doesn't"""
        os.makedirs(directory_path, exist_ok=True)
        self.logger.info(f"Ensured directory exists: {directory_path}")
        return True

