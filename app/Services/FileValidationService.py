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
        # Check filename first
        filename = file.filename or ""
        filename_lower = filename.lower()
        
        # Check content-type
        content_type = getattr(file, 'content_type', None) or ""
        if hasattr(file, 'headers'):
            content_type = content_type or file.headers.get('content-type', '')
        
        # Check file content by reading first bytes (PDF files start with %PDF)
        is_pdf_by_content = False
        try:
            # For FastAPI UploadFile, try to read the first bytes to check PDF magic number
            if hasattr(file, 'file') and hasattr(file.file, 'read'):
                # Save current position
                try:
                    current_pos = file.file.tell()
                except:
                    current_pos = None
                
                # Read first 4 bytes
                file.file.seek(0)
                first_bytes = file.file.read(4)
                
                # Reset position
                if current_pos is not None:
                    file.file.seek(current_pos)
                else:
                    file.file.seek(0)
                
                if first_bytes and first_bytes.startswith(b'%PDF'):
                    is_pdf_by_content = True
        except Exception as e:
            self.logger.debug(f"Could not read file content for validation (this is OK): {e}")
        
        # Validate: must pass at least one check
        is_pdf = (
            filename_lower.endswith(".pdf") or
            'pdf' in content_type.lower() or
            is_pdf_by_content
        )
        
        if not is_pdf:
            self.logger.warning(f"Invalid file type uploaded: filename='{filename}', content-type='{content_type}'")
            file_ext = filename_lower.split('.')[-1] if '.' in filename_lower else "unknown"
            raise InvalidFileTypeException(
                file_type=file_ext,
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

