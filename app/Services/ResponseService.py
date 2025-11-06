from fastapi import HTTPException
from app.Logger.ocr_logger import get_standard_logger
from app.Exceptions.custom_exceptions import (
    InvalidFileTypeException, FileNotFoundException, NoTableFoundException,
    FileSaveException, ValidationException, FileOperationException
)
from app.Exceptions.custom_exceptions import (
    handle_general_operations, log_method_entry_exit, ExceptionSeverity
)


class ResponseService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ResponseService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.logger = get_standard_logger("ResponseService")
            self._initialized = True

    @log_method_entry_exit
    @handle_general_operations(severity=ExceptionSeverity.MEDIUM)
    def handle_table_extraction_response(self, service_response):
        """Handle table extraction service response"""
        return service_response

    @log_method_entry_exit
    @handle_general_operations(severity=ExceptionSeverity.MEDIUM)
    def handle_file_download_response(self, service_response):
        """Handle file download service response"""
        return service_response

    @log_method_entry_exit
    @handle_general_operations(severity=ExceptionSeverity.MEDIUM)
    def handle_health_check_response(self, service_response):
        """Handle health check service response"""
        return service_response

    @log_method_entry_exit
    def handle_table_extraction_error(self, error, operation_type="extraction"):
        """Handle table extraction errors and convert to HTTP exceptions"""
        if isinstance(error, (InvalidFileTypeException, FileNotFoundException, NoTableFoundException,
                             FileSaveException, ValidationException, FileOperationException)):
            self.logger.error(f"Document processing failed: {error.message}")
            raise HTTPException(status_code=error.status_code, detail=error.message)
        else:
            self.logger.error(f"Unexpected error in {operation_type}: {error}")
            raise HTTPException(status_code=500, detail=f"Failed to process document: {str(error)}")

    @log_method_entry_exit
    def handle_file_download_error(self, error):
        """Handle file download errors and convert to HTTP exceptions"""
        if isinstance(error, FileNotFoundException):
            self.logger.warning(f"CSV file not found: {error.message}")
            raise HTTPException(status_code=error.status_code, detail=error.message)
        else:
            self.logger.error(f"Download failed: {error}")
            raise HTTPException(status_code=500, detail=f"Failed to download file: {str(error)}")

    @log_method_entry_exit
    def handle_test_download_error(self, error):
        """Handle test download errors and convert to HTTP exceptions"""
        self.logger.error(f"Test download failed: {error}")
        raise HTTPException(status_code=500, detail=f"Failed to create test download: {str(error)}")

