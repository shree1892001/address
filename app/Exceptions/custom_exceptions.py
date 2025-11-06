

import functools
import inspect
import traceback
from typing import Optional, Dict, Any, Callable, Type
from enum import Enum
import logging

from app.Logger.ocr_logger import get_standard_logger


class BaseOCRException(Exception):
    """Base exception class for all OCR-related exceptions"""
    
    def __init__(self, message: str, status_code: int = 500, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary format for API responses"""
        return {
            "error": True,
            "message": self.message,
            "status_code": self.status_code,
            "details": self.details,
            "exception_type": self.__class__.__name__
        }


class PDFProcessingException(BaseOCRException):
    """Exception raised when PDF processing fails"""
    
    def __init__(self, message: str = "PDF processing failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=422, details=details)


class PDFOpenException(PDFProcessingException):
    """Exception raised when PDF file cannot be opened"""
    
    def __init__(self, file_path: str, details: Optional[Dict[str, Any]] = None):
        message = f"Failed to open PDF file: {file_path}"
        details = details or {}
        details["file_path"] = file_path
        super().__init__(message, details=details)


class PDFCorruptedException(PDFProcessingException):
    """Exception raised when PDF file is corrupted"""
    
    def __init__(self, file_path: str, details: Optional[Dict[str, Any]] = None):
        message = f"PDF file is corrupted: {file_path}"
        details = details or {}
        details["file_path"] = file_path
        super().__init__(message, details=details)


class PDFEmptyException(PDFProcessingException):
    """Exception raised when PDF file is empty or has no content"""
    
    def __init__(self, file_path: str, details: Optional[Dict[str, Any]] = None):
        message = f"PDF file is empty or has no extractable content: {file_path}"
        details = details or {}
        details["file_path"] = file_path
        super().__init__(message, details=details)


class TextExtractionException(BaseOCRException):
    """Exception raised when text extraction fails"""
    
    def __init__(self, message: str = "Text extraction failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=422, details=details)


class OCRProcessingException(TextExtractionException):
    """Exception raised when OCR processing fails"""
    
    def __init__(self, page_number: int, details: Optional[Dict[str, Any]] = None):
        message = f"OCR processing failed on page {page_number}"
        details = details or {}
        details["page_number"] = page_number
        super().__init__(message, details=details)


class TableExtractionException(BaseOCRException):
    """Exception raised when table extraction fails"""
    
    def __init__(self, message: str = "Table extraction failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=422, details=details)


class NoTableFoundException(TableExtractionException):
    """Exception raised when no table is found in the document"""
    
    def __init__(self, file_path: str, details: Optional[Dict[str, Any]] = None):
        message = f"No table found in document: {file_path}"
        details = details or {}
        details["file_path"] = file_path
        super().__init__(message, details=details)


class ColumnDetectionException(TableExtractionException):
    """Exception raised when column detection fails"""
    
    def __init__(self, details: Optional[Dict[str, Any]] = None):
        message = "Failed to detect table columns"
        super().__init__(message, details=details)


class RowClusteringException(TableExtractionException):
    """Exception raised when row clustering fails"""
    
    def __init__(self, details: Optional[Dict[str, Any]] = None):
        message = "Failed to cluster table rows"
        super().__init__(message, details=details)


class FileOperationException(BaseOCRException):
    """Exception raised when file operations fail"""
    
    def __init__(self, message: str = "File operation failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=500, details=details)


class FileNotFoundException(FileOperationException):
    """Exception raised when file is not found"""
    
    def __init__(self, file_path: str, details: Optional[Dict[str, Any]] = None):
        message = f"File not found: {file_path}"
        details = details or {}
        details["file_path"] = file_path
        super().__init__(message, status_code=404, details=details)


class FilePermissionException(FileOperationException):
    """Exception raised when file permission is denied"""
    
    def __init__(self, file_path: str, details: Optional[Dict[str, Any]] = None):
        message = f"Permission denied accessing file: {file_path}"
        details = details or {}
        details["file_path"] = file_path
        super().__init__(message, status_code=403, details=details)


class FileSaveException(FileOperationException):
    """Exception raised when saving file fails"""
    
    def __init__(self, file_path: str, details: Optional[Dict[str, Any]] = None):
        message = f"Failed to save file: {file_path}"
        details = details or {}
        details["file_path"] = file_path
        super().__init__(message, details=details)


class ValidationException(BaseOCRException):
    """Exception raised when input validation fails"""
    
    def __init__(self, message: str = "Validation failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=400, details=details)


class InvalidFileTypeException(ValidationException):
    """Exception raised when file type is not supported"""
    
    def __init__(self, file_type: str, supported_types: list, details: Optional[Dict[str, Any]] = None):
        message = f"Unsupported file type: {file_type}. Supported types: {', '.join(supported_types)}"
        details = details or {}
        details["file_type"] = file_type
        details["supported_types"] = supported_types
        super().__init__(message, details=details)


class InvalidFilePathException(ValidationException):
    """Exception raised when file path is invalid"""
    
    def __init__(self, file_path: str, details: Optional[Dict[str, Any]] = None):
        message = f"Invalid file path: {file_path}"
        details = details or {}
        details["file_path"] = file_path
        super().__init__(message, details=details)


class ConfigurationException(BaseOCRException):
    """Exception raised when configuration is invalid"""
    
    def __init__(self, message: str = "Configuration error", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=500, details=details)


class MissingConfigurationException(ConfigurationException):
    """Exception raised when required configuration is missing"""
    
    def __init__(self, config_key: str, details: Optional[Dict[str, Any]] = None):
        message = f"Missing required configuration: {config_key}"
        details = details or {}
        details["config_key"] = config_key
        super().__init__(message, details=details)


class ExternalServiceException(BaseOCRException):
    """Exception raised when external service calls fail"""
    
    def __init__(self, service_name: str, message: str = "External service error", details: Optional[Dict[str, Any]] = None):
        message = f"{service_name}: {message}"
        details = details or {}
        details["service_name"] = service_name
        super().__init__(message, status_code=503, details=details)


class TesseractException(ExternalServiceException):
    """Exception raised when Tesseract OCR fails"""
    
    def __init__(self, details: Optional[Dict[str, Any]] = None):
        super().__init__("Tesseract OCR", "OCR processing failed", details)


class PyMuPDFException(ExternalServiceException):
    """Exception raised when PyMuPDF operations fail"""
    
    def __init__(self, operation: str, details: Optional[Dict[str, Any]] = None):
        message = f"PyMuPDF {operation} failed"
        details = details or {}
        details["operation"] = operation
        super().__init__("PyMuPDF", message, details)


class ArabicTextProcessingException(BaseOCRException):
    """Exception raised when Arabic text processing fails"""
    
    def __init__(self, details: Optional[Dict[str, Any]] = None):
        message = "Arabic text processing failed"
        super().__init__(message, status_code=422, details=details)


class MemoryException(BaseOCRException):
    """Exception raised when memory allocation fails"""
    
    def __init__(self, details: Optional[Dict[str, Any]] = None):
        message = "Insufficient memory for processing"
        super().__init__(message, status_code=507, details=details)


class TimeoutException(BaseOCRException):
    """Exception raised when processing times out"""
    
    def __init__(self, timeout_seconds: int, details: Optional[Dict[str, Any]] = None):
        message = f"Processing timed out after {timeout_seconds} seconds"
        details = details or {}
        details["timeout_seconds"] = timeout_seconds
        super().__init__(message, status_code=408, details=details)


# Status code constants for easy reference
STATUS_CODES = {
    "BAD_REQUEST": 400,
    "UNAUTHORIZED": 401,
    "FORBIDDEN": 403,
    "NOT_FOUND": 404,
    "REQUEST_TIMEOUT": 408,
    "UNPROCESSABLE_ENTITY": 422,
    "INTERNAL_SERVER_ERROR": 500,
    "INSUFFICIENT_STORAGE": 507,
    "SERVICE_UNAVAILABLE": 503
}


# ============================================================================
# AOP-BASED EXCEPTION HANDLING FUNCTIONALITY
# ============================================================================

class ExceptionSeverity(Enum):
    """Exception severity levels for logging and handling"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ExceptionContext(Enum):
    """Context where exceptions can occur"""
    VALIDATION = "validation"
    FILE_OPERATION = "file_operation"
    PDF_PROCESSING = "pdf_processing"
    TEXT_EXTRACTION = "text_extraction"
    TABLE_EXTRACTION = "table_extraction"
    OCR_PROCESSING = "ocr_processing"
    EXTERNAL_SERVICE = "external_service"
    CONFIGURATION = "configuration"
    GENERAL = "general"


class AOPExceptionHandler:
    """
    AOP-based exception handler that provides centralized exception handling
    with decorators and aspect-like functionality
    """
    
    def __init__(self, logger_name: str = "AOPExceptionHandler"):
        self.logger = get_standard_logger(logger_name)
        self.exception_mappings = self._build_exception_mappings()
        self.context_handlers = self._build_context_handlers()
    
    def _build_exception_mappings(self) -> Dict[Type[Exception], Dict[str, Any]]:
        """Build mappings for different exception types"""
        return {
            # PDF Processing Exceptions
            PDFProcessingException: {
                "severity": ExceptionSeverity.HIGH,
                "context": ExceptionContext.PDF_PROCESSING,
                "retryable": False,
                "default_status": 422
            },
            
            # Text Extraction Exceptions
            TextExtractionException: {
                "severity": ExceptionSeverity.HIGH,
                "context": ExceptionContext.TEXT_EXTRACTION,
                "retryable": True,
                "default_status": 422
            },
            
            # Table Extraction Exceptions
            TableExtractionException: {
                "severity": ExceptionSeverity.HIGH,
                "context": ExceptionContext.TABLE_EXTRACTION,
                "retryable": False,
                "default_status": 422
            },
            
            # File Operation Exceptions
            FileOperationException: {
                "severity": ExceptionSeverity.MEDIUM,
                "context": ExceptionContext.FILE_OPERATION,
                "retryable": True,
                "default_status": 500
            },
            
            # Validation Exceptions
            ValidationException: {
                "severity": ExceptionSeverity.LOW,
                "context": ExceptionContext.VALIDATION,
                "retryable": False,
                "default_status": 400
            },
            
            # Configuration Exceptions
            ConfigurationException: {
                "severity": ExceptionSeverity.CRITICAL,
                "context": ExceptionContext.CONFIGURATION,
                "retryable": False,
                "default_status": 500
            },
            
            # External Service Exceptions
            ExternalServiceException: {
                "severity": ExceptionSeverity.HIGH,
                "context": ExceptionContext.EXTERNAL_SERVICE,
                "retryable": True,
                "default_status": 503
            }
        }
    
    def _build_context_handlers(self) -> Dict[ExceptionContext, Callable]:
        """Build context-specific exception handlers"""
        return {
            ExceptionContext.VALIDATION: self._handle_validation_exception,
            ExceptionContext.FILE_OPERATION: self._handle_file_operation_exception,
            ExceptionContext.PDF_PROCESSING: self._handle_pdf_processing_exception,
            ExceptionContext.TEXT_EXTRACTION: self._handle_text_extraction_exception,
            ExceptionContext.TABLE_EXTRACTION: self._handle_table_extraction_exception,
            ExceptionContext.OCR_PROCESSING: self._handle_ocr_processing_exception,
            ExceptionContext.EXTERNAL_SERVICE: self._handle_external_service_exception,
            ExceptionContext.CONFIGURATION: self._handle_configuration_exception,
            ExceptionContext.GENERAL: self._handle_general_exception
        }
    
    def handle_exceptions(
        self,
        context: ExceptionContext = ExceptionContext.GENERAL,
        severity: ExceptionSeverity = ExceptionSeverity.MEDIUM,
        retryable: bool = False,
        max_retries: int = 0,
        custom_exceptions: Optional[list] = None,
        log_args: bool = True,
        log_result: bool = False,
        cleanup_func: Optional[Callable] = None
    ):
        """
        Main decorator for exception handling with AOP functionality
        
        Args:
            context: The context where the exception can occur
            severity: Severity level of potential exceptions
            retryable: Whether the operation can be retried
            max_retries: Maximum number of retry attempts
            custom_exceptions: List of custom exceptions to handle
            log_args: Whether to log function arguments
            log_result: Whether to log function result
            cleanup_func: Function to call for cleanup on exception
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Get function metadata
                func_name = func.__name__
                module_name = func.__module__
                
                # Log function entry
                if log_args:
                    self.logger.info(
                        f"Entering {module_name}.{func_name} with args: {args}, kwargs: {kwargs}"
                    )
                else:
                    self.logger.info(f"Entering {module_name}.{func_name}")
                
                retry_count = 0
                last_exception = None
                
                while retry_count <= max_retries:
                    try:
                        # Execute the function
                        result = await func(*args, **kwargs)
                        
                        # Log successful execution
                        if log_result:
                            self.logger.info(
                                f"Successfully executed {module_name}.{func_name}, result: {result}"
                            )
                        else:
                            self.logger.info(f"Successfully executed {module_name}.{func_name}")
                        
                        return result
                        
                    except Exception as e:
                        last_exception = e
                        retry_count += 1
                        
                        # Handle the exception
                        handled_exception = self._handle_exception(
                            e, context, severity, func_name, module_name, retry_count, max_retries
                        )
                        
                        # Check if we should retry
                        if retryable and retry_count <= max_retries and self._is_retryable_exception(e):
                            self.logger.warning(
                                f"Retrying {module_name}.{func_name} (attempt {retry_count}/{max_retries})"
                            )
                            continue
                        
                        # Perform cleanup if specified
                        if cleanup_func:
                            try:
                                cleanup_func(*args, **kwargs)
                            except Exception as cleanup_error:
                                self.logger.error(f"Cleanup failed: {cleanup_error}")
                        
                        # Re-raise the handled exception
                        raise handled_exception
                
                # If we get here, all retries failed
                self.logger.error(
                    f"All retry attempts failed for {module_name}.{func_name}"
                )
                raise last_exception
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Get function metadata
                func_name = func.__name__
                module_name = func.__module__
                
                # Log function entry
                if log_args:
                    self.logger.info(
                        f"Entering {module_name}.{func_name} with args: {args}, kwargs: {kwargs}"
                    )
                else:
                    self.logger.info(f"Entering {module_name}.{func_name}")
                
                retry_count = 0
                last_exception = None
                
                while retry_count <= max_retries:
                    try:
                        # Execute the function
                        result = func(*args, **kwargs)
                        
                        # Log successful execution
                        if log_result:
                            self.logger.info(
                                f"Successfully executed {module_name}.{func_name}, result: {result}"
                            )
                        else:
                            self.logger.info(f"Successfully executed {module_name}.{func_name}")
                        
                        return result
                        
                    except Exception as e:
                        last_exception = e
                        retry_count += 1
                        
                        # Handle the exception
                        handled_exception = self._handle_exception(
                            e, context, severity, func_name, module_name, retry_count, max_retries
                        )
                        
                        # Check if we should retry
                        if retryable and retry_count <= max_retries and self._is_retryable_exception(e):
                            self.logger.warning(
                                f"Retrying {module_name}.{func_name} (attempt {retry_count}/{max_retries})"
                            )
                            continue
                        
                        # Perform cleanup if specified
                        if cleanup_func:
                            try:
                                cleanup_func(*args, **kwargs)
                            except Exception as cleanup_error:
                                self.logger.error(f"Cleanup failed: {cleanup_error}")
                        
                        # Re-raise the handled exception
                        raise handled_exception
                
                # If we get here, all retries failed
                self.logger.error(
                    f"All retry attempts failed for {module_name}.{func_name}"
                )
                raise last_exception
            
            # Check if the function is async
            if inspect.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        return decorator
    
    def _handle_exception(
        self,
        exception: Exception,
        context: ExceptionContext,
        severity: ExceptionSeverity,
        func_name: str,
        module_name: str,
        retry_count: int,
        max_retries: int
    ) -> Exception:
        """Handle an exception based on its type and context"""
        
        # Get exception mapping
        exception_mapping = self.exception_mappings.get(type(exception), {})
        exception_context = exception_mapping.get("context", context)
        exception_severity = exception_mapping.get("severity", severity)
        
        # Log the exception
        self._log_exception(
            exception, exception_context, exception_severity,
            func_name, module_name, retry_count, max_retries
        )
        
        # Get context-specific handler
        handler = self.context_handlers.get(exception_context, self._handle_general_exception)
        
        # Handle the exception
        return handler(exception, func_name, module_name)
    
    def _log_exception(
        self,
        exception: Exception,
        context: ExceptionContext,
        severity: ExceptionSeverity,
        func_name: str,
        module_name: str,
        retry_count: int,
        max_retries: int
    ):
        """Log exception with appropriate level based on severity"""
        
        log_message = (
            f"Exception in {module_name}.{func_name} "
            f"(Context: {context.value}, Severity: {severity.value}, "
            f"Retry: {retry_count}/{max_retries}): {str(exception)}"
        )
        
        if severity == ExceptionSeverity.CRITICAL:
            self.logger.critical(log_message)
            self.logger.exception(f"Critical exception traceback:")
        elif severity == ExceptionSeverity.HIGH:
            self.logger.error(log_message)
            self.logger.exception(f"High severity exception traceback:")
        elif severity == ExceptionSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def _is_retryable_exception(self, exception: Exception) -> bool:
        """Check if an exception is retryable"""
        exception_mapping = self.exception_mappings.get(type(exception), {})
        return exception_mapping.get("retryable", False)
    
    # Context-specific exception handlers
    def _handle_validation_exception(
        self, exception: Exception, func_name: str, module_name: str
    ) -> Exception:
        """Handle validation exceptions"""
        if isinstance(exception, BaseOCRException):
            return exception
        return ValidationException(
            f"Validation failed in {module_name}.{func_name}: {str(exception)}",
            details={"function": func_name, "module": module_name}
        )
    
    def _handle_file_operation_exception(
        self, exception: Exception, func_name: str, module_name: str
    ) -> Exception:
        """Handle file operation exceptions"""
        if isinstance(exception, BaseOCRException):
            return exception
        return FileOperationException(
            f"File operation failed in {module_name}.{func_name}: {str(exception)}",
            details={"function": func_name, "module": module_name}
        )
    
    def _handle_pdf_processing_exception(
        self, exception: Exception, func_name: str, module_name: str
    ) -> Exception:
        """Handle PDF processing exceptions"""
        if isinstance(exception, BaseOCRException):
            return exception
        return PDFProcessingException(
            f"PDF processing failed in {module_name}.{func_name}: {str(exception)}",
            details={"function": func_name, "module": module_name}
        )
    
    def _handle_text_extraction_exception(
        self, exception: Exception, func_name: str, module_name: str
    ) -> Exception:
        """Handle text extraction exceptions"""
        if isinstance(exception, BaseOCRException):
            return exception
        return TextExtractionException(
            f"Text extraction failed in {module_name}.{func_name}: {str(exception)}",
            details={"function": func_name, "module": module_name}
        )
    
    def _handle_table_extraction_exception(
        self, exception: Exception, func_name: str, module_name: str
    ) -> Exception:
        """Handle table extraction exceptions"""
        if isinstance(exception, BaseOCRException):
            return exception
        return TableExtractionException(
            f"Table extraction failed in {module_name}.{func_name}: {str(exception)}",
            details={"function": func_name, "module": module_name}
        )
    
    def _handle_ocr_processing_exception(
        self, exception: Exception, func_name: str, module_name: str
    ) -> Exception:
        """Handle OCR processing exceptions"""
        if isinstance(exception, BaseOCRException):
            return exception
        return TextExtractionException(
            f"OCR processing failed in {module_name}.{func_name}: {str(exception)}",
            details={"function": func_name, "module": module_name}
        )
    
    def _handle_external_service_exception(
        self, exception: Exception, func_name: str, module_name: str
    ) -> Exception:
        """Handle external service exceptions"""
        if isinstance(exception, BaseOCRException):
            return exception
        return ExternalServiceException(
            "External Service",
            f"External service call failed in {module_name}.{func_name}: {str(exception)}",
            details={"function": func_name, "module": module_name}
        )
    
    def _handle_configuration_exception(
        self, exception: Exception, func_name: str, module_name: str
    ) -> Exception:
        """Handle configuration exceptions"""
        if isinstance(exception, BaseOCRException):
            return exception
        return ConfigurationException(
            f"Configuration error in {module_name}.{func_name}: {str(exception)}",
            details={"function": func_name, "module": module_name}
        )
    
    def _handle_general_exception(
        self, exception: Exception, func_name: str, module_name: str
    ) -> Exception:
        """Handle general exceptions"""
        if isinstance(exception, BaseOCRException):
            return exception
        return BaseOCRException(
            f"General error in {module_name}.{func_name}: {str(exception)}",
            details={"function": func_name, "module": module_name}
        )


# Global AOP exception handler instance
aop_handler = AOPExceptionHandler()


# Convenience decorators for different contexts
def handle_validation_exceptions(
    severity: ExceptionSeverity = ExceptionSeverity.LOW,
    log_args: bool = True,
    log_result: bool = False,
    cleanup_func: Optional[Callable] = None
):
    """Decorator for validation-related functions"""
    return aop_handler.handle_exceptions(
        context=ExceptionContext.VALIDATION,
        severity=severity,
        retryable=False,
        log_args=log_args,
        log_result=log_result,
        cleanup_func=cleanup_func
    )


def handle_file_operations(
    severity: ExceptionSeverity = ExceptionSeverity.MEDIUM,
    retryable: bool = True,
    max_retries: int = 3,
    log_args: bool = True,
    log_result: bool = False,
    cleanup_func: Optional[Callable] = None
):
    """Decorator for file operation functions"""
    return aop_handler.handle_exceptions(
        context=ExceptionContext.FILE_OPERATION,
        severity=severity,
        retryable=retryable,
        max_retries=max_retries,
        log_args=log_args,
        log_result=log_result,
        cleanup_func=cleanup_func
    )


def handle_pdf_processing(
    severity: ExceptionSeverity = ExceptionSeverity.HIGH,
    retryable: bool = False,
    log_args: bool = True,
    log_result: bool = False,
    cleanup_func: Optional[Callable] = None
):
    """Decorator for PDF processing functions"""
    return aop_handler.handle_exceptions(
        context=ExceptionContext.PDF_PROCESSING,
        severity=severity,
        retryable=retryable,
        log_args=log_args,
        log_result=log_result,
        cleanup_func=cleanup_func
    )


def handle_text_extraction(
    severity: ExceptionSeverity = ExceptionSeverity.HIGH,
    retryable: bool = True,
    max_retries: int = 2,
    log_args: bool = True,
    log_result: bool = False,
    cleanup_func: Optional[Callable] = None
):
    """Decorator for text extraction functions"""
    return aop_handler.handle_exceptions(
        context=ExceptionContext.TEXT_EXTRACTION,
        severity=severity,
        retryable=retryable,
        max_retries=max_retries,
        log_args=log_args,
        log_result=log_result,
        cleanup_func=cleanup_func
    )


def handle_table_extraction(
    severity: ExceptionSeverity = ExceptionSeverity.HIGH,
    retryable: bool = False,
    log_args: bool = True,
    log_result: bool = False,
    cleanup_func: Optional[Callable] = None
):
    """Decorator for table extraction functions"""
    return aop_handler.handle_exceptions(
        context=ExceptionContext.TABLE_EXTRACTION,
        severity=severity,
        retryable=retryable,
        log_args=log_args,
        log_result=log_result,
        cleanup_func=cleanup_func
    )


def handle_ocr_processing(
    severity: ExceptionSeverity = ExceptionSeverity.HIGH,
    retryable: bool = True,
    max_retries: int = 2,
    log_args: bool = True,
    log_result: bool = False,
    cleanup_func: Optional[Callable] = None
):
    """Decorator for OCR processing functions"""
    return aop_handler.handle_exceptions(
        context=ExceptionContext.OCR_PROCESSING,
        severity=severity,
        retryable=retryable,
        max_retries=max_retries,
        log_args=log_args,
        log_result=log_result,
        cleanup_func=cleanup_func
    )


def handle_external_services(
    severity: ExceptionSeverity = ExceptionSeverity.HIGH,
    retryable: bool = True,
    max_retries: int = 3,
    log_args: bool = True,
    log_result: bool = False,
    cleanup_func: Optional[Callable] = None
):
    """Decorator for external service calls"""
    return aop_handler.handle_exceptions(
        context=ExceptionContext.EXTERNAL_SERVICE,
        severity=severity,
        retryable=retryable,
        max_retries=max_retries,
        log_args=log_args,
        log_result=log_result,
        cleanup_func=cleanup_func
    )


def handle_configuration(
    severity: ExceptionSeverity = ExceptionSeverity.CRITICAL,
    retryable: bool = False,
    log_args: bool = True,
    log_result: bool = False,
    cleanup_func: Optional[Callable] = None
):
    """Decorator for configuration-related functions"""
    return aop_handler.handle_exceptions(
        context=ExceptionContext.CONFIGURATION,
        severity=severity,
        retryable=retryable,
        log_args=log_args,
        log_result=log_result,
        cleanup_func=cleanup_func
    )


def handle_general_operations(
    severity: ExceptionSeverity = ExceptionSeverity.MEDIUM,
    retryable: bool = False,
    log_args: bool = True,
    log_result: bool = False,
    cleanup_func: Optional[Callable] = None
):
    """Decorator for general operations"""
    return aop_handler.handle_exceptions(
        context=ExceptionContext.GENERAL,
        severity=severity,
        retryable=retryable,
        log_args=log_args,
        log_result=log_result,
        cleanup_func=cleanup_func
    )


# Utility function for cleanup operations
def cleanup_temp_files(*args, **kwargs):
    """Utility function for cleaning up temporary files"""
    import os
    import glob
    
    # Look for temp files in common locations
    temp_patterns = [
        "temp_uploads/*",
        "temp_files/*",
        "*.tmp",
        "*.temp"
    ]
    
    for pattern in temp_patterns:
        try:
            files = glob.glob(pattern)
            for file_path in files:
                if os.path.exists(file_path) and os.path.isfile(file_path):
                    os.remove(file_path)
        except Exception as e:
            logging.warning(f"Failed to cleanup files matching {pattern}: {e}")


# Aspect for method entry/exit logging
def log_method_entry_exit(func: Callable) -> Callable:
    """Aspect for logging method entry and exit"""
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        logger = get_standard_logger("MethodAspect")
        func_name = f"{func.__module__}.{func.__name__}"
        
        logger.info(f"Entering method: {func_name}")
        try:
            result = await func(*args, **kwargs)
            logger.info(f"Exiting method: {func_name} successfully")
            return result
        except Exception as e:
            logger.error(f"Exiting method: {func_name} with exception: {e}")
            raise
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        logger = get_standard_logger("MethodAspect")
        func_name = f"{func.__module__}.{func.__name__}"
        
        logger.info(f"Entering method: {func_name}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"Exiting method: {func_name} successfully")
            return result
        except Exception as e:
            logger.error(f"Exiting method: {func_name} with exception: {e}")
            raise
    
    # Check if the function is async
    if inspect.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


# Aspect for performance monitoring
def monitor_performance(func: Callable) -> Callable:
    """Aspect for monitoring method performance"""
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        import time
        logger = get_standard_logger("PerformanceAspect")
        func_name = f"{func.__module__}.{func.__name__}"
        
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"Method {func_name} executed in {execution_time:.4f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Method {func_name} failed after {execution_time:.4f} seconds: {e}")
            raise
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        import time
        logger = get_standard_logger("PerformanceAspect")
        func_name = f"{func.__module__}.{func.__name__}"
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"Method {func_name} executed in {execution_time:.4f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Method {func_name} failed after {execution_time:.4f} seconds: {e}")
            raise
    
    # Check if the function is async
    if inspect.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper
