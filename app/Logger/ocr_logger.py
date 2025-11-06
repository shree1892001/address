import logging
import os
import threading
from datetime import datetime
from logging.handlers import RotatingFileHandler

class OCRLogger:
    """
    Dedicated logger configuration for OCR Service
    Handles file logging with rotation and console output
    """
    
    _lock = threading.Lock()
    _loggers = {}
    
    def __init__(self, logger_name="OCRService"):
        self.logger_name = logger_name
        with self._lock:
            if logger_name not in self._loggers:
                self.logger = logging.getLogger(logger_name)
                self.setup_logger()
                self._loggers[logger_name] = self.logger
            else:
                self.logger = self._loggers[logger_name]
        

    
    def setup_logger(self):
        """
        Setup logger with file and console handlers
        """
        # Create logs directory if it doesn't exist
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        # Create log file path with timestamp
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = os.path.join(logs_dir, f"ocr_service_{timestamp}.log")
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Set logger level
        self.logger.setLevel(logging.INFO)
        
        # Create formatters
        file_formatter = logging.Formatter(
            "[%(asctime)s] %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        console_formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(message)s",
            datefmt="%H:%M:%S"
        )
        
        # Use a custom file handler that handles Windows file locking issues
        try:
            # Try to use RotatingFileHandler with error handling
            file_handler = SafeRotatingFileHandler(
                log_file,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
        except Exception:
            # Fallback to simple file handler if rotation fails
            file_handler = logging.FileHandler(
                log_file,
                mode='a',
                encoding='utf-8'
            )
        
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Prevent propagation to root logger
        self.logger.propagate = False


class SafeRotatingFileHandler(RotatingFileHandler):
    """
    A RotatingFileHandler that safely handles file rotation on Windows
    """
    
    def doRollover(self):
        """
        Override doRollover to handle Windows file locking issues
        """
        try:
            super().doRollover()
        except PermissionError:
            # If rotation fails due to file being in use, just continue
            # The log will continue to grow but won't crash the application
            pass
        except Exception:
            # Handle any other rotation errors gracefully
            pass
    
    def info(self, message):
        """Log info message"""
        try:
            self.logger.info(message)
        except Exception:
            # If logging fails, just print to console
            print(f"[INFO] {message}")
    
    def error(self, message):
        """Log error message"""
        try:
            self.logger.error(message)
        except Exception:
            # If logging fails, just print to console
            print(f"[ERROR] {message}")
    
    def warning(self, message):
        """Log warning message"""
        try:
            self.logger.warning(message)
        except Exception:
            # If logging fails, just print to console
            print(f"[WARNING] {message}")
    
    def debug(self, message):
        """Log debug message"""
        try:
            self.logger.debug(message)
        except Exception:
            # If logging fails, just print to console
            print(f"[DEBUG] {message}")
    
    def critical(self, message):
        """Log critical message"""
        try:
            self.logger.critical(message)
        except Exception:
            # If logging fails, just print to console
            print(f"[CRITICAL] {message}")
    
    def exception(self, message):
        """Log exception with traceback"""
        try:
            self.logger.exception(message)
        except Exception:
            # If logging fails, just print to console
            print(f"[EXCEPTION] {message}")

# Create a default OCR logger instance
ocr_logger = OCRLogger("OCRService")

def get_ocr_logger(name="OCRService"):
    """
    Get OCR logger instance
    """
    return OCRLogger(name)

def get_standard_logger(name="OCRService"):
    """
    Get standard Python logger instance for compatibility with custom exceptions
    """
    logger_instance = OCRLogger(name)
    return logger_instance.logger 