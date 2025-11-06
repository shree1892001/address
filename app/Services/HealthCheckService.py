from datetime import datetime
from app.Logger.ocr_logger import get_standard_logger
from app.Exceptions.custom_exceptions import (
    handle_general_operations, log_method_entry_exit, ExceptionSeverity
)


class HealthCheckService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(HealthCheckService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.logger = get_standard_logger("HealthCheckService")
            self._initialized = True

    @log_method_entry_exit
    @handle_general_operations(severity=ExceptionSeverity.LOW)
    def get_health_status(self):
        """Get health check status"""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "OCR Document Processing Service"
        }

