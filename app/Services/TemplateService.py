from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.Logger.ocr_logger import get_standard_logger
from app.Exceptions.custom_exceptions import (
    handle_general_operations, log_method_entry_exit, ExceptionSeverity
)


class TemplateService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TemplateService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.logger = get_standard_logger("TemplateService")
            self.templates = Jinja2Templates(directory="Templates")
            self._initialized = True

    @log_method_entry_exit
    @handle_general_operations(severity=ExceptionSeverity.LOW)
    def render_upload_interface(self, request: Request):
        """Render the upload interface template"""
        return self.templates.TemplateResponse("upload.html", {"request": request})

