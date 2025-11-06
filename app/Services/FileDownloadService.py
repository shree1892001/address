import os
import csv
from fastapi.responses import FileResponse
from app.Logger.ocr_logger import get_standard_logger
from app.Exceptions.custom_exceptions import FileNotFoundException
from app.Exceptions.custom_exceptions import (
    handle_file_operations, log_method_entry_exit, ExceptionSeverity
)


class FileDownloadService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FileDownloadService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.logger = get_standard_logger("FileDownloadService")
            self._initialized = True

    @log_method_entry_exit
    @handle_file_operations(severity=ExceptionSeverity.MEDIUM)
    def download_csv_file(self, file_id, temp_directory="temp_uploads"):
        """Download extracted table as CSV file"""
        csv_path = f"{temp_directory}/{file_id}.csv"
        
        if not os.path.exists(csv_path):
            self.logger.warning(f"CSV file not found: {csv_path}")
            raise FileNotFoundException(csv_path)
        
        self.logger.info(f"Downloading CSV file: {csv_path}")
        return FileResponse(
            path=csv_path,
            filename=f"extracted_table_{file_id}.csv",
            media_type="text/csv"
        )

    @log_method_entry_exit
    @handle_file_operations(severity=ExceptionSeverity.LOW)
    def create_test_csv_file(self, temp_directory="temp_uploads"):
        """Create a test CSV file for download testing"""
        test_file_path = f"{temp_directory}/test.csv"
        
        # Create a test CSV file if it doesn't exist
        if not os.path.exists(test_file_path):
            os.makedirs(temp_directory, exist_ok=True)
            test_data = [
                ["Column1", "Column2", "Column3"],
                ["Data1", "Data2", "Data3"],
                ["Data4", "Data5", "Data6"]
            ]
            
            with open(test_file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(test_data)
            
            self.logger.info(f"Created test CSV file: {test_file_path}")
        
        return test_file_path

    @log_method_entry_exit
    @handle_file_operations(severity=ExceptionSeverity.LOW)
    def download_test_file(self):
        """Download test CSV file"""
        test_file_path = self.create_test_csv_file()
        
        self.logger.info(f"Test download requested: {test_file_path}")
        return FileResponse(
            path=test_file_path,
            filename="test_download.csv",
            media_type="text/csv"
        )

