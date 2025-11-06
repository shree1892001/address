import csv
import json
from app.Logger.ocr_logger import get_standard_logger
from app.Exceptions.custom_exceptions import FileSaveException
from app.Exceptions.custom_exceptions import (
    handle_file_operations, log_method_entry_exit, ExceptionSeverity
)


class FileSaveService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FileSaveService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.logger = get_standard_logger("FileSaveService")
            self._initialized = True

    @log_method_entry_exit
    @handle_file_operations(severity=ExceptionSeverity.MEDIUM)
    def save_results(self, table, json_path, csv_path):
        """Save table data to JSON and CSV files"""
        try:
            # Save to JSON
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(table, jf, ensure_ascii=False, indent=2)
            
            # Save to CSV
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as cf:
                writer = csv.writer(cf)
                writer.writerows(table)
            
            self.logger.info(f"Saved {len(table)} rows to {csv_path} and {json_path}")
            return True
            
        except Exception as e:
            raise FileSaveException(csv_path, details={"error": str(e), "table_rows": len(table)})

    @log_method_entry_exit
    @handle_file_operations(severity=ExceptionSeverity.MEDIUM)
    def save_json(self, data, json_path):
        """Save data to JSON file"""
        try:
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(data, jf, ensure_ascii=False, indent=2)
            self.logger.info(f"Saved data to JSON: {json_path}")
            return True
        except Exception as e:
            raise FileSaveException(json_path, details={"error": str(e)})

    @log_method_entry_exit
    @handle_file_operations(severity=ExceptionSeverity.MEDIUM)
    def save_csv(self, table, csv_path):
        """Save table data to CSV file"""
        try:
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as cf:
                writer = csv.writer(cf)
                writer.writerows(table)
            self.logger.info(f"Saved {len(table)} rows to CSV: {csv_path}")
            return True
        except Exception as e:
            raise FileSaveException(csv_path, details={"error": str(e), "table_rows": len(table)})
