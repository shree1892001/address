import os
import tempfile
import shutil
import time
import concurrent.futures
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
import fitz  # PyMuPDF
from PIL import Image, ImageEnhance, ImageFilter
import pandas as pd
import numpy as np
import threading
import asyncio

from app.Logger.ocr_logger import get_standard_logger
from app.Exceptions.custom_exceptions import (
    NoTableFoundException, FileProcessingError, ValidationError
)
from app.Exceptions.custom_exceptions import (
    handle_general_operations, log_method_entry_exit, monitor_performance, ExceptionSeverity
)
from app.Services.FileValidationService import FileValidationService
from app.Services.FileUploadService import FileUploadService
from app.Services.OCRService import OCRService
from app.Services.TableProcessingService import TableProcessingService
from app.Services.FileSaveService import FileSaveService
from app.Services.DocumentProcessor import DocumentProcessor

# Configure logging
logger = get_standard_logger("TableExtractionServiceV2")

class TableExtractionServiceV2:
    """
    Enhanced TableExtractionService that supports multiple file types (PDF, PNG, JPEG, DOCX)
    while preserving the original PDF processing logic.
    
    Features:
    - Multi-threaded processing for better performance
    - Automatic cleanup of temporary files
    - Comprehensive error handling and logging
    - Support for various document formats
    - OCR for scanned documents and images
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(TableExtractionServiceV2, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized') or not self._initialized:
            self.logger = logger
            self.file_validation_service = FileValidationService()
            self.file_upload_service = FileUploadService()
            self.ocr_service = OCRService()
            self.table_processing_service = TableProcessingService()
            self.file_save_service = FileSaveService()
            self.document_processor = DocumentProcessor()
            
            # Thread pool for parallel processing
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
            
            # Track resources
            self.temp_dirs = []
            self._initialized = True
            
            self.logger.info("TableExtractionServiceV2 initialized")

    def __del__(self):
        """Clean up resources when the object is destroyed"""
        self.cleanup()
        try:
            self.executor.shutdown(wait=False)
        except:
            pass

    @log_method_entry_exit
    @monitor_performance
    @handle_general_operations(severity=ExceptionSeverity.HIGH)
    async def extract_table(self, file) -> Dict[str, Any]:
        """
        Extract table from uploaded document (PDF, PNG, JPEG, DOCX).
        
        Args:
            file: Uploaded file object (FastAPI UploadFile or similar)
            
        Returns:
            Dictionary containing extraction results with the following structure:
            {
                'status': 'success'|'partial'|'error',
                'message': str,
                'data': List[Dict] | None,
                'excel_file': str | None,
                'processing_time': float,
                'page_count': int,
                'table_count': int
            }
        """
        start_time = time.time()
        temp_files = []
        temp_dir = None
        response = None  # Initialize response variable
        
        try:
            self.logger.info(f"Starting table extraction for file: {getattr(file, 'filename', 'unknown')}")
            
            # 1. Validate and save uploaded file
            try:
                file_paths = await self.file_upload_service.save_uploaded_file(file)
                temp_files.append(file_paths["pdf_path"])
                self.logger.debug(f"Saved uploaded file to: {file_paths['pdf_path']}")
            except Exception as e:
                raise FileProcessingError(f"Failed to save uploaded file: {str(e)}")
            
            # 2. Process the document (convert to PDF if needed)
            try:
                pdf_path = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.document_processor.process_file(file_paths["pdf_path"])
                )
                temp_files.append(pdf_path)
                self.logger.info(f"Processed document to PDF: {pdf_path}")
            except Exception as e:
                raise FileProcessingError(f"Document processing failed: {str(e)}")
            
            # 3. Extract table data from PDF
            table_data = await self._extract_table_from_pdf(pdf_path)
            
            if not table_data:
                self.logger.warning(f"No table data found in document: {file.filename}")
                raise NoTableFoundException(pdf_path)
            
            # 4. Process and clean the extracted data
            cleaned_data = self._clean_table_data(table_data)
            
            # 5. Convert to DataFrame for better CSV/JSON handling
            df = self._create_dataframe(cleaned_data)
            
            # 6. Save results
            output_dir = os.path.dirname(pdf_path)
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            json_path = os.path.join(output_dir, f"{base_name}.json")
            csv_path = os.path.join(output_dir, f"{base_name}.csv")
            
            # Save as CSV using pandas
            df.to_csv(csv_path, index=False)
            
            # Save as JSON with proper formatting
            df.to_json(json_path, orient='records', indent=2)
            
            # Add to temp files for cleanup
            temp_files.extend([json_path, csv_path])
            
            result_paths = {
                'json_path': json_path,
                'csv_path': csv_path
            }
            
            # 7. Prepare response
            processing_time = time.time() - start_time
            response = {
                'status': 'success',
                'message': 'Table extracted successfully',
                'data': cleaned_data,
                'excel_file': result_paths.get('excel_path'),
                'processing_time': round(processing_time, 2),
                'page_count': self._get_page_count(pdf_path),
                'table_count': len(cleaned_data) if isinstance(cleaned_data, list) else 1
            }
            
            self.logger.info(
                f"Successfully processed {file.filename} in {processing_time:.2f}s. "
                f"Extracted {response['table_count']} tables from {response['page_count']} pages."
            )
            
            return response
                
        except NoTableFoundException as e:
            # Handle case where no tables are found but we have text content
            if hasattr(e, 'text_content') and e.text_content:
                return self._create_text_response(file, e.text_content)
            raise
                
        except Exception as e:
            self.logger.error(f"Error processing file {getattr(file, 'filename', 'unknown')}: {str(e)}", exc_info=True)
            raise
            
        finally:
            try:
                # Ensure temp_files is a list before cleanup
                temp_files = temp_files or []
                if not isinstance(temp_files, (list, tuple, set)):
                    temp_files = [temp_files] if temp_files else []
                
                # Clean up temporary files
                if temp_files:  # Only clean up if there are files to clean
                    await self._cleanup_temp_files(temp_files)
                
                # Also clean up any temporary directories
                if hasattr(self, 'temp_dirs') and self.temp_dirs:
                    await self._cleanup_temp_files(self.temp_dirs)
                    self.temp_dirs.clear()
                    
            except Exception as cleanup_error:
                self.logger.error(f"Error during cleanup: {str(cleanup_error)}", exc_info=True)
                # Don't re-raise cleanup errors to avoid masking the original error
    
    def _get_page_count(self, pdf_path: str) -> int:
        """Get the number of pages in a PDF file
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            int: Number of pages in the PDF, or 0 if there was an error
        """
        try:
            with fitz.open(pdf_path) as doc:
                return len(doc)
        except Exception as e:
            self.logger.warning(f"Error getting page count for {pdf_path}: {str(e)}")
            return 0
            
    async def _cleanup_temp_files(self, temp_files):
        """Clean up temporary files asynchronously
        
        Args:
            temp_files: List of file paths to clean up. Can be None, a string, or any iterable.
        """
        if not temp_files:
            self.logger.debug("No temporary files to clean up")
            return
            
        # Convert single string to list
        if isinstance(temp_files, str):
            temp_files = [temp_files]
            
        # Ensure we have an iterable
        if not isinstance(temp_files, (list, tuple, set)):
            self.logger.warning(f"Invalid temp_files type: {type(temp_files).__name__}")
            return
            
        # Filter out any non-string items and empty strings
        temp_files = [f for f in temp_files if isinstance(f, str) and f.strip()]
        
        if not temp_files:
            self.logger.debug("No valid file paths to clean up")
            return
            
        self.logger.debug(f"Cleaning up {len(temp_files)} temporary files")
        
        def _delete_file_sync(file_path: str) -> None:
            """Synchronously delete a single file or directory."""
            if not file_path or not isinstance(file_path, str):
                return
                
            try:
                file_path = os.path.abspath(file_path)
                
                # Safety check to prevent accidental deletion of important directories
                if any(part in file_path.lower() for part in ['system32', 'windows', 'program files']):
                    self.logger.warning(f"Prevented deletion of system file: {file_path}")
                    return
                    
                if os.path.exists(file_path):
                    if os.path.isdir(file_path):
                        shutil.rmtree(file_path, ignore_errors=True)
                        self.logger.debug(f"Deleted temporary directory: {file_path}")
                    else:
                        os.unlink(file_path)
                        self.logger.debug(f"Deleted temporary file: {file_path}")
            except PermissionError as pe:
                self.logger.warning(f"Permission denied deleting {file_path}: {str(pe)}")
            except FileNotFoundError:
                pass  # File already deleted, nothing to do
            except Exception as e:
                self.logger.warning(f"Error deleting {file_path}: {str(e)}", exc_info=True)
        
        try:
            # Use ThreadPoolExecutor for parallel I/O bound operations
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(temp_files) or 1)) as executor:
                loop = asyncio.get_event_loop()
                # Process files in chunks to avoid too many open files
                chunk_size = 50
                for i in range(0, len(temp_files), chunk_size):
                    chunk = temp_files[i:i + chunk_size]
                    await loop.run_in_executor(
                        None,  # Use default executor
                        lambda: list(executor.map(_delete_file_sync, chunk))
                    )
        except Exception as e:
            self.logger.error(f"Error during temp file cleanup: {str(e)}", exc_info=True)
            
        Returns:
            Dictionary containing extraction results with the following structure:
            {
                'status': 'success'|'partial'|'error',
                'message': str,
                'data': List[Dict] | None,
                'excel_file': str | None,
                'processing_time': float,
                'page_count': int,
                'table_count': int
            }
        """
        start_time = time.time()
        # Initialize temp_files as a list to ensure it's always iterable
        temp_files = []
        temp_dir = None
        response = None  # Initialize response variable
    
    try:
        self.logger.info(f"Starting table extraction for file: {getattr(file, 'filename', 'unknown')}")
        
        # 1. Validate and save uploaded file
        try:
            file_paths = await self.file_upload_service.save_uploaded_file(file)
            temp_files.append(file_paths["pdf_path"])
            self.logger.debug(f"Saved uploaded file to: {file_paths['pdf_path']}")
        except Exception as e:
            raise FileProcessingError(f"Failed to save uploaded file: {str(e)}")
        
        # 2. Process the document (convert to PDF if needed)
        try:
            pdf_path = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.document_processor.process_file(file_paths["pdf_path"])
            )
            temp_files.append(pdf_path)
            self.logger.info(f"Processed document to PDF: {pdf_path}")
        except Exception as e:
            raise FileProcessingError(f"Document processing failed: {str(e)}")
        doc = None
        try:
            # Open PDF
            doc = self.ocr_service.open_pdf(pdf_path)
            
            # Process each page
            all_tables = []
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Extract text spans from page
                spans = self.ocr_service.extract_page_spans(page)
                if not spans:
                    continue
                    
                # Cluster spans into rows
                rows = self.table_processing_service.cluster_rows(spans)
                if not rows:
                    continue
                    
                # Process rows into table data
                table_data = self._process_rows_into_table(rows)
                if table_data:
                    all_tables.extend(table_data)
            
            return all_tables if all_tables else None
            
        except Exception as e:
            self.logger.error(f"Error extracting table from PDF: {e}", exc_info=True)
            raise
            
        finally:
            if doc:
                doc.close()
    
    def _process_rows_into_table(self, rows: List[List[Dict]]) -> List[List]:
        """Process clustered rows into a structured table"""
        if not rows:
            return []
            
        # Convert spans to text rows
        text_rows = []
        for row in rows:
            # Sort spans left to right
            row.sort(key=lambda s: s.get('x0', 0))
            # Join text from spans
            row_text = [span.get('text', '').strip() for span in row]
            text_rows.append(row_text)
            
        return text_rows
        
    def _is_likely_table_row(self, row: List[Dict]) -> bool:
        """Determine if a row of spans is likely part of a table"""
        if not row:
            return False
            
        # Check if row has multiple columns
        if len(row) < 2:
            return False
            
        # Check for numeric content in at least one cell
        has_numeric = any(any(c.isdigit() for c in span.get('text', '')) for span in row)
        if not has_numeric:
            return False
            
        # Check for reasonable horizontal spacing
        x_positions = [span.get('x0', 0) for span in row]
        avg_gap = sum(x_positions[i+1] - x_positions[i] for i in range(len(x_positions)-1)) / (len(x_positions) - 1)
        
        return avg_gap > 10  # Minimum gap between columns (pixels)
    
    def _clean_table_data(self, table_data: List[List]) -> List[List]:
        """Clean and process the extracted table data"""
        if not table_data:
            return []
            
        # Clean header row
        if table_data:
            header_row = table_data[0]
            cleaned_header = []
            
            for cell in header_row:
                cell_str = str(cell).strip()
                words = cell_str.split()
                if len(words) > 1:
                    # Find where header ends and data begins
                    header_part = []
                    for i, word in enumerate(words):
                        if any(c.isdigit() for c in word) or word.startswith(('$', '€', '£', '₹')):
                            break
                        header_part.append(word)
                    cleaned_header.append(' '.join(header_part) if header_part else words[0])
                else:
                    cleaned_header.append(cell_str)
            
            table_data[0] = cleaned_header
            
            # Clean first data row if it contains header remnants
            if len(table_data) > 1:
                first_data_row = table_data[1]
                cleaned_first_row = []
                
                for i, cell in enumerate(first_data_row):
                    cell_str = str(cell).strip()
                    if i < len(header_row):
                        orig_header_cell = str(header_row[i]).strip()
                        orig_words = orig_header_cell.split()
                        if len(orig_words) > 1 and cell_str == orig_header_cell:
                            data_part = []
                            for j, word in enumerate(orig_words):
                                if any(c.isdigit() for c in word) or word.startswith(('$', '€', '£', '₹')) or j > 0:
                                    data_part.extend(orig_words[j:])
                                    break
                            cleaned_first_row.append(' '.join(data_part) if data_part else orig_words[-1])
                            continue
                    cleaned_first_row.append(cell_str)
                
                table_data[1] = cleaned_first_row
        
        return table_data
    
    def _create_dataframe(self, table_data: List[List]) -> pd.DataFrame:
        """Create a DataFrame from the cleaned table data
        
        Args:
            table_data: List of lists where each inner list represents a row
            
        Returns:
            pd.DataFrame: DataFrame containing the table data
        """
        if not table_data:
            return pd.DataFrame()
            
        try:
            # Ensure all rows are lists and have the same length
            max_cols = max(len(row) for row in table_data) if table_data else 0
            table_data = [row + [''] * (max_cols - len(row)) if isinstance(row, list) else [''] * max_cols 
                         for row in table_data]
            
            # Create DataFrame with first row as header
            if len(table_data) > 1:
                df = pd.DataFrame(table_data[1:], columns=table_data[0])
            else:
                df = pd.DataFrame(columns=table_data[0] if table_data else [])
                
            # Clean up column names
            df.columns = [str(col).strip() for col in df.columns]
            
            # Remove empty rows and columns
            df = df.dropna(how='all').reset_index(drop=True)
            df = df.loc[:, (df != '').any(axis=0)]
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error creating DataFrame: {str(e)}", exc_info=True)
            # Fallback to simple DataFrame creation if above fails
            return pd.DataFrame(table_data[1:], columns=table_data[0]) if table_data else pd.DataFrame()
            data_rows.append(padded_row)
            
        return pd.DataFrame(data_rows, columns=header)
    
    def _create_success_response(self, filename: str, rows: int, cols: int, excel_path: str) -> Dict:
        """Create a success response dictionary"""
        return {
            "success": True,
            "message": f"Successfully extracted table from {filename}",
            "rows_extracted": rows,
            "columns": cols,
            "excel_file": excel_path,
            "timestamp": datetime.now().isoformat()
        }
    
    def _create_text_response(self, filename: str, text_content: Dict) -> Dict:
        """Create a response for text-only documents"""
        return {
            "success": True,
            "message": f"No tables detected in {filename}. Returning extracted text.",
            "rows_extracted": 0,
            "columns": 0,
            "excel_file": None,
            "text_content": text_content,
            "timestamp": datetime.now().isoformat()
        }
    
    def _cleanup_temp_files(self, temp_files: List[str]):
        """Clean up temporary files"""
        for file_path in temp_files:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    self.logger.warning(f"Error deleting temp file {file_path}: {e}")
