import os
import tempfile
import shutil
import fitz  # PyMuPDF
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import io
import logging
import time
import concurrent.futures
from typing import Optional, Tuple, List, Union, Dict, Any
from datetime import datetime
from functools import wraps
import glob

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_execution_time(func):
    """Decorator to log execution time of methods"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} executed in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            raise
    return wrapper

class DocumentProcessor:
    """
    A utility class for processing various document types (PNG, JPEG, DOCX) to PDF format.
    Handles conversion, validation, and cleanup of temporary files with enhanced performance.
    """
    
    # Maximum file size (20MB)
    MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
    
    # Timeout for processing operations (seconds)
    PROCESS_TIMEOUT = 300  # 5 minutes
    
    # Supported file extensions
    SUPPORTED_EXTENSIONS = {
        '.pdf': 'application/pdf',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.doc': 'application/msword'
    }
    
    def __init__(self):
        self.temp_dir = None
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        
    def __del__(self):
        self.cleanup()
        self._executor.shutdown(wait=False)
    
    @log_execution_time
    def _create_temp_dir(self) -> str:
        """Create a temporary directory for processing files with a unique name"""
        if not self.temp_dir or not os.path.exists(self.temp_dir):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.temp_dir = tempfile.mkdtemp(prefix=f"doc_processor_{timestamp}_")
            logger.info(f"Created temporary directory: {self.temp_dir}")
        return self.temp_dir
    
    @log_execution_time
    def _validate_file(self, file_path: str) -> Tuple[bool, str]:
        """
        Validate the input file
        Returns (is_valid, error_message)
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                return False, f"File not found: {file_path}"
                
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                return False, "File is empty"
                
            if file_size > self.MAX_FILE_SIZE:
                return False, f"File size ({file_size/1024/1024:.1f}MB) exceeds maximum allowed size of {self.MAX_FILE_SIZE/1024/1024}MB"
            
            # Get file extension
            _, ext = os.path.splitext(file_path.lower())
            if not ext or ext not in self.SUPPORTED_EXTENSIONS:
                return False, f"Unsupported file type: {ext}. Allowed types: {', '.join(self.SUPPORTED_EXTENSIONS.keys())}"
            
            # Additional validation based on file type
            if ext == '.pdf':
                return self._validate_pdf(file_path)
            elif ext in ('.png', '.jpg', '.jpeg'):
                return self._validate_image(file_path)
            elif ext in ('.docx', '.doc'):
                return self._validate_docx(file_path)
                
            return True, ""
            
        except Exception as e:
            logger.error(f"Validation error for {file_path}: {str(e)}", exc_info=True)
            return False, f"Error validating file: {str(e)}"
    
    def _validate_pdf(self, file_path: str) -> Tuple[bool, str]:
        """Validate PDF file"""
        try:
            with fitz.open(file_path) as doc:
                if doc.is_encrypted:
                    return False, "Encrypted PDFs are not supported"
                if doc.page_count == 0:
                    return False, "PDF contains no pages"
                # Check if the PDF is image-based (scanned)
                text = ""
                for page in doc:
                    text += page.get_text()
                if len(text.strip()) < 100:  # Arbitrary threshold for text content
                    logger.info("PDF appears to be image-based (scanned document)")
                return True, ""
        except Exception as e:
            return False, f"Invalid PDF file: {str(e)}"
    
    def _validate_image(self, file_path: str) -> Tuple[bool, str]:
        """Validate image file"""
        try:
            with Image.open(file_path) as img:
                img.verify()  # Verify that it is, in fact, an image
                return True, ""
        except Exception as e:
            return False, f"Invalid image file: {str(e)}"
    
    def _validate_docx(self, file_path: str) -> Tuple[bool, str]:
        """Validate DOCX file"""
        try:
            import zipfile
            # Basic validation by checking if it's a valid zip file
            with zipfile.ZipFile(file_path, 'r') as zf:
                # Check for required document files
                required_files = ['[Content_Types].xml', '_rels/.rels']
                for f in required_files:
                    if f not in zf.namelist():
                        return False, f"Invalid DOCX file: missing {f}"
            return True, ""
        except Exception as e:
            return False, f"Invalid DOCX file: {str(e)}"
    
    @log_execution_time
    def _image_to_pdf(self, image_path: str, output_pdf_path: str) -> bool:
        """
        Convert an image file to PDF with enhanced image processing for better OCR results
        """
        try:
            # Open the image and process it in a separate thread
            def process_image():
                with Image.open(image_path) as img:
                    # Convert to RGB if necessary (required for PDF)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Enhance image for better OCR
                    img = self._enhance_image_for_ocr(img)
                    
                    # Save as PDF with high quality
                    img.save(output_pdf_path, "PDF", resolution=300, quality=95, optimize=True)
                    
                    return True
            
            # Run with timeout
            future = self._executor.submit(process_image)
            return future.result(timeout=self.PROCESS_TIMEOUT)
                
        except concurrent.futures.TimeoutError:
            logger.error(f"Image to PDF conversion timed out after {self.PROCESS_TIMEOUT} seconds")
            return False
        except Exception as e:
            logger.error(f"Error converting image to PDF: {str(e)}", exc_info=True)
            return False
    
    def _enhance_image_for_ocr(self, img: Image.Image) -> Image.Image:
        """Apply image enhancements to improve OCR accuracy"""
        try:
            # Convert to grayscale for better contrast
            if img.mode != 'L':
                img = img.convert('L')
            
            # Auto-contrast and auto-levels
            img = ImageOps.autocontrast(img, cutoff=1)
            
            # Increase contrast
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.8)  # Increased from 1.5 for better text clarity
            
            # Sharpen the image
            img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
            
            # Apply adaptive thresholding for better text extraction
            img = img.point(lambda x: 0 if x < 180 else 255, '1')
            
            # Convert back to RGB for PDF
            return img.convert('RGB')
            
        except Exception as e:
            logger.warning(f"Error enhancing image: {str(e)}")
            return img.convert('RGB')  # Return original if enhancement fails
    
    @log_execution_time
    def _docx_to_pdf(self, docx_path: str, output_pdf_path: str) -> bool:
        """
        Convert a DOCX file to PDF using python-docx and PyMuPDF
        This is more reliable than using reportlab for complex documents
        """
        try:
            # First try using docx2pdf if available
            try:
                from docx2pdf import convert
                convert(docx_path, output_pdf_path)
                return os.path.exists(output_pdf_path)
            except ImportError:
                logger.warning("docx2pdf not available, falling back to python-docx")
                
            # Fallback to python-docx and PyMuPDF
            import docx
            from docx.shared import Inches, RGBColor
            
            # Open the DOCX file
            doc = docx.Document(docx_path)
            
            # Create a new PDF document
            pdf_doc = fitz.open()
            
            # Create a new page
            page = pdf_doc.new_page()
            
            # Process each paragraph
            y_position = 72  # Start 1 inch from top
            for para in doc.paragraphs:
                # Skip empty paragraphs
                if not para.text.strip():
                    y_position += 12  # Add some space between paragraphs
                    continue
                
                # Check if we need a new page
                if y_position > 720:  # 10 inches down the page
                    page = pdf_doc.new_page()
                    y_position = 72
                
                # Add the paragraph text
                text = para.text
                
                # Get paragraph style
                style = para.style
                font_size = style.font.size.pt if style.font and style.font.size else 12
                
                # Add the text to the PDF
                page.insert_text(
                    point=(72, y_position),  # 1 inch from left
                    text=text,
                    fontsize=font_size,
                    fontname="helv"  # Use Helvetica as a standard font
                )
                
                # Move down for next paragraph
                y_position += font_size * 1.5
            
            # Save the PDF
            pdf_doc.save(output_pdf_path)
            pdf_doc.close()
            
            return os.path.exists(output_pdf_path)
            
        except Exception as e:
            logger.error(f"Error converting DOCX to PDF: {str(e)}", exc_info=True)
            return False
    
    @log_execution_time
    def process_file(self, file_path: str, output_dir: str = None) -> str:
        """
        Process a file and convert it to PDF if needed
        Returns the path to the processed PDF file
        """
        start_time = time.time()
        
        # Validate input
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Validate the file
        is_valid, error_msg = self._validate_file(file_path)
        if not is_valid:
            raise ValueError(f"Invalid file: {error_msg}")
        
        # Create output directory if not specified
        if not output_dir:
            output_dir = self._create_temp_dir()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get file info
        file_name = os.path.basename(file_path)
        base_name, ext = os.path.splitext(file_name)
        output_pdf = os.path.join(output_dir, f"{base_name}.pdf")
        
        logger.info(f"Processing file: {file_name} (Size: {os.path.getsize(file_path)/1024:.1f}KB)")
        
        try:
            # Process based on file type
            if ext.lower() == '.pdf':
                # Just copy the PDF if it's already in the right format
                shutil.copy2(file_path, output_pdf)
                
            elif ext.lower() in ('.png', '.jpg', '.jpeg'):
                # Convert image to PDF
                if not self._image_to_pdf(file_path, output_pdf):
                    raise RuntimeError(f"Failed to convert image to PDF: {file_path}")
                    
            elif ext.lower() in ('.docx', '.doc'):
                # Convert DOCX to PDF
                if not self._docx_to_pdf(file_path, output_pdf):
                    raise RuntimeError(f"Failed to convert document to PDF: {file_path}")
                    
            else:
                raise ValueError(f"Unsupported file type: {ext}")
            
            # Verify the output file
            if not os.path.exists(output_pdf):
                raise RuntimeError(f"Output file was not created: {output_pdf}")
                
            output_size = os.path.getsize(output_pdf) / 1024  # KB
            logger.info(f"Successfully processed {file_name} -> {os.path.basename(output_pdf)} "
                      f"({output_size:.1f}KB) in {time.time() - start_time:.2f} seconds")
            
            return output_pdf
            
        except Exception as e:
            logger.error(f"Error processing file {file_name}: {str(e)}", exc_info=True)
            # Clean up any partial output file
            if os.path.exists(output_pdf):
                try:
                    os.remove(output_pdf)
                except:
                    pass
            raise
    
    @log_execution_time
    def cleanup(self):
        """Clean up temporary files and directories"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                logger.info(f"Cleaning up temporary directory: {self.temp_dir}")
                shutil.rmtree(self.temp_dir)
                self.temp_dir = None
            except Exception as e:
                logger.warning(f"Error cleaning up temp directory: {str(e)}")
        
        # Clean up any other temporary files
        temp_dir = tempfile.gettempdir()
        for pattern in ["doc_processor_*"]:
            for temp_path in glob.glob(os.path.join(temp_dir, pattern)):
                try:
                    if os.path.isdir(temp_path):
                        shutil.rmtree(temp_path, ignore_errors=True)
                    else:
                        os.remove(temp_path)
                except Exception as e:
                    logger.debug(f"Could not remove temp file {temp_path}: {str(e)}")