import os
from typing import Optional
from app.Logger.ocr_logger import get_standard_logger
from app.Services.OCRService import OCRService
from app.Services.Extractors.BaseDocumentExtractor import BaseDocumentExtractor
from app.Services.Extractors.PDFExtractor import PDFExtractor
from app.Services.Extractors.DocxExtractor import DocxExtractor
from app.Services.Extractors.ImageExtractor import ImageExtractor


class DocumentExtractorFactory:
    """Factory for creating document extractors based on file type"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DocumentExtractorFactory, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.logger = get_standard_logger("DocumentExtractorFactory")
            self.ocr_service = OCRService()
            self._initialized = True
    
    def detect_file_type(self, file_path: str) -> str:
        """Detect file type based on extension and content.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File type: 'pdf', 'docx', 'jpeg', 'jpg', 'png', or 'unknown'
        """
        if not file_path:
            return 'unknown'
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Map extensions to file types
        extension_map = {
            '.pdf': 'pdf',
            '.docx': 'docx',
            '.doc': 'docx',  # Treat .doc as .docx for extraction
            '.jpeg': 'jpeg',
            '.jpg': 'jpeg',
            '.png': 'png',
            '.gif': 'png',  # Treat GIF as image
            '.bmp': 'png',  # Treat BMP as image
            '.tiff': 'png',  # Treat TIFF as image
            '.tif': 'png'   # Treat TIF as image
        }
        
        file_type = extension_map.get(file_ext, 'unknown')
        
        # Additional validation: check file content for images
        if file_type in ['jpeg', 'png']:
            try:
                with open(file_path, 'rb') as f:
                    # Check image magic numbers
                    header = f.read(8)
                    if file_type == 'jpeg' and not header.startswith(b'\xff\xd8'):
                        self.logger.warning(f"File extension suggests JPEG but magic number doesn't match: {file_path}")
                    elif file_type == 'png' and not header.startswith(b'\x89PNG\r\n\x1a\n'):
                        self.logger.warning(f"File extension suggests PNG but magic number doesn't match: {file_path}")
            except Exception as e:
                self.logger.debug(f"Could not verify image magic number: {e}")
        
        self.logger.info(f"Detected file type: {file_type} for {file_path}")
        return file_type
    
    def create_extractor(self, file_path: str) -> Optional[BaseDocumentExtractor]:
        """Create appropriate extractor based on file type.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Document extractor instance or None if unsupported
        """
        file_type = self.detect_file_type(file_path)
        
        if file_type == 'pdf':
            return PDFExtractor(ocr_service=self.ocr_service)
        elif file_type == 'docx':
            return DocxExtractor(ocr_service=self.ocr_service)
        elif file_type in ['jpeg', 'png']:
            return ImageExtractor(ocr_service=self.ocr_service)
        else:
            self.logger.warning(f"Unsupported file type: {file_type} for {file_path}")
            return None
    
    async def extract_text(self, file_path: str) -> str:
        """Extract text from a file using the appropriate extractor.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Extracted text as a string
        """
        extractor = self.create_extractor(file_path)
        
        if extractor is None:
            file_type = self.detect_file_type(file_path)
            return f"Error: Unsupported file type ({file_type}). Supported types: PDF, DOCX, JPEG, PNG"
        
        return await extractor.extract_text(file_path)


# Singleton instance
document_extractor_factory = DocumentExtractorFactory()

