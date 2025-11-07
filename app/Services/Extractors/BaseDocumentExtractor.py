from abc import ABC, abstractmethod
from typing import Optional
from app.Logger.ocr_logger import get_standard_logger


class BaseDocumentExtractor(ABC):
    """Base class for all document extractors"""
    
    def __init__(self, ocr_service=None):
        self.logger = get_standard_logger(self.__class__.__name__)
        self.ocr_service = ocr_service
    
    @abstractmethod
    async def extract_text(self, file_path: str) -> str:
        """Extract text from a document file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Extracted text as a string
        """
        pass
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        if not text or not text.strip():
            return ""
        
        import re
        # Normalize whitespace
        cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)
        return cleaned.strip()

