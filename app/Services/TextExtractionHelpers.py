"""
Helper methods for enhanced text extraction from PDFs
"""

import re
from app.Logger.ocr_logger import get_standard_logger

logger = get_standard_logger("TextExtractionHelpers")

def extract_from_text_dict(text_dict):
    """Extract text from PyMuPDF text dictionary with better structure preservation"""
    try:
        structured_text = ""
        
        for block in text_dict.get("blocks", []):
            if block.get("type") == 0:  # Text block
                block_text = ""
                for line in block.get("lines", []):
                    line_text = ""
                    for span in line.get("spans", []):
                        span_text = span.get("text", "")
                        if span_text.strip():
                            line_text += span_text
                    
                    if line_text.strip():
                        block_text += line_text.strip() + "\n"
                
                if block_text.strip():
                    structured_text += block_text + "\n"
        
        return structured_text.strip()
    except Exception as e:
        logger.debug(f"Text dict extraction failed: {e}")
        return ""

def extract_from_blocks(blocks):
    """Extract text from PyMuPDF blocks with layout preservation"""
    try:
        block_text = ""
        
        for block in blocks:
            if len(block) >= 5:  # Valid text block
                text = block[4].strip()
                if text:
                    block_text += text + "\n"
        
        return block_text.strip()
    except Exception as e:
        logger.debug(f"Block extraction failed: {e}")
        return ""

def clean_raw_text(raw_text):
    """Clean raw text while preserving structure"""
    try:
        # Remove excessive whitespace but preserve line breaks
        cleaned = re.sub(r'[ \t]+', ' ', raw_text)
        cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)
        
        return cleaned.strip()
    except Exception as e:
        logger.debug(f"Raw text cleaning failed: {e}")
        return raw_text

def reconstruct_text_from_words(words):
    """Reconstruct text from pdfplumber words with proper spacing"""
    try:
        if not words:
            return ""
        
        # Sort words by position (top to bottom, left to right)
        sorted_words = sorted(words, key=lambda w: (w.get('top', 0), w.get('x0', 0)))
        
        reconstructed = ""
        current_line_top = None
        line_tolerance = 5  # pixels
        
        for word in sorted_words:
            word_text = word.get('text', '').strip()
            if not word_text:
                continue
            
            word_top = word.get('top', 0)
            
            # Check if this word is on a new line
            if current_line_top is None or abs(word_top - current_line_top) > line_tolerance:
                if reconstructed:
                    reconstructed += "\n"
                current_line_top = word_top
            else:
                # Same line, add space
                if reconstructed and not reconstructed.endswith("\n"):
                    reconstructed += " "
            
            reconstructed += word_text
        
        return reconstructed.strip()
    except Exception as e:
        logger.debug(f"Word reconstruction failed: {e}")
        return ""

def post_process_text(text):
    """Post-process extracted text for better readability"""
    try:
        # Remove excessive whitespace
        processed = re.sub(r'[ \t]+', ' ', text)
        processed = re.sub(r'\n\s*\n\s*\n+', '\n\n', processed)
        
        # Fix common extraction issues
        processed = re.sub(r'([a-z])([A-Z])', r'\1 \2', processed)  # Add space between camelCase
        processed = re.sub(r'(\d)([A-Za-z])', r'\1 \2', processed)  # Add space between number and letter
        processed = re.sub(r'([A-Za-z])(\d)', r'\1 \2', processed)  # Add space between letter and number
        
        # Clean up phone numbers and emails
        processed = re.sub(r'(\+\d{1,3})\s*(\d)', r'\1 \2', processed)
        processed = re.sub(r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', r'\1', processed)
        
        return processed.strip()
    except Exception as e:
        logger.debug(f"Text post-processing failed: {e}")
        return text