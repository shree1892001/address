from app.Logger.ocr_logger import get_standard_logger
from app.Exceptions.custom_exceptions import (
    handle_general_operations, log_method_entry_exit, ExceptionSeverity
)
from app.Services.OCRService import OCRService


class TextOnlyExtractionService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TextOnlyExtractionService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.logger = get_standard_logger("TextOnlyExtractionService")
            self.ocr_service = OCRService()
            self._initialized = True

    @log_method_entry_exit
    @handle_general_operations(severity=ExceptionSeverity.MEDIUM)
    async def extract_text_from_pdf(self, pdf_path):
        """Extract structured text from PDF when no tables are found"""
        doc = None
        try:
            doc = self.ocr_service.open_pdf(pdf_path)
            all_text = []
            
            for page_idx, page in enumerate(doc):
                self.logger.info(f"Extracting text from page {page_idx + 1}")
                
                # Try direct text extraction first
                page_text = page.get_text()
                
                if len(page_text.strip()) < 50:  # Scanned page
                    # Use OCR for scanned pages
                    spans = self.ocr_service.extract_page_spans(page)
                    page_text = self._format_spans_as_text(spans)
                
                if page_text.strip():
                    all_text.append({
                        "page": page_idx + 1,
                        "content": page_text.strip()
                    })
            
            return self._structure_extracted_text(all_text)
            
        finally:
            if doc:
                doc.close()

    def _format_spans_as_text(self, spans):
        """Format OCR spans into readable text"""
        if not spans:
            return ""
        
        # Sort spans by position
        spans.sort(key=lambda s: (s["y0"], s["x0"]))
        
        # Group into lines
        lines = []
        current_line = []
        current_y = spans[0]["y0"] if spans else 0
        
        for span in spans:
            if abs(span["y0"] - current_y) < 15:  # Same line
                current_line.append(span["text"])
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [span["text"]]
                current_y = span["y0"]
        
        if current_line:
            lines.append(" ".join(current_line))
        
        return "\n".join(lines)

    def _structure_extracted_text(self, pages_text):
        """Structure the extracted text into sections"""
        structured_text = {
            "total_pages": len(pages_text),
            "full_text": "",
            "pages": pages_text,
            "sections": []
        }
        
        # Combine all text
        full_text = "\n\n".join([page["content"] for page in pages_text])
        structured_text["full_text"] = full_text
        
        # Basic section detection
        sections = self._detect_text_sections(full_text)
        structured_text["sections"] = sections
        
        return structured_text

    def _detect_text_sections(self, text):
        """Detect basic text sections"""
        sections = []
        lines = text.split('\n')
        current_section = []
        section_title = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line looks like a heading
            if self._is_heading(line):
                # Save previous section
                if current_section and section_title:
                    sections.append({
                        "title": section_title,
                        "content": "\n".join(current_section)
                    })
                
                # Start new section
                section_title = line
                current_section = []
            else:
                current_section.append(line)
        
        # Add last section
        if current_section and section_title:
            sections.append({
                "title": section_title,
                "content": "\n".join(current_section)
            })
        
        return sections

    def _is_heading(self, line):
        """Check if line is likely a heading"""
        if len(line) < 3 or len(line) > 100:
            return False
        
        # Check for heading patterns
        heading_indicators = [
            line.isupper(),  # ALL CAPS
            line.istitle(),  # Title Case
            line.endswith(':'),  # Ends with colon
            any(word in line.lower() for word in ['chapter', 'section', 'part', 'summary'])
        ]
        
        return any(heading_indicators)