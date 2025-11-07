import re
from typing import Optional
from app.Services.Extractors.BaseDocumentExtractor import BaseDocumentExtractor


class PDFExtractor(BaseDocumentExtractor):
    """PDF document extractor - preserves all existing PDF extraction logic"""
    
    def _is_scanned_pdf(self, pdf_path: str) -> bool:
        """Detect if PDF is scanned (image-based) by analyzing text content.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            True if PDF appears to be scanned, False otherwise
        """
        try:
            import fitz
            doc = fitz.open(pdf_path)
            total_chars = 0
            total_pages = len(doc)
            
            # Sample first few pages to determine if scanned
            sample_pages = min(3, total_pages)
            
            for page_num in range(sample_pages):
                page = doc[page_num]
                # Try multiple extraction methods
                page_text = page.get_text() or ""
                page_text += page.get_text("text") or ""
                page_text += page.get_text("raw") or ""
                
                # Also check dict blocks
                try:
                    text_dict = page.get_text("dict", flags=11)
                    for block in text_dict.get("blocks", []):
                        if block.get("type") == 0:  # Text block
                            for line in block.get("lines", []):
                                for span in line.get("spans", []):
                                    page_text += span.get("text", "")
                except:
                    pass
                
                total_chars += len(page_text.strip())
            
            doc.close()
            
            # Threshold: if average chars per page is less than 50, likely scanned
            avg_chars_per_page = total_chars / sample_pages if sample_pages > 0 else 0
            is_scanned = avg_chars_per_page < 50
            
            self.logger.info(
                f"Scanned PDF detection: {total_chars} chars from {sample_pages} pages "
                f"(avg: {avg_chars_per_page:.1f} chars/page) - {'SCANNED' if is_scanned else 'TEXT-BASED'}"
            )
            
            return is_scanned
            
        except Exception as e:
            self.logger.warning(f"Scanned PDF detection failed: {e}")
            # Default to not scanned if detection fails
            return False
    
    def _perform_ocr_on_page(self, page, page_num: int) -> str:
        """Perform OCR on a single page using Tesseract via OCRService.
        
        Args:
            page: PyMuPDF page object
            page_num: Page number (0-indexed)
            
        Returns:
            Extracted text from OCR
        """
        try:
            # Use OCRService's enhanced OCR method
            spans = self.ocr_service.extract_page_spans(page)
            
            if not spans:
                return ""
            
            # Extract text from spans and combine
            ocr_text_parts = []
            for span in spans:
                text = span.get("text", "").strip()
                if text:
                    ocr_text_parts.append(text)
            
            ocr_text = "\n".join(ocr_text_parts)
            
            if ocr_text.strip():
                self.logger.info(f"OCR extracted {len(ocr_text)} characters from page {page_num + 1}")
                return ocr_text
            else:
                return ""
                
        except ImportError:
            self.logger.debug("pytesseract not available for OCR")
            return ""
        except Exception as ocr_err:
            self.logger.warning(f"OCR failed for page {page_num + 1}: {ocr_err}")
            return ""
    
    async def extract_text(self, file_path: str) -> str:
        """Extract text from PDF file - COMPLETE PRESERVATION OF EXISTING LOGIC.
        
        This method uses multiple extraction strategies to handle:
        - Regular PDFs with direct text
        - Canva PDFs with XObject Forms
        - PDFs with structured content trees
        - Scanned/image-based PDFs (with automatic OCR detection and processing)
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Complete extracted text as a string
        """
        all_text = ""
        
        try:
            import fitz
            doc = fitz.open(file_path)
            
            # Step 1: Detect if PDF is scanned
            is_scanned = self._is_scanned_pdf(file_path)
            
            if is_scanned:
                self.logger.info("Detected scanned PDF - using OCR for all pages")
                # For scanned PDFs, use OCR directly for all pages
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    ocr_text = self._perform_ocr_on_page(page, page_num)
                    if ocr_text.strip():
                        all_text += f"\n--- PAGE {page_num + 1} ---\n{ocr_text}\n"
                doc.close()
                
                # Clean and return OCR results
                if all_text.strip():
                    return self.clean_text(all_text)
                else:
                    return "No text content found in scanned PDF"
            
            # Step 2: For text-based PDFs, use comprehensive extraction strategies
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_texts = []
                
                # Strategy 1: Standard PyMuPDF extraction methods
                try:
                    page_texts.append(page.get_text())
                    page_texts.append(page.get_text("text"))
                    page_texts.append(page.get_text("raw"))
                    page_texts.append(page.get_text("layout"))
                except Exception as e:
                    self.logger.warning(f"Standard extraction failed for page {page_num + 1}: {e}")
                
                # Strategy 2: Extract from dict blocks (catches structured content)
                try:
                    td = page.get_text("dict", flags=11)  # flags=11 for comprehensive extraction
                    block_lines = []
                    for block in td.get("blocks", []):
                        if block.get("type") == 0:  # Text block
                            for line in block.get("lines", []):
                                line_text = " ".join([span.get("text", "") for span in line.get("spans", [])]).strip()
                                if line_text:
                                    block_lines.append(line_text)
                    if block_lines:
                        page_texts.append("\n".join(block_lines))
                except Exception as e:
                    self.logger.warning(f"Dict block extraction failed for page {page_num + 1}: {e}")
                
                # Strategy 3: Extract from structured content tree (for Canva PDFs with StructTreeRoot)
                try:
                    struct_text = []
                    # Get structured text with comprehensive flags
                    text_dict = page.get_text("dict", flags=11)
                    for block in text_dict.get("blocks", []):
                        if block.get("type") == 0:  # Text block
                            for line in block.get("lines", []):
                                line_parts = []
                                for span in line.get("spans", []):
                                    span_text = span.get("text", "").strip()
                                    if span_text:
                                        line_parts.append(span_text)
                                if line_parts:
                                    struct_text.append(" ".join(line_parts))
                    
                    if struct_text:
                        page_texts.append("\n".join(struct_text))
                except Exception as e:
                    self.logger.warning(f"Structured content extraction failed for page {page_num + 1}: {e}")
                
                # Strategy 4: Extract from XObject Forms (critical for Canva PDFs)
                try:
                    # Canva PDFs store content in XObject Forms referenced in page resources
                    xobject_texts = []
                    
                    # Get page resources to find XObjects
                    try:
                        # Try to extract text with different extraction flags that handle XObjects
                        xobj_text = page.get_text(flags=11)  # Comprehensive flags
                        if xobj_text and xobj_text.strip():
                            xobject_texts.append(xobj_text)
                    except:
                        pass
                    
                    # Also try extracting from the document's structure tree if available
                    try:
                        if doc.is_pdf:
                            # Try to get text from all content streams
                            for xref in page.get_contents():
                                try:
                                    # PyMuPDF should handle XObjects automatically, but we can also
                                    # try to get additional text from content streams
                                    stream_text = page.get_text(flags=11)
                                    if stream_text:
                                        xobject_texts.append(stream_text)
                                except:
                                    pass
                    except:
                        pass
                    
                    if xobject_texts:
                        # Deduplicate and combine
                        unique_xobj = list(set([t.strip() for t in xobject_texts if t.strip()]))
                        if unique_xobj:
                            page_texts.append("\n".join(unique_xobj))
                except Exception as e:
                    self.logger.warning(f"XObject extraction failed for page {page_num + 1}: {e}")
                
                # Combine all extraction results for this page
                combined = "\n".join([t for t in page_texts if t and t.strip()])
                
                # If we got substantial text, add it
                if combined.strip() and len(combined.strip()) > 20:
                    all_text += f"\n--- PAGE {page_num + 1} ---\n{combined}\n"
                # Otherwise try OCR as fallback for individual pages
                elif len(combined.strip()) < 20:
                    self.logger.info(f"Minimal text found on page {page_num + 1}, attempting OCR fallback")
                    ocr_text = self._perform_ocr_on_page(page, page_num)
                    if ocr_text.strip():
                        all_text += f"\n--- PAGE {page_num + 1} ---\n{ocr_text}\n"
            
            doc.close()
            
        except Exception as e:
            self.logger.error(f"PyMuPDF comprehensive extraction failed: {e}")
        
        # Fallback: Try pdfplumber if PyMuPDF didn't get enough
        if len(all_text.strip()) < 200:
            try:
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    plumber_text = ""
                    for page_num, page in enumerate(pdf.pages):
                        page_text = page.extract_text()
                        if page_text:
                            plumber_text += f"\n--- PAGE {page_num + 1} ---\n{page_text}\n"
                    
                    if len(plumber_text.strip()) > len(all_text.strip()):
                        all_text = plumber_text
                        self.logger.info("pdfplumber provided better extraction results")
            except Exception as e:
                self.logger.warning(f"pdfplumber fallback failed: {e}")
        
        # Final fallback: PyPDF2
        if len(all_text.strip()) < 100:
            try:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    pypdf2_text = ""
                    for page_num, page in enumerate(reader.pages):
                        page_text = page.extract_text()
                        if page_text:
                            pypdf2_text += f"\n--- PAGE {page_num + 1} ---\n{page_text}\n"
                    
                    if len(pypdf2_text.strip()) > len(all_text.strip()):
                        all_text = pypdf2_text
                        self.logger.info("PyPDF2 provided fallback extraction results")
            except Exception as e:
                self.logger.warning(f"PyPDF2 fallback failed: {e}")
        
        # Clean and return
        if all_text.strip():
            return self.clean_text(all_text)
        else:
            return "No text content found in PDF"

