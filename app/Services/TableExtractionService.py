import pandas as pd
import subprocess
import os
import re
from datetime import datetime

try:
    import ocrmypdf
except ImportError:
    ocrmypdf = None
from app.Logger.ocr_logger import get_standard_logger
from app.Exceptions.custom_exceptions import NoTableFoundException
from app.Exceptions.custom_exceptions import (
    handle_general_operations, log_method_entry_exit, monitor_performance, ExceptionSeverity
)
from app.Services.FileValidationService import FileValidationService
from app.Common.Constants import (
    CLASS_TEXT_CONF_MIN,
    CLASS_TOP_TABLE_CONF_MAX,
    TABLE_MIN_QUALITY,
    SINGLE_COL_MIN_ROWS,
    SINGLE_COL_MIN_QUALITY,
    SCANNED_TEXT_MIN_CHARS,
    SPATIAL_SAME_LINE_TOL_PX,
    WORD_TOKEN_MAX_LEN,
    SHORT_WORD_RATIO_MIN,
    TABLE_MIN_NUMERIC_CHAR_RATIO,
    ALPHA_CHAR_RATIO_MIN,
    PARAGRAPH_LIKE_MIN_COLS,
    AVG_TOKEN_LEN_MIN,
    AVG_TOKEN_LEN_MAX,
)
from app.Services.FileUploadService import FileUploadService
from app.Services.OCRService import OCRService
from app.Services.TableProcessingService import TableProcessingService
from app.Services.FileSaveService import FileSaveService
from app.Services.DocumentClassificationService import DocumentClassificationService
from app.Services.TextExtractionService import TextExtractionService
from app.Services.TextOnlyExtractionService import TextOnlyExtractionService
from app.Services.TextExtractionHelpers import (
    extract_from_text_dict, extract_from_blocks, clean_raw_text,
    reconstruct_text_from_words, post_process_text
)


class TableExtractionService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TableExtractionService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.logger = get_standard_logger("TableExtractionService")
            self.file_validation_service = FileValidationService()
            self.file_upload_service = FileUploadService()
            self.ocr_service = OCRService()
            self.table_processing_service = TableProcessingService()
            self.file_save_service = FileSaveService()
            self.document_classification_service = DocumentClassificationService()
            self.text_extraction_service = TextExtractionService()
            self.text_only_extraction_service = TextOnlyExtractionService()
            self._initialized = True

    @log_method_entry_exit
    @monitor_performance
    @handle_general_operations(severity=ExceptionSeverity.HIGH)
    async def extract_table_from_pdf(self, file):
        """Extract table from uploaded PDF document"""
        temp_files = []
        
        try:
            # Validate file type
            self.file_validation_service.validate_pdf_file(file)
            
            # Save uploaded file
            file_paths = await self.file_upload_service.save_uploaded_file(file)
            temp_files.extend([file_paths["pdf_path"], file_paths["excel_path"]])
            
            # Extract table using new services
            table_data = await self._extract_table_data(file_paths["pdf_path"])
            
            if not table_data:
                self.logger.warning(f"No table data found in document: {file.filename}")
                raise NoTableFoundException(file_paths["pdf_path"])
            
            # If the result is a structured text fallback (no tables), return text JSON response
            if isinstance(table_data, dict) and table_data.get("no_table", False):
                text_content = table_data.get("text_content", {})
                response = {
                    "success": True,
                    "message": f"No tables detected in {file.filename}. Returning extracted text.",
                    "rows_extracted": 0,
                    "columns": 0,
                    "excel_file": None,
                    "text_content": text_content,
                    "timestamp": datetime.now().isoformat()
                }
                self.logger.info(f"No tables found. Returned paragraph/JSON content for {file.filename}")
                return response

            # Clean merged header-data before saving
            if table_data and len(table_data) > 0:
                self.logger.debug(f"Original header row: {table_data[0]}")
                header_row = table_data[0]
                cleaned_header = []
                for cell in header_row:
                    cell_str = str(cell).strip()
                    words = cell_str.split()
                    if len(words) > 1:
                        # Find where header ends and data begins
                        header_part = []
                        for i, word in enumerate(words):
                            # If word contains digits or currency, it's likely data
                            if any(c.isdigit() for c in word) or word.startswith(('$', '€', '£', '₹')):
                                break
                            header_part.append(word)
                        
                        # If we found a split point, use header part, otherwise use first word
                        if header_part:
                            cleaned_header.append(' '.join(header_part))
                        else:
                            cleaned_header.append(words[0])
                    else:
                        cleaned_header.append(cell_str)
                
                # Also ensure first data row is properly separated
                if len(table_data) > 1:
                    first_data_row = table_data[1]
                    self.logger.debug(f"Original first data row: {first_data_row}")
                    # If first data row looks like it has header remnants, clean it
                    cleaned_first_row = []
                    for i, cell in enumerate(first_data_row):
                        cell_str = str(cell).strip()
                        # If this cell matches pattern from original header, extract data part
                        if i < len(header_row):
                            orig_header_cell = str(header_row[i]).strip()
                            orig_words = orig_header_cell.split()
                            if len(orig_words) > 1 and cell_str == orig_header_cell:
                                # This data cell is same as original merged header, extract data part
                                data_part = []
                                for j, word in enumerate(orig_words):
                                    # Skip header words, keep data words
                                    if any(c.isdigit() for c in word) or word.startswith(('$', '€', '£', '₹')) or j > 0:
                                        data_part.extend(orig_words[j:])
                                        break
                                cleaned_first_row.append(' '.join(data_part) if data_part else orig_words[-1])
                            else:
                                cleaned_first_row.append(cell_str)
                        else:
                            cleaned_first_row.append(cell_str)
                    table_data[1] = cleaned_first_row
                    self.logger.debug(f"Cleaned first data row: {table_data[1]}")
                
                table_data[0] = cleaned_header
                self.logger.debug(f"Final cleaned header row: {table_data[0]}")
            
            # Save results
            self.file_save_service.save_results(table_data, file_paths["json_path"], file_paths["csv_path"])
            
            # Convert to DataFrame and save as Excel
            # Ensure all rows have same number of columns as header
            header = table_data[0] if table_data else []
            data_rows = []
            for row in table_data[1:]:
                # Pad or trim row to match header length
                padded_row = row[:len(header)] + [''] * (len(header) - len(row))
                data_rows.append(padded_row)
            df = pd.DataFrame(data_rows, columns=header)
            df.to_excel(file_paths["excel_path"], index=False)
            
            # Prepare response
            data_rows_count = len(data_rows)  # Actual data rows processed
            response = {
                "success": True,
                "message": f"Successfully extracted table from {file.filename}",
                "rows_extracted": data_rows_count,
                "columns": len(header) if header else 0,
                "excel_file": file_paths["excel_path"],
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"Successfully processed {file.filename}: {response['rows_extracted']} rows extracted")
            return response
            
        except Exception as e:
            self.logger.error(f"Table extraction failed: {e}")
            # Cleanup temp files on error
            self.file_upload_service.cleanup_temp_files(temp_files)
            raise

    async def _extract_table_data(self, pdf_path):
        """Extract table data from PDF using new services"""
        doc = None
        try:
            # Open PDF
            doc = self.ocr_service.open_pdf(pdf_path)
            all_spans = []
            all_rows = []

            # Process each page
            for page_idx, page in enumerate(doc):
                self.logger.info(f"Processing page {page_idx + 1}")
                spans = self.ocr_service.extract_page_spans(page)
                all_spans.extend(spans)
                
                # Process page individually to maintain table structure
                if spans:
                    page_rows = self.table_processing_service.cluster_rows(spans)
                    all_rows.extend(page_rows)

            # Early document-level classification to avoid false table positives on text docs
            try:
                classification = self.document_classification_service.classify_document_content(all_spans)
                doc_type = classification.get("document_type", "unknown")
                table_regions = classification.get("table_regions", [])
                class_conf = classification.get("confidence", 0.0)
                top_table_conf = (table_regions[0].get("confidence", 0.0) if table_regions else 0.0)
                if doc_type == "text_document" and (
                        not all_rows or len(all_rows) < 2 or top_table_conf < CLASS_TOP_TABLE_CONF_MAX) and class_conf >= CLASS_TEXT_CONF_MIN:
                    self.logger.info(
                        f"Document classified as text (conf={class_conf:.2f}, top_table_conf={top_table_conf:.2f}); returning text content"
                    )
                    text_result = await self.text_only_extraction_service.extract_text_from_pdf(pdf_path)
                    if text_result and text_result.get("full_text"):
                        return {"no_table": True, "text_content": text_result}
            except Exception as e:
                self.logger.warning(f"Document classification shortcut failed: {e}")

            if not all_rows or len(all_spans) < 2:
                self.logger.warning(f"Insufficient text found: {len(all_spans)} spans, {len(all_rows)} rows")
                # For scanned PDFs, use ocrmypdf to get better OCR
                if self._is_scanned_pdf(pdf_path):
                    ocr_pdf_path = await self._convert_to_searchable_pdf(pdf_path)
                    if ocr_pdf_path:
                        return await self._extract_table_data(ocr_pdf_path)
                # Fallback to spatial extraction if OCR fails
                if all_spans:
                    spatial_table = self._extract_spatial_table(all_spans)
                    if spatial_table:
                        return spatial_table
                # Only use text extraction if no tables found
                text_result = await self.text_only_extraction_service.extract_text_from_pdf(pdf_path)
                if text_result and text_result.get("full_text"):
                    self.logger.info("No tables found, extracted text content instead")
                    return {"no_table": True, "text_content": text_result}
                raise NoTableFoundException(pdf_path, details={"reason": "No text spans found in document"})

            # Use adaptive column detection for multi-page documents
            self.logger.info(f"Detecting columns from {len(all_rows)} rows across {len(doc)} pages")
            
            # Find the row with maximum columns as template
            max_col_row = max(all_rows, key=len) if all_rows else []
            if len(max_col_row) >= 2:
                global_centers = [span["x0"] for span in max_col_row]
            else:
                global_centers = self.table_processing_service.detect_template_columns(all_rows)
            
            global_min_x = min(s["x0"] for s in all_spans) if all_spans else 0
            global_max_x = max(s["x1"] for s in all_spans) if all_spans else 0
            global_bounds = self.table_processing_service.centers_to_bounds(
                global_centers, global_min_x, global_max_x
            )
            expected_cols = len(global_centers)

            self.logger.info(f"Building table with {expected_cols} columns")
            table = self.table_processing_service.build_table_with_bounds(all_rows, global_bounds, global_centers)

            # Debug: log a small sample of the constructed table before header detection
            try:
                sample_rows = table[:8] if isinstance(table, list) else []
                self.logger.debug(f"Constructed table sample (first {len(sample_rows)} rows): {sample_rows}")
            except Exception:
                self.logger.debug("Constructed table sample: <unserializable>")

            # Process table structure with dynamic header extraction
            header_found = False
            header_row = None
            pre_table = []
            i = 0
            
            # First, try to identify the header row more accurately
            potential_header_indices = []
            for idx, row in enumerate(table):
                clean_row = [str(cell).strip() for cell in row]
                if any(clean_row):  # Skip empty rows
                    if self.table_processing_service.is_likely_header(clean_row, expected_cols):
                        potential_header_indices.append(idx)
            
            # If we found potential headers, use the first one
            if potential_header_indices:
                header_idx = potential_header_indices[0]
                header_row = self._extract_best_header_dynamically(table, header_idx, expected_cols)
                if header_row:
                    # Merge split headers across adjacent cells (e.g., "Joining" + "Date" -> "Joining Date")
                    header_row = self._merge_split_headers_in_row(header_row, table, header_idx, expected_cols)
                    header_found = True
                    i = header_idx + 1
                    # Skip any continuation rows that are part of the header
                    while i < len(table) and self._is_header_continuation(header_row, [str(cell).strip() for cell in table[i]]):
                        i += 1
            
            # Now process the rest of the table
            while i < len(table):
                clean_row = [str(cell).strip() for cell in table[i]]
                
                # Skip empty rows
                if not any(clean_row):
                    i += 1
                    continue
                
                # If we have a header, process data rows
                if header_found:
                    # First, check if first data row has header continuation words that should be merged with header
                    # This handles cases where "Date" appears in the first data row but should be part of "Joining Date" header
                    if len(pre_table) == 0:  # This is the first data row
                        header_row, clean_row = self._fix_split_headers_in_data_row(header_row, clean_row, expected_cols)
                    
                    # Check for merged header-data cells (but be careful not to split multi-word headers)
                    for col_idx, cell in enumerate(clean_row):
                        if '\n' in cell and col_idx < len(header_row):
                            parts = [p.strip() for p in cell.split('\n') if p.strip()]
                            if len(parts) >= 2:
                                current_header = header_row[col_idx].strip() if col_idx < len(header_row) else ""
                                
                                # Generic logic: Determine if parts should be merged into header or split
                                should_merge_with_header = self._should_merge_parts_with_header(
                                    current_header, parts, col_idx, header_row
                                )
                                
                                if should_merge_with_header:
                                    # Merge first part(s) with header, keep rest as data
                                    parts_to_merge = []
                                    parts_for_data = []
                                    
                                    # Determine how many parts belong to header
                                    for i, part in enumerate(parts):
                                        if self._is_likely_header_part(part, current_header, i, len(parts)):
                                            parts_to_merge.append(part)
                                        else:
                                            parts_for_data = parts[i:]
                                            break
                                    
                                    if parts_to_merge:
                                        # Complete the header
                                        merged_header = f"{current_header} {' '.join(parts_to_merge)}".strip()
                                        header_row[col_idx] = merged_header
                                        # Keep the rest as data
                                        clean_row[col_idx] = '\n'.join(parts_for_data) if parts_for_data else ""
                                # If header cell is empty or very short, and first part looks like header
                                elif (not current_header or len(current_header) < 3) and \
                                   self.table_processing_service.is_header_like(parts[0]) and \
                                   not self.table_processing_service.is_header_like(parts[1]):
                                    # Update the header
                                    header_row[col_idx] = parts[0]
                                    # Keep the rest as data
                                    clean_row[col_idx] = '\n'.join(parts[1:])
                    
                    # Only add non-header rows that look like data
                    if (not self.table_processing_service.is_repeat_header(clean_row, header_row) and 
                        self.table_processing_service.is_likely_table_row(clean_row, expected_cols)):
                        pre_table.append(clean_row)
                
                i += 1
                
            # If we didn't find a header but have data, use the first non-empty row as header
            if not header_found and pre_table:
                header_row = pre_table[0]
                pre_table = pre_table[1:] if len(pre_table) > 1 else []
                header_found = True

            # If we effectively have a single column, don't immediately reject; we'll validate later
            if expected_cols < 2:
                self.logger.info("Detected one effective column; will validate as potential single-column table")

            self.logger.info(f"Post-processing table with {len(pre_table)} data rows")
            final_table = self.table_processing_service.post_process_table(pre_table, header_row)

            if not final_table or len(final_table) <= 1:  # Only header or empty
                # Try fallback extraction
                fallback_result = self._try_fallback_extraction(all_rows, all_spans)
                if fallback_result:
                    return fallback_result
                # Only use text extraction if no valid tables found
                text_result = await self.text_only_extraction_service.extract_text_from_pdf(pdf_path)
                if text_result and text_result.get("full_text"):
                    self.logger.info("No valid tables found, extracted text content instead")
                    return {"no_table": True, "text_content": text_result}
                raise NoTableFoundException(pdf_path, details={"reason": "No valid table data found after processing"})

            # Validate table quality to filter false positives (plain text laid out as rows)
            try:
                quality_score = self._score_fallback_table(final_table)
            except Exception:
                quality_score = 0.0

            # Validate columns and quality (support legitimate single-column tables)
            cols_detected = (len(final_table[0]) if final_table else 0)
            data_rows_count = max(0, len(final_table) - 1)

            # Define helper functions for table quality analysis
            def _short_word_ratio(rows):
                """Calculate ratio of short words (<= WORD_TOKEN_MAX_LEN) in all cells."""
                total_cells = 0
                short_cells = 0
                for r in rows:
                    for cell in r:
                        token = str(cell).strip()
                        if not token:
                            continue
                        total_cells += 1
                        if len(token) <= WORD_TOKEN_MAX_LEN:
                            short_cells += 1
                return (short_cells / total_cells) if total_cells else 0.0

            def _numeric_char_ratio(rows):
                """Calculate ratio of numeric characters in all cells."""
                total_chars = 0
                numeric_chars = 0
                for r in rows:
                    for cell in r:
                        s = str(cell)
                        total_chars += len(s)
                        numeric_chars += sum(1 for c in s if c.isdigit())
                return (numeric_chars / total_chars) if total_chars else 0.0

            def _alpha_char_ratio(rows):
                """Calculate ratio of alphabetic characters in all cells."""
                total_chars = 0
                alpha_chars = 0
                for r in rows:
                    for cell in r:
                        s = str(cell)
                        total_chars += len(s)
                        alpha_chars += sum(1 for c in s if c.isalpha())
                return (alpha_chars / total_chars) if total_chars else 0.0

            def _avg_token_length(rows):
                """Calculate average token length across all cells."""
                tokens = []
                for r in rows:
                    for cell in r:
                        for t in str(cell).split():
                            if t:
                                tokens.append(t)
                if not tokens:
                    return 0.0
                return sum(len(t) for t in tokens) / len(tokens)

            data_rows_only = final_table[1:]  # exclude header
            short_ratio = _short_word_ratio(data_rows_only)
            num_char_ratio = _numeric_char_ratio(data_rows_only)
            alpha_ratio = _alpha_char_ratio(data_rows_only)
            avg_token_len = _avg_token_length(data_rows_only)

            if cols_detected < 2:
                # Heuristics for single-column table acceptance
                min_rows_single_col = SINGLE_COL_MIN_ROWS
                min_quality_single_col = SINGLE_COL_MIN_QUALITY
                if (
                    data_rows_count < min_rows_single_col
                    or quality_score < min_quality_single_col
                    or short_ratio >= SHORT_WORD_RATIO_MIN
                ) and num_char_ratio < TABLE_MIN_NUMERIC_CHAR_RATIO:
                    self.logger.info(
                        f"Rejecting single-col as table (rows={data_rows_count}, score={quality_score:.2f}, short_ratio={short_ratio:.2f}, num_ratio={num_char_ratio:.3f}); returning text"
                    )
                    text_result = await self.text_only_extraction_service.extract_text_from_pdf(pdf_path)
                    if text_result and text_result.get("full_text"):
                        return {"no_table": True, "text_content": text_result}
            elif (
                quality_score < TABLE_MIN_QUALITY
                or (short_ratio >= SHORT_WORD_RATIO_MIN and num_char_ratio < TABLE_MIN_NUMERIC_CHAR_RATIO)
                or (
                    cols_detected >= PARAGRAPH_LIKE_MIN_COLS
                    and alpha_ratio >= ALPHA_CHAR_RATIO_MIN
                    and AVG_TOKEN_LEN_MIN <= avg_token_len <= AVG_TOKEN_LEN_MAX
                )
            ):
                self.logger.info(
                    f"Rejecting low-quality/paragraph-like table (cols={cols_detected}, score={quality_score:.2f}, short_ratio={short_ratio:.2f}, num_ratio={num_char_ratio:.3f}, alpha_ratio={alpha_ratio:.2f}, avg_token_len={avg_token_len:.2f}); returning text"
                )
                text_result = await self.text_only_extraction_service.extract_text_from_pdf(pdf_path)
                if text_result and text_result.get("full_text"):
                    return {"no_table": True, "text_content": text_result}

            self.logger.info(f"Successfully extracted table with {len(final_table)} rows")
            
            # Check if we should also extract additional text content
            if len(final_table) > 1:  # Tables found
                try:
                    text_result = await self.text_only_extraction_service.extract_text_from_pdf(pdf_path)
                    if text_result and text_result.get("full_text"):
                        # Check if there's significant text content beyond the table
                        table_text_length = sum(len(str(cell)) for row in final_table for cell in row)
                        full_text_length = len(text_result["full_text"])
                        
                        # If text is significantly longer than table content, include both
                        if full_text_length > table_text_length * 2:
                            self.logger.info("Found both tables and additional text content")
                            # Add text content as additional rows
                            final_table.append(["--- Additional Text Content ---"])
                            final_table.append([text_result["full_text"]])
                except Exception as e:
                    self.logger.warning(f"Failed to extract additional text: {e}")
            
            return final_table
            
        except Exception as e:
            # Try fallback extraction before giving up
            if 'all_rows' in locals() and 'all_spans' in locals():
                fallback_result = self._try_fallback_extraction(all_rows, all_spans)
                if fallback_result:
                    self.logger.info("Fallback extraction succeeded after main extraction failed")
                    return fallback_result
            
            # Only use text extraction if table extraction completely failed
            text_result = await self.text_only_extraction_service.extract_text_from_pdf(pdf_path)
            if text_result and text_result.get("full_text"):
                self.logger.info("Table extraction failed, extracted text content instead")
                return {"no_table": True, "text_content": text_result}
            
            self.logger.error(f"Table extraction failed: {e}", exc_info=True)
            raise
            
        finally:
            if doc is not None:
                doc.close()

    def _try_fallback_extraction(self, all_rows, all_spans):
        """Extract all tables from document without gaps"""
        try:
            if not all_rows or len(all_rows) < 2:
                return None
            
            # Detect all table regions
            table_regions = self._detect_all_table_regions(all_rows)
            
            if not table_regions:
                return self._extract_single_fallback_table(all_rows)
            
            # Extract all tables and combine into one comprehensive table
            all_tables = []
            for region in table_regions:
                table_data = self._extract_clean_table(region['rows'])
                if table_data and len(table_data) >= 2:
                    all_tables.extend(table_data)
            
            if all_tables:
                self.logger.info(f"Extracted {len(table_regions)} tables with total {len(all_tables)} rows")
                return all_tables
            
            return self._extract_single_fallback_table(all_rows)
            
        except Exception as e:
            self.logger.warning(f"Fallback extraction failed: {e}")
            return self._extract_single_fallback_table(all_rows)

    def _extract_single_fallback_table(self, all_rows):
        """Original single table fallback extraction"""
        try:
            fallback_table = []
            header_row = None
            
            for row in all_rows:
                row_data = [span["text"].strip() for span in row if span["text"].strip()]
                if len(row_data) >= 1:  # At least 1 non-empty cell (lowered requirement)
                    # Use existing header detection logic
                    if header_row and self.table_processing_service.is_repeat_header(row_data, header_row):
                        continue
                    if not header_row:
                        header_row = row_data
                    fallback_table.append(row_data)
            
            if len(fallback_table) >= 1:  # Accept even single row tables
                self.logger.info(f"Single fallback extraction found {len(fallback_table)} rows")
                return fallback_table
            
            return None
        except Exception as e:
            self.logger.warning(f"Single fallback extraction failed: {e}")
            return None

    def _is_scanned_pdf(self, pdf_path):
        """Check if PDF is scanned (image-based) by analyzing text content"""
        try:
            doc = self.ocr_service.open_pdf(pdf_path)
            total_chars = 0
            
            for page in doc:
                text = page.get_text()
                total_chars += len(text.strip())
                
            doc.close()
            
            # If very little text found, likely scanned
            is_scanned = total_chars < SCANNED_TEXT_MIN_CHARS
            self.logger.info(f"PDF analysis: {total_chars} characters found, scanned: {is_scanned}")
            return is_scanned
            
        except Exception as e:
            self.logger.warning(f"Scanned PDF detection failed: {e}")
            return False

    def _extract_spatial_table(self, spans):
        """Extract table using spatial positioning for scanned PDFs"""
        try:
            # Sort spans by Y position first, then X position
            spans.sort(key=lambda s: (s["y0"], s["x0"]))
            
            # Group spans into rows by Y position
            rows = []
            current_row = [spans[0]]
            current_y = spans[0]["y0"]
            
            for span in spans[1:]:
                # If Y position is close, add to current row
                if abs(span["y0"] - current_y) < SPATIAL_SAME_LINE_TOL_PX:  # tolerance from config
                    current_row.append(span)
                else:
                    # Sort current row by X position and add to rows
                    current_row.sort(key=lambda s: s["x0"])
                    rows.append([s["text"] for s in current_row])
                    current_row = [span]
                    current_y = span["y0"]
            
            # Add last row
            if current_row:
                current_row.sort(key=lambda s: s["x0"])
                rows.append([s["text"] for s in current_row])
            
            if rows:
                self.logger.info(f"Spatial extraction found {len(rows)} rows")
                return rows
            
            return None
        except Exception as e:
            self.logger.warning(f"Spatial extraction failed: {e}")
            return None

    async def _convert_to_searchable_pdf(self, pdf_path):
        """Convert scanned PDF to searchable PDF using ocrmypdf"""
        try:
            ocr_pdf_path = pdf_path.replace('.pdf', '_ocr.pdf')
            
            if ocrmypdf:
                # Use Python library
                ocrmypdf.ocr(pdf_path, ocr_pdf_path, 
                           force_ocr=True, 
                           deskew=True, 
                           clean=True,
                           optimize=1,
                           tesseract_config=['--psm', '6'])
                self.logger.info(f"Successfully converted scanned PDF to searchable: {ocr_pdf_path}")
                return ocr_pdf_path
            else:
                # Fallback to command line
                cmd = ['ocrmypdf', '--force-ocr', '--deskew', '--clean', pdf_path, ocr_pdf_path]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0 and os.path.exists(ocr_pdf_path):
                    self.logger.info(f"Successfully converted scanned PDF to searchable: {ocr_pdf_path}")
                    return ocr_pdf_path
                else:
                    self.logger.warning(f"OCR conversion failed: {result.stderr}")
                    return None
                
        except Exception as e:
            self.logger.warning(f"OCR conversion error: {e}")
            return None

    def _detect_all_table_regions(self, all_rows):
        """Detect all table regions"""
        return []
    
    def _extract_clean_table(self, rows):
        """Extract clean table from rows"""
        return None

    def _try_fallback_extraction(self, all_rows, all_spans):
        """Extract all tables from document without gaps"""
        try:
            if not all_rows or len(all_rows) < 2:
                return None
            
            # Detect all table regions
            table_regions = self._detect_all_table_regions(all_rows)
            
            if not table_regions:
                return self._extract_single_fallback_table(all_rows)
            
            # Extract all tables and combine into one comprehensive table
            all_tables = []
            for region in table_regions:
                table_data = self._extract_clean_table(region['rows'])
                if table_data and len(table_data) >= 2:
                    all_tables.extend(table_data)
            
            if all_tables:
                self.logger.info(f"Extracted {len(table_regions)} tables with total {len(all_tables)} rows")
                return all_tables
            
            return self._extract_single_fallback_table(all_rows)
            
        except Exception as e:
            self.logger.warning(f"Fallback extraction failed: {e}")
            return self._extract_single_fallback_table(all_rows)

    @log_method_entry_exit
    @monitor_performance
    @handle_general_operations(severity=ExceptionSeverity.MEDIUM)
    async def extract_table_simple(self, file):
        """Simplified table extraction endpoint"""
        temp_files = []
        
        try:
            # Validate file type
            self.file_validation_service.validate_pdf_file(file)
            
            # Save uploaded file
            file_paths = await self.file_upload_service.save_uploaded_file(file)
            temp_files.append(file_paths["pdf_path"])
            
            # Extract table using new services
            table_data = await self._extract_table_data(file_paths["pdf_path"])
            
            if not table_data:
                self.logger.warning(f"No table data found in document: {file.filename}")
                raise NoTableFoundException(file_paths["pdf_path"])
            
# If the result is a structured text fallback (no tables), return text JSON response
            if isinstance(table_data, dict) and table_data.get("no_table", False):
                response = {
                    "success": True,
                    "message": f"No tables detected in {file.filename}. Returning extracted text.",
                    "data_type": "text",
                    "text_content": table_data.get("text_content", {}),
                    "rows": 0,
                    "columns": 0
                }
            else:
                # Prepare simplified response
                response = {
                    "success": True,
                    "message": f"Successfully extracted table from {file.filename}",
                    "data": table_data,
                    "rows": len(table_data) - 1,  # Exclude header
                    "columns": len(table_data[0]) if table_data else 0
                }
            
            self.logger.info(f"Successfully processed {file.filename}: {response['rows']} rows extracted")
            return response
            
        except Exception as e:
            self.logger.error(f"Simple extraction failed: {e}")
            # Cleanup temp files on error
            self.file_upload_service.cleanup_temp_files(temp_files)
            raise

    @log_method_entry_exit
    @monitor_performance
    @handle_general_operations(severity=ExceptionSeverity.HIGH)
    async def extract_all_content_from_pdf(self, file):
        """Extract all text content from PDF - works with various templates and formats"""
        temp_files = []
        
        try:
            # Validate file type
            self.file_validation_service.validate_pdf_file(file)
            
            # Save uploaded file
            file_paths = await self.file_upload_service.save_uploaded_file(file)
            temp_files.append(file_paths["pdf_path"])
            
            # Extract all text content using comprehensive method
            content_result = await self._extract_all_content_data(file_paths["pdf_path"])
            
            if not content_result or not content_result.get("extracted_text"):
                self.logger.warning(f"No readable content found in document: {file.filename}")
                # Try one more fallback with basic file reading
                try:
                    with open(file_paths["pdf_path"], 'rb') as f:
                        # Check if file is readable
                        f.read(1024)
                    raise NoTableFoundException(file_paths["pdf_path"], details={
                        "reason": "PDF appears to be image-based or corrupted",
                        "suggestion": "Try converting to searchable PDF first"
                    })
                except Exception:
                    raise NoTableFoundException(file_paths["pdf_path"], details={
                        "reason": "Unable to read PDF file",
                        "suggestion": "Check if file is valid PDF format"
                    })
            
            # Return the extracted text directly
            extracted_text = content_result.get("extracted_text", "")
            
            # Basic validation of extracted content
            if len(extracted_text.strip()) < 10:
                self.logger.warning(f"Very little content extracted from {file.filename}: {len(extracted_text)} characters")
            
            response = {
                "extracted_text": extracted_text
            }
            
            self.logger.info(f"Successfully processed {file.filename}: {len(extracted_text)} characters extracted")
            return response
            
        except Exception as e:
            self.logger.error(f"Content extraction failed for {file.filename}: {e}")
            # Cleanup temp files on error
            self.file_upload_service.cleanup_temp_files(temp_files)
            raise

    async def _extract_all_content_data(self, pdf_path):
        """Comprehensive PDF text extraction - extracts ALL readable text from the document"""
        try:
            self.logger.info(f"Starting complete text extraction from: {pdf_path}")
            
            all_extracted_text = []
            extraction_methods = []
            
            # Method 1: PyMuPDF comprehensive extraction
            try:
                import fitz
                doc = fitz.open(pdf_path)
                
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    page_text_parts = []
                    
                    # Extract using multiple methods and combine
                    methods_tried = []
                    
                    # 1. Standard text extraction
                    try:
                        standard_text = page.get_text()
                        if standard_text.strip():
                            page_text_parts.append(standard_text)
                            methods_tried.append("standard")
                    except:
                        pass
                    
                    # 2. Dictionary-based detailed extraction (CRITICAL for Canva PDFs)
                    try:
                        text_dict = page.get_text("dict", flags=11)  # flags=11 for comprehensive extraction
                        dict_text = ""
                        for block in text_dict.get("blocks", []):
                            if block.get("type") == 0:  # Text block
                                for line in block.get("lines", []):
                                    line_text = ""
                                    for span in line.get("spans", []):
                                        span_text = span.get("text", "")
                                        if span_text.strip():
                                            line_text += span_text + " "
                                    if line_text.strip():
                                        dict_text += line_text.strip() + "\n"
                        
                        if dict_text.strip():
                            page_text_parts.append(dict_text)
                            methods_tried.append("dictionary")
                    except:
                        pass
                    
                    # 2b. Extract from structured content tree (for Canva PDFs with StructTreeRoot)
                    try:
                        struct_text = ""
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
                                        struct_text += " ".join(line_parts) + "\n"
                        
                        if struct_text.strip():
                            page_text_parts.append(struct_text)
                            methods_tried.append("structured")
                    except:
                        pass
                    
                    # 2c. Try multiple PyMuPDF extraction modes
                    try:
                        # Try raw extraction
                        raw_text = page.get_text("raw")
                        if raw_text.strip():
                            page_text_parts.append(raw_text)
                            methods_tried.append("raw")
                    except:
                        pass
                    
                    try:
                        # Try layout extraction
                        layout_text = page.get_text("layout")
                        if layout_text.strip():
                            page_text_parts.append(layout_text)
                            methods_tried.append("layout")
                    except:
                        pass
                    
                    # 3. Block-based extraction
                    try:
                        blocks = page.get_text("blocks")
                        block_text = ""
                        for block in blocks:
                            if len(block) >= 5:  # Valid text block
                                text = block[4].strip()
                                if text:
                                    block_text += text + "\n"
                        
                        if block_text.strip() and len(block_text) > len(page_text_parts[0] if page_text_parts else ""):
                            page_text_parts = [block_text]  # Replace if better
                            methods_tried = ["blocks"]
                    except:
                        pass
                    
                    # Combine all extractions to get complete text (don't just use the longest)
                    if page_text_parts:
                        # Combine all unique text parts
                        combined_text = ""
                        seen_text = set()
                        for text_part in page_text_parts:
                            # Add unique lines to avoid duplicates
                            for line in text_part.split('\n'):
                                line_stripped = line.strip()
                                if line_stripped and line_stripped.lower() not in seen_text:
                                    combined_text += line_stripped + "\n"
                                    seen_text.add(line_stripped.lower())
                        
                        if combined_text.strip():
                            all_extracted_text.append(f"\n=== PAGE {page_num + 1} ===\n{combined_text.strip()}")
                            self.logger.debug(f"Page {page_num + 1}: Extracted {len(combined_text)} chars using {methods_tried}")
                        else:
                            # Fallback to longest if combination failed
                            best_text = max(page_text_parts, key=len)
                            all_extracted_text.append(f"\n=== PAGE {page_num + 1} ===\n{best_text.strip()}")
                            self.logger.debug(f"Page {page_num + 1}: Using best extraction: {len(best_text)} chars")
                
                doc.close()
                
                if all_extracted_text:
                    pymupdf_result = "\n\n".join(all_extracted_text)
                    extraction_methods.append("PyMuPDF-Complete")
                    self.logger.info(f"PyMuPDF complete extraction: {len(pymupdf_result)} characters")
                    
                    # Clean and return immediately if we got good content
                    if len(pymupdf_result) > 500:  # Substantial content
                        final_text = self._clean_extracted_text(pymupdf_result)
                        return {"extracted_text": final_text}
                    
            except Exception as e:
                self.logger.warning(f"PyMuPDF complete extraction failed: {e}")
            
            # Method 2: pdfplumber comprehensive extraction
            try:
                import pdfplumber
                with pdfplumber.open(pdf_path) as pdf:
                    plumber_pages = []
                    
                    for page_num, page in enumerate(pdf.pages):
                        page_methods = []
                        
                        # Try standard extraction
                        try:
                            text1 = page.extract_text()
                            if text1 and text1.strip():
                                page_methods.append(("standard", text1))
                        except:
                            pass
                        
                        # Try layout-aware extraction
                        try:
                            text2 = page.extract_text(layout=True)
                            if text2 and text2.strip():
                                page_methods.append(("layout", text2))
                        except:
                            pass
                        
                        # Try word-based reconstruction
                        try:
                            words = page.extract_words()
                            if words:
                                reconstructed = self._reconstruct_from_words(words)
                                if reconstructed and reconstructed.strip():
                                    page_methods.append(("words", reconstructed))
                        except:
                            pass
                        
                        # Use the method that extracted the most text
                        if page_methods:
                            best_method, best_text = max(page_methods, key=lambda x: len(x[1]))
                            plumber_pages.append(f"\n=== PAGE {page_num + 1} ===\n{best_text.strip()}")
                    
                    if plumber_pages:
                        plumber_result = "\n\n".join(plumber_pages)
                        extraction_methods.append("pdfplumber-Complete")
                        self.logger.info(f"pdfplumber complete extraction: {len(plumber_result)} characters")
                        
                        # Use if better than PyMuPDF or if PyMuPDF failed
                        if len(plumber_result) > len("\n\n".join(all_extracted_text) if all_extracted_text else ""):
                            final_text = self._clean_extracted_text(plumber_result)
                            return {"extracted_text": final_text}
                        
            except Exception as e:
                self.logger.warning(f"pdfplumber complete extraction failed: {e}")
            
            # Method 3: PyPDF2 as fallback
            try:
                import PyPDF2
                with open(pdf_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    pypdf2_pages = []
                    
                    for page_num, page in enumerate(reader.pages):
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            pypdf2_pages.append(f"\n=== PAGE {page_num + 1} ===\n{page_text.strip()}")
                    
                    if pypdf2_pages:
                        pypdf2_result = "\n\n".join(pypdf2_pages)
                        extraction_methods.append("PyPDF2-Complete")
                        self.logger.info(f"PyPDF2 complete extraction: {len(pypdf2_result)} characters")
                        
                        final_text = self._clean_extracted_text(pypdf2_result)
                        return {"extracted_text": final_text}
                        
            except Exception as e:
                self.logger.warning(f"PyPDF2 complete extraction failed: {e}")
            
            # If we have any extracted text, return it
            if all_extracted_text:
                final_result = "\n\n".join(all_extracted_text)
                final_text = self._clean_extracted_text(final_result)
                self.logger.info(f"Returning partial extraction: {len(final_text)} characters")
                return {"extracted_text": final_text}
            
            # Method 4: OCR as last resort
            try:
                if self._is_scanned_pdf(pdf_path):
                    self.logger.info("Attempting OCR extraction...")
                    ocr_pdf_path = await self._convert_to_searchable_pdf(pdf_path)
                    if ocr_pdf_path and ocr_pdf_path != pdf_path:
                        ocr_result = await self._extract_all_content_data(ocr_pdf_path)
                        if ocr_result and ocr_result.get("extracted_text"):
                            return ocr_result
            except Exception as e:
                self.logger.warning(f"OCR extraction failed: {e}")
            
            raise NoTableFoundException(pdf_path, details={
                "reason": "No readable text content found in PDF",
                "methods_tried": extraction_methods,
                "suggestion": "PDF may be image-based or corrupted"
            })
            
        except Exception as e:
            self.logger.error(f"Complete text extraction failed: {e}", exc_info=True)
            raise
    
    def _reconstruct_from_words(self, words):
        """Reconstruct text from word objects with proper spacing and line breaks"""
        try:
            if not words:
                return ""
            
            # Sort words by position (top to bottom, left to right)
            sorted_words = sorted(words, key=lambda w: (w.get('top', 0), w.get('x0', 0)))
            
            lines = []
            current_line = []
            current_top = None
            line_tolerance = 3  # pixels
            
            for word in sorted_words:
                word_text = word.get('text', '').strip()
                if not word_text:
                    continue
                
                word_top = word.get('top', 0)
                
                # Check if this word starts a new line
                if current_top is None or abs(word_top - current_top) > line_tolerance:
                    # Save previous line
                    if current_line:
                        lines.append(' '.join(current_line))
                    current_line = [word_text]
                    current_top = word_top
                else:
                    # Same line
                    current_line.append(word_text)
            
            # Add the last line
            if current_line:
                lines.append(' '.join(current_line))
            
            return '\n'.join(lines)
            
        except Exception as e:
            self.logger.debug(f"Word reconstruction failed: {e}")
            return ""
    
    def _clean_extracted_text(self, text):
        """Clean and format extracted text for better readability"""
        try:
            import re
            
            # Remove page separators for cleaner output
            cleaned = re.sub(r'\n=== PAGE \d+ ===\n', '\n\n', text)
            
            # Fix spacing issues
            cleaned = re.sub(r'[ \t]+', ' ', cleaned)  # Multiple spaces to single
            cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)  # Multiple newlines to double
            
            # Fix common extraction artifacts
            cleaned = re.sub(r'([a-z])([A-Z])', r'\1 \2', cleaned)  # camelCase spacing
            cleaned = re.sub(r'(\w)(\d)', r'\1 \2', cleaned)  # word-number spacing
            cleaned = re.sub(r'(\d)(\w)', r'\1 \2', cleaned)  # number-word spacing
            
            # Clean up email and phone formatting
            cleaned = re.sub(r'(\+\d{1,3})\s*(\d)', r'\1 \2', cleaned)
            
            return cleaned.strip()
            
        except Exception as e:
            self.logger.debug(f"Text cleaning failed: {e}")
            return text

    async def _extract_table_from_region(self, table_region, all_spans):
        """Extract table data from a specific region"""
        try:
            region_spans = table_region.get("spans", [])
            if not region_spans:
                return None
            
            # Use advanced table detection
            detected_tables = self.table_processing_service.advanced_table_detection(region_spans)
            
            if detected_tables:
                # Get the best table (highest quality score)
                best_table = detected_tables[0]
                return {
                    "data": best_table.get("data", []),
                    "method": best_table.get("method", "unknown"),
                    "confidence": best_table.get("confidence", 0.0),
                    "quality_score": best_table.get("quality_score", 0.0),
                    "bounds": table_region.get("bounds", {}),
                    "region_confidence": table_region.get("confidence", 0.0)
                }
            
            # Fallback to traditional method
            rows = self.table_processing_service.cluster_rows(region_spans)
            if len(rows) >= 2:
                columns = self.table_processing_service.detect_template_columns(rows)
                if columns:
                    global_min_x = min(s["x0"] for s in region_spans)
                    global_max_x = max(s["x1"] for s in region_spans)
                    global_bounds = self.table_processing_service.centers_to_bounds(columns, global_min_x, global_max_x)
                    table_data = self.table_processing_service.build_table_with_bounds(rows, global_bounds, columns)
                    
                    if len(table_data) > 1:
                        return {
                            "data": table_data,
                            "method": "traditional_fallback",
                            "confidence": 0.6,
                            "quality_score": 0.6,
                            "bounds": table_region.get("bounds", {}),
                            "region_confidence": table_region.get("confidence", 0.0)
                        }
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Table extraction from region failed: {e}")
            return None

    def _extract_best_header_dynamically(self, table, current_index, expected_cols):
        """Extract and merge multi-line English headers, handling multi-word headers properly"""
        try:
            if current_index >= len(table):
                return None
            
            current_row = [cell.strip() for cell in table[current_index]]
            
            # First, check if any cells in current row have newlines that might be multi-word headers
            # For example: "Joining\nDate" should become "Joining Date"
            processed_row = []
            for cell in current_row:
                if '\n' in cell:
                    # Check if this looks like a multi-word header (all parts are header-like)
                    parts = [p.strip() for p in cell.split('\n') if p.strip()]
                    if len(parts) >= 2:
                        # Generic check: Are all parts likely header components?
                        if self._are_parts_multi_word_header(parts):
                            # Merge into single header: "Joining Date"
                            processed_row.append(" ".join(parts))
                        else:
                            # Keep as is (might be header + data)
                            processed_row.append(cell)
                    else:
                        processed_row.append(cell)
                else:
                    processed_row.append(cell)
            
            current_row = processed_row
            
            # Check if current row is English header
            if self.table_processing_service.is_english_row(current_row):
                # Check if next row continues the header (multi-line)
                if current_index + 1 < len(table):
                    next_row = [cell.strip() for cell in table[current_index + 1]]
                    if self.table_processing_service.is_english_row(next_row):
                        return self._merge_header_rows(current_row, next_row)
                return current_row
            
            # Check next row for English header
            if current_index + 1 < len(table):
                next_row = [cell.strip() for cell in table[current_index + 1]]
                if self.table_processing_service.is_likely_header(next_row,
                                                                  expected_cols) and self.table_processing_service.is_english_row(
                        next_row):
                    return next_row
            
            return None
                
        except Exception as e:
            self.logger.warning(f"Dynamic header extraction failed: {e}")
            return None
    
    def _is_header_continuation(self, first_row, second_row):
        """Check if second row continues first row header"""
        try:
            # If either row is empty, it's not a continuation
            if not first_row or not second_row:
                return False

            first_cells = [str(c).strip() for c in first_row if str(c).strip()]
            second_cells = [str(c).strip() for c in second_row if str(c).strip()]
            if not first_cells or not second_cells:
                return False

            # Use the table_processing service header-like detector if available, else fall back
            header_like_fn = getattr(self.table_processing_service, "is_header_like", None) or self._is_header_like

            # If second row has many header-like tokens compared to its length, consider it a continuation
            header_like_count = sum(1 for c in second_cells if header_like_fn(c))
            if header_like_count >= max(1, len(second_cells) // 2):
                # Ensure second row is not predominantly numeric (which would indicate data)
                numeric_count = sum(1 for c in second_cells if self._is_numeric_content(c))
                if numeric_count / max(1, len(second_cells)) < 0.5:
                    return True

            # If second row is noticeably shorter and both rows look English/header-like,
            # treat as continuation (covers split headers across lines)
            if len(second_cells) < len(first_cells) and header_like_fn(" ".join(first_cells)) and header_like_fn(" ".join(second_cells)):
                return True

            return False
        except Exception:
            return False
    
    def _merge_header_rows(self, first_row, second_row):
        """Merge two header rows"""
        merged = []
        for i in range(max(len(first_row), len(second_row))):
            first_cell = first_row[i] if i < len(first_row) else ""
            second_cell = second_row[i] if i < len(second_row) else ""
            
            if first_cell.strip() and second_cell.strip():
                merged.append(f"{first_cell.strip()} {second_cell.strip()}")
            else:
                merged.append((first_cell or second_cell).strip())
        return merged
    
    def _are_parts_multi_word_header(self, parts):
        """
        Generic method to determine if parts should be merged as a multi-word header.
        Uses heuristics instead of hardcoded word lists.
        """
        if not parts or len(parts) < 2:
            return False
        
        # Check if all parts are header-like
        all_header_like = all(
            self.table_processing_service.is_header_like(part) 
            for part in parts
        )
        
        if not all_header_like:
            return False
        
        # Additional heuristics:
        # 1. All parts should be relatively short (typical header words)
        avg_length = sum(len(part) for part in parts) / len(parts)
        if avg_length > 20:  # Too long for typical header words
            return False
        
        # 2. Parts should have similar characteristics (all capitalized, all lowercase, etc.)
        capitalization_patterns = []
        for part in parts:
            if part and part[0].isupper():
                capitalization_patterns.append('title')
            elif part.isupper():
                capitalization_patterns.append('upper')
            elif part.islower():
                capitalization_patterns.append('lower')
            else:
                capitalization_patterns.append('mixed')
        
        # If patterns are consistent, more likely to be a multi-word header
        pattern_consistency = len(set(capitalization_patterns)) <= 2
        
        # 3. Parts should not look like data (dates, numbers, etc.)
        looks_like_data = any(
            self._looks_like_data_value(part) for part in parts
        )
        
        # 4. Combined length should be reasonable for a header
        combined = " ".join(parts)
        reasonable_header_length = 3 <= len(combined) <= 50
        
        return (pattern_consistency and 
                not looks_like_data and 
                reasonable_header_length and
                all_header_like)
    
    def _should_merge_parts_with_header(self, current_header, parts, col_idx, header_row):
        """
        Generic method to determine if parts should be merged with existing header.
        Uses heuristics to detect incomplete headers.
        """
        if not parts or len(parts) < 2:
            return False
        
        # If header is empty or very short, first part might be header
        if not current_header or len(current_header.strip()) < 3:
            first_part = parts[0].strip()
            return (self.table_processing_service.is_header_like(first_part) and
                    not self.table_processing_service.is_header_like(parts[1]))
        
        # If header exists but seems incomplete
        header_words = current_header.split()
        if len(header_words) <= 2:  # Short header might be incomplete
            first_part = parts[0].strip()
            
            # Check if first part complements the header
            # Heuristics:
            # 1. First part should be header-like
            # 2. First part should not be data-like
            # 3. Combined should make sense as a header
            if (self.table_processing_service.is_header_like(first_part) and
                not self._looks_like_data_value(first_part)):
                
                # Check if combining makes sense (similar capitalization, reasonable length)
                combined = f"{current_header} {first_part}".strip()
                if 3 <= len(combined) <= 50:
                    # Check if second part looks like data (not header)
                    if len(parts) > 1 and not self.table_processing_service.is_header_like(parts[1]):
                        return True
        
        return False
    
    def _is_likely_header_part(self, part, current_header, index, total_parts):
        """
        Generic method to determine if a part is likely a header component.
        """
        if not part or not part.strip():
            return False
        
        part = part.strip()
        
        # Must be header-like
        if not self.table_processing_service.is_header_like(part):
            return False
        
        # Should not look like data
        if self._looks_like_data_value(part):
            return False
        
        # If it's the first part and header is incomplete, more likely to be header
        if index == 0 and current_header and len(current_header.split()) <= 2:
            # Check capitalization consistency
            header_first_char = current_header[0] if current_header else ''
            part_first_char = part[0] if part else ''
            
            # Similar capitalization suggests continuation
            if (header_first_char.isupper() == part_first_char.isupper() or
                header_first_char.islower() == part_first_char.islower()):
                return True
        
        # Reasonable length for header word
        return 2 <= len(part) <= 25
    
    def _looks_like_data_value(self, text):
        """
        Generic method to determine if text looks like a data value rather than header.
        """
        if not text:
            return False
        
        text = text.strip()
        
        # Check for date patterns (YYYY-MM-DD, MM/DD/YYYY, etc.)
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',   # MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}',   # MM-DD-YYYY
            r'\d{1,2}/\d{1,2}/\d{2,4}',  # Various date formats
        ]
        if any(re.search(pattern, text) for pattern in date_patterns):
            return True
        
        # Check for numeric patterns (currency, numbers, etc.)
        if re.match(r'^[\d,.$€£¥]+$', text.replace(' ', '')):
            return True
        
        # Check for email patterns
        if '@' in text and '.' in text:
            return True
        
        # Check for URL patterns
        if text.startswith('http://') or text.startswith('https://') or text.startswith('www.'):
            return True
        
        # Very long text is likely data, not header
        if len(text) > 30:
            return True
        
        return False
    
    def _merge_split_headers_in_row(self, header_row, table, header_idx, expected_cols):
        """
        Generic method to merge headers that are split across adjacent cells in the same row.
        Example: ["ID", "Name", "Department", "Salary", "Joining", "Date"] 
                 -> ["ID", "Name", "Department", "Salary", "Joining Date", ""]
        """
        if not header_row or len(header_row) < 2:
            return header_row
        
        merged_header = list(header_row)  # Work with a copy
        i = 0
        
        while i < len(merged_header) - 1:
            current_cell = merged_header[i].strip()
            next_cell = merged_header[i + 1].strip() if i + 1 < len(merged_header) else ""
            
            # Check if current cell is incomplete and next cell should be merged
            if current_cell and next_cell:
                # Heuristics to determine if they should be merged:
                # 1. Current cell is short (likely incomplete header)
                # 2. Next cell is header-like
                # 3. Combined makes sense as a header
                # 4. Next cell doesn't look like data
                
                current_words = len(current_cell.split())
                next_words = len(next_cell.split())
                
                # If current is short (1-2 words) and next is header-like
                if (current_words <= 2 and 
                    next_words <= 2 and
                    self.table_processing_service.is_header_like(next_cell) and
                    not self._looks_like_data_value(next_cell)):
                    
                    # Check if combining makes sense
                    combined = f"{current_cell} {next_cell}".strip()
                    if (3 <= len(combined) <= 50 and
                        self.table_processing_service.is_header_like(combined)):
                        
                        # Check capitalization consistency
                        current_first = current_cell[0] if current_cell else ''
                        next_first = next_cell[0] if next_cell else ''
                        
                        # Similar capitalization suggests they belong together
                        if (current_first.isupper() == next_first.isupper() or
                            (current_first.isupper() and next_first.isupper())):
                            
                            # Merge them
                            merged_header[i] = combined
                            merged_header[i + 1] = ""  # Clear the next cell
                            i += 2  # Skip the next cell since we merged it
                            continue
            
            i += 1
        
        return merged_header
    
    def _fix_split_headers_in_data_row(self, header_row, data_row, expected_cols):
        """
        Generic method to fix cases where header continuation words appear in the first data row.
        Example: Header=["Joining"], Data=["Date 2021-02-15"] 
                 -> Header=["Joining Date"], Data=["2021-02-15"]
        """
        if not header_row or not data_row:
            return header_row, data_row
        
        fixed_header = list(header_row)
        fixed_data = list(data_row)
        
        for col_idx in range(min(len(fixed_header), len(fixed_data))):
            header_cell = fixed_header[col_idx].strip() if col_idx < len(fixed_header) else ""
            data_cell = fixed_data[col_idx].strip() if col_idx < len(fixed_data) else ""
            
            if not header_cell or not data_cell:
                continue
            
            # Check if data cell starts with a header continuation word
            data_words = data_cell.split()
            if len(data_words) >= 2:
                first_word = data_words[0].strip()
                rest_data = " ".join(data_words[1:])
                
                # Check if first word should be merged with header
                if (self._is_likely_header_continuation(first_word, header_cell) and
                    not self._looks_like_data_value(first_word) and
                    self.table_processing_service.is_header_like(first_word)):
                    
                    # Check if rest looks like actual data
                    if rest_data and (self._looks_like_data_value(rest_data) or 
                                      len(rest_data) > len(first_word)):
                        # Merge first word with header, keep rest as data
                        fixed_header[col_idx] = f"{header_cell} {first_word}".strip()
                        fixed_data[col_idx] = rest_data
        
        return fixed_header, fixed_data
    
    def _is_likely_header_continuation(self, word, existing_header):
        """
        Generic method to determine if a word is likely a continuation of an existing header.
        """
        if not word or not existing_header:
            return False
        
        word = word.strip()
        existing_header = existing_header.strip()
        
        # Must be header-like
        if not self.table_processing_service.is_header_like(word):
            return False
        
        # Should not look like data
        if self._looks_like_data_value(word):
            return False
        
        # Existing header should be short (incomplete)
        header_words = existing_header.split()
        if len(header_words) > 3:  # Already complete header
            return False
        
        # Check capitalization consistency
        header_first = existing_header[0] if existing_header else ''
        word_first = word[0] if word else ''
        
        # Similar capitalization suggests continuation
        capitalization_match = (
            header_first.isupper() == word_first.isupper() or
            (header_first.isupper() and word_first.isupper())
        )
        
        # Word should be relatively short (typical header words)
        reasonable_length = 2 <= len(word) <= 20
        
        return capitalization_match and reasonable_length

    def _detect_multiple_table_regions(self, all_rows, all_spans):
        """Detect multiple table regions with different structures using enhanced detection"""
        try:
            if len(all_rows) < 2:  # Reduced minimum requirement
                return []
            
            regions = []
            current_region = []
            current_structure = None
            
            # Track spatial gaps to detect table boundaries
            y_positions = []
            for row in all_rows:
                if row:
                    avg_y = sum(span.get("y_center", span.get("y0", 0)) for span in row) / len(row)
                    y_positions.append(avg_y)
            
            # Detect large vertical gaps that might indicate table boundaries
            gap_threshold = self._calculate_gap_threshold(y_positions)
            
            for i, row in enumerate(all_rows):
                row_data = [span["text"].strip() for span in row if span["text"].strip()]
                if len(row_data) < 1:  # Allow single column tables
                    # Check if this is a significant gap
                    if current_region and i < len(y_positions) - 1:
                        current_y = y_positions[i] if i < len(y_positions) else 0
                        next_y = y_positions[i + 1] if i + 1 < len(y_positions) else 0
                        if abs(next_y - current_y) > gap_threshold:
                            # Large gap detected - end current region
                            if len(current_region) >= 2:
                                regions.append({
                                    'rows': current_region,
                                    'structure': current_structure,
                                    'start_index': i - len(current_region),
                                    'end_index': i - 1
                                })
                            current_region = []
                            current_structure = None
                    continue
                
                # Analyze row structure
                row_structure = self._analyze_row_structure(row_data)
                
                # Check for table boundary indicators
                is_boundary = self._is_table_boundary(row_data, current_region, i, y_positions, gap_threshold)
                
                # Check if structure changed significantly or boundary detected
                if (current_structure and (self._structure_changed(current_structure, row_structure) or is_boundary)):
                    # Save current region if it has enough rows
                    if len(current_region) >= 2:
                        regions.append({
                            'rows': current_region,
                            'structure': current_structure,
                            'start_index': i - len(current_region),
                            'end_index': i - 1
                        })
                    current_region = [row_data]
                    current_structure = row_structure
                else:
                    current_region.append(row_data)
                    if not current_structure:
                        current_structure = row_structure
            
            # Add the last region
            if len(current_region) >= 2:
                regions.append({
                    'rows': current_region,
                    'structure': current_structure,
                    'start_index': len(all_rows) - len(current_region),
                    'end_index': len(all_rows) - 1
                })
            
            # If no regions found with strict criteria, try more lenient detection
            if not regions:
                regions = self._detect_regions_lenient(all_rows)
            
            self.logger.info(f"Detected {len(regions)} potential table regions")
            return regions
            
        except Exception as e:
            self.logger.warning(f"Multiple table region detection failed: {e}")
            return []

    def _analyze_row_structure(self, row_data):
        """Analyze the structure of a row"""
        try:
            structure = {
                'column_count': len(row_data),
                'numeric_columns': sum(1 for cell in row_data if self._is_numeric_content(cell)),
                'text_columns': sum(1 for cell in row_data if not self._is_numeric_content(cell) and cell.strip()),
                'empty_columns': sum(1 for cell in row_data if not cell.strip()),
                'avg_cell_length': sum(len(cell) for cell in row_data) / len(row_data) if row_data else 0,
                'has_header_indicators': any(self._is_header_like(cell) for cell in row_data)
            }
            return structure
        except:
            return {'column_count': 0, 'numeric_columns': 0, 'text_columns': 0, 'empty_columns': 0,
                    'avg_cell_length': 0, 'has_header_indicators': False}

    def _is_numeric_content(self, text):
        """Check if text contains primarily numeric content"""
        if not text or not text.strip():
            return False
        cleaned = text.strip()
        # Check for numbers, decimals, percentages, currency
        numeric_chars = sum(1 for c in cleaned if c.isdigit() or c in '.,%-$€£¥')
        return numeric_chars / len(cleaned) > 0.5 if cleaned else False

    def _is_header_like(self, text):
        """Check if text looks like a header"""
        if not text or not text.strip():
            return False
        text = text.strip()
        
        # Headers often contain specific words or patterns
        header_indicators = [
            'total', 'sum', 'average', 'count', 'year', 'month', 'date', 'name', 'type', 'category',
            'expenditure', 'amount', 'function', 'policy', 'financial', 'information', 'contingency',
            'remunerated', 'agency', 'services', 'payments', 'banking', 'other', 'assets', 'property',
            'investment', 'intangibles', 'trade', 'receivables', 'cash', 'equivalents', 'rainfall',
            'americas', 'asia', 'europe', 'africa', 'current', 'non-current', 'high', 'low',
            'column', 'header', 'row', 'data', 'cell', 'table'
        ]
        
        text_lower = text.lower()
        
        # Check for header indicator words
        if any(indicator in text_lower for indicator in header_indicators):
            return True
        
        # Check if text is all uppercase (common for headers)
        if text.isupper() and len(text) > 1:
            return True
        
        # Check for title case (First Letter Capitalized)
        if text.istitle() and len(text.split()) <= 3:
            return True
        
        # Check for patterns like "TH" (table header indicators)
        if text_lower in ['th', 'td', 'header', 'data']:
            return True
        
        # Check for numeric patterns that might be years or codes
        if len(text) == 4 and text.isdigit() and text.startswith(('19', '20')):
            return True
        
        # Check for patterns with slashes (like 2009/10)
        if '/' in text and any(part.isdigit() for part in text.split('/')):
            return True
        
        return False

    def _structure_changed(self, old_structure, new_structure):
        """Check if table structure changed significantly"""
        try:
            # Column count change is a strong indicator (more sensitive)
            if abs(old_structure['column_count'] - new_structure['column_count']) > 0:
                return True
            
            # Significant change in numeric vs text ratio (more sensitive)
            old_numeric_ratio = old_structure['numeric_columns'] / max(1, old_structure['column_count'])
            new_numeric_ratio = new_structure['numeric_columns'] / max(1, new_structure['column_count'])
            if abs(old_numeric_ratio - new_numeric_ratio) > 0.3:
                return True
            
            # Significant change in average cell length (more sensitive)
            if abs(old_structure['avg_cell_length'] - new_structure['avg_cell_length']) > 5:
                return True
            
            # Check for header pattern changes
            if old_structure['has_header_indicators'] != new_structure['has_header_indicators']:
                return True
            
            return False
        except:
            return False

    def _calculate_gap_threshold(self, y_positions):
        """Calculate threshold for detecting significant vertical gaps"""
        try:
            if len(y_positions) < 2:
                return 50  # Default threshold
            
            # Calculate gaps between consecutive rows
            gaps = []
            for i in range(len(y_positions) - 1):
                gap = abs(y_positions[i + 1] - y_positions[i])
                if gap > 0:
                    gaps.append(gap)
            
            if not gaps:
                return 50
            
            # Use median gap * 2 as threshold for significant gaps
            gaps.sort()
            median_gap = gaps[len(gaps) // 2]
            return max(median_gap * 2, 30)  # Minimum threshold of 30
            
        except:
            return 50

    def _is_table_boundary(self, row_data, current_region, row_index, y_positions, gap_threshold):
        """Check if current row indicates a table boundary"""
        try:
            # Check for large vertical gap before this row
            if row_index > 0 and row_index < len(y_positions):
                prev_y = y_positions[row_index - 1] if row_index - 1 < len(y_positions) else 0
                curr_y = y_positions[row_index]
                if abs(curr_y - prev_y) > gap_threshold:
                    return True
            
            # Check for content that suggests new table (like repeated headers)
            if current_region and len(current_region) > 0:
                # Check if this row looks like a header after data rows
                if self._looks_like_new_table_header(row_data, current_region):
                    return True
            
            return False
        except:
            return False

    def _looks_like_new_table_header(self, row_data, current_region):
        """Check if row looks like a header for a new table"""
        try:
            # If current region has mostly numeric data and this row is mostly text
            region_numeric_ratio = 0
            region_rows = 0
            
            for region_row in current_region[-3:]:  # Check last 3 rows
                region_rows += 1
                numeric_count = sum(1 for cell in region_row if self._is_numeric_content(cell))
                region_numeric_ratio += numeric_count / max(1, len(region_row))
            
            if region_rows > 0:
                region_numeric_ratio /= region_rows
            
            # Current row analysis
            current_numeric_ratio = sum(1 for cell in row_data if self._is_numeric_content(cell)) / max(1,
                                                                                                        len(row_data))
            current_text_ratio = sum(
                1 for cell in row_data if not self._is_numeric_content(cell) and cell.strip()) / max(1, len(row_data))
            
            # Header indicators
            has_header_words = any(self._is_header_like(cell) for cell in row_data)
            
            # If previous region was numeric-heavy and current row is text-heavy with header words
            if region_numeric_ratio > 0.5 and current_text_ratio > 0.6 and has_header_words:
                return True
            
            return False
        except:
            return False

    def _detect_regions_lenient(self, all_rows):
        """More lenient region detection for cases where strict detection fails"""
        try:
            regions = []
            current_region = []
            
            for i, row in enumerate(all_rows):
                row_data = [span["text"].strip() for span in row if span["text"].strip()]
                if len(row_data) < 1:
                    # Empty row might indicate table boundary
                    if len(current_region) >= 2:
                        regions.append({
                            'rows': current_region,
                            'structure': self._analyze_row_structure(current_region[0]) if current_region else {},
                            'start_index': i - len(current_region),
                            'end_index': i - 1
                        })
                    current_region = []
                    continue
                
                current_region.append(row_data)
            
            # Add the last region
            if len(current_region) >= 2:
                regions.append({
                    'rows': current_region,
                    'structure': self._analyze_row_structure(current_region[0]) if current_region else {},
                    'start_index': len(all_rows) - len(current_region),
                    'end_index': len(all_rows) - 1
                })
            
            return regions
        except:
            return []

    def _detect_table_regions_dynamic(self, all_rows):
        """Dynamically detect table regions based on structure changes"""
        try:
            regions = []
            current_region = []
            prev_cols = 0
            
            for row in all_rows:
                row_data = [span["text"].strip() for span in row if span["text"].strip()]
                if not row_data:
                    if len(current_region) >= 2:
                        regions.append({'rows': current_region})
                    current_region = []
                    prev_cols = 0
                    continue
                
                curr_cols = len(row_data)
                
                # Structure change detection
                if prev_cols > 0 and abs(curr_cols - prev_cols) > 0:
                    if len(current_region) >= 2:
                        regions.append({'rows': current_region})
                    current_region = [row_data]
                else:
                    current_region.append(row_data)
                
                prev_cols = curr_cols
            
            if len(current_region) >= 2:
                regions.append({'rows': current_region})
            
            return regions
        except:
            return []

    def _extract_table_from_region_fallback(self, region):
        """Extract table data from a specific region"""
        try:
            rows = region['rows']
            if len(rows) < 2:
                return None
            
            # Find header row
            header_row = None
            data_rows = []
            
            for i, row in enumerate(rows):
                if not header_row and self._is_likely_header_row(row, region['structure']):
                    header_row = row
                elif header_row and not self.table_processing_service.is_repeat_header(row, header_row):
                    data_rows.append(row)
                elif not header_row:
                    # If no clear header found, use first row as header
                    if i == 0:
                        header_row = row
                    else:
                        data_rows.append(row)
            
            # Ensure consistent column count
            if header_row and data_rows:
                max_cols = max(len(header_row), max(len(row) for row in data_rows))
                
                # Pad header row
                while len(header_row) < max_cols:
                    header_row.append('')
                
                # Pad data rows
                for row in data_rows:
                    while len(row) < max_cols:
                        row.append('')
                
                return [header_row] + data_rows
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Region table extraction failed: {e}")
            return None

    def _is_likely_header_row(self, row, structure):
        """Check if a row is likely a header based on region structure"""
        try:
            # More text than numbers suggests header
            text_count = sum(1 for cell in row if not self._is_numeric_content(cell) and cell.strip())
            numeric_count = sum(1 for cell in row if self._is_numeric_content(cell))
            
            # Header indicators
            has_header_words = any(self._is_header_like(cell) for cell in row)
            
            return text_count > numeric_count or has_header_words
        except:
            return False

    def _score_fallback_table(self, table_data):
        """Score the quality of a fallback extracted table"""
        try:
            if not table_data or len(table_data) < 2:
                return 0.0
            
            score = 0.0
            
            # Score based on size
            rows = len(table_data)
            cols = len(table_data[0]) if table_data else 0
            
            if rows >= 3 and cols >= 3:
                score += 0.4
            elif rows >= 2 and cols >= 2:
                score += 0.3
            
            # Score based on data consistency
            col_counts = [len(row) for row in table_data]
            if len(set(col_counts)) == 1:  # All rows same length
                score += 0.3
            
            # Score based on content quality
            filled_cells = sum(1 for row in table_data for cell in row if cell.strip())
            total_cells = sum(len(row) for row in table_data)
            fill_ratio = filled_cells / total_cells if total_cells > 0 else 0
            score += fill_ratio * 0.3
            
            return min(score, 1.0)
            
        except:
            return 0.0

    def _extract_single_fallback_table(self, all_rows):
        """Original single table fallback extraction"""
        try:
            fallback_table = []
            header_row = None
            
            for row in all_rows:
                row_data = [span["text"].strip() for span in row if span["text"].strip()]
                if len(row_data) >= 1:  # At least 1 non-empty cell (lowered requirement)
                    # Use existing header detection logic
                    if header_row and self.table_processing_service.is_repeat_header(row_data, header_row):
                        continue
                    if not header_row:
                        header_row = row_data
                    fallback_table.append(row_data)
            
            if len(fallback_table) >= 1:  # Accept even single row tables
                self.logger.info(f"Single fallback extraction found {len(fallback_table)} rows")
                return fallback_table
            
            return None
        except Exception as e:
            self.logger.warning(f"Single fallback extraction failed: {e}")
            return None

    def _try_fallback_extraction_all_tables(self, all_rows, all_spans):
        """Enhanced fallback method to extract ALL tables with different structures"""
        try:
            if not all_rows or len(all_rows) < 2:
                return []
            
            self.logger.info("Attempting enhanced fallback extraction for ALL tables")
            
            # Detect multiple table regions based on structure changes
            table_regions = self._detect_multiple_table_regions(all_rows, all_spans)
            
            if not table_regions:
                # Fall back to single table extraction
                single_table = self._extract_single_fallback_table(all_rows)
                return [single_table] if single_table else []
            
            # Extract ALL valid tables from regions
            extracted_tables = []
            
            for i, region in enumerate(table_regions):
                table_data = self._extract_table_from_region_fallback(region)
                if table_data and len(table_data) >= 2:
                    score = self._score_fallback_table(table_data)
                    extracted_tables.append({
                        'data': table_data,
                        'score': score,
                        'region_index': i,
                        'rows': len(table_data),
                        'columns': len(table_data[0]) if table_data else 0,
                        'structure': region.get('structure', {})
                    })
            
            if extracted_tables:
                # Sort by score
                extracted_tables.sort(key=lambda x: x['score'], reverse=True)
                
                self.logger.info(f"Enhanced fallback extraction found {len(extracted_tables)} tables:")
                for i, table_info in enumerate(extracted_tables):
                    self.logger.info(
                        f"  Table {i + 1}: {table_info['rows']} rows x {table_info['columns']} columns (score: {table_info['score']:.2f})")
                
                # Return all tables
                return [table_info['data'] for table_info in extracted_tables]
            
            # Final fallback to simple extraction
            single_table = self._extract_single_fallback_table(all_rows)
            return [single_table] if single_table else []
            
        except Exception as e:
            self.logger.warning(f"Enhanced fallback extraction for all tables failed: {e}")
            single_table = self._extract_single_fallback_table(all_rows)
            return [single_table] if single_table else []

