import fitz  # PyMuPDF
import csv
import json
import statistics
import arabic_reshaper
from bidi.algorithm import get_display
import io
import os
from PIL import Image
import pytesseract
import re
from app.Common.Constants import (
    DEFAULT_ROW_TOL_MIN, DEFAULT_ROW_TOL_FACTOR, DEFAULT_SHIFT_TOL, DEFAULT_BOUND_MARGIN,
    DEFAULT_MIN_NONEMPTY_CELLS, DEFAULT_HEADER_FILL_THRESHOLD, DEFAULT_DATA_ROW_NUMERIC_THRESHOLD
) 
from app.Logger.ocr_logger import get_standard_logger
from app.Exceptions.custom_exceptions import (
    PDFOpenException, PDFCorruptedException, NoTableFoundException, ColumnDetectionException,
    FileNotFoundException, ArabicTextProcessingException, FileSaveException,
    # AOP functionality
    handle_pdf_processing, handle_text_extraction, handle_table_extraction,
    handle_file_operations, handle_ocr_processing, handle_validation_exceptions, cleanup_temp_files, log_method_entry_exit,
    monitor_performance, ExceptionSeverity
)




class PDFTableExtractor:
    def __init__(self, pdf_path, out_json="output.json", out_csv="output.csv"):
        self.pdf_path = pdf_path
        self.out_json = out_json
        self.out_csv = out_csv
        self.doc = None
        self.logger = get_standard_logger("PDFTableExtractor")

    @handle_text_extraction(severity=ExceptionSeverity.LOW)
    def clean_text(self, text):
        if not text:
            return ""
        text = " ".join(str(text).split())
        try:
            reshaped_text = arabic_reshaper.reshape(text)
            bidi_text = get_display(reshaped_text)
            return bidi_text
        except Exception as e:
            self.logger.warning(f"Arabic text processing failed: {e}")
            raise ArabicTextProcessingException(details={"original_text": text, "error": str(e)})

    @handle_pdf_processing(severity=ExceptionSeverity.HIGH)
    def open_pdf(self):
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundException(self.pdf_path)
        
        try:
            self.doc = fitz.open(self.pdf_path)
            self.logger.info(f"Opened PDF: {self.pdf_path}")
        except Exception as e:
            if "corrupted" in str(e).lower() or "invalid" in str(e).lower():
                raise PDFCorruptedException(self.pdf_path, details={"error": str(e)})
            else:
                raise PDFOpenException(self.pdf_path, details={"error": str(e)})

    @handle_text_extraction(severity=ExceptionSeverity.MEDIUM)
    def extract_page_spans(self, page):
        spans = []
        try:
            d = page.get_text(
                "dict",
                flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_SPANS
            )
            if not d["blocks"]:
                return self.ocr_page(page)

            for block in d.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = self.clean_text(span.get("text", ""))
                        if not text:
                            continue
                        x0, y0, x1, y1 = span["bbox"]
                        spans.append({
                            "x0": float(x0),
                            "x1": float(x1),
                            "y0": float(y0),
                            "y1": float(y1),
                            "y_center": (y0 + y1) / 2.0,
                            "text": text
                        })
            return spans
        except Exception as e:
            self.logger.warning(f"Text extraction failed, falling back to OCR: {e}")
            return self.ocr_page(page)

    @handle_ocr_processing(severity=ExceptionSeverity.HIGH)
    def ocr_page(self, page):
        spans = []
        try:
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes()))
            ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, lang="ara+eng")
            
            for i in range(len(ocr_data["level"])):
                t = ocr_data["text"][i].strip()
                if not t:
                    continue
                text = self.clean_text(t)
                x0 = float(ocr_data["left"][i])
                y0 = float(ocr_data["top"][i])
                x1 = x0 + float(ocr_data["width"][i])
                y1 = y0 + float(ocr_data["height"][i])
                spans.append({
                    "x0": x0,
                    "x1": x1,
                    "y0": y0,
                    "y1": y1,
                    "y_center": (y0 + y1) / 2.0,
                    "text": text
                })
            return spans
        except Exception as e:
            self.logger.error(f"OCR processing failed: {e}")
            raise ArabicTextProcessingException(details={"error": str(e), "page_number": page.number})

    def cluster_rows(self, spans):
        if not spans:
            return []
        spans.sort(key=lambda s: (s["y_center"], s["x0"]))
        avg_h = statistics.mean((s["y1"] - s["y0"]) for s in spans)
        row_tol = max(DEFAULT_ROW_TOL_MIN, avg_h * DEFAULT_ROW_TOL_FACTOR)

        rows = []
        current_row = [spans[0]]
        current_y = spans[0]["y_center"]

        for s in spans[1:]:
            if abs(s["y_center"] - current_y) <= row_tol:
                current_row.append(s)
                current_y = (current_y * (len(current_row) - 1) + s["y_center"]) / len(current_row)
            else:
                if len(current_row) >= DEFAULT_MIN_NONEMPTY_CELLS:
                    rows.append(current_row)
                current_row = [s]
                current_y = s["y_center"]
        if current_row and len(current_row) >= DEFAULT_MIN_NONEMPTY_CELLS:
            rows.append(current_row)
        return rows

    def _row_x0s(self, row):
        return sorted(s["x0"] for s in row)

    @handle_table_extraction(severity=ExceptionSeverity.HIGH)
    def detect_template_columns(self, all_rows):
        if not all_rows:
            raise ColumnDetectionException(details={"reason": "No rows available for column detection"})
        
        template_row = max(all_rows, key=lambda r: len(r), default=[])
        if not template_row:
            raise ColumnDetectionException(details={"reason": "No template row found"})
        
        columns = self._row_x0s(template_row)
        if not columns:
            raise ColumnDetectionException(details={"reason": "No columns detected in template row"})
        
        return columns

    def centers_to_bounds(self, centers, min_x, max_x, margin=DEFAULT_BOUND_MARGIN):
        if not centers:
            return []
        centers = sorted(centers)
        bounds = []
        left = min_x - margin
        right = (centers[0] + centers[1]) / 2.0 if len(centers) > 1 else max_x + margin
        bounds.append((left, right))
        for i in range(1, len(centers) - 1):
            left = (centers[i - 1] + centers[i]) / 2.0
            right = (centers[i] + centers[i + 1]) / 2.0
            bounds.append((left, right))
        if len(centers) > 1:
            left = (centers[-2] + centers[-1]) / 2.0
            right = max_x + margin
            bounds.append((left, right))
        return bounds

    def assign_span_to_col(self, span, bounds, centers):
        if not bounds:
            return 0
        cx = (span["x0"] + span["x1"]) / 2.0
        for i, (l, r) in enumerate(bounds):
            if l <= cx < r:
                return i
        return min(range(len(centers)), key=lambda i: abs(cx - centers[i]))

    def build_table_with_bounds(self, rows, bounds, centers):
        table = []
        for row in rows:
            cells = {i: [] for i in range(len(centers))}
            for s in row:
                idx = self.assign_span_to_col(s, bounds, centers)
                cells[idx].append((s["x0"], s["text"]))
            ordered = [" ".join(t for _, t in sorted(cells[i], key=lambda x: x[0])).strip()
                       for i in range(len(centers))]
            if sum(1 for c in ordered if c) >= DEFAULT_MIN_NONEMPTY_CELLS:
                table.append(ordered)
        return table

    def is_english_row(self, row):
        text = ' '.join(row)
        latin = sum(1 for c in text if ('A' <= c <= 'Z' or 'a' <= c <= 'z'))
        total = sum(1 for c in text if c.isalpha())
        return latin / total > 0.5 if total > 0 else False

    def post_process_table(self, table, header):
        fixed = []
        for row in table:
            if self.is_repeat_header(row, header):
                continue
            if not fixed:
                fixed.append(row)
                continue
            filled = sum(1 for c in row if c.strip())
            if filled < 3:  # Likely continuation of previous row
                prev = fixed[-1]
                for i, c in enumerate(row):
                    if c.strip():
                        prev[i] = (prev[i] + " " + c).strip()
            else:
                fixed.append(row)

        # Filter unwanted rows
        filtered = [header] if header else []
        for row in fixed:
            joined = ' '.join(row)
            if "SUB TOTAL" in joined or "Dubai Airport" in joined or "distance" in joined.lower() or not row[0].isdigit():
                continue
            filtered.append(row)

        # Combine second and third columns for data rows if split, but keep both merged and unmerged
        for i in range(1, len(filtered)):
            row = filtered[i]
            if len(row) > 2 and "Overflights -" in row[1] and re.match(r'^\d+$', row[2]):
                row[1] = row[1] + " " + row[2]
                # Keep row[2] as is

        return filtered

    def is_likely_header(self, row, expected_cols):
        if not row or len(row) < expected_cols:
            return False
        filled_cells = sum(1 for cell in row if cell.strip())
        non_numeric_count = sum(1 for cell in row if cell.strip() and not re.match(r'^[\d,.]+$', cell.strip()))
        return (filled_cells / len(row) >= DEFAULT_HEADER_FILL_THRESHOLD and
                non_numeric_count >= 2 and
                filled_cells >= DEFAULT_MIN_NONEMPTY_CELLS)

    def is_repeat_header(self, row, header_row):
        if not header_row or len(row) != len(header_row):
            return False
        match_ratio = sum(1 for a, b in zip(row, header_row) if a.strip().lower() == b.strip().lower()) / len(header_row)
        return match_ratio > 0.8

    def is_likely_table_row(self, row, expected_cols):
        if not row or len(row) < expected_cols:
            return False
        filled_cells = sum(1 for cell in row if cell.strip())
        numeric_count = sum(1 for cell in row if cell.strip() and re.match(r'^[\d,.]+$', cell.strip()))
        return (filled_cells >= DEFAULT_MIN_NONEMPTY_CELLS and
                numeric_count >= DEFAULT_DATA_ROW_NUMERIC_THRESHOLD)

    @monitor_performance
    @log_method_entry_exit
    @handle_table_extraction(
        severity=ExceptionSeverity.HIGH,
        retryable=False,
        cleanup_func=cleanup_temp_files
    )
    def extract_table(self):
        try:
            self.logger.info(f"Starting table extraction from: {self.pdf_path}")
            self.open_pdf()
            all_spans = []
            all_rows = []

            for page_idx, page in enumerate(self.doc):
                self.logger.info(f"Processing page {page_idx + 1}")
                spans = self.extract_page_spans(page)
                all_spans.extend(spans)
                rows = self.cluster_rows(spans)
                all_rows.extend(rows)

            if not all_rows:
                self.logger.warning("No text found in document.")
                self.doc.close()
                raise NoTableFoundException(self.pdf_path, details={"reason": "No text spans found in document"})

            self.logger.info(f"Detecting columns from {len(all_rows)} rows")
            global_centers = self.detect_template_columns(all_rows)
            global_min_x = min(s["x0"] for s in all_spans) if all_spans else 0
            global_max_x = max(s["x1"] for s in all_spans) if all_spans else 0
            global_bounds = self.centers_to_bounds(global_centers, global_min_x, global_max_x)
            expected_cols = len(global_centers)

            self.logger.info(f"Building table with {expected_cols} columns")
            table = self.build_table_with_bounds(all_rows, global_bounds, global_centers)

            header_found = False
            header_row = None
            pre_table = []
            i = 0
            while i < len(table):
                clean_row = [cell.strip() for cell in table[i]]
                if not header_found and self.is_likely_header(clean_row, expected_cols):
                    if self.is_english_row(clean_row):
                        header_row = clean_row
                        header_found = True
                    elif i + 1 < len(table):
                        next_row = [cell.strip() for cell in table[i+1]]
                        if self.is_likely_header(next_row, expected_cols) and self.is_english_row(next_row):
                            header_row = next_row
                            header_found = True
                            i += 1
                elif header_found:
                    if self.is_repeat_header(clean_row, header_row):
                        pass
                    elif self.is_likely_table_row(clean_row, expected_cols):
                        pre_table.append(clean_row)
                i += 1

            self.logger.info(f"Post-processing table with {len(pre_table)} data rows")
            final_table = self.post_process_table(pre_table, header_row)

            if not final_table or len(final_table) <= 1:  # Only header or empty
                self.doc.close()
                raise NoTableFoundException(self.pdf_path, details={"reason": "No valid table data found after processing"})

            self.doc.close()
            self.save_results(final_table)
            self.logger.info(f"Successfully extracted table with {len(final_table)} rows")
            return final_table
            
        except Exception as e:
            self.logger.error(f"Table extraction failed: {e}")
            if hasattr(self, 'doc') and self.doc:
                self.doc.close()
            raise

    @handle_file_operations(severity=ExceptionSeverity.MEDIUM)
    def save_results(self, table):
        try:
            with open(self.out_json, "w", encoding="utf-8") as jf:
                json.dump(table, jf, ensure_ascii=False, indent=2)
            with open(self.out_csv, "w", newline="", encoding="utf-8-sig") as cf:
                writer = csv.writer(cf)
                writer.writerows(table)
            self.logger.info(f"Saved {len(table)} rows to {self.out_csv} and {self.out_json}")
        except Exception as e:
            raise FileSaveException(self.out_csv, details={"error": str(e), "table_rows": len(table)})


if __name__ == "__main__":
    extractor = PDFTableExtractor(
        pdf_path="sample.pdf",
        out_json="all_pages_table.json",
        out_csv="all_pages_table_clean4.csv"
    )
    extractor.extract_table()
    def _extract_best_header_dynamically(self, table, current_index, expected_cols):
        """Extract and merge multi-line English headers"""
        if current_index >= len(table):
            return None
        
        current_row = [cell.strip() for cell in table[current_index]]
        
        if self.is_english_row(current_row):
            # Check for continuation row
            if current_index + 1 < len(table):
                next_row = [cell.strip() for cell in table[current_index + 1]]
                if self.is_english_row(next_row):
                    return self._merge_header_rows(current_row, next_row)
            return current_row
        
        if current_index + 1 < len(table):
            next_row = [cell.strip() for cell in table[current_index + 1]]
            if self.is_likely_header(next_row, expected_cols) and self.is_english_row(next_row):
                return next_row
        
        return None
    
    def _is_header_continuation(self, first_row, second_row):
        """Check if second row continues first row header"""
        return True
    
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