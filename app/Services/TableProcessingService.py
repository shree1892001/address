import statistics
import re
import numpy as np
from typing import List, Dict, Tuple, Optional
from app.Common.Constants import (
    DEFAULT_ROW_TOL_MIN, DEFAULT_ROW_TOL_FACTOR, DEFAULT_SHIFT_TOL, DEFAULT_BOUND_MARGIN,
    DEFAULT_MIN_NONEMPTY_CELLS, DEFAULT_HEADER_FILL_THRESHOLD, DEFAULT_DATA_ROW_NUMERIC_THRESHOLD
)
from app.Logger.ocr_logger import get_standard_logger
from app.Exceptions.custom_exceptions import (
    ColumnDetectionException
)
from app.Exceptions.custom_exceptions import (
    handle_table_extraction, log_method_entry_exit, ExceptionSeverity
)


class TableProcessingService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TableProcessingService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.logger = get_standard_logger("TableProcessingService")
            self._initialized = True

    @log_method_entry_exit
    @handle_table_extraction(severity=ExceptionSeverity.MEDIUM)
    def cluster_rows(self, spans, dynamic_params=None):
        """Cluster text spans into rows using original logic with optional dynamic parameters"""
        if not spans:
            return []
        
        # Use original default values to maintain accuracy
        row_tol_min = DEFAULT_ROW_TOL_MIN
        row_tol_factor = DEFAULT_ROW_TOL_FACTOR
        min_nonempty_cells = DEFAULT_MIN_NONEMPTY_CELLS
        
        # Only use dynamic parameters if explicitly provided and needed
        if dynamic_params:
            row_tol_min = dynamic_params.get("row_tol_min", DEFAULT_ROW_TOL_MIN)
            row_tol_factor = dynamic_params.get("row_tol_factor", DEFAULT_ROW_TOL_FACTOR)
            min_nonempty_cells = dynamic_params.get("min_nonempty_cells", DEFAULT_MIN_NONEMPTY_CELLS)
        
        spans.sort(key=lambda s: (s["y_center"], s["x0"]))
        avg_h = statistics.mean((s["y1"] - s["y0"]) for s in spans)
        row_tol = max(row_tol_min, avg_h * row_tol_factor)

        rows = []
        current_row = [spans[0]]
        current_y = spans[0]["y_center"]

        for s in spans[1:]:
            if abs(s["y_center"] - current_y) <= row_tol:
                current_row.append(s)
                current_y = (current_y * (len(current_row) - 1) + s["y_center"]) / len(current_row)
            else:
                if len(current_row) >= min_nonempty_cells:
                    rows.append(current_row)
                current_row = [s]
                current_y = s["y_center"]
        if current_row and len(current_row) >= min_nonempty_cells:
            rows.append(current_row)
        return rows

    def _calculate_dynamic_clustering_params(self, spans):
        """Calculate dynamic clustering parameters based on span characteristics"""
        try:
            if not spans:
                return DEFAULT_ROW_TOL_MIN, DEFAULT_ROW_TOL_FACTOR, DEFAULT_MIN_NONEMPTY_CELLS
            
            # Calculate span height statistics
            heights = [s["y1"] - s["y0"] for s in spans]
            avg_height = statistics.mean(heights)
            height_std = statistics.stdev(heights) if len(heights) > 1 else 0
            
            # Calculate span count statistics
            span_count = len(spans)
            
            # Adapt parameters based on characteristics
            if span_count < 10:
                # Few spans - be more lenient
                row_tol_min = DEFAULT_ROW_TOL_MIN * 0.8
                row_tol_factor = DEFAULT_ROW_TOL_FACTOR * 0.8
                min_nonempty_cells = max(2, DEFAULT_MIN_NONEMPTY_CELLS - 1)
            elif span_count > 50:
                # Many spans - be more strict
                row_tol_min = DEFAULT_ROW_TOL_MIN * 1.2
                row_tol_factor = DEFAULT_ROW_TOL_FACTOR * 1.2
                min_nonempty_cells = DEFAULT_MIN_NONEMPTY_CELLS + 1
            else:
                # Normal case
                row_tol_min = DEFAULT_ROW_TOL_MIN
                row_tol_factor = DEFAULT_ROW_TOL_FACTOR
                min_nonempty_cells = DEFAULT_MIN_NONEMPTY_CELLS
            
            # Adjust based on height variance
            if height_std > avg_height * 0.5:
                # High variance in heights - be more lenient
                row_tol_factor *= 1.2
            
            return row_tol_min, row_tol_factor, min_nonempty_cells
            
        except Exception as e:
            self.logger.warning(f"Dynamic parameter calculation failed: {e}")
            return DEFAULT_ROW_TOL_MIN, DEFAULT_ROW_TOL_FACTOR, DEFAULT_MIN_NONEMPTY_CELLS

    def _is_numeric_cell(self, cell_text: str) -> bool:
        """
        Dynamically check if a cell contains numeric data without relying on predefined patterns.
        
        Args:
            cell_text: The text to check
            
        Returns:
            bool: True if the cell contains numeric data
        """
        try:
            if not cell_text or not cell_text.strip():
                return False
                
            text = cell_text.strip()
            
            # Check for empty string after stripping
            if not text:
                return False
                
            # Remove all non-numeric characters except '.', '-', and ','
            cleaned = ''
            decimal_seen = False
            for char in text:
                if char.isdigit():
                    cleaned += char
                elif char in '.,' and not decimal_seen:
                    cleaned += '.'
                    decimal_seen = True
                elif char == '-' and not cleaned:  # Only allow minus at start
                    cleaned += char
                elif char in '/%':  # Common numeric symbols
                    continue
                elif not char.isspace() and not char.isalpha():
                    # If we find non-numeric, non-whitespace, non-alpha, likely not a pure number
                    return False
                    
            # If we removed all non-numeric characters and have something left
            if cleaned and (cleaned.replace('-', '').replace('.', '').isdigit() or 
                           (cleaned.endswith('%') and cleaned[:-1].replace('-', '').replace('.', '').isdigit())):
                return True
                
            # Check for number ranges (e.g., "10-20")
            if '-' in text[1:]:  # Not at start (for negative numbers)
                parts = text.split('-')
                if len(parts) == 2 and all(p.strip().replace(',', '').replace('.', '').isdigit() for p in parts):
                    return True
                    
            return False
                
        except Exception as e:
            self.logger.debug(f"Error in _is_numeric_cell: {str(e)}")
            return False

    def _contains_digits(self, text: str) -> bool:
        """Check if text contains digits without regex"""
        try:
            if not text:
                return False
            return any(char.isdigit() for char in text)
        except:
            return False

    def _normalize_text(self, text: str) -> str:
        """Normalize OCR text for robust comparisons: collapse whitespace, replace NBSP/ZWSP, trim."""
        try:
            if text is None:
                return ""
            # Replace common non-printing/nbsp characters
            s = text.replace('\u00A0', ' ').replace('\u200B', '').replace('\ufeff', '')
            # Collapse multiple whitespace to single space
            s = re.sub(r'\s+', ' ', s)
            return s.strip()
        except:
            return text

    def _is_pure_number(self, text: str) -> bool:
        """Check if text is a pure number without regex"""
        try:
            if not text:
                return False
            
            # Remove common numeric separators
            cleaned = text.replace(',', '').replace('.', '').replace(' ', '')
            
            # Check if all remaining characters are digits
            return cleaned.isdigit() and len(cleaned) > 0
        except:
            return False

    def _calculate_dynamic_header_threshold(self, row: List[str]) -> float:
        """Calculate dynamic header fill threshold based on row characteristics"""
        try:
            if not row:
                return DEFAULT_HEADER_FILL_THRESHOLD
            
            # Base threshold
            base_threshold = DEFAULT_HEADER_FILL_THRESHOLD
            
            # Adjust based on row length
            row_length = len(row)
            if row_length > 10:
                return base_threshold * 0.8  # Be more lenient for long rows
            elif row_length < 3:
                return base_threshold * 1.2  # Be more strict for short rows
            else:
                return base_threshold
                
        except:
            return DEFAULT_HEADER_FILL_THRESHOLD

    def _calculate_dynamic_min_cells(self, row: List[str]) -> int:
        """Calculate dynamic minimum cells threshold based on row characteristics"""
        try:
            if not row:
                return DEFAULT_MIN_NONEMPTY_CELLS
            
            # Base threshold
            base_threshold = DEFAULT_MIN_NONEMPTY_CELLS
            
            # Adjust based on row length
            row_length = len(row)
            if row_length > 8:
                return max(2, base_threshold - 1)  # Be more lenient for long rows
            elif row_length < 4:
                return base_threshold + 1  # Be more strict for short rows
            else:
                return base_threshold
                
        except:
            return DEFAULT_MIN_NONEMPTY_CELLS

    def _calculate_dynamic_numeric_threshold(self, row: List[str]) -> int:
        """Calculate dynamic numeric threshold based on row characteristics"""
        try:
            if not row:
                return DEFAULT_DATA_ROW_NUMERIC_THRESHOLD
            
            # Base threshold
            base_threshold = DEFAULT_DATA_ROW_NUMERIC_THRESHOLD
            
            # Adjust based on row length
            row_length = len(row)
            if row_length > 6:
                return max(1, base_threshold - 1)  # Be more lenient for long rows
            elif row_length < 3:
                return base_threshold + 1  # Be more strict for short rows
            else:
                return base_threshold
                
        except:
            return DEFAULT_DATA_ROW_NUMERIC_THRESHOLD

    def _row_x0s(self, row):
        """Get x-coordinates of spans in a row"""
        return sorted(s["x0"] for s in row)

    @log_method_entry_exit
    @handle_table_extraction(severity=ExceptionSeverity.HIGH)
    def detect_template_columns(self, all_rows):
        """Detect column structure from rows"""
        if not all_rows:
            raise ColumnDetectionException(details={"reason": "No rows available for column detection"})
        
        template_row = max(all_rows, key=lambda r: len(r), default=[])
        if not template_row:
            raise ColumnDetectionException(details={"reason": "No template row found"})
        
        columns = self._row_x0s(template_row)
        if not columns:
            raise ColumnDetectionException(details={"reason": "No columns detected in template row"})
        
        return columns

    @log_method_entry_exit
    @handle_table_extraction(severity=ExceptionSeverity.MEDIUM)
    def centers_to_bounds(self, centers, min_x, max_x, margin=DEFAULT_BOUND_MARGIN):
        """Convert column centers to boundaries"""
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

    @log_method_entry_exit
    @handle_table_extraction(severity=ExceptionSeverity.MEDIUM)
    def assign_span_to_col(self, span, bounds, centers):
        """Assign a span to the appropriate column"""
        if not bounds:
            return 0
        cx = (span["x0"] + span["x1"]) / 2.0
        for i, (l, r) in enumerate(bounds):
            if l <= cx < r:
                return i
        return min(range(len(centers)), key=lambda i: abs(cx - centers[i]))

    @log_method_entry_exit
    @handle_table_extraction(severity=ExceptionSeverity.MEDIUM)
    def build_table_with_bounds(self, rows, bounds, centers):
        """Build table structure using column bounds"""
        table = []
        for row in rows:
            cells = {i: [] for i in range(len(centers))}
            for s in row:
                idx = self.assign_span_to_col(s, bounds, centers)
                cells[idx].append((s["x0"], s["text"]))
            ordered = [" ".join(t for _, t in sorted(cells[i], key=lambda x: x[0])).strip()
                       for i in range(len(centers))]
            if sum(1 for c in ordered if c) >= 1:  # Accept rows with at least 1 cell
                table.append(ordered)
        return table

    @log_method_entry_exit
    @handle_table_extraction(severity=ExceptionSeverity.LOW)
    def is_english_row(self, row):
        """Check if a row contains primarily English text"""
        text = ' '.join(row)
        latin = sum(1 for c in text if ('A' <= c <= 'Z' or 'a' <= c <= 'z'))
        total = sum(1 for c in text if c.isalpha())
        return latin / total > 0.5 if total > 0 else False

    @log_method_entry_exit
    @handle_table_extraction(severity=ExceptionSeverity.MEDIUM)
    def post_process_table(self, table, header):
        """Post-process extracted table data"""
        # Debug: show incoming header and a preview of table rows
        try:
            preview = table[:6] if table else []
            # Use repr to expose invisible/control characters in logs
            self.logger.debug(f"post_process_table called - header: {repr(header)} | preview rows: {repr(preview)}")
        except Exception:
            self.logger.debug("post_process_table called - header or table preview unprintable")

        # Extract clean header from the header parameter for comparison
        clean_header = [self._normalize_text(str(h)) for h in header] if header else []
        
        fixed = []
        for row in table:
            # Check if row is a repeat of the clean header
            if self._is_duplicate_header_row(row, clean_header):
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

        # Normalize header and first data row strings to handle NBSP/ZWSP and weird whitespace
        try:
            if header:
                header = [self._normalize_text(h) for h in header]
        except Exception:
            pass

        # Fix header cells that accidentally captured trailing data tokens (e.g., "ID 1")
        try:
            if header and fixed:
                first_data = fixed[0]
                
                # Enhanced header-data separation logic
                for i in range(min(len(header), len(first_data))):
                    try:
                        h_raw = header[i] if i < len(header) else ""
                        h = self._normalize_text(h_raw)
                        if not h:
                            continue

                        # Split header cell into words
                        h_words = h.split()
                        if len(h_words) < 2:
                            continue
                            
                        # Look for pattern: HeaderWord + DataValue
                        # Try different split points to find the best separation
                        best_split = None
                        for split_idx in range(1, len(h_words)):
                            header_part = ' '.join(h_words[:split_idx]).strip()
                            data_part = ' '.join(h_words[split_idx:]).strip()
                            
                            # Check if data_part looks like actual data (numeric or currency)
                            if (self._contains_digits(data_part) or 
                                data_part.startswith(('$', '€', '£', '₹', '¥', '₽', '₩')) or
                                self._is_numeric_cell(data_part)):
                                best_split = (header_part, data_part)
                                break
                        
                        if best_split:
                            header_part, data_part = best_split
                            header[i] = header_part
                            # Set the data part as the first data row value
                            first_data[i] = data_part
                            
                    except Exception:
                        continue

                fixed[0] = first_data
        except Exception:
            pass

        # If header still looks like it contains data (digits/currency), prefer a clean
        # header row found in the extracted rows (some PDFs place the real header as a
        # separate row in the table). If found, use it and remove it from the data rows.
        try:
            header_contains_digits = any(self._contains_digits(str(h)) for h in header) if header else False
            if header_contains_digits and table:
                clean_header = None
                for r in table:
                    # Candidate must be header-like and not look like a data row
                    try:
                        if self.is_likely_header(r, len(r)) and not any(self._contains_digits(str(c)) for c in r):
                            clean_header = [self._normalize_text(c) for c in r]
                            break
                    except Exception:
                        continue

                if clean_header:
                    # Remove the clean header row from fixed if present
                    fixed = [row for row in fixed if row != clean_header]
                    header = clean_header
        except Exception:
            pass



        # Final header cleaning: if header still contains data, clean it aggressively
        if header and fixed:
            cleaned_header = []
            for i, h_cell in enumerate(header):
                h_words = str(h_cell).split()
                if len(h_words) > 1:
                    # Take only the first word as header if it looks like a header term
                    first_word = h_words[0]
                    # If first word is likely a header (alphabetic, not purely numeric)
                    if first_word.isalpha() or (first_word.replace('_', '').isalpha()):
                        cleaned_header.append(first_word)
                    else:
                        cleaned_header.append(h_cell)
                else:
                    cleaned_header.append(h_cell)
            header = cleaned_header

        # Keep rows generically without hardcoded content filters
        filtered = [header] if header else []
        filtered.extend(fixed)

        # Debug: show header and first data row after post-processing adjustments
        try:
            after_preview = filtered[:3] if filtered else []
            self.logger.debug(f"post_process_table result preview (header+first rows): {after_preview}")
        except Exception:
            self.logger.debug("post_process_table result preview: <unserializable>")

        # Combine second and third columns for data rows if split
        for i in range(1, len(filtered)):
            row = filtered[i]
            if len(row) > 2 and "Overflights -" in row[1] and re.match(r'^\d+$', row[2]):
                row[1] = row[1] + " " + row[2]

        # Final cleanup: if header cells still contain the first data row's content
        # (common when OCR merges header+first-row), remove the duplicate suffix from header.
        try:
            if len(filtered) > 1:
                hdr = filtered[0]
                first = filtered[1]
                for i in range(min(len(hdr), len(first))):
                    h = hdr[i].strip() if hdr[i] else ""
                    d = first[i].strip() if first[i] else ""
                    if not h or not d:
                        continue
                    # If the entire data cell appears inside the header, remove it from header
                    if d and d in h:
                        new_h = h.replace(d, '').strip()
                        # Remove leftover punctuation/extra separators
                        new_h = re.sub(r'[\-:/,\s]+$', '', new_h).strip()
                        hdr[i] = new_h
        except Exception:
            pass

        return filtered

    @log_method_entry_exit
    @handle_table_extraction(severity=ExceptionSeverity.LOW)
    def is_likely_header(self, row, expected_cols):
        """
        Dynamically determine if a row is likely a header row based on content analysis.
        
        Args:
            row: The row to check
            expected_cols: Expected number of columns
            
        Returns:
            bool: True if the row is likely a header
        """
        if not row or len(row) < expected_cols:
            return False
            
        cleaned_row = [cell.strip() for cell in row]
        filled_cells = sum(1 for cell in cleaned_row if cell)
        
        # Calculate basic statistics
        non_numeric_count = sum(1 for cell in cleaned_row if cell and not self._is_numeric_cell(cell))
        avg_word_length = sum(len(cell.split()) for cell in cleaned_row if cell) / max(1, filled_cells)
        
        # Dynamic thresholds based on expected columns
        min_filled = max(2, expected_cols // 3)  # At least 2 or 1/3 of expected columns
        min_non_numeric = max(2, expected_cols * 0.3)  # At least 2 or 30% non-numeric
        
        # Header characteristics:
        # 1. Has a good number of filled cells
        # 2. Contains mostly non-numeric text
        # 3. Has reasonable word lengths (not too long, not too short)
        is_header = (
            filled_cells >= min_filled and
            non_numeric_count >= min_non_numeric and
            1 <= avg_word_length <= 5  # Average word length between 1-5 words
        )
        
        return is_header

    def _is_duplicate_header_row(self, row, clean_header):
        """
        Check if a row is a duplicate of the header row using generic pattern matching.
        
        Args:
            row: Current row to check
            clean_header: The normalized header row to compare against
            
        Returns:
            bool: True if the row is a duplicate header
        """
        try:
            if not clean_header or not row or len(row) != len(clean_header):
                return False
                
            # Normalize the current row
            norm_row = [self._normalize_text(str(cell)) for cell in row]
            
            # Check for exact match
            if norm_row == clean_header:
                return True
                
            # Check word overlap between row and clean header (generic approach)
            clean_header_words = set(' '.join(clean_header).lower().split())
            row_words = set(' '.join(norm_row).lower().split())
            
            if not clean_header_words:
                return False
                
            word_overlap = len(clean_header_words.intersection(row_words))
            overlap_ratio = word_overlap / len(clean_header_words)
            
            # If significant word overlap (70% or more), likely a duplicate header
            if overlap_ratio >= 0.7:
                return True
                
            # Check if row has mostly non-numeric content like a header should
            non_numeric_cells = sum(1 for cell in norm_row if cell and not self._is_numeric_cell(cell))
            non_numeric_ratio = non_numeric_cells / len(norm_row) if norm_row else 0
            
            # If high word overlap (50%+) AND mostly non-numeric, likely header
            if overlap_ratio >= 0.5 and non_numeric_ratio >= 0.8:
                return True
                
            return False
            
        except Exception as e:
            self.logger.warning(f"Error in _is_duplicate_header_row: {str(e)}")
            return False

    @log_method_entry_exit
    @handle_table_extraction(severity=ExceptionSeverity.LOW)
    def is_repeat_header(self, row, header_row):
        """
        Check if a row is a repeat of the header by comparing column-wise similarity.
        
        Args:
            row: Current row to check
            header_row: The header row to compare against
            
        Returns:
            bool: True if the row is likely a repeated header
        """
        try:
            # Basic validation
            if not header_row or not row or len(row) != len(header_row):
                return False
                
            # Normalize both rows
            norm_row = [str(cell).strip().lower() for cell in row]
            norm_header = [str(cell).strip().lower() for cell in header_row]
            
            # Check for exact match
            if norm_row == norm_header:
                return True
                
            # Calculate column-wise similarity
            total_columns = len(header_row)
            if total_columns == 0:
                return False
                
            # Count how many columns have significant text match
            matching_columns = 0
            for col in range(total_columns):
                row_cell = norm_row[col]
                header_cell = norm_header[col]
                
                # Skip empty cells in header
                if not header_cell:
                    continue
                    
                # Exact match
                if row_cell == header_cell:
                    matching_columns += 1
                    continue
                    
                # Partial match (header text is contained in row cell or vice versa)
                if (header_cell in row_cell) or (row_cell and row_cell in header_cell):
                    matching_columns += 0.5  # Partial match scores less
            
            # Calculate match ratio considering only non-empty header columns
            non_empty_headers = sum(1 for h in norm_header if h)
            if non_empty_headers == 0:
                return False
                
            match_ratio = matching_columns / non_empty_headers
            
            # Consider it a match if most columns match
            is_match = match_ratio >= 0.6  # 60% of non-empty header columns match
            
            self.logger.debug(f"Header match ratio: {match_ratio:.2f} (threshold: 0.60) - {'Match' if is_match else 'No match'}")
            return is_match
            
        except Exception as e:
            self.logger.warning(f"Error in is_repeat_header: {str(e)}")
            return False

    @log_method_entry_exit
    @handle_table_extraction(severity=ExceptionSeverity.LOW)
    def is_likely_table_row(self, row, expected_cols):
        """
        Dynamically determine if a row is likely a data row based on content analysis.
        
        Args:
            row: The row to check
            expected_cols: Expected number of columns
            
        Returns:
            bool: True if the row is likely a data row
        """
        if not row or len(row) < expected_cols:
            return False
            
        cleaned_row = [cell.strip() for cell in row]
        filled_cells = sum(1 for cell in cleaned_row if cell)
        
        # Calculate basic statistics
        numeric_count = sum(1 for cell in cleaned_row if cell and self._is_numeric_cell(cell))
        numeric_ratio = numeric_count / max(1, filled_cells)
        
        # Dynamic thresholds based on expected columns
        min_filled = max(2, expected_cols // 3)  # At least 2 or 1/3 of expected columns
        min_numeric = max(1, expected_cols * 0.2)  # At least 1 or 20% numeric
        
        # Data row characteristics:
        # 1. Has a reasonable number of filled cells
        # 2. Contains some numeric data or has good fill ratio
        is_data_row = (
            filled_cells >= min_filled and
            (numeric_count >= min_numeric or filled_cells >= expected_cols * 0.7)
        )
        
        return is_data_row

    @log_method_entry_exit
    @handle_table_extraction(severity=ExceptionSeverity.MEDIUM)
    def detect_table_regions(self, spans: List[Dict]) -> List[Dict]:
        """Detect potential table regions in the document"""
        try:
            if not spans:
                return []
            
            # Group spans by spatial proximity
            regions = self._group_spans_by_proximity(spans)
            
            # Analyze each region for table characteristics
            table_regions = []
            for region in regions:
                if self._is_potential_table_region(region):
                    table_regions.append(region)
            
            self.logger.info(f"Detected {len(table_regions)} potential table regions")
            return table_regions
            
        except Exception as e:
            self.logger.error(f"Table region detection failed: {e}")
            return []

    def _group_spans_by_proximity(self, spans: List[Dict]) -> List[List[Dict]]:
        """Group text spans by spatial proximity"""
        if not spans:
            return []
        
        # Sort spans by position
        spans.sort(key=lambda s: (s["y0"], s["x0"]))
        
        regions = []
        current_region = [spans[0]]
        
        for span in spans[1:]:
            # Check if span is close to current region
            if self._is_spatially_close(span, current_region):
                current_region.append(span)
            else:
                if len(current_region) >= 3:  # Minimum spans for a region
                    regions.append(current_region)
                current_region = [span]
        
        # Add the last region
        if len(current_region) >= 3:
            regions.append(current_region)
        
        return regions

    def _is_spatially_close(self, span: Dict, region: List[Dict]) -> bool:
        """Check if a span is spatially close to a region"""
        try:
            # Calculate region bounds
            region_x0 = min(s["x0"] for s in region)
            region_x1 = max(s["x1"] for s in region)
            region_y0 = min(s["y0"] for s in region)
            region_y1 = max(s["y1"] for s in region)
            
            # Check proximity
            x_distance = min(abs(span["x0"] - region_x1), abs(span["x1"] - region_x0))
            y_distance = min(abs(span["y0"] - region_y1), abs(span["y1"] - region_y0))
            
            # Allow some overlap or close proximity
            return x_distance < 50 and y_distance < 20
            
        except:
            return False

    def _is_potential_table_region(self, region: List[Dict]) -> bool:
        """Check if a region has table-like characteristics"""
        try:
            if len(region) < 3:
                return False
            
            # Check for regular spacing patterns
            x_positions = [s["x0"] for s in region]
            y_positions = [s["y0"] for s in region]
            
            # Check for column-like alignment
            x_alignment_score = self._calculate_alignment_score(x_positions)
            y_alignment_score = self._calculate_alignment_score(y_positions)
            
            # Check for numeric content (common in tables)
            numeric_ratio = self._calculate_numeric_ratio(region)
            
            # Table-like if it has good alignment and some numeric content
            return (x_alignment_score > 0.3 or y_alignment_score > 0.3) and numeric_ratio > 0.1
            
        except:
            return False

    def _calculate_alignment_score(self, positions: List[float]) -> float:
        """Calculate how well positions are aligned"""
        try:
            if len(positions) < 2:
                return 0.0
            
            # Group positions by proximity
            positions.sort()
            clusters = []
            current_cluster = [positions[0]]
            
            for pos in positions[1:]:
                if pos - current_cluster[-1] < 20:  # Within 20 pixels
                    current_cluster.append(pos)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [pos]
            clusters.append(current_cluster)
            
            # Score based on number of distinct clusters
            return min(len(clusters) / len(positions), 1.0)
            
        except:
            return 0.0

    def _calculate_numeric_ratio(self, region: List[Dict]) -> float:
        """Calculate ratio of numeric content in region"""
        try:
            total_text = " ".join(s["text"] for s in region)
            numeric_chars = sum(1 for c in total_text if c.isdigit())
            total_chars = len(total_text)
            
            return numeric_chars / total_chars if total_chars > 0 else 0.0
            
        except:
            return 0.0

    @log_method_entry_exit
    @handle_table_extraction(severity=ExceptionSeverity.HIGH)
    def advanced_table_detection(self, spans: List[Dict]) -> List[Dict]:
        """Advanced table detection using original logic with adaptive fallback"""
        try:
            if not spans:
                return []
            
            # Use original traditional method first to maintain accuracy
            traditional_tables = self._detect_tables_traditional(spans)
            
            # Only use adaptive detection if traditional method fails
            if not traditional_tables:
                self.logger.info("Traditional detection found no tables, trying adaptive methods")
                try:
                    from app.Services.AdaptiveTableDetectionService import AdaptiveTableDetectionService
                    adaptive_detector = AdaptiveTableDetectionService()
                    adaptive_tables = adaptive_detector.detect_tables_adaptively(spans)
                    if adaptive_tables:
                        self.logger.info(f"Adaptive detection found {len(adaptive_tables)} tables")
                        return adaptive_tables
                except Exception as e:
                    self.logger.warning(f"Adaptive detection failed: {e}")
            
            self.logger.info(f"Traditional detection found {len(traditional_tables)} tables")
            return traditional_tables
            
        except Exception as e:
            self.logger.error(f"Advanced table detection failed: {e}")
            return []

    def _detect_tables_traditional(self, spans: List[Dict]) -> List[Dict]:
        """Traditional table detection using row clustering"""
        try:
            rows = self.cluster_rows(spans)
            if len(rows) < 2:
                return []
            
            # Use existing column detection
            columns = self.detect_template_columns(rows)
            if not columns:
                return []
            
            # Build table structure
            table_data = self.build_table_with_bounds(rows, 
                self.centers_to_bounds(columns, 
                    min(s["x0"] for s in spans), 
                    max(s["x1"] for s in spans)), 
                columns)
            
            if len(table_data) > 1:
                return [{
                    "method": "traditional",
                    "data": table_data,
                    "rows": rows,
                    "columns": columns,
                    "confidence": 0.8
                }]
            
            return []
            
        except Exception as e:
            self.logger.warning(f"Traditional table detection failed: {e}")
            return []

    def _detect_tables_grid_based(self, spans: List[Dict]) -> List[Dict]:
        """Grid-based table detection"""
        try:
            # Create a grid of potential table cells
            grid = self._create_text_grid(spans)
            
            # Find connected regions in the grid
            table_regions = self._find_connected_regions(grid)
            
            tables = []
            for region in table_regions:
                if self._is_valid_table_region(region):
                    table_data = self._extract_table_from_grid(region, grid)
                    if table_data:
                        tables.append({
                            "method": "grid_based",
                            "data": table_data,
                            "confidence": 0.7
                        })
            
            return tables
            
        except Exception as e:
            self.logger.warning(f"Grid-based table detection failed: {e}")
            return []

    def _detect_tables_pattern_based(self, spans: List[Dict]) -> List[Dict]:
        """Pattern-based table detection using text patterns"""
        try:
            # Look for common table patterns
            patterns = [
                self._detect_separated_columns_pattern(spans),
                self._detect_aligned_text_pattern(spans),
                self._detect_mixed_content_pattern(spans)
            ]
            
            tables = []
            for pattern in patterns:
                if pattern:
                    tables.append({
                        "method": "pattern_based",
                        "data": pattern,
                        "confidence": 0.6
                    })
            
            return tables
            
        except Exception as e:
            self.logger.warning(f"Pattern-based table detection failed: {e}")
            return []

    def _create_text_grid(self, spans: List[Dict]) -> np.ndarray:
        """Create a grid representation of text spans"""
        try:
            if not spans:
                return np.array([])
            
            # Calculate grid dimensions
            min_x = min(s["x0"] for s in spans)
            max_x = max(s["x1"] for s in spans)
            min_y = min(s["y0"] for s in spans)
            max_y = max(s["y1"] for s in spans)
            
            # Create grid with appropriate resolution
            grid_width = int((max_x - min_x) / 10) + 1
            grid_height = int((max_y - min_y) / 10) + 1
            
            grid = np.zeros((grid_height, grid_width), dtype=int)
            
            # Fill grid with text spans
            for span in spans:
                x_start = int((span["x0"] - min_x) / 10)
                x_end = int((span["x1"] - min_x) / 10)
                y_start = int((span["y0"] - min_y) / 10)
                y_end = int((span["y1"] - min_y) / 10)
                
                # Ensure bounds
                x_start = max(0, min(x_start, grid_width - 1))
                x_end = max(0, min(x_end, grid_width - 1))
                y_start = max(0, min(y_start, grid_height - 1))
                y_end = max(0, min(y_end, grid_height - 1))
                
                grid[y_start:y_end+1, x_start:x_end+1] = 1
            
            return grid
            
        except Exception as e:
            self.logger.warning(f"Grid creation failed: {e}")
            return np.array([])

    def _find_connected_regions(self, grid: np.ndarray) -> List[Dict]:
        """Find connected regions in the grid"""
        try:
            if grid.size == 0:
                return []
            
            # Simple connected component analysis
            regions = []
            visited = np.zeros_like(grid, dtype=bool)
            
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    if grid[i, j] == 1 and not visited[i, j]:
                        region = self._flood_fill(grid, visited, i, j)
                        if len(region) > 10:  # Minimum region size
                            regions.append({
                                "cells": region,
                                "bounds": self._calculate_region_bounds(region)
                            })
            
            return regions
            
        except Exception as e:
            self.logger.warning(f"Connected region detection failed: {e}")
            return []

    def _flood_fill(self, grid: np.ndarray, visited: np.ndarray, start_i: int, start_j: int) -> List[Tuple[int, int]]:
        """Flood fill algorithm for connected components"""
        try:
            stack = [(start_i, start_j)]
            region = []
            
            while stack:
                i, j = stack.pop()
                
                if (i < 0 or i >= grid.shape[0] or 
                    j < 0 or j >= grid.shape[1] or 
                    visited[i, j] or grid[i, j] == 0):
                    continue
                
                visited[i, j] = True
                region.append((i, j))
                
                # Add neighbors
                stack.extend([(i+1, j), (i-1, j), (i, j+1), (i, j-1)])
            
            return region
            
        except:
            return []

    def _calculate_region_bounds(self, region: List[Tuple[int, int]]) -> Dict:
        """Calculate bounds of a region"""
        try:
            if not region:
                return {}
            
            rows = [cell[0] for cell in region]
            cols = [cell[1] for cell in region]
            
            return {
                "min_row": min(rows),
                "max_row": max(rows),
                "min_col": min(cols),
                "max_col": max(cols)
            }
            
        except:
            return {}

    def _is_valid_table_region(self, region: Dict) -> bool:
        """Check if a region represents a valid table"""
        try:
            bounds = region.get("bounds", {})
            cells = region.get("cells", [])
            
            # Check minimum size
            if len(cells) < 6:  # At least 2x3 or 3x2
                return False
            
            # Check aspect ratio
            width = bounds.get("max_col", 0) - bounds.get("min_col", 0)
            height = bounds.get("max_row", 0) - bounds.get("min_row", 0)
            
            if width == 0 or height == 0:
                return False
            
            aspect_ratio = width / height
            return 0.5 < aspect_ratio < 3.0  # Reasonable table aspect ratio
            
        except:
            return False

    def _extract_table_from_grid(self, region: Dict, grid: np.ndarray) -> List[List[str]]:
        """Extract table data from grid region"""
        try:
            bounds = region["bounds"]
            cells = region["cells"]
            
            # Group cells by rows
            rows = {}
            for cell in cells:
                row = cell[0]
                if row not in rows:
                    rows[row] = []
                rows[row].append(cell)
            
            # Sort rows and extract data
            table_data = []
            for row in sorted(rows.keys()):
                row_cells = sorted(rows[row], key=lambda x: x[1])
                row_data = []
                
                for cell in row_cells:
                    # Extract text from grid cell (simplified)
                    row_data.append(f"Cell_{cell[0]}_{cell[1]}")
                
                table_data.append(row_data)
            
            return table_data
            
        except Exception as e:
            self.logger.warning(f"Table extraction from grid failed: {e}")
            return []

    def _detect_separated_columns_pattern(self, spans: List[Dict]) -> Optional[List[List[str]]]:
        """Detect tables with clearly separated columns"""
        try:
            # Look for spans with significant horizontal gaps
            spans.sort(key=lambda s: s["x0"])
            
            gaps = []
            for i in range(len(spans) - 1):
                gap = spans[i+1]["x0"] - spans[i]["x1"]
                if gap > 20:  # Significant gap
                    gaps.append((i, gap))
            
            if len(gaps) < 2:  # Need at least 2 gaps for columns
                return None
            
            # Group spans by columns based on gaps
            columns = []
            current_column = []
            
            for i, span in enumerate(spans):
                current_column.append(span)
                
                # Check if next span starts a new column
                if i < len(spans) - 1:
                    gap = spans[i+1]["x0"] - span["x1"]
                    if gap > 20:
                        columns.append(current_column)
                        current_column = []
            
            if current_column:
                columns.append(current_column)
            
            if len(columns) >= 2:
                # Convert to table format
                max_rows = max(len(col) for col in columns)
                table_data = []
                
                for row_idx in range(max_rows):
                    row = []
                    for col in columns:
                        if row_idx < len(col):
                            row.append(col[row_idx]["text"])
                        else:
                            row.append("")
                    table_data.append(row)
                
                return table_data
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Separated columns pattern detection failed: {e}")
            return None

    def _detect_aligned_text_pattern(self, spans: List[Dict]) -> Optional[List[List[str]]]:
        """Detect tables with aligned text patterns"""
        try:
            # Group spans by similar x-coordinates (columns)
            x_groups = {}
            tolerance = 30  # Pixels
            
            for span in spans:
                x_pos = span["x0"]
                grouped = False
                
                for group_x in x_groups:
                    if abs(x_pos - group_x) < tolerance:
                        x_groups[group_x].append(span)
                        grouped = True
                        break
                
                if not grouped:
                    x_groups[x_pos] = [span]
            
            # Check if we have multiple columns
            if len(x_groups) < 2:
                return None
            
            # Sort columns by x-position
            sorted_columns = sorted(x_groups.items(), key=lambda x: x[0])
            
            # Create table data
            max_rows = max(len(col[1]) for col in sorted_columns)
            table_data = []
            
            for row_idx in range(max_rows):
                row = []
                for _, column_spans in sorted_columns:
                    if row_idx < len(column_spans):
                        row.append(column_spans[row_idx]["text"])
                    else:
                        row.append("")
                table_data.append(row)
            
            return table_data if len(table_data) > 1 else None
            
        except Exception as e:
            self.logger.warning(f"Aligned text pattern detection failed: {e}")
            return None

    def _detect_mixed_content_pattern(self, spans: List[Dict]) -> Optional[List[List[str]]]:
        """Detect tables with mixed content (text and numbers)"""
        try:
            # Look for patterns of mixed content
            text_spans = [s for s in spans if not self._contains_digits(s["text"])]
            numeric_spans = [s for s in spans if self._contains_digits(s["text"])]
            
            if len(numeric_spans) < 2 or len(text_spans) < 2:
                return None
            
            # Try to organize by spatial proximity
            all_spans = text_spans + numeric_spans
            rows = self.cluster_rows(all_spans)
            
            if len(rows) < 2:
                return None
            
            # Check if rows have mixed content
            mixed_rows = 0
            for row in rows:
                has_text = any(not self._contains_digits(s["text"]) for s in row)
                has_numbers = any(self._contains_digits(s["text"]) for s in row)
                if has_text and has_numbers:
                    mixed_rows += 1
            
            if mixed_rows >= 2:
                # Convert to table format
                columns = self.detect_template_columns(rows)
                if columns:
                    table_data = self.build_table_with_bounds(rows, 
                        self.centers_to_bounds(columns, 
                            min(s["x0"] for s in all_spans), 
                            max(s["x1"] for s in all_spans)), 
                        columns)
                    return table_data
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Mixed content pattern detection failed: {e}")
            return None

    def _deduplicate_tables(self, tables: List[Dict]) -> List[Dict]:
        """Remove duplicate tables from the list"""
        try:
            unique_tables = []
            
            for table in tables:
                is_duplicate = False
                table_data = table.get("data", [])
                
                for existing_table in unique_tables:
                    existing_data = existing_table.get("data", [])
                    
                    # Check if tables are similar
                    if self._are_tables_similar(table_data, existing_data):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    unique_tables.append(table)
            
            return unique_tables
            
        except Exception as e:
            self.logger.warning(f"Table deduplication failed: {e}")
            return tables

    def _are_tables_similar(self, table1: List[List[str]], table2: List[List[str]]) -> bool:
        """Check if two tables are similar"""
        try:
            if len(table1) != len(table2):
                return False
            
            if len(table1) == 0:
                return True
            
            if len(table1[0]) != len(table2[0]):
                return False
            
            # Check similarity of content
            matches = 0
            total_cells = len(table1) * len(table1[0])
            
            for i in range(len(table1)):
                for j in range(len(table1[0])):
                    if table1[i][j].strip() == table2[i][j].strip():
                        matches += 1
            
            similarity = matches / total_cells if total_cells > 0 else 0
            return similarity > 0.8  # 80% similarity threshold
            
        except:
            return False

    def _score_table_quality(self, tables: List[Dict]) -> List[Dict]:
        """Score and rank tables by quality"""
        try:
            scored_tables = []
            
            for table in tables:
                score = self._calculate_table_quality_score(table)
                table["quality_score"] = score
                scored_tables.append(table)
            
            # Sort by quality score (descending)
            scored_tables.sort(key=lambda x: x.get("quality_score", 0), reverse=True)
            
            return scored_tables
            
        except Exception as e:
            self.logger.warning(f"Table scoring failed: {e}")
            return tables

    def _calculate_table_quality_score(self, table: Dict) -> float:
        """Calculate quality score for a table"""
        try:
            table_data = table.get("data", [])
            if not table_data:
                return 0.0
            
            score = 0.0
            
            # Score based on number of rows and columns
            rows = len(table_data)
            cols = len(table_data[0]) if table_data else 0
            
            if rows >= 3 and cols >= 2:
                score += 0.3
            
            # Score based on data consistency
            consistency_score = self._calculate_data_consistency(table_data)
            score += consistency_score * 0.4
            
            # Score based on header detection
            if self._has_good_header(table_data):
                score += 0.2
            
            # Score based on numeric content
            numeric_score = self._calculate_numeric_content_score(table_data)
            score += numeric_score * 0.1
            
            return min(score, 1.0)
            
        except:
            return 0.0

    def _calculate_data_consistency(self, table_data: List[List[str]]) -> float:
        """Calculate data consistency score"""
        try:
            if len(table_data) < 2:
                return 0.0
            
            # Check column width consistency
            col_widths = [len(row) for row in table_data]
            if len(set(col_widths)) == 1:  # All rows have same width
                return 1.0
            else:
                return 0.5
            
        except:
            return 0.0

    def _has_good_header(self, table_data: List[List[str]]) -> bool:
        """Check if table has a good header row"""
        try:
            if len(table_data) < 2:
                return False
            
            header_row = table_data[0]
            # Check if header has mostly non-numeric content
            non_numeric_count = sum(1 for cell in header_row 
                                  if not self._is_pure_number(cell.strip()))
            
            return non_numeric_count >= len(header_row) * 0.7
            
        except:
            return False

    def _calculate_numeric_content_score(self, table_data: List[List[str]]) -> float:
        """Calculate score based on numeric content"""
        try:
            if not table_data:
                return 0.0
            
            total_cells = 0
            numeric_cells = 0
            
            for row in table_data:
                for cell in row:
                    total_cells += 1
                    if self._contains_digits(cell.strip()):
                        numeric_cells += 1
            
            return numeric_cells / total_cells if total_cells > 0 else 0.0
            
        except:
            return 0.0
