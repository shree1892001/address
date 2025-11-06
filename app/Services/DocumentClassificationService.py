import re
import statistics
from typing import List, Dict, Tuple, Optional
from app.Logger.ocr_logger import get_standard_logger
from app.Exceptions.custom_exceptions import (
    handle_general_operations, log_method_entry_exit, ExceptionSeverity
)


class DocumentClassificationService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DocumentClassificationService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.logger = get_standard_logger("DocumentClassificationService")
            self._initialized = True

    @log_method_entry_exit
    @handle_general_operations(severity=ExceptionSeverity.MEDIUM)
    def classify_document_content(self, spans: List[Dict]) -> Dict:
        """Classify document content and prioritize tables"""
        try:
            if not spans:
                return {
                    "document_type": "empty",
                    "content_priority": [],
                    "table_regions": [],
                    "text_regions": [],
                    "confidence": 0.0
                }
            
            # Analyze document structure
            document_analysis = self._analyze_document_structure(spans)
            
            # Detect table regions
            table_regions = self._detect_table_regions(spans)
            
            # Detect text regions
            text_regions = self._detect_text_regions(spans, table_regions)
            
            # Classify document type
            document_type = self._classify_document_type(document_analysis, table_regions, text_regions)
            
            # Prioritize content (tables first, then text)
            content_priority = self._prioritize_content(table_regions, text_regions)
            
            # Calculate confidence
            confidence = self._calculate_classification_confidence(document_analysis, table_regions, text_regions)
            
            result = {
                "document_type": document_type,
                "content_priority": content_priority,
                "table_regions": table_regions,
                "text_regions": text_regions,
                "confidence": confidence,
                "analysis": document_analysis
            }
            
            self.logger.info(f"Document classified as: {document_type} (confidence: {confidence:.2f})")
            return result
            
        except Exception as e:
            self.logger.error(f"Document classification failed: {e}")
            return {
                "document_type": "unknown",
                "content_priority": [],
                "table_regions": [],
                "text_regions": [],
                "confidence": 0.0
            }

    def _analyze_document_structure(self, spans: List[Dict]) -> Dict:
        """Analyze the overall structure of the document"""
        try:
            if not spans:
                return {}
            
            # Calculate document dimensions
            min_x = min(s["x0"] for s in spans)
            max_x = max(s["x1"] for s in spans)
            min_y = min(s["y0"] for s in spans)
            max_y = max(s["y1"] for s in spans)
            
            # Analyze text density
            total_area = (max_x - min_x) * (max_y - min_y)
            text_area = sum((s["x1"] - s["x0"]) * (s["y1"] - s["y0"]) for s in spans)
            text_density = text_area / total_area if total_area > 0 else 0
            
            # Analyze spacing patterns
            spacing_analysis = self._analyze_spacing_patterns(spans)
            
            # Analyze content types
            content_analysis = self._analyze_content_types(spans)
            
            return {
                "dimensions": {
                    "width": max_x - min_x,
                    "height": max_y - min_y,
                    "min_x": min_x,
                    "max_x": max_x,
                    "min_y": min_y,
                    "max_y": max_y
                },
                "text_density": text_density,
                "spacing_analysis": spacing_analysis,
                "content_analysis": content_analysis,
                "total_spans": len(spans)
            }
            
        except Exception as e:
            self.logger.warning(f"Document structure analysis failed: {e}")
            return {}

    def _analyze_spacing_patterns(self, spans: List[Dict]) -> Dict:
        """Analyze spacing patterns in the document"""
        try:
            if len(spans) < 2:
                return {"regular_spacing": False, "spacing_variance": 0.0}
            
            # Calculate horizontal and vertical gaps
            spans_sorted_x = sorted(spans, key=lambda s: s["x0"])
            spans_sorted_y = sorted(spans, key=lambda s: s["y0"])
            
            x_gaps = []
            y_gaps = []
            
            for i in range(len(spans_sorted_x) - 1):
                gap = spans_sorted_x[i+1]["x0"] - spans_sorted_x[i]["x1"]
                if gap > 0:
                    x_gaps.append(gap)
            
            for i in range(len(spans_sorted_y) - 1):
                gap = spans_sorted_y[i+1]["y0"] - spans_sorted_y[i]["y1"]
                if gap > 0:
                    y_gaps.append(gap)
            
            # Analyze gap patterns
            x_gap_variance = statistics.variance(x_gaps) if len(x_gaps) > 1 else 0
            y_gap_variance = statistics.variance(y_gaps) if len(y_gaps) > 1 else 0
            
            # Check for regular spacing (low variance indicates regular patterns)
            regular_x_spacing = x_gap_variance < 100 if x_gaps else False
            regular_y_spacing = y_gap_variance < 100 if y_gaps else False
            
            return {
                "regular_spacing": regular_x_spacing or regular_y_spacing,
                "x_gap_variance": x_gap_variance,
                "y_gap_variance": y_gap_variance,
                "avg_x_gap": statistics.mean(x_gaps) if x_gaps else 0,
                "avg_y_gap": statistics.mean(y_gaps) if y_gaps else 0
            }
            
        except Exception as e:
            self.logger.warning(f"Spacing pattern analysis failed: {e}")
            return {"regular_spacing": False, "spacing_variance": 0.0}

    def _analyze_content_types(self, spans: List[Dict]) -> Dict:
        """Analyze the types of content in the document"""
        try:
            total_spans = len(spans)
            if total_spans == 0:
                return {}
            
            # Count different content types
            numeric_spans = 0
            text_spans = 0
            mixed_spans = 0
            short_spans = 0
            long_spans = 0
            
            for span in spans:
                text = span.get("text", "")
                
                # Check if numeric
                if re.search(r'^\d+$', text.strip()):
                    numeric_spans += 1
                elif re.search(r'\d', text):
                    mixed_spans += 1
                else:
                    text_spans += 1
                
                # Check length
                if len(text.strip()) <= 3:
                    short_spans += 1
                elif len(text.strip()) > 20:
                    long_spans += 1
            
            return {
                "numeric_ratio": numeric_spans / total_spans,
                "text_ratio": text_spans / total_spans,
                "mixed_ratio": mixed_spans / total_spans,
                "short_text_ratio": short_spans / total_spans,
                "long_text_ratio": long_spans / total_spans
            }
            
        except Exception as e:
            self.logger.warning(f"Content type analysis failed: {e}")
            return {}

    def _detect_table_regions(self, spans: List[Dict]) -> List[Dict]:
        """Detect regions that likely contain tables"""
        try:
            if not spans:
                return []
            
            # Group spans by spatial proximity
            regions = self._group_spans_by_proximity(spans)
            
            table_regions = []
            for region in regions:
                if self._is_likely_table_region(region):
                    table_regions.append({
                        "spans": region,
                        "bounds": self._calculate_region_bounds(region),
                        "confidence": self._calculate_table_confidence(region),
                        "type": "table"
                    })
            
            # Sort by confidence (highest first)
            table_regions.sort(key=lambda x: x["confidence"], reverse=True)
            
            self.logger.info(f"Detected {len(table_regions)} table regions")
            return table_regions
            
        except Exception as e:
            self.logger.error(f"Table region detection failed: {e}")
            return []

    def _detect_text_regions(self, spans: List[Dict], table_regions: List[Dict]) -> List[Dict]:
        """Detect regions that contain regular text (non-table)"""
        try:
            if not spans:
                return []
            
            # Get spans that are not part of table regions
            table_span_ids = set()
            for table_region in table_regions:
                for span in table_region["spans"]:
                    table_span_ids.add(id(span))
            
            non_table_spans = [span for span in spans if id(span) not in table_span_ids]
            
            if not non_table_spans:
                return []
            
            # Group remaining spans by proximity
            text_regions = self._group_spans_by_proximity(non_table_spans)
            
            # Filter and format text regions
            formatted_regions = []
            for region in text_regions:
                if len(region) >= 2:  # Minimum spans for a text region
                    formatted_regions.append({
                        "spans": region,
                        "bounds": self._calculate_region_bounds(region),
                        "confidence": self._calculate_text_confidence(region),
                        "type": "text"
                    })
            
            # Sort by position (top to bottom, left to right)
            formatted_regions.sort(key=lambda x: (x["bounds"]["min_y"], x["bounds"]["min_x"]))
            
            self.logger.info(f"Detected {len(formatted_regions)} text regions")
            return formatted_regions
            
        except Exception as e:
            self.logger.error(f"Text region detection failed: {e}")
            return []

    def _group_spans_by_proximity(self, spans: List[Dict]) -> List[List[Dict]]:
        """Group spans by spatial proximity"""
        try:
            if not spans:
                return []
            
            # Sort spans by position
            spans.sort(key=lambda s: (s["y0"], s["x0"]))
            
            regions = []
            current_region = [spans[0]]
            
            for span in spans[1:]:
                if self._is_spatially_close(span, current_region):
                    current_region.append(span)
                else:
                    if len(current_region) >= 2:  # Minimum spans for a region
                        regions.append(current_region)
                    current_region = [span]
            
            # Add the last region
            if len(current_region) >= 2:
                regions.append(current_region)
            
            return regions
            
        except Exception as e:
            self.logger.warning(f"Span grouping failed: {e}")
            return []

    def _is_spatially_close(self, span: Dict, region: List[Dict]) -> bool:
        """Check if a span is spatially close to a region"""
        try:
            if not region:
                return False
            
            # Calculate region bounds
            region_x0 = min(s["x0"] for s in region)
            region_x1 = max(s["x1"] for s in region)
            region_y0 = min(s["y0"] for s in region)
            region_y1 = max(s["y1"] for s in region)
            
            # Check proximity
            x_distance = min(abs(span["x0"] - region_x1), abs(span["x1"] - region_x0))
            y_distance = min(abs(span["y0"] - region_y1), abs(span["y1"] - region_y0))
            
            # Allow some overlap or close proximity
            return x_distance < 50 and y_distance < 30
            
        except:
            return False

    def _is_likely_table_region(self, region: List[Dict]) -> bool:
        """Check if a region is likely to contain a table"""
        try:
            if len(region) < 3:
                return False
            
            # Check for table-like characteristics
            x_positions = [s["x0"] for s in region]
            y_positions = [s["y0"] for s in region]
            
            # Check for column-like alignment
            x_alignment_score = self._calculate_alignment_score(x_positions)
            y_alignment_score = self._calculate_alignment_score(y_positions)
            
            # Check for numeric content (common in tables)
            numeric_ratio = self._calculate_numeric_ratio(region)
            
            # Check for regular spacing
            spacing_score = self._calculate_spacing_score(region)
            
            # Table-like if it has good alignment, some numeric content, and regular spacing
            return (x_alignment_score > 0.3 or y_alignment_score > 0.3) and \
                   numeric_ratio > 0.1 and \
                   spacing_score > 0.3
            
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
                if pos - current_cluster[-1] < 25:  # Within 25 pixels
                    current_cluster.append(pos)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [pos]
            clusters.append(current_cluster)
            
            # Score based on number of distinct clusters (more clusters = better alignment)
            return min(len(clusters) / len(positions), 1.0)
            
        except:
            return 0.0

    def _calculate_numeric_ratio(self, region: List[Dict]) -> float:
        """Calculate ratio of numeric content in region"""
        try:
            total_text = " ".join(s.get("text", "") for s in region)
            numeric_chars = sum(1 for c in total_text if c.isdigit())
            total_chars = len(total_text)
            
            return numeric_chars / total_chars if total_chars > 0 else 0.0
            
        except:
            return 0.0

    def _calculate_spacing_score(self, region: List[Dict]) -> float:
        """Calculate spacing regularity score"""
        try:
            if len(region) < 2:
                return 0.0
            
            # Calculate gaps between spans
            region_sorted = sorted(region, key=lambda s: s["x0"])
            gaps = []
            
            for i in range(len(region_sorted) - 1):
                gap = region_sorted[i+1]["x0"] - region_sorted[i]["x1"]
                if gap > 0:
                    gaps.append(gap)
            
            if not gaps:
                return 0.0
            
            # Calculate variance in gaps (lower variance = more regular spacing)
            if len(gaps) > 1:
                gap_variance = statistics.variance(gaps)
                # Convert variance to score (lower variance = higher score)
                return max(0, 1 - (gap_variance / 1000))
            else:
                return 0.5
            
        except:
            return 0.0

    def _calculate_table_confidence(self, region: List[Dict]) -> float:
        """Calculate confidence that a region contains a table"""
        try:
            if not region:
                return 0.0
            
            confidence = 0.0
            
            # Base confidence from region size
            if len(region) >= 6:
                confidence += 0.3
            elif len(region) >= 3:
                confidence += 0.2
            
            # Alignment confidence
            x_positions = [s["x0"] for s in region]
            y_positions = [s["y0"] for s in region]
            x_alignment = self._calculate_alignment_score(x_positions)
            y_alignment = self._calculate_alignment_score(y_positions)
            confidence += max(x_alignment, y_alignment) * 0.3
            
            # Numeric content confidence
            numeric_ratio = self._calculate_numeric_ratio(region)
            confidence += numeric_ratio * 0.2
            
            # Spacing confidence
            spacing_score = self._calculate_spacing_score(region)
            confidence += spacing_score * 0.2
            
            return min(confidence, 1.0)
            
        except:
            return 0.0

    def _calculate_text_confidence(self, region: List[Dict]) -> float:
        """Calculate confidence that a region contains regular text"""
        try:
            if not region:
                return 0.0
            
            confidence = 0.0
            
            # Base confidence from region size
            if len(region) >= 3:
                confidence += 0.4
            elif len(region) >= 2:
                confidence += 0.3
            
            # Text content confidence (less numeric = more likely text)
            numeric_ratio = self._calculate_numeric_ratio(region)
            confidence += (1 - numeric_ratio) * 0.3
            
            # Length confidence (longer text = more likely regular text)
            avg_text_length = statistics.mean(len(s.get("text", "")) for s in region)
            if avg_text_length > 10:
                confidence += 0.3
            elif avg_text_length > 5:
                confidence += 0.2
            
            return min(confidence, 1.0)
            
        except:
            return 0.0

    def _calculate_region_bounds(self, region: List[Dict]) -> Dict:
        """Calculate bounds of a region"""
        try:
            if not region:
                return {}
            
            return {
                "min_x": min(s["x0"] for s in region),
                "max_x": max(s["x1"] for s in region),
                "min_y": min(s["y0"] for s in region),
                "max_y": max(s["y1"] for s in region),
                "width": max(s["x1"] for s in region) - min(s["x0"] for s in region),
                "height": max(s["y1"] for s in region) - min(s["y0"] for s in region)
            }
            
        except:
            return {}

    def _classify_document_type(self, analysis: Dict, table_regions: List[Dict], text_regions: List[Dict]) -> str:
        """Classify the type of document"""
        try:
            if not analysis:
                return "unknown"
            
            # Count regions
            table_count = len(table_regions)
            text_count = len(text_regions)
            total_regions = table_count + text_count
            
            if total_regions == 0:
                return "empty"
            
            # Calculate ratios
            table_ratio = table_count / total_regions
            text_ratio = text_count / total_regions
            
            # Get content analysis
            content_analysis = analysis.get("content_analysis", {})
            numeric_ratio = content_analysis.get("numeric_ratio", 0)
            
            # Classify based on characteristics
            if table_ratio > 0.7:
                return "table_dominant"
            elif table_ratio > 0.3:
                return "mixed_content"
            elif numeric_ratio > 0.3:
                return "data_document"
            elif text_ratio > 0.7:
                return "text_document"
            else:
                return "mixed_content"
                
        except Exception as e:
            self.logger.warning(f"Document type classification failed: {e}")
            return "unknown"

    def _prioritize_content(self, table_regions: List[Dict], text_regions: List[Dict]) -> List[Dict]:
        """Prioritize content with tables first, then text"""
        try:
            priority_list = []
            
            # Add table regions first (sorted by confidence)
            for table_region in table_regions:
                priority_list.append({
                    "type": "table",
                    "region": table_region,
                    "priority": 1,  # Highest priority
                    "confidence": table_region.get("confidence", 0.0)
                })
            
            # Add text regions second (sorted by position)
            for text_region in text_regions:
                priority_list.append({
                    "type": "text",
                    "region": text_region,
                    "priority": 2,  # Lower priority
                    "confidence": text_region.get("confidence", 0.0)
                })
            
            # Sort by priority, then by confidence
            priority_list.sort(key=lambda x: (x["priority"], -x["confidence"]))
            
            self.logger.info(f"Content prioritized: {len(table_regions)} tables, {len(text_regions)} text regions")
            return priority_list
            
        except Exception as e:
            self.logger.error(f"Content prioritization failed: {e}")
            return []

    def _calculate_classification_confidence(self, analysis: Dict, table_regions: List[Dict], text_regions: List[Dict]) -> float:
        """Calculate overall confidence in the classification"""
        try:
            if not analysis:
                return 0.0
            
            confidence = 0.0
            
            # Base confidence from analysis completeness
            if analysis.get("dimensions") and analysis.get("content_analysis"):
                confidence += 0.3
            
            # Confidence from region detection
            total_regions = len(table_regions) + len(text_regions)
            if total_regions > 0:
                confidence += 0.3
            
            # Confidence from region quality
            if table_regions:
                avg_table_confidence = statistics.mean(r.get("confidence", 0) for r in table_regions)
                confidence += avg_table_confidence * 0.2
            
            if text_regions:
                avg_text_confidence = statistics.mean(r.get("confidence", 0) for r in text_regions)
                confidence += avg_text_confidence * 0.2
            
            return min(confidence, 1.0)
            
        except:
            return 0.0
