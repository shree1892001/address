import statistics
import numpy as np
from typing import List, Dict, Tuple, Optional
from app.Logger.ocr_logger import get_standard_logger
from app.Exceptions.custom_exceptions import (
    handle_table_extraction, log_method_entry_exit, ExceptionSeverity
)


class AdaptiveTableDetectionService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AdaptiveTableDetectionService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.logger = get_standard_logger("AdaptiveTableDetectionService")
            self._initialized = True

    @log_method_entry_exit
    @handle_table_extraction(severity=ExceptionSeverity.HIGH)
    def detect_tables_adaptively(self, spans: List[Dict]) -> List[Dict]:
        """Dynamically detect tables without hardcoded logic"""
        try:
            if not spans:
                return []
            
            # Analyze document characteristics dynamically
            doc_characteristics = self._analyze_document_characteristics(spans)
            
            # Generate adaptive thresholds based on document
            adaptive_thresholds = self._generate_adaptive_thresholds(doc_characteristics)
            
            # Detect table regions using adaptive methods
            table_regions = self._detect_table_regions_adaptively(spans, adaptive_thresholds)
            
            # Extract tables from regions
            tables = []
            for region in table_regions:
                table_data = self._extract_table_from_region_adaptively(region, doc_characteristics)
                if table_data:
                    tables.append(table_data)
            
            # Score and rank tables
            scored_tables = self._score_tables_adaptively(tables, doc_characteristics)
            
            self.logger.info(f"Adaptive detection found {len(scored_tables)} tables")
            return scored_tables
            
        except Exception as e:
            self.logger.error(f"Adaptive table detection failed: {e}")
            return []

    def _analyze_document_characteristics(self, spans: List[Dict]) -> Dict:
        """Analyze document characteristics to adapt detection parameters"""
        try:
            if not spans:
                return {}
            
            # Calculate document dimensions
            min_x = min(s["x0"] for s in spans)
            max_x = max(s["x1"] for s in spans)
            min_y = min(s["y0"] for s in spans)
            max_y = max(s["y1"] for s in spans)
            
            # Analyze text distribution
            text_distribution = self._analyze_text_distribution(spans)
            
            # Analyze spacing patterns
            spacing_patterns = self._analyze_spacing_patterns(spans)
            
            # Analyze content types
            content_analysis = self._analyze_content_types(spans)
            
            # Calculate density metrics
            density_metrics = self._calculate_density_metrics(spans, min_x, max_x, min_y, max_y)
            
            return {
                "dimensions": {
                    "width": max_x - min_x,
                    "height": max_y - min_y,
                    "min_x": min_x,
                    "max_x": max_x,
                    "min_y": min_y,
                    "max_y": max_y
                },
                "text_distribution": text_distribution,
                "spacing_patterns": spacing_patterns,
                "content_analysis": content_analysis,
                "density_metrics": density_metrics,
                "total_spans": len(spans)
            }
            
        except Exception as e:
            self.logger.warning(f"Document characteristics analysis failed: {e}")
            return {}

    def _analyze_text_distribution(self, spans: List[Dict]) -> Dict:
        """Analyze how text is distributed across the document"""
        try:
            if not spans:
                return {}
            
            # Calculate text density in different regions
            x_positions = [s["x0"] for s in spans]
            y_positions = [s["y0"] for s in spans]
            
            # Divide document into regions and analyze density
            x_quartiles = np.percentile(x_positions, [25, 50, 75])
            y_quartiles = np.percentile(y_positions, [25, 50, 75])
            
            # Calculate density in each region
            region_densities = {}
            for i, x_q in enumerate(x_quartiles):
                for j, y_q in enumerate(y_quartiles):
                    region_spans = [s for s in spans 
                                  if (i == 0 or s["x0"] >= x_quartiles[i-1]) and 
                                     (i == 3 or s["x0"] < x_q) and
                                     (j == 0 or s["y0"] >= y_quartiles[j-1]) and 
                                     (j == 3 or s["y0"] < y_q)]
                    region_densities[f"region_{i}_{j}"] = len(region_spans)
            
            # Calculate distribution metrics
            densities = list(region_densities.values())
            distribution_variance = np.var(densities) if densities else 0
            distribution_mean = np.mean(densities) if densities else 0
            
            return {
                "region_densities": region_densities,
                "distribution_variance": distribution_variance,
                "distribution_mean": distribution_mean,
                "is_uniform": distribution_variance < distribution_mean * 0.5
            }
            
        except Exception as e:
            self.logger.warning(f"Text distribution analysis failed: {e}")
            return {}

    def _analyze_spacing_patterns(self, spans: List[Dict]) -> Dict:
        """Analyze spacing patterns in the document"""
        try:
            if len(spans) < 2:
                return {"regular_spacing": False, "spacing_variance": 0.0}
            
            # Calculate gaps between spans
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
            x_gap_analysis = self._analyze_gap_patterns(x_gaps)
            y_gap_analysis = self._analyze_gap_patterns(y_gaps)
            
            return {
                "x_gaps": x_gap_analysis,
                "y_gaps": y_gap_analysis,
                "has_regular_spacing": x_gap_analysis.get("is_regular", False) or y_gap_analysis.get("is_regular", False)
            }
            
        except Exception as e:
            self.logger.warning(f"Spacing pattern analysis failed: {e}")
            return {"has_regular_spacing": False}

    def _analyze_gap_patterns(self, gaps: List[float]) -> Dict:
        """Analyze patterns in gaps"""
        try:
            if len(gaps) < 2:
                return {"is_regular": False, "variance": 0.0, "mean": 0.0}
            
            mean_gap = np.mean(gaps)
            variance = np.var(gaps)
            std_dev = np.std(gaps)
            
            # Determine if gaps are regular (low coefficient of variation)
            cv = std_dev / mean_gap if mean_gap > 0 else float('inf')
            is_regular = cv < 0.3  # 30% coefficient of variation threshold
            
            # Find common gap sizes
            gap_clusters = self._cluster_gaps(gaps)
            
            return {
                "is_regular": is_regular,
                "variance": variance,
                "mean": mean_gap,
                "std_dev": std_dev,
                "cv": cv,
                "gap_clusters": gap_clusters
            }
            
        except Exception as e:
            self.logger.warning(f"Gap pattern analysis failed: {e}")
            return {"is_regular": False, "variance": 0.0, "mean": 0.0}

    def _cluster_gaps(self, gaps: List[float]) -> List[Dict]:
        """Cluster gaps to find common sizes"""
        try:
            if len(gaps) < 2:
                return []
            
            # Simple clustering based on proximity
            gaps_sorted = sorted(gaps)
            clusters = []
            current_cluster = [gaps_sorted[0]]
            
            for gap in gaps_sorted[1:]:
                # If gap is close to current cluster, add it
                if gap - current_cluster[-1] < np.std(gaps) * 0.5:
                    current_cluster.append(gap)
                else:
                    # Start new cluster
                    if len(current_cluster) > 1:
                        clusters.append({
                            "center": np.mean(current_cluster),
                            "size": len(current_cluster),
                            "gaps": current_cluster
                        })
                    current_cluster = [gap]
            
            # Add the last cluster
            if len(current_cluster) > 1:
                clusters.append({
                    "center": np.mean(current_cluster),
                    "size": len(current_cluster),
                    "gaps": current_cluster
                })
            
            return clusters
            
        except Exception as e:
            self.logger.warning(f"Gap clustering failed: {e}")
            return []

    def _analyze_content_types(self, spans: List[Dict]) -> Dict:
        """Analyze content types without hardcoded patterns"""
        try:
            if not spans:
                return {}
            
            # Analyze character types dynamically
            all_text = " ".join(s.get("text", "") for s in spans)
            
            # Count different character categories
            char_counts = {
                "letters": sum(1 for c in all_text if c.isalpha()),
                "digits": sum(1 for c in all_text if c.isdigit()),
                "spaces": sum(1 for c in all_text if c.isspace()),
                "punctuation": sum(1 for c in all_text if not c.isalnum() and not c.isspace()),
                "total": len(all_text)
            }
            
            # Calculate ratios
            total_chars = char_counts["total"]
            if total_chars == 0:
                return {"char_ratios": {}, "content_type": "empty"}
            
            char_ratios = {
                "letter_ratio": char_counts["letters"] / total_chars,
                "digit_ratio": char_counts["digits"] / total_chars,
                "space_ratio": char_counts["spaces"] / total_chars,
                "punctuation_ratio": char_counts["punctuation"] / total_chars
            }
            
            # Determine content type based on ratios
            content_type = self._determine_content_type(char_ratios)
            
            return {
                "char_counts": char_counts,
                "char_ratios": char_ratios,
                "content_type": content_type
            }
            
        except Exception as e:
            self.logger.warning(f"Content type analysis failed: {e}")
            return {"content_type": "unknown"}

    def _determine_content_type(self, char_ratios: Dict) -> str:
        """Determine content type based on character ratios"""
        try:
            letter_ratio = char_ratios.get("letter_ratio", 0)
            digit_ratio = char_ratios.get("digit_ratio", 0)
            space_ratio = char_ratios.get("space_ratio", 0)
            
            # Dynamic classification based on ratios
            if digit_ratio > 0.3:
                return "numeric_heavy"
            elif letter_ratio > 0.7:
                return "text_heavy"
            elif 0.2 < digit_ratio < 0.4 and 0.3 < letter_ratio < 0.6:
                return "mixed_content"
            elif space_ratio > 0.2:
                return "sparse_content"
            else:
                return "balanced_content"
                
        except Exception as e:
            self.logger.warning(f"Content type determination failed: {e}")
            return "unknown"

    def _calculate_density_metrics(self, spans: List[Dict], min_x: float, max_x: float, min_y: float, max_y: float) -> Dict:
        """Calculate density metrics for the document"""
        try:
            if not spans:
                return {}
            
            # Calculate total text area
            total_text_area = sum((s["x1"] - s["x0"]) * (s["y1"] - s["y0"]) for s in spans)
            document_area = (max_x - min_x) * (max_y - min_y)
            
            # Calculate density
            text_density = total_text_area / document_area if document_area > 0 else 0
            
            # Calculate span density (number of spans per unit area)
            span_density = len(spans) / document_area if document_area > 0 else 0
            
            # Calculate average span size
            avg_span_size = total_text_area / len(spans) if spans else 0
            
            return {
                "text_density": text_density,
                "span_density": span_density,
                "avg_span_size": avg_span_size,
                "total_text_area": total_text_area,
                "document_area": document_area
            }
            
        except Exception as e:
            self.logger.warning(f"Density metrics calculation failed: {e}")
            return {}

    def _generate_adaptive_thresholds(self, doc_characteristics: Dict) -> Dict:
        """Generate adaptive thresholds based on document characteristics"""
        try:
            dimensions = doc_characteristics.get("dimensions", {})
            density_metrics = doc_characteristics.get("density_metrics", {})
            content_analysis = doc_characteristics.get("content_analysis", {})
            
            # Base thresholds
            base_thresholds = {
                "min_spans_per_region": 3,
                "proximity_tolerance": 50,
                "alignment_tolerance": 25,
                "min_region_size": 6,
                "quality_threshold": 0.5
            }
            
            # Adapt based on document size
            doc_width = dimensions.get("width", 600)
            doc_height = dimensions.get("height", 800)
            
            if doc_width > 1000 or doc_height > 1000:
                # Large document - be more lenient
                base_thresholds["proximity_tolerance"] *= 1.5
                base_thresholds["alignment_tolerance"] *= 1.5
                base_thresholds["min_spans_per_region"] = 2
            elif doc_width < 400 or doc_height < 400:
                # Small document - be more strict
                base_thresholds["proximity_tolerance"] *= 0.7
                base_thresholds["alignment_tolerance"] *= 0.7
                base_thresholds["min_spans_per_region"] = 4
            
            # Adapt based on density
            text_density = density_metrics.get("text_density", 0.1)
            if text_density > 0.3:
                # High density - be more strict
                base_thresholds["quality_threshold"] = 0.6
                base_thresholds["min_region_size"] = 8
            elif text_density < 0.05:
                # Low density - be more lenient
                base_thresholds["quality_threshold"] = 0.3
                base_thresholds["min_region_size"] = 4
            
            # Adapt based on content type
            content_type = content_analysis.get("content_type", "unknown")
            if content_type == "numeric_heavy":
                # Numeric content - likely tables
                base_thresholds["quality_threshold"] *= 0.8
            elif content_type == "text_heavy":
                # Text content - less likely to be tables
                base_thresholds["quality_threshold"] *= 1.2
            
            return base_thresholds
            
        except Exception as e:
            self.logger.warning(f"Adaptive threshold generation failed: {e}")
            return {
                "min_spans_per_region": 3,
                "proximity_tolerance": 50,
                "alignment_tolerance": 25,
                "min_region_size": 6,
                "quality_threshold": 0.5
            }

    def _detect_table_regions_adaptively(self, spans: List[Dict], thresholds: Dict) -> List[Dict]:
        """Detect table regions using adaptive methods"""
        try:
            if not spans:
                return []
            
            # Group spans by spatial proximity
            regions = self._group_spans_adaptively(spans, thresholds)
            
            # Analyze each region for table characteristics
            table_regions = []
            for region in regions:
                if self._is_likely_table_region_adaptively(region, thresholds):
                    table_regions.append({
                        "spans": region,
                        "bounds": self._calculate_region_bounds(region),
                        "confidence": self._calculate_region_confidence(region, thresholds),
                        "type": "table"
                    })
            
            # Sort by confidence
            table_regions.sort(key=lambda x: x["confidence"], reverse=True)
            
            return table_regions
            
        except Exception as e:
            self.logger.error(f"Adaptive table region detection failed: {e}")
            return []

    def _group_spans_adaptively(self, spans: List[Dict], thresholds: Dict) -> List[List[Dict]]:
        """Group spans by spatial proximity using adaptive thresholds"""
        try:
            if not spans:
                return []
            
            # Sort spans by position
            spans.sort(key=lambda s: (s["y0"], s["x0"]))
            
            regions = []
            current_region = [spans[0]]
            
            proximity_tolerance = thresholds.get("proximity_tolerance", 50)
            
            for span in spans[1:]:
                if self._is_spatially_close_adaptively(span, current_region, proximity_tolerance):
                    current_region.append(span)
                else:
                    min_spans = thresholds.get("min_spans_per_region", 3)
                    if len(current_region) >= min_spans:
                        regions.append(current_region)
                    current_region = [span]
            
            # Add the last region
            if len(current_region) >= thresholds.get("min_spans_per_region", 3):
                regions.append(current_region)
            
            return regions
            
        except Exception as e:
            self.logger.warning(f"Adaptive span grouping failed: {e}")
            return []

    def _is_spatially_close_adaptively(self, span: Dict, region: List[Dict], tolerance: float) -> bool:
        """Check if a span is spatially close to a region using adaptive tolerance"""
        try:
            if not region:
                return False
            
            # Calculate region bounds
            region_x0 = min(s["x0"] for s in region)
            region_x1 = max(s["x1"] for s in region)
            region_y0 = min(s["y0"] for s in region)
            region_y1 = max(s["y1"] for s in region)
            
            # Check proximity with adaptive tolerance
            x_distance = min(abs(span["x0"] - region_x1), abs(span["x1"] - region_x0))
            y_distance = min(abs(span["y0"] - region_y1), abs(span["y1"] - region_y0))
            
            return x_distance < tolerance and y_distance < tolerance * 0.6
            
        except:
            return False

    def _is_likely_table_region_adaptively(self, region: List[Dict], thresholds: Dict) -> bool:
        """Check if a region is likely to contain a table using adaptive criteria"""
        try:
            if len(region) < thresholds.get("min_region_size", 6):
                return False
            
            # Calculate alignment scores
            x_positions = [s["x0"] for s in region]
            y_positions = [s["y0"] for s in region]
            
            x_alignment_score = self._calculate_alignment_score_adaptively(x_positions, thresholds)
            y_alignment_score = self._calculate_alignment_score_adaptively(y_positions, thresholds)
            
            # Calculate content diversity
            content_diversity = self._calculate_content_diversity(region)
            
            # Calculate spacing regularity
            spacing_score = self._calculate_spacing_score_adaptively(region, thresholds)
            
            # Adaptive scoring
            alignment_score = max(x_alignment_score, y_alignment_score)
            quality_score = (alignment_score + content_diversity + spacing_score) / 3.0
            
            return quality_score >= thresholds.get("quality_threshold", 0.5)
            
        except:
            return False

    def _calculate_alignment_score_adaptively(self, positions: List[float], thresholds: Dict) -> float:
        """Calculate alignment score using adaptive parameters"""
        try:
            if len(positions) < 2:
                return 0.0
            
            # Group positions by proximity
            positions.sort()
            clusters = []
            current_cluster = [positions[0]]
            
            alignment_tolerance = thresholds.get("alignment_tolerance", 25)
            
            for pos in positions[1:]:
                if pos - current_cluster[-1] < alignment_tolerance:
                    current_cluster.append(pos)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [pos]
            clusters.append(current_cluster)
            
            # Score based on number of distinct clusters
            return min(len(clusters) / len(positions), 1.0)
            
        except:
            return 0.0

    def _calculate_content_diversity(self, region: List[Dict]) -> float:
        """Calculate content diversity in a region"""
        try:
            if not region:
                return 0.0
            
            # Analyze text content
            all_text = " ".join(s.get("text", "") for s in region)
            
            # Calculate character diversity
            unique_chars = len(set(all_text.lower()))
            total_chars = len(all_text)
            char_diversity = unique_chars / total_chars if total_chars > 0 else 0
            
            # Calculate word diversity
            words = all_text.split()
            unique_words = len(set(word.lower() for word in words))
            total_words = len(words)
            word_diversity = unique_words / total_words if total_words > 0 else 0
            
            # Calculate length diversity
            word_lengths = [len(word) for word in words]
            length_variance = np.var(word_lengths) if word_lengths else 0
            length_diversity = min(length_variance / 10, 1.0)  # Normalize
            
            return (char_diversity + word_diversity + length_diversity) / 3.0
            
        except:
            return 0.0

    def _calculate_spacing_score_adaptively(self, region: List[Dict], thresholds: Dict) -> float:
        """Calculate spacing regularity score using adaptive parameters"""
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
            
            # Calculate regularity
            if len(gaps) > 1:
                gap_variance = np.var(gaps)
                gap_mean = np.mean(gaps)
                cv = np.sqrt(gap_variance) / gap_mean if gap_mean > 0 else 1.0
                regularity = max(0, 1 - cv)
            else:
                regularity = 0.5
            
            return regularity
            
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

    def _calculate_region_confidence(self, region: List[Dict], thresholds: Dict) -> float:
        """Calculate confidence that a region contains a table"""
        try:
            if not region:
                return 0.0
            
            confidence = 0.0
            
            # Base confidence from region size
            min_size = thresholds.get("min_region_size", 6)
            if len(region) >= min_size * 2:
                confidence += 0.3
            elif len(region) >= min_size:
                confidence += 0.2
            
            # Alignment confidence
            x_positions = [s["x0"] for s in region]
            y_positions = [s["y0"] for s in region]
            x_alignment = self._calculate_alignment_score_adaptively(x_positions, thresholds)
            y_alignment = self._calculate_alignment_score_adaptively(y_positions, thresholds)
            confidence += max(x_alignment, y_alignment) * 0.3
            
            # Content diversity confidence
            content_diversity = self._calculate_content_diversity(region)
            confidence += content_diversity * 0.2
            
            # Spacing confidence
            spacing_score = self._calculate_spacing_score_adaptively(region, thresholds)
            confidence += spacing_score * 0.2
            
            return min(confidence, 1.0)
            
        except:
            return 0.0

    def _extract_table_from_region_adaptively(self, region: Dict, doc_characteristics: Dict) -> Optional[Dict]:
        """Extract table data from a region using adaptive methods"""
        try:
            spans = region.get("spans", [])
            if not spans:
                return None
            
            # Try multiple extraction methods
            extraction_methods = [
                self._extract_by_spatial_clustering,
                self._extract_by_alignment_analysis,
                self._extract_by_content_patterns
            ]
            
            best_table = None
            best_score = 0
            
            for method in extraction_methods:
                try:
                    table_data = method(spans, doc_characteristics)
                    if table_data:
                        score = self._score_table_quality(table_data, doc_characteristics)
                        if score > best_score:
                            best_score = score
                            best_table = table_data
                except Exception as e:
                    self.logger.warning(f"Table extraction method failed: {e}")
                    continue
            
            if best_table:
                best_table["extraction_confidence"] = best_score
                best_table["bounds"] = region.get("bounds", {})
                best_table["region_confidence"] = region.get("confidence", 0.0)
            
            return best_table
            
        except Exception as e:
            self.logger.warning(f"Adaptive table extraction failed: {e}")
            return None

    def _extract_by_spatial_clustering(self, spans: List[Dict], doc_characteristics: Dict) -> Optional[List[List[str]]]:
        """Extract table by spatial clustering"""
        try:
            # Group spans into rows by y-coordinate
            rows = self._cluster_into_rows(spans)
            
            if len(rows) < 2:
                return None
            
            # Group spans into columns by x-coordinate
            columns = self._cluster_into_columns(spans)
            
            if len(columns) < 2:
                return None
            
            # Build table structure
            table_data = self._build_table_structure(rows, columns)
            
            return table_data if len(table_data) > 1 else None
            
        except Exception as e:
            self.logger.warning(f"Spatial clustering extraction failed: {e}")
            return None

    def _extract_by_alignment_analysis(self, spans: List[Dict], doc_characteristics: Dict) -> Optional[List[List[str]]]:
        """Extract table by alignment analysis"""
        try:
            # Find alignment patterns
            x_alignments = self._find_alignment_patterns([s["x0"] for s in spans])
            y_alignments = self._find_alignment_patterns([s["y0"] for s in spans])
            
            if len(x_alignments) < 2 or len(y_alignments) < 2:
                return None
            
            # Build table based on alignments
            table_data = self._build_table_from_alignments(spans, x_alignments, y_alignments)
            
            return table_data if len(table_data) > 1 else None
            
        except Exception as e:
            self.logger.warning(f"Alignment analysis extraction failed: {e}")
            return None

    def _extract_by_content_patterns(self, spans: List[Dict], doc_characteristics: Dict) -> Optional[List[List[str]]]:
        """Extract table by content pattern analysis"""
        try:
            # Analyze content patterns
            content_patterns = self._analyze_content_patterns(spans)
            
            if not content_patterns.get("has_table_pattern", False):
                return None
            
            # Extract based on patterns
            table_data = self._extract_from_content_patterns(spans, content_patterns)
            
            return table_data if len(table_data) > 1 else None
            
        except Exception as e:
            self.logger.warning(f"Content pattern extraction failed: {e}")
            return None

    def _cluster_into_rows(self, spans: List[Dict]) -> List[List[Dict]]:
        """Cluster spans into rows"""
        try:
            if not spans:
                return []
            
            # Sort by y-coordinate
            spans_sorted = sorted(spans, key=lambda s: s["y0"])
            
            rows = []
            current_row = [spans_sorted[0]]
            current_y = spans_sorted[0]["y0"]
            
            # Calculate average height for tolerance
            avg_height = np.mean([s["y1"] - s["y0"] for s in spans])
            tolerance = avg_height * 0.5
            
            for span in spans_sorted[1:]:
                if abs(span["y0"] - current_y) <= tolerance:
                    current_row.append(span)
                else:
                    if len(current_row) >= 2:
                        rows.append(current_row)
                    current_row = [span]
                    current_y = span["y0"]
            
            if len(current_row) >= 2:
                rows.append(current_row)
            
            return rows
            
        except Exception as e:
            self.logger.warning(f"Row clustering failed: {e}")
            return []

    def _cluster_into_columns(self, spans: List[Dict]) -> List[List[Dict]]:
        """Cluster spans into columns"""
        try:
            if not spans:
                return []
            
            # Sort by x-coordinate
            spans_sorted = sorted(spans, key=lambda s: s["x0"])
            
            columns = []
            current_column = [spans_sorted[0]]
            current_x = spans_sorted[0]["x0"]
            
            # Calculate average width for tolerance
            avg_width = np.mean([s["x1"] - s["x0"] for s in spans])
            tolerance = avg_width * 0.5
            
            for span in spans_sorted[1:]:
                if abs(span["x0"] - current_x) <= tolerance:
                    current_column.append(span)
                else:
                    if len(current_column) >= 2:
                        columns.append(current_column)
                    current_column = [span]
                    current_x = span["x0"]
            
            if len(current_column) >= 2:
                columns.append(current_column)
            
            return columns
            
        except Exception as e:
            self.logger.warning(f"Column clustering failed: {e}")
            return []

    def _build_table_structure(self, rows: List[List[Dict]], columns: List[List[Dict]]) -> List[List[str]]:
        """Build table structure from rows and columns"""
        try:
            if not rows or not columns:
                return []
            
            # Create a grid to map spans to cells
            all_spans = []
            for row in rows:
                all_spans.extend(row)
            
            # Sort columns by x-position
            columns_sorted = sorted(columns, key=lambda col: min(s["x0"] for s in col))
            
            # Build table
            table_data = []
            for row in rows:
                row_data = []
                for col in columns_sorted:
                    # Find spans that belong to this cell
                    cell_spans = []
                    for span in row:
                        if any(self._spans_overlap(span, col_span) for col_span in col):
                            cell_spans.append(span)
                    
                    # Combine text from cell spans
                    cell_text = " ".join(span["text"] for span in cell_spans)
                    row_data.append(cell_text)
                
                table_data.append(row_data)
            
            return table_data
            
        except Exception as e:
            self.logger.warning(f"Table structure building failed: {e}")
            return []

    def _spans_overlap(self, span1: Dict, span2: Dict) -> bool:
        """Check if two spans overlap"""
        try:
            return not (span1["x1"] < span2["x0"] or span2["x1"] < span1["x0"] or
                       span1["y1"] < span2["y0"] or span2["y1"] < span1["y0"])
        except:
            return False

    def _find_alignment_patterns(self, positions: List[float]) -> List[float]:
        """Find alignment patterns in positions"""
        try:
            if len(positions) < 2:
                return []
            
            # Group positions by proximity
            positions_sorted = sorted(positions)
            clusters = []
            current_cluster = [positions_sorted[0]]
            
            # Use adaptive tolerance based on position variance
            position_variance = np.var(positions)
            tolerance = np.sqrt(position_variance) * 0.3
            
            for pos in positions_sorted[1:]:
                if pos - current_cluster[-1] <= tolerance:
                    current_cluster.append(pos)
                else:
                    if len(current_cluster) >= 2:
                        clusters.append(np.mean(current_cluster))
                    current_cluster = [pos]
            
            if len(current_cluster) >= 2:
                clusters.append(np.mean(current_cluster))
            
            return clusters
            
        except Exception as e:
            self.logger.warning(f"Alignment pattern finding failed: {e}")
            return []

    def _build_table_from_alignments(self, spans: List[Dict], x_alignments: List[float], y_alignments: List[float]) -> List[List[str]]:
        """Build table from alignment patterns"""
        try:
            if not x_alignments or not y_alignments:
                return []
            
            # Sort alignments
            x_alignments_sorted = sorted(x_alignments)
            y_alignments_sorted = sorted(y_alignments)
            
            # Create table grid
            table_data = []
            for y_align in y_alignments_sorted:
                row_data = []
                for x_align in x_alignments_sorted:
                    # Find spans near this alignment point
                    cell_spans = []
                    for span in spans:
                        if (abs(span["x0"] - x_align) < 20 and 
                            abs(span["y0"] - y_align) < 20):
                            cell_spans.append(span)
                    
                    # Combine text from cell spans
                    cell_text = " ".join(span["text"] for span in cell_spans)
                    row_data.append(cell_text)
                
                if any(cell.strip() for cell in row_data):  # Only add non-empty rows
                    table_data.append(row_data)
            
            return table_data
            
        except Exception as e:
            self.logger.warning(f"Table building from alignments failed: {e}")
            return []

    def _analyze_content_patterns(self, spans: List[Dict]) -> Dict:
        """Analyze content patterns to detect table structure"""
        try:
            if not spans:
                return {"has_table_pattern": False}
            
            # Analyze text patterns
            all_text = " ".join(s.get("text", "") for s in spans)
            
            # Calculate pattern metrics
            char_diversity = len(set(all_text.lower())) / len(all_text) if all_text else 0
            word_count = len(all_text.split())
            avg_word_length = np.mean([len(word) for word in all_text.split()]) if all_text.split() else 0
            
            # Determine if patterns suggest table structure
            has_table_pattern = (
                char_diversity > 0.3 and  # Good character diversity
                word_count > 5 and        # Sufficient content
                avg_word_length > 2       # Not just single characters
            )
            
            return {
                "has_table_pattern": has_table_pattern,
                "char_diversity": char_diversity,
                "word_count": word_count,
                "avg_word_length": avg_word_length
            }
            
        except Exception as e:
            self.logger.warning(f"Content pattern analysis failed: {e}")
            return {"has_table_pattern": False}

    def _extract_from_content_patterns(self, spans: List[Dict], patterns: Dict) -> List[List[str]]:
        """Extract table from content patterns"""
        try:
            if not patterns.get("has_table_pattern", False):
                return []
            
            # Simple extraction based on spatial distribution
            rows = self._cluster_into_rows(spans)
            
            if len(rows) < 2:
                return []
            
            # Build table from rows
            table_data = []
            for row in rows:
                # Sort spans in row by x-coordinate
                row_sorted = sorted(row, key=lambda s: s["x0"])
                row_data = [span["text"] for span in row_sorted]
                table_data.append(row_data)
            
            return table_data
            
        except Exception as e:
            self.logger.warning(f"Content pattern extraction failed: {e}")
            return []

    def _score_table_quality(self, table_data: List[List[str]], doc_characteristics: Dict) -> float:
        """Score table quality adaptively"""
        try:
            if not table_data:
                return 0.0
            
            score = 0.0
            
            # Score based on size
            rows = len(table_data)
            cols = len(table_data[0]) if table_data else 0
            
            if rows >= 3 and cols >= 2:
                score += 0.3
            elif rows >= 2 and cols >= 2:
                score += 0.2
            
            # Score based on content consistency
            consistency_score = self._calculate_table_consistency(table_data)
            score += consistency_score * 0.4
            
            # Score based on content diversity
            diversity_score = self._calculate_table_diversity(table_data)
            score += diversity_score * 0.3
            
            return min(score, 1.0)
            
        except:
            return 0.0

    def _calculate_table_consistency(self, table_data: List[List[str]]) -> float:
        """Calculate table consistency"""
        try:
            if len(table_data) < 2:
                return 0.0
            
            # Check column width consistency
            col_widths = [len(row) for row in table_data]
            if len(set(col_widths)) == 1:
                return 1.0
            else:
                return 0.5
            
        except:
            return 0.0

    def _calculate_table_diversity(self, table_data: List[List[str]]) -> float:
        """Calculate table content diversity"""
        try:
            if not table_data:
                return 0.0
            
            # Calculate content diversity across cells
            all_cells = [cell for row in table_data for cell in row]
            unique_cells = len(set(cell.strip().lower() for cell in all_cells if cell.strip()))
            total_cells = len(all_cells)
            
            return unique_cells / total_cells if total_cells > 0 else 0.0
            
        except:
            return 0.0

    def _score_tables_adaptively(self, tables: List[Dict], doc_characteristics: Dict) -> List[Dict]:
        """Score and rank tables adaptively"""
        try:
            scored_tables = []
            
            for table in tables:
                score = self._calculate_adaptive_table_score(table, doc_characteristics)
                table["adaptive_score"] = score
                scored_tables.append(table)
            
            # Sort by adaptive score
            scored_tables.sort(key=lambda x: x.get("adaptive_score", 0), reverse=True)
            
            return scored_tables
            
        except Exception as e:
            self.logger.warning(f"Adaptive table scoring failed: {e}")
            return tables

    def _calculate_adaptive_table_score(self, table: Dict, doc_characteristics: Dict) -> float:
        """Calculate adaptive table score"""
        try:
            table_data = table.get("data", [])
            if not table_data:
                return 0.0
            
            score = 0.0
            
            # Base score from extraction confidence
            extraction_confidence = table.get("extraction_confidence", 0.0)
            score += extraction_confidence * 0.4
            
            # Score from region confidence
            region_confidence = table.get("region_confidence", 0.0)
            score += region_confidence * 0.3
            
            # Score from table quality
            quality_score = self._score_table_quality(table_data, doc_characteristics)
            score += quality_score * 0.3
            
            return min(score, 1.0)
            
        except:
            return 0.0
