import statistics
import numpy as np
from typing import List, Dict, Optional
from app.Logger.ocr_logger import get_standard_logger
from app.Exceptions.custom_exceptions import (
    handle_general_operations, log_method_entry_exit, ExceptionSeverity
)


class DynamicTextProcessingService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DynamicTextProcessingService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.logger = get_standard_logger("DynamicTextProcessingService")
            self._initialized = True

    @log_method_entry_exit
    @handle_general_operations(severity=ExceptionSeverity.MEDIUM)
    def process_text_dynamically(self, text: str) -> Dict:
        """Process text dynamically without hardcoded patterns"""
        try:
            if not text:
                return {
                    "processed_text": "",
                    "text_type": "empty",
                    "confidence": 0.0,
                    "characteristics": {}
                }
            
            # Analyze text characteristics
            characteristics = self._analyze_text_characteristics(text)
            
            # Determine text type dynamically
            text_type = self._determine_text_type_dynamically(characteristics)
            
            # Apply appropriate processing
            processed_text = self._apply_dynamic_processing(text, characteristics, text_type)
            
            # Calculate confidence
            confidence = self._calculate_processing_confidence(characteristics, text_type)
            
            return {
                "processed_text": processed_text,
                "text_type": text_type,
                "confidence": confidence,
                "characteristics": characteristics
            }
            
        except Exception as e:
            self.logger.error(f"Dynamic text processing failed: {e}")
            return {
                "processed_text": text,
                "text_type": "unknown",
                "confidence": 0.0,
                "characteristics": {}
            }

    def _analyze_text_characteristics(self, text: str) -> Dict:
        """Analyze text characteristics without hardcoded patterns"""
        try:
            if not text:
                return {}
            
            # Character analysis
            char_analysis = self._analyze_characters(text)
            
            # Word analysis
            word_analysis = self._analyze_words(text)
            
            # Structure analysis
            structure_analysis = self._analyze_structure(text)
            
            # Pattern analysis
            pattern_analysis = self._analyze_patterns(text)
            
            return {
                "char_analysis": char_analysis,
                "word_analysis": word_analysis,
                "structure_analysis": structure_analysis,
                "pattern_analysis": pattern_analysis
            }
            
        except Exception as e:
            self.logger.warning(f"Text characteristics analysis failed: {e}")
            return {}

    def _analyze_characters(self, text: str) -> Dict:
        """Analyze character composition dynamically"""
        try:
            if not text:
                return {}
            
            total_chars = len(text)
            
            # Count character types
            char_counts = {
                "letters": 0,
                "digits": 0,
                "spaces": 0,
                "punctuation": 0,
                "special": 0
            }
            
            for char in text:
                if char.isalpha():
                    char_counts["letters"] += 1
                elif char.isdigit():
                    char_counts["digits"] += 1
                elif char.isspace():
                    char_counts["spaces"] += 1
                elif char in ".,!?;:()[]{}":
                    char_counts["punctuation"] += 1
                else:
                    char_counts["special"] += 1
            
            # Calculate ratios
            char_ratios = {}
            for char_type, count in char_counts.items():
                char_ratios[char_type] = count / total_chars if total_chars > 0 else 0
            
            # Calculate diversity
            unique_chars = len(set(text.lower()))
            char_diversity = unique_chars / total_chars if total_chars > 0 else 0
            
            return {
                "char_counts": char_counts,
                "char_ratios": char_ratios,
                "char_diversity": char_diversity,
                "total_chars": total_chars
            }
            
        except Exception as e:
            self.logger.warning(f"Character analysis failed: {e}")
            return {}

    def _analyze_words(self, text: str) -> Dict:
        """Analyze word composition dynamically"""
        try:
            if not text:
                return {}
            
            words = text.split()
            total_words = len(words)
            
            if total_words == 0:
                return {"total_words": 0}
            
            # Word length analysis
            word_lengths = [len(word) for word in words]
            avg_word_length = statistics.mean(word_lengths)
            word_length_std = statistics.stdev(word_lengths) if len(word_lengths) > 1 else 0
            
            # Word diversity
            unique_words = len(set(word.lower() for word in words))
            word_diversity = unique_words / total_words if total_words > 0 else 0
            
            # Word type analysis
            word_types = {
                "short_words": sum(1 for w in words if len(w) <= 3),
                "medium_words": sum(1 for w in words if 4 <= len(w) <= 7),
                "long_words": sum(1 for w in words if len(w) > 7)
            }
            
            # Calculate word type ratios
            word_type_ratios = {}
            for word_type, count in word_types.items():
                word_type_ratios[word_type] = count / total_words if total_words > 0 else 0
            
            return {
                "total_words": total_words,
                "avg_word_length": avg_word_length,
                "word_length_std": word_length_std,
                "word_diversity": word_diversity,
                "word_types": word_types,
                "word_type_ratios": word_type_ratios
            }
            
        except Exception as e:
            self.logger.warning(f"Word analysis failed: {e}")
            return {}

    def _analyze_structure(self, text: str) -> Dict:
        """Analyze text structure dynamically"""
        try:
            if not text:
                return {}
            
            # Line analysis
            lines = text.split('\n')
            line_count = len(lines)
            
            # Calculate line lengths
            line_lengths = [len(line.strip()) for line in lines if line.strip()]
            avg_line_length = statistics.mean(line_lengths) if line_lengths else 0
            line_length_std = statistics.stdev(line_lengths) if len(line_lengths) > 1 else 0
            
            # Paragraph analysis
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            paragraph_count = len(paragraphs)
            
            # Calculate paragraph lengths
            paragraph_lengths = [len(p.split()) for p in paragraphs]
            avg_paragraph_length = statistics.mean(paragraph_lengths) if paragraph_lengths else 0
            
            # Sentence analysis (approximate)
            sentence_indicators = ['.', '!', '?']
            sentence_count = sum(text.count(indicator) for indicator in sentence_indicators)
            
            return {
                "line_count": line_count,
                "avg_line_length": avg_line_length,
                "line_length_std": line_length_std,
                "paragraph_count": paragraph_count,
                "avg_paragraph_length": avg_paragraph_length,
                "sentence_count": sentence_count
            }
            
        except Exception as e:
            self.logger.warning(f"Structure analysis failed: {e}")
            return {}

    def _analyze_patterns(self, text: str) -> Dict:
        """Analyze text patterns dynamically"""
        try:
            if not text:
                return {}
            
            # Repetition analysis
            repetition_score = self._calculate_repetition_score(text)
            
            # Sequence analysis
            sequence_score = self._calculate_sequence_score(text)
            
            # Balance analysis
            balance_score = self._calculate_balance_score(text)
            
            # Complexity analysis
            complexity_score = self._calculate_complexity_score(text)
            
            return {
                "repetition_score": repetition_score,
                "sequence_score": sequence_score,
                "balance_score": balance_score,
                "complexity_score": complexity_score
            }
            
        except Exception as e:
            self.logger.warning(f"Pattern analysis failed: {e}")
            return {}

    def _calculate_repetition_score(self, text: str) -> float:
        """Calculate repetition score dynamically"""
        try:
            if not text:
                return 0.0
            
            words = text.split()
            if len(words) < 2:
                return 0.0
            
            # Count word repetitions
            word_counts = {}
            for word in words:
                word_lower = word.lower()
                word_counts[word_lower] = word_counts.get(word_lower, 0) + 1
            
            # Calculate repetition ratio
            repeated_words = sum(1 for count in word_counts.values() if count > 1)
            total_unique_words = len(word_counts)
            
            repetition_ratio = repeated_words / total_unique_words if total_unique_words > 0 else 0
            
            # Convert to score (higher repetition = lower score for most text types)
            return max(0, 1 - repetition_ratio)
            
        except:
            return 0.5

    def _calculate_sequence_score(self, text: str) -> float:
        """Calculate sequence score dynamically"""
        try:
            if not text:
                return 0.0
            
            # Analyze character sequences
            char_sequences = []
            for i in range(len(text) - 1):
                char_sequences.append(text[i:i+2])
            
            if not char_sequences:
                return 0.0
            
            # Calculate sequence diversity
            unique_sequences = len(set(char_sequences))
            total_sequences = len(char_sequences)
            
            sequence_diversity = unique_sequences / total_sequences if total_sequences > 0 else 0
            
            return sequence_diversity
            
        except:
            return 0.5

    def _calculate_balance_score(self, text: str) -> float:
        """Calculate balance score dynamically"""
        try:
            if not text:
                return 0.0
            
            # Analyze character balance
            char_analysis = self._analyze_characters(text)
            char_ratios = char_analysis.get("char_ratios", {})
            
            # Calculate balance based on character type distribution
            letter_ratio = char_ratios.get("letters", 0)
            digit_ratio = char_ratios.get("digits", 0)
            space_ratio = char_ratios.get("spaces", 0)
            punct_ratio = char_ratios.get("punctuation", 0)
            
            # Ideal balance varies by text type, but generally:
            # - Letters should dominate
            # - Some spaces for readability
            # - Moderate punctuation
            # - Few digits unless numeric content
            
            balance_score = 0.0
            
            # Letter dominance (good for most text)
            if 0.5 <= letter_ratio <= 0.8:
                balance_score += 0.4
            elif 0.3 <= letter_ratio <= 0.9:
                balance_score += 0.2
            
            # Space balance (good for readability)
            if 0.1 <= space_ratio <= 0.3:
                balance_score += 0.3
            elif 0.05 <= space_ratio <= 0.4:
                balance_score += 0.15
            
            # Punctuation balance
            if 0.01 <= punct_ratio <= 0.1:
                balance_score += 0.2
            elif 0.005 <= punct_ratio <= 0.15:
                balance_score += 0.1
            
            # Digit balance (context-dependent)
            if digit_ratio <= 0.2:  # Low digits for text
                balance_score += 0.1
            
            return min(balance_score, 1.0)
            
        except:
            return 0.5

    def _calculate_complexity_score(self, text: str) -> float:
        """Calculate complexity score dynamically"""
        try:
            if not text:
                return 0.0
            
            word_analysis = self._analyze_words(text)
            char_analysis = self._analyze_characters(text)
            
            complexity_score = 0.0
            
            # Word complexity
            avg_word_length = word_analysis.get("avg_word_length", 0)
            if avg_word_length > 5:
                complexity_score += 0.3
            elif avg_word_length > 3:
                complexity_score += 0.2
            
            # Vocabulary complexity
            word_diversity = word_analysis.get("word_diversity", 0)
            complexity_score += word_diversity * 0.3
            
            # Character complexity
            char_diversity = char_analysis.get("char_diversity", 0)
            complexity_score += char_diversity * 0.2
            
            # Structure complexity
            structure_analysis = self._analyze_structure(text)
            paragraph_count = structure_analysis.get("paragraph_count", 0)
            if paragraph_count > 1:
                complexity_score += 0.2
            
            return min(complexity_score, 1.0)
            
        except:
            return 0.5

    def _determine_text_type_dynamically(self, characteristics: Dict) -> str:
        """Determine text type based on characteristics"""
        try:
            char_analysis = characteristics.get("char_analysis", {})
            word_analysis = characteristics.get("word_analysis", {})
            structure_analysis = characteristics.get("structure_analysis", {})
            pattern_analysis = characteristics.get("pattern_analysis", {})
            
            char_ratios = char_analysis.get("char_ratios", {})
            word_count = word_analysis.get("total_words", 0)
            avg_word_length = word_analysis.get("avg_word_length", 0)
            complexity_score = pattern_analysis.get("complexity_score", 0)
            
            # Dynamic classification based on characteristics
            letter_ratio = char_ratios.get("letters", 0)
            digit_ratio = char_ratios.get("digits", 0)
            space_ratio = char_ratios.get("spaces", 0)
            
            # Classification logic
            if word_count == 0:
                return "empty"
            elif word_count == 1:
                return "single_word"
            elif digit_ratio > 0.5:
                return "numeric_heavy"
            elif letter_ratio > 0.8 and space_ratio > 0.1:
                return "text_heavy"
            elif avg_word_length > 8 and complexity_score > 0.7:
                return "complex_text"
            elif avg_word_length < 3 and word_count > 10:
                return "simple_text"
            elif space_ratio < 0.05:
                return "dense_text"
            else:
                return "balanced_text"
                
        except Exception as e:
            self.logger.warning(f"Text type determination failed: {e}")
            return "unknown"

    def _apply_dynamic_processing(self, text: str, characteristics: Dict, text_type: str) -> str:
        """Apply appropriate processing based on text type and characteristics"""
        try:
            if not text:
                return ""
            
            processed_text = text
            
            # Apply processing based on text type
            if text_type == "numeric_heavy":
                processed_text = self._process_numeric_text(processed_text, characteristics)
            elif text_type == "text_heavy":
                processed_text = self._process_text_heavy(processed_text, characteristics)
            elif text_type == "dense_text":
                processed_text = self._process_dense_text(processed_text, characteristics)
            elif text_type == "simple_text":
                processed_text = self._process_simple_text(processed_text, characteristics)
            else:
                processed_text = self._process_general_text(processed_text, characteristics)
            
            # Apply common processing
            processed_text = self._apply_common_processing(processed_text)
            
            return processed_text
            
        except Exception as e:
            self.logger.warning(f"Dynamic processing failed: {e}")
            return text

    def _process_numeric_text(self, text: str, characteristics: Dict) -> str:
        """Process numeric-heavy text"""
        try:
            # For numeric text, focus on number formatting
            processed = text
            
            # Normalize number spacing
            processed = self._normalize_number_spacing(processed)
            
            # Clean up numeric separators
            processed = self._clean_numeric_separators(processed)
            
            return processed
            
        except:
            return text

    def _process_text_heavy(self, text: str, characteristics: Dict) -> str:
        """Process text-heavy content"""
        try:
            # For text-heavy content, focus on readability
            processed = text
            
            # Normalize whitespace
            processed = self._normalize_whitespace(processed)
            
            # Fix common text issues
            processed = self._fix_common_text_issues(processed)
            
            return processed
            
        except:
            return text

    def _process_dense_text(self, text: str, characteristics: Dict) -> str:
        """Process dense text (low spacing)"""
        try:
            # For dense text, add appropriate spacing
            processed = text
            
            # Add spacing around punctuation
            processed = self._add_punctuation_spacing(processed)
            
            # Normalize spacing
            processed = self._normalize_whitespace(processed)
            
            return processed
            
        except:
            return text

    def _process_simple_text(self, text: str, characteristics: Dict) -> str:
        """Process simple text (short words)"""
        try:
            # For simple text, basic cleaning
            processed = text
            
            # Basic normalization
            processed = self._normalize_whitespace(processed)
            
            return processed
            
        except:
            return text

    def _process_general_text(self, text: str, characteristics: Dict) -> str:
        """Process general text"""
        try:
            # General processing for balanced text
            processed = text
            
            # Normalize whitespace
            processed = self._normalize_whitespace(processed)
            
            # Basic cleaning
            processed = self._basic_text_cleaning(processed)
            
            return processed
            
        except:
            return text

    def _normalize_number_spacing(self, text: str) -> str:
        """Normalize spacing around numbers without regex"""
        try:
            # Add spaces around numbers if missing using string operations
            processed = text
            
            # Find positions where numbers and letters are adjacent
            i = 0
            while i < len(processed) - 1:
                char1 = processed[i]
                char2 = processed[i + 1]
                
                # Check if we need to add space between number and letter
                if char1.isdigit() and char2.isalpha():
                    processed = processed[:i+1] + ' ' + processed[i+1:]
                    i += 2  # Skip the added space
                elif char1.isalpha() and char2.isdigit():
                    processed = processed[:i+1] + ' ' + processed[i+1:]
                    i += 2  # Skip the added space
                else:
                    i += 1
            
            return processed
        except:
            return text

    def _clean_numeric_separators(self, text: str) -> str:
        """Clean up numeric separators"""
        try:
            # Normalize decimal separators
            processed = text.replace(',', '.')
            return processed
        except:
            return text

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace without regex"""
        try:
            # Replace multiple spaces with single space
            words = text.split()
            return ' '.join(words)
        except:
            return text

    def _fix_common_text_issues(self, text: str) -> str:
        """Fix common text issues without hardcoded patterns"""
        try:
            processed = text
            
            # Fix common OCR errors dynamically
            processed = self._fix_ocr_errors_dynamically(processed)
            
            return processed
            
        except:
            return text

    def _fix_ocr_errors_dynamically(self, text: str) -> str:
        """Fix OCR errors dynamically based on context"""
        try:
            # This would analyze the text and fix common OCR errors
            # without using hardcoded patterns
            processed = text
            
            # Example: Fix common character substitutions
            # This could be enhanced with machine learning
            char_fixes = {
                '0': 'O',  # In word contexts
                '1': 'I',  # In word contexts
                '5': 'S',  # In word contexts
            }
            
            # Apply fixes only in appropriate contexts
            words = processed.split()
            for i, word in enumerate(words):
                if len(word) > 2:  # Only for longer words
                    for wrong_char, correct_char in char_fixes.items():
                        if wrong_char in word and self._is_likely_word_context(word):
                            word = word.replace(wrong_char, correct_char)
                    words[i] = word
            
            return ' '.join(words)
            
        except:
            return text

    def _is_likely_word_context(self, word: str) -> bool:
        """Determine if a word is likely in a word context (not numeric)"""
        try:
            # Simple heuristic: if word has more letters than digits, it's likely a word
            letter_count = sum(1 for c in word if c.isalpha())
            digit_count = sum(1 for c in word if c.isdigit())
            return letter_count > digit_count
        except:
            return False

    def _add_punctuation_spacing(self, text: str) -> str:
        """Add spacing around punctuation"""
        try:
            processed = text
            
            # Add spaces around punctuation
            punctuation = '.,!?;:'
            for punct in punctuation:
                processed = processed.replace(punct, f' {punct} ')
            
            # Clean up multiple spaces
            processed = self._normalize_whitespace(processed)
            
            return processed
            
        except:
            return text

    def _basic_text_cleaning(self, text: str) -> str:
        """Basic text cleaning"""
        try:
            processed = text
            
            # Normalize whitespace
            processed = self._normalize_whitespace(processed)
            
            # Basic punctuation normalization
            processed = processed.replace('"', '"').replace('"', '"')
            processed = processed.replace(''', "'").replace(''', "'")
            
            return processed
            
        except:
            return text

    def _apply_common_processing(self, text: str) -> str:
        """Apply common processing to all text"""
        try:
            processed = text
            
            # Final normalization
            processed = self._normalize_whitespace(processed)
            
            # Remove leading/trailing whitespace
            processed = processed.strip()
            
            return processed
            
        except:
            return text

    def _calculate_processing_confidence(self, characteristics: Dict, text_type: str) -> float:
        """Calculate confidence in the processing"""
        try:
            confidence = 0.0
            
            # Base confidence from text type determination
            if text_type != "unknown":
                confidence += 0.3
            
            # Confidence from characteristics completeness
            if characteristics.get("char_analysis") and characteristics.get("word_analysis"):
                confidence += 0.3
            
            # Confidence from pattern analysis
            pattern_analysis = characteristics.get("pattern_analysis", {})
            if pattern_analysis:
                complexity_score = pattern_analysis.get("complexity_score", 0)
                confidence += complexity_score * 0.2
            
            # Confidence from structure analysis
            structure_analysis = characteristics.get("structure_analysis", {})
            if structure_analysis.get("total_words", 0) > 0:
                confidence += 0.2
            
            return min(confidence, 1.0)
            
        except:
            return 0.5
