BASE_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "CompleteAddressLookup/1.0 (contact@example.com)"

LOG_FILE_PATH = "/logs/extractor.log"


# Dynamic Configuration - These will be calculated adaptively
# Base values for fallback scenarios only
DEFAULT_ROW_TOL_MIN = 2.0
DEFAULT_ROW_TOL_FACTOR = 0.6
DEFAULT_SHIFT_TOL = 12.0
DEFAULT_BOUND_MARGIN = 2.0
DEFAULT_MIN_NONEMPTY_CELLS = 3
DEFAULT_HEADER_FILL_THRESHOLD = 0.6
DEFAULT_DATA_ROW_NUMERIC_THRESHOLD = 1

# Dynamic thresholds will be calculated based on document characteristics
# No hardcoded values will be used in the new adaptive system

 
API_PORT =8000

API_HOST = "0.0.0.0"

# OCR/Table Extraction thresholds (configurable)
# Classification-based short-circuit
CLASS_TEXT_CONF_MIN = 0.5            # Minimum confidence to treat doc as text document
CLASS_TOP_TABLE_CONF_MAX = 0.4       # Max confidence of top table region to still skip tables

# Table quality thresholds
TABLE_MIN_QUALITY = 0.5              # Minimum quality score to accept detected table

# Single-column table heuristics
SINGLE_COL_MIN_ROWS = 8              # Minimum data rows to accept single-column table
SINGLE_COL_MIN_QUALITY = 0.6         # Higher quality bar for single-column tables

# Scanned PDF detection
SCANNED_TEXT_MIN_CHARS = 50          # If native text chars below this, treat as scanned

# Spatial grouping tolerances
SPATIAL_SAME_LINE_TOL_PX = 15        # Pixels to group spans into same line

# Paragraph-as-table rejection heuristics
WORD_TOKEN_MAX_LEN = 4               # Max length to consider a token a short word
SHORT_WORD_RATIO_MIN = 0.75          # If >= this fraction of cells are short words → likely paragraph
TABLE_MIN_NUMERIC_CHAR_RATIO = 0.1   # Require at least this numeric char ratio for table acceptance

# Additional prose-vs-table heuristics
ALPHA_CHAR_RATIO_MIN = 0.9           # If alphabetic chars dominate, likely prose
PARAGRAPH_LIKE_MIN_COLS = 3          # If many columns but all prose-like, reject as table
AVG_TOKEN_LEN_MIN = 3                # Average token length lower bound for prose-like rejection
AVG_TOKEN_LEN_MAX = 9                # Average token length upper bound for prose-like rejection