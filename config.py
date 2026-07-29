r"""
Central configuration for EMB-Finder.

All paths default to living next to this file (BASE_DIR), so the app
works out-of-the-box on any machine/OS as long as the folder structure
(qdrant_db/, design_database.csv, match_memory.csv, dst_images/) is kept
together. Any value can be overridden with an environment variable, e.g.:

    set EMB_QDRANT_PATH=D:\some\other\path      (Windows)
    export EMB_QDRANT_PATH=/some/other/path     (Linux/Mac)

This removes the old hardcoded C:\Users\Asus\... paths so the project can
run on a different laptop, a cloud VM, or inside a Docker container
without editing source code.
"""

import os
import platform
from pathlib import Path

# Root folder = the folder this config.py lives in (i.e. the repo root)
BASE_DIR = Path(__file__).resolve().parent

def _env_path(env_var, default_path):
    """Return env var value if set, else the given default (as a Path)."""
    return Path(os.environ.get(env_var, default_path))

# --- Core data paths (override via env vars if you keep data elsewhere) ---
QDRANT_PATH = str(_env_path("EMB_QDRANT_PATH", BASE_DIR / "qdrant_db"))
DATABASE_CSV = str(_env_path("EMB_DATABASE_CSV", BASE_DIR / "design_database.csv"))
MEMORY_CSV = str(_env_path("EMB_MEMORY_CSV", BASE_DIR / "match_memory.csv"))
DST_IMAGES_DIR = str(_env_path("EMB_DST_IMAGES_DIR", BASE_DIR / "dst_images"))

# --- Tesseract OCR binary ---
# On Windows we default to the standard Tesseract-OCR install path (matches
# the original setup). On Linux/Mac, `tesseract` is normally already on
# PATH after `apt install tesseract-ocr` / `brew install tesseract`, so we
# leave pytesseract to find it automatically unless overridden.
if "EMB_TESSERACT_CMD" in os.environ:
    TESSERACT_CMD = os.environ["EMB_TESSERACT_CMD"]
elif platform.system() == "Windows":
    TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:
    TESSERACT_CMD = None  # let pytesseract use PATH

# --- Qdrant collections & model settings ---
DESIGN_COLLECTION = "embroidery_designs"
MEMORY_COLLECTION = "embroidery_match_memory"
VECTOR_SIZE = 768
HIGH_CONFIDENCE_SCORE = 0.92
MEMORY_SCORE_THRESHOLD = 0.82

# --- Re-ranking (Step 3) ---
# When the same design shows up multiple times in the raw top-N (e.g. its
# front AND sleeve image both land in the results, or it matches across
# multiple rotated/cropped search variants), that's a real corroboration
# signal - not just a tiebreaker. Each extra hit nudges the design's score
# up by REPEAT_MATCH_BONUS, capped so that hit-count alone can never fully
# override a genuinely stronger single match.
REPEAT_MATCH_BONUS = 0.01          # +1% confidence per extra corroborating hit
REPEAT_MATCH_BONUS_CAP = 5         # capped at +5% total (5 extra hits)

# --- View-type classification (Step 2) ---
# Keyword groups used to auto-tag each catalog image with a view type
# (front / sleeve / butta / back / full / other) from its file_name.
# Order matters: more specific categories are checked first.
VIEW_TYPE_KEYWORDS = {
    "butta": ["butta", "boota", "booti", "buti"],
    "sleeve": ["sleeve", "slee", "hand", "hend", "patti", "border", "left", "right"],
    "back": ["back", "bk"],
    "front": ["front", "frount", "neck"],
    "full": ["full"],
}
