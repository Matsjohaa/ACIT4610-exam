"""Global configuration constants for Problem 4 NSA spam detector.

Only constants used by comprehensive_grid_search.py and supporting scripts.
"""

from __future__ import annotations

from pathlib import Path


# Project root for NSA is therefore parent of this file's directory.
_THIS_DIR = Path(__file__).resolve().parent
_NSA_ROOT = _THIS_DIR.parent  # .../NSA

# Dataset selection - SMS Spam Collection
DATASET: str = "sms_spam"

# Dataset path
DATA_PATH = _NSA_ROOT / "data" / "sms_spam.tsv"

# Train/test split ratio, rest will go to training
TEST_RATIO: float = 0.2 
SEED: int = 42  # Fixed seed for reproducibility

# Bit-based r-contiguous matching configuration
BIT_LENGTH: int = 256  # Length of bit vector for character n-gram hashing
CHAR_NGRAM_SIZE: int = 4  # Default size of character n-grams (overridden in grid search)
R_CONTIGUOUS: int = 8  # Default r-contiguous value (overridden in grid search)

# Output / results configuration
RESULTS_DIR = _NSA_ROOT / "results"

__all__ = [
	"DATA_PATH",
	"TEST_RATIO",
	"SEED",
	"BIT_LENGTH",
	"CHAR_NGRAM_SIZE",
	"R_CONTIGUOUS",
	"RESULTS_DIR",
]

