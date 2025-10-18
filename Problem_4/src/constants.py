"""Global configuration constants for Problem 4 NSA spam detector.

Adjust values here to tune experiments without editing source logic.
"""

from __future__ import annotations

from pathlib import Path


# SEED = None or a number
SEED: int | None = None

# Data (resolve relative to the project structure so it works no matter the CWD)
# constants.py lives in: <repo>/Problem_4/src/constants.py
# Project root for Problem_4 is therefore parent of this file's directory.
_THIS_DIR = Path(__file__).resolve().parent
_PROBLEM_ROOT = _THIS_DIR.parent  # .../Problem_4
DATA_PATH = _PROBLEM_ROOT / "data" / "sms_spam.tsv"
TEST_RATIO: float = 0.2

# Vocabulary / text processing
VOCAB_MIN_FREQ: int = 2
VOCAB_MAX_SIZE: int = 4000

# NSA (Negative Selection Algorithm) hyperparameters
NSA_NUM_DETECTORS: int = 600
NSA_DETECTOR_SIZE: int = 4
NSA_OVERLAP_THRESHOLD: int = 2
NSA_MAX_ATTEMPTS: int = 30_000

# Future configurable options (placeholders for extensions)
# Require at least this many detectors to fire to call spam (currently unused)
NSA_MIN_ACTIVATIONS: int = 1

# Output / results configuration
# Store misclassification files inside a dedicated 'results/' subdirectory under Problem_4
# to keep the root clean and group experiment artifacts.
RESULTS_DIR = _PROBLEM_ROOT / "results"
# File format choice: TSV keeps consistency with source dataset, human readable, simple to parse.
# (If messages could contain tabs/newlines, consider JSON Lines instead.)
MISCLASS_FP_FILENAME = "false_positives.tsv"  # ham predicted spam
MISCLASS_FN_FILENAME = "false_negatives.tsv"  # spam predicted ham
MISCLASS_INCLUDE_HEADER = True

__all__ = [
	"SEED",
	"DATA_PATH",
	"TEST_RATIO",
	"VOCAB_MIN_FREQ",
	"VOCAB_MAX_SIZE",
	"NSA_NUM_DETECTORS",
	"NSA_DETECTOR_SIZE",
	"NSA_OVERLAP_THRESHOLD",
	"NSA_MAX_ATTEMPTS",
	"NSA_MIN_ACTIVATIONS",
	"RESULTS_DIR",
	"MISCLASS_FP_FILENAME",
	"MISCLASS_FN_FILENAME",
	"MISCLASS_INCLUDE_HEADER",
]

