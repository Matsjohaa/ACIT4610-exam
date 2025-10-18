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
NSA_NUM_DETECTORS: int = 1000  # increase detector pool for broader spam coverage
NSA_DETECTOR_SIZE: int = 4
NSA_OVERLAP_THRESHOLD: int = 1  # lower threshold to allow weaker overlaps, boosting recall
NSA_MAX_ATTEMPTS: int = 60_000  # allow more attempts to find non-self detectors

# Future configurable options (placeholders for extensions)
# Require at least this many detectors to fire to call spam (currently unused)
NSA_MIN_ACTIVATIONS: int = 2  # require at least 2 detectors to fire to mitigate precision loss from lower threshold

# Optional spam-guided detector weighting:
# If enabled, detector tokens are sampled with probability proportional to (spam_freq+1)/(ham_freq+1) ** NSA_WEIGHT_EXP
# to emphasize tokens characteristic of spam. This can improve recall on imbalanced datasets.
NSA_USE_SPAM_WEIGHTS: bool = True
NSA_WEIGHT_EXP: float = 1.5  # increase (>1) to amplify differences

# Output / results configuration
# Store misclassification files inside a dedicated 'results/' subdirectory under Problem_4
# to keep the root clean and group experiment artifacts.
RESULTS_DIR = _PROBLEM_ROOT / "results"
# File format choice: TSV keeps consistency with source dataset, human readable, simple to parse.
# (If messages could contain tabs/newlines, consider JSON Lines instead.)
MISCLASS_FP_FILENAME = "false_positives.tsv"  # ham predicted spam
MISCLASS_FN_FILENAME = "false_negatives.tsv"  # spam predicted ham
MISCLASS_INCLUDE_HEADER = True

# Additional result files for correctly classified examples
TRUE_POS_FILENAME = "true_positives.tsv"   # spam predicted spam
TRUE_NEG_FILENAME = "true_negatives.tsv"   # ham predicted ham

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
	"TRUE_POS_FILENAME",
	"TRUE_NEG_FILENAME",
]

