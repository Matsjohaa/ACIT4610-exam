"""Utility functions for Problem 4 (NSA spam detection).

Contains:
- Reproducibility helpers
- Confusion matrix & metrics
"""

from __future__ import annotations

import random
from typing import Iterable, List, Sequence, Tuple, Dict

import numpy as np


def set_seed(seed: int) -> None:
	"""Set Python & NumPy RNG seeds for reproducibility.

	Seed value is provided explicitly (e.g., from constants.SEED) to centralize configuration.
	"""
	random.seed(seed)
	np.random.seed(seed)


def compute_confusion(y_true: Sequence[int], y_pred: Sequence[int]) -> Tuple[int, int, int, int]:
	"""Return (tp, tn, fp, fn). Assumes positive class == 1 (spam)."""
	tp = sum((yt == 1) and (yp == 1) for yt, yp in zip(y_true, y_pred))
	tn = sum((yt == 0) and (yp == 0) for yt, yp in zip(y_true, y_pred))
	fp = sum((yt == 0) and (yp == 1) for yt, yp in zip(y_true, y_pred))
	fn = sum((yt == 1) and (yp == 0) for yt, yp in zip(y_true, y_pred))
	return tp, tn, fp, fn


def classification_report(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
	"""Compute standard binary classification metrics.

	Returns a dict containing accuracy, precision, recall, f1 and raw counts.

	accuracy = (tp + tn) / total
	precision = tp / (tp + fp)
	recall = tp / (tp + fn)
	f1 = 2 * (precision * recall) / (precision + recall)
	"""
	tp, tn, fp, fn = compute_confusion(y_true, y_pred)
	precision = tp / (tp + fp) if (tp + fp) else 0.0
	recall = tp / (tp + fn) if (tp + fn) else 0.0
	f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
	acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
	return {
		"accuracy": acc,
		"precision": precision,
		"recall": recall,
		"f1": f1,
		"tp": float(tp),
		"tn": float(tn),
		"fp": float(fp),
		"fn": float(fn),
	}


__all__ = [
	"set_seed",
	"compute_confusion",
	"classification_report",
]

