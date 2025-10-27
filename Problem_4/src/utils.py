"""Utility functions for Problem 4 (NSA spam detection).

Contains:
- Reproducibility helpers
- Confusion matrix & metrics
"""

from __future__ import annotations

import random
from typing import Iterable, List, Sequence, Tuple, Dict, Iterable as _Iterable

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


from typing import Iterable as _It, Any

def coverage_curve(detectors: _It[Any], X_sets: Sequence[Sequence[int]], y_true: Sequence[int], checkpoints: Sequence[int], overlap_threshold: int, min_activations: int) -> List[Dict[str, float]]:
	"""Compute recall/precision/F1 at different numbers of detectors (incremental prefix usage).

	Assumes detectors is a list of sets (or sequences convertible to sets)."""
	results = []
	# Pre-convert detectors to list of sets for speed
	_det = [set(d) for d in detectors]
	for k in checkpoints:
		k = min(k, len(_det))
		active = _det[:k]
		preds = []
		for s in X_sets:
			s_set = set(s)
			activations = 0
			for d in active:
				if len(s_set & d) >= overlap_threshold:
					activations += 1
					# short-circuit if only need threshold
					if activations >= min_activations:
						break
			preds.append(1 if activations >= min_activations else 0)
		metrics = classification_report(y_true, preds)
		metrics["k_detectors"] = float(k)
		results.append(metrics)
	return results


def roc_pr_points(y_true: Sequence[int], scores: Sequence[float]):
	"""Return FPR, TPR (ROC) and precision, recall (PR) point lists (step-wise).

	Uses a pure-Python accumulation (no sklearn dependency here) so notebook can plot raw points.
	"""
	# Sort by descending score
	paired = list(zip(scores, y_true))
	paired.sort(key=lambda x: -x[0])
	P = sum(y_true)
	N = len(y_true) - P
	if P == 0 or N == 0:
		return {"fpr": [], "tpr": [], "precision": [], "recall": []}
	TP = 0
	FP = 0
	prev_score = None
	roc_fpr = []
	roc_tpr = []
	pr_precision = []
	pr_recall = []
	for score, label in paired:
		if label == 1:
			TP += 1
		else:
			FP += 1
		# ROC points per step
		roc_tpr.append(TP / P)
		roc_fpr.append(FP / N)
		# PR points
		prec = TP / (TP + FP) if (TP + FP) else 1.0
		rec = TP / P
		pr_precision.append(prec)
		pr_recall.append(rec)
	return {"fpr": roc_fpr, "tpr": roc_tpr, "precision": pr_precision, "recall": pr_recall}

__all__.extend(["coverage_curve", "roc_pr_points"])

# ---------------- Detector diversity helpers ---------------- #
from math import comb
def detector_diversity_stats(detectors, sample_limit: int | None = 400):
	"""Compute diversity metrics for a list of detector sets.

	Returns dict with:
	- avg_pairwise_intersection
	- zero_overlap_fraction
	- mean_jaccard
	- std_jaccard
	Sampling: if number of detectors > sample_limit, randomly sample sample_limit detectors
	to keep O(n^2) manageable.
	"""
	import random, statistics
	D = list(detectors)
	if not D:
		return {
			"avg_pairwise_intersection": 0.0,
			"zero_overlap_fraction": 0.0,
			"mean_jaccard": 0.0,
			"std_jaccard": 0.0,
		}
	if sample_limit and len(D) > sample_limit:
		D = random.sample(D, sample_limit)
	# Ensure sets
	D_sets = [set(d) for d in D]
	n = len(D_sets)
	if n < 2:
		return {
			"avg_pairwise_intersection": 0.0,
			"zero_overlap_fraction": 0.0,
			"mean_jaccard": 0.0,
			"std_jaccard": 0.0,
		}
	intersections = []
	zero_overlap = 0
	jaccards = []
	# Brute force pairs
	for i in range(n):
		for j in range(i + 1, n):
			a = D_sets[i]
			b = D_sets[j]
			inter = len(a & b)
			union = len(a | b)
			intersections.append(inter)
			if inter == 0:
				zero_overlap += 1
			jac = inter / union if union else 0.0
			jaccards.append(jac)
	pair_count = comb(n, 2)
	avg_inter = sum(intersections) / pair_count if pair_count else 0.0
	zero_frac = zero_overlap / pair_count if pair_count else 0.0
	mean_j = sum(jaccards) / pair_count if pair_count else 0.0
	std_j = statistics.pstdev(jaccards) if len(jaccards) > 1 else 0.0
	return {
		"avg_pairwise_intersection": float(avg_inter),
		"zero_overlap_fraction": float(zero_frac),
		"mean_jaccard": float(mean_j),
		"std_jaccard": float(std_j),
	}

__all__.append("detector_diversity_stats")

