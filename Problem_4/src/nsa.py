"""Negative Selection Algorithm classifier for spam detection (Problem 4).

This is a simple token-set based variant:
Self space: ham messages (class 0) represented as sets of token indices.
Detector generation:
  Randomly sample fixed-size subsets of vocabulary indices.
  Reject a candidate if it *matches* any self sample (overlap >= overlap_threshold).
Classification:
  A message is predicted spam if ANY detector matches it.
"""

from __future__ import annotations

import random
from typing import List, Set


class NegativeSelectionClassifier:
	def __init__(
		self,
		vocab_size: int,
		num_detectors: int = 500,
		detector_size: int = 3,
		overlap_threshold: int = 2,
		max_attempts: int = 20000,
		seed: int = 42,
	) -> None:
		self.vocab_size = vocab_size
		self.num_detectors = num_detectors
		self.detector_size = detector_size
		self.overlap_threshold = overlap_threshold
		self.max_attempts = max_attempts
		self.seed = seed
		self.detectors: List[Set[int]] = []

	# ---------------- Internal helpers ---------------- #
	def _random_detector(self) -> Set[int]:
		return set(random.sample(range(self.vocab_size), self.detector_size))

	def _matches(self, sample: Set[int], detector: Set[int]) -> bool:
		return len(sample & detector) >= self.overlap_threshold

	# ---------------- Public API ---------------- #
	def fit(self, X_sets: List[Set[int]], y: List[int]):
		random.seed(self.seed)
		self.detectors = []
		self_samples = [s for s, label in zip(X_sets, y) if label == 0]
		attempts = 0
		while len(self.detectors) < self.num_detectors and attempts < self.max_attempts:
			cand = self._random_detector()
			if any(self._matches(s, cand) for s in self_samples):
				attempts += 1
				continue
			self.detectors.append(cand)
			attempts += 1
		return self

	def predict(self, X_sets: List[Set[int]]):
		preds: List[int] = []
		for s in X_sets:
			is_spam = any(self._matches(s, d) for d in self.detectors)
			preds.append(1 if is_spam else 0)
		return preds


__all__ = ["NegativeSelectionClassifier"]

