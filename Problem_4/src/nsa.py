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
from typing import List, Set, Sequence


class NegativeSelectionClassifier:
	def __init__(
		self,
		vocab_size: int,
		num_detectors: int = 500,
		detector_size: int = 3,
		overlap_threshold: int = 2,
		max_attempts: int = 20000,
		seed: int = 42,
		min_activations: int = 1,
		weights: Sequence[float] | None = None,
	) -> None:
		self.vocab_size = vocab_size
		self.num_detectors = num_detectors
		self.detector_size = detector_size
		self.overlap_threshold = overlap_threshold
		self.max_attempts = max_attempts
		self.seed = seed
		self.weights = list(weights) if weights is not None else None
		self.min_activations = min_activations
		self.detectors: List[Set[int]] = []

	# ---------------- Internal helpers ---------------- #
	def _random_detector(self) -> Set[int]:
		if not self.weights:
			return set(random.sample(range(self.vocab_size), self.detector_size))
		# Weighted sampling without replacement: simple approach using cumulative weights.
		indices = list(range(self.vocab_size))
		chosen: Set[int] = set()
		weights_local = self.weights
		for _ in range(self.detector_size):
			# Compute cumulative distribution over remaining indices
			remaining = [i for i in indices if i not in chosen]
			w_sum = sum(weights_local[i] for i in remaining)
			# Guard against zero division
			if w_sum == 0:
				# Fallback to uniform
				picked = random.choice(remaining)
				chosen.add(picked)
				continue
			threshold = random.random() * w_sum
			cuml = 0.0
			picked = remaining[-1]
			for i in remaining:
				cuml += weights_local[i]
				if cuml >= threshold:
					picked = i
					break
			chosen.add(picked)
		return chosen

	def _matches(self, sample: Set[int], detector: Set[int]) -> bool:
		return len(sample & detector) >= self.overlap_threshold

	# ---------------- Public API ---------------- #
	def fit(self, X_sets: List[Set[int]], y: List[int]):
		"Generates random candidate detectors (sets of token indices of size detector_size). "
		"Rejects any candidate that “matches” (overlaps >= overlap_threshold) ANY ham training sample. Stops when it has num_detectors or hits max_attempts."
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
		# Stats
		self.attempts_used = attempts
		self.detectors_count = len(self.detectors)
		return self

	def predict(self, X_sets: List[Set[int]]):
		"""Classify each sample.

		Count activations (detectors whose overlap >= overlap_threshold) and
		predict spam if activations >= min_activations; else ham.
		"""
		preds: List[int] = []
		for s in X_sets:
			activations = sum(1 for d in self.detectors if self._matches(s, d))
			preds.append(1 if activations >= self.min_activations else 0)
		return preds

	def predict_with_scores(self, X_sets: List[Set[int]]):
		"""Return tuple (predictions, activation_counts) where activation_counts is the
		number of detectors that matched each sample. This can be used as a score for
		ROC/PR curves (higher means more evidence of spam)."""
		preds: List[int] = []
		acts: List[int] = []
		for s in X_sets:
			activations = sum(1 for d in self.detectors if self._matches(s, d))
			acts.append(activations)
			preds.append(1 if activations >= self.min_activations else 0)
		return preds, acts

	def activation_counts(self, X_sets: List[Set[int]]):
		"""Return activation counts only (helper for coverage curve)."""
		return [sum(1 for d in self.detectors if self._matches(s, d)) for s in X_sets]


__all__ = ["NegativeSelectionClassifier"]

