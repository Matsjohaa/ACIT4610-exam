"""Negative Selection Algorithm classifier for spam detection (Problem 4).

Supports both binary Hamming distance and vocabulary-based r-contiguous matching.
"""

from __future__ import annotations

import random
import numpy as np
from typing import List, Sequence, Union, Set
from collections import Counter


class NegativeSelectionClassifier:
    def __init__(
        self,
        feature_length: int = 128,
        num_detectors: int = 500,
        detector_size: int = 3,
        hamming_threshold: int = 5,
        r_contiguous: int = 2,
        matching_rule: str = "hamming",  # "hamming" or "r_contiguous"
        representation: str = "binary",  # "binary" or "vocabulary" 
        vocab_size: int = 5000,
        min_word_freq: int = 2,
        max_attempts: int = 10000,
        seed: int = 42,
        min_activations: int = 1,
        weights: Sequence[float] | None = None,
    ) -> None:
        self.feature_length = feature_length
        self.num_detectors = num_detectors
        self.detector_size = detector_size
        self.hamming_threshold = hamming_threshold
        self.r_contiguous = r_contiguous
        self.matching_rule = matching_rule
        self.representation = representation
        self.vocab_size = vocab_size
        self.min_word_freq = min_word_freq
        self.max_attempts = max_attempts
        self.seed = seed
        self.min_activations = min_activations
        self.detectors: List[Union[np.ndarray, tuple]] = []
        
        # For vocabulary-based representation
        self.vocabulary = []
        self.word_to_idx = {}
        
        # For backward compatibility
        self.overlap_threshold = hamming_threshold

    def _build_vocabulary(self, texts: Sequence[str]) -> None:
        """Build vocabulary from texts."""
        word_counts = Counter()
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)
        
        vocab_words = [
            word for word, count in word_counts.items() 
            if count >= self.min_word_freq
        ]
        vocab_words.sort(key=lambda w: (-word_counts[w], w))
        self.vocabulary = vocab_words[:self.vocab_size]
        self.word_to_idx = {word: i for i, word in enumerate(self.vocabulary)}

    def _text_to_tokens(self, text: str) -> List[int]:
        """Convert text to token indices."""
        words = text.lower().split()
        return [self.word_to_idx[word] for word in words if word in self.word_to_idx]

    def _get_patterns_from_tokens(self, tokens: List[int]) -> Set[tuple]:
        """Extract detector-sized patterns from tokens."""
        patterns = set()
        for i in range(len(tokens) - self.detector_size + 1):
            pattern = tuple(tokens[i:i + self.detector_size])
            patterns.add(pattern)
        return patterns

    def _r_contiguous_match(self, detector: tuple, target: tuple) -> bool:
        """Check r-contiguous match."""
        if len(detector) != len(target):
            return False
        
        consecutive_matches = 0
        max_consecutive = 0
        
        for i in range(len(detector)):
            if detector[i] == target[i]:
                consecutive_matches += 1
                max_consecutive = max(max_consecutive, consecutive_matches)
            else:
                consecutive_matches = 0
        
        return max_consecutive >= self.r_contiguous

    def _hamming_match_vocab(self, detector: tuple, target: tuple) -> bool:
        """Check Hamming match for vocabulary patterns."""
        if len(detector) != len(target):
            return False
        
        mismatches = sum(1 for i in range(len(detector)) if detector[i] != target[i])
        return mismatches <= self.hamming_threshold

    def _matches_pattern(self, detector: Union[np.ndarray, tuple], target: Union[np.ndarray, tuple]) -> bool:
        """Check if detector matches target using selected rule."""
        if self.representation == "binary":
            # Binary Hamming distance
            return np.sum(detector != target) <= self.hamming_threshold
        else:
            # Vocabulary-based matching
            if self.matching_rule == "r_contiguous":
                return self._r_contiguous_match(detector, target)
            elif self.matching_rule == "hamming":
                return self._hamming_match_vocab(detector, target)
        return False

    def _random_detector_binary(self) -> np.ndarray:
        """Generate random binary detector."""
        return np.random.randint(0, 2, size=self.feature_length, dtype=np.uint8)

    def _random_detector_vocab(self) -> tuple:
        """Generate random vocabulary detector."""
        return tuple(random.randint(0, len(self.vocabulary) - 1) for _ in range(self.detector_size))

    def fit(self, X: Union[List[np.ndarray], List[str]], y: List[int]):
        """Generate detectors and train on features."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.detectors = []
        
        if self.representation == "binary":
            # Binary representation
            self_samples = np.array([x for x, label in zip(X, y) if label == 0])
            if len(self_samples) == 0:
                print("Warning: No ham samples found for training")
                return self
                
            attempts = 0
            print(f"Generating {self.num_detectors} detectors from {len(self_samples)} ham samples...")
            
            while len(self.detectors) < self.num_detectors and attempts < self.max_attempts:
                # Generate candidate detector
                cand = self._random_detector_binary()
                
                # Vectorized distance computation
                distances = np.sum(self_samples != cand, axis=1)
                
                # Check if ANY distance is <= threshold (reject if match found)
                if np.any(distances <= self.hamming_threshold):
                    attempts += 1
                    continue
                
                # Valid detector found
                self.detectors.append(cand)
                attempts += 1
                
                # Progress indicators for slow generations
                if len(self.detectors) % 500 == 0:
                    print(f"  Generated {len(self.detectors)}/{self.num_detectors} detectors...")
        
        else:
            # Vocabulary representation
            ham_texts = [text for text, label in zip(X, y) if label == 0]
            print(f"Training on {len(ham_texts)} ham samples")
            
            # Build vocabulary
            self._build_vocabulary(ham_texts)
            if len(self.vocabulary) == 0:
                print("Warning: No vocabulary built")
                return self
            
            # Convert ham texts to pattern sets
            ham_pattern_sets = []
            for text in ham_texts:
                tokens = self._text_to_tokens(text)
                patterns = self._get_patterns_from_tokens(tokens)
                ham_pattern_sets.append(patterns)
            
            # Generate detectors
            print(f"Generating {self.num_detectors} detectors...")
            attempts = 0
            
            while len(self.detectors) < self.num_detectors and attempts < self.max_attempts:
                # Generate candidate detector
                candidate = self._random_detector_vocab()
                
                # Check if it matches any ham pattern
                matches_ham = False
                for pattern_set in ham_pattern_sets:
                    for pattern in pattern_set:
                        if self._matches_pattern(candidate, pattern):
                            matches_ham = True
                            break
                    if matches_ham:
                        break
                
                if not matches_ham:
                    self.detectors.append(candidate)
                
                attempts += 1
                
                if len(self.detectors) % 100 == 0:
                    print(f"  Generated {len(self.detectors)}/{self.num_detectors} detectors...")
        
        print(f"Detector generation complete: {len(self.detectors)} detectors generated")
        return self

        return self

    def predict(self, X_binary: List[np.ndarray]):
        """Classify samples using detector activations."""
        if not self.detectors:
            return [0] * len(X_binary)
            
        # Convert to numpy arrays for vectorization
        X_array = np.array(X_binary)
        detector_array = np.array(self.detectors)
        
        preds = []
        for sample in X_array:
            # Vectorized activation counting
            distances = np.sum(detector_array != sample, axis=1)
            activations = np.sum(distances <= self.hamming_threshold)
            preds.append(1 if activations >= self.min_activations else 0)
        return preds

    def predict_with_scores(self, X_binary: List[np.ndarray]):
        """Return predictions and activation counts."""
        if not self.detectors:
            return [0] * len(X_binary), [0] * len(X_binary)
            
        # Convert to numpy arrays for vectorization
        X_array = np.array(X_binary)
        detector_array = np.array(self.detectors)
        
        preds, acts = [], []
        for sample in X_array:
            # Vectorized activation counting
            distances = np.sum(detector_array != sample, axis=1)
            activations = np.sum(distances <= self.hamming_threshold)
            acts.append(activations)
            preds.append(1 if activations >= self.min_activations else 0)
        return preds, acts

    def activation_counts(self, X_binary: List[np.ndarray]):
        """Return activation counts for coverage curves."""
        if not self.detectors:
            return [0] * len(X_binary)
            
        # Convert to numpy arrays for vectorization
        X_array = np.array(X_binary)
        detector_array = np.array(self.detectors)
        
        counts = []
        for sample in X_array:
            distances = np.sum(detector_array != sample, axis=1)
            activations = np.sum(distances <= self.hamming_threshold)
            counts.append(activations)
        return counts


__all__ = ["NegativeSelectionClassifier"]
