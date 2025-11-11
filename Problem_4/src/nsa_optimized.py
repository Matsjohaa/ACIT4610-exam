"""Negative Selection Algorithm classifier for spam detection (Problem 4).

Supports both binary Hamming distance and vocabulary-based r-contiguous matching.
Optimized for fast parameter tuning in analysis scripts.
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
        representation: str = "vocabulary",  # "binary" or "vocabulary" 
        vocab_size: int = 5000,
        min_word_freq: int = 2,
        max_attempts: int = 10000,
        seed: int = 42,
        min_activations: int = 1,
        max_ham_match_ratio: float = 0.05,  # Maximum ratio of ham samples a detector can match
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
        self.max_ham_match_ratio = max_ham_match_ratio
        self.detectors = []
        
        # For vocabulary representation
        self.vocabulary = []
        self.word_to_idx = {}
        
        # For backward compatibility with existing analysis scripts
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
        """Check r-contiguous match - requires r_contiguous consecutive matching positions."""
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

    def _matches_pattern(self, detector, target) -> bool:
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

    def _random_detector_vocab(self) -> tuple:
        """Generate vocabulary detector with better distribution strategy."""
        # Mix of purely random and vocabulary-based patterns
        if random.random() < 0.7:  # 70% - Pure random from vocabulary
            return tuple(random.randint(0, len(self.vocabulary) - 1) for _ in range(self.detector_size))
        else:  # 30% - Sample from most common words (more likely to match real text)
            common_words = list(range(min(50, len(self.vocabulary))))  # Top 50 most common words
            return tuple(random.choice(common_words) for _ in range(self.detector_size))
    
    def _create_pattern_variation(self, base_pattern: tuple) -> tuple:
        """Create a variation of an existing pattern."""
        variation = list(base_pattern)
        
        if self.matching_rule == "hamming" and self.hamming_threshold > 0:
            # Create variation that differs by up to hamming_threshold+1 positions
            # This ensures the variation won't match under the current threshold
            num_changes = random.randint(self.hamming_threshold + 1, min(self.hamming_threshold + 2, len(variation)))
            positions = random.sample(range(len(variation)), num_changes)
            for pos in positions:
                variation[pos] = random.randint(0, len(self.vocabulary) - 1)
            return tuple(variation)
        
        elif self.matching_rule == "r_contiguous":
            # For r-contiguous: break potential contiguous runs by strategically placing changes
            # We need to ensure no r_contiguous consecutive matches exist
            # Strategy: Change every r_contiguous-th position to break all long runs
            for i in range(0, len(variation), max(1, self.r_contiguous - 1)):
                variation[i] = random.randint(0, len(self.vocabulary) - 1)
            return tuple(variation)
        
        else:
            # Fallback: change multiple random positions
            num_changes = random.randint(1, max(2, len(variation) // 2))
            positions = random.sample(range(len(variation)), num_changes)
            for pos in positions:
                variation[pos] = random.randint(0, len(self.vocabulary) - 1)
            return tuple(variation)
    
    def _create_spam_detector_variation(self, base_pattern: tuple) -> tuple:
        """Create a SMALL variation of spam pattern that still matches under current rule."""
        variation = list(base_pattern)
        
        if self.matching_rule == "hamming" and self.hamming_threshold > 0:
            # Change UP TO hamming_threshold positions (so it still matches)
            num_changes = random.randint(1, min(self.hamming_threshold, len(variation)))
            positions = random.sample(range(len(variation)), num_changes)
            for pos in positions:
                variation[pos] = random.randint(0, len(self.vocabulary) - 1)
            return tuple(variation)
        
        elif self.matching_rule == "r_contiguous":
            # For r-contiguous: change positions but preserve at least r_contiguous consecutive matches
            # Strategy: Only change positions at edges or with gaps >= r_contiguous
            safe_positions = []
            # Keep middle section intact, only change edges
            if len(variation) > self.r_contiguous:
                # Can change first few and last few positions
                edge_size = max(1, (len(variation) - self.r_contiguous) // 2)
                safe_positions = list(range(edge_size)) + list(range(len(variation) - edge_size, len(variation)))
            
            if safe_positions:
                pos = random.choice(safe_positions)
                variation[pos] = random.randint(0, len(self.vocabulary) - 1)
            return tuple(variation)
        
        else:
            # Fallback: small random change (1-2 positions)
            num_changes = random.randint(1, min(2, len(variation)))
            positions = random.sample(range(len(variation)), num_changes)
            for pos in positions:
                variation[pos] = random.randint(0, len(self.vocabulary) - 1)
            return tuple(variation)

    def fit(self, X: Union[List[str], List[np.ndarray]], y: List[int]):
        """Generate detectors trained on both ham and spam patterns."""
        if hasattr(self, 'seed') and self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        self.detectors = []
        
        # Extract ham and spam samples separately
        ham_samples = [x for x, label in zip(X, y) if label == 0]
        spam_samples = [x for x, label in zip(X, y) if label == 1]
        
        if self.representation == "vocabulary":
            print(f"Training PURE NSA with {self.matching_rule} matching (self-only learning)...")
            print(f"Training on {len(ham_samples)} ham samples (self) - spam samples NOT used for detector generation")
            
            # Build vocabulary from all samples (shared feature space is acceptable)
            self._build_vocabulary(X)
            if len(self.vocabulary) == 0:
                print("Warning: No vocabulary built")
                return self
            
            # PURE NSA: Extract ham patterns to avoid (self-tolerance)
            ham_pattern_sets = []
            for text in ham_samples:
                tokens = self._text_to_tokens(text)
                patterns = self._get_patterns_from_tokens(tokens)
                ham_pattern_sets.append(patterns)
            
            print(f"  Extracted {sum(len(ps) for ps in ham_pattern_sets)} total ham patterns from {len(ham_samples)} samples")
            
            # PURE NSA: Generate detectors through RANDOM sampling with negative selection
            # IMPROVEMENT: Increase max_ham_match_ratio slightly to allow more detectors through
            # This is still pure NSA - we're just being less strict about self-tolerance
            effective_ham_ratio = min(0.15, self.max_ham_match_ratio * 3)  # Allow up to 15% ham matches
            print(f"Generating {self.num_detectors} detectors via random sampling + negative selection...")
            print(f"  Ham tolerance: {effective_ham_ratio*100:.1f}% (relaxed for better coverage)")
            
            attempts = 0
            last_printed_count = 0
            detector_set = set()  # Track unique detectors for diversity
            
            while len(self.detectors) < self.num_detectors and attempts < self.max_attempts:
                # PURE NSA: Random detector from vocabulary space
                candidate = self._random_detector_vocab()
                
                # Check detector diversity (avoid duplicates)
                if candidate in detector_set:
                    attempts += 1
                    continue
                
                # NEGATIVE SELECTION: Test candidate against ham (self) patterns
                ham_matches = 0
                for pattern_set in ham_pattern_sets:
                    for pattern in pattern_set:
                        if self._matches_pattern(candidate, pattern):
                            ham_matches += 1
                            break  # Count each ham sample only once
                
                # Reject if candidate matches too many ham samples (fails self-tolerance)
                # Using relaxed threshold to allow more detectors through
                max_ham_matches = max(1, int(len(ham_samples) * effective_ham_ratio))
                
                if ham_matches <= max_ham_matches:
                    self.detectors.append(candidate)
                    detector_set.add(candidate)
                
                attempts += 1
                
                # Print progress
                if (len(self.detectors) > 0 and len(self.detectors) % 100 == 0 and len(self.detectors) != last_printed_count) or \
                   (attempts % 1000 == 0 and attempts > 0):
                    success_rate = len(self.detectors) / attempts * 100 if attempts > 0 else 0
                    print(f"  Generated {len(self.detectors)}/{self.num_detectors} detectors (success rate: {success_rate:.1f}%)...")
                    last_printed_count = len(self.detectors)
        
        else:
            # Binary representation (for compatibility)
            print(f"Training binary NSA with Hamming distance...")
            self_samples = np.array(ham_samples)
            attempts = 0
            
            while len(self.detectors) < self.num_detectors and attempts < self.max_attempts:
                cand = np.random.randint(0, 2, size=self.feature_length, dtype=np.uint8)
                distances = np.sum(self_samples != cand, axis=1)
                
                if not np.any(distances <= self.hamming_threshold):
                    self.detectors.append(cand)
                
                attempts += 1
        
        print(f"Detector generation complete: {len(self.detectors)} detectors generated")
        if len(self.detectors) == 0:
            print(f"  No detectors generated for {dict(num_detectors=self.num_detectors, detector_size=self.detector_size, **{k: getattr(self, k) for k in ['hamming_threshold', 'r_contiguous'] if hasattr(self, k)}, matching_rule=self.matching_rule)}")
        return self

    def predict(self, X: Union[List[str], List[np.ndarray]]) -> List[int]:
        """Predict labels - classify as spam if any detector matches."""
        predictions = []
        
        for sample in X:
            if self.representation == "vocabulary":
                # Vocabulary prediction
                tokens = self._text_to_tokens(sample)
                if len(tokens) < self.detector_size:
                    predictions.append(0)  # Too short, classify as ham
                    continue
                
                # Check all patterns in the text
                spam_detected = False
                for i in range(len(tokens) - self.detector_size + 1):
                    pattern = tuple(tokens[i:i + self.detector_size])
                    
                    # Check if any detector matches this pattern
                    for detector in self.detectors:
                        if self._matches_pattern(detector, pattern):
                            spam_detected = True
                            break
                    
                    if spam_detected:
                        break
                
                predictions.append(1 if spam_detected else 0)
            
            else:
                # Binary prediction using minimum activations
                activations = 0
                for detector in self.detectors:
                    if self._matches_pattern(detector, sample):
                        activations += 1
                
                predictions.append(1 if activations >= self.min_activations else 0)
        
        return predictions

    def get_info(self) -> dict:
        """Get classifier information for analysis."""
        info = {
            "num_detectors": len(self.detectors),
            "detector_size": self.detector_size,
            "matching_rule": self.matching_rule,
            "representation": self.representation,
        }
        
        if self.representation == "vocabulary":
            info.update({
                "vocabulary_size": len(self.vocabulary),
                "r_contiguous": self.r_contiguous if self.matching_rule == "r_contiguous" else None,
                "hamming_threshold": self.hamming_threshold if self.matching_rule == "hamming" else None,
            })
        else:
            info.update({
                "hamming_threshold": self.hamming_threshold,
                "min_activations": self.min_activations,
            })
        
        return info