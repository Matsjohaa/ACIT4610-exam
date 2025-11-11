"""Preprocessing utilities for SMS spam dataset (Problem 4).

Responsibilities:
- Load TSV file with columns: label<TAB>message
- Clean & normalize text
- Train/test split
- Convert texts to binary representations for authentic NSA
"""

from __future__ import annotations

import csv
import re
import hashlib
import numpy as np
from collections import Counter
from typing import Dict, List, Sequence, Tuple


def clean_text(text: str) -> str:
	"""Return text unchanged, preserving original casing and punctuation.

	The previous implementation aggressively normalized (lowercased, stripped non-alphanumerics),
	which can remove potentially discriminative spam signals (e.g., repeated !!!, currency symbols, casing).
	Per user request, this function now acts as a passthrough.

	If later you want a middle ground, consider:
	- Adding lightweight normalization (e.g., collapse whitespace) without losing symbols.
	- Tagging patterns (URLs, numbers) rather than deleting characters.
	"""
	#text = text.lower()
	#text = re.sub(r"[^a-z0-9\s]", " ", text)
	#text = re.sub(r"\s+", " ", text).strip()
	return text


def load_data(tsv_path: str) -> Tuple[List[str], List[int]]:
	"""Load dataset and return cleaned texts & binary labels (spam=1, ham=0)."""
	texts: List[str] = []
	labels: List[int] = []
	with open(tsv_path, "r", encoding="utf-8") as f:
		reader = csv.reader(f, delimiter="\t")
		for row in reader:
			if len(row) < 2:
				continue
			label_raw, msg = row[0], row[1]
			labels.append(1 if label_raw.strip().lower() == "spam" else 0)
			texts.append(clean_text(msg))
	return texts, labels


def text_to_binary_features(text: str, feature_length: int = 128) -> np.ndarray:
	"""Convert text to sparse binary feature representation using selective hashing.
	
	Creates a discriminative binary vector by:
	1. Extracting key features: words, special patterns, bigrams
	2. Using controlled hashing with different seeds per feature type
	3. Limiting bits set per feature type to maintain sparsity
	
	This creates a sparse, discriminative representation suitable for NSA matching.
	"""
	binary_features = np.zeros(feature_length, dtype=np.uint8)
	text_lower = text.lower()
	
	# Feature type allocation (bits reserved for each type)
	word_bits = feature_length // 2      # 64 bits for words
	pattern_bits = feature_length // 4   # 32 bits for special patterns  
	bigram_bits = feature_length // 4    # 32 bits for bigrams
	
	# 1. Word features (most discriminative)
	words = [w for w in text_lower.split() if len(w) >= 2]
	for word in words[:10]:  # Limit to first 10 words to control density
		hash_val = hash(word + "word") % (2**31)
		bit_pos = hash_val % word_bits
		binary_features[bit_pos] = 1
	
	# 2. Special pattern features (spam indicators)
	patterns_found = []
	if re.search(r'[!]{2,}', text):  # Multiple exclamation marks
		patterns_found.append('multi_excl')
	if re.search(r'[A-Z]{3,}', text):  # All caps words
		patterns_found.append('all_caps')
	if re.search(r'\d+', text):  # Contains numbers
		patterns_found.append('has_numbers')
	if re.search(r'[£$€]', text):  # Currency symbols
		patterns_found.append('currency')
	if re.search(r'www\.|http|\.com', text.lower()):  # URLs
		patterns_found.append('url')
	if re.search(r'\bfree\b', text.lower()):  # "free" keyword
		patterns_found.append('free_word')
	if re.search(r'\bwin\b|\bwinner\b', text.lower()):  # Win keywords
		patterns_found.append('win_word')
	
	for pattern in patterns_found:
		hash_val = hash(pattern + "pattern") % (2**31)
		bit_pos = word_bits + (hash_val % pattern_bits)
		binary_features[bit_pos] = 1
	
	# 3. Character bigrams (selective - skip common ones)
	bigrams = []
	for i in range(len(text_lower) - 1):
		bigram = text_lower[i:i+2]
		if not bigram.isspace() and not bigram.isalnum():  # Special char bigrams
			bigrams.append(bigram)
	
	for bigram in bigrams[:5]:  # Limit bigrams
		hash_val = hash(bigram + "bigram") % (2**31)
		bit_pos = word_bits + pattern_bits + (hash_val % bigram_bits)
		binary_features[bit_pos] = 1
	
	return binary_features


def train_test_split(texts: Sequence[str], labels: Sequence[int], test_ratio: float = 0.2, seed: int = 42):
	"Divides the dataset into training and testing sets, based on the specified test ratio in constants.py"
	
	import random

	idx = list(range(len(texts)))
	random.Random(seed).shuffle(idx)
	cut = int(len(idx) * (1 - test_ratio))
	train_idx, test_idx = idx[:cut], idx[cut:]
	X_train = [texts[i] for i in train_idx]
	y_train = [labels[i] for i in train_idx]
	X_test = [texts[i] for i in test_idx]
	y_test = [labels[i] for i in test_idx]
	return X_train, y_train, X_test, y_test


def train_val_test_split(
	texts: Sequence[str],
	labels: Sequence[int],
	test_ratio: float = 0.2,
	val_ratio: float = 0.1,
	seed: int = 42,
):
	"""Split into train/validation/test.

	All ratios are applied to the total dataset size.
	Example: test_ratio=0.2, val_ratio=0.15 => 20% test, 15% validation, 65% train.
	"""
	import random
	idx = list(range(len(texts)))
	random.Random(seed).shuffle(idx)
	
	# Calculate splits based on total dataset size
	total_size = len(idx)
	test_size = int(total_size * test_ratio)
	val_size = int(total_size * val_ratio)
	train_size = total_size - test_size - val_size
	
	# Split indices
	test_idx = idx[:test_size]
	val_idx = idx[test_size:test_size + val_size]
	train_idx = idx[test_size + val_size:]
	
	X_train = [texts[i] for i in train_idx]
	y_train = [labels[i] for i in train_idx]
	X_val = [texts[i] for i in val_idx]
	y_val = [labels[i] for i in val_idx]
	X_test = [texts[i] for i in test_idx]
	y_test = [labels[i] for i in test_idx]
	return X_train, y_train, X_val, y_val, X_test, y_test


def texts_to_binary_arrays(texts: Sequence[str], feature_length: int = 128) -> List[np.ndarray]:
	"""Convert texts to binary feature arrays."""
	return [text_to_binary_features(text, feature_length) for text in texts]


# Legacy functions for backward compatibility (now deprecated)
def build_vocabulary(texts: Sequence[str], min_freq: int = 2, max_size: int = 5000):
	"""Build vocabulary from texts. Return list of words and word-to-index mapping.
	
	DEPRECATED: This function is kept for backward compatibility but should not
	be used in authentic NSA implementation as it violates self-learning principles.
	"""
	counter: Counter[str] = Counter()
	for t in texts:
		counter.update(t.split())
	vocab = [w for w, c in counter.items() if c >= min_freq]
	vocab = sorted(vocab, key=lambda w: (-counter[w], w))[:max_size]
	index = {w: i for i, w in enumerate(vocab)}
	return list(vocab), index


def texts_to_sets(texts: Sequence[str], vocab_index: Dict[str, int]):
	"""Convert texts to list of sets of token IDs based on the provided vocabulary index.
	
	DEPRECATED: This function is kept for backward compatibility but should not
	be used in authentic NSA implementation as it violates self-learning principles.
	"""
	reps = []
	for t in texts:
		token_ids = {vocab_index[w] for w in t.split() if w in vocab_index}
		reps.append(token_ids)
	return reps


__all__ = [
	"clean_text",
	"load_data",
	"train_test_split",
	"train_val_test_split",
	"text_to_binary_features",
	"texts_to_binary_arrays",
	# Legacy functions (deprecated)
	"build_vocabulary",
	"texts_to_sets",
]

