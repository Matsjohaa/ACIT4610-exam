"""Preprocessing utilities for SMS spam dataset (Problem 4).

Responsibilities:
- Load TSV file with columns: label<TAB>message
- Clean & normalize text
- Train/test split
- Build vocabulary and represent texts as token-id sets
"""

from __future__ import annotations

import csv
import re
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


def build_vocabulary(texts: Sequence[str], min_freq: int = 2, max_size: int = 5000):
	"Build vocabulary from texts. Return list of words and word-to-index mapping."
	counter: Counter[str] = Counter()
	for t in texts:
		counter.update(t.split())
	vocab = [w for w, c in counter.items() if c >= min_freq]
	vocab = sorted(vocab, key=lambda w: (-counter[w], w))[:max_size]
	index = {w: i for i, w in enumerate(vocab)}
	return list(vocab), index


def texts_to_sets(texts: Sequence[str], vocab_index: Dict[str, int]):
	"Convert texts to list of sets of token IDs based on the provided vocabulary index."
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
	"build_vocabulary",
	"texts_to_sets",
]

