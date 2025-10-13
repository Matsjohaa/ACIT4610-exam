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
	text = text.lower()
	text = re.sub(r"[^a-z0-9\s]", " ", text)
	text = re.sub(r"\s+", " ", text).strip()
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


def build_vocabulary(texts: Sequence[str], min_freq: int = 2, max_size: int = 5000):
	counter: Counter[str] = Counter()
	for t in texts:
		counter.update(t.split())
	vocab = [w for w, c in counter.items() if c >= min_freq]
	vocab = sorted(vocab, key=lambda w: (-counter[w], w))[:max_size]
	index = {w: i for i, w in enumerate(vocab)}
	return list(vocab), index


def texts_to_sets(texts: Sequence[str], vocab_index: Dict[str, int]):
	reps = []
	for t in texts:
		token_ids = {vocab_index[w] for w in t.split() if w in vocab_index}
		reps.append(token_ids)
	return reps


__all__ = [
	"clean_text",
	"load_data",
	"train_test_split",
	"build_vocabulary",
	"texts_to_sets",
]

