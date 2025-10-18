"""Evaluation entry point for Problem 4 NSA spam detector."""

from __future__ import annotations

import sys, pathlib
_THIS_DIR = pathlib.Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
	sys.path.insert(0, str(_THIS_DIR))

from preprocessing import load_data, train_test_split, build_vocabulary, texts_to_sets
from nsa import NegativeSelectionClassifier
from utils import set_seed, classification_report
from constants import (
	SEED,
	DATA_PATH,
	TEST_RATIO,
	VOCAB_MIN_FREQ,
	VOCAB_MAX_SIZE,
	NSA_NUM_DETECTORS,
	NSA_DETECTOR_SIZE,
	NSA_OVERLAP_THRESHOLD,
	NSA_MAX_ATTEMPTS,
	NSA_MIN_ACTIVATIONS,
	NSA_USE_SPAM_WEIGHTS,
	NSA_WEIGHT_EXP,
	RESULTS_DIR,
	MISCLASS_FP_FILENAME,
	MISCLASS_FN_FILENAME,
	MISCLASS_INCLUDE_HEADER,
	TRUE_POS_FILENAME,
	TRUE_NEG_FILENAME,
)
from pathlib import Path
import csv
import random, time



def _write_results(
	X_test_texts,
	y_test,
	y_pred,
	results_dir: Path,
	fp_filename: str,
	fn_filename: str,
	tp_filename: str,
	tn_filename: str,
	include_header: bool,
	run_seed: int,
):
	"""Persist classification outcomes to TSV files: false positives, false negatives,
	true positives, true negatives.

	Columns: label\ttext (label is TRUE class value: ham/spam).
	"""
	results_dir.mkdir(parents=True, exist_ok=True)

	# Helper to convert numeric label to string
	def lbl(val: int) -> str:
		return "spam" if val == 1 else "ham"

	fps = []
	fns = []
	tp = []
	tn = []
	for txt, yt, yp in zip(X_test_texts, y_test, y_pred):
		if yt == 0 and yp == 1:
			fps.append((yt, yp, txt))
		elif yt == 1 and yp == 0:
			fns.append((yt, yp, txt))
		elif yt == 1 and yp == 1:
			tp.append((yt, yp, txt))
		elif yt == 0 and yp == 0:
			tn.append((yt, yp, txt))

	# False positives
	fp_path = results_dir / fp_filename
	with open(fp_path, "w", encoding="utf-8", newline="") as f:
		writer = csv.writer(f, delimiter="\t")
		if include_header:
			writer.writerow([f"# seed={run_seed} fp_count={len(fps)}"])
			writer.writerow(["label", "text"])
		for yt, yp, txt in fps:
			writer.writerow([lbl(yt), txt])

	# False negatives
	fn_path = results_dir / fn_filename
	with open(fn_path, "w", encoding="utf-8", newline="") as f:
		writer = csv.writer(f, delimiter="\t")
		if include_header:
			writer.writerow([f"# seed={run_seed} fn_count={len(fns)}"])
			writer.writerow(["label", "text"])
		for yt, yp, txt in fns:
			writer.writerow([lbl(yt), txt])

	# True positives
	tp_path = results_dir / tp_filename
	with open(tp_path, "w", encoding="utf-8", newline="") as f:
		writer = csv.writer(f, delimiter="\t")
		if include_header:
			writer.writerow([f"# seed={run_seed} tp_count={len(tp)}"])
			writer.writerow(["label", "text"])
		for yt, yp, txt in tp:
			writer.writerow([lbl(yt), txt])

	# True negatives
	tn_path = results_dir / tn_filename
	with open(tn_path, "w", encoding="utf-8", newline="") as f:
		writer = csv.writer(f, delimiter="\t")
		if include_header:
			writer.writerow([f"# seed={run_seed} tn_count={len(tn)}"])
			writer.writerow(["label", "text"])
		for yt, yp, txt in tn:
			writer.writerow([lbl(yt), txt])

	print(f"[INFO] Wrote false positives (count={len(fps)}) to: {fp_path}")
	print(f"[INFO] Wrote false negatives (count={len(fns)}) to: {fn_path}")
	print(f"[INFO] Wrote true positives (count={len(tp)}) to: {tp_path}")
	print(f"[INFO] Wrote true negatives (count={len(tn)}) to: {tn_path}")


def run_evaluation(save_misclassifications: bool = True):
	# Handle dynamic seed: if SEED is None, generate one (time & randomness based) and print it.
	if SEED is None:
		generated_seed = random.randrange(0, 2**31 - 1) ^ int(time.time())
		print(f"[INFO] Generated random SEED={generated_seed} (set this in constants.py to reproduce)")
		seed_value = generated_seed
	else:
		seed_value = SEED
	set_seed(seed_value)
	texts, labels = load_data(str(DATA_PATH))
	X_train_texts, y_train, X_test_texts, y_test = train_test_split(texts, labels, test_ratio=TEST_RATIO)
	vocab, vocab_index = build_vocabulary(X_train_texts, min_freq=VOCAB_MIN_FREQ, max_size=VOCAB_MAX_SIZE)
	X_train_sets = texts_to_sets(X_train_texts, vocab_index)
	X_test_sets = texts_to_sets(X_test_texts, vocab_index)

	# Optional spam-guided weighting: compute token frequencies separately for spam and ham.
	weights = None
	if NSA_USE_SPAM_WEIGHTS:
		spam_counts = [0] * len(vocab)
		ham_counts = [0] * len(vocab)
		for text, label in zip(X_train_texts, y_train):
			tokens = text.split()
			unique = set(tokens)
			for tok in unique:  # presence-based to align with set model
				idx = vocab_index.get(tok)
				if idx is None:
					continue
				if label == 1:
					spam_counts[idx] += 1
				else:
					ham_counts[idx] += 1
		# Build weight vector emphasizing spam-skewed tokens
		weights = []
		for s_c, h_c in zip(spam_counts, ham_counts):
			ratio = (s_c + 1) / (h_c + 1)
			weights.append(ratio ** NSA_WEIGHT_EXP)

	model = NegativeSelectionClassifier(
		vocab_size=len(vocab),
		num_detectors=NSA_NUM_DETECTORS,
		detector_size=NSA_DETECTOR_SIZE,
		overlap_threshold=NSA_OVERLAP_THRESHOLD,
		max_attempts=NSA_MAX_ATTEMPTS,
		seed=seed_value,
		min_activations=NSA_MIN_ACTIVATIONS,
		weights=weights,
	).fit(X_train_sets, y_train)

	y_pred = model.predict(X_test_sets)
	report = classification_report(y_test, y_pred)

	if save_misclassifications:
		_write_results(
			X_test_texts,
			y_test,
			y_pred,
			RESULTS_DIR,
			MISCLASS_FP_FILENAME,
			MISCLASS_FN_FILENAME,
			TRUE_POS_FILENAME,
			TRUE_NEG_FILENAME,
			MISCLASS_INCLUDE_HEADER,
			run_seed=seed_value,
		)

	return report, y_test, y_pred


if __name__ == "__main__":
	report, *_ = run_evaluation()
	for k, v in report.items():
		print(f"{k}: {v}")

