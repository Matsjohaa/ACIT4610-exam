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
	RESULTS_DIR,
	MISCLASS_FP_FILENAME,
	MISCLASS_FN_FILENAME,
	MISCLASS_INCLUDE_HEADER,
)
from pathlib import Path
import csv
import random, time


def _write_misclassifications(
	X_test_texts,
	y_test,
	y_pred,
	results_dir: Path,
	fp_filename: str,
	fn_filename: str,
	include_header: bool,
	run_seed: int,
):
	"""Persist false positives & false negatives to TSV files.

	Columns now: label\ttext
	Where 'label' is the TRUE class (ham/spam).
	"""
	results_dir.mkdir(parents=True, exist_ok=True)

	# Helper to convert numeric label to string
	def lbl(val: int) -> str:
		return "spam" if val == 1 else "ham"

	fps = []
	fns = []
	for txt, yt, yp in zip(X_test_texts, y_test, y_pred):
		if yt == 0 and yp == 1:
			fps.append((yt, yp, txt))
		elif yt == 1 and yp == 0:
			fns.append((yt, yp, txt))

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

	print(f"[INFO] Wrote false positives (count={len(fps)}) to: {fp_path}")
	print(f"[INFO] Wrote false negatives (count={len(fns)}) to: {fn_path}")


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

	model = NegativeSelectionClassifier(
		vocab_size=len(vocab),
		num_detectors=NSA_NUM_DETECTORS,
		detector_size=NSA_DETECTOR_SIZE,
		overlap_threshold=NSA_OVERLAP_THRESHOLD,
		max_attempts=NSA_MAX_ATTEMPTS,
		seed=seed_value,
	).fit(X_train_sets, y_train)

	y_pred = model.predict(X_test_sets)
	report = classification_report(y_test, y_pred)

	if save_misclassifications:
		_write_misclassifications(
			X_test_texts,
			y_test,
			y_pred,
			RESULTS_DIR,
			MISCLASS_FP_FILENAME,
			MISCLASS_FN_FILENAME,
			MISCLASS_INCLUDE_HEADER,
			run_seed=seed_value,
		)

	return report, y_test, y_pred


if __name__ == "__main__":
	report, *_ = run_evaluation()
	for k, v in report.items():
		print(f"{k}: {v}")

