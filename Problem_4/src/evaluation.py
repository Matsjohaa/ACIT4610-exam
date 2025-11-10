"""Evaluation entry point for A SINGLE RUN, validation phase will be useless here."""

from __future__ import annotations

import sys, pathlib
_THIS_DIR = pathlib.Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
	sys.path.insert(0, str(_THIS_DIR))

from preprocessing import load_data, train_val_test_split, build_vocabulary, texts_to_sets
from nsa import NegativeSelectionClassifier
from utils import set_seed, classification_report
from constants import (
	SEED,
	DATA_PATH,
	TEST_RATIO,
	VAL_RATIO,
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
from utils import coverage_curve, roc_pr_points, detector_diversity_stats
from pathlib import Path
import csv
import random, time
from sklearn.metrics import roc_auc_score, average_precision_score



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


def run_evaluation(save_misclassifications: bool = True, save_artifacts: bool = True):
	# Handle dynamic seed: if SEED is None, generate one (time & randomness based) and print it.
	if SEED is None:
		generated_seed = random.randrange(0, 2**31 - 1) ^ int(time.time())
		print(f"[INFO] Generated random SEED={generated_seed} (set this in constants.py to reproduce)")
		seed_value = generated_seed
	else:
		seed_value = SEED
	set_seed(seed_value)
	texts, labels = load_data(str(DATA_PATH))
	X_train_texts, y_train, X_val_texts, y_val, X_test_texts, y_test = train_val_test_split(
		texts, labels, test_ratio=TEST_RATIO, val_ratio=VAL_RATIO, seed=seed_value
	)
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

	gen_start = time.time()
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
	gen_end = time.time()

	# Get predictions + activation score (acts) for curves
	pred_start = time.time()
	y_pred, acts = model.predict_with_scores(X_test_sets)
	pred_end = time.time()
	# Validation predictions for early stopping / future tuning context (not used for training NSA currently)
	val_sets = texts_to_sets(X_val_texts, vocab_index)
	val_pred, val_acts = model.predict_with_scores(val_sets)
	val_report = classification_report(y_val, val_pred)

	# Test evaluation
	report = classification_report(y_test, y_pred)

	# Compute ROC-AUC & PR-AUC (average precision) treating acts as scores.
	# Guard against degenerate cases (only one class present in test labels).
	import math
	if len(set(y_test)) == 2 and len(set(acts)) > 1:
		try:
			roc_auc = roc_auc_score(y_test, acts)
			pr_auc = average_precision_score(y_test, acts)
			report["roc_auc"] = float(roc_auc)
			report["pr_auc"] = float(pr_auc)
		except Exception:
			report["roc_auc"] = math.nan
			report["pr_auc"] = math.nan
	else:
		report["roc_auc"] = math.nan
		report["pr_auc"] = math.nan

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

	# Attach validation metrics namespaced
	for k, v in val_report.items():
		report[f"val_{k}"] = v

	# Detector stats
	report["detectors_count"] = getattr(model, "detectors_count", len(model.detectors))
	import math as _m
	attempts_used = getattr(model, "attempts_used", float("nan"))
	if attempts_used is None:
		attempts_used = _m.nan
	report["detector_attempts_used"] = float(attempts_used)
	# Timing
	report["generation_seconds"] = float(gen_end - gen_start)
	report["prediction_seconds"] = float(pred_end - pred_start)

	# Diversity stats
	div_stats = detector_diversity_stats(model.detectors)
	for k, v in div_stats.items():
		report[f"div_{k}"] = v

	# Activation distributions (validation spam/ham)
	val_spam_acts = [a for a, lbl in zip(val_acts, y_val) if lbl == 1]
	val_ham_acts = [a for a, lbl in zip(val_acts, y_val) if lbl == 0]
	def _dist_summary(arr):
		import math as _m
		if not arr:
			return {"min": _m.nan, "max": _m.nan, "mean": _m.nan}
		return {"min": float(min(arr)), "max": float(max(arr)), "mean": float(sum(arr)/len(arr))}
	spam_summary = _dist_summary(val_spam_acts)
	ham_summary = _dist_summary(val_ham_acts)
	for k,v in spam_summary.items():
		report[f"val_spam_act_{k}"] = v
	for k,v in ham_summary.items():
		report[f"val_ham_act_{k}"] = v

	# Coverage curve (subset checkpoints)
	checkpoints = [50, 100, 200, 400, 600, 800, report["detectors_count"]]
	# Cast detectors to list of sequences (each detector already a set)
	curve = coverage_curve(list(model.detectors), X_test_sets, y_test, checkpoints, model.overlap_threshold, model.min_activations)

	# ROC/PR raw points
	rocpr = roc_pr_points(y_test, acts)

	# Artifact persistence (JSON/TSV for notebook plots)
	if save_artifacts:
		RESULTS_DIR.mkdir(parents=True, exist_ok=True)
		import json, csv as _csv
		# coverage curve JSON
		curve_path = RESULTS_DIR / "coverage_curve.json"
		with open(curve_path, "w", encoding="utf-8") as f:
			json.dump(curve, f, ensure_ascii=False, indent=2)
			print(f"[INFO] Saved coverage curve to {curve_path}")
		# ROC curve TSV
		roc_path = RESULTS_DIR / "roc_curve.tsv"
		with open(roc_path, "w", encoding="utf-8", newline="") as f:
			w = _csv.writer(f, delimiter="\t")
			w.writerow(["fpr", "tpr"]) 
			for fpr, tpr in zip(rocpr.get("fpr", []), rocpr.get("tpr", [])):
				w.writerow([fpr, tpr])
			print(f"[INFO] Saved ROC curve points to {roc_path}")
		# PR curve TSV
		pr_path = RESULTS_DIR / "pr_curve.tsv"
		with open(pr_path, "w", encoding="utf-8", newline="") as f:
			w = _csv.writer(f, delimiter="\t")
			w.writerow(["recall", "precision"]) 
			for rec, prec in zip(rocpr.get("recall", []), rocpr.get("precision", [])):
				w.writerow([rec, prec])
			print(f"[INFO] Saved PR curve points to {pr_path}")
		# Detector stats JSON
		stats_path = RESULTS_DIR / "detector_stats.json"
		stats_payload = {
			"detectors_count": report["detectors_count"],
			"detector_attempts_used": report["detector_attempts_used"],
			"diversity": div_stats,
			"generation_seconds": report["generation_seconds"],
			"prediction_seconds": report["prediction_seconds"],
			"val_spam_activation_summary": spam_summary,
			"val_ham_activation_summary": ham_summary,
		}
		with open(stats_path, "w", encoding="utf-8") as f:
			json.dump(stats_payload, f, ensure_ascii=False, indent=2)
			print(f"[INFO] Saved detector stats to {stats_path}")

	return {
		"report": report,
		"y_test": y_test,
		"y_pred": y_pred,
		"scores_test": acts,
		"scores_val": val_acts,
		"coverage_curve": curve,
		"roc_pr_points": rocpr,
	}
	return report, y_test, y_pred, acts


if __name__ == "__main__":
	out = run_evaluation()
	rep = out["report"]
	for k, v in rep.items():
		print(f"{k}: {v}")

