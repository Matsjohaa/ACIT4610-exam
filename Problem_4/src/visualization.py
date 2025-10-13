"""Visualization utilities for Problem 4."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from utils import compute_confusion


def plot_confusion(y_true, y_pred):
	tp, tn, fp, fn = compute_confusion(y_true, y_pred)
	# Confusion matrix with rows = true (ham=0, spam=1), cols = pred
	# [[TN, FP], [FN, TP]]
	mat = np.array([[tn, fp], [fn, tp]])
	fig, ax = plt.subplots()
	im = ax.imshow(mat, cmap="Blues")
	ax.set_xticks([0, 1])
	ax.set_yticks([0, 1])
	ax.set_xticklabels(["Ham", "Spam"])
	ax.set_yticklabels(["Ham", "Spam"])
	for i in range(2):
		for j in range(2):
			ax.text(j, i, int(mat[i, j]), ha="center", va="center", color="black")
	ax.set_xlabel("Predicted")
	ax.set_ylabel("True")
	ax.set_title("Confusion Matrix")
	plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
	plt.tight_layout()
	return fig


__all__ = ["plot_confusion"]

