#!/usr/bin/env python3
"""
Generate tables and plots from final_precision_results.csv
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from constants import RESULTS_DIR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("="*80)
print("GENERATING TABLES AND PLOTS")
print("="*80)

# Load results
results_path = RESULTS_DIR / "final_precision_results.csv"
results_df = pd.read_csv(results_path)
print(f"\nLoaded {len(results_df)} configurations from {results_path}")

# Create plots directory
plots_dir = RESULTS_DIR / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

# Add strategy name column
results_df['name'] = results_df['detector_set'] + '_min' + results_df['min_activations'].astype(str)

# ============================================================================
# TABLE 1: Score Table (Top 15 by F1)
# ============================================================================
print("\n" + "="*80)
print("TABLE 1: Performance Metrics (Top 15 by F1)")
print("="*80)

# Calculate additional metrics
results_df['accuracy'] = (results_df['tp'] + results_df['tn']) / (results_df['tp'] + results_df['tn'] + results_df['fp'] + results_df['fn'])
results_df['fpr'] = results_df['fp'] / (results_df['fp'] + results_df['tn'])

# Take top 15 by F1
top_15_f1 = results_df.nlargest(15, 'f1')

score_table = top_15_f1[['name', 'precision', 'recall', 'f1', 'accuracy', 'fpr', 'fp', 'fn']].copy()
score_table.columns = ['Strategy', 'Precision', 'Recall', 'F1', 'Accuracy', 'FPR', 'FP', 'FN']

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: f'{x:.4f}' if pd.notna(x) and isinstance(x, float) else x)
print(score_table.to_string(index=False))

# Save table
score_table.to_csv(RESULTS_DIR / "score_table.csv", index=False)
print(f"\n✓ Saved to: {RESULTS_DIR / 'score_table.csv'}")

# ============================================================================
# TABLE 2: Hyperparameters (Top 15 by F1)
# ============================================================================
print("\n" + "="*80)
print("TABLE 2: Hyperparameters (Top 15 by F1)")
print("="*80)

hyperparams_table = top_15_f1[['name', 'num_detectors', 'min_activations', 'max_ham_match_ratio']].copy()
hyperparams_table.columns = ['Strategy', 'Num Detectors', 'Min Activations', 'Max Ham Ratio']
hyperparams_table['r_contiguous'] = 2
hyperparams_table['detector_size'] = 4
hyperparams_table['vocab_size'] = 2000

hyperparams_table = hyperparams_table[['Strategy', 'Num Detectors', 'Min Activations', 'Max Ham Ratio', 
                                        'r_contiguous', 'detector_size', 'vocab_size']]

print(hyperparams_table.to_string(index=False))

hyperparams_table.to_csv(RESULTS_DIR / "hyperparameters_table.csv", index=False)
print(f"\n✓ Saved to: {RESULTS_DIR / 'hyperparameters_table.csv'}")

# ============================================================================
# PLOT 1: Pareto Front (Precision vs Recall)
# ============================================================================
print("\n" + "="*80)
print("PLOT 1: Pareto Front (Precision vs Recall)")
print("="*80)

plt.style.use('default')
fig, ax = plt.subplots(figsize=(12, 8))

# F1 contour lines
recall_range = np.linspace(0.01, 1, 100)
f1_levels = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

for f1_val in f1_levels:
    precision_for_f1 = (f1_val * recall_range) / (2 * recall_range - f1_val)
    precision_for_f1 = np.clip(precision_for_f1, 0, 1)
    ax.plot(recall_range, precision_for_f1, '--', alpha=0.3, linewidth=1, color='gray')
    valid_idx = np.where((precision_for_f1 > 0.1) & (precision_for_f1 < 0.95))[0]
    if len(valid_idx) > 0:
        mid_idx = valid_idx[len(valid_idx)//2]
        ax.text(recall_range[mid_idx], precision_for_f1[mid_idx], f'F1={f1_val:.1f}', 
               fontsize=9, alpha=0.6)

# Plot all results - color by detector set
detector_sets = results_df['detector_set'].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(detector_sets)))
color_map = dict(zip(detector_sets, colors))

for det_set in detector_sets:
    subset = results_df[results_df['detector_set'] == det_set]
    ax.scatter(subset['recall'], subset['precision'], s=100, alpha=0.6, 
              color=color_map[det_set], label=det_set, edgecolors='black', linewidth=0.5)

# Highlight best F1
best_f1 = results_df.loc[results_df['f1'].idxmax()]
ax.scatter(best_f1['recall'], best_f1['precision'], s=400, alpha=0.8, 
          color='red', marker='*', edgecolors='black', linewidth=2, zorder=10, label='Best F1')

# Highlight best precision
best_prec = results_df.loc[results_df['precision'].idxmax()]
ax.scatter(best_prec['recall'], best_prec['precision'], s=400, alpha=0.8, 
          color='gold', marker='*', edgecolors='black', linewidth=2, zorder=10, label='Best Precision')

# Target region
ax.axhline(y=0.80, color='red', linestyle='--', linewidth=1.5, alpha=0.4, label='Target P≥0.80')
ax.axvline(x=0.40, color='red', linestyle='--', linewidth=1.5, alpha=0.4, label='Target R≥0.40')
ax.fill_between([0.40, 1], 0.80, 1, alpha=0.05, color='green')

ax.set_xlabel('Recall', fontsize=13, fontweight='bold')
ax.set_ylabel('Precision', fontsize=13, fontweight='bold')
ax.set_title('Pareto Front: Precision vs Recall (60 Configurations)', fontsize=15, fontweight='bold')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend(loc='lower left', fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(plots_dir / "pareto_front.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved: {plots_dir / 'pareto_front.png'}")
plt.close()

# ============================================================================
# PLOT 2: Performance vs Min Activations (by detector set)
# ============================================================================
print("\n" + "="*80)
print("PLOT 2: Performance vs Min Activations")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, det_set in enumerate(sorted(detector_sets)):
    subset = results_df[results_df['detector_set'] == det_set].sort_values('min_activations')
    
    ax = axes[idx]
    ax.plot(subset['min_activations'], subset['precision'], marker='o', label='Precision', linewidth=2)
    ax.plot(subset['min_activations'], subset['recall'], marker='s', label='Recall', linewidth=2)
    ax.plot(subset['min_activations'], subset['f1'], marker='^', label='F1', linewidth=2)
    
    ax.set_xlabel('Min Activations', fontsize=11, fontweight='bold')
    ax.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax.set_title(f'{det_set} ({subset["num_detectors"].iloc[0]} detectors)', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(plots_dir / "performance_vs_min_activations.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved: {plots_dir / 'performance_vs_min_activations.png'}")
plt.close()

# ============================================================================
# PLOT 3: Best configs comparison
# ============================================================================
print("\n" + "="*80)
print("PLOT 3: Top 10 Configurations Comparison")
print("="*80)

fig, ax = plt.subplots(figsize=(14, 8))

top_10 = results_df.nlargest(10, 'f1')
x = np.arange(len(top_10))
width = 0.25

ax.bar(x - width, top_10['precision'], width, label='Precision', alpha=0.8)
ax.bar(x, top_10['recall'], width, label='Recall', alpha=0.8)
ax.bar(x + width, top_10['f1'], width, label='F1', alpha=0.8)

ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Top 10 Configurations by F1-Score', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(top_10['name'].values, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(plots_dir / "top_10_comparison.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved: {plots_dir / 'top_10_comparison.png'}")
plt.close()

print("\n" + "="*80)
print("✓ ALL TABLES AND PLOTS GENERATED")
print(f"✓ Location: {plots_dir}")
print("="*80)
