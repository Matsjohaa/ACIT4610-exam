#!/usr/bin/env python3
"""
Comprehensive Results Analyzer
- Adds ROC-AUC and PR-AUC values to comprehensive_results.csv
- Creates Pareto front plot with F1 contours
- Shows performance highlights
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from preprocessing import load_data, train_test_split
from nsa_optimized import NegativeSelectionClassifier
from constants import DATA_PATH
import pickle

RESULTS_DIR = Path(__file__).parent.parent / "results"
results_file = RESULTS_DIR / "comprehensive_results.csv"
cache_dir = RESULTS_DIR / "detector_cache_bits"

if not results_file.exists():
    print(f"❌ Results file not found: {results_file}")
    print("Run comprehensive_grid_search.py first!")
    exit(1)

print("="*80)
print("COMPREHENSIVE RESULTS ANALYZER")
print("="*80)

# Load data
print("\n📂 Loading data and results...")
texts, labels = load_data(str(DATA_PATH))
X_train, y_train, X_test, y_test = train_test_split(texts, labels, seed=42)
df = pd.read_csv(results_file)
print(f"   {len(X_test)} test samples ({sum(y_test)} spam, {len(y_test)-sum(y_test)} ham)")
print(f"   {len(df)} configurations loaded")

# ============================================================================
# PHASE 1: Calculate and Add ROC-AUC / PR-AUC if missing
# ============================================================================
if 'roc_auc' not in df.columns or 'pr_auc' not in df.columns:
    print("\n" + "="*80)
    print("PHASE 1: Calculating ROC-AUC and PR-AUC for all configurations")
    print("="*80)
    
    # Get unique detector sets
    detector_sets = df['detector_set'].unique()
    print(f"Processing {len(detector_sets)} detector sets...")
    
    # Dictionary to store AUC values per detector set
    auc_by_detector = {}
    
    for det_name in detector_sets:
        cache_file = cache_dir / f"det_{det_name}.pkl"
        
        if not cache_file.exists():
            print(f"   ⚠️  Skipping {det_name}: cache file not found")
            continue
        
        # Load detector
        with open(cache_file, 'rb') as f:
            nsa = pickle.load(f)
        
        # Get all configurations for this detector set
        det_configs = df[df['detector_set'] == det_name].copy()
        
        # Collect predictions at different thresholds
        y_scores = []
        thresholds = sorted(det_configs['min_activations'].unique())
        
        for thresh in thresholds:
            nsa.min_activations = thresh
            y_pred = nsa.predict(X_test)
            y_scores.append(y_pred)
        
        # Calculate AUC using predictions as scores
        # Use max prediction across thresholds as confidence score
        y_score_max = np.max(y_scores, axis=0)
        
        try:
            roc_auc = roc_auc_score(y_test, y_score_max)
            pr_auc = average_precision_score(y_test, y_score_max)
        except:
            # Fallback: use best F1 config's predictions
            best_config = det_configs.loc[det_configs['f1'].idxmax()]
            nsa.min_activations = int(best_config['min_activations'])
            y_pred = nsa.predict(X_test)
            roc_auc = roc_auc_score(y_test, y_pred) if len(np.unique(y_pred)) > 1 else 0.5
            pr_auc = average_precision_score(y_test, y_pred) if len(np.unique(y_pred)) > 1 else 0.0
        
        auc_by_detector[det_name] = {'roc_auc': roc_auc, 'pr_auc': pr_auc}
        print(f"   ✓ {det_name}: ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}")
    
    # Add AUC values to dataframe
    df['roc_auc'] = df['detector_set'].map(lambda x: auc_by_detector.get(x, {}).get('roc_auc', np.nan))
    df['pr_auc'] = df['detector_set'].map(lambda x: auc_by_detector.get(x, {}).get('pr_auc', np.nan))
    
    # Save updated results
    df.to_csv(results_file, index=False)
    print(f"\n✓ Updated {results_file} with ROC-AUC and PR-AUC values")
else:
    print("\n✓ ROC-AUC and PR-AUC values already present in results file")

# ============================================================================
# PHASE 2: Create Pareto Front Plot with F1 Contours
# ============================================================================
print("\n" + "="*80)
print("PHASE 2: Creating Pareto Front Plot")
print("="*80)

fig, ax = plt.subplots(figsize=(12, 10))

# Create F1 contours in background
recall_range = np.linspace(0, 1, 100)
precision_range = np.linspace(0, 1, 100)
R, P = np.meshgrid(recall_range, precision_range)

# Calculate F1 for each point
F1 = 2 * P * R / (P + R + 1e-10)  # Add small epsilon to avoid division by zero

# Plot F1 contours
contour_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
contours = ax.contour(R, P, F1, levels=contour_levels, colors='gray', alpha=0.3, linewidths=1)
ax.clabel(contours, inline=True, fontsize=9, fmt='F1=%.1f')

# Plot all configurations colored by n-gram size
ngram_colors = {3: '#ff7f0e', 4: '#2ca02c', 5: '#1f77b4'}  # Orange, Green, Blue
ngram_labels = {3: 'N-gram=3', 4: 'N-gram=4', 5: 'N-gram=5'}

for ngram in [3, 4, 5]:
    subset = df[df['ngram'] == ngram]
    ax.scatter(subset['recall'], subset['precision'], 
               c=ngram_colors[ngram], alpha=0.4, s=30,
               label=ngram_labels[ngram], edgecolors='none')

# Identify Pareto front (non-dominated points)
# A point is Pareto optimal if no other point has both higher precision AND higher recall
pareto_mask = np.ones(len(df), dtype=bool)
for i in range(len(df)):
    for j in range(len(df)):
        if i != j:
            # If point j dominates point i (higher or equal in both dimensions, strictly higher in at least one)
            if (df.iloc[j]['recall'] >= df.iloc[i]['recall'] and 
                df.iloc[j]['precision'] >= df.iloc[i]['precision'] and
                (df.iloc[j]['recall'] > df.iloc[i]['recall'] or 
                 df.iloc[j]['precision'] > df.iloc[i]['precision'])):
                pareto_mask[i] = False
                break

pareto_points = df[pareto_mask].copy()
pareto_points = pareto_points.sort_values('recall')

# Plot Pareto front
ax.plot(pareto_points['recall'], pareto_points['precision'], 
        'r-', linewidth=2, label='Pareto Front', zorder=10)
ax.scatter(pareto_points['recall'], pareto_points['precision'],
           c='red', s=100, marker='*', edgecolors='black', linewidths=1,
           label='Pareto Optimal', zorder=11)

# Highlight best F1 configuration
best_f1 = df.loc[df['f1'].idxmax()]
ax.scatter([best_f1['recall']], [best_f1['precision']], 
           c='gold', s=200, marker='★', edgecolors='black', linewidths=2,
           label=f"Best F1={best_f1['f1']:.3f}", zorder=12)

# Labels and styling
ax.set_xlabel('Recall (Spam Detection Rate)', fontsize=12, fontweight='bold')
ax.set_ylabel('Precision (Spam Accuracy)', fontsize=12, fontweight='bold')
ax.set_title('Pareto Front: Precision vs. Recall Trade-off\nwith F1 Score Contours', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='lower left', fontsize=10, framealpha=0.9)

# Add text with Pareto points info
pareto_info = f"Pareto optimal points: {len(pareto_points)}"
ax.text(0.98, 0.02, pareto_info, transform=ax.transAxes,
        fontsize=10, verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plot_path = RESULTS_DIR / "plots" / "pareto_front.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved Pareto front plot: {plot_path}")
plt.close()

print(f"\n📊 Pareto Analysis:")
print(f"   Total configurations: {len(df)}")
print(f"   Pareto optimal points: {len(pareto_points)}")
print(f"\n   Top 5 Pareto points by F1:")
for i, (_, row) in enumerate(pareto_points.nlargest(5, 'f1').iterrows(), 1):
    print(f"      {i}. {row['detector_set']}_min{int(row['min_activations'])} (n={int(row['ngram'])}): "
          f"P={row['precision']:.3f}, R={row['recall']:.3f}, F1={row['f1']:.3f}")

# ============================================================================
# PHASE 3: Performance Highlights
# ============================================================================
print("\n" + "="*80)
print("PHASE 3: PERFORMANCE HIGHLIGHTS")
print("="*80)

# ============================================================================
# PHASE 3: Performance Highlights
# ============================================================================
print("\n" + "="*80)
print("PHASE 3: PERFORMANCE HIGHLIGHTS")
print("="*80)

print(f"\n📊 Dataset: {len(df)} configurations tested")
print(f"   Detector sets: {df['detector_set'].nunique()}")
print(f"   N-gram sizes: {sorted(df['ngram'].unique())}")
print(f"   R values tested: {sorted(df['r_contiguous'].unique())}")

# Best by each metric
best_f1 = df.loc[df['f1'].idxmax()]
print(f"\n🥇 BEST F1: {best_f1['f1']:.4f}")
print(f"   Config: {best_f1['detector_set']}, min_act={int(best_f1['min_activations'])}, n-gram={int(best_f1['ngram'])}")
print(f"   P={best_f1['precision']:.4f}, R={best_f1['recall']:.4f}, FPR={best_f1['fpr']:.4f}")
if not pd.isna(best_f1.get('roc_auc')):
    print(f"   ROC-AUC={best_f1['roc_auc']:.4f}, PR-AUC={best_f1['pr_auc']:.4f}")

best_p = df.loc[df['precision'].idxmax()]
print(f"\n🥇 BEST PRECISION: {best_p['precision']:.4f}")
print(f"   Config: {best_p['detector_set']}, min_act={int(best_p['min_activations'])}, n-gram={int(best_p['ngram'])}")
print(f"   R={best_p['recall']:.4f}, F1={best_p['f1']:.4f}, FPR={best_p['fpr']:.4f}")

best_r = df.loc[df['recall'].idxmax()]
print(f"\n🥇 BEST RECALL: {best_r['recall']:.4f}")
print(f"   Config: {best_r['detector_set']}, min_act={int(best_r['min_activations'])}, n-gram={int(best_r['ngram'])}")
print(f"   P={best_r['precision']:.4f}, F1={best_r['f1']:.4f}, FPR={best_r['fpr']:.4f}")

# Best by N-gram size
print(f"\n📊 BEST RESULTS BY N-GRAM SIZE:")
for ngram in sorted(df['ngram'].unique()):
    subset = df[df['ngram'] == ngram]
    best = subset.loc[subset['f1'].idxmax()]
    print(f"\n   N-gram={int(ngram)} ({len(subset)} configs):")
    print(f"      Best F1: {best['f1']:.4f} (P={best['precision']:.4f}, R={best['recall']:.4f}, FPR={best['fpr']:.4f})")
    print(f"      Config: {best['detector_set']}_min{int(best['min_activations'])}")
    if not pd.isna(best.get('roc_auc')):
        print(f"      ROC-AUC: {best['roc_auc']:.4f}, PR-AUC: {best['pr_auc']:.4f}")

# Target achievement
print(f"\n🎯 TARGET ANALYSIS (P≥0.80, R≥0.40):")
target = df[(df['precision'] >= 0.80) & (df['recall'] >= 0.40)]
if len(target) > 0:
    print(f"   ✅ {len(target)} configs achieved target!")
    print(f"\n   Top 5:")
    for _, row in target.nlargest(5, 'f1').iterrows():
        print(f"      {row['detector_set']}_min{int(row['min_activations'])} (n={int(row['ngram'])}): "
              f"P={row['precision']:.3f}, R={row['recall']:.3f}, F1={row['f1']:.3f}")
else:
    print(f"   ❌ No configs achieved target")
    print(f"\n   Closest (P≥0.45, R≥0.70):")
    close = df[(df['precision'] >= 0.45) & (df['recall'] >= 0.70)]
    if len(close) > 0:
        for _, row in close.nlargest(10, 'f1').iterrows():
            print(f"      {row['detector_set']}_min{int(row['min_activations'])} (n={int(row['ngram'])}): "
                  f"P={row['precision']:.3f}, R={row['recall']:.3f}, F1={row['f1']:.3f}")

# Top 10 F1
print(f"\n📊 TOP 10 BY F1:")
for i, (_, row) in enumerate(df.nlargest(10, 'f1').iterrows(), 1):
    print(f"   {i:2d}. {row['detector_set']}_min{int(row['min_activations'])} (n={int(row['ngram'])}): "
          f"P={row['precision']:.3f}, R={row['recall']:.3f}, F1={row['f1']:.3f}, FPR={row['fpr']:.3f}")

# Performance by R value
print(f"\n📊 PERFORMANCE BY R VALUE:")
for r in sorted(df['r_contiguous'].unique()):
    r_data = df[df['r_contiguous'] == r]
    best = r_data.loc[r_data['f1'].idxmax()]
    print(f"   r={int(r)}: Best F1={best['f1']:.3f} (P={best['precision']:.3f}, R={best['recall']:.3f}) "
          f"[{best['detector_set']}_min{int(best['min_activations'])}, n={int(best['ngram'])}]")

# Low FPR configs
print(f"\n📊 LOW FALSE POSITIVE RATE (FPR<0.10, R>0.60):")
low_fpr = df[(df['fpr'] < 0.10) & (df['recall'] > 0.60)]
if len(low_fpr) > 0:
    print(f"   Found {len(low_fpr)} configs:")
    for _, row in low_fpr.nlargest(5, 'f1').iterrows():
        print(f"      {row['detector_set']}_min{int(row['min_activations'])} (n={int(row['ngram'])}): "
              f"P={row['precision']:.3f}, R={row['recall']:.3f}, F1={row['f1']:.3f}, FPR={row['fpr']:.3f}")
else:
    print(f"   None found - relaxing to FPR<0.15:")
    low_fpr = df[(df['fpr'] < 0.15) & (df['recall'] > 0.60)]
    for _, row in low_fpr.nlargest(5, 'f1').iterrows():
        print(f"      {row['detector_set']}_min{int(row['min_activations'])} (n={int(row['ngram'])}): "
              f"P={row['precision']:.3f}, R={row['recall']:.3f}, F1={row['f1']:.3f}, FPR={row['fpr']:.3f}")

# Best ROC-AUC and PR-AUC
if 'roc_auc' in df.columns and not df['roc_auc'].isna().all():
    print(f"\n📊 BEST AUC SCORES:")
    best_roc = df.loc[df['roc_auc'].idxmax()]
    print(f"   Best ROC-AUC: {best_roc['roc_auc']:.4f} [{best_roc['detector_set']}, n={int(best_roc['ngram'])}]")
    best_pr = df.loc[df['pr_auc'].idxmax()]
    print(f"   Best PR-AUC: {best_pr['pr_auc']:.4f} [{best_pr['detector_set']}, n={int(best_pr['ngram'])}]")

print(f"\n" + "="*80)
print(f"✓ Full results: {results_file}")
print(f"✓ Plots: {RESULTS_DIR / 'plots'}")
print(f"   - Pareto front: pareto_front.png")
print(f"   - Detector coverage: detector_coverage_curve.png")
print(f"   - Precision-Recall: precision_recall_curve.png")
print("="*80)

