#!/usr/bin/env python3
"""
THIS IS THE MAIN FILE TO RUN THE ALGORITHM
COMPREHENSIVE BIT-BASED NSA GRID SEARCH with Full Metrics and Plots
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from preprocessing import load_data, train_test_split
from nsa_optimized import NegativeSelectionClassifier
from constants import DATA_PATH, RESULTS_DIR, BIT_LENGTH, CHAR_NGRAM_SIZE, R_CONTIGUOUS

import pandas as pd
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

print("="*80)
print("COMPREHENSIVE BIT-BASED NSA GRID SEARCH")
print("Full Metrics, Plots, and N-gram Analysis")
print("="*80)

# Load SMS dataset
print(f"\nLoading SMS dataset...")
texts, labels = load_data(str(DATA_PATH))
X_train, y_train, X_test, y_test = train_test_split(texts, labels, seed=42)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")
print(f"Test set: {sum(y_test)} spam, {len(y_test) - sum(y_test)} ham")

# Cache directory
cache_dir = RESULTS_DIR / "detector_cache_bits"
cache_dir.mkdir(parents=True, exist_ok=True)

# Plot directory
plot_dir = RESULTS_DIR / "plots"
plot_dir.mkdir(parents=True, exist_ok=True)

# Check for existing results to avoid re-running
existing_results_file = RESULTS_DIR / "comprehensive_results.csv"
existing_configs = set()
if existing_results_file.exists():
    print(f"\n✓ Found existing results: {existing_results_file}")
    existing_df = pd.read_csv(existing_results_file)
    # Create unique keys for already-tested configs
    for _, row in existing_df.iterrows():
        # Extract ngram from detector_set name (e.g., "1k_r6_n4" or "1k_r6")
        det_name = row['detector_set']
        if '_n' in det_name:
            ngram = int(det_name.split('_n')[1])
        else:
            ngram = 4  # Default from first run
        key = (row['num_detectors'], row['r_contiguous'], ngram, row['min_activations'])
        existing_configs.add(key)
    print(f"  {len(existing_configs)} configurations already tested (will skip)")
else:
    print(f"\n⚠️  No existing results found - will test all configurations")

# PHASE 1: ENHANCED DETECTOR GENERATION (WITH N-GRAM VARIATIONS)
print("\n" + "="*80)
print("PHASE 1: DETECTOR GENERATION (MULTI-NGRAM CONFIGURATIONS)")
print("="*80)
print(f"Configuration:")
print(f"  Bit vector length: {BIT_LENGTH}")
print(f"  Testing character n-gram sizes: 3, 4, 5")

# Test different n-gram sizes with best-performing r values from previous run
# Based on results: r=7, r=8, r=9 performed best
ngram_sizes = [3, 4, 5]  # Test shorter, current, and longer n-grams

detector_configs = []

# For each n-gram size, test best-performing configurations
for ngram in ngram_sizes:
    if ngram == 4:
        # n-gram=4: Already tested in first run, but include full range for comparison
        # Skip r=6 (underperformed), focus on r=7,8,9
        configs = [
            {'num': 1000, 'r': 6, 'ratio': 0.05, 'ngram': 4, 'name': '1k_r6_n4'},
            {'num': 1000, 'r': 7, 'ratio': 0.05, 'ngram': 4, 'name': '1k_r7_n4'},
            {'num': 2000, 'r': 7, 'ratio': 0.05, 'ngram': 4, 'name': '2k_r7_n4'},
            {'num': 3000, 'r': 7, 'ratio': 0.05, 'ngram': 4, 'name': '3k_r7_n4'},
            {'num': 1000, 'r': 8, 'ratio': 0.05, 'ngram': 4, 'name': '1k_r8_n4'},
            {'num': 2000, 'r': 8, 'ratio': 0.05, 'ngram': 4, 'name': '2k_r8_n4'},
            {'num': 3000, 'r': 8, 'ratio': 0.05, 'ngram': 4, 'name': '3k_r8_n4'},
            {'num': 5000, 'r': 8, 'ratio': 0.05, 'ngram': 4, 'name': '5k_r8_n4'},
            {'num': 2000, 'r': 9, 'ratio': 0.05, 'ngram': 4, 'name': '2k_r9_n4'},
            {'num': 3000, 'r': 9, 'ratio': 0.05, 'ngram': 4, 'name': '3k_r9_n4'},
            {'num': 2000, 'r': 10, 'ratio': 0.05, 'ngram': 4, 'name': '2k_r10_n4'},
            {'num': 3000, 'r': 10, 'ratio': 0.05, 'ngram': 4, 'name': '3k_r10_n4'},
        ]
    elif ngram == 3:
        # n-gram=3 (NEW): Shorter patterns, expect higher recall, lower precision
        # Test with best r values (7, 8, 9)
        configs = [
            {'num': 1000, 'r': 7, 'ratio': 0.05, 'ngram': 3, 'name': '1k_r7_n3'},
            {'num': 2000, 'r': 7, 'ratio': 0.05, 'ngram': 3, 'name': '2k_r7_n3'},
            {'num': 2000, 'r': 8, 'ratio': 0.05, 'ngram': 3, 'name': '2k_r8_n3'},
            {'num': 3000, 'r': 8, 'ratio': 0.05, 'ngram': 3, 'name': '3k_r8_n3'},
            {'num': 2000, 'r': 9, 'ratio': 0.05, 'ngram': 3, 'name': '2k_r9_n3'},
        ]
    else:  # ngram == 5
        # Test with best r values (7, 8, 9)
        configs = [
            {'num': 1000, 'r': 7, 'ratio': 0.05, 'ngram': 5, 'name': '1k_r7_n5'},
            {'num': 2000, 'r': 7, 'ratio': 0.05, 'ngram': 5, 'name': '2k_r7_n5'},
            {'num': 2000, 'r': 8, 'ratio': 0.05, 'ngram': 5, 'name': '2k_r8_n5'},
            {'num': 3000, 'r': 8, 'ratio': 0.05, 'ngram': 5, 'name': '3k_r8_n5'},
            {'num': 2000, 'r': 9, 'ratio': 0.05, 'ngram': 5, 'name': '2k_r9_n5'},
        ]
    
    detector_configs.extend(configs)

trained_classifiers = {}

for cfg in detector_configs:
    cache_file = cache_dir / f"det_{cfg['name']}.pkl"
    
    if cache_file.exists():
        print(f"\n✓ Loading cached: {cfg['name']}")
        with open(cache_file, 'rb') as f:
            nsa = pickle.load(f)
        print(f"  {len(nsa.detectors)} detectors loaded (r={cfg['r']}, ngram={cfg['ngram']})")
    else:
        print(f"\n🔧 Generating: {cfg['name']} ({cfg['num']} detectors, r={cfg['r']}, ngram={cfg['ngram']})")
        start = time.time()
        
        nsa = NegativeSelectionClassifier(
            representation="binary",
            matching_rule="r_contiguous",
            feature_length=BIT_LENGTH,
            char_ngram_size=cfg['ngram'],  # Use n-gram from config
            r_contiguous=cfg['r'],
            num_detectors=cfg['num'],
            min_activations=1,
            max_ham_match_ratio=cfg['ratio'],
            seed=42
        )
        
        nsa.fit(X_train, y_train)
        elapsed = time.time() - start
        
        print(f"  ✓ {len(nsa.detectors)} detectors in {elapsed:.1f}s")
        
        with open(cache_file, 'wb') as f:
            pickle.dump(nsa, f)
        print(f"  💾 Cached to {cache_file.name}")
    
    trained_classifiers[cfg['name']] = nsa

print(f"\n✓ {len(trained_classifiers)} detector sets ready")


# PHASE 2: COMPREHENSIVE GRID SEARCH
print("\n" + "="*80)
print("PHASE 2: GRID SEARCH (EXTENDED THRESHOLDS)")
print("="*80)

# Extended min_activations ranges based on results
# r=6: Best at low thresholds (1-10)
# r=7-8: Extended range to find precision sweet spot (1-60)
# r=9-10: Full range (1-100)
grid = []
for cfg in detector_configs:
    if cfg['r'] == 6:
        min_acts = [1, 2, 3, 5, 7, 10, 15, 20]
    elif cfg['r'] in [7, 8]:
        # These showed good results at high thresholds - extend further
        if cfg['num'] <= 2000:
            min_acts = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60]
        else:
            min_acts = [1, 3, 5, 10, 15, 20, 30, 40, 50, 60, 80, 100]
    else:  # r=9, 10
        # r=10 showed precision plateauing around 30-50
        if cfg['num'] <= 2000:
            min_acts = [1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50, 60, 80]
        else:
            min_acts = [1, 3, 5, 10, 15, 20, 30, 40, 50, 75, 100, 150]
    
    for min_act in min_acts:
        # Check if this config was already tested
        config_tuple = (cfg['num'], cfg['r'], cfg['ngram'], min_act)
        if config_tuple in existing_configs:
            continue  # Skip this configuration
        
        grid.append({
            'detector_set': cfg['name'],
            'num_detectors': cfg['num'],
            'r_contiguous': cfg['r'],
            'ngram': cfg['ngram'],
            'min_activations': min_act,
            'max_ham_match_ratio': cfg['ratio']
        })

print(f"\nTotal configs to test: {len(grid)}")
print(f"Estimated runtime: ~{len(grid) * 10 / 60:.1f} min")
print("="*80)

results = []
predictions_for_curves = []  # Store predictions for ROC/PR curves
start_time = time.time()

for i, config in enumerate(grid):
    iter_start = time.time()
    nsa = trained_classifiers[config['detector_set']]
    nsa.min_activations = config['min_activations']
    nsa.r_contiguous = config['r_contiguous']
    
    y_pred = nsa.predict(X_test)
    iter_time = time.time() - iter_start
    
    # Calculate confusion matrix
    tp = sum(1 for p, t in zip(y_pred, y_test) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(y_pred, y_test) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(y_pred, y_test) if p == 0 and t == 1)
    tn = sum(1 for p, t in zip(y_pred, y_test) if p == 0 and t == 0)
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # False Positive Rate on ham
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # Store predictions for ROC/PR curve calculation
    predictions_for_curves.append({
        'config': f"{config['detector_set']}_min{config['min_activations']}",
        'y_true': y_test,
        'y_pred': y_pred,
        'detector_set': config['detector_set'],
        'r': config['r_contiguous'],
        'min_act': config['min_activations']
    })
    
    results.append({
        'detector_set': config['detector_set'],
        'num_detectors': config['num_detectors'],
        'r_contiguous': config['r_contiguous'],
        'ngram': config['ngram'],
        'min_activations': config['min_activations'],
        'max_ham_match_ratio': config['max_ham_match_ratio'],
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fpr': fpr,  # False positive rate
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    })
    
    # Progress
    elapsed = time.time() - start_time
    avg_time = elapsed / (i + 1)
    eta = (len(grid) - (i + 1)) * avg_time
    print(f"[{i+1}/{len(grid)}] {config['detector_set']}_min{config['min_activations']} (n={config['ngram']}): "
          f"P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, FPR={fpr:.3f} "
          f"({iter_time:.1f}s | ETA:{eta/60:.1f}m)")

print(f"\n✓ Complete in {(time.time() - start_time)/60:.1f} min")

# PHASE 3: ROC-AUC AND PR-AUC CALCULATION
print("\n" + "="*80)
print("PHASE 3: ROC-AUC AND PR-AUC CALCULATION")
print("="*80)

# For each detector set, calculate AUC using predictions at different thresholds
auc_results = []

for detector_name in trained_classifiers.keys():
    # Get all predictions for this detector set
    detector_preds = [p for p in predictions_for_curves if p['detector_set'] == detector_name]
    
    if len(detector_preds) == 0:
        continue
    
    # Create score array (using min_activations as inverse score - lower threshold = higher score)
    # We'll use recall as the "score" since higher min_act → lower recall
    y_true = detector_preds[0]['y_true']
    
    # Sort by min_activations
    detector_preds_sorted = sorted(detector_preds, key=lambda x: x['min_act'])
    
    # Calculate ROC-AUC and PR-AUC using the range of predictions
    try:
        # Use predictions from lowest threshold (most permissive) as scores
        y_pred_scores = detector_preds_sorted[0]['y_pred']
        
        if len(set(y_pred_scores)) > 1:  # Need at least 2 classes in predictions
            roc_auc = roc_auc_score(y_true, y_pred_scores)
            pr_auc = average_precision_score(y_true, y_pred_scores)
        else:
            roc_auc = 0.5  # Random classifier
            pr_auc = sum(y_true) / len(y_true)  # Baseline = proportion of positives
        
        auc_results.append({
            'detector_set': detector_name,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc
        })
        
        print(f"  {detector_name}: ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}")
    except Exception as e:
        print(f"  {detector_name}: Could not calculate AUC ({str(e)})")

# PHASE 4: DETECTOR COVERAGE CURVE
print("\n" + "="*80)
print("PHASE 4: DETECTOR COVERAGE CURVE (Recall vs. Num Detectors)")
print("="*80)

# For r=8 (best performer), show how recall changes with number of detectors
coverage_data = []

for r_val in [6, 7, 8, 9, 10]:
    r_configs = [cfg for cfg in detector_configs if cfg['r'] == r_val]
    
    for cfg in r_configs:
        # Get best recall for this detector set (lowest min_activations)
        cfg_results = [r for r in results if r['detector_set'] == cfg['name']]
        if cfg_results:
            best_recall = max([r['recall'] for r in cfg_results])
            coverage_data.append({
                'r': r_val,
                'num_detectors': cfg['num'],
                'best_recall': best_recall
            })

# Plot detector coverage curve
plt.figure(figsize=(10, 6))
for r_val in sorted(set([d['r'] for d in coverage_data])):
    r_data = [d for d in coverage_data if d['r'] == r_val]
    r_data_sorted = sorted(r_data, key=lambda x: x['num_detectors'])
    
    nums = [d['num_detectors'] for d in r_data_sorted]
    recalls = [d['best_recall'] for d in r_data_sorted]
    
    plt.plot(nums, recalls, marker='o', label=f'r={r_val}', linewidth=2)

plt.xlabel('Number of Detectors', fontsize=12)
plt.ylabel('Maximum Recall (Spam Detection Rate)', fontsize=12)
plt.title('Detector Coverage Curve: Recall vs. Number of Detectors', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

coverage_plot_path = plot_dir / "detector_coverage_curve.png"
plt.savefig(coverage_plot_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {coverage_plot_path}")
plt.close()

# PHASE 5: PRECISION-RECALL CURVES
print("\n" + "="*80)
print("PHASE 5: PRECISION-RECALL CURVES")
print("="*80)

# For each r value, plot PR curve across different min_activations
plt.figure(figsize=(12, 8))

for r_val in sorted(set([r['r_contiguous'] for r in results])):
    # Get results for this r value (use largest detector set for each r)
    r_results = [r for r in results if r['r_contiguous'] == r_val]
    
    # Group by detector set and pick the largest
    detector_sets = set([r['detector_set'] for r in r_results])
    largest_set = max(detector_sets, key=lambda s: int(s.split('_')[0].replace('k', '000')))
    
    set_results = [r for r in r_results if r['detector_set'] == largest_set]
    set_results_sorted = sorted(set_results, key=lambda x: -x['recall'])  # Sort by descending recall
    
    precisions = [r['precision'] for r in set_results_sorted]
    recalls = [r['recall'] for r in set_results_sorted]
    
    plt.plot(recalls, precisions, marker='o', label=f'{largest_set} (r={r_val})', linewidth=2, markersize=4)

plt.xlabel('Recall (Spam Detection Rate)', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve: Spam Classification Performance', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.tight_layout()

pr_curve_path = plot_dir / "precision_recall_curve.png"
plt.savefig(pr_curve_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {pr_curve_path}")
plt.close()

# PHASE 6: RESULTS AND ANALYSIS
print("\n" + "="*80)
print("🏆 COMPREHENSIVE RESULTS")
print("="*80)

results_df = pd.DataFrame(results)
results_path = RESULTS_DIR / "comprehensive_results.csv"

# Merge with existing results if available
if existing_results_file.exists() and len(existing_configs) > 0:
    print(f"\n📋 Merging with existing results...")
    existing_df = pd.read_csv(existing_results_file)
    # Combine old and new results
    results_df = pd.concat([existing_df, results_df], ignore_index=True)
    print(f"   Previous: {len(existing_df)} configs")
    print(f"   New: {len(results)} configs")
    print(f"   Combined: {len(results_df)} configs")

results_df.to_csv(results_path, index=False)
print(f"\n✓ Saved: {results_path} ({len(results_df)} configs)")

# Best F1
best_f1 = results_df.loc[results_df['f1'].idxmax()]
print(f"\n🥇 Best F1: {best_f1['f1']:.4f}")
print(f"   {best_f1['detector_set']} (det={int(best_f1['num_detectors'])}, "
      f"r={int(best_f1['r_contiguous'])}, min_act={int(best_f1['min_activations'])})")
print(f"   P={best_f1['precision']:.4f}, R={best_f1['recall']:.4f}, FPR={best_f1['fpr']:.4f}")

# Best Precision
best_prec = results_df.loc[results_df['precision'].idxmax()]
print(f"\n🥇 Best Precision: {best_prec['precision']:.4f}")
print(f"   {best_prec['detector_set']} (det={int(best_prec['num_detectors'])}, "
      f"r={int(best_prec['r_contiguous'])}, min_act={int(best_prec['min_activations'])})")
print(f"   R={best_prec['recall']:.4f}, F1={best_prec['f1']:.4f}, FPR={best_prec['fpr']:.4f}")

# Best Recall
best_rec = results_df.loc[results_df['recall'].idxmax()]
print(f"\n🥇 Best Recall: {best_rec['recall']:.4f}")
print(f"   {best_rec['detector_set']} (det={int(best_rec['num_detectors'])}, "
      f"r={int(best_rec['r_contiguous'])}, min_act={int(best_rec['min_activations'])})")
print(f"   P={best_rec['precision']:.4f}, F1={best_rec['f1']:.4f}, FPR={best_rec['fpr']:.4f}")

# Target: P≥0.80, R≥0.40
target = results_df[(results_df['precision'] >= 0.80) & (results_df['recall'] >= 0.40)]
if len(target) > 0:
    print(f"\n🎯 TARGET ACHIEVED! ({len(target)} configs with P≥0.80, R≥0.40):")
    for _, row in target.head(10).iterrows():
        print(f"   {row['detector_set']}_min{int(row['min_activations'])}: "
              f"P={row['precision']:.4f}, R={row['recall']:.4f}, F1={row['f1']:.4f}, FPR={row['fpr']:.4f}")
else:
    print(f"\n⚠️  No configs achieved P≥0.80, R≥0.40")
    # Show closest configs (high precision with reasonable recall)
    print(f"\n📊 Closest to target (P≥0.70, R≥0.30):")
    close = results_df[(results_df['precision'] >= 0.70) & (results_df['recall'] >= 0.30)]
    if len(close) > 0:
        for _, row in close.nlargest(10, 'f1').iterrows():
            print(f"   {row['detector_set']}_min{int(row['min_activations'])}: "
                  f"P={row['precision']:.4f}, R={row['recall']:.4f}, F1={row['f1']:.4f}, FPR={row['fpr']:.4f}")

# Top 10 by F1
print(f"\n📊 Top 10 by F1:")
top_f1 = results_df.nlargest(10, 'f1')
for _, row in top_f1.iterrows():
    print(f"   {row['detector_set']}_min{int(row['min_activations'])}: "
          f"P={row['precision']:.4f}, R={row['recall']:.4f}, F1={row['f1']:.4f}, FPR={row['fpr']:.4f}")

# ROC-AUC and PR-AUC Summary
if auc_results:
    print(f"\n📊 ROC-AUC and PR-AUC by Detector Set:")
    for auc_res in sorted(auc_results, key=lambda x: -x['pr_auc']):
        print(f"   {auc_res['detector_set']}: ROC-AUC={auc_res['roc_auc']:.4f}, PR-AUC={auc_res['pr_auc']:.4f}")
    
    # Save AUC results
    auc_df = pd.DataFrame(auc_results)
    auc_path = RESULTS_DIR / "auc_results.csv"
    auc_df.to_csv(auc_path, index=False)
    print(f"\n✓ Saved AUC results: {auc_path}")

print("\n" + "="*80)
print("✓ COMPREHENSIVE ANALYSIS COMPLETE")
print(f"✓ Total configs: {len(results_df)}")
print(f"✓ Best F1: {results_df['f1'].max():.4f}")
print(f"✓ Best Precision: {results_df['precision'].max():.4f}")
print(f"✓ Best Recall: {results_df['recall'].max():.4f}")
print(f"✓ Total time: {(time.time() - start_time)/60:.1f} minutes")
print("="*80)
print(f"\n📁 Results saved to:")
print(f"   {results_path}")
print(f"   {coverage_plot_path}")
print(f"   {pr_curve_path}")
print(f"   Detector cache: {cache_dir}")
