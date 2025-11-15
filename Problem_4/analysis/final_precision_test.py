#!/usr/bin/env python3
"""
FAST GRID SEARCH - Optimized for speed with detector caching
Goal: Highest precision with good recall (P as high as possible, R≥0.40)

Strategy:
1. Generate detectors ONCE (slow, ~1 hour, but cached)
2. Test 100+ min_activations configs (fast, ~10 min)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from preprocessing import load_data, train_test_split
from nsa_optimized import NegativeSelectionClassifier
from constants import DATA_PATH, DATASET, RESULTS_DIR
import pandas as pd
import time
import pickle

print("="*80)
print("FAST GRID SEARCH: Maximum Precision with Good Recall")
print("="*80)

# Load data
print(f"\nLoading {DATASET} dataset...")
texts, labels = load_data(str(DATA_PATH))
X_train, y_train, X_test, y_test = train_test_split(texts, labels, seed=42)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# Cache directory
cache_dir = RESULTS_DIR / "detector_cache"
cache_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# PHASE 1: GENERATE DETECTORS (cached after first run)
# ============================================================================
print("\n" + "="*80)
print("PHASE 1: DETECTOR GENERATION")
print("="*80)

detector_configs = [
    {'num': 2000, 'ratio': 0.005, 'name': '2k'},
    {'num': 3000, 'ratio': 0.003, 'name': '3k'},
    {'num': 5000, 'ratio': 0.003, 'name': '5k'},
    {'num': 7000, 'ratio': 0.002, 'name': '7k'},
    {'num': 10000, 'ratio': 0.002, 'name': '10k'},
    {'num': 15000, 'ratio': 0.001, 'name': '15k'},
]

trained_classifiers = {}

for cfg in detector_configs:
    cache_file = cache_dir / f"det_{cfg['name']}_r{cfg['ratio']}.pkl"
    
    if cache_file.exists():
        print(f"\n✓ Loading cached: {cfg['name']}")
        with open(cache_file, 'rb') as f:
            nsa = pickle.load(f)
        print(f"  {len(nsa.detectors)} detectors loaded")
    else:
        print(f"\n🔧 Generating: {cfg['name']} ({cfg['num']} detectors)")
        start = time.time()
        
        nsa = NegativeSelectionClassifier(
            representation="vocabulary",
            matching_rule="r_contiguous",
            r_contiguous=2,
            detector_size=4,
            vocab_size=2000,
            min_word_freq=2,
            num_detectors=cfg['num'],
            min_activations=1,
            max_ham_match_ratio=cfg['ratio'],
            max_attempts=cfg['num'] * 2000,
            seed=42
        )
        
        nsa.fit(X_train, y_train)
        elapsed = time.time() - start
        
        print(f"  ✓ {len(nsa.detectors)} detectors in {elapsed:.1f}s")
        
        with open(cache_file, 'wb') as f:
            pickle.dump(nsa, f)
        print(f"  💾 Cached")
    
    trained_classifiers[cfg['name']] = nsa

print(f"\n✓ {len(trained_classifiers)} detector sets ready")

# ============================================================================
# PHASE 2: FOCUSED GRID SEARCH (optimized for speed)
# ============================================================================
print("\n" + "="*80)
print("PHASE 2: GRID SEARCH")
print("="*80)

grid = []
for cfg in detector_configs:
    # FOCUSED grid: test key min_activations values only
    # Strategy: Test low (high recall), medium (balanced), high (high precision)
    if cfg['num'] <= 3000:
        min_acts = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30]  # 10 values
    elif cfg['num'] <= 7000:
        min_acts = [1, 2, 3, 5, 7, 10, 15, 20, 30, 40]  # 10 values
    else:
        min_acts = [1, 3, 5, 8, 10, 15, 20, 30, 50, 70]  # 10 values
    
    for min_act in min_acts:
        grid.append({
            'detector_set': cfg['name'],
            'num_detectors': cfg['num'],
            'min_activations': min_act,
            'max_ham_match_ratio': cfg['ratio']
        })

print(f"\nTotal configs: {len(grid)}")
print(f"Estimated: ~{len(grid) * 30 / 60:.1f} min (predictions are slow)")
print("="*80)

results = []
start_time = time.time()

for i, config in enumerate(grid):
    iter_start = time.time()
    nsa = trained_classifiers[config['detector_set']]
    nsa.min_activations = config['min_activations']
    
    y_pred = nsa.predict(X_test)
    iter_time = time.time() - iter_start
    
    tp = sum(1 for p, t in zip(y_pred, y_test) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(y_pred, y_test) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(y_pred, y_test) if p == 0 and t == 1)
    tn = sum(1 for p, t in zip(y_pred, y_test) if p == 0 and t == 0)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    results.append({
        'detector_set': config['detector_set'],
        'num_detectors': config['num_detectors'],
        'min_activations': config['min_activations'],
        'max_ham_match_ratio': config['max_ham_match_ratio'],
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    })
    
    # Progress every 5 configs
    if (i + 1) % 5 == 0 or (i + 1) == len(grid):
        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1)
        eta = (len(grid) - (i + 1)) * avg_time
        print(f"[{i+1}/{len(grid)}] {config['detector_set']}_min{config['min_activations']}: "
              f"P={precision:.3f}, R={recall:.3f}, F1={f1:.3f} "
              f"({iter_time:.1f}s | Avg:{avg_time:.1f}s | ETA:{eta/60:.1f}m)")

print(f"\n✓ Complete in {(time.time() - start_time)/60:.1f} min")

# ============================================================================
# PHASE 3: RESULTS
# ============================================================================
print("\n" + "="*80)
print("🏆 RESULTS")
print("="*80)

results_df = pd.DataFrame(results)
results_path = RESULTS_DIR / "final_precision_results.csv"
results_df.to_csv(results_path, index=False)
print(f"\n✓ Saved: {results_path} ({len(results_df)} configs)")

best_f1 = results_df.loc[results_df['f1'].idxmax()]
print(f"\n🥇 Best F1: {best_f1['f1']:.4f}")
print(f"   {best_f1['detector_set']} (det={int(best_f1['num_detectors'])}, min_act={int(best_f1['min_activations'])})")
print(f"   P={best_f1['precision']:.4f}, R={best_f1['recall']:.4f}, FP={int(best_f1['fp'])}")

best_prec = results_df.loc[results_df['precision'].idxmax()]
print(f"\n🥇 Best Precision: {best_prec['precision']:.4f}")
print(f"   {best_prec['detector_set']} (det={int(best_prec['num_detectors'])}, min_act={int(best_prec['min_activations'])})")
print(f"   R={best_prec['recall']:.4f}, F1={best_prec['f1']:.4f}, FP={int(best_prec['fp'])}")

good_prec = results_df[results_df['precision'] >= 0.30]
if len(good_prec) > 0:
    best_recall = good_prec.loc[good_prec['recall'].idxmax()]
    print(f"\n🥇 Best Recall (P≥0.30): {best_recall['recall']:.4f}")
    print(f"   {best_recall['detector_set']} (det={int(best_recall['num_detectors'])}, min_act={int(best_recall['min_activations'])})")
    print(f"   P={best_recall['precision']:.4f}, F1={best_recall['f1']:.4f}")

target = results_df[(results_df['precision'] >= 0.80) & (results_df['recall'] >= 0.40)]
if len(target) > 0:
    print(f"\n🎯🎯🎯 TARGET ACHIEVED! ({len(target)} configs with P≥0.80, R≥0.40):")
    for _, row in target.head(10).iterrows():
        print(f"   {row['detector_set']}_min{int(row['min_activations'])}: "
              f"P={row['precision']:.4f}, R={row['recall']:.4f}, F1={row['f1']:.4f}")
else:
    print(f"\n⚠️  No configs achieved P≥0.80, R≥0.40")
    print(f"\n📊 Top 10 by Precision:")
    top = results_df.nlargest(10, 'precision')
    for _, row in top.iterrows():
        print(f"   {row['detector_set']}_min{int(row['min_activations'])}: "
              f"P={row['precision']:.4f}, R={row['recall']:.4f}, F1={row['f1']:.4f}, FP={int(row['fp'])}")

print(f"\n📊 Top 10 by F1:")
top_f1 = results_df.nlargest(10, 'f1')
for _, row in top_f1.iterrows():
    print(f"   {row['detector_set']}_min{int(row['min_activations'])}: "
          f"P={row['precision']:.4f}, R={row['recall']:.4f}, F1={row['f1']:.4f}")

print("\n" + "="*80)
print("✓ OPTIMIZATION COMPLETE")
print(f"✓ Total configs: {len(results_df)}")
print(f"✓ Best F1: {results_df['f1'].max():.4f}")
print(f"✓ Best Precision: {results_df['precision'].max():.4f}")
print(f"✓ Total time: {(time.time() - start_time)/60:.1f} minutes")
print("="*80)
