#!/usr/bin/env python3
"""
Explore match-rule hyperparameters (r_contiguous) using cached detectors.

Usage:
  - Quick smoke test: python explore_match_rules.py --mode smoke
  - Full run: python explore_match_rules.py --mode full

Smoke mode: runs on a single detector set (2k) and tests r_contiguous in [1,2,3]
Full mode: runs on all cached detector sets and tests r_contiguous in [1,2,3,4]

This script only performs prediction (no detector regeneration) and writes results to results/explore_match_rules_results.csv
"""

import argparse
from pathlib import Path
import pickle
import time
import pandas as pd
import numpy as np
import sys

sys.path.append(str(Path(__file__).parent.parent / 'src'))
from preprocessing import load_data, train_test_split

PROBLEM_DIR = Path(__file__).parent.parent  # /.../Problem_4
RESULTS_DIR = PROBLEM_DIR / 'results'
CACHE_DIR = RESULTS_DIR / 'detector_cache'
OUT_FILE = RESULTS_DIR / 'explore_match_rules_results.csv'


def load_cached_detectors():
    pkls = list(CACHE_DIR.glob('det_*.pkl'))
    detectors = {}
    for p in pkls:
        name = p.stem.replace('det_', '').split('_')[0]
        with open(p, 'rb') as f:
            try:
                nsa = pickle.load(f)
                detectors[name] = nsa
            except Exception as e:
                print(f"Failed to load {p}: {e}")
    return detectors


def evaluate(nsa, X_test, y_test, min_acts):
    res = []
    for min_act in min_acts:
        nsa.min_activations = int(min_act)
        y_pred = nsa.predict(X_test)
        tp = sum(1 for p, t in zip(y_pred, y_test) if p == 1 and t == 1)
        fp = sum(1 for p, t in zip(y_pred, y_test) if p == 1 and t == 0)
        fn = sum(1 for p, t in zip(y_pred, y_test) if p == 0 and t == 1)
        tn = sum(1 for p, t in zip(y_pred, y_test) if p == 0 and t == 0)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        res.append({'min_activations': min_act, 'precision': precision, 'recall': recall, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn})
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['smoke', 'full'], default='smoke')
    parser.add_argument('--use-best', action='store_true', help='Auto-select detector set and base min_activations from results/final_precision_results.csv (best F1)')
    parser.add_argument('--detector-set', type=str, default=None, help='Name of cached detector set to evaluate (e.g. 15k). Overrides --use-best if provided')
    args = parser.parse_args()

    print('Loading dataset...')
    # Use configured DATA_PATH from constants so the selected dataset (e.g. enron1) is loaded
    from constants import DATA_PATH
    texts, labels = load_data(str(DATA_PATH))
    X_train, y_train, X_test, y_test = train_test_split(texts, labels, seed=42)
    print(f'Train: {len(X_train)}, Test: {len(X_test)}')

    detectors = load_cached_detectors()
    if not detectors:
        print('No cached detectors found in', CACHE_DIR)
        return

    print('Found cached detector sets:', list(detectors.keys()))

    # Allow auto-selection of the best config from previous runs
    base_min_acts = None
    if args.use_best:
        best_csv = RESULTS_DIR / 'final_precision_results.csv'
        if best_csv.exists():
            try:
                df_best = pd.read_csv(best_csv)
                best_row = df_best.loc[df_best['f1'].idxmax()]
                best_set = str(best_row['detector_set'])
                best_min = int(best_row['min_activations'])
                print(f"Auto-selected best config from {best_csv}: {best_set} (min_activations={best_min})")
                base_min_acts = best_min
                # If detector-set not explicitly passed, use the best one
                if args.detector_set is None:
                    args.detector_set = best_set
            except Exception as e:
                print('Failed to read best config:', e)
        else:
            print('No final_precision_results.csv found for --use-best; continuing with defaults')

    if args.mode == 'smoke':
        # Default smoke should focus on the best performing detector set (if available)
        if args.detector_set:
            target_sets = [args.detector_set] if args.detector_set in detectors else ([ '15k' ] if '15k' in detectors else list(detectors.keys())[:1])
        else:
            target_sets = ['15k'] if '15k' in detectors else (['2k'] if '2k' in detectors else list(detectors.keys())[:1])
        r_values = [1, 2, 3]
        min_acts_map = None
    else:
        target_sets = sorted(detectors.keys())
        r_values = [1,2,3,4]
        min_acts_map = None

    results = []
    start = time.time()

    for ds in target_sets:
        nsa = detectors[ds]
        print(f"\nEvaluating detector set: {ds} ({len(nsa.detectors)} detectors)")
        # Ensure we start from a clean baseline matching rule
        nsa.matching_rule = 'r_contiguous'
        # If the cached classifier has attributes that can be tuned at predict time, reset them
        nsa.min_activations = int(base_min_acts) if base_min_acts is not None else int(nsa.min_activations)
        for r in r_values:
            # Override r_contiguous
            print(f"  r_contiguous={r}")
            nsa.r_contiguous = int(r)
            # Also ensure matching rule set to r_contiguous
            nsa.matching_rule = 'r_contiguous'
            # Choose min_acts
            if min_acts_map and ds in min_acts_map:
                min_acts = min_acts_map[ds]
            else:
                # If we auto-selected a best min_activations, create a focused grid around it
                if base_min_acts is not None:
                    # include small values plus the best and a few larger thresholds
                    candidates = sorted(list({1,2,3,5,10,20,30,50,70, base_min_acts}))
                    # Place the best value first for convenience
                    if base_min_acts in candidates:
                        candidates.remove(base_min_acts)
                        candidates = [base_min_acts] + candidates
                    min_acts = candidates
                else:
                    min_acts = [1,2,3,5,10,20,30]

            t0 = time.time()
            rows = evaluate(nsa, X_test, y_test, min_acts)
            t1 = time.time()
            for row in rows:
                results.append({'detector_set': ds, 'num_detectors': len(nsa.detectors), 'r_contiguous': r, **row})
            print(f"    done (took {t1-t0:.1f}s for {len(min_acts)} min_acts)")

    df = pd.DataFrame(results)
    df.to_csv(OUT_FILE, index=False)
    print(f"\nSaved: {OUT_FILE} ({len(df)} rows) in {(time.time()-start)/60:.1f} minutes")

if __name__ == '__main__':
    main()
