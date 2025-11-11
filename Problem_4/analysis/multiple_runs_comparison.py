#!/usr/bin/env python3
"""
Multiple runs optimization comparison for Vocabulary NSA spam detection with 7 strategies.
Runs the optimization comparison 5 times to calculate average performance and confidence intervals.
Ge    # Get activation scores for each sample (count how many detectors match)
    # Use NSA's own tokenizer to ensure consistency
    test_scores = []
    for text in test_texts:
        tokens = nsa._text_to_tokens(text)  # Use NSA's tokenizer
        activations = 0
        
        if len(tokens) >= nsa.detector_size:
            for i in range(len(tokens) - nsa.detector_size + 1):
                pattern = tuple(tokens[i:i + nsa.detector_size])
                for detector in nsa.detectors:
                    if nsa._matches_pattern(detector, pattern):
                        activations += 1
        
        test_scores.append(activations)
    test_scores = np.array(test_scores)
    
    # Debug: Print score statistics
    spam_scores = test_scores[np.array(test_labels) == 1]
    ham_scores = test_scores[np.array(test_labels) == 0]
    print(f"    Activation scores - Spam: {spam_scores.mean():.2f} ± {spam_scores.std():.2f}, Ham: {ham_scores.mean():.2f} ± {ham_scores.std():.2f}")nsive plots and tables including Pareto front, ROC/PR curves, and detector statistics.
"""

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from typing import Dict, List, Tuple
from sklearn.metrics import roc_curve, precision_recall_curve, auc, roc_auc_score
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Add the src directory to path
script_dir = Path(__file__).resolve().parent
src_dir = script_dir.parent / 'src'
sys.path.insert(0, str(src_dir))

# Import our modules
from preprocessing import load_data, train_val_test_split
from nsa_optimized import NegativeSelectionClassifier
import constants


def run_parameter_grid_search_single_run(train_texts, train_labels, val_texts, val_labels, run_id):
    """Run parameter grid search for both r-contiguous and Hamming NSA for one run."""
    print(f"  Run {run_id}: Running grid search...")
    
    # Balanced parameter grids (medium speed, some variety)
    import itertools
    param_grids = {
        "r_contiguous": {
            'r_contiguous': [3],
            'detector_size': [4],
            'num_detectors': [700, 900],  # 2 options
            'vocab_size': [1200],
            'min_word_freq': [3],
            'max_ham_match_ratio': [0.05],
            'min_activations': [1],
            'matching_rule': ['r_contiguous']
        },
        "hamming": {
            'hamming_threshold': [1],
            'detector_size': [5],
            'num_detectors': [3500, 4000],  # 2 options
            'vocab_size': [700],
            'min_word_freq': [3],
            'max_ham_match_ratio': [0.05],
            'min_activations': [1],
            'matching_rule': ['hamming']
        }
    }
    
    results = []
    
    for rule_name, param_grid in param_grids.items():
        param_combinations = list(itertools.product(*param_grid.values()))
        param_names = list(param_grid.keys())
        
        for params in param_combinations:
            param_dict = dict(zip(param_names, params))
            
            try:
                nsa = NegativeSelectionClassifier(
                    representation="vocabulary",
                    max_attempts=5000,
                    **param_dict
                )
                
                nsa.fit(train_texts, train_labels)
                
                if len(nsa.detectors) == 0:
                    continue
                
                val_pred = nsa.predict(val_texts)
                
                # Calculate metrics
                tp = sum(1 for p, t in zip(val_pred, val_labels) if p == 1 and t == 1)
                fp = sum(1 for p, t in zip(val_pred, val_labels) if p == 1 and t == 0)
                fn = sum(1 for p, t in zip(val_pred, val_labels) if p == 0 and t == 1)
                tn = sum(1 for p, t in zip(val_pred, val_labels) if p == 0 and t == 0)
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                accuracy = (tp + tn) / len(val_labels)
                
                # Store results
                result = {
                    **param_dict,
                    'detectors_generated': len(nsa.detectors),
                    'val_accuracy': accuracy,
                    'val_precision': precision,
                    'val_recall': recall,
                    'val_f1': f1,
                }
                results.append(result)
            
            except Exception as e:
                continue
    
    return pd.DataFrame(results)


def find_optimal_parameters_seven_strategies(results_df):
    """Find optimal parameters for all seven strategies."""
    if len(results_df) == 0:
        return {}
    
    strategies = {}
    
    # 1. F1-Optimized
    best_f1 = results_df.loc[results_df['val_f1'].idxmax()]
    strategies['F1-Optimized'] = {
        'params': best_f1.to_dict(),
        'metric_focus': 'F1 Score',
        'weights': 'N/A (harmonic mean)'
    }
    
    # 2. Precision-Optimized (99% P, 1% R)
    results_df['precision_score'] = results_df['val_precision'] * 0.99 + results_df['val_recall'] * 0.01
    best_precision = results_df.loc[results_df['precision_score'].idxmax()]
    strategies['Precision-Optimized'] = {
        'params': best_precision.to_dict(),
        'metric_focus': 'Precision',
        'weights': 'P=99%, R=1%'
    }
    
    # 3. Recall-Optimized (99% R, 1% P)
    results_df['recall_score'] = results_df['val_recall'] * 0.99 + results_df['val_precision'] * 0.01
    best_recall = results_df.loc[results_df['recall_score'].idxmax()]
    strategies['Recall-Optimized'] = {
        'params': best_recall.to_dict(),
        'metric_focus': 'Recall',
        'weights': 'R=99%, P=1%'
    }
    
    # 4. Precision-Weighted (75% P, 25% R)
    results_df['precision_weighted_score'] = results_df['val_precision'] * 0.75 + results_df['val_recall'] * 0.25
    best_precision_weighted = results_df.loc[results_df['precision_weighted_score'].idxmax()]
    strategies['Precision-Weighted'] = {
        'params': best_precision_weighted.to_dict(),
        'metric_focus': 'Precision-biased',
        'weights': 'P=75%, R=25%'
    }
    
    # 5. Recall-Weighted (25% P, 75% R)
    results_df['recall_weighted_score'] = results_df['val_precision'] * 0.25 + results_df['val_recall'] * 0.75
    best_recall_weighted = results_df.loc[results_df['recall_weighted_score'].idxmax()]
    strategies['Recall-Weighted'] = {
        'params': best_recall_weighted.to_dict(),
        'metric_focus': 'Recall-biased',
        'weights': 'P=25%, R=75%'
    }
    
    # 6. Balance-Weighted (50% P, 50% R)
    results_df['balanced_weighted_score'] = results_df['val_precision'] * 0.50 + results_df['val_recall'] * 0.50
    best_balanced = results_df.loc[results_df['balanced_weighted_score'].idxmax()]
    strategies['Balance-Weighted'] = {
        'params': best_balanced.to_dict(),
        'metric_focus': 'Balanced',
        'weights': 'P=50%, R=50%'
    }
    
    # 7. Conservative (Recall ≥ 0.90, then max F1)
    conservative_candidates = results_df[results_df['val_recall'] >= 0.90]
    if len(conservative_candidates) > 0:
        best_conservative = conservative_candidates.loc[conservative_candidates['val_f1'].idxmax()]
        strategies['Conservative'] = {
            'params': best_conservative.to_dict(),
            'metric_focus': 'High recall with best F1',
            'weights': 'Filter: R≥90%, then max F1'
        }
    else:
        best_conservative = results_df.loc[results_df['val_recall'].idxmax()]
        strategies['Conservative'] = {
            'params': best_conservative.to_dict(),
            'metric_focus': 'Highest available recall',
            'weights': f'Max recall: {best_conservative["val_recall"]:.2%}'
        }
    
    return strategies


def evaluate_strategy_with_scores(params, train_texts, train_labels, test_texts, test_labels):
    """Evaluate a strategy on test set and return detailed metrics including scores for ROC/PR curves."""
    nsa = NegativeSelectionClassifier(
        representation="vocabulary",
        num_detectors=int(params['num_detectors']),
        detector_size=int(params['detector_size']),
        matching_rule=params['matching_rule'],
        max_attempts=5000
    )
    
    # Set rule-specific parameters
    if params['matching_rule'] == 'r_contiguous':
        nsa.r_contiguous = int(params['r_contiguous'])
        nsa.min_activations = int(params.get('min_activations', 1))
    else:
        nsa.hamming_threshold = int(params['hamming_threshold'])
        nsa.min_activations = int(params.get('min_activations', 1))
    
    # Set additional parameters
    nsa.vocab_size = int(params.get('vocab_size', 1000))
    nsa.min_word_freq = int(params.get('min_word_freq', 3))
    nsa.max_ham_match_ratio = float(params.get('max_ham_match_ratio', 0.05))
    
    # Train
    nsa.fit(train_texts, train_labels)
    
    # Predict with scores
    test_pred = nsa.predict(test_texts)
    
    # Get activation scores for each sample (count how many detectors match)
    # Use NSA's own tokenizer to ensure consistency
    test_scores = []
    for text in test_texts:
        tokens = nsa._text_to_tokens(text)  # Use NSA's tokenizer
        activations = 0
        
        if len(tokens) >= nsa.detector_size:
            for i in range(len(tokens) - nsa.detector_size + 1):
                pattern = tuple(tokens[i:i + nsa.detector_size])
                for detector in nsa.detectors:
                    if nsa._matches_pattern(detector, pattern):
                        activations += 1
        
        test_scores.append(activations)
    test_scores = np.array(test_scores)
    
    # Calculate all metrics
    tp = sum(1 for p, t in zip(test_pred, test_labels) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(test_pred, test_labels) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(test_pred, test_labels) if p == 0 and t == 1)
    tn = sum(1 for p, t in zip(test_pred, test_labels) if p == 0 and t == 0)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(test_labels)
    
    # Calculate ROC-AUC and PR-AUC
    try:
        roc_auc = roc_auc_score(test_labels, test_scores)
        fpr, tpr, _ = roc_curve(test_labels, test_scores)
        precision_curve, recall_curve, _ = precision_recall_curve(test_labels, test_scores)
        pr_auc = auc(recall_curve, precision_curve)
    except Exception as e:
        print(f"    Warning: ROC/PR calculation failed: {e}")
        roc_auc = 0.5
        pr_auc = precision
        fpr, tpr = np.array([0, 1]), np.array([0, 1])
        precision_curve, recall_curve = np.array([1, 0]), np.array([0, 1])
    
    # Calculate detector statistics manually
    ham_matches_list = []
    spam_matches_list = []
    
    for text, label in zip(train_texts, train_labels):
        tokens = nsa._text_to_tokens(text)  # Use NSA's tokenizer
        matches = 0
        
        if len(tokens) >= nsa.detector_size:
            for i in range(len(tokens) - nsa.detector_size + 1):
                pattern = tuple(tokens[i:i + nsa.detector_size])
                for detector in nsa.detectors:
                    if nsa._matches_pattern(detector, pattern):
                        matches += 1
                        break  # Count only once per position
        
        if label == 0:  # ham
            ham_matches_list.append(matches)
        else:  # spam
            spam_matches_list.append(matches)
    
    ham_matches = np.mean(ham_matches_list) if ham_matches_list else 0
    spam_matches = np.mean(spam_matches_list) if spam_matches_list else 0
    
    return {
        'test_accuracy': accuracy,
        'test_precision': precision,
        'test_recall': recall,
        'test_f1': f1,
        'test_roc_auc': roc_auc,
        'test_pr_auc': pr_auc,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp,
        'true_negatives': tn,
        'total_errors': fp + fn,
        'detectors_generated': len(nsa.detectors),
        'detector_size': int(params['detector_size']),
        'num_detectors_target': int(params['num_detectors']),
        'matching_rule': params['matching_rule'],
        'rule_parameter': int(params.get('r_contiguous', params.get('hamming_threshold', 0))) if not pd.isna(params.get('r_contiguous', params.get('hamming_threshold', 0))) else 0,
        'min_activations': int(params.get('min_activations', 1)),
        'vocab_size': int(params.get('vocab_size', 1000)),
        'ham_matches_avg': ham_matches,
        'spam_matches_avg': spam_matches,
        'discrimination_rate': spam_matches / ham_matches if ham_matches > 0 else 0,
        # ROC/PR curve data
        'roc_fpr': fpr,
        'roc_tpr': tpr,
        'pr_recall': recall_curve,
        'pr_precision': precision_curve,
    }


def run_single_comparison(train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, run_id):
    """Run a single complete optimization comparison."""
    print(f"\nRun {run_id}/3:")
    
    # Run grid search
    results_df = run_parameter_grid_search_single_run(train_texts, train_labels, val_texts, val_labels, run_id)
    
    # Find optimal parameters for all 7 strategies
    strategies = find_optimal_parameters_seven_strategies(results_df)
    
    # Evaluate each strategy on test set
    print(f"  Run {run_id}: Evaluating {len(strategies)} strategies on test set...")
    strategy_results = {}
    
    for strategy_name, strategy_info in strategies.items():
        test_results = evaluate_strategy_with_scores(
            strategy_info['params'], 
            train_texts, train_labels, 
            test_texts, test_labels
        )
        strategy_results[strategy_name] = test_results
    
    return strategy_results


def calculate_statistics(all_runs_results):
    """Calculate mean, std, and 95% CI for all metrics across runs."""
    strategy_names = list(all_runs_results[0].keys())
    
    # Exclude non-scalar metrics
    excluded_metrics = {'roc_fpr', 'roc_tpr', 'pr_precision', 'pr_recall'}
    all_metrics = list(all_runs_results[0][strategy_names[0]].keys())
    scalar_metrics = [m for m in all_metrics if m not in excluded_metrics]
    
    statistics = {}
    
    for strategy in strategy_names:
        statistics[strategy] = {}
        
        for metric in scalar_metrics:
            values = [run[strategy][metric] for run in all_runs_results if strategy in run]
            
            if len(values) > 0 and isinstance(values[0], (int, float, np.number)):
                mean_val = np.mean(values)
                std_val = np.std(values, ddof=1) if len(values) > 1 else 0
                sem_val = std_val / np.sqrt(len(values)) if len(values) > 1 else 0
                ci_95 = 1.96 * sem_val
                
                statistics[strategy][metric] = {
                    'mean': float(mean_val),
                    'std': float(std_val),
                    'sem': float(sem_val),
                    'ci_95': float(ci_95),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'n_runs': len(values)
                }
            else:
                # For non-numeric values, store the most common
                statistics[strategy][metric] = {
                    'mean': values[0] if values else None,
                    'mode': Counter(values).most_common(1)[0][0] if values else None
                }
    
    return statistics


def average_curves(all_runs_results):
    """Calculate average ROC and PR curves across runs."""
    strategy_curves = defaultdict(lambda: {'roc_fpr': [], 'roc_tpr': [], 'pr_recall': [], 'pr_precision': []})
    
    for run in all_runs_results:
        for strategy_name, results in run.items():
            if 'roc_fpr' in results:
                strategy_curves[strategy_name]['roc_fpr'].append(results['roc_fpr'])
                strategy_curves[strategy_name]['roc_tpr'].append(results['roc_tpr'])
                strategy_curves[strategy_name]['pr_recall'].append(results['pr_recall'])
                strategy_curves[strategy_name]['pr_precision'].append(results['pr_precision'])
    
    averaged_curves = {}
    for strategy_name, curves in strategy_curves.items():
        if curves['roc_fpr']:
            common_fpr = np.linspace(0, 1, 100)
            common_recall = np.linspace(0, 1, 100)
            
            # Average ROC curves
            roc_tprs = []
            for fpr, tpr in zip(curves['roc_fpr'], curves['roc_tpr']):
                roc_tprs.append(np.interp(common_fpr, fpr, tpr))
            avg_roc_tpr = np.mean(roc_tprs, axis=0)
            
            # Average PR curves
            pr_precisions = []
            for recall, precision in zip(curves['pr_recall'], curves['pr_precision']):
                recall_rev = recall[::-1]
                precision_rev = precision[::-1]
                pr_precisions.append(np.interp(common_recall, recall_rev, precision_rev))
            avg_pr_precision = np.mean(pr_precisions, axis=0)
            
            averaged_curves[strategy_name] = {
                'roc_fpr': common_fpr,
                'roc_tpr': avg_roc_tpr,
                'pr_recall': common_recall,
                'pr_precision': avg_pr_precision
            }
    
    return averaged_curves


def generate_pareto_front_with_f1_contours(statistics, output_dir):
    """Generate Pareto front plot with F1 contours."""
    print("  Generating Pareto front with F1 contours...")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    strategy_names = list(statistics.keys())
    colors = plt.cm.Set1(np.linspace(0, 1, len(strategy_names)))
    
    # Plot each strategy
    recalls = []
    precisions = []
    for i, strategy in enumerate(strategy_names):
        if 'test_precision' in statistics[strategy] and 'test_recall' in statistics[strategy]:
            precision = statistics[strategy]['test_precision']['mean']
            recall = statistics[strategy]['test_recall']['mean']
            precision_ci = statistics[strategy]['test_precision']['ci_95']
            recall_ci = statistics[strategy]['test_recall']['ci_95']
            
            recalls.append(recall)
            precisions.append(precision)
            
            ax.errorbar(recall, precision, xerr=recall_ci, yerr=precision_ci,
                       fmt='o', markersize=12, alpha=0.8, color=colors[i],
                       label=strategy, capsize=5, capthick=2, linewidth=2)
            
            # Annotate
            ax.annotate(strategy.replace('-', '\n'), (recall, precision),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=9, alpha=0.7,
                       bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[i], alpha=0.2))
    
    # Draw F1 contours
    recall_range = np.linspace(0.01, 1, 100)
    precision_range = np.linspace(0.01, 1, 100)
    R, P = np.meshgrid(recall_range, precision_range)
    F1 = 2 * P * R / (P + R)
    
    contours = ax.contour(R, P, F1, levels=[0.5, 0.6, 0.7, 0.75, 0.8, 0.85],
                          colors='gray', alpha=0.3, linewidths=1)
    ax.clabel(contours, inline=True, fontsize=8, fmt='F1=%.2f')
    
    # Find and draw Pareto frontier
    points = list(zip(recalls, precisions, strategy_names))
    points_sorted = sorted(points, key=lambda x: x[0])
    
    pareto_points = []
    max_precision = 0
    for recall, precision, strategy in points_sorted:
        if precision > max_precision:
            pareto_points.append((recall, precision, strategy))
            max_precision = precision
    
    if len(pareto_points) > 1:
        pareto_recalls = [p[0] for p in pareto_points]
        pareto_precisions = [p[1] for p in pareto_points]
        ax.plot(pareto_recalls, pareto_precisions, 'r--', 
               linewidth=3, alpha=0.7, label='Pareto Frontier', zorder=10)
    
    ax.set_xlabel('Recall', fontsize=14, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=14, fontweight='bold')
    ax.set_title('Pareto Front Analysis with F1 Contours\n(Error bars show 95% confidence intervals)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    pareto_path = output_dir / 'plots' / 'pareto_front_with_f1_contours.png'
    pareto_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(pareto_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {pareto_path}")


def generate_roc_pr_curves(averaged_curves, statistics, output_dir):
    """Generate ROC and PR curves."""
    print("  Generating ROC and PR curves...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    colors = plt.cm.Set1(np.linspace(0, 1, 7))
    
    strategy_names = list(averaged_curves.keys())
    
    # ROC Curves
    for i, strategy in enumerate(strategy_names):
        if strategy in averaged_curves:
            curves = averaged_curves[strategy]
            roc_auc = statistics[strategy]['test_roc_auc']['mean']
            ax1.plot(curves['roc_fpr'], curves['roc_tpr'],
                    color=colors[i], linewidth=2.5, alpha=0.8,
                    label=f'{strategy} (AUC={roc_auc:.3f})')
    
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1.5, label='Random')
    ax1.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    ax1.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
    ax1.set_title('ROC Curves (Averaged over 3 runs)', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # PR Curves
    for i, strategy in enumerate(strategy_names):
        if strategy in averaged_curves:
            curves = averaged_curves[strategy]
            pr_auc = statistics[strategy]['test_pr_auc']['mean']
            ax2.plot(curves['pr_recall'], curves['pr_precision'],
                    color=colors[i], linewidth=2.5, alpha=0.8,
                    label=f'{strategy} (AUC={pr_auc:.3f})')
    
    ax2.set_xlabel('Recall', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Precision', fontsize=13, fontweight='bold')
    ax2.set_title('Precision-Recall Curves (Averaged over 3 runs)', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    roc_pr_path = output_dir / 'plots' / 'roc_pr_curves.png'
    roc_pr_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(roc_pr_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {roc_pr_path}")


def generate_detector_coverage_plot(statistics, output_dir):
    """Generate detector coverage plot showing recall vs number of detectors."""
    print("  Generating detector coverage plot...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.Set1(np.linspace(0, 1, 7))
    
    strategy_names = list(statistics.keys())
    
    # For each strategy, show target vs generated detectors and resulting recall
    for i, strategy in enumerate(strategy_names):
        if 'num_detectors_target' in statistics[strategy] and 'detectors_generated' in statistics[strategy]:
            target = statistics[strategy]['num_detectors_target']['mean']
            generated = statistics[strategy]['detectors_generated']['mean']
            recall = statistics[strategy]['test_recall']['mean']
            
            # Plot point
            ax.scatter(generated, recall, s=200, alpha=0.7, color=colors[i],
                      label=f'{strategy} (target={int(target)})', marker='o', edgecolors='black', linewidth=2)
            
            # Annotate with target and generated
            ax.annotate(f'{int(generated)}', (generated, recall),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Number of Detectors Generated', fontsize=13, fontweight='bold')
    ax.set_ylabel('Recall', fontsize=13, fontweight='bold')
    ax.set_title('Detector Coverage: Recall vs Number of Detectors\n(Averaged over 3 runs)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    coverage_path = output_dir / 'plots' / 'detector_coverage.png'
    coverage_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(coverage_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {coverage_path}")


def save_score_table(statistics, output_dir):
    """Save score table with all metrics."""
    print("  Saving score table...")
    
    score_data = []
    for strategy in statistics.keys():
        row = {
            'Strategy': strategy,
            'Precision': f"{statistics[strategy]['test_precision']['mean']:.4f} ± {statistics[strategy]['test_precision']['ci_95']:.4f}",
            'Recall': f"{statistics[strategy]['test_recall']['mean']:.4f} ± {statistics[strategy]['test_recall']['ci_95']:.4f}",
            'F1': f"{statistics[strategy]['test_f1']['mean']:.4f} ± {statistics[strategy]['test_f1']['ci_95']:.4f}",
            'ROC-AUC': f"{statistics[strategy]['test_roc_auc']['mean']:.4f} ± {statistics[strategy]['test_roc_auc']['ci_95']:.4f}",
            'PR-AUC': f"{statistics[strategy]['test_pr_auc']['mean']:.4f} ± {statistics[strategy]['test_pr_auc']['ci_95']:.4f}",
            'Accuracy': f"{statistics[strategy]['test_accuracy']['mean']:.4f} ± {statistics[strategy]['test_accuracy']['ci_95']:.4f}",
            'FP': f"{statistics[strategy]['false_positives']['mean']:.1f} ± {statistics[strategy]['false_positives']['ci_95']:.1f}",
            'FN': f"{statistics[strategy]['false_negatives']['mean']:.1f} ± {statistics[strategy]['false_negatives']['ci_95']:.1f}",
        }
        score_data.append(row)
    
    score_df = pd.DataFrame(score_data)
    score_path = output_dir / 'score_table.csv'
    score_df.to_csv(score_path, index=False)
    print(f"    Saved: {score_path}")


def save_hyperparameters_table(statistics, all_runs_results, output_dir):
    """Save optimal hyperparameters table."""
    print("  Saving hyperparameters table...")
    
    hyperparam_data = []
    for strategy in statistics.keys():
        # Find most common hyperparameters across runs
        matching_rules = []
        rule_params = []
        detector_sizes = []
        min_activations = []
        vocab_sizes = []
        
        for run in all_runs_results:
            if strategy in run:
                matching_rules.append(run[strategy]['matching_rule'])
                rule_params.append(run[strategy]['rule_parameter'])
                detector_sizes.append(run[strategy]['detector_size'])
                min_activations.append(run[strategy]['min_activations'])
                vocab_sizes.append(run[strategy]['vocab_size'])
        
        if matching_rules:
            most_common_rule = Counter(matching_rules).most_common(1)[0][0]
            most_common_rule_param = Counter(rule_params).most_common(1)[0][0]
            most_common_size = Counter(detector_sizes).most_common(1)[0][0]
            most_common_min_act = Counter(min_activations).most_common(1)[0][0]
            most_common_vocab = Counter(vocab_sizes).most_common(1)[0][0]
            
            row = {
                'Strategy': strategy,
                'Matching Rule': most_common_rule,
                'Rule Parameter': most_common_rule_param,
                'Detector Size': most_common_size,
                'Num Detectors (target)': f"{statistics[strategy]['num_detectors_target']['mean']:.0f}",
                'Detectors Generated': f"{statistics[strategy]['detectors_generated']['mean']:.0f} ± {statistics[strategy]['detectors_generated']['ci_95']:.0f}",
                'Min Activations': most_common_min_act,
                'Vocab Size': most_common_vocab,
            }
            hyperparam_data.append(row)
    
    hyperparam_df = pd.DataFrame(hyperparam_data)
    hyperparam_path = output_dir / 'optimal_hyperparameters.csv'
    hyperparam_df.to_csv(hyperparam_path, index=False)
    print(f"    Saved: {hyperparam_path}")


def save_detector_generation_table(statistics, output_dir):
    """Save detector generation statistics table."""
    print("  Saving detector generation table...")
    
    detector_data = []
    for strategy in statistics.keys():
        try:
            # Check if ham_matches_avg exists and has the mean field
            if 'ham_matches_avg' in statistics[strategy] and 'mean' in statistics[strategy]['ham_matches_avg']:
                # Get detector_size - might be in mean or mode
                detector_size = statistics[strategy]['detector_size'].get('mean', 
                                statistics[strategy]['detector_size'].get('mode', 0))
                
                row = {
                    'Strategy': strategy,
                    'Detectors Generated': f"{statistics[strategy]['detectors_generated']['mean']:.0f} ± {statistics[strategy]['detectors_generated']['ci_95']:.0f}",
                    'Detector Size': f"{detector_size:.1f}" if isinstance(detector_size, (int, float)) else str(detector_size),
                    'Ham Matches (avg)': f"{statistics[strategy]['ham_matches_avg']['mean']:.2f} ± {statistics[strategy]['ham_matches_avg']['ci_95']:.2f}",
                    'Spam Matches (avg)': f"{statistics[strategy]['spam_matches_avg']['mean']:.2f} ± {statistics[strategy]['spam_matches_avg']['ci_95']:.2f}",
                    'Discrimination Rate': f"{statistics[strategy]['discrimination_rate']['mean']:.2f} ± {statistics[strategy]['discrimination_rate']['ci_95']:.2f}",
                    'Matching Rule': statistics[strategy]['matching_rule'].get('mode', 'N/A'),
                }
                detector_data.append(row)
        except Exception as e:
            print(f"    Warning: Could not add {strategy} to detector table: {e}")
            continue
    
    detector_df = pd.DataFrame(detector_data)
    
    if len(detector_df) == 0:
        print(f"    Warning: No detector data to save")
        return
    
    detector_path = output_dir / 'detector_generation_statistics.csv'
    detector_df.to_csv(detector_path, index=False)
    print(f"    Saved: {detector_path}")


def main():
    """Main execution pipeline."""
    print("=" * 70)
    print("NSA MULTIPLE RUNS (3x) OPTIMIZATION COMPARISON")
    print("Seven Optimization Strategies with Vocabulary-based NSA")
    print("=" * 70)
    
    # 1. Load data
    print("\n1. Loading and preparing data...")
    texts, labels = load_data(str(constants.DATA_PATH))
    
    # Split data once (fixed seed for consistency across runs)
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = train_val_test_split(
        texts, labels,
        test_ratio=constants.TEST_RATIO,
        val_ratio=constants.VAL_RATIO,
        seed=42  # Fixed seed so all runs use same split
    )
    
    print(f"Data: {len(train_texts)} train, {len(val_texts)} val, {len(test_texts)} test")
    print("Using vocabulary-based NSA with r-contiguous and Hamming matching")
    
    # 2. Run 3 independent comparisons
    print(f"\n2. Running 3 independent optimization comparisons...")
    all_runs_results = []
    
    for run_id in range(1, 4):
        try:
            run_results = run_single_comparison(
                train_texts, train_labels, val_texts, val_labels,
                test_texts, test_labels, run_id
            )
            all_runs_results.append(run_results)
            print(f"  Run {run_id}: ✓ Completed")
        except Exception as e:
            print(f"  Run {run_id}: ✗ Failed with error:")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_runs_results:
        print("\n❌ No runs completed successfully!")
        return
    
    print(f"\n✓ Completed {len(all_runs_results)} successful runs")
    
    # 3. Calculate statistics
    print(f"\n3. Calculating statistics...")
    statistics = calculate_statistics(all_runs_results)
    averaged_curves = average_curves(all_runs_results)
    
    # 4. Generate all plots
    print(f"\n4. Generating plots...")
    output_dir = constants.RESULTS_DIR
    
    try:
        generate_pareto_front_with_f1_contours(statistics, output_dir)
    except Exception as e:
        print(f"  ⚠️  Error generating Pareto front: {e}")
    
    try:
        generate_roc_pr_curves(averaged_curves, statistics, output_dir)
    except Exception as e:
        print(f"  ⚠️  Error generating ROC/PR curves: {e}")
    
    try:
        generate_detector_coverage_plot(statistics, output_dir)
    except Exception as e:
        print(f"  ⚠️  Error generating detector coverage plot: {e}")
    
    # 5. Save all tables
    print(f"\n5. Saving tables...")
    try:
        save_score_table(statistics, output_dir)
    except Exception as e:
        print(f"  ⚠️  Error saving score table: {e}")
    
    try:
        save_hyperparameters_table(statistics, all_runs_results, output_dir)
    except Exception as e:
        print(f"  ⚠️  Error saving hyperparameters table: {e}")
    
    try:
        save_detector_generation_table(statistics, output_dir)
    except Exception as e:
        print(f"  ⚠️  Error saving detector generation table: {e}")
    
    # 6. Print summary
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY (Mean ± 95% CI)")
    print(f"{'='*70}")
    
    for strategy in ['F1-Optimized', 'Precision-Optimized', 'Recall-Optimized', 
                     'Precision-Weighted', 'Recall-Weighted', 'Balance-Weighted', 'Conservative']:
        if strategy in statistics:
            stats = statistics[strategy]
            print(f"\n{strategy}:")
            print(f"  F1:        {stats['test_f1']['mean']:.4f} ± {stats['test_f1']['ci_95']:.4f}")
            print(f"  Precision: {stats['test_precision']['mean']:.4f} ± {stats['test_precision']['ci_95']:.4f}")
            print(f"  Recall:    {stats['test_recall']['mean']:.4f} ± {stats['test_recall']['ci_95']:.4f}")
            print(f"  ROC-AUC:   {stats['test_roc_auc']['mean']:.4f} ± {stats['test_roc_auc']['ci_95']:.4f}")
    
    print(f"\n{'='*70}")
    print("✓ Analysis complete!")
    print(f"{'='*70}")
    print(f"\nResults saved to: {output_dir}")
    print(f"\nGenerated files:")
    print(f"  Plots:")
    print(f"    - pareto_front_with_f1_contours.png")
    print(f"    - roc_pr_curves.png")
    print(f"    - detector_coverage.png")
    print(f"  Tables:")
    print(f"    - score_table.csv")
    print(f"    - optimal_hyperparameters.csv")
    print(f"    - detector_generation_statistics.csv")


if __name__ == "__main__":
    main()
