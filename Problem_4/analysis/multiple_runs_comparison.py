#!/usr/bin/env python3
"""
Multiple runs optimization comparison for NSA spam detection.
Runs the optimization comparison multiple times to calculate average performance and confidence intervals.
Will take about 2 hours to run on a standard laptop.
"""

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from typing import Dict, List, Tuple
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from collections import defaultdict

# Add the src directory to the path so we can import modules
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import constants
from preprocessing import load_data, train_val_test_split, build_vocabulary, texts_to_sets
from nsa import NegativeSelectionClassifier
from utils import set_seed, classification_report
from sklearn.metrics import roc_auc_score, average_precision_score
import itertools

def run_parameter_grid_search(train_sets, train_labels, val_sets, val_labels, vocab_size, run_id):
    """Run comprehensive parameter grid search for one run"""
    print(f"  Run {run_id}: Running parameter grid search...")
    
    param_grid = {
        'num_detectors': [100, 300, 500, 700, 1000, 1500, 2000, 3000],
        'detector_size': [2, 3, 4, 5],
        'overlap_threshold': [1, 2, 3, 4],
        'min_activations': [1, 2, 3]
    }
    
    param_combinations = list(itertools.product(*param_grid.values()))
    param_names = list(param_grid.keys())
    total_combinations = len(param_combinations)
    
    results = []
    
    for i, param_values in enumerate(param_combinations):
        if i % 50 == 0:
            print(f"    Progress: {i+1}/{total_combinations} experiments...")
        
        param_dict = dict(zip(param_names, param_values))
        
        # Train NSA with current parameters
        nsa = NegativeSelectionClassifier(
            vocab_size=vocab_size,
            num_detectors=param_dict['num_detectors'],
            detector_size=param_dict['detector_size'],
            overlap_threshold=param_dict['overlap_threshold'],
            min_activations=param_dict['min_activations'],
            max_attempts=constants.NSA_MAX_ATTEMPTS,
            seed=None  # Keep randomness for multiple runs
        )
        
        nsa.fit(train_sets, train_labels)
        val_pred, val_scores = nsa.predict_with_scores(val_sets)
        val_metrics = classification_report(val_labels, val_pred)
        
        # Store results
        result = {
            **param_dict,
            'detectors_generated': nsa.detectors_count,
            'attempts_used': nsa.attempts_used,
            'val_accuracy': val_metrics['accuracy'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1'],
            'val_roc_auc': roc_auc_score(val_labels, val_scores),
            'val_pr_auc': average_precision_score(val_labels, val_scores),
        }
        
        # Calculate weighted scores
        result['precision_weighted'] = 0.75 * result['val_precision'] + 0.25 * result['val_recall']
        result['recall_weighted'] = 0.25 * result['val_precision'] + 0.75 * result['val_recall']
        result['balanced_weighted'] = 0.50 * result['val_precision'] + 0.50 * result['val_recall']
        result['precision_optimized_weighted'] = 0.99 * result['val_precision'] + 0.01 * result['val_recall']
        
        results.append(result)
    
    return pd.DataFrame(results)

def find_optimal_parameters_single_run(results_df):
    """Find optimal parameters using different strategies for a single run"""
    strategies = {}
    
    # Strategy 1: F1-Score Optimization
    f1_best = results_df.loc[results_df['val_f1'].idxmax()]
    strategies['F1-Optimized'] = {'params': f1_best}
    
    # Strategy 2: Precision Optimization (99% precision, 1% recall)
    precision_best = results_df.loc[results_df['precision_optimized_weighted'].idxmax()]
    strategies['Precision-Optimized'] = {'params': precision_best}
    
    # Strategy 3: Recall Optimization
    recall_best = results_df.loc[results_df['val_recall'].idxmax()]
    strategies['Recall-Optimized'] = {'params': recall_best}
    
    # Strategy 4: Precision-Weighted (75% precision, 25% recall)
    weighted_best = results_df.loc[results_df['precision_weighted'].idxmax()]
    strategies['Precision-Weighted'] = {'params': weighted_best}
    
    # Strategy 5: Recall-Weighted (25% precision, 75% recall)
    recall_weighted_best = results_df.loc[results_df['recall_weighted'].idxmax()]
    strategies['Recall-Weighted'] = {'params': recall_weighted_best}
    
    # Strategy 6: Balanced (50% precision, 50% recall)
    balanced_best = results_df.loc[results_df['balanced_weighted'].idxmax()]
    strategies['Balanced-Weighted'] = {'params': balanced_best}
    
    # Strategy 7: Conservative (High precision threshold)
    high_precision = results_df[results_df['val_precision'] >= 0.93]
    if not high_precision.empty:
        conservative_best = high_precision.loc[high_precision['val_f1'].idxmax()]
        strategies['Conservative'] = {'params': conservative_best}
    
    return strategies

def evaluate_strategy_on_test_single_run(strategy_params, train_sets, train_labels, test_sets, test_labels, vocab_size):
    """Evaluate a strategy on test set for a single run"""
    nsa = NegativeSelectionClassifier(
        vocab_size=vocab_size,
        num_detectors=int(strategy_params['num_detectors']),
        detector_size=int(strategy_params['detector_size']),
        overlap_threshold=int(strategy_params['overlap_threshold']),
        min_activations=int(strategy_params['min_activations']),
        max_attempts=constants.NSA_MAX_ATTEMPTS,
        seed=None  # Keep randomness
    )
    
    nsa.fit(train_sets, train_labels)
    test_pred, test_scores = nsa.predict_with_scores(test_sets)
    test_metrics = classification_report(test_labels, test_pred)
    
    # Calculate confusion matrix components
    true_positives = sum(1 for true_label, pred_label in zip(test_labels, test_pred) 
                        if true_label == 1 and pred_label == 1)
    true_negatives = sum(1 for true_label, pred_label in zip(test_labels, test_pred) 
                        if true_label == 0 and pred_label == 0)
    false_positives = sum(1 for true_label, pred_label in zip(test_labels, test_pred) 
                         if true_label == 0 and pred_label == 1)
    false_negatives = sum(1 for true_label, pred_label in zip(test_labels, test_pred) 
                         if true_label == 1 and pred_label == 0)
    
    # Calculate ROC and PR curves
    fpr, tpr, _ = roc_curve(test_labels, test_scores)
    precision_curve, recall_curve, _ = precision_recall_curve(test_labels, test_scores)
    
    return {
        'test_accuracy': test_metrics['accuracy'],
        'test_precision': test_metrics['precision'],
        'test_recall': test_metrics['recall'],
        'test_f1': test_metrics['f1'],
        'test_roc_auc': roc_auc_score(test_labels, test_scores),
        'test_pr_auc': average_precision_score(test_labels, test_scores),
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'true_positives': true_positives,
        'true_negatives': true_negatives,
        'total_errors': false_positives + false_negatives,
        'detectors_generated': nsa.detectors_count,
        'detector_size': int(strategy_params['detector_size']),
        'overlap_threshold': int(strategy_params['overlap_threshold']),
        'min_activations': int(strategy_params['min_activations']),
        # Store curve data for averaging
        'roc_fpr': fpr,
        'roc_tpr': tpr,
        'pr_precision': precision_curve,
        'pr_recall': recall_curve,
        'test_labels': test_labels,
        'test_scores': test_scores
    }

def generate_detector_coverage_single_run(train_sets, train_labels, test_sets, test_labels, vocab_size, strategy_params, run_id):
    """Generate detector coverage curve for a single strategy and run"""
    detector_counts = [50, 100, 200, 300, 500, 700, 1000, 1500, 2000, 3000]
    coverage_results = []
    
    base_params = {
        'detector_size': int(strategy_params['detector_size']),
        'overlap_threshold': int(strategy_params['overlap_threshold']),
        'min_activations': int(strategy_params['min_activations'])
    }
    
    for num_det in detector_counts:
        nsa = NegativeSelectionClassifier(
            vocab_size=vocab_size,
            num_detectors=num_det,
            detector_size=base_params['detector_size'],
            overlap_threshold=base_params['overlap_threshold'],
            min_activations=base_params['min_activations'],
            max_attempts=constants.NSA_MAX_ATTEMPTS,
            seed=None
        )
        
        nsa.fit(train_sets, train_labels)
        test_pred, test_scores = nsa.predict_with_scores(test_sets)
        test_metrics = classification_report(test_labels, test_pred)
        
        coverage_results.append({
            'num_detectors': num_det,
            'detectors_generated': nsa.detectors_count,
            'recall': test_metrics['recall'],
            'precision': test_metrics['precision'],
            'f1': test_metrics['f1']
        })
    
    return coverage_results

def run_single_comparison(train_sets, train_labels, val_sets, val_labels, test_sets, test_labels, vocab_size, run_id):
    """Run a single complete optimization comparison"""
    print(f"Run {run_id}/10:")
    
    # Run parameter grid search
    results_df = run_parameter_grid_search(train_sets, train_labels, val_sets, val_labels, vocab_size, run_id)
    
    # Find optimal parameters for each strategy
    strategies = find_optimal_parameters_single_run(results_df)
    
    # Evaluate each strategy on test set
    print(f"  Run {run_id}: Evaluating strategies on test set...")
    strategy_results = {}
    
    for strategy_name, strategy_info in strategies.items():
        test_results = evaluate_strategy_on_test_single_run(
            strategy_info['params'], train_sets, train_labels, test_sets, test_labels, vocab_size
        )
        strategy_results[strategy_name] = test_results
    
    return strategy_results

def calculate_statistics(all_runs_results):
    """Calculate mean, std, and 95% confidence intervals for all metrics"""
    strategy_names = list(all_runs_results[0].keys())
    
    # Only include scalar metrics in statistics calculation
    excluded_metrics = {'roc_fpr', 'roc_tpr', 'pr_precision', 'pr_recall', 'test_labels', 'test_scores'}
    all_metrics = list(all_runs_results[0][strategy_names[0]].keys())
    scalar_metrics = [m for m in all_metrics if m not in excluded_metrics]
    
    statistics = {}
    
    for strategy in strategy_names:
        statistics[strategy] = {}
        
        for metric in scalar_metrics:
            values = [run[strategy][metric] for run in all_runs_results if strategy in run]
            
            if len(values) > 0:
                mean_val = np.mean(values)
                std_val = np.std(values, ddof=1) if len(values) > 1 else 0
                sem_val = std_val / np.sqrt(len(values)) if len(values) > 1 else 0
                ci_95 = 1.96 * sem_val  # 95% confidence interval
                
                statistics[strategy][metric] = {
                    'mean': mean_val,
                    'std': std_val,
                    'sem': sem_val,
                    'ci_95': ci_95,
                    'min': np.min(values),
                    'max': np.max(values),
                    'n_runs': len(values)
                }
    
    return statistics

def interpolate_curve(x_vals, y_vals, common_x):
    """Interpolate curve to common x values for averaging"""
    return np.interp(common_x, x_vals, y_vals)

def average_curves(all_runs_results):
    """Calculate average ROC and PR curves across runs"""
    strategy_curves = defaultdict(lambda: {'roc_fpr': [], 'roc_tpr': [], 'pr_recall': [], 'pr_precision': []})
    
    # Collect all curves
    for run in all_runs_results:
        for strategy_name, results in run.items():
            if 'roc_fpr' in results:
                strategy_curves[strategy_name]['roc_fpr'].append(results['roc_fpr'])
                strategy_curves[strategy_name]['roc_tpr'].append(results['roc_tpr'])
                strategy_curves[strategy_name]['pr_recall'].append(results['pr_recall'])
                strategy_curves[strategy_name]['pr_precision'].append(results['pr_precision'])
    
    # Calculate averages
    averaged_curves = {}
    for strategy_name, curves in strategy_curves.items():
        if curves['roc_fpr']:
            # Common x values for interpolation
            common_fpr = np.linspace(0, 1, 100)
            common_recall = np.linspace(0, 1, 100)
            
            # Interpolate and average ROC curves
            roc_tprs = []
            for fpr, tpr in zip(curves['roc_fpr'], curves['roc_tpr']):
                roc_tprs.append(interpolate_curve(fpr, tpr, common_fpr))
            avg_roc_tpr = np.mean(roc_tprs, axis=0)
            
            # Interpolate and average PR curves (reverse order for recall)
            pr_precisions = []
            for recall, precision in zip(curves['pr_recall'], curves['pr_precision']):
                # Reverse for interpolation (recall should be increasing)
                recall_rev = recall[::-1]
                precision_rev = precision[::-1]
                pr_precisions.append(interpolate_curve(recall_rev, precision_rev, common_recall))
            avg_pr_precision = np.mean(pr_precisions, axis=0)
            
            averaged_curves[strategy_name] = {
                'roc_fpr': common_fpr,
                'roc_tpr': avg_roc_tpr,
                'pr_recall': common_recall,
                'pr_precision': avg_pr_precision
            }
    
    return averaged_curves

def generate_averaged_plots(statistics, averaged_curves, output_dir):
    """Generate averaged plots from multiple runs"""
    print("  Generating averaged plots...")
    
    # Set style
    plt.style.use('default')
    colors = plt.cm.Set1(np.linspace(0, 1, 7))
    
    # Create main comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    strategy_names = list(statistics.keys())
    
    # 1. ROC Curves
    for i, strategy in enumerate(strategy_names):
        if strategy in averaged_curves:
            curves = averaged_curves[strategy]
            roc_auc = statistics[strategy]['test_roc_auc']['mean']
            axes[0].plot(curves['roc_fpr'], curves['roc_tpr'], 
                        color=colors[i], linewidth=2, alpha=0.8,
                        label=f'{strategy} (AUC={roc_auc:.3f})')
    
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curves (Averaged)')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Precision-Recall Curves
    for i, strategy in enumerate(strategy_names):
        if strategy in averaged_curves:
            curves = averaged_curves[strategy]
            pr_auc = statistics[strategy]['test_pr_auc']['mean']
            axes[1].plot(curves['pr_recall'], curves['pr_precision'],
                        color=colors[i], linewidth=2, alpha=0.8,
                        label=f'{strategy} (AUC={pr_auc:.3f})')
    
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curves (Averaged)')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Performance Metrics Comparison
    x = np.arange(len(strategy_names))
    width = 0.2
    
    precisions = [statistics[s]['test_precision']['mean'] for s in strategy_names]
    recalls = [statistics[s]['test_recall']['mean'] for s in strategy_names]
    f1_scores = [statistics[s]['test_f1']['mean'] for s in strategy_names]
    
    axes[2].bar(x - width, precisions, width, label='Precision', alpha=0.8, color='skyblue')
    axes[2].bar(x, recalls, width, label='Recall', alpha=0.8, color='lightcoral')
    axes[2].bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8, color='lightgreen')
    
    axes[2].set_xlabel('Strategy')
    axes[2].set_ylabel('Score')
    axes[2].set_title('Performance Metrics (Averaged)')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([s.replace('-', '\n') for s in strategy_names], rotation=45, ha='right')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / 'plots' / 'multiple_runs_averaged_comparison.png'
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {plot_path}")

def generate_detector_coverage_plots(all_runs_results, statistics, train_sets, train_labels, test_sets, test_labels, vocab_size, output_dir):
    """Generate averaged detector coverage plots"""
    print("  Generating detector coverage analysis...")
    
    # Generate coverage data for all strategies
    all_strategies = ['F1-Optimized', 'Precision-Optimized', 'Recall-Optimized', 
                      'Precision-Weighted', 'Recall-Weighted', 'Balanced-Weighted', 'Conservative']
    
    strategy_coverage = defaultdict(list)
    
    # Use the most common parameters from the first few runs
    for strategy in all_strategies:
        if strategy in statistics:
            print(f"    Generating coverage for {strategy}...")
            
            # Get most common parameters across runs
            param_combinations = []
            for run in all_runs_results[:3]:  # Use first 3 runs to determine common params
                if strategy in run:
                    params = {
                        'detector_size': run[strategy].get('detector_size', 3),
                        'overlap_threshold': run[strategy].get('overlap_threshold', 1),
                        'min_activations': run[strategy].get('min_activations', 2)
                    }
                    param_combinations.append(params)
            
            if param_combinations:
                # Use most common parameters
                common_params = param_combinations[0]  # Simplified - use first
                
                # Generate coverage for 3 runs (to save time)
                for run_id in range(1, 4):
                    coverage_data = generate_detector_coverage_single_run(
                        train_sets, train_labels, test_sets, test_labels, 
                        vocab_size, common_params, run_id
                    )
                    strategy_coverage[strategy].append(coverage_data)
    
    # Average coverage data
    averaged_coverage = {}
    for strategy, runs_data in strategy_coverage.items():
        if runs_data:
            detector_counts = [50, 100, 200, 300, 500, 700, 1000, 1500, 2000, 3000]
            avg_recalls = []
            avg_precisions = []
            avg_f1s = []
            
            for i, num_det in enumerate(detector_counts):
                recalls = [run[i]['recall'] for run in runs_data if i < len(run)]
                precisions = [run[i]['precision'] for run in runs_data if i < len(run)]
                f1s = [run[i]['f1'] for run in runs_data if i < len(run)]
                
                avg_recalls.append(np.mean(recalls) if recalls else 0)
                avg_precisions.append(np.mean(precisions) if precisions else 0)
                avg_f1s.append(np.mean(f1s) if f1s else 0)
            
            averaged_coverage[strategy] = {
                'detector_counts': detector_counts,
                'recalls': avg_recalls,
                'precisions': avg_precisions,
                'f1s': avg_f1s
            }
    
    # Plot detector coverage
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    colors = plt.cm.Set1(np.linspace(0, 1, len(all_strategies)))
    
    for i, strategy in enumerate(all_strategies):
        if strategy in averaged_coverage:
            data = averaged_coverage[strategy]
            ax.plot(data['detector_counts'], data['recalls'], 
                   marker='o', linewidth=2, markersize=6, color=colors[i],
                   label=f'{strategy}', alpha=0.8)
    
    ax.set_xlabel('Number of Detectors (Target)')
    ax.set_ylabel('Recall')
    ax.set_title('Detector Coverage Analysis: Recall vs Number of Detectors (Averaged)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 3200)
    ax.set_ylim(0, 1)
    
    # Save plot
    coverage_path = output_dir / 'plots' / 'multiple_runs_detector_coverage.png'
    coverage_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(coverage_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {coverage_path}")

def save_results(statistics, all_runs_results, output_dir):
    """Save statistical results to files"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed statistics as JSON
    with open(output_dir / 'multiple_runs_statistics.json', 'w') as f:
        # Convert numpy types to regular Python types for JSON serialization
        stats_for_json = {}
        for strategy, metrics in statistics.items():
            stats_for_json[strategy] = {}
            for metric, values in metrics.items():
                stats_for_json[strategy][metric] = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                                                   for k, v in values.items()}
        json.dump(stats_for_json, f, indent=2)
    
    # Create Table 1: Performance and Error Analysis (clean means for LaTeX)
    performance_data = []
    strategy_names = list(statistics.keys())
    
    for strategy in strategy_names:
        if strategy in statistics:
            stats = statistics[strategy]
            row = {
                'Strategy': strategy,
                'Precision': f"{stats['test_precision']['mean']:.3f}",
                'Recall': f"{stats['test_recall']['mean']:.3f}",
                'F1-Score': f"{stats['test_f1']['mean']:.3f}",
                'ROC-AUC': f"{stats['test_roc_auc']['mean']:.3f}",
                'PR-AUC': f"{stats['test_pr_auc']['mean']:.3f}",
                'Accuracy': f"{stats['test_accuracy']['mean']:.3f}",
                'False Positives': f"{stats['false_positives']['mean']:.0f}",
                'False Negatives': f"{stats['false_negatives']['mean']:.0f}",
            }
            performance_data.append(row)
    
    performance_df = pd.DataFrame(performance_data)
    performance_df.to_csv(output_dir / 'performance_table_means.csv', index=False)
    
    # Create Table 2: Optimal Hyperparameters (most frequent selection)
    hyperparameter_data = []
    
    for strategy in strategy_names:
        if strategy in statistics:
            # Find most frequently selected hyperparameters across runs
            param_combinations = []
            for run in all_runs_results:
                if strategy in run:
                    params = (
                        run[strategy].get('detector_size', 0),
                        run[strategy].get('overlap_threshold', 0), 
                        run[strategy].get('min_activations', 0)
                    )
                    param_combinations.append(params)
            
            if param_combinations:
                # Find most common parameter combination
                from collections import Counter
                most_common_params = Counter(param_combinations).most_common(1)[0][0]
                
                # Get average number of detectors generated for this strategy
                avg_detectors = statistics[strategy]['detectors_generated']['mean']
                
                row = {
                    'Strategy': strategy,
                    'Detectors Generated': f"{avg_detectors:.0f}",
                    'Detector Size': int(most_common_params[0]),
                    'Overlap Threshold': int(most_common_params[1]),
                    'Min Activations': int(most_common_params[2])
                }
                hyperparameter_data.append(row)
    
    hyperparameter_df = pd.DataFrame(hyperparameter_data)
    hyperparameter_df.to_csv(output_dir / 'hyperparameters_table.csv', index=False)
    
    # Create summary with confidence intervals for academic reporting
    summary_data = []
    for strategy in strategy_names:
        if strategy in statistics:
            stats = statistics[strategy]
            row = {
                'Strategy': strategy,
                'Precision (Mean ± 95% CI)': f"{stats['test_precision']['mean']:.3f} ± {stats['test_precision']['ci_95']:.3f}",
                'Recall (Mean ± 95% CI)': f"{stats['test_recall']['mean']:.3f} ± {stats['test_recall']['ci_95']:.3f}",
                'F1-Score (Mean ± 95% CI)': f"{stats['test_f1']['mean']:.3f} ± {stats['test_f1']['ci_95']:.3f}",
                'ROC-AUC (Mean ± 95% CI)': f"{stats['test_roc_auc']['mean']:.3f} ± {stats['test_roc_auc']['ci_95']:.3f}",
                'Runs': stats['test_precision']['n_runs']
            }
            summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / 'multiple_runs_summary.csv', index=False)
    
    print(f"\nResults saved:")
    print(f"- Performance table (means): {output_dir / 'performance_table_means.csv'}")
    print(f"- Hyperparameters table: {output_dir / 'hyperparameters_table.csv'}")
    print(f"- Statistical summary: {output_dir / 'multiple_runs_summary.csv'}")
    print(f"- Detailed statistics: {output_dir / 'multiple_runs_statistics.json'}")
    """Save statistical results to files"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed statistics as JSON
    with open(output_dir / 'multiple_runs_statistics.json', 'w') as f:
        # Convert numpy types to regular Python types for JSON serialization
        stats_for_json = {}
        for strategy, metrics in statistics.items():
            stats_for_json[strategy] = {}
            for metric, values in metrics.items():
                stats_for_json[strategy][metric] = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                                                   for k, v in values.items()}
        json.dump(stats_for_json, f, indent=2)
    
    # Create Table 1: Performance and Error Analysis (clean means for LaTeX)
    performance_data = []
    strategy_names = list(statistics.keys())
    
    for strategy in strategy_names:
        if strategy in statistics:
            stats = statistics[strategy]
            row = {
                'Strategy': strategy,
                'Precision': f"{stats['test_precision']['mean']:.3f}",
                'Recall': f"{stats['test_recall']['mean']:.3f}",
                'F1-Score': f"{stats['test_f1']['mean']:.3f}",
                'ROC-AUC': f"{stats['test_roc_auc']['mean']:.3f}",
                'PR-AUC': f"{stats['test_pr_auc']['mean']:.3f}",
                'Accuracy': f"{stats['test_accuracy']['mean']:.3f}",
                'False Positives': f"{stats['false_positives']['mean']:.0f}",
                'False Negatives': f"{stats['false_negatives']['mean']:.0f}",
            }
            performance_data.append(row)
    
    performance_df = pd.DataFrame(performance_data)
    performance_df.to_csv(output_dir / 'performance_table_means.csv', index=False)
    
    # Create Table 2: Optimal Hyperparameters (most frequent selection)
    hyperparameter_data = []
    
    for strategy in strategy_names:
        if strategy in statistics:
            # Find most frequently selected hyperparameters across runs
            param_combinations = []
            for run in all_runs_results:
                if strategy in run:
                    params = (
                        run[strategy].get('detector_size', 0),
                        run[strategy].get('overlap_threshold', 0), 
                        run[strategy].get('min_activations', 0)
                    )
                    param_combinations.append(params)
            
            if param_combinations:
                # Find most common parameter combination
                from collections import Counter
                most_common_params = Counter(param_combinations).most_common(1)[0][0]
                
                # Get average number of detectors generated for this strategy
                avg_detectors = statistics[strategy]['detectors_generated']['mean']
                
                row = {
                    'Strategy': strategy,
                    'Detectors Generated': f"{avg_detectors:.0f}",
                    'Detector Size': int(most_common_params[0]),
                    'Overlap Threshold': int(most_common_params[1]),
                    'Min Activations': int(most_common_params[2])
                }
                hyperparameter_data.append(row)
    
    hyperparameter_df = pd.DataFrame(hyperparameter_data)
    hyperparameter_df.to_csv(output_dir / 'hyperparameters_table.csv', index=False)
    
    # Create summary with confidence intervals for academic reporting
    summary_data = []
    for strategy in strategy_names:
        if strategy in statistics:
            stats = statistics[strategy]
            row = {
                'Strategy': strategy,
                'Precision (Mean ± 95% CI)': f"{stats['test_precision']['mean']:.3f} ± {stats['test_precision']['ci_95']:.3f}",
                'Recall (Mean ± 95% CI)': f"{stats['test_recall']['mean']:.3f} ± {stats['test_recall']['ci_95']:.3f}",
                'F1-Score (Mean ± 95% CI)': f"{stats['test_f1']['mean']:.3f} ± {stats['test_f1']['ci_95']:.3f}",
                'ROC-AUC (Mean ± 95% CI)': f"{stats['test_roc_auc']['mean']:.3f} ± {stats['test_roc_auc']['ci_95']:.3f}",
                'Runs': stats['test_precision']['n_runs']
            }
            summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / 'multiple_runs_summary.csv', index=False)
    
    print(f"\nResults saved:")
    print(f"- Performance table (means): {output_dir / 'performance_table_means.csv'}")
    print(f"- Hyperparameters table: {output_dir / 'hyperparameters_table.csv'}")
    print(f"- Statistical summary: {output_dir / 'multiple_runs_summary.csv'}")
    print(f"- Detailed statistics: {output_dir / 'multiple_runs_statistics.json'}")

def main():
    """Run multiple optimization comparisons and calculate statistics"""
    print("="*70)
    print("NSA MULTIPLE RUNS OPTIMIZATION COMPARISON")
    print("="*70)
    
    # Load and prepare data once
    print("1. Loading and preparing data...")
    texts, labels = load_data(str(constants.DATA_PATH))
    
    # Split data once for all runs (keep consistent across runs)
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = train_val_test_split(
        texts, labels, 
        test_ratio=constants.TEST_RATIO, 
        val_ratio=constants.VAL_RATIO,
        seed=42
    )
    
    print(f"Data loaded: {len(train_texts)} train, {len(val_texts)} val, {len(test_texts)} test")
    
    # Build vocabulary and create token sets
    vocab_list, vocab_index = build_vocabulary(train_texts, 
                                             min_freq=constants.VOCAB_MIN_FREQ, 
                                             max_size=constants.VOCAB_MAX_SIZE)
    vocab_size = len(vocab_list)
    print(f"Vocabulary size: {vocab_size}")
    
    train_sets = texts_to_sets(train_texts, vocab_index)
    val_sets = texts_to_sets(val_texts, vocab_index)
    test_sets = texts_to_sets(test_texts, vocab_index)
    
    # Run multiple comparisons
    print(f"\n2. Running 10 independent optimization comparisons...")
    all_runs_results = []
    
    for run_id in range(1, 11):
        try:
            run_results = run_single_comparison(
                train_sets, train_labels, val_sets, val_labels, 
                test_sets, test_labels, vocab_size, run_id
            )
            all_runs_results.append(run_results)
            print(f"  Run {run_id}: Completed successfully")
        except Exception as e:
            print(f"  Run {run_id}: Failed with error: {e}")
            continue
    
    if not all_runs_results:
        print("ERROR: No runs completed successfully!")
        return
    
    print(f"\n3. Calculating statistics from {len(all_runs_results)} successful runs...")
    statistics = calculate_statistics(all_runs_results)
    
    print(f"\n4. Generating averaged plots and detector coverage analysis...")
    # Calculate averaged curves
    averaged_curves = average_curves(all_runs_results)
    
    # Save results
    output_dir = Path(__file__).parent.parent / 'results'
    save_results(statistics, all_runs_results, output_dir)
    
    # Generate plots
    generate_averaged_plots(statistics, averaged_curves, output_dir)
    generate_detector_coverage_plots(all_runs_results, statistics, train_sets, train_labels, test_sets, test_labels, vocab_size, output_dir)
    
    # Print summary
    print("\n" + "="*70)
    print("MULTIPLE RUNS SUMMARY")
    print("="*70)
    
    for strategy in ['F1-Optimized', 'Precision-Optimized', 'Recall-Optimized', 'Conservative']:
        if strategy in statistics:
            stats = statistics[strategy]
            print(f"\n{strategy}:")
            print(f"  Precision: {stats['test_precision']['mean']:.3f} ± {stats['test_precision']['ci_95']:.3f}")
            print(f"  Recall:    {stats['test_recall']['mean']:.3f} ± {stats['test_recall']['ci_95']:.3f}")
            print(f"  F1-Score:  {stats['test_f1']['mean']:.3f} ± {stats['test_f1']['ci_95']:.3f}")
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)

if __name__ == "__main__":
    main()