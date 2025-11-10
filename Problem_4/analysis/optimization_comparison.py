#!/usr/bin/env python3
"""
Single run for Optimization Strategy (f1-optimzed, precision-optimized, recall-optimized, etc.)
Compare different parameter optimization approaches for NSA spam detection.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import json
from pathlib import Path
import itertools
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add the src directory to path
script_dir = Path(__file__).resolve().parent
src_dir = script_dir.parent / 'src'
sys.path.insert(0, str(src_dir))

# Import our modules
from preprocessing import load_data, train_val_test_split, build_vocabulary, texts_to_sets
from nsa import NegativeSelectionClassifier
from utils import set_seed, classification_report
import constants

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def save_plot(fig, filename, output_dir):
    """Save plot with high DPI for reports"""
    plot_path = output_dir / 'plots' / filename
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {plot_path.resolve()}")
    plt.close(fig)

def run_parameter_grid_search(train_sets, train_labels, val_sets, val_labels, vocab_size):
    """Run comprehensive parameter grid search"""
    print("Running parameter grid search...")
    
    param_grid = {
        'num_detectors': [100, 300, 500, 700, 1000, 1500, 2000, 3000],
        'detector_size': [2, 3, 4, 5],
        'overlap_threshold': [1, 2, 3, 4],
        'min_activations': [1, 2, 3]
    }
    
    param_combinations = list(itertools.product(*param_grid.values()))
    param_names = list(param_grid.keys())
    results = []
    
    for i, params in enumerate(param_combinations):
        if i % 20 == 0:  # Progress indicator
            print(f"Progress: {i+1}/{len(param_combinations)} experiments...")
        
        param_dict = dict(zip(param_names, params))
        
        # Create and train NSA
        nsa = NegativeSelectionClassifier(
            vocab_size=vocab_size,
            num_detectors=param_dict['num_detectors'],
            detector_size=param_dict['detector_size'],
            overlap_threshold=param_dict['overlap_threshold'],
            min_activations=param_dict['min_activations'],
            max_attempts=constants.NSA_MAX_ATTEMPTS,
            seed=constants.SEED
        )
        
        nsa.fit(train_sets, train_labels)
        val_pred, val_scores = nsa.predict_with_scores(val_sets)
        val_metrics = classification_report(val_labels, val_pred)
        
        result = {
            **param_dict,
            'detectors_generated': nsa.detectors_count,
            'attempts_used': nsa.attempts_used,
            'val_accuracy': val_metrics['accuracy'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1'],
            'val_roc_auc': roc_auc_score(val_labels, val_scores) if len(set(val_labels)) > 1 else 0,
            'val_pr_auc': average_precision_score(val_labels, val_scores) if len(set(val_labels)) > 1 else 0
        }
        results.append(result)
    
    print("Grid search completed!")
    return pd.DataFrame(results)

def find_optimal_parameters(results_df):
    """Find optimal parameters using different strategies"""
    strategies = {}
    
    # Strategy 1: F1-Score Optimization
    f1_best = results_df.loc[results_df['val_f1'].idxmax()]
    strategies['F1-Optimized'] = {
        'params': f1_best,
        'strategy': 'Maximize F1-Score',
        'description': 'Balanced precision-recall optimization'
    }
    
    # Strategy 2: Precision Optimization (99% precision, 1% recall)
    results_df['precision_optimized_weighted'] = 0.99 * results_df['val_precision'] + 0.01 * results_df['val_recall']
    precision_best = results_df.loc[results_df['precision_optimized_weighted'].idxmax()]
    strategies['Precision-Optimized'] = {
        'params': precision_best,
        'strategy': 'Maximize Precision (99% weight)',
        'description': 'Minimize false positives (ham→spam)'
    }
    
    # Strategy 3: Recall Optimization
    recall_best = results_df.loc[results_df['val_recall'].idxmax()]
    strategies['Recall-Optimized'] = {
        'params': recall_best,
        'strategy': 'Maximize Recall',
        'description': 'Minimize false negatives (spam→ham)'
    }
    
    # Strategy 4: Precision-Weighted (75% precision, 25% recall)
    results_df['precision_weighted'] = 0.75 * results_df['val_precision'] + 0.25 * results_df['val_recall']
    weighted_best = results_df.loc[results_df['precision_weighted'].idxmax()]
    strategies['Precision-Weighted'] = {
        'params': weighted_best,
        'strategy': 'Weighted Score (0.75*Precision + 0.25*Recall)',
        'description': 'Prioritize precision while considering recall'
    }
    
    # Strategy 5: Recall-Weighted (25% precision, 75% recall)
    results_df['recall_weighted'] = 0.25 * results_df['val_precision'] + 0.75 * results_df['val_recall']
    recall_weighted_best = results_df.loc[results_df['recall_weighted'].idxmax()]
    strategies['Recall-Weighted'] = {
        'params': recall_weighted_best,
        'strategy': 'Weighted Score (0.25*Precision + 0.75*Recall)',
        'description': 'Prioritize recall while considering precision'
    }
    
    # Strategy 6: Balanced (50% precision, 50% recall)
    results_df['balanced_weighted'] = 0.50 * results_df['val_precision'] + 0.50 * results_df['val_recall']
    balanced_best = results_df.loc[results_df['balanced_weighted'].idxmax()]
    strategies['Balanced-Weighted'] = {
        'params': balanced_best,
        'strategy': 'Balanced Score (0.50*Precision + 0.50*Recall)',
        'description': 'Equal weight to precision and recall'
    }
    
    # Strategy 7: Conservative (High precision threshold)
    # Only consider models with precision > 0.93
    high_precision = results_df[results_df['val_precision'] >= 0.93]
    if not high_precision.empty:
        conservative_best = high_precision.loc[high_precision['val_f1'].idxmax()]
        strategies['Conservative'] = {
            'params': conservative_best,
            'strategy': 'High Precision Threshold (≥0.93) + Best F1',
            'description': 'Conservative approach ensuring high precision'
        }
    
    return strategies

def evaluate_strategy_on_test(strategy_params, train_sets, train_labels, test_sets, test_labels, vocab_size):
    """Evaluate a strategy on test set"""
    nsa = NegativeSelectionClassifier(
        vocab_size=vocab_size,
        num_detectors=int(strategy_params['num_detectors']),
        detector_size=int(strategy_params['detector_size']),
        overlap_threshold=int(strategy_params['overlap_threshold']),
        min_activations=int(strategy_params['min_activations']),
        max_attempts=constants.NSA_MAX_ATTEMPTS,
        seed=constants.SEED
    )
    
    nsa.fit(train_sets, train_labels)
    test_pred, test_scores = nsa.predict_with_scores(test_sets)
    test_metrics = classification_report(test_labels, test_pred)
    
    # Calculate confusion matrix components
    false_positives = sum(1 for true_label, pred_label in zip(test_labels, test_pred) 
                         if true_label == 0 and pred_label == 1)
    false_negatives = sum(1 for true_label, pred_label in zip(test_labels, test_pred) 
                         if true_label == 1 and pred_label == 0)
    true_positives = sum(1 for true_label, pred_label in zip(test_labels, test_pred) 
                        if true_label == 1 and pred_label == 1)
    true_negatives = sum(1 for true_label, pred_label in zip(test_labels, test_pred) 
                        if true_label == 0 and pred_label == 0)
    
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
        'test_pred': test_pred,
        'test_scores': test_scores
    }

def generate_detector_coverage_curves(train_sets, train_labels, test_sets, test_labels, vocab_size, best_params_per_strategy):
    """Generate detector coverage curves for different strategies"""
    detector_counts = [50, 100, 200, 300, 500, 700, 1000, 1500, 2000, 3000]
    coverage_data = {}
    
    for strategy_name, params in best_params_per_strategy.items():
        strategy_coverage = []
        print(f"   Generating coverage curve for {strategy_name}...")
        
        for num_det in detector_counts:
            nsa = NegativeSelectionClassifier(
                vocab_size=vocab_size,
                num_detectors=num_det,
                detector_size=int(params['detector_size']),
                overlap_threshold=int(params['overlap_threshold']),
                min_activations=int(params['min_activations']),
                max_attempts=constants.NSA_MAX_ATTEMPTS,
                seed=constants.SEED
            )
            
            nsa.fit(train_sets, train_labels)
            test_pred, test_scores = nsa.predict_with_scores(test_sets)
            metrics = classification_report(test_labels, test_pred)
            
            strategy_coverage.append({
                'num_detectors': num_det,
                'detectors_generated': nsa.detectors_count,
                'recall': metrics['recall'],
                'precision': metrics['precision']
            })
        
        coverage_data[strategy_name] = strategy_coverage
    
    return coverage_data

def create_comparison_table(strategies_results):
    """Create comparison table for different strategies"""
    comparison_data = []
    
    for strategy_name, data in strategies_results.items():
        row = {
            'Strategy': strategy_name,
            'Description': data['strategy_info']['description'],
            'Precision': f"{data['test_results']['test_precision']:.4f}",
            'Recall': f"{data['test_results']['test_recall']:.4f}",
            'F1-Score': f"{data['test_results']['test_f1']:.4f}",
            'ROC-AUC': f"{data['test_results']['test_roc_auc']:.4f}",
            'False Positives': data['test_results']['false_positives'],
            'False Negatives': data['test_results']['false_negatives'],
            'Total Errors': data['test_results']['total_errors'],
            'Detectors Generated': data['test_results']['detectors_generated'],
            'Detector Size': int(data['strategy_info']['params']['detector_size']),
            'Overlap Threshold': int(data['strategy_info']['params']['overlap_threshold']),
            'Min Activations': int(data['strategy_info']['params']['min_activations'])
        }
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)

def plot_strategy_comparison(strategies_results, coverage_data, output_dir):
    """Create comprehensive comparison plots"""
    
    # Extract data for plotting
    strategies = list(strategies_results.keys())
    precisions = [strategies_results[s]['test_results']['test_precision'] for s in strategies]
    recalls = [strategies_results[s]['test_results']['test_recall'] for s in strategies]
    f1_scores = [strategies_results[s]['test_results']['test_f1'] for s in strategies]
    roc_aucs = [strategies_results[s]['test_results']['test_roc_auc'] for s in strategies]
    pr_aucs = [strategies_results[s]['test_results']['test_pr_auc'] for s in strategies]
    false_positives = [strategies_results[s]['test_results']['false_positives'] for s in strategies]
    false_negatives = [strategies_results[s]['test_results']['false_negatives'] for s in strategies]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Precision vs Recall scatter
    colors = plt.cm.Set1(np.linspace(0, 1, len(strategies)))
    for i, (strategy, precision, recall) in enumerate(zip(strategies, precisions, recalls)):
        axes[0,0].scatter(recall, precision, s=150, c=[colors[i]], label=strategy, alpha=0.8)
        axes[0,0].annotate(strategy.replace('-', '\n'), (recall, precision), 
                          xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    axes[0,0].set_xlabel('Recall')
    axes[0,0].set_ylabel('Precision')
    axes[0,0].set_title('Precision vs Recall Trade-off')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_xlim(0, 1)
    axes[0,0].set_ylim(0, 1)
    
    # 2. Metrics comparison (bar chart)
    x = np.arange(len(strategies))
    width = 0.15  # Reduced width to fit 5 metrics
    
    axes[0,1].bar(x - 2*width, precisions, width, label='Precision', alpha=0.8)
    axes[0,1].bar(x - width, recalls, width, label='Recall', alpha=0.8)
    axes[0,1].bar(x, f1_scores, width, label='F1-Score', alpha=0.8)
    axes[0,1].bar(x + width, roc_aucs, width, label='ROC-AUC', alpha=0.8)
    axes[0,1].bar(x + 2*width, pr_aucs, width, label='PR-AUC', alpha=0.8)
    
    axes[0,1].set_xlabel('Optimization Strategy')
    axes[0,1].set_ylabel('Score')
    axes[0,1].set_title('Performance Metrics Comparison')
    axes[0,1].set_xticks(x)
    axes[0,1].set_xticklabels([s.replace('-', '\n') for s in strategies], rotation=45, ha='right')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Error analysis
    axes[0,2].bar(x - width/2, false_positives, width, label='False Positives', alpha=0.8)
    axes[0,2].bar(x + width/2, false_negatives, width, label='False Negatives', alpha=0.8)
    
    axes[0,2].set_xlabel('Optimization Strategy')
    axes[0,2].set_ylabel('Number of Errors')
    axes[0,2].set_title('Error Type Analysis')
    axes[0,2].set_xticks(x)
    axes[0,2].set_xticklabels([s.replace('-', '\n') for s in strategies], rotation=45, ha='right')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    # 4. ROC curves comparison
    for i, strategy in enumerate(strategies):
        test_labels = strategies_results[strategy]['test_labels']
        test_scores = strategies_results[strategy]['test_results']['test_scores']
        fpr, tpr, _ = roc_curve(test_labels, test_scores)
        auc_score = strategies_results[strategy]['test_results']['test_roc_auc']
        axes[1,0].plot(fpr, tpr, label=f'{strategy} (AUC={auc_score:.3f})', 
                      color=colors[i], linewidth=2)
    
    axes[1,0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[1,0].set_xlabel('False Positive Rate')
    axes[1,0].set_ylabel('True Positive Rate')
    axes[1,0].set_title('ROC Curves Comparison')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 5. PR curves comparison
    for i, strategy in enumerate(strategies):
        test_labels = strategies_results[strategy]['test_labels']
        test_scores = strategies_results[strategy]['test_results']['test_scores']
        precision_curve, recall_curve, _ = precision_recall_curve(test_labels, test_scores)
        pr_auc_score = strategies_results[strategy]['test_results']['test_pr_auc']
        axes[1,1].plot(recall_curve, precision_curve, 
                      label=f'{strategy} (AUC={pr_auc_score:.3f})', 
                      color=colors[i], linewidth=2)
    
    # Add baseline (random classifier)
    spam_ratio = sum(strategies_results[strategies[0]]['test_labels']) / len(strategies_results[strategies[0]]['test_labels'])
    axes[1,1].axhline(y=spam_ratio, color='k', linestyle='--', alpha=0.5, label=f'Random ({spam_ratio:.3f})')
    
    axes[1,1].set_xlabel('Recall')
    axes[1,1].set_ylabel('Precision')
    axes[1,1].set_title('Precision-Recall Curves Comparison')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # 6. Detector Coverage Curves (replaced total errors)
    for i, strategy in enumerate(strategies):
        if strategy in coverage_data:
            coverage_curve = coverage_data[strategy]
            detector_counts = [point['detectors_generated'] for point in coverage_curve]
            recall_values = [point['recall'] for point in coverage_curve]
            
            axes[1,2].plot(detector_counts, recall_values, 'o-', 
                          label=strategy, color=colors[i], linewidth=2, markersize=6)
    
    axes[1,2].set_xlabel('Number of Detectors Generated')
    axes[1,2].set_ylabel('Recall')
    axes[1,2].set_title('Detector Coverage Curves: Recall vs Number of Detectors')
    axes[1,2].legend()
    axes[1,2].grid(True, alpha=0.3)
    axes[1,2].set_ylim(0, 1)
    
    plt.tight_layout()
    save_plot(fig, 'optimization_strategies_comparison.png', output_dir)

def main():
    print("=" * 70)
    print("NSA OPTIMIZATION STRATEGIES COMPARISON")
    print("=" * 70)
    
    # 1. Load and prepare data
    print("\n1. Loading and preparing data...")
    texts, labels = load_data(str(constants.DATA_PATH))
    
    set_seed(constants.SEED)
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = train_val_test_split(
        texts, labels, 
        test_ratio=constants.TEST_RATIO, 
        val_ratio=constants.VAL_RATIO,
        seed=constants.SEED
    )
    
    vocab_list, vocab_index = build_vocabulary(train_texts, 
                                              min_freq=constants.VOCAB_MIN_FREQ, 
                                              max_size=constants.VOCAB_MAX_SIZE)
    
    train_sets = texts_to_sets(train_texts, vocab_index)
    val_sets = texts_to_sets(val_texts, vocab_index)
    test_sets = texts_to_sets(test_texts, vocab_index)
    
    print(f"Data loaded: {len(train_texts)} train, {len(val_texts)} val, {len(test_texts)} test")
    print(f"Vocabulary size: {len(vocab_list)}")
    
    # 2. Run parameter grid search
    print("\n2. Running comprehensive parameter grid search...")
    results_df = run_parameter_grid_search(train_sets, train_labels, val_sets, val_labels, len(vocab_list))
    
    # 3. Find optimal parameters for different strategies
    print("\n3. Finding optimal parameters for different strategies...")
    strategies = find_optimal_parameters(results_df)
    
    # 4. Evaluate each strategy on test set
    print("\n4. Evaluating strategies on test set...")
    strategies_results = {}
    
    for strategy_name, strategy_data in strategies.items():
        print(f"   Evaluating {strategy_name}...")
        test_results = evaluate_strategy_on_test(
            strategy_data['params'], 
            train_sets, train_labels, 
            test_sets, test_labels, 
            len(vocab_list)
        )
        
        strategies_results[strategy_name] = {
            'strategy_info': strategy_data,
            'test_results': test_results,
            'test_labels': test_labels
        }
        
        print(f"     Precision: {test_results['test_precision']:.4f}, "
              f"Recall: {test_results['test_recall']:.4f}, "
              f"F1: {test_results['test_f1']:.4f}")
        print(f"     FP: {test_results['false_positives']}, "
              f"FN: {test_results['false_negatives']}")
    
    # 5. Generate detector coverage curves
    print("\n5. Generating detector coverage curves...")
    best_params_per_strategy = {
        strategy_name: data['strategy_info']['params'] 
        for strategy_name, data in strategies_results.items()
    }
    coverage_data = generate_detector_coverage_curves(
        train_sets, train_labels, test_sets, test_labels, 
        len(vocab_list), best_params_per_strategy
    )
    
    # 6. Create comparison table
    print("\n6. Creating comparison analysis...")
    comparison_df = create_comparison_table(strategies_results)
    
    # 7. Generate plots (including coverage curves)
    print("\n7. Generating comparison plots...")
    output_dir = script_dir.parent / 'results'
    plot_strategy_comparison(strategies_results, coverage_data, output_dir)
    
    # 8. Save results
    print("\n8. Saving results...")
    
    # Save comparison table
    comparison_df.to_csv(output_dir / 'optimization_strategies_comparison.csv', index=False)
    
    # Save detailed results
    detailed_results = {}
    for strategy_name, data in strategies_results.items():
        detailed_results[strategy_name] = {
            'strategy': data['strategy_info']['strategy'],
            'description': data['strategy_info']['description'],
            'optimal_parameters': {
                'num_detectors': int(data['strategy_info']['params']['num_detectors']),
                'detector_size': int(data['strategy_info']['params']['detector_size']),
                'overlap_threshold': int(data['strategy_info']['params']['overlap_threshold']),
                'min_activations': int(data['strategy_info']['params']['min_activations'])
            },
            'test_performance': {
                'accuracy': float(data['test_results']['test_accuracy']),
                'precision': float(data['test_results']['test_precision']),
                'recall': float(data['test_results']['test_recall']),
                'f1_score': float(data['test_results']['test_f1']),
                'roc_auc': float(data['test_results']['test_roc_auc']),
                'pr_auc': float(data['test_results']['test_pr_auc'])
            },
            'error_analysis': {
                'false_positives': int(data['test_results']['false_positives']),
                'false_negatives': int(data['test_results']['false_negatives']),
                'true_positives': int(data['test_results']['true_positives']),
                'true_negatives': int(data['test_results']['true_negatives']),
                'total_errors': int(data['test_results']['total_errors'])
            },
            'detectors_generated': int(data['test_results']['detectors_generated'])
        }
    
    with open(output_dir / 'optimization_strategies_detailed.json', 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    # Save grid search results
    results_df.to_csv(output_dir / 'full_parameter_grid_results.csv', index=False)
    
    # 8. Print summary
    print("\n" + "=" * 70)
    print("OPTIMIZATION STRATEGIES COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\nComparison Table:")
    print(comparison_df.to_string(index=False))
    
    print(f"\n\nFiles saved:")
    print(f"- Comparison table: {(output_dir / 'optimization_strategies_comparison.csv').resolve()}")
    print(f"- Detailed results: {(output_dir / 'optimization_strategies_detailed.json').resolve()}")
    print(f"- Full grid search: {(output_dir / 'full_parameter_grid_results.csv').resolve()}")
    print(f"- Comparison plots: {(output_dir / 'plots' / 'optimization_strategies_comparison.png').resolve()}")
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()