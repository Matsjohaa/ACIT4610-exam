#!/usr/bin/env python3
"""
NSA Parameter Analysis Script
Run the complete NSA spam detection analysis with parameter optimization.
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
from visualization import plot_confusion
import constants

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

def save_plot(fig, filename, output_dir):
    """Save plot with high DPI for reports"""
    plot_path = output_dir / 'plots' / filename
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {plot_path.resolve()}")  # Show absolute path

def plot_baseline_analysis(test_labels, test_pred, test_scores, roc_auc, pr_auc, output_dir):
    """Create baseline analysis plots"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(test_labels, test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    ax1.set_title('Baseline NSA Confusion Matrix')
    ax1.set_xticklabels(['Ham', 'Spam'])
    ax1.set_yticklabels(['Ham', 'Spam'])
    
    # ROC curve
    fpr, tpr, _ = roc_curve(test_labels, test_scores)
    ax2.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})', linewidth=2)
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve - Baseline NSA')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # PR curve
    precision, recall, _ = precision_recall_curve(test_labels, test_scores)
    ax3.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.3f})', linewidth=2)
    ax3.axhline(y=sum(test_labels)/len(test_labels), color='k', linestyle='--', alpha=0.5, label='Random')
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.set_title('Precision-Recall Curve - Baseline NSA')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_plot(fig, 'baseline_analysis.png', output_dir)
    plt.close()

def plot_parameter_effects(results_df, param_grid, output_dir):
    """Visualize parameter effects on performance"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # F1 vs num_detectors
    for detector_size in param_grid['detector_size']:
        subset = results_df[results_df['detector_size'] == detector_size]
        grouped = subset.groupby('num_detectors')['val_f1'].mean()
        axes[0,0].plot(grouped.index, grouped.values, marker='o', label=f'detector_size={detector_size}')
    axes[0,0].set_xlabel('Number of Detectors')
    axes[0,0].set_ylabel('Validation F1')
    axes[0,0].set_title('F1 vs Number of Detectors')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # F1 vs detector_size
    for overlap_thresh in param_grid['overlap_threshold']:
        subset = results_df[results_df['overlap_threshold'] == overlap_thresh]
        grouped = subset.groupby('detector_size')['val_f1'].mean()
        axes[0,1].plot(grouped.index, grouped.values, marker='o', label=f'overlap_thresh={overlap_thresh}')
    axes[0,1].set_xlabel('Detector Size')
    axes[0,1].set_ylabel('Validation F1')
    axes[0,1].set_title('F1 vs Detector Size')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Precision vs Recall scatter
    scatter = axes[1,0].scatter(results_df['val_recall'], results_df['val_precision'], 
                              c=results_df['val_f1'], cmap='viridis', alpha=0.7)
    axes[1,0].set_xlabel('Validation Recall')
    axes[1,0].set_ylabel('Validation Precision')
    axes[1,0].set_title('Precision vs Recall (colored by F1)')
    plt.colorbar(scatter, ax=axes[1,0])
    axes[1,0].grid(True, alpha=0.3)
    
    # F1 vs overlap_threshold
    for min_act in param_grid['min_activations']:
        subset = results_df[results_df['min_activations'] == min_act]
        grouped = subset.groupby('overlap_threshold')['val_f1'].mean()
        axes[1,1].plot(grouped.index, grouped.values, marker='o', label=f'min_activations={min_act}')
    axes[1,1].set_xlabel('Overlap Threshold')
    axes[1,1].set_ylabel('Validation F1')
    axes[1,1].set_title('F1 vs Overlap Threshold')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_plot(fig, 'parameter_effects.png', output_dir)
    plt.close()

def plot_best_model_evaluation(test_labels, test_pred, test_scores, test_roc_auc, test_pr_auc, output_dir):
    """Comprehensive evaluation plots for best model"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(test_labels, test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
    axes[0,0].set_xlabel('Predicted')
    axes[0,0].set_ylabel('True')
    axes[0,0].set_title('Best Model Confusion Matrix')
    axes[0,0].set_xticklabels(['Ham', 'Spam'])
    axes[0,0].set_yticklabels(['Ham', 'Spam'])
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(test_labels, test_scores)
    axes[0,1].plot(fpr, tpr, label=f'ROC Curve (AUC = {test_roc_auc:.3f})', linewidth=2)
    axes[0,1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0,1].set_xlabel('False Positive Rate')
    axes[0,1].set_ylabel('True Positive Rate')
    axes[0,1].set_title('ROC Curve - Best Model')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(test_labels, test_scores)
    axes[1,0].plot(recall, precision, label=f'PR Curve (AUC = {test_pr_auc:.3f})', linewidth=2)
    axes[1,0].axhline(y=sum(test_labels)/len(test_labels), color='k', linestyle='--', alpha=0.5, label='Random')
    axes[1,0].set_xlabel('Recall')
    axes[1,0].set_ylabel('Precision')
    axes[1,0].set_title('Precision-Recall Curve - Best Model')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Score distribution
    spam_scores = [test_scores[i] for i in range(len(test_scores)) if test_labels[i] == 1]
    ham_scores = [test_scores[i] for i in range(len(test_scores)) if test_labels[i] == 0]
    
    axes[1,1].hist(ham_scores, bins=20, alpha=0.7, label='Ham', density=True)
    axes[1,1].hist(spam_scores, bins=20, alpha=0.7, label='Spam', density=True)
    axes[1,1].set_xlabel('Activation Score')
    axes[1,1].set_ylabel('Density')
    axes[1,1].set_title('Score Distribution by Class')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_plot(fig, 'best_model_evaluation.png', output_dir)
    plt.close()

def plot_coverage_curves(coverage_df, output_dir):
    """Plot detector coverage curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Recall vs Number of Detectors
    ax1.plot(coverage_df['detectors_generated'], coverage_df['recall'], 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Detectors Generated')
    ax1.set_ylabel('Recall on Test Set')
    ax1.set_title('Detector Coverage Curve: Recall vs Number of Detectors')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Trade-off: Precision vs Recall
    for i, row in coverage_df.iterrows():
        ax2.scatter(row['recall'], row['precision'], s=100, alpha=0.7)
        ax2.annotate(f"{int(row['detectors_generated'])}", 
                    (row['recall'], row['precision']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax2.plot(coverage_df['recall'], coverage_df['precision'], '--', alpha=0.5)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Trade-off (numbers show detector count)')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    save_plot(fig, 'detector_coverage_curves.png', output_dir)
    plt.close()

def plot_detector_statistics(detector_overlaps, test_ham_activations, test_spam_activations, best_nsa, vocab_list, output_dir):
    """Visualize detector statistics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Detector overlap distribution
    axes[0,0].hist(detector_overlaps, bins=30, alpha=0.7, edgecolor='black')
    axes[0,0].axvline(np.mean(detector_overlaps), color='red', linestyle='--', 
                     label=f'Mean: {np.mean(detector_overlaps):.3f}')
    axes[0,0].set_xlabel('Jaccard Similarity')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].set_title('Distribution of Detector Pairwise Overlaps')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Activation counts by class (test set)
    bins = range(max(max(test_ham_activations), max(test_spam_activations)) + 2)
    axes[0,1].hist(test_ham_activations, bins=bins, alpha=0.7, label='Ham', density=True)
    axes[0,1].hist(test_spam_activations, bins=bins, alpha=0.7, label='Spam', density=True)
    axes[0,1].set_xlabel('Number of Activations')
    axes[0,1].set_ylabel('Density')
    axes[0,1].set_title('Distribution of Activations by Class (Test Set)')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Detector size distribution (should be constant)
    detector_sizes = [len(det) for det in best_nsa.detectors]
    axes[1,0].hist(detector_sizes, bins=20, alpha=0.7, edgecolor='black')
    axes[1,0].set_xlabel('Detector Size')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title('Distribution of Detector Sizes')
    axes[1,0].grid(True, alpha=0.3)
    
    # Most common tokens in detectors
    token_counts_in_detectors = {}
    vocab_reverse = {v: k for k, v in enumerate(vocab_list)}
    
    for detector in best_nsa.detectors:
        for token_id in detector:
            token = vocab_reverse.get(token_id, f'token_{token_id}')
            if token_id < len(vocab_list):
                token = vocab_list[token_id]
            token_counts_in_detectors[token] = token_counts_in_detectors.get(token, 0) + 1
    
    # Top tokens in detectors
    top_tokens = sorted(token_counts_in_detectors.items(), key=lambda x: x[1], reverse=True)[:15]
    if top_tokens:
        tokens, counts = zip(*top_tokens)
        
        axes[1,1].barh(range(len(tokens)), counts)
        axes[1,1].set_yticks(range(len(tokens)))
        axes[1,1].set_yticklabels(tokens)
        axes[1,1].set_xlabel('Frequency in Detectors')
        axes[1,1].set_title('Most Common Tokens in Detectors')
        axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_plot(fig, 'detector_statistics.png', output_dir)
    plt.close()

def main():
    print("=" * 60)
    print("NSA SPAM DETECTION - PARAMETER ANALYSIS")
    print("=" * 60)
    
    # 1. Data Loading and Preprocessing
    print("\n1. Loading SMS Spam Collection dataset...")
    texts, labels = load_data(str(constants.DATA_PATH))
    
    print(f"Total samples: {len(texts)}")
    print(f"Ham samples: {labels.count(0)}")
    print(f"Spam samples: {labels.count(1)}")
    print(f"Spam ratio: {labels.count(1) / len(labels):.3f}")
    
    # Split data
    set_seed(42)
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = train_val_test_split(
        texts, labels, 
        test_ratio=constants.TEST_RATIO, 
        val_ratio=constants.VAL_RATIO,
        seed=42
    )
    
    print(f"\nDataset splits:")
    print(f"Training: {len(train_texts)} samples ({train_labels.count(0)} ham, {train_labels.count(1)} spam)")
    print(f"Validation: {len(val_texts)} samples ({val_labels.count(0)} ham, {val_labels.count(1)} spam)")
    print(f"Test: {len(test_texts)} samples ({test_labels.count(0)} ham, {test_labels.count(1)} spam)")
    
    # Build vocabulary
    vocab_list, vocab_index = build_vocabulary(
        train_texts, 
        min_freq=constants.VOCAB_MIN_FREQ, 
        max_size=constants.VOCAB_MAX_SIZE
    )
    
    print(f"Vocabulary size: {len(vocab_list)}")
    print(f"Sample vocabulary words: {vocab_list[:10]}")
    
    # Convert texts to token sets
    train_sets = texts_to_sets(train_texts, vocab_index)
    val_sets = texts_to_sets(val_texts, vocab_index)
    test_sets = texts_to_sets(test_texts, vocab_index)
    
    print(f"Average tokens per message (train): {np.mean([len(s) for s in train_sets]):.1f}")
    
    # 2. Baseline NSA Experiment
    print("\n2. Running baseline NSA experiment...")
    
    baseline_nsa = NegativeSelectionClassifier(
        vocab_size=len(vocab_list),
        num_detectors=constants.NSA_NUM_DETECTORS,
        detector_size=constants.NSA_DETECTOR_SIZE,
        overlap_threshold=constants.NSA_OVERLAP_THRESHOLD,
        max_attempts=constants.NSA_MAX_ATTEMPTS,
        min_activations=constants.NSA_MIN_ACTIVATIONS,
        seed=42
    )
    
    baseline_nsa.fit(train_sets, train_labels)
    
    print(f"Detectors generated: {baseline_nsa.detectors_count}/{baseline_nsa.num_detectors}")
    print(f"Training attempts used: {baseline_nsa.attempts_used}/{baseline_nsa.max_attempts}")
    
    # Test predictions
    test_pred, test_scores = baseline_nsa.predict_with_scores(test_sets)
    baseline_metrics = classification_report(test_labels, test_pred)
    
    print(f"\nBaseline Results on Test Set:")
    for metric, value in baseline_metrics.items():
        if isinstance(value, float):
            print(f"{metric.capitalize()}: {value:.4f}")
        else:
            print(f"{metric.capitalize()}: {value}")
    
    roc_auc = roc_auc_score(test_labels, test_scores)
    pr_auc = average_precision_score(test_labels, test_scores)
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    
    # Create baseline plots
    output_dir = script_dir.parent / 'results'
    plot_baseline_analysis(test_labels, test_pred, test_scores, roc_auc, pr_auc, output_dir)
    
    # 3. Parameter Grid Search
    print("\n3. Running parameter grid search...")
    
    param_grid = {
        'num_detectors': [500, 1000, 2000],
        'detector_size': [3, 4, 5],
        'overlap_threshold': [1, 2, 3],
        'min_activations': [1, 2]
    }
    
    print("Parameter grid:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    
    param_combinations = list(itertools.product(*param_grid.values()))
    print(f"Total combinations to test: {len(param_combinations)}")
    
    # Run grid search
    results = []
    param_names = list(param_grid.keys())
    
    for i, params in enumerate(param_combinations):
        print(f"\nExperiment {i+1}/{len(param_combinations)}")
        
        param_dict = dict(zip(param_names, params))
        print(f"Parameters: {param_dict}")
        
        # Create and train NSA
        nsa = NegativeSelectionClassifier(
            vocab_size=len(vocab_list),
            num_detectors=param_dict['num_detectors'],
            detector_size=param_dict['detector_size'],
            overlap_threshold=param_dict['overlap_threshold'],
            min_activations=param_dict['min_activations'],
            max_attempts=constants.NSA_MAX_ATTEMPTS,
            seed=42
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
        
        print(f"F1: {val_metrics['f1']:.4f}, Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}")
    
    print("\nGrid search completed!")
    
    # 4. Analyze results
    results_df = pd.DataFrame(results)
    
    # Create parameter analysis plots
    plot_parameter_effects(results_df, param_grid, output_dir)
    
    # 4. Best Model Evaluation
    results_df = pd.DataFrame(results)
    best_params = results_df.loc[results_df['val_f1'].idxmax()]
    
    print(f"\n4. Best parameters found:")
    print(f"num_detectors: {best_params['num_detectors']:.0f}")
    print(f"detector_size: {best_params['detector_size']:.0f}")
    print(f"overlap_threshold: {best_params['overlap_threshold']:.0f}")
    print(f"min_activations: {best_params['min_activations']:.0f}")
    print(f"Validation F1: {best_params['val_f1']:.4f}")
    
    # Train best model
    best_nsa = NegativeSelectionClassifier(
        vocab_size=len(vocab_list),
        num_detectors=int(best_params['num_detectors']),
        detector_size=int(best_params['detector_size']),
        overlap_threshold=int(best_params['overlap_threshold']),
        min_activations=int(best_params['min_activations']),
        max_attempts=constants.NSA_MAX_ATTEMPTS,
        seed=42
    )
    
    best_nsa.fit(train_sets, train_labels)
    
    # Test set evaluation
    test_pred, test_scores = best_nsa.predict_with_scores(test_sets)
    test_metrics = classification_report(test_labels, test_pred)
    
    print(f"\nBest Model Test Set Results:")
    for metric, value in test_metrics.items():
        if isinstance(value, float):
            print(f"{metric.capitalize()}: {value:.4f}")
        else:
            print(f"{metric.capitalize()}: {value}")
    
    test_roc_auc = roc_auc_score(test_labels, test_scores)
    test_pr_auc = average_precision_score(test_labels, test_scores)
    print(f"ROC-AUC: {test_roc_auc:.4f}")
    print(f"PR-AUC: {test_pr_auc:.4f}")
    
    # Create best model evaluation plots
    plot_best_model_evaluation(test_labels, test_pred, test_scores, test_roc_auc, test_pr_auc, output_dir)
    
    # 5. Detector Coverage Analysis
    print(f"\n5. Detector coverage analysis...")
    
    detector_counts = [50, 100, 200, 500, 1000, 1500, 2000]
    coverage_results = []
    
    for num_det in detector_counts:
        print(f"Testing with {num_det} detectors...")
        
        nsa = NegativeSelectionClassifier(
            vocab_size=len(vocab_list),
            num_detectors=num_det,
            detector_size=int(best_params['detector_size']),
            overlap_threshold=int(best_params['overlap_threshold']),
            min_activations=int(best_params['min_activations']),
            max_attempts=constants.NSA_MAX_ATTEMPTS,
            seed=42
        )
        
        nsa.fit(train_sets, train_labels)
        test_pred_cov, test_scores_cov = nsa.predict_with_scores(test_sets)
        metrics = classification_report(test_labels, test_pred_cov)
        
        coverage_results.append({
            'num_detectors': num_det,
            'detectors_generated': nsa.detectors_count,
            'recall': metrics['recall'],
            'precision': metrics['precision'],
            'f1': metrics['f1']
        })
    
    coverage_df = pd.DataFrame(coverage_results)
    print(f"\nDetector Coverage Results:")
    print(coverage_df)
    
    # Create coverage curve plots
    plot_coverage_curves(coverage_df, output_dir)
    
    # 6. Detector Statistics
    print(f"\n6. Analyzing detector statistics...")
    
    # Detector diversity
    detector_overlaps = []
    for i, det1 in enumerate(best_nsa.detectors):
        for j, det2 in enumerate(best_nsa.detectors[i+1:], i+1):
            overlap = len(det1 & det2) / len(det1 | det2)  # Jaccard similarity
            detector_overlaps.append(overlap)
    
    print(f"Average pairwise detector overlap (Jaccard): {np.mean(detector_overlaps):.4f}")
    
    # Activation statistics
    test_activations = best_nsa.activation_counts(test_sets)
    test_ham_activations = [test_activations[i] for i in range(len(test_activations)) if test_labels[i] == 0]
    test_spam_activations = [test_activations[i] for i in range(len(test_activations)) if test_labels[i] == 1]
    
    print(f"Average activations per ham message: {np.mean(test_ham_activations):.2f}")
    print(f"Average activations per spam message: {np.mean(test_spam_activations):.2f}")
    
    # Create detector statistics plots
    plot_detector_statistics(detector_overlaps, test_ham_activations, test_spam_activations, best_nsa, vocab_list, output_dir)
    
    # 7. Error Analysis
    print(f"\n7. Error analysis...")
    
    # Identify misclassified examples
    false_positives = []  # Ham predicted as spam
    false_negatives = []  # Spam predicted as ham
    true_positives = []   # Spam predicted as spam
    true_negatives = []   # Ham predicted as ham
    
    for i, (true_label, pred_label, score) in enumerate(zip(test_labels, test_pred, test_scores)):
        example = {
            'index': i,
            'text': test_texts[i],
            'true_label': true_label,
            'pred_label': pred_label,
            'score': score
        }
        
        if true_label == 0 and pred_label == 1:
            false_positives.append(example)
        elif true_label == 1 and pred_label == 0:
            false_negatives.append(example)
        elif true_label == 1 and pred_label == 1:
            true_positives.append(example)
        else:
            true_negatives.append(example)
    
    print(f"False Positives: {len(false_positives)}")
    print(f"False Negatives: {len(false_negatives)}")
    print(f"True Positives: {len(true_positives)}")
    print(f"True Negatives: {len(true_negatives)}")
    
    # Show examples
    if false_positives:
        print(f"\n=== FALSE POSITIVES (Ham predicted as Spam) ===")
        for i, fp in enumerate(sorted(false_positives, key=lambda x: x['score'], reverse=True)[:5]):
            print(f"{i+1}. Score: {fp['score']}, Text: {fp['text'][:100]}...")
    
    if false_negatives:
        print(f"\n=== FALSE NEGATIVES (Spam predicted as Ham) ===")
        for i, fn in enumerate(sorted(false_negatives, key=lambda x: x['score'])[:5]):
            print(f"{i+1}. Score: {fn['score']}, Text: {fn['text'][:100]}...")
    
    # 8. Save Results
    print(f"\n8. Saving results...")
    
    results_summary = {
        'dataset': 'SMS Spam Collection',
        'total_samples': len(texts),
        'train_samples': len(train_texts),
        'val_samples': len(val_texts),
        'test_samples': len(test_texts),
        'spam_ratio': labels.count(1) / len(labels),
        'vocab_size': len(vocab_list),
        'best_params': {
            'num_detectors': int(best_params['num_detectors']),
            'detector_size': int(best_params['detector_size']),
            'overlap_threshold': int(best_params['overlap_threshold']),
            'min_activations': int(best_params['min_activations'])
        },
        'test_metrics': {
            'accuracy': float(test_metrics['accuracy']),
            'precision': float(test_metrics['precision']),
            'recall': float(test_metrics['recall']),
            'f1': float(test_metrics['f1']),
            'roc_auc': float(test_roc_auc),
            'pr_auc': float(test_pr_auc)
        },
        'detector_stats': {
            'detectors_generated': int(best_nsa.detectors_count),
            'avg_pairwise_overlap': float(np.mean(detector_overlaps)),
            'avg_activations_ham': float(np.mean(test_ham_activations)),
            'avg_activations_spam': float(np.mean(test_spam_activations))
        }
    }
    
    # Save to file
    output_dir = script_dir.parent / 'results'
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'nsa_analysis_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    results_df.to_csv(output_dir / 'parameter_grid_results.csv', index=False)
    coverage_df.to_csv(output_dir / 'detector_coverage_curve.csv', index=False)
    
    print(f"Results saved to:")
    print(f"- {(output_dir / 'nsa_analysis_summary.json').resolve()}")
    print(f"- {(output_dir / 'parameter_grid_results.csv').resolve()}")
    print(f"- {(output_dir / 'detector_coverage_curve.csv').resolve()}")
    
    # 9. Summary
    print(f"\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Dataset: SMS Spam Collection ({len(texts)} samples)")
    print(f"Best F1-Score: {test_metrics['f1']:.4f}")
    print(f"Best Parameters: {dict(best_params[['num_detectors', 'detector_size', 'overlap_threshold', 'min_activations']])}")
    print(f"Detectors Generated: {best_nsa.detectors_count}")
    print(f"ROC-AUC: {test_roc_auc:.4f}")
    print(f"PR-AUC: {test_pr_auc:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()