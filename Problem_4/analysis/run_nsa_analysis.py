#!/usr/bin/env python3
"""
Binary NSA F1-Optimization Script
Quick analysis focused on F1-optimization for binary NSA spam detection.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, f1_score
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

# Import our modules - UPDATED FOR BINARY NSA
from preprocessing import load_data, train_val_test_split, texts_to_binary_arrays
from nsa import NegativeSelectionClassifier
import constants

def main():
    print("=" * 60)
    print("BINARY NSA F1-OPTIMIZATION ANALYSIS")
    print("=" * 60)
    
    # 1. Data Loading and Preprocessing
    print("\n1. Loading and preparing data...")
    texts, labels = load_data(str(constants.DATA_PATH))
    
    print(f"Total samples: {len(texts)}")
    print(f"Ham samples: {labels.count(0)} ({labels.count(0)/len(labels)*100:.1f}%)")
    print(f"Spam samples: {labels.count(1)} ({labels.count(1)/len(labels)*100:.1f}%)")
    
    # Split data
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = train_val_test_split(
        texts, labels, seed=42
    )
    
    print(f"\nDataset splits:")
    print(f"Training: {len(train_texts)} ({train_labels.count(0)} ham, {train_labels.count(1)} spam)")
    print(f"Validation: {len(val_texts)} ({val_labels.count(0)} ham, {val_labels.count(1)} spam)")
    print(f"Test: {len(test_texts)} ({test_labels.count(0)} ham, {test_labels.count(1)} spam)")
    
    # Convert to binary features
    print("Converting texts to binary feature vectors...")
    train_binary = texts_to_binary_arrays(train_texts)
    val_binary = texts_to_binary_arrays(val_texts)
    test_binary = texts_to_binary_arrays(test_texts)
    print(f"Binary feature length: {len(train_binary[0])}")
    
    # 2. F1-Optimized Parameter Grid Search (FOCUSED)
    print("\n2. Running focused F1-optimization grid search...")
    
    # Focused parameter grid for quick testing
    param_grid = {
        'num_detectors': [100, 300, 500, 700, 1000, 1500, 2000, 3000],  # Same as other files
        'hamming_threshold': [35, 40, 45, 50, 55, 60, 65, 70],  # Same conservative range
        'min_activations': [1, 2, 3]
    }
    
    print("Parameter grid:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    
    param_combinations = list(itertools.product(*param_grid.values()))
    print(f"Total combinations to test: {len(param_combinations)}")
    
    # Run grid search
    results = []
    param_names = list(param_grid.keys())
    
    print("\\nRunning parameter optimization...")
    for i, params in enumerate(param_combinations):
        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{len(param_combinations)} experiments...")
        
        param_dict = dict(zip(param_names, params))
        
        # Create and train NSA
        nsa = NegativeSelectionClassifier(
            num_detectors=param_dict['num_detectors'],
            hamming_threshold=param_dict['hamming_threshold'],
            min_activations=param_dict['min_activations'],
            max_attempts=20000,
            seed=42
        )
        
        nsa.fit(train_binary, train_labels)
        
        # Quick validation
        val_pred = nsa.predict(val_binary)
        
        # Calculate metrics
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        
        val_precision = precision_score(val_labels, val_pred, zero_division=0)
        val_recall = recall_score(val_labels, val_pred, zero_division=0)
        val_f1 = f1_score(val_labels, val_pred, zero_division=0)
        val_accuracy = accuracy_score(val_labels, val_pred)
        
        result = {
            **param_dict,
            'detectors_generated': len(nsa.detectors),
            'val_accuracy': val_accuracy,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1
        }
        results.append(result)
    
    print("Grid search completed!")
    
    # 3. Find Best F1 Parameters
    results_df = pd.DataFrame(results)
    best_idx = results_df['val_f1'].idxmax()
    best_params = results_df.loc[best_idx]
    
    print(f"\\n3. Best F1-optimized parameters found:")
    print(f"num_detectors: {best_params['num_detectors']:.0f}")
    print(f"hamming_threshold: {best_params['hamming_threshold']:.0f}")
    print(f"min_activations: {best_params['min_activations']:.0f}")
    print(f"Validation F1: {best_params['val_f1']:.4f}")
    print(f"Validation Precision: {best_params['val_precision']:.4f}")
    print(f"Validation Recall: {best_params['val_recall']:.4f}")
    print(f"Detectors Generated: {best_params['detectors_generated']:.0f}")
    
    # 4. Final Model Evaluation
    print("\\n4. Training final model and testing...")
    
    final_nsa = NegativeSelectionClassifier(
        num_detectors=int(best_params['num_detectors']),
        hamming_threshold=int(best_params['hamming_threshold']),
        min_activations=int(best_params['min_activations']),
        max_attempts=20000,
        seed=42
    )
    
    final_nsa.fit(train_binary, train_labels)
    
    # Test set evaluation
    test_pred = final_nsa.predict(test_binary)
    test_pred_scores, test_scores = final_nsa.predict_with_scores(test_binary)
    
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
    
    test_precision = precision_score(test_labels, test_pred, zero_division=0)
    test_recall = recall_score(test_labels, test_pred, zero_division=0)
    test_f1 = f1_score(test_labels, test_pred, zero_division=0)
    test_accuracy = accuracy_score(test_labels, test_pred)
    
    print(f"\\nFinal Test Set Results:")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1-Score: {test_f1:.4f}")
    
    if len(set(test_labels)) > 1:
        test_roc_auc = roc_auc_score(test_labels, test_scores)
        test_pr_auc = average_precision_score(test_labels, test_scores)
        print(f"ROC-AUC: {test_roc_auc:.4f}")
        print(f"PR-AUC: {test_pr_auc:.4f}")
    
    # Classification report
    print(f"\\nDetailed Classification Report:")
    print(classification_report(test_labels, test_pred, target_names=['Ham', 'Spam']))
    
    # 5. Save Results
    output_dir = script_dir.parent / 'results'
    output_dir.mkdir(exist_ok=True)
    
    results_summary = {
        'algorithm': 'Binary NSA',
        'optimization_target': 'F1-Score',
        'dataset_size': len(texts),
        'best_params': {
            'num_detectors': int(best_params['num_detectors']),
            'hamming_threshold': int(best_params['hamming_threshold']),
            'min_activations': int(best_params['min_activations'])
        },
        'test_metrics': {
            'accuracy': float(test_accuracy),
            'precision': float(test_precision),
            'recall': float(test_recall),
            'f1': float(test_f1),
            'roc_auc': float(test_roc_auc) if len(set(test_labels)) > 1 else None,
            'pr_auc': float(test_pr_auc) if len(set(test_labels)) > 1 else None
        },
        'detectors_generated': int(len(final_nsa.detectors))
    }
    
    # Save results
    with open(output_dir / 'binary_nsa_f1_optimization.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    results_df.to_csv(output_dir / 'binary_nsa_f1_grid_results.csv', index=False)
    
    print(f"\\nResults saved:")
    print(f"- {(output_dir / 'binary_nsa_f1_optimization.json').resolve()}")
    print(f"- {(output_dir / 'binary_nsa_f1_grid_results.csv').resolve()}")
    
    # 6. Summary
    print(f"\\n" + "=" * 60)
    print("BINARY NSA F1-OPTIMIZATION SUMMARY")
    print("=" * 60)
    print(f"Best F1-Score: {test_f1:.4f}")
    print(f"Best Precision: {test_precision:.4f}")
    print(f"Best Recall: {test_recall:.4f}")
    print(f"Hamming Threshold: {int(best_params['hamming_threshold'])}")
    print(f"Detectors: {int(len(final_nsa.detectors))}")
    print("=" * 60)

if __name__ == "__main__":
    main()