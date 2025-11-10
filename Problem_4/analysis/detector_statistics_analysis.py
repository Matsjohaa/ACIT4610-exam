#!/usr/bin/env python3
"""
Detector Statistics Analysis Script
Analyzes detector generation statistics, success rates, and matching patterns
using data from multiple runs optimization results.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import pandas as pd
import seaborn as sns
import sys

# Add the src directory to the path so we can import modules
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import constants
from preprocessing import load_data, train_val_test_split, build_vocabulary, texts_to_sets
from nsa import NegativeSelectionClassifier
from utils import set_seed, classification_report

def load_statistics(stats_file):
    """Load the existing statistics from JSON file"""
    with open(stats_file, 'r') as f:
        return json.load(f)

def extract_detector_statistics(statistics):
    """Extract detector-related statistics from the JSON data"""
    detector_stats = {}
    
    for strategy, stats in statistics.items():
        detector_stats[strategy] = {
            'detectors_generated': {
                'mean': stats['detectors_generated']['mean'],
                'std': stats['detectors_generated']['std'],
                'min': stats['detectors_generated']['min'],
                'max': stats['detectors_generated']['max']
            },
            'detector_size': {
                'mean': stats['detector_size']['mean'],
                'std': stats['detector_size']['std']
            },
            'overlap_threshold': {
                'mean': stats['overlap_threshold']['mean']
            },
            'min_activations': {
                'mean': stats['min_activations']['mean'],
                'std': stats['min_activations']['std']
            }
        }
    
    return detector_stats

def calculate_detector_success_rate(detector_stats):
    """Calculate detector generation success rates based on max_attempts constraint"""
    # The actual success rate should be calculated as: detectors_generated / max_attempts
    # But since we don't have max_attempts data, we'll estimate based on realistic constraints
    
    success_rates = {}
    for strategy, stats in detector_stats.items():
        mean_generated = stats['detectors_generated']['mean']
        max_generated = stats['detectors_generated']['max']
        min_generated = stats['detectors_generated']['min']
        
        # Estimate success rate based on the assumption that max_attempts = 20000 (from constants)
        # and that variation in detectors_generated reflects success rate differences
        estimated_attempts = 20000  # From constants.NSA_MAX_ATTEMPTS
        estimated_success_rate = mean_generated / estimated_attempts
        
        success_rates[strategy] = {
            'mean_generated': mean_generated,
            'estimated_attempts': estimated_attempts,
            'estimated_success_rate': min(estimated_success_rate, 1.0),
            'generation_consistency': 1.0 - (stats['detectors_generated']['std'] / mean_generated) if mean_generated > 0 else 0
        }
    
    return success_rates

def analyze_matching_patterns(statistics_file, detector_stats, output_dir):
    """Analyze average matches per spam/ham on validation set using representative parameters.
    
    Note: This is an illustrative analysis that creates new NSA models with average 
    parameters from the 10-run experiments, rather than reusing the exact same models.
    Results provide representative insights into detector behavior patterns.
    """
    print("Analyzing matching patterns on validation set...")
    print("Note: Using representative parameters from 10-run analysis for illustration")
    
    # Load and prepare data
    texts, labels = load_data(str(constants.DATA_PATH))
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = train_val_test_split(
        texts, labels, 
        test_ratio=constants.TEST_RATIO, 
        val_ratio=constants.VAL_RATIO,
        seed=42
    )
    
    vocab_list, vocab_index = build_vocabulary(train_texts, 
                                             min_freq=constants.VOCAB_MIN_FREQ, 
                                             max_size=constants.VOCAB_MAX_SIZE)
    
    train_sets = texts_to_sets(train_texts, vocab_index)
    val_sets = texts_to_sets(val_texts, vocab_index)
    
    # Separate validation data by class
    val_ham_sets = [s for s, label in zip(val_sets, val_labels) if label == 0]
    val_spam_sets = [s for s, label in zip(val_sets, val_labels) if label == 1]
    
    matching_analysis = {}
    
    for strategy, stats in detector_stats.items():
        print(f"  Analyzing {strategy}...")
        
        # Use average parameters for this strategy
        params = {
            'num_detectors': int(stats['detectors_generated']['mean']),
            'detector_size': int(round(stats['detector_size']['mean'])),
            'overlap_threshold': int(stats['overlap_threshold']['mean']),
            'min_activations': int(round(stats['min_activations']['mean']))
        }
        
        # Create NSA with these parameters and analyze 3 runs
        matches_per_ham = []
        matches_per_spam = []
        
        for run in range(3):  # Analyze 3 runs for speed
            set_seed(42 + run)
            nsa = NegativeSelectionClassifier(
                vocab_size=len(vocab_list),
                **params,
                max_attempts=constants.NSA_MAX_ATTEMPTS,
                seed=None
            )
            
            try:
                nsa.fit(train_sets, train_labels)
                
                # Calculate matches per message (not per detector)
                ham_activations = [sum(1 for d in nsa.detectors if nsa._matches(s, d)) for s in val_ham_sets]
                spam_activations = [sum(1 for d in nsa.detectors if nsa._matches(s, d)) for s in val_spam_sets]
                
                matches_per_ham.extend(ham_activations)
                matches_per_spam.extend(spam_activations)
                
            except Exception as e:
                print(f"    Error in run {run}: {e}")
                continue
        
        if matches_per_ham and matches_per_spam:
            ham_mean = np.mean(matches_per_ham)
            spam_mean = np.mean(matches_per_spam)
            
            # Calculate proper discrimination ratio
            if ham_mean > 0:
                discrimination_ratio = spam_mean / ham_mean
            else:
                discrimination_ratio = float('inf')  # Perfect discrimination if no ham activations
                
            matching_analysis[strategy] = {
                'avg_matches_ham': ham_mean,
                'std_matches_ham': np.std(matches_per_ham),
                'avg_matches_spam': spam_mean,
                'std_matches_spam': np.std(matches_per_spam),
                'discrimination_ratio': discrimination_ratio
            }
    
    return matching_analysis

def create_detector_analysis_plots(detector_stats, success_rates, matching_analysis, output_dir):
    """Create comprehensive detector analysis plots"""
    
    # Set up the figure with multiple subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    strategies = list(detector_stats.keys())
    
    # Plot 1: Detectors Generated vs Strategy
    detector_means = [detector_stats[s]['detectors_generated']['mean'] for s in strategies]
    detector_stds = [detector_stats[s]['detectors_generated']['std'] for s in strategies]
    
    bars1 = ax1.bar(range(len(strategies)), detector_means, yerr=detector_stds, 
                    capsize=5, alpha=0.7, color='skyblue', edgecolor='navy')
    ax1.set_xlabel('Optimization Strategy', fontweight='bold')
    ax1.set_ylabel('Number of Detectors Generated', fontweight='bold')
    ax1.set_title('Average Detectors Generated per Strategy', fontweight='bold')
    ax1.set_xticks(range(len(strategies)))
    ax1.set_xticklabels(strategies, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mean in zip(bars1, detector_means):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(detector_stds)/20,
                f'{mean:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Success Rates (Estimated)
    if success_rates:
        success_rate_values = [success_rates[s]['estimated_success_rate'] for s in strategies if s in success_rates]
        success_strategies = [s for s in strategies if s in success_rates]
        
        bars2 = ax2.bar(range(len(success_strategies)), success_rate_values, 
                        alpha=0.7, color='lightgreen', edgecolor='darkgreen')
        ax2.set_xlabel('Optimization Strategy', fontweight='bold')
        ax2.set_ylabel('Estimated Success Rate', fontweight='bold')
        ax2.set_title('Detector Generation Efficiency (Estimated)', fontweight='bold')
        ax2.set_xticks(range(len(success_strategies)))
        ax2.set_xticklabels(success_strategies, rotation=45, ha='right')
        ax2.set_ylim(0, max(success_rate_values) * 1.2)
        ax2.grid(True, alpha=0.3)
        
        # Add percentage labels
        for bar, rate in zip(bars2, success_rate_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(success_rate_values)/50,
                    f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Average Matches per Message Type
    if matching_analysis:
        match_strategies = list(matching_analysis.keys())
        ham_matches = [matching_analysis[s]['avg_matches_ham'] for s in match_strategies]
        spam_matches = [matching_analysis[s]['avg_matches_spam'] for s in match_strategies]
        
        x = np.arange(len(match_strategies))
        width = 0.35
        
        bars3a = ax3.bar(x - width/2, ham_matches, width, label='Ham Messages', 
                        alpha=0.7, color='lightcoral')
        bars3b = ax3.bar(x + width/2, spam_matches, width, label='Spam Messages', 
                        alpha=0.7, color='gold')
        
        ax3.set_xlabel('Optimization Strategy', fontweight='bold')
        ax3.set_ylabel('Average Detector Matches per Message', fontweight='bold')
        ax3.set_title('Detector Matching Patterns by Message Type', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(match_strategies, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Discrimination Ratio
    if matching_analysis:
        discrimination_ratios = [matching_analysis[s]['discrimination_ratio'] for s in match_strategies]
        
        bars4 = ax4.bar(range(len(match_strategies)), discrimination_ratios, 
                       alpha=0.7, color='mediumpurple', edgecolor='indigo')
        ax4.set_xlabel('Optimization Strategy', fontweight='bold')
        ax4.set_ylabel('Discrimination Ratio (Spam/Ham Matches)', fontweight='bold')
        ax4.set_title('Detector Discrimination Capability', fontweight='bold')
        ax4.set_xticks(range(len(match_strategies)))
        ax4.set_xticklabels(match_strategies, rotation=45, ha='right')
        ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Equal Matching')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add ratio labels
        for bar, ratio in zip(bars4, discrimination_ratios):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + max(discrimination_ratios)/50,
                    f'{ratio:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = output_dir / 'detector_statistics_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Detector statistics plot saved to: {output_path}")
    
    plt.show()

def generate_statistics_summary(detector_stats, success_rates, matching_analysis, output_dir):
    """Generate a comprehensive statistics summary"""
    
    print("\n" + "="*80)
    print("DETECTOR STATISTICS ANALYSIS SUMMARY")
    print("="*80)
    
    # Create summary table
    summary_data = []
    
    for strategy in detector_stats.keys():
        row = {
            'Strategy': strategy,
            'Detectors (Mean ± SD)': f"{detector_stats[strategy]['detectors_generated']['mean']:.1f} ± {detector_stats[strategy]['detectors_generated']['std']:.1f}",
            'Detector Size (Mean)': f"{detector_stats[strategy]['detector_size']['mean']:.1f}",
            'Est. Success Rate': f"{success_rates.get(strategy, {}).get('estimated_success_rate', 0):.1%}" if success_rates else "N/A"
        }
        
        if matching_analysis and strategy in matching_analysis:
            row.update({
                'Ham Matches (Avg)': f"{matching_analysis[strategy]['avg_matches_ham']:.2f}",
                'Spam Matches (Avg)': f"{matching_analysis[strategy]['avg_matches_spam']:.2f}",
                'Discrimination Ratio': f"{matching_analysis[strategy]['discrimination_ratio']:.2f}"
            })
        
        summary_data.append(row)
    
    # Create DataFrame and save
    df_summary = pd.DataFrame(summary_data)
    
    print("\nDetector Generation Summary:")
    print(df_summary.to_string(index=False))
    
    # Save to CSV
    summary_path = output_dir / 'detector_statistics_summary.csv'
    df_summary.to_csv(summary_path, index=False)
    print(f"\nDetector statistics summary saved to: {summary_path}")
    
    # Additional analysis
    if matching_analysis:
        print("\n" + "-"*60)
        print("MATCHING PATTERN ANALYSIS")
        print("-"*60)
        print("\nNote: Discrimination ratios calculated from illustrative runs using")
        print("representative parameters. Values may differ slightly from displayed")
        print("averages due to randomness in detector generation.")
        
        for strategy, data in matching_analysis.items():
            print(f"\n{strategy}:")
            print(f"  Average matches per ham message: {data['avg_matches_ham']:.2f} ± {data['std_matches_ham']:.2f}")
            print(f"  Average matches per spam message: {data['avg_matches_spam']:.2f} ± {data['std_matches_spam']:.2f}")
            print(f"  Discrimination ratio (spam/ham): {data['discrimination_ratio']:.2f}")
            
            if data['discrimination_ratio'] > 2.0:
                print(f"  → Excellent discrimination (spam gets {data['discrimination_ratio']:.1f}x more matches)")
            elif data['discrimination_ratio'] > 1.5:
                print(f"  → Good discrimination")
            else:
                print(f"  → Limited discrimination ability")

def main():
    """Main function to run detector statistics analysis"""
    print("="*80)
    print("DETECTOR STATISTICS ANALYSIS")
    print("="*80)
    
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    results_dir = project_dir / 'results'
    plots_dir = results_dir / 'plots'
    statistics_file = results_dir / 'multiple_runs_statistics.json'
    
    # Ensure output directory exists
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and analyze data
    print("1. Loading statistics from multiple runs...")
    statistics = load_statistics(statistics_file)
    
    print("2. Extracting detector statistics...")
    detector_stats = extract_detector_statistics(statistics)
    
    print("3. Calculating detector success rates...")
    success_rates = calculate_detector_success_rate(detector_stats)
    
    print("4. Analyzing matching patterns...")
    matching_analysis = analyze_matching_patterns(statistics_file, detector_stats, plots_dir)
    
    print("5. Creating visualization plots...")
    create_detector_analysis_plots(detector_stats, success_rates, matching_analysis, plots_dir)
    
    print("6. Generating summary report...")
    generate_statistics_summary(detector_stats, success_rates, matching_analysis, results_dir)
    
    print("\n" + "="*80)
    print("DETECTOR STATISTICS ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
