#!/usr/bin/env python3
"""
GENERATES DETECTOR COVERAGE ANALYSIS VS RECALL PLOT
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['lines.markersize'] = 8

def load_results():
    """Load comprehensive results"""
    results_path = Path(__file__).parent.parent / 'results' / 'comprehensive_results.csv'
    df = pd.read_csv(results_path)
    return df

def analyze_detector_scaling(df):
    """
    Analyze how recall scales with detector count
    For each detector size, find:
    - Maximum recall achieved (best case)
    - Average recall across all configs
    - Recall at different percentiles (25th, 50th, 75th)
    """
    detector_sizes = sorted(df['num_detectors'].unique())
    
    stats = []
    for size in detector_sizes:
        subset = df[df['num_detectors'] == size]
        
        stats.append({
            'detectors': size,
            'num_configs': len(subset),
            'max_recall': subset['recall'].max(),
            'mean_recall': subset['recall'].mean(),
            'median_recall': subset['recall'].median(),
            'q25_recall': subset['recall'].quantile(0.25),
            'q75_recall': subset['recall'].quantile(0.75),
            'std_recall': subset['recall'].std(),
            'max_f1': subset['f1'].max(),
            'mean_f1': subset['f1'].mean()
        })
    
    stats_df = pd.DataFrame(stats)
    return stats_df

def plot_detector_coverage(stats_df):
    """
    Create a comprehensive detector coverage plot showing:
    1. Maximum recall line (best possible)
    2. Mean recall line (expected performance)
    3. Shaded region showing variance (25th to 75th percentile)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    detectors = stats_df['detectors']
    
    # Plot maximum recall (upper bound)
    ax.plot(detectors, stats_df['max_recall'], 
            marker='o', linewidth=2.5, markersize=10,
            color='#d62728', label='Maximum Recall', zorder=3)
    
    # Plot mean recall (expected performance)
    ax.plot(detectors, stats_df['mean_recall'], 
            marker='s', linewidth=2.5, markersize=9,
            color='#2ca02c', label='Mean Recall', zorder=3)
    
    # Plot median recall
    ax.plot(detectors, stats_df['median_recall'], 
            marker='^', linewidth=2, markersize=8,
            color='#1f77b4', label='Median Recall', alpha=0.8, zorder=2)
    
    # Shaded region for 25th to 75th percentile (interquartile range)
    ax.fill_between(detectors, 
                    stats_df['q25_recall'], 
                    stats_df['q75_recall'],
                    alpha=0.2, color='#1f77b4', 
                    label='25th-75th Percentile Range')
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Annotate key insights
    # Find plateau point (where improvement becomes marginal)
    recall_improvements = stats_df['mean_recall'].diff()
    if len(recall_improvements) > 1:
        plateau_idx = 1  # Start checking from 2nd point
        for i in range(1, len(recall_improvements)):
            if recall_improvements.iloc[i] < 0.03:  # Less than 3% improvement
                plateau_idx = i
                break
        
        plateau_detectors = stats_df['detectors'].iloc[plateau_idx]
        plateau_recall = stats_df['mean_recall'].iloc[plateau_idx]
        
        ax.annotate(f'Plateau at {plateau_detectors:,} detectors\n(Mean recall: {plateau_recall:.3f})',
                   xy=(plateau_detectors, plateau_recall),
                   xytext=(plateau_detectors + 500, plateau_recall - 0.08),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                   fontsize=10, ha='left',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    # Formatting
    ax.set_xlabel('Number of Detectors', fontweight='bold')
    ax.set_ylabel('Recall (Spam Detection Rate)', fontweight='bold')
    ax.set_title('Detector Set Size vs. Recall Performance\n(Aggregated across all r-contiguous and n-gram configurations)',
                fontweight='bold', pad=20)
    
    # Set x-axis to show detector counts clearly
    ax.set_xticks(detectors)
    ax.set_xticklabels([f'{int(d/1000)}k' if d >= 1000 else str(d) for d in detectors])
    
    # Set y-axis limits for better visibility
    y_min = max(0.5, stats_df['q25_recall'].min() - 0.05)
    y_max = min(1.0, stats_df['max_recall'].max() + 0.05)
    ax.set_ylim(y_min, y_max)
    
    # Format y-axis as percentages
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    
    # Add legend
    ax.legend(loc='lower right', framealpha=0.9)
    
    # Tight layout
    plt.tight_layout()
    
    return fig

def plot_f1_comparison(stats_df):
    """
    Create a bar plot comparing F1 scores across detector sizes
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(stats_df))
    width = 0.35
    
    # Plot max and mean F1
    bars1 = ax.bar(x - width/2, stats_df['max_f1'], width, 
                   label='Maximum F1', color='#d62728', alpha=0.8)
    bars2 = ax.bar(x + width/2, stats_df['mean_f1'], width,
                   label='Mean F1', color='#2ca02c', alpha=0.8)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    # Formatting
    ax.set_xlabel('Number of Detectors', fontweight='bold')
    ax.set_ylabel('F1-Score', fontweight='bold')
    ax.set_title('F1-Score Performance Across Detector Set Sizes',
                fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(d/1000)}k' for d in stats_df['detectors']])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    
    return fig

def print_statistics(stats_df):
    """Print detailed statistics"""
    print("\n" + "="*80)
    print("DETECTOR COVERAGE ANALYSIS")
    print("="*80)
    
    print("\nRecall Performance by Detector Set Size:")
    print("-" * 80)
    print(f"{'Detectors':<12} {'Configs':<10} {'Max Recall':<12} {'Mean Recall':<12} "
          f"{'Median':<10} {'Std Dev':<10}")
    print("-" * 80)
    
    for _, row in stats_df.iterrows():
        print(f"{int(row['detectors']):>10,}  {int(row['num_configs']):>8}  "
              f"{row['max_recall']:>10.3f}  {row['mean_recall']:>11.3f}  "
              f"{row['median_recall']:>8.3f}  {row['std_recall']:>8.3f}")
    
    print("\n" + "-" * 80)
    print("\nF1-Score Performance:")
    print("-" * 80)
    print(f"{'Detectors':<12} {'Max F1':<12} {'Mean F1':<12}")
    print("-" * 80)
    
    for _, row in stats_df.iterrows():
        print(f"{int(row['detectors']):>10,}  {row['max_f1']:>10.3f}  {row['mean_f1']:>10.3f}")
    
    # Calculate improvement rates
    print("\n" + "-" * 80)
    print("\nMarginal Improvements (Mean Recall):")
    print("-" * 80)
    
    for i in range(1, len(stats_df)):
        prev = stats_df.iloc[i-1]
        curr = stats_df.iloc[i]
        improvement = curr['mean_recall'] - prev['mean_recall']
        pct_improvement = (improvement / prev['mean_recall']) * 100
        
        print(f"{int(prev['detectors']):,} → {int(curr['detectors']):,}: "
              f"+{improvement:+.3f} ({pct_improvement:+.1f}%)")
    
    print("="*80)

def main():
    """Main execution"""
    print("Loading comprehensive results...")
    df = load_results()
    
    print(f"Loaded {len(df)} configurations")
    print(f"Detector sizes: {sorted(df['num_detectors'].unique())}")
    
    # Analyze detector scaling
    stats_df = analyze_detector_scaling(df)
    
    # Print statistics
    print_statistics(stats_df)
    
    # Create plots
    print("\nGenerating detector coverage plot...")
    fig1 = plot_detector_coverage(stats_df)
    
    output_dir = Path(__file__).parent.parent / 'results' / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'detector_coverage_curve.png'
    fig1.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    print("\nGenerating F1 comparison plot...")
    fig2 = plot_f1_comparison(stats_df)
    
    output_path2 = output_dir / 'detector_f1_comparison.png'
    fig2.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path2}")
    
    plt.show()
    
    print("\nAnalysis complete!")

if __name__ == '__main__':
    main()
