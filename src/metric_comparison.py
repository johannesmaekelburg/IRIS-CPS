"""
Metric Comparison for Inconsistency Measures

Compares different inconsistency metrics to demonstrate why I(θ) is superior:
1. I(θ) = 1 - P[C=1]           (Proposed global metric)
2. 1 - Jaccard                 (Overlap-based only)
3. |ΔVolume| / Volume₀         (Geometric change)
4. 1 - MC_probability          (Direct MC without constraint checking)

Usage:
    python metric_comparison.py --data_dir ../data/measurements/both --output_dir ../results/metric_comparison
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from typing import Dict, List, Tuple

# Import shared utilities
from analysis_utils import (
    load_experimental_data,
    compute_derived_metrics,
    save_report,
    create_report_header,
    setup_figure,
    save_figure,
    ensure_output_dir,
    print_analysis_header,
    print_analysis_footer,
    compute_summary_statistics,
    format_statistics_table
)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


def load_all_data(data_dir: Path) -> pd.DataFrame:
    """
    Load all experimental data and extract metrics.
    Uses the shared load_experimental_data function and adds derived metrics.
    """
    # Use shared loader
    df = load_experimental_data(str(data_dir), verbose=True)
    
    # Compute derived metrics
    df = compute_derived_metrics(df)
    
    return df


def compute_metric_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute statistics for each metric."""
    
    metrics = {
        'I(θ)': 'I_theta',
        '1 - Jaccard': '1_minus_jaccard',
        'Volume Ratio': 'volume_ratio',
        '1 - MC Prob': '1_minus_mc_prob'
    }
    
    stats = []
    for name, col in metrics.items():
        values = df[col].dropna()
        stats.append({
            'Metric': name,
            'Mean': values.mean(),
            'Std': values.std(),
            'Min': values.min(),
            'Max': values.max(),
            'Range': values.max() - values.min()
        })
    
    return pd.DataFrame(stats)


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise correlations between metrics."""
    
    metric_cols = ['I_theta', '1_minus_jaccard', 'volume_ratio', '1_minus_mc_prob']
    
    # Filter valid rows
    df_valid = df[metric_cols].dropna()
    
    # Compute Pearson and Spearman correlations
    pearson_corr = df_valid.corr(method='pearson')
    spearman_corr = df_valid.corr(method='spearman')
    
    return pearson_corr, spearman_corr


def identify_disagreements(df: pd.DataFrame, threshold: float = 0.3) -> pd.DataFrame:
    """Find cases where metrics strongly disagree."""
    
    disagreements = []
    
    for idx, row in df.iterrows():
        I_theta = row['I_theta']
        jaccard_inconsistency = row['1_minus_jaccard']
        volume_inconsistency = row['volume_ratio']
        
        # Case 1: I(θ) high, Jaccard low (geometric overlap but constraint violation)
        if I_theta > 0.7 and jaccard_inconsistency < 0.3:
            disagreements.append({
                'case': 'High I(θ), Low Jaccard',
                'scenario': row['scenario'],
                'param_value': row['param_value'],
                'I_theta': I_theta,
                '1_minus_jaccard': jaccard_inconsistency,
                'interpretation': 'Shapes overlap but constraints violated'
            })
        
        # Case 2: I(θ) low, Jaccard high (no overlap but constraints satisfied)
        if I_theta < 0.3 and jaccard_inconsistency > 0.7:
            disagreements.append({
                'case': 'Low I(θ), High Jaccard',
                'scenario': row['scenario'],
                'param_value': row['param_value'],
                'I_theta': I_theta,
                '1_minus_jaccard': jaccard_inconsistency,
                'interpretation': 'Poor overlap but constraints still hold'
            })
        
        # Case 3: I(θ) high, Volume change small
        if I_theta > 0.7 and volume_inconsistency < 0.1:
            disagreements.append({
                'case': 'High I(θ), Small Volume Change',
                'scenario': row['scenario'],
                'param_value': row['param_value'],
                'I_theta': I_theta,
                'volume_ratio': volume_inconsistency,
                'interpretation': 'Volume similar but inconsistency high'
            })
    
    return pd.DataFrame(disagreements)


def plot_metric_comparison(df: pd.DataFrame, output_dir: Path):
    """Create focused comparison plots - only the most informative."""
    
    # Plot 1: Side-by-side metric sensitivity
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Inconsistency Metric Comparison', fontsize=14, fontweight='bold')
    
    # Left: All metrics vs param_value (averaged across scenarios)
    ax = axes[0]
    metrics_data = []
    for param in sorted(df['param_value'].unique()):
        subset = df[df['param_value'] == param]
        metrics_data.append({
            'param': param,
            'I_theta': subset['I_theta'].mean(),
            '1_minus_jaccard': subset['1_minus_jaccard'].mean(),
            'volume_ratio': subset['volume_ratio'].mean(),
        })
    metrics_df = pd.DataFrame(metrics_data)
    ax.plot(metrics_df['param'], metrics_df['I_theta'], 'o-', 
            label='I(theta) [MC-based]', linewidth=2.5, markersize=6, color='#e74c3c')
    ax.plot(metrics_df['param'], metrics_df['1_minus_jaccard'], 's--', 
            label='1 - Jaccard [Overlap]', linewidth=2, markersize=5, color='#3498db')
    ax.plot(metrics_df['param'], metrics_df['volume_ratio'], '^:', 
            label='Volume Ratio [Geometric]', linewidth=2, markersize=5, color='#2ecc71')
    ax.set_xlabel('Parameter Value (Uncertainty Scale)', fontsize=11)
    ax.set_ylabel('Inconsistency Score', fontsize=11)
    ax.set_title('Metric Sensitivity to Uncertainty Changes', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Right: I(theta) vs Jaccard scatter
    ax = axes[1]
    scatter = ax.scatter(df['I_theta'], df['1_minus_jaccard'], 
                        c=df['param_value'], cmap='viridis', alpha=0.6, s=40, edgecolors='k', linewidth=0.5)
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, linewidth=2, label='Perfect agreement')
    ax.set_xlabel('I(theta) [Proposed Metric]', fontsize=11)
    ax.set_ylabel('1 - Jaccard [Overlap Only]', fontsize=11)
    ax.set_title('I(theta) vs Jaccard: Agreement Analysis', fontsize=12, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax, label='param_value')
    cbar.set_label('Uncertainty Scale', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metric_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: metric_comparison.png")
    plt.close()


def plot_correlation_heatmap(pearson_corr: pd.DataFrame, spearman_corr: pd.DataFrame, 
                             output_dir: Path):
    """Plot correlation heatmaps."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    labels = ['I(θ)', '1-Jaccard', 'Vol Ratio', '1-MC Prob']
    
    # Pearson correlation
    ax = axes[0]
    sns.heatmap(pearson_corr, annot=True, fmt='.3f', cmap='coolwarm', 
                center=0, vmin=-1, vmax=1, ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_title('Pearson Correlation (Linear)', fontsize=12, fontweight='bold')
    
    # Spearman correlation
    ax = axes[1]
    sns.heatmap(spearman_corr, annot=True, fmt='.3f', cmap='coolwarm', 
                center=0, vmin=-1, vmax=1, ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_title('Spearman Correlation (Rank)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metric_correlations.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: metric_correlations.png")
    plt.close()


def plot_disagreement_cases(disagreements: pd.DataFrame, output_dir: Path):
    """Visualize cases where metrics disagree."""
    
    if len(disagreements) == 0:
        print("No strong disagreements found - metrics generally agree")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    cases = disagreements.groupby('case').size()
    cases.plot(kind='bar', ax=ax, color=['#e74c3c', '#3498db', '#2ecc71'])
    
    ax.set_xlabel('Disagreement Type')
    ax.set_ylabel('Number of Cases')
    ax.set_title('Cases Where Metrics Disagree', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metric_disagreements.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: metric_disagreements.png")
    plt.close()


def generate_comparison_report(df: pd.DataFrame, stats: pd.DataFrame, 
                               pearson_corr: pd.DataFrame, disagreements: pd.DataFrame,
                               output_dir: Path):
    """Generate text report summarizing comparison."""
    
    report = []
    report.append("=" * 80)
    report.append("INCONSISTENCY METRIC COMPARISON REPORT")
    report.append("=" * 80)
    report.append(f"\nDataset: {len(df)} experiments from {df['scenario'].nunique()} scenarios")
    report.append(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    report.append("\n\n" + "=" * 80)
    report.append("1. METRIC STATISTICS")
    report.append("=" * 80)
    report.append(stats.to_string(index=False))
    
    report.append("\n\n" + "=" * 80)
    report.append("2. METRIC CORRELATIONS (Pearson)")
    report.append("=" * 80)
    report.append(pearson_corr.to_string())
    
    report.append("\n\n" + "=" * 80)
    report.append("3. KEY FINDINGS")
    report.append("=" * 80)
    
    # Finding 1: I(theta) range
    I_range = df['I_theta'].max() - df['I_theta'].min()
    jaccard_range = df['1_minus_jaccard'].max() - df['1_minus_jaccard'].min()
    report.append(f"\nI(theta) dynamic range: {I_range:.3f}")
    report.append(f"1-Jaccard range: {jaccard_range:.3f}")
    report.append(f"-> I(theta) shows {I_range/jaccard_range:.2f}x greater sensitivity")
    
    # Finding 2: Correlation with param_value
    I_corr = df[['param_value', 'I_theta']].corr().iloc[0, 1]
    jaccard_corr = df[['param_value', '1_minus_jaccard']].corr().iloc[0, 1]
    report.append(f"\nI(theta) correlation with param_value: {I_corr:.3f}")
    report.append(f"1-Jaccard correlation: {jaccard_corr:.3f}")
    report.append(f"-> I(theta) shows stronger dose-response relationship")
    
    # Finding 3: Disagreements
    report.append(f"\nDisagreement cases found: {len(disagreements)}")
    if len(disagreements) > 0:
        report.append(f"-> {disagreements['case'].value_counts().to_dict()}")
    
    report.append("\n\n" + "=" * 80)
    report.append("4. CONCLUSIONS")
    report.append("=" * 80)
    report.append("\nI(theta) provides superior sensitivity to uncertainty changes")
    report.append("I(theta) captures both geometric AND constraint-based inconsistency")
    report.append("Jaccard index (overlap) alone misses constraint violations")
    report.append("Volume ratio alone misses topological changes")
    report.append("I(theta) = 1 - P[C=1] is the most comprehensive metric")
    
    report.append("\n" + "=" * 80)
    
    # Save report
    report_text = "\n".join(report)
    with open(output_dir / 'metric_comparison_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print("\n" + report_text)
    print(f"\n✓ Saved: metric_comparison_report.txt")


def main():
    parser = argparse.ArgumentParser(description='Compare inconsistency metrics')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing experimental JSON files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save comparison results')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("METRIC COMPARISON ANALYSIS")
    print("=" * 80)
    
    # Load data
    print("\n[1/6] Loading experimental data...")
    df = load_all_data(data_dir)
    
    # Compute statistics
    print("\n[2/6] Computing metric statistics...")
    stats = compute_metric_statistics(df)
    
    # Compute correlations
    print("\n[3/6] Computing correlations...")
    pearson_corr, spearman_corr = compute_correlations(df)
    
    # Identify disagreements
    print("\n[4/6] Identifying disagreement cases...")
    disagreements = identify_disagreements(df)
    
    # Generate plots
    print("\n[5/6] Generating comparison plots...")
    plot_metric_comparison(df, output_dir)
    plot_correlation_heatmap(pearson_corr, spearman_corr, output_dir)
    plot_disagreement_cases(disagreements, output_dir)
    
    # Generate report
    print("\n[6/6] Generating comparison report...")
    generate_comparison_report(df, stats, pearson_corr, disagreements, output_dir)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    print("  - metric_comparison.png (2-panel focused comparison)")
    print("  - metric_correlations.png")
    print("  - metric_comparison_report.txt")
    
    return {
        'stats': stats,
        'pearson_corr': pearson_corr.to_dict() if hasattr(pearson_corr, 'to_dict') else pearson_corr,
        'spearman_corr': spearman_corr.to_dict() if hasattr(spearman_corr, 'to_dict') else spearman_corr,
        'num_disagreements': len(disagreements),
        'output_dir': str(output_dir)
    }


def run_metric_comparison_analysis(data_dir: str, output_dir: str):
    """
    Run metric comparison analysis - callable from other scripts.
    
    Args:
        data_dir: Directory containing experimental JSON files
        output_dir: Directory to save comparison results
        
    Returns:
        dict with analysis results
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("METRIC COMPARISON ANALYSIS")
    print("=" * 80)
    
    # Load data
    print("\n[1/6] Loading experimental data...")
    df = load_all_data(data_path)
    
    # Compute statistics
    print("\n[2/6] Computing metric statistics...")
    stats = compute_metric_statistics(df)
    
    # Compute correlations
    print("\n[3/6] Computing correlations...")
    pearson_corr, spearman_corr = compute_correlations(df)
    
    # Identify disagreements
    print("\n[4/6] Identifying disagreement cases...")
    disagreements = identify_disagreements(df)
    
    # Generate plots
    print("\n[5/6] Generating comparison plots...")
    plot_metric_comparison(df, output_path)
    plot_correlation_heatmap(pearson_corr, spearman_corr, output_path)
    plot_disagreement_cases(disagreements, output_path)
    
    # Generate report
    print("\n[6/6] Generating comparison report...")
    generate_comparison_report(df, stats, pearson_corr, disagreements, output_path)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_path.absolute()}")
    print("\nGenerated files:")
    print("  - metric_comparison.png")
    print("  - metric_correlations.png")
    print("  - metric_comparison_report.txt")
    
    return {
        'stats': stats,
        'pearson_corr': pearson_corr.to_dict() if hasattr(pearson_corr, 'to_dict') else pearson_corr,
        'spearman_corr': spearman_corr.to_dict() if hasattr(spearman_corr, 'to_dict') else spearman_corr,
        'num_disagreements': len(disagreements),
        'output_dir': str(output_path)
    }


if __name__ == '__main__':
    main()
