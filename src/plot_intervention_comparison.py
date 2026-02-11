"""
Plot Intervention Comparison Analysis

Creates plots comparing:
1. Effect of different intervention types (widen, shrink, correlate)
2. I(θ) behavior for each dimension (2D, 3D, 4D)

Author: Generated for ConViDe Analysis
Date: February 2026
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
import argparse

# Use a clean style
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        pass  # Use default style


def load_all_data(data_dir: str) -> pd.DataFrame:
    """Load all experimental data from JSON files."""
    data_path = Path(data_dir)
    json_files = sorted(data_path.glob('results_*.json'))
    
    all_records = []
    
    for json_file in json_files:
        # Extract scenario info from filename
        # Format: results_convide_Xd_scenario_Y.json
        name = json_file.stem
        parts = name.split('_')
        
        # Find dimension (2d, 3d, 4d)
        dim = None
        scenario_id = None
        for i, p in enumerate(parts):
            if p in ['2d', '3d', '4d']:
                dim = int(p[0])
            if p == 'scenario' and i + 1 < len(parts):
                scenario_id = int(parts[i + 1])
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        experiments = data.get('experiments', data)
        
        for exp in experiments:
            record = {
                'dimension': dim,
                'scenario_id': scenario_id,
                'file': json_file.name,
                'intervention': exp.get('intervention', 'unknown'),
                'param_value': exp.get('param_value', np.nan),
            }
            
            # Extract post_inconsistency metrics
            post_inc = exp.get('post_inconsistency', {})
            record['I_theta'] = post_inc.get('I_theta', np.nan)
            record['jaccard_index'] = post_inc.get('jaccard_index', np.nan)
            record['mc_probability'] = post_inc.get('mc_probability', np.nan)
            
            # Extract pre_inconsistency for delta computation
            pre_inc = exp.get('pre_inconsistency', {})
            record['I_theta_pre'] = pre_inc.get('I_theta', np.nan)
            
            # Compute delta
            if not np.isnan(record['I_theta']) and not np.isnan(record['I_theta_pre']):
                record['delta_I_theta'] = record['I_theta'] - record['I_theta_pre']
            else:
                record['delta_I_theta'] = np.nan
            
            all_records.append(record)
    
    return pd.DataFrame(all_records)


def plot_intervention_comparison(df: pd.DataFrame, output_dir: Path):
    """Create a comparison plot of different intervention types."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    interventions = df['intervention'].unique()
    colors = {'widen': '#e74c3c', 'shrink': '#3498db', 'correlate': '#2ecc71'}
    markers = {'widen': 'o', 'shrink': 's', 'correlate': '^'}
    
    # Plot 1: Mean I(θ) vs param_value for each intervention
    ax1 = axes[0]
    for intervention in sorted(interventions):
        subset = df[df['intervention'] == intervention]
        grouped = subset.groupby('param_value')['I_theta'].agg(['mean', 'std']).reset_index()
        
        color = colors.get(intervention, '#95a5a6')
        marker = markers.get(intervention, 'o')
        
        ax1.errorbar(grouped['param_value'], grouped['mean'], 
                    yerr=grouped['std'], 
                    label=intervention.capitalize(),
                    marker=marker, color=color, 
                    capsize=3, linewidth=2, markersize=8)
    
    ax1.set_xlabel('Parameter Value', fontsize=12)
    ax1.set_ylabel('Mean I(θ)', fontsize=12)
    ax1.set_title('Global Inconsistency by Intervention Type', fontsize=14)
    ax1.legend(loc='best')
    ax1.set_ylim(0, 1.05)
    ax1.set_xscale('symlog', linthresh=0.1)
    
    # Plot 2: Delta I(θ) distribution by intervention
    ax2 = axes[1]
    delta_data = []
    labels = []
    for intervention in sorted(interventions):
        subset = df[df['intervention'] == intervention]
        delta = subset['delta_I_theta'].dropna()
        if len(delta) > 0:
            delta_data.append(delta.values)
            labels.append(intervention.capitalize())
    
    bp = ax2.boxplot(delta_data, labels=labels, patch_artist=True)
    for i, (patch, intervention) in enumerate(zip(bp['boxes'], sorted(interventions))):
        patch.set_facecolor(colors.get(intervention, '#95a5a6'))
        patch.set_alpha(0.7)
    
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Intervention Type', fontsize=12)
    ax2.set_ylabel('ΔI(θ) = I(θ)_post - I(θ)_pre', fontsize=12)
    ax2.set_title('Change in Inconsistency by Intervention', fontsize=14)
    
    # Plot 3: Summary statistics bar chart
    ax3 = axes[2]
    summary_data = []
    for intervention in sorted(interventions):
        subset = df[df['intervention'] == intervention]
        summary_data.append({
            'intervention': intervention.capitalize(),
            'mean_I_theta': subset['I_theta'].mean(),
            'max_I_theta': subset['I_theta'].max(),
            'min_I_theta': subset['I_theta'].min(),
        })
    
    summary_df = pd.DataFrame(summary_data)
    x = np.arange(len(summary_df))
    width = 0.25
    
    bars1 = ax3.bar(x - width, summary_df['min_I_theta'], width, 
                    label='Min', color='#3498db', alpha=0.8)
    bars2 = ax3.bar(x, summary_df['mean_I_theta'], width, 
                    label='Mean', color='#2ecc71', alpha=0.8)
    bars3 = ax3.bar(x + width, summary_df['max_I_theta'], width, 
                    label='Max', color='#e74c3c', alpha=0.8)
    
    ax3.set_xlabel('Intervention Type', fontsize=12)
    ax3.set_ylabel('I(θ)', fontsize=12)
    ax3.set_title('I(θ) Statistics by Intervention', fontsize=14)
    ax3.set_xticks(x)
    ax3.set_xticklabels(summary_df['intervention'])
    ax3.legend(loc='upper left')
    ax3.set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    output_file = output_dir / 'intervention_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot_by_dimension(df: pd.DataFrame, output_dir: Path):
    """Create separate plots for each dimension (2D, 3D, 4D)."""
    
    dimensions = sorted(df['dimension'].dropna().unique())
    
    for dim in dimensions:
        dim_df = df[df['dimension'] == dim]
        
        if len(dim_df) == 0:
            continue
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'{int(dim)}D Scenarios - Inconsistency Analysis', fontsize=16, fontweight='bold')
        
        interventions = dim_df['intervention'].unique()
        colors = {'widen': '#e74c3c', 'shrink': '#3498db', 'correlate': '#2ecc71'}
        markers = {'widen': 'o', 'shrink': 's', 'correlate': '^'}
        
        # Plot 1: I(θ) vs param_value
        ax1 = axes[0, 0]
        for intervention in sorted(interventions):
            subset = dim_df[dim_df['intervention'] == intervention]
            grouped = subset.groupby('param_value')['I_theta'].agg(['mean', 'std']).reset_index()
            
            color = colors.get(intervention, '#95a5a6')
            marker = markers.get(intervention, 'o')
            
            ax1.errorbar(grouped['param_value'], grouped['mean'],
                        yerr=grouped['std'],
                        label=intervention.capitalize(),
                        marker=marker, color=color,
                        capsize=3, linewidth=2, markersize=7)
        
        ax1.set_xlabel('Parameter Value')
        ax1.set_ylabel('Mean I(θ)')
        ax1.set_title(f'I(θ) vs Parameter Value')
        ax1.legend(loc='best')
        ax1.set_ylim(0, 1.05)
        ax1.set_xscale('symlog', linthresh=0.1)
        
        # Plot 2: Jaccard-based consistency vs MC probability
        ax2 = axes[0, 1]
        valid_df = dim_df.dropna(subset=['jaccard_index', 'mc_probability'])
        if len(valid_df) > 0:
            for intervention in sorted(interventions):
                subset = valid_df[valid_df['intervention'] == intervention]
                if len(subset) > 0:
                    color = colors.get(intervention, '#95a5a6')
                    ax2.scatter(subset['jaccard_index'], subset['mc_probability'],
                               alpha=0.5, color=color, label=intervention.capitalize(), s=30)
            
            # Add diagonal line
            ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='y=x')
            ax2.set_xlabel('Jaccard Index (Consistency)')
            ax2.set_ylabel('MC Probability (Consistency)')
            ax2.set_title('Jaccard vs MC Probability')
            ax2.legend(loc='best')
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
        else:
            ax2.text(0.5, 0.5, 'No data available', ha='center', va='center')
        
        # Plot 3: Distribution of I(θ) by intervention
        ax3 = axes[1, 0]
        data_for_violin = []
        labels = []
        colors_list = []
        for intervention in sorted(interventions):
            subset = dim_df[dim_df['intervention'] == intervention]
            data_for_violin.append(subset['I_theta'].dropna().values)
            labels.append(intervention.capitalize())
            colors_list.append(colors.get(intervention, '#95a5a6'))
        
        vp = ax3.violinplot(data_for_violin, positions=range(len(labels)), showmeans=True, showmedians=True)
        for i, body in enumerate(vp['bodies']):
            body.set_facecolor(colors_list[i])
            body.set_alpha(0.7)
        
        ax3.set_xticks(range(len(labels)))
        ax3.set_xticklabels(labels)
        ax3.set_xlabel('Intervention Type')
        ax3.set_ylabel('I(θ)')
        ax3.set_title('I(θ) Distribution by Intervention')
        ax3.set_ylim(0, 1.05)
        
        # Plot 4: Scenarios within this dimension
        ax4 = axes[1, 1]
        scenario_ids = sorted(dim_df['scenario_id'].dropna().unique())
        
        x_positions = []
        y_means = []
        y_stds = []
        bar_colors = []
        scenario_labels = []
        
        for scenario_id in scenario_ids:
            scenario_df = dim_df[dim_df['scenario_id'] == scenario_id]
            for intervention in sorted(interventions):
                subset = scenario_df[scenario_df['intervention'] == intervention]
                if len(subset) > 0:
                    x_positions.append(f"S{int(scenario_id)}\n{intervention[:3]}")
                    y_means.append(subset['I_theta'].mean())
                    y_stds.append(subset['I_theta'].std())
                    bar_colors.append(colors.get(intervention, '#95a5a6'))
        
        if len(x_positions) > 0:
            bars = ax4.bar(range(len(x_positions)), y_means, yerr=y_stds,
                          color=bar_colors, alpha=0.8, capsize=2)
            ax4.set_xticks(range(len(x_positions)))
            ax4.set_xticklabels(x_positions, fontsize=8, rotation=45)
            ax4.set_xlabel('Scenario / Intervention')
            ax4.set_ylabel('Mean I(θ)')
            ax4.set_title(f'I(θ) by Scenario ({int(dim)}D)')
            ax4.set_ylim(0, 1.05)
        
        plt.tight_layout()
        
        output_file = output_dir / f'dimension_{int(dim)}d_analysis.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()


def plot_dimension_comparison(df: pd.DataFrame, output_dir: Path):
    """Compare behavior across dimensions."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    dimensions = sorted(df['dimension'].dropna().unique())
    dim_colors = {2: '#3498db', 3: '#2ecc71', 4: '#e74c3c'}
    
    interventions = sorted(df['intervention'].unique())
    
    # Plot 1: Mean I(θ) by dimension for each intervention
    ax1 = axes[0]
    width = 0.25
    x = np.arange(len(interventions))
    
    for i, dim in enumerate(dimensions):
        means = []
        stds = []
        for intervention in interventions:
            subset = df[(df['dimension'] == dim) & (df['intervention'] == intervention)]
            means.append(subset['I_theta'].mean() if len(subset) > 0 else 0)
            stds.append(subset['I_theta'].std() if len(subset) > 0 else 0)
        
        offset = (i - len(dimensions)/2 + 0.5) * width
        bars = ax1.bar(x + offset, means, width, yerr=stds,
                      label=f'{int(dim)}D', color=dim_colors.get(dim, '#95a5a6'),
                      alpha=0.8, capsize=3)
    
    ax1.set_xlabel('Intervention Type', fontsize=12)
    ax1.set_ylabel('Mean I(θ)', fontsize=12)
    ax1.set_title('I(θ) by Dimension and Intervention', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels([i.capitalize() for i in interventions])
    ax1.legend(title='Dimension')
    ax1.set_ylim(0, 1.05)
    
    # Plot 2: I(θ) trend across dimensions
    ax2 = axes[1]
    for intervention in interventions:
        dim_means = []
        dim_stds = []
        dims_present = []
        for dim in dimensions:
            subset = df[(df['dimension'] == dim) & (df['intervention'] == intervention)]
            if len(subset) > 0:
                dim_means.append(subset['I_theta'].mean())
                dim_stds.append(subset['I_theta'].std())
                dims_present.append(dim)
        
        if len(dims_present) > 0:
            ax2.errorbar(dims_present, dim_means, yerr=dim_stds,
                        marker='o', linewidth=2, markersize=10, capsize=5,
                        label=intervention.capitalize())
    
    ax2.set_xlabel('Dimension', fontsize=12)
    ax2.set_ylabel('Mean I(θ)', fontsize=12)
    ax2.set_title('I(θ) Trend Across Dimensions', fontsize=14)
    ax2.set_xticks(dimensions)
    ax2.set_xticklabels([f'{int(d)}D' for d in dimensions])
    ax2.legend(loc='best')
    ax2.set_ylim(0, 1.05)
    
    # Plot 3: Box plot by dimension
    ax3 = axes[2]
    data_by_dim = []
    labels = []
    colors_list = []
    for dim in dimensions:
        dim_data = df[df['dimension'] == dim]['I_theta'].dropna().values
        if len(dim_data) > 0:
            data_by_dim.append(dim_data)
            labels.append(f'{int(dim)}D')
            colors_list.append(dim_colors.get(dim, '#95a5a6'))
    
    bp = ax3.boxplot(data_by_dim, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax3.set_xlabel('Dimension', fontsize=12)
    ax3.set_ylabel('I(θ)', fontsize=12)
    ax3.set_title('I(θ) Distribution by Dimension', fontsize=14)
    
    plt.tight_layout()
    
    output_file = output_dir / 'dimension_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def main(data_dir: str, output_dir: str):
    """Main function to generate all comparison plots."""
    
    print("="*60)
    print("INTERVENTION & DIMENSION COMPARISON PLOTS")
    print("="*60)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\nLoading data from: {data_dir}")
    df = load_all_data(data_dir)
    print(f"Loaded {len(df)} experiments")
    print(f"Dimensions: {sorted(df['dimension'].dropna().unique())}")
    print(f"Interventions: {sorted(df['intervention'].unique())}")
    print(f"Scenarios: {sorted(df['scenario_id'].dropna().unique())}")
    
    # Generate plots
    print("\n--- Generating plots ---")
    
    # 1. Intervention comparison (all dimensions combined)
    print("\n1. Intervention comparison plot...")
    plot_intervention_comparison(df, output_path)
    
    # 2. Plots for each dimension
    print("\n2. Dimension-specific plots...")
    plot_by_dimension(df, output_path)
    
    # 3. Dimension comparison
    print("\n3. Dimension comparison plot...")
    plot_dimension_comparison(df, output_path)
    
    print("\n" + "="*60)
    print(f"All plots saved to: {output_path}")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate intervention and dimension comparison plots')
    parser.add_argument('--data_dir', type=str, default='../data/measurements',
                       help='Directory containing result JSON files')
    parser.add_argument('--output_dir', type=str, default='../figures/comparison',
                       help='Directory for output plots')
    
    args = parser.parse_args()
    main(args.data_dir, args.output_dir)
