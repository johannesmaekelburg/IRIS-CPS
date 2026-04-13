"""
Master Analysis Pipeline

Runs both sensitivity analysis and metric comparison sequentially or independently.

Usage:
    # Run both analyses:
    python run_full_analysis.py --data_dir ../data/measurements/both --output_dir ../results/full_analysis
    
    # Run only sensitivity analysis:
    python run_full_analysis.py --data_dir ../data/measurements/both --output_dir ../results --only sensitivity
    
    # Run only metric comparison:
    python run_full_analysis.py --data_dir ../data/measurements/both --output_dir ../results --only comparison

    # Analyze Jaccard-only measurements:
    python run_full_analysis.py --data_dir ../data/measurements/jaccard --output_dir ../results/jaccard_analysis

Author: CPS Uncertainty-Propagation Framework / Causality Extension
Date: 2026-02-10
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import individual analysis modules
try:
    from sensitivity_analysis import run_full_sensitivity_analysis
    SENSITIVITY_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import sensitivity_analysis: {e}")
    SENSITIVITY_AVAILABLE = False

try:
    from metric_comparison import run_metric_comparison_analysis
    COMPARISON_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import metric_comparison: {e}")
    COMPARISON_AVAILABLE = False

from analysis_utils import ensure_output_dir, print_analysis_header, print_analysis_footer

try:
    import matplotlib.tri as mtri
    HEATMAP_AVAILABLE = True
except ImportError:
    HEATMAP_AVAILABLE = False


# ============================================================================
# Saltelli Experiment Detection
# ============================================================================

def detect_experiment_type(data_dir: str) -> str:
    """Detect if experiments are CONVIDE or Saltelli mode."""
    data_path = Path(data_dir)
    json_files = list(data_path.glob('results_*.json'))
    
    if not json_files:
        raise FileNotFoundError(f"No results files found in {data_dir}")
    
    # Check first file
    with open(json_files[0], 'r') as f:
        data = json.load(f)
    
    experiments = data.get('experiments', [])
    if not experiments:
        return 'unknown'
    
    first_exp = experiments[0]
    
    # Check for Saltelli-specific fields
    if first_exp.get('intervention_type') == 'compound':
        if 'scale_factor' in first_exp and 'center_delta' in first_exp:
            return 'saltelli'
    
    # Check for CONVIDE-specific fields
    if 'param_name' in first_exp or 'intervention_value' in first_exp:
        return 'convide'
    
    return 'unknown'


def run_saltelli_analysis(data_dir: str, output_dir: Path, problem_path: str):
    """Run Saltelli sensitivity analysis on all scenarios."""
    from pathlib import Path
    import subprocess
    import sys
    
    data_path = Path(data_dir)
    json_files = sorted(data_path.glob('results_scenario_*.json'))
    
    if not json_files:
        print("[WARNING]  No scenario files found - skipping Saltelli analysis")
        return
    
    print(f"Found {len(json_files)} scenario files to analyze")
    print()
    
    results = []
    
    for json_file in json_files:
        scenario_name = json_file.stem.replace('results_', '')
        print(f"Analyzing {scenario_name}...")

        try:
            with open(json_file, 'r') as f:
                scenario_data = json.load(f)
            scenario_experiments = scenario_data.get('experiments', [])
            has_compound = any(exp.get('intervention_type') == 'compound' for exp in scenario_experiments)
        except Exception as e:
            print(f"  [ERROR] Could not inspect {scenario_name}: {e}")
            results.append({'scenario': scenario_name, 'status': 'failed', 'error': str(e)})
            print()
            continue

        if not has_compound:
            print(f"  [WARNING] Skipping {scenario_name}: no Saltelli/compound interventions found")
            results.append({'scenario': scenario_name, 'status': 'skipped', 'reason': 'no compound interventions'})
            print()
            continue
        
        # Run analyze_saltelli_results.py script
        cmd = [
            sys.executable,
            'analyze_saltelli_results.py',
            '--results', str(json_file.absolute()),
            '--problem', str(Path(problem_path).absolute()),
            '--output_dir', str(Path(output_dir).absolute())
        ]
        
        try:
            result = subprocess.run(cmd, cwd=Path(__file__).parent, 
                                  capture_output=True, text=True, check=True)
            print(f"  [OK] {scenario_name} completed")
            results.append({'scenario': scenario_name, 'status': 'success'})
        except subprocess.CalledProcessError as e:
            print(f"  [ERROR] {scenario_name} failed: {e}")
            print(f"  Error output: {e.stderr}")
            results.append({'scenario': scenario_name, 'status': 'failed', 'error': str(e)})
        
        print()
    
    # Summary
    success_count = sum(1 for r in results if r['status'] == 'success')
    skipped_count = sum(1 for r in results if r['status'] == 'skipped')
    print(f"Saltelli analysis summary: {success_count}/{len(results)} scenarios completed successfully")
    if skipped_count > 0:
        print(f"Skipped non-Saltelli scenarios: {skipped_count}")
    
    return results


# ============================================================================
# Saltelli Mode Detection
# ============================================================================

def detect_saltelli_mode(data_dir):
    """
    Detect if experiments are Saltelli-mode (compound interventions) or CONVIDE-mode (discrete).
    
    Returns:
        'saltelli' if compound interventions detected
        'convide' if discrete interventions detected
        None if no data found
    """
    data_path = Path(data_dir)
    json_files = list(data_path.glob('results_*.json'))
    
    if not json_files:
        return None
    
    # Check first file
    with open(json_files[0], 'r') as f:
        data = json.load(f)
    
    experiments = data.get('experiments', [])
    if not experiments:
        return None
    
    # Check first experiment for intervention type
    first_exp = experiments[0]
    if first_exp.get('intervention_type') == 'compound':
        return 'saltelli'
    else:
        return 'convide'


# ============================================================================
# Intervention & Dimension Comparison Plots
# ============================================================================

def load_data_for_comparison(data_dir: str) -> pd.DataFrame:
    """Load all experimental data with dimension and intervention info."""
    data_path = Path(data_dir)
    json_files = sorted(data_path.glob('results_*.json'))
    
    all_records = []
    
    for json_file in json_files:
        name = json_file.stem
        parts = name.split('_')
        
        # Extract scenario_id from filename
        scenario_id = None
        for i, p in enumerate(parts):
            if p == 'scenario' and i + 1 < len(parts):
                scenario_id = int(parts[i + 1])
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        experiments = data.get('experiments', data if isinstance(data, list) else [])
        
        for exp in experiments:
            # Extract dimension from data structure (length of center array)
            dim = None
            pre_state = exp.get('pre_state', {})
            uncertainty = pre_state.get('uncertainty', {})
            source_center = uncertainty.get('source_center', [])
            if source_center:
                dim = len(source_center)
            # Extract intervention info from MATLAB export format
            intervention = exp.get('intervention_type', 'unknown')
            param_value = exp.get('intervention_value', exp.get('param_value', np.nan))
            
            record = {
                'dimension': dim,
                'scenario_id': scenario_id,
                'intervention': intervention,
                'param_value': param_value,
            }
            
            # Extract post-intervention inconsistency
            post_state = exp.get('post_state', {})
            post_inc = exp.get('post_inconsistency', post_state.get('inconsistency', {}))
            
            # Compute I_theta from MC probability if not directly available
            I_theta_post = post_inc.get('I_theta')
            if I_theta_post is None or (isinstance(I_theta_post, float) and np.isnan(I_theta_post)):
                mc_p_consistent = post_inc.get('mc_p_consistent_sobol', post_inc.get('mc_probability_sobol'))
                if mc_p_consistent is not None:
                    I_theta_post = 1.0 - mc_p_consistent
            
            record['I_theta'] = I_theta_post
            record['jaccard_index'] = post_inc.get('jaccard_index', np.nan)
            
            # Extract pre-intervention inconsistency
            pre_state = exp.get('pre_state', {})
            pre_inc = exp.get('pre_inconsistency', pre_state.get('inconsistency', {}))
            
            # Compute pre I_theta
            I_theta_pre = pre_inc.get('I_theta')
            if I_theta_pre is None or (isinstance(I_theta_pre, float) and np.isnan(I_theta_pre)):
                mc_p_consistent_pre = pre_inc.get('mc_p_consistent_sobol', pre_inc.get('mc_probability_sobol'))
                if mc_p_consistent_pre is not None:
                    I_theta_pre = 1.0 - mc_p_consistent_pre
            
            record['I_theta_pre'] = I_theta_pre
            
            if not np.isnan(record['I_theta']) and not np.isnan(record['I_theta_pre']):
                record['delta_I_theta'] = record['I_theta'] - record['I_theta_pre']
            else:
                record['delta_I_theta'] = np.nan
            
            # Classify initial consistency state (threshold at 0.1)
            if not np.isnan(record['I_theta_pre']):
                record['initially_consistent'] = record['I_theta_pre'] < 0.1
            else:
                record['initially_consistent'] = None
            
            all_records.append(record)
    
    return pd.DataFrame(all_records)


def plot_intervention_comparison(df: pd.DataFrame, output_path: Path, subset_name: str = 'all'):
    """Create intervention comparison plot (widen vs shrink vs correlate).
    
    Args:
        df: DataFrame with experimental results
        output_path: Directory to save plots
        subset_name: 'all', 'consistent', or 'inconsistent' for plot subset
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    interventions = sorted(df['intervention'].unique())
    colors = {'widen': '#e74c3c', 'shrink': '#3498db', 'correlate': '#2ecc71'}
    markers = {'widen': 'o', 'shrink': 's', 'correlate': '^'}
    
    # Plot 1: Mean I(θ) vs param_value for each intervention
    ax1 = axes[0]
    for intervention in interventions:
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
    title_suffix = '' if subset_name == 'all' else f' ({subset_name.capitalize()} Start)'
    ax1.set_title(f'Global Inconsistency by Intervention Type{title_suffix}', fontsize=14)
    ax1.legend(loc='best')
    ax1.set_ylim(0, 1.05)
    ax1.set_xscale('symlog', linthresh=0.1)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Delta I(θ) distribution by intervention (box plot)
    ax2 = axes[1]
    delta_data = []
    labels = []
    box_colors = []
    for intervention in interventions:
        subset = df[df['intervention'] == intervention]
        delta = subset['delta_I_theta'].dropna()
        if len(delta) > 0:
            delta_data.append(delta.values)
            labels.append(intervention.capitalize())
            box_colors.append(colors.get(intervention, '#95a5a6'))
    
    bp = ax2.boxplot(delta_data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Intervention Type', fontsize=12)
    ax2.set_ylabel('ΔI(θ) = I(θ)_post - I(θ)_pre', fontsize=12)
    title_suffix = '' if subset_name == 'all' else f' ({subset_name.capitalize()} Start)'
    ax2.set_title(f'Change in Inconsistency by Intervention{title_suffix}', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    suffix = '' if subset_name == 'all' else f'_{subset_name}'
    output_file = output_path / f'intervention_comparison{suffix}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  [OK] Saved: {output_file.name}")
    plt.close()


def plot_dimension_comparison(df: pd.DataFrame, output_path: Path, subset_name: str = 'all'):
    """Create dimension comparison plot (2D vs 3D vs 4D).
    
    Args:
        df: DataFrame with experimental results
        output_path: Directory to save plots
        subset_name: 'all', 'consistent', or 'inconsistent' for plot subset
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
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
        ax1.bar(x + offset, means, width, yerr=stds,
               label=f'{int(dim)}D', color=dim_colors.get(dim, '#95a5a6'),
               alpha=0.8, capsize=3)
    
    ax1.set_xlabel('Intervention Type', fontsize=12)
    ax1.set_ylabel('Mean I(θ)', fontsize=12)
    title_suffix = '' if subset_name == 'all' else f' ({subset_name.capitalize()} Start)'
    ax1.set_title(f'I(θ) by Dimension and Intervention{title_suffix}', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels([i.capitalize() for i in interventions])
    ax1.legend(title='Dimension')
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3, axis='y')
    
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
    title_suffix = '' if subset_name == 'all' else f' ({subset_name.capitalize()} Start)'
    ax2.set_title(f'I(θ) Trend Across Dimensions{title_suffix}', fontsize=14)
    ax2.set_xticks(dimensions)
    ax2.set_xticklabels([f'{int(d)}D' for d in dimensions])
    ax2.legend(loc='best')
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    suffix = '' if subset_name == 'all' else f'_{subset_name}'
    output_file = output_path / f'dimension_comparison{suffix}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  [OK] Saved: {output_file.name}")
    plt.close()


def plot_causal_effects_grouped_ordered(df: pd.DataFrame, output_path: Path):
    """Create 3 plots grouping total causal effects by intervention type (scale, shift, correlation),
    ordered by effect magnitude.
    
    Args:
        df: DataFrame with experimental results
        output_path: Directory to save plots
    """
    
    # Define intervention type categories
    intervention_categories = {
        'scale': ['widen', 'shrink'],
        'shift': ['shift_forward', 'shift_backward', 'shift_up', 'shift_down', 'shift'],
        'correlation': ['correlate', 'increase_correlation', 'decrease_correlation']
    }
    
    # Detect which interventions are present in the data
    present_interventions = df['intervention'].unique()
    
    # Auto-categorize based on what's present
    categorized = {cat: [] for cat in ['scale', 'shift', 'correlation']}
    for interv in present_interventions:
        interv_lower = str(interv).lower()
        categorized_flag = False
        
        for cat, keywords in intervention_categories.items():
            if any(kw in interv_lower for kw in keywords):
                categorized[cat].append(interv)
                categorized_flag = True
                break
        
        # If not categorized, try to infer from name
        if not categorized_flag:
            if 'scale' in interv_lower or 'size' in interv_lower:
                categorized['scale'].append(interv)
            elif 'shift' in interv_lower or 'move' in interv_lower:
                categorized['shift'].append(interv)
            elif 'corr' in interv_lower:
                categorized['correlation'].append(interv)
    
    # Remove empty categories
    categorized = {k: v for k, v in categorized.items() if len(v) > 0}
    
    if len(categorized) == 0:
        print("  ⚠ No intervention categories detected - skipping grouped causal effect plots")
        return
    
    # Create figure with subplots (one per category)
    n_cats = len(categorized)
    fig, axes = plt.subplots(1, n_cats, figsize=(6 * n_cats, 5))
    
    if n_cats == 1:
        axes = [axes]
    
    colors_by_cat = {
        'scale': '#e74c3c',
        'shift': '#3498db',
        'correlation': '#2ecc71'
    }
    
    for idx, (cat_name, interventions) in enumerate(categorized.items()):
        ax = axes[idx]
        
        # Filter data for this category
        cat_df = df[df['intervention'].isin(interventions)].copy()
        
        if len(cat_df) == 0:
            continue
        
        # Group by scenario and intervention, compute mean causal effect
        grouped = cat_df.groupby(['scenario_id', 'intervention', 'param_value']).agg({
            'delta_I_theta': ['mean', 'std', 'count']
        }).reset_index()
        
        grouped.columns = ['scenario_id', 'intervention', 'param_value', 
                          'mean_delta', 'std_delta', 'n_samples']
        
        # Create label for each scenario
        grouped['label'] = grouped.apply(
            lambda row: f"S{row['scenario_id']} {row['intervention']} ({row['param_value']:.2f})", 
            axis=1
        )
        
        # Sort by absolute magnitude of causal effect
        grouped['abs_delta'] = grouped['mean_delta'].abs()
        grouped_sorted = grouped.sort_values('abs_delta', ascending=False)
        
        # Plot
        y_pos = np.arange(len(grouped_sorted))
        bars = ax.barh(y_pos, grouped_sorted['mean_delta'], 
                      xerr=grouped_sorted['std_delta'],
                      color=colors_by_cat.get(cat_name, '#95a5a6'),
                      alpha=0.7, capsize=3)
        
        # Color bars by positive/negative
        for i, (bar, val) in enumerate(zip(bars, grouped_sorted['mean_delta'])):
            if val < 0:
                bar.set_color('#2ecc71')  # Green for decrease (good)
            else:
                bar.set_color('#e74c3c')  # Red for increase (bad)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(grouped_sorted['label'], fontsize=8)
        ax.set_xlabel('ΔI(θ) = I(θ)_post - I(θ)_pre', fontsize=11)
        ax.set_title(f'{cat_name.capitalize()} Interventions\n(ordered by |effect|)', 
                    fontsize=13, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add sample count annotations
        for i, (y, n) in enumerate(zip(y_pos, grouped_sorted['n_samples'])):
            if n > 1:
                ax.text(0, y, f'  n={int(n)}', 
                       va='center', ha='left' if grouped_sorted.iloc[i]['mean_delta'] < 0 else 'right',
                       fontsize=7, alpha=0.7)
    
    plt.tight_layout()
    
    output_file = output_path / 'causal_effects_grouped_ordered.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  [OK] Saved: {output_file.name}")
    plt.close()


# ============================================================================
# Saltelli Comparison Plots
# ============================================================================

def load_saltelli_data_for_plots(data_dir: str) -> pd.DataFrame:
    """Load Saltelli compound experiment data into a DataFrame for comparison plots."""
    data_path = Path(data_dir)
    json_files = sorted(data_path.glob('results_scenario_*.json'))

    all_records = []
    for json_file in json_files:
        parts = json_file.stem.split('_')
        scenario_id = None
        for i, p in enumerate(parts):
            if p == 'scenario' and i + 1 < len(parts):
                try:
                    scenario_id = int(parts[i + 1])
                except ValueError:
                    pass

        with open(json_file) as f:
            data = json.load(f)

        for exp in data.get('experiments', []):
            if exp.get('intervention_type') != 'compound':
                continue

            pre_unc = exp.get('pre_state', {}).get('uncertainty', {})
            pre_inc = exp.get('pre_state', {}).get('inconsistency', {})
            post_inc = exp.get('post_state', {}).get('inconsistency', {})
            causal = exp.get('causal_effect', {}).get('inconsistency', {})

            source_center = pre_unc.get('source_center', [])
            dim = len(source_center) if source_center else None

            I_theta_pre = pre_inc.get('I_theta')
            I_theta_post = post_inc.get('I_theta')
            delta = causal.get('delta_I_theta')
            if delta is None and I_theta_pre is not None and I_theta_post is not None:
                delta = I_theta_post - I_theta_pre

            all_records.append({
                'scenario_id': scenario_id,
                'scenario_description': exp.get('scenario_description', ''),
                'dimension': dim,
                'scale_factor': exp.get('scale_factor'),
                'center_delta': exp.get('center_delta'),
                'correlation_strength': exp.get('correlation_strength'),
                'I_theta_pre': I_theta_pre,
                'I_theta_post': I_theta_post,
                'delta_I_theta': delta,
            })

    return pd.DataFrame(all_records)


def run_saltelli_comparison_plots(data_dir: str, output_dir: Path):
    """Generate comparison plots for Saltelli compound experiment data."""

    print("Loading Saltelli data for comparison plots...")
    df = load_saltelli_data_for_plots(data_dir)
    if df.empty:
        print("  [WARNING] No Saltelli compound data found — skipping comparison plots")
        return

    df = df.dropna(subset=['I_theta_post'])
    dims = sorted(df['dimension'].dropna().unique())
    scenarios = sorted(df['scenario_id'].dropna().unique())
    params = ['scale_factor', 'center_delta', 'correlation_strength']
    labels = {p: HEATMAP_PARAM_LABELS.get(p, p) for p in params}

    print(f"  Loaded {len(df)} experiments across {len(scenarios)} scenarios")
    print(f"  Dimensions present: {[f'{int(d)}D' for d in dims]}")
    print()

    # -- Plot 1: Marginal I(theta) vs each parameter, per scenario --------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(scenarios), 1)))

    for ax, param in zip(axes, params):
        for color, sid in zip(colors, scenarios):
            sub = df[df['scenario_id'] == sid].dropna(subset=[param])
            if sub.empty:
                continue
            sub_sorted = sub.sort_values(param)
            # Binned mean trend
            try:
                sub_sorted['bin'] = pd.qcut(sub_sorted[param], q=12, duplicates='drop')
                trend = sub_sorted.groupby('bin', observed=False).agg(
                    x=(param, 'median'), y=('I_theta_post', 'mean')
                ).dropna()
                ax.plot(trend['x'], trend['y'], linewidth=1.5,
                        color=color, label=f'S{int(sid)}')
            except Exception:
                pass
        ax.set_xlabel(labels[param], fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25)
        ax.set_title(f'I(\u03b8) vs {labels[param]}', fontsize=12, fontweight='bold')

    axes[0].set_ylabel(r'$I(\theta)$', fontsize=12)
    handles, lbls = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, lbls, title='Scenario', loc='center right',
                   fontsize=8, ncol=1)
    fig.suptitle('Marginal Effects on Inconsistency — Saltelli Experiments',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_dir / 'saltelli_marginal_effects.png', dpi=150, bbox_inches='tight')
    print(f"  [OK] Saved: saltelli_marginal_effects.png")
    plt.close(fig)

    # -- Plot 2: delta-I(theta) distribution per scenario (box plot) -------
    df_delta = df.dropna(subset=['delta_I_theta'])
    if not df_delta.empty:
        scenario_groups = [df_delta[df_delta['scenario_id'] == sid]['delta_I_theta'].values
                           for sid in scenarios
                           if len(df_delta[df_delta['scenario_id'] == sid]) > 0]
        scenario_labels = [f"S{int(sid)}" for sid in scenarios
                           if len(df_delta[df_delta['scenario_id'] == sid]) > 0]

        fig, ax = plt.subplots(figsize=(max(8, len(scenario_labels) * 1.1), 5))
        bp = ax.boxplot(scenario_groups, labels=scenario_labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('#4878CF')
            patch.set_alpha(0.7)
        ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Scenario', fontsize=12)
        ax.set_ylabel(r'$\Delta I(\theta) = I(\theta)_{post} - I(\theta)_{pre}$', fontsize=12)
        ax.set_title('Change in Inconsistency per Scenario — Saltelli Experiments',
                     fontsize=13, fontweight='bold')
        ax.grid(alpha=0.25, axis='y')
        plt.tight_layout()
        fig.savefig(output_dir / 'saltelli_delta_I_theta_by_scenario.png', dpi=150, bbox_inches='tight')
        print(f"  [OK] Saved: saltelli_delta_I_theta_by_scenario.png")
        plt.close(fig)

    # -- Plot 3: I(theta) by dimension ------------------------------------
    if len(dims) > 1:
        dim_groups = [df[df['dimension'] == d]['I_theta_post'].dropna().values for d in dims]
        fig, ax = plt.subplots(figsize=(6, 5))
        bp = ax.boxplot(dim_groups, labels=[f'{int(d)}D' for d in dims], patch_artist=True)
        dim_colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
        for patch, color in zip(bp['boxes'], dim_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xlabel('Dimension', fontsize=12)
        ax.set_ylabel(r'$I(\theta)$', fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.set_title('Inconsistency by Dimension — Saltelli Experiments',
                     fontsize=13, fontweight='bold')
        ax.grid(alpha=0.25, axis='y')
        plt.tight_layout()
        fig.savefig(output_dir / 'saltelli_I_theta_by_dimension.png', dpi=150, bbox_inches='tight')
        print(f"  [OK] Saved: saltelli_I_theta_by_dimension.png")
        plt.close(fig)


def run_comparison_plots(data_dir: str, output_dir: Path):
    """Generate intervention and dimension comparison plots."""
    
    print("Loading data for comparison plots...")
    df = load_data_for_comparison(data_dir)
    
    dims = sorted(df['dimension'].dropna().unique())
    interventions = sorted(df['intervention'].unique())
    
    # Count initially consistent vs inconsistent scenarios
    df_clean = df[df['initially_consistent'].notna()]
    n_consistent = df_clean['initially_consistent'].sum()
    n_inconsistent = (~df_clean['initially_consistent']).sum()
    
    print(f"  Loaded {len(df)} experiments")
    print(f"  Dimensions: {[f'{int(d)}D' for d in dims]}")
    print(f"  Interventions: {interventions}")
    print(f"  Initially consistent: {n_consistent} experiments")
    print(f"  Initially inconsistent: {n_inconsistent} experiments")
    print()
    
    # Generate plots for all scenarios
    print("Generating plots for all scenarios...")
    plot_intervention_comparison(df, output_dir, subset_name='all')
    plot_dimension_comparison(df, output_dir, subset_name='all')
    
    # Generate plots for initially consistent scenarios
    if n_consistent > 0:
        print("\nGenerating plots for initially consistent scenarios...")
        df_consistent = df[df['initially_consistent'] == True]
        plot_intervention_comparison(df_consistent, output_dir, subset_name='consistent')
        plot_dimension_comparison(df_consistent, output_dir, subset_name='consistent')
    
    # Generate plots for initially inconsistent scenarios
    if n_inconsistent > 0:
        print("\nGenerating plots for initially inconsistent scenarios...")
        df_inconsistent = df[df['initially_consistent'] == False]
        plot_intervention_comparison(df_inconsistent, output_dir, subset_name='inconsistent')
        plot_dimension_comparison(df_inconsistent, output_dir, subset_name='inconsistent')
    
    # Generate grouped and ordered causal effect plots
    print("\nGenerating grouped causal effect plots (scale, shift, correlation)...")
    plot_causal_effects_grouped_ordered(df, output_dir)


# ============================================================================
# Data-driven Inconsistency Landscape Heatmaps
# ============================================================================

HEATMAP_PARAM_LABELS = {
    'scale_factor': r'Scale $s_u$',
    'center_delta': r'Shift $\Delta c_u$',
    'correlation_strength': r'Correlation $\beta$',
}


def _load_scenario_saltelli_data(json_path: Path):
    """Extract (scale_factor, center_delta, I_theta) from a Saltelli measurement file."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    experiments = data.get('experiments', [])
    records = []
    for exp in experiments:
        if exp.get('intervention_type') != 'compound':
            continue
        I_theta = exp.get('post_state', {}).get('inconsistency', {}).get('I_theta')
        if I_theta is None:
            continue
        records.append({
            'scale_factor': exp['scale_factor'],
            'center_delta': exp['center_delta'],
            'correlation_strength': exp.get('correlation_strength'),
            'I_theta': I_theta,
            'sample_idx': exp.get('sample_idx', 0),
        })
    if not records:
        return None, None

    df = pd.DataFrame(records)
    description = experiments[0].get('scenario_description', json_path.stem)
    scenario_type = experiments[0].get('scenario_type', '')
    sid = scenario_type.replace('scenario_', 'S') if scenario_type else ''
    name = f"{sid}: {description}" if sid else description
    return df, name


def _plot_data_heatmap(df, scenario_name, output_path,
                       param1='scale_factor', param2='center_delta',
                       grid_n=100, figsize=(7, 5.5)):
    """Plot an inconsistency landscape heatmap from measurement data using gridded interpolation."""
    from scipy.interpolate import griddata

    # Aggregate duplicates (same param combo, multiple repeats)
    agg = df.groupby([param1, param2])['I_theta'].mean().reset_index()
    x = agg[param1].to_numpy()
    y = agg[param2].to_numpy()
    z = agg['I_theta'].to_numpy()

    # Use highest non-saturated value for color scaling when possible.
    non_sat = z[z < (1.0 - 1e-9)]
    vmax_display = float(non_sat.max()) if non_sat.size > 0 else float(z.max())
    vmax_display = max(vmax_display, 1e-3)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    if len(agg) >= 3:
        # Interpolate onto a regular grid covering the full parameter range
        xi = np.linspace(x.min(), x.max(), grid_n)
        yi = np.linspace(y.min(), y.max(), grid_n)
        Xi, Yi = np.meshgrid(xi, yi)
        Zi = griddata((x, y), z, (Xi, Yi), method='cubic')
        # Fill any remaining NaN at edges with nearest-neighbour
        mask = np.isnan(Zi)
        if mask.any():
            Zi[mask] = griddata((x, y), z, (Xi[mask], Yi[mask]), method='nearest')
        # Clip overshoots from cubic interpolation
        Zi = np.clip(Zi, 0.0, vmax_display)

        levels = np.linspace(0.0, vmax_display, 20)
        cf = ax.contourf(Xi, Yi, Zi, levels=levels, cmap='RdYlGn_r', vmin=0.0, vmax=vmax_display)
        cs = ax.contour(Xi, Yi, Zi, levels=8, colors='k', linewidths=0.4, alpha=0.5)
        ax.clabel(cs, inline=True, fontsize=7, fmt='%.2f')
    else:
        cf = ax.scatter(x, y, c=np.minimum(z, vmax_display), cmap='RdYlGn_r', s=30,
                        vmin=0.0, vmax=vmax_display)

    fig.colorbar(cf, ax=ax, label=rf'$I(\theta)$ (max shown: {vmax_display:.3f})', shrink=0.9)
    ax.set_xlabel(HEATMAP_PARAM_LABELS.get(param1, param1), fontsize=12)
    ax.set_ylabel(HEATMAP_PARAM_LABELS.get(param2, param2), fontsize=12)
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(float(y.min()), float(y.max()))
    ux = np.unique(np.round(x, 6))
    uy = np.unique(np.round(y, 6))
    if len(ux) <= 8:
        ax.set_xticks(ux)
    if len(uy) <= 8:
        ax.set_yticks(uy)
    ax.set_title(f'Inconsistency Landscape \u2014 {scenario_name}',
                 fontsize=13, fontweight='bold')

    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] Saved: {output_path.name}")
    plt.close(fig)


def _generate_measurement_heatmaps(data_dir: str, heatmap_dir: Path):
    """Generate heatmaps from actual measurement JSON files."""
    data_path = Path(data_dir)

    # Find all scenario measurement files
    json_files = sorted(data_path.glob('results_scenario_*.json'))

    # Also check parent directory for Saltelli files
    if not json_files:
        parent = data_path.parent
        json_files = sorted(parent.glob('results_scenario_*.json'))

    if not json_files:
        print("  [WARNING] No scenario measurement files found")
        return

    for json_file in json_files:
        df, scenario_name = _load_scenario_saltelli_data(json_file)
        if df is None or df.empty:
            continue

        tag = scenario_name.replace(': ', '_').replace(' ', '_')
        tag = ''.join(c if (c.isalnum() and ord(c) < 128) or c in '_-' else '_' for c in tag)
        param_pairs = [
            ('scale_factor', 'center_delta'),
            ('scale_factor', 'correlation_strength'),
            ('center_delta', 'correlation_strength'),
        ]
        for p1, p2 in param_pairs:
            if df[p1].isna().all() or df[p2].isna().all():
                continue
            pair_tag = f"{p1}_vs_{p2}"
            out_file = heatmap_dir / f'inconsistency_landscape_{pair_tag}_{tag}.png'
            print(f"  Generating heatmap for {scenario_name} [{p1} vs {p2}] ({len(df)} data points)...")
            _plot_data_heatmap(df, scenario_name, out_file, param1=p1, param2=p2)


def run_complete_analysis(data_dir: str,
                         output_dir: str,
                         analysis_type: str = 'both',
                         param_name: str = 'auto',
                         threshold: float = 0.5,
                         run_sobol: bool = True,
                         problem_file: str = None):
    """
    Run complete analysis pipeline.
    
    Args:
        data_dir: Directory containing experimental JSON files
        output_dir: Root directory for all results
        analysis_type: 'both', 'sensitivity', 'comparison', or 'saltelli'
        param_name: Parameter to analyze (for sensitivity)
        threshold: Inconsistency threshold (for robustness)
        run_sobol: Whether to compute Sobol indices
        problem_file: Path to Saltelli problem definition JSON (auto-detect if None)
    """
    
    print_analysis_header("COMPLETE INCONSISTENCY ANALYSIS PIPELINE")
    
    start_time = datetime.now()
    
    # Detect experiment type
    exp_type = detect_experiment_type(data_dir)
    print(f"Experiment type detected: {exp_type.upper()}")
    
    # Create timestamped output directory
    timestamp = start_time.strftime('%Y%m%d_%H%M%S')
    analysis_name = analysis_type if analysis_type != 'both' else 'complete'
    output_dir_timestamped = Path(output_dir) / f"{analysis_name}_analysis_{timestamp}"
    output_path = ensure_output_dir(str(output_dir_timestamped))
    
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_path}")
    print(f"Analysis type: {analysis_type}")
    print()
    
    results_summary = {
        'data_dir': data_dir,
        'output_dir': str(output_path),
        'output_dir_name': output_path.name,
        'experiment_type': exp_type,
        'analysis_type': analysis_type,
        'timestamp': timestamp,
        'start_time': start_time.isoformat(),
        'completed': []
    }
    
    # === SALTELLI EXPERIMENTS ===
    if exp_type == 'saltelli' or analysis_type == 'saltelli':
        print("\n" + "="*80)
        print("SALTELLI SENSITIVITY ANALYSIS")
        print("="*80)
        print("Computing Sobol indices (S₁, Sᵀ, S₂) for all scenarios...")
        print()
        
        # Auto-detect problem file
        if problem_file is None:
            problem_file = Path(data_dir).parent / 'saltelli_problem_3param.json'
        else:
            problem_file = Path(problem_file)  # Convert string to Path if provided
            
        if not problem_file.exists():
            print(f"[WARNING]  Problem file not found: {problem_file}")
            print("Creating default problem definition...")
            problem_data = {
                'problem': {
                    'num_vars': 3,
                    'names': ['scale_factor', 'center_delta', 'correlation_strength'],
                    'bounds': [[0.5, 5.0], [0.0, 0.3], [0.0, 0.95]]
                },
                'N': 64,
                'total_samples': 512
            }
            problem_file.parent.mkdir(parents=True, exist_ok=True)
            with open(problem_file, 'w') as f:
                json.dump(problem_data, f, indent=2)
            print(f"[OK] Created: {problem_file}")
        
        try:
            saltelli_results = run_saltelli_analysis(data_dir, output_path, str(problem_file.absolute()))
            results_summary['completed'].append('saltelli_analysis')
            results_summary['saltelli_results'] = saltelli_results
            print("\n[OK] Saltelli analysis completed successfully")
            
        except Exception as e:
            print(f"\n[ERROR] Saltelli analysis failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Run comparison plots adapted for Saltelli data
        print("\n" + "="*80)
        print("SALTELLI COMPARISON PLOTS")
        print("="*80)
        print("Generating parameter effect and scenario comparison plots...")
        print()
        try:
            run_saltelli_comparison_plots(data_dir, output_path)
            results_summary['completed'].append('saltelli_comparison_plots')
            print("\n[OK] Saltelli comparison plots completed")
        except Exception as e:
            print(f"\n[ERROR] Saltelli comparison plots failed: {e}")
            import traceback
            traceback.print_exc()

    # === CONVIDE EXPERIMENTS ===
    elif exp_type == 'convide':
        
        # Run metric comparison (validates I(θ) metric choice)
        if analysis_type in ['both', 'comparison']:
            if not COMPARISON_AVAILABLE:
                print("[ERROR] Metric comparison not available - skipping")
            else:
                print("\n" + "="*80)
                print("PHASE 1: METRIC COMPARISON")
                print("="*80)
                print("Validating I(θ) metric against alternatives...")
                print()
                
                comparison_dir = output_path / 'metric_comparison'
                comparison_dir.mkdir(exist_ok=True)
                
                try:
                    # Call the metric comparison analysis function directly
                    run_metric_comparison_analysis(
                        data_dir=data_dir,
                        output_dir=str(comparison_dir)
                    )
                    
                    results_summary['completed'].append('metric_comparison')
                    print("\n[OK] Metric comparison completed successfully")
                    
                except Exception as e:
                    print(f"\n[ERROR] Metric comparison failed: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Run sensitivity analysis (parameter effects and robustness)
        if analysis_type in ['both', 'sensitivity']:
            if not SENSITIVITY_AVAILABLE:
                print("[ERROR] Sensitivity analysis not available - skipping")
            else:
                print("\n" + "="*80)
                print("PHASE 2: SENSITIVITY ANALYSIS")
                print("="*80)
                print("Computing parameter sensitivities and robustness margins...")
                print()
                
                sensitivity_dir = output_path / 'sensitivity_analysis'
                sensitivity_dir.mkdir(exist_ok=True)
                
                try:
                    run_full_sensitivity_analysis(
                        data_dir=data_dir,
                        output_dir=str(sensitivity_dir),
                        param_name=param_name,
                        threshold=threshold,
                        run_sobol=run_sobol
                    )
                    
                    results_summary['completed'].append('sensitivity_analysis')
                    print("\n[OK] Sensitivity analysis completed successfully")
                    
                except Exception as e:
                    print(f"\n[ERROR] Sensitivity analysis failed: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Run intervention and dimension comparison plots
        if analysis_type == 'both':
            print("\n" + "="*80)
            print("PHASE 3: INTERVENTION & DIMENSION COMPARISON")
            print("="*80)
            print("Comparing intervention effects and dimension scaling...")
            print()
            
            try:
                run_comparison_plots(data_dir, output_path)
                
                results_summary['completed'].append('comparison_plots')
                print("\n[OK] Comparison plots completed successfully")
                
            except Exception as e:
                print(f"\n[ERROR] Comparison plots failed: {e}")
                import traceback
                traceback.print_exc()

    # === INCONSISTENCY LANDSCAPE HEATMAPS FROM MEASUREMENT DATA ===
    if HEATMAP_AVAILABLE:
        print("\n" + "="*80)
        print("INCONSISTENCY LANDSCAPE HEATMAPS")
        print("="*80)
        print("Generating I(theta) heatmaps from measurement data...")
        print()

        try:
            _generate_measurement_heatmaps(data_dir, output_path)
            results_summary['completed'].append('inconsistency_heatmaps')
            print("\n[OK] Inconsistency landscape heatmaps completed")
        except Exception as e:
            print(f"\n[ERROR] Heatmap generation failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Final summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    results_summary['end_time'] = end_time.isoformat()
    results_summary['duration_seconds'] = duration
    
    print("\n" + "="*80)
    print("ANALYSIS PIPELINE COMPLETE")
    print("="*80)
    print(f"\nEnd time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    print(f"\nCompleted analyses: {', '.join(results_summary['completed'])}")
    print(f"\nAll results saved to: {output_path.absolute()}")
    print()
    
    # Save summary
    summary_file = output_path / 'analysis_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"[OK] Analysis summary saved: {summary_file.name}")
    
    return results_summary


def main():
    parser = argparse.ArgumentParser(
        description='Run complete inconsistency analysis pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run both analyses:
  python run_full_analysis.py --data_dir ../data --output_dir ../results
  
  # Run only sensitivity analysis:
  python run_full_analysis.py --data_dir ../data --output_dir ../results --only sensitivity
  
  # Run only metric comparison:
  python run_full_analysis.py --data_dir ../data --output_dir ../results --only comparison
  
  # Custom parameter analysis:
  python run_full_analysis.py --data_dir ../data --output_dir ../results \\
      --param param_alpha --threshold 0.3 --no-sobol
        """
    )
    
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing experimental JSON files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Root directory for all analysis results')
    parser.add_argument('--only', type=str, choices=['sensitivity', 'comparison', 'saltelli'],
                       help='Run only one analysis (default: auto-detect and run appropriate analyses)')
    parser.add_argument('--param', type=str, default='auto',
                       help='Parameter name for sensitivity analysis (default: auto-detect)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Inconsistency threshold for robustness margins (default: 0.5)')
    parser.add_argument('--no-sobol', action='store_true',
                       help='Skip Sobol indices computation (faster)')
    parser.add_argument('--problem', type=str, default=None,
                       help='Path to Saltelli problem definition JSON (auto-detect if not specified)')
    
    args = parser.parse_args()
    
    # Determine analysis type
    if args.only:
        analysis_type = args.only
    else:
        analysis_type = 'both'
    
    # Detect if data is Saltelli mode
    is_saltelli = detect_saltelli_mode(args.data_dir)
    
    if is_saltelli:
        print("\n[OK] Detected Saltelli-mode experiments")
        print("  → Running Saltelli sensitivity analysis (Sobol indices)")
        # Skip old analysis module checks for Saltelli mode
    else:
        # Check availability for CONVIDE mode
        if analysis_type == 'sensitivity' and not SENSITIVITY_AVAILABLE:
            print("Error: Sensitivity analysis module not available")
            sys.exit(1)
        
        if analysis_type == 'comparison' and not COMPARISON_AVAILABLE:
            print("Error: Metric comparison module not available")
            sys.exit(1)
        
        if analysis_type == 'both' and not (SENSITIVITY_AVAILABLE and COMPARISON_AVAILABLE):
            print("Error: Both analysis modules must be available to run complete pipeline")
            sys.exit(1)
    
    # Run analysis
    try:
        run_complete_analysis(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            analysis_type=analysis_type,
            param_name=args.param,
            threshold=args.threshold,
            run_sobol=not args.no_sobol,
            problem_file=args.problem
        )
        
        sys.exit(0)
        
    except Exception as e:
        print(f"\n[ERROR] Analysis pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()



