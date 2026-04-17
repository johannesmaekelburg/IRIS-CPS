#!/usr/bin/env python3
"""
Analyze MATLAB Saltelli Results and Compute Sobol Indices

Features:
- Loads MATLAB experiment results (JSON format)
- Extracts Saltelli sample parameters and I(theta) values
- Computes Sobol indices (S1, ST, S2) for sensitivity analysis
- Generates comprehensive visualization plots
- Saves results to timestamped folder
"""

import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.tri as mtri
import matplotlib.patches as mpatches
from pathlib import Path
from datetime import datetime
from SALib.analyze import sobol
from matplotlib.colors import ListedColormap, BoundaryNorm

# Set plotting style
sns.set_style("white")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


PARAM_LABELS = {
    'scale_factor': 'Scale factor',
    'center_delta': 'Center shift',
    'correlation_strength': 'Correlation strength',
}

CLR_SOURCE = "#4878CF"
CLR_TARGET = "#E8A838"
CLR_CONSISTENT = "#2CA02C"
CLR_INCONSISTENT = "#D62728"
HEATMAP_CMAP = "RdYlGn_r"

def create_paper_summary_figure(results, Si, problem, output_dir):
    """Create one paper-ready overview figure in the visual style of zonotope_plots.py."""

    output_dir = Path(output_dir)
    param_names = problem['names']
    df = pd.DataFrame(results).copy().dropna(subset=['I_theta_mean'])

    if df.empty or len(param_names) < 2:
        return

    def _label(name):
        return PARAM_LABELS.get(name, name)

    def _contour_panel(ax, x, y, z, xlabel, ylabel, title, vmin, vmax, levels=18):
        plot_df = pd.DataFrame({'x': x, 'y': y, 'z': z}).dropna()
        if len(plot_df) < 3:
            sc = ax.scatter(plot_df['x'], plot_df['y'], c=plot_df['z'], cmap=HEATMAP_CMAP,
                            s=24, alpha=0.85, vmin=vmin, vmax=vmax)
            ax.set_xlabel(xlabel, fontsize=10, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.grid(alpha=0.2)
            return sc

        triang = mtri.Triangulation(plot_df['x'].to_numpy(), plot_df['y'].to_numpy())
        contourf = ax.tricontourf(triang, plot_df['z'].to_numpy(), levels=levels,
                                  cmap=HEATMAP_CMAP, vmin=vmin, vmax=vmax)
        cs = ax.tricontour(triang, plot_df['z'].to_numpy(), levels=8, colors='k', linewidths=0.35, alpha=0.35)
        ax.clabel(cs, inline=True, fontsize=7, fmt='%.2f')
        ax.scatter(plot_df['x'], plot_df['y'], s=6, color='black', alpha=0.16, linewidths=0)
        ax.set_xlabel(xlabel, fontsize=10, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(alpha=0.18)
        return contourf

    ranking_order = np.argsort(Si['ST'])[::-1]
    top_two = ranking_order[:2]
    p1 = param_names[top_two[0]]
    p2 = param_names[top_two[1]]
    interaction = Si['ST'] - Si['S1']
    vmin = df['I_theta_mean'].min()
    vmax = df['I_theta_mean'].max()

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 4, height_ratios=[1.1, 1.0, 0.95])

    ax_main = fig.add_subplot(gs[0, :2])
    contour = _contour_panel(
        ax_main,
        df[p1], df[p2], df['I_theta_mean'],
        _label(p1), _label(p2),
        'A. Inconsistency landscape',
        vmin, vmax, levels=20,
    )
    fig.colorbar(contour, ax=ax_main, label='I(theta)', shrink=0.9)

    ax_rank = fig.add_subplot(gs[0, 2:])
    y = np.arange(len(param_names))
    ax_rank.barh(y, Si['ST'][ranking_order], color=CLR_SOURCE, alpha=0.85, label='ST')
    ax_rank.scatter(Si['S1'][ranking_order], y, color=CLR_INCONSISTENT, zorder=3, label='S1')
    ax_rank.set_yticks(y)
    ax_rank.set_yticklabels([_label(param_names[i]) for i in ranking_order])
    ax_rank.set_xlabel('Importance', fontsize=10, fontweight='bold')
    ax_rank.set_title('B. Parameter importance', fontsize=11, fontweight='bold')
    ax_rank.grid(axis='x', alpha=0.25)
    ax_rank.legend(fontsize=9)

    if len(param_names) >= 3:
        remaining = [idx for idx in range(len(param_names)) if idx not in top_two]
        slice_idx = remaining[np.argmax(Si['ST'][remaining])] if remaining else None
    else:
        slice_idx = None

    slice_axes = [fig.add_subplot(gs[1, i]) for i in range(4)]
    if slice_idx is not None:
        p3 = param_names[slice_idx]
        try:
            slice_labels = pd.qcut(df[p3], q=4, duplicates='drop')
        except ValueError:
            slice_labels = None

        if slice_labels is None or getattr(slice_labels, 'cat', None) is None or len(slice_labels.cat.categories) < 2:
            bins = np.linspace(df[p3].min(), df[p3].max(), 5)
            if np.unique(bins).size >= 3:
                slice_labels = pd.cut(df[p3], bins=bins, include_lowest=True, duplicates='drop')

        if slice_labels is not None and getattr(slice_labels, 'cat', None) is not None:
            categories = list(slice_labels.cat.categories)
            while len(categories) < 4:
                categories.append(None)

            for idx, ax in enumerate(slice_axes):
                cat = categories[idx]
                if cat is None:
                    ax.axis('off')
                    continue
                mask = slice_labels == cat
                slice_df = df.loc[mask]
                _contour_panel(
                    ax,
                    slice_df[p1], slice_df[p2], slice_df['I_theta_mean'],
                    _label(p1), _label(p2),
                    f'C{idx + 1}. {_label(p3)} in {cat}',
                    vmin, vmax, levels=14,
                )
        else:
            for ax in slice_axes:
                ax.axis('off')
    else:
        for ax in slice_axes:
            ax.axis('off')

    ax_dist = fig.add_subplot(gs[2, :2])
    ax_dist.hist(df['I_theta_mean'], bins=30, color=CLR_SOURCE, edgecolor='black', alpha=0.75)
    ax_dist.axvline(df['I_theta_mean'].mean(), color=CLR_INCONSISTENT, linestyle='--', linewidth=1.8, label='mean')
    ax_dist.axvline(df['I_theta_mean'].median(), color=CLR_CONSISTENT, linestyle='--', linewidth=1.8, label='median')
    ax_dist.set_xlabel('I(theta)', fontsize=10, fontweight='bold')
    ax_dist.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax_dist.set_title('D. Distribution of inconsistency values', fontsize=11, fontweight='bold')
    ax_dist.grid(axis='y', alpha=0.25)
    ax_dist.legend(fontsize=9)

    ax_table = fig.add_subplot(gs[2, 2:])
    ax_table.axis('off')
    summary_data = []
    for idx in ranking_order:
        summary_data.append([
            _label(param_names[idx]),
            f"{Si['S1'][idx]:.3f}",
            f"{Si['ST'][idx]:.3f}",
            f"{interaction[idx]:.3f}"
        ])
    table = ax_table.table(
        cellText=summary_data,
        colLabels=['Parameter', 'S1', 'ST', 'ST-S1'],
        cellLoc='center',
        loc='center',
        colWidths=[0.42, 0.16, 0.16, 0.18]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.7)
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    for i in range(1, len(summary_data) + 1):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f4f4f4')
    ax_table.set_title('E. Sensitivity summary table', fontsize=11, fontweight='bold', pad=10)

    fig.suptitle('Saltelli sensitivity overview', fontsize=15, fontweight='bold')
    paper_path = output_dir / '8_paper_summary_figure.png'
    fig.savefig(paper_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"[OK] Saved paper summary figure: {paper_path}")


def load_matlab_results(json_path):
    """Load MATLAB results and extract Saltelli data."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    experiments = data.get('experiments', [])
    
    # Extract relevant fields
    results = []
    for exp in experiments:
        if exp.get('intervention_type') == 'compound':
            results.append({
                'sample_idx': exp.get('sample_idx', 0),
                'repeat_idx': exp.get('repeat_idx', 1),
                'scale_factor': exp.get('scale_factor'),
                'center_delta': exp.get('center_delta'),
                'correlation_strength': exp.get('correlation_strength'),
                'I_theta': 1.0 - exp.get('post_state', {}).get('inconsistency', {}).get('I_MF_random', np.nan)
            })
    
    return results


def aggregate_repeats(results):
    """Aggregate multiple repeats by taking mean I(theta) per sample."""
    from collections import defaultdict
    
    sample_groups = defaultdict(list)
    
    for r in results:
        sample_groups[r['sample_idx']].append(r)
    
    aggregated = []
    for sample_idx in sorted(sample_groups.keys()):
        group = sample_groups[sample_idx]
        
        # Check consistency of parameters within sample
        assert all(g['scale_factor'] == group[0]['scale_factor'] for g in group)
        assert all(g['center_delta'] == group[0]['center_delta'] for g in group)
        assert all(g['correlation_strength'] ==group[0]['correlation_strength'] for g in group)
        
        # Average I(theta) across repeats
        I_theta_values = [g['I_theta'] for g in group if not np.isnan(g['I_theta'])]
        
        aggregated.append({
            'sample_idx': sample_idx,
            'scale_factor': group[0]['scale_factor'],
            'center_delta': group[0]['center_delta'],
            'correlation_strength': group[0]['correlation_strength'],
            'I_theta_mean': np.mean(I_theta_values) if I_theta_values else np.nan,
            'I_theta_std': np.std(I_theta_values) if len(I_theta_values) > 1 else 0.0,
            'n_repeats': len(I_theta_values)
        })
    
    return aggregated


def compute_sobol_indices(results, problem):
    """Compute Sobol indices from Saltelli samples."""
    if not results:
        raise ValueError("No aggregated Saltelli samples found in results file")
    
    # Sort by sample index
    results_sorted = sorted(results, key=lambda r: r['sample_idx'])
    
    # Extract Y values
    Y = np.array([r['I_theta_mean'] for r in results_sorted])

    if Y.size == 0:
        raise ValueError("No I(theta) values available for Sobol analysis")
    
    # Check for NaN values
    n_nan = np.isnan(Y).sum()
    if n_nan == Y.size:
        raise ValueError("All I(theta) values are NaN; cannot compute Sobol indices")
    if n_nan > 0:
        print(f"[WARNING] Warning: {n_nan} NaN values in I(theta) - replacing with mean")
        Y[np.isnan(Y)] = np.nanmean(Y)
    
    print(f"\n{'='*80}")
    print("SOBOL SENSITIVITY ANALYSIS")
    print(f"{'='*80}")
    print(f"Total samples: {len(Y)}")
    print(f"I(theta) range: [{np.nanmin(Y):.4f}, {np.nanmax(Y):.4f}]")
    print(f"I(theta) mean ± std: {np.nanmean(Y):.4f} ± {np.nanstd(Y):.4f}")
    
    # Compute Sobol indices
    Si = sobol.analyze(problem, Y, calc_second_order=True, print_to_console=False)
    
    # Display results
    print(f"\n{'='*80}")
    print("FIRST-ORDER INDICES (S1) - Individual Parameter Effects")
    print(f"{'='*80}")
    print(f"{'Parameter':<30} {'S1':<10} {'Conf. (95%)':<15}")
    print(f"{'-'*80}")
    for i, name in enumerate(problem['names']):
        s1 = Si['S1'][i]
        s1_conf = Si['S1_conf'][i]
        print(f"{name:<30} {s1:>8.4f}   ± {s1_conf:<8.4f}")
    
    print(f"\n{'='*80}")
    print("TOTAL-EFFECT INDICES (ST) - Parameter + Interactions")
    print(f"{'='*80}")
    print(f"{'Parameter':<30} {'ST':<10} {'Conf. (95%)':<15}")
    print(f"{'-'*80}")
    for i, name in enumerate(problem['names']):
        st = Si['ST'][i]
        st_conf = Si['ST_conf'][i]
        print(f"{name:<30} {st:>8.4f}   ± {st_conf:<8.4f}")
    
    print(f"\n{'='*80}")
    print("SECOND-ORDER INDICES (S2) - Pairwise Interactions")
    print(f"{'='*80}")
    for i, name_i in enumerate(problem['names']):
        for j, name_j in enumerate(problem['names']):
            if j > i:
                s2 = Si['S2'][i, j]
                s2_conf = Si['S2_conf'][i, j]
                print(f"{name_i} × {name_j:<20}: {s2:>8.4f} ± {s2_conf:<8.4f}")
    
    # Interpretation
    print(f"\n{'='*80}")
    print("INTERPRETATION")
    print(f"{'='*80}")
    most_influential = np.argmax(Si['ST'])
    print(f"Most influential parameter: {problem['names'][most_influential]} (ST={Si['ST'][most_influential]:.4f})")
    
    interaction_strength = Si['ST'] - Si['S1']
    print(f"\nInteraction effects (ST - S1):")
    for i, name in enumerate(problem['names']):
        print(f"  {name:<30} {interaction_strength[i]:>8.4f}")
    
    return Si


def create_plots(results, Si, problem, output_dir):
    """Generate a more intuitive set of Saltelli sensitivity plots."""

    output_dir = Path(output_dir)
    param_names = problem['names']
    n_params = len(param_names)

    df = pd.DataFrame(results).copy()
    df = df.dropna(subset=['I_theta_mean'])

    if df.empty:
        raise ValueError("No valid results available for plotting")

    def _label(name):
        return PARAM_LABELS.get(name, name)

    def _contour_landscape(ax, x, y, z, xlabel, ylabel, title, levels=20, cmap='RdYlGn_r',
                           show_points=True, vmin=None, vmax=None):
        plot_df = pd.DataFrame({'x': x, 'y': y, 'z': z}).dropna()
        if len(plot_df) < 3:
            sc = ax.scatter(plot_df['x'], plot_df['y'], c=plot_df['z'], cmap=cmap,
                            s=26, alpha=0.85, vmin=vmin, vmax=vmax)
            ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.grid(alpha=0.25)
            return sc

        triang = mtri.Triangulation(plot_df['x'].to_numpy(), plot_df['y'].to_numpy())
        contourf = ax.tricontourf(triang, plot_df['z'].to_numpy(), levels=levels, cmap=cmap,
                                  vmin=vmin, vmax=vmax)
        cs = ax.tricontour(triang, plot_df['z'].to_numpy(), levels=8, colors='k', linewidths=0.4, alpha=0.45)
        ax.clabel(cs, inline=True, fontsize=7, fmt='%.2f')
        if show_points:
            ax.scatter(plot_df['x'], plot_df['y'], s=8, color='black', alpha=0.18, linewidths=0)
        ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(alpha=0.2)
        return contourf

    def _binned_mean_curve(x, y, max_bins=12):
        curve_df = pd.DataFrame({'x': x, 'y': y}).dropna().sort_values('x')
        if curve_df.empty:
            return np.array([]), np.array([])

        n_unique = curve_df['x'].nunique()
        n_bins = min(max_bins, n_unique)

        if n_bins < 3:
            grouped = curve_df.groupby('x', as_index=False)['y'].mean()
            return grouped['x'].to_numpy(), grouped['y'].to_numpy()

        curve_df['bin'] = pd.qcut(curve_df['x'], q=n_bins, duplicates='drop')
        grouped = curve_df.groupby('bin', observed=False).agg(
            x=('x', 'median'),
            y=('y', 'mean')
        ).dropna()
        return grouped['x'].to_numpy(), grouped['y'].to_numpy()

    interaction = Si['ST'] - Si['S1']

    # Plot 1: Sobol overview
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(n_params)
    width = 0.35

    ax1.bar(x - width / 2, Si['S1'], width, label='S1',
            color=CLR_INCONSISTENT, yerr=Si['S1_conf'], capsize=5)
    ax1.bar(x + width / 2, Si['ST'], width, label='ST',
            color=CLR_SOURCE, yerr=Si['ST_conf'], capsize=5)
    ax1.set_xlabel('Parameters', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Sobol index', fontsize=12, fontweight='bold')
    ax1.set_title('Direct vs total parameter importance', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(param_names, rotation=30, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    ranking_order = np.argsort(Si['ST'])[::-1]
    ax2.barh(np.arange(n_params), Si['ST'][ranking_order], color=CLR_SOURCE, alpha=0.85, label='ST')
    ax2.scatter(Si['S1'][ranking_order], np.arange(n_params), color=CLR_INCONSISTENT, zorder=3, label='S1')
    ax2.set_yticks(np.arange(n_params))
    ax2.set_yticklabels([param_names[i] for i in ranking_order])
    ax2.set_xlabel('Importance', fontsize=12, fontweight='bold')
    ax2.set_title('Parameter ranking', fontsize=13, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_dir / '1_sobol_overview.png', bbox_inches='tight')
    plt.close()

    # Plot 2: Interaction heatmap
    if n_params > 1:
        fig, ax = plt.subplots(figsize=(8, 6))
        S2_matrix = Si['S2']
        S2_full = np.full((n_params, n_params), np.nan)

        for i in range(n_params):
            for j in range(i + 1, n_params):
                S2_full[i, j] = S2_matrix[i, j]
                S2_full[j, i] = S2_matrix[i, j]

        im = ax.imshow(np.nan_to_num(S2_full, nan=0.0), cmap='YlOrRd', aspect='auto')
        ax.set_xticks(np.arange(n_params))
        ax.set_yticks(np.arange(n_params))
        ax.set_xticklabels(param_names, rotation=30, ha='right')
        ax.set_yticklabels(param_names)
        ax.set_title('Pairwise interaction strength (S2)', fontsize=13, fontweight='bold', pad=14)

        for i in range(n_params):
            for j in range(n_params):
                if i != j and not np.isnan(S2_full[i, j]):
                    ax.text(j, i, f'{S2_full[i, j]:.3f}', ha='center', va='center', fontsize=9)

        plt.colorbar(im, ax=ax, label='S2 index')
        plt.tight_layout()
        plt.savefig(output_dir / '2_interaction_heatmap.png', bbox_inches='tight')
        plt.close()

    # Plot 3: Marginal effects
    fig, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 4), sharey=True)
    if n_params == 1:
        axes = [axes]

    for i, (param, ax) in enumerate(zip(param_names, axes)):
        ax.scatter(df[param], df['I_theta_mean'], alpha=0.18, s=20, color='gray', edgecolors='none')

        x_curve, y_curve = _binned_mean_curve(df[param], df['I_theta_mean'])
        if x_curve.size > 0:
            ax.plot(x_curve, y_curve, color=CLR_SOURCE, linewidth=2.5, marker='o', markersize=4)

        ax.set_xlabel(_label(param), fontsize=11, fontweight='bold')
        ax.set_title(f'{param}\nS1={Si["S1"][i]:.3f}, ST={Si["ST"][i]:.3f}', fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)

    axes[0].set_ylabel('I(theta)', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / '3_marginal_effects.png', bbox_inches='tight')
    plt.close()

    # Plot 4: Landscape plot for the two most influential parameters
    if n_params >= 2:
        top_two = np.argsort(Si['ST'])[-2:][::-1]
        p1 = param_names[top_two[0]]
        p2 = param_names[top_two[1]]
        vmin = df['I_theta_mean'].min()
        vmax = df['I_theta_mean'].max()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        contour = _contour_landscape(
            ax1,
            df[p1], df[p2], df['I_theta_mean'],
            _label(p1), _label(p2),
            'Inconsistency landscape',
            levels=20,
            cmap=HEATMAP_CMAP,
            show_points=True,
            vmin=vmin,
            vmax=vmax,
        )
        plt.colorbar(contour, ax=ax1, label='I(theta)')

        q_low = df['I_theta_mean'].quantile(0.2)
        q_high = df['I_theta_mean'].quantile(0.8)
        low_mask = df['I_theta_mean'] <= q_low
        high_mask = df['I_theta_mean'] >= q_high

        ax2.scatter(df[p1], df[p2], s=16, color='lightgray', alpha=0.35, label='middle 60%')
        ax2.scatter(df.loc[low_mask, p1], df.loc[low_mask, p2], s=28, color=CLR_CONSISTENT, alpha=0.8, label='lowest 20%')
        ax2.scatter(df.loc[high_mask, p1], df.loc[high_mask, p2], s=28, color=CLR_INCONSISTENT, alpha=0.8, label='highest 20%')
        ax2.set_xlabel(_label(p1), fontsize=11, fontweight='bold')
        ax2.set_ylabel(_label(p2), fontsize=11, fontweight='bold')
        ax2.set_title('Where low and high inconsistency occur', fontsize=13, fontweight='bold')
        ax2.grid(alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(output_dir / '4_response_landscape.png', bbox_inches='tight')
        plt.close()

        # Plot 4b: Representative low / median / high samples on landscape
        fig, ax = plt.subplots(figsize=(7, 5.5))
        contour = _contour_landscape(
            ax,
            df[p1], df[p2], df['I_theta_mean'],
            _label(p1), _label(p2),
            'Representative parameter regimes',
            levels=20,
            cmap=HEATMAP_CMAP,
            show_points=True,
            vmin=vmin,
            vmax=vmax,
        )
        plt.colorbar(contour, ax=ax, label='I(theta)')

        idx_low = df['I_theta_mean'].idxmin()
        idx_med = (df['I_theta_mean'] - df['I_theta_mean'].median()).abs().idxmin()
        idx_high = df['I_theta_mean'].idxmax()
        reps = [
            ('Low', idx_low, CLR_CONSISTENT),
            ('Median', idx_med, CLR_TARGET),
            ('High', idx_high, CLR_INCONSISTENT),
        ]
        used_labels = set()
        for label, idx_rep, color in reps:
            if idx_rep in used_labels:
                continue
            used_labels.add(idx_rep)
            row = df.loc[idx_rep]
            ax.scatter(row[p1], row[p2], s=90, color=color, edgecolor='black', linewidth=0.8, zorder=4)
            ax.annotate(
                f"{label}: I={row['I_theta_mean']:.3f}",
                (row[p1], row[p2]),
                xytext=(6, 6),
                textcoords='offset points',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=color, alpha=0.9)
            )

        ax.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(output_dir / '4b_representative_samples.png', bbox_inches='tight')
        plt.close()

    # Plot 5: Sliced landscapes over the remaining parameter
    if n_params >= 3:
        remaining = [idx for idx in range(n_params) if idx not in top_two]
        slice_idx = remaining[np.argmax(Si['ST'][remaining])] if remaining else None

        if slice_idx is not None:
            p3 = param_names[slice_idx]
            slice_labels = None
            try:
                slice_groups = pd.qcut(df[p3], q=4, duplicates='drop')
                if getattr(slice_groups, 'cat', None) is not None and len(slice_groups.cat.categories) >= 2:
                    slice_labels = slice_groups
            except ValueError:
                slice_labels = None

            if slice_labels is None:
                bins = np.linspace(df[p3].min(), df[p3].max(), 5)
                if np.unique(bins).size >= 3:
                    slice_labels = pd.cut(df[p3], bins=bins, include_lowest=True, duplicates='drop')

            if slice_labels is not None:
                categories = list(slice_labels.cat.categories)
                n_slices = len(categories)
                fig, axes = plt.subplots(1, n_slices, figsize=(5 * n_slices, 4.5), sharex=True, sharey=True)
                if n_slices == 1:
                    axes = [axes]

                vmin = df['I_theta_mean'].min()
                vmax = df['I_theta_mean'].max()
                last_contour = None

                for ax, cat in zip(axes, categories):
                    mask = slice_labels == cat
                    slice_df = df.loc[mask]
                    last_contour = _contour_landscape(
                        ax,
                        slice_df[p1], slice_df[p2], slice_df['I_theta_mean'],
                        _label(p1), _label(p2),
                        f'{_label(p3)} in {cat}',
                        levels=14,
                        cmap=HEATMAP_CMAP,
                        show_points=True,
                        vmin=vmin,
                        vmax=vmax,
                    )
                    ax.set_xlabel(_label(p1), fontsize=10, fontweight='bold')
                    ax.grid(alpha=0.2)

                axes[0].set_ylabel(_label(p2), fontsize=10, fontweight='bold')
                if last_contour is not None:
                    fig.colorbar(last_contour, ax=axes, label='I(theta)', shrink=0.85)
                fig.suptitle(f'Sliced inconsistency landscapes across {_label(p3)}', fontsize=13, fontweight='bold')
                fig.tight_layout()
                fig.savefig(output_dir / '5_sliced_response_landscapes.png', bbox_inches='tight')
                plt.close(fig)

    # Plot 6: Distribution and ECDF
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.hist(df['I_theta_mean'], bins=30, color=CLR_SOURCE, edgecolor='black', alpha=0.75)
    ax1.axvline(df['I_theta_mean'].mean(), color=CLR_INCONSISTENT, linestyle='--', linewidth=2,
                label=f'Mean = {df["I_theta_mean"].mean():.3f}')
    ax1.axvline(df['I_theta_mean'].median(), color=CLR_CONSISTENT, linestyle='--', linewidth=2,
                label=f'Median = {df["I_theta_mean"].median():.3f}')
    ax1.set_xlabel('I(theta)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution of inconsistency values', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    sorted_vals = np.sort(df['I_theta_mean'].to_numpy())
    ecdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    ax2.plot(sorted_vals, ecdf, color=CLR_TARGET, linewidth=2.5)
    ax2.set_xlabel('I(theta)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative fraction', fontsize=12, fontweight='bold')
    ax2.set_title('Empirical cumulative distribution', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3)
    for q in [0.1, 0.5, 0.9]:
        x_q = df['I_theta_mean'].quantile(q)
        ax2.axvline(x_q, linestyle='--', linewidth=1, alpha=0.5, color='gray')

    plt.tight_layout()
    plt.savefig(output_dir / '6_inconsistency_distribution.png', bbox_inches='tight')
    plt.close()

    # Plot 7: Summary table with rank and interaction gap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, n_params))
    ax1.barh(np.arange(n_params), interaction[ranking_order], color=colors)
    ax1.set_yticks(np.arange(n_params))
    ax1.set_yticklabels([param_names[i] for i in ranking_order])
    ax1.set_xlabel('Interaction contribution (ST - S1)', fontsize=12, fontweight='bold')
    ax1.set_title('How much interactions matter', fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)

    summary_data = []
    for idx in ranking_order:
        summary_data.append([
            param_names[idx],
            f"{Si['S1'][idx]:.3f}",
            f"{Si['ST'][idx]:.3f}",
            f"{interaction[idx]:.3f}"
        ])

    ax2.axis('off')
    table = ax2.table(
        cellText=summary_data,
        colLabels=['Parameter', 'S1', 'ST', 'ST-S1'],
        cellLoc='center',
        loc='center',
        colWidths=[0.40, 0.18, 0.18, 0.20]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.9)

    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    for i in range(1, len(summary_data) + 1):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f4f4f4')

    ax2.set_title('Compact sensitivity summary', fontsize=13, fontweight='bold', pad=14)

    plt.tight_layout()
    plt.savefig(output_dir / '7_summary_and_interactions.png', bbox_inches='tight')
    plt.close()

    print(f"\n[OK] Generated up to 8 visualization plots in {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Saltelli experiment results from MATLAB',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python analyze_saltelli_results.py \\
    --results ../data/measurements/results_scenario_1.json \\
    --problem ../data/saltelli_problem_3param.json \\
    --output_dir ../results
        """
    )
    parser.add_argument('--results', type=str, required=True,
                       help='Path to MATLAB results JSON file')
    parser.add_argument('--problem', type=str, required=True,
                       help='Path to problem definition JSON file')
    parser.add_argument('--output_dir', type=str, default='../results',
                       help='Output directory for results (default: ../results)')
    parser.add_argument('--name', type=str, default=None,
                       help='Analysis name (default: extracted from results filename)')
    
    args = parser.parse_args()
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Extract scenario name from results file if not provided
    if args.name is None:
        results_path = Path(args.results)
        # Extract scenario from filename like "results_scenario_1.json"
        scenario_match = results_path.stem.replace('results_', '')
        analysis_name = f"saltelli_analysis_{scenario_match}_{timestamp}"
    else:
        analysis_name = f"{args.name}_{timestamp}"
    
    output_dir = Path(args.output_dir) / analysis_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"SALTELLI SENSITIVITY ANALYSIS")
    print(f"{'='*80}")
    print(f"Results file: {args.results}")
    print(f"Problem file: {args.problem}")
    print(f"Output directory: {output_dir}")
    print(f"Timestamp: {timestamp}")
    print(f"{'='*80}")
    
    # Load problem definition
    with open(args.problem, 'r') as f:
        problem_data = json.load(f)
        problem = problem_data['problem']
    
    # Load results
    print(f"\n[1/4] Loading results from: {args.results}")
    results = load_matlab_results(args.results)
    print(f"[OK] Loaded {len(results)} experiments")

    if not results:
        raise ValueError("No compound/Saltelli experiments found in the provided results file")
    
    # Aggregate repeats
    print(f"Aggregating repeats...")
    results_agg = aggregate_repeats(results)
    print(f"[OK] Aggregated to {len(results_agg)} unique samples")

    if not results_agg:
        raise ValueError("No valid aggregated Saltelli samples found after repeat aggregation")
    
    # Compute Sobol indices
    print(f"\n[3/4] Computing Sobol indices...")
    Si = compute_sobol_indices(results_agg, problem)
    
    # Create plots
    print(f"\n[4/4] Generating plots...")
    create_plots(results_agg, Si, problem, output_dir)
    create_paper_summary_figure(results_agg, Si, problem, output_dir)
    
    # Save JSON output
    output_json = {
        'metadata': {
            'timestamp': timestamp,
            'results_file': str(args.results),
            'problem_file': str(args.problem),
            'n_experiments': len(results),
            'n_samples': len(results_agg)
        },
        'problem': problem,
        'summary_statistics': {
            'I_theta_mean': float(np.mean([r['I_theta_mean'] for r in results_agg])),
            'I_theta_std': float(np.std([r['I_theta_mean'] for r in results_agg])),
            'I_theta_min': float(np.min([r['I_theta_mean'] for r in results_agg])),
            'I_theta_max': float(np.max([r['I_theta_mean'] for r in results_agg]))
        },
        'sobol_indices': {
            'first_order': {
                name: {
                    'S1': float(Si['S1'][i]),
                    'S1_conf': float(Si['S1_conf'][i])
                }
                for i, name in enumerate(problem['names'])
            },
            'total_effect': {
                name: {
                    'ST': float(Si['ST'][i]),
                    'ST_conf': float(Si['ST_conf'][i])
                }
                for i, name in enumerate(problem['names'])
            },
            'second_order': {
                f"{problem['names'][i]}_{problem['names'][j]}": {
                    'S2': float(Si['S2'][i, j]),
                    'S2_conf': float(Si['S2_conf'][i, j])
                }
                for i in range(len(problem['names']))
                for j in range(len(problem['names']))
                if j > i
            },
            'interaction_strength': {
                name: float(Si['ST'][i] - Si['S1'][i])
                for i, name in enumerate(problem['names'])
            }
        }
    }
    
    json_path = output_dir / 'sobol_indices.json'
    with open(json_path, 'w') as f:
        json.dump(output_json, f, indent=2)
    
    print(f"\n[OK] Saved Sobol indices: {json_path}")
    
    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"All results saved to: {output_dir}")
    print(f"  - sobol_indices.json (numerical results)")
    print(f"  - up to 8 visualization plots (PNG format)")
    print(f"  - 8_paper_summary_figure.png (paper-ready overview)")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()




