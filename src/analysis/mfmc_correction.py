#!/usr/bin/env python3
"""
Multi-Fidelity Monte Carlo (MFMC) Correction

Uses AABB Jaccard as a cheap control variate to reduce variance in the
MC Probability estimator of I(theta).

Estimator:
    I_MF(theta) = I_MC(theta) + alpha * (mu_AABB - I_AABB(theta))

where:
    alpha = Cov(I_MC, I_AABB) / Var(I_AABB)   [optimal control variate coefficient]
    mu_AABB = mean(I_AABB) over all N experiments in the scenario

Variance reduction factor: rho^2, where rho = corr(I_MC, I_AABB).

Usage:
    python mfmc_correction.py --input data/measurements --output results/mfmc
    python mfmc_correction.py --input data/measurements --scenario 103 104 107
"""

import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
try:
    import seaborn as sns
    sns.set_style("white")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
from pathlib import Path
from datetime import datetime

# Sampling methods present in the data
SAMPLING_METHODS = ['sobol', 'halton', 'lhs', 'random']

plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


# ─────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────

def load_scenario_results(json_path: Path) -> dict:
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


SALTELLI_PARAMS = ['scale_factor', 'center_delta', 'correlation_strength']

PARAM_LABELS = {
    'scale_factor':          r'Scale factor $\alpha$',
    'center_delta':          r'Shift $\Delta c_u$',
    'correlation_strength':  r'Correlation $\beta$',
}


def extract_fidelity_pairs(data: dict) -> pd.DataFrame:
    """
    Extract (I_AABB, I_MC_*) pairs from all experiments in a scenario result file.
    Also extracts Saltelli parameter columns when present (compound interventions).
    Returns a DataFrame with one row per experiment.
    """
    records = []
    for exp in data.get('experiments', []):
        inc     = exp.get('post_state', {}).get('inconsistency', {})
        pre_inc = exp.get('pre_state',  {}).get('inconsistency', {})

        if 'jaccard_index' not in inc:
            continue

        # Track whether zonotopes actually intersect.
        # Only intersecting experiments satisfy the MFMC control-variate assumption.
        j = inc.get('jaccard_index')
        jaccard_valid = j is not None          # False → no intersection
        I_AABB = float(j) if jaccard_valid else np.nan

        row = {
            'exp_id':             exp.get('exp_id', ''),
            'intervention_type':  exp.get('intervention_type', ''),
            'intervention_value': exp.get('intervention_value', np.nan),
            'sample_idx':         exp.get('sample_idx', np.nan),
            'jaccard_valid':      jaccard_valid,
            'I_AABB':             I_AABB,
            'I_theta_pre':        1.0 - pre_inc.get('I_MF_random', np.nan),
        }

        # Saltelli parameter values (only present for compound interventions)
        for p in SALTELLI_PARAMS:
            row[p] = exp.get(p, np.nan)

        # MC Probability per sampling method
        for m in SAMPLING_METHODS:
            row[f'I_MC_{m}']  = inc.get(f'mc_probability_{m}', np.nan)
            row[f'I_MC_se_{m}'] = inc.get(f'mc_standard_error_{m}', np.nan)

        # Primary I_theta: MFMC-corrected estimate (inverted: I_MF_random is consistency prob)
        row['I_theta'] = 1.0 - inc.get('I_MF_random', np.nan)
        records.append(row)

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────
# MFMC core
# ─────────────────────────────────────────────────────────────

def compute_mfmc(df: pd.DataFrame, mc_col: str) -> dict:
    """
    Compute MFMC control-variate correction for one (scenario, sampling_method) pair.

    α and μ_AABB are estimated exclusively from experiments where the zonotopes
    intersect (jaccard_valid=True).  The correction is then applied only to those
    same experiments; non-intersecting experiments receive NaN (no correction).

    Returns a dict with:
        alpha, rho, variance_reduction, mu_AABB, n_intersecting, I_MF (array)
    """
    # Restrict estimation to intersecting experiments
    intersecting = df[df['jaccard_valid'] == True] if 'jaccard_valid' in df.columns else df
    valid = intersecting[['I_AABB', mc_col]].dropna()
    if len(valid) < 3:
        return None

    aabb = valid['I_AABB'].values
    mc   = valid[mc_col].values

    mu_aabb = aabb.mean()
    cov     = np.cov(mc, aabb, ddof=1)        # 2×2 covariance matrix
    var_aabb = cov[1, 1]
    cov_mc_aabb = cov[0, 1]

    if var_aabb < 1e-12:
        # AABB has no variation — can't use as control variate
        return None

    alpha = cov_mc_aabb / var_aabb
    rho   = cov_mc_aabb / (np.std(mc, ddof=1) * np.std(aabb, ddof=1) + 1e-12)

    # Apply correction only where zonotopes intersect AND MC value is present
    mc_all   = df[mc_col].values
    aabb_all = df['I_AABB'].values
    valid_mask = (df['jaccard_valid'].values.astype(bool)
                  if 'jaccard_valid' in df.columns
                  else ~np.isnan(aabb_all))
    apply_mask = valid_mask & ~np.isnan(mc_all)

    I_MF = np.full(len(df), np.nan)
    I_MF[apply_mask] = np.clip(
        mc_all[apply_mask] + alpha * (mu_aabb - aabb_all[apply_mask]),
        0.0, 1.0,
    )

    var_mc = np.var(mc, ddof=1)
    mf_finite = I_MF[~np.isnan(I_MF)]
    var_mf = np.var(mf_finite, ddof=1) if len(mf_finite) > 1 else 0.0
    variance_reduction = 1.0 - var_mf / (var_mc + 1e-12)   # = rho^2 in theory

    return {
        'alpha':              alpha,
        'rho':                rho,
        'rho_sq':             rho ** 2,
        'variance_reduction': variance_reduction,
        'mu_AABB':            mu_aabb,
        'n_intersecting':     int(apply_mask.sum()),
        'I_MF':               I_MF,
        'n_experiments':      len(df),
    }


def process_scenario(data: dict, scenario_id: int) -> dict:
    """Run MFMC for all sampling methods in one scenario. Returns summary dict."""
    df = extract_fidelity_pairs(data)
    if df.empty:
        print(f"  [WARN] No experiments extracted for scenario {scenario_id}")
        return None

    scenario_name = data.get('scenario_name', data.get('scenario', f'scenario_{scenario_id}'))
    results = {
        'scenario_id':   scenario_id,
        'scenario_name': scenario_name,
        'n_experiments': len(df),
        'methods':       {},
    }

    # Detect which sampling methods are actually present
    available = [m for m in SAMPLING_METHODS if f'I_MC_{m}' in df.columns
                 and df[f'I_MC_{m}'].notna().sum() > 2]

    if not available:
        print(f"  [WARN] No MC probability columns found for scenario {scenario_id}")
        return None

    for m in available:
        r = compute_mfmc(df, f'I_MC_{m}')
        if r is None:
            print(f"  [WARN] Could not compute MFMC for method={m} (no AABB variance)")
            continue
        results['methods'][m] = r
        print(f"  [{m:8s}]  rho={r['rho']:+.4f}  alpha={r['alpha']:+.6f}"
              f"  var_reduction={r['variance_reduction']*100:.1f}%"
              f"  n_intersecting={r['n_intersecting']}/{r['n_experiments']}")

    results['df'] = df
    return results


# ─────────────────────────────────────────────────────────────
# Augmented JSON output
# ─────────────────────────────────────────────────────────────

def augment_json(data: dict, scenario_results: dict, primary_method: str = 'sobol') -> dict:
    """
    Add I_MF_* fields to each experiment's inconsistency block.
    Also adds a top-level 'mfmc_summary' block.
    """
    if scenario_results is None:
        return data

    df = scenario_results['df']
    methods = scenario_results['methods']

    # Build lookup exp_id → row index
    id_to_idx = {row['exp_id']: i for i, row in df.iterrows()}

    for exp in data.get('experiments', []):
        eid = exp.get('exp_id', '')
        if eid not in id_to_idx:
            continue
        idx = id_to_idx[eid]
        inc = exp['post_state']['inconsistency']

        for m, r in methods.items():
            inc[f'I_MF_{m}'] = float(r['I_MF'][idx]) if not np.isnan(r['I_MF'][idx]) else None

        # Primary MFMC estimate (one value used downstream)
        if primary_method in methods:
            inc['I_MF'] = inc.get(f'I_MF_{primary_method}')
            # Recompute I_theta_mfmc as 1 - I_MF (inconsistency = 1 - consistency)
            # Note: I_theta = 1 - mc_probability, I_MF replaces mc_probability
            if inc['I_MF'] is not None:
                inc['I_theta_mfmc'] = float(1.0 - inc['I_MF'])

    # Summary block
    summary = {
        'computed_at': datetime.now().isoformat(),
        'primary_method': primary_method,
    }
    for m, r in methods.items():
        summary[m] = {
            'alpha':              float(r['alpha']),
            'rho':                float(r['rho']),
            'rho_sq':             float(r['rho_sq']),
            'variance_reduction': float(r['variance_reduction']),
            'mu_AABB':            float(r['mu_AABB']),
            'n_intersecting':     int(r.get('n_intersecting', 0)),
            'n_experiments':      int(r.get('n_experiments', 0)),
        }
    data['mfmc_summary'] = summary
    return data


# ─────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────

CLR_AABB  = '#E8A838'
CLR_MC    = '#4878CF'
CLR_MF    = '#2CA02C'
CLR_DIAG  = '#888888'


def plot_scenario(scenario_results: dict, output_dir: Path):
    """
    Three-panel figure per scenario:
      Left:   Scatter I_AABB vs I_MC (with I_MF overlay)
      Centre: Distribution comparison (AABB / MC / MF)
      Right:  Variance reduction bar chart per sampling method
    """
    sid   = scenario_results['scenario_id']
    sname = scenario_results['scenario_name']
    df    = scenario_results['df']
    methods = scenario_results['methods']

    if not methods:
        return

    primary = 'sobol' if 'sobol' in methods else next(iter(methods))
    r = methods[primary]

    fig = plt.figure(figsize=(14, 4.5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

    # ── Left: scatter I_AABB vs I_MC + I_MF ──
    ax0 = fig.add_subplot(gs[0])
    mc_vals  = df[f'I_MC_{primary}'].values
    mf_vals  = r['I_MF']
    aabb_vals = df['I_AABB'].values

    mask = ~(np.isnan(aabb_vals) | np.isnan(mc_vals))
    ax0.scatter(aabb_vals[mask], mc_vals[mask],  s=18, alpha=0.6, color=CLR_MC,   label='MC')
    ax0.scatter(aabb_vals[mask], mf_vals[mask],  s=18, alpha=0.6, color=CLR_MF,   label='MF (corrected)', marker='^')
    lim = [0, 1]
    ax0.plot(lim, lim, '--', color=CLR_DIAG, lw=0.9, label='y=x')
    ax0.set_xlabel('I_AABB (cheap)')
    ax0.set_ylabel('Estimate')
    ax0.set_title(f'Fidelity correlation\nρ={r["rho"]:.3f}  (method: {primary})')
    ax0.legend(fontsize=8, frameon=False)
    ax0.set_xlim(0, 1); ax0.set_ylim(0, 1)

    # ── Centre: distribution comparison ──
    ax1 = fig.add_subplot(gs[1])
    bins = np.linspace(0, 1, 30)
    ax1.hist(aabb_vals[mask],  bins=bins, alpha=0.5, color=CLR_AABB, label='AABB',        density=True)
    ax1.hist(mc_vals[mask],    bins=bins, alpha=0.5, color=CLR_MC,   label=f'MC ({primary})', density=True)
    ax1.hist(mf_vals[mask],    bins=bins, alpha=0.5, color=CLR_MF,   label='MF',           density=True)
    ax1.set_xlabel('Consistency estimate')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution comparison')
    ax1.legend(fontsize=8, frameon=False)

    # ── Right: variance reduction per method ──
    ax2 = fig.add_subplot(gs[2])
    mnames = list(methods.keys())
    vr     = [methods[m]['variance_reduction'] * 100 for m in mnames]
    rhos   = [methods[m]['rho'] for m in mnames]
    colors = [CLR_MF if v > 0 else CLR_DIAG for v in vr]
    bars   = ax2.bar(mnames, vr, color=colors, alpha=0.8)
    for bar, rho in zip(bars, rhos):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'ρ={rho:.2f}', ha='center', va='bottom', fontsize=8)
    ax2.axhline(0, color='black', lw=0.7)
    ax2.set_ylabel('Variance reduction (%)')
    ax2.set_title('MFMC gain per sampling method')
    ax2.set_ylim(min(-5, min(vr) - 5), max(vr) + 12)

    fig.suptitle(f'Scenario {sid}: {sname}', fontsize=11, fontweight='bold', y=1.01)
    plt.tight_layout()

    out = output_dir / f'mfmc_scenario_{sid}.png'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved plot: {out.name}")


def plot_summary(all_results: list, output_dir: Path):
    """
    Summary figure: rho and variance reduction across all scenarios and methods.
    """
    rows = []
    for sr in all_results:
        if sr is None:
            continue
        for m, r in sr['methods'].items():
            rows.append({
                'scenario_id':        sr['scenario_id'],
                'scenario':           f"S{sr['scenario_id']}",
                'method':             m,
                'rho':                r['rho'],
                'variance_reduction': r['variance_reduction'] * 100,
                'alpha':              r['alpha'],
            })

    if not rows:
        return
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    def _heatmap(ax, pivot, title, ylabel, fmt, cmap, vmin=None, vmax=None):
        data_arr = pivot.values.astype(float)
        im = ax.imshow(data_arr, cmap=cmap, aspect='auto',
                       vmin=vmin, vmax=vmax, interpolation='nearest')
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=30, ha='right')
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        for i in range(data_arr.shape[0]):
            for j in range(data_arr.shape[1]):
                v = data_arr[i, j]
                ax.text(j, i, fmt % v, ha='center', va='center', fontsize=9,
                        color='black')
        ax.set_title(title)
        ax.set_xlabel('Sampling method')
        ax.set_ylabel(ylabel)

    pivot_rho = df.pivot(index='scenario', columns='method', values='rho')
    _heatmap(axes[0], pivot_rho, 'Correlation ρ (AABB vs MC)', 'Scenario',
             '%.2f', 'RdYlGn', vmin=-1, vmax=1)

    pivot_vr = df.pivot(index='scenario', columns='method', values='variance_reduction')
    _heatmap(axes[1], pivot_vr, 'Variance reduction (%) from MFMC', 'Scenario',
             '%.1f', 'RdYlGn')

    plt.suptitle('MFMC Summary — All Scenarios', fontsize=12, fontweight='bold')
    plt.tight_layout()
    out = output_dir / 'mfmc_summary.png'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved summary plot: {out.name}")


# ─────────────────────────────────────────────────────────────
# Ablation study: AABB vs MC vs MFMC comparison heatmaps
# ─────────────────────────────────────────────────────────────

METHOD_COLORS = {
    'AABB':         '#E8A838',
    'MC (sobol)':   '#4878CF',
    'MC (halton)':  '#6aaed6',
    'MC (lhs)':     '#9ecae1',
    'MC (random)':  '#c6dbef',
    'MFMC (sobol)': '#2CA02C',
}


def _collect_ablation_df(all_results: list) -> pd.DataFrame:
    """
    Build a long-form DataFrame with columns:
        scenario_id, scenario_label, exp_id, method, value
    covering AABB, MC (all sampling methods), and MFMC (all sampling methods).
    """
    rows = []
    for sr in all_results:
        if sr is None:
            continue
        df = sr['df']
        sid = sr['scenario_id']
        label = f"S{sid}"
        methods = sr['methods']

        for _, row in df.iterrows():
            eid = row['exp_id']
            # AABB
            rows.append(dict(scenario_id=sid, scenario=label, exp_id=eid,
                             method='AABB', value=row['I_AABB']))
            # MC variants
            for m in SAMPLING_METHODS:
                col = f'I_MC_{m}'
                if col in df.columns:
                    rows.append(dict(scenario_id=sid, scenario=label, exp_id=eid,
                                     method=f'MC ({m})', value=row[col]))
            # MFMC variants
            for m, r in methods.items():
                idx = df.index.get_loc(row.name)
                rows.append(dict(scenario_id=sid, scenario=label, exp_id=eid,
                                 method=f'MFMC ({m})', value=float(r['I_MF'][idx])))
    return pd.DataFrame(rows)


def plot_ablation_study(all_results: list, output_dir: Path, primary: str = 'sobol'):
    """
    Ablation study figure: three panels
      1. Scenario-mean consistency per method (grouped bar chart)
      2. Score distribution per method across all scenarios (violin / box)
      3. Per-scenario AABB vs MC vs MFMC scatter grid
    """
    abl = _collect_ablation_df(all_results)
    if abl.empty:
        return

    # Focus on primary method for cleaner plots
    methods_to_show = ['AABB', f'MC ({primary})', f'MFMC ({primary})']
    colors_to_show  = [METHOD_COLORS.get(m, '#888888') for m in methods_to_show]
    sub = abl[abl['method'].isin(methods_to_show)].copy()
    scenarios = sorted(sub['scenario_id'].unique())
    scenario_labels = [f"S{s}" for s in scenarios]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── Panel 1: mean per scenario, grouped bars ──
    ax = axes[0]
    means = sub.groupby(['scenario', 'method'])['value'].mean().unstack('method')
    means = means.reindex(columns=methods_to_show)
    x = np.arange(len(means))
    w = 0.25
    for i, (m, c) in enumerate(zip(methods_to_show, colors_to_show)):
        if m in means.columns:
            ax.bar(x + (i - 1) * w, means[m], w, label=m, color=c, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(means.index, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Mean consistency score')
    ax.set_title('Mean score per scenario\n(AABB vs MC vs MFMC)')
    ax.legend(fontsize=8, frameon=False)
    ax.set_ylim(0, 1.05)

    # ── Panel 2: distribution across all experiments (box plot) ──
    ax = axes[1]
    data_by_method = [sub[sub['method'] == m]['value'].dropna().values
                      for m in methods_to_show]
    bp = ax.boxplot(data_by_method, patch_artist=True, notch=False,
                    medianprops=dict(color='black', linewidth=1.5))
    for patch, c in zip(bp['boxes'], colors_to_show):
        patch.set_facecolor(c)
        patch.set_alpha(0.8)
    ax.set_xticklabels([m.replace(' (', '\n(') for m in methods_to_show], fontsize=8)
    ax.set_ylabel('Consistency score')
    ax.set_title('Score distribution\nacross all experiments')
    ax.set_ylim(0, 1.05)

    # ── Panel 3: AABB vs MFMC scatter (MC as reference diagonal) ──
    ax = axes[2]
    aabb_vals = sub[sub['method'] == 'AABB'][['scenario_id', 'exp_id', 'value']].rename(columns={'value': 'I_AABB'})
    mc_vals   = sub[sub['method'] == f'MC ({primary})'][['scenario_id', 'exp_id', 'value']].rename(columns={'value': 'I_MC'})
    mf_vals   = sub[sub['method'] == f'MFMC ({primary})'][['scenario_id', 'exp_id', 'value']].rename(columns={'value': 'I_MF'})
    merged = aabb_vals.merge(mc_vals, on=['scenario_id', 'exp_id']).merge(mf_vals, on=['scenario_id', 'exp_id'])

    ax.scatter(merged['I_MC'], merged['I_AABB'], s=10, alpha=0.4, color=METHOD_COLORS['AABB'],    label='AABB')
    ax.scatter(merged['I_MC'], merged['I_MF'],   s=10, alpha=0.4, color=METHOD_COLORS['MC (sobol)'] if primary != 'sobol' else METHOD_COLORS['MFMC (sobol)'], label=f'MFMC ({primary})')
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, label='y = x (MC reference)')
    ax.set_xlabel(f'MC ({primary})')
    ax.set_ylabel('Score')
    ax.set_title('AABB & MFMC vs MC reference')
    ax.legend(fontsize=8, frameon=False)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    fig.suptitle('Ablation Study: Consistency Scoring Methods', fontsize=12, fontweight='bold')
    plt.tight_layout()
    out = output_dir / 'ablation_study.png'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved ablation study: {out.name}")


def plot_method_heatmap(all_results: list, output_dir: Path, primary: str = 'sobol'):
    """
    Heatmap grid: rows = scenarios, columns = methods.
    Cell value = mean consistency score across all experiments in that scenario.
    One heatmap per panel: AABB | MC | MFMC | difference (MFMC - MC).
    """
    abl = _collect_ablation_df(all_results)
    if abl.empty:
        return

    methods_to_show = ['AABB', f'MC ({primary})', f'MFMC ({primary})']
    sub = abl[abl['method'].isin(methods_to_show)]

    pivot = sub.groupby(['scenario', 'method'])['value'].mean().unstack('method')
    pivot = pivot.reindex(columns=methods_to_show)

    # Add difference column
    if f'MC ({primary})' in pivot.columns and f'MFMC ({primary})' in pivot.columns:
        pivot['MFMC - MC'] = pivot[f'MFMC ({primary})'] - pivot[f'MC ({primary})']

    cols = list(pivot.columns)
    n_cols = len(cols)

    fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, max(4, len(pivot) * 0.45 + 1.5)))

    cmaps   = ['YlOrRd_r', 'YlOrRd_r', 'YlOrRd_r', 'RdYlGn']
    vmins   = [0,           0,           0,           -0.05]
    vmaxs   = [1,           1,           1,            0.05]

    for ax, col, cmap, vmin, vmax in zip(axes, cols, cmaps, vmins, vmaxs):
        data = pivot[col].values.reshape(-1, 1)
        im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax,
                       interpolation='nearest')
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xticks([0])
        ax.set_xticklabels([col], rotation=30, ha='right', fontsize=9)
        ax.set_yticks(range(len(pivot)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        for i, v in enumerate(pivot[col].values):
            ax.text(0, i, f'{v:.3f}', ha='center', va='center', fontsize=9,
                    color='white' if (vmin == 0 and v > 0.6) else 'black')
        ax.set_title(col, fontsize=10, fontweight='bold')

    fig.suptitle(f'Method Comparison Heatmap (primary: {primary})',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    out = output_dir / 'method_heatmap.png'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved method heatmap: {out.name}")


# ─────────────────────────────────────────────────────────────
# Inconsistency landscape plots (Saltelli data only)
# ─────────────────────────────────────────────────────────────

import matplotlib.tri as mtri

HEATMAP_CMAP = 'RdYlGn_r'


def _contour_panel(ax, x, y, z, xlabel, ylabel, title, vmin, vmax, levels=18):
    """Filled contour panel with contour lines, matching the style of analyze_saltelli_results."""
    df_plot = pd.DataFrame({'x': x, 'y': y, 'z': z}).dropna()
    if len(df_plot) < 3:
        sc = ax.scatter(df_plot['x'], df_plot['y'], c=df_plot['z'],
                        cmap=HEATMAP_CMAP, s=24, alpha=0.85, vmin=vmin, vmax=vmax)
        ax.set_xlabel(xlabel, fontsize=10, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
        ax.set_title(title, fontsize=11, fontweight='bold')
        return sc
    triang = mtri.Triangulation(df_plot['x'].to_numpy(), df_plot['y'].to_numpy())
    cf = ax.tricontourf(triang, df_plot['z'].to_numpy(), levels=levels,
                        cmap=HEATMAP_CMAP, vmin=vmin, vmax=vmax)
    cs = ax.tricontour(triang, df_plot['z'].to_numpy(), levels=8,
                       colors='k', linewidths=0.35, alpha=0.35)
    ax.clabel(cs, inline=True, fontsize=7, fmt='%.2f')
    ax.scatter(df_plot['x'], df_plot['y'], s=6, color='black', alpha=0.16, linewidths=0)
    ax.set_xlabel(xlabel, fontsize=10, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(alpha=0.18)
    return cf


def plot_mfmc_landscape(scenario_results: dict, output_dir: Path, primary: str = 'sobol'):
    """
    Inconsistency landscape for Saltelli scenarios — three panels side by side:
      Left:   I_theta (raw MC)
      Centre: I_MF (MFMC-corrected)
      Right:  Difference (I_MF - I_theta)

    Only runs if the scenario has Saltelli parameter columns.
    Axes are the two most-varying Saltelli parameters.
    """
    if scenario_results is None:
        return

    df      = scenario_results['df']
    sid     = scenario_results['scenario_id']
    sname   = scenario_results['scenario_name']
    methods = scenario_results['methods']

    if primary not in methods:
        return

    # Check this is Saltelli data
    saltelli_cols = [p for p in SALTELLI_PARAMS if p in df.columns and df[p].notna().any()]
    if len(saltelli_cols) < 2:
        return  # CONVIDE discrete data — skip landscape

    # Attach I_MF to df
    r     = methods[primary]
    df    = df.copy()
    df['I_MF']  = r['I_MF']
    df['I_MC']  = df[f'I_MC_{primary}']
    df['delta'] = df['I_MF'] - df['I_MC']

    # Aggregate repeats per sample_idx
    if 'sample_idx' in df.columns and df['sample_idx'].notna().any():
        grp = df.groupby('sample_idx').agg(
            {**{p: 'first' for p in saltelli_cols},
             'I_MC': 'mean', 'I_MF': 'mean', 'delta': 'mean'}
        ).reset_index()
    else:
        grp = df.copy()

    # Pick the two axes: highest-variance Saltelli parameters
    variances  = {p: grp[p].var() for p in saltelli_cols}
    top2       = sorted(variances, key=variances.get, reverse=True)[:2]
    x_col, y_col = top2[0], top2[1]
    xlabel = PARAM_LABELS.get(x_col, x_col)
    ylabel = PARAM_LABELS.get(y_col, y_col)

    vmin = min(grp['I_MC'].min(), grp['I_MF'].min())
    vmax = max(grp['I_MC'].max(), grp['I_MF'].max())

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    cf0 = _contour_panel(axes[0], grp[x_col], grp[y_col], grp['I_MC'],
                         xlabel, ylabel,
                         f'MC ({primary})\nI(\u03b8) landscape',
                         vmin, vmax)
    fig.colorbar(cf0, ax=axes[0], label='I(\u03b8)', shrink=0.88)

    cf1 = _contour_panel(axes[1], grp[x_col], grp[y_col], grp['I_MF'],
                         xlabel, ylabel,
                         f'MFMC — corrected\nI(\u03b8) landscape',
                         vmin, vmax)
    fig.colorbar(cf1, ax=axes[1], label='I(\u03b8)', shrink=0.88)

    # Difference panel — symmetric colormap around 0
    d_abs = grp['delta'].abs().max()
    d_abs = max(d_abs, 1e-4)
    cf2 = _contour_panel(axes[2], grp[x_col], grp[y_col], grp['delta'],
                         xlabel, ylabel,
                         'Difference\n(MFMC \u2212 MC)',
                         -d_abs, d_abs, levels=14)
    # Override colormap for difference panel
    axes[2].cla()
    df_d = pd.DataFrame({'x': grp[x_col], 'y': grp[y_col], 'z': grp['delta']}).dropna()
    if len(df_d) >= 3:
        triang = mtri.Triangulation(df_d['x'].to_numpy(), df_d['y'].to_numpy())
        cf2 = axes[2].tricontourf(triang, df_d['z'].to_numpy(), levels=14,
                                   cmap='RdBu_r', vmin=-d_abs, vmax=d_abs)
        cs2 = axes[2].tricontour(triang, df_d['z'].to_numpy(), levels=6,
                                  colors='k', linewidths=0.35, alpha=0.35)
        axes[2].clabel(cs2, inline=True, fontsize=7, fmt='%.3f')
        axes[2].scatter(df_d['x'], df_d['y'], s=6, color='black', alpha=0.16, linewidths=0)
    fig.colorbar(cf2, ax=axes[2], label='\u0394I(\u03b8)', shrink=0.88)
    axes[2].set_xlabel(xlabel, fontsize=10, fontweight='bold')
    axes[2].set_ylabel(ylabel, fontsize=10, fontweight='bold')
    axes[2].set_title('Difference\n(MFMC \u2212 MC)', fontsize=11, fontweight='bold')
    axes[2].grid(alpha=0.18)

    fig.suptitle(f'Inconsistency Landscape — S{sid}: {sname}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = output_dir / f'landscape_mfmc_scenario_{sid}.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved landscape: {out.name}")


# ─────────────────────────────────────────────────────────────
# CSV export
# ─────────────────────────────────────────────────────────────

def export_csv(all_results: list, output_dir: Path):
    rows = []
    for sr in all_results:
        if sr is None:
            continue
        for m, r in sr['methods'].items():
            rows.append({
                'scenario_id':        sr['scenario_id'],
                'scenario_name':      sr['scenario_name'],
                'n_experiments':      sr['n_experiments'],
                'sampling_method':    m,
                'alpha':              r['alpha'],
                'rho':                r['rho'],
                'rho_sq':             r['rho_sq'],
                'variance_reduction': r['variance_reduction'],
            })

    if not rows:
        return
    df = pd.DataFrame(rows)
    out = output_dir / 'mfmc_coefficients.csv'
    df.to_csv(out, index=False, float_format='%.6f')
    print(f"Saved coefficients: {out.name}")
    return df


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='MFMC post-processing for I(theta) estimates')
    parser.add_argument('--input',    required=True,  help='Measurements directory with results_scenario_*.json')
    parser.add_argument('--output',   required=True,  help='Output directory for plots and corrected JSONs')
    parser.add_argument('--scenario', nargs='*', type=int, default=None,
                        help='Scenario IDs to process (default: all)')
    parser.add_argument('--save-json', action='store_true',
                        help='Write augmented JSON files with I_MF fields added')
    parser.add_argument('--primary-method', default='sobol',
                        help='Primary MC sampling method for I_MF (default: sobol)')
    args = parser.parse_args()

    input_dir  = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob('results_scenario_*.json'))
    if not json_files:
        print(f"ERROR: No results_scenario_*.json found in {input_dir}")
        return

    # Filter by scenario IDs if requested
    if args.scenario:
        json_files = [f for f in json_files
                      if any(f.name == f'results_scenario_{sid}.json' for sid in args.scenario)]

    print(f"Processing {len(json_files)} scenario file(s)...\n")

    all_results = []
    for jf in json_files:
        sid = int(jf.stem.replace('results_scenario_', ''))
        print(f"Scenario {sid}: {jf.name}")
        data = load_scenario_results(jf)
        sr   = process_scenario(data, sid)
        all_results.append(sr)

        if sr is not None:
            plot_scenario(sr, output_dir)
            plot_mfmc_landscape(sr, output_dir, primary=args.primary_method)

            if args.save_json:
                augmented = augment_json(data, sr, primary_method=args.primary_method)
                out_json = output_dir / jf.name
                with open(out_json, 'w', encoding='utf-8') as f:
                    json.dump(augmented, f, indent=2, allow_nan=False)
                print(f"  Saved augmented JSON: {out_json.name}")
        print()

    plot_summary(all_results, output_dir)
    plot_ablation_study(all_results, output_dir, primary=args.primary_method)
    plot_method_heatmap(all_results, output_dir, primary=args.primary_method)
    df_coef = export_csv(all_results, output_dir)

    if df_coef is not None:
        print("\n--- MFMC Coefficient Summary ---")
        summary = df_coef.groupby('sampling_method')[['rho', 'variance_reduction']].mean()
        summary.columns = ['mean rho', 'mean var_reduction (%)']
        print(summary.to_string(float_format='{:.4f}'.format))

    print(f"\nDone. Outputs in: {output_dir}")


if __name__ == '__main__':
    main()
