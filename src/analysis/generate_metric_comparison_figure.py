#!/usr/bin/env python3
"""
generate_metric_comparison_figure.py
=====================================
Figure G — Estimator Accuracy Comparison (Jaccard vs. MC vs. MFMC)

Four panels:
  Top-left  : Scatter I(θ) [plain MC] vs 1 − Jaccard_C, Spearman ρ annotated.
  Top-right : Scatter I(θ) [plain MC] vs MFMC (1 − I_MF_random), Spearman ρ.
  Bot-left  : Scatter Jaccard vs MFMC, Spearman ρ.
  Bot-right : Violin of all three metrics side-by-side, with median markers.

All metrics on [0, 1] where 1 = fully inconsistent.

Run from project root:
    python src/analysis/generate_metric_comparison_figure.py \\
        --data data/measurements
"""

import argparse
import json
import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':     'serif',
    'font.size':       10,
    'axes.titlesize':  10,
    'axes.labelsize':  10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi':      150,
    'savefig.dpi':     300,
    'savefig.bbox':    'tight',
})

_C_ITHETA  = '#e74c3c'   # red
_C_JACCARD = '#3498db'   # blue
_C_MFMC    = '#2ecc71'   # green

METRIC_COLORS = {
    'MC':           _C_ITHETA,
    r'$1-$Jaccard': _C_JACCARD,
    'MFMC':         _C_MFMC,
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_experiments(data_dirs: list[str]) -> list[dict]:
    experiments = []
    n_files = 0
    for d in data_dirs:
        files = sorted(glob.glob(str(Path(d) / 'results_scenario_*.json')))
        for f in files:
            try:
                with open(f, encoding='utf-8') as fh:
                    data = json.load(fh)
                experiments.extend(data.get('experiments', []))
                n_files += 1
            except Exception:
                pass
    print(f'Loaded {len(experiments)} experiments from {n_files} files.')
    return experiments


def _scalar(exp, state_key, field):
    try:
        v = exp[state_key]['inconsistency'][field]
        if isinstance(v, list):
            v = v[0]
        return float(v) if v is not None else None
    except (KeyError, TypeError, ValueError):
        return None


def build_records(experiments: list[dict]) -> dict[str, np.ndarray]:
    I_theta_list  = []
    jaccard_list  = []
    mfmc_list     = []

    for exp in experiments:
        I_theta  = _scalar(exp, 'post_state', 'I_theta')
        jac_raw  = _scalar(exp, 'post_state', 'jaccard_C')
        mfmc_raw = _scalar(exp, 'post_state', 'I_MF_random')

        if I_theta is None or jac_raw is None or mfmc_raw is None:
            continue

        I_theta_list.append(I_theta)
        jaccard_list.append(1.0 - jac_raw)
        mfmc_list.append(1.0 - mfmc_raw)

    print(f'  {len(I_theta_list)} valid records with all three metrics.')
    return {
        'MC':           np.array(I_theta_list),
        r'$1-$Jaccard': np.array(jaccard_list),
        'MFMC':         np.array(mfmc_list),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _scatter_panel(ax, x_vals, y_vals, x_label, y_label, color_y,
                   max_pts=2000):
    """Draw scatter with y=x reference and Spearman ρ annotation."""
    N = len(x_vals)
    if N > max_pts:
        rng = np.random.default_rng(42)
        idx = rng.choice(N, max_pts, replace=False)
        x_plot, y_plot = x_vals[idx], y_vals[idx]
    else:
        x_plot, y_plot = x_vals, y_vals

    ax.scatter(x_plot, y_plot, color=color_y, s=18, alpha=0.45,
               linewidths=0, rasterized=True)
    ax.plot([0, 1], [0, 1], 'k--', lw=0.9, alpha=0.45, label='$y = x$')

    rho, pval = spearmanr(x_vals, y_vals)
    p_str = f'p<0.001' if pval < 0.001 else f'p={pval:.3f}'
    ax.text(0.04, 0.96, fr'$\rho = {rho:.3f}$  ({p_str})',
            transform=ax.transAxes, va='top', fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.7))

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.spines[['top', 'right']].set_visible(False)
    return rho


# ── Main figure ───────────────────────────────────────────────────────────────

def plot_metric_comparison(data: dict[str, np.ndarray], out_path: Path) -> None:
    I_theta  = data['MC']
    jaccard  = data[r'$1-$Jaccard']
    mfmc     = data['MFMC']

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 7.0))

    # ── Top-left: MC vs Jaccard ───────────────────────────────────────────────
    rho1 = _scatter_panel(
        axes[0, 0], I_theta, jaccard,
        x_label=r'MC',
        y_label=r'$1 - \mathrm{Jaccard}_C$',
        color_y=_C_JACCARD,
    )

    # ── Top-right: MC vs MFMC ─────────────────────────────────────────────────
    rho2 = _scatter_panel(
        axes[0, 1], I_theta, mfmc,
        x_label=r'MC',
        y_label=r'MFMC estimate  $(1 - I_\mathrm{MF})$',
        color_y=_C_MFMC,
    )

    # ── Bot-left: Jaccard vs MFMC ────────────────────────────────────────────
    rho3 = _scatter_panel(
        axes[1, 0], jaccard, mfmc,
        x_label=r'$1 - \mathrm{Jaccard}_C$',
        y_label=r'MFMC estimate  $(1 - I_\mathrm{MF})$',
        color_y=_C_MFMC,
    )

    # ── Bot-right: Violin of all three metrics ────────────────────────────────
    ax = axes[1, 1]
    mc_label = 'MC'
    labels = [mc_label, r'$1-$Jaccard', 'MFMC']
    values = [I_theta, jaccard, mfmc]
    colors = [_C_ITHETA, _C_JACCARD, _C_MFMC]

    vp = ax.violinplot(values, positions=[0, 1, 2],
                       showmedians=True, showextrema=True, widths=0.65)
    for pc, col in zip(vp['bodies'], colors):
        pc.set_facecolor(col)
        pc.set_alpha(0.55)
    vp['cmedians'].set_color('black')
    vp['cmedians'].set_linewidth(1.8)

    # Annotate medians
    for i, (vals, col) in enumerate(zip(values, colors)):
        med = float(np.median(vals))
        ax.text(i + 0.22, med + 0.015, f'{med:.3f}',
                ha='left', va='bottom', fontsize=8, color=col)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Inconsistency score')
    ax.set_ylim(-0.05, 1.1)
    ax.spines[['top', 'right']].set_visible(False)

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        p = out_path.with_suffix(f'.{ext}')
        fig.savefig(p, dpi=300, bbox_inches='tight')
        print(f'  Saved: {p.name}')
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate paper-ready metric comparison figure (Figure G)')
    parser.add_argument('--data', nargs='+',
                        default=['data/measurements'],
                        help='One or more data directories with results_scenario_*.json')
    parser.add_argument('--output', default=None,
                        help='Output directory (default: results/paper_figures)')
    args = parser.parse_args()

    out_dir = Path(args.output) if args.output else Path('results/paper_figures')
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = load_experiments(args.data)
    data        = build_records(experiments)

    if not any(len(v) for v in data.values()):
        print('[ERROR] No valid records found.')
        return

    plot_metric_comparison(data, out_dir / 'figG_metric_comparison')
    print(f'\nDone. Figures saved to {out_dir}')


if __name__ == '__main__':
    main()
