#!/usr/bin/env python3
"""
generate_causal_sensitivity_figure.py
======================================
Figure H — Causal Effects and Robustness Margins

Three rows (one per uncertainty parameter: scale_factor, center_delta, correlation_strength).
Two columns:

  Col 1 — Total causal effects τ(a,b):
           Bar chart of τ = E[I(θ)|do(θ=b)] − E[I(θ)|do(θ=a)] for quantile
           pairs (Q1→Q3, Q1→Q4, Q2→Q4).  Error bars = propagated SE.

  Col 2 — Robustness margins s*(τ):
           Interpolated I(θ) curve vs parameter value with threshold lines
           (τ = 0.3, 0.5, 0.7) and shaded safe region below lowest threshold.

Run from project root:
    python src/analysis/generate_causal_sensitivity_figure.py \\
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
from scipy.interpolate import interp1d

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':     'serif',
    'font.size':       9,
    'axes.titlesize':  9,
    'axes.labelsize':  9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'figure.dpi':      150,
    'savefig.dpi':     300,
    'savefig.bbox':    'tight',
})

PARAMS = ['scale_factor', 'center_delta', 'correlation_strength']
PARAM_LABELS = {
    'scale_factor':         r'Scale $s_u$',
    'center_delta':         r'Center shift $\Delta c_u$',
    'correlation_strength': r'Correlation $R_u$',
}

THRESHOLDS = [0.9]
THRESHOLD_COLORS = ['#d6604d']

_C_TAU   = '#2166ac'
_C_SENS  = '#7b3294'
_C_CURVE = '#d6604d'


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(data_dirs: list[str]) -> list[dict]:
    rows = []
    n_files = 0
    for d in data_dirs:
        for f in sorted(glob.glob(str(Path(d) / 'results_scenario_*.json'))):
            try:
                data = json.loads(Path(f).read_text(encoding='utf-8'))
            except Exception:
                continue
            n_files += 1
            for exp in data.get('experiments', []):
                if exp.get('intervention_type') != 'compound':
                    continue
                mfmc_raw = _s(exp, 'post_state', 'I_MF_random')
                if mfmc_raw is None:
                    continue
                row = {'I_theta': 1.0 - mfmc_raw}
                for p in PARAMS:
                    v = exp.get(p)
                    if v is not None:
                        try:
                            row[p] = float(v)
                        except (TypeError, ValueError):
                            pass
                if all(p in row for p in PARAMS):
                    rows.append(row)
    print(f'Loaded {len(rows)} compound experiments from {n_files} files.')
    return rows


def _s(exp, state_key, field):
    try:
        v = exp[state_key]['inconsistency'][field]
        if isinstance(v, list):
            v = v[0]
        return float(v) if v is not None else None
    except (KeyError, TypeError, ValueError):
        return None


# ── Analysis helpers ──────────────────────────────────────────────────────────

def _binned_means(vals: np.ndarray, I: np.ndarray, n_bins: int = 20):
    """Return (bin_centers, mean_I, sem_I) after binning vals."""
    lo, hi = np.nanpercentile(vals, 1), np.nanpercentile(vals, 99)
    edges = np.linspace(lo, hi, n_bins + 1)
    centers, means, sems = [], [], []
    for i in range(n_bins):
        mask = (vals >= edges[i]) & (vals < edges[i + 1])
        if mask.sum() < 3:
            continue
        centers.append((edges[i] + edges[i + 1]) / 2)
        means.append(np.mean(I[mask]))
        sems.append(np.std(I[mask]) / np.sqrt(mask.sum()))
    return np.array(centers), np.array(means), np.array(sems)


def _causal_effects(vals: np.ndarray, I: np.ndarray):
    """
    Compute τ for quantile pairs: Q1→Q3, Q1→Q4 (large shift), Q2→Q3 (medium).
    Returns list of (label, tau, tau_se).
    """
    q1, q2, q3, q4 = np.percentile(vals, [10, 33, 67, 90])
    pairs = [
        (r'$Q_{10}→Q_{90}$', q1, q4),
        (r'$Q_{10}→Q_{67}$', q1, q3),
        (r'$Q_{33}→Q_{90}$', q2, q4),
    ]
    tol = (vals.max() - vals.min()) * 0.15
    results = []
    for label, a, b in pairs:
        mask_a = np.abs(vals - a) < tol
        mask_b = np.abs(vals - b) < tol
        if mask_a.sum() < 2 or mask_b.sum() < 2:
            continue
        Ia, Ib = I[mask_a], I[mask_b]
        tau    = Ib.mean() - Ia.mean()
        tau_se = np.sqrt(Ia.std(ddof=1)**2 / len(Ia) + Ib.std(ddof=1)**2 / len(Ib))
        results.append((label, tau, tau_se))
    return results


def _local_sensitivity(centers: np.ndarray, means: np.ndarray):
    """Finite-difference dI/dθ at each bin center."""
    n = len(centers)
    grad = np.empty(n)
    for i in range(n):
        if i == 0:
            grad[i] = (means[1] - means[0]) / (centers[1] - centers[0] + 1e-12)
        elif i == n - 1:
            grad[i] = (means[-1] - means[-2]) / (centers[-1] - centers[-2] + 1e-12)
        else:
            grad[i] = (means[i + 1] - means[i - 1]) / (centers[i + 1] - centers[i - 1] + 1e-12)
    return grad


def _robustness_curve(centers: np.ndarray, means: np.ndarray):
    """Return smooth interpolated curve over a dense grid."""
    kind = 'linear' if len(centers) <= 3 else ('quadratic' if len(centers) <= 5 else 'cubic')
    f = interp1d(centers, means, kind=kind, fill_value='extrapolate')
    x_dense = np.linspace(centers[0], centers[-1], 300)
    return x_dense, np.clip(f(x_dense), 0, 1)


# ── Main figure ───────────────────────────────────────────────────────────────

def plot_causal_sensitivity(rows: list[dict], out_path: Path) -> None:
    import pandas as pd
    df = pd.DataFrame(rows)
    I  = df['I_theta'].values

    # Two columns: causal effects (left) + robustness margins (right)
    fig, axes = plt.subplots(len(PARAMS), 2,
                              figsize=(8, 2.8 * len(PARAMS)))

    for row_idx, param in enumerate(PARAMS):
        label  = PARAM_LABELS[param]
        vals   = df[param].values
        ctrs, mn, se = _binned_means(vals, I, n_bins=25)

        if len(ctrs) < 2:
            for c in range(2):
                axes[row_idx, c].text(0.5, 0.5, 'insufficient data',
                                       ha='center', va='center',
                                       transform=axes[row_idx, c].transAxes)
            continue

        # ── Row label ─────────────────────────────────────────────────────────
        axes[row_idx, 0].annotate(
            label, xy=(-0.22, 0.5), xycoords='axes fraction',
            ha='right', va='center', fontsize=9, fontweight='bold',
            rotation=90,
        )

        # ── Col 0: Total causal effects ───────────────────────────────────────
        ax = axes[row_idx, 0]
        effects = _causal_effects(vals, I)
        if effects:
            eff_labels = [e[0] for e in effects]
            taus       = np.array([e[1] for e in effects])
            tau_ses    = np.array([e[2] for e in effects])
            x = np.arange(len(effects))
            # One distinct colour per bar — matches legend patches
            _BAR_COLORS = [_C_TAU, '#5b9bd5', '#a8c8e8']
            bar_colors = [_BAR_COLORS[i] if taus[i] >= 0 else _C_CURVE
                          for i in range(len(taus))]
            ax.bar(x, taus, yerr=tau_ses, capsize=5,
                   color=bar_colors,
                   alpha=0.85, error_kw={'elinewidth': 1.2})
            ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(eff_labels, fontsize=7.5)
            ax.set_ylabel(r'$\tau(a,b)$')
        else:
            ax.text(0.5, 0.5, 'insufficient data', ha='center', va='center',
                    transform=ax.transAxes)
        ax.spines[['top', 'right']].set_visible(False)

        # ── Col 1: Robustness margins ─────────────────────────────────────────
        ax = axes[row_idx, 1]
        x_dense, I_dense = _robustness_curve(ctrs, mn)

        # Shade safe region (below threshold) in green
        tau_min = THRESHOLDS[0]
        ax.fill_between(x_dense, 0, tau_min,
                        color='#4dac26', alpha=0.08,
                        label=f'Safe ($I < {tau_min}$)')
        # Shade unsafe region (above threshold) in red
        ax.fill_between(x_dense, tau_min, 1.0,
                        color='#d6604d', alpha=0.06,
                        label=f'Unsafe ($I \\geq {tau_min}$)')

        ax.plot(x_dense, I_dense, color=_C_CURVE, lw=2.0, label=r'$E[I(\theta)]$')
        ax.scatter(ctrs, mn, color=_C_CURVE, s=20, zorder=4, alpha=0.8)

        for tau, tc in zip(THRESHOLDS, THRESHOLD_COLORS):
            ax.axhline(tau, color=tc, lw=1.1, ls='--',
                       label=fr'$\tau={tau}$')
            for i in range(len(I_dense) - 1):
                if (I_dense[i] <= tau < I_dense[i + 1]) or \
                   (I_dense[i] > tau >= I_dense[i + 1]):
                    alpha = (tau - I_dense[i]) / (I_dense[i + 1] - I_dense[i] + 1e-12)
                    s_star = x_dense[i] + alpha * (x_dense[i + 1] - x_dense[i])
                    ax.axvline(s_star, color=tc, lw=0.8, ls=':', alpha=0.7)
                    ax.text(s_star, tau + 0.03,
                            fr'$s^*={s_star:.2f}$',
                            color=tc, fontsize=6.5, ha='center')
                    break

        ax.set_xlabel(label)
        ax.set_ylabel(r'$E[I(\theta)]$')
        # Y-axis: start just below data min, end just above 1
        y_lo = max(0.0, np.nanmin(mn) - 0.08)
        ax.set_ylim(y_lo, 1.05)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.legend_.remove() if ax.get_legend() else None
        ax.spines[['top', 'right']].set_visible(False)

    fig.tight_layout(rect=[0.08, 0.08, 1, 1])  # left margin for row labels, bottom for legend

    # ── Single shared legend below the whole figure ───────────────────────────
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines

    all_handles = [
        # Bar label group — colours must match _BAR_COLORS above
        mpatches.Patch(color=_C_TAU,    alpha=0.85,
                       label=r'$Q_{10}{\to}Q_{90}$: full range'),
        mpatches.Patch(color='#5b9bd5', alpha=0.85,
                       label=r'$Q_{10}{\to}Q_{67}$: low$\to$mid'),
        mpatches.Patch(color='#a8c8e8', alpha=0.85,
                       label=r'$Q_{33}{\to}Q_{90}$: mid$\to$high'),
        # Spacer
        mpatches.Patch(color='none', label=''),
        # Margin symbol group
        mlines.Line2D([], [], color=_C_CURVE, lw=2.0,
                      label=r'$E[I(\theta)]$'),
        mlines.Line2D([], [], color=THRESHOLD_COLORS[0], lw=1.1, ls='--',
                      label=fr'$\tau={THRESHOLDS[0]}$'),
        mpatches.Patch(color='#4dac26', alpha=0.3,
                       label=r'Safe ($I<\tau$)'),
        mpatches.Patch(color='#d6604d', alpha=0.25,
                       label=r'Unsafe ($I\geq\tau$)'),
        mlines.Line2D([], [], color=THRESHOLD_COLORS[0], lw=0.8, ls=':',
                      label=r'$s^*$ crossing'),
    ]
    fig.legend(handles=all_handles,
               loc='lower center', bbox_to_anchor=(0.5, 0.0),
               ncol=9, fontsize=7.5, frameon=True,
               columnspacing=0.8, handlelength=1.2)

    for ext in ('pdf', 'png'):
        p = out_path.with_suffix(f'.{ext}')
        fig.savefig(p, dpi=300, bbox_inches='tight')
        print(f'  Saved: {p.name}')
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate causal effects / sensitivity / robustness figure (Figure H)')
    parser.add_argument('--data', nargs='+',
                        default=['data/measurements'])
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    out_dir = Path(args.output) if args.output else Path('results/paper_figures')
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_data(args.data)
    if not rows:
        print('[ERROR] No compound experiments found.')
        return

    plot_causal_sensitivity(rows, out_dir / 'figH_causal_sensitivity')
    print(f'\nDone. Figures saved to {out_dir}')


if __name__ == '__main__':
    main()
