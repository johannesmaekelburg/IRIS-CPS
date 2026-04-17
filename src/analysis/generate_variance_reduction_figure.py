#!/usr/bin/env python3
"""
generate_variance_reduction_figure.py
======================================
Figure F — MFMC Variance Reduction vs. Plain MC

Three panels:
  Left   : Sorted estimate plot — plain MC I(θ) with ±2 SE error band vs.
            MFMC estimate (1 − I_MF_random).  MFMC should track inside the
            MC band but with visibly less scatter.
  Middle : Scatter of MC standard error vs. |correction| = |I_theta − MFMC|.
            Shows that larger corrections occur where MC variance is higher,
            validating the control-variate mechanism.
  Right  : Per-scenario: std(I_theta) vs std(MFMC) as paired bars.
            Directly compares within-scenario variability.

Run from project root:
    python src/analysis/generate_variance_reduction_figure.py \\
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

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':     'serif',
    'font.size':       9,
    'axes.titlesize':  9,
    'axes.labelsize':  9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi':      150,
    'savefig.dpi':     300,
    'savefig.bbox':    'tight',
})

_C_MC   = '#d6604d'
_C_MFMC = '#2166ac'


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


def build_records(experiments: list[dict]) -> list[dict]:
    rows = []
    for exp in experiments:
        I_theta   = _scalar(exp, 'post_state', 'I_theta')
        I_MF_raw  = _scalar(exp, 'post_state', 'I_MF_random')
        mc_se     = _scalar(exp, 'post_state', 'mc_standard_error_random')

        if I_theta is None or I_MF_raw is None or mc_se is None:
            continue

        mfmc      = 1.0 - I_MF_raw
        correction = abs(I_theta - mfmc)
        scen = exp.get('scenario_type', 'unknown')

        rows.append({
            'scenario':   scen,
            'I_theta':    I_theta,
            'mfmc':       mfmc,
            'mc_se':      mc_se,
            'correction': correction,
        })
    print(f'  {len(rows)} valid records extracted.')
    return rows


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_variance_reduction(records: list[dict], out_path: Path) -> None:
    I_theta    = np.array([r['I_theta']    for r in records])
    mfmc       = np.array([r['mfmc']       for r in records])
    mc_se      = np.array([r['mc_se']      for r in records])
    correction = np.array([r['correction'] for r in records])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    # ── Panel 1: Sorted estimate comparison ──────────────────────────────────
    ax = axes[0]
    # Sample up to 400 points for clarity; sort by MC estimate
    N_show = min(400, len(I_theta))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(I_theta), N_show, replace=False)
    order = np.argsort(I_theta[idx])
    x_plot  = np.arange(N_show)
    mc_s    = I_theta[idx][order]
    mf_s    = mfmc[idx][order]
    se_s    = mc_se[idx][order]

    ax.fill_between(x_plot, mc_s - 2*se_s, mc_s + 2*se_s,
                    color=_C_MC, alpha=0.20, label=r'MC ± 2 SE')
    ax.plot(x_plot, mc_s, color=_C_MC, lw=1.2, alpha=0.7,
            label=r'MC $I(\theta)$')
    ax.scatter(x_plot, mf_s, color=_C_MFMC, s=3, alpha=0.55,
               linewidths=0, zorder=3, label='MFMC estimate')

    ax.set_xlabel('Experiments (sorted by MC estimate)')
    ax.set_ylabel('Inconsistency score')
    ax.set_title(f'MC vs. MFMC Estimates\n(N={N_show} sample, sorted)', pad=5)
    ax.legend(fontsize=7.5, frameon=False)
    ax.set_ylim(-0.05, 1.05)
    ax.spines[['top', 'right']].set_visible(False)

    # ── Panel 2: MC SE vs correction magnitude ────────────────────────────────
    ax = axes[1]
    sc = ax.scatter(mc_se, correction, c=I_theta,
                    cmap='RdYlBu_r', s=5, alpha=0.35,
                    linewidths=0, rasterized=True)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r'$I(\theta)$ (MC)', fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # Trend line
    if len(mc_se) > 10:
        from numpy.polynomial.polynomial import polyfit as pfit
        c0, c1 = pfit(mc_se, correction, 1)
        x_line = np.linspace(mc_se.min(), mc_se.max(), 100)
        ax.plot(x_line, c0 + c1*x_line, 'k--', lw=1.2, alpha=0.6,
                label=f'Linear fit (slope={c1:.2f})')
        ax.legend(fontsize=7.5, frameon=False)

    med_corr = np.median(correction)
    ax.axhline(med_corr, color='grey', lw=0.8, ls=':')
    ax.text(ax.get_xlim()[1] * 0.98, med_corr + 0.003,
            f'median={med_corr:.3f}', ha='right', fontsize=7.5, color='grey')

    ax.set_xlabel('MC standard error  (SE)')
    ax.set_ylabel(r'|Correction|  $= |I_\mathrm{MC} - I_\mathrm{MFMC}|$')
    ax.set_title('MFMC Correction vs. MC Variance\n(colour = MC estimate)', pad=5)
    ax.spines[['top', 'right']].set_visible(False)

    # ── Panel 3: Per-scenario std comparison ─────────────────────────────────
    ax = axes[2]
    scenarios = sorted(set(r['scenario'] for r in records))
    std_mc   = []
    std_mfmc = []
    for s in scenarios:
        sub = [r for r in records if r['scenario'] == s]
        std_mc.append(np.std([r['I_theta'] for r in sub]))
        std_mfmc.append(np.std([r['mfmc']  for r in sub]))

    std_mc   = np.array(std_mc)
    std_mfmc = np.array(std_mfmc)
    order_s  = np.argsort(std_mc)[::-1]
    x_s = np.arange(len(scenarios))
    w   = 0.35

    bars1 = ax.bar(x_s - w/2, std_mc[order_s],   w, color=_C_MC,   alpha=0.7,
                   label=r'std MC $I(\theta)$')
    bars2 = ax.bar(x_s + w/2, std_mfmc[order_s], w, color=_C_MFMC, alpha=0.7,
                   label='std MFMC')

    scen_labels = [scenarios[i].replace('scenario_', 'S') for i in order_s]
    ax.set_xticks(x_s)
    ax.set_xticklabels(scen_labels, rotation=60, ha='right', fontsize=6.5)
    ax.set_ylabel('Within-scenario std')
    ax.set_title('Per-Scenario Variability\n(MC vs. MFMC)', pad=5)
    ax.legend(fontsize=7.5, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)

    # Overall reduction
    mean_reduction = float(np.mean(std_mc - std_mfmc))
    pct_reduction  = float(np.mean((std_mc - std_mfmc) / (std_mc + 1e-12))) * 100
    fig.suptitle(
        f'MFMC Variance Reduction  '
        f'(mean std reduction: {mean_reduction:.4f},  {pct_reduction:.1f}%)',
        fontsize=10, y=1.01,
    )

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        p = out_path.with_suffix(f'.{ext}')
        fig.savefig(p, dpi=300, bbox_inches='tight')
        print(f'  Saved: {p.name}')
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate MFMC variance reduction figure (Figure F)')
    parser.add_argument('--data', nargs='+',
                        default=['data/measurements'],
                        help='One or more data directories with results_scenario_*.json')
    parser.add_argument('--output', default=None,
                        help='Output directory (default: results/paper_figures)')
    args = parser.parse_args()

    out_dir = Path(args.output) if args.output else Path('results/paper_figures')
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = load_experiments(args.data)
    records     = build_records(experiments)

    if not records:
        print('[ERROR] No valid records found.')
        return

    plot_variance_reduction(records, out_dir / 'figF_variance_reduction')
    print(f'\nDone. Figures saved to {out_dir}')


if __name__ == '__main__':
    main()
