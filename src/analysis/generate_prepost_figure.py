#!/usr/bin/env python3
"""
generate_prepost_figure.py
==========================
Figure E — Pre- vs. Post-Intervention Inconsistency

Three panels:
  Left   : Scatter pre vs post I(θ), y=x reference, coloured by scale_factor.
            Answers: "do interventions increase inconsistency?"
  Middle : Violin of Δ = post − pre per scenario (sorted by median Δ).
            Answers: "by how much, and which scenarios are most affected?"
  Right  : Paired box pre/post for each metric (I_theta, MFMC, Jaccard).
            Answers: "is the shift consistent across all metrics?"

Run from project root:
    python src/analysis/generate_prepost_figure.py \\
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
import matplotlib.ticker as mticker

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'serif',
    'font.size':        9,
    'axes.titlesize':   9,
    'axes.labelsize':   9,
    'xtick.labelsize':  8,
    'ytick.labelsize':  8,
    'legend.fontsize':  8,
    'figure.dpi':       150,
    'savefig.dpi':      300,
    'savefig.bbox':     'tight',
})

_C_PRE   = '#2166ac'
_C_POST  = '#d6604d'
_C_DELTA = '#4dac26'


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


def _inc(exp, state_key, field):
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
        # Use MFMC-corrected inconsistency (= 1 - I_MF_random) as primary metric
        pre_mfmc_raw  = _inc(exp, 'pre_state',  'I_MF_random')
        post_mfmc_raw = _inc(exp, 'post_state', 'I_MF_random')
        if pre_mfmc_raw is None or post_mfmc_raw is None:
            continue

        pre_I   = 1.0 - pre_mfmc_raw
        post_I  = 1.0 - post_mfmc_raw

        # Plain MC I_theta for the right panel
        pre_Itheta  = _inc(exp, 'pre_state',  'I_theta')
        post_Itheta = _inc(exp, 'post_state', 'I_theta')

        # Jaccard inconsistency (1 - jaccard_C)
        pre_jac_raw  = _inc(exp, 'pre_state',  'jaccard_C')
        post_jac_raw = _inc(exp, 'post_state', 'jaccard_C')
        pre_jac  = (1.0 - pre_jac_raw)  if pre_jac_raw  is not None else None
        post_jac = (1.0 - post_jac_raw) if post_jac_raw is not None else None

        sf  = exp.get('scale_factor')
        cd  = exp.get('center_delta')
        cs  = exp.get('correlation_strength')
        scen = exp.get('scenario_type', 'unknown')

        rows.append({
            'scenario':    scen,
            'scale_factor': float(sf) if sf is not None else np.nan,
            'center_delta': float(cd) if cd is not None else np.nan,
            'corr_strength': float(cs) if cs is not None else np.nan,
            'pre_I':    pre_I,
            'post_I':   post_I,
            'delta_I':  post_I - pre_I,
            'pre_Itheta':  pre_Itheta,
            'post_Itheta': post_Itheta,
            'pre_jac':  pre_jac,
            'post_jac': post_jac,
        })
    print(f'  {len(rows)} valid pre/post pairs extracted.')
    return rows


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_prepost(records: list[dict], out_path: Path) -> None:
    pre_I  = np.array([r['pre_I']  for r in records])
    post_I = np.array([r['post_I'] for r in records])
    delta  = np.array([r['delta_I'] for r in records])
    sf     = np.array([r['scale_factor'] for r in records])

    # Scenarios sorted by median delta
    scenarios = sorted(set(r['scenario'] for r in records))
    deltas_by_scen = {s: [r['delta_I'] for r in records if r['scenario'] == s]
                      for s in scenarios}
    scenarios_sorted = sorted(scenarios,
                               key=lambda s: np.median(deltas_by_scen[s]))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    # ── Panel 1: Pre vs Post scatter ──────────────────────────────────────────
    ax = axes[0]
    sc = ax.scatter(pre_I, post_I, c=sf, cmap='viridis',
                    s=8, alpha=0.45, linewidths=0, rasterized=True)
    ax.plot([0, 1], [0, 1], 'k--', lw=1.0, alpha=0.5, label='$y = x$ (no change)')
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Scale factor $s_u$', fontsize=8)
    cb.ax.tick_params(labelsize=7)

    frac_above = np.mean(post_I > pre_I)
    ax.text(0.04, 0.96,
            f'{100*frac_above:.0f}% have post > pre',
            transform=ax.transAxes, va='top', fontsize=8,
            color=_C_POST)
    ax.set_xlabel(r'Pre-intervention $I(\theta)$')
    ax.set_ylabel(r'Post-intervention $I(\theta)$')
    ax.set_title('Pre vs. Post Inconsistency\n(MFMC estimate)', pad=5)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=7, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)

    # ── Panel 2: Violin of Δ per scenario ─────────────────────────────────────
    ax = axes[1]
    vdata   = [deltas_by_scen[s] for s in scenarios_sorted]
    vp = ax.violinplot(vdata, positions=range(len(scenarios_sorted)),
                       showmedians=True, showextrema=False, widths=0.7)
    for pc in vp['bodies']:
        pc.set_facecolor(_C_DELTA)
        pc.set_alpha(0.55)
    vp['cmedians'].set_color('black')
    vp['cmedians'].set_linewidth(1.5)

    ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
    ax.set_xticks(range(len(scenarios_sorted)))
    ax.set_xticklabels([s.replace('scenario_', 'S') for s in scenarios_sorted],
                       rotation=60, ha='right', fontsize=6.5)
    ax.set_ylabel(r'$\Delta I(\theta) = I_\mathrm{post} - I_\mathrm{pre}$')
    ax.set_title('Inconsistency Shift per Scenario\n(sorted by median Δ)', pad=5)
    ax.spines[['top', 'right']].set_visible(False)

    # ── Panel 3: Paired box pre/post per metric ────────────────────────────────
    ax = axes[2]

    metrics = {
        'MFMC':   ('pre_I',      'post_I'),
        r'$I(\theta)$': ('pre_Itheta',  'post_Itheta'),
        'Jaccard':('pre_jac',    'post_jac'),
    }
    positions = []
    labels    = []
    bdata     = []
    colors    = []
    x = 0
    xtick_pos = []
    xtick_lab = []
    for name, (pre_k, post_k) in metrics.items():
        pre_vals  = [r[pre_k]  for r in records if r[pre_k]  is not None]
        post_vals = [r[post_k] for r in records if r[post_k] is not None]
        if not pre_vals:
            x += 3
            continue
        bdata.extend([pre_vals, post_vals])
        positions.extend([x, x + 1])
        colors.extend([_C_PRE, _C_POST])
        xtick_pos.append(x + 0.5)
        xtick_lab.append(name)
        x += 3

    bp = ax.boxplot(bdata, positions=positions, widths=0.6,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color='black', linewidth=1.5),
                    whiskerprops=dict(linewidth=1.0),
                    boxprops=dict(linewidth=1.0),
                    capprops=dict(linewidth=1.0))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    # Legend patches
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=_C_PRE,  alpha=0.65, label='Pre'),
                        Patch(facecolor=_C_POST, alpha=0.65, label='Post')],
              fontsize=8, frameon=False, loc='upper left')

    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_lab, fontsize=8)
    ax.set_ylabel('Inconsistency score')
    ax.set_title('Pre vs. Post by Metric\n(box = IQR, line = median)', pad=5)
    ax.set_ylim(-0.05, 1.1)
    ax.spines[['top', 'right']].set_visible(False)

    n = len(records)
    med_delta = float(np.median(delta))
    fig.suptitle(
        fr'Pre/Post Intervention Inconsistency  '
        fr'(N={n},  median $\Delta I(\theta)$ = {med_delta:+.3f})',
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
    parser = argparse.ArgumentParser(description='Generate pre/post intervention figure (Figure E)')
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
        print('[ERROR] No valid pre/post pairs found.')
        return

    plot_prepost(records, out_dir / 'figE_prepost_intervention')
    print(f'\nDone. Figures saved to {out_dir}')


if __name__ == '__main__':
    main()
