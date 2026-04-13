#!/usr/bin/env python3
"""
generate_paper_figures.py
=========================
Generates two ICDM-ready paper figures from existing analysis output:

  Figure A (RQ3): Aggregate Sobol indices (S1 + ST) across all scenarios
                  with mean ± std error bars, grouped by parameter.

  Figure B (RQ2+RQ4): Causal (forward, Sobol ST) vs. Anticausal (backward,
                       η² / conditional means) side-by-side for each parameter.

Run from project root:
    python src/analysis/generate_paper_figures.py --results results/measurements_cps_plus_measurements_mfmc
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Style ────────────────────────────────────────────────────────────────────
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

PARAM_LABELS = {
    'scale_factor':        r'Scale $s_u$',
    'center_delta':        r'Center shift $\Delta c_u$',
    'correlation_strength': r'Correlation $R_u$',
}
PARAM_ORDER = ['scale_factor', 'center_delta', 'correlation_strength']

ANTICAUSAL_PARAM_MAP = {
    'scale_factor':         's_u',
    'center_delta':         'delta_c_u',
    'correlation_strength': 'R_u',
}

COLORS = {
    'S1':  '#2166ac',
    'ST':  '#d6604d',
    'eta2': '#4dac26',
    'MI':   '#7b3294',
}


# ── Data loading ─────────────────────────────────────────────────────────────

def _find_sobol_files(results_dir: Path) -> list[Path]:
    return sorted(results_dir.rglob('sobol_indices.json'))


def _load_all_sobol(results_dir: Path) -> dict:
    """
    Returns {param_name: {'S1': [...], 'ST': [...]}} across all scenarios.
    Clips S1 to [0, 1] (negative values from small N are noise).
    """
    files = _find_sobol_files(results_dir)
    if not files:
        raise FileNotFoundError(f'No sobol_indices.json found under {results_dir}')

    data = {p: {'S1': [], 'ST': []} for p in PARAM_ORDER}
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        si = d.get('sobol_indices', {})
        fo = si.get('first_order', {})
        te = si.get('total_effect', {})
        for p in PARAM_ORDER:
            if p in fo and p in te:
                s1 = max(0.0, float(fo[p]['S1']))   # clip negative noise
                st = max(0.0, float(te[p]['ST']))
                data[p]['S1'].append(s1)
                data[p]['ST'].append(st)

    print(f'  Loaded Sobol from {len(files)} scenario files.')
    return data


def _load_anticausal_summary(results_dir: Path) -> dict:
    """Returns the anticausal_summary.json as a dict."""
    path = results_dir / 'anticausal_inference' / 'anticausal_summary.json'
    if not path.exists():
        raise FileNotFoundError(f'anticausal_summary.json not found at {path}')
    with open(path) as fh:
        return json.load(fh)


# ── Figure A: Aggregate Sobol ─────────────────────────────────────────────────

def plot_aggregate_sobol(sobol_data: dict, out_path: Path) -> None:
    """
    Grouped bar chart: for each parameter, S1 and ST bars with ±1 std error.
    """
    params  = PARAM_ORDER
    labels  = [PARAM_LABELS[p] for p in params]
    n_params = len(params)

    s1_means = [np.mean(sobol_data[p]['S1']) for p in params]
    s1_stds  = [np.std(sobol_data[p]['S1'])  for p in params]
    st_means = [np.mean(sobol_data[p]['ST']) for p in params]
    st_stds  = [np.std(sobol_data[p]['ST'])  for p in params]
    n_scen   = len(sobol_data[params[0]]['S1'])

    x     = np.arange(n_params)
    width = 0.35

    fig, ax = plt.subplots(figsize=(5.5, 3.4))

    bars1 = ax.bar(x - width / 2, s1_means, width,
                   yerr=s1_stds, capsize=4,
                   color=COLORS['S1'], alpha=0.85, label=r'First-order $S_1$',
                   error_kw={'elinewidth': 1.2})
    bars2 = ax.bar(x + width / 2, st_means, width,
                   yerr=st_stds, capsize=4,
                   color=COLORS['ST'], alpha=0.85, label=r'Total-effect $S_T$',
                   error_kw={'elinewidth': 1.2})

    # Annotate mean values above bars
    for bar, mean in zip(bars1, s1_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f'{mean:.2f}', ha='center', va='bottom', fontsize=7)
    for bar, mean in zip(bars2, st_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f'{mean:.2f}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Sobol Index')
    ax.set_ylim(0, min(1.05, max(st_means) + max(st_stds) + 0.18))
    ax.set_title(f'Aggregate Sobol Indices across {n_scen} Scenarios (mean ± std)',
                 pad=6)
    ax.legend(loc='upper right')
    ax.spines[['top', 'right']].set_visible(False)
    ax.axhline(0, color='k', linewidth=0.5)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f'  Saved: {out_path.name}')


# ── Figure B: Causal vs Anticausal ───────────────────────────────────────────

def plot_causal_vs_anticausal(sobol_data: dict,
                               anticausal: dict,
                               out_path: Path) -> None:
    """
    Side-by-side for each parameter:
      Left panel:  Sobol ST distribution (violin / box)
      Right panel: Anticausal η² bar + conditional mean trace
    """
    params   = PARAM_ORDER
    n_params = len(params)

    # Extract anticausal data
    effect_map = {e['param']: e for e in anticausal['effect_sizes']}
    cond_stats  = anticausal['conditional_stats']
    bin_labels  = anticausal['bin_labels']
    n_bins      = len(bin_labels)

    fig = plt.figure(figsize=(7.0, 3.0 * n_params))
    gs  = gridspec.GridSpec(n_params, 2, figure=fig,
                            hspace=0.55, wspace=0.38,
                            left=0.10, right=0.97, top=0.93, bottom=0.06)

    for row, p in enumerate(params):
        ac_key = ANTICAUSAL_PARAM_MAP[p]
        label  = PARAM_LABELS[p]

        # ── Left: Sobol ST distribution ───────────────────────────────────
        ax_l = fig.add_subplot(gs[row, 0])
        st_vals = np.array(sobol_data[p]['ST'])
        s1_vals = np.array(sobol_data[p]['S1'])

        vp = ax_l.violinplot([s1_vals, st_vals], positions=[0, 1],
                             showmedians=True, showextrema=True)
        for pc in vp['bodies']:
            pc.set_alpha(0.6)
        vp['bodies'][0].set_facecolor(COLORS['S1'])
        vp['bodies'][1].set_facecolor(COLORS['ST'])

        ax_l.set_xticks([0, 1])
        ax_l.set_xticklabels([r'$S_1$', r'$S_T$'])
        ax_l.set_ylabel('Sobol index')
        ax_l.set_title(f'{label}\nForward causal (Sobol)', fontsize=8)
        ax_l.set_ylim(-0.05, 1.05)
        ax_l.axhline(0, color='k', linewidth=0.5, linestyle='--')
        ax_l.spines[['top', 'right']].set_visible(False)

        # ── Right: η² + conditional mean trace ────────────────────────────
        ax_r = fig.add_subplot(gs[row, 1])

        # Conditional mean trace (primary, left axis)
        cond_means = []
        bin_mean_I = []
        for b in range(n_bins):
            row_data = next(
                (r for r in cond_stats if r['bin'] == b and r['param'] == ac_key),
                None
            )
            if row_data:
                cond_means.append(row_data['cond_mean'])
                bin_mean_I.append(row_data['bin_mean_I'])

        ax_r.plot(range(len(cond_means)), cond_means,
                  color=COLORS['eta2'], marker='o', linewidth=1.8,
                  markersize=5, label=r'$E[\theta | \mathrm{bin}]$')
        ax_r.set_xticks(range(n_bins))
        ax_r.set_xticklabels([f'B{i}' for i in range(n_bins)], fontsize=7)
        ax_r.set_xlabel(r'Inconsistency bin $I(\theta)$')
        ax_r.set_ylabel(r'Conditional mean of $\theta$')
        ax_r.spines[['top', 'right']].set_visible(False)

        # η² annotation
        eta2 = effect_map.get(ac_key, {}).get('eta2', float('nan'))
        mi   = effect_map.get(ac_key, {}).get('MI',   float('nan'))
        ax_r.set_title(
            f'{label}\nAnticausal: '
            r'$\eta^2$' + f'={eta2:.3f},  MI={mi:.3f}',
            fontsize=8
        )

        # η² as a shaded band height indicator on secondary axis
        ax_r2 = ax_r.twinx()
        ax_r2.set_ylim(0, 1)
        ax_r2.bar(n_bins - 0.5, eta2, width=0.4,
                  color=COLORS['eta2'], alpha=0.25, label=r'$\eta^2$')
        ax_r2.set_ylabel(r'$\eta^2$', color=COLORS['eta2'], fontsize=8)
        ax_r2.tick_params(axis='y', labelcolor=COLORS['eta2'])
        ax_r2.spines[['top']].set_visible(False)

    fig.suptitle('Causal (Forward) vs. Anticausal (Backward) Analysis per Parameter',
                 fontsize=10, y=0.98)

    fig.savefig(out_path)
    plt.close(fig)
    print(f'  Saved: {out_path.name}')


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description='Generate ICDM paper figures')
    parser.add_argument(
        '--results',
        default='results/measurements_cps_plus_measurements_mfmc',
        help='Root results directory (default: %(default)s)',
    )
    args = parser.parse_args()

    results_dir = Path(args.results)
    out_dir     = results_dir / 'paper_figures'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Results dir: {results_dir}')
    print(f'Output dir:  {out_dir}\n')

    # Load data
    print('Loading Sobol indices...')
    sobol_data = _load_all_sobol(results_dir)

    print('Loading anticausal summary...')
    anticausal = _load_anticausal_summary(results_dir)

    # Generate figures
    print('\nGenerating Figure A — Aggregate Sobol...')
    plot_aggregate_sobol(sobol_data, out_dir / 'figA_aggregate_sobol.pdf')
    plot_aggregate_sobol(sobol_data, out_dir / 'figA_aggregate_sobol.png')

    print('\nGenerating Figure B — Causal vs Anticausal...')
    plot_causal_vs_anticausal(sobol_data, anticausal,
                               out_dir / 'figB_causal_vs_anticausal.pdf')
    plot_causal_vs_anticausal(sobol_data, anticausal,
                               out_dir / 'figB_causal_vs_anticausal.png')

    print(f'\nDone. Figures saved to {out_dir}')


if __name__ == '__main__':
    main()
