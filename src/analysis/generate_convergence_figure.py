#!/usr/bin/env python3
"""
generate_convergence_figure.py
==============================
Reads mc_convergence_* arrays (written by the instrumented MATLAB engine)
and produces Figure D for the ICDM paper:

  "MC Consistency Score Convergence vs. Sample Count"

  - X-axis: number of MC samples (100, 200, …, N_max)
  - Y-axis: running mean consistency score estimate
  - One curve per sampling method (Sobol, Random, …)
  - Mean ± std band across all experiments / states

Run from project root:
    python src/analysis/generate_convergence_figure.py --data data/measurements_cps/measurements_cps
"""

import argparse
import json
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

# Display name and colour per sampling method suffix
METHOD_STYLE = {
    'sobol':  {'label': 'MC Sobol (QMC)',   'color': '#2166ac', 'ls': '-'},
    'random': {'label': 'MC Random (pseudo-random)', 'color': '#d6604d', 'ls': '--'},
    'halton': {'label': 'MC Halton (QMC)',  'color': '#4dac26', 'ls': '-.'},
}


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_experiments(data_dir: str) -> list[dict]:
    files = sorted(Path(data_dir).glob('results_scenario_*.json'))
    if not files:
        raise FileNotFoundError(f'No results_scenario_*.json in {data_dir}')
    experiments = []
    for f in files:
        with open(f, encoding='utf-8') as fh:
            data = json.load(fh)
        experiments.extend(data.get('experiments', []))
    print(f'  Loaded {len(experiments)} experiments from {len(files)} files.')
    return experiments


def _get_inc(exp: dict, state_key: str, field: str):
    try:
        return exp[state_key]['inconsistency'][field]
    except (KeyError, TypeError):
        return None


# ── Data extraction ───────────────────────────────────────────────────────────

def collect_convergence_data(experiments: list[dict]) -> dict:
    """
    Returns {method_suffix: {'checkpoints': array, 'curves': list_of_arrays}}
    where each curve is a 1-D array of running-mean estimates at each checkpoint.
    """
    data = {}   # method -> {checkpoints, curves}

    for exp in experiments:
        for state_key in ('pre_state', 'post_state'):
            chk = _get_inc(exp, state_key, 'mc_convergence_checkpoints')
            if chk is None:
                continue
            chk = np.array(chk, dtype=float)

            for method in METHOD_STYLE:
                curve = _get_inc(exp, state_key, f'mc_convergence_{method}')
                if curve is None:
                    continue
                curve = np.array(curve, dtype=float)
                if len(curve) != len(chk):
                    continue  # length mismatch — skip

                if method not in data:
                    data[method] = {'checkpoints': chk, 'curves': []}
                data[method]['curves'].append(curve)

    return data


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_convergence(data: dict, out_path: Path, final_values: dict | None = None) -> None:
    """
    One figure: mean convergence curve ± std band per method.
    Optionally draws a dashed horizontal line at the final (N_max) mean estimate.
    """
    if not data:
        print('  [convergence] No mc_convergence_* fields found — skipping figure.')
        print('  Make sure to re-run the MATLAB engine after the instrumentation changes.')
        return

    fig, ax = plt.subplots(figsize=(6.0, 3.8))

    for method, style in METHOD_STYLE.items():
        if method not in data:
            continue
        chk    = data[method]['checkpoints']
        curves = np.array(data[method]['curves'])   # shape (n_exp, n_checkpoints)

        mean_c = np.mean(curves, axis=0)
        std_c  = np.std(curves,  axis=0)
        n_exp  = curves.shape[0]

        ax.plot(chk, mean_c,
                color=style['color'], linestyle=style['ls'], linewidth=1.8,
                label=f"{style['label']}  (n={n_exp})")
        ax.fill_between(chk,
                         mean_c - std_c, mean_c + std_c,
                         color=style['color'], alpha=0.15)

        # Mark final value with a dot
        ax.scatter([chk[-1]], [mean_c[-1]],
                   color=style['color'], s=40, zorder=5)

    ax.set_xlabel('Number of MC samples')
    ax.set_ylabel('Consistency score (running mean)')
    ax.set_title('MC Consistency Score Convergence vs. Sample Count', pad=6)
    ax.legend(loc='lower right')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.axhline(0, color='k', linewidth=0.4)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f'  Saved: {out_path.name}')


def plot_convergence_per_method(data: dict, out_dir: Path) -> None:
    """
    Per-method detailed figure: each individual experiment as a thin grey line,
    mean in bold, ±std band.  One PDF per method.
    """
    for method, style in METHOD_STYLE.items():
        if method not in data:
            continue
        chk    = data[method]['checkpoints']
        curves = np.array(data[method]['curves'])
        mean_c = np.mean(curves, axis=0)
        std_c  = np.std(curves,  axis=0)

        fig, ax = plt.subplots(figsize=(5.5, 3.5))

        # Individual trajectories (thin, transparent)
        for curve in curves:
            ax.plot(chk, curve, color='#888888', linewidth=0.4, alpha=0.3)

        # Mean ± std
        ax.fill_between(chk, mean_c - std_c, mean_c + std_c,
                        color=style['color'], alpha=0.25)
        ax.plot(chk, mean_c,
                color=style['color'], linewidth=2.0,
                label=f'Mean ± std  (n={len(curves)})')

        ax.set_xlabel('Number of MC samples')
        ax.set_ylabel('Consistency score (running mean)')
        ax.set_title(f'Convergence — {style["label"]}', pad=6)
        ax.legend(loc='lower right')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.spines[['top', 'right']].set_visible(False)

        fig.tight_layout()
        out_path = out_dir / f'figD_convergence_{method}.pdf'
        fig.savefig(out_path)
        plt.close(fig)
        print(f'  Saved: {out_path.name}')


def print_summary(data: dict) -> None:
    print()
    print('=' * 60)
    print('MC CONVERGENCE SUMMARY')
    print('=' * 60)
    for method in METHOD_STYLE:
        if method not in data:
            print(f'  {method:<10}  — no data')
            continue
        curves = np.array(data[method]['curves'])
        chk    = data[method]['checkpoints']
        final  = curves[:, -1]
        print(f'  {method:<10}  N_max={int(chk[-1]):>6}   '
              f'final mean={np.mean(final):.4f} ± {np.std(final):.4f}   '
              f'n_exp={len(curves)}')
    print('=' * 60)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description='Generate MC convergence figure (Figure D)')
    parser.add_argument('--data',   default='data/measurements_cps/measurements_cps',
                        help='Data directory with results_scenario_*.json')
    parser.add_argument('--output', default=None,
                        help='Output directory (default: results/<data_name>/convergence)')
    args = parser.parse_args()

    data_dir = args.data
    out_dir  = Path(args.output) if args.output else \
               Path('results') / Path(data_dir).name / 'convergence'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Data:   {data_dir}')
    print(f'Output: {out_dir}\n')

    experiments = _load_experiments(data_dir)
    data = collect_convergence_data(experiments)

    print_summary(data)

    # Combined figure (all methods on one axes)
    plot_convergence(data, out_dir / 'figD_convergence_all.pdf')
    plot_convergence(data, out_dir / 'figD_convergence_all.png')

    # Per-method detailed figures
    plot_convergence_per_method(data, out_dir)

    print(f'\nDone. Results in {out_dir}')


if __name__ == '__main__':
    main()
