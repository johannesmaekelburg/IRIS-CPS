#!/usr/bin/env python3
"""
generate_runtime_table.py
=========================
Reads timing fields (timing_*_s) written by the instrumented MATLAB engine
and produces the RQ1 cost-vs-accuracy table:

    method | N_samples | mean_time_ms | rho_vs_MFMC | rho_vs_reference

Run from project root:
    python src/analysis/generate_runtime_table.py --data data/measurements_cps/measurements_cps
"""

import argparse
import json
from pathlib import Path
from itertools import chain

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────

# Timing field prefix → display name
METHOD_MAP = {
    'timing_jaccard_s':          ('Jaccard (AABB)',   None),
    'timing_jaccard_mc_sobol_s': ('Jaccard-MC Sobol', None),
    'timing_jaccard_mc_random_s':('Jaccard-MC Random',None),
    'timing_mc_sobol_s':         ('MC Sobol',         'I_MF_random'),
    'timing_mc_random_s':        ('MC Random',        'I_theta'),
    'timing_mc_halton_s':        ('MC Halton',        None),
    'timing_I_theta_s':          ('I(θ)',             'I_theta'),
}

# Reference metric for correlations (Spearman ρ)
REFERENCE_FIELD = 'I_MF_random'   # MFMC-Sobol as gold standard


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_experiments(data_dir: str) -> list[dict]:
    """Flatten all experiments from all results_scenario_*.json files."""
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


def _extract(exp: dict, field: str):
    """Navigate exp.pre_state.inconsistency.<field>."""
    try:
        return exp['pre_state']['inconsistency'][field]
    except (KeyError, TypeError):
        return None


# ── Analysis ──────────────────────────────────────────────────────────────────

def build_table(experiments: list[dict]) -> list[dict]:
    """
    For each timing field found in the data, compute:
      - mean wall time (ms per experiment, both pre and post state)
      - Spearman ρ vs REFERENCE_FIELD
      - N samples (from mc_num_samples if applicable)
    """
    from scipy.stats import spearmanr

    # Collect reference values
    ref_vals = np.array([
        v for exp in experiments
        for state_key in ('pre_state', 'post_state')
        if (v := _get_state(exp, state_key, REFERENCE_FIELD)) is not None
    ], dtype=float)

    rows = []
    for timing_field, (label, score_field) in METHOD_MAP.items():
        times = []
        scores = []
        n_samples_vals = []

        for exp in experiments:
            for state_key in ('pre_state', 'post_state'):
                t = _get_state(exp, state_key, timing_field)
                if t is None:
                    continue
                times.append(float(t) * 1000)  # → ms

                if score_field:
                    s = _get_state(exp, state_key, score_field)
                    if s is not None:
                        scores.append(float(s))

                # Try to find n_samples
                ns = _get_n_samples(exp, state_key, timing_field)
                if ns is not None:
                    n_samples_vals.append(ns)

        if not times:
            continue

        mean_ms = float(np.mean(times))
        std_ms  = float(np.std(times))

        # Spearman ρ vs reference
        rho = float('nan')
        if scores and len(scores) == len(ref_vals[:len(scores)]):
            try:
                rho, _ = spearmanr(scores, ref_vals[:len(scores)])
            except Exception:
                pass

        n_samples = int(np.median(n_samples_vals)) if n_samples_vals else None

        rows.append({
            'method':     label,
            'n_samples':  n_samples,
            'mean_ms':    mean_ms,
            'std_ms':     std_ms,
            'rho_vs_ref': rho,
            'n_obs':      len(times),
        })

    rows.sort(key=lambda r: r['mean_ms'])
    return rows


def _get_state(exp: dict, state_key: str, field: str):
    try:
        return exp[state_key]['inconsistency'][field]
    except (KeyError, TypeError):
        return None


def _get_n_samples(exp: dict, state_key: str, timing_field: str):
    """Infer n_samples from the matching mc_num_samples field."""
    # timing_mc_sobol_s → mc_num_samples_sobol
    suffix = timing_field.replace('timing_', '').replace('_s', '')
    candidates = [f'mc_num_samples_{suffix}', 'mc_num_samples']
    try:
        inc = exp[state_key]['inconsistency']
        for c in candidates:
            if c in inc:
                return int(inc[c])
    except (KeyError, TypeError):
        pass
    return None


# ── Output ────────────────────────────────────────────────────────────────────

def print_table(rows: list[dict]) -> None:
    print()
    print('=' * 72)
    print('RQ1 — RUNTIME vs ACCURACY TABLE')
    print(f'Reference metric: {REFERENCE_FIELD}')
    print('=' * 72)
    header = f"{'Method':<22} {'N_samp':>7} {'Time (ms)':>12} {'±std':>8} {'ρ vs ref':>10} {'N_obs':>8}"
    print(header)
    print('-' * 72)
    for r in rows:
        ns  = str(r['n_samples']) if r['n_samples'] else '—'
        rho = f"{r['rho_vs_ref']:.3f}" if not np.isnan(r['rho_vs_ref']) else '—'
        print(f"  {r['method']:<20} {ns:>7} {r['mean_ms']:>10.2f}   {r['std_ms']:>7.2f}   {rho:>8}   {r['n_obs']:>6}")
    print('=' * 72)


def save_csv(rows: list[dict], out_path: Path) -> None:
    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write('method,n_samples,mean_ms,std_ms,rho_vs_ref,n_obs\n')
        for r in rows:
            ns  = r['n_samples'] if r['n_samples'] else ''
            rho = f"{r['rho_vs_ref']:.6f}" if not np.isnan(r['rho_vs_ref']) else ''
            fh.write(f"{r['method']},{ns},{r['mean_ms']:.4f},{r['std_ms']:.4f},{rho},{r['n_obs']}\n")
    print(f'  CSV saved: {out_path.name}')


def plot_table(rows: list[dict], out_path: Path) -> None:
    """Scatter plot: time (x) vs ρ (y), sized by N_samples."""
    valid = [r for r in rows if not np.isnan(r['rho_vs_ref'])]
    if not valid:
        print('  [plot] No rows with ρ — skipping figure.')
        return

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    sizes  = [max(40, (r['n_samples'] or 500) / 10) for r in valid]
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(valid)))

    for r, s, c in zip(valid, sizes, colors):
        ax.scatter(r['mean_ms'], r['rho_vs_ref'], s=s, color=c,
                   edgecolors='k', linewidths=0.5, zorder=3)
        ax.annotate(r['method'], (r['mean_ms'], r['rho_vs_ref']),
                    textcoords='offset points', xytext=(5, 2), fontsize=7)

    ax.set_xlabel('Mean wall time per state (ms)')
    ax.set_ylabel(f'Spearman ρ vs {REFERENCE_FIELD}')
    ax.set_title('RQ1: Cost vs. Accuracy of Consistency Metrics')
    ax.axhline(1.0, color='gray', linewidth=0.7, linestyle='--')
    ax.set_ylim(max(0, min(r['rho_vs_ref'] for r in valid) - 0.05), 1.05)
    ax.spines[['top', 'right']].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Plot saved: {out_path.name}')


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description='Generate RQ1 runtime table')
    parser.add_argument('--data',   default='data/measurements_cps/measurements_cps',
                        help='Data directory with results_scenario_*.json')
    parser.add_argument('--output', default=None,
                        help='Output directory (default: results/<data_name>/runtime_table)')
    args = parser.parse_args()

    data_dir = args.data
    out_dir  = Path(args.output) if args.output else \
               Path('results') / Path(data_dir).name / 'runtime_table'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Data:   {data_dir}')
    print(f'Output: {out_dir}\n')

    experiments = _load_experiments(data_dir)

    # Check whether timing fields are present
    sample = experiments[0] if experiments else {}
    has_timing = any(
        k.startswith('timing_')
        for k in (sample.get('pre_state', {}).get('inconsistency', {}) or {}).keys()
    )
    if not has_timing:
        print()
        print('WARNING: No timing_* fields found in the data.')
        print('  The MATLAB engine must be re-run after the instrumentation changes')
        print('  (causal_experiment_engine_twostep.m) to populate timing data.')
        print('  Re-generate measurements, then run this script again.')
        return

    rows = build_table(experiments)
    print_table(rows)
    save_csv(rows, out_dir / 'runtime_table.csv')
    plot_table(rows, out_dir / 'figC_runtime_vs_accuracy.pdf')
    plot_table(rows, out_dir / 'figC_runtime_vs_accuracy.png')
    print(f'\nDone. Results in {out_dir}')


if __name__ == '__main__':
    main()
