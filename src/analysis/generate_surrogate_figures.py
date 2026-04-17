#!/usr/bin/env python3
"""
generate_surrogate_figures.py
==============================
Figures I–L: surrogate model evaluation (GPR on 3 Saltelli params → I_MF).

  Fig I  — Surrogate accuracy (RQ2):
            Scatter I_hat vs I_MC on held-out 20% test set.
            RMSE + Spearman ρ annotated.  One colour per scenario.

  Fig J  — Generalization (RQ2):
            Bar chart of RMSE per domain: orange = train, blue = held-out.
            Train on all-but-one scenario, test on the left-out one (LOO).

  Fig K  — Speedup (RQ3):
            Grouped bars: MC sample count vs GPR prediction cost per scenario.
            Shows relative speedup multiplier.

  Fig L  — Ranking preservation (RQ4):
            Side-by-side Sobol S_T bars (exact vs surrogate) per parameter.
            Kendall τ annotated above each parameter group.

Run from project root:
    python src/analysis/generate_surrogate_figures.py \\
        --data data/measurements
"""

import argparse
import json
import glob
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel as C, RBF, WhiteKernel,
)
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

PARAMS      = ['scale_factor', 'center_delta', 'correlation_strength']
PARAM_LABELS = {
    'scale_factor':         r'Scale $s_u$',
    'center_delta':         r'Center shift $\Delta c_u$',
    'correlation_strength': r'Correlation $R_u$',
}
_C_EXACT = '#2166ac'
_C_SURR  = '#d6604d'
_C_TRAIN = '#f4a582'
_C_TEST  = '#2166ac'

MAX_GPR_SAMPLES = 500    # GPR kernel matrix ∝ N²; keep small for speed


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
                n_mc     = _s(exp, 'post_state', 'mc_num_samples_random')
                if mfmc_raw is None:
                    continue
                row = {
                    'I':       1.0 - mfmc_raw,
                    'scenario': exp.get('scenario_type', Path(f).stem),
                    'n_mc':    int(n_mc) if n_mc is not None else 2000,
                }
                ok = True
                for p in PARAMS:
                    v = exp.get(p)
                    if v is None:
                        ok = False
                        break
                    try:
                        row[p] = float(v)
                    except (TypeError, ValueError):
                        ok = False
                        break
                if ok:
                    rows.append(row)
    print(f'Loaded {len(rows)} compound experiments from {n_files} files.')
    return rows


def _s(exp, state_key, field):
    try:
        v = exp[state_key]['inconsistency'][field]
        return v[0] if isinstance(v, list) else v
    except (KeyError, TypeError):
        return None


def to_xy(rows: list[dict]):
    X = np.array([[r[p] for p in PARAMS] for r in rows], dtype=float)
    y = np.array([r['I'] for r in rows], dtype=float)
    return X, y


# ── GPR fitting ───────────────────────────────────────────────────────────────

def fit_gpr(X_train: np.ndarray, y_train: np.ndarray) -> tuple:
    """Return (fitted_model, scaler_X). Subsamples to MAX_GPR_SAMPLES if needed."""
    if len(X_train) > MAX_GPR_SAMPLES:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_train), MAX_GPR_SAMPLES, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]
    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X_train)
    kernel = C(1.0) * RBF([1.0] * X_train.shape[1]) + WhiteKernel(noise_level=0.01)
    gpr    = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2,
                                      normalize_y=True, random_state=42)
    gpr.fit(Xs, y_train)
    return gpr, scaler


def predict(gpr, scaler, X: np.ndarray) -> np.ndarray:
    return gpr.predict(scaler.transform(X))


# ── Figure I: Surrogate accuracy ──────────────────────────────────────────────

def plot_surrogate_accuracy(rows: list[dict], out_path: Path) -> None:
    X, y = to_xy(rows)
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, np.arange(len(rows)), test_size=0.2, random_state=42)

    gpr, scaler = fit_gpr(X_tr, y_tr)
    y_hat       = predict(gpr, scaler, X_te)

    rmse = float(np.sqrt(mean_squared_error(y_te, y_hat)))
    rho, pval = spearmanr(y_te, y_hat)

    # Colour by scenario
    scenarios = [rows[i]['scenario'] for i in idx_te]
    unique_s  = sorted(set(scenarios))
    cmap      = plt.cm.get_cmap('tab20', len(unique_s))
    s_to_c    = {s: cmap(i) for i, s in enumerate(unique_s)}
    colors    = [s_to_c[s] for s in scenarios]

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.scatter(y_te, y_hat, c=colors, s=12, alpha=0.6,
               linewidths=0, rasterized=True)
    lim = [-0.03, 1.03]
    ax.plot(lim, lim, 'k--', lw=1.0, alpha=0.5, label='$y=x$')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r'True $I(\theta)$  [MFMC]')
    ax.set_ylabel(r'Surrogate $\hat{I}(\theta)$  [GPR]')

    p_str = 'p<0.001' if pval < 0.001 else f'p={pval:.3f}'
    ax.text(0.04, 0.96,
            f'RMSE = {rmse:.4f}\n'
            fr'Spearman $\rho$ = {rho:.3f}  ({p_str})',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='grey', alpha=0.8))

    ax.set_title(f'Surrogate Accuracy — GPR on held-out 20%  (N={len(y_te)})',
                 pad=6)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)

    fig.tight_layout()
    _save(fig, out_path)
    print(f'  RMSE={rmse:.4f}  ρ={rho:.3f}')


# ── Figure J: Generalization (LOO per scenario) ───────────────────────────────

def plot_generalization(rows: list[dict], out_path: Path) -> None:
    from collections import defaultdict
    by_scen = defaultdict(list)
    for r in rows:
        by_scen[r['scenario']].append(r)

    scenarios = sorted(by_scen.keys())
    if len(scenarios) < 2:
        print('[Fig J] Need ≥2 scenarios for LOO. Skipping.')
        return

    train_rmse, test_rmse = [], []
    valid_scens = []

    for held_out in scenarios:
        train_rows = [r for r in rows if r['scenario'] != held_out]
        test_rows  = by_scen[held_out]
        if len(train_rows) < 5 or len(test_rows) < 2:
            continue
        X_tr, y_tr = to_xy(train_rows)
        X_te, y_te = to_xy(test_rows)

        gpr, scaler = fit_gpr(X_tr, y_tr)
        y_hat_tr    = predict(gpr, scaler, X_tr)
        y_hat_te    = predict(gpr, scaler, X_te)

        train_rmse.append(float(np.sqrt(mean_squared_error(y_tr, y_hat_tr))))
        test_rmse.append(float(np.sqrt(mean_squared_error(y_te, y_hat_te))))
        valid_scens.append(held_out.replace('scenario_', 'S'))

    if not valid_scens:
        print('[Fig J] No valid LOO folds. Skipping.')
        return

    x     = np.arange(len(valid_scens))
    w     = 0.35
    order = np.argsort(test_rmse)[::-1]

    fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(valid_scens)), 4.2))
    ax.bar(x - w/2, [train_rmse[i] for i in order], w,
           color=_C_TRAIN, alpha=0.8, label='Train RMSE')
    ax.bar(x + w/2, [test_rmse[i]  for i in order], w,
           color=_C_TEST,  alpha=0.8, label='Held-out RMSE')

    ax.set_xticks(x)
    ax.set_xticklabels([valid_scens[i] for i in order],
                       rotation=55, ha='right', fontsize=7)
    ax.set_ylabel('RMSE')
    ax.set_title('Surrogate Generalization — Leave-One-Scenario-Out\n'
                 '(sorted by held-out RMSE)', pad=5)
    ax.legend(frameon=False)
    ax.spines[['top', 'right']].set_visible(False)

    fig.tight_layout()
    _save(fig, out_path)


# ── Figure K: Speedup ─────────────────────────────────────────────────────────

def plot_speedup(rows: list[dict], out_path: Path) -> None:
    from collections import defaultdict

    # Measure GPR predict time on all data
    X, y = to_xy(rows)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    gpr, scaler = fit_gpr(X_tr, y_tr)

    # Time per surrogate prediction (microseconds)
    N_bench = 500
    X_bench = X[:N_bench] if len(X) >= N_bench else X
    t0 = time.perf_counter()
    predict(gpr, scaler, X_bench)
    t_surr_per_sample = (time.perf_counter() - t0) / len(X_bench) * 1e6  # µs

    # MC cost proxy: n_mc samples needed per evaluation
    # Relative speedup = n_mc / (n_surr) where n_surr is effectively 1
    by_scen = defaultdict(list)
    for r in rows:
        by_scen[r['scenario']].append(r['n_mc'])

    scenarios  = sorted(by_scen.keys())
    mean_n_mc  = [np.mean(by_scen[s]) for s in scenarios]

    # Speedup = MC_cost_proxy / surrogate_cost_proxy
    # MC cost: N_mc evaluations of containment check
    # Surrogate cost: 1 GPR prediction ≈ t_surr_per_sample µs
    # Use N_mc as dimensionless speedup (surrogate avoids all MC samples)
    speedups = mean_n_mc  # speedup factor = N_mc avoided

    x = np.arange(len(scenarios))
    order = np.argsort(speedups)[::-1]
    s_labels = [scenarios[i].replace('scenario_', 'S') for i in order]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    # Left: N_MC per scenario
    ax = axes[0]
    ax.bar(x, [mean_n_mc[i] for i in order],
           color=_C_EXACT, alpha=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels(s_labels, rotation=55, ha='right', fontsize=7)
    ax.set_ylabel('Mean MC samples per evaluation')
    ax.set_title('MC Cost Proxy (N samples avoided\nby surrogate per query)', pad=5)
    ax.spines[['top', 'right']].set_visible(False)

    # Right: Speedup bar
    ax = axes[1]
    bars = ax.bar(x, [speedups[i] for i in order],
                  color=_C_SURR, alpha=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels(s_labels, rotation=55, ha='right', fontsize=7)
    ax.set_ylabel('Speedup factor  (N_MC / N_surrogate)')
    ax.set_title(f'Surrogate Speedup per Scenario\n'
                 f'(GPR predict ≈ {t_surr_per_sample:.1f} µs/sample)', pad=5)
    ax.spines[['top', 'right']].set_visible(False)

    fig.tight_layout()
    _save(fig, out_path)
    print(f'  GPR predict time: {t_surr_per_sample:.2f} µs/sample')
    print(f'  Mean speedup: {np.mean(speedups):.0f}×')


# ── Figure L: Ranking preservation ───────────────────────────────────────────

def plot_ranking_preservation(rows: list[dict], out_path: Path) -> None:
    """
    Sobol total-effect S_T: exact (from data variance) vs surrogate.
    Uses Morris-style one-at-a-time sensitivity as a proxy since we don't
    have the full Saltelli design here — computes Pearson eta² per param
    from raw data (exact) and from surrogate predictions on same X.
    """
    X, y_exact = to_xy(rows)
    X_tr, _, y_tr, _ = train_test_split(X, y_exact, test_size=0.2, random_state=42)
    gpr, scaler  = fit_gpr(X_tr, y_tr)
    y_surr = predict(gpr, scaler, X)

    def eta2(X_col: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
        """η² via binned ANOVA-like variance decomposition."""
        edges = np.percentile(X_col, np.linspace(0, 100, n_bins + 1))
        edges = np.unique(edges)
        if len(edges) < 3:
            return np.nan
        grand_mean = np.mean(y)
        ss_between = 0.0
        ss_total   = np.sum((y - grand_mean) ** 2)
        for i in range(len(edges) - 1):
            mask = (X_col >= edges[i]) & (X_col < edges[i + 1])
            if mask.sum() < 2:
                continue
            bin_mean    = np.mean(y[mask])
            ss_between += mask.sum() * (bin_mean - grand_mean) ** 2
        return ss_between / (ss_total + 1e-12)

    exact_eta2 = [eta2(X[:, i], y_exact) for i in range(len(PARAMS))]
    surr_eta2  = [eta2(X[:, i], y_surr)  for i in range(len(PARAMS))]

    # Kendall τ on the ranking
    tau, p_tau = kendalltau(
        np.argsort(np.argsort(exact_eta2)),
        np.argsort(np.argsort(surr_eta2))
    )

    labels = [PARAM_LABELS[p] for p in PARAMS]
    x      = np.arange(len(PARAMS))
    w      = 0.35

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    bars1 = ax.bar(x - w/2, exact_eta2, w, color=_C_EXACT, alpha=0.8,
                   label=r'Exact  ($\eta^2$ from data)')
    bars2 = ax.bar(x + w/2, surr_eta2,  w, color=_C_SURR,  alpha=0.8,
                   label=r'Surrogate  ($\eta^2$ from GPR)')

    # Annotate values
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                f'{h:.3f}', ha='center', va='bottom', fontsize=7)

    p_str = 'p<0.001' if p_tau < 0.001 else f'p={p_tau:.3f}'
    ax.text(0.97, 0.97,
            fr'Kendall $\tau$ = {tau:.3f}  ({p_str})',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='grey', alpha=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(r'Effect size $\eta^2$  (proxy for $S_T$)')
    ax.set_title('Ranking Preservation: Exact vs Surrogate\n'
                 r'($\eta^2$ as $S_T$ proxy, Kendall $\tau$ on ranking)', pad=5)
    ax.legend(frameon=False)
    ax.spines[['top', 'right']].set_visible(False)

    fig.tight_layout()
    _save(fig, out_path)
    print(f'  Kendall τ = {tau:.3f}')


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save(fig, out_path: Path) -> None:
    for ext in ('pdf', 'png'):
        p = out_path.with_suffix(f'.{ext}')
        fig.savefig(p, dpi=300, bbox_inches='tight')
        print(f'  Saved: {p.name}')
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate surrogate figures I-L')
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

    print(f'\n[Fig I] Surrogate accuracy...')
    plot_surrogate_accuracy(rows, out_dir / 'figI_surrogate_accuracy')

    print(f'\n[Fig J] Generalization...')
    plot_generalization(rows, out_dir / 'figJ_generalization')

    print(f'\n[Fig K] Speedup...')
    plot_speedup(rows, out_dir / 'figK_speedup')

    print(f'\n[Fig L] Ranking preservation...')
    plot_ranking_preservation(rows, out_dir / 'figL_ranking_preservation')

    print(f'\nDone. Figures saved to {out_dir}')


if __name__ == '__main__':
    main()
