#!/usr/bin/env python3
"""
Anticausal Inference / Bayesian Inversion over θ → I(θ)

Given dataset D = {(θᵢ, I(θᵢ))} with θ = [s_u, Δc_u, R_u], answers:
  "Given high inconsistency I(θ), which uncertainty parameters are most responsible?"

Implements:
  1. Binning of D by inconsistency level into k bins
  2. Empirical conditional distributions P(θ | I(θ) ∈ bin) per dimension
  3. Visualizations: conditional histograms, violin plots, heatmaps, ECDFs
  4. Diagnostics: conditional means/variances, η² effect sizes, KS tests,
     mutual information, and a ranking of which θ is most diagnostic

Usage:
    python anticausal_inference.py --data <csv_or_json> [--k 5] [--out <dir>]
    python anticausal_inference.py --demo
"""

import argparse
import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

# ── plotting style (consistent with project) ────────────────────────────────
sns.set_style("white")
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

PARAM_LABELS = {
    "s_u":        r"Scale $s_u$",
    "delta_c_u":  r"Center shift $\Delta c_u$",
    "R_u":        r"Correlation $R_u$",
    # also accept the Saltelli CSV column names
    "scale_factor":           r"Scale $s_u$",
    "center_delta":           r"Center shift $\Delta c_u$",
    "correlation_strength":   r"Correlation $R_u$",
}

INCONSISTENCY_COL = "inconsistency"   # expected column name

CMAP_BINS   = "RdYlGn_r"   # low I → green, high I → red
CLR_NEUTRAL = "#4878CF"


# ════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_dataset(path: str) -> pd.DataFrame:
    """Load CSV or JSON into a DataFrame with canonical column names."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")

    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    elif p.suffix.lower() == ".json":
        raw = json.loads(p.read_text(encoding="utf-8"))
        df = pd.DataFrame(raw) if isinstance(raw, list) else pd.json_normalize(raw)
    else:
        raise ValueError(f"Unsupported format: {p.suffix}")

    df = _canonicalise_columns(df)
    _validate(df)
    return df


def load_saltelli_results(results_dir: str, pattern: str = "results_scenario_*.json",
                          scenario_filter: Optional[list] = None) -> pd.DataFrame:
    """
    Load compound-intervention Saltelli results from data/measurements/measurements/.

    Each experiment has scale_factor, center_delta, correlation_strength at the
    top level and I_theta inside post_state.inconsistency.I_theta.

    Parameters
    ----------
    results_dir     : directory containing results_scenario_*.json files
    pattern         : glob pattern for result files
    scenario_filter : optional list of scenario names to include (None = all)
    """
    import glob as _glob
    rows = []

    json_files = sorted(_glob.glob(str(Path(results_dir) / pattern)))
    if not json_files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' in {results_dir}"
        )

    skipped = 0
    for fpath in json_files:
        try:
            d = json.loads(Path(fpath).read_text(encoding="utf-8"))
        except PermissionError:
            skipped += 1
            continue

        experiments = d.get("experiments", [])
        for e in experiments:
            if e.get("intervention_type") != "compound":
                continue

            scenario = e.get("scenario_type", Path(fpath).stem)
            if scenario_filter and scenario not in scenario_filter:
                continue

            _v = e.get("post_state", {}).get("inconsistency", {}).get("I_MF_random")
            if _v is None or (isinstance(_v, float) and np.isnan(_v)):
                continue
            I = 1.0 - float(_v)

            rows.append({
                "s_u":             float(e["scale_factor"]),
                "delta_c_u":       float(e["center_delta"]),
                "R_u":             float(e["correlation_strength"]),
                INCONSISTENCY_COL: I,
                "scenario":        scenario,
                "sample_idx":      e.get("sample_idx"),
            })

    if not rows:
        raise ValueError(
            "No compound experiments found. Check that the files contain "
            "intervention_type='compound' and post_state.inconsistency.I_theta."
        )

    df = pd.DataFrame(rows)
    if skipped:
        warnings.warn(f"Skipped {skipped} files due to permission errors.")
    print(f"Loaded {len(df)} compound experiments from {len(json_files) - skipped} files.")
    print(f"  Scenarios: {sorted(df['scenario'].unique())}")
    print(f"  I(theta) range: [{df[INCONSISTENCY_COL].min():.3f}, "
          f"{df[INCONSISTENCY_COL].max():.3f}]")
    return df


def load_mfmc_results(results_dir: str, pattern: str = "results_scenario_*.json",
                      normalise: bool = True) -> pd.DataFrame:
    """
    Build a (s_u, delta_c_u, R_u, inconsistency) DataFrame from mfmc JSON exports.

    Each experiment records one post-intervention state.  The three θ dimensions
    are extracted as:
      s_u       = post_state.uncertainty.source_radius   (normalised by pre-state value)
      delta_c_u = ||post_center - pre_center||_2         (L2 shift magnitude, normalised)
      R_u       = post_state.uncertainty.source_correlation
      inconsistency = post_state.inconsistency.I_theta

    WARNING — design limitation
    ---------------------------
    The mfmc data is one-at-a-time (OAT): each experiment varies one parameter
    while the others are fixed at their baseline.  The joint distribution
    P(s_u, delta_c_u, R_u) is therefore highly structured, NOT space-filling.
    Anticausal estimates from this dataset can be confounded by intervention type.
    Use Saltelli-sampled data (Option B) for unconfounded estimates.

    Parameters
    ----------
    normalise : if True, divide s_u and delta_c_u by their per-scenario
                pre-state values so they become dimensionless scale factors.
    """
    import glob as _glob
    rows = []

    json_files = sorted(_glob.glob(str(Path(results_dir) / pattern)))
    if not json_files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' in {results_dir}"
        )

    for fpath in json_files:
        d = json.loads(Path(fpath).read_text(encoding="utf-8"))
        experiments = d.get("experiments", [])
        if not experiments:
            continue

        # pre-state reference (first experiment, pre side)
        pre0 = experiments[0]["pre_state"]["uncertainty"]
        pre_radius = float(pre0["source_radius"])
        pre_center = np.array(pre0["source_center"], dtype=float)

        for e in experiments:
            u    = e["post_state"]["uncertainty"]
            inc  = e["post_state"]["inconsistency"]
            _v   = inc.get("I_MF_random")
            if _v is None or (isinstance(_v, float) and np.isnan(_v)):
                continue
            I = 1.0 - float(_v)

            post_radius = float(u["source_radius"])
            post_center = np.array(u["source_center"], dtype=float)
            post_corr   = float(u["source_correlation"])

            delta_c = float(np.linalg.norm(post_center - pre_center))

            if normalise and pre_radius > 0:
                s_u       = post_radius / pre_radius
                delta_c_u = delta_c / pre_radius   # same unit as radius
            else:
                s_u       = post_radius
                delta_c_u = delta_c

            rows.append({
                "s_u":           s_u,
                "delta_c_u":     delta_c_u,
                "R_u":           post_corr,
                INCONSISTENCY_COL: float(I),
                "scenario":      d.get("scenario_name", Path(fpath).stem),
                "intervention":  e.get("intervention_type", "unknown"),
            })

    if not rows:
        raise ValueError("No valid experiments found in mfmc results.")

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} experiments from {len(json_files)} scenario files.")
    print(f"  Intervention types: {sorted(df['intervention'].unique())}")
    print(f"  I(theta) range: [{df[INCONSISTENCY_COL].min():.3f}, "
          f"{df[INCONSISTENCY_COL].max():.3f}]")
    return df


def _canonicalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Accept both naming conventions; rename to canonical s_u / delta_c_u / R_u."""
    rename = {
        "scale_factor":         "s_u",
        "center_delta":         "delta_c_u",
        "correlation_strength": "R_u",
        "I_theta":              INCONSISTENCY_COL,
        "I_theta_mean":         INCONSISTENCY_COL,
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    return df


def _validate(df: pd.DataFrame) -> None:
    required = {"s_u", "delta_c_u", "R_u", INCONSISTENCY_COL}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"DataFrame is missing columns: {missing}.\n"
            f"Available: {list(df.columns)}"
        )
    n_before = len(df)
    df.dropna(subset=list(required), inplace=True)
    if len(df) < n_before:
        warnings.warn(f"Dropped {n_before - len(df)} rows with NaN values.")


PARAM_COLS = ["s_u", "delta_c_u", "R_u"]


# ════════════════════════════════════════════════════════════════════════════
# 2. BINNING
# ════════════════════════════════════════════════════════════════════════════

def bin_by_inconsistency(
    df: pd.DataFrame,
    k: int = 5,
    strategy: str = "quantile",   # "quantile" | "uniform" | "fixed"
    fixed_edges: Optional[list] = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Add a 'bin' column (integer 0..k-1) and return readable bin labels.

    strategy='quantile'  → equal-count bins
    strategy='uniform'   → equal-width bins over the observed I range
    strategy='fixed'     → explicit bin edges via fixed_edges parameter
                           e.g. fixed_edges=[0, 0.5, 0.8, 0.95, 1.0]
                           Last edge is made inclusive automatically.
    """
    I = df[INCONSISTENCY_COL].values
    df = df.copy()

    if strategy == "fixed":
        if fixed_edges is None or len(fixed_edges) < 2:
            raise ValueError("strategy='fixed' requires fixed_edges with at least 2 values.")
        edges = np.array(sorted(fixed_edges), dtype=float)
        edges[-1] += 1e-12   # make last edge inclusive
    elif strategy == "quantile":
        quantiles = np.linspace(0, 100, k + 1)
        edges = np.percentile(I, quantiles)
        edges[-1] += 1e-12
    elif strategy == "uniform":
        edges = np.linspace(I.min(), I.max() + 1e-12, k + 1)
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Use 'quantile', 'uniform', or 'fixed'.")

    k_edges = len(edges) - 1
    raw_bins = np.digitize(I, edges[1:]).clip(0, k_edges - 1)

    # Compact: remove empty bins and re-index contiguously
    occupied = sorted(set(raw_bins))
    non_empty = len(occupied)
    if non_empty < k_edges:
        warnings.warn(
            f"Only {non_empty} of {k_edges} bins are non-empty. "
            f"Compacting to {non_empty} bins."
        )
    remap = {old: new for new, old in enumerate(occupied)}
    df["bin"] = np.array([remap[b] for b in raw_bins])

    labels = [
        f"[{edges[occupied[i]]:.2f}, {edges[occupied[i]+1]:.2f})"
        for i in range(non_empty)
    ]
    return df, labels


# ════════════════════════════════════════════════════════════════════════════
# 3. CONDITIONAL STATISTICS  P(θ | bin)
# ════════════════════════════════════════════════════════════════════════════

def conditional_stats(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    Return a DataFrame with per-(bin, param) conditional mean and std,
    plus the bin centre (mean inconsistency) and bin size.
    """
    rows = []
    for b in range(k):
        sub = df[df["bin"] == b]
        bin_mean_I = sub[INCONSISTENCY_COL].mean()
        bin_size   = len(sub)
        for col in PARAM_COLS:
            rows.append({
                "bin":       b,
                "param":     col,
                "cond_mean": sub[col].mean(),
                "cond_std":  sub[col].std(ddof=1),
                "bin_mean_I": bin_mean_I,
                "bin_size":  bin_size,
            })
    return pd.DataFrame(rows)


def effect_sizes(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    Compute η² (eta-squared) for each parameter:
        η²_j = SS_between / SS_total
    Measures the fraction of variance in θ_j explained by the bin (inconsistency level).
    Higher η² → θ_j is more diagnostic.

    Also reports:
    - Spearman ρ between θ_j and I(θ)   (monotone association)
    - KS statistic: max separation between highest and lowest bin CDFs
    - Mutual information (binned)
    """
    from sklearn.feature_selection import mutual_info_regression

    records = []
    I = df[INCONSISTENCY_COL].values

    for col in PARAM_COLS:
        x = df[col].values

        # η² via one-way ANOVA decomposition
        grand_mean = x.mean()
        ss_total   = np.sum((x - grand_mean) ** 2)
        ss_between = sum(
            len(g) * (g[col].mean() - grand_mean) ** 2
            for _, g in df.groupby("bin")[[col]]
        )
        eta2 = ss_between / ss_total if ss_total > 0 else 0.0

        # Spearman rank correlation with I(θ)
        rho, p_rho = stats.spearmanr(x, I)

        # KS: top bin vs bottom bin
        bot = df[df["bin"] == 0][col].values
        top = df[df["bin"] == df["bin"].max()][col].values
        ks_stat, ks_p = stats.ks_2samp(bot, top)

        # Mutual information (non-parametric)
        mi = mutual_info_regression(x.reshape(-1, 1), I, random_state=42)[0]

        records.append({
            "param":    col,
            "eta2":     eta2,
            "spearman_rho": rho,
            "spearman_p":   p_rho,
            "ks_stat":  ks_stat,
            "ks_p":     ks_p,
            "MI":       mi,
        })

    return pd.DataFrame(records).sort_values("eta2", ascending=False)


# ════════════════════════════════════════════════════════════════════════════
# 4. VISUALISATIONS
# ════════════════════════════════════════════════════════════════════════════

def _bin_palette(k: int) -> list:
    cmap = plt.get_cmap(CMAP_BINS)
    if k == 1:
        return [cmap(0.5)]
    return [cmap(i / (k - 1)) for i in range(k)]


# ── 4a. Conditional histograms (one row per param, one column per bin) ──────

def plot_conditional_histograms(
    df: pd.DataFrame, labels: list[str], out_dir: Path
) -> None:
    k = len(labels)
    palette = _bin_palette(k)
    fig, axes = plt.subplots(len(PARAM_COLS), k,
                             figsize=(2.5 * k, 2.5 * len(PARAM_COLS)),
                             sharey="row", sharex="row")

    for r, col in enumerate(PARAM_COLS):
        x_all  = df[col].values
        x_range = (x_all.min(), x_all.max())

        for b in range(k):
            ax  = axes[r, b]
            sub = df[df["bin"] == b][col].values
            if len(sub) == 0:
                ax.text(0.5, 0.5, "empty", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="grey")
            else:
                ax.hist(sub, bins=20, range=x_range, color=palette[b],
                        edgecolor="white", linewidth=0.4, density=True)
                ax.axvline(sub.mean(), color="k", lw=1.2, ls="--", label="mean")
                ax.axvline(x_all.mean(), color="grey", lw=0.8, ls=":", label="global")

            if r == 0:
                ax.set_title(f"Bin {b}\n{labels[b]}", fontsize=8)
            if b == 0:
                ax.set_ylabel(PARAM_LABELS.get(col, col), fontsize=9)
            ax.tick_params(labelsize=7)

    fig.suptitle(r"Conditional distributions $P(\theta_j \mid I(\theta) \in \mathrm{bin})$",
                 y=1.01, fontsize=11)
    fig.tight_layout()
    _save(fig, out_dir / "cond_histograms.pdf")


# ── 4b. Violin plots ─────────────────────────────────────────────────────────

def plot_violin(df: pd.DataFrame, labels: list[str], out_dir: Path) -> None:
    k = len(labels)
    palette = _bin_palette(k)
    fig, axes = plt.subplots(1, len(PARAM_COLS),
                             figsize=(4 * len(PARAM_COLS), 4.5))

    for ax, col in zip(axes, PARAM_COLS):
        non_empty = [(b, df[df["bin"] == b][col].values)
                     for b in range(k) if len(df[df["bin"] == b]) > 1]
        if non_empty:
            positions, data_per_bin = zip(*non_empty)
            parts = ax.violinplot(list(data_per_bin), positions=list(positions),
                                  showmedians=True, showextrema=True)
            for i, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(palette[positions[i]])
                pc.set_alpha(0.8)
            parts["cmedians"].set_color("black")
            parts["cbars"].set_color("black")
            parts["cmaxes"].set_color("black")
            parts["cmins"].set_color("black")

        # overlay global mean
        ax.axhline(df[col].mean(), color="grey", lw=1, ls="--", label="global mean")

        ax.set_xticks(range(k))
        ax.set_xticklabels([f"B{b}" for b in range(k)], fontsize=8)
        ax.set_xlabel("Inconsistency bin", fontsize=9)
        ax.set_title(PARAM_LABELS.get(col, col), fontsize=10)
        ax.tick_params(labelsize=8)

    fig.suptitle(r"$P(\theta_j \mid I(\theta) \in \mathrm{bin})$ — violin plots",
                 fontsize=11)
    fig.tight_layout()
    _save(fig, out_dir / "cond_violins.pdf")


# ── 4c. Conditional mean ± std traces ────────────────────────────────────────

def plot_cond_mean_traces(
    cstats: pd.DataFrame, labels: list[str], out_dir: Path
) -> None:
    k = len(labels)
    palette = _bin_palette(k)
    x_bins = np.arange(k)

    fig, axes = plt.subplots(1, len(PARAM_COLS),
                             figsize=(4 * len(PARAM_COLS), 3.5))

    for ax, col in zip(axes, PARAM_COLS):
        sub = cstats[cstats["param"] == col].sort_values("bin")
        means = sub["cond_mean"].values
        stds  = sub["cond_std"].values

        ax.plot(x_bins, means, "o-", color=CLR_NEUTRAL, lw=1.8, zorder=3)
        ax.fill_between(x_bins, means - stds, means + stds,
                        color=CLR_NEUTRAL, alpha=0.25, label="±1 std")

        ax.set_xticks(x_bins)
        ax.set_xticklabels([f"B{b}" for b in range(k)], fontsize=8)
        ax.set_xlabel("Inconsistency bin", fontsize=9)
        ax.set_title(PARAM_LABELS.get(col, col), fontsize=10)
        ax.tick_params(labelsize=8)

    fig.suptitle(r"Conditional mean $\pm$ std of $\theta_j$ per inconsistency bin",
                 fontsize=11)
    fig.tight_layout()
    _save(fig, out_dir / "cond_mean_traces.pdf")


# ── 4d. Heatmap: conditional mean (normalised) ──────────────────────────────

def plot_cond_mean_heatmap(
    cstats: pd.DataFrame, labels: list[str], out_dir: Path
) -> None:
    k = len(labels)
    matrix = np.full((len(PARAM_COLS), k), np.nan)

    for r, col in enumerate(PARAM_COLS):
        sub = cstats[cstats["param"] == col].sort_values("bin")["cond_mean"].values
        valid = ~np.isnan(sub)
        if valid.sum() < 2:
            continue
        col_min, col_max = sub[valid].min(), sub[valid].max()
        rng = col_max - col_min
        if rng < 1e-12:               # no variation across bins
            matrix[r, valid] = 0.5
        else:
            matrix[r] = np.where(valid, (sub - col_min) / rng, np.nan)

    masked = np.ma.masked_invalid(matrix)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="#eeeeee")     # grey for empty/NaN cells

    fig, ax = plt.subplots(figsize=(max(5, 1.4 * k), 2.8))
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Normalised conditional mean", shrink=0.85)

    ax.set_xticks(range(k))
    ax.set_xticklabels([f"B{b}\n{labels[b]}" for b in range(k)], fontsize=7)
    ax.set_yticks(range(len(PARAM_COLS)))
    ax.set_yticklabels([PARAM_LABELS.get(c, c) for c in PARAM_COLS], fontsize=9)
    ax.set_xlabel("Inconsistency bin", fontsize=9)
    ax.set_title("Normalised conditional mean — anticausal view", fontsize=10)

    for r in range(len(PARAM_COLS)):
        for c in range(k):
            val = matrix[r, c]
            label_txt = f"{val:.2f}" if not np.isnan(val) else "—"
            ax.text(c, r, label_txt, ha="center", va="center",
                    fontsize=7, color="black")

    fig.tight_layout()
    _save(fig, out_dir / "cond_mean_heatmap.pdf")


# ── 4e. Effect-size ranking bar chart ────────────────────────────────────────

def plot_effect_sizes(es: pd.DataFrame, out_dir: Path) -> None:
    metrics = [
        ("eta2",         r"$\eta^2$ (ANOVA)",         "#4878CF"),
        ("spearman_rho", r"Spearman $|\rho|$",          "#E8A838"),
        ("ks_stat",      r"KS statistic",                "#D62728"),
        ("MI",           r"Mutual information",          "#2CA02C"),
    ]
    n_met = len(metrics)
    fig, axes = plt.subplots(1, n_met, figsize=(3.2 * n_met, 3.5))

    for ax, (col, title, color) in zip(axes, metrics):
        vals = es[col].abs().values        # abs for signed Spearman
        labels = [PARAM_LABELS.get(p, p) for p in es["param"].values]
        y_pos  = np.arange(len(labels))
        ax.barh(y_pos, vals, color=color, edgecolor="white")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Value", fontsize=8)
        ax.tick_params(labelsize=8)
        # mark top bar
        ax.barh(np.argmax(vals), vals.max(), color=color,
                edgecolor="black", linewidth=1.2)

    fig.suptitle("Anticausal diagnostics — which θ is most responsible for I(θ)?",
                 fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir / "effect_sizes.pdf")


# ── 4f. 2-D heatmap: joint P(θⱼ, θₖ | top bin) vs. P(θⱼ, θₖ | bottom bin) ─

def plot_joint_heatmaps(df: pd.DataFrame, out_dir: Path,
                        max_points: int = 2000) -> None:
    from itertools import combinations
    pairs = list(combinations(PARAM_COLS, 2))
    k_max = df["bin"].max()
    bot   = df[df["bin"] == 0]
    top   = df[df["bin"] == k_max]

    if len(bot) == 0 or len(top) == 0:
        print("  skipped joint_heatmaps (empty top or bottom bin)")
        return

    # subsample for KDE speed
    rng = np.random.default_rng(42)
    def _sub(d):
        if len(d) > max_points:
            return d.iloc[rng.choice(len(d), max_points, replace=False)]
        return d
    bot_s, top_s = _sub(bot), _sub(top)

    n_pairs = len(pairs)
    fig, axes = plt.subplots(n_pairs, 2, figsize=(7, 3.0 * n_pairs))
    if n_pairs == 1:
        axes = axes[np.newaxis, :]

    for row, (cx, cy) in enumerate(pairs):
        for col_idx, (subset, title, cmap) in enumerate([
            (bot_s, "Low I(θ) — bottom bin", "Blues"),
            (top_s, "High I(θ) — top bin",   "Reds"),
        ]):
            ax = axes[row, col_idx]
            sns.kdeplot(
                data=subset, x=cx, y=cy,
                fill=True, cmap=cmap, ax=ax,
                levels=8, thresh=0.05,
            )
            ax.scatter(subset[cx], subset[cy], s=5, alpha=0.3,
                       color="black", zorder=2)
            ax.set_xlabel(PARAM_LABELS.get(cx, cx), fontsize=8)
            ax.set_ylabel(PARAM_LABELS.get(cy, cy), fontsize=8)
            ax.set_title(title, fontsize=8)
            ax.tick_params(labelsize=7)

    fig.suptitle(r"Joint $P(\theta_j, \theta_k)$ conditioned on inconsistency level",
                 fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir / "joint_heatmaps.pdf")


# ── 4g. ECDF per bin ─────────────────────────────────────────────────────────

def plot_ecdfs(df: pd.DataFrame, labels: list[str], out_dir: Path) -> None:
    k = len(labels)
    palette = _bin_palette(k)
    fig, axes = plt.subplots(1, len(PARAM_COLS),
                             figsize=(4 * len(PARAM_COLS), 3.5))

    for ax, col in zip(axes, PARAM_COLS):
        for b in range(k):
            x = np.sort(df[df["bin"] == b][col].values)
            y = np.arange(1, len(x) + 1) / len(x)
            ax.step(x, y, where="post", color=palette[b],
                    lw=1.5, label=f"B{b}")

        ax.set_xlabel(PARAM_LABELS.get(col, col), fontsize=9)
        ax.set_ylabel("ECDF", fontsize=9)
        ax.set_title(PARAM_LABELS.get(col, col), fontsize=10)
        ax.legend(fontsize=7, title="bin")
        ax.tick_params(labelsize=8)

    fig.suptitle("Empirical CDFs of θ conditioned on inconsistency bin", fontsize=11)
    fig.tight_layout()
    _save(fig, out_dir / "cond_ecdfs.pdf")


# ── helper ───────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    # also save PNG for quick preview
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved → {path.name}")


# ════════════════════════════════════════════════════════════════════════════
# 5. TEXTUAL REPORT
# ════════════════════════════════════════════════════════════════════════════

def print_report(df: pd.DataFrame, es: pd.DataFrame,
                 cstats: pd.DataFrame, labels: list[str]) -> None:
    k = len(labels)
    print("\n" + "=" * 68)
    print("  ANTICAUSAL INFERENCE REPORT")
    print("=" * 68)
    print(f"  Dataset:  {len(df)} samples,  k={k} bins")
    print(f"  I(θ) range:  [{df[INCONSISTENCY_COL].min():.4f}, "
          f"{df[INCONSISTENCY_COL].max():.4f}]")
    print()

    print("  ── Bin sizes ──────────────────────────────────────────────")
    for b in range(k):
        n_b = (df["bin"] == b).sum()
        print(f"  Bin {b}  {labels[b]}:  {n_b} samples")

    print()
    print("  ── Effect-size ranking (η²) ───────────────────────────────")
    print(f"  {'Rank':<5} {'Parameter':<25} {'η²':>6}  {'|ρ|':>6}  {'KS':>6}  {'MI':>6}")
    for rank, row in es.reset_index(drop=True).iterrows():
        print(f"  {rank+1:<5} {PARAM_LABELS.get(row['param'], row['param']):<25} "
              f"{row['eta2']:>6.3f}  {abs(row['spearman_rho']):>6.3f}  "
              f"{row['ks_stat']:>6.3f}  {row['MI']:>6.3f}")

    print()
    print("  ── Conditional means per bin ──────────────────────────────")
    header = f"  {'Param':<22}"
    for b in range(k):
        header += f"  B{b}"
    print(header)
    for col in PARAM_COLS:
        sub    = cstats[cstats["param"] == col].sort_values("bin")
        line   = f"  {PARAM_LABELS.get(col, col):<22}"
        for _, r in sub.iterrows():
            line += f"  {r['cond_mean']:>5.3f}"
        print(line)

    print()
    winner = es.iloc[0]
    print(f"  >> Most diagnostic parameter: "
          f"{PARAM_LABELS.get(winner['param'], winner['param'])}  "
          f"(η² = {winner['eta2']:.3f})")
    print("=" * 68 + "\n")


# ════════════════════════════════════════════════════════════════════════════
# 6. EXPORT
# ════════════════════════════════════════════════════════════════════════════

def export_results(df: pd.DataFrame, es: pd.DataFrame,
                   cstats: pd.DataFrame, labels: list[str],
                   out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    es.to_csv(out_dir / "effect_sizes.csv", index=False)
    cstats.to_csv(out_dir / "conditional_stats.csv", index=False)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_samples": len(df),
        "k_bins": len(labels),
        "bin_labels": labels,
        "effect_sizes": es.to_dict(orient="records"),
        "conditional_stats": cstats.to_dict(orient="records"),
        "most_diagnostic_param": es.iloc[0]["param"],
        "most_diagnostic_eta2":  es.iloc[0]["eta2"],
    }
    (out_dir / "anticausal_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"  saved → anticausal_summary.json")


# ════════════════════════════════════════════════════════════════════════════
# 7. MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════════════

def run_anticausal_inference(
    df: pd.DataFrame,
    k: int = 5,
    strategy: str = "quantile",
    fixed_edges: Optional[list] = None,
    out_dir: Optional[str] = None,
) -> dict:
    """
    Full anticausal inference pipeline.

    Parameters
    ----------
    df          : DataFrame with columns s_u / delta_c_u / R_u / inconsistency
    k           : number of bins (ignored when strategy='fixed')
    strategy    : 'quantile' | 'uniform' | 'fixed'
    fixed_edges : bin boundaries for strategy='fixed', e.g. [0, 0.5, 0.8, 0.95, 1.0]
    out_dir     : directory for figures and CSV/JSON; None → skip saving

    Returns
    -------
    dict with keys: df_binned, cond_stats, effect_sizes, bin_labels
    """
    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
    else:
        out_path = None

    # ── step 1: bin ──────────────────────────────────────────────────────────
    df_binned, labels = bin_by_inconsistency(
        df, k=k, strategy=strategy, fixed_edges=fixed_edges
    )
    k_actual = len(labels)

    # ── step 2: statistics ───────────────────────────────────────────────────
    cstats = conditional_stats(df_binned, k_actual)
    es     = effect_sizes(df_binned, k_actual)

    # ── step 3: report ───────────────────────────────────────────────────────
    print_report(df_binned, es, cstats, labels)

    # ── step 4: visualise ────────────────────────────────────────────────────
    if out_path is not None:
        print("\nGenerating figures...")
        plot_conditional_histograms(df_binned, labels, out_path)
        plot_violin(df_binned, labels, out_path)
        plot_cond_mean_traces(cstats, labels, out_path)
        plot_cond_mean_heatmap(cstats, labels, out_path)
        plot_effect_sizes(es, out_path)
        plot_joint_heatmaps(df_binned, out_path)
        plot_ecdfs(df_binned, labels, out_path)
        export_results(df_binned, es, cstats, labels, out_path)

    return {
        "df_binned":    df_binned,
        "cond_stats":   cstats,
        "effect_sizes": es,
        "bin_labels":   labels,
    }


# ════════════════════════════════════════════════════════════════════════════
# 8. DEMO / CLI
# ════════════════════════════════════════════════════════════════════════════

def _make_demo_df(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic dataset that mimics realistic behaviour:
    - high s_u  → pushes the system toward inconsistency
    - large |R_u| has a moderate effect
    - delta_c_u is weakly diagnostic
    """
    rng = np.random.default_rng(seed)

    s_u        = rng.uniform(0.5, 5.0, n)
    delta_c_u  = rng.uniform(0.0, 0.4, n)
    R_u        = rng.uniform(0.0, 0.95, n)

    # synthetic I(θ) ∈ [0,1]
    logit = (
        -3.5
        + 1.2 * np.log(s_u)         # strong positive effect of scale
        + 0.5 * delta_c_u           # weak positive effect of center shift
        + 0.8 * R_u                 # moderate positive effect of correlation
        + 0.3 * rng.normal(0, 1, n) # noise
    )
    I = 1.0 / (1.0 + np.exp(-logit))

    return pd.DataFrame({
        "s_u":         s_u,
        "delta_c_u":   delta_c_u,
        "R_u":         R_u,
        INCONSISTENCY_COL: I,
    })


def main():
    parser = argparse.ArgumentParser(
        description="Anticausal inference: P(θ | I(θ) ∈ bin)"
    )
    parser.add_argument("--data",     type=str,  default=None,
                        help="Path to CSV or JSON dataset")
    parser.add_argument("--saltelli",  type=str,  default=None,
                        help="Path to directory of compound Saltelli results (e.g. data/measurements/measurements)")
    parser.add_argument("--mfmc",     type=str,  default=None,
                        help="Path to directory of mfmc OAT results_scenario_*.json files")
    parser.add_argument("--k",        type=int,  default=5,
                        help="Number of bins for quantile/uniform strategy (default: 5)")
    parser.add_argument("--strategy", type=str,  default="quantile",
                        choices=["quantile", "uniform", "fixed"],
                        help="Binning strategy (default: quantile)")
    parser.add_argument("--bins",     type=str,  default=None,
                        help="Explicit bin edges for --strategy fixed, "
                             "comma-separated, e.g. '0,0.5,0.8,0.95,1.0'")
    parser.add_argument("--out",      type=str,  default=None,
                        help="Output directory (default: figures/anticausal_<timestamp>)")
    parser.add_argument("--demo",     action="store_true",
                        help="Run on synthetic demo data")
    args = parser.parse_args()

    # resolve data
    if args.demo:
        print("Running on synthetic demo data (n=2000).")
        df = _make_demo_df()
    elif args.saltelli is not None:
        print(f"Loading Saltelli compound data from: {args.saltelli}")
        df = load_saltelli_results(args.saltelli)
    elif args.mfmc is not None:
        print(f"Loading mfmc OAT data from: {args.mfmc}")
        print("NOTE: mfmc data is one-at-a-time (OAT) — results may be confounded.")
        df = load_mfmc_results(args.mfmc)
    elif args.data is not None:
        df = load_dataset(args.data)
    else:
        print("No data source specified. Running demo. Use --data, --saltelli, --mfmc, or --demo.")
        df = _make_demo_df()

    # resolve output directory
    if args.out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = str(Path("figures") / f"anticausal_{ts}")
    else:
        out_dir = args.out

    fixed_edges = None
    if args.bins is not None:
        fixed_edges = [float(x) for x in args.bins.split(",")]
        args.strategy = "fixed"

    run_anticausal_inference(
        df, k=args.k, strategy=args.strategy,
        fixed_edges=fixed_edges, out_dir=out_dir
    )


if __name__ == "__main__":
    main()
