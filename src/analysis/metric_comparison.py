"""
Metric Comparison for Inconsistency Measures  (Saltelli / compound-intervention data)

All metrics are stored and compared as INCONSISTENCY scores (higher = more inconsistent):
  I(θ)                   — proposed global metric  (already inconsistency)
  1 - jaccard_C          — AABB Jaccard inconsistency  (jaccard_C is a consistency score)
  1 - mc_probability_*   — MC inconsistency per sampling method  (mc_probability = P(consistent))
  1 - I_MF_*             — MFMC-corrected inconsistency  (I_MF = corrected P(consistent))

Auto-detects which metrics and sampling methods are present in the data.

Entry point (called from run_analysis.py):
    run_metric_comparison_analysis(data_dir, output_dir)
"""

import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

sns.set_style("white")
plt.rcParams.update({
    "figure.dpi":        150,
    "font.size":         10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

SALTELLI_PARAMS = ["scale_factor", "center_delta", "correlation_strength"]
MC_METHODS      = ["sobol", "halton", "lhs", "random"]

# Column names used internally — all are INCONSISTENCY scores
# jaccard_C, mc_probability_*, I_MF_* are stored as (1 - raw_value) during loading
_LABELS = {
    "I_theta":          r"$I(\theta)$",
    "jaccard_inc":      r"$1 - \mathrm{Jaccard}_C$",
    **{f"mc_inc_{m}":   f"MC({m})"    for m in MC_METHODS},
    **{f"mfmc_inc_{m}": f"MFMC({m})"  for m in MC_METHODS},
}

_COLORS = {
    "I_theta":      "#e74c3c",
    "jaccard_inc":  "#3498db",
    **{f"mc_inc_{m}":   "#27ae60" for m in MC_METHODS},
    **{f"mfmc_inc_{m}": "#9b59b6" for m in MC_METHODS},
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_saltelli_data(data_dir: str,
                       pattern: str = "results_scenario_*.json") -> pd.DataFrame:
    """
    Load Saltelli/compound experiments into a flat DataFrame.
    Collects every metric column that exists in the data.
    """
    json_files = sorted(Path(data_dir).glob(pattern))
    if not json_files:
        raise FileNotFoundError(f"No files matching '{pattern}' in {data_dir}")

    def _scalar(v):
        """Extract a scalar float from a value that may be a 1-element list (MATLAB JSON)."""
        if isinstance(v, list):
            v = v[0] if v else None
        if v is None:
            return None
        return float(v)

    rows, skipped = [], 0
    for fpath in json_files:
        try:
            data = json.loads(fpath.read_text(encoding="utf-8"))
        except Exception:
            skipped += 1
            continue

        for exp in data.get("experiments", []):
            if exp.get("intervention_type") != "compound":
                continue
            inc = (exp.get("post_state") or {}).get("inconsistency", {})
            if not inc:
                continue

            row = {
                "scenario":   exp.get("scenario_type", fpath.stem),
                "exp_id":     exp.get("exp_id"),
                "sample_idx": exp.get("sample_idx"),
            }
            for p in SALTELLI_PARAMS:
                row[p] = exp.get(p, np.nan)

            # I_theta: already an inconsistency score
            s = _scalar(inc.get("I_theta"))
            if s is not None and not np.isnan(s):
                row["I_theta"] = s

            # jaccard_C is a consistency score → flip to inconsistency
            s = _scalar(inc.get("jaccard_C"))
            if s is not None and not np.isnan(s):
                row["jaccard_inc"] = 1.0 - s

            # mc_probability_* = P(consistent) → flip to inconsistency
            # I_MF_*           = corrected P(consistent) → flip to inconsistency
            for m in MC_METHODS:
                s = _scalar(inc.get(f"mc_probability_{m}"))
                if s is not None and not np.isnan(s):
                    row[f"mc_inc_{m}"] = 1.0 - s

                s = _scalar(inc.get(f"I_MF_{m}"))
                if s is not None and not np.isnan(s):
                    row[f"mfmc_inc_{m}"] = 1.0 - s

            rows.append(row)

    if not rows:
        raise ValueError(
            "No compound experiments found. "
            "Check that files contain intervention_type='compound'."
        )

    df = pd.DataFrame(rows)
    if skipped:
        warnings.warn(f"Skipped {skipped} files due to read errors.")
    print(f"Loaded {len(df):,} experiments  "
          f"({df['scenario'].nunique()} scenarios, "
          f"{len(json_files) - skipped} files)")
    return df


def discover_metrics(df: pd.DataFrame) -> dict:
    """
    Return OrderedDict {column: label} for every metric present in df.
    Ordering: I_theta, jaccard_inc, mc_inc_*, mfmc_inc_*.
    Only includes columns with at least 10 non-NaN values.
    """
    candidates = ["I_theta", "jaccard_inc"]
    for m in MC_METHODS:
        candidates.append(f"mc_inc_{m}")
    for m in MC_METHODS:
        candidates.append(f"mfmc_inc_{m}")

    return {
        col: _LABELS.get(col, col)
        for col in candidates
        if col in df.columns and df[col].notna().sum() >= 10
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. STATISTICS & CORRELATIONS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metric_statistics(df: pd.DataFrame, metrics: dict) -> pd.DataFrame:
    rows = []
    for col, label in metrics.items():
        v = df[col].dropna()
        rows.append({
            "Metric": label,
            "N":      len(v),
            "Mean":   v.mean(),
            "Std":    v.std(),
            "Min":    v.min(),
            "Median": v.median(),
            "Max":    v.max(),
        })
    return pd.DataFrame(rows)


def compute_correlations(df: pd.DataFrame, metrics: dict):
    """Return (pearson_df, spearman_df) indexed by short metric labels."""
    cols   = list(metrics.keys())
    labels = list(metrics.values())
    sub    = df[cols].dropna()

    pearson  = sub.corr(method="pearson")
    spearman = sub.corr(method="spearman")

    pearson.columns  = pearson.index  = labels
    spearman.columns = spearman.index = labels
    return pearson, spearman


# ─────────────────────────────────────────────────────────────────────────────
# 3. DISAGREEMENT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def identify_disagreements(df: pd.DataFrame,
                            threshold: float = 0.3) -> pd.DataFrame:
    """
    Find experiments where I_theta and jaccard_inc differ by more than threshold.
    Returns empty DataFrame when either column is missing.
    """
    if "I_theta" not in df.columns or "jaccard_inc" not in df.columns:
        return pd.DataFrame()

    keep = ["scenario", "exp_id", "I_theta", "jaccard_inc"] + \
           [p for p in SALTELLI_PARAMS if p in df.columns]
    sub  = df[keep].dropna(subset=["I_theta", "jaccard_inc"])

    rows = []
    for _, row in sub.iterrows():
        diff = float(row["I_theta"]) - float(row["jaccard_inc"])
        if abs(diff) < threshold:
            continue
        rows.append({
            **row.to_dict(),
            "diff": diff,
            "case": "I(θ) >> Jaccard" if diff > 0 else "Jaccard >> I(θ)",
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 4. PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path.name}")


def plot_scatter_matrix(df: pd.DataFrame, metrics: dict, out_dir: Path) -> None:
    """Lower-triangular pairwise scatter; diagonal = histogram."""
    cols   = list(metrics.keys())
    labels = list(metrics.values())
    n = len(cols)
    if n < 2:
        return

    fig, axes = plt.subplots(n, n, figsize=(3 * n, 3 * n))
    fig.suptitle("Pairwise Metric Comparison", fontsize=13, fontweight="bold", y=1.01)

    for i in range(n):
        for j in range(n):
            ax = axes[i][j]
            if i == j:
                ax.hist(df[cols[i]].dropna(), bins=50,
                        color=_COLORS.get(cols[i], "#888"),
                        alpha=0.75, edgecolor="none")
                ax.set_xlabel(labels[i], fontsize=9)
                ax.set_yticks([])
            elif i > j:
                valid = df[[cols[j], cols[i]]].dropna()
                ax.scatter(valid[cols[j]], valid[cols[i]],
                           s=2, alpha=0.25,
                           color=_COLORS.get(cols[i], "#888"),
                           rasterized=True)
                ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
                rho, _ = spearmanr(valid[cols[j]], valid[cols[i]])
                ax.set_title(f"ρ = {rho:.2f}", fontsize=8, pad=2)
                ax.set_xlabel(labels[j], fontsize=9)
                ax.set_ylabel(labels[i], fontsize=9)
                ax.set_xlim(-0.05, 1.05)
                ax.set_ylim(-0.05, 1.05)
            else:
                ax.axis("off")

    fig.tight_layout()
    _save(fig, out_dir / "metric_scatter_matrix.png")


def plot_correlation_heatmap(pearson: pd.DataFrame, spearman: pd.DataFrame,
                              out_dir: Path) -> None:
    w = max(6, 2.5 * len(pearson))
    fig, axes = plt.subplots(1, 2, figsize=(w * 2, w * 0.7))
    kw = dict(annot=True, fmt=".2f", cmap="coolwarm", center=0,
              vmin=-1, vmax=1, linewidths=0.5, square=True, annot_kws={"size": 9})
    sns.heatmap(pearson,  ax=axes[0], **kw)
    sns.heatmap(spearman, ax=axes[1], **kw)
    axes[0].set_title("Pearson $r$",  fontsize=11, fontweight="bold")
    axes[1].set_title("Spearman $ρ$", fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, out_dir / "metric_correlations.png")


def plot_distributions(df: pd.DataFrame, metrics: dict, out_dir: Path) -> None:
    """Overlaid density histogram for all metrics."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for col, label in metrics.items():
        ax.hist(df[col].dropna(), bins=60, alpha=0.4, label=label,
                color=_COLORS.get(col, "#888"), density=True, edgecolor="none")
    ax.set_xlabel("Inconsistency score", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Metric Distributions", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, out_dir / "metric_distributions.png")


def plot_disagreements(df_dis: pd.DataFrame, df: pd.DataFrame,
                       out_dir: Path) -> None:
    """I_theta vs jaccard_inc scatter coloured by scale_factor; disagreement bar."""
    if "I_theta" not in df.columns or "jaccard_inc" not in df.columns:
        return

    valid = df[["I_theta", "jaccard_inc", "scale_factor"]].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: full scatter
    ax = axes[0]
    sc = ax.scatter(valid["I_theta"], valid["jaccard_inc"],
                    c=valid["scale_factor"], cmap="viridis",
                    s=3, alpha=0.35, rasterized=True)
    plt.colorbar(sc, ax=ax, label="scale factor")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6, label="$y=x$")
    rho, _ = spearmanr(valid["I_theta"], valid["jaccard_inc"])
    ax.set_xlabel(r"$I(\theta)$", fontsize=11)
    ax.set_ylabel(r"$1 - \mathrm{Jaccard}_C$", fontsize=11)
    ax.set_title(fr"$I(\theta)$ vs $1-\mathrm{{Jaccard}}_C$  (Spearman $\rho$={rho:.2f})",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

    # Right: disagreement counts
    ax = axes[1]
    if len(df_dis) > 0:
        counts = df_dis["case"].value_counts()
        bars = ax.bar(counts.index, counts.values,
                      color=["#e74c3c", "#3498db"][:len(counts)])
        ax.bar_label(bars, fontsize=10)
        ax.set_ylabel("Number of experiments", fontsize=11)
        ax.set_title(f"Strong disagreements  (|diff| > 0.3)\n"
                     f"N = {len(df_dis):,} / {len(valid):,}  "
                     f"({100*len(df_dis)/len(valid):.1f}%)",
                     fontsize=11, fontweight="bold")
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    else:
        ax.text(0.5, 0.5, "No strong disagreements found",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_title("Strong disagreements  (|diff| > 0.3)",
                     fontsize=11, fontweight="bold")

    fig.tight_layout()
    _save(fig, out_dir / "metric_disagreements.png")


def plot_per_scenario(df: pd.DataFrame, metrics: dict, out_dir: Path) -> None:
    """Median + IQR of each metric broken down by scenario."""
    cols      = list(metrics.keys())
    labels    = list(metrics.values())
    scenarios = sorted(df["scenario"].unique())
    n_s = len(scenarios)
    if n_s < 2:
        return

    fig, axes = plt.subplots(1, len(cols),
                              figsize=(3.2 * len(cols), max(4, 0.45 * n_s)),
                              sharey=True)
    if len(cols) == 1:
        axes = [axes]

    y = list(range(n_s))
    for ax, col, label in zip(axes, cols, labels):
        medians = [df[df["scenario"] == s][col].median() for s in scenarios]
        q25     = [df[df["scenario"] == s][col].quantile(0.25) for s in scenarios]
        q75     = [df[df["scenario"] == s][col].quantile(0.75) for s in scenarios]

        ax.barh(y, medians, color=_COLORS.get(col, "#888"), alpha=0.65)
        xerr_lo = np.array(medians) - np.array(q25)
        xerr_hi = np.array(q75)     - np.array(medians)
        ax.errorbar(medians, y, xerr=[xerr_lo, xerr_hi],
                    fmt="none", color="black", capsize=3, lw=1)
        ax.set_xlabel(label, fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_yticks(y)
        ax.set_yticklabels(scenarios, fontsize=7)

    fig.suptitle("Median inconsistency per scenario  "
                 "(bar = median, whiskers = IQR)",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    _save(fig, out_dir / "metric_per_scenario.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. TEXT REPORT
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(df: pd.DataFrame, stats: pd.DataFrame,
                    pearson: pd.DataFrame, spearman: pd.DataFrame,
                    disagreements: pd.DataFrame, metrics: dict,
                    out_dir: Path) -> None:
    lines = [
        "=" * 72,
        "INCONSISTENCY METRIC COMPARISON  (Saltelli / compound data)",
        "=" * 72,
        f"Date       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Experiments: {len(df):,}  ({df['scenario'].nunique()} scenarios)",
        f"Metrics    : {', '.join(metrics.values())}",
        "",
        "─" * 72,
        "1. DESCRIPTIVE STATISTICS",
        "─" * 72,
        stats.to_string(index=False),
        "",
        "─" * 72,
        "2. SPEARMAN RANK CORRELATIONS",
        "─" * 72,
        spearman.round(3).to_string(),
    ]

    if "I_theta" in df.columns and "jaccard_inc" in df.columns:
        valid = df[["I_theta", "jaccard_inc"]].dropna()
        rho, _ = spearmanr(valid["I_theta"], valid["jaccard_inc"])
        lines += [
            "",
            "─" * 72,
            "3. KEY FINDINGS",
            "─" * 72,
            f"I(θ) range                 : "
            f"{df['I_theta'].max() - df['I_theta'].min():.3f}",
            f"1 - Jaccard_C range        : "
            f"{df['jaccard_inc'].max() - df['jaccard_inc'].min():.3f}",
            f"I(θ) vs 1-Jaccard_C ρ      : {rho:.3f}",
            f"Strong disagreements       : {len(disagreements):,} / {len(valid):,}  "
            f"({100*len(disagreements)/max(len(valid),1):.1f}%)",
        ]
        if len(disagreements) > 0:
            for case, cnt in disagreements["case"].value_counts().items():
                lines.append(f"  {case:<25}: {cnt:,}")

    lines.append("=" * 72)
    text = "\n".join(lines)
    (out_dir / "metric_comparison_report.txt").write_text(text, encoding="utf-8")
    print(text)
    print("\n  saved: metric_comparison_report.txt")


# ─────────────────────────────────────────────────────────────────────────────
# 6. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_metric_comparison_analysis(data_dir: str, output_dir: str) -> dict:
    """
    Run metric comparison on Saltelli/compound data.
    Called from run_full_analysis.py / run_analysis.py.
    Works with any subset of sampling methods present in the data.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("METRIC COMPARISON  (Saltelli)")
    print("=" * 60)

    print("\n[1/6] Loading data...")
    df = load_saltelli_data(data_dir)

    print("\n[2/6] Discovering available metrics...")
    metrics = discover_metrics(df)
    if len(metrics) < 2:
        print(f"  Only {len(metrics)} metric(s) found — skipping comparison.")
        return {"n_experiments": len(df), "metrics": list(metrics.keys()),
                "n_disagreements": 0, "output_dir": str(out_dir)}
    print(f"  Found: {', '.join(metrics.values())}")

    print("\n[3/6] Computing statistics...")
    stats = compute_metric_statistics(df, metrics)

    print("\n[4/6] Computing correlations...")
    pearson, spearman = compute_correlations(df, metrics)

    print("\n[5/6] Identifying disagreements...")
    disagreements = identify_disagreements(df)
    print(f"  Strong disagreements (|diff| > 0.3): {len(disagreements):,}")

    print("\n[6/6] Generating plots...")
    plot_scatter_matrix(df, metrics, out_dir)
    plot_correlation_heatmap(pearson, spearman, out_dir)
    plot_distributions(df, metrics, out_dir)
    plot_disagreements(disagreements, df, out_dir)
    plot_per_scenario(df, metrics, out_dir)

    print("\nGenerating report...")
    generate_report(df, stats, pearson, spearman, disagreements, metrics, out_dir)

    print(f"\nResults saved to: {out_dir.absolute()}")
    return {
        "n_experiments":    len(df),
        "metrics":          list(metrics.keys()),
        "n_disagreements":  len(disagreements),
        "output_dir":       str(out_dir),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Metric comparison for Saltelli/compound experiments"
    )
    parser.add_argument("--data",   required=True, help="data/measurements directory")
    parser.add_argument("--output", required=True, help="output directory")
    args = parser.parse_args()
    run_metric_comparison_analysis(args.data, args.output)
