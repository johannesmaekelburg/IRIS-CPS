#!/usr/bin/env python3
"""
Likelihood-Weighted Inconsistency Estimation
=============================================

Computes  I_LW = Σ wᵢ · I(θᵢ)  where  wᵢ ∝ p(θᵢ) / q(θᵢ)  via importance
sampling on any existing dataset D = {(θᵢ, I(θᵢ))}.

The target distribution p is pluggable — use the built-in TruncatedNormalTarget
(fitted from data) or supply your own callable for MFMC-derived or any other
distribution.  q is assumed uniform over the empirical bounding box of D.

Quick start
-----------
    from likelihood_weighted_inconsistency import (
        load_saltelli_dataset, TruncatedNormalTarget, estimate, plot_lw
    )
    df      = load_saltelli_dataset("data/measurements")
    target  = TruncatedNormalTarget(df)          # fit p from data
    results = estimate(df, target)
    fig     = plot_lw(results, out_dir="figures/lw")

Custom target (e.g. MFMC-derived)
----------------------------------
    from likelihood_weighted_inconsistency import CallableTarget, estimate
    mfmc_target = CallableTarget(lambda df: my_mfmc_pdf(df[["s_u","delta_c_u","R_u"]].values))
    results_mfmc = estimate(df, mfmc_target, label="MFMC prior")

CLI
---
    python likelihood_weighted_inconsistency.py --data ../data/measurements
    python likelihood_weighted_inconsistency.py --demo
"""

from __future__ import annotations

import argparse
import json
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
from scipy.stats import truncnorm

# ── plotting style ────────────────────────────────────────────────────────────
sns.set_style("white")
plt.rcParams.update({
    "figure.dpi":        150,
    "font.size":         10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ── canonical names ───────────────────────────────────────────────────────────
PARAM_COLS = ["s_u", "delta_c_u", "R_u"]
INC_COL    = "inconsistency"

PARAM_LABELS = {
    "s_u":       r"Scale $s_u$",
    "delta_c_u": r"Center shift $\Delta c_u$",
    "R_u":       r"Correlation $R_u$",
}

# Physical bounds — hard constraints for the truncation
PARAM_BOUNDS = {
    "s_u":       (0.0,      np.inf),  # must be positive
    "delta_c_u": (-np.inf,  np.inf),  # unconstrained
    "R_u":       (0.0,      1.0),     # correlation ∈ [0, 1]
}


# ═════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_saltelli_dataset(results_dir: str,
                          pattern: str = "results_scenario_*.json") -> pd.DataFrame:
    """
    Load compound-intervention experiments from data/measurements/.

    Column mapping
    ──────────────
      scale_factor        → s_u
      center_delta        → delta_c_u
      correlation_strength→ R_u
      post_state.inconsistency.I_theta → inconsistency
    """
    import glob as _glob

    files = sorted(_glob.glob(str(Path(results_dir) / pattern)))
    if not files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' in {results_dir}"
        )

    rows, skipped = [], 0
    for fpath in files:
        try:
            d = json.loads(Path(fpath).read_text(encoding="utf-8"))
        except PermissionError:
            skipped += 1
            continue

        for e in d.get("experiments", []):
            if e.get("intervention_type") != "compound":
                continue
            I = (e.get("post_state") or {}).get("inconsistency", {}).get("I_theta")
            if I is None or (isinstance(I, float) and np.isnan(I)):
                continue
            try:
                rows.append({
                    "s_u":       float(e["scale_factor"]),
                    "delta_c_u": float(e["center_delta"]),
                    "R_u":       float(e["correlation_strength"]),
                    INC_COL:     float(I),
                    "scenario":  e.get("scenario_type", Path(fpath).stem),
                })
            except (KeyError, TypeError, ValueError):
                skipped += 1

    if not rows:
        raise ValueError(
            "No valid compound experiments found. "
            "Check that files contain intervention_type='compound' and "
            "post_state.inconsistency.I_theta."
        )

    df = pd.DataFrame(rows)
    if skipped:
        warnings.warn(f"Skipped {skipped} records due to errors/missing fields.")
    print(f"Loaded {len(df)} experiments from {len(files) - skipped} files.")
    for col in PARAM_COLS:
        print(f"  {PARAM_LABELS[col]:<28} in [{df[col].min():.4f}, {df[col].max():.4f}]"
              f"  (mu={df[col].mean():.4f}, sd={df[col].std():.4f})")
    print(f"  {INC_COL:<28} in [{df[INC_COL].min():.4f}, {df[INC_COL].max():.4f}]"
          f"  (mu={df[INC_COL].mean():.4f})")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# 2.  TARGET DISTRIBUTION  p(θ)   — pluggable interface
# ═════════════════════════════════════════════════════════════════════════════

class TargetDistribution(ABC):
    """Base class — implement pdf() to plug in any target distribution."""

    label: str = "p(θ)"

    @abstractmethod
    def pdf(self, df: pd.DataFrame) -> np.ndarray:
        """Return p(θᵢ) for every row of df.  Shape: (N,)."""

    def log_pdf(self, df: pd.DataFrame) -> np.ndarray:
        """Default log-pdf via pdf(); override for numerical stability."""
        return np.log(self.pdf(df) + 1e-300)


class TruncatedNormalTarget(TargetDistribution):
    """
    Product of marginal truncated normals.

        p(θ) = ∏ⱼ  TN(θⱼ ; μⱼ, σⱼ, aⱼ, bⱼ)

    μ, σ estimated from data (or supplied explicitly).
    (aⱼ, bⱼ) are the physical bounds from PARAM_BOUNDS.

    Parameters
    ----------
    df    : DataFrame used to estimate μ, σ.  Pass None if supplying mu/sigma.
    mu    : dict {col: value} — override fitted means.
    sigma : dict {col: value} — override fitted standard deviations.
    label : human-readable name shown in plots.
    """

    def __init__(self,
                 df:    Optional[pd.DataFrame] = None,
                 mu:    Optional[dict]         = None,
                 sigma: Optional[dict]         = None,
                 label: str                    = "Truncated Normal") -> None:
        self.label = label
        self._params: dict[str, dict] = {}

        for col in PARAM_COLS:
            # Estimate or accept mean/std
            if df is not None:
                x = df[col].dropna().values
                mu_j    = float(x.mean())    if (mu    is None or col not in mu)    else float(mu[col])
                sigma_j = float(x.std(ddof=1)) if (sigma is None or col not in sigma) else float(sigma[col])
            else:
                if mu is None or col not in mu:
                    raise ValueError(f"Must provide df or mu['{col}'].")
                mu_j    = float(mu[col])
                sigma_j = float(sigma[col]) if (sigma and col in sigma) else 1.0

            sigma_j = max(sigma_j, 1e-8)

            lo, hi = PARAM_BOUNDS[col]
            a = (lo - mu_j) / sigma_j if np.isfinite(lo) else -np.inf
            b = (hi - mu_j) / sigma_j if np.isfinite(hi) else  np.inf

            self._params[col] = {
                "mu": mu_j, "sigma": sigma_j,
                "a": a, "b": b, "lo": lo, "hi": hi,
                "dist": truncnorm(a=a, b=b, loc=mu_j, scale=sigma_j),
            }

    def pdf(self, df: pd.DataFrame) -> np.ndarray:
        log_p = np.zeros(len(df))
        for col in PARAM_COLS:
            log_p += self._params[col]["dist"].logpdf(df[col].values)
        return np.exp(log_p)

    def marginal(self, col: str) -> truncnorm:
        """Return the fitted scipy truncnorm object for one parameter."""
        return self._params[col]["dist"]

    def summary(self) -> None:
        print(f"\nTarget distribution: {self.label}")
        print(f"  {'param':<24}  {'mu':>8}  {'sigma':>8}  bounds")
        print("  " + "-" * 54)
        for col in PARAM_COLS:
            p  = self._params[col]
            lo = f"{p['lo']:.3g}" if np.isfinite(p['lo']) else "-inf"
            hi = f"{p['hi']:.3g}" if np.isfinite(p['hi']) else "+inf"
            print(f"  {PARAM_LABELS[col]:<24}  {p['mu']:8.4f}  {p['sigma']:8.4f}"
                  f"  [{lo}, {hi}]")
        print()


class CallableTarget(TargetDistribution):
    """
    Wrap any user-supplied PDF function as a target distribution.

    Example (MFMC-derived density)
    ──────────────────────────────
        def my_mfmc_pdf(theta: np.ndarray) -> np.ndarray:
            # theta shape (N, 3) with columns [s_u, delta_c_u, R_u]
            ...
            return density_values   # shape (N,)

        target = CallableTarget(my_mfmc_pdf, label="MFMC prior")
        results = estimate(df, target)

    The callable receives a (N, 3) numpy array ordered [s_u, delta_c_u, R_u]
    and must return a (N,) array of non-negative density values.
    """

    def __init__(self, fn: Callable[[np.ndarray], np.ndarray],
                 label: str = "Custom p(θ)") -> None:
        self.label = label
        self._fn   = fn

    def pdf(self, df: pd.DataFrame) -> np.ndarray:
        theta = df[PARAM_COLS].values  # shape (N, 3)
        vals  = self._fn(theta)
        return np.asarray(vals, dtype=float)


# ═════════════════════════════════════════════════════════════════════════════
# 3.  IMPORTANCE SAMPLING
# ═════════════════════════════════════════════════════════════════════════════

def importance_weights(df: pd.DataFrame,
                       target: TargetDistribution) -> np.ndarray:
    """
    Normalised IS weights  wᵢ = p(θᵢ) / q(θᵢ)  /  Σⱼ p(θⱼ)/q(θⱼ).

    q is assumed uniform over the empirical bounding box, so q(θ) = 1/V is
    constant and cancels after normalisation.  Returns weights summing to 1.
    """
    p_vals = target.pdf(df)

    if np.all(p_vals == 0):
        warnings.warn(
            f"All p(θ) = 0 under '{target.label}'. "
            "The target may be too narrow relative to the data range. "
            "Returning uniform weights."
        )
        return np.full(len(df), 1.0 / len(df))

    p_vals = np.clip(p_vals, 0, None)   # guard against numerical negatives
    total  = p_vals.sum()
    return p_vals / total


def effective_sample_size(w: np.ndarray) -> float:
    """ESS = 1 / Σ wᵢ²  (equals N when all weights are equal)."""
    return float(1.0 / np.sum(w ** 2))


def _bootstrap_ci(I_arr: np.ndarray, w: np.ndarray,
                  n_boot: int, ci: float, seed: int) -> tuple[float, float]:
    """Weighted bootstrap CI via stratified resampling."""
    rng      = np.random.default_rng(seed)
    N        = len(I_arr)
    indices  = rng.choice(N, size=(n_boot, N), replace=True, p=w)
    boot_lw  = I_arr[indices].mean(axis=1)   # unweighted mean of weighted resample
    alpha    = (1.0 - ci) / 2
    return float(np.quantile(boot_lw, alpha)), float(np.quantile(boot_lw, 1 - alpha))


# ═════════════════════════════════════════════════════════════════════════════
# 4.  MAIN ESTIMATOR
# ═════════════════════════════════════════════════════════════════════════════

def estimate(df:      pd.DataFrame,
             target:  TargetDistribution,
             n_boot:  int  = 4000,
             ci:      float = 0.95,
             verbose: bool  = True) -> dict:
    """
    Compute likelihood-weighted inconsistency estimate.

    Parameters
    ----------
    df      : DataFrame with columns [s_u, delta_c_u, R_u, inconsistency]
    target  : any TargetDistribution (TruncatedNormalTarget, CallableTarget, …)
    n_boot  : bootstrap replicates for the CI
    ci      : confidence level (default 0.95)
    verbose : print summary table

    Returns
    -------
    dict with keys:
        w          – normalised IS weights, shape (N,)
        I_LW       – likelihood-weighted estimate  Σ wᵢ I(θᵢ)
        I_uniform  – unweighted mean  (1/N) Σ I(θᵢ)
        I_LW_ci    – (lower, upper) bootstrap CI for I_LW
        ess        – effective sample size
        ess_pct    – ESS as % of N
        target     – the TargetDistribution used
        df_aug     – df with added columns: weight, weight_ratio, log2_wr
    """
    I_arr     = df[INC_COL].values
    w         = importance_weights(df, target)
    I_LW      = float(np.dot(w, I_arr))
    I_uniform = float(I_arr.mean())
    ess       = effective_sample_size(w)
    ci_lo, ci_hi = _bootstrap_ci(I_arr, w, n_boot=n_boot, ci=ci, seed=42)

    df_aug = df.copy()
    df_aug["weight"]       = w
    df_aug["weight_ratio"] = w * len(df)          # wᵢ N  — 1 = neutral
    df_aug["log2_wr"]      = np.log2(np.clip(df_aug["weight_ratio"], 1e-9, None))

    if verbose:
        N = len(df)
        if isinstance(target, TruncatedNormalTarget):
            target.summary()
        print(f"Likelihood-weighted estimate  [{target.label}]")
        print("  " + "-" * 44)
        print(f"  N                  : {N}")
        print(f"  ESS                : {ess:.1f}  ({100*ess/N:.1f}% of N)")
        print(f"  I_uniform          : {I_uniform:.4f}")
        print(f"  I_LW               : {I_LW:.4f}")
        print(f"  delta (I_LW-I_unif): {I_LW - I_uniform:+.4f}")
        print(f"  {int(ci*100)}% CI (bootstrap)  : [{ci_lo:.4f}, {ci_hi:.4f}]")
        print()

    return {
        "w":        w,
        "I_LW":     I_LW,
        "I_uniform": I_uniform,
        "I_LW_ci":  (ci_lo, ci_hi),
        "ess":       ess,
        "ess_pct":  100.0 * ess / len(df),
        "target":   target,
        "df_aug":   df_aug,
    }


def compare_targets(df: pd.DataFrame,
                    targets: list[TargetDistribution],
                    **estimate_kwargs) -> pd.DataFrame:
    """
    Run estimate() for multiple targets and return a summary DataFrame.

    Useful for sweeping over different p distributions without re-loading data.

    Example
    -------
        targets = [
            TruncatedNormalTarget(df, label="Fitted TN"),
            TruncatedNormalTarget(df, sigma={"s_u": 0.1}, label="Narrow s_u"),
            CallableTarget(my_mfmc_pdf, label="MFMC prior"),
        ]
        summary = compare_targets(df, targets)
        print(summary)
    """
    rows = []
    for t in targets:
        r = estimate(df, t, verbose=False, **estimate_kwargs)
        rows.append({
            "label":     t.label,
            "I_LW":      r["I_LW"],
            "I_uniform": r["I_uniform"],
            "delta":     r["I_LW"] - r["I_uniform"],
            "CI_lo":     r["I_LW_ci"][0],
            "CI_hi":     r["I_LW_ci"][1],
            "ESS":       r["ess"],
            "ESS_%":     r["ess_pct"],
        })
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# 5.  VISUALISATION
# ═════════════════════════════════════════════════════════════════════════════

# Colour palette consistent with the rest of the project
_C_BLUE   = "#4878CF"
_C_ORANGE = "#E07B39"
_C_RED    = "#D65F5F"
_C_GREY   = "#BBBBBB"


def plot_lw(results:  dict,
            out_dir:  Optional[Path | str] = None,
            fname:    str = "likelihood_weighted_inconsistency") -> plt.Figure:
    """
    Three-row paper figure.

    Row A – Marginal distributions: observed data histogram (grey = q)
            with fitted target p(θ) overlaid (orange).
    Row B – Scatter I(θ) vs θⱼ, coloured by log₂(wᵢN):
            blue = downweighted, red = upweighted, white = neutral.
    Row C – Left: bar chart comparing I_uniform vs I_LW with bootstrap CI.
            Right: histogram of weight ratios wᵢN.
    """
    df_aug  = results["df_aug"]
    target  = results["target"]
    w       = results["w"]
    I_LW    = results["I_LW"]
    I_unif  = results["I_uniform"]
    ci_lo, ci_hi = results["I_LW_ci"]
    N       = len(df_aug)
    ess     = results["ess"]

    log_wr = df_aug["log2_wr"].values
    vmax   = max(float(np.abs(np.quantile(log_wr, [0.01, 0.99])).max()), 1.0)
    norm   = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    # ── layout ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 10))
    gs  = gridspec.GridSpec(
        3, 3, figure=fig,
        hspace=0.48, wspace=0.36,
        height_ratios=[1.0, 1.0, 1.15],
    )
    ax_marg = [fig.add_subplot(gs[0, j]) for j in range(3)]
    ax_scat = [fig.add_subplot(gs[1, j]) for j in range(3)]
    ax_bar  = fig.add_subplot(gs[2, :2])
    ax_wt   = fig.add_subplot(gs[2, 2])

    for ax, lbl in zip(ax_marg + ax_scat + [ax_bar, ax_wt],
                       "ABCDEFGH"):
        ax.text(-0.13, 1.07, lbl, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")

    # ════════════════════════════════════════════════════════════════════════
    # Row A — marginal distributions
    # ════════════════════════════════════════════════════════════════════════
    for j, (col, ax) in enumerate(zip(PARAM_COLS, ax_marg)):
        x = df_aug[col].values
        lo_b, hi_b = PARAM_BOUNDS[col]
        x_lo = x.min() - 0.05 * (x.max() - x.min()) if not np.isfinite(lo_b) else max(lo_b, x.min() - 0.05 * (x.max() - x.min()))
        x_hi = x.max() + 0.05 * (x.max() - x.min()) if not np.isfinite(hi_b) else min(hi_b, x.max() + 0.05 * (x.max() - x.min()))

        ax.hist(x, bins=30, density=True,
                color=_C_GREY, edgecolor="white", linewidth=0.3, zorder=1,
                label=r"Data ($q$, uniform)")

        # Target density (works for any TargetDistribution that has a
        # TruncatedNormalTarget marginal; for other targets we skip the overlay)
        if isinstance(target, TruncatedNormalTarget):
            x_line = np.linspace(x_lo, x_hi, 400)
            p_line = target.marginal(col).pdf(x_line)
            ax.plot(x_line, p_line, color=_C_ORANGE, lw=2.0, zorder=3,
                    label=r"Target $p(\theta)$")
            ax.fill_between(x_line, 0, p_line,
                            color=_C_ORANGE, alpha=0.12, zorder=2)
            ax.axvline(target._params[col]["mu"],
                       color=_C_ORANGE, lw=1.0, ls="--", zorder=4)
        ax.axvline(x.mean(), color=_C_GREY, lw=1.0, ls=":", zorder=4)

        ax.set_xlabel(PARAM_LABELS[col], fontsize=9)
        if j == 0:
            ax.set_ylabel("Density", fontsize=9)
            ax.legend(fontsize=7.5, frameon=False, loc="upper right")
        ax.tick_params(labelsize=8)

    ax_marg[1].set_title(
        r"Marginal distributions: data $q(\theta)$ vs target $p(\theta)$",
        fontsize=9, pad=5,
    )

    # ════════════════════════════════════════════════════════════════════════
    # Row B — scatter I(θ) vs θⱼ, coloured by log₂(wᵢN)
    # ════════════════════════════════════════════════════════════════════════
    I_vals = df_aug[INC_COL].values
    sc = None
    for j, (col, ax) in enumerate(zip(PARAM_COLS, ax_scat)):
        sc = ax.scatter(
            df_aug[col], I_vals,
            c=log_wr, cmap="RdBu_r", norm=norm,
            s=12, alpha=0.65, linewidths=0, zorder=2,
        )
        ax.set_xlabel(PARAM_LABELS[col], fontsize=9)
        if j == 0:
            ax.set_ylabel(r"$I(\theta)$", fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.tick_params(labelsize=8)

    cbar = fig.colorbar(sc, ax=ax_scat, fraction=0.015, pad=0.03,
                        shrink=0.85, aspect=18)
    cbar.set_label(r"$\log_2(w_i N)$" + "\n← down  ·  up →", fontsize=8)
    cbar.ax.tick_params(labelsize=7.5)

    ax_scat[1].set_title(
        r"$I(\theta)$ coloured by weight ratio $w_i N$",
        fontsize=9, pad=5,
    )

    # ════════════════════════════════════════════════════════════════════════
    # Row C-left — estimation comparison bar chart
    # ════════════════════════════════════════════════════════════════════════
    labels_b = [r"Uniform mean  $\bar{I}$",
                fr"LW estimate  $\hat{{I}}_{{LW}}$" + f"\n[{target.label}]"]
    vals_b   = [I_unif, I_LW]
    bars = ax_bar.bar(labels_b, vals_b,
                      color=[_C_BLUE, _C_ORANGE],
                      width=0.40, edgecolor="white", linewidth=0.5, zorder=3)

    ax_bar.errorbar(
        x=1, y=I_LW,
        yerr=[[I_LW - ci_lo], [ci_hi - I_LW]],
        fmt="none", color="black", capsize=5, lw=1.5, zorder=4,
    )

    for bar, val in zip(bars, vals_b):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    val + 0.004, f"{val:.4f}",
                    ha="center", va="bottom", fontsize=9)

    delta = I_LW - I_unif
    mid_y = (I_LW + I_unif) / 2
    ax_bar.annotate(
        "", xy=(1.0, I_LW), xytext=(1.0, I_unif),
        arrowprops=dict(arrowstyle="<->", color=_C_RED, lw=1.5),
    )
    ax_bar.text(1.22, mid_y, f"Δ = {delta:+.4f}",
                va="center", color=_C_RED, fontsize=9.5)

    ax_bar.text(0.02, 0.96,
                f"ESS = {ess:.0f}  ({ess/N*100:.1f}% of N = {N})",
                transform=ax_bar.transAxes,
                va="top", fontsize=8.5, color="#555555")

    y_lo = max(0.0, min(I_unif, I_LW) * 0.90)
    y_hi = max(I_unif, I_LW) * 1.10 + 0.005
    ax_bar.set_ylim(y_lo, y_hi)
    ax_bar.set_ylabel(r"Inconsistency $I(\theta)$", fontsize=9)
    ax_bar.tick_params(axis="x", labelsize=9)
    ax_bar.tick_params(axis="y", labelsize=8)
    ax_bar.set_title("Estimation comparison", fontsize=9, pad=5)
    ax_bar.grid(axis="y", lw=0.4, alpha=0.5, zorder=1)
    ax_bar.set_axisbelow(True)

    # ════════════════════════════════════════════════════════════════════════
    # Row C-right — weight-ratio histogram
    # ════════════════════════════════════════════════════════════════════════
    wr      = df_aug["weight_ratio"].values
    up_mask = wr >= 1.0
    bins_h  = np.linspace(0.0, min(wr.max() * 1.02, 6.0), 50)

    ax_wt.axvline(1.0, color="#888888", lw=1.0, ls="--", zorder=1,
                  label="Neutral (wᵢN = 1)")
    ax_wt.hist(wr[~up_mask], bins=bins_h, color=_C_BLUE,  alpha=0.70,
               edgecolor="white", linewidth=0.3, label="Downweighted", zorder=2)
    ax_wt.hist(wr[up_mask],  bins=bins_h, color=_C_RED,   alpha=0.70,
               edgecolor="white", linewidth=0.3, label="Upweighted",   zorder=2)

    ax_wt.set_xlabel(r"Weight ratio $w_i N$", fontsize=9)
    ax_wt.set_ylabel("Count", fontsize=9)
    ax_wt.set_title("Weight distribution", fontsize=9, pad=5)
    ax_wt.legend(fontsize=8, frameon=False)
    ax_wt.tick_params(labelsize=8)
    ax_wt.text(0.97, 0.97,
               f"{up_mask.sum()/N*100:.0f}% up\n{(~up_mask).sum()/N*100:.0f}% down",
               transform=ax_wt.transAxes,
               ha="right", va="top", fontsize=8.5, color="#333333")

    # ── global title ──────────────────────────────────────────────────────────
    fig.suptitle(
        "Likelihood-Weighted Inconsistency Estimation  —  "
        r"$\hat{I}_{LW} = \sum_i w_i\,I(\theta_i)$,  "
        r"$w_i \propto p(\theta_i)/q(\theta_i)$",
        fontsize=10.5, y=1.003,
    )

    # ── save ─────────────────────────────────────────────────────────────────
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(out_dir / f"{fname}.{ext}",
                        bbox_inches="tight", dpi=300)
        print(f"Saved: {out_dir / fname}.{{pdf,png}}")

    return fig


def plot_compare_targets(df: pd.DataFrame,
                         targets: list[TargetDistribution],
                         out_dir: Optional[Path | str] = None,
                         fname:   str = "lw_target_comparison",
                         **estimate_kwargs) -> plt.Figure:
    """
    Forest-plot style comparison of I_LW across multiple target distributions.

    Runs estimate() for each target, then plots a dot + CI for each,
    together with the unweighted baseline I_uniform.
    """
    summary = compare_targets(df, targets, **estimate_kwargs)
    I_unif  = float(df[INC_COL].mean())

    fig, ax = plt.subplots(figsize=(7, 1.0 + 0.55 * len(targets)))

    y_pos = np.arange(len(summary))
    ax.axvline(I_unif, color=_C_BLUE, lw=1.2, ls="--",
               label=f"Uniform mean ({I_unif:.4f})", zorder=1)

    for i, row in summary.iterrows():
        err_lo = row["I_LW"] - row["CI_lo"]
        err_hi = row["CI_hi"] - row["I_LW"]
        ax.errorbar(row["I_LW"], y_pos[i],
                    xerr=[[err_lo], [err_hi]],
                    fmt="o", color=_C_ORANGE,
                    capsize=4, lw=1.5, ms=6, zorder=3)
        ax.text(row["CI_hi"] + 0.002, y_pos[i],
                f"Δ={row['delta']:+.4f}  ESS={row['ESS']:.0f}({row['ESS_%']:.0f}%)",
                va="center", fontsize=8, color="#333333")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(summary["label"].tolist(), fontsize=9)
    ax.set_xlabel(r"Inconsistency estimate $I$", fontsize=9)
    ax.set_title("LW estimates across target distributions", fontsize=10, pad=6)
    ax.legend(fontsize=8.5, frameon=False)
    ax.tick_params(labelsize=8)
    ax.grid(axis="x", lw=0.4, alpha=0.5)
    fig.tight_layout()

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(out_dir / f"{fname}.{ext}",
                        bbox_inches="tight", dpi=300)
        print(f"Saved: {out_dir / fname}.{{pdf,png}}")

    return fig


# ═════════════════════════════════════════════════════════════════════════════
# 6.  LANDSCAPE ANALYSIS  (Option 3)
# ═════════════════════════════════════════════════════════════════════════════

def landscape_analysis(df: pd.DataFrame) -> dict:
    """
    Inspect where I(theta) is low and which parameters drive it.

    Returns a dict with keys:
        spearman   – DataFrame with rho, p-value per parameter
        thresholds – dict {threshold: sub-DataFrame} for I < 0.5, 0.7, 0.9
        dominant   – name of the parameter with the strongest |rho|
    """
    from scipy import stats as _stats

    # Spearman correlations
    rows = []
    for col in PARAM_COLS:
        rho, p = _stats.spearmanr(df[col], df[INC_COL])
        rows.append({"param": col, "label": PARAM_LABELS[col],
                     "rho": rho, "p_value": p})
    spearman = pd.DataFrame(rows).sort_values("rho", key=abs, ascending=False)

    # Low-I clusters
    thresholds = {}
    for thresh in [0.5, 0.7, 0.9]:
        sub = df[df[INC_COL] < thresh]
        thresholds[thresh] = sub

    dominant = spearman.iloc[0]["param"]

    print("=== Landscape analysis ===")
    print(f"\n  Spearman rho with I(theta):")
    for _, r in spearman.iterrows():
        bar = "#" * int(abs(r["rho"]) * 30)
        print(f"    {r['param']:<14} rho={r['rho']:+.4f}  {bar}")

    print(f"\n  Low-I clusters (mean of each param vs full dataset mean):")
    header = f"  {'':14}  {'full':>10}" + "".join(f"  {'I<'+str(t):>8}" for t in [0.5, 0.7, 0.9])
    print(header)
    print("  " + "-" * (len(header) - 2))
    for col in PARAM_COLS:
        full_mu = df[col].mean()
        line = f"  {col:<14}  {full_mu:>10.4f}"
        for thresh in [0.5, 0.7, 0.9]:
            sub = thresholds[thresh]
            mu_str = f"{sub[col].mean():>8.4f}" if len(sub) > 0 else f"{'(none)':>8}"
            line += f"  {mu_str}"
        print(line)

    print(f"\n  Dominant driver: {PARAM_LABELS[dominant]}  (|rho|={spearman.iloc[0]['rho']:.4f})")
    print()

    return {"spearman": spearman, "thresholds": thresholds, "dominant": dominant}


def sweep_prior_mean(df:          pd.DataFrame,
                     sweep_param: str,
                     sweep_values: np.ndarray,
                     fixed_mu:    Optional[dict] = None,
                     sigma_scale: float = 0.25,
                     n_boot:      int   = 1000) -> pd.DataFrame:
    """
    Sweep the prior mean of one parameter and record I_LW at each step.

    All other parameter means are fixed at fixed_mu (defaults to low-I cluster
    centre if None).  Sigma = sigma_scale * observed std for each parameter.

    Returns a DataFrame with columns:
        sweep_val, I_LW, CI_lo, CI_hi, ESS, ESS_pct
    """
    # Default fixed means: low-I cluster centre (I < 0.7)
    low_I = df[df[INC_COL] < 0.7]
    default_mu = {col: float(low_I[col].mean()) if len(low_I) > 0
                  else float(df[col].mean())
                  for col in PARAM_COLS}
    if fixed_mu:
        default_mu.update(fixed_mu)

    sigma = {col: df[col].std() * sigma_scale for col in PARAM_COLS}

    rows = []
    for val in sweep_values:
        mu = dict(default_mu)
        mu[sweep_param] = float(val)
        target = TruncatedNormalTarget(df, mu=mu, sigma=sigma,
                                       label=f"{sweep_param}={val:.3f}")
        r = estimate(df, target, n_boot=n_boot, verbose=False)
        rows.append({
            "sweep_val": float(val),
            "I_LW":      r["I_LW"],
            "CI_lo":     r["I_LW_ci"][0],
            "CI_hi":     r["I_LW_ci"][1],
            "ESS":       r["ess"],
            "ESS_pct":   r["ess_pct"],
        })
        print(f"  {sweep_param}={val:.3f}  ->  I_LW={r['I_LW']:.4f}"
              f"  ESS={r['ess']:.0f} ({r['ess_pct']:.1f}%)")

    return pd.DataFrame(rows)


def plot_prior_sweep(sweep_df:    pd.DataFrame,
                     sweep_param: str,
                     I_uniform:   float,
                     out_dir:     Optional[Path | str] = None,
                     fname:       str = "prior_mean_sweep") -> plt.Figure:
    """
    Two-panel figure: (A) I_LW vs prior mean with CI band, (B) ESS %.
    """
    fig, (ax_I, ax_ess) = plt.subplots(
        2, 1, figsize=(7, 5.5), sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.10},
    )

    x   = sweep_df["sweep_val"].values
    y   = sweep_df["I_LW"].values
    lo  = sweep_df["CI_lo"].values
    hi  = sweep_df["CI_hi"].values
    ess = sweep_df["ESS_pct"].values

    # Panel A — I_LW
    ax_I.axhline(I_uniform, color=_C_BLUE, lw=1.3, ls="--",
                 label=f"Uniform mean ({I_uniform:.4f})")
    ax_I.fill_between(x, lo, hi, color=_C_ORANGE, alpha=0.20, zorder=2)
    ax_I.plot(x, y, color=_C_ORANGE, lw=2.0, zorder=3,
              label=r"$\hat{I}_{LW}$ (95% CI)")
    ax_I.scatter(x, y, color=_C_ORANGE, s=18, zorder=4)

    ax_I.set_ylabel(r"Expected inconsistency $I$", fontsize=9)
    ax_I.legend(fontsize=8.5, frameon=False)
    ax_I.tick_params(labelsize=8)
    ax_I.set_title(
        f"Effect of prior mean on $\\hat{{I}}_{{LW}}$  "
        f"(sweeping {PARAM_LABELS[sweep_param]})",
        fontsize=9.5, pad=5,
    )
    ax_I.text(-0.10, 1.06, "A", transform=ax_I.transAxes,
              fontsize=11, fontweight="bold", va="top")

    # Panel B — ESS %
    ax_ess.fill_between(x, 0, ess, color="#999999", alpha=0.30)
    ax_ess.plot(x, ess, color="#555555", lw=1.5)
    ax_ess.axhline(20, color=_C_RED, lw=0.9, ls=":", label="ESS = 20% (reliability floor)")
    ax_ess.set_xlabel(PARAM_LABELS[sweep_param], fontsize=9)
    ax_ess.set_ylabel("ESS (%)", fontsize=9)
    ax_ess.legend(fontsize=8, frameon=False)
    ax_ess.tick_params(labelsize=8)
    ax_ess.set_ylim(0, 105)
    ax_ess.text(-0.10, 1.06, "B", transform=ax_ess.transAxes,
                fontsize=11, fontweight="bold", va="top")

    fig.tight_layout()

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(out_dir / f"{fname}.{ext}", bbox_inches="tight", dpi=300)
        print(f"Saved: {out_dir / fname}.{{pdf,png}}")

    return fig


def plot_three_priors(df:      pd.DataFrame,
                      targets: list[TruncatedNormalTarget],
                      out_dir: Optional[Path | str] = None,
                      fname:   str = "three_prior_comparison") -> plt.Figure:
    """
    Side-by-side panels showing the full plot_lw figure for 3 representative
    priors (uniform, moderate, concentrated), arranged for a paper figure.

    Left:   forest plot of I_LW estimates with CI
    Right:  marginal distributions for delta_c_u (dominant driver)
    """
    I_uniform = float(df[INC_COL].mean())
    results_list = [estimate(df, t, n_boot=1000, verbose=False) for t in targets]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5),
                             gridspec_kw={"width_ratios": [1, 1.4]})
    ax_forest, ax_marg = axes

    # ── Left: forest plot ────────────────────────────────────────────────────
    ax_forest.axvline(I_uniform, color=_C_BLUE, lw=1.3, ls="--",
                      label=f"Uniform mean\n({I_uniform:.4f})", zorder=1)

    colors = [_C_GREY, _C_ORANGE, _C_RED]
    y_pos  = np.arange(len(targets))

    for i, (t, r) in enumerate(zip(targets, results_list)):
        err_lo = r["I_LW"] - r["I_LW_ci"][0]
        err_hi = r["I_LW_ci"][1] - r["I_LW"]
        ax_forest.errorbar(r["I_LW"], y_pos[i],
                           xerr=[[err_lo], [err_hi]],
                           fmt="o", color=colors[i], capsize=5,
                           lw=1.8, ms=7, zorder=3)
        ax_forest.text(
            r["I_LW_ci"][1] + 0.001, y_pos[i],
            f"  {r['I_LW']:.4f}  (ESS={r['ess']:.0f}, {r['ess_pct']:.0f}%)",
            va="center", fontsize=8.5, color="#333333",
        )

    ax_forest.set_yticks(y_pos)
    ax_forest.set_yticklabels([t.label for t in targets], fontsize=9)
    ax_forest.set_xlabel(r"Expected inconsistency $\hat{I}_{LW}$", fontsize=9)
    ax_forest.set_title("LW estimates by prior", fontsize=9.5, pad=5)
    ax_forest.legend(fontsize=8.5, frameon=False)
    ax_forest.tick_params(labelsize=8)
    ax_forest.grid(axis="x", lw=0.4, alpha=0.5)
    ax_forest.text(-0.12, 1.06, "A", transform=ax_forest.transAxes,
                   fontsize=11, fontweight="bold", va="top")

    # ── Right: marginal distributions of delta_c_u ───────────────────────────
    col   = "delta_c_u"
    x_all = df[col].values
    lo_b, hi_b = PARAM_BOUNDS[col]
    x_lo = x_all.min() - 0.02 * (x_all.max() - x_all.min())
    x_hi = x_all.max() + 0.02 * (x_all.max() - x_all.min())
    x_line = np.linspace(x_lo, x_hi, 400)

    ax_marg.hist(x_all, bins=40, density=True,
                 color=_C_GREY, edgecolor="white", linewidth=0.3,
                 zorder=1, label="Data (uniform q)")

    for i, (t, clr) in enumerate(zip(targets, colors)):
        if isinstance(t, TruncatedNormalTarget):
            p_line = t.marginal(col).pdf(x_line)
            ax_marg.plot(x_line, p_line, color=clr, lw=2.0,
                         zorder=3 + i, label=t.label)
            ax_marg.fill_between(x_line, 0, p_line,
                                 color=clr, alpha=0.10, zorder=2)

    ax_marg.set_xlabel(PARAM_LABELS[col] + "  (dominant driver)", fontsize=9)
    ax_marg.set_ylabel("Density", fontsize=9)
    ax_marg.set_title(r"Prior shapes for $\Delta c_u$", fontsize=9.5, pad=5)
    ax_marg.legend(fontsize=8.5, frameon=False)
    ax_marg.tick_params(labelsize=8)
    ax_marg.text(-0.10, 1.06, "B", transform=ax_marg.transAxes,
                 fontsize=11, fontweight="bold", va="top")

    fig.suptitle(
        "Prior sensitivity: effect of concentrating p on low-inconsistency region",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(out_dir / f"{fname}.{ext}", bbox_inches="tight", dpi=300)
        print(f"Saved: {out_dir / fname}.{{pdf,png}}")

    return fig


# ═════════════════════════════════════════════════════════════════════════════
# 6.  DEMO
# ═════════════════════════════════════════════════════════════════════════════

def _demo_data(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    s_u       = rng.uniform(0.1, 2.0, n)
    delta_c_u = rng.uniform(-1.0, 1.0, n)
    R_u       = rng.uniform(0.0, 1.0, n)
    I = (
        0.6 * np.exp(-1.5 * s_u)
        + 0.25 * R_u ** 2
        + 0.05 * np.abs(delta_c_u)
        + 0.08 * rng.standard_normal(n)
    ).clip(0, 1)
    return pd.DataFrame({"s_u": s_u, "delta_c_u": delta_c_u,
                         "R_u": R_u, INC_COL: I})


# ═════════════════════════════════════════════════════════════════════════════
# 7.  CLI
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Likelihood-weighted inconsistency estimation."
    )
    ap.add_argument("--data", type=str, default=None,
                    help="Path to measurements directory or a CSV file.")
    ap.add_argument("--out", type=str,
                    default=str(Path(__file__).parent.parent / "figures" / "lw_inconsistency"),
                    help="Output directory for figures.")
    ap.add_argument("--n-boot", type=int, default=4000,
                    help="Bootstrap replicates (default: 4000).")
    ap.add_argument("--demo", action="store_true",
                    help="Run on synthetic demo data.")
    args = ap.parse_args()

    if args.demo:
        print("Demo mode -- synthetic data")
        df = _demo_data()
    elif args.data is None:
        ap.error("Provide --data <path> or --demo.")
    else:
        p = Path(args.data)
        df = load_saltelli_dataset(str(p)) if p.is_dir() else pd.read_csv(p)

    out = Path(args.out)

    # ── Baseline: fitted truncated normal ────────────────────────────────
    baseline_target  = TruncatedNormalTarget(df, label="Fitted TN (baseline)")
    baseline_results = estimate(df, baseline_target, n_boot=args.n_boot)
    plot_lw(baseline_results, out_dir=out)
    I_uniform = baseline_results["I_uniform"]

    # ── Option 3: landscape analysis — where is I(theta) low? ────────────
    print("\n" + "=" * 55)
    print("Option 3: Landscape analysis")
    print("=" * 55)
    landscape = landscape_analysis(df)
    dominant  = landscape["dominant"]   # parameter with strongest |rho|

    # ── Option 2: sweep prior mean along the dominant driver ─────────────
    print("=" * 55)
    print(f"Option 2: Prior-mean sweep along {PARAM_LABELS[dominant]}")
    print("=" * 55)

    # Sweep the dominant parameter from near-zero to its observed max
    d_min  = float(df[dominant].min())
    d_max  = float(df[dominant].max())
    n_pts  = 20
    sweep_vals = np.linspace(d_min + 0.01 * (d_max - d_min),
                             d_max - 0.01 * (d_max - d_min),
                             n_pts)

    sweep_df = sweep_prior_mean(
        df, dominant, sweep_vals,
        sigma_scale=0.15,    # narrow prior: 15% of observed std
        n_boot=min(args.n_boot, 1000),
    )
    plot_prior_sweep(sweep_df, dominant, I_uniform, out_dir=out)

    # ── Three representative priors for the paper figure ─────────────────
    print("\nBuilding three-prior comparison figure...")
    low_I_mu = {col: float(df[df[INC_COL] < 0.7][col].mean())
                for col in PARAM_COLS}
    sigma_narrow = {col: df[col].std() * 0.15 for col in PARAM_COLS}

    three_targets = [
        TruncatedNormalTarget(df, label="Uniform (baseline)"),
        TruncatedNormalTarget(
            df,
            mu={dominant: float(df[dominant].quantile(0.25))},
            sigma={col: df[col].std() * 0.40 for col in PARAM_COLS},
            label="Moderate prior",
        ),
        TruncatedNormalTarget(
            df,
            mu=low_I_mu,
            sigma=sigma_narrow,
            label="Concentrated prior\n(low-I region)",
        ),
    ]
    plot_three_priors(df, three_targets, out_dir=out)

    plt.show()


if __name__ == "__main__":
    main()
