#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/run_combined_figures.py
===========================
Generate combined paper figures comparing the 2D/3D model and 4D model.

Two figure types produced:

  COMBINED  — both models overlaid in a single figure (solid fill = 2D/3D,
              hatched = 4D).  Suitable for:
                efficiency, pareto, acc_by_dim, accuracy_by_domain,
                exploration_budget, sobol_by_domain, sensitivity_compare,
                acc_scatter, consistency_rate

  SEPARATED — two labelled side-by-side panels in one PDF.  Suitable for
              plots where per-domain rows differ or where overlapping lines
              would obscure detail:
                threshold_by_domain, error_by_regime

The script loads each model once, runs inference, then calls bespoke
combined-plot functions that mirror the originals in surrogate/run_analysis.py
but accept a pair of data dicts (A = 2D/3D, B = 4D).

Usage
-----
    # Full run (val scenarios, both models)
    python src/run_combined_figures.py \\
        --model-2d3d product_transformer_2d3d/product_transformer_exact.pt \\
        --model-4d   product_transformer_4d_real/product_transformer_exact.pt \\
        --scenarios-2d3d 69 32 36 3 25 42 1 57 62 51 58 7 \\
        --scenarios-4d   9 202 216 232 251 256 291 296 304 315 317 \\
        --output results/combined_figures

    # Quick smoke test
    python src/run_combined_figures.py \\
        --model-2d3d product_transformer_2d3d/product_transformer_exact.pt \\
        --model-4d   product_transformer_4d_real/product_transformer_exact.pt \\
        --max-samples 20000 --max-scenarios 5

    # Only a subset of plots
    python src/run_combined_figures.py ... --plots efficiency pareto acc_by_dim
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
_SRC  = Path(__file__).resolve().parent
_ROOT = _SRC.parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# ── Windows: force UTF-8 stdout ───────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Import shared utilities from surrogate.run_analysis ───────────────────────
from surrogate.run_analysis import (
    _load_model,
    _load_accuracy_data,
    _load_timing_and_sobol,
    _sobol_per_scenario,
    _metrics,
    _style_ax,
    PARAMS,
    PARAM_LABELS,
    C_MC, C_MFMC, C_AABB, C_SURR,
    FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND, FS_ANNOT,
    LW_MAIN, LW_GRID, ALPHA_BAND, EPS,
)

# ── Extra colour / style for two-model figures ────────────────────────────────
# Model A = 2D/3D  (solid bars, marker filled)
# Model B = 4D     (hatched bars, marker hollow)
HATCH_B  = "//"          # hatch pattern for model B bars
ALPHA_A  = 0.82
ALPHA_B  = 0.68

LABEL_A  = "2D/3D Model"
LABEL_B  = "4D Model"

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 7, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})


# =============================================================================
# Helpers
# =============================================================================

def _savefig(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {stem}.png/pdf")


def _mean_sobol(sobol_results, key):
    """Average a Sobol index dict over all scenarios."""
    if not sobol_results:
        return {p: 0.0 for p in PARAMS}
    return {p: float(np.mean([r[key][p] for r in sobol_results])) for p in PARAMS}


def _legend_below(fig, handles_or_ax, pad: float = 0.16, ncol: int = None):
    """Place a legend centred below the figure.

    Parameters
    ----------
    handles_or_ax : Axes | list[Artist]
        Either a matplotlib Axes (handles/labels are extracted automatically)
        or an explicit list of artist handles.
    pad : float
        Bottom margin reserved for the legend (figure fraction).
    ncol : int | None
        Number of columns.  Defaults to len(handles) (single row).
        Pass a smaller value to wrap to multiple rows.
    """
    if hasattr(handles_or_ax, "get_legend_handles_labels"):
        handles, labels = handles_or_ax.get_legend_handles_labels()
    else:
        handles = list(handles_or_ax)
        labels  = [h.get_label() for h in handles]
    # De-duplicate while preserving order; skip private labels (_nolegend_ etc.)
    seen, h_out, l_out = set(), [], []
    for h, lbl in zip(handles, labels):
        if lbl not in seen and not lbl.startswith("_"):
            seen.add(lbl); h_out.append(h); l_out.append(lbl)
    if not h_out:
        return
    n_cols = ncol if ncol is not None else len(h_out)
    fig.legend(h_out, l_out,
               loc="lower center",
               bbox_to_anchor=(0.5, 0),
               ncol=n_cols,
               fontsize=FS_LEGEND,
               frameon=True,
               framealpha=0.9,
               handlelength=2.0)
    fig.subplots_adjust(bottom=pad)


# =============================================================================
# COMBINED plots
# =============================================================================

def plot_combined_efficiency(timing_a: dict, timing_b: dict,
                              out_dir: Path) -> None:
    """Grouped bar chart: per-method timing for model A and model B.

    Groups: AABB | Surrogate | MFMC | MC
    Within each group: model-A bar (solid) + model-B bar (hatched).
    """
    print("\n[Combined] Efficiency...")

    methods = ["AABB", "Surrogate", "MFMC"]
    keys    = ["us_aabb", "us_surr", "us_mfmc"]
    colors  = [C_AABB,    C_SURR,    C_MFMC]
    if timing_a.get("us_mc", 0) > 0 or timing_b.get("us_mc", 0) > 0:
        methods.append("MC");  keys.append("us_mc");  colors.append(C_MC)

    x     = np.arange(len(methods))
    w     = 0.35
    fig, ax = plt.subplots(figsize=(7.16, 4.0))

    bars_a = ax.bar(x - w / 2, [max(timing_a.get(k, 0), 0) for k in keys],
                    width=w * 0.92, color=colors, alpha=ALPHA_A, zorder=3,
                    edgecolor="white", linewidth=0.6, label=LABEL_A)
    bars_b = ax.bar(x + w / 2, [max(timing_b.get(k, 0), 0) for k in keys],
                    width=w * 0.92, color=colors, alpha=ALPHA_B, zorder=3,
                    edgecolor="white", linewidth=0.6, hatch=HATCH_B,
                    label=LABEL_B)

    all_vals = ([timing_a.get(k, 0) for k in keys] +
                [timing_b.get(k, 0) for k in keys])
    ymax = max(v for v in all_vals if np.isfinite(v) and v > 0) * 1.18

    for bar, val in list(zip(bars_a, [timing_a.get(k, 0) for k in keys])) + \
                    list(zip(bars_b, [timing_b.get(k, 0) for k in keys])):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ymax * 0.01,
                    f"{val:.1f}", ha="center", va="bottom",
                    fontsize=FS_ANNOT, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel(r"Mean inference time ($\mu$s / sample)", fontsize=FS_LABEL)
    ax.set_title("Inference time per sample — 2D/3D vs 4D", fontsize=FS_TITLE)
    ax.set_ylim(0, ymax)
    _style_ax(ax)
    fig.tight_layout()
    _legend_below(fig, ax)
    _savefig(fig, out_dir, "efficiency")


def plot_combined_pareto(acc_a: dict, timing_a: dict,
                          acc_b: dict, timing_b: dict,
                          out_dir: Path) -> None:
    """Accuracy-vs-speed Pareto scatter for both models.

    Filled markers = model A (2D/3D).  Hollow markers = model B (4D).
    """
    print("\n[Combined] Pareto...")

    _MARKER = {"MC": "*", "Surrogate": "D", "MFMC": "s", "AABB": "o"}
    _SIZE   = {"MC": 200, "Surrogate": 220, "MFMC": 140, "AABB": 120}

    def _pts(acc, timing):
        ref = acc["i_theta"]
        pts = []
        for name, pred, col, t_key in [
            ("AABB",      acc["i_aabb"], C_AABB, "us_aabb"),
            ("MFMC",      acc["i_mfmc"], C_MFMC, "us_mfmc"),
            ("Surrogate", acc["i_surr"], C_SURR, "us_surr"),
        ]:
            m = _metrics(pred, ref)
            t = max(timing.get(t_key, 1e-3), 1e-3)
            pts.append((name, m["rho"], m["r2"], t, col))
        if timing.get("us_mc", 0) > 0:
            m = _metrics(ref, ref)
            pts.append(("MC", m["rho"], m["r2"], timing["us_mc"], C_MC))
        return pts

    pts_a = _pts(acc_a, timing_a)
    pts_b = _pts(acc_b, timing_b)

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.8))

    for ax, y_idx, ylabel, title in [
        (axes[0], 1, r"Spearman $\rho$", r"Accuracy ($\rho$) vs Speed"),
        (axes[1], 2, r"$R^2$",           r"Accuracy ($R^2$) vs Speed"),
    ]:
        for pts, model_lbl, filled in [(pts_a, LABEL_A, True),
                                        (pts_b, LABEL_B, False)]:
            for i, row in enumerate(pts):
                name, rho, r2, t, col = row
                y = rho if y_idx == 1 else r2
                mk = _MARKER.get(name, "o")
                sz = _SIZE.get(name, 140)
                fc = col if filled else "none"
                ax.scatter(t, y, s=sz, color=col, facecolors=fc,
                           marker=mk, zorder=4,
                           edgecolors=col, linewidth=1.2)
                # Annotate model A only — direction depends on method
                if filled:
                    # SW for MFMC/MC, SE for Surrogate, NW for AABB
                    _ann = {"MFMC":      ((-5, -14), "right"),
                            "MC":        ((0,  -14), "center"),
                            "Surrogate": ((5,  -14), "left"),
                            "AABB":      ((0,   8),  "center")}
                    (dx, dy), ha = _ann.get(name, ((-5, 6), "right"))
                    ax.annotate(name, (t, y), xytext=(dx, dy),
                                textcoords="offset points",
                                ha=ha, fontsize=8, color=col,
                                fontweight="bold")

        ax.set_xscale("log")
        ax.set_xlabel(r"Inference time ($\mu$s / sample, log scale)",
                      fontsize=FS_LABEL)
        ax.set_ylabel(ylabel, fontsize=FS_LABEL)
        ax.set_title(title, fontsize=FS_TITLE)
        _style_ax(ax, grid_axis="both")

    from matplotlib.lines import Line2D
    _pareto_handles = [
        Line2D([0], [0], marker="o", color="gray",
               markerfacecolor="gray", markersize=7, label=LABEL_A, linewidth=0),
        Line2D([0], [0], marker="o", color="gray",
               markerfacecolor="none", markersize=7, label=LABEL_B,
               linewidth=0, markeredgewidth=1.5),
    ]
    # Legend in the empty middle band of the right panel
    axes[1].legend(handles=_pareto_handles, fontsize=FS_LEGEND,
                   bbox_to_anchor=(0.65, 0.52), loc="upper left",
                   framealpha=0.9, edgecolor="#cccccc")
    fig.tight_layout()
    _savefig(fig, out_dir, "pareto")


def plot_combined_acc_by_dim(acc_a: dict, acc_b: dict,
                              out_dir: Path) -> None:
    """Bar chart: accuracy by zonotope dimension, combining both models.

    Dims 2D/3D come from model A; dim 4D comes from model B.
    A subtle background band marks which model covers which dims.
    """
    print("\n[Combined] Accuracy by dimension...")

    def _dim_rows(acc):
        dims_all = acc["dim"]
        ref, aabb, mfmc, surr = (acc[k] for k in
                                  ("i_theta", "i_aabb", "i_mfmc", "i_surr"))
        rows = []
        for d in sorted(np.unique(dims_all)):
            m = dims_all == d
            rows.append({
                "dim": d, "N": int(m.sum()),
                **{f"rho_{k}": _metrics(v, ref[m])["rho"]
                   for k, v in [("AABB", aabb[m]), ("MFMC", mfmc[m]),
                                 ("Surr", surr[m])]},
                **{f"r2_{k}":  _metrics(v, ref[m])["r2"]
                   for k, v in [("AABB", aabb[m]), ("MFMC", mfmc[m]),
                                 ("Surr", surr[m])]},
                **{f"mae_{k}": _metrics(v, ref[m])["mae"]
                   for k, v in [("AABB", aabb[m]), ("MFMC", mfmc[m]),
                                 ("Surr", surr[m])]},
            })
        return rows

    rows_a = _dim_rows(acc_a)
    rows_b = _dim_rows(acc_b)
    all_rows = rows_a + rows_b   # non-overlapping dims expected (2,3 + 4)
    all_rows.sort(key=lambda r: r["dim"])

    n_a = len(rows_a)
    x = np.arange(len(all_rows))
    width = 0.24

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.2), sharey=False)
    metric_cfg = [
        ("rho", r"Spearman $\rho$", axes[0]),
        ("r2",  r"$R^2$",           axes[1]),
        ("mae", r"MAE",             axes[2]),
    ]

    for prefix, ylabel, ax in metric_cfg:
        for i_off, (key, col, lbl) in enumerate([
            ("AABB", C_AABB, "AABB"),
            ("MFMC", C_MFMC, "MFMC"),
            ("Surr", C_SURR, "Surrogate"),
        ]):
            vals = [r[f"{prefix}_{key}"] for r in all_rows]
            ax.bar(x + (i_off - 1) * width, vals, width=width * 0.9,
                   color=col, alpha=0.80, label=lbl, zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{r['dim']}D" for r in all_rows])
        ax.set_xlabel("Zonotope dimension", fontsize=FS_LABEL)
        ax.set_ylabel(ylabel, fontsize=FS_LABEL)
        if prefix in ("rho", "r2"):
            ax.set_ylim(0, 1.05)
        _style_ax(ax)

    fig.subplots_adjust(left=0.06, right=0.99, top=0.96,
                        bottom=0.28, wspace=0.35)
    # Legend below the "Zonotope dimension" x-axis label
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="lower center", bbox_to_anchor=(0.5, 0.02),
               ncol=len(handles), fontsize=FS_LEGEND,
               frameon=True, framealpha=0.9, handlelength=2.0)
    _savefig(fig, out_dir, "acc_by_dim")


def plot_combined_accuracy_by_domain(acc_a: dict, domain_a: np.ndarray,
                                      acc_b: dict, domain_b: np.ndarray,
                                      out_dir: Path,
                                      domain_short_a: dict = None,
                                      domain_short_b: dict = None,
                                      domain_order_a: list = None,
                                      domain_order_b: list = None) -> None:
    """Spearman ρ by domain for both models.

    Each domain appears once per model; bars are grouped by (AABB, MFMC, Surr)
    within each model, using solid fill for A and hatching for B.
    """
    print("\n[Combined] Accuracy by domain...")

    def _domain_rows(acc, domain, short, order):
        if len(domain) == 0:
            return []
        ref, aabb, mfmc, surr = (acc[k] for k in
                                  ("i_theta", "i_aabb", "i_mfmc", "i_surr"))
        short   = short or {}
        present = set(domain)
        unique  = [d for d in (order or []) if d in present]
        if not unique:          # order doesn't cover this model's domains
            unique = sorted(present)
        rows = []
        for d in unique:
            m = domain == d
            if m.sum() < 5:
                continue
            r, a, f, s = ref[m], aabb[m], mfmc[m], surr[m]
            rows.append({
                "domain": d,
                "label": short.get(d, d),
                "N": int(m.sum()),
                "rho_AABB": _metrics(a, r)["rho"],
                "rho_MFMC": _metrics(f, r)["rho"],
                "rho_Surr": _metrics(s, r)["rho"],
            })
        return rows

    rows_a = _domain_rows(acc_a, domain_a, domain_short_a, domain_order_a)
    rows_b = _domain_rows(acc_b, domain_b, domain_short_b, domain_order_b)
    if not rows_a and not rows_b:
        print("  [warn] No domain rows — skipping.")
        return

    # Collect unique domain labels (may overlap between models)
    domains_a = {r["domain"] for r in rows_a}
    domains_b = {r["domain"] for r in rows_b}
    all_domains = sorted(domains_a | domains_b)

    # Build a unified row per domain
    def _lookup(rows, dom):
        for r in rows:
            if r["domain"] == dom:
                return r
        return None

    methods = [("AABB", C_AABB), ("MFMC", C_MFMC), ("Surr", C_SURR)]
    n_methods = len(methods)
    group_w   = n_methods * 2 + 1     # width slots per domain group
    positions = []
    xlabels   = []
    bar_data  = []   # list of (x, height, color, hatch, alpha, label)

    x0 = 0
    for dom in all_domains:
        ra = _lookup(rows_a, dom)
        rb = _lookup(rows_b, dom)
        short = (domain_short_a or {}).get(dom,
                 (domain_short_b or {}).get(dom, dom))
        xlabels.append(short)
        group_center = x0 + (group_w - 1) / 2
        positions.append(group_center)

        for i, (key, col) in enumerate(methods):
            xa = x0 + i * 2
            xb = x0 + i * 2 + 1
            va = ra[f"rho_{key}"] if ra else np.nan
            vb = rb[f"rho_{key}"] if rb else np.nan
            bar_data.append((xa, va, col, None,     ALPHA_A))
            bar_data.append((xb, vb, col, HATCH_B,  ALPHA_B))

        x0 += group_w + 0.5

    # Dummy for domain separator lines
    domain_bounds = []
    x0 = 0
    for dom in all_domains:
        domain_bounds.append(x0 - 0.5)
        x0 += group_w + 0.5

    fig, ax = plt.subplots(figsize=(7.16, 4.0))

    for xi, val, col, hatch, alpha in bar_data:
        if np.isfinite(val):
            ax.bar(xi, val, width=0.88, color=col, alpha=alpha,
                   hatch=hatch, zorder=3, edgecolor="white", linewidth=0.4)

    ax.set_xticks(positions)
    ax.set_xticklabels(xlabels, rotation=30, ha="right",
                       fontsize=max(FS_LABEL - 0.5, 6.5))
    ax.set_ylabel(r"Spearman $\rho$", fontsize=FS_LABEL)
    ax.set_xlabel("Domain", fontsize=FS_LABEL)
    ax.set_ylim(0, 1.05)
    _style_ax(ax)

    # ── Legend ────────────────────────────────────────────────────────────────
    from matplotlib.patches import Patch
    handles = []
    for key, col in methods:
        handles.append(Patch(facecolor=col, alpha=ALPHA_A,
                             label=f"{key} ({LABEL_A})", edgecolor="white"))
        handles.append(Patch(facecolor=col, alpha=ALPHA_B, hatch=HATCH_B,
                             label=f"{key} ({LABEL_B})", edgecolor="white"))
    fig.tight_layout()
    _legend_below(fig, handles)
    _savefig(fig, out_dir, "accuracy_by_domain")


def plot_combined_exploration_budget(timing_a: dict, timing_b: dict,
                                      out_dir: Path,
                                      budgets_s=(0.001, 0.01, 0.1, 1.0,
                                                 10.0, 60.0)) -> None:
    """Evaluations per time budget — both models on one log-log plot.

    Surrogate lines are drawn for each model (solid A, dashed B).
    AABB / MFMC shown once (model A, lighter) as baseline reference.
    """
    print("\n[Combined] Exploration budget...")

    budgets_us = np.array(budgets_s) * 1e6

    us_surr_a = max(timing_a.get("us_surr", 1.0), EPS)
    us_surr_b = max(timing_b.get("us_surr", 1.0), EPS)
    us_aabb_a = max(timing_a.get("us_aabb", 1.0), EPS)
    us_mfmc_a = max(timing_a.get("us_mfmc", 1.0), EPS)

    fig, ax = plt.subplots(figsize=(7.16, 4.0))

    # Baselines (model A only, de-emphasised)
    for name, t_us, col, ls, lw, alpha, zo in [
        ("AABB",       us_aabb_a, C_AABB, ":", LW_MAIN - 0.6, 0.50, 2),
        ("MFMC",       us_mfmc_a, C_MFMC, "--", LW_MAIN - 0.2, 0.65, 3),
    ]:
        ax.plot(budgets_s, budgets_us / t_us,
                color=col, lw=lw, ls=ls, alpha=alpha,
                marker="o", markersize=3, label=name, zorder=zo)

    # Surrogate — both models
    ax.plot(budgets_s, budgets_us / us_surr_a,
            color=C_SURR, lw=LW_MAIN + 1.4, ls="-",
            marker="^", markersize=5,
            label=f"Surrogate ({LABEL_A})", zorder=5, alpha=1.0)
    ax.plot(budgets_s, budgets_us / us_surr_b,
            color=C_SURR, lw=LW_MAIN + 0.8, ls=(0, (4, 2)),
            marker="s", markersize=4,
            label=f"Surrogate ({LABEL_B})", zorder=4, alpha=0.82)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Time budget (s)", fontsize=FS_LABEL)
    ax.set_ylabel("Evaluations achievable", fontsize=FS_LABEL)
    ax.set_title("Scalable exploration: evaluations per time budget",
                 fontsize=FS_TITLE)

    _ymax = max(1e8, float(budgets_us[-1] / min(us_surr_a, us_surr_b)) * 2)
    _shade_kw = dict(alpha=0.045, zorder=0)
    ax.axvspan(budgets_s[0], 0.05,  color="#4dac26", **_shade_kw)
    ax.axvspan(0.05,          5.0,  color="#f1a340", **_shade_kw)
    ax.axvspan(5.0,  budgets_s[-1], color="#d01c8b", **_shade_kw)
    # Use axes-fraction coordinates so text never escapes the log-scale axis
    ax.text(0.04, 0.88, "Interactive",  fontsize=FS_ANNOT - 0.5,
            color="#2e7d32", alpha=0.85, ha="center",
            transform=ax.transAxes)
    ax.text(0.43, 0.88, "Large-scale",  fontsize=FS_ANNOT - 0.5,
            color="#b45309", alpha=0.85, ha="center",
            transform=ax.transAxes)
    ax.text(0.94, 0.88, "Exhaustive",   fontsize=FS_ANNOT - 0.5,
            color="#9b2226", alpha=0.85, ha="right",
            transform=ax.transAxes)

    _style_ax(ax, grid_axis="both")
    fig.tight_layout()
    _legend_below(fig, ax)
    _savefig(fig, out_dir, "exploration_budget")


def _sobol_bootstrap_ci(records, key, param, n_boot=1000):
    """Bootstrap 95% CI on mean Sobol index across scenarios.

    records: list of dicts with key -> {param: float}
    Returns (mean, lo, hi) or (val, nan, nan) if < 3 records.
    """
    vals = np.array([r[key][param] for r in records
                     if key in r and param in r[key]])
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(vals))
    if len(vals) < 3:
        return mean, np.nan, np.nan
    rng = np.random.default_rng(42)
    boot = np.array([np.mean(rng.choice(vals, size=len(vals), replace=True))
                     for _ in range(n_boot)])
    return mean, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def plot_combined_sobol_by_domain(sobol_a: list, sobol_b: list,
                                   out_dir: Path,
                                   domain_short_a: dict = None,
                                   domain_short_b: dict = None,
                                   domain_order_a: list = None,
                                   domain_order_b: list = None,
                                   domain_remap_b: dict = None) -> None:
    """Sobol heatmap combining both models.

    Layout: 2 column-pairs (MC | Surrogate) × 2 rows (S1 | ST).
    Domains from model A (2D/3D) are listed first, then model B (4D),
    separated by a bold horizontal rule.

    domain_remap_b: optional dict mapping domain names in sobol_b to a merged
        label, e.g. {"Phys. Coupling": "Engineering", "Spec. Gap": "Engineering"}.
        All records whose domain maps to the same target are pooled together.
    """
    if not sobol_a and not sobol_b:
        print("\n[Combined] No Sobol results — skipping sobol_by_domain.")
        return
    print("\n[Combined] Sobol by domain heatmap...")

    def _collect(sobol_results, short, order, remap=None):
        by_dom = defaultdict(list)
        for r in sobol_results:
            dom = r.get("domain", "Unknown")
            if remap:
                dom = remap.get(dom, dom)
            by_dom[dom].append(r)
        unique = order if order else sorted(by_dom.keys())
        unique = [d for d in unique if d in by_dom]
        if not unique:
            unique = sorted(by_dom.keys())
        labels = [short.get(d, d) for d in unique]
        return unique, labels, by_dom

    doms_a, dlbls_a, bdom_a = _collect(sobol_a, domain_short_a or {},
                                        domain_order_a)
    doms_b, dlbls_b, bdom_b = _collect(sobol_b, domain_short_b or {},
                                        domain_order_b, remap=domain_remap_b)

    # Combine: model A domains first, then model B
    all_doms  = doms_a  + doms_b
    all_dlbls = dlbls_a + dlbls_b
    n_a = len(doms_a)
    n_dom = len(all_doms)
    plbls = [PARAM_LABELS[p] for p in PARAMS]

    configs = [
        (0, 0, "S1_mc",   r"MC — first-order $S_1$"),
        (0, 1, "S1_surr", r"Surrogate — first-order $S_1$"),
        (1, 0, "ST_mc",   r"MC — total-effect $S_T$"),
        (1, 1, "ST_surr", r"Surrogate — total-effect $S_T$"),
    ]

    def _mean_row(dom, key, bdom):
        rlist = bdom.get(dom, [])
        if not rlist:
            return [np.nan] * len(PARAMS)
        return [float(np.mean([r[key][p] for r in rlist])) for p in PARAMS]

    Z_dict = {}
    for _, _, key, _ in configs:
        rows = []
        for i, dom in enumerate(all_doms):
            if i < n_a:
                rows.append(_mean_row(dom, key, bdom_a))
            else:
                rows.append(_mean_row(dom, key, bdom_b))
        Z_dict[key] = np.array(rows)

    # ── Bootstrap CI matrices (lo/hi per domain × param) ──────────────────────
    CI_lo = {}; CI_hi = {}
    for _, _, key, _ in configs:
        lo_rows = []; hi_rows = []
        for i, dom in enumerate(all_doms):
            bdom = bdom_a if i < n_a else bdom_b
            rlist = bdom.get(dom, [])
            lo_row = []; hi_row = []
            for p in PARAMS:
                _, lo, hi = _sobol_bootstrap_ci(rlist, key, p)
                lo_row.append(lo); hi_row.append(hi)
            lo_rows.append(lo_row); hi_rows.append(hi_row)
        CI_lo[key] = np.array(lo_rows, dtype=float)
        CI_hi[key] = np.array(hi_rows, dtype=float)

    # ── MC mean lookup: mc_means[dom][key][param_idx] ─────────────────────────
    mc_key_map = {"S1_surr": "S1_mc", "ST_surr": "ST_mc"}
    mc_means = {}
    for i, dom in enumerate(all_doms):
        bdom = bdom_a if i < n_a else bdom_b
        mc_means[dom] = {}
        for mc_key in ("S1_mc", "ST_mc"):
            mc_means[dom][mc_key] = _mean_row(dom, mc_key, bdom)

    row_h = max(0.42 * n_dom, 2.5)
    fig, axes = plt.subplots(
        2, 2,
        figsize=(7.16, row_h * 2 + 0.8),
        gridspec_kw={"hspace": 0.28, "wspace": 0.08},
    )

    im_last = None
    for row, col, key, title in configs:
        ax = axes[row, col]
        Z  = Z_dict[key]
        Zlo = CI_lo[key]
        Zhi = CI_hi[key]
        is_surrogate = key in ("S1_surr", "ST_surr")
        mc_key = mc_key_map.get(key)

        im = ax.imshow(Z, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
        im_last = im

        ax.set_xticks(np.arange(len(PARAMS)))
        ax.set_xticklabels(plbls, fontsize=FS_TICK)

        if col == 0:
            ax.set_yticks(np.arange(n_dom))
            ax.set_yticklabels(all_dlbls, fontsize=FS_TICK)
        else:
            ax.set_yticks(np.arange(n_dom))
            ax.set_yticklabels([], fontsize=FS_TICK)

        dom_max_col = np.nanargmax(Z, axis=1)
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                v = Z[i, j]
                if not np.isfinite(v):
                    continue
                clr  = "white" if v > 0.60 else "black"
                bold = "bold" if j == dom_max_col[i] else "normal"
                # Line 1: value (same as before)
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=FS_ANNOT - 0.5, color=clr, fontweight=bold)
                # Line 2: [lo, hi] CI (non-diff panels only)
                lo_v = Zlo[i, j]; hi_v = Zhi[i, j]
                if np.isfinite(lo_v) and np.isfinite(hi_v):
                    ci_txt = f"\u00b1{(hi_v - lo_v) / 2:.2f}"
                    ax.text(j, i + 0.28, ci_txt, ha="center", va="center",
                            fontsize=max(5, FS_ANNOT - 2),
                            color="#555555", fontweight="normal")
                # Line 3: Δ vs MC (surrogate panels only)
                if is_surrogate and mc_key is not None:
                    dom = all_doms[i]
                    mc_val = mc_means[dom][mc_key][j]
                    if np.isfinite(mc_val) and np.isfinite(v):
                        delta = v - mc_val
                        sign = "+" if delta >= 0 else "-"
                        delta_txt = f"\u0394{sign}{abs(delta):.2f}"
                        ax.text(j, i - 0.28, delta_txt, ha="center", va="center",
                                fontsize=max(4, FS_ANNOT - 3),
                                color="#777777", fontweight="normal")

        ax.set_title(title, fontsize=FS_TITLE, pad=4)
        ax.spines[["top", "right"]].set_visible(False)

        # Bold separator line between model A and B domains
        if n_a > 0 and n_a < n_dom:
            ax.axhline(n_a - 0.5, color="#555", lw=1.6, zorder=5)

        pass  # model labels in left margin removed

    fig.subplots_adjust(right=0.87)
    cbar_ax = fig.add_axes([0.89, 0.12, 0.018, 0.76])
    cb = fig.colorbar(im_last, cax=cbar_ax)
    cb.set_label("Mean Sobol index", fontsize=FS_LABEL)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cb.ax.tick_params(labelsize=FS_TICK)

    for row_i, lbl in enumerate([r"First-order ($S_1$)",
                                   r"Total-effect ($S_T$)"]):
        axes[row_i, 0].set_ylabel(lbl, fontsize=FS_LABEL, labelpad=6)

    _savefig(fig, out_dir, "sobol_by_domain")


def plot_combined_sensitivity_compare(sobol_a: list, sobol_b: list,
                                       out_dir: Path) -> None:
    """S1 and ST bar charts for MC/MFMC/Surrogate — both models overlaid.

    Solid fill = model A (2D/3D); hatched = model B (4D).
    Two panels: S1 (left) and ST (right).
    """
    if not sobol_a and not sobol_b:
        return
    print("\n[Combined] Sensitivity compare...")

    param_lbls = [PARAM_LABELS[p] for p in PARAMS]
    x = np.arange(len(PARAMS))
    n_methods = 3   # MC, MFMC, Surr
    bar_slot_w = 0.13
    group_w    = n_methods * 2 * bar_slot_w

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.8), sharey=True)
    method_cfg = [
        ("mc",   C_MC,   "MC ($I_\\theta$)"),
        ("mfmc", C_MFMC, "MFMC"),
        ("surr", C_SURR, "Surrogate"),
    ]

    for ax, prefix, ylabel, title in [
        (axes[0], "S1", r"Sobol $S_1$", r"First-order indices $S_1$"),
        (axes[1], "ST", r"Sobol $S_T$", r"Total-effect indices $S_T$"),
    ]:
        for i, (method, col, lbl) in enumerate(method_cfg):
            k = f"{prefix}_{method}"
            sa = _mean_sobol(sobol_a, k)
            sb = _mean_sobol(sobol_b, k)
            offset_a = (i * 2 - n_methods + 0.5) * bar_slot_w
            offset_b = offset_a + bar_slot_w
            ax.bar(x + offset_a,
                   [sa[p] for p in PARAMS],
                   width=bar_slot_w * 0.9,
                   color=col, alpha=ALPHA_A, zorder=3,
                   label=f"{lbl} — {LABEL_A}")
            ax.bar(x + offset_b,
                   [sb[p] for p in PARAMS],
                   width=bar_slot_w * 0.9,
                   color=col, alpha=ALPHA_B, zorder=3,
                   hatch=HATCH_B, edgecolor="white", linewidth=0.3,
                   label=f"{lbl} — {LABEL_B}")

        ax.set_xticks(x)
        ax.set_xticklabels(param_lbls, fontsize=FS_LABEL)
        ax.set_ylabel(ylabel, fontsize=FS_LABEL)
        ax.set_title(title, fontsize=FS_TITLE)
        ax.set_ylim(0, 1.05)
        _style_ax(ax)

    fig.tight_layout()
    _legend_below(fig, axes[0])
    _savefig(fig, out_dir, "sensitivity_compare")


def plot_combined_acc_scatter(acc_a: dict, acc_b: dict,
                               out_dir: Path,
                               max_pts: int = 40_000) -> None:
    """6-panel dot scatter (predicted vs MC) — 3 methods × 2 models.

    Rows: model A (2D/3D) top, model B (4D) bottom.
    Columns: AABB | MFMC | Surrogate.
    Same dot style as single-model eval.
    """
    import matplotlib.colors as mcolors
    print("\n[Combined] Accuracy scatter...")

    method_cfg = [
        ("AABB",      "i_aabb", C_AABB, r"AABB $(1-J_C)$"),
        ("MFMC",      "i_mfmc", C_MFMC, r"MFMC $(1-I_\mathrm{MF})$"),
        ("Surrogate", "i_surr", C_SURR, r"$\hat{I}$ (Surrogate)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(7.16, 5.0),
                             gridspec_kw={"wspace": 0.10, "hspace": 0.38})
    rng = np.random.default_rng(0)

    for row_idx, (acc, model_lbl) in enumerate([(acc_a, LABEL_A),
                                                  (acc_b, LABEL_B)]):
        ref = acc["i_theta"]
        for col_idx, (method_name, key, col, ylabel) in enumerate(method_cfg):
            ax  = axes[row_idx, col_idx]
            est = acc[key]
            valid = np.isfinite(ref) & np.isfinite(est)
            ref_v, est_v = ref[valid], est[valid]

            n = len(ref_v)
            if n > max_pts:
                idx = rng.choice(n, size=max_pts, replace=False)
                ref_p, est_p = ref_v[idx], est_v[idx]
            else:
                ref_p, est_p = ref_v, est_v

            alpha = max(0.08, min(0.40, 6_000 / max(len(ref_p), 1)))
            ax.set_facecolor(mcolors.to_rgba(col, alpha=0.06))
            ax.scatter(ref_p, est_p, s=2.5, color=col, alpha=alpha,
                       linewidths=0, rasterized=True, zorder=2)
            ax.plot([0, 1], [0, 1], color="#333333", lw=1.4, ls="--",
                    alpha=0.6, zorder=3)

            m = _metrics(est_v, ref_v)
            ann = (rf"$\rho={m['rho']:.3f}$" + "\n"
                   rf"$R^2={m['r2']:.3f}$"   + "\n"
                   rf"MAE$={m['mae']:.4f}$")
            ax.text(0.04, 0.97, ann, transform=ax.transAxes, va="top",
                    fontsize=FS_ANNOT + 0.5, linespacing=1.45,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec="#cccccc", alpha=0.85, lw=0.6))

            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
            ax.set_aspect("equal", adjustable="box")
            # Column titles only on top row
            if row_idx == 0:
                ax.set_title(method_name, fontsize=FS_TITLE)
            _style_ax(ax, grid_axis="both")

    # Row labels on far left, rotated
    row_labels = [LABEL_A, LABEL_B]
    for row_idx, lbl in enumerate(row_labels):
        axes[row_idx, 0].set_ylabel(lbl, fontsize=FS_LABEL, labelpad=8)

    # Single shared x-axis label centred below all panels
    fig.text(0.5, 0.01, r"$I_\theta$ — MC reference", ha="center",
             fontsize=FS_LABEL)
    fig.subplots_adjust(left=0.10, bottom=0.08)
    _savefig(fig, out_dir, "acc_scatter")


def plot_combined_consistency_rate(acc_a: dict, acc_b: dict,
                                    out_dir: Path,
                                    gammas: np.ndarray = None) -> None:
    """Overlaid CDFs P(I ≤ γ) for both models — all methods."""
    print("\n[Combined] Consistency rate...")

    if gammas is None:
        gammas = np.linspace(0, 1, 200)

    fig, ax = plt.subplots(figsize=(7.16, 4.0))

    for acc, ls_surr, model_lbl in [
        (acc_a, "-",         LABEL_A),
        (acc_b, (0, (5, 3)), LABEL_B),
    ]:
        ref  = acc["i_theta"]
        aabb = acc["i_aabb"]
        mfmc = acc["i_mfmc"]
        surr = acc["i_surr"]

        for arr, col, base_lbl, ls in [
            (ref,  C_MC,   "MC",      ":"),
            (aabb, C_AABB, "AABB",    "--"),
            (mfmc, C_MFMC, "MFMC",    "-."),
            (surr, C_SURR, "Surrogate", ls_surr),
        ]:
            valid = np.isfinite(arr)
            cdf = np.array([float((arr[valid] <= g).mean()) for g in gammas])
            ax.plot(gammas, cdf, color=col, lw=LW_MAIN - 0.4, ls=ls, alpha=0.80,
                    label=f"{base_lbl} ({model_lbl})", zorder=3)

    ax.axvline(0.5, color="#888", lw=0.9, ls=":", alpha=0.6, zorder=2,
               label=r"$\gamma = 0.5$")
    ax.set_xlabel(r"Threshold $\gamma$", fontsize=FS_LABEL)
    ax.set_ylabel(r"$P(I \leq \gamma)$", fontsize=FS_LABEL)
    ax.set_title("Consistency rate CDF — 2D/3D vs 4D", fontsize=FS_TITLE)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    _style_ax(ax, grid_axis="both")
    fig.tight_layout()
    # 9 items (4 methods × 2 models + gamma line) → wrap to 5 columns, 2 rows
    _legend_below(fig, ax, pad=0.30, ncol=5)
    _savefig(fig, out_dir, "consistency_rate")


# =============================================================================
# SEPARATED plots  (two labelled panels, one PDF)
# =============================================================================

def plot_separated_threshold_by_domain(acc_a: dict, domain_a: np.ndarray,
                                        acc_b: dict, domain_b: np.ndarray,
                                        out_dir: Path,
                                        threshold: float = 0.5,
                                        domain_short_a: dict = None,
                                        domain_short_b: dict = None,
                                        domain_order_a: list = None,
                                        domain_order_b: list = None) -> None:
    """FPR/FNR by domain — FPR panel on top, FNR panel below (vertical).

    Style matches the single-model Q1 figure 7: wide bars per domain group,
    horizontal x-labels, legend inside top panel.
    """
    print(f"\n[Separated] Threshold by domain (gamma={threshold})...")

    def _domain_rows(acc, domain, short, order):
        if len(domain) == 0:
            return []
        ref, aabb, mfmc, surr = (acc[k] for k in
                                  ("i_theta", "i_aabb", "i_mfmc", "i_surr"))
        short  = short or {}
        present = set(domain)
        unique  = [d for d in (order or []) if d in present]
        if not unique:
            unique = sorted(present)
        rows = []
        for d in unique:
            m = domain == d
            if m.sum() < 10:
                continue
            rv = ref[m]
            row = {"domain": d, "label": short.get(d, d)}
            for key, pred in [("AABB", aabb[m]), ("MFMC", mfmc[m]),
                               ("Surr", surr[m])]:
                valid = np.isfinite(pred) & np.isfinite(rv)
                xv, rv2 = pred[valid], rv[valid]
                pp = xv >= threshold; ap = rv2 >= threshold; an = rv2 < threshold
                n_pos = ap.sum(); n_neg = an.sum()
                row[f"fpr_{key}"] = float((pp & an).sum() / n_neg) if n_neg else np.nan
                row[f"fnr_{key}"] = float((~pp & ap).sum() / n_pos) if n_pos else np.nan
            rows.append(row)
        return rows

    rows_a = _domain_rows(acc_a, domain_a, domain_short_a, domain_order_a)
    rows_b = _domain_rows(acc_b, domain_b, domain_short_b, domain_order_b)
    if not rows_a and not rows_b:
        print("  [warn] No domain rows — skipping.")
        return

    def _lookup(rows, dom):
        for r in rows:
            if r["domain"] == dom:
                return r
        return None

    doms_a = [r["domain"] for r in rows_a]
    doms_b = [r["domain"] for r in rows_b]
    all_domains = list(dict.fromkeys(doms_a + [d for d in doms_b if d not in doms_a]))
    short_all   = {**(domain_short_a or {}), **(domain_short_b or {})}
    xlabels     = [short_all.get(d, d) for d in all_domains]

    methods   = [("AABB", C_AABB), ("MFMC", C_MFMC), ("Surr", C_SURR)]
    n_methods = len(methods)        # 3 methods
    n_dom     = len(all_domains)

    # ── Bar geometry — wide bars matching figure-7 style ──────────────────────
    # Within each domain group: 6 bars (3 methods × 2 models), paired by method.
    # Constraints: total span of 6 bars must stay < 1.0 (domain spacing) so
    # bars from adjacent groups never overlap.
    # With BAR_W=0.13, PAIR_GAP=0.02, METHOD_GAP=0.15 the span = 0.88 < 1.0.
    BAR_W      = 0.13          # individual bar width
    PAIR_GAP   = 0.02          # gap within A/B pair (between solid and hatched)
    METHOD_GAP = 0.15          # extra gap between method pairs
    # offsets from domain centre for each of the 6 bars
    pair_w  = BAR_W + PAIR_GAP                  # centre-to-centre within pair
    step    = pair_w + METHOD_GAP               # centre-to-centre between pairs
    # method pair centres: -step, 0, +step
    m_ctrs  = np.array([-step, 0.0, step])
    off_a   = m_ctrs - PAIR_GAP / 2 - BAR_W / 2   # solid bar (model A)
    off_b   = m_ctrs + PAIR_GAP / 2 + BAR_W / 2   # hatched bar (model B)

    x = np.arange(n_dom, dtype=float)

    _LOG_FLOOR = 1e-4

    def _draw_panel(ax, rate, ylabel, title):
        for i_dom, dom in enumerate(all_domains):
            ra = _lookup(rows_a, dom)
            rb = _lookup(rows_b, dom)
            for i_m, (key, col) in enumerate(methods):
                va = ra[f"{rate}_{key}"] if ra else np.nan
                vb = rb[f"{rate}_{key}"] if rb else np.nan
                if np.isfinite(va):
                    ax.bar(x[i_dom] + off_a[i_m], max(va, _LOG_FLOOR),
                           width=BAR_W, color=col, alpha=ALPHA_A,
                           zorder=3, edgecolor="white", linewidth=0.5)
                if np.isfinite(vb):
                    ax.bar(x[i_dom] + off_b[i_m], max(vb, _LOG_FLOOR),
                           width=BAR_W, color=col, alpha=ALPHA_B,
                           hatch=HATCH_B, zorder=3,
                           edgecolor="white", linewidth=0.5)

        ax.set_yscale("log")
        ax.set_ylim(_LOG_FLOOR * 0.5, 2.0)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=FS_LABEL,
                           rotation=30, ha="right")
        ax.set_ylabel(ylabel, fontsize=FS_LABEL)
        ax.set_title(title, fontsize=FS_TITLE)
        _style_ax(ax)

    fig, (ax_fpr, ax_fnr) = plt.subplots(
        2, 1, figsize=(7.16, 5.5),
        gridspec_kw={"hspace": 0.55},
    )

    _draw_panel(ax_fpr, "fpr",
                r"FPR (log scale)",
                rf"FPR at $\gamma={threshold}$")
    _draw_panel(ax_fnr, "fnr",
                r"FNR (log scale)",
                rf"FNR at $\gamma={threshold}$")

    # Legend inside the FPR panel (top-right), matching figure-7 placement
    from matplotlib.patches import Patch
    handles = []
    for key, col in methods:
        handles.append(Patch(facecolor=col, alpha=ALPHA_A,
                             label=f"{key} ({LABEL_A})", edgecolor="white"))
        handles.append(Patch(facecolor=col, alpha=ALPHA_B, hatch=HATCH_B,
                             label=f"{key} ({LABEL_B})", edgecolor="white"))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.96,
                        bottom=0.16, hspace=0.55)
    thr_labels = [h.get_label() for h in handles]
    fig.legend(handles, thr_labels,
               loc="lower center", bbox_to_anchor=(0.5, 0.01),
               ncol=3, fontsize=FS_LEGEND,
               frameon=True, framealpha=0.9, handlelength=2.0)
    _savefig(fig, out_dir, "threshold_by_domain")


def plot_auroc_ap_by_domain(acc_a: dict, domain_a: np.ndarray,
                             acc_b: dict, domain_b: np.ndarray,
                             out_dir: Path,
                             threshold: float = 0.5,
                             domain_short_a: dict = None,
                             domain_short_b: dict = None,
                             domain_order_a: list = None,
                             domain_order_b: list = None,
                             domain_remap_b: dict = None) -> None:
    """AUROC and Average Precision (AP) per domain per method, both models.

    Two panels stacked vertically: top = AUROC, bottom = AP.
    Within each domain group: 6 bars (3 methods × 2 models).
    Binary label: y_true = (i_theta >= threshold).
    """
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
    except ImportError:
        print("\n[Separated] scikit-learn not found — skipping auroc_ap_by_domain.")
        return

    print(f"\n[Separated] AUROC / AP by domain (gamma={threshold})...")

    # Apply domain remapping for model B (e.g. merge 4D Engineering → "Engineering")
    if domain_remap_b and len(domain_b):
        domain_b = np.array([domain_remap_b.get(d, d) for d in domain_b])

    def _domain_rows(acc, domain, short, order):
        if len(domain) == 0:
            return []
        ref = acc["i_theta"]
        short = short or {}
        present = set(domain)
        unique = [d for d in (order or []) if d in present]
        if not unique:
            unique = sorted(present)
        rows = []
        for d in unique:
            m = domain == d
            if m.sum() < 20:
                continue
            rv = ref[m]
            y_true = (rv >= threshold).astype(int)
            if y_true.sum() < 5 or (len(y_true) - y_true.sum()) < 5:
                continue  # need both classes
            row = {"domain": d, "label": short.get(d, d)}
            for key, pred_key in [("AABB", "i_aabb"), ("MFMC", "i_mfmc"), ("Surr", "i_surr")]:
                pred = acc[pred_key][m].copy()
                # NaN in MFMC means no intersection found → predict 0 (consistent)
                if pred_key == "i_mfmc":
                    pred = np.where(np.isfinite(pred), pred, 0.0)
                valid = np.isfinite(pred) & np.isfinite(rv)
                if valid.sum() < 20:
                    row[f"auroc_{key}"] = np.nan
                    row[f"ap_{key}"] = np.nan
                    continue
                yt, ys = y_true[valid], pred[valid]
                try:
                    row[f"auroc_{key}"] = float(roc_auc_score(yt, ys))
                    row[f"ap_{key}"]    = float(average_precision_score(yt, ys))
                except Exception:
                    row[f"auroc_{key}"] = np.nan
                    row[f"ap_{key}"]    = np.nan
            rows.append(row)
        return rows

    rows_a = _domain_rows(acc_a, domain_a, domain_short_a, domain_order_a)
    rows_b = _domain_rows(acc_b, domain_b, domain_short_b, domain_order_b)
    if not rows_a and not rows_b:
        print("  [warn] No domain rows — skipping.")
        return

    doms_a = [r["domain"] for r in rows_a]
    doms_b = [r["domain"] for r in rows_b]
    all_domains = list(dict.fromkeys(doms_a + [d for d in doms_b if d not in doms_a]))
    short_all   = {**(domain_short_a or {}), **(domain_short_b or {})}
    xlabels     = [short_all.get(d, d) for d in all_domains]

    methods   = [("AABB", C_AABB), ("MFMC", C_MFMC), ("Surr", C_SURR)]
    n_dom     = len(all_domains)
    BAR_W     = 0.13
    PAIR_GAP  = 0.02
    METHOD_GAP = 0.15
    pair_w    = BAR_W + PAIR_GAP
    step      = pair_w + METHOD_GAP
    m_ctrs    = np.array([-step, 0.0, step])
    off_a     = m_ctrs - PAIR_GAP / 2 - BAR_W / 2
    off_b     = m_ctrs + PAIR_GAP / 2 + BAR_W / 2
    xs        = np.arange(n_dom)

    def _lookup(rows, dom):
        for r in rows:
            if r["domain"] == dom:
                return r
        return None

    fig, axes = plt.subplots(2, 1, figsize=(7.16, 5.5), sharex=True)
    metrics_cfg = [
        (axes[0], "auroc", "AUROC"),
        (axes[1], "ap",    "Average Precision (AP)"),
    ]

    legend_handles = []
    for ax, metric_key, ylabel in metrics_cfg:
        for mi, (mname, mcol) in enumerate(methods):
            vals_a = [(_lookup(rows_a, d) or {}).get(f"{metric_key}_{mname}", np.nan)
                      for d in all_domains]
            vals_b = [(_lookup(rows_b, d) or {}).get(f"{metric_key}_{mname}", np.nan)
                      for d in all_domains]
            ba = ax.bar(xs + off_a[mi], vals_a, BAR_W, color=mcol, alpha=0.85,
                        label=mname if metric_key == "auroc" else None)
            bb = ax.bar(xs + off_b[mi], vals_b, BAR_W, color=mcol, alpha=0.85,
                        hatch="///", edgecolor="white", linewidth=0.4,
                        label=None)
            if metric_key == "auroc":
                legend_handles.append(ba)

        ax.axhline(0.5, color="grey", lw=0.8, ls="--", alpha=0.6)  # random baseline
        ax.set_ylabel(ylabel, fontsize=FS_LABEL)
        ax.set_ylim(0.0, 1.05)
        ax.tick_params(axis="y", labelsize=FS_TICK)
        ax.yaxis.grid(True, lw=LW_GRID, alpha=0.4)
        ax.set_axisbelow(True)

    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(xlabels, rotation=30, ha="right", fontsize=FS_TICK)

    # Dummy hatched patch for legend
    import matplotlib.patches as mpatches
    patch_a = mpatches.Patch(facecolor="grey", alpha=0.85, label="Model A (2D/3D)")
    patch_b = mpatches.Patch(facecolor="grey", alpha=0.85, hatch="///",
                              edgecolor="white", label="Model B (4D)")
    all_handles = legend_handles + [patch_a, patch_b]
    all_labels  = [h.get_label() for h in legend_handles] + ["Model A (2D/3D)", "Model B (4D)"]
    fig.legend(all_handles, all_labels, loc="lower center",
               bbox_to_anchor=(0.5, 0.01), ncol=5,
               fontsize=FS_LEGEND, frameon=True, framealpha=0.9, handlelength=1.8)

    fig.subplots_adjust(left=0.08, right=0.99, top=0.96, bottom=0.22, hspace=0.25)
    _savefig(fig, out_dir, "auroc_ap_by_domain")


def plot_pr_curves(acc_a: dict, domain_a: np.ndarray,
                   acc_b: dict, domain_b: np.ndarray,
                   out_dir: Path,
                   threshold: float = 0.5,
                   domain_short_a: dict = None,
                   domain_short_b: dict = None,
                   domain_order_a: list = None,
                   domain_order_b: list = None,
                   domain_remap_b: dict = None) -> None:
    """Precision-recall curves per domain, all methods, both models.

    One subplot per domain. Solid lines = Model A, dashed = Model B.
    AP score annotated in legend.
    """
    try:
        from sklearn.metrics import precision_recall_curve, average_precision_score
    except ImportError:
        print("\n[Separated] scikit-learn not found — skipping pr_curves.")
        return

    print(f"\n[Separated] PR curves by domain (gamma={threshold})...")

    short_all   = {**(domain_short_a or {}), **(domain_short_b or {})}
    domain_a_arr = np.asarray(domain_a) if not isinstance(domain_a, np.ndarray) else domain_a
    domain_b_arr = np.asarray(domain_b) if not isinstance(domain_b, np.ndarray) else domain_b

    # Apply domain remapping for model B (e.g. merge 4D Engineering → "Engineering")
    if domain_remap_b and len(domain_b_arr):
        domain_b_arr = np.array([domain_remap_b.get(d, d) for d in domain_b_arr])

    present_a = set(domain_a_arr)
    present_b = set(domain_b_arr)
    order_a = [d for d in (domain_order_a or []) if d in present_a] or sorted(present_a)
    all_domains = list(dict.fromkeys(
        order_a + [d for d in (domain_order_b or sorted(present_b)) if d in present_b and d not in present_a]
    ))

    n_dom = len(all_domains)
    if n_dom == 0:
        return

    ncols = min(3, n_dom)
    nrows = (n_dom + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.16, 2.4 * nrows),
                             squeeze=False)

    method_cfg = [
        ("AABB", "i_aabb", C_AABB),
        ("MFMC", "i_mfmc", C_MFMC),
        ("Surr", "i_surr", C_SURR),
    ]

    for idx, dom in enumerate(all_domains):
        ax = axes[idx // ncols][idx % ncols]
        label = short_all.get(dom, dom)
        ax.set_title(label, fontsize=FS_TICK + 1)

        for mname, pred_key, mcol in method_cfg:
            for acc, dom_arr, ls, model_tag in [
                (acc_a, domain_a_arr, "-",  "A"),
                (acc_b, domain_b_arr, "--", "B"),
            ]:
                m = dom_arr == dom
                if m.sum() < 20:
                    continue
                rv   = acc["i_theta"][m]
                pred = acc[pred_key][m].copy()
                # NaN in MFMC means no intersection found → predict 0 (consistent)
                if pred_key == "i_mfmc":
                    pred = np.where(np.isfinite(pred), pred, 0.0)
                valid = np.isfinite(pred) & np.isfinite(rv)
                if valid.sum() < 20:
                    continue
                yt = (rv[valid] >= threshold).astype(int)
                ys = pred[valid]
                if yt.sum() < 5 or (len(yt) - yt.sum()) < 5:
                    continue
                try:
                    prec, rec, _ = precision_recall_curve(yt, ys)
                    ap = average_precision_score(yt, ys)
                    ax.plot(rec, prec, color=mcol, lw=1.2, ls=ls, alpha=0.85,
                            label=f"{mname}-{model_tag} (AP={ap:.2f})")
                except Exception:
                    pass

        # Baseline: fraction of positives
        for acc, dom_arr in [(acc_a, domain_a_arr), (acc_b, domain_b_arr)]:
            m = dom_arr == dom
            if m.sum() > 0:
                rv = acc["i_theta"][m]
                frac = float((rv >= threshold).mean())
                ax.axhline(frac, color="grey", lw=0.7, ls=":", alpha=0.5)
                break

        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.set_xlabel("Recall", fontsize=FS_TICK)
        ax.set_ylabel("Precision", fontsize=FS_TICK)
        ax.tick_params(labelsize=FS_TICK - 1)
        ax.legend(fontsize=FS_TICK - 2, loc="lower left", framealpha=0.7)

    # Hide unused subplots
    for idx in range(n_dom, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(f"Precision-Recall Curves (γ = {threshold})", fontsize=FS_TITLE)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _savefig(fig, out_dir, "pr_curves")


def plot_calibration(acc_a: dict, acc_b: dict, out_dir: Path,
                     n_bins: int = 10) -> None:
    """Reliability diagram comparing all three estimators per model.

    Bins by ground-truth I_theta (shared x-axis) and plots mean predicted
    value per bin for AABB, MFMC, and Surrogate against the diagonal.
    A histogram of I_theta sample counts is shown below each panel.

    Binning by i_theta keeps the x-axis consistent across all methods so
    all three lines are directly comparable on the same axes.

    Colors
    ------
    Surrogate : primblue  #3B82F6
    AABB      : C_AABB  (from palette)
    MFMC      : C_MFMC  (from palette)
    Diagonal  : neutral gray  #9CA3AF
    Histogram : consteal  #0D9488
    """
    print("\n[Calibration] Reliability diagram (all estimators)...")

    C_HIST = "#0D9488"   # consteal
    C_DIAG = "#9CA3AF"   # neutral gray

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ctrs = (bins[:-1] + bins[1:]) / 2.0

    # Method configs: (label, acc_key, color, marker)
    method_cfg = [
        ("AABB",      "i_aabb", C_AABB, "s"),
        ("MFMC",      "i_mfmc", C_MFMC, "^"),
        ("Surrogate", "i_surr", "#3B82F6", "o"),
    ]

    def _cal_stats(acc, pred_key):
        """Bin by i_theta; return (mean_pred_per_bin, bin_counts)."""
        ref  = acc["i_theta"]
        pred = acc[pred_key].copy()
        # NaN in MFMC = no intersection → predict 0
        if pred_key == "i_mfmc":
            pred = np.where(np.isfinite(pred), pred, 0.0)
        valid = np.isfinite(pred) & np.isfinite(ref)
        pred, ref = pred[valid], ref[valid]
        mean_pred, counts = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (ref >= lo) & (ref < hi)
            counts.append(int(m.sum()))
            mean_pred.append(float(pred[m].mean()) if m.sum() >= 5 else np.nan)
        return np.array(mean_pred), np.array(counts, dtype=float)

    def _ece(mean_pred, counts):
        valid = np.isfinite(mean_pred) & (counts > 0)
        if not valid.any():
            return np.nan
        w = counts[valid] / counts[valid].sum()
        return float((w * np.abs(mean_pred[valid] - ctrs[valid])).sum())

    configs = [(acc_a, LABEL_A), (acc_b, LABEL_B)]

    fig = plt.figure(figsize=(7.16, 4.4))
    gs_outer = fig.add_gridspec(1, 2, wspace=0.30)

    for col, (acc, model_lbl) in enumerate(configs):
        if acc["n"] == 0:
            continue

        # Compute i_theta histogram (same for all methods)
        ref_valid = acc["i_theta"][np.isfinite(acc["i_theta"])]
        counts_ref = np.array([
            int(((ref_valid >= lo) & (ref_valid < hi)).sum())
            for lo, hi in zip(bins[:-1], bins[1:])
        ], dtype=float)

        gs_inner = gs_outer[col].subgridspec(
            2, 1, height_ratios=[0.70, 0.30], hspace=0.08
        )
        ax_cal  = fig.add_subplot(gs_inner[0])
        ax_hist = fig.add_subplot(gs_inner[1], sharex=ax_cal)

        # ── Calibration curves ────────────────────────────────────────────────
        ax_cal.plot([0, 1], [0, 1], color=C_DIAG, lw=1.2, ls="--",
                    label="Perfect", zorder=1)

        for mname, pred_key, mcol, mk in method_cfg:
            mean_pred, _ = _cal_stats(acc, pred_key)
            valid = np.isfinite(mean_pred)
            ece = _ece(mean_pred, counts_ref)
            lbl = f"{mname} (ECE={ece:.3f})"
            ax_cal.plot(ctrs[valid], mean_pred[valid],
                        color=mcol, lw=1.6, marker=mk, ms=3.5,
                        label=lbl, zorder=2)

        ax_cal.set_xlim(0, 1); ax_cal.set_ylim(0, 1)
        ax_cal.set_ylabel("Mean predicted $\\hat{I}(\\theta)$", fontsize=FS_LABEL)
        ax_cal.set_title(model_lbl, fontsize=FS_TITLE)
        ax_cal.legend(fontsize=FS_LEGEND - 0.5, loc="upper left")
        ax_cal.yaxis.grid(True, lw=LW_GRID, alpha=0.4)
        ax_cal.set_axisbelow(True)
        plt.setp(ax_cal.get_xticklabels(), visible=False)

        # ── I_theta distribution histogram ────────────────────────────────────
        bar_w = (bins[1] - bins[0]) * 0.85
        ax_hist.bar(ctrs, counts_ref, width=bar_w, color=C_HIST, alpha=0.80)
        ax_hist.set_xlabel("Ground-truth $I(\\theta)$ bin", fontsize=FS_LABEL)
        ax_hist.set_ylabel("Count", fontsize=FS_TICK)
        ax_hist.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x)))
        )
        ax_hist.yaxis.grid(True, lw=LW_GRID, alpha=0.4)
        ax_hist.set_axisbelow(True)
        ax_hist.tick_params(axis="both", labelsize=FS_TICK - 1)

    fig.suptitle("Estimator Calibration (Reliability Diagram)",
                 fontsize=FS_TITLE, y=1.01)
    _savefig(fig, out_dir, "calibration")


def plot_separated_error_by_regime(acc_a: dict, acc_b: dict,
                                    out_dir: Path) -> None:
    """MAE vs I_theta level — left panel = model A, right panel = model B.

    Shared y-axis scale to allow direct visual comparison.
    """
    print("\n[Separated] Error by regime...")

    n_bins = 20
    bins   = np.linspace(0, 1, n_bins + 1)
    ctrs   = (bins[:-1] + bins[1:]) / 2

    def _bin_stats(acc):
        ref  = acc["i_theta"]
        means_dict, stds_dict = {}, {}
        for name, arr in [("AABB", acc["i_aabb"]),
                           ("MFMC", acc["i_mfmc"]),
                           ("Surr", acc["i_surr"])]:
            err = np.abs(arr - ref)
            means, stds = [], []
            for i in range(n_bins):
                m = (ref >= bins[i]) & (ref < bins[i + 1])
                if m.sum() >= 10:
                    means.append(float(np.mean(err[m])))
                    stds.append(float(np.std(err[m])))
                else:
                    means.append(np.nan); stds.append(np.nan)
            means_dict[name] = np.array(means)
            stds_dict[name]  = np.array(stds)
        return means_dict, stds_dict

    stats_a = _bin_stats(acc_a)
    stats_b = _bin_stats(acc_b)

    # Shared y-axis: find global max for consistent comparison
    all_means = [v for d in [stats_a[0], stats_b[0]] for v in d.values()
                 for x in [v] if np.isfinite(x).any()]
    ymax = max(
        float(np.nanmax(np.concatenate([v for v in stats_a[0].values()]))),
        float(np.nanmax(np.concatenate([v for v in stats_b[0].values()]))),
    ) * 1.25

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.8),
                              sharey=True, sharex=True)
    method_style = [
        ("AABB", C_AABB, "o"),
        ("MFMC", C_MFMC, "s"),
        ("Surr", C_SURR, "^"),
    ]

    for ax, (means, stds), model_lbl in [
        (axes[0], stats_a, LABEL_A),
        (axes[1], stats_b, LABEL_B),
    ]:
        for name, col, mk in method_style:
            m_vals = means[name]; s_vals = stds[name]
            valid  = ~np.isnan(m_vals)
            ax.fill_between(ctrs[valid],
                            m_vals[valid] - s_vals[valid],
                            m_vals[valid] + s_vals[valid],
                            color=col, alpha=0.10)
            ax.plot(ctrs[valid], m_vals[valid], color=col, lw=LW_MAIN,
                    marker=mk, markersize=4, label=name, zorder=3)

        ax.set_xlabel(r"$I_\theta$ (reference)", fontsize=FS_LABEL)
        ax.set_ylabel(r"MAE $= |\hat{I} - I_\theta|$", fontsize=FS_LABEL)
        ax.set_title(f"MAE vs inconsistency level\n({model_lbl})",
                     fontsize=FS_TITLE)
        ax.set_ylim(0, ymax)
        _style_ax(ax)

    fig.tight_layout()
    _legend_below(fig, axes[0])
    _savefig(fig, out_dir, "error_by_regime")


# =============================================================================
# Table generators
# =============================================================================

def _fmt(v, decimals=3):
    """Format a float, or '--' if NaN."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    return f"{v:.{decimals}f}"


def _fmt_ci(val, lo, hi, dec):
    """Format a value with a ±half-width CI as a LaTeX string.

    Returns ``0.932{\\tiny$\\pm$0.011}`` where 0.011 = max(val-lo, hi-val).
    Falls back to ``_fmt(val, dec)`` if lo or hi is NaN.
    """
    if (lo is None or hi is None
            or (isinstance(lo, float) and np.isnan(lo))
            or (isinstance(hi, float) and np.isnan(hi))):
        return _fmt(val, dec)
    half = max(val - lo, hi - val)
    return rf"{val:.{dec}f}{{\tiny$\pm${half:.{dec}f}}}"


def table_acc_by_dim(acc_a: dict, acc_b: dict, out_dir: Path) -> None:
    """LaTeX + CSV table: accuracy metrics by zonotope dimension.

    Single-column layout: rows = Dim (group) × Method, cols = N | ρ | R² | MAE.
    """
    print("\n[Table] Accuracy by dimension...")
    out_dir.mkdir(parents=True, exist_ok=True)

    def _bootstrap_ci(ref, pred, sc_ids, n_boot=500):
        """Bootstrap 95% CI on ρ, R², MAE by resampling Saltelli samples.

        Resamples individual samples with replacement for all dimensions.
        For 2D this spans multiple scenarios; for 3D/4D it is within a
        single validation scenario.
        """
        nan_result = {
            "rho_lo": np.nan, "rho_hi": np.nan,
            "r2_lo":  np.nan, "r2_hi":  np.nan,
            "mae_lo": np.nan, "mae_hi": np.nan,
        }
        n = len(ref)
        if n < 10:
            return nan_result
        rng = np.random.default_rng(42)
        boot_rho, boot_r2, boot_mae = [], [], []
        idx = np.arange(n)
        for _ in range(n_boot):
            chosen = rng.choice(idx, size=n, replace=True)
            m = _metrics(pred[chosen], ref[chosen])
            boot_rho.append(m["rho"]); boot_r2.append(m["r2"]); boot_mae.append(m["mae"])
        if len(boot_rho) < 5:
            return nan_result
        boot_rho = np.array(boot_rho)
        boot_r2  = np.array(boot_r2)
        boot_mae = np.array(boot_mae)
        return {
            "rho_lo": float(np.nanpercentile(boot_rho, 2.5)),
            "rho_hi": float(np.nanpercentile(boot_rho, 97.5)),
            "r2_lo":  float(np.nanpercentile(boot_r2,  2.5)),
            "r2_hi":  float(np.nanpercentile(boot_r2,  97.5)),
            "mae_lo": float(np.nanpercentile(boot_mae, 2.5)),
            "mae_hi": float(np.nanpercentile(boot_mae, 97.5)),
        }

    def _ece(ref, pred, n_bins=10):
        """Expected Calibration Error: Σ (|bin|/N) × |mean_pred - mean_actual|."""
        pred = pred.copy()
        # NaN in MFMC = no intersection → predict 0
        pred = np.where(np.isfinite(pred), pred, 0.0)
        valid = np.isfinite(ref)
        ref, pred = ref[valid], pred[valid]
        if len(ref) == 0:
            return np.nan
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece_sum = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (ref >= lo) & (ref < hi)
            if m.sum() < 5:
                continue
            ece_sum += (m.sum() / len(ref)) * abs(pred[m].mean() - ref[m].mean())
        return float(ece_sum)

    def _dim_rows(acc):
        dims_all = acc["dim"]
        sc_ids   = acc["scenario_id"]
        ref, aabb, mfmc, surr = (acc[k] for k in
                                  ("i_theta", "i_aabb", "i_mfmc", "i_surr"))
        rows = []
        for d in sorted(np.unique(dims_all)):
            m = dims_all == d
            row = {"dim": d, "N": int(m.sum())}
            for k, v in [("AABB", aabb[m]), ("MFMC", mfmc[m]), ("Surr", surr[m])]:
                row[f"rho_{k}"] = _metrics(v, ref[m])["rho"]
                row[f"r2_{k}"]  = _metrics(v, ref[m])["r2"]
                row[f"mae_{k}"] = _metrics(v, ref[m])["mae"]
                ci = _bootstrap_ci(ref[m], v, sc_ids[m])
                row[f"rho_lo_{k}"] = ci["rho_lo"]; row[f"rho_hi_{k}"] = ci["rho_hi"]
                row[f"r2_lo_{k}"]  = ci["r2_lo"];  row[f"r2_hi_{k}"]  = ci["r2_hi"]
                row[f"mae_lo_{k}"] = ci["mae_lo"]; row[f"mae_hi_{k}"] = ci["mae_hi"]
            rows.append(row)
        return rows

    rows_a = _dim_rows(acc_a)
    rows_b = _dim_rows(acc_b)
    all_rows = sorted(rows_a + rows_b, key=lambda r: r["dim"])

    methods = [("AABB", "AABB"), ("MFMC", "MFMC"), ("Surr", "Surr.")]

    # ── LaTeX ─────────────────────────────────────────────────────────────────
    # 5 columns: Dim | Method | ρ | R² | MAE
    col_spec = "l l r r r"
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Accuracy metrics by zonotope dimension. "
         r"95\% bootstrap CIs computed by resampling Saltelli samples.}",
        r"\label{tab:acc_by_dim}",
        r"\footnotesize",
        rf"\begin{{tabular*}}{{\columnwidth}}{{@{{\extracolsep{{\fill}}}} {col_spec}}}",
        r"\toprule",
        r"Dim & Method & $\rho$ & $R^2$ & MAE \\",
        r"\midrule",
    ]

    for r in all_rows:
        dim_lbl = f"{r['dim']}D"
        n_methods = len(methods)
        for i, (key, lbl) in enumerate(methods):
            dim_cell = rf"\multirow{{{n_methods}}}{{*}}{{{dim_lbl}}}" if i == 0 else ""
            cells = [dim_cell, lbl,
                     _fmt_ci(r.get(f"rho_{key}"), r.get(f"rho_lo_{key}"),
                              r.get(f"rho_hi_{key}"), 3),
                     _fmt_ci(r.get(f"r2_{key}"),  r.get(f"r2_lo_{key}"),
                              r.get(f"r2_hi_{key}"),  3),
                     _fmt_ci(r.get(f"mae_{key}"), r.get(f"mae_lo_{key}"),
                              r.get(f"mae_hi_{key}"), 4)]
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\midrule")

    # Remove last \midrule and replace with \bottomrule
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]

    tex_path = out_dir / "table_acc_by_dim.tex"
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved: {tex_path.name}")

    # ── CSV ───────────────────────────────────────────────────────────────────
    import csv
    csv_path = out_dir / "table_acc_by_dim.csv"
    header = ["Dim", "Method", "rho", "R2", "MAE"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in all_rows:
            for key, lbl in methods:
                w.writerow([f"{r['dim']}D", lbl,
                             _fmt(r.get(f"rho_{key}"), 3),
                             _fmt(r.get(f"r2_{key}"),  3),
                             _fmt(r.get(f"mae_{key}"), 4)])
    print(f"  Saved: {csv_path.name}")


def table_threshold_by_domain(acc_a: dict, domain_a: np.ndarray,
                               acc_b: dict, domain_b: np.ndarray,
                               out_dir: Path,
                               threshold: float = 0.5,
                               domain_short_a: dict = None,
                               domain_short_b: dict = None,
                               domain_order_a: list = None,
                               domain_order_b: list = None) -> None:
    """LaTeX + CSV table: FPR / FNR by domain.

    Rows = domains.  Columns = Method × Model × Rate (FPR / FNR).
    """
    print(f"\n[Table] Threshold by domain (gamma={threshold})...")
    out_dir.mkdir(parents=True, exist_ok=True)

    def _domain_rows(acc, domain, short, order):
        if len(domain) == 0:
            return []
        ref, aabb, mfmc, surr = (acc[k] for k in
                                  ("i_theta", "i_aabb", "i_mfmc", "i_surr"))
        short   = short or {}
        present = set(domain)
        unique  = [d for d in (order or []) if d in present]
        if not unique:
            unique = sorted(present)
        rows = []
        for d in unique:
            m = domain == d
            if m.sum() < 10:
                continue
            rv  = ref[m]
            row = {"domain": d, "label": short.get(d, d)}
            for key, pred in [("AABB", aabb[m]), ("MFMC", mfmc[m]),
                               ("Surr", surr[m])]:
                valid = np.isfinite(pred) & np.isfinite(rv)
                xv, rv2 = pred[valid], rv[valid]
                pp = xv >= threshold
                ap = rv2 >= threshold; an = rv2 < threshold
                n_pos = ap.sum(); n_neg = an.sum()
                row[f"fpr_{key}"] = float((pp & an).sum() / n_neg) if n_neg else np.nan
                row[f"fnr_{key}"] = float((~pp & ap).sum() / n_pos) if n_pos else np.nan
            rows.append(row)
        return rows

    rows_a = _domain_rows(acc_a, domain_a, domain_short_a, domain_order_a)
    rows_b = _domain_rows(acc_b, domain_b, domain_short_b, domain_order_b)

    doms_a   = {r["domain"] for r in rows_a}
    doms_b   = {r["domain"] for r in rows_b}
    all_doms = list(dict.fromkeys(
        [r["domain"] for r in rows_a] +
        [r["domain"] for r in rows_b if r["domain"] not in doms_a]
    ))
    short_all = {**(domain_short_a or {}), **(domain_short_b or {})}

    def _lu(rows, dom):
        for r in rows:
            if r["domain"] == dom:
                return r
        return None

    methods = [("AABB", "AABB"), ("MFMC", "MFMC"), ("Surr", "Surr.")]

    # ── LaTeX ─────────────────────────────────────────────────────────────────
    # Columns: Domain | FPR | FNR (per method) — 1 + 3*2 = 7 cols total
    # Model-B rows are italicised; no separate model column needed.
    col_spec = "l " + " ".join(["r r"] * len(methods))
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{FPR and FNR at $\gamma = {threshold}$ by domain and method."
        r"  \textit{Italic rows} = 4D model; roman rows = 2D/3D model.}",
        r"\label{tab:threshold_by_domain}",
        r"\footnotesize",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]
    # Header row 1 — method groups (each spans 2 cols)
    meth_hdr = " & ".join(
        [rf"\multicolumn{{2}}{{c}}{{{lbl}}}" for _, lbl in methods]
    )
    lines.append(r"Domain & " + meth_hdr + r" \\")
    cmidrule = " ".join(
        [rf"\cmidrule(lr){{{2 + i*2}-{3 + i*2}}}" for i in range(len(methods))]
    )
    lines.append(cmidrule)
    sub_hdr = " & ".join(["FPR & FNR"] * len(methods))
    lines.append(r" & " + sub_hdr + r" \\")
    lines.append(r"\midrule")

    for dom in all_doms:
        ra = _lu(rows_a, dom)
        rb = _lu(rows_b, dom)
        rd = ra if ra is not None else rb   # one is always set
        is_b = (ra is None)
        label = short_all.get(dom, dom)
        cells = [label]
        for key, _ in methods:
            cells.append(_fmt(rd[f"fpr_{key}"] if rd else np.nan, 3))
            cells.append(_fmt(rd[f"fnr_{key}"] if rd else np.nan, 3))
        row_str = " & ".join(cells) + r" \\"
        if is_b:
            row_str = r"\textit{" + row_str.replace(r" \\", r"} \\")
        lines.append(row_str)

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    tex_path = out_dir / "table_threshold_by_domain.tex"
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved: {tex_path.name}")

    # ── CSV ───────────────────────────────────────────────────────────────────
    import csv
    csv_path = out_dir / "table_threshold_by_domain.csv"
    header = ["Domain", "Model"] + [
        f"{key}_{rate}" for key, _ in methods for rate in ("FPR", "FNR")
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for dom in all_doms:
            ra = _lu(rows_a, dom)
            rb = _lu(rows_b, dom)
            rd = ra if ra is not None else rb
            model_lbl = LABEL_B if ra is None else LABEL_A
            row = [short_all.get(dom, dom), model_lbl]
            for key, _ in methods:
                row.append(_fmt(rd[f"fpr_{key}"] if rd else np.nan, 3))
                row.append(_fmt(rd[f"fnr_{key}"] if rd else np.nan, 3))
            w.writerow(row)
    print(f"  Saved: {csv_path.name}")


# =============================================================================
# Data loading helpers
# =============================================================================

def _build_file_filter(data_dirs, scenario_ids):
    """Return list of Path objects for the given scenario IDs."""
    import re
    sc_ids = set(scenario_ids)
    files, seen = [], set()
    for d in data_dirs:
        for jf in sorted(Path(d).glob("results_scenario_*.json")):
            m = re.search(r"results_scenario_(\d+)\.json", jf.name)
            if m:
                sc_id = int(m.group(1))
                if sc_id in sc_ids and sc_id not in seen:
                    files.append(jf)
                    seen.add(sc_id)
    return files


def _load_eval_data(path: Path) -> dict:
    """Load pre-computed eval data exported by run_analysis.py --export-data.

    Returns the same structure as _load_all().
    Domain names are normalised to human-readable labels via domain_utils so
    that exports created before domain_utils was wired in still display correctly.
    """
    import json as _json
    path = Path(path)
    arr  = np.load(path / "acc.npz")
    meta = _json.loads((path / "meta.json").read_text(encoding="utf-8"))

    acc = {
        "i_surr":      arr["i_surr"],
        "i_theta":     arr["i_theta"],
        "i_aabb":      arr["i_aabb"],
        "i_mfmc":      arr["i_mfmc"],
        "dim":         arr["dim"],
        "scenario_id": arr["scenario_id"],
        "t_surr_total": float(meta["t_surr_total"]),
        "n":            int(meta["n"]),
    }

    domain_list  = meta.get("domain", [])
    domain_short = meta.get("domain_short", {})
    domain_order = meta.get("domain_order", [])
    domain = np.array(domain_list, dtype=object) if domain_list else np.array([])

    # ── Normalise raw filename stems → human-readable labels ──────────────────
    # Exports created before domain_utils was wired in may store raw stems
    # such as "building_hvac_full" or "building_hvac_full.json" instead of
    # "Building HVAC".  The mapping is inlined here so no import is needed.
    _SRC2DOM = {
        "automotive_full":             "Automotive",
        "building_hvac_full":          "Building HVAC",
        "industrial_robot_full":       "Industrial Robot",
        "medical_device_full":         "Medical Device",
        "railway_full":                "Railway",
        "satellite_aerospace_full":    "Satellite/Aerospace",
        "smart_grid_full":             "Smart Grid",
        "water_chemical_process_full": "Water/Chemical",
        "wind_turbine_full":           "Wind Turbine",
    }
    _DOM_SHORT = {
        "Automotive":          "Auto",
        "Building HVAC":       "HVAC",
        "Industrial Robot":    "Robot",
        "Medical Device":      "Medical",
        "Railway":             "Rail",
        "Satellite/Aerospace": "Space",
        "Smart Grid":          "Grid",
        "Water/Chemical":      "Water",
        "Wind Turbine":        "Wind",
        "CAD Export Drift":    "CAD Drift",
        "MBSE Mismatch":       "MBSE",
        "Doc. Sync":           "Doc",
        "Ctrl. Conflict":      "Ctrl",
        "Sensor Drift":        "Sensor",
        "Req. Ambiguity":      "Req.",
        "Config Mismatch":     "Config",
        "Sim. Error":          "Sim",
        "Phys. Coupling":      "Physics",
        "Spec. Gap":           "Spec",
        "Param. Bias":         "Param",
        "Traceability":        "Trace",
        "Unknown":             "?",
    }
    _DEFAULT_ORDER = [
        "Automotive", "Building HVAC", "Industrial Robot", "Medical Device",
        "Railway", "Satellite/Aerospace", "Smart Grid", "Water/Chemical",
        "Wind Turbine",
    ]

    def _norm_domain(raw: str) -> str:
        if raw in _DOM_SHORT:           # already a valid human-readable label
            return raw
        # strip .json suffix, then match against known stems
        stem = raw.removesuffix(".json").removesuffix(".JSON").lower()
        for key, label in _SRC2DOM.items():
            if stem == key or stem.startswith(key):
                return label
        return raw                      # unknown — keep as-is

    if len(domain):
        domain = np.array([_norm_domain(d) for d in domain], dtype=object)

    # Always rebuild short-name and order from the inlined tables
    present      = set(domain.tolist()) if len(domain) else set()
    domain_short = {d: _DOM_SHORT.get(d, d) for d in present}
    if not domain_order:
        domain_order = _DEFAULT_ORDER

    print(f"  [precomputed] Loaded {acc['n']:,} samples from {path}/")
    return dict(
        acc          = acc,
        timing       = meta.get("timing", {}),
        domain       = domain,
        domain_short = domain_short,
        domain_order = domain_order,
        sobol_results= meta.get("sobol_results", []),
    )


def _load_all(model_path, data_dirs, scenario_ids, dims_filter,
              max_samples, max_scenarios, device):
    """Load model, accuracy data, timing, Sobol scenarios, and compute Sobol."""
    print(f"\n{'─'*60}")
    print(f"  Loading model: {model_path}")
    model, trained_dims = _load_model(device, model_path=str(model_path))
    if dims_filter is None:
        dims_filter = trained_dims
    if dims_filter is not None:
        print(f"  Dims filter: {dims_filter}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {type(model).__name__}  {n_params} parameters")

    files = _build_file_filter(data_dirs, scenario_ids) if scenario_ids else None
    print(f"  Files: {len(files) if files else 'all'} scenario(s)")

    print("  Loading accuracy data...")
    acc = _load_accuracy_data(data_dirs, model, device,
                              max_samples=max_samples,
                              dims_filter=dims_filter,
                              files=files)
    print(f"  n={acc['n']:,}")

    print("  Loading timing + Sobol parameters...")
    ts = _load_timing_and_sobol(data_dirs, acc,
                                max_scenarios=max_scenarios,
                                files=files)
    timing = {
        "us_aabb": ts.get("us_aabb", 0.0),
        "us_mfmc": ts.get("us_mfmc", 0.0),
        "us_mc":   ts.get("us_mc",   0.0),
        "us_surr": acc["t_surr_total"] / acc["n"] * 1e6,
    }

    sobol_scenarios = ts["sobol_scenarios"]
    sobol_results   = []
    if sobol_scenarios:
        print("  Computing Sobol indices (may take a minute)...")
        sobol_results = _sobol_per_scenario(sobol_scenarios)

    return dict(
        acc=acc,
        timing=timing,
        domain=ts["domain"],
        domain_short=ts["_domain_short"],
        domain_order=ts["_domain_order"],
        sobol_results=sobol_results,
    )


# =============================================================================
# Paper statistics export
# =============================================================================

def _save_paper_stats(da: dict, db: dict, out_dir: Path,
                      threshold: float = 0.5) -> None:
    """Collect all plot-level numbers and write paper_stats.json.

    Structure
    ---------
    models            — sample counts and dims for each model
    overall_accuracy  — rho / R² / MAE per method per model (all samples)
    efficiency        — mean inference time (µs/sample) per method per model
    accuracy_by_dim   — rho / R² / MAE per dim per method (model that covers it)
    accuracy_by_domain— rho per domain per method per model
    threshold_by_domain — FPR / FNR at `threshold` per domain per method per model
    sobol_overall     — mean S1 & ST per parameter per index type per model
    sobol_by_domain   — mean S1 & ST per domain per parameter per index type
    consistency_cdf   — P(I ≤ γ) at γ ∈ {0.1,0.25,0.5,0.75,0.9} per method/model
    error_by_regime   — mean MAE in I_theta bins per method per model
    """
    import json as _json
    import datetime

    def _r(v):
        """Round to 6 sig figs for compact JSON."""
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return None
        return round(float(v), 6)

    def _met(pred, ref):
        m = _metrics(pred, ref)
        return {"rho": _r(m["rho"]), "r2": _r(m["r2"]), "mae": _r(m["mae"])}

    stats = {
        "_generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "_threshold": threshold,
    }

    # ── Model metadata ─────────────────────────────────────────────────────────
    stats["models"] = {
        "2d3d": {
            "n_samples": int(da["acc"]["n"]),
            "dims": sorted(int(d) for d in np.unique(da["acc"]["dim"]).tolist()),
        },
        "4d": {
            "n_samples": int(db["acc"]["n"]),
            "dims": sorted(int(d) for d in np.unique(db["acc"]["dim"]).tolist()),
        },
    }

    # ── Overall accuracy ───────────────────────────────────────────────────────
    def _overall(acc):
        ref = acc["i_theta"]
        return {
            "AABB":      _met(acc["i_aabb"], ref),
            "MFMC":      _met(acc["i_mfmc"], ref),
            "Surrogate": _met(acc["i_surr"],  ref),
        }

    stats["overall_accuracy"] = {
        "2d3d": _overall(da["acc"]),
        "4d":   _overall(db["acc"]),
    }

    # ── Efficiency (µs / sample) ───────────────────────────────────────────────
    def _timing_dict(acc, timing):
        surr_us = acc["t_surr_total"] / acc["n"] * 1e6
        return {
            "AABB_us":      _r(timing.get("us_aabb")),
            "MFMC_us":      _r(timing.get("us_mfmc")),
            "MC_us":        _r(timing.get("us_mc")),
            "Surrogate_us": _r(surr_us),
        }

    stats["efficiency"] = {
        "2d3d": _timing_dict(da["acc"], da["timing"]),
        "4d":   _timing_dict(db["acc"], db["timing"]),
    }

    # ── Accuracy by dimension ──────────────────────────────────────────────────
    by_dim = {}
    for model_lbl, acc in [("2d3d", da["acc"]), ("4d", db["acc"])]:
        ref = acc["i_theta"]
        for d in sorted(np.unique(acc["dim"]).tolist()):
            m = acc["dim"] == d
            key = str(int(d))
            by_dim.setdefault(key, {})[model_lbl] = {
                "n": int(m.sum()),
                "AABB":      _met(acc["i_aabb"][m], ref[m]),
                "MFMC":      _met(acc["i_mfmc"][m], ref[m]),
                "Surrogate": _met(acc["i_surr"][m],  ref[m]),
            }
    stats["accuracy_by_dim"] = by_dim

    # ── Accuracy by domain ─────────────────────────────────────────────────────
    by_dom = {}
    for model_lbl, acc, domain in [("2d3d", da["acc"], da["domain"]),
                                    ("4d",   db["acc"], db["domain"])]:
        if not len(domain):
            continue
        ref = acc["i_theta"]
        for d in np.unique(domain):
            m = domain == d
            if m.sum() < 5:
                continue
            by_dom.setdefault(str(d), {})[model_lbl] = {
                "n": int(m.sum()),
                "AABB":      _met(acc["i_aabb"][m], ref[m]),
                "MFMC":      _met(acc["i_mfmc"][m], ref[m]),
                "Surrogate": _met(acc["i_surr"][m],  ref[m]),
            }
    stats["accuracy_by_domain"] = by_dom

    # ── Threshold metrics by domain ────────────────────────────────────────────
    thr_dom = {}
    for model_lbl, acc, domain in [("2d3d", da["acc"], da["domain"]),
                                    ("4d",   db["acc"], db["domain"])]:
        if not len(domain):
            continue
        ref = acc["i_theta"]
        for d in np.unique(domain):
            m = domain == d
            if m.sum() < 10:
                continue
            rv = ref[m]
            dom_key = str(d)
            thr_dom.setdefault(dom_key, {})[model_lbl] = {}
            for key, pred in [("AABB",  acc["i_aabb"][m]),
                               ("MFMC",  acc["i_mfmc"][m]),
                               ("Surrogate", acc["i_surr"][m])]:
                valid = np.isfinite(pred) & np.isfinite(rv)
                xv, rv2 = pred[valid], rv[valid]
                pp = xv >= threshold
                ap = rv2 >= threshold
                an = rv2 < threshold
                n_pos, n_neg = int(ap.sum()), int(an.sum())
                fpr = float((pp & an).sum() / n_neg) if n_neg else None
                fnr = float((~pp & ap).sum() / n_pos) if n_pos else None
                thr_dom[dom_key][model_lbl][key] = {
                    "FPR": _r(fpr), "FNR": _r(fnr),
                    "n_pos": n_pos, "n_neg": n_neg,
                }
    stats["threshold_by_domain"] = thr_dom

    # ── Sobol indices ──────────────────────────────────────────────────────────
    def _sobol_summary(sobol_results):
        if not sobol_results:
            return None
        out = {}
        for index_key in ("S1_mc", "S1_surr", "ST_mc", "ST_surr"):
            out[index_key] = {
                p: _r(float(np.mean([r[index_key][p] for r in sobol_results
                                     if index_key in r])))
                for p in PARAMS
            }
        return out

    def _sobol_by_domain(sobol_results):
        if not sobol_results:
            return {}
        by_dom = defaultdict(list)
        for r in sobol_results:
            by_dom[r.get("domain", "Unknown")].append(r)
        out = {}
        for dom, rlist in by_dom.items():
            out[str(dom)] = {}
            for index_key in ("S1_mc", "S1_surr", "ST_mc", "ST_surr"):
                valid = [r for r in rlist if index_key in r]
                if not valid:
                    continue
                out[str(dom)][index_key] = {
                    p: _r(float(np.mean([r[index_key][p] for r in valid])))
                    for p in PARAMS
                }
        return out

    stats["sobol_overall"] = {
        "2d3d": _sobol_summary(da["sobol_results"]),
        "4d":   _sobol_summary(db["sobol_results"]),
        "param_labels": {p: PARAM_LABELS[p] for p in PARAMS},
    }
    stats["sobol_by_domain"] = {
        "2d3d": _sobol_by_domain(da["sobol_results"]),
        "4d":   _sobol_by_domain(db["sobol_results"]),
    }

    # ── Consistency CDF at key thresholds ─────────────────────────────────────
    gammas = [0.10, 0.25, 0.50, 0.75, 0.90]
    cdf_stats = {}
    for model_lbl, acc in [("2d3d", da["acc"]), ("4d", db["acc"])]:
        cdf_stats[model_lbl] = {}
        for method, arr in [("MC",        acc["i_theta"]),
                             ("AABB",      acc["i_aabb"]),
                             ("MFMC",      acc["i_mfmc"]),
                             ("Surrogate", acc["i_surr"])]:
            valid = arr[np.isfinite(arr)]
            cdf_stats[model_lbl][method] = {
                f"P_leq_{g}".replace(".", "_"): _r(float((valid <= g).mean()))
                for g in gammas
            }
    stats["consistency_cdf"] = cdf_stats

    # ── Error by I_theta regime ────────────────────────────────────────────────
    n_bins = 10
    bins   = np.linspace(0, 1, n_bins + 1)
    ctrs   = ((bins[:-1] + bins[1:]) / 2).tolist()
    regime_stats = {}
    for model_lbl, acc in [("2d3d", da["acc"]), ("4d", db["acc"])]:
        ref = acc["i_theta"]
        regime_stats[model_lbl] = {"bin_centers": [_r(c) for c in ctrs]}
        for method, pred in [("AABB",      acc["i_aabb"]),
                              ("MFMC",      acc["i_mfmc"]),
                              ("Surrogate", acc["i_surr"])]:
            err = np.abs(pred - ref)
            mae_bins, std_bins = [], []
            for i in range(n_bins):
                m = (ref >= bins[i]) & (ref < bins[i + 1]) & np.isfinite(err)
                mae_bins.append(_r(float(np.mean(err[m]))) if m.sum() >= 5 else None)
                std_bins.append(_r(float(np.std(err[m])))  if m.sum() >= 5 else None)
            regime_stats[model_lbl][method] = {
                "mae_per_bin": mae_bins, "std_per_bin": std_bins,
            }
    stats["error_by_regime"] = regime_stats

    # ── Write ──────────────────────────────────────────────────────────────────
    out_path = out_dir / "paper_stats.json"
    out_path.write_text(
        _json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n  Saved paper stats → {out_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Combined paper figures for the 2D/3D and 4D surrogate models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Plot names (--plots):
  combined  : efficiency  pareto  acc_by_dim  accuracy_by_domain
              exploration_budget  sobol_by_domain  sensitivity_compare
              acc_scatter  consistency_rate
  separated : threshold_by_domain  error_by_regime
  shortcut  : all  (default)
""")

    # ── Pre-computed results (bypass model inference) ─────────────────────────
    p.add_argument("--precomputed-2d3d", default=None,
                   help="Directory produced by run_analysis.py --export-data for the "
                        "2D/3D model.  When given, skips model loading and inference.")
    p.add_argument("--precomputed-4d", default=None,
                   help="Directory produced by run_analysis.py --export-data for the "
                        "4D model.  When given, skips model loading and inference.")

    # ── Models (only used when --precomputed-* are NOT given) ─────────────────
    p.add_argument("--model-2d3d", default=
                   "product_transformer_2d3d/product_transformer_exact.pt",
                   help="Path to the 2D/3D model checkpoint.")
    p.add_argument("--model-4d", default=
                   "product_transformer_4d_real/product_transformer_exact.pt",
                   help="Path to the 4D model checkpoint.")

    # ── Data ──────────────────────────────────────────────────────────────────
    p.add_argument("--data-2d3d", nargs="+",
                   default=["data/measurements_v6", "data/measurements_cps_v6"],
                   help="Data directories for the 2D/3D model.")
    p.add_argument("--data-4d", nargs="+",
                   default=["data/measurements_v6", "data/measurements_cps_v6",
                            "data/synthetic_v6"],
                   help="Data directories for the 4D model. "
                        "Includes synthetic_v6 because the 4D model was not "
                        "trained on any synthetic data — those scenarios are "
                        "fully held-out evaluation data.")
    p.add_argument("--scenarios-2d3d", nargs="*", type=int, default=None,
                   help="Scenario IDs to load for the 2D/3D model. "
                        "Omit to use all available scenarios.")
    p.add_argument("--scenarios-4d", nargs="*", type=int, default=None,
                   help="Scenario IDs to load for the 4D model. "
                        "Omit to use all available scenarios.")

    # ── Dimension filters (auto-detected from run_config.json by default) ─────
    p.add_argument("--dims-2d3d", nargs="*", type=int, default=None,
                   help="Zonotope dims for model A (e.g. --dims-2d3d 2 3). "
                        "Defaults to run_config.json dims.")
    p.add_argument("--dims-4d", nargs="*", type=int, default=None,
                   help="Zonotope dims for model B (e.g. --dims-4d 4). "
                        "Defaults to run_config.json dims.")

    # ── Output ────────────────────────────────────────────────────────────────
    p.add_argument("--output", default="results/combined_figures",
                   help="Output directory.")

    # ── Plot selection ────────────────────────────────────────────────────────
    p.add_argument("--plots", nargs="*", default=None,
                   help="Which plots to generate.  Omit for all.")

    # ── Performance ───────────────────────────────────────────────────────────
    p.add_argument("--max-samples",   type=int, default=None)
    p.add_argument("--max-scenarios", type=int, default=None)
    p.add_argument("--gamma", type=float, default=0.5,
                   help="Inconsistency threshold for threshold_by_domain.")

    args = p.parse_args()

    _ALL_PLOTS = {
        "efficiency", "pareto", "acc_by_dim", "accuracy_by_domain",
        "exploration_budget", "sobol_by_domain", "sensitivity_compare",
        "acc_scatter", "consistency_rate",
        "threshold_by_domain", "error_by_regime",
        "auroc_ap_by_domain", "pr_curves", "calibration",
        "table_acc_by_dim", "table_threshold_by_domain",
    }
    requested = (set(args.plots) & _ALL_PLOTS) if args.plots else _ALL_PLOTS

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dirs_a   = [Path(d) for d in args.data_2d3d]
    data_dirs_b   = [Path(d) for d in args.data_4d]

    t_start = time.time()
    print("=" * 60)
    print("  Combined Figure Pipeline")
    print(f"  Output:  {out_root}")
    print(f"  Plots:   {sorted(requested)}")
    print("=" * 60)

    # ── Load both models (or pre-computed results) ────────────────────────────
    if args.precomputed_2d3d:
        print(f"\nLoading pre-computed 2D/3D results from {args.precomputed_2d3d}")
        da = _load_eval_data(Path(args.precomputed_2d3d))
    else:
        da = _load_all(
            model_path    = Path(args.model_2d3d),
            data_dirs     = data_dirs_a,
            scenario_ids  = args.scenarios_2d3d,
            dims_filter   = args.dims_2d3d,
            max_samples   = args.max_samples,
            max_scenarios = args.max_scenarios,
            device        = device,
        )

    if args.precomputed_4d:
        print(f"\nLoading pre-computed 4D results from {args.precomputed_4d}")
        db = _load_eval_data(Path(args.precomputed_4d))
    else:
        db = _load_all(
            model_path    = Path(args.model_4d),
            data_dirs     = data_dirs_b,
            scenario_ids  = args.scenarios_4d,
            dims_filter   = args.dims_4d,
            max_samples   = args.max_samples,
            max_scenarios = args.max_scenarios,
            device        = device,
        )

    def want(name):
        return name in requested

    # Shared remap: merge all 4D Engineering domain labels into "Engineering"
    _engineering_4d_remap = {
        "Phys. Coupling": "Engineering",
        "Spec. Gap":      "Engineering",
        "Param. Bias":    "Engineering",
        "Traceability":   "Engineering",
        "CAD Export Drift": "Engineering",
        "MBSE Mismatch":  "Engineering",
        "Doc. Sync":      "Engineering",
        "Ctrl. Conflict": "Engineering",
        "Sensor Drift":   "Engineering",
        "Req. Ambiguity": "Engineering",
        "Config Mismatch": "Engineering",
        "Sim. Error":     "Engineering",
    }

    # ── COMBINED plots ────────────────────────────────────────────────────────
    if want("efficiency"):
        plot_combined_efficiency(da["timing"], db["timing"], out_root)

    if want("pareto"):
        plot_combined_pareto(da["acc"], da["timing"],
                              db["acc"], db["timing"], out_root)

    if want("acc_by_dim"):
        plot_combined_acc_by_dim(da["acc"], db["acc"], out_root)

    if want("accuracy_by_domain"):
        plot_combined_accuracy_by_domain(
            da["acc"], da["domain"],
            db["acc"], db["domain"],
            out_root,
            domain_short_a=da["domain_short"],
            domain_short_b=db["domain_short"],
            domain_order_a=da["domain_order"],
            domain_order_b=db["domain_order"],
        )

    if want("exploration_budget"):
        plot_combined_exploration_budget(da["timing"], db["timing"], out_root)

    if want("sobol_by_domain"):
        plot_combined_sobol_by_domain(
            da["sobol_results"], db["sobol_results"],
            out_root,
            domain_short_a=da["domain_short"],
            domain_short_b=db["domain_short"],
            domain_order_a=da["domain_order"],
            domain_order_b=db["domain_order"],
            domain_remap_b=_engineering_4d_remap,
        )

    if want("sensitivity_compare"):
        plot_combined_sensitivity_compare(
            da["sobol_results"], db["sobol_results"], out_root)

    if want("acc_scatter"):
        plot_combined_acc_scatter(da["acc"], db["acc"], out_root)

    if want("consistency_rate"):
        plot_combined_consistency_rate(da["acc"], db["acc"], out_root)

    # ── SEPARATED plots ───────────────────────────────────────────────────────
    if want("threshold_by_domain"):
        plot_separated_threshold_by_domain(
            da["acc"], da["domain"],
            db["acc"], db["domain"],
            out_root, threshold=args.gamma,
            domain_short_a=da["domain_short"],
            domain_short_b=db["domain_short"],
            domain_order_a=da["domain_order"],
            domain_order_b=db["domain_order"],
        )

    if want("error_by_regime"):
        plot_separated_error_by_regime(da["acc"], db["acc"], out_root)

    if want("calibration"):
        plot_calibration(da["acc"], db["acc"], out_root)

    if want("auroc_ap_by_domain"):
        plot_auroc_ap_by_domain(
            da["acc"], da["domain"],
            db["acc"], db["domain"],
            out_root, threshold=args.gamma,
            domain_short_a=da["domain_short"],
            domain_short_b=db["domain_short"],
            domain_order_a=da["domain_order"],
            domain_order_b=db["domain_order"],
            domain_remap_b=_engineering_4d_remap,
        )

    if want("pr_curves"):
        plot_pr_curves(
            da["acc"], da["domain"],
            db["acc"], db["domain"],
            out_root, threshold=args.gamma,
            domain_short_a=da["domain_short"],
            domain_short_b=db["domain_short"],
            domain_order_a=da["domain_order"],
            domain_order_b=db["domain_order"],
            domain_remap_b=_engineering_4d_remap,
        )

    if want("table_acc_by_dim"):
        table_acc_by_dim(da["acc"], db["acc"], out_root)

    if want("table_threshold_by_domain"):
        table_threshold_by_domain(
            da["acc"], da["domain"],
            db["acc"], db["domain"],
            out_root, threshold=args.gamma,
            domain_short_a=da["domain_short"],
            domain_short_b=db["domain_short"],
            domain_order_a=da["domain_order"],
            domain_order_b=db["domain_order"],
        )

    _save_paper_stats(da, db, out_root, threshold=args.gamma)

    elapsed = time.time() - t_start
    m, s = divmod(int(elapsed), 60)
    print()
    print("=" * 60)
    print(f"  All combined figures saved to: {out_root}")
    print(f"  Total time: {m}m {s}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
