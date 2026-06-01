#!/usr/bin/env python3
"""
surrogate/run_analysis.py
=========================
Accuracy, efficiency, and sensitivity analysis for the DeepSets surrogate,
with direct comparison against AABB and MFMC.

Output structure
----------------
    results/paper_figures/
    ├── Q1_estimation/
    │   ├── acc_scatter.png/pdf          — hexbin scatter: surr / AABB / MFMC vs MC
    │   ├── per_domain_table.png/pdf/tex — per-domain ρ, MAE, FPR for all methods
    │   ├── threshold_by_domain.png/pdf  — FPR/FNR at γ by domain
    │   ├── efficiency.png/pdf           — inference time bar chart (all methods)
    │   └── pareto.png/pdf               — accuracy vs speed Pareto front
    ├── Q2_surrogate/
    │   ├── acc_by_dim.png/pdf           — accuracy split by zonotope dimension
    │   ├── accuracy_by_domain.png/pdf   — ρ / MAE by engineering domain
    │   ├── error_by_regime.png/pdf      — signed error vs I_theta level
    │   ├── exploration_budget.png/pdf   — evaluations per time budget
    │   └── consistency_rate.png/pdf     — CDF P(I ≤ γ) per method
    ├── Q3_sensitivity/
    │   ├── sensitivity_surrogate.png/pdf
    │   ├── sensitivity_compare.png/pdf
    │   ├── sobol_by_domain.png/pdf
    │   ├── response_surfaces.png/pdf
    │   ├── conditional_sensitivity.png/pdf
    │   ├── paper_landscape_a/b.png/pdf
    │   └── phase diagrams (one per domain)
    ├── Q4_counterfactual/
    │   ├── q4_overview.png/pdf
    │   ├── q4_landscape.png/pdf
    │   ├── q4_fix_direction.png/pdf
    │   └── q4_trajectories.png/pdf
    └── summary_panel.png/pdf            — all RQs in one figure

Usage
-----
    python -m surrogate.run_analysis
    python -m surrogate.run_analysis --max_scenarios 20   # quick test
    python -m surrogate.run_analysis --output results/my_run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

# ── path setup so we can be run as   python -m surrogate.run_analysis ──────
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from surrogate.dataset import _make_features
from surrogate.model import DeepSetsZonotope, MAX_DIM

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_DIRS = [
    _ROOT / "data" / "measurements_v6",
    _ROOT / "data" / "measurements_cps_v6",
]
MODEL_PATH = _ROOT / "surrogate" / "model.pt"

# ── V2 model registry (populated lazily so surrogate.models_v2 is optional) ──
_V2_MODEL_CLASS_NAMES: set[str] = set()
try:
    from surrogate.models_v2 import MODEL_REGISTRY as _V2_REGISTRY, load_checkpoint as _load_v2_checkpoint
    _V2_MODEL_CLASS_NAMES = {
        (cls.func if hasattr(cls, "func") else cls).__name__
        for cls in _V2_REGISTRY.values()
    }
except ImportError:
    _load_v2_checkpoint = None
PARAMS = ["scale_factor", "center_delta", "correlation_strength"]
PARAM_LABELS = {
    "scale_factor":         r"Scale $s_u$",
    "center_delta":         r"Center $\Delta c_u$",
    "correlation_strength": r"Corr. $R_u$",
}

# ── Publication color palette ─────────────────────────────────────────────────
C_MC   = "#333333"
C_MFMC = "#2166ac"
C_AABB = "#c45b2c"
C_SURR = "#5e3c99"   # surrogate — purple

# ── Figure style ──────────────────────────────────────────────────────────────
FS_TITLE  = 10
FS_LABEL  = 9
FS_TICK   = 7.5
FS_LEGEND = 8
FS_ANNOT  = 7
LW_MAIN   = 1.8
LW_GRID   = 0.55
ALPHA_BAND = 0.18
SPINE_COLOR = "#555555"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

EPS = 1e-10


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _style_ax(ax, grid_axis="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(SPINE_COLOR)
    ax.tick_params(labelsize=FS_TICK, colors=SPINE_COLOR)
    ax.xaxis.label.set_size(FS_LABEL)
    ax.yaxis.label.set_size(FS_LABEL)
    if grid_axis == "y":
        ax.yaxis.grid(True, ls="--", alpha=0.35, lw=LW_GRID, color="#aaaaaa")
    elif grid_axis == "both":
        ax.grid(True, ls="--", alpha=0.35, lw=LW_GRID, color="#aaaaaa")
    ax.set_axisbelow(True)


def _metrics(y_hat, y_true):
    valid = np.isfinite(y_hat) & np.isfinite(y_true)
    yh, yt = y_hat[valid], y_true[valid]
    mae  = float(np.mean(np.abs(yh - yt)))
    rho  = float(spearmanr(yh, yt).statistic)
    ss_res = np.sum((yt - yh) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    r2   = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    bias = float(np.mean(yh - yt))
    return dict(mae=mae, rho=rho, r2=r2, bias=bias, n=int(valid.sum()))


def _strong_cmap(name, lo=0.25):
    from matplotlib.colors import LinearSegmentedColormap
    base = plt.get_cmap(name)
    return LinearSegmentedColormap.from_list(
        f"{name}_strong", base(np.linspace(lo, 1.0, 256))
    )


def _scatter_panel(ax, ref, est, accent, ylabel, annotate=True,
                   max_pts=40_000, rng_seed=0):
    """Scatter panel with transparent points.

    Subsamples to *max_pts* when the dataset is larger so rendering stays fast
    while density is still visible through alpha-blending.
    """
    import matplotlib.colors as mcolors
    valid = np.isfinite(ref) & np.isfinite(est)
    ref, est = ref[valid], est[valid]

    # Subsample for large datasets
    n = len(ref)
    if n > max_pts:
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(n, size=max_pts, replace=False)
        ref_p, est_p = ref[idx], est[idx]
    else:
        ref_p, est_p = ref, est

    # Alpha scales with dataset size so denser clouds look denser
    alpha = max(0.08, min(0.40, 6_000 / max(len(ref_p), 1)))

    ax.set_facecolor(mcolors.to_rgba(accent, alpha=0.06))
    ax.scatter(ref_p, est_p, s=2.5, color=accent, alpha=alpha,
               linewidths=0, rasterized=True, zorder=2)

    ax.plot([0, 1], [0, 1], color="#333333", lw=1.4, ls="--", alpha=0.6, zorder=3)

    m = _metrics(est, ref)
    if annotate:
        ann = (rf"$\rho={m['rho']:.3f}$" + "\n"
               rf"$R^2={m['r2']:.3f}$"   + "\n"
               rf"MAE$={m['mae']:.4f}$"  + "\n"
               rf"bias$={m['bias']:+.4f}$")
        ax.text(0.04, 0.97, ann, transform=ax.transAxes, va="top",
                fontsize=FS_ANNOT + 0.5, linespacing=1.45,
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="#cccccc", alpha=0.85, lw=0.6))
    ax.set_xlabel(r"$I_\theta$ — MC reference", fontsize=FS_LABEL)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal", adjustable="box")
    _style_ax(ax, grid_axis="both")
    return m


# ═════════════════════════════════════════════════════════════════════════════
# Data loading + surrogate inference
# Uses ZonotopeDataset (same code path as evaluate.py) to guarantee identical
# sample counts. Timing and Sobol params are loaded separately from JSON.
# ═════════════════════════════════════════════════════════════════════════════

from surrogate.dataset import ZonotopeDataset, collate_fn
from torch.utils.data import DataLoader


def _load_accuracy_data(data_dirs, model, device, max_samples=None,
                         dims_filter=None, files=None):
    """
    Run surrogate on the full dataset.  Automatically selects v1 or v2
    dataset/features based on the model type.

    dims_filter : list[int] | None
        If set (e.g. [2]), only samples whose zonotope dimension is in the
        list are retained.  Use this when the model was trained on a subset
        of dimensions.
    files : list[Path] | None
        Explicit file list (overrides data_dirs glob). Use to restrict to
        specific scenario files (e.g. val split only).
    """
    if _is_v2_model(model):
        from surrogate.dataset_v2 import ZonotopeDatasetV2
        from surrogate.dataset_v2 import collate_fn as collate_fn_v2
        ds     = ZonotopeDatasetV2(data_dirs, label_key="I_theta",
                                   max_samples=max_samples, files=files)
        loader = DataLoader(ds, batch_size=2048, shuffle=False,
                            collate_fn=collate_fn_v2, num_workers=0)
        pd_key, gf_key = "per_dim_v2", "global_v2"
        label_key = "label"
    else:
        ds     = ZonotopeDataset(data_dirs, label_key="I_theta",
                                 max_samples=max_samples, files=files)
        loader = DataLoader(ds, batch_size=2048, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)
        pd_key, gf_key = "per_dim", "global_feats"
        label_key = "i_theta"

    preds, i_thetas, i_aabbs, i_mfmcs, dims, sc_ids = [], [], [], [], [], []
    t_total = 0.0

    with torch.no_grad():
        for batch in loader:
            pd = batch[pd_key].to(device)
            mk = batch["mask"].to(device)
            gf = batch[gf_key].to(device)

            t0 = time.perf_counter()
            out = model(pd, mk, gf).cpu().numpy()
            t_total += time.perf_counter() - t0

            preds.append(out)
            i_thetas.extend(batch["i_theta"])
            i_aabbs.extend(batch["i_aabb"])
            i_mfmcs.extend(batch["i_mfmc"])
            sc_ids.extend(batch.get("scenario_id", [-1] * len(batch["i_theta"])))
            mask_np = batch["mask"].numpy()
            dims.extend(mask_np.sum(axis=1).astype(int).tolist())

    i_surr  = np.concatenate(preds)
    i_theta = np.array(i_thetas, dtype=np.float32)
    i_aabb  = np.array(i_aabbs,  dtype=np.float32)
    i_mfmc  = np.array(i_mfmcs,  dtype=np.float32)
    dim_arr = np.array(dims,      dtype=np.int32)
    sc_arr  = np.array(sc_ids,    dtype=np.int32)

    if dims_filter is not None:
        keep = np.isin(dim_arr, dims_filter)
        i_surr  = i_surr[keep]
        i_theta = i_theta[keep]
        i_aabb  = i_aabb[keep]
        i_mfmc  = i_mfmc[keep]
        dim_arr = dim_arr[keep]
        sc_arr  = sc_arr[keep]
        print(f"  dims_filter={dims_filter}: kept {keep.sum():,} / {len(keep):,} samples")

    return dict(
        i_surr  = i_surr,
        i_theta = i_theta,
        i_aabb  = i_aabb,
        i_mfmc  = i_mfmc,
        dim     = dim_arr,
        scenario_id = sc_arr,
        t_surr_total = t_total,
        n = len(i_surr),
    )


def _load_timing_and_sobol(data_dirs, acc, max_scenarios=None, files=None):
    """
    Second pass over JSONs to collect timing, Sobol params, and domain labels.

    Returns a dict with:
      t_aabb, t_mfmc         — per-sample timing arrays (aligned with acc)
      sobol_scenarios        — list of {domain, rows} per scenario
      domain                 — per-sample domain string array (aligned with acc)

    files : list[Path] | None
        Explicit file list (overrides data_dirs glob). Use to restrict to
        specific scenario files (e.g. val split only).
    """
    # Try to import domain utilities from src/analysis
    try:
        _SRC_ANALYSIS = _ROOT / "src" / "analysis"
        sys.path.insert(0, str(_SRC_ANALYSIS))
        from domain_utils import get_scenario_domain, DOMAIN_SHORT, DOMAIN_ORDER
    except ImportError:
        def get_scenario_domain(d): return d.get("dataset_source", "Unknown") or "Unknown"
        DOMAIN_SHORT = {}
        DOMAIN_ORDER = []

    if files is not None:
        all_files = list(files)
    else:
        all_files = []
        for d in data_dirs:
            all_files.extend(sorted(Path(d).glob("results_scenario_*.json")))
    if max_scenarios:
        all_files = all_files[:max_scenarios]

    t_aabb_list, t_mfmc_list, t_mc_list, domain_list = [], [], [], []
    sobol_scenarios = []
    global_idx = 0

    for jf in all_files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue

        domain = get_scenario_domain(data)

        sc_rows = []
        n_acc   = acc["n"]
        for exp in data.get("experiments", []):
            # Stop collecting if we've already consumed all acc data
            if global_idx + len(sc_rows) >= n_acc:
                break
            try:
                inc = exp["post_state"]["inconsistency"]
                i_theta_v = inc.get("I_theta")
                if i_theta_v is None or not np.isfinite(float(i_theta_v)):
                    continue

                t_aabb_list.append(float(inc.get("timing_jaccard_s", 0.0)))
                t_mfmc_list.append(float(inc.get("timing_mfmc_adaptive_sobol_s", 0.0)))
                t_mc_list.append(float(inc.get("timing_mc_sobol_s", 0.0)))
                domain_list.append(domain)

                params = {}
                for p in PARAMS:
                    v = exp.get(p)
                    if v is None:
                        params = None; break
                    params[p] = float(v)
                sc_rows.append(params)
            except (KeyError, TypeError, ValueError):
                continue

        n_sc = len(sc_rows)
        n_acc = acc["n"]  # total samples available (may be capped by max_samples)
        if n_sc > 0 and global_idx < n_acc:
            # Clamp to available acc data (handles max_samples cap)
            n_avail   = min(n_sc, n_acc - global_idx)
            i_surr_sc  = acc["i_surr"] [global_idx: global_idx + n_avail]
            i_theta_sc = acc["i_theta"][global_idx: global_idx + n_avail]
            i_mfmc_sc  = acc["i_mfmc"] [global_idx: global_idx + n_avail]
            i_aabb_sc  = acc["i_aabb"] [global_idx: global_idx + n_avail]
            global_idx += n_avail

            params_ok = []
            for idx, p in enumerate(sc_rows[:n_avail]):
                if p is not None:
                    params_ok.append({
                        **p,
                        "i_theta": float(i_theta_sc[idx]),
                        "i_mfmc":  float(i_mfmc_sc[idx]),
                        "i_surr":  float(i_surr_sc[idx]),
                        "i_aabb":  float(i_aabb_sc[idx]),
                    })
            if params_ok:
                # Extract numeric scenario id from filename, e.g. results_scenario_42.json → 42
                try:
                    sc_id = int(jf.stem.split("_")[-1])
                except ValueError:
                    sc_id = -1
                sobol_scenarios.append({"domain": domain, "rows": params_ok,
                                        "scenario_id": sc_id,
                                        "json_file": str(jf)})

        if global_idx >= acc["n"]:
            break  # used up all acc data

    def _mean_us(lst):
        arr = np.array(lst, dtype=np.float32)
        pos = arr[arr > 0]
        return float(pos.mean()) * 1e6 if len(pos) > 0 else 0.0

    return dict(
        t_aabb   = np.array(t_aabb_list,  dtype=np.float32),
        t_mfmc   = np.array(t_mfmc_list,  dtype=np.float32),
        t_mc     = np.array(t_mc_list,    dtype=np.float32),
        us_aabb  = _mean_us(t_aabb_list),
        us_mfmc  = _mean_us(t_mfmc_list),
        us_mc    = _mean_us(t_mc_list),
        domain   = np.array(domain_list,  dtype=object),
        sobol_scenarios = sobol_scenarios,
        _domain_short   = DOMAIN_SHORT,
        _domain_order   = DOMAIN_ORDER,
    )


def _concat(data, key):
    """For backward compat — just returns data[key] since data is now flat."""
    return data[key]


# ═════════════════════════════════════════════════════════════════════════════
# RQ1 — Accuracy scatter
# ═════════════════════════════════════════════════════════════════════════════

def plot_acc_scatter(acc: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n[RQ1] Accuracy scatter...")

    ref  = acc["i_theta"]
    aabb = acc["i_aabb"]
    mfmc = acc["i_mfmc"]
    surr = acc["i_surr"]

    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.8),
                             gridspec_kw={"width_ratios": [1, 1, 1, 0.62],
                                          "wspace": 0.30})

    m_aabb = _scatter_panel(axes[0], ref, aabb, C_AABB,
                            r"AABB $(1-J_C)$")
    m_mfmc = _scatter_panel(axes[1], ref, mfmc, C_MFMC,
                            r"MFMC $(1-I_\mathrm{MF})$")
    m_surr = _scatter_panel(axes[2], ref, surr, C_SURR,
                            r"Surrogate $\hat{I}$")

    axes[0].set_title("AABB",      fontsize=FS_TITLE, pad=4, color=C_AABB)
    axes[1].set_title("MFMC",      fontsize=FS_TITLE, pad=4, color=C_MFMC)
    axes[2].set_title("Surrogate", fontsize=FS_TITLE, pad=4, color=C_SURR)

    # Annotate the upper-left (high I_theta, low surrogate) region
    axes[2].annotate("ranking\npreserved;\nextreme\nvalues\nsmoothed",
                     xy=(0.82, 0.18), xycoords="axes fraction",
                     fontsize=5.5, color="#444444",
                     ha="center", va="center",
                     bbox=dict(boxstyle="round,pad=0.3", fc="white",
                               ec="#cccccc", alpha=0.80, lw=0.5))

    # Error violin — enlarged width ratio above ensures more room
    ax_v = axes[3]
    for i, (err, col, lbl) in enumerate([
        (aabb - ref, C_AABB, "AABB"),
        (mfmc - ref, C_MFMC, "MFMC"),
        (surr - ref, C_SURR, "Surr"),
    ]):
        vp = ax_v.violinplot([err], positions=[i], showmedians=False,
                             showextrema=False, widths=0.72)
        for pc in vp["bodies"]:
            pc.set_facecolor(col); pc.set_edgecolor(col); pc.set_alpha(0.55)
        q1_v, med, q3_v = np.percentile(err, [25, 50, 75])
        ax_v.plot([i - 0.20, i + 0.20], [med, med],
                  color=col, lw=2.2, solid_capstyle="round", zorder=4)
        rect = plt.Rectangle((i - 0.20, q1_v), 0.40, q3_v - q1_v,
                              fc="none", ec=col, lw=1.3, zorder=3)
        ax_v.add_patch(rect)
        # Bias value only in tooltip / caption — remove inline annotation

    ax_v.axhline(0, color="#444", lw=0.85, ls="--", alpha=0.55, zorder=2)
    ax_v.set_xticks([0, 1, 2])
    ax_v.set_xticklabels(["AABB", "MFMC", "Surr"], fontsize=FS_LABEL)
    ax_v.set_ylabel(r"Signed error $(\hat{I}-I_\theta)$", fontsize=FS_LABEL)
    ax_v.set_title("Error distribution", fontsize=FS_TITLE)
    ax_v.set_xlim(-0.60, 2.60)
    _style_ax(ax_v)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"acc_scatter.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"  AABB: rho={m_aabb['rho']:.4f}  R2={m_aabb['r2']:.4f}  "
          f"MAE={m_aabb['mae']:.4f}  bias={m_aabb['bias']:+.4f}")
    print(f"  MFMC: rho={m_mfmc['rho']:.4f}  R2={m_mfmc['r2']:.4f}  "
          f"MAE={m_mfmc['mae']:.4f}  bias={m_mfmc['bias']:+.4f}")
    print(f"  Surr: rho={m_surr['rho']:.4f}  R2={m_surr['r2']:.4f}  "
          f"MAE={m_surr['mae']:.4f}  bias={m_surr['bias']:+.4f}")
    print(f"  Saved: acc_scatter.png/pdf")
    return {"AABB": m_aabb, "MFMC": m_mfmc, "Surrogate": m_surr}


# ═════════════════════════════════════════════════════════════════════════════
# RQ1 — Accuracy by dimension
# ═════════════════════════════════════════════════════════════════════════════

def plot_acc_by_dim(acc: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n[RQ1] Accuracy by dimension...")

    dims_all = acc["dim"]
    ref_all  = acc["i_theta"]
    aabb_all = acc["i_aabb"]
    mfmc_all = acc["i_mfmc"]
    surr_all = acc["i_surr"]

    unique_dims = sorted(np.unique(dims_all))
    rows_data = []
    for d in unique_dims:
        m = dims_all == d
        ref, aabb, mfmc, surr = ref_all[m], aabb_all[m], mfmc_all[m], surr_all[m]
        rows_data.append({
            "dim": d, "N": int(m.sum()),
            **{f"rho_{k}": _metrics(v, ref)["rho"]
               for k, v in [("AABB", aabb), ("MFMC", mfmc), ("Surr", surr)]},
            **{f"r2_{k}":  _metrics(v, ref)["r2"]
               for k, v in [("AABB", aabb), ("MFMC", mfmc), ("Surr", surr)]},
            **{f"mae_{k}": _metrics(v, ref)["mae"]
               for k, v in [("AABB", aabb), ("MFMC", mfmc), ("Surr", surr)]},
        })
        print(f"  dim={d} (N={m.sum():,}):  "
              f"surr rho={rows_data[-1]['rho_Surr']:.3f} "
              f"R2={rows_data[-1]['r2_Surr']:.3f} "
              f"MAE={rows_data[-1]['mae_Surr']:.4f}")

    x = np.arange(len(unique_dims))
    width = 0.24
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.8), sharey=False)
    metric_cfg = [
        ("rho", r"Spearman $\rho$", axes[0]),
        ("r2",  r"$R^2$",           axes[1]),
        ("mae", r"MAE",             axes[2]),
    ]

    for prefix, ylabel, ax in metric_cfg:
        for offset, (key, col, lbl) in enumerate([
            ("AABB", C_AABB, "AABB"),
            ("MFMC", C_MFMC, "MFMC"),
            ("Surr", C_SURR, "Surrogate"),
        ]):
            vals = [r[f"{prefix}_{key}"] for r in rows_data]
            ax.bar(x + (offset - 1) * width, vals, width=width * 0.9,
                   color=col, alpha=0.80, label=lbl, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{d}D" for d in unique_dims])
        ax.set_xlabel("Zonotope dimension", fontsize=FS_LABEL)
        ax.set_ylabel(ylabel, fontsize=FS_LABEL)
        if prefix in ("rho", "r2"):
            ax.set_ylim(0, 1.05)
        _style_ax(ax)

    axes[0].legend(fontsize=FS_LEGEND, ncol=1)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"acc_by_dim.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: acc_by_dim.png/pdf")


# ═════════════════════════════════════════════════════════════════════════════
# RQ1 — Error by I_theta regime
# ═════════════════════════════════════════════════════════════════════════════

def plot_error_by_regime(acc: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n[RQ1] Error by regime...")

    ref  = acc["i_theta"]
    aabb = acc["i_aabb"]
    mfmc = acc["i_mfmc"]
    surr = acc["i_surr"]

    n_bins = 20
    bins = np.linspace(0, 1, n_bins + 1)
    ctrs = (bins[:-1] + bins[1:]) / 2

    def _bin_stats(err):
        means, stds = [], []
        for i in range(n_bins):
            m = (ref >= bins[i]) & (ref < bins[i + 1])
            if m.sum() >= 10:
                means.append(float(np.mean(err[m])))
                stds.append(float(np.std(err[m])))
            else:
                means.append(np.nan); stds.append(np.nan)
        return np.array(means), np.array(stds)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    for est, col, lbl, mk in [
        (aabb, C_AABB, "AABB",      "o"),
        (mfmc, C_MFMC, "MFMC",      "s"),
        (surr, C_SURR, "Surrogate", "^"),
    ]:
        err = np.abs(est - ref)
        mean, std = _bin_stats(err)
        valid = ~np.isnan(mean)
        ax.fill_between(ctrs[valid],
                        mean[valid] - std[valid],
                        mean[valid] + std[valid],
                        color=col, alpha=0.10)
        ax.plot(ctrs[valid], mean[valid], color=col, lw=LW_MAIN,
                marker=mk, markersize=4, label=lbl, zorder=3)

    ax.set_xlabel(r"$I_\theta$ (reference)", fontsize=FS_LABEL)
    ax.set_ylabel(r"MAE $= |\hat{I} - I_\theta|$", fontsize=FS_LABEL)
    ax.set_title("MAE vs ground-truth inconsistency level", fontsize=FS_TITLE)
    ax.legend(fontsize=FS_LEGEND)
    _style_ax(ax)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"error_by_regime.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: error_by_regime.png/pdf")


# ═════════════════════════════════════════════════════════════════════════════
# RQ2 — Efficiency
# ═════════════════════════════════════════════════════════════════════════════

def plot_efficiency(acc: dict, t_aabb_all: np.ndarray, t_mfmc_all: np.ndarray,
                    out_dir: Path, t_mc_all: np.ndarray = None,
                    us_aabb: float = 0.0, us_mfmc: float = 0.0,
                    us_mc: float = 0.0):
    """Plot inference-time comparison.

    us_aabb / us_mfmc / us_mc : pre-computed mean µs/sample from JSON timing.
        If not provided (or zero), fall back to computing from the raw arrays.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n[RQ2] Efficiency...")

    us_surr = acc["t_surr_total"] / acc["n"] * 1e6

    # Use JSON-sourced means when available; otherwise compute from raw arrays
    if us_aabb <= 0 and (t_aabb_all > 0).any():
        us_aabb = float(np.mean(t_aabb_all[t_aabb_all > 0])) * 1e6
    if us_mfmc <= 0 and (t_mfmc_all > 0).any():
        us_mfmc = float(np.mean(t_mfmc_all[t_mfmc_all > 0])) * 1e6
    if us_mc <= 0 and t_mc_all is not None and (t_mc_all > 0).any():
        us_mc = float(np.mean(t_mc_all[t_mc_all > 0])) * 1e6

    print(f"  AABB:      {us_aabb:.1f} µs/sample")
    print(f"  Surrogate: {us_surr:.2f} µs/sample")
    print(f"  MFMC:      {us_mfmc:.1f} µs/sample")
    if us_mc > 0:
        print(f"  MC:        {us_mc:.1f} µs/sample")
    if us_surr > 0 and us_mfmc > 0:
        print(f"  Speedup vs MFMC: {us_mfmc/us_surr:.0f}x")

    # ── Main paper figure: bar chart only ────────────────────────────────────
    methods = ["AABB", "Surrogate", "MFMC", "MC"]
    times   = [us_aabb, us_surr, us_mfmc, us_mc]
    colors  = [C_AABB,  C_SURR,   C_MFMC,  C_MC]
    if us_mc <= 0:
        methods = methods[:-1]; times = times[:-1]; colors = colors[:-1]

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    bars = ax.bar(methods, times, color=colors, alpha=0.82, zorder=3,
                  edgecolor="white", linewidth=0.5)
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(times) * 0.01,
                f"{t:.1f}", ha="center", va="bottom",
                fontsize=FS_ANNOT + 1, fontweight="bold")
    ax.set_ylabel(r"Mean inference time ($\mu$s / sample)", fontsize=FS_LABEL)
    ax.set_title("Inference time per sample", fontsize=FS_TITLE)
    _style_ax(ax)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"efficiency.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: efficiency.png/pdf  (bar chart, main paper)")

    # ── Appendix figure: per-scenario scatter ────────────────────────────────
    valid = (t_aabb_all > 0) & (t_mfmc_all > 0)
    t_aabb_sc = t_aabb_all[valid] * 1e6
    t_mfmc_sc = t_mfmc_all[valid] * 1e6

    if len(t_aabb_sc) > 0 and len(t_mfmc_sc) > 0:
        fig2, ax2 = plt.subplots(figsize=(5.0, 4.0))
        n = min(len(t_aabb_sc), len(t_mfmc_sc))
        ax2.scatter(t_aabb_sc[:n], t_mfmc_sc[:n], alpha=0.45, s=18,
                    color="#888888", zorder=2, label="Scenarios")
        ax2.axhline(us_surr, color=C_SURR, lw=LW_MAIN, ls="--", zorder=3,
                    label=f"Surrogate ({us_surr:.2f} µs)")
        if us_mc > 0:
            ax2.axhline(us_mc, color=C_MC, lw=LW_MAIN, ls=":", zorder=3,
                        label=f"MC ({us_mc:.0f} µs)")
        ax2.set_xlabel(r"AABB time ($\mu$s/sample)", fontsize=FS_LABEL)
        ax2.set_ylabel(r"MFMC time ($\mu$s/sample)", fontsize=FS_LABEL)
        ax2.set_title("Per-scenario timing scatter", fontsize=FS_TITLE)
        ax2.legend(fontsize=FS_LEGEND)
        _style_ax(ax2, grid_axis="both")
        fig2.tight_layout()
        for ext in ("png", "pdf"):
            fig2.savefig(out_dir / f"efficiency_scatter.{ext}", dpi=300,
                         bbox_inches="tight")
        plt.close(fig2)
        print(f"  Saved: efficiency_scatter.png/pdf  (appendix)")

    return dict(us_aabb=us_aabb, us_surr=us_surr, us_mfmc=us_mfmc, us_mc=us_mc)


# ═════════════════════════════════════════════════════════════════════════════
# RQ2 — Surrogate generalization (train vs held-out val scenarios)
# ═════════════════════════════════════════════════════════════════════════════

def _reproduce_split(val_fraction: float = 0.15):
    """Reproduce the exact train/val file split from train.py (seed=42)."""
    from collections import defaultdict
    from surrogate.dataset import ZonotopeDataset  # noqa: F401

    all_files = []
    for d in DATA_DIRS:
        all_files.extend(sorted(Path(d).glob("results_scenario_*.json")))

    by_dim: dict = defaultdict(list)
    for jf in all_files:
        import json
        with open(jf) as fh:
            data = json.load(fh)
        c1 = data["experiments"][0]["post_state"]["uncertainty"]["source_center"]
        by_dim[len(c1)].append(jf)

    rng = np.random.default_rng(42)
    train_files, val_files = [], []
    for dim in sorted(by_dim):
        files = list(rng.permutation(by_dim[dim]))
        n_val = max(1, int(val_fraction * len(files)))
        val_files.extend(files[:n_val])
        train_files.extend(files[n_val:])
    return train_files, val_files


def _per_scenario_accuracy(file_list, model, device, label_key="I_theta",
                            dims_filter=None):
    """Run surrogate inference per scenario; return list of per-scenario dicts.

    Automatically uses v1 or v2 feature extraction based on the model type.
    dims_filter: list[int] | None — skip samples whose zonotope dimension is
    not in this list.
    """
    import json, torch
    from surrogate.model import MAX_DIM

    use_v2 = _is_v2_model(model)
    if use_v2:
        from surrogate.dataset_v2 import _compute_features, _extract_upr
        from surrogate.models_v2 import N_DIM_FEAT, N_GLOBAL
    else:
        from surrogate.dataset import _make_features, _upr_onehot
        from surrogate.model import N_DIM_FEAT, N_GLOBAL

    results = []
    model.eval()
    for jf in file_list:
        with open(jf) as fh:
            data = json.load(fh)

        # Infer domain from file path / first experiment
        domain = "unknown"
        exp0 = data["experiments"][0]
        for key in ("domain", "scenario_domain", "experiment_domain"):
            if key in exp0:
                domain = exp0[key]; break
        if domain == "unknown":
            parent = Path(jf).parent.name
            domain = "CPS" if "cps" in parent.lower() else "CONVIDE"

        preds, refs = [], []
        for exp in data["experiments"]:
            try:
                unc = exp["post_state"]["uncertainty"]
                inc = exp["post_state"]["inconsistency"]
                label = inc.get(label_key)
                if not isinstance(label, (int, float)) or not np.isfinite(label):
                    continue
                c1 = np.array(unc["source_center"],     dtype=np.float64)
                G1 = np.array(unc["source_generators"], dtype=np.float64)
                c2 = np.array(unc["target_center"],     dtype=np.float64)
                G2 = np.array(unc["target_generators"], dtype=np.float64)
                if G1.ndim == 1: G1 = G1[:, None]
                if G2.ndim == 1: G2 = G2[:, None]
                d = len(c1)
                if d > MAX_DIM:
                    continue
                if dims_filter is not None and d not in dims_filter:
                    continue

                if use_v2:
                    scales, offsets, upr_oh = _extract_upr(exp, d)
                    feats = _compute_features(c1, G1, c2, G2, scales, offsets, upr_oh)
                    pd_pad = np.zeros((MAX_DIM, N_DIM_FEAT), dtype=np.float32)
                    pd_pad[:d] = feats["per_dim_v2"]
                    global_feats = feats["global_v2"]
                else:
                    upr_scales = upr_offsets = upr_oh = None
                    cr = exp.get("consistency_relations")
                    if cr and isinstance(cr, list) and len(cr) >= d:
                        upr_scales, upr_offsets = [], []
                        dom_type = "unknown"
                        for rel in cr[:d]:
                            m = rel.get("mapping", {})
                            upr_scales.append(float(m.get("scale", 1.0)))
                            upr_offsets.append(float(m.get("offset", 0.0)))
                            if dom_type == "unknown":
                                dom_type = rel.get("upr_type", "unknown")
                        upr_oh = _upr_onehot(dom_type)
                    per_dim, global_feats = _make_features(
                        c1, G1, c2, G2,
                        upr_scales=upr_scales,
                        upr_offsets=upr_offsets,
                        upr_type_onehot=upr_oh,
                    )
                    pd_pad = np.zeros((MAX_DIM, N_DIM_FEAT), dtype=np.float32)
                    pd_pad[:d] = per_dim

                mask = np.zeros(MAX_DIM, dtype=np.float32)
                mask[:d] = 1.0

                with torch.no_grad():
                    out = model(
                        torch.tensor(pd_pad[None]).to(device),
                        torch.tensor(mask[None]).to(device),
                        torch.tensor(global_feats[None]).to(device),
                    )
                preds.append(float(out[0]))
                refs.append(float(label))
            except (KeyError, ValueError, TypeError):
                continue

        if len(preds) < 10:
            continue

        preds_a = np.array(preds)
        refs_a  = np.array(refs)
        m = _metrics(preds_a, refs_a)
        # Infer dim from first sample (already filtered to MAX_DIM)
        first_exp = data["experiments"][0]
        dim = len(first_exp["post_state"]["uncertainty"]["source_center"])

        results.append(dict(
            file=jf.name,
            domain=domain,
            dim=dim,
            n=len(preds),
            rho=m["rho"],
            mae=m["mae"],
            r2=m["r2"],
            bias=m["bias"],
        ))
    return results


def plot_generalization(model, device, out_dir: Path,
                        val_fraction: float = 0.15, dims_filter=None):
    """Inductive generalization figure: train vs held-out val scenarios.

    Three panels:
      A — Per-scenario ρ strip, sorted by dimension then split (train=circle, val=star)
      B — Train vs val ρ by zonotope dimension (grouped box)
      C — Per-domain mean ρ: train bar vs val marker overlay
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n[RQ2] Generalization (train vs val scenarios)...")

    train_files, val_files = _reproduce_split(val_fraction)
    print(f"  Split: {len(train_files)} train / {len(val_files)} val scenarios")

    train_res = _per_scenario_accuracy(train_files, model, device, dims_filter=dims_filter)
    val_res   = _per_scenario_accuracy(val_files,   model, device, dims_filter=dims_filter)

    if not train_res and not val_res:
        print("  [warn] No per-scenario data — skipping generalization figure.")
        return

    # Tag each result with split label
    for r in train_res: r["split"] = "train"
    for r in val_res:   r["split"] = "val"
    all_res = train_res + val_res

    dims   = sorted({r["dim"] for r in all_res})
    domains = sorted({r["domain"] for r in all_res})

    from matplotlib.lines import Line2D

    # ── Figure: 2 panels ─────────────────────────────────────────────────────
    fig, (ax_dim, ax_dom) = plt.subplots(1, 2, figsize=(10.0, 4.2),
                                          gridspec_kw={"wspace": 0.38})

    # ── Panel A: val-only scatter by dimension ────────────────────────────────
    rng_j = np.random.default_rng(0)

    for di, dim in enumerate(dims):
        tr_rhos = np.array([r["rho"] for r in train_res if r["dim"] == dim])
        vl_rhos = [r["rho"] for r in val_res if r["dim"] == dim]

        # Train: IQR band + median line as background reference
        if len(tr_rhos) >= 2:
            q1_t, med_t, q3_t = np.percentile(tr_rhos, [25, 50, 75])
            ax_dim.bar(di, q3_t - q1_t, bottom=q1_t,
                       color=C_SURR, alpha=0.18, width=0.55, zorder=1)
            ax_dim.plot([di - 0.27, di + 0.27], [med_t, med_t],
                        color=C_SURR, lw=2.0, alpha=0.65, zorder=2)
        elif len(tr_rhos) == 1:
            ax_dim.plot([di - 0.22, di + 0.22], [tr_rhos[0], tr_rhos[0]],
                        color=C_SURR, lw=2.0, alpha=0.65, zorder=2)

        # Val: red stars with jitter
        if vl_rhos:
            jitter = rng_j.uniform(-0.14, 0.14, len(vl_rhos))
            for jit, rv in zip(jitter, vl_rhos):
                ax_dim.scatter(di + jit, rv, color="#cc2222", marker="*",
                               s=100, zorder=4,
                               edgecolors="white", linewidth=0.4)

        # Annotate sparse 3D/4D coverage
        if dim >= 3:
            n_tr = len(tr_rhos)
            ax_dim.text(di, -0.12,
                        f"{n_tr} train\nscenario{'s' if n_tr!=1 else ''}",
                        ha="center", va="top", fontsize=FS_ANNOT - 0.5,
                        color="#777", style="italic")

    ax_dim.set_xticks(range(len(dims)))
    ax_dim.set_xticklabels([f"{d}D" for d in dims], fontsize=FS_LABEL)
    ax_dim.set_ylabel(r"Spearman $\rho$ (per scenario)", fontsize=FS_LABEL)
    ax_dim.set_title("Generalization by dimension", fontsize=FS_TITLE)
    ax_dim.set_ylim(-0.22, 1.05)
    ax_dim.axhline(0, color="#aaa", lw=0.7, ls="--", alpha=0.5)
    ax_dim.legend(
        handles=[
            Line2D([0], [0], color=C_SURR, lw=6, alpha=0.40,
                   label="Train IQR / median"),
            Line2D([0], [0], marker="*", color="w",
                   markerfacecolor="#cc2222", markersize=10,
                   label="Val (unseen scenarios)"),
        ], fontsize=FS_LEGEND, framealpha=0.88, loc="lower right")
    _style_ax(ax_dim)

    # ── Panel B: grouped bars train vs val per domain ─────────────────────────
    dom_tr: dict = {}
    dom_vl: dict = {}
    for r in train_res:
        dom_tr.setdefault(r["domain"], []).append(r["rho"])
    for r in val_res:
        dom_vl.setdefault(r["domain"], []).append(r["rho"])

    dom_list = sorted(set(dom_tr) | set(dom_vl))
    xd = np.arange(len(dom_list))
    bw = 0.35

    tr_means = [float(np.mean(dom_tr[d])) if dom_tr.get(d) else float("nan")
                for d in dom_list]
    vl_means = [float(np.mean(dom_vl[d])) if dom_vl.get(d) else float("nan")
                for d in dom_list]
    tr_errs  = [float(np.std(dom_tr[d]))  if dom_tr.get(d) and len(dom_tr[d]) > 1
                else 0.0 for d in dom_list]
    vl_errs  = [float(np.std(dom_vl[d]))  if dom_vl.get(d) and len(dom_vl[d]) > 1
                else 0.0 for d in dom_list]

    ax_dom.bar(xd - bw / 2, tr_means, bw, color=C_SURR, alpha=0.75,
               yerr=tr_errs, capsize=3, error_kw=dict(lw=1.0),
               zorder=3, label="Train")
    ax_dom.bar(xd + bw / 2, vl_means, bw, color="#cc2222", alpha=0.75,
               yerr=vl_errs, capsize=3, error_kw=dict(lw=1.0),
               zorder=3, label="Val (unseen)")

    ax_dom.set_xticks(xd)
    ax_dom.set_xticklabels(dom_list, rotation=30, ha="right",
                           fontsize=max(FS_LABEL - 1.0, 6.5))
    ax_dom.set_ylabel(r"Mean Spearman $\rho$", fontsize=FS_LABEL)
    ax_dom.set_title("Generalization by domain", fontsize=FS_TITLE)
    ax_dom.set_ylim(0, 1.05)
    ax_dom.legend(fontsize=FS_LEGEND, framealpha=0.85)
    _style_ax(ax_dom)

    fig.suptitle(
        "Surrogate inductive generalization — unseen scenarios and domains",
        fontsize=FS_TITLE + 0.5, y=1.01)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"generalization.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: generalization.png/pdf  "
          f"({len(train_res)} train, {len(val_res)} val scenarios)")

    # Print summary
    tr_rho = np.array([r["rho"] for r in train_res])
    vl_rho = np.array([r["rho"] for r in val_res])
    print(f"  Train ρ: mean={tr_rho.mean():.3f}  median={np.median(tr_rho):.3f}")
    print(f"  Val   ρ: mean={vl_rho.mean():.3f}  median={np.median(vl_rho):.3f}")


# ═════════════════════════════════════════════════════════════════════════════
# RQ3 — Sensitivity analysis (Sobol per scenario, then aggregate)
# ═════════════════════════════════════════════════════════════════════════════

def _sobol_per_scenario(sobol_scenarios: list):
    """
    Compute Sobol S1/ST per scenario.
    sobol_scenarios: list of {domain, rows} dicts (or plain list-of-rows for
                     backwards compat).  Each row has keys: scale_factor,
                     center_delta, correlation_strength, i_theta, i_mfmc, i_surr.
    """
    try:
        from SALib.analyze import sobol as sobol_analyze
    except ImportError:
        print("  [warn] SALib not found — skipping Sobol analysis.")
        return []

    problem = {
        "num_vars": 3,
        "names": PARAMS,
        "bounds": [[0, 1]] * 3,
    }

    results = []
    for sc in sobol_scenarios:
        # Accept both {domain, rows} dict format and plain list-of-rows
        if isinstance(sc, dict):
            rows   = sc.get("rows", [])
            domain = sc.get("domain", "Unknown")
        else:
            rows   = sc
            domain = "Unknown"

        if not rows:
            continue
        N = len(rows)
        # Saltelli: N = n_base * (D + 2) with D=3 → n_base = N/5
        if N % 5 != 0:
            continue

        # Build Y arrays in scenario order
        Y_mc   = np.array([r["i_theta"] for r in rows])
        Y_mfmc = np.array([r["i_mfmc"]  for r in rows])
        Y_surr = np.array([r["i_surr"]  for r in rows])

        if np.std(Y_mc) < 1e-9:
            continue   # constant scenario — Sobol undefined

        def _si(Y):
            try:
                si = sobol_analyze.analyze(problem, Y,
                                           calc_second_order=False,
                                           print_to_console=False)
                return (
                    {p: max(0.0, float(si["S1"][i]))  for i, p in enumerate(PARAMS)},
                    {p: max(0.0, float(si["ST"][i]))  for i, p in enumerate(PARAMS)},
                )
            except Exception:
                return None, None

        s1_mc,   st_mc   = _si(Y_mc)
        s1_mfmc, st_mfmc = _si(Y_mfmc)
        s1_surr, st_surr = _si(Y_surr)

        if any(x is None for x in [s1_mc, s1_mfmc, s1_surr]):
            continue

        results.append(dict(
            S1_mc=s1_mc,   ST_mc=st_mc,
            S1_mfmc=s1_mfmc, ST_mfmc=st_mfmc,
            S1_surr=s1_surr, ST_surr=st_surr,
            domain=domain,
        ))

    print(f"  Sobol computed for {len(results)} scenarios.")
    return results


def plot_sensitivity_surrogate(sobol_results, out_dir: Path):
    if not sobol_results:
        print("\n[RQ3] No Sobol results — skipping sensitivity plots.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n[RQ3] Sensitivity surrogate...")

    # Average S1 and ST across scenarios
    def _mean_si(key):
        return {p: float(np.mean([r[key][p] for r in sobol_results]))
                for p in PARAMS}

    s1_mc   = _mean_si("S1_mc");   st_mc   = _mean_si("ST_mc")
    s1_mfmc = _mean_si("S1_mfmc"); st_mfmc = _mean_si("ST_mfmc")
    s1_surr = _mean_si("S1_surr"); st_surr = _mean_si("ST_surr")

    param_lbls = [PARAM_LABELS[p] for p in PARAMS]
    x = np.arange(len(PARAMS))
    width = 0.26

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharey=True)
    for ax, s1_dict, st_dict, title in [
        (axes[0], s1_surr, st_surr, "Surrogate — Sobol indices"),
        (axes[1], s1_mc,   st_mc,   r"MC ($I_\theta$) — Sobol indices"),
    ]:
        s1_vals = [s1_dict[p] for p in PARAMS]
        st_vals = [st_dict[p] for p in PARAMS]
        ax.bar(x - width / 2, s1_vals, width=width * 0.9,
               color=C_SURR if "Surrogate" in title else C_MC,
               alpha=0.75, label=r"$S_1$ (first-order)", zorder=3)
        ax.bar(x + width / 2, st_vals, width=width * 0.9,
               color=C_SURR if "Surrogate" in title else C_MC,
               alpha=0.40, label=r"$S_T$ (total effect)",
               hatch="//", zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(param_lbls, fontsize=FS_LABEL)
        ax.set_ylabel("Sobol index", fontsize=FS_LABEL)
        ax.set_title(title, fontsize=FS_TITLE)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=FS_LEGEND)
        _style_ax(ax)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"sensitivity_surrogate.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: sensitivity_surrogate.png/pdf")

    print(f"  Surrogate S1: { {p: f'{s1_surr[p]:.3f}' for p in PARAMS} }")
    print(f"  MC       S1: { {p: f'{s1_mc[p]:.3f}'   for p in PARAMS} }")


def plot_sensitivity_compare(sobol_results, out_dir: Path):
    if not sobol_results:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n[RQ3] Sensitivity comparison (surrogate vs MFMC vs MC)...")

    def _mean_si(key):
        return {p: float(np.mean([r[key][p] for r in sobol_results]))
                for p in PARAMS}

    s1 = {k: _mean_si(k) for k in ["S1_mc", "S1_mfmc", "S1_surr"]}
    st = {k: _mean_si(k) for k in ["ST_mc", "ST_mfmc", "ST_surr"]}

    param_lbls = [PARAM_LABELS[p] for p in PARAMS]
    x = np.arange(len(PARAMS))
    width = 0.22

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True)
    # Each tuple: (ax, source_dict, key_prefix, ylabel, title)
    panels = [
        (axes[0], s1, "S1", r"Sobol $S_1$", r"First-order indices $S_1$"),
        (axes[1], st, "ST", r"Sobol $S_T$", r"Total-effect indices $S_T$"),
    ]
    method_cfg = [
        ("mc",   C_MC,   "MC ($I_\\theta$)"),
        ("mfmc", C_MFMC, "MFMC"),
        ("surr", C_SURR, "Surrogate"),
    ]

    for ax, si_dict, prefix, ylabel, title in panels:
        for i, (method, col, lbl) in enumerate(method_cfg):
            k = f"{prefix}_{method}"   # e.g. "S1_mc" or "ST_mfmc"
            vals = [si_dict[k][p] for p in PARAMS]
            ax.bar(x + (i - 1) * width, vals, width=width * 0.9,
                   color=col, alpha=0.78, label=lbl, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(param_lbls, fontsize=FS_LABEL)
        ax.set_ylabel(ylabel, fontsize=FS_LABEL)
        ax.set_title(title, fontsize=FS_TITLE)
        ax.set_ylim(0, 1.05)
        _style_ax(ax)

    axes[0].legend(fontsize=FS_LEGEND)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"sensitivity_compare.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: sensitivity_compare.png/pdf")


# ═════════════════════════════════════════════════════════════════════════════
# Summary panel
# ═════════════════════════════════════════════════════════════════════════════

def plot_summary_panel(acc: dict, acc_metrics, timing, sobol_results, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n[Summary] Combined panel...")

    fig = plt.figure(figsize=(15.0, 4.5))
    gs  = fig.add_gridspec(1, 4, wspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])

    # Panel 1 — surrogate scatter
    ref  = acc["i_theta"]
    surr = acc["i_surr"]
    _scatter_panel(ax1, ref, surr, C_SURR, r"Surrogate $\hat{I}$")
    ax1.set_title("Accuracy (vs MC)", fontsize=FS_TITLE)

    # Panel 2 — metric bars for all methods
    methods  = ["AABB", "MFMC", "Surrogate"]
    colors   = [C_AABB, C_MFMC, C_SURR]
    rho_vals = [acc_metrics[m]["rho"] for m in methods]
    r2_vals  = [acc_metrics[m]["r2"]  for m in methods]
    mae_vals = [acc_metrics[m]["mae"] for m in methods]

    x = np.arange(3)
    w = 0.25
    ax2.bar(x - w, rho_vals, w * 0.9, color=colors, alpha=0.80, label=r"$\rho$")
    ax2.bar(x,     r2_vals,  w * 0.9, color=colors, alpha=0.50, hatch="//")
    for i, mae in enumerate(mae_vals):
        ax2.text(i + w * 0.5, 0.02, f"MAE\n{mae:.3f}",
                 ha="center", va="bottom", fontsize=5.5, color=colors[i])
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=FS_LABEL - 0.5)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel(r"$\rho$ (solid) / $R^2$ (hatched)", fontsize=FS_LABEL - 0.5)
    ax2.set_title("Metrics summary", fontsize=FS_TITLE)
    _style_ax(ax2)

    # Panel 3 — efficiency bar
    t_vals  = [timing["us_aabb"], timing["us_surr"], timing["us_mfmc"]]
    t_lbls  = ["AABB", "Surrogate", "MFMC"]
    t_cols  = [C_AABB, C_SURR, C_MFMC]
    bars = ax3.bar(t_lbls, t_vals, color=t_cols, alpha=0.80, zorder=3)
    for bar, t in zip(bars, t_vals):
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(t_vals) * 0.01,
                 f"{t:.1f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax3.set_ylabel(r"µs / sample", fontsize=FS_LABEL)
    ax3.set_title("Inference time", fontsize=FS_TITLE)
    _style_ax(ax3)

    # Panel 4 — Sobol S1 comparison
    if sobol_results:
        def _mean_si(key):
            return [float(np.mean([r[key][p] for r in sobol_results]))
                    for p in PARAMS]
        x_s = np.arange(len(PARAMS))
        w_s = 0.26
        for off, (key, col, lbl) in enumerate([
            ("S1_mc",   C_MC,   "MC"),
            ("S1_mfmc", C_MFMC, "MFMC"),
            ("S1_surr", C_SURR, "Surr"),
        ]):
            ax4.bar(x_s + (off - 1) * w_s, _mean_si(key), w_s * 0.9,
                    color=col, alpha=0.78, label=lbl, zorder=3)
        ax4.set_xticks(x_s)
        ax4.set_xticklabels([PARAM_LABELS[p] for p in PARAMS],
                            fontsize=FS_LABEL - 1.5, rotation=10, ha="right")
        ax4.set_ylim(0, 1.05)
        ax4.set_ylabel(r"Sobol $S_1$", fontsize=FS_LABEL)
        ax4.set_title("Sensitivity", fontsize=FS_TITLE)
        ax4.legend(fontsize=FS_LEGEND - 1)
        _style_ax(ax4)
    else:
        ax4.text(0.5, 0.5, "SALib not available\n(install SALib)",
                 ha="center", va="center", transform=ax4.transAxes,
                 fontsize=FS_LABEL, color="#888888")
        _style_ax(ax4)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"summary_panel.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: summary_panel.png/pdf")


# ═════════════════════════════════════════════════════════════════════════════
# Paper figures — CDF, robustness margin, Pareto, exploration budget
# ═════════════════════════════════════════════════════════════════════════════

def plot_consistency_rate(acc: dict, out_dir: Path):
    """CDF: P(I ≤ γ) for each estimator, plus deviation from MC reference."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n[Paper] Consistency-rate CDF...")

    ref  = acc["i_theta"]
    aabb = acc["i_aabb"]
    mfmc = acc["i_mfmc"]
    surr = acc["i_surr"]

    gamma = np.linspace(0.0, 1.0, 500)

    def _cdf(x):
        fx = x[np.isfinite(x)]
        return np.array([np.mean(fx <= g) for g in gamma])

    cdf_ref  = _cdf(ref)
    cdf_aabb = _cdf(aabb)
    cdf_mfmc = _cdf(mfmc)
    cdf_surr = _cdf(surr)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.5))

    ax = axes[0]
    for cdf, col, lbl, ls in [
        (cdf_ref,  C_MC,   r"MC $I_\theta$ (ref)", "-"),
        (cdf_aabb, C_AABB, "AABB",                 "--"),
        (cdf_mfmc, C_MFMC, "MFMC",                 "-."),
        (cdf_surr, C_SURR, "Surrogate",             ":"),
    ]:
        ax.plot(gamma, cdf, color=col, lw=LW_MAIN, ls=ls, label=lbl)
    ax.set_xlabel(r"Threshold $\gamma$", fontsize=FS_LABEL)
    ax.set_ylabel(r"$P(I \leq \gamma)$", fontsize=FS_LABEL)
    ax.set_title("Consistency-rate CDF", fontsize=FS_TITLE)
    ax.legend(fontsize=FS_LEGEND)
    _style_ax(ax, grid_axis="both")

    ax2 = axes[1]
    for cdf, col, lbl, ls in [
        (cdf_aabb, C_AABB, "AABB",      "--"),
        (cdf_mfmc, C_MFMC, "MFMC",      "-."),
        (cdf_surr, C_SURR, "Surrogate", ":"),
    ]:
        ax2.plot(gamma, cdf - cdf_ref, color=col, lw=LW_MAIN, ls=ls, label=lbl)
    ax2.axhline(0.0, color="#888", lw=0.8, alpha=0.5)
    ax2.set_xlabel(r"Threshold $\gamma$", fontsize=FS_LABEL)
    ax2.set_ylabel(r"$\Delta P(I \leq \gamma)$ vs MC", fontsize=FS_LABEL)
    ax2.set_title("CDF deviation from MC reference", fontsize=FS_TITLE)
    ax2.legend(fontsize=FS_LEGEND)
    _style_ax(ax2, grid_axis="both")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"consistency_rate.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: consistency_rate.png/pdf")


def plot_robustness_margin(acc: dict, out_dir: Path, threshold: float = 0.5):
    """Margin distribution + FPR/FNR at a decision threshold."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Paper] Robustness margin (threshold={threshold})...")

    ref  = acc["i_theta"]
    aabb = acc["i_aabb"]
    mfmc = acc["i_mfmc"]
    surr = acc["i_surr"]

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.5))

    # Left: signed margin violin
    ax = axes[0]
    for i, (x, col, lbl) in enumerate([
        (ref,  C_MC,   r"MC $I_\theta$"),
        (aabb, C_AABB, "AABB"),
        (mfmc, C_MFMC, "MFMC"),
        (surr, C_SURR, "Surrogate"),
    ]):
        margin = x[np.isfinite(x)] - threshold
        vp = ax.violinplot([margin], positions=[i], showmedians=False,
                           showextrema=False, widths=0.65)
        for pc in vp["bodies"]:
            pc.set_facecolor(col); pc.set_edgecolor(col); pc.set_alpha(0.50)
        q1, med, q3 = np.percentile(margin, [25, 50, 75])
        ax.plot([i - 0.18, i + 0.18], [med, med], color=col, lw=2.0,
                solid_capstyle="round", zorder=4)
        rect = plt.Rectangle((i - 0.18, q1), 0.36, q3 - q1,
                              fc="none", ec=col, lw=1.2, zorder=3)
        ax.add_patch(rect)
    ax.axhline(0, color="#444", lw=0.9, ls="--", alpha=0.6)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["MC", "AABB", "MFMC", "Surr"], fontsize=FS_LABEL - 0.5)
    ax.set_ylabel(rf"Margin $(\hat{{I}} - {threshold})$", fontsize=FS_LABEL)
    ax.set_title(rf"Margin to threshold $\gamma={threshold}$", fontsize=FS_TITLE)
    _style_ax(ax)

    # Right: FPR / FNR bars
    ax2 = axes[1]
    labels, fprs, fnrs = [], [], []
    for x, col, lbl in [(aabb, C_AABB, "AABB"), (mfmc, C_MFMC, "MFMC"),
                         (surr, C_SURR, "Surrogate")]:
        valid = np.isfinite(x) & np.isfinite(ref)
        xv, rv = x[valid], ref[valid]
        pred_pos   = xv >= threshold
        actual_pos = rv >= threshold
        actual_neg = rv < threshold
        n_pos = actual_pos.sum()
        n_neg = actual_neg.sum()
        fpr = float((pred_pos & actual_neg).sum() / n_neg) if n_neg > 0 else 0.0
        fnr = float((~pred_pos & actual_pos).sum() / n_pos) if n_pos > 0 else 0.0
        labels.append(lbl); fprs.append(fpr); fnrs.append(fnr)

    x_pos  = np.arange(len(labels))
    width  = 0.3
    colors = [C_AABB, C_MFMC, C_SURR]
    bars_fpr = ax2.bar(x_pos - width / 2, fprs, width * 0.9,
                       color=colors, alpha=0.80, label="FPR", zorder=3)
    bars_fnr = ax2.bar(x_pos + width / 2, fnrs, width * 0.9,
                       color=colors, alpha=0.45, hatch="//", label="FNR", zorder=3)
    for i, (fpr, fnr) in enumerate(zip(fprs, fnrs)):
        ax2.text(i - width / 2, fpr + 0.004, f"{fpr:.3f}", ha="center",
                 va="bottom", fontsize=6.5)
        ax2.text(i + width / 2, fnr + 0.004, f"{fnr:.3f}", ha="center",
                 va="bottom", fontsize=6.5)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, fontsize=FS_LABEL)
    ax2.set_ylabel("Rate", fontsize=FS_LABEL)
    ax2.set_title(rf"FPR / FNR at threshold $\gamma={threshold}$", fontsize=FS_TITLE)
    ax2.legend(fontsize=FS_LEGEND)
    _style_ax(ax2)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"robustness_margin.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: robustness_margin.png/pdf")
    print(f"  FPR — AABB:{fprs[0]:.4f}  MFMC:{fprs[1]:.4f}  Surr:{fprs[2]:.4f}")
    print(f"  FNR — AABB:{fnrs[0]:.4f}  MFMC:{fnrs[1]:.4f}  Surr:{fnrs[2]:.4f}")


def plot_pareto(acc: dict, timing: dict, out_dir: Path):
    """Accuracy vs. speed Pareto front among AABB, MFMC, Surrogate, MC."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n[Paper] Pareto plot (accuracy vs speed)...")

    ref = acc["i_theta"]
    methods_cfg = [
        ("AABB",      acc["i_aabb"], C_AABB, timing.get("us_aabb", 0.0)),
        ("MFMC",      acc["i_mfmc"], C_MFMC, timing.get("us_mfmc", 0.0)),
        ("Surrogate", acc["i_surr"], C_SURR, timing.get("us_surr", 0.0)),
    ]
    # Add MC as perfect-accuracy reference if timing is available
    us_mc = timing.get("us_mc", 0.0)
    if us_mc > 0:
        methods_cfg.append(("MC", ref, C_MC, us_mc))

    names, rhos, maes, times, cols = [], [], [], [], []
    color_map = [C_AABB, C_MFMC, C_SURR, C_MC]
    for i, (name, pred, col, t) in enumerate(methods_cfg):
        m = _metrics(pred, ref)
        names.append(name); rhos.append(m["rho"])
        maes.append(m["mae"]); times.append(max(t, 1e-3))
        cols.append(col)

    # Per-method marker/size config
    _MARKER = {"MC": "*", "Surrogate": "D", "MFMC": "s", "AABB": "o"}
    _SIZE   = {"MC": 220, "Surrogate": 240, "MFMC": 160, "AABB": 130}

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.5))

    r2s = [_metrics(pred, ref)["r2"] for _, pred, _, _ in methods_cfg]

    for ax, y_vals, ylabel, title, better in [
        (axes[0], rhos, r"Spearman $\rho$", r"Accuracy ($\rho$) vs Speed", "up"),
        (axes[1], r2s,  r"$R^2$",           r"Accuracy ($R^2$) vs Speed",  "up"),
    ]:
        for i, (name, y, t) in enumerate(zip(names, y_vals, times)):
            mk = _MARKER.get(name, "o")
            sz = _SIZE.get(name, 160)
            ax.scatter(t, y, s=sz, color=cols[i], zorder=4,
                       edgecolors="white", linewidth=0.9, marker=mk)
            label = name
            if name == "Surrogate":
                label = "Surrogate\n(learned)"
            ax.annotate(label, (t, y), xytext=(9, 5), textcoords="offset points",
                        fontsize=FS_LABEL, color=cols[i], fontweight="bold")

        ax.set_xscale("log")
        ax.set_xlabel(r"Inference time ($\mu$s / sample, log scale)", fontsize=FS_LABEL)
        ax.set_ylabel(ylabel, fontsize=FS_LABEL)
        ax.set_title(title, fontsize=FS_TITLE)
        _style_ax(ax, grid_axis="both")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"pareto.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: pareto.png/pdf")


def plot_response_surface_comparison(sobol_scenarios: list, out_dir: Path,
                                     gamma: float = 0.5, n_grid: int = 80,
                                     exclude_domains: set = None):
    """MC vs Surrogate response-surface comparison for RQ2.

    Two scenario rows chosen from domains NOT already shown in paper_landscape:
      col 1 — MC  I_theta
      col 2 — Surrogate  Î          (shared colorbar with col 1)
      col 3 — absolute error |MC - Surrogate|

    Parameters
    ----------
    exclude_domains : set of str, optional
        Domain names already used in paper_landscape — these are skipped so the
        reader does not see the same scenario twice in 2D and 3D.
    """
    if not sobol_scenarios:
        print("\n[RQ2] No sobol scenarios — skipping response surface comparison.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    exclude_domains = set(exclude_domains or [])
    print("\n[RQ2] Response-surface comparison (MC vs Surrogate)...")
    if exclude_domains:
        print(f"  Excluding domains from paper_landscape: {sorted(exclude_domains)}")

    xparam, yparam, fixed_param = _SURF_PAIRS[0]   # scale vs center
    meta_x   = _SURF_META[xparam]
    meta_y   = _SURF_META[yparam]
    op, thresh, slice_desc = _SURF_SLICE[fixed_param]

    # Build per-scenario slices
    sc_slices = []
    for sc in sobol_scenarios:
        rows   = sc["rows"]   if isinstance(sc, dict) else sc
        domain = sc.get("domain", "Unknown") if isinstance(sc, dict) else "Unknown"
        xs, ys, mcs, sus = [], [], [], []
        for r in rows:
            fval = r.get(fixed_param)
            if fval is None:
                continue
            fval = float(fval)
            if op == "<"   and fval >= thresh:                       continue
            if op == "ab"  and abs(fval) >= thresh:                  continue
            if op == "rng" and not (thresh[0] <= fval <= thresh[1]): continue
            try:
                xs.append(float(r[xparam]));  ys.append(float(r[yparam]))
                mcs.append(float(r["i_theta"])); sus.append(float(r["i_surr"]))
            except (KeyError, TypeError, ValueError):
                continue
        if len(xs) >= 30:
            sc_slices.append({"domain": domain, "x": xs, "y": ys,
                               "mc": mcs, "surr": sus})

    if not sc_slices:
        print("  SKIP: no scenarios with sufficient data for surface comparison.")
        return

    # Compute per-scenario ρ to rank
    from scipy.stats import spearmanr as _spear
    for sc in sc_slices:
        rho, _ = _spear(sc["mc"], sc["surr"])
        sc["rho"] = float(rho) if np.isfinite(rho) else -1.0

    sc_slices.sort(key=lambda s: s["rho"], reverse=True)

    # Prefer scenarios not already in paper_landscape
    preferred = [s for s in sc_slices if s["domain"] not in exclude_domains]
    pool = preferred if len(preferred) >= 2 else sc_slices  # fallback if too few

    # Pick 2 contrasting scenarios: best ρ + most different domain
    pick_simple  = pool[0]
    pick_complex = None
    for sc in reversed(pool):
        if sc["domain"] != pick_simple["domain"]:
            pick_complex = sc
            break
    if pick_complex is None:
        pick_complex = pool[-1] if len(pool) > 1 else pool[0]

    scenarios = [pick_simple, pick_complex]

    xi = np.linspace(meta_x["lo"], meta_x["hi"], n_grid)
    yi = np.linspace(meta_y["lo"], meta_y["hi"], n_grid)
    XX, YY = np.meshgrid(xi, yi)
    levels_01 = np.linspace(0, 1, 38)   # same as paper_landscape

    n_rows = len(scenarios)
    fig, axes = plt.subplots(n_rows, 3,
                             figsize=(12.0, 4.0 * n_rows),
                             gridspec_kw={"wspace": 0.08, "hspace": 0.38})
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row_idx, sc in enumerate(scenarios):
        pts  = np.column_stack([sc["x"], sc["y"]])
        mc_a = np.array(sc["mc"])
        su_a = np.array(sc["surr"])
        ZZ_mc   = _interp_surf(pts, mc_a, XX, YY)
        ZZ_surr = _interp_surf(pts, su_a, XX, YY)
        ZZ_err  = np.abs(ZZ_mc - ZZ_surr)

        # ── MC panel (col 0) ─────────────────────────────────────────────────
        ax_mc   = axes[row_idx, 0]
        ax_surr = axes[row_idx, 1]
        ax_err  = axes[row_idx, 2]

        cf_mc = ax_mc.contourf(XX, YY, ZZ_mc, levels=levels_01,
                               cmap="RdYlBu_r", vmin=0, vmax=1, extend="both")
        ax_mc.contour(XX, YY, ZZ_mc, levels=[gamma],
                      colors=["#222"], linewidths=[1.8], linestyles=["--"])
        ax_mc.scatter(sc["x"], sc["y"], s=2, color="k",
                      alpha=0.12, linewidths=0, rasterized=True)

        # ── Surrogate panel (col 1) — same cmap/scale as MC ──────────────────
        cf_surr = ax_surr.contourf(XX, YY, ZZ_surr, levels=levels_01,
                                   cmap="RdYlBu_r", vmin=0, vmax=1, extend="both")
        ax_surr.contour(XX, YY, ZZ_surr, levels=[gamma],
                        colors=["#222"], linewidths=[1.8], linestyles=["--"])
        ax_surr.scatter(sc["x"], sc["y"], s=2, color="k",
                        alpha=0.12, linewidths=0, rasterized=True)

        # Shared colorbar for MC + Surrogate, placed to the right of col 1
        fig.colorbar(cf_surr, ax=[ax_mc, ax_surr], shrink=0.88, pad=0.02,
                     label=r"$I(\theta)$").ax.tick_params(labelsize=FS_TICK)

        # ── Error panel (col 2) ───────────────────────────────────────────────
        cf_err = ax_err.contourf(XX, YY, ZZ_err, levels=32,
                                 cmap="Oranges", extend="max")
        ax_err.scatter(sc["x"], sc["y"], s=2, color="k",
                       alpha=0.12, linewidths=0, rasterized=True)
        fig.colorbar(cf_err, ax=ax_err, shrink=0.88, pad=0.02,
                     label=r"$|\hat{I}-I_\theta|$").ax.tick_params(labelsize=FS_TICK)

        # ── Axis decoration ───────────────────────────────────────────────────
        for col_idx, ax in enumerate([ax_mc, ax_surr, ax_err]):
            ax.set_xlim(meta_x["lo"], meta_x["hi"])
            ax.set_ylim(meta_y["lo"], meta_y["hi"])
            ax.set_xlabel(meta_x["label"], fontsize=FS_LABEL)
            ax.spines[["top", "right"]].set_visible(False)
            if col_idx == 0:
                ax.set_ylabel(meta_y["label"], fontsize=FS_LABEL)
            else:
                ax.set_yticklabels([])

        ax_mc.set_title(rf"MC $I_\theta$",          fontsize=FS_TITLE)
        ax_surr.set_title(rf"Surrogate $\hat{{I}}$", fontsize=FS_TITLE)
        ax_err.set_title(r"$|\hat{I} - I_\theta|$",  fontsize=FS_TITLE)

        # Row label: domain + ρ
        axes[row_idx, 0].text(
            -0.22, 0.5,
            f"{sc['domain']}\n" + r"$\rho=$" + f"{sc['rho']:.2f}",
            transform=axes[row_idx, 0].transAxes,
            va="center", ha="right", fontsize=FS_LABEL,
            color="#333333", rotation=90)

    fig.text(0.5, 0.01, f"Fixed: {slice_desc}",
             ha="center", fontsize=7.5, color="#555")
    fig.suptitle("Surrogate vs MC response surfaces",
                 fontsize=FS_TITLE + 0.5, y=1.01)
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"response_surface_comparison.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: response_surface_comparison.png/pdf  "
          f"({scenarios[0]['domain']} ρ={scenarios[0]['rho']:.2f}  |  "
          f"{scenarios[1]['domain']} ρ={scenarios[1]['rho']:.2f})")


def plot_exploration_budget(timing: dict, out_dir: Path,
                            budgets_s=(0.001, 0.01, 0.1, 1.0, 10.0, 60.0)):
    """Evaluations achievable per time budget for each method.

    ICDM style:
    - Surrogate line is visually dominant (thick, saturated colour).
    - AABB is de-emphasised (lighter, dotted) — it is the baseline, not the hero.
    - Subtle region annotations (Interactive / Large-scale / Exhaustive)
      communicate practical operation regimes without cluttering the plot.
    - Distinct line styles for grayscale printing.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n[Paper] Exploration budget...")

    us_aabb = max(timing.get("us_aabb", 1.0), EPS)
    us_surr = max(timing.get("us_surr", 1.0), EPS)
    us_mfmc = max(timing.get("us_mfmc", 1.0), EPS)
    us_mc   = timing.get("us_mc", 0.0)

    budgets_us = np.array(budgets_s) * 1e6

    # Method styling: (name, t_us, color, linestyle, marker, lw, alpha, zorder)
    methods = [
        ("AABB",      us_aabb, C_AABB, ":",  "o", LW_MAIN - 0.4, 0.60, 2),
        ("MFMC",      us_mfmc, C_MFMC, "--", "s", LW_MAIN,       0.80, 3),
        ("Surrogate", us_surr, C_SURR, "-",  "^", LW_MAIN + 1.6, 1.00, 5),
    ]
    if us_mc > 0:
        methods.append(("MC", max(us_mc, EPS), C_MC, "-.", "D",
                        LW_MAIN - 0.2, 0.70, 2))

    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    for name, t_us, col, ls, mk, lw, alpha, zo in methods:
        n_evals = budgets_us / t_us
        ax.plot(budgets_s, n_evals,
                color=col, lw=lw, ls=ls,
                marker=mk, markersize=5 if name == "Surrogate" else 4,
                label=name, alpha=alpha, zorder=zo)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Time budget (s)", fontsize=FS_LABEL)
    ax.set_ylabel("Evaluations achievable", fontsize=FS_LABEL)
    ax.set_title("Scalable exploration: evaluations per time budget",
                 fontsize=FS_TITLE)

    # Subtle region shading (low alpha so it doesn't compete with lines)
    _ymin, _ymax = ax.get_ylim() if ax.get_ylim()[1] > 1 else (1, 1e8)
    _ymax = max(_ymax, 1e8)
    _shade_kw = dict(alpha=0.045, zorder=0)
    ax.axvspan(budgets_s[0],  0.05,  color="#4dac26", **_shade_kw)
    ax.axvspan(0.05,           5.0,  color="#f1a340", **_shade_kw)
    ax.axvspan(5.0,  budgets_s[-1], color="#d01c8b", **_shade_kw)

    # Region text labels at top of shaded bands
    ax.text(0.007,  1.5e7, "Interactive",  fontsize=FS_ANNOT - 0.5,
            color="#2e7d32", alpha=0.75, ha="left")
    ax.text(0.18,   1.5e7, "Large-scale",  fontsize=FS_ANNOT - 0.5,
            color="#b45309", alpha=0.75, ha="left")
    ax.text(6.0,    1.5e7, "Exhaustive",   fontsize=FS_ANNOT - 0.5,
            color="#9b2226", alpha=0.75, ha="left")

    ax.legend(fontsize=FS_LEGEND, framealpha=0.90,
              handlelength=2.2, loc="lower right")
    _style_ax(ax, grid_axis="both")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"exploration_budget.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: exploration_budget.png/pdf")


# ═════════════════════════════════════════════════════════════════════════════
# Paper figures — domain-stratified accuracy and threshold classification
# ═════════════════════════════════════════════════════════════════════════════

def plot_per_domain_table(acc: dict, domain: np.ndarray, out_dir: Path,
                          threshold: float = 0.85,
                          domain_short: dict = None,
                          domain_order: list = None):
    """Per-domain accuracy table for AABB, MFMC, and Surrogate vs MC.

    Outputs:
      per_domain_table.png  — matplotlib rendered table
      per_domain_table.tex  — LaTeX tabular, paste directly into paper
    Columns: Domain | N | MFMC (ρ, MAE, FPR) | AABB (ρ, MAE, FPR) | Surrogate (ρ, MAE, FPR)
    FPR = false positive rate at threshold γ (consistent → predicted inconsistent).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Table] Per-domain accuracy table (γ={threshold})...")

    if len(domain) == 0:
        print("  [warn] No domain data — skipping table.")
        return

    ref  = acc["i_theta"]
    aabb = acc["i_aabb"]
    mfmc = acc["i_mfmc"]
    surr = acc["i_surr"]

    short    = domain_short or {}
    unique_d = domain_order if domain_order else sorted(set(domain))
    unique_d = [d for d in unique_d if d in set(domain)]
    if not unique_d:
        unique_d = sorted(set(domain))

    rows = []
    for d in unique_d:
        m = domain == d
        if m.sum() < 10:
            continue
        rv = ref[m]
        row = {"domain": d, "N": int(m.sum())}
        for tag, pred in [("MFMC", mfmc[m]), ("AABB", aabb[m]), ("Surrogate", surr[m])]:
            valid = np.isfinite(pred) & np.isfinite(rv)
            xv, rv2 = pred[valid], rv[valid]
            met = _metrics(xv, rv2)
            row[f"rho_{tag}"] = met["rho"]
            row[f"mae_{tag}"] = met["mae"]
            # FPR: consistent in reality (rv < threshold) predicted as inconsistent
            actual_neg = rv2 < threshold
            pred_pos   = xv >= threshold
            n_neg = actual_neg.sum()
            row[f"fpr_{tag}"] = float((pred_pos & actual_neg).sum() / n_neg) \
                                 if n_neg > 0 else float("nan")
        rows.append(row)

    if not rows:
        print("  [warn] No rows after filtering — skipping table.")
        return

    # ── PNG rendered table ────────────────────────────────────────────────────
    col_groups = [
        ("MFMC",      C_MFMC, ["rho_MFMC", "mae_MFMC", "fpr_MFMC"]),
        ("AABB",      C_AABB, ["rho_AABB", "mae_AABB", "fpr_AABB"]),
        ("Surrogate", C_SURR, ["rho_Surrogate", "mae_Surrogate", "fpr_Surrogate"]),
    ]
    col_headers = ["Domain", "N"] + [
        f"{g}\nρ" for g, _, _ in col_groups
    ] + [
        f"{g}\nMAE" for g, _, _ in col_groups
    ] + [
        f"{g}\nFPR" for g, _, _ in col_groups
    ]
    # Flat column order: Domain, N, then for each row all metrics grouped by method
    flat_cols = (["domain", "N"]
                 + [f"rho_{g}" for g, _, _ in col_groups]
                 + [f"mae_{g}" for g, _, _ in col_groups]
                 + [f"fpr_{g}" for g, _, _ in col_groups])

    dlbls = [short.get(r["domain"], r["domain"]) for r in rows]
    table_data = []
    for r, dlbl in zip(rows, dlbls):
        row_vals = [dlbl, f"{r['N']:,}"]
        for g, _, _ in col_groups:
            row_vals.append(f"{r[f'rho_{g}']:.3f}")
        for g, _, _ in col_groups:
            row_vals.append(f"{r[f'mae_{g}']:.4f}")
        for g, _, _ in col_groups:
            v = r[f"fpr_{g}"]
            row_vals.append(f"{v:.3f}" if np.isfinite(v) else "—")
        table_data.append(row_vals)

    n_cols = len(flat_cols)
    n_rows = len(rows)
    fig_w  = max(14.0, n_cols * 1.2)
    fig_h  = max(3.0, (n_rows + 2) * 0.38)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    tbl = ax.table(cellText=table_data,
                   colLabels=col_headers,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.35)

    # Colour header cells by method group
    method_col_ranges = {
        "MFMC":      (2, 4),   # cols 2,3,4 → rho, mae, fpr
        "AABB":      (5, 7),
        "Surrogate": (8, 10),
    }
    for col_idx, (g, col, _) in enumerate(col_groups):
        for ci in range(2 + col_idx, n_cols, 3):
            tbl[(0, ci)].set_facecolor(col)
            tbl[(0, ci)].set_text_props(color="white", fontweight="bold")

    # Shade alternate data rows
    for ri in range(n_rows):
        if ri % 2 == 0:
            for ci in range(n_cols):
                tbl[(ri + 1, ci)].set_facecolor("#f7f7f7")

    ax.set_title(
        rf"Per-domain accuracy vs MC reference ($\gamma={threshold}$)",
        fontsize=FS_TITLE, pad=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"per_domain_table.{ext}", dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    print("  Saved: per_domain_table.png/pdf")

    # ── LaTeX table ───────────────────────────────────────────────────────────
    tex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{Per-domain accuracy of MFMC, AABB, and Surrogate vs.\ MC"
        rf" at $\gamma={threshold}$. "
        r"$\rho$ = Spearman rank correlation; MAE = mean absolute error; "
        r"FPR = false positive rate (fraction of safe scenarios mis-classified "
        r"as inconsistent).}}",
        r"\label{tab:accuracy_by_domain}",
        r"\begin{tabular}{@{}llrrrrrrrrr@{}}",
        r"\toprule",
        r"Domain & $N$ & "
        r"\multicolumn{3}{c}{MFMC} & "
        r"\multicolumn{3}{c}{AABB} & "
        r"\multicolumn{3}{c}{Surrogate} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
        r"& & $\rho$ & MAE & FPR & $\rho$ & MAE & FPR & $\rho$ & MAE & FPR \\",
        r"\midrule",
    ]
    for r, dlbl in zip(rows, dlbls):
        def _fpr(tag):
            v = r[f"fpr_{tag}"]
            return r"---" if not np.isfinite(v) else f"{v:.3f}"
        line = (f"  {dlbl} & {r['N']:,} & "
                f"{r['rho_MFMC']:.3f} & {r['mae_MFMC']:.4f} & {_fpr('MFMC')} & "
                f"{r['rho_AABB']:.3f} & {r['mae_AABB']:.4f} & {_fpr('AABB')} & "
                f"{r['rho_Surrogate']:.3f}  & {r['mae_Surrogate']:.4f}  & {_fpr('Surrogate')} \\\\")
        tex_lines.append(line)
    tex_lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    tex_path = out_dir / "per_domain_table.tex"
    tex_path.write_text("\n".join(tex_lines), encoding="utf-8")
    print("  Saved: per_domain_table.tex")
    print(f"  Domains: {len(rows)}")


def plot_accuracy_by_domain(acc: dict, domain: np.ndarray, out_dir: Path,
                             domain_short: dict = None,
                             domain_order: list = None):
    """Accuracy metrics (ρ, MAE) grouped by engineering domain."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n[Paper] Accuracy by domain...")

    if len(domain) == 0:
        print("  [warn] No domain data — skipping.")
        return

    ref  = acc["i_theta"]
    aabb = acc["i_aabb"]
    mfmc = acc["i_mfmc"]
    surr = acc["i_surr"]

    short = domain_short or {}
    unique_d = domain_order if domain_order else sorted(set(domain))
    unique_d = [d for d in unique_d if d in set(domain)]
    if not unique_d:
        unique_d = sorted(set(domain))

    rows = []
    for d in unique_d:
        m = domain == d
        if m.sum() < 5:
            continue
        r, a, f, s = ref[m], aabb[m], mfmc[m], surr[m]
        entry = {"domain": d, "N": int(m.sum())}
        for k, v in [("AABB", a), ("MFMC", f), ("Surr", s)]:
            met = _metrics(v, r)
            entry[f"rho_{k}"] = met["rho"]
            entry[f"mae_{k}"] = met["mae"]
        rows.append(entry)
        print(f"  {d} (N={m.sum():,}):  "
              f"Surr rho={entry['rho_Surr']:.3f}  MAE={entry['mae_Surr']:.4f}")

    if not rows:
        print("  [warn] No domain rows after filtering — skipping.")
        return

    x     = np.arange(len(rows))
    width = 0.28
    dlbls = [short.get(r["domain"], r["domain"]) for r in rows]

    fig, ax = plt.subplots(figsize=(max(7.0, 1.6 * len(rows)), 4.2))
    for offset, (key, col, lbl) in enumerate([
        ("AABB", C_AABB, "AABB"),
        ("MFMC", C_MFMC, "MFMC"),
        ("Surr", C_SURR, "Surrogate"),
    ]):
        vals = [r[f"rho_{key}"] for r in rows]
        ax.bar(x + (offset - 1) * width, vals, width * 0.88,
               color=col, alpha=0.82, label=lbl, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(dlbls, rotation=30, ha="right",
                       fontsize=max(FS_LABEL - 0.5, 6.5))
    ax.set_ylabel(r"Spearman $\rho$", fontsize=FS_LABEL)
    ax.set_xlabel("Domain", fontsize=FS_LABEL)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=FS_LEGEND)
    _style_ax(ax)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"accuracy_by_domain.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: accuracy_by_domain.png/pdf")


def plot_threshold_by_domain(acc: dict, domain: np.ndarray, out_dir: Path,
                              threshold: float = 0.5,
                              domain_short: dict = None,
                              domain_order: list = None):
    """FPR / FNR at threshold γ, split by engineering domain."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Paper] Threshold classification by domain (gamma={threshold})...")

    if len(domain) == 0:
        print("  [warn] No domain data — skipping.")
        return

    ref  = acc["i_theta"]
    aabb = acc["i_aabb"]
    mfmc = acc["i_mfmc"]
    surr = acc["i_surr"]

    short    = domain_short or {}
    unique_d = domain_order if domain_order else sorted(set(domain))
    unique_d = [d for d in unique_d if d in set(domain)]
    if not unique_d:
        unique_d = sorted(set(domain))

    rows = []
    for d in unique_d:
        m = domain == d
        if m.sum() < 10:
            continue
        rv = ref[m]
        row = {"domain": d}
        for key, pred in [("AABB", aabb[m]), ("MFMC", mfmc[m]), ("Surr", surr[m])]:
            valid = np.isfinite(pred) & np.isfinite(rv)
            xv, rv2 = pred[valid], rv[valid]
            pp = xv >= threshold
            ap = rv2 >= threshold
            an = rv2 < threshold
            n_pos = ap.sum(); n_neg = an.sum()
            row[f"fpr_{key}"] = float((pp & an).sum() / n_neg) if n_neg > 0 else 0.0
            row[f"fnr_{key}"] = float((~pp & ap).sum() / n_pos) if n_pos > 0 else 0.0
        rows.append(row)

    if not rows:
        print("  [warn] No domain rows — skipping.")
        return

    x     = np.arange(len(rows))
    width = 0.22
    dlbls = [short.get(r["domain"], r["domain"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(max(8.0, 1.5 * len(rows)), 4.5))
    for ax, rate, ylabel, title in [
        (axes[0], "fpr", "FPR (false positive rate)",
         rf"FPR at $\gamma={threshold}$"),
        (axes[1], "fnr", "FNR (false negative rate)",
         rf"FNR at $\gamma={threshold}$"),
    ]:
        for offset, (key, col, lbl) in enumerate([
            ("AABB", C_AABB, "AABB"),
            ("MFMC", C_MFMC, "MFMC"),
            ("Surr", C_SURR, "Surrogate"),
        ]):
            vals = [r[f"{rate}_{key}"] for r in rows]
            ax.bar(x + (offset - 1) * width, vals, width * 0.9,
                   color=col, alpha=0.80, label=lbl, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(dlbls, rotation=30, ha="right",
                           fontsize=max(FS_LABEL - 1, 6))
        ax.set_ylabel(ylabel, fontsize=FS_LABEL)
        ax.set_title(title, fontsize=FS_TITLE)
        _style_ax(ax)

    axes[0].legend(fontsize=FS_LEGEND)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"threshold_by_domain.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: threshold_by_domain.png/pdf")


# ═════════════════════════════════════════════════════════════════════════════
# Paper figures — response surfaces, domain Sobol heatmap, conditional sensitivity
# ═════════════════════════════════════════════════════════════════════════════

# ── Shared helpers for full 2D/3D landscape figures ──────────────────────────

_SURF_META = {
    "scale_factor":         {"label": r"Scale $s_u$",          "lo": 0.1,  "hi": 3.0},
    "center_delta":         {"label": r"Center $\Delta c_u$",  "lo": -1.0, "hi": 1.0},
    "correlation_strength": {"label": r"Corr. $\rho_u$",       "lo": 0.0,  "hi": 0.95},
}

_SURF_SLICE = {
    # fixed_param → (operator, threshold, description)
    "correlation_strength": ("<",  0.10,         r"$\rho_u < 0.10$"),
    "center_delta":         ("ab", 0.05,         r"$|\Delta c_u| < 0.05$"),
    "scale_factor":         ("rng",(0.85, 1.15), r"$s_u \in [0.85,1.15]$"),
}

_SURF_PAIRS = [
    ("scale_factor",  "center_delta",         "correlation_strength"),
    ("scale_factor",  "correlation_strength",  "center_delta"),
    ("center_delta",  "correlation_strength",  "scale_factor"),
]

_SURF_FNAMES_2D = [
    "surface_scale_center",
    "surface_scale_corr",
    "surface_center_corr",
]

_SURF_FNAMES_3D = [
    "landscape_scale_center",
    "landscape_scale_corr",
    "landscape_center_corr",
]


def _interp_surf(pts, vals, XX, YY, sigma=2.0):
    """Interpolate scattered (x,y)→val onto meshgrid XX,YY; smooth and clip."""
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter
    ZZ = griddata(pts, vals, (XX, YY), method="linear")
    mask = np.isnan(ZZ)
    if mask.any():
        ZZ[mask] = griddata(pts, vals, (XX[mask], YY[mask]), method="nearest")
    ZZ = np.clip(np.nan_to_num(ZZ, nan=float(np.nanmean(vals))), 0.0, 1.0)
    if sigma > 0:
        ZZ = np.clip(gaussian_filter(ZZ, sigma=sigma), 0.0, 1.0)
    return ZZ


def _collect_slices(sobol_scenarios, xparam, yparam, fixed_param):
    """Aggregate Saltelli rows by domain, applying slice condition on fixed_param.

    Returns dict: domain → {"x", "y", "mc", "surr"} (lists of float).
    """
    op, thresh, _ = _SURF_SLICE[fixed_param]
    by_dom: dict = {}

    for sc in sobol_scenarios:
        rows   = sc["rows"]              if isinstance(sc, dict) else sc
        domain = sc.get("domain", "Unknown") if isinstance(sc, dict) else "Unknown"
        if domain not in by_dom:
            by_dom[domain] = {"x": [], "y": [], "mc": [], "surr": []}
        d = by_dom[domain]
        for r in rows:
            fval = r.get(fixed_param)
            if fval is None:
                continue
            fval = float(fval)
            if op == "<"   and fval >= thresh:                    continue
            if op == "ab"  and abs(fval) >= thresh:               continue
            if op == "rng" and not (thresh[0] <= fval <= thresh[1]): continue
            try:
                x  = float(r[xparam]);  y  = float(r[yparam])
                mc = float(r["i_theta"]); su = float(r["i_surr"])
            except (KeyError, TypeError, ValueError):
                continue
            if not all(np.isfinite([x, y, mc, su])):
                continue
            d["x"].append(x);   d["y"].append(y)
            d["mc"].append(mc); d["surr"].append(su)

    return by_dom


def _geometry_label(z_mc: np.ndarray) -> str:
    """Derive a short geometry descriptor from MC inconsistency values."""
    z = z_mc[np.isfinite(z_mc)]
    if len(z) == 0:
        return "—"
    frac_above = float((z > 0.5).mean())
    std_z      = float(z.std())
    if std_z > 0.30:
        return "sharp transition"
    if frac_above > 0.65:
        return "broad basin"
    if frac_above < 0.20:
        return "consistent regime"
    return "shifted basin"


def plot_response_surfaces(sobol_scenarios: list, out_dir: Path,
                           n_scenarios: int = 2):
    """2D inconsistency landscape for the top Saltelli params (MC vs Surrogate).

    ICDM style:
    - Two representative scenarios (rows), MC | Surrogate (columns).
    - Identical colormap (RdYlBu_r) and scale [0,1] for both columns so the
      reader can compare geometry directly.
    - Shared colorbar per row (between MC and Surrogate).
    - Threshold contour (γ) drawn thick on both panels.
    - Descriptive subtitle (broad basin / shifted basin / sharp transition).
    """
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter

    if not sobol_scenarios:
        print("\n[Paper] No sobol scenarios — skipping response surfaces.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Paper] Response surfaces ({n_scenarios} scenarios)...")

    # Pick the two scenarios with the most data, from different domains
    def _nrows(sc):
        return len(sc["rows"] if isinstance(sc, dict) else sc)

    sorted_sc = sorted(sobol_scenarios, key=_nrows, reverse=True)
    chosen, seen_doms = [], set()
    for sc in sorted_sc:
        dom = sc.get("domain", "") if isinstance(sc, dict) else ""
        if dom not in seen_doms or len(chosen) == 0:
            chosen.append(sc)
            seen_doms.add(dom)
        if len(chosen) >= n_scenarios:
            break

    p1, p2  = PARAMS[0], PARAMS[1]   # scale_factor, center_delta
    CMAP    = "RdYlBu_r"
    LEVELS  = np.linspace(0, 1, 38)
    GAMMA   = 0.5

    fig, axes = plt.subplots(
        len(chosen), 2,
        figsize=(8.8, 3.8 * len(chosen)),
        gridspec_kw={"wspace": 0.06, "hspace": 0.42},
        squeeze=False,
    )

    for row_idx, sc in enumerate(chosen):
        rows        = sc["rows"] if isinstance(sc, dict) else sc
        domain_name = sc.get("domain", "") if isinstance(sc, dict) else ""

        x    = np.array([r[p1]        for r in rows], dtype=float)
        y    = np.array([r[p2]        for r in rows], dtype=float)
        z_mc = np.array([r["i_theta"] for r in rows], dtype=float)
        z_su = np.array([r["i_surr"]  for r in rows], dtype=float)

        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z_mc)
        geom  = _geometry_label(z_mc[valid])

        xi = np.linspace(float(x[valid].min()), float(x[valid].max()), 80)
        yi = np.linspace(float(y[valid].min()), float(y[valid].max()), 80)
        XX, YY = np.meshgrid(xi, yi)

        def _surface(z):
            try:
                ZZ = griddata((x[valid], y[valid]), z[valid], (XX, YY), method="linear")
                fill = np.nan_to_num(ZZ, nan=float(np.nanmean(z[valid])))
                return gaussian_filter(fill, sigma=1.2)
            except Exception:
                return np.full_like(XX, float(np.nanmean(z[valid])))

        ZZ_mc = _surface(z_mc)
        ZZ_su = _surface(z_su)

        cf_mc = axes[row_idx, 0].contourf(
            XX, YY, ZZ_mc, levels=LEVELS, cmap=CMAP, vmin=0, vmax=1, extend="both")
        cf_su = axes[row_idx, 1].contourf(
            XX, YY, ZZ_su, levels=LEVELS, cmap=CMAP, vmin=0, vmax=1, extend="both")

        for col_idx, (ax, ZZ, col_title) in enumerate([
            (axes[row_idx, 0], ZZ_mc, rf"MC $I_\theta$"),
            (axes[row_idx, 1], rf"Surrogate $\hat{{I}}$", ZZ_su),
        ]):
            # unpack swapped tuple above
            if col_idx == 1:
                ax, col_title = axes[row_idx, 1], rf"Surrogate $\hat{{I}}$"
                ZZ = ZZ_su

            ax.contour(XX, YY, ZZ, levels=[GAMMA],
                       colors=["#111"], linewidths=[2.4], linestyles=["--"])
            ax.scatter(x[valid], y[valid], s=3, color="k",
                       alpha=0.10, linewidths=0, rasterized=True)
            ax.set_xlabel(PARAM_LABELS[p1], fontsize=FS_LABEL)
            ax.set_xlim(xi[0], xi[-1])
            ax.set_ylim(yi[0], yi[-1])
            ax.set_title(col_title, fontsize=FS_TITLE)
            ax.spines[["top", "right"]].set_visible(False)
            if col_idx == 0:
                ax.set_ylabel(PARAM_LABELS[p2], fontsize=FS_LABEL)
            else:
                ax.set_yticklabels([])

        # Shared colorbar between MC and Surrogate for this row
        fig.colorbar(cf_su, ax=[axes[row_idx, 0], axes[row_idx, 1]],
                     shrink=0.88, pad=0.02,
                     label=r"$I(\theta)$").ax.tick_params(labelsize=FS_TICK)

        # Row label: domain + geometry descriptor
        axes[row_idx, 0].set_ylabel(
            f"{domain_name}\n({geom})\n" + PARAM_LABELS[p2],
            fontsize=FS_LABEL)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"response_surfaces.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: response_surfaces.png/pdf")


def plot_response_surfaces_full(sobol_scenarios: list, out_dir: Path,
                                gamma: float = 0.5,
                                cols_per_page: int = 3,
                                n_grid: int = 80):
    """Full 2D response surfaces: all 3 param pairs × all domains, MC vs Surrogate.

    Layout per page: 2 rows (top=MC I_theta, bottom=Surrogate) × cols_per_page domains.
    Three sets of paginated figures, one per parameter pair.
    """
    from collections import defaultdict

    if not sobol_scenarios:
        print("\n[Q3] No sobol scenarios — skipping full response surfaces.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n[Q3] Full 2D response surfaces (MC vs Surrogate, all domains)...")

    # collect domain order
    seen_domains = []
    seen_set = set()
    for sc in sobol_scenarios:
        d = sc.get("domain", "Unknown") if isinstance(sc, dict) else "Unknown"
        if d not in seen_set:
            seen_domains.append(d)
            seen_set.add(d)

    cmap_mc   = "RdYlBu_r"
    cmap_surr = "PuRd"
    levels    = np.linspace(0, 1, 64)

    for (xparam, yparam, fixed_param), fname_base in zip(_SURF_PAIRS, _SURF_FNAMES_2D):
        _, _, slice_desc = _SURF_SLICE[fixed_param]
        meta_x = _SURF_META[xparam]
        meta_y = _SURF_META[yparam]

        by_dom = _collect_slices(sobol_scenarios, xparam, yparam, fixed_param)

        xi = np.linspace(meta_x["lo"], meta_x["hi"], n_grid)
        yi = np.linspace(meta_y["lo"], meta_y["hi"], n_grid)
        XX, YY = np.meshgrid(xi, yi)

        pages = [seen_domains[i:i + cols_per_page]
                 for i in range(0, len(seen_domains), cols_per_page)]

        for pg_idx, page_doms in enumerate(pages):
            n_cols = len(page_doms)
            fig, axes = plt.subplots(
                2, n_cols,
                figsize=(max(5.0, n_cols * 4.0 + 1.2), 7.8),
                squeeze=False,
                sharey="row",
            )

            for col_i, dom in enumerate(page_doms):
                d = by_dom.get(dom, {"x": [], "y": [], "mc": [], "surr": []})
                pts = np.column_stack([d["x"], d["y"]]) if d["x"] else np.empty((0, 2))

                for row_i, (zvals, cmap, row_label) in enumerate([
                    (d["mc"],   cmap_mc,   r"MC $I_\theta$"),
                    (d["surr"], cmap_surr, r"Surrogate $\hat{I}$"),
                ]):
                    ax = axes[row_i][col_i]
                    ax.spines[["top", "right"]].set_visible(False)

                    if len(d["x"]) >= 30:
                        ZZ = _interp_surf(pts, np.array(zvals), XX, YY)
                        cf = ax.contourf(XX, YY, ZZ, levels=levels, cmap=cmap,
                                         vmin=0, vmax=1, extend="both")
                        ax.contour(XX, YY, ZZ, levels=[gamma],
                                   colors=["#222222"], linewidths=[1.8], linestyles=["--"])
                        ax.scatter(d["x"], d["y"], s=2, color="k",
                                   alpha=0.12, linewidths=0, rasterized=True)
                        if col_i == n_cols - 1:
                            fig.colorbar(cf, ax=ax, shrink=0.85,
                                         label=r"$I(\theta)$")
                    else:
                        ax.text(0.5, 0.5,
                                f"Insufficient\ndata ({len(d['x'])} pts)",
                                ha="center", va="center",
                                transform=ax.transAxes, fontsize=8, color="#888")

                    ax.set_xlim(meta_x["lo"], meta_x["hi"])
                    ax.set_ylim(meta_y["lo"], meta_y["hi"])
                    ax.set_xlabel(meta_x["label"], fontsize=FS_LABEL)
                    if col_i == 0:
                        ax.set_ylabel(f"{row_label}\n{meta_y['label']}", fontsize=FS_LABEL)

                    # domain label on top row only
                    if row_i == 0:
                        ax.set_title(dom, fontsize=FS_TITLE, fontweight="bold")

            page_suf = f"_page{pg_idx + 1}" if len(pages) > 1 else ""
            fig.text(0.5, 0.01, f"Fixed: {slice_desc}",
                     ha="center", fontsize=7.5, color="#555555")
            fig.tight_layout(rect=[0, 0.03, 1, 1])
            fname = f"{fname_base}{page_suf}"
            for ext in ("png", "pdf"):
                fig.savefig(out_dir / f"{fname}.{ext}", dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {fname}.png/pdf  ({len(d['x'])} pts, {n_cols} domains)")

    print("  Done: full response surfaces.")


def plot_phase_diagrams(sobol_scenarios: list, out_dir: Path,
                        gamma: float = 0.5,
                        cols_per_page: int = 3,
                        n_grid: int = 55):
    """3D response landscape: all 3 param pairs × all domains (surrogate).

    Shows I_surr as a 3D surface, with a semi-transparent gamma plane
    and a γ-contour projected onto the base.
    Three sets of paginated figures, one per parameter pair.
    """
    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:
        print("[Q3] mpl_toolkits not available — skipping phase diagrams.")
        return

    if not sobol_scenarios:
        print("\n[Q3] No sobol scenarios — skipping phase diagrams.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n[Q3] 3D phase diagrams (Surrogate, all domains)...")

    seen_domains = []
    seen_set = set()
    for sc in sobol_scenarios:
        d = sc.get("domain", "Unknown") if isinstance(sc, dict) else "Unknown"
        if d not in seen_set:
            seen_domains.append(d)
            seen_set.add(d)

    def _plot3d(ax, xs, ys, zs, title, xlabel, ylabel, xlim, ylim):
        if len(xs) < 30:
            ax.set_title(f"{title}\n(insuf. {len(xs)} pts)", fontsize=8)
            return None
        xi = np.linspace(xlim[0], xlim[1], n_grid)
        yi = np.linspace(ylim[0], ylim[1], n_grid)
        XX, YY = np.meshgrid(xi, yi)
        pts = np.column_stack([xs, ys])
        ZZ  = _interp_surf(pts, np.array(zs), XX, YY)

        surf = ax.plot_surface(XX, YY, ZZ, cmap="RdYlGn_r",
                               vmin=0, vmax=1, alpha=0.88,
                               linewidth=0, antialiased=True,
                               rcount=40, ccount=40)
        # gamma plane
        ax.plot_surface(XX, YY, np.full_like(ZZ, gamma),
                        color="#666", alpha=0.12, linewidth=0)
        try:
            ax.contour(XX, YY, ZZ, levels=[gamma], zdir="z", offset=0.0,
                       colors=["#111"], linewidths=[1.4], linestyles=["--"])
        except Exception:
            pass
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(0, 1)
        ax.set_xlabel(xlabel, fontsize=7, labelpad=2)
        ax.set_ylabel(ylabel, fontsize=7, labelpad=2)
        ax.set_zlabel(r"$\hat{I}$", fontsize=7, labelpad=2)
        ax.set_title(title, fontsize=FS_TITLE - 0.5, pad=3)
        ax.tick_params(labelsize=6, pad=1)
        ax.view_init(elev=24, azim=-55)
        try:
            ax.set_box_aspect([1.8, 0.7, 1.0])
        except AttributeError:
            pass
        return surf

    for (xparam, yparam, fixed_param), fname_base in zip(_SURF_PAIRS, _SURF_FNAMES_3D):
        _, _, slice_desc = _SURF_SLICE[fixed_param]
        meta_x = _SURF_META[xparam]
        meta_y = _SURF_META[yparam]

        by_dom = _collect_slices(sobol_scenarios, xparam, yparam, fixed_param)

        pages = [seen_domains[i:i + cols_per_page]
                 for i in range(0, len(seen_domains), cols_per_page)]

        for pg_idx, page_doms in enumerate(pages):
            n_cols = len(page_doms)
            fig = plt.figure(figsize=(max(5.5, n_cols * 4.6 + 1.2), 4.8))
            axes = [fig.add_subplot(1, n_cols, i + 1, projection="3d")
                    for i in range(n_cols)]
            surf_last = None

            for ax, dom in zip(axes, page_doms):
                d = by_dom.get(dom, {"x": [], "y": [], "surr": []})
                s = _plot3d(ax, d["x"], d["y"], d["surr"], dom,
                            meta_x["label"], meta_y["label"],
                            (meta_x["lo"], meta_x["hi"]),
                            (meta_y["lo"], meta_y["hi"]))
                if s is not None:
                    surf_last = s

            if surf_last is not None:
                cbar_ax = fig.add_axes([0.93, 0.20, 0.014, 0.62])
                cb = fig.colorbar(surf_last, cax=cbar_ax)
                cb.set_label(r"$\hat{I}$ (Surrogate)", fontsize=8)
                cb.set_ticks([0.0, 0.5, gamma, 1.0])
                cb.set_ticklabels(["0", "0.5", rf"$\gamma$={gamma}", "1"], fontsize=7)
                cb.ax.axhline(gamma, color="#333", lw=1.2, ls="--")

            fig.text(0.5, 0.01, f"Fixed: {slice_desc}",
                     ha="center", fontsize=7.5, color="#555")
            fig.subplots_adjust(left=0.03, right=0.92,
                                top=0.95, bottom=0.06, wspace=0.05)
            page_suf = f"_page{pg_idx + 1}" if len(pages) > 1 else ""
            fname = f"{fname_base}{page_suf}"
            for ext in ("png", "pdf"):
                fig.savefig(out_dir / f"{fname}.{ext}", dpi=300)
            plt.close(fig)
            print(f"  Saved: {fname}.png/pdf")

    print("  Done: phase diagrams.")


def plot_landscape_a_compare(sobol_scenarios: list, out_dir: Path,
                             sc_ids: list, gamma: float = 0.5,
                             n_grid: int = 80):
    """Generate paper_landscape_a side-by-side for multiple scenarios.

    Produces one row per scenario (MC | Surrogate), saved as
    paper_landscape_a_compare.png/pdf.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Compare] Landscape-A comparison for scenarios {sc_ids}...")

    xparam, yparam, fixed_param = _SURF_PAIRS[0]
    meta_x = _SURF_META[xparam]
    meta_y = _SURF_META[yparam]
    op, thresh, slice_desc = _SURF_SLICE[fixed_param]

    # Build per-scenario slices keyed by scenario_id
    sc_map = {}
    for sc in sobol_scenarios:
        sc_id  = sc.get("scenario_id", -1) if isinstance(sc, dict) else -1
        rows   = sc["rows"] if isinstance(sc, dict) else sc
        domain = sc.get("domain", "?") if isinstance(sc, dict) else "?"
        xs, ys, mcs, sus = [], [], [], []
        for r in rows:
            fval = r.get(fixed_param)
            if fval is None: continue
            fval = float(fval)
            if op == "<"   and fval >= thresh:                       continue
            if op == "ab"  and abs(fval) >= thresh:                  continue
            if op == "rng" and not (thresh[0] <= fval <= thresh[1]): continue
            try:
                xs.append(float(r[xparam]));  ys.append(float(r[yparam]))
                mcs.append(float(r["i_theta"])); sus.append(float(r["i_surr"]))
            except (KeyError, TypeError, ValueError):
                continue
        if len(xs) >= 10 and sc_id not in sc_map:
            # First-wins: if two files share the same numeric id (e.g. both
            # measurements_v6 and measurements_cps_v6 have results_scenario_1.json)
            # keep the first one encountered, which matches find_best_landscape_a.
            sc_map[sc_id] = {"domain": domain, "x": xs, "y": ys,
                             "mc": mcs, "surr": sus}

    picks = []
    for sid in sc_ids:
        if sid in sc_map:
            picks.append((sid, sc_map[sid]))
        else:
            print(f"  [warn] Sc {sid} not found or too few points — skipping.")

    if not picks:
        print("  No valid scenarios — skipping compare figure.")
        return

    xi     = np.linspace(meta_x["lo"], meta_x["hi"], n_grid)
    yi     = np.linspace(meta_y["lo"], meta_y["hi"], n_grid)
    XX, YY = np.meshgrid(xi, yi)
    LEVELS = np.linspace(0, 1, 38)
    CMAP   = "RdYlBu_r"

    n_rows = len(picks)
    fig, axes = plt.subplots(n_rows, 2,
                             figsize=(9.0, 3.8 * n_rows),
                             gridspec_kw={"wspace": 0.08, "hspace": 0.42},
                             sharey="row")

    if n_rows == 1:
        axes = axes[np.newaxis, :]

    from scipy.stats import spearmanr as _sp

    for row_i, (sid, sc) in enumerate(picks):
        pts   = np.column_stack([sc["x"], sc["y"]])
        ZZ_mc = _interp_surf(pts, np.array(sc["mc"]),   XX, YY)
        ZZ_su = _interp_surf(pts, np.array(sc["surr"]), XX, YY)
        rho   = float(_sp(sc["mc"], sc["surr"]).statistic)

        ax_mc   = axes[row_i, 0]
        ax_surr = axes[row_i, 1]

        cf_mc = ax_mc.contourf(XX, YY, ZZ_mc, levels=LEVELS, cmap=CMAP,
                               vmin=0, vmax=1, extend="both")
        cf_su = ax_surr.contourf(XX, YY, ZZ_su, levels=LEVELS, cmap=CMAP,
                                 vmin=0, vmax=1, extend="both")

        for ax, ZZ, col_title in [
            (ax_mc,   ZZ_mc, rf"MC $I_\theta$"),
            (ax_surr, ZZ_su, rf"Surrogate $\hat{{I}}$"),
        ]:
            ax.contour(XX, YY, ZZ, levels=[gamma],
                       colors=["#111"], linewidths=[2.8], linestyles=["--"])
            ax.scatter(sc["x"], sc["y"], s=3, color="k",
                       alpha=0.12, linewidths=0, rasterized=True)
            ax.set_xlabel(meta_x["label"], fontsize=FS_LABEL)
            ax.set_title(col_title, fontsize=FS_TITLE)
            ax.set_xlim(meta_x["lo"], meta_x["hi"])
            ax.set_ylim(meta_y["lo"], meta_y["hi"])
            ax.spines[["top", "right"]].set_visible(False)

        # Shared colorbar between MC and Surrogate for this row
        fig.colorbar(cf_su, ax=[ax_mc, ax_surr], shrink=0.88, pad=0.02,
                     label=r"$I(\theta)$").ax.tick_params(labelsize=FS_TICK)

        # Row label with domain
        ax_mc.set_ylabel(f"Sc {sid} · {sc['domain']}\n" + meta_y["label"],
                         fontsize=FS_LABEL)
        ax_surr.set_yticklabels([])

        # Compact ρ annotation inside the Surrogate panel (top-right)
        ax_surr.text(
            0.97, 0.96, rf"$\rho = {rho:.3f}$",
            transform=ax_surr.transAxes,
            va="top", ha="right", fontsize=FS_ANNOT + 1,
            color="#111",
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#aaa",
                      alpha=0.85, lw=0.7),
        )

    fig.text(0.5, 0.01, f"Fixed: {slice_desc}  |  γ={gamma} (dashed)",
             ha="center", fontsize=7.5, color="#555")
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"paper_landscape_a_compare.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: paper_landscape_a_compare.png/pdf  "
          f"({len(picks)} scenarios)")


def find_best_landscape_a(sobol_scenarios: list, gamma: float = 0.5,
                          top_n: int = 15):
    """Score every scenario for panel-A suitability and print a ranked table.

    Scoring criteria (all computed on the scale×center slice):
      rho       — Spearman ρ between MC and surrogate (want high)
      boundary  — fraction of points near the γ boundary, defined as
                  min(frac_below, frac_above) where frac = share on each side
                  (want ~0.5 — boundary cuts through the middle of the plot)
      n         — number of slice points (want large for smooth surface)
      score     — rho * boundary * log(n+1)  (combined rank)
    """
    from scipy.stats import spearmanr as _spear

    xparam, yparam, fixed_param = _SURF_PAIRS[0]
    op, thresh, _ = _SURF_SLICE[fixed_param]

    rows = []
    for sc in sobol_scenarios:
        sc_rows = sc["rows"] if isinstance(sc, dict) else sc
        domain  = sc.get("domain", "?") if isinstance(sc, dict) else "?"
        sc_id   = sc.get("scenario_id", -1) if isinstance(sc, dict) else -1

        xs, mcs, sus = [], [], []
        for r in sc_rows:
            fval = r.get(fixed_param)
            if fval is None: continue
            fval = float(fval)
            if op == "<"   and fval >= thresh:                       continue
            if op == "ab"  and abs(fval) >= thresh:                  continue
            if op == "rng" and not (thresh[0] <= fval <= thresh[1]): continue
            try:
                xs.append(float(r[xparam]))
                mcs.append(float(r["i_theta"]))
                sus.append(float(r["i_surr"]))
            except (KeyError, TypeError, ValueError):
                continue

        n = len(xs)
        if n < 30:
            continue

        mc_a  = np.array(mcs)
        su_a  = np.array(sus)
        rho   = float(_spear(mc_a, su_a).statistic) if n >= 5 else 0.0
        if not np.isfinite(rho): rho = 0.0

        frac_below = float((mc_a < gamma).mean())
        boundary   = min(frac_below, 1.0 - frac_below)   # 0.5 = boundary in centre
        score      = rho * boundary * np.log(n + 1)

        rows.append(dict(sc_id=sc_id, domain=domain, n=n,
                         rho=rho, boundary=boundary, score=score))

    rows.sort(key=lambda r: r["score"], reverse=True)

    print(f"\n{'='*68}")
    print(f"  Landscape-A scenario search  (γ={gamma}, top {top_n})")
    print(f"  Slice: {_SURF_SLICE[fixed_param][2]}")
    print(f"{'='*68}")
    print(f"  {'Sc':>4}  {'Domain':<22}  {'N':>5}  {'ρ':>6}  {'bdry':>5}  {'score':>7}")
    print(f"  {'-'*4}  {'-'*22}  {'-'*5}  {'-'*6}  {'-'*5}  {'-'*7}")
    for r in rows[:top_n]:
        print(f"  {r['sc_id']:>4}  {r['domain']:<22}  {r['n']:>5}  "
              f"{r['rho']:>6.3f}  {r['boundary']:>5.3f}  {r['score']:>7.3f}")
    print(f"{'='*68}")
    if rows:
        best = rows[0]
        print(f"  → Best candidate: Sc {best['sc_id']}  "
              f"({best['domain']})  score={best['score']:.3f}")
    print()
    return rows


def find_best_landscape_b(sobol_scenarios: list, exclude_sc_a: int = None,
                          top_n: int = 15):
    """Score every scenario for panel-B suitability and print a ranked table.

    Panel B is the 3D surface that illustrates HOW WELL the surrogate reproduces
    the Sobol sensitivity structure — not just the predicted values.

    Scoring criterion:
      sobol_mae  — mean over {S1, ST} × {3 params} of |surrogate - MC|
                   (want small — surrogate indices match MC indices)
      N          — number of Saltelli rows (want large)
      score      = (1 - sobol_mae) * log(N+1)   (higher is better)

    Also reports the per-param S1 values so you can see what type of
    sensitivity profile each scenario has.
    """
    try:
        from SALib.analyze import sobol as sobol_analyze
    except ImportError:
        print("  [warn] SALib not installed — cannot rank landscape-B candidates.")
        return []

    problem = {"num_vars": 3, "names": PARAMS, "bounds": [[0, 1]] * 3}

    def _si(Y):
        try:
            si = sobol_analyze.analyze(problem, Y,
                                       calc_second_order=False,
                                       print_to_console=False)
            return (
                {p: max(0.0, float(si["S1"][i])) for i, p in enumerate(PARAMS)},
                {p: max(0.0, float(si["ST"][i])) for i, p in enumerate(PARAMS)},
            )
        except Exception:
            return None, None

    rows_out = []
    for sc in sobol_scenarios:
        sc_rows = sc["rows"] if isinstance(sc, dict) else sc
        domain  = sc.get("domain", "?") if isinstance(sc, dict) else "?"
        sc_id   = sc.get("scenario_id", -1) if isinstance(sc, dict) else -1

        if exclude_sc_a is not None and sc_id == exclude_sc_a:
            continue

        N = len(sc_rows)
        if N < 10 or N % 5 != 0:
            continue

        Y_mc   = np.array([r["i_theta"] for r in sc_rows])
        Y_surr = np.array([r["i_surr"]  for r in sc_rows])

        if np.std(Y_mc) < 1e-9:
            continue  # constant — Sobol undefined

        s1_mc,   st_mc   = _si(Y_mc)
        s1_surr, st_surr = _si(Y_surr)
        if s1_mc is None or s1_surr is None:
            continue

        # Mean absolute error between surrogate Sobol and MC Sobol
        s1_mae = float(np.mean([abs(s1_surr[p] - s1_mc[p]) for p in PARAMS]))
        st_mae = float(np.mean([abs(st_surr[p] - st_mc[p]) for p in PARAMS]))
        sobol_mae = (s1_mae + st_mae) / 2.0

        score = (1.0 - sobol_mae) * np.log(N + 1)

        rows_out.append(dict(
            sc_id=sc_id, domain=domain, N=N,
            sobol_mae=sobol_mae, score=score,
            s1_mc=s1_mc, s1_surr=s1_surr,
        ))

    rows_out.sort(key=lambda r: r["score"], reverse=True)

    # Short param abbreviations for table header
    abbr = {PARAMS[0]: "sc", PARAMS[1]: "cd", PARAMS[2]: "cs"}

    print(f"\n{'='*80}")
    print(f"  Landscape-B scenario search — balanced S1_surr ≈ S1_mc  (top {top_n})")
    print(f"  Columns: sobol_mae = mean|S1_surr-S1_mc| + |ST_surr-ST_mc| / 2")
    print(f"{'='*80}")
    hdr_s1 = "  ".join(f"S1_mc[{abbr[p]}]" for p in PARAMS)
    hdr_d  = "  ".join(f"Δ[{abbr[p]}]   " for p in PARAMS)
    print(f"  {'Sc':>4}  {'Domain':<22}  {'N':>5}  {'s_mae':>6}  {'score':>7}  "
          f"  {hdr_s1}   {hdr_d}")
    print(f"  {'-'*4}  {'-'*22}  {'-'*5}  {'-'*6}  {'-'*7}"
          + "  " + "  ".join(["-"*10]*3) + "  " + "  ".join(["-"*10]*3))
    for r in rows_out[:top_n]:
        s1_str = "  ".join(f"{r['s1_mc'][p]:>10.3f}" for p in PARAMS)
        d_str  = "  ".join(f"{abs(r['s1_surr'][p]-r['s1_mc'][p]):>10.3f}"
                           for p in PARAMS)
        print(f"  {r['sc_id']:>4}  {r['domain']:<22}  {r['N']:>5}  "
              f"{r['sobol_mae']:>6.3f}  {r['score']:>7.3f}  "
              f"  {s1_str}   {d_str}")
    print(f"{'='*80}")
    if rows_out:
        best = rows_out[0]
        print(f"  → Best candidate: Sc {best['sc_id']}  "
              f"({best['domain']})  sobol_mae={best['sobol_mae']:.3f}  "
              f"score={best['score']:.3f}")
    print()
    return rows_out


def plot_paper_landscape(sobol_scenarios: list, out_dir: Path,
                         gamma: float = 0.5,
                         n_grid: int = 80,
                         sc_a: int = None,
                         sc_b: list = None):
    """Curated paper-ready landscape figures (2 panels).

    panel_a — 2D response surface (scale vs center):
        One scenario, one setting: MC I_theta and Surrogate side by side.
        If sc_a is given, uses that scenario id; otherwise picks the one with
        the most data points in the scale/center slice.

    panel_b — 3D phase diagram, different examples:
        Three different scenarios, Surrogate I surface + gamma plane.
        If sc_b is given (list of ints), uses those scenario ids in order;
        otherwise picks the 3 most-data scenarios from different domains,
        excluding panel_a's.
    """
    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        has_3d = True
    except ImportError:
        has_3d = False

    if not sobol_scenarios:
        print("\n[Q3] No sobol scenarios — skipping paper landscape figures.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n[Q3] Paper landscape figures (curated)...")

    # Use scale vs center pair (most interpretable for paper)
    xparam, yparam, fixed_param = _SURF_PAIRS[0]
    meta_x = _SURF_META[xparam]
    meta_y = _SURF_META[yparam]
    _, _, slice_desc = _SURF_SLICE[fixed_param]

    # Per-scenario slice (not aggregated by domain)
    op, thresh, _ = _SURF_SLICE[fixed_param]
    sc_slices = []
    for sc in sobol_scenarios:
        rows   = sc["rows"]              if isinstance(sc, dict) else sc
        domain = sc.get("domain", "Unknown") if isinstance(sc, dict) else "Unknown"
        xs, ys, mcs, sus = [], [], [], []
        for r in rows:
            fval = r.get(fixed_param)
            if fval is None:
                continue
            fval = float(fval)
            if op == "<"   and fval >= thresh:                    continue
            if op == "ab"  and abs(fval) >= thresh:               continue
            if op == "rng" and not (thresh[0] <= fval <= thresh[1]): continue
            try:
                xs.append(float(r[xparam]));  ys.append(float(r[yparam]))
                mcs.append(float(r["i_theta"])); sus.append(float(r["i_surr"]))
            except (KeyError, TypeError, ValueError):
                continue
        sc_slices.append({"domain": domain, "x": xs, "y": ys, "mc": mcs, "surr": sus,
                           "scenario_id": sc.get("scenario_id", -1) if isinstance(sc, dict) else -1})

    # ── Panel A: 2D surface (MC vs Surrogate), best single scenario ──────────
    if sc_a is not None:
        best = next((s for s in sc_slices if s.get("scenario_id") == sc_a), None)
        if best is None:
            print(f"  [warn] sc_a={sc_a} not found — falling back to most-data selection.")
            best = max(sc_slices, key=lambda s: len(s["x"]), default=None)
    else:
        best = max(sc_slices, key=lambda s: len(s["x"]), default=None)

    if best and len(best["x"]) >= 30:
        print(f"  Panel A: scenario={best.get('scenario_id','?')}  "
              f"domain='{best['domain']}'  N={len(best['x'])} pts")
        xi = np.linspace(meta_x["lo"], meta_x["hi"], n_grid)
        yi = np.linspace(meta_y["lo"], meta_y["hi"], n_grid)
        XX, YY = np.meshgrid(xi, yi)
        pts = np.column_stack([best["x"], best["y"]])

        CMAP_AB = "RdYlBu_r"   # same colormap for both panels
        LEVELS  = np.linspace(0, 1, 38)   # ~40% fewer contour lines

        fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharey=True)
        for ax, zvals, title in [
            (axes[0], best["mc"],   rf"MC $I_\theta$  [{best['domain']}]"),
            (axes[1], best["surr"], rf"Surrogate $\hat{{I}}$  [{best['domain']}]"),
        ]:
            ZZ = _interp_surf(pts, np.array(zvals), XX, YY)
            cf = ax.contourf(XX, YY, ZZ, levels=LEVELS, cmap=CMAP_AB,
                             vmin=0, vmax=1, extend="both")
            ax.contour(XX, YY, ZZ, levels=[gamma],
                       colors=["#111"], linewidths=[2.8], linestyles=["--"])
            ax.scatter(best["x"], best["y"], s=3, color="k",
                       alpha=0.15, linewidths=0, rasterized=True)
            fig.colorbar(cf, ax=ax, shrink=0.85, label=r"$I(\theta)$")
            ax.set_xlabel(meta_x["label"], fontsize=FS_LABEL)
            ax.set_ylabel(meta_y["label"], fontsize=FS_LABEL)
            ax.set_title(title, fontsize=FS_TITLE)
            ax.set_xlim(meta_x["lo"], meta_x["hi"])
            ax.set_ylim(meta_y["lo"], meta_y["hi"])
            ax.spines[["top", "right"]].set_visible(False)

        fig.text(0.5, 0.01, f"Fixed: {slice_desc}",
                 ha="center", fontsize=7.5, color="#555")
        fig.tight_layout(rect=[0, 0.03, 1, 1])
        for ext in ("png", "pdf"):
            fig.savefig(out_dir / f"paper_landscape_a.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("  Saved: paper_landscape_a.png/pdf  (2D MC vs Surrogate)")
    else:
        print("  SKIP panel_a: insufficient data")

    # ── Panel B: 3D landscape, 3 different example scenarios ─────────────────
    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        has_3d = True
    except ImportError:
        has_3d = False

    if not has_3d:
        print("  SKIP panel_b: mpl_toolkits not available")
        return

    # First-wins: keep first occurrence when two files share the same numeric id
    sc_id_map = {}
    for s in sc_slices:
        sid = s.get("scenario_id")
        if sid not in sc_id_map:
            sc_id_map[sid] = s

    if sc_b is not None:
        picks = []
        for sid in sc_b:
            sc = sc_id_map.get(sid)
            if sc is None:
                print(f"  [warn] sc_b scenario {sid} not found in slices — skipping.")
            elif len(sc["x"]) < 30:
                print(f"  [warn] sc_b scenario {sid} has too few points ({len(sc['x'])}) — skipping.")
            else:
                picks.append(sc)
    else:
        seen_doms = set()
        if best:
            seen_doms.add(best["domain"])
        picks = []
        for sc in sorted(sc_slices, key=lambda s: len(s["x"]), reverse=True):
            if len(sc["x"]) < 30 or sc is best:
                continue
            if sc["domain"] not in seen_doms or len(picks) < 3:
                picks.append(sc)
                seen_doms.add(sc["domain"])
            if len(picks) == 3:
                break

    if not picks:
        print("  SKIP panel_b: no scenarios with sufficient data")
        return

    for i, sc in enumerate(picks):
        print(f"  Panel B[{i+1}]: scenario={sc.get('scenario_id','?')}  "
              f"domain='{sc['domain']}'  N={len(sc['x'])} pts")

    n_cols = len(picks)
    fig = plt.figure(figsize=(n_cols * 4.6 + 1.2, 4.8))
    surf_last = None

    xi = np.linspace(meta_x["lo"], meta_x["hi"], n_grid // 2 + 10)
    yi = np.linspace(meta_y["lo"], meta_y["hi"], n_grid // 2 + 10)
    XX3, YY3 = np.meshgrid(xi, yi)

    for i, sc in enumerate(picks):
        ax = fig.add_subplot(1, n_cols, i + 1, projection="3d")
        pts = np.column_stack([sc["x"], sc["y"]])
        ZZ  = _interp_surf(pts, np.array(sc["surr"]), XX3, YY3)

        surf = ax.plot_surface(XX3, YY3, ZZ, cmap="RdYlBu_r",
                               vmin=0, vmax=1, alpha=0.90,
                               linewidth=0, antialiased=True,
                               rcount=40, ccount=40)
        # Translucent γ-plane to show the operational threshold
        ax.plot_surface(XX3, YY3, np.full_like(ZZ, gamma),
                        color="#444", alpha=0.10, linewidth=0)
        # γ-contour projected onto z=0 — thicker for visibility
        try:
            ax.contour(XX3, YY3, ZZ, levels=[gamma], zdir="z", offset=0.0,
                       colors=["#111"], linewidths=[2.2], linestyles=["--"])
        except Exception:
            pass

        ax.set_xlim(meta_x["lo"], meta_x["hi"])
        ax.set_ylim(meta_y["lo"], meta_y["hi"])
        ax.set_zlim(0, 1)
        ax.set_xlabel(meta_x["label"], fontsize=7.5, labelpad=3)
        ax.set_ylabel(meta_y["label"], fontsize=7.5, labelpad=3)
        ax.set_zlabel(r"$\hat{I}$",   fontsize=7.5, labelpad=3)
        ax.set_title(sc["domain"], fontsize=FS_TITLE + 0.5, pad=5)
        ax.tick_params(labelsize=6.5, pad=1)
        # Slightly higher elevation reduces distortion and improves readability
        ax.view_init(elev=28, azim=-48)
        try:
            ax.set_box_aspect([1.6, 0.7, 1.0])
        except AttributeError:
            pass
        surf_last = surf

    if surf_last is not None:
        cbar_ax = fig.add_axes([0.93, 0.18, 0.015, 0.64])
        cb = fig.colorbar(surf_last, cax=cbar_ax)
        cb.set_label(r"$\hat{I}$ (Surrogate)", fontsize=FS_LABEL)
        cb.set_ticks([0.0, gamma, 1.0])
        cb.set_ticklabels(["0", rf"$\gamma$={gamma}", "1"], fontsize=FS_TICK)
        cb.ax.axhline(gamma, color="#333", lw=1.4, ls="--")

    fig.text(0.5, 0.005, f"Fixed: {slice_desc}  |  γ={gamma} (dashed contour)",
             ha="center", fontsize=7.5, color="#555")
    fig.subplots_adjust(left=0.02, right=0.92,
                        top=0.94, bottom=0.07, wspace=0.04)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"paper_landscape_b.{ext}", dpi=300)
    plt.close(fig)
    print(f"  Saved: paper_landscape_b.png/pdf  (3D Surrogate, {n_cols} scenarios)")


def plot_sobol_by_domain(sobol_results: list, out_dir: Path,
                          domain_short: dict = None,
                          domain_order: list = None):
    """Heatmap: domain × parameter, coloured by mean S1 and ST (MC + Surrogate).

    Layout (ICDM style):
      Row 0: MC S1  |  Surrogate S1      ← first-order effects
      Row 1: MC ST  |  Surrogate ST      ← total effects (includes interactions)

    A shared colorbar is placed outside the right column.  The dominant
    parameter per domain is bolded to guide the reader's eye.  An annotation
    in the right margin highlights that ST > S1 signals interaction-driven
    inconsistency.
    """
    if not sobol_results:
        print("\n[Paper] No Sobol results — skipping sobol_by_domain.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n[Paper] Sobol by domain heatmap...")

    from collections import defaultdict
    short    = domain_short or {}
    by_dom   = defaultdict(list)
    for r in sobol_results:
        by_dom[r.get("domain", "Unknown")].append(r)

    unique_d = domain_order if domain_order else sorted(by_dom.keys())
    unique_d = [d for d in unique_d if d in by_dom]
    if not unique_d:
        unique_d = sorted(by_dom.keys())

    dlbls = [short.get(d, d) for d in unique_d]
    plbls = [PARAM_LABELS[p] for p in PARAMS]

    # Row 0: S1 pair;  Row 1: ST pair — emphasises interaction gap visually
    configs = [
        (0, 0, "S1_mc",   r"MC — first-order $S_1$"),
        (0, 1, "S1_surr", r"Surrogate — first-order $S_1$"),
        (1, 0, "ST_mc",   r"MC — total-effect $S_T$"),
        (1, 1, "ST_surr", r"Surrogate — total-effect $S_T$"),
    ]

    # Compute all four matrices first so we can share a single colorbar
    Z_dict = {}
    for _, _, key, _ in configs:
        Z_dict[key] = np.array([
            [float(np.mean([r[key][p] for r in by_dom[d]])) for p in PARAMS]
            for d in unique_d
        ])

    n_dom = len(unique_d)
    row_h = max(0.55 * n_dom, 3.2)
    fig, axes = plt.subplots(
        2, 2,
        figsize=(9.8, row_h * 2 + 0.8),
        gridspec_kw={"hspace": 0.28, "wspace": 0.08},
    )

    im_last = None
    for row, col, key, title in configs:
        ax = axes[row, col]
        Z  = Z_dict[key]

        im = ax.imshow(Z, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
        im_last = im

        ax.set_xticks(np.arange(len(PARAMS)))
        ax.set_xticklabels(plbls, fontsize=FS_TICK)

        if col == 0:
            ax.set_yticks(np.arange(n_dom))
            ax.set_yticklabels(dlbls, fontsize=FS_TICK)
        else:
            ax.set_yticks(np.arange(n_dom))
            ax.set_yticklabels([], fontsize=FS_TICK)

        # Find dominant parameter per domain (per row)
        dom_max_col = Z.argmax(axis=1)   # column index of max per domain

        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                v    = Z[i, j]
                clr  = "white" if v > 0.60 else "black"
                bold = "bold" if j == dom_max_col[i] else "normal"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=FS_ANNOT - 0.5, color=clr, fontweight=bold)

        ax.set_title(title, fontsize=FS_TITLE, pad=4)

        # Horizontal divider between S1 rows and ST rows
        if row == 0:
            ax.spines["bottom"].set_linewidth(1.8)
        ax.spines[["top", "right"]].set_visible(False)

    # Single shared colorbar outside the right column
    fig.subplots_adjust(right=0.87)
    cbar_ax = fig.add_axes([0.89, 0.12, 0.018, 0.76])
    cb = fig.colorbar(im_last, cax=cbar_ax)
    cb.set_label("Mean Sobol index", fontsize=FS_LABEL)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cb.ax.tick_params(labelsize=FS_TICK)

    # Right-margin annotation: ST > S1 ⟹ interaction effects
    fig.text(
        0.995, 0.32,
        r"$S_T > S_1$" + "\n⟹ interactions",
        fontsize=FS_ANNOT - 0.5, color="#555", ha="right", va="center",
        style="italic",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#bbb", lw=0.6),
    )

    # Row labels in the far left margin
    for row_i, lbl in enumerate([r"First-order ($S_1$)", r"Total-effect ($S_T$)"]):
        axes[row_i, 0].set_ylabel(lbl, fontsize=FS_LABEL, labelpad=6)

    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"sobol_by_domain.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: sobol_by_domain.png/pdf")


def plot_conditional_sensitivity(sobol_scenarios: list, out_dir: Path,
                                  n_bins: int = 4, n_boot: int = 2000,
                                  rng_seed: int = 0):
    """Sobol S1/ST/interaction by inconsistency regime (scenario-level binning).

    Replaces the previous per-sample subsetting approach, which broke the
    Saltelli sample structure required by Sobol estimators.

    Method
    ------
    1. Compute S1 and ST from each scenario's *intact* Saltelli design
       (N % 5 == 0 guard preserved from _sobol_per_scenario).
    2. Bin *scenarios* by their median I_theta into ``n_bins`` quantile
       groups (Low → Saturated).  This preserves sample structure within
       every scenario; only the aggregation is regime-conditional.
    3. Bootstrap *across scenarios* within each bin (B = n_boot resamples)
       to obtain 95 % CIs for S1, ST, and the interaction proxy ST − S1.
    4. Plot as disconnected dot + error-bar panels (3 panels: S1 | ST |
       ST − S1).  X-axis is categorical; connecting lines are omitted.
    5. Regime legend entries include the quantile interval [lo, hi] and n.

    Label
    -----
    Titled "Sobol indices by I_θ regime" — valid because Saltelli structure
    is respected per scenario; only regime *aggregation* is conditional.
    """
    try:
        from SALib.analyze import sobol as sobol_analyze
    except ImportError:
        print("\n[Paper] SALib not found — skipping conditional sensitivity.")
        return

    if not sobol_scenarios:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Paper] Conditional sensitivity — scenario-level binning "
          f"({n_bins} regimes, {n_boot} bootstrap resamples)...")

    problem = {"num_vars": 3, "names": PARAMS, "bounds": [[0, 1]] * 3}

    def _si_intact(rows):
        """Sobol on an intact Saltelli block; returns (s1, st) dicts or (None, None)."""
        N = len(rows)
        if N < 10 or N % 5 != 0:
            return None, None
        try:
            Y = np.array([r["i_theta"] for r in rows], dtype=float)
            if np.std(Y) < 1e-9:
                return None, None
            si = sobol_analyze.analyze(problem, Y,
                                       calc_second_order=False,
                                       print_to_console=False)
            s1 = {p: max(0.0, float(si["S1"][i])) for i, p in enumerate(PARAMS)}
            st = {p: max(0.0, float(si["ST"][i])) for i, p in enumerate(PARAMS)}
            return s1, st
        except Exception:
            return None, None

    def _si_surr_intact(rows):
        N = len(rows)
        if N < 10 or N % 5 != 0:
            return None, None
        try:
            Y = np.array([r["i_surr"] for r in rows], dtype=float)
            if np.std(Y) < 1e-9:
                return None, None
            si = sobol_analyze.analyze(problem, Y,
                                       calc_second_order=False,
                                       print_to_console=False)
            s1 = {p: max(0.0, float(si["S1"][i])) for i, p in enumerate(PARAMS)}
            st = {p: max(0.0, float(si["ST"][i])) for i, p in enumerate(PARAMS)}
            return s1, st
        except Exception:
            return None, None

    # ── Step 1: per-scenario Sobol + median I_theta ───────────────────────────
    sc_records = []
    for sc in sobol_scenarios:
        rows   = sc["rows"] if isinstance(sc, dict) else sc
        domain = sc.get("domain", "Unknown") if isinstance(sc, dict) else "Unknown"
        if not rows:
            continue
        med_i = float(np.median([r["i_theta"] for r in rows
                                 if np.isfinite(r["i_theta"])]))
        s1_mc, st_mc   = _si_intact(rows)
        s1_su, st_su   = _si_surr_intact(rows)
        if s1_mc is None or s1_su is None:
            continue
        sc_records.append(dict(med_i=med_i, s1_mc=s1_mc, st_mc=st_mc,
                                s1_su=s1_su, st_su=st_su, domain=domain))

    if len(sc_records) < n_bins * 2:
        print(f"  [warn] Only {len(sc_records)} valid scenarios — "
              "not enough for regime analysis.")
        return

    # ── Step 2: bin scenarios by median I_theta quartiles ────────────────────
    med_vals = np.array([r["med_i"] for r in sc_records])
    bin_edges = np.quantile(med_vals, np.linspace(0, 1, n_bins + 1))
    bin_edges[0]  -= 1e-9
    bin_edges[-1] += 1e-9

    _regime_names = ["Low", "Mid", "High", "Saturated"]
    bins_data = []
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        members = [r for r in sc_records if lo < r["med_i"] <= hi]
        if not members:
            continue
        bins_data.append(dict(lo=lo, hi=hi, members=members,
                              label=_regime_names[b] if b < len(_regime_names)
                              else f"Bin {b}"))

    if not bins_data:
        print("  [warn] Empty bins after scenario grouping.")
        return

    # ── Step 3: bootstrap S1, ST, ST−S1 across scenarios per bin ────────────
    rng = np.random.default_rng(rng_seed)

    def _boot_stats(members, key_s1, key_st, n_boot):
        """Bootstrap mean ± 95 % CI for S1, ST, ST-S1 over scenario list."""
        n = len(members)
        s1_boot = np.zeros((n_boot, len(PARAMS)))
        st_boot = np.zeros((n_boot, len(PARAMS)))
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            s1_boot[b] = [np.mean([members[i][key_s1][p] for i in idx])
                          for p in PARAMS]
            st_boot[b] = [np.mean([members[i][key_st][p] for i in idx])
                          for p in PARAMS]
        inter_boot = np.clip(st_boot - s1_boot, 0, None)
        def _ci(arr):
            mn  = arr.mean(axis=0)
            lo  = np.percentile(arr, 2.5,  axis=0)
            hi  = np.percentile(arr, 97.5, axis=0)
            return mn, lo, hi
        return _ci(s1_boot), _ci(st_boot), _ci(inter_boot)

    bin_stats_mc   = []
    bin_stats_surr = []
    for bd in bins_data:
        mc_res   = _boot_stats(bd["members"], "s1_mc", "st_mc",   n_boot)
        surr_res = _boot_stats(bd["members"], "s1_su", "st_su",   n_boot)
        bin_stats_mc.append(mc_res)
        bin_stats_surr.append(surr_res)

    # ── Step 4: three-panel dot + CI figure ───────────────────────────────────
    from matplotlib.lines import Line2D

    _ramp       = ["#b3cde3", "#6baed6", "#2171b5", "#084594"]
    regime_cols = [_ramp[i % len(_ramp)] for i in range(len(bins_data))]

    n_p   = len(PARAMS)
    x_pos = np.arange(n_p, dtype=float)

    # Horizontal jitter per regime so dots don't stack
    n_reg  = len(bins_data)
    jitter = np.linspace(-0.18, 0.18, n_reg) if n_reg > 1 else [0.0]

    panel_titles = [
        r"First-order $S_1$",
        r"Total-effect $S_T$",
        r"Interaction $S_T - S_1$",
    ]

    fig, axes = plt.subplots(
        1, 3,
        figsize=(13.0, 4.4),
        gridspec_kw={"wspace": 0.36},
    )

    for panel_idx, (ax, title) in enumerate(zip(axes, panel_titles)):
        for b_idx, (bd, col, jit) in enumerate(
                zip(bins_data, regime_cols, jitter)):
            mc_res, surr_res = bin_stats_mc[b_idx], bin_stats_surr[b_idx]
            mn_mc,  lo_mc,  hi_mc  = mc_res[panel_idx]
            mn_su,  lo_su,  hi_su  = surr_res[panel_idx]

            xj = x_pos + jit
            # MC — filled circle
            ax.errorbar(
                xj, mn_mc,
                yerr=[mn_mc - lo_mc, hi_mc - mn_mc],
                fmt="o", color=col, markersize=5.5,
                capsize=3, elinewidth=1.0, lw=0,
                zorder=5 + b_idx, label=None,
            )
            # Surrogate — open triangle, same colour, slight inner jitter
            ax.errorbar(
                xj + 0.04, mn_su,
                yerr=[mn_su - lo_su, hi_su - mn_su],
                fmt="^", color=col, markersize=4.5,
                capsize=3, elinewidth=0.8, lw=0,
                markerfacecolor="none", markeredgewidth=1.2,
                zorder=5 + b_idx, label=None,
            )

        ax.axhline(0, color="#aaa", lw=0.7, ls="--", zorder=1)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([PARAM_LABELS[p] for p in PARAMS],
                           fontsize=FS_LABEL)
        ax.set_ylim(-0.05, 1.08)
        ax.set_ylabel("Sobol index" if panel_idx == 0 else "",
                      fontsize=FS_LABEL)
        ax.set_title(title, fontsize=FS_TITLE)
        _style_ax(ax)

    # Legend: regime rows (colour) + estimator column (marker style)
    regime_handles = [
        Line2D([0], [0], color=regime_cols[i], marker="o", markersize=5,
               lw=0,
               label=(f"{bd['label']}  "
                      f"[{bd['lo']:.2f}, {bd['hi']:.2f}]  "
                      f"n={len(bd['members'])}"))
        for i, bd in enumerate(bins_data)
    ]
    style_handles = [
        Line2D([0], [0], color="#555", marker="o", markersize=5,
               lw=0, label="MC  (filled ●)"),
        Line2D([0], [0], color="#555", marker="^", markersize=5,
               lw=0, markerfacecolor="none",
               markeredgewidth=1.2, label="Surrogate  (open ▲)"),
    ]
    axes[0].legend(
        handles=regime_handles + style_handles,
        fontsize=FS_LEGEND - 0.5, ncol=1,
        framealpha=0.88, handlelength=1.2,
        title="Regime  [med. $I_θ$ interval]",
        title_fontsize=FS_LEGEND - 1,
    )

    fig.suptitle(
        "Sobol indices by $I_\\theta$ regime  "
        "(scenario-level binning, 95 % bootstrap CI)",
        fontsize=FS_TITLE + 0.5, y=1.01,
    )

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"conditional_sensitivity.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: conditional_sensitivity.png/pdf")


# ═════════════════════════════════════════════════════════════════════════════
# Q4: Counterfactual explanations
# ═════════════════════════════════════════════════════════════════════════════

def _grid_counterfactual(rows, gamma=0.5):
    """Find nearest consistent neighbour for each inconsistent sample.

    Parameters
    ----------
    rows : list of dict
        Each dict must contain keys: scale_factor, center_delta,
        correlation_strength, i_surr.
    gamma : float
        Inconsistency threshold; samples with i_surr > gamma are considered
        inconsistent.

    Returns
    -------
    list of dict — one entry per inconsistent sample:
        dist         : normalised L2 distance to nearest consistent sample
        delta_scale/center/corr : signed normalised deltas (fix direction)
        dominant     : which parameter has the largest |delta|
        *_star       : raw parameter values of the inconsistent point
        *_prime      : raw parameter values of the counterfactual
        i_surr_star  : surrogate I value of the inconsistent point
        i_surr_prime : surrogate I value of the counterfactual
    Returns [] if no inconsistent or no consistent samples exist.
    """
    if not rows:
        return []

    keys = ["scale_factor", "center_delta", "correlation_strength"]
    vals = {k: np.array([r[k] for r in rows], dtype=float) for k in keys}
    i_surr = np.array([r["i_surr"] for r in rows], dtype=float)

    # Normalise each parameter to [0, 1] within this scenario's range
    mins   = {k: vals[k].min() for k in keys}
    maxs   = {k: vals[k].max() for k in keys}
    ranges = {k: (maxs[k] - mins[k]) if maxs[k] > mins[k] else 1.0 for k in keys}
    norm   = {k: (vals[k] - mins[k]) / ranges[k] for k in keys}

    X = np.stack([norm[k] for k in keys], axis=1)   # (N, 3)

    incon_idx = np.where(i_surr >  gamma)[0]
    con_idx   = np.where(i_surr <= gamma)[0]

    if len(incon_idx) == 0 or len(con_idx) == 0:
        return []

    X_con = X[con_idx]

    results = []
    for idx in incon_idx:
        x_star   = X[idx]
        dists    = np.linalg.norm(X_con - x_star, axis=1)
        near_pos = int(np.argmin(dists))
        nearest  = con_idx[near_pos]
        dist_val = float(dists[near_pos])

        delta     = X[nearest] - x_star        # signed normalised deltas
        abs_delta = np.abs(delta)
        dominant  = keys[int(np.argmax(abs_delta))]

        results.append({
            "dist":          dist_val,
            "delta_scale":   float(delta[0]),
            "delta_center":  float(delta[1]),
            "delta_corr":    float(delta[2]),
            "dominant":      dominant,
            # raw coordinates
            "scale_star":    float(vals["scale_factor"][idx]),
            "center_star":   float(vals["center_delta"][idx]),
            "corr_star":     float(vals["correlation_strength"][idx]),
            "scale_prime":   float(vals["scale_factor"][nearest]),
            "center_prime":  float(vals["center_delta"][nearest]),
            "corr_prime":    float(vals["correlation_strength"][nearest]),
            # inconsistency values
            "i_surr_star":   float(i_surr[idx]),
            "i_surr_prime":  float(i_surr[nearest]),
        })

    return results


# ── Q4 colour scheme ──────────────────────────────────────────────────────────
_CF_COLORS = {
    "scale_factor":         "#2166ac",   # blue  — scale
    "center_delta":         "#d6604d",   # red   — center
    "correlation_strength": "#4dac26",   # green — correlation
}
_CF_SHORT = {
    "scale_factor":         r"Scale $s_u$",
    "center_delta":         r"Center $\Delta c_u$",
    "correlation_strength": r"Corr. $\rho_u$",
}


def _cf_domain_short(domain: str) -> str:
    """Return a compact domain label for axis ticks."""
    _MAP = {
        "automotive":          "Auto.",
        "building":            "HVAC",
        "hvac":                "HVAC",
        "industrial":          "Robot",
        "robot":               "Robot",
        "medical":             "Med.",
        "railway":             "Rail",
        "satellite":           "Sat.",
        "aerospace":           "Sat.",
        "smart":               "Grid",
        "grid":                "Grid",
        "water":               "Water",
        "chemical":            "Chem.",
        "wind":                "Wind",
        "cad":                 "CAD",
        "mbse":                "MBSE",
        "sensor":              "Sensor",
        "systems engineering": "SysEng",
        "convide":             "ConVIDe",
    }
    low = domain.lower()
    for key, short in _MAP.items():
        if key in low:
            return short
    # Fallback: capitalise first word, max 7 chars
    first = domain.split("_")[0].split("/")[-1]
    return first[:7].capitalize()


def plot_counterfactual(sobol_scenarios, out_dir, model=None,
                        device=None, gamma=0.5, max_queries=40,
                        max_workers=4, use_v2=False):
    """Q4 counterfactual figures — publication-quality outputs.

    Uses batched gradient-based counterfactual search (Adam + penalty
    escalation) through the frozen DeepSets surrogate.  All queries within
    a scenario are vectorised into a single (Q*R, 3) batch; scenarios are
    run in parallel via a ThreadPoolExecutor.

    Falls back to grid-search per scenario when geometry is unavailable.

    Per scenario (one figure each):
        q4_scenario_{id}.png/pdf : 2-panel landscape (left) + trajectories (right)

    Across all scenarios:
        q4_overview.png/pdf      : stacked bar (dominant param) + box (repair dist)
        q4_fix_direction.png/pdf : mean signed repair shift per domain
    """
    import matplotlib.patches as mpatches
    from .counterfactual import (
        run_counterfactuals_parallel, PARAM_NAMES, PARAM_BOUNDS,
    )

    print(f"\n[Q4] Counterfactual explanations (gamma={gamma}, "
          f"workers={max_workers})...")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cpu")

    PARAMS_3 = ["scale_factor", "center_delta", "correlation_strength"]

    # -- 1. Run gradient-based search for all scenarios in parallel
    sc_list = [s for s in sobol_scenarios if isinstance(s, dict)]

    if model is not None:
        sc_cfs = run_counterfactuals_parallel(
            sc_list, model,
            gamma=gamma, max_queries=max_queries,
            device=device, max_workers=max_workers,
            use_v2=use_v2)
    else:
        sc_cfs = {}

    # -- 2. Collect results per scenario
    domain_dists   = {}
    domain_dom     = {}
    domain_cfs_raw = {}

    # Per-scenario data for individual figures
    sc_records = []   # list of (sc_id, domain, rows, cfs_raw)

    for sc_idx, scenario in enumerate(sobol_scenarios):
        if isinstance(scenario, dict):
            domain    = scenario.get("domain", "unknown")
            rows      = scenario.get("rows", [])
            sc_id     = scenario.get("scenario_id", sc_idx)
        else:
            domain = "unknown"; rows = scenario; sc_id = sc_idx

        cfs_raw = sc_cfs.get(sc_idx, [])

        # Fall back to grid search if no gradient results
        if not cfs_raw:
            grid = _grid_counterfactual(rows, gamma)
            if not grid:
                continue

            class _GCF:
                pass
            for g in grid:
                o = _GCF()
                o.dist_normalised = g["dist"]
                o.dominant_param  = g["dominant"]
                o.delta_theta     = np.array([g["delta_scale"],
                                              g["delta_center"],
                                              g["delta_corr"]], dtype=np.float32)
                o.theta_star  = np.array([g["scale_star"],
                                          g["center_star"],
                                          g["corr_star"]], dtype=np.float32)
                o.theta_prime = np.array([g["scale_prime"],
                                          g["center_prime"],
                                          g["corr_prime"]], dtype=np.float32)
                o.i_surr_star  = g["i_surr_star"]
                o.i_surr_prime = g["i_surr_prime"]
                cfs_raw.append(o)

        if not cfs_raw:
            continue

        sc_records.append((sc_id, domain, rows, cfs_raw))

        if domain not in domain_dists:
            domain_dists[domain]   = []
            domain_dom[domain]     = {k: 0 for k in PARAM_NAMES}
            domain_cfs_raw[domain] = []

        domain_dists[domain].extend([cf.dist_normalised for cf in cfs_raw])
        domain_cfs_raw[domain].extend(cfs_raw)
        for cf in cfs_raw:
            domain_dom[domain][cf.dominant_param] += 1

    if not domain_dists:
        print("[Q4] No counterfactuals found -- skipping.")
        return

    n_total = sum(len(v) for v in domain_dists.values())
    print(f"  Found {n_total} counterfactual pairs across "
          f"{len(domain_dists)} domain(s), {len(sc_records)} scenario(s).")

    # -- Export raw per-repair data to CSV (used by combine_q4_figures.py for
    #    merged q4_directions and q4_fix_direction figures across eval sets)
    import csv
    csv_path = out_dir / "q4_repairs_raw.csv"
    _sc_domain_map = {sc_id: dom for sc_id, dom, _, _ in sc_records}
    with open(csv_path, "w", newline="") as _csv_fh:
        _writer = csv.writer(_csv_fh)
        _writer.writerow([
            "domain", "scenario_id",
            "delta_scale", "delta_center", "delta_corr",
            "theta_star_scale", "theta_star_center", "theta_star_corr",
            "theta_prime_scale", "theta_prime_center", "theta_prime_corr",
            "dist_normalised", "dominant_param",
            "i_surr_star", "i_surr_prime",
        ])
        for sc_id, domain, _rows, cfs in sc_records:
            for cf in cfs:
                dt = np.asarray(cf.delta_theta, dtype=float)
                ts = np.asarray(cf.theta_star,  dtype=float)
                tp = np.asarray(cf.theta_prime, dtype=float)
                _writer.writerow([
                    domain, sc_id,
                    round(float(dt[0]), 6), round(float(dt[1]), 6), round(float(dt[2]), 6),
                    round(float(ts[0]), 6), round(float(ts[1]), 6), round(float(ts[2]), 6),
                    round(float(tp[0]), 6), round(float(tp[1]), 6), round(float(tp[2]), 6),
                    round(float(cf.dist_normalised), 6),
                    getattr(cf, "dominant_param", ""),
                    round(float(getattr(cf, "i_surr_star",  float("nan"))), 6),
                    round(float(getattr(cf, "i_surr_prime", float("nan"))), 6),
                ])
    print(f"  Saved: q4_repairs_raw.csv  ({n_total} rows)")

    def _dom_dominant(d):
        counts = domain_dom[d]
        return max(counts, key=counts.get)

    # -- 3. Per-scenario 2-panel figure
    def _draw_landscape_ax(ax, fig, rows, cfs, domain, panel_title, show_cbar=True):
        """Draw response surface + hero arrow on ax. Returns hero_cf."""
        scales  = np.array([r["scale_factor"] for r in rows], dtype=float)
        centers = np.array([r["center_delta"]  for r in rows], dtype=float)
        i_surr  = np.array([r["i_surr"]        for r in rows], dtype=float)
        valid   = np.isfinite(scales) & np.isfinite(centers) & np.isfinite(i_surr)

        if valid.sum() >= 20:
            xi = np.linspace(scales[valid].min(), scales[valid].max(), 90)
            yi = np.linspace(centers[valid].min(), centers[valid].max(), 90)
            XX, YY = np.meshgrid(xi, yi)
            ZZ = _interp_surf(
                np.column_stack([scales[valid], centers[valid]]),
                i_surr[valid], XX, YY, sigma=1.5)
            cf_fill = ax.contourf(XX, YY, ZZ, levels=np.linspace(0, 1, 64),
                                  cmap="RdYlGn_r", vmin=0, vmax=1,
                                  alpha=0.75, zorder=1)
            ax.contour(XX, YY, ZZ, levels=[gamma],
                       colors=["#222222"], linewidths=[2.5],
                       linestyles=["--"], zorder=3)
            if show_cbar:
                cb = fig.colorbar(cf_fill, ax=ax, shrink=0.92, pad=0.03,
                                  aspect=25)
                cb.set_label(r"$\hat{I}(\theta)$", fontsize=FS_LABEL)
                cb.set_ticks([0.0, gamma, 1.0])
                cb.set_ticklabels(["0", rf"$\gamma\!=\!{gamma}$", "1"],
                                  fontsize=FS_TICK)
                cb.ax.axhline(gamma, color="#333", lw=1.2, ls="--")
        else:
            sc_bg = ax.scatter(scales[valid], centers[valid], c=i_surr[valid],
                               cmap="RdYlGn_r", vmin=0, vmax=1, s=14,
                               alpha=0.6, zorder=1)
            if show_cbar:
                fig.colorbar(sc_bg, ax=ax, shrink=0.92,
                             label=r"$\hat{I}(\theta)$")

        hero_cf = max(cfs, key=lambda c: c.dist_normalised)

        # All theta* as faint gray dots (reduced to avoid clutter)
        ax.scatter([c.theta_star[0] for c in cfs],
                   [c.theta_star[1] for c in cfs],
                   s=9, c="#999999", alpha=0.18, linewidths=0, zorder=4)

        # Hero theta* -- red circle
        ax.scatter([hero_cf.theta_star[0]], [hero_cf.theta_star[1]],
                   s=160, c="#cc2222", edgecolors="white", linewidths=1.6,
                   zorder=7, label=r"Inconsistent $\theta^*$")

        # Hero theta' -- green triangle (enlarged, black border for emphasis)
        ax.scatter([hero_cf.theta_prime[0]], [hero_cf.theta_prime[1]],
                   s=195, marker="^", c="#1a7a1a", edgecolors="#111111",
                   linewidths=2.0, zorder=7, label=r"Consistent $\theta'$")

        # Repair arrow — reduced thickness so landscape remains visually primary
        ax.annotate("",
                    xy    =(hero_cf.theta_prime[0], hero_cf.theta_prime[1]),
                    xytext=(hero_cf.theta_star[0],  hero_cf.theta_star[1]),
                    arrowprops=dict(arrowstyle="-|>", color="#111111",
                                    lw=1.55, mutation_scale=13),
                    zorder=8)

        # "Minimal repair" label — offset perpendicular, semi-transparent box
        mx = (hero_cf.theta_star[0] + hero_cf.theta_prime[0]) / 2
        my = (hero_cf.theta_star[1] + hero_cf.theta_prime[1]) / 2
        dx = hero_cf.theta_prime[0] - hero_cf.theta_star[0]
        dy = hero_cf.theta_prime[1] - hero_cf.theta_star[1]
        norm = (dx**2 + dy**2) ** 0.5 + 1e-10
        perp_scale = norm * 0.24
        ox = -dy / norm * perp_scale
        oy =  dx / norm * perp_scale
        ax.text(mx + ox, my + oy, "Minimal repair",
                ha="center", va="bottom", fontsize=FS_ANNOT + 1,
                color="#111111", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.30", fc="white",
                          ec="#888888", alpha=0.93, lw=0.9),
                zorder=9)

        # Distance + I(θ*) before + I(θ') after annotations (bottom-right)
        i_prime = getattr(hero_cf, "i_surr_prime", float("nan"))
        # Look up I(θ*) from rows by nearest-neighbour match
        dists_r = ((scales - hero_cf.theta_star[0])**2
                   + (centers - hero_cf.theta_star[1])**2)
        i_star_val = float(i_surr[int(np.argmin(dists_r))]) if valid.any() else float("nan")
        ann_lines = rf"$d = {hero_cf.dist_normalised:.2f}$"
        if np.isfinite(i_star_val):
            ann_lines += "\n" + rf"$\hat{{I}}(\theta^*) = {i_star_val:.2f}$"
        if np.isfinite(i_prime):
            ann_lines += "\n" + rf"$\hat{{I}}(\theta') = {i_prime:.2f}$"
        ax.text(0.97, 0.04,
                ann_lines,
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=FS_ANNOT + 1, linespacing=1.5,
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec="#cccccc", alpha=0.85, lw=0.7))

        ax.set_xlabel(PARAM_LABELS["scale_factor"],  fontsize=FS_LABEL)
        ax.set_ylabel(PARAM_LABELS["center_delta"],  fontsize=FS_LABEL)
        ax.set_title(panel_title, fontsize=FS_TITLE, pad=6)
        ax.legend(fontsize=FS_LEGEND - 0.5, loc="upper left",
                  framealpha=0.88, edgecolor="#cccccc",
                  handlelength=1.2, borderpad=0.6)
        _style_ax(ax, grid_axis="both")
        return hero_cf

    def _draw_trajectories_ax(ax, rows, cfs, domain, panel_title):
        """Draw all repair arrows coloured by dominant param on ax."""
        scales  = np.array([r["scale_factor"] for r in rows], dtype=float)
        centers = np.array([r["center_delta"]  for r in rows], dtype=float)
        i_surr  = np.array([r["i_surr"]        for r in rows], dtype=float)
        valid   = np.isfinite(scales) & np.isfinite(centers) & np.isfinite(i_surr)

        if valid.sum() >= 20:
            xi = np.linspace(scales[valid].min(), scales[valid].max(), 90)
            yi = np.linspace(centers[valid].min(), centers[valid].max(), 90)
            XX, YY = np.meshgrid(xi, yi)
            ZZ = _interp_surf(
                np.column_stack([scales[valid], centers[valid]]),
                i_surr[valid], XX, YY, sigma=1.5)
            ax.contourf(XX, YY, ZZ, levels=np.linspace(0, 1, 64),
                        cmap="RdYlGn_r", vmin=0, vmax=1, alpha=0.55, zorder=1)
            ax.contour(XX, YY, ZZ, levels=[gamma],
                       colors=["#222222"], linewidths=[2.0],
                       linestyles=["--"], zorder=3)

        # Arrowheads explicit: larger mutation_scale, slightly reduced lw+alpha
        arrow_kw = dict(lw=1.2, mutation_scale=13, arrowstyle="-|>", alpha=0.68)
        for cf in cfs:
            ax.annotate("",
                        xy    =(cf.theta_prime[0], cf.theta_prime[1]),
                        xytext=(cf.theta_star[0],  cf.theta_star[1]),
                        arrowprops=dict(color=_CF_COLORS[cf.dominant_param],
                                        **arrow_kw),
                        zorder=5)

        handles_traj = [
            mpatches.Patch(color=_CF_COLORS[p], label=_CF_SHORT[p])
            for p in PARAMS_3
        ]
        ax.legend(handles=handles_traj,
                  title="Dominant repair param.",
                  title_fontsize=FS_LEGEND - 1,
                  fontsize=FS_LEGEND - 1, loc="upper left",
                  framealpha=0.88, edgecolor="#cccccc",
                  handlelength=1.1, borderpad=0.5,
                  labelspacing=0.35)

        # Semantic clarification below the panel title
        ax.set_xlabel(PARAM_LABELS["scale_factor"], fontsize=FS_LABEL)
        ax.set_ylabel(PARAM_LABELS["center_delta"], fontsize=FS_LABEL)
        ax.set_title(panel_title + "\n"
                     r"$\it{Arrows\ terminate\ at\ nearest\ consistent\ config.}$",
                     fontsize=FS_TITLE, pad=6)
        _style_ax(ax, grid_axis="both")

    for sc_id, domain, rows, cfs in sc_records:
        if len(cfs) < 2:
            continue
        dlabel = _cf_domain_short(domain)
        fig, axes = plt.subplots(
            1, 2,
            figsize=(13.0, 5.2),
            gridspec_kw={"wspace": 0.38})

        _draw_landscape_ax(
            axes[0], fig, rows, cfs, domain,
            panel_title=rf"(a) Minimal-repair landscape -- {dlabel}",
            show_cbar=True)

        _draw_trajectories_ax(
            axes[1], rows, cfs, domain,
            panel_title=rf"(b) All repair trajectories -- {dlabel}")

        fig.suptitle(
            rf"Q4 Counterfactual analysis -- Scenario {sc_id} ({dlabel})",
            fontsize=FS_TITLE + 1, y=1.01, fontweight="semibold")

        fig.tight_layout()
        fname = f"q4_scenario_{sc_id}"
        for ext in ("png", "pdf"):
            fig.savefig(out_dir / f"{fname}.{ext}", dpi=300,
                        bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}.png/pdf  ({len(cfs)} repairs, domain={dlabel})")

    # -- 3b. Standalone q4_landscape figure (prefer scenario 49, else global hero)
    _LANDSCAPE_SC_ID = 49
    _best_sc_land = None
    _best_dist_land = -1.0
    for _sc_id_l, _dom_l, _rows_l, _cfs_l in sc_records:
        if len(_cfs_l) < 2:
            continue
        if _sc_id_l == _LANDSCAPE_SC_ID:
            _best_sc_land = (_sc_id_l, _dom_l, _rows_l, _cfs_l)
            break
        _md = max(c.dist_normalised for c in _cfs_l)
        if _md > _best_dist_land:
            _best_dist_land = _md
            _best_sc_land = (_sc_id_l, _dom_l, _rows_l, _cfs_l)
    if _best_sc_land is not None:
        _sc_id_l, _dom_l, _rows_l, _cfs_l = _best_sc_land
        _dlabel_l = _cf_domain_short(_dom_l)
        fig_l, ax_l = plt.subplots(1, 1, figsize=(7.2, 4.8))
        _draw_landscape_ax(
            ax_l, fig_l, _rows_l, _cfs_l, _dom_l,
            panel_title="",
            show_cbar=True)
        fig_l.tight_layout()
        for ext in ("png", "pdf"):
            fig_l.savefig(out_dir / f"q4_landscape.{ext}", dpi=300,
                          bbox_inches="tight")
        plt.close(fig_l)
        print(f"  Saved: q4_landscape.png/pdf  "
              f"(scenario={_sc_id_l}, domain={_dlabel_l})")

    # -- 4. Repair distance figure (boxplot only) → q4_distances
    domains  = sorted(domain_dists.keys(),
                      key=lambda d: (PARAMS_3.index(_dom_dominant(d)), d))
    dlabels  = [_cf_domain_short(d) for d in domains]
    n_dom    = len(domains)

    bar_w  = max(0.55, min(0.75, 2.5 / max(n_dom, 1)))
    fig_w  = max(7.0, n_dom * 1.35 + 2.5)
    fig, ax_box = plt.subplots(1, 1,
                               figsize=(fig_w, max(3.4, n_dom * 0.22 + 2.0)))

    bp_data = [domain_dists[d] for d in domains]
    medians = [float(np.median(domain_dists[d])) for d in domains]
    hardest_idx = int(np.argmax(medians))

    bp = ax_box.boxplot(bp_data,
                        positions=np.arange(n_dom),
                        widths=bar_w * 0.85,
                        patch_artist=True,
                        showfliers=True,
                        medianprops=dict(color="#111111", lw=2.2),
                        whiskerprops=dict(color="#666666", lw=1.3),
                        capprops=dict(color="#666666", lw=1.3),
                        flierprops=dict(marker=".", color="#bbbbbb",
                                        markersize=4, alpha=0.55),
                        boxprops=dict(linewidth=1.3))

    cmap_d = plt.cm.tab10
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(cmap_d(i % 10))
        patch.set_alpha(0.50)

    y_maxes = [max(domain_dists[d]) for d in domains]
    for i, (med, ymax) in enumerate(zip(medians, y_maxes)):
        offset = (max(y_maxes) - min(medians)) * 0.07 + 0.02
        ax_box.text(i, ymax + offset, f"{med:.2f}",
                    ha="center", va="bottom", fontsize=FS_ANNOT + 1.0,
                    color="#333333", fontweight="semibold")

    # "Hardest to repair" annotation removed — median labels are sufficient

    ax_box.set_xticks(np.arange(n_dom))
    ax_box.set_xticklabels(dlabels, rotation=30 if n_dom > 4 else 0,
                           ha="right" if n_dom > 4 else "center",
                           fontsize=FS_LABEL)
    ax_box.set_ylabel(r"Normalised repair distance $\|\theta'-\theta^*\|$",
                      fontsize=FS_LABEL)
    # title removed — caption carries this information
    ax_box.set_xlim(-0.6, n_dom - 0.4)
    _style_ax(ax_box, grid_axis="y")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"q4_distances.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print("  Saved: q4_distances.png/pdf")

    # -- 5. Fix-direction: mean signed shift, one panel per parameter
    if domain_cfs_raw:
        param_full  = ["scale_factor", "center_delta", "correlation_strength"]
        param_xlbls = ["Scale repair", "Center repair", "Correlation repair"]

        fig_h = max(3.8, n_dom * 0.52 + 1.8)
        fig, axes = plt.subplots(1, 3,
                                 figsize=(13.0, fig_h), sharey=True,
                                 gridspec_kw={"wspace": 0.12})
        y_pos = np.arange(n_dom, 0, -1)   # top-to-bottom

        for col_idx, (ax, p_idx, pf, pxlbl) in enumerate(
                zip(axes, range(3), param_full, param_xlbls)):
            col = _CF_COLORS[pf]
            dom_means = []
            for d in domains:
                deltas = [float(cf.delta_theta[p_idx])
                          for cf in domain_cfs_raw.get(d, [])]
                dom_means.append(float(np.mean(deltas)) if deltas else 0.0)

            arr   = np.array(dom_means, dtype=float)
            sigma = arr.std() if arr.std() > 1e-9 else 1.0
            pad   = max(abs(arr).max() * 0.28, 0.04)
            x_lo  = min(arr.mean() - 2.8 * sigma, -pad)
            x_hi  = max(arr.mean() + 2.8 * sigma,  pad)

            for yi, val in enumerate(dom_means):
                clipped  = val < x_lo or val > x_hi
                val_draw = float(np.clip(val, x_lo * 0.92, x_hi * 0.92))
                arrow_c  = col if val >= 0 else "#cc4444"

                ax.annotate("",
                            xy=(val_draw, y_pos[yi]),
                            xytext=(0.0,  y_pos[yi]),
                            arrowprops=dict(arrowstyle="-|>",
                                            color=arrow_c, lw=2.2,
                                            mutation_scale=12))

                gap    = (x_hi - x_lo) * 0.03
                lbl_x  = val_draw + (gap if val >= 0 else -gap)
                suffix = "\u22ef" if clipped else ""
                ax.text(lbl_x, y_pos[yi],
                        f"{val:+.3f}{suffix}",
                        ha="left" if val >= 0 else "right",
                        va="center", fontsize=FS_ANNOT + 0.5, color="#222222")

            ax.axvline(0, color="#444444", lw=1.0, ls="-", alpha=0.70, zorder=2)
            ax.fill_betweenx(
                [y_pos[-1] - 0.5, y_pos[0] + 0.5],
                x_lo, 0, color="#ffcccc", alpha=0.10, zorder=0)
            ax.fill_betweenx(
                [y_pos[-1] - 0.5, y_pos[0] + 0.5],
                0, x_hi, color="#ccffcc", alpha=0.10, zorder=0)

            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(y_pos[-1] - 0.7, y_pos[0] + 0.7)
            ax.set_yticks(y_pos)

            if col_idx == 0:
                ax.set_yticklabels(dlabels, fontsize=FS_LABEL)
            else:
                ax.set_yticklabels([])

            ax.set_xlabel(
                r"$\leftarrow$ decrease    increase $\rightarrow$",
                fontsize=FS_TICK + 0.5)
            ax.set_title(pxlbl, fontsize=FS_TITLE - 0.5, pad=6,
                         color=_CF_COLORS[pf])
            _style_ax(ax, grid_axis="x")

        axes[0].set_ylabel("Domain", fontsize=FS_LABEL)
        fig.suptitle(
            "Mean normalised repair shift required to restore consistency\n"
            r"(positive = increase parameter, negative = decrease)",
            fontsize=FS_TITLE, y=1.03)

        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out_dir / f"q4_fix_direction.{ext}", dpi=300,
                        bbox_inches="tight")
        plt.close(fig)
        print("  Saved: q4_fix_direction.png/pdf")

    # -- 6. Repair structure: ternary (dominant param) + parallel coords (signed)
    # Collect all repairs in one flat pass over domain_cfs_raw
    _SPANS = np.array([b[1] - b[0] for b in PARAM_BOUNDS], dtype=float)  # [3.05, 2.10, 1.05]
    _SQRT3_2 = 3.0 ** 0.5 / 2.0
    # Vertices in 2D cartesian: Scale=bottom-left, Center=bottom-right, Corr=top
    _VERTS = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, _SQRT3_2]])

    def _bary2cart(pts):
        """(N,3) barycentric → (N,2) cartesian via vertex linear combination."""
        return np.asarray(pts) @ _VERTS

    all_tern, all_par, all_dom = [], [], []
    for d in sorted(domain_cfs_raw.keys()):
        for cf in domain_cfs_raw[d]:
            dt = np.array(cf.delta_theta, dtype=float)
            abs_n = np.abs(dt) / (_SPANS + 1e-10)
            total = abs_n.sum() + 1e-10
            all_tern.append(abs_n / total)          # normalised absolute → ternary
            all_par.append(dt / (_SPANS + 1e-10))   # signed, span-scaled
            all_dom.append(d)

    if not all_tern:
        return

    all_tern = np.array(all_tern)   # (N, 3)
    all_par  = np.array(all_par)    # (N, 3)
    all_dom  = np.array(all_dom)
    dom_uniq = sorted(set(all_dom))
    n_du     = len(dom_uniq)

    _cmap_rs  = plt.cm.tab10 if n_du <= 10 else plt.cm.tab20
    _dc       = {d: _cmap_rs(i % 20) for i, d in enumerate(dom_uniq)}
    _ds       = {d: _cf_domain_short(d) for d in dom_uniq}

    # ══ Figure A: Ternary — repair composition in 3D simplex ═════════════════
    fig_t, ax_t = plt.subplots(1, 1, figsize=(7.2, 5.8))

    tri_pts = np.vstack([_VERTS, _VERTS[0]])
    ax_t.plot(tri_pts[:, 0], tri_pts[:, 1],
              color="#444444", lw=1.6, zorder=2)

    for lvl in (0.25, 0.50, 0.75):
        for axis in range(3):
            p = np.zeros((2, 3))
            p[0, axis] = lvl;  p[0, (axis+1) % 3] = 1 - lvl; p[0, (axis+2) % 3] = 0
            p[1, axis] = lvl;  p[1, (axis+1) % 3] = 0;        p[1, (axis+2) % 3] = 1 - lvl
            xy = _bary2cart(p)
            ax_t.plot(xy[:, 0], xy[:, 1],
                      color="#aaaaaa", lw=0.5, alpha=0.45, zorder=1, ls="--")

    for d in dom_uniq:
        mask = all_dom == d
        if not mask.any():
            continue
        xy = _bary2cart(all_tern[mask])
        ax_t.scatter(xy[:, 0], xy[:, 1],
                     s=30, color=_dc[d], alpha=0.44,
                     linewidths=0, zorder=3)

    legend_handles = []
    for d in dom_uniq:
        mask = all_dom == d
        if not mask.any():
            continue
        cen = all_tern[mask].mean(axis=0)
        cen /= cen.sum() + 1e-10
        xy_c = _bary2cart(cen.reshape(1, 3))[0]
        ax_t.scatter([xy_c[0]], [xy_c[1]],
                     s=195, color=_dc[d], marker="D",
                     edgecolors="#111111", linewidths=1.9,
                     zorder=6, alpha=1.0)
        ax_t.text(xy_c[0], xy_c[1] + 0.027, _ds[d],
                  ha="center", va="bottom",
                  fontsize=FS_ANNOT - 0.5, color=_dc[d],
                  fontweight="bold", zorder=7)
        legend_handles.append(
            mpatches.Patch(color=_dc[d], label=_ds[d]))

    _vlabels  = [r"Scale $s_u$", r"Center $\Delta c_u$", r"Corr. $\rho_u$"]
    _vcolors  = [_CF_COLORS["scale_factor"],
                 _CF_COLORS["center_delta"],
                 _CF_COLORS["correlation_strength"]]
    _voffsets = [(-0.08, -0.07), (0.08, -0.07), (0.0, 0.06)]
    for (ox, oy), lbl, col, v in zip(_voffsets, _vlabels, _vcolors, _VERTS):
        ax_t.text(v[0] + ox, v[1] + oy, lbl,
                  ha="center", va="center",
                  fontsize=FS_LABEL, fontweight="bold", color=col, zorder=8)

    _vdom_lbl = ["Scale-dominated", "Center-dominated", "Corr.-dominated"]
    _vdom_off = [(-0.16, 0.08), (0.18, 0.08), (0.0, 0.10)]
    for (ox, oy), lbl, col, v in zip(_vdom_off, _vdom_lbl, _vcolors, _VERTS):
        ax_t.text(v[0] + ox, v[1] + oy, lbl,
                  ha="center", va="center",
                  fontsize=FS_ANNOT - 0.5, color=col,
                  style="italic", alpha=0.60, zorder=8)

    ax_t.set_xlim(-0.20, 1.20)
    ax_t.set_ylim(-0.15, _SQRT3_2 + 0.18)
    ax_t.set_aspect("equal", adjustable="box")
    ax_t.axis("off")
    # title and normalisation subtitle removed — caption carries this information

    fig_t.legend(handles=legend_handles,
                 title="Domain  (\u25c6 = centroid)",
                 title_fontsize=FS_LEGEND,
                 fontsize=FS_LEGEND,
                 loc="lower center",
                 bbox_to_anchor=(0.5, -0.01),
                 ncol=n_du,
                 framealpha=0.93, edgecolor="#cccccc",
                 handlelength=1.2, borderpad=0.5,
                 handletextpad=0.4, columnspacing=1.0)

    for ext in ("png", "pdf"):
        fig_t.savefig(out_dir / f"q4_ternary.{ext}", dpi=300,
                      bbox_inches="tight")
    plt.close(fig_t)
    print("  Saved: q4_ternary.png/pdf")

    # ══ Figure B: Parallel coordinates — signed repair directions ═════════════
    fig_p, ax_p = plt.subplots(1, 1, figsize=(10.0, 4.4))

    x_pos = np.array([0, 1, 2])
    y_abs = max(np.abs(all_par).max() * 1.18, 0.15)

    ax_p.fill_between([-0.35, 2.35], [0, 0], [y_abs, y_abs],
                      color="#ccffcc", alpha=0.08, zorder=0)
    ax_p.fill_between([-0.35, 2.35], [-y_abs, -y_abs], [0, 0],
                      color="#ffcccc", alpha=0.08, zorder=0)
    ax_p.text(2.30, y_abs * 0.90,  "increase",
              ha="right", va="top",
              fontsize=FS_ANNOT, color="#448844", style="italic")
    ax_p.text(2.30, -y_abs * 0.90, "decrease",
              ha="right", va="bottom",
              fontsize=FS_ANNOT, color="#884444", style="italic")

    for xi in x_pos:
        ax_p.axvline(xi, color="#999999", lw=0.9, alpha=0.50, zorder=1)
    ax_p.axhline(0, color="#444444", lw=1.0, ls="--", alpha=0.55, zorder=3)

    for d in dom_uniq:
        mask = all_dom == d
        for row in all_par[mask]:
            ax_p.plot(x_pos, row, color=_dc[d],
                      alpha=0.09, lw=0.50, zorder=2)

    for d in dom_uniq:
        mask = all_dom == d
        if not mask.any():
            continue
        mean_row = all_par[mask].mean(axis=0)
        ax_p.plot(x_pos, mean_row, color=_dc[d],
                  alpha=0.95, lw=2.5, zorder=5,
                  marker="o", markersize=5.5,
                  markeredgecolor="white", markeredgewidth=0.9)
        ax_p.text(2.04, mean_row[2], _ds[d],
                  ha="left", va="center",
                  fontsize=FS_ANNOT - 0.5, color=_dc[d],
                  fontweight="bold", zorder=7)

    p_xlbls = [r"Scale $s_u$", r"Center $\Delta c_u$", r"Corr. $\rho_u$"]
    ax_p.set_xlim(-0.35, 2.55)
    ax_p.set_ylim(-y_abs, y_abs)
    ax_p.set_xticks(x_pos)
    ax_p.set_xticklabels(p_xlbls, fontsize=FS_LABEL)
    ax_p.set_ylabel(r"Repair shift (span-normalised $\delta\theta_i / \mathrm{range}_i$)",
                    fontsize=FS_LABEL)
    # title removed — caption carries this information
    ax_p.spines["bottom"].set_visible(False)
    _style_ax(ax_p, grid_axis="y")

    for ext in ("png", "pdf"):
        fig_p.savefig(out_dir / f"q4_directions.{ext}", dpi=300,
                      bbox_inches="tight")
    plt.close(fig_p)
    print("  Saved: q4_directions.png/pdf")

    # -- 7. Three-panel repair decomposition:
    #        (a) mean signed components  (b) absolute composition  (c) distance
    if not domain_cfs_raw:
        return

    # Domain ordering consistent with overview (by dominant repair param)
    domains_3p = sorted(domain_cfs_raw.keys(),
                        key=lambda d: (PARAMS_3.index(_dom_dominant(d)), d))
    dlabels_3p = [_cf_domain_short(d) for d in domains_3p]
    n_d3 = len(domains_3p)
    x_d  = np.arange(n_d3)

    # Per-domain statistics
    signed_means = np.zeros((n_d3, 3))
    abs_means    = np.zeros((n_d3, 3))
    dists_3p     = []

    for di, d in enumerate(domains_3p):
        cfs_d  = domain_cfs_raw[d]
        dt_all = np.array([cf.delta_theta for cf in cfs_d], dtype=float)
        dt_n   = dt_all / (_SPANS + 1e-10)   # span-normalised
        signed_means[di] = dt_n.mean(axis=0)
        abs_means[di]    = np.abs(dt_n).mean(axis=0)
        dists_3p.append([cf.dist_normalised for cf in cfs_d])

    # Absolute fractions normalised to 100 %
    abs_tot  = abs_means.sum(axis=1, keepdims=True).clip(min=1e-10)
    abs_frac = abs_means / abs_tot * 100.0

    p_colors = [_CF_COLORS[p] for p in PARAMS_3]
    p_labels = [r"Scale $s_u$", r"Center $\Delta c_u$", r"Corr. $\rho_u$"]

    fig_w = max(7.5, n_d3 * 1.05 + 2.5)
    fig, axes = plt.subplots(
        3, 1, sharex=True,
        figsize=(fig_w, 10.0),
        gridspec_kw={"hspace": 0.08, "height_ratios": [1.0, 0.82, 0.82]})

    bw      = 0.24
    offsets = np.array([-bw, 0.0, bw])

    # ── (a) Mean signed repair components ────────────────────────────────────
    ax_a = axes[0]
    handles_leg = []
    for pi, (col, lbl) in enumerate(zip(p_colors, p_labels)):
        bars = ax_a.bar(x_d + offsets[pi], signed_means[:, pi],
                        width=bw * 0.88, color=col, alpha=0.84,
                        label=lbl, zorder=3)
        handles_leg.append(bars[0])

    ax_a.axhline(0, color="#333333", lw=1.0, zorder=4)
    y_a = max(np.abs(signed_means).max() * 1.28, 0.06)
    ax_a.fill_between([-0.6, n_d3 - 0.4], 0, y_a,
                      color="#ccffcc", alpha=0.07, zorder=0)
    ax_a.fill_between([-0.6, n_d3 - 0.4], -y_a, 0,
                      color="#ffcccc", alpha=0.07, zorder=0)
    ax_a.text(n_d3 - 0.48,  y_a * 0.88, "increase",
              ha="right", va="top",
              fontsize=FS_ANNOT, color="#448844", style="italic")
    ax_a.text(n_d3 - 0.48, -y_a * 0.88, "decrease",
              ha="right", va="bottom",
              fontsize=FS_ANNOT, color="#884444", style="italic")
    ax_a.set_xlim(-0.6, n_d3 - 0.4)
    ax_a.set_ylim(-y_a, y_a)
    ax_a.set_ylabel("Mean signed\nrepair shift", fontsize=FS_LABEL)
    ax_a.set_title("(a) Mean signed repair components — direction of intervention",
                   fontsize=FS_TITLE, pad=6)
    _style_ax(ax_a, grid_axis="y")

    # ── (b) Absolute repair composition (stacked, 100 %) ─────────────────────
    ax_b = axes[1]
    bottoms_b = np.zeros(n_d3)
    for pi, col in enumerate(p_colors):
        bars_b = ax_b.bar(x_d, abs_frac[:, pi],
                          bottom=bottoms_b, width=0.54,
                          color=col, alpha=0.86, zorder=3)
        for xi, (bot, h) in enumerate(zip(bottoms_b, abs_frac[:, pi])):
            if h > 15:
                ax_b.text(xi, bot + h / 2, f"{h:.0f}%",
                          ha="center", va="center",
                          fontsize=FS_ANNOT + 1.5, color="white",
                          fontweight="bold", zorder=5)
        bottoms_b += abs_frac[:, pi]

    ax_b.axhline(100, color="#666666", lw=0.8, ls="--", alpha=0.40, zorder=2)
    ax_b.set_xlim(-0.6, n_d3 - 0.4)
    ax_b.set_ylim(0, 114)
    ax_b.set_ylabel("Repair composition\n(%)", fontsize=FS_LABEL)
    ax_b.set_title("(b) Absolute repair composition — which parameter dominates",
                   fontsize=FS_TITLE, pad=6)
    _style_ax(ax_b, grid_axis="y")

    # ── (c) Repair distance distribution ─────────────────────────────────────
    ax_c = axes[2]
    medians_c   = [float(np.median(d)) for d in dists_3p]
    hardest_idx = int(np.argmax(medians_c))
    y_maxes_c   = [max(d) if d else 0.0 for d in dists_3p]
    glob_max    = max(y_maxes_c) if y_maxes_c else 1.0

    bp = ax_c.boxplot(
        dists_3p,
        positions=x_d,
        widths=0.50,
        patch_artist=True,
        showfliers=True,
        medianprops=dict(color="#111111", lw=2.1),
        whiskerprops=dict(color="#666666", lw=1.2),
        capprops=dict(color="#666666", lw=1.2),
        flierprops=dict(marker=".", color="#bbbbbb", markersize=3.5, alpha=0.55),
        boxprops=dict(linewidth=1.2))

    cmap_c = plt.cm.tab10
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(cmap_c(i % 10))
        patch.set_alpha(0.45)
        if i == hardest_idx:
            patch.set_edgecolor("#111111")
            patch.set_linewidth(2.5)
            patch.set_alpha(0.82)

    # Median labels above each box
    for i, (med, ymax) in enumerate(zip(medians_c, y_maxes_c)):
        ax_c.text(i, ymax + glob_max * 0.05, f"{med:.2f}",
                  ha="center", va="bottom",
                  fontsize=FS_ANNOT + 0.5, color="#333333")

    if n_d3 > 0:
        dx_ann = min(0.70, (n_d3 - 1 - hardest_idx) * 0.5 + 0.3)
        ax_c.annotate(
            "Hardest\nto repair",
            xy=(hardest_idx, medians_c[hardest_idx]),
            xytext=(hardest_idx + dx_ann,
                    medians_c[hardest_idx] + glob_max * 0.25),
            fontsize=FS_ANNOT, color="#111111",
            arrowprops=dict(arrowstyle="->", color="#555555", lw=0.9),
            bbox=dict(boxstyle="round,pad=0.22", fc="white",
                      ec="#aaaaaa", alpha=0.90, lw=0.6))

    ax_c.set_xlim(-0.6, n_d3 - 0.4)
    ax_c.set_ylim(0, glob_max * 1.40)
    ax_c.set_xticks(x_d)
    ax_c.set_xticklabels(
        dlabels_3p,
        rotation=30 if n_d3 > 4 else 0,
        ha="right" if n_d3 > 4 else "center",
        fontsize=FS_LABEL)
    ax_c.set_ylabel(r"Repair distance $\|\theta'-\theta^*\|$",
                    fontsize=FS_LABEL)
    ax_c.set_title("(c) Repair distance distribution — cost of consistency restoration",
                   fontsize=FS_TITLE, pad=6)
    _style_ax(ax_c, grid_axis="y")

    # Shared legend — single horizontal row below all panels
    fig.legend(
        handles=handles_leg,
        labels=p_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.025),
        ncol=3,
        fontsize=FS_LEGEND + 0.5,
        framealpha=0.93,
        edgecolor="#cccccc",
        handlelength=1.5,
        handletextpad=0.6,
        columnspacing=2.0)

    fig.suptitle(
        "Counterfactual repair decomposition across CPS domains",
        fontsize=FS_TITLE + 1, y=1.01, fontweight="semibold")

    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"q4_repair_decomposition.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print("  Saved: q4_repair_decomposition.png/pdf")

    # -- 8. Save machine-readable statistics for fact-checking
    stats = {}
    for d in sorted(domain_dom.keys(), key=_cf_domain_short):
        total = sum(domain_dom[d].values())
        dists = np.array(domain_dists[d]) if d in domain_dists else np.array([])
        stats[_cf_domain_short(d)] = {
            "n_repairs":    int(total),
            "scale_pct":    round(domain_dom[d]["scale_factor"]    / max(total,1) * 100, 1),
            "center_pct":   round(domain_dom[d]["center_delta"]    / max(total,1) * 100, 1),
            "corr_pct":     round(domain_dom[d]["correlation_strength"] / max(total,1) * 100, 1),
            "median_dist":  round(float(np.median(dists)), 4) if len(dists) else None,
            "mean_dist":    round(float(np.mean(dists)),   4) if len(dists) else None,
            "max_dist":     round(float(np.max(dists)),    4) if len(dists) else None,
        }
    # Hero repair
    best_cf2 = None; best_d2 = -1; best_sc2 = None; best_dom2 = None
    for sc_idx2, scenario2 in enumerate(sobol_scenarios):
        dom2  = (scenario2.get("domain","?") if isinstance(scenario2, dict)
                 else "unknown")
        rows2 = (scenario2.get("rows",[]) if isinstance(scenario2, dict)
                 else scenario2)
        for cf2 in sc_cfs.get(sc_idx2, []):
            if cf2.dist_normalised > best_d2:
                best_d2 = cf2.dist_normalised; best_cf2 = cf2
                best_sc2 = scenario2; best_dom2 = dom2
    if best_cf2:
        rows2   = best_sc2.get("rows",[]) if isinstance(best_sc2, dict) else best_sc2
        scales2 = np.array([r.get("scale_factor",np.nan) for r in rows2])
        centr2  = np.array([r.get("center_delta", np.nan) for r in rows2])
        isurr2  = np.array([r.get("i_surr",       np.nan) for r in rows2])
        valid2  = np.isfinite(scales2) & np.isfinite(centr2) & np.isfinite(isurr2)
        dr2     = (scales2 - best_cf2.theta_star[0])**2 + (centr2 - best_cf2.theta_star[1])**2
        i_star2 = float(isurr2[int(np.argmin(dr2))]) if valid2.any() else float("nan")
        stats["_hero_repair"] = {
            "domain":         _cf_domain_short(best_dom2),
            "scenario_id":    best_sc2.get("scenario_id","?") if isinstance(best_sc2,dict) else "?",
            "dist_normalised":round(float(best_cf2.dist_normalised), 4),
            "i_theta_star":   round(i_star2, 4),
            "i_theta_prime":  round(float(getattr(best_cf2,"i_surr_prime",float("nan"))), 4),
            "theta_star":     [round(x,4) for x in best_cf2.theta_star.tolist()],
            "theta_prime":    [round(x,4) for x in best_cf2.theta_prime.tolist()],
        }
    stats_path = out_dir / "q4_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"  Saved: q4_stats.json")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def _load_model(device, model_path=None):
    """Load surrogate checkpoint.

    Returns
    -------
    model       : nn.Module in eval mode
    trained_dims: list[int] | None  — dimensions the model was trained on,
                  auto-read from run_config.json next to the checkpoint.
                  None means "all dimensions".
    """
    path = Path(model_path) if model_path else MODEL_PATH
    ckpt = torch.load(path, map_location=device, weights_only=False)

    # Auto-detect training dims from run_config.json in the same directory
    trained_dims = None
    cfg_path = path.parent / "run_config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            dims_cfg = cfg.get("args", {}).get("dims") or cfg.get("dims")
            if dims_cfg:
                trained_dims = [int(d) for d in dims_cfg]
        except Exception:
            pass

    # V2 checkpoint: saved by train_compare.py — has 'model_name' + 'state_dict'
    if "model_name" in ckpt and _load_v2_checkpoint is not None:
        model = _load_v2_checkpoint(path, map_location=device)
        model.to(device)
        return model, trained_dims

    # V1 checkpoint: saved by surrogate/train.py — has 'args' + 'model_state'
    saved_args = ckpt["args"]
    model = DeepSetsZonotope(
        phi_hidden=saved_args["phi_hidden"],
        rho_hidden=saved_args["rho_hidden"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, trained_dims


def _is_v2_model(model) -> bool:
    """True when the model expects v2 features (per_dim_v2 / global_v2)."""
    return type(model).__name__ in _V2_MODEL_CLASS_NAMES


def _export_eval_data(out_path, acc, ts, domain, domain_short,
                      domain_order, sobol_results):
    """Serialize acc/timing/domain/Sobol data for later use by combined figures.

    Writes two files to *out_path* (a directory):
      acc.npz   — large numeric arrays (i_surr, i_theta, i_aabb, i_mfmc, dim, scenario_id)
      meta.json — timing scalars, domain labels, Sobol results, etc.
    """
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── Numeric arrays ────────────────────────────────────────────────────────
    np.savez_compressed(
        out_path / "acc.npz",
        i_surr     = acc["i_surr"].astype(np.float32),
        i_theta    = acc["i_theta"].astype(np.float32),
        i_aabb     = acc["i_aabb"].astype(np.float32),
        i_mfmc     = acc["i_mfmc"].astype(np.float32),
        dim        = acc["dim"].astype(np.int32),
        scenario_id= np.array(acc["scenario_id"], dtype=np.int32),
    )

    # ── Metadata / small objects ───────────────────────────────────────────────
    timing = {
        "us_aabb": float(ts.get("us_aabb", 0.0)),
        "us_mfmc": float(ts.get("us_mfmc", 0.0)),
        "us_mc":   float(ts.get("us_mc",   0.0)),
        "us_surr": float(acc["t_surr_total"] / max(acc["n"], 1) * 1e6),
    }

    # domain / domain_short may be numpy arrays or lists — normalise to lists
    def _to_list(x):
        if x is None:
            return []
        if hasattr(x, "tolist"):
            return x.tolist()
        return list(x)

    def _make_json_safe(obj):
        """Recursively convert numpy scalars / arrays in obj to plain Python."""
        if isinstance(obj, dict):
            return {k: _make_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_make_json_safe(v) for v in obj]
        if hasattr(obj, "item"):          # numpy scalar
            return obj.item()
        if hasattr(obj, "tolist"):        # numpy array
            return obj.tolist()
        return obj

    meta = {
        "t_surr_total": float(acc["t_surr_total"]),
        "n":            int(acc["n"]),
        "timing":       timing,
        "domain":       _to_list(domain),
        "domain_short": _make_json_safe(domain_short),
        "domain_order": _to_list(domain_order),
        "sobol_results": _make_json_safe(sobol_results),
    }
    (out_path / "meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\n  [export] Saved eval data to {out_path}/")
    print(f"           acc.npz ({acc['n']:,} samples)  meta.json")


def main(args):
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    # ── Resolve requested plot set ────────────────────────────────────────────
    # RQ-level shortcuts expand to their member plots.
    _RQ_PLOTS = {
        "q1": {"acc_scatter", "per_domain_table", "threshold_by_domain",
               "efficiency", "pareto"},
        "q2": {"acc_by_dim", "accuracy_by_domain", "error_by_regime",
               "exploration_budget", "consistency_rate",
               "response_surface_comparison", "paper_landscape"},
        "q3": {"sensitivity_surrogate", "sensitivity_compare", "sobol_by_domain",
               "response_surfaces", "conditional_sensitivity",
               "response_surfaces_full", "phase_diagrams", "paper_landscape",
               "find_landscape_a", "landscape_a_compare", "find_landscape_b"},
        "q4": {"counterfactual"},
    }
    _ALL_PLOTS = set().union(*_RQ_PLOTS.values()) | {"summary_panel"}

    if args.plots:
        requested: set[str] = set()
        for token in args.plots.replace(",", " ").split():
            token = token.lower()
            if token in _RQ_PLOTS:
                requested |= _RQ_PLOTS[token]
            elif token in _ALL_PLOTS:
                requested.add(token)
            else:
                print(f"  [warn] Unknown plot name '{token}' — ignored.")
        if not requested:
            print("No valid plot names given. Exiting.")
            return
    else:
        requested = _ALL_PLOTS.copy()

    def want(*names):
        return any(n in requested for n in names)

    print(f"Plots to generate: {sorted(requested)}")

    # ── Data loading — only what is needed ───────────────────────────────────
    # Groups:
    #   acc_group   : all Q1/Q2 plots + summary
    #   timing_group: efficiency, pareto, exploration_budget, summary
    #   ts_group    : domain/sobol_scenarios (Q3/Q4 + domain plots)
    #   sobol_group : sensitivity_surrogate, sensitivity_compare, sobol_by_domain, summary

    _ACC_PLOTS    = _RQ_PLOTS["q1"] | _RQ_PLOTS["q2"] | {"summary_panel"}
    _TIMING_PLOTS = {"efficiency", "pareto", "exploration_budget", "summary_panel"}
    _TS_PLOTS     = (_RQ_PLOTS["q3"] | _RQ_PLOTS["q4"]
                     | {"per_domain_table", "threshold_by_domain",
                        "accuracy_by_domain",
                        "efficiency", "pareto", "exploration_budget",
                        "response_surface_comparison",
                        "find_landscape_a", "landscape_a_compare",
                        "find_landscape_b"})
    _SOBOL_PLOTS  = {"sensitivity_surrogate", "sensitivity_compare",
                     "sobol_by_domain", "summary_panel"}

    need_acc    = bool(requested & _ACC_PLOTS)
    need_ts     = bool(requested & _TS_PLOTS)
    need_timing = bool(requested & _TIMING_PLOTS)
    need_sobol  = bool(requested & _SOBOL_PLOTS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _mpath = args.model_path or MODEL_PATH
    print(f"\nLoading surrogate model from {_mpath} (device={device})...")
    model, trained_dims = _load_model(device, model_path=args.model_path)
    print(f"  Model: {type(model).__name__}  "
          f"{sum(p.numel() for p in model.parameters())} parameters")

    # Resolve dims filter: explicit --dims > auto-detected from run_config.json
    if args.dims:
        dims_filter = [int(d) for d in args.dims.replace(",", " ").split()]
    else:
        dims_filter = trained_dims
    if dims_filter is not None:
        print(f"  Dims filter: {dims_filter}  "
              f"(source: {'--dims arg' if args.dims else 'run_config.json'})")

    # Resolve data directories: --data arg overrides built-in DATA_DIRS
    if args.data:
        data_dirs = [Path(p) for p in args.data]
    else:
        data_dirs = DATA_DIRS
    print(f"  Data dirs: {[str(d) for d in data_dirs]}")

    # Build scenario-filtered file list if --scenarios is given.
    # First match across dirs wins (avoids double-loading when the same ID
    # exists in both e.g. measurements_v6 and synthetic_v6).
    _files_filter = None
    if args.scenarios:
        sc_ids = {int(s.strip()) for s in args.scenarios.replace(",", " ").split()
                  if s.strip().isdigit()}
        import re as _re
        _files_filter = []
        _seen_ids: set = set()
        for d in data_dirs:
            for jf in sorted(Path(d).glob("results_scenario_*.json")):
                m = _re.search(r"results_scenario_(\d+)\.json", jf.name)
                if m:
                    sc_id = int(m.group(1))
                    if sc_id in sc_ids and sc_id not in _seen_ids:
                        _files_filter.append(jf)
                        _seen_ids.add(sc_id)
        print(f"  Scenario filter: {sorted(sc_ids)} → {len(_files_filter)} files")

    acc = domain = sobol_scenarios = domain_short = domain_order = None
    t_aabb = t_mfmc = t_mc = None
    ts = {}
    timing = acc_metrics = sobol_results = None

    if need_acc:
        print("\nLoading accuracy data and running surrogate inference...")
        acc = _load_accuracy_data(data_dirs, model, device,
                                  max_samples=args.max_samples,
                                  dims_filter=dims_filter,
                                  files=_files_filter)
        print(f"  n={acc['n']:,}  finite_aabb={np.isfinite(acc['i_aabb']).sum():,}  "
              f"finite_mfmc={np.isfinite(acc['i_mfmc']).sum():,}")

    if need_ts:
        # _load_timing_and_sobol also needs acc for domain alignment;
        # load a minimal acc if not already loaded.
        if acc is None:
            print("\nLoading accuracy data (needed for domain alignment)...")
            acc = _load_accuracy_data(data_dirs, model, device,
                                      max_samples=args.max_samples,
                                      dims_filter=dims_filter,
                                      files=_files_filter)
        print("\nLoading timing, Sobol parameters, and domain labels from JSON...")
        ts = _load_timing_and_sobol(data_dirs, acc,
                                    max_scenarios=args.max_scenarios,
                                    files=_files_filter)
        t_aabb          = ts["t_aabb"]
        t_mfmc          = ts["t_mfmc"]
        t_mc            = ts["t_mc"]
        domain          = ts["domain"]
        sobol_scenarios = ts["sobol_scenarios"]
        domain_short    = ts["_domain_short"]
        domain_order    = ts["_domain_order"]
    elif need_acc:
        # Domain not available — use empty arrays so domain plots skip gracefully
        domain       = np.array([])
        domain_short = {}
        domain_order = []

    if need_sobol and sobol_scenarios:
        print("\nComputing Sobol sensitivity indices (this may take a minute)...")
        sobol_results = _sobol_per_scenario(sobol_scenarios)

    # ── Export pre-computed data (for combined figures) ───────────────────────
    if getattr(args, "export_data", None) and acc is not None:
        _export_eval_data(
            out_path     = Path(args.export_data),
            acc          = acc,
            ts           = ts if need_ts else {},
            domain       = domain       if need_ts else [],
            domain_short = domain_short if need_ts else {},
            domain_order = domain_order if need_ts else [],
            sobol_results= sobol_results or [],
        )

    # ── Q1 ────────────────────────────────────────────────────────────────────
    if want(*_RQ_PLOTS["q1"]):
        print("\n" + "=" * 60)
        print("Q1 -- Accuracy and efficiency of I(theta) estimation")
        print("=" * 60)
    q1 = out_root / "Q1_estimation"

    if want("acc_scatter"):
        q1.mkdir(parents=True, exist_ok=True)
        acc_metrics = plot_acc_scatter(acc, q1)

    if want("per_domain_table"):
        q1.mkdir(parents=True, exist_ok=True)
        plot_per_domain_table(acc, domain, q1, threshold=0.85,
                              domain_short=domain_short, domain_order=domain_order)

    if want("threshold_by_domain"):
        q1.mkdir(parents=True, exist_ok=True)
        plot_threshold_by_domain(acc, domain, q1,
                                 domain_short=domain_short, domain_order=domain_order)

    if want("efficiency"):
        q1.mkdir(parents=True, exist_ok=True)
        timing = plot_efficiency(acc, t_aabb, t_mfmc, q1,
                                 t_mc_all=t_mc,
                                 us_aabb=ts.get("us_aabb", 0.0),
                                 us_mfmc=ts.get("us_mfmc", 0.0),
                                 us_mc=ts.get("us_mc",   0.0))

    if want("pareto"):
        q1.mkdir(parents=True, exist_ok=True)
        if timing is None:
            print("  [warn] pareto needs efficiency — running efficiency first.")
            timing = plot_efficiency(acc, t_aabb, t_mfmc, q1,
                                     t_mc_all=t_mc,
                                     us_aabb=ts.get("us_aabb", 0.0),
                                     us_mfmc=ts.get("us_mfmc", 0.0),
                                     us_mc=ts.get("us_mc",   0.0))
        plot_pareto(acc, timing, q1)

    # ── Q2 ────────────────────────────────────────────────────────────────────
    if want(*_RQ_PLOTS["q2"]):
        print("\n" + "=" * 60)
        print("Q2 -- Surrogate model accuracy, generalization, runtime")
        print("=" * 60)
    q2 = out_root / "Q2_surrogate"

    if want("acc_by_dim"):
        q2.mkdir(parents=True, exist_ok=True)
        plot_acc_by_dim(acc, q2)

    if want("accuracy_by_domain"):
        q2.mkdir(parents=True, exist_ok=True)
        plot_accuracy_by_domain(acc, domain, q2,
                                domain_short=domain_short, domain_order=domain_order)

    if want("error_by_regime"):
        q2.mkdir(parents=True, exist_ok=True)
        plot_error_by_regime(acc, q2)

    if want("exploration_budget"):
        q2.mkdir(parents=True, exist_ok=True)
        if timing is None:
            print("  [warn] exploration_budget needs efficiency — running efficiency first.")
            timing = plot_efficiency(acc, t_aabb, t_mfmc,
                                     out_root / "Q1_estimation",
                                     t_mc_all=t_mc,
                                     us_aabb=ts.get("us_aabb", 0.0),
                                     us_mfmc=ts.get("us_mfmc", 0.0),
                                     us_mc=ts.get("us_mc",   0.0))
        plot_exploration_budget(timing, q2)

    if want("consistency_rate"):
        q2.mkdir(parents=True, exist_ok=True)
        plot_consistency_rate(acc, q2)


    if want("response_surface_comparison"):
        q2.mkdir(parents=True, exist_ok=True)
        # Collect domains already used in paper_landscape so we can show different ones
        _landscape_domains: set = set()
        _sc_b_ids = ([int(x) for x in args.landscape_b.split(",")]
                     if args.landscape_b else [])
        for sc in (sobol_scenarios or []):
            sid = sc.get("scenario_id", -1) if isinstance(sc, dict) else -1
            dom = sc.get("domain", "") if isinstance(sc, dict) else ""
            if sid == args.landscape_a or sid in _sc_b_ids:
                _landscape_domains.add(dom)
        plot_response_surface_comparison(sobol_scenarios, q2, gamma=args.gamma,
                                         exclude_domains=_landscape_domains)

    # ── Q3 ────────────────────────────────────────────────────────────────────
    if want(*_RQ_PLOTS["q3"]):
        print("\n" + "=" * 60)
        print("Q3 -- Sensitivity analysis")
        print("=" * 60)
    q3 = out_root / "Q3_sensitivity"

    if want("sensitivity_surrogate"):
        q3.mkdir(parents=True, exist_ok=True)
        plot_sensitivity_surrogate(sobol_results, q3)

    if want("sensitivity_compare"):
        q3.mkdir(parents=True, exist_ok=True)
        plot_sensitivity_compare(sobol_results, q3)

    if want("sobol_by_domain"):
        q3.mkdir(parents=True, exist_ok=True)
        plot_sobol_by_domain(sobol_results, q3,
                             domain_short=domain_short, domain_order=domain_order)

    if want("response_surfaces"):
        q3.mkdir(parents=True, exist_ok=True)
        plot_response_surfaces(sobol_scenarios, q3)

    if want("conditional_sensitivity"):
        q3.mkdir(parents=True, exist_ok=True)
        plot_conditional_sensitivity(sobol_scenarios, q3)

    if want("response_surfaces_full"):
        q3.mkdir(parents=True, exist_ok=True)
        plot_response_surfaces_full(sobol_scenarios, q3, gamma=args.gamma)

    if want("phase_diagrams"):
        q3.mkdir(parents=True, exist_ok=True)
        plot_phase_diagrams(sobol_scenarios, q3, gamma=args.gamma)

    if want("find_landscape_a"):
        find_best_landscape_a(sobol_scenarios, gamma=args.gamma)

    if want("find_landscape_b"):
        find_best_landscape_b(sobol_scenarios, exclude_sc_a=args.landscape_a)

    if want("landscape_a_compare"):
        _sc_compare = [int(x) for x in args.landscape_b.split(",")] \
                      if args.landscape_b else [1, 3, 11]
        q3.mkdir(parents=True, exist_ok=True)
        plot_landscape_a_compare(sobol_scenarios, q3, _sc_compare,
                                 gamma=args.gamma)

    if want("paper_landscape"):
        _sc_b = [int(x) for x in args.landscape_b.split(",")] \
                if args.landscape_b else None
        q3.mkdir(parents=True, exist_ok=True)
        plot_paper_landscape(sobol_scenarios, q3, gamma=args.gamma,
                             sc_a=args.landscape_a, sc_b=_sc_b)
        # Also written to Q2 — primary MC-vs-Surrogate visual for RQ2
        q2.mkdir(parents=True, exist_ok=True)
        plot_paper_landscape(sobol_scenarios, q2, gamma=args.gamma,
                             sc_a=args.landscape_a, sc_b=_sc_b)

    # ── Q4 ────────────────────────────────────────────────────────────────────
    if want("counterfactual"):
        print("\n" + "=" * 60)
        print("Q4 -- Counterfactual: minimal fix to restore consistency")
        print("=" * 60)
        q4 = out_root / "Q4_counterfactual"
        q4.mkdir(parents=True, exist_ok=True)

        cf_sc = sobol_scenarios
        cf_ids_arg = getattr(args, "cf_scenarios", None)
        if cf_ids_arg:
            wanted_ids = {int(x) for x in cf_ids_arg.replace(",", " ").split()}
            # Deduplicate: keep first match per scenario_id to avoid
            # double-counting IDs that appear in both data directories
            _seen_ids = set()
            cf_sc = []
            for s in sobol_scenarios:
                if isinstance(s, dict) and s.get("scenario_id") in wanted_ids:
                    sid = s.get("scenario_id")
                    if sid not in _seen_ids:
                        cf_sc.append(s)
                        _seen_ids.add(sid)
            print(f"  Filtering to {len(cf_sc)} scenario(s): {sorted(wanted_ids)}")

        plot_counterfactual(cf_sc, q4, model=model,
                            device=device, gamma=args.gamma,
                            use_v2=_is_v2_model(model))

    # ── Summary panel ─────────────────────────────────────────────────────────
    if want("summary_panel"):
        if acc_metrics is None:
            acc_metrics = {"AABB": {}, "MFMC": {}, "Surrogate": {}}
        if timing is None:
            timing = {}
        if sobol_results is None:
            sobol_results = []
        plot_summary_panel(acc, acc_metrics, timing, sobol_results, out_root)

    print(f"\nAll outputs saved to: {out_root}")


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Plot names (--plots):
  RQ shortcuts : q1  q2  q3  q4          (all plots in that group)
  Q1 plots     : acc_scatter  per_domain_table  threshold_by_domain
                 efficiency  pareto
  Q2 plots     : acc_by_dim  accuracy_by_domain  error_by_regime
                 exploration_budget  consistency_rate
                 response_surface_comparison
  Q3 plots     : sensitivity_surrogate  sensitivity_compare  sobol_by_domain
                 response_surfaces  conditional_sensitivity
                 response_surfaces_full  phase_diagrams  paper_landscape
  Q4 plots     : counterfactual
  Other        : summary_panel

Examples:
  --plots paper_landscape
  --plots paper_landscape,counterfactual
  --plots q4
  --plots q1,q2
""")
    p.add_argument("--model-path",     type=str, default=None,
                   help="Path to a surrogate checkpoint (.pt). Supports both v1 "
                        "(DeepSetsZonotope, saved by surrogate/train.py) and v2 "
                        "(ProductSetTransformer etc., saved by surrogate/train_compare.py). "
                        "Defaults to surrogate/model.pt.")
    p.add_argument("--data",          type=str, nargs='+', default=None,
                   help="One or more data directories containing "
                        "results_scenario_*.json files. Overrides the built-in "
                        "DATA_DIRS constant (e.g. --data data/measurements_v6 "
                        "data/measurements_cps_v6).")
    p.add_argument("--scenarios",     type=str, default=None,
                   help="Comma-separated scenario IDs to restrict analysis to "
                        "(e.g. --scenarios 1,3,7,25). Files not matching any "
                        "listed ID are skipped. Useful for val-split evaluation.")
    p.add_argument("--dims",           type=str, default=None,
                   help="Comma-separated zonotope dimensions to evaluate on "
                        "(e.g. --dims 2  or  --dims 2,3). Auto-detected from "
                        "run_config.json next to the checkpoint when present. "
                        "Omit to use all dimensions.")
    p.add_argument("--output",        type=str, default="results/paper_figures",
                   help="Root output directory. Sub-folders Q1..Q4 are created automatically.")
    p.add_argument("--plots",         type=str, default=None,
                   help="Comma- or space-separated list of plot names / RQ shortcuts. "
                        "Omit to regenerate all.")
    p.add_argument("--max_samples",   type=int, default=None,
                   help="Cap total samples loaded for a quick test")
    p.add_argument("--max_scenarios", type=int, default=None,
                   help="Cap number of scenarios for timing/Sobol (quick test)")
    p.add_argument("--gamma",         type=float, default=0.5,
                   help="Inconsistency threshold for Q4 counterfactual search")
    p.add_argument("--landscape_a",   type=int,   default=None,
                   help="Scenario id for paper_landscape panel A (e.g. --landscape_a 1)")
    p.add_argument("--landscape_b",   type=str,   default=None,
                   help="Comma-separated scenario ids for panel B (e.g. --landscape_b 49,1,35)")
    p.add_argument("--cf_scenarios",  type=str,   default=None,
                   help="Comma-separated scenario ids to use for Q4 counterfactual "
                        "(e.g. --cf_scenarios 35,49,7). Omit to use all loaded scenarios.")
    p.add_argument("--export-data",   type=str,   default=None,
                   help="If given, save all computed accuracy/timing/domain/Sobol data "
                        "to this directory (creates acc.npz + meta.json) so that "
                        "run_combined_figures.py can load them without re-running inference.")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
