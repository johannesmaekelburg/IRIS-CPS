#!/usr/bin/env python3
"""
generate_rq2_figures.py
=======================
RQ2 — GNN Surrogate: Learning I(θ)

Produces one figure with three panels:

  Panel 1 — Accuracy scatter:
    Î(θ) vs I_MC(θ) on held-out test scenarios, one colour per scenario.
    Annotated with Spearman ρ and MAE.

  Panel 2 — Generalization bar chart:
    Spearman ρ per scenario, val scenarios (in-domain) vs test scenarios
    (cross-domain), grouped by train/test split.

  Panel 3 — Runtime comparison:
    Mean per-sample inference time (ms) for MC, MFMC, and GNN.
    Log-scale x-axis with speedup annotations.

Reads from:
  --results_dir   results/surrogate/          (contains results.json, best_model*.pt)
  --data_root     data/surrogate/             (surrogate dataset)
  --measurements  data/measurements/          (for MC/MFMC timing)

Run from project root:
    python -m src.learned_surrogate.generate_rq2_figures \\
        --results_dir results/surrogate \\
        --data_root data/surrogate \\
        --measurements data/measurements
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       10,
    "axes.labelsize":  10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi":      150,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
})

_C_VAL  = "#2166ac"   # blue  — val / in-domain
_C_TEST = "#d6604d"   # red   — test / cross-domain
_C_GNN  = "#2ecc71"   # green — GNN bar
_C_MFMC = "#9b59b6"   # purple — MFMC bar
_C_MC   = "#e74c3c"   # red   — MC bar


# ── Model + data helpers ───────────────────────────────────────────────────────

def _load_model_and_config(results_dir: Path, device: str):
    """Load best model checkpoint from results_dir."""
    import torch
    from .model import ZonotopeGINE
    from .config import SurrogateConfig

    # Prefer best_model_test.pt, fall back to best_model.pt
    ckpt_path = results_dir / "best_model_test.pt"
    if not ckpt_path.exists():
        ckpt_path = results_dir / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found in {results_dir}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = ckpt["config"]

    # Reconstruct config
    from .config import ModelConfig, TrainConfig, PretrainConfig
    import dataclasses

    def _dc_from_dict(cls, d):
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in fields})

    model_cfg = _dc_from_dict(ModelConfig, cfg_dict["model"])
    train_cfg = _dc_from_dict(TrainConfig, cfg_dict["train"])

    model = ZonotopeGINE(model_cfg)
    # Use strict=False to ignore pretrain heads that may still be in the checkpoint
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    pretrain_prefixes = ("volume_head.", "containment_head.",
                         "pairwise_aabb_head.", "affine_map_head.")
    unexpected_non_pretrain = [k for k in unexpected
                                if not k.startswith(pretrain_prefixes)]
    if unexpected_non_pretrain:
        raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected_non_pretrain}")
    if missing:
        raise RuntimeError(f"Missing keys in checkpoint: {missing}")
    model.to(device)
    model.eval()

    return model, train_cfg, ckpt.get("val_metrics", {}), ckpt.get("test_metrics", {})


def _collect_predictions(model, loader, device):
    """Return (preds, targets, scenario_ids) arrays."""
    import torch
    preds_list, targets_list, scenario_list = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch).squeeze(-1).cpu().numpy()
            y   = batch.y.squeeze(-1).cpu().numpy()
            s   = batch.scenario_idx.squeeze(-1).cpu().numpy()
            preds_list.append(out)
            targets_list.append(y)
            scenario_list.append(s)
    return (np.concatenate(preds_list),
            np.concatenate(targets_list),
            np.concatenate(scenario_list))


def _per_scenario_rho(preds, targets, scenario_ids):
    """Return {scenario_idx: spearman_rho} for each unique scenario."""
    result = {}
    for s in np.unique(scenario_ids):
        m = scenario_ids == s
        if m.sum() < 5:
            continue
        rho, _ = spearmanr(targets[m], preds[m])
        result[int(s)] = float(rho)
    return result


def _measure_gnn_time_ms(model, loader, device, n_repeats: int = 5) -> float:
    """Mean per-sample GNN inference time in milliseconds."""
    import torch
    # Warm up
    for batch in loader:
        batch = batch.to(device)
        with torch.no_grad():
            model(batch)
        break

    times = []
    for _ in range(n_repeats):
        n_samples = 0
        t0 = time.perf_counter()
        for batch in loader:
            batch = batch.to(device)
            with torch.no_grad():
                model(batch)
            n_samples += batch.num_graphs
        dt = time.perf_counter() - t0
        times.append(dt / n_samples * 1000)   # ms per sample
    return float(np.mean(times))


def _load_timing_from_measurements(measurements_dir: Path):
    """
    Extract mean per-sample timing (ms) for MC and MFMC from measurement JSONs.
    Returns (mc_ms, mfmc_ms) or (None, None) if not available.
    """
    import glob
    mc_times, mfmc_times = [], []
    for f in sorted(glob.glob(str(measurements_dir / "results_scenario_*.json"))):
        try:
            data = json.loads(Path(f).read_text(encoding="utf-8"))
        except Exception:
            continue
        for exp in data.get("experiments", []):
            # Look for timing fields — various naming conventions observed
            for key in ("timing_mc_ms", "timing_I_theta_ms", "mc_time_ms"):
                v = exp.get(key)
                if v is not None:
                    try:
                        mc_times.append(float(v))
                    except (TypeError, ValueError):
                        pass
                    break
            for key in ("timing_mfmc_ms", "timing_I_MF_ms", "mfmc_time_ms"):
                v = exp.get(key)
                if v is not None:
                    try:
                        mfmc_times.append(float(v))
                    except (TypeError, ValueError):
                        pass
                    break

    mc_ms   = float(np.median(mc_times))   if mc_times   else None
    mfmc_ms = float(np.median(mfmc_times)) if mfmc_times else None
    return mc_ms, mfmc_ms


# ── Figure panels ──────────────────────────────────────────────────────────────

def _panel_accuracy(ax, preds, targets, scenario_ids):
    """Panel 1: scatter Î vs I_MC coloured by scenario."""
    unique_s = np.unique(scenario_ids)
    cmap = plt.cm.get_cmap("tab20", len(unique_s))
    colors = {s: cmap(i) for i, s in enumerate(unique_s)}

    for s in unique_s:
        m = scenario_ids == s
        ax.scatter(targets[m], preds[m], s=14, alpha=0.5,
                   color=colors[s], linewidths=0, rasterized=True,
                   label=f"S{s:02d}")

    lo = min(targets.min(), preds.min())
    hi = max(targets.max(), preds.max())
    margin = (hi - lo) * 0.04
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
            "k--", lw=0.9, alpha=0.4)

    rho, _ = spearmanr(targets, preds)
    mae    = float(np.mean(np.abs(preds - targets)))
    ax.text(0.04, 0.96,
            fr"$\rho={rho:.3f}$""\n"f"MAE$={mae:.4f}$",
            transform=ax.transAxes, va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.7))

    ax.set_xlabel(r"$I(\theta)$ — MC")
    ax.set_ylabel(r"$\hat{I}(\theta)$ — GNN")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_aspect("equal", adjustable="box")
    ax.spines[["top", "right"]].set_visible(False)

    # Compact legend — only if ≤ 15 scenarios
    if len(unique_s) <= 15:
        ax.legend(fontsize=6.5, ncol=2, loc="lower right",
                  framealpha=0.7, handlelength=0.8)


def _panel_generalization(ax, val_rho: dict, test_rho: dict):
    """Panel 2: grouped bar chart of Spearman ρ per scenario."""
    val_items  = sorted(val_rho.items())
    test_items = sorted(test_rho.items())

    # Val scenarios on left, test on right with a gap
    n_val  = len(val_items)
    n_test = len(test_items)
    x_val  = np.arange(n_val)
    x_test = np.arange(n_val + 1, n_val + 1 + n_test)   # gap of 1

    bars_val  = ax.bar(x_val,  [v for _, v in val_items],  color=_C_VAL,
                       alpha=0.85, width=0.7, label="Val (in-domain)")
    bars_test = ax.bar(x_test, [v for _, v in test_items], color=_C_TEST,
                       alpha=0.85, width=0.7, label="Test (cross-domain)")

    # Value annotations
    for bar in list(bars_val) + list(bars_test):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.2f}", ha="center", va="bottom", fontsize=6.5)

    all_x      = np.concatenate([x_val, x_test])
    all_labels = [f"S{s:02d}" for s, _ in val_items + test_items]
    ax.set_xticks(all_x)
    ax.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(r"Spearman $\rho$")
    ax.set_ylim(0, 1.12)
    ax.axhline(1.0, color="grey", ls="--", lw=0.7, alpha=0.5)
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    # Divider between val and test
    if n_val > 0 and n_test > 0:
        ax.axvline(n_val - 0.3, color="grey", lw=0.8, ls=":", alpha=0.6)


def _panel_runtime(ax, gnn_ms: float, mc_ms: float | None, mfmc_ms: float | None):
    """Panel 3: horizontal bar chart of per-sample inference time (ms)."""
    methods, times, colors = [], [], []

    if mc_ms is not None:
        methods.append("MC")
        times.append(mc_ms)
        colors.append(_C_MC)
    if mfmc_ms is not None:
        methods.append("MFMC")
        times.append(mfmc_ms)
        colors.append(_C_MFMC)
    methods.append("GNN")
    times.append(gnn_ms)
    colors.append(_C_GNN)

    y = np.arange(len(methods))
    ax.barh(y, times, color=colors, alpha=0.85, height=0.55)

    # Speedup annotations relative to slowest (MC)
    ref = max(times)
    for i, (t, m) in enumerate(zip(times, methods)):
        speedup = ref / t
        label = f"{t:.1f} ms"
        if speedup > 1.5:
            label += f"  ({speedup:.0f}×)"
        ax.text(t * 1.05, i, label, va="center", fontsize=8.5)

    ax.set_yticks(y)
    ax.set_yticklabels(methods)
    ax.set_xlabel("Time per sample (ms)")
    ax.set_xscale("log")
    ax.spines[["top", "right"]].set_visible(False)


# ── Main ───────────────────────────────────────────────────────────────────────

def _resolve_results_dir(base: str) -> Path:
    """Return the path to use as results_dir.

    If *base* directly contains a checkpoint (best_model*.pt), use it as-is.
    Otherwise look for timestamped subdirectories (YYYYMMDD_HHMMSS or any name)
    and return the most recently modified one.
    """
    p = Path(base)
    if (p / "best_model.pt").exists() or (p / "best_model_test.pt").exists():
        return p
    # Find subdirectories that contain a checkpoint
    candidates = sorted(
        [d for d in p.iterdir() if d.is_dir() and
         ((d / "best_model.pt").exists() or (d / "best_model_test.pt").exists())],
        key=lambda d: d.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found under {p}. "
            "Pass --results_dir pointing to a specific run directory."
        )
    chosen = candidates[-1]   # most recently modified
    print(f"  Auto-detected latest run: {chosen.name}")
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate RQ2 surrogate evaluation figures")
    parser.add_argument("--results_dir", default="results/surrogate",
                        help="Surrogate results root or specific run directory. "
                             "If a root is given, the latest run is used automatically.")
    parser.add_argument("--data_root", default="data/surrogate",
                        help="Surrogate dataset root")
    parser.add_argument("--measurements", default="data/measurements",
                        help="Measurement JSON directory (for MC/MFMC timing)")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: results/paper_figures)")
    parser.add_argument("--device", default="cpu",
                        help="Device for inference timing (default: cpu)")
    args = parser.parse_args()

    results_dir = _resolve_results_dir(args.results_dir)
    data_root   = Path(args.data_root)
    out_dir     = Path(args.output) if args.output else Path("results/paper_figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Lazy imports (torch / torch_geometric only needed at runtime) ──────────
    import torch
    from torch_geometric.loader import DataLoader
    from .dataset import InconsistencyDataset
    from .config import load_config

    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # ── Load config + model ────────────────────────────────────────────────────
    print("Loading model...")
    model, train_cfg, val_ckpt_metrics, test_ckpt_metrics = \
        _load_model_and_config(results_dir, device)

    # ── Build val & test datasets ──────────────────────────────────────────────
    print("Building datasets...")
    val_ds = InconsistencyDataset(
        data_root, train_cfg.scenarios_train,
        split="val", val_fraction=train_cfg.val_fraction,
    )
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    test_loader = None
    if train_cfg.scenarios_test:
        try:
            test_ds = InconsistencyDataset(
                data_root, train_cfg.scenarios_test, split="all",
            )
            test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
        except FileNotFoundError as e:
            print(f"  Warning: test dataset unavailable — {e}")

    # ── Collect predictions ────────────────────────────────────────────────────
    print("Running inference on val set...")
    val_preds, val_targets, val_sids = _collect_predictions(model, val_loader, device)
    val_rho = _per_scenario_rho(val_preds, val_targets, val_sids)

    test_preds = test_targets = test_sids = None
    test_rho = {}
    if test_loader is not None:
        print("Running inference on test set...")
        test_preds, test_targets, test_sids = \
            _collect_predictions(model, test_loader, device)
        test_rho = _per_scenario_rho(test_preds, test_targets, test_sids)

    # Use test for accuracy scatter if available, else val
    acc_preds   = test_preds   if test_preds   is not None else val_preds
    acc_targets = test_targets if test_targets is not None else val_targets
    acc_sids    = test_sids    if test_sids    is not None else val_sids

    # ── Timing ────────────────────────────────────────────────────────────────
    print("Measuring GNN inference time...")
    gnn_ms = _measure_gnn_time_ms(model, val_loader, device)
    print(f"  GNN: {gnn_ms:.2f} ms/sample")

    mc_ms, mfmc_ms = _load_timing_from_measurements(Path(args.measurements))
    if mc_ms is None:
        # Fallback: use known value from data generation (1783s / 5000 samples)
        mc_ms = 356.0
        print(f"  MC timing not found in measurements — using fallback {mc_ms} ms/sample")
    else:
        print(f"  MC: {mc_ms:.2f} ms/sample  |  MFMC: {mfmc_ms:.2f} ms/sample")

    # ── Build figure ───────────────────────────────────────────────────────────
    print("Building figure...")
    has_test = test_preds is not None

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    _panel_accuracy(axes[0], acc_preds, acc_targets, acc_sids)
    _panel_generalization(axes[1], val_rho, test_rho)
    _panel_runtime(axes[2], gnn_ms, mc_ms, mfmc_ms)

    fig.tight_layout(w_pad=2.5)

    for ext in ("pdf", "png"):
        p = out_dir / f"figRQ2_surrogate.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"  Saved: {p.name}")
    plt.close(fig)

    # ── Print summary metrics ──────────────────────────────────────────────────
    print("\n--- Summary ---")
    if val_rho:
        print(f"Val  Spearman ρ: mean={np.mean(list(val_rho.values())):.4f}  "
              f"per-scenario: {val_rho}")
    if test_rho:
        print(f"Test Spearman ρ: mean={np.mean(list(test_rho.values())):.4f}  "
              f"per-scenario: {test_rho}")
    print(f"GNN inference: {gnn_ms:.2f} ms/sample")
    if mc_ms:
        print(f"Speedup vs MC: {mc_ms / gnn_ms:.0f}×")
    print(f"\nDone. Figures saved to {out_dir}")


if __name__ == "__main__":
    main()
