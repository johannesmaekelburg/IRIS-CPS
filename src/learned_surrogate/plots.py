"""Plotting utilities for the inconsistency surrogate training pipeline.

All functions write figures to *output_dir* and return nothing.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Consistent style
sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = sns.color_palette("Set2")


# ---------------------------------------------------------------------------
# CSV logging helpers
# ---------------------------------------------------------------------------

class MetricsLogger:
    """Append-mode CSV logger for per-epoch metrics."""

    def __init__(self, path: Path):
        self.path = path
        self._file = None
        self._writer = None
        self._fields: Optional[List[str]] = None

    def log(self, row: Dict[str, float]) -> None:
        if self._file is None:
            self._fields = list(row.keys())
            self._file = open(self.path, "w", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=self._fields)
            self._writer.writeheader()
        self._writer.writerow({k: f"{v:.6g}" if isinstance(v, float) else v
                               for k, v in row.items()})
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()


def read_metrics_csv(path: Path) -> Dict[str, np.ndarray]:
    """Read a metrics CSV into {column_name: array}."""
    import csv as _csv
    with open(path) as f:
        reader = _csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


# ---------------------------------------------------------------------------
# Pretraining plots
# ---------------------------------------------------------------------------

def plot_pretrain_curves(output_dir: Path) -> None:
    """Plot per-task pretraining loss curves from the CSV log."""
    csv_path = output_dir / "pretrain_log.csv"
    if not csv_path.exists():
        return

    data = read_metrics_csv(csv_path)
    epochs = data.get("epoch")
    if epochs is None:
        return

    # Collect task names from columns
    train_tasks = [c.replace("train_", "") for c in data if c.startswith("train_")]
    val_tasks = [c.replace("val_", "") for c in data if c.startswith("val_")]
    tasks = sorted(set(train_tasks) & set(val_tasks))

    if not tasks:
        return

    n_tasks = len(tasks)
    fig, axes = plt.subplots(1, n_tasks, figsize=(5 * n_tasks, 4), squeeze=False)

    for i, task in enumerate(tasks):
        ax = axes[0, i]
        ax.plot(epochs, data[f"train_{task}"], label="train", color=PALETTE[0])
        ax.plot(epochs, data[f"val_{task}"], label="val", color=PALETTE[1])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(task.replace("_", " ").title())
        ax.legend()
        ax.set_yscale("log")

    fig.suptitle("Pretraining Loss Curves", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "pretrain_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fine-tuning training curve plots
# ---------------------------------------------------------------------------

def plot_finetune_curves(output_dir: Path) -> None:
    """Plot fine-tuning training curves from the CSV log."""
    csv_path = output_dir / "finetune_log.csv"
    if not csv_path.exists():
        return

    data = read_metrics_csv(csv_path)
    epochs = data.get("epoch")
    if epochs is None:
        return

    has_test = "test_mse" in data

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Train loss
    ax = axes[0, 0]
    ax.plot(epochs, data["train_loss"], color=PALETTE[0])
    ax.set_ylabel("Huber / MSE Loss")
    ax.set_title("Training Loss")
    ax.set_yscale("log")

    # Val/Test MSE & MAE
    ax = axes[0, 1]
    ax.plot(epochs, data["val_mse"], color=PALETTE[1], label="Val MSE")
    ax.plot(epochs, data["val_mae"], color=PALETTE[2], label="Val MAE")
    if has_test:
        ax.plot(epochs, data["test_mse"], color=PALETTE[1], ls="--", label="Test MSE")
        ax.plot(epochs, data["test_mae"], color=PALETTE[2], ls="--", label="Test MAE")
    ax.set_ylabel("Error")
    ax.set_title("Validation / Test Error")
    ax.legend(fontsize=8)
    ax.set_yscale("log")

    # Val/Test R2
    ax = axes[1, 0]
    ax.plot(epochs, data["val_r2"], color=PALETTE[3], label="Val")
    if has_test:
        ax.plot(epochs, data["test_r2"], color=PALETTE[3], ls="--", label="Test")
    ax.axhline(1.0, ls="--", color="grey", alpha=0.5)
    ax.set_ylabel("R$^2$")
    ax.set_title("R$^2$")
    ax.set_xlabel("Epoch")
    if has_test:
        ax.legend(fontsize=8)

    # Val/Test Spearman
    ax = axes[1, 1]
    ax.plot(epochs, data["val_spearman_rho"], color=PALETTE[4], label="Val")
    if has_test:
        ax.plot(epochs, data["test_spearman_rho"], color=PALETTE[4], ls="--", label="Test")
    ax.axhline(1.0, ls="--", color="grey", alpha=0.5)
    ax.set_ylabel("Spearman $\\rho$")
    ax.set_title("Rank Correlation")
    ax.set_xlabel("Epoch")
    if has_test:
        ax.legend(fontsize=8)

    # Mark best epochs
    if "val_mse" in data:
        best_val_idx = int(np.argmin(data["val_mse"]))
        best_val_ep = epochs[best_val_idx]
        for a in axes.flat:
            a.axvline(best_val_ep, ls=":", color="red", alpha=0.4,
                      label=f"best val (ep {int(best_val_ep)})")
    if has_test:
        best_test_idx = int(np.argmin(data["test_mse"]))
        best_test_ep = epochs[best_test_idx]
        for a in axes.flat:
            a.axvline(best_test_ep, ls=":", color="blue", alpha=0.4,
                      label=f"best test (ep {int(best_test_ep)})")

    fig.suptitle("Fine-tuning Training Curves", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "finetune_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Prediction quality plots (best model)
# ---------------------------------------------------------------------------

def plot_predictions(
    preds: np.ndarray,
    targets: np.ndarray,
    scenario_ids: np.ndarray,
    split_name: str,
    output_dir: Path,
) -> None:
    """Generate prediction quality plots for a given data split.

    Creates:
      - scatter plot (true vs predicted, colored by scenario)
      - residual plot
      - error histogram
      - per-scenario error box plot
    """
    residuals = preds - targets
    abs_errors = np.abs(residuals)

    # Unique scenarios for coloring
    unique_scenarios = np.unique(scenario_ids)
    scenario_colors = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(unique_scenarios)}

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    # --- 1. Scatter: true vs predicted ---
    ax = axes[0, 0]
    for s in unique_scenarios:
        mask = scenario_ids == s
        ax.scatter(targets[mask], preds[mask], alpha=0.5, s=15,
                   color=scenario_colors[s], label=f"S{s:02d}", edgecolors="none")
    lo = min(targets.min(), preds.min())
    hi = max(targets.max(), preds.max())
    margin = (hi - lo) * 0.05
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
            "k--", alpha=0.4, lw=1)
    ax.set_xlabel("True $I(\\theta)$")
    ax.set_ylabel("Predicted $I(\\theta)$")
    ax.set_title(f"True vs Predicted ({split_name})")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.set_aspect("equal", adjustable="box")

    # --- 2. Residuals vs true ---
    ax = axes[0, 1]
    for s in unique_scenarios:
        mask = scenario_ids == s
        ax.scatter(targets[mask], residuals[mask], alpha=0.5, s=15,
                   color=scenario_colors[s], edgecolors="none")
    ax.axhline(0, color="k", ls="--", alpha=0.4)
    ax.set_xlabel("True $I(\\theta)$")
    ax.set_ylabel("Residual (pred $-$ true)")
    ax.set_title(f"Residuals ({split_name})")

    # --- 3. Error histogram ---
    ax = axes[1, 0]
    ax.hist(residuals, bins=40, color=PALETTE[0], edgecolor="white", alpha=0.8)
    ax.axvline(0, color="k", ls="--", alpha=0.4)
    ax.set_xlabel("Prediction Error")
    ax.set_ylabel("Count")
    ax.set_title(f"Error Distribution ({split_name})")
    # Annotate stats
    ax.text(0.97, 0.95,
            f"mean={residuals.mean():.4f}\nstd={residuals.std():.4f}\n"
            f"MAE={abs_errors.mean():.4f}",
            transform=ax.transAxes, va="top", ha="right", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # --- 4. Per-scenario box plot ---
    ax = axes[1, 1]
    scenario_data = []
    scenario_labels = []
    for s in unique_scenarios:
        mask = scenario_ids == s
        scenario_data.append(abs_errors[mask])
        scenario_labels.append(f"S{s:02d}")
    bp = ax.boxplot(scenario_data, labels=scenario_labels, patch_artist=True)
    for patch, s in zip(bp["boxes"], unique_scenarios):
        patch.set_facecolor(scenario_colors[s])
        patch.set_alpha(0.7)
    ax.set_ylabel("|Error|")
    ax.set_title(f"Per-Scenario Absolute Error ({split_name})")
    ax.tick_params(axis="x", rotation=45)

    fig.suptitle(f"Prediction Analysis — {split_name}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / f"predictions_{split_name.lower()}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_predictions_combined(
    val_preds: np.ndarray, val_targets: np.ndarray, val_scenarios: np.ndarray,
    test_preds: Optional[np.ndarray], test_targets: Optional[np.ndarray],
    test_scenarios: Optional[np.ndarray],
    output_dir: Path,
) -> None:
    """Side-by-side scatter: val and test on one figure."""
    n_cols = 2 if test_preds is not None else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5.5), squeeze=False)

    for col, (preds, targets, scenarios, name) in enumerate([
        (val_preds, val_targets, val_scenarios, "Validation"),
        (test_preds, test_targets, test_scenarios, "Test"),
    ]):
        if preds is None:
            continue
        ax = axes[0, col]
        unique_s = np.unique(scenarios)
        colors = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(unique_s)}
        for s in unique_s:
            m = scenarios == s
            ax.scatter(targets[m], preds[m], alpha=0.5, s=18,
                       color=colors[s], label=f"S{s:02d}", edgecolors="none")
        lo = min(targets.min(), preds.min())
        hi = max(targets.max(), preds.max())
        margin = (hi - lo) * 0.05
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                "k--", alpha=0.4, lw=1)

        # Compute metrics for annotation
        mse = np.mean((preds - targets) ** 2)
        ss_res = np.sum((targets - preds) ** 2)
        ss_tot = np.sum((targets - targets.mean()) ** 2)
        r2 = 1 - ss_res / max(ss_tot, 1e-8)
        ax.text(0.03, 0.97, f"R$^2$={r2:.3f}\nMSE={mse:.5f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        ax.set_xlabel("True $I(\\theta)$")
        ax.set_ylabel("Predicted $I(\\theta)$")
        ax.set_title(name)
        ax.legend(fontsize=7, ncol=2, loc="lower right")
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle("True vs Predicted Inconsistency", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "scatter_combined.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
