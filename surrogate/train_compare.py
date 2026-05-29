"""
Train and compare all model variants on the same data split.

Usage
-----
    python -m surrogate.train_compare                        # all models, defaults
    python -m surrogate.train_compare --epochs 30 --lr 1e-3
    python -m surrogate.train_compare --models deepsets_v2 flat_mlp
    python -m surrogate.train_compare --max_samples 50000    # quick test
"""

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .dataset_v2 import (ZonotopeDatasetV2, collate_fn,
                         N_DIM_FEAT_V2, N_GLOBAL_V2)
from .model import DeepSetsZonotope
from .models_v2 import MODEL_REGISTRY, create_model

ROOT = Path(__file__).resolve().parents[1]
DATA_DIRS = [
    ROOT / "data" / "measurements_v6",
    ROOT / "data" / "measurements_cps_v6",
]

COLORS = {
    "original": "#888888",
    "deepsets_v2": "#4878CF",
    "flat_mlp": "#6ACC65",
    "siamese": "#D65F5F",
    "set_transformer": "#B47CC7",
    "product_transformer": "#E5A832",
    "product_transformer_large": "#C44E52",
}


def _get_scenario_dim(json_file):
    with open(json_file) as f:
        data = json.load(f)
    if not data.get("experiments"):
        return data.get("dim", 0)
    c1 = data["experiments"][0]["post_state"]["uncertainty"]["source_center"]
    return len(c1)


def _forward(model, batch, model_name, device):
    """Dispatch forward pass based on model type."""
    if model_name == "original":
        return model(
            batch["per_dim_v1"].to(device),
            batch["mask"].to(device),
            batch["global_v1"].to(device),
        )
    elif model_name == "siamese":
        return model(
            batch["src_center"].to(device),
            batch["src_generators"].to(device),
            batch["src_gen_mask"].to(device),
            batch["tgt_center"].to(device),
            batch["tgt_generators"].to(device),
            batch["tgt_gen_mask"].to(device),
            batch["global_v2"].to(device),
        )
    else:
        return model(
            batch["per_dim_v2"].to(device),
            batch["mask"].to(device),
            batch["global_v2"].to(device),
        )


@torch.no_grad()
def collect_predictions(model, loader, model_name, device):
    """Collect predictions, labels, and per-sample dims for plotting."""
    model.eval()
    all_preds, all_labels, all_dims = [], [], []
    for batch in loader:
        pred = _forward(model, batch, model_name, device)
        all_preds.append(pred.cpu().numpy())
        all_labels.append(batch["label"].numpy())
        all_dims.append(batch["mask"].sum(dim=1).numpy().astype(int))

    return (np.concatenate(all_preds),
            np.concatenate(all_labels),
            np.concatenate(all_dims))


def compute_metrics(preds, labels):
    mse = float(np.mean((preds - labels) ** 2))
    mae = float(np.mean(np.abs(preds - labels)))
    ss_res = np.sum((labels - preds) ** 2)
    ss_tot = np.sum((labels - labels.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    rho = float(spearmanr(preds, labels).statistic)
    return {"mse": mse, "mae": mae, "r2": r2, "spearman": rho}


def _run_phase(model, model_name, train_loader, val_loader, optimizer,
               criterion, n_epochs, device, history, desc, track_best=True,
               select_metric="balanced", epoch_scatter_dir=None,
               train_eval_loader=None, scatter_stride=10):
    """Run a training phase. Returns (best_metrics, best_state).

    select_metric: "balanced" picks best by mean of per-dim MSE (so a late
    4D grok isn't drowned out by the dominant 2D data); "overall" uses
    aggregate val MSE.
    """
    n_train = len(train_loader.dataset)
    best_select = float("inf")
    best_metrics, best_state = {}, None

    pbar = tqdm(range(1, n_epochs + 1), desc=desc, unit="ep")
    for epoch in pbar:
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            pred = _forward(model, batch, model_name, device)
            loss = criterion(pred, batch["label"].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch["label"])
        train_loss /= n_train

        preds, labels, dims = collect_predictions(
            model, val_loader, model_name, device)
        val_metrics = compute_metrics(preds, labels)

        # Balanced selection score: mean of per-dimension MSE
        per_dim_mse = [float(np.mean((preds[dims == d] - labels[dims == d]) ** 2))
                       for d in sorted(set(dims))]
        balanced_mse = float(np.mean(per_dim_mse))
        select_val = balanced_mse if select_metric == "balanced" else val_metrics["mse"]

        history["train_mse"].append(train_loss)
        history["val_mse"].append(val_metrics["mse"])
        history["val_mae"].append(val_metrics["mae"])
        history["val_r2"].append(val_metrics["r2"])

        if track_best and select_val < best_select:
            best_select = select_val
            best_metrics = val_metrics.copy()
            best_metrics["epoch"] = epoch
            best_metrics["balanced_mse"] = balanced_mse
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}

        pbar.set_postfix({
            "loss": f"{train_loss:.5f}",
            "val_MSE": f"{val_metrics['mse']:.5f}",
            "bal_MSE": f"{balanced_mse:.5f}",
            "R²": f"{val_metrics['r2']:.3f}",
            "best": f"{best_select:.5f}" if track_best else "—",
        })

        # ── Epoch scatter snapshots (every scatter_stride epochs + last) ──
        if epoch_scatter_dir is not None and (
                epoch % scatter_stride == 0 or epoch == n_epochs):
            _epoch_scatter(preds, labels, dims, f"val epoch {epoch}",
                           epoch_scatter_dir / f"epoch_{epoch:03d}_val.png")
            if train_eval_loader is not None:
                tp, tl, td = collect_predictions(
                    model, train_eval_loader, model_name, device)
                for d in sorted(set(td)):
                    mm = td == d
                    _epoch_scatter(
                        tp[mm], tl[mm], td[mm],
                        f"train {d}D epoch {epoch}",
                        epoch_scatter_dir / f"epoch_{epoch:03d}_train{d}d.png")

    return best_metrics, best_state


def train_one_model(model, model_name, train_loader, val_loader,
                    args, device, pretrain_loader=None,
                    epoch_scatter_dir=None, train_eval_loader=None,
                    scatter_stride=10):
    """Train a single model, save best state dict, return best val metrics.

    If pretrain_loader is given, first pretrain on it (synth), then
    fine-tune on train_loader (real) with a fresh optimizer.
    Otherwise trains jointly on train_loader (default).
    """
    criterion = nn.HuberLoss(delta=0.05)
    history = {"train_mse": [], "val_mse": [], "val_mae": [], "val_r2": []}

    # ── Phase 1: pretrain on synthetic (no best tracking) ──
    if pretrain_loader is not None:
        pre_opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay)
        _run_phase(model, model_name, pretrain_loader, val_loader, pre_opt,
                   criterion, args.pretrain_epochs, device, history,
                   desc=f"  {model_name} [pretrain]", track_best=False)

    # ── Phase 2: fine-tune on real (fresh optimizer, track best) ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    best_metrics, best_state = _run_phase(
        model, model_name, train_loader, val_loader, optimizer,
        criterion, args.epochs, device, history,
        desc=f"  {model_name}", track_best=True,
        epoch_scatter_dir=epoch_scatter_dir,
        train_eval_loader=train_eval_loader,
        scatter_stride=scatter_stride)

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    return best_metrics, history


# ═══════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════

DIM_MARKERS = {2: "o", 3: "s", 4: "D"}
DIM_LABELS = {2: "2D", 3: "3D", 4: "4D"}
DIM_COLORS = {2: "#4878CF", 3: "#6ACC65", 4: "#D65F5F"}


def _epoch_scatter(preds, labels, dims, title, path):
    """Lightweight pred-vs-true scatter for a single epoch snapshot."""
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    mse = float(np.mean((preds - labels) ** 2)) if len(preds) else float("nan")
    for d in sorted(set(dims)):
        m = dims == d
        ax.scatter(labels[m], preds[m], s=4, alpha=0.15,
                   marker=DIM_MARKERS.get(d, "o"),
                   color=DIM_COLORS.get(d, "#333"),
                   label=DIM_LABELS.get(d, f"{d}D"), rasterized=True)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("True I(θ)")
    ax.set_ylabel("Predicted I(θ)")
    ax.set_title(f"{title}  (MSE={mse:.4f})", fontsize=9)
    ax.set_aspect("equal")
    if len(set(dims)) > 1:
        ax.legend(fontsize=7, markerscale=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_scatter_single(preds, labels, dims, model_name, metrics, ax):
    """Pred vs true scatter for one model on a single Axes."""
    for d in sorted(set(dims)):
        mask = dims == d
        ax.scatter(labels[mask], preds[mask],
                   s=4, alpha=0.15, marker=DIM_MARKERS.get(d, "o"),
                   color=DIM_COLORS.get(d, "#333333"),
                   label=DIM_LABELS.get(d, f"{d}D"),
                   rasterized=True)

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("True I(θ)")
    ax.set_ylabel("Predicted I(θ)")
    ax.set_title(f"{model_name}\n"
                 f"MSE={metrics['mse']:.4f}  "
                 f"R²={metrics['r2']:.3f}  "
                 f"ρ={metrics['spearman']:.3f}",
                 fontsize=9)
    ax.set_aspect("equal")
    ax.legend(fontsize=7, markerscale=3, loc="upper left")


def plot_error_by_dim(all_predictions, ax):
    """Bar chart of MAE per dimension for each model."""
    model_names = list(all_predictions.keys())
    unique_dims = sorted({d for _, _, dims in all_predictions.values()
                          for d in dims})
    n_models = len(model_names)
    n_dims = len(unique_dims)
    width = 0.8 / n_models
    x = np.arange(n_dims)

    for i, name in enumerate(model_names):
        preds, labels, dims = all_predictions[name]
        maes = []
        for d in unique_dims:
            mask = dims == d
            if mask.any():
                maes.append(float(np.mean(np.abs(preds[mask] - labels[mask]))))
            else:
                maes.append(0.0)
        ax.bar(x + i * width, maes, width, label=name,
               color=COLORS.get(name, "#333333"), alpha=0.85)

    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels([DIM_LABELS.get(d, f"{d}D") for d in unique_dims])
    ax.set_ylabel("MAE")
    ax.set_title("MAE by Dimension")
    ax.legend(fontsize=7)


def plot_comparison_bar(results, ax):
    """Grouped bar chart of key metrics across models."""
    names = list(results.keys())
    metrics_keys = ["mse", "mae"]
    x = np.arange(len(names))
    width = 0.35

    for i, mk in enumerate(metrics_keys):
        vals = [results[n][mk] for n in names]
        bars = ax.bar(x + i * width, vals, width, label=mk.upper(),
                      alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.4f}", ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Error")
    ax.set_title("MSE / MAE Comparison")
    ax.legend(fontsize=7)


def plot_learning_curves(all_histories, output_dir):
    """Train/val MSE and val R² curves for all models on shared axes."""
    model_names = list(all_histories.keys())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for name in model_names:
        h = all_histories[name]
        epochs = np.arange(1, len(h["train_mse"]) + 1)
        c = COLORS.get(name, "#333333")
        axes[0].plot(epochs, h["train_mse"], color=c, ls="--", alpha=0.5)
        axes[0].plot(epochs, h["val_mse"], color=c, label=name)
        axes[1].plot(epochs, h["val_mae"], color=c, label=name)
        axes[2].plot(epochs, h["val_r2"], color=c, label=name)

    axes[0].set_ylabel("MSE")
    axes[0].set_title("Train (dashed) / Val (solid) MSE")
    axes[1].set_ylabel("MAE")
    axes[1].set_title("Val MAE")
    axes[2].set_ylabel("R²")
    axes[2].set_title("Val R²")

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Learning Curves", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_dir / "learning_curves.png", dpi=150)
    fig.savefig(output_dir / "learning_curves.pdf")
    plt.close(fig)


def generate_plots(all_predictions, results, all_histories, output_dir):
    """Generate all comparison figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_names = list(all_predictions.keys())
    n = len(model_names)

    # ── 1. Individual scatter plots (one per model) ──
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows),
                             squeeze=False)
    for i, name in enumerate(model_names):
        r, c = divmod(i, cols)
        preds, labels, dims = all_predictions[name]
        plot_scatter_single(preds, labels, dims, name, results[name],
                            axes[r][c])
    for i in range(n, rows * cols):
        r, c = divmod(i, cols)
        axes[r][c].set_visible(False)
    fig.suptitle("Predicted vs True I(θ) — Validation Set", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_dir / "scatter_all.png", dpi=150)
    fig.savefig(output_dir / "scatter_all.pdf")
    plt.close(fig)

    # ── 2. Summary figure: bar charts + error by dim ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    plot_comparison_bar(results, ax1)
    plot_error_by_dim(all_predictions, ax2)
    fig.suptitle("Model Comparison — Validation Set", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_dir / "comparison_summary.png", dpi=150)
    fig.savefig(output_dir / "comparison_summary.pdf")
    plt.close(fig)

    # ── 3. Learning curves ──
    plot_learning_curves(all_histories, output_dir)

    print(f"  Plots saved to {output_dir}/")


# ═══════════════════════════════════════════════════════════════════════
# Leave-one-scenario-out cross-validation (for few-scenario dims)
# ═══════════════════════════════════════════════════════════════════════

def run_loso(args, model_names, device):
    """Leave-one-scenario-out CV over the scenarios of the chosen dim(s).

    For each held-out scenario: train on all the others (of that dim),
    evaluate on the held-out one. Aggregates predictions across all folds
    so every scenario is tested exactly once. Robust for 3D/4D where a
    single random holdout is unreliable (some scenarios are near-constant).
    """
    dims = args.dims or [2, 3, 4]

    all_files = []
    for d in DATA_DIRS:
        all_files.extend(sorted(Path(d).glob("results_scenario_*.json")))
    by_dim = defaultdict(list)
    for jf in all_files:
        by_dim[_get_scenario_dim(jf)].append(jf)

    # Synthetic scenarios grouped by dim (added to every fold's training set)
    synth_by_dim = defaultdict(list)
    if args.synthetic_dir:
        synth_dir = Path(args.synthetic_dir)
        if not synth_dir.is_absolute():
            synth_dir = ROOT / synth_dir
        caps = {2: args.n_synth_2d, 3: args.n_synth_3d, 4: args.n_synth_4d}
        synth_rng = np.random.default_rng(123)
        for f in sorted(synth_dir.glob("results_scenario_*.json")):
            synth_by_dim[_get_scenario_dim(f)].append(f)
        for d in list(synth_by_dim):
            cap = caps.get(d)
            if cap is not None and cap < len(synth_by_dim[d]):
                synth_by_dim[d] = list(synth_rng.choice(
                    synth_by_dim[d], size=cap, replace=False))

    run_id = args.run_id or (datetime.now().strftime("%Y%m%d_%H%M%S") + "_loso")
    out_dir = ROOT / "results" / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"LOSO-CV run id: {run_id}\nOutput: {out_dir}\n")

    loso_results = {}

    for dim in dims:
        scenarios = sorted(by_dim.get(dim, []))
        if len(scenarios) < 2:
            print(f"dim={dim}: only {len(scenarios)} scenario(s), skipping LOSO")
            continue
        synth_dim_files = synth_by_dim.get(dim, [])
        if synth_dim_files:
            print(f"dim={dim}: adding {len(synth_dim_files)} synth scenarios "
                  f"to each fold's training set")

        for name in model_names:
            print(f"\n{'='*64}")
            print(f"  LOSO-CV  dim={dim}  model={name}  ({len(scenarios)} folds)")
            print(f"{'='*64}")

            fold_preds, fold_labels = [], []
            fold_rows = []

            for held_out in scenarios:
                # Real scenarios of this dim minus the held-out one, plus
                # synthetic scenarios of this dim (synth never used for val).
                train_files = [f for f in scenarios if f != held_out]
                train_files = train_files + synth_dim_files
                train_ds = ZonotopeDatasetV2(files=train_files,
                                             label_key="I_theta")
                val_ds = ZonotopeDatasetV2(files=[held_out],
                                           label_key="I_theta")
                train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                          shuffle=True, collate_fn=collate_fn)
                val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                        shuffle=False, collate_fn=collate_fn)

                if name == "original":
                    model = DeepSetsZonotope(phi_hidden=16, rho_hidden=32)
                else:
                    model = create_model(name)
                model = model.to(device)

                history = {"train_mse": [], "val_mse": [],
                           "val_mae": [], "val_r2": []}
                optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                              weight_decay=args.weight_decay)
                criterion = nn.HuberLoss(delta=0.05)
                _, best_state = _run_phase(
                    model, name, train_loader, val_loader, optimizer,
                    criterion, args.epochs, device, history,
                    desc=f"  {held_out.name}", track_best=True,
                    select_metric="overall")
                if best_state is not None:
                    model.load_state_dict(best_state)

                preds, labels, _ = collect_predictions(
                    model, val_loader, name, device)
                m = compute_metrics(preds, labels)
                fold_preds.append(preds)
                fold_labels.append(labels)
                # R²/ρ are meaningless for near-constant scenarios; flag it
                near_const = labels.std() < 0.02
                fold_rows.append((held_out.name, m, float(labels.std()),
                                  bool(near_const)))
                print(f"    {held_out.name}: MSE={m['mse']:.5f} "
                      f"MAE={m['mae']:.4f} "
                      + (f"R²={m['r2']:.3f} ρ={m['spearman']:.3f}"
                         if not near_const
                         else f"(near-constant I, std={labels.std():.4f})"))

            # ── Aggregate across folds ──
            all_p = np.concatenate(fold_preds)
            all_l = np.concatenate(fold_labels)
            agg = compute_metrics(all_p, all_l)
            # Mean per-fold MSE/MAE (equal weight per scenario)
            mean_mse = float(np.mean([r[1]["mse"] for r in fold_rows]))
            mean_mae = float(np.mean([r[1]["mae"] for r in fold_rows]))
            print(f"\n  dim={dim} {name} aggregate over {len(scenarios)} folds:")
            print(f"    pooled:   MSE={agg['mse']:.5f}  MAE={agg['mae']:.4f}  "
                  f"R²={agg['r2']:.4f}  ρ={agg['spearman']:.4f}")
            print(f"    per-fold: mean MSE={mean_mse:.5f}  mean MAE={mean_mae:.4f}")

            loso_results[f"dim{dim}_{name}"] = {
                "dim": dim, "model": name, "n_folds": len(scenarios),
                "n_synth_scenarios": len(synth_dim_files),
                "pooled": agg, "mean_fold_mse": mean_mse,
                "mean_fold_mae": mean_mae,
                "folds": [{"scenario": r[0], "metrics": r[1],
                           "label_std": r[2], "near_constant": r[3]}
                          for r in fold_rows],
            }

            # Scatter of pooled held-out predictions
            fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
            ax.scatter(all_l, all_p, s=4, alpha=0.15,
                       color=COLORS.get(name, "#333"), rasterized=True)
            ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
            ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
            ax.set_xlabel("True I(θ)"); ax.set_ylabel("Predicted I(θ)")
            ax.set_title(f"LOSO {dim}D — {name}\n"
                         f"pooled MSE={agg['mse']:.4f} ρ={agg['spearman']:.3f}",
                         fontsize=9)
            ax.set_aspect("equal")
            fig.tight_layout()
            fig.savefig(out_dir / f"loso_dim{dim}_{name}.png", dpi=150)
            plt.close(fig)

    with open(out_dir / "loso_results.json", "w") as f:
        json.dump(loso_results, f, indent=2)
    print(f"\nLOSO results saved to {out_dir}/loso_results.json")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="AdamW weight decay (0 = plain Adam behavior)")
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to train. Default: all. "
                             "Choices: original, deepsets_v2, flat_mlp, "
                             "siamese, set_transformer")
    parser.add_argument("--synthetic_dir", type=str, default=None,
                        help="Directory with synthetic data. "
                             "Added to training only (not validation).")
    parser.add_argument("--synthetic_3d4d_only", action="store_true",
                        help="Only include 3D/4D synthetic scenarios "
                             "(skip 2D to preserve real 2D fit).")
    parser.add_argument("--dims", type=int, nargs="+", default=None,
                        help="Only use these dimensions for train/val "
                             "(e.g., --dims 4 for 4D-only)")
    parser.add_argument("--balance_dims", action="store_true",
                        help="Downsample training data so each dimension "
                             "has equal sample count (cap to min)")
    parser.add_argument("--pretrain_synth", action="store_true",
                        help="Two-phase: pretrain on synth, fine-tune on real. "
                             "Default is joint training (synth mixed into real).")
    parser.add_argument("--pretrain_epochs", type=int, default=20,
                        help="Epochs for the synth pretrain phase")
    parser.add_argument("--n_synth_2d", type=int, default=None,
                        help="Cap number of 2D synth scenarios (default: all)")
    parser.add_argument("--n_synth_3d", type=int, default=None,
                        help="Cap number of 3D synth scenarios (default: all)")
    parser.add_argument("--n_synth_4d", type=int, default=None,
                        help="Cap number of 4D synth scenarios (default: all)")
    parser.add_argument("--run_id", type=str, default=None,
                        help="Unique run id for output dir "
                             "(default: timestamp). Saved under results/runs/")
    parser.add_argument("--loso", action="store_true",
                        help="Leave-one-scenario-out CV over the scenarios of "
                             "the chosen --dims. Robust eval for few scenarios.")
    parser.add_argument("--n_val_synth", type=int, default=0,
                        help="Hold out this many synthetic scenarios (per dim) "
                             "as a separate validation set; train on the rest. "
                             "Reported separately from the real held-out set.")
    parser.add_argument("--holdout", nargs="+", default=None,
                        help="Explicit real scenario file(s) to hold out for "
                             "validation, e.g. --holdout results_scenario_7.json. "
                             "Overrides the random split. Use to avoid "
                             "degenerate (constant I) holdouts.")
    parser.add_argument("--save_epoch_scatters", action="store_true",
                        help="Save train (per-dim) and val scatter plots into "
                             "<run_dir>/epoch_scatters/ (see --scatter_stride).")
    parser.add_argument("--scatter_stride", type=int, default=10,
                        help="Save epoch scatters every N epochs (and the last "
                             "epoch). Default 10.")
    args = parser.parse_args()

    all_model_names = ["original", "deepsets_v2", "flat_mlp",
                       "siamese", "set_transformer"]
    model_names = args.models or all_model_names

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    if args.loso:
        run_loso(args, model_names, device)
        return

    # ── Inductive split (stratified by dimension) ──
    all_files = []
    for d in DATA_DIRS:
        all_files.extend(sorted(Path(d).glob("results_scenario_*.json")))

    by_dim = defaultdict(list)
    for jf in all_files:
        by_dim[_get_scenario_dim(jf)].append(jf)

    rng = np.random.default_rng(42)
    holdout_set = set(args.holdout) if args.holdout else None
    train_files, val_files = [], []
    for dim, files in sorted(by_dim.items()):
        if args.dims and dim not in args.dims:
            continue
        files = list(rng.permutation(files))
        if holdout_set is not None:
            def _is_holdout(f):
                return (f.name in holdout_set
                        or any(str(f).endswith(h) for h in holdout_set))
            val = [f for f in files if _is_holdout(f)]
            train = [f for f in files if not _is_holdout(f)]
            if val:  # only apply to dims that contain a named holdout
                val_files.extend(val)
                train_files.extend(train)
                print(f"  dim={dim}: {len(train)} train, {len(val)} val "
                      f"(explicit holdout)")
                continue
        n_val = max(1, int(args.val_fraction * len(files)))
        val_files.extend(files[:n_val])
        train_files.extend(files[n_val:])
        print(f"  dim={dim}: {len(files)-n_val} train, {n_val} val")

    print(f"Inductive split: {len(train_files)} train scenarios, "
          f"{len(val_files)} val scenarios")
    print(f"  Held-out REAL: {[f.name for f in val_files]}\n")

    # ── Synthetic data ──
    # synth_files → training; synth_val_files → separate held-out eval.
    synth_files = []
    synth_val_files = []
    if args.synthetic_dir:
        synth_dir = Path(args.synthetic_dir)
        if not synth_dir.is_absolute():
            synth_dir = ROOT / synth_dir
        all_synth = sorted(synth_dir.glob("results_scenario_*.json"))

        # Group by dimension
        synth_by_dim = defaultdict(list)
        for f in all_synth:
            synth_by_dim[_get_scenario_dim(f)].append(f)

        caps = {2: args.n_synth_2d, 3: args.n_synth_3d, 4: args.n_synth_4d}
        synth_rng = np.random.default_rng(123)
        for dim in sorted(synth_by_dim):
            if args.synthetic_3d4d_only and dim < 3:
                continue
            if args.dims and dim not in args.dims:
                continue  # match synth dims to the real dims being trained
            files = synth_by_dim[dim]
            cap = caps.get(dim)
            if cap is not None and cap < len(files):
                files = list(synth_rng.choice(files, size=cap, replace=False))
            files = list(synth_rng.permutation(files))
            # Hold out n_val_synth scenarios (per dim) for separate evaluation
            n_hold = min(args.n_val_synth, max(0, len(files) - 1))
            synth_val_files.extend(files[:n_hold])
            synth_files.extend(files[n_hold:])
            print(f"  Synthetic dim={dim}: {len(files)-n_hold} train, "
                  f"{n_hold} held-out for eval"
                  + (f" (capped from {len(synth_by_dim[dim])})"
                     if cap is not None and cap < len(synth_by_dim[dim]) else ""))
        if synth_val_files:
            print(f"  Held-out SYNTH: {[f.name for f in synth_val_files]}")

    # ── Load data ──
    # Pretrain mode: synth → pretrain loader, real → fine-tune loader.
    # Joint mode (default): synth mixed into the real training set.
    print("Loading data...")
    t0 = time.time()
    if args.pretrain_synth and synth_files:
        train_ds = ZonotopeDatasetV2(files=train_files, label_key="I_theta",
                                     max_samples=args.max_samples)
        pretrain_ds = ZonotopeDatasetV2(files=synth_files, label_key="I_theta")
    else:
        train_ds = ZonotopeDatasetV2(files=train_files + synth_files,
                                     label_key="I_theta",
                                     max_samples=args.max_samples)
        pretrain_ds = None
    val_ds = ZonotopeDatasetV2(files=val_files, label_key="I_theta")
    synth_val_ds = (ZonotopeDatasetV2(files=synth_val_files, label_key="I_theta")
                    if synth_val_files else None)
    dt = time.time() - t0
    print(f"  {len(train_ds):,} train + {len(val_ds):,} val samples "
          f"loaded in {dt:.1f}s")
    if synth_val_ds is not None:
        print(f"  {len(synth_val_ds):,} held-out synth eval samples")
    if pretrain_ds is not None:
        print(f"  {len(pretrain_ds):,} synth pretrain samples (separate phase)")
    elif synth_files:
        print(f"  (synth data mixed into training)")

    if args.balance_dims:
        by_dim_samples = defaultdict(list)
        for i, s in enumerate(train_ds.samples):
            by_dim_samples[s["dim"]].append(i)
        min_count = min(len(v) for v in by_dim_samples.values())
        bal_rng = np.random.default_rng(42)
        keep = []
        for dim in sorted(by_dim_samples):
            indices = by_dim_samples[dim]
            if len(indices) > min_count:
                indices = bal_rng.choice(indices, size=min_count,
                                         replace=False).tolist()
            keep.extend(indices)
        keep.sort()
        train_ds.samples = [train_ds.samples[i] for i in keep]
        print(f"  Balanced dims: {min_count:,} samples per dim, "
              f"{len(train_ds):,} total")

    print()

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn,
                              num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_fn,
                            num_workers=0)
    pretrain_loader = None
    if pretrain_ds is not None:
        pretrain_loader = DataLoader(pretrain_ds, batch_size=args.batch_size,
                                     shuffle=True, collate_fn=collate_fn,
                                     num_workers=0)
    synth_val_loader = None
    if synth_val_ds is not None:
        synth_val_loader = DataLoader(synth_val_ds, batch_size=args.batch_size,
                                      shuffle=False, collate_fn=collate_fn,
                                      num_workers=0)

    # Subsampled training set used only for per-epoch scatter snapshots
    train_eval_loader = None
    if args.save_epoch_scatters:
        n_eval = min(len(train_ds), 40000)
        idx = np.random.default_rng(0).choice(
            len(train_ds), size=n_eval, replace=False).tolist()
        train_eval_loader = DataLoader(
            Subset(train_ds, idx), batch_size=args.batch_size,
            shuffle=False, collate_fn=collate_fn, num_workers=0)

    # ── Train each model ──
    suffix = "_synth" if synth_files else ""
    run_id = args.run_id or (datetime.now().strftime("%Y%m%d_%H%M%S")
                             + suffix)
    out_dir = ROOT / "results" / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run id: {run_id}")
    print(f"Output: {out_dir}\n")

    # Save the run configuration for reproducibility
    run_config = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "models": model_names,
        "device": str(device),
        "n_train_samples": len(train_ds),
        "n_val_samples": len(val_ds),
        "val_real_scenarios": [f.name for f in val_files],
        "val_synth_scenarios": [f.name for f in synth_val_files],
        "n_synth_train_scenarios": len(synth_files),
        "n_dim_feat": N_DIM_FEAT_V2,
        "n_global": N_GLOBAL_V2,
    }
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    results = {}
    all_predictions = {}
    all_histories = {}

    for name in model_names:
        print(f"{'='*60}")
        print(f"  Training: {name}")
        print(f"{'='*60}")

        if name == "original":
            model = DeepSetsZonotope(phi_hidden=16, rho_hidden=32)
        else:
            model = create_model(name)
        model = model.to(device)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}  |  Device: {device}\n")

        epoch_scatter_dir = None
        if args.save_epoch_scatters:
            epoch_scatter_dir = out_dir / "epoch_scatters" / name
            epoch_scatter_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        metrics, history = train_one_model(
            model, name, train_loader, val_loader, args, device,
            pretrain_loader=pretrain_loader,
            epoch_scatter_dir=epoch_scatter_dir,
            train_eval_loader=train_eval_loader,
            scatter_stride=args.scatter_stride)
        elapsed = time.time() - t0

        metrics["n_params"] = n_params
        metrics["wall_time_s"] = elapsed
        results[name] = metrics
        all_histories[name] = history

        preds, labels, dims = collect_predictions(
            model, val_loader, name, device)
        all_predictions[name] = (preds, labels, dims)

        print(f"\n  Best: epoch {metrics['epoch']}, "
              f"MSE={metrics['mse']:.5f}, "
              f"R²={metrics['r2']:.4f}, "
              f"ρ={metrics['spearman']:.4f}")
        print(f"  Per-dimension (real val):")
        for d in sorted(set(dims)):
            m = dims == d
            dm = compute_metrics(preds[m], labels[m])
            print(f"    dim={d}: MSE={dm['mse']:.5f}  MAE={dm['mae']:.4f}  "
                  f"R²={dm['r2']:.4f}  ρ={dm['spearman']:.4f}  N={int(m.sum()):,}")

        # ── Evaluate on held-out synthetic scenarios (separate) ──
        synth_metrics = None
        if synth_val_loader is not None:
            sp, sl, _ = collect_predictions(
                model, synth_val_loader, name, device)
            synth_metrics = compute_metrics(sp, sl)
            metrics["synth_val"] = synth_metrics
            print(f"  Held-out SYNTH val: MSE={synth_metrics['mse']:.5f}  "
                  f"MAE={synth_metrics['mae']:.4f}  R²={synth_metrics['r2']:.4f}  "
                  f"ρ={synth_metrics['spearman']:.4f}  N={len(sl):,}")
        print(f"  Time: {elapsed:.1f}s")

        # ── Save model checkpoint (loadable by a colleague) ──
        checkpoint = {
            "model_name": name,
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "n_dim_feat": N_DIM_FEAT_V2,
            "n_global": N_GLOBAL_V2,
            "n_params": n_params,
            "val_metrics": metrics,
            "val_real_scenarios": [f.name for f in val_files],
            "val_synth_scenarios": [f.name for f in synth_val_files],
            "run_id": run_id,
            "feature_order": {
                "per_dim": ["delta_c", "r1_norm", "r2_norm", "cos", "width_ratio"],
                "global": ["log_vol_ratio", "norm_center_dist", "dim",
                           "sep_ratio", "off_over_tgt"],
            },
        }
        torch.save(checkpoint, out_dir / f"{name}.pt")

        # ── Save incrementally after each model ──
        # Per-model scatter
        fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
        plot_scatter_single(preds, labels, dims, name, metrics, ax)
        fig.tight_layout()
        fig.savefig(out_dir / f"scatter_{name}.png", dpi=150)
        plt.close(fig)

        # Update learning curves (all models trained so far)
        plot_learning_curves(all_histories, out_dir)

        # Update results JSON
        with open(out_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"  Saved to {out_dir}/\n")

    # ── Final combined plots ──
    generate_plots(all_predictions, results, all_histories, out_dir)

    # ── Comparison table ──
    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}")
    print(f"{'Model':<20s} {'Params':>8s} {'MSE':>10s} {'MAE':>8s} "
          f"{'R²':>8s} {'ρ':>8s} {'Best ep':>8s} {'Time':>8s}")
    print("-" * 80)

    ranked = sorted(results.items(), key=lambda kv: kv[1]["mse"])
    for name, m in ranked:
        print(f"{name:<20s} {m['n_params']:>8,} {m['mse']:>10.5f} "
              f"{m['mae']:>8.4f} {m['r2']:>8.4f} {m['spearman']:>8.4f} "
              f"{m['epoch']:>8d} {m['wall_time_s']:>7.1f}s")

    print(f"{'='*80}")
    print(f"\nSettings: {args.epochs} epochs, lr={args.lr}, "
          f"batch_size={args.batch_size}, device={device}")
    print(f"All results in {out_dir}/")


if __name__ == "__main__":
    main()
