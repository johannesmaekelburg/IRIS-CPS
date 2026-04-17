"""Train the GINE-based inconsistency surrogate.

Supports optional self-supervised pretraining (geometry tasks) followed by
supervised fine-tuning on MC-computed I(theta) targets.

Usage:
    python train.py --config configs/surrogate_default.yaml
    python train.py --config my_config.yaml --pretrain_only
    python train.py --config my_config.yaml --finetune_only
    python train.py --config my_config.yaml --checkpoint results/surrogate/pretrained.pt
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as TF
from torch_geometric.loader import DataLoader

from .config import SurrogateConfig, load_config, resolve_device
from .dataset import (
    AffineMapDataset,
    ContainmentDataset,
    InconsistencyDataset,
    PairwiseAABBDataset,
    VolumeDataset,
)
from .model import ZonotopeGINE
from .plots import (
    MetricsLogger,
    plot_finetune_curves,
    plot_predictions,
    plot_predictions_combined,
    plot_pretrain_curves,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


class EarlyStopping:
    """Stop training when metric has not improved for *patience* epochs."""

    def __init__(self, patience: int, mode: str = "min"):
        self.patience = patience
        self.mode = mode
        self.best: Optional[float] = None
        self.counter = 0
        self.best_epoch = 0

    def step(self, score: float, epoch: int) -> bool:
        if self.patience <= 0:
            return False
        improved = (
            self.best is None
            or (self.mode == "min" and score < self.best)
            or (self.mode == "max" and score > self.best)
        )
        if improved:
            self.best = score
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
        return self.counter >= self.patience


def build_scheduler(optimizer, scheduler_name: str, epochs: int):
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3))
    else:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)


# ---------------------------------------------------------------------------
# Pretraining
# ---------------------------------------------------------------------------

def pretrain_epoch(
    model: ZonotopeGINE,
    loaders: Dict[str, DataLoader],
    optimizer: torch.optim.Optimizer,
    loss_weights: Dict[str, float],
    device: str,
) -> Dict[str, float]:
    """One epoch of multi-task pretraining (round-robin over tasks)."""
    model.train()
    task_losses = {t: 0.0 for t in loaders}
    task_counts = {t: 0 for t in loaders}

    iterators = {t: iter(l) for t, l in loaders.items()}
    active = set(loaders.keys())

    while active:
        total_loss = torch.tensor(0.0, device=device)
        for task in list(active):
            try:
                batch = next(iterators[task])
            except StopIteration:
                active.discard(task)
                continue

            batch = batch.to(device)

            if task == "volume":
                pred = model.forward_volume(batch)
                loss = TF.mse_loss(pred.squeeze(-1), batch.y.squeeze(-1))
            elif task == "containment":
                pred = model.forward_containment(batch)
                loss = TF.binary_cross_entropy(pred, batch.labels)
            elif task == "pairwise_aabb":
                pred = model.forward_pairwise_aabb(batch)
                loss = TF.mse_loss(pred.squeeze(-1), batch.y.squeeze(-1))
            elif task == "affine_map":
                pred_c, pred_g = model.forward_affine_map(batch)
                loss_c = TF.mse_loss(pred_c, batch.y_center)
                mask = batch.y_generator_mask.unsqueeze(-1).float()
                loss_g = TF.mse_loss(pred_g * mask, batch.y_generators * mask)
                loss = loss_c + loss_g
            else:
                continue

            total_loss = total_loss + loss * loss_weights.get(task, 1.0)
            task_losses[task] += loss.item()
            task_counts[task] += 1

        if total_loss.requires_grad:
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    return {t: task_losses[t] / max(task_counts[t], 1) for t in task_losses}


@torch.no_grad()
def pretrain_validate(
    model: ZonotopeGINE,
    loaders: Dict[str, DataLoader],
    device: str,
) -> Dict[str, float]:
    model.eval()
    metrics: Dict[str, float] = {}
    for task, loader in loaders.items():
        total_loss = 0.0
        n = 0
        for batch in loader:
            batch = batch.to(device)
            if task == "volume":
                pred = model.forward_volume(batch)
                loss = TF.mse_loss(pred.squeeze(-1), batch.y.squeeze(-1))
            elif task == "containment":
                pred = model.forward_containment(batch)
                loss = TF.binary_cross_entropy(pred, batch.labels)
            elif task == "pairwise_aabb":
                pred = model.forward_pairwise_aabb(batch)
                loss = TF.mse_loss(pred.squeeze(-1), batch.y.squeeze(-1))
            elif task == "affine_map":
                pred_c, pred_g = model.forward_affine_map(batch)
                loss_c = TF.mse_loss(pred_c, batch.y_center)
                mask = batch.y_generator_mask.unsqueeze(-1).float()
                loss_g = TF.mse_loss(pred_g * mask, batch.y_generators * mask)
                loss = loss_c + loss_g
            else:
                continue
            total_loss += loss.item()
            n += 1
        metrics[task] = total_loss / max(n, 1)
    return metrics


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------

def train_epoch(
    model: ZonotopeGINE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: str,
    auxiliary_aabb_weight: float = 0.0,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        batch = batch.to(device)
        pred = model.forward_inconsistency(batch)
        loss = loss_fn(pred.squeeze(-1), batch.y.squeeze(-1))

        if auxiliary_aabb_weight > 0:
            loss_aabb = TF.mse_loss(pred.squeeze(-1), batch.y_aabb.squeeze(-1))
            loss = loss + auxiliary_aabb_weight * loss_aabb

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n += 1
    return {"loss": total_loss / max(n, 1)}


@torch.no_grad()
def evaluate(
    model: ZonotopeGINE,
    loader: DataLoader,
    device: str,
) -> Dict[str, float]:
    """Evaluate: MSE, MAE, R-squared, Spearman correlation."""
    model.eval()
    all_preds, all_targets = [], []
    for batch in loader:
        batch = batch.to(device)
        pred = model.forward_inconsistency(batch)
        all_preds.append(pred.squeeze(-1).cpu())
        all_targets.append(batch.y.squeeze(-1).cpu())

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    mse = TF.mse_loss(preds, targets).item()
    mae = TF.l1_loss(preds, targets).item()
    ss_res = ((targets - preds) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    r2 = (1.0 - ss_res / ss_tot.clamp(min=1e-8)).item()

    from scipy.stats import spearmanr
    rho, _ = spearmanr(preds.numpy(), targets.numpy())

    return {"mse": mse, "mae": mae, "r2": r2, "spearman_rho": float(rho)}


@torch.no_grad()
def collect_predictions(
    model: ZonotopeGINE,
    loader: DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect predictions, targets, and scenario IDs for plotting."""
    model.eval()
    all_preds, all_targets, all_scenarios = [], [], []
    for batch in loader:
        batch = batch.to(device)
        pred = model.forward_inconsistency(batch)
        all_preds.append(pred.squeeze(-1).cpu().numpy())
        all_targets.append(batch.y.squeeze(-1).cpu().numpy())
        if hasattr(batch, "scenario_idx"):
            all_scenarios.append(batch.scenario_idx.squeeze(-1).cpu().numpy())
        else:
            all_scenarios.append(np.zeros(len(pred), dtype=np.int64))
    return (
        np.concatenate(all_preds),
        np.concatenate(all_targets),
        np.concatenate(all_scenarios),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the GINE-based inconsistency surrogate.",
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config")
    parser.add_argument("--pretrain_only", action="store_true")
    parser.add_argument("--finetune_only", action="store_true",
                        help="Skip pretraining, load checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to pretrained checkpoint")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Optional run name appended to output_dir. "
                             "Defaults to a timestamp (YYYYMMDD_HHMMSS).")
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config.device)
    set_seed(config.seed)

    project_root = Path(__file__).resolve().parent.parent.parent
    data_root = project_root / config.data_root
    run_tag = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / config.output_dir / run_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Save config file to output dir ────────────────────────────────────────
    if args.config and Path(args.config).exists():
        shutil.copy(args.config, output_dir / "config_used.yaml")
    else:
        # No YAML provided — save resolved dataclass config as YAML
        import yaml
        with open(output_dir / "config_used.yaml", "w") as _f:
            yaml.dump(dataclasses.asdict(config), _f, default_flow_style=False)

    # ── Tee stdout/stderr to output_dir/train.log ─────────────────────────────
    class _Tee:
        def __init__(self, *streams):
            self._streams = streams
        def write(self, data):
            for s in self._streams:
                s.write(data)
        def flush(self):
            for s in self._streams:
                s.flush()

    _log_file = open(output_dir / "train.log", "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, _log_file)
    sys.stderr = _Tee(sys.__stderr__, _log_file)

    print(f"\n{'=' * 60}")
    print("GINE INCONSISTENCY SURROGATE — TRAINING")
    print(f"{'=' * 60}")
    print(f"Device      : {device}")
    print(f"Data root   : {data_root}")
    print(f"Output      : {output_dir}")
    print(f"Pretrain    : {config.pretrain.enabled and not args.finetune_only}")
    print(f"{'=' * 60}\n")

    model = ZonotopeGINE(config.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # ==================================================================
    # Pretraining
    # ==================================================================
    if config.pretrain.enabled and not args.finetune_only:
        print(f"\n--- Pretraining ({config.pretrain.tasks}) ---")
        model.init_pretrain_heads()
        model = model.to(device)

        pretrain_dir = data_root / "pretrain"

        # Build datasets
        loaders_train: Dict[str, DataLoader] = {}
        loaders_val: Dict[str, DataLoader] = {}
        task_cls = {
            "volume": (VolumeDataset, pretrain_dir / "volume.npz"),
            "containment": (ContainmentDataset, pretrain_dir / "containment.npz"),
            "pairwise_aabb": (PairwiseAABBDataset, pretrain_dir / "pairwise_aabb.npz"),
            "affine_map": (AffineMapDataset, pretrain_dir / "affine_map.npz"),
        }

        for task in config.pretrain.tasks:
            cls, path = task_cls[task]
            if not path.exists():
                print(f"  WARNING: {path} not found, skipping {task}")
                continue
            ds_t = cls(path, split="train", val_fraction=config.pretrain.val_fraction,
                       seed=config.seed)
            ds_v = cls(path, split="val", val_fraction=config.pretrain.val_fraction,
                       seed=config.seed)
            loaders_train[task] = DataLoader(
                ds_t, batch_size=config.pretrain.batch_size, shuffle=True,
                num_workers=config.num_workers, drop_last=False,
            )
            loaders_val[task] = DataLoader(
                ds_v, batch_size=config.pretrain.batch_size,
                num_workers=config.num_workers,
            )
            print(f"  {task}: {ds_t.len()} train, {ds_v.len()} val")

        if not loaders_train:
            print("  No pretraining data found — skipping pretraining.")
        else:
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=config.pretrain.lr,
                weight_decay=config.pretrain.weight_decay,
            )
            scheduler = build_scheduler(
                optimizer, config.pretrain.scheduler, config.pretrain.epochs,
            )
            stopper = EarlyStopping(config.pretrain.patience)
            pretrain_logger = MetricsLogger(output_dir / "pretrain_log.csv")

            for epoch in range(config.pretrain.epochs):
                t0 = time.perf_counter()
                train_metrics = pretrain_epoch(
                    model, loaders_train, optimizer,
                    config.pretrain.loss_weights, device,
                )
                val_metrics = pretrain_validate(model, loaders_val, device)
                scheduler.step()
                dt = time.perf_counter() - t0

                # Aggregate val loss
                agg_val = sum(val_metrics.values()) / max(len(val_metrics), 1)

                # Log to CSV
                log_row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
                for t, v in train_metrics.items():
                    log_row[f"train_{t}"] = v
                for t, v in val_metrics.items():
                    log_row[f"val_{t}"] = v
                log_row["val_agg"] = agg_val
                pretrain_logger.log(log_row)

                if epoch % 5 == 0 or epoch == config.pretrain.epochs - 1:
                    parts = [f"{t}={v:.4f}" for t, v in val_metrics.items()]
                    print(f"  Epoch {epoch:3d} | val: {', '.join(parts)} "
                          f"| agg={agg_val:.4f} | {dt:.1f}s")

                if stopper.step(agg_val, epoch):
                    print(f"  Early stopping at epoch {epoch}")
                    break

            pretrain_logger.close()
            plot_pretrain_curves(output_dir)

            # Save pretrained backbone
            ckpt_path = output_dir / "pretrained_backbone.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": dataclasses.asdict(config),
                "epoch": epoch,
            }, ckpt_path)
            print(f"  Saved pretrained backbone -> {ckpt_path}")

    if args.pretrain_only:
        print("\nDone (pretrain only).")
        return

    # ==================================================================
    # Load pretrained weights
    # ==================================================================
    ckpt_path = args.checkpoint or (output_dir / "pretrained_backbone.pt")
    if Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        # Load backbone weights, skip pretrain heads
        model_dict = model.state_dict()
        skip_prefixes = ("volume_head.", "containment_head.",
                         "pairwise_aabb_head.", "affine_map_head.")
        pretrained = {
            k: v for k, v in ckpt["model_state_dict"].items()
            if k in model_dict and not k.startswith(skip_prefixes)
        }
        model_dict.update(pretrained)
        model.load_state_dict(model_dict)
        print(f"Loaded backbone from {ckpt_path} "
              f"({len(pretrained)}/{len(model_dict)} keys)")

    # ==================================================================
    # Fine-tuning datasets
    # ==================================================================
    print(f"\n--- Fine-tuning ---")
    print(f"  Train scenarios: {config.train.scenarios_train}")
    print(f"  Test scenarios : {config.train.scenarios_test}")

    train_ds = InconsistencyDataset(
        data_root, config.train.scenarios_train, split="train",
        val_fraction=config.train.val_fraction, seed=config.seed,
    )
    val_ds = InconsistencyDataset(
        data_root, config.train.scenarios_train, split="val",
        val_fraction=config.train.val_fraction, seed=config.seed,
    )
    test_ds = InconsistencyDataset(
        data_root, config.train.scenarios_test, split="all",
        seed=config.seed,
    ) if config.train.scenarios_test else None

    print(f"  Samples: {train_ds.len()} train, {val_ds.len()} val"
          + (f", {test_ds.len()} test" if test_ds else ""))

    train_loader = DataLoader(
        train_ds, batch_size=config.train.batch_size, shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.train.batch_size,
        num_workers=config.num_workers,
    )
    test_loader = (
        DataLoader(test_ds, batch_size=config.train.batch_size,
                   num_workers=config.num_workers)
        if test_ds else None
    )

    # ==================================================================
    # Optimizer with differential LR
    # ==================================================================
    if config.train.backbone_lr_factor < 1.0:
        param_groups = [
            {"params": model.backbone_parameters(),
             "lr": config.train.lr * config.train.backbone_lr_factor},
            {"params": model.head_parameters("inconsistency"),
             "lr": config.train.lr},
        ]
    else:
        param_groups = [{"params": model.parameters(), "lr": config.train.lr}]

    optimizer = torch.optim.AdamW(
        param_groups, weight_decay=config.train.weight_decay,
    )
    scheduler = build_scheduler(optimizer, config.train.scheduler, config.train.epochs)

    if config.train.loss == "huber":
        loss_fn = nn.HuberLoss(delta=config.train.huber_delta)
    elif config.train.loss == "combined":
        # Weighted sum of Huber (robust, good for ρ) + MSE (good for R²)
        _huber = nn.HuberLoss(delta=config.train.huber_delta)
        _mse   = nn.MSELoss()
        _w     = getattr(config.train, "mse_weight", 0.5)
        class _CombinedLoss(nn.Module):
            def forward(self, pred, target):
                return (1.0 - _w) * _huber(pred, target) + _w * _mse(pred, target)
        loss_fn = _CombinedLoss()
    else:
        loss_fn = nn.MSELoss()

    stopper = EarlyStopping(config.train.patience, mode="max")
    best_val_rho = -float("inf")
    best_test_rho = -float("inf")
    finetune_logger = MetricsLogger(output_dir / "finetune_log.csv")

    # ==================================================================
    # Training loop
    # ==================================================================
    for epoch in range(config.train.epochs):
        # Optional backbone freezing
        if epoch < config.train.freeze_backbone_epochs:
            for p in model.backbone_parameters():
                p.requires_grad = False
        elif epoch == config.train.freeze_backbone_epochs and epoch > 0:
            for p in model.backbone_parameters():
                p.requires_grad = True
            print(f"  Epoch {epoch}: unfreezing backbone")

        t0 = time.perf_counter()
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, device,
            config.train.auxiliary_aabb_weight,
        )
        val_metrics = evaluate(model, val_loader, device)
        test_metrics = evaluate(model, test_loader, device) if test_loader else None
        scheduler.step()
        dt = time.perf_counter() - t0

        # Log to CSV
        log_row = {
            "epoch": epoch,
            "lr_backbone": optimizer.param_groups[0]["lr"],
            "lr_head": optimizer.param_groups[-1]["lr"],
            "train_loss": train_metrics["loss"],
            "val_mse": val_metrics["mse"],
            "val_mae": val_metrics["mae"],
            "val_r2": val_metrics["r2"],
            "val_spearman_rho": val_metrics["spearman_rho"],
        }
        if test_metrics is not None:
            log_row.update({
                "test_mse": test_metrics["mse"],
                "test_mae": test_metrics["mae"],
                "test_r2": test_metrics["r2"],
                "test_spearman_rho": test_metrics["spearman_rho"],
            })
        finetune_logger.log(log_row)

        if epoch % 5 == 0 or epoch == config.train.epochs - 1:
            test_str = ""
            if test_metrics is not None:
                test_str = (f" | test_mse={test_metrics['mse']:.6f}"
                            f" | test_r2={test_metrics['r2']:.4f}")
            print(f"  Epoch {epoch:3d} | loss={train_metrics['loss']:.4f} "
                  f"| val_mse={val_metrics['mse']:.6f} "
                  f"| val_r2={val_metrics['r2']:.4f} "
                  f"| rho={val_metrics['spearman_rho']:.4f}"
                  f"{test_str} | {dt:.1f}s")

        if val_metrics["spearman_rho"] > best_val_rho:
            best_val_rho = val_metrics["spearman_rho"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": dataclasses.asdict(config),
                "epoch": epoch,
                "val_metrics": val_metrics,
            }, output_dir / "best_model.pt")

        if test_metrics is not None and test_metrics["spearman_rho"] > best_test_rho:
            best_test_rho = test_metrics["spearman_rho"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": dataclasses.asdict(config),
                "epoch": epoch,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
            }, output_dir / "best_model_test.pt")

        if stopper.step(val_metrics["spearman_rho"], epoch):
            print(f"  Early stopping at epoch {epoch}")
            break

    finetune_logger.close()
    plot_finetune_curves(output_dir)

    # ==================================================================
    # Final evaluation
    # ==================================================================
    print(f"\n--- Evaluation ---")

    # Use best-test model for final plots if available, else best-val
    best_test_path = output_dir / "best_model_test.pt"
    best_val_path = output_dir / "best_model.pt"
    if test_loader is not None and best_test_path.exists():
        final_ckpt_path = best_test_path
        print("  Using best-test-MSE checkpoint for final evaluation.")
    else:
        final_ckpt_path = best_val_path
        print("  Using best-val-MSE checkpoint for final evaluation.")

    best_ckpt = torch.load(final_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    best_epoch = best_ckpt["epoch"]

    val_final = evaluate(model, val_loader, device)
    print(f"  Best epoch: {best_epoch}")
    print(f"  Val  — MSE={val_final['mse']:.6f}, MAE={val_final['mae']:.4f}, "
          f"R2={val_final['r2']:.4f}, rho={val_final['spearman_rho']:.4f}")

    results = {
        "best_epoch": best_epoch,
        "val_metrics": val_final,
        "config": dataclasses.asdict(config),
    }

    # Collect predictions for plots
    val_preds, val_targets, val_scenarios = collect_predictions(
        model, val_loader, device,
    )
    plot_predictions(val_preds, val_targets, val_scenarios, "Val", output_dir)

    test_preds = test_targets = test_scenarios = None

    if test_loader is not None:
        test_final = evaluate(model, test_loader, device)
        print(f"  Test — MSE={test_final['mse']:.6f}, MAE={test_final['mae']:.4f}, "
              f"R2={test_final['r2']:.4f}, rho={test_final['spearman_rho']:.4f}")
        results["test_metrics"] = test_final

        test_preds, test_targets, test_scenarios = collect_predictions(
            model, test_loader, device,
        )
        plot_predictions(test_preds, test_targets, test_scenarios, "Test", output_dir)

        # Per-scenario breakdown
        per_scenario = {}
        for s_idx in config.train.scenarios_test:
            try:
                s_ds = InconsistencyDataset(data_root, [s_idx], split="all", seed=config.seed)
                s_loader = DataLoader(s_ds, batch_size=config.train.batch_size)
                s_metrics = evaluate(model, s_loader, device)
                per_scenario[f"S{s_idx:02d}"] = s_metrics
                print(f"    S{s_idx:02d} — MSE={s_metrics['mse']:.6f}, "
                      f"R2={s_metrics['r2']:.4f}")
            except FileNotFoundError:
                pass
        results["per_scenario_test"] = per_scenario

    # Combined scatter plot (val + test side by side)
    plot_predictions_combined(
        val_preds, val_targets, val_scenarios,
        test_preds, test_targets, test_scenarios,
        output_dir,
    )

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved -> {output_dir / 'results.json'}")
    print(f"  Best model   -> {final_ckpt_path}")
    print(f"  Plots        -> {output_dir}/*.png")


if __name__ == "__main__":
    main()
