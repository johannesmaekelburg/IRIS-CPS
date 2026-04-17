"""Optuna hyperparameter optimisation for the GINE inconsistency surrogate.

Each trial:
  1. Samples hyperparameters from the search space.
  2. Writes a temporary YAML config.
  3. Runs train.py as a subprocess with reduced epochs.
  4. Reads the resulting results.json and returns val Spearman rho.

Multiple SLURM jobs can run in parallel against the same SQLite study DB.

Usage (single worker):
    python -m src.learned_surrogate.optuna_train \\
        --base_config src/learned_surrogate/configs/surrogate_default.yaml \\
        --study_name surrogate_hpo \\
        --n_trials 5

Usage (parallel SLURM workers — each job runs one trial):
    python -m src.learned_surrogate.optuna_train \\
        --base_config src/learned_surrogate/configs/surrogate_default.yaml \\
        --study_name surrogate_hpo \\
        --n_trials 1          # one trial per SLURM job
        --storage results/optuna/study.db

After all trials:
    python -m src.learned_surrogate.optuna_train --report_only \\
        --study_name surrogate_hpo \\
        --storage results/optuna/study.db
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

def _sample_config(trial, base_cfg: dict) -> dict:
    """Sample hyperparameters and return an updated config dict."""
    import copy
    cfg = copy.deepcopy(base_cfg)

    # ── Model architecture ──────────────────────────────────────────────────
    node_embed_dim = trial.suggest_categorical("node_embed_dim", [64, 128])
    n_gine_layers  = trial.suggest_int("n_gine_layers", 2, 4)

    cfg["model"]["node_embed_dim"]   = node_embed_dim
    cfg["model"]["n_gine_layers"]    = n_gine_layers
    cfg["model"]["gine_hidden_dim"]  = node_embed_dim
    cfg["model"]["edge_mlp_dims"]    = [20, node_embed_dim, node_embed_dim]
    cfg["model"]["head_mlp_dims"]    = [node_embed_dim, node_embed_dim // 2, 1]

    # ── Fine-tuning ─────────────────────────────────────────────────────────
    cfg["train"]["lr"] = trial.suggest_float("lr", 5e-5, 5e-4, log=True)
    cfg["train"]["huber_delta"] = trial.suggest_float(
        "huber_delta", 0.01, 0.2, log=True
    )
    cfg["train"]["freeze_backbone_epochs"] = trial.suggest_categorical(
        "freeze_backbone_epochs", [0, 5, 10, 20]
    )
    cfg["train"]["backbone_lr_factor"] = trial.suggest_float(
        "backbone_lr_factor", 0.01, 0.3, log=True
    )

    # ── Fixed (known good values) ────────────────────────────────────────────
    cfg["train"]["loss"]      = "huber"
    cfg["train"]["scheduler"] = "step"
    cfg["device"]             = "cpu"

    return cfg


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------

def _run_trial(trial, base_cfg: dict, project_root: Path,
               trial_epochs: int, storage_dir: Path) -> float:
    """Write config, run train.py, return best val Spearman rho."""
    import optuna

    cfg = _sample_config(trial, base_cfg)

    # Reduced epochs for HPO speed
    cfg["train"]["epochs"] = trial_epochs
    cfg["pretrain"]["epochs"] = 500   # patience will stop it early anyway

    # Each trial gets its own output dir
    trial_dir = storage_dir / f"trial_{trial.number:03d}"
    cfg["output_dir"] = str(trial_dir.relative_to(project_root))

    # Write temporary config
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, dir=project_root
    ) as f:
        yaml.dump(cfg, f, default_flow_style=False)
        tmp_cfg_path = Path(f.name)

    try:
        cmd = [
            sys.executable, "-u", "-m", "src.learned_surrogate.train",
            "--config", str(tmp_cfg_path.relative_to(project_root)),
        ]
        print(f"\n[Trial {trial.number}] Running: {' '.join(cmd)}")
        print(f"[Trial {trial.number}] Params: {trial.params}")

        result = subprocess.run(
            cmd, cwd=str(project_root),
            capture_output=False,   # stream output live
        )

        if result.returncode != 0:
            raise optuna.exceptions.TrialPruned(
                f"Training subprocess failed (exit {result.returncode})"
            )

        # Find the run directory (timestamped subdir inside trial_dir)
        run_dirs = sorted(trial_dir.glob("*/results.json"),
                          key=lambda p: p.stat().st_mtime)
        if not run_dirs:
            # Fallback: results.json directly in trial_dir
            results_path = trial_dir / "results.json"
        else:
            results_path = run_dirs[-1]

        if not results_path.exists():
            raise optuna.exceptions.TrialPruned("results.json not found")

        results = json.loads(results_path.read_text())
        val_rho = results["val_metrics"]["spearman_rho"]
        print(f"[Trial {trial.number}] val_rho={val_rho:.4f}")

        # Log extra metrics as trial user attributes
        trial.set_user_attr("val_r2",       results["val_metrics"]["r2"])
        trial.set_user_attr("val_mse",      results["val_metrics"]["mse"])
        trial.set_user_attr("best_epoch",   results["best_epoch"])
        if "test_metrics" in results:
            trial.set_user_attr("test_rho", results["test_metrics"]["spearman_rho"])
            trial.set_user_attr("test_r2",  results["test_metrics"]["r2"])

        return val_rho

    finally:
        tmp_cfg_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _print_report(study) -> None:
    import pandas as pd
    print(f"\n{'=' * 70}")
    print(f"Study: {study.study_name}  |  {len(study.trials)} trials")
    print(f"Best val rho: {study.best_value:.4f}")
    print(f"Best params:  {study.best_params}")
    print(f"Best trial user attrs: {study.best_trial.user_attrs}")
    print(f"{'=' * 70}\n")

    rows = []
    for t in study.trials:
        if t.value is None:
            continue
        row = {"trial": t.number, "val_rho": t.value}
        row.update(t.params)
        row.update(t.user_attrs)
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows).sort_values("val_rho", ascending=False)
        print(df.to_string(index=False))


# ---------------------------------------------------------------------------
# Seed study with previous manual runs
# ---------------------------------------------------------------------------

# Known runs: (params_dict, val_rho, extra_attrs)
_PREVIOUS_RUNS = [
    # 20260416_103330 — Large + Huber (best overall)
    (
        {"node_embed_dim": 64, "n_gine_layers": 3, "lr": 1e-4,
         "huber_delta": 0.05, "freeze_backbone_epochs": 10,
         "backbone_lr_factor": 0.1},
        0.9442,
        {"val_r2": 0.959, "test_rho": 0.867, "test_r2": 0.753,
         "best_epoch": 45, "run": "20260416_103330"},
    ),
    # 20260416_140401 — Large + Combined loss
    (
        {"node_embed_dim": 64, "n_gine_layers": 3, "lr": 1e-4,
         "huber_delta": 0.05, "freeze_backbone_epochs": 10,
         "backbone_lr_factor": 0.1},
        0.9432,
        {"val_r2": 0.963, "test_rho": 0.767, "test_r2": 0.456,
         "best_epoch": 83, "run": "20260416_140401"},
    ),
]


def _seed_study_with_previous_runs(study) -> None:
    """Add previous manual runs as completed trials so Optuna learns from them."""
    import optuna

    existing_runs = {
        t.user_attrs.get("run") for t in study.trials
        if t.user_attrs.get("run")
    }

    added = 0
    for params, val_rho, attrs in _PREVIOUS_RUNS:
        if attrs.get("run") in existing_runs:
            continue  # already added
        trial = optuna.trial.create_trial(
            params=params,
            distributions={
                "node_embed_dim":          optuna.distributions.CategoricalDistribution([64, 128]),
                "n_gine_layers":           optuna.distributions.IntDistribution(2, 4),
                "lr":                      optuna.distributions.FloatDistribution(5e-5, 5e-4, log=True),
                "huber_delta":             optuna.distributions.FloatDistribution(0.01, 0.2, log=True),
                "freeze_backbone_epochs":  optuna.distributions.CategoricalDistribution([0, 5, 10, 20]),
                "backbone_lr_factor":      optuna.distributions.FloatDistribution(0.01, 0.3, log=True),
            },
            value=val_rho,
            user_attrs=attrs,
        )
        study.add_trial(trial)
        added += 1

    if added:
        print(f"  Seeded study with {added} previous manual run(s).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optuna HPO for GINE inconsistency surrogate"
    )
    parser.add_argument("--base_config",
                        default="src/learned_surrogate/configs/surrogate_default.yaml",
                        help="Base YAML config to override")
    parser.add_argument("--study_name", default="surrogate_hpo")
    parser.add_argument("--storage",    default=None,
                        help="Optuna storage URL or path to SQLite file. "
                             "Default: results/optuna/<study_name>.db")
    parser.add_argument("--n_trials",   type=int, default=5,
                        help="Number of trials to run in this worker")
    parser.add_argument("--trial_epochs", type=int, default=150,
                        help="Fine-tuning epochs per trial (default: 150)")
    parser.add_argument("--report_only", action="store_true",
                        help="Print study results and exit, no new trials")
    args = parser.parse_args()

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    project_root = Path(__file__).resolve().parent.parent.parent

    # Storage
    if args.storage is None:
        optuna_dir = project_root / "results" / "optuna"
        optuna_dir.mkdir(parents=True, exist_ok=True)
        storage_url = f"sqlite:///{optuna_dir / (args.study_name + '.db')}"
    elif args.storage.startswith("sqlite://"):
        storage_url = args.storage
    else:
        db_path = Path(args.storage)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        storage_url = f"sqlite:///{db_path}"

    storage_dir = project_root / "results" / "optuna" / args.study_name
    storage_dir.mkdir(parents=True, exist_ok=True)

    # ── Tee stdout/stderr to a per-worker log file ────────────────────────
    # Named by SLURM job/array ID if available, otherwise timestamp
    import os
    slurm_job  = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID", "")
    slurm_task = os.environ.get("SLURM_ARRAY_TASK_ID", "")
    log_tag    = f"{slurm_job}_{slurm_task}" if slurm_job else time.strftime("%Y%m%d_%H%M%S")
    log_path   = storage_dir / f"worker_{log_tag}.log"

    class _Tee:
        def __init__(self, *streams):
            self._streams = streams
        def write(self, data):
            for s in self._streams:
                s.write(data)
        def flush(self):
            for s in self._streams:
                s.flush()

    _log_file  = open(log_path, "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, _log_file)
    sys.stderr = _Tee(sys.__stderr__, _log_file)
    print(f"Logging to: {log_path}")

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        direction="maximize",
        load_if_exists=True,
    )

    # Seed with previous manual runs so Optuna learns from them
    _seed_study_with_previous_runs(study)

    if args.report_only:
        _print_report(study)
        return

    # Load base config
    base_cfg_path = project_root / args.base_config
    with open(base_cfg_path) as f:
        base_cfg = yaml.safe_load(f)

    # Run trials
    study.optimize(
        lambda trial: _run_trial(
            trial, base_cfg, project_root, args.trial_epochs, storage_dir
        ),
        n_trials=args.n_trials,
        catch=(Exception,),
    )

    _print_report(study)

    # Save best config
    best_cfg = _sample_config(study.best_trial, base_cfg)
    best_cfg["train"]["epochs"] = 400   # restore full epochs for final run
    best_cfg_path = project_root / "src" / "learned_surrogate" / "configs" / \
                    f"surrogate_best_{args.study_name}.yaml"
    with open(best_cfg_path, "w") as f:
        yaml.dump(best_cfg, f, default_flow_style=False)
    print(f"\nBest config saved -> {best_cfg_path}")
    print("Run the best config with:")
    print(f"  python -m src.learned_surrogate.train "
          f"--config src/learned_surrogate/configs/"
          f"surrogate_best_{args.study_name}.yaml")


if __name__ == "__main__":
    main()
