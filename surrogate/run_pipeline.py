"""
Full surrogate pipeline: train → evaluate → plot analysis.

Runs all three stages in sequence with a single command.

Usage
-----
    python -m surrogate.run_pipeline                          # full run
    python -m surrogate.run_pipeline --max_samples 50000     # quick test
    python -m surrogate.run_pipeline --epochs 30 --lr 5e-4   # custom training
    python -m surrogate.run_pipeline --skip_train            # eval + plots only

Outputs
-------
    surrogate/model.pt           -- best checkpoint (train stage)
    results/paper_figures/  -- all figures and summaries (analysis stage)
"""

import argparse
import sys
import time
from pathlib import Path


def run_train(args):
    print("=" * 60)
    print("STAGE 1 / 3  —  Training")
    print("=" * 60)

    import types
    from .train import train as _train

    train_args = types.SimpleNamespace(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        phi_hidden=args.phi_hidden,
        rho_hidden=args.rho_hidden,
        label_key=args.label_key,
        val_fraction=args.val_fraction,
        max_samples=args.max_samples,
    )
    _train(train_args)


def run_evaluate(args):
    print()
    print("=" * 60)
    print("STAGE 2 / 3  —  Evaluation")
    print("=" * 60)

    import types
    from .evaluate import evaluate as _evaluate

    eval_args = types.SimpleNamespace(
        label_key=args.label_key,
        max_samples=args.max_samples,
    )
    _evaluate(eval_args)


def run_analysis(args):
    print()
    print("=" * 60)
    print("STAGE 3 / 3  —  Analysis & Figures")
    print("=" * 60)

    import types
    from .run_analysis import main as _main

    analysis_args = types.SimpleNamespace(
        max_samples=args.max_samples,
        max_scenarios=args.max_scenarios,
        output=args.output,
        plots=getattr(args, "plots", None),
        landscape_a=getattr(args, "landscape_a", None),
        landscape_b=getattr(args, "landscape_b", None),
        gamma=getattr(args, "gamma", 0.5),
        cf_scenarios=getattr(args, "cf_scenarios", None),
    )
    _main(analysis_args)


def parse_args():
    p = argparse.ArgumentParser(
        description="Run full surrogate pipeline: train → evaluate → plot"
    )
    # ── stage control ─────────────────────────────────────────────────────
    p.add_argument("--skip_train",    action="store_true",
                   help="Skip training stage (use existing model.pt)")
    p.add_argument("--skip_evaluate", action="store_true",
                   help="Skip evaluation stage")
    p.add_argument("--skip_analysis", action="store_true",
                   help="Skip figure generation stage")

    # ── training hyperparameters ──────────────────────────────────────────
    p.add_argument("--epochs",       type=int,   default=20)
    p.add_argument("--batch_size",   type=int,   default=2048)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--phi_hidden",   type=int,   default=16)
    p.add_argument("--rho_hidden",   type=int,   default=32)
    p.add_argument("--label_key",    type=str,   default="I_theta")
    p.add_argument("--val_fraction", type=float, default=0.15)

    # ── shared / analysis ─────────────────────────────────────────────────
    p.add_argument("--max_samples",  type=int,   default=None,
                   help="Cap samples (all stages)")
    p.add_argument("--max_scenarios", type=int,  default=None,
                   help="Cap scenarios for Sobol computation (analysis stage)")
    p.add_argument("--output",       type=str,   default="results/paper_figures",
                   help="Output directory for figures")
    p.add_argument("--plots",        type=str,   default=None,
                   help="Comma-separated plot names or RQ shortcuts (q1/q2/q3/q4). "
                        "Example: --plots paper_landscape,landscape_a_compare")
    p.add_argument("--landscape_a",  type=int,   default=None,
                   help="Scenario id for paper_landscape panel A (e.g. --landscape_a 11)")
    p.add_argument("--landscape_b",  type=str,   default=None,
                   help="Comma-separated scenario ids for panel B (e.g. --landscape_b 49,1,35)")
    p.add_argument("--cf_scenarios", type=str,   default=None,
                   help="Comma-separated scenario ids for Q4 counterfactual (e.g. --cf_scenarios 35,49,7)")

    return p.parse_args()


def main(args=None):
    if args is None:
        args = parse_args()

    t_start = time.time()

    if not args.skip_train:
        run_train(args)
    else:
        print("Skipping training stage (--skip_train).")

    if not args.skip_evaluate:
        run_evaluate(args)
    else:
        print("Skipping evaluation stage (--skip_evaluate).")

    if not args.skip_analysis:
        run_analysis(args)
    else:
        print("Skipping analysis stage (--skip_analysis).")

    elapsed = time.time() - t_start
    m, s = divmod(int(elapsed), 60)
    print()
    print("=" * 60)
    print(f"Pipeline complete in {m}m {s}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
