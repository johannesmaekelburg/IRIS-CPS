"""CLI for counterfactual explanation evaluation.

Run from the repo root:

    # Sequential (12 CONVIDE scenarios, 20 queries each)
    python src/run_counterfactual_eval.py

    # Parallel with 8 workers (all 48 scenarios)
    python src/run_counterfactual_eval.py --parallel --workers 8 --scenarios all

    # Quick smoke-test (1 scenario, 3 queries)
    python src/run_counterfactual_eval.py --scenarios 1 --queries 3 --no-baseline

    # Custom checkpoint + output file
    python src/run_counterfactual_eval.py \
        --checkpoint results/surrogate/20260420_132706/best_model.pt \
        --output results/counterfactual/eval_results.json \
        --parallel --workers 12
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow running from the repo root or from src/
_repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Counterfactual explanation evaluation for MMS inconsistency."
    )

    # Model + data
    p.add_argument(
        "--checkpoint",
        default="results/surrogate/20260416_103330/best_model.pt",
        help="Path to best_model.pt (default: latest run).",
    )
    p.add_argument(
        "--data-root",
        default="data/surrogate",
        help="Path to data/surrogate/ directory.",
    )
    p.add_argument(
        "--measurement-root",
        default="data/measurements",
        help=(
            "Optional measurement directory with results_scenario_*.json. "
            "If set, theta* queries are drawn from measured experiments instead "
            "of uniform box sampling. Defaults to data/measurements and will "
            "auto-resolve CPS/CONVIDE subfolders by scenario index."
        ),
    )
    p.add_argument(
        "--max-pre-inconsistency",
        type=float,
        default=None,
        help=(
            "Optional filter when using --measurement-root: only use measured "
            "queries whose pre_state I(theta) is at most this value."
        ),
    )
    p.add_argument(
        "--max-query-inconsistency",
        type=float,
        default=None,
        help=(
            "Cap starting inconsistency when sampling random queries: only keep "
            "theta* with gamma < I(theta*) <= this value (e.g. 0.6). "
            "Focuses evaluation on near-boundary queries where the surrogate "
            "is better calibrated and fixes are more plausible."
        ),
    )

    # Scenarios
    p.add_argument(
        "--scenarios",
        default="convide",
        help=(
            "Which scenarios to evaluate. Options: "
            "'convide' (S01-S12, default), 'cps' (S13-S48), 'all' (S01-S48), "
            "or a comma-separated list of integers e.g. '1,3,5'."
        ),
    )

    # Evaluation config
    p.add_argument("--queries", type=int, default=20, help="Queries per scenario.")
    p.add_argument("--gamma", type=float, default=0.3, help="Consistency threshold.")
    p.add_argument(
        "--gamma-surrogate",
        type=float,
        default=None,
        help=(
            "Optional surrogate-side threshold used during optimization. "
            "MC/MFMC validity is still judged against --gamma."
        ),
    )
    p.add_argument(
        "--mc-verify", type=int, default=2000, help="MC samples for final verification."
    )
    p.add_argument(
        "--mc-screen", type=int, default=500, help="MC samples for screening theta*."
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip naive baseline comparison.",
    )

    # Optimizer
    p.add_argument("--lr", type=float, default=0.01, help="Adam lr for theta.")
    p.add_argument("--max-iter", type=int, default=500, help="Max optimizer iterations.")
    p.add_argument("--lambda-init", type=float, default=10.0, help="Initial penalty lambda.")
    p.add_argument(
        "--search-mode",
        choices=["hybrid", "surrogate", "cem", "mppi", "spsa", "multifidelity"],
        default="hybrid",
        help=(
            "Counterfactual search mode. 'hybrid' lets the GNN propose multiple "
            "repairs and uses MC/MFMC to select the winner, 'multifidelity' adds "
            "MFMC-guided SPSA refinement on top of the hybrid proposals, and the "
            "others are direct search baselines in theta-space."
        ),
    )
    p.add_argument(
        "--candidate-starts",
        type=int,
        default=3,
        help="Number of extra random hybrid proposal starts.",
    )
    p.add_argument(
        "--rerank-top-k",
        type=int,
        default=4,
        help="How many GNN proposals to verify with MC/MFMC in hybrid mode.",
    )
    p.add_argument(
        "--raw-proximity",
        action="store_true",
        help="Use raw parameter-space distance instead of normalized box-scaled distance.",
    )
    p.add_argument(
        "--population-size",
        type=int,
        default=64,
        help="Population size for CEM/MPPI proposal search.",
    )
    p.add_argument(
        "--elite-frac",
        type=float,
        default=0.2,
        help="Elite fraction for CEM updates.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.05,
        help="Softmax temperature for MPPI-style updates.",
    )
    p.add_argument(
        "--init-sigma-frac",
        type=float,
        default=0.2,
        help="Initial search std as a fraction of parameter box width for CEM/MPPI.",
    )
    p.add_argument(
        "--spsa-perturb-scale",
        type=float,
        default=0.1,
        help="Initial perturbation scale as a fraction of box width for SPSA.",
    )
    # Multi-fidelity (Phase-2 SPSA) parameters
    p.add_argument(
        "--mf-spsa-iter",
        type=int,
        default=40,
        help="SPSA refinement iterations in multi-fidelity mode (default: 40).",
    )
    p.add_argument(
        "--mf-mc-per-eval",
        type=int,
        default=50,
        help="MC samples per MFMC evaluation during SPSA refinement (default: 50).",
    )
    # Parallelism
    p.add_argument(
        "--parallel",
        action="store_true",
        help="Run scenarios in parallel (one process per scenario).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: all CPU cores).",
    )

    # Output
    p.add_argument(
        "--output",
        default=None,
        help=(
            "Path for JSON output. "
            "Defaults to results/counterfactual/eval_<timestamp>.json."
        ),
    )
    p.add_argument(
        "--device",
        default="auto",
        help="Torch device (auto / cpu / cuda:0).",
    )

    return p.parse_args()


def resolve_scenarios(spec: str):
    from causal_engine import (
        create_all_scenarios,
        create_convide_scenarios,
        create_cps_scenarios,
    )

    if spec == "convide":
        scenarios = create_convide_scenarios()
        indices = list(range(1, 13))
    elif spec == "cps":
        scenarios = create_cps_scenarios()
        indices = list(range(13, 49))
    elif spec == "all":
        scenarios = create_all_scenarios()
        indices = list(range(1, 49))
    else:
        # Comma-separated list: e.g. "1,3,5"
        all_scenarios = create_all_scenarios()
        all_indices = list(range(1, 49))
        idx_map = {i: s for i, s in zip(all_indices, all_scenarios)}
        requested = [int(x.strip()) for x in spec.split(",")]
        scenarios = [idx_map[i] for i in requested if i in idx_map]
        indices = [i for i in requested if i in idx_map]
        missing = [i for i in requested if i not in idx_map]
        if missing:
            print(f"WARNING: scenario indices not found: {missing}")

    return scenarios, indices


def main() -> None:
    args = parse_args()

    from learned_surrogate.counterfactual import (
        CounterfactualEvalConfig,
        CounterfactualExplainer,
        evaluate_counterfactuals_parallel,
    )
    from learned_surrogate.config import resolve_device

    requested_device = args.device
    args.device = resolve_device(args.device)
    if args.device != requested_device:
        print(
            f"Requested device '{requested_device}' is unavailable; "
            f"falling back to '{args.device}'."
        )
    print(f"Using torch device: {args.device}")

    print(f"Loading scenarios: {args.scenarios}")
    scenarios, indices = resolve_scenarios(args.scenarios)
    print(f"  -> {len(scenarios)} scenarios: {indices}")

    cfg = CounterfactualEvalConfig(
        gamma=args.gamma,
        n_queries=args.queries,
        mc_samples_verify=args.mc_verify,
        mc_samples_screen=args.mc_screen,
        seed=args.seed,
        run_baseline=not args.no_baseline,
        lr=args.lr,
        max_iter=args.max_iter,
        lambda_init=args.lambda_init,
        measurement_root=args.measurement_root,
        max_pre_inconsistency=args.max_pre_inconsistency,
        max_query_inconsistency=args.max_query_inconsistency,
        search_mode=args.search_mode,
        n_candidate_starts=args.candidate_starts,
        rerank_top_k=args.rerank_top_k,
        normalize_proximity=not args.raw_proximity,
        gamma_surrogate=args.gamma_surrogate,
        population_size=args.population_size,
        elite_fraction=args.elite_frac,
        mppi_temperature=args.temperature,
        init_sigma_frac=args.init_sigma_frac,
        spsa_perturb_scale=args.spsa_perturb_scale,
        mf_spsa_iter=args.mf_spsa_iter,
        mf_mc_per_eval=args.mf_mc_per_eval,
    )

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        checkpoint = _repo / args.checkpoint
    if not checkpoint.exists():
        print(f"ERROR: checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    data_root = Path(args.data_root)
    if not data_root.exists():
        data_root = _repo / args.data_root

    if args.output is None:
        out_dir = _repo / "results" / "counterfactual"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"eval_{ts}.json"
    else:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    snapshot_meta = {
        "checkpoint": str(checkpoint),
        "requested_device": requested_device,
        "device": args.device,
        "scenarios": args.scenarios,
        "n_scenarios": len(scenarios),
        "gamma": args.gamma,
        "gamma_surrogate": args.gamma_surrogate,
        "n_queries": args.queries,
        "parallel": args.parallel,
        "n_workers": args.workers,
        "measurement_root": args.measurement_root,
        "max_pre_inconsistency": args.max_pre_inconsistency,
        "max_query_inconsistency": args.max_query_inconsistency,
        "search_mode": args.search_mode,
        "n_candidate_starts": args.candidate_starts,
        "rerank_top_k": args.rerank_top_k,
        "normalize_proximity": not args.raw_proximity,
        "population_size": args.population_size,
        "elite_fraction": args.elite_frac,
        "temperature": args.temperature,
        "init_sigma_frac": args.init_sigma_frac,
        "spsa_perturb_scale": args.spsa_perturb_scale,
        "mf_spsa_iter": args.mf_spsa_iter,
        "mf_mc_per_eval": args.mf_mc_per_eval,
        "output_path": str(out_path),
    }
    print(f"Streaming incremental results to: {out_path}")
    print(
        "Search mode: "
        f"{args.search_mode}"
        + (
            f" (top-{args.rerank_top_k} MC/MFMC rerank, {args.candidate_starts} extra starts)"
            if args.search_mode in ("hybrid", "multifidelity")
            else f" (population={args.population_size}, elite={args.elite_frac:.2f})"
            if args.search_mode == "cem"
            else f" (population={args.population_size}, temp={args.temperature:.3f})"
            if args.search_mode == "mppi"
            else f" (perturb={args.spsa_perturb_scale:.3f})"
            if args.search_mode == "spsa"
            else ""
        )
        + (
            " with normalized proximity"
            if not args.raw_proximity
            else " with raw proximity"
        )
    )
    if args.gamma_surrogate is not None:
        print(
            f"Thresholds: surrogate gamma={args.gamma_surrogate}  "
            f"MC/MFMC gamma={args.gamma}"
        )
    if args.measurement_root:
        print(
            "Using measured query source: "
            f"{args.measurement_root}"
            + (
                f" (pre_state I(theta) <= {args.max_pre_inconsistency})"
                if args.max_pre_inconsistency is not None
                else ""
            )
        )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "status": "started",
                "meta": snapshot_meta,
                "per_scenario_results": {},
                "per_scenario_baseline": {},
                "per_scenario_summary": [],
                "aggregate": {},
            },
            f,
            indent=2,
        )

    t0 = time.perf_counter()

    if args.parallel:
        import os

        n_workers = args.workers or min(len(scenarios), os.cpu_count() or 1)
        print(f"Parallel mode: {n_workers} workers")

        results = evaluate_counterfactuals_parallel(
            checkpoint_path=str(checkpoint),
            scenarios=scenarios,
            scenario_indices=indices,
            data_root=str(data_root),
            cfg=cfg,
            device=args.device,
            n_workers=n_workers,
            snapshot_path=str(out_path),
            snapshot_meta=snapshot_meta,
        )
    else:
        print("Sequential mode")
        explainer = CounterfactualExplainer.from_checkpoint(
            checkpoint_path=str(checkpoint),
            data_root=str(data_root),
            gamma=args.gamma,
            device=args.device,
        )
        results = explainer.evaluate(
            scenarios,
            indices,
            cfg=cfg,
            snapshot_path=str(out_path),
            snapshot_meta=snapshot_meta,
        )

    elapsed = time.perf_counter() - t0

    agg = results.get("aggregate", {})
    print("\n" + "=" * 60)
    print(f"AGGREGATE RESULTS ({elapsed:.1f}s total)")
    print("=" * 60)
    print(f"  Scenarios evaluated : {agg.get('n_scenarios', 0)}")
    print(f"  Total queries       : {agg.get('n_queries_total', 0)}")
    print(
        f"  Validity rate       : {agg.get('mean_validity_rate', 0):.1%}  "
        f"(+/-{agg.get('std_validity_rate', 0):.1%})"
    )
    print(f"  Mean proximity ||delta theta|| : {agg.get('mean_proximity_l2', 0):.4f}")
    print(
        f"  Mean start / end    : {agg.get('mean_I_surrogate_star', float('nan')):.3f}"
        f"  -> surr {agg.get('mean_I_surrogate_prime', float('nan')):.3f}"
        f"  -> mc {agg.get('mean_I_mc_prime', float('nan')):.3f}"
    )
    print(
        f"  Mean improvement    : d_surr={agg.get('mean_surrogate_improvement', float('nan')):.3f}"
        f"  d_mc={agg.get('mean_mc_improvement_vs_surrogate_start', float('nan')):.3f}"
    )
    print(
        f"  Near-feasible rates : MC<=0.4 {agg.get('mc_below_040_rate', 0):.1%}"
        f"  MC<=0.5 {agg.get('mc_below_050_rate', 0):.1%}"
        f"  improved {agg.get('surrogate_improvement_rate', 0):.1%}"
    )
    print(f"  Mean MC gap to gamma: {agg.get('mean_mc_gap_to_gamma', float('nan')):+.3f}")
    print(f"  Dominant fix param  : {agg.get('dominant_param', '?')}")
    print()

    delta = agg.get("mean_abs_delta_theta", {})
    for name, val in delta.items():
        print(f"    |delta {name[:12]:<12}| = {val:.4f}")
    print()

    dom_freq = agg.get("dominant_param_freq", {})
    for name, freq in sorted(dom_freq.items(), key=lambda x: -x[1]):
        print(f"    dom freq {name[:20]:<20} = {freq:.1%}")

    def _to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, dict):
            return {k: _to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_serializable(v) for v in obj]
        return obj

    results["meta"] = {
        "checkpoint": str(checkpoint),
        "scenarios": args.scenarios,
        "n_scenarios": len(scenarios),
        "gamma": args.gamma,
        "n_queries": args.queries,
        "parallel": args.parallel,
        "n_workers": args.workers,
        "measurement_root": args.measurement_root,
        "max_pre_inconsistency": args.max_pre_inconsistency,
        "wall_time_s": elapsed,
        "output_path": str(out_path),
    }
    results["status"] = "complete"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(results), f, indent=2)

    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    # Required on Windows for multiprocessing spawn.
    main()
