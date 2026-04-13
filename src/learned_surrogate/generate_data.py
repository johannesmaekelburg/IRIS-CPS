"""
Generate training data for the learned inconsistency surrogate model.

Creates two kinds of data:
  1. Pretraining data (scenario-independent) -- teaches zonotope geometry.
  2. Training data (per-scenario) -- I(theta) via MC probability (high-fidelity).

Output structure
================
<output_dir>/
    meta.json                                   # generation config
    pretrain/
        volume.npz                              # random zonotopes -> interval hull volume
        containment.npz                         # random zonotopes + points -> in/out
        pairwise_aabb.npz                       # random zonotope pairs -> AABB Jaccard
        affine_map.npz                          # source + affine map -> target zonotope
    scenario_S01_CAD_Export_Drift/
        meta.json                               # scenario geometry (source, target, F, f)
        train_inconsistency.npz                 # theta -> I(theta) via MC + AABB
    scenario_S02_.../
        ...

Usage
=====
    # Generate everything for all 12 scenarios
    python generate_data.py --tasks all --scenario all

    # Only HF training data for scenario 4
    python generate_data.py --tasks train --scenario 4

    # Only pretraining data (scenario-independent)
    python generate_data.py --tasks pretrain

    # Only specific pretraining objectives
    python generate_data.py --tasks pretrain_volume pretrain_pairwise_aabb

    # Single scenario, more samples, higher MC fidelity
    python generate_data.py --tasks train --scenario 8 --n_train 20000 --mc_samples 5000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Import from causal_engine (parent directory)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from causal_engine import (
    Zonotope,
    Scenario,
    affine_map,
    contains_points_batch,
    aabb_jaccard,
    apply_compound_intervention,
    compute_I_theta,
    compute_I_theta_aabb,
    create_convide_scenarios,
    PARAM_NAMES,
    PARAM_BOUNDS,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

D_MAX = 4     # max spatial dimension (zonotopes padded to this)
P_MAX = 12    # max generator columns (base generators + correlate adds 1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pad_center(c: np.ndarray, d_max: int = D_MAX) -> np.ndarray:
    """Pad center vector to d_max with trailing zeros."""
    out = np.zeros(d_max)
    out[: len(c)] = c
    return out


def pad_generators(
    G: np.ndarray, d_max: int = D_MAX, p_max: int = P_MAX,
) -> np.ndarray:
    """Pad generator matrix to (d_max, p_max) with zeros."""
    d, p = G.shape
    out = np.zeros((d_max, p_max))
    out[:d, :p] = G
    return out


def random_zonotope(
    rng: np.random.Generator,
    dim: int,
    n_generators: int,
    center_scale: float = 100.0,
    generator_scale: float = 3.0,
) -> Zonotope:
    """Sample a random zonotope with given dimension and generator count."""
    c = rng.uniform(-center_scale, center_scale, size=dim)
    G = rng.uniform(-generator_scale, generator_scale, size=(dim, n_generators))
    return Zonotope(c, G)


def scenario_dir_name(scenario: Scenario) -> str:
    """Directory name for a scenario, e.g. 'scenario_S04_Control_Design_Conflict'."""
    parts = scenario.name.replace(":", "").split()
    num = int(parts[0][1:])  # "S4" -> 4
    rest = "_".join(parts[1:])
    return f"scenario_S{num:02d}_{rest}"


def save_scenario_meta(scenario: Scenario, output_dir: Path) -> None:
    """Save scenario geometry and parameter ranges as meta.json."""
    meta = {
        "scenario_name": scenario.name,
        "dim": scenario.dim,
        "source_center": scenario.source.c.tolist(),
        "source_generators": scenario.source.G.tolist(),
        "source_n_generators": scenario.source.n_generators,
        "target_center": scenario.target.c.tolist(),
        "target_generators": scenario.target.G.tolist(),
        "target_n_generators": scenario.target.n_generators,
        "F": scenario.F.tolist(),
        "f": scenario.f.tolist(),
        "param_names": PARAM_NAMES,
        "param_bounds": PARAM_BOUNDS,
        "d_max": D_MAX,
        "p_max": P_MAX,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)


# ---------------------------------------------------------------------------
# Pretraining: volume prediction
# ---------------------------------------------------------------------------

def generate_pretrain_volume(
    output_path: Path, n_samples: int, seed: int,
) -> None:
    """Random zonotopes -> interval-hull volume.

    Teaches the encoder to predict the size of a zonotope from (c, G).
    """
    print(f"\n  [pretrain_volume] Generating {n_samples} samples ...")
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()

    centers = np.zeros((n_samples, D_MAX))
    generators = np.zeros((n_samples, D_MAX, P_MAX))
    dims = np.zeros(n_samples, dtype=np.int32)
    n_gens = np.zeros(n_samples, dtype=np.int32)
    volumes = np.zeros(n_samples)

    for i in tqdm(range(n_samples), desc="volume", unit="Z"):
        d = int(rng.integers(2, D_MAX + 1))
        p = int(rng.integers(2, P_MAX + 1))
        Z = random_zonotope(rng, d, p)

        centers[i] = pad_center(Z.c)
        generators[i] = pad_generators(Z.G)
        dims[i] = d
        n_gens[i] = p
        volumes[i] = Z.volume_interval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        centers=centers,
        generators=generators,
        dim=dims,
        n_generators=n_gens,
        volume=volumes,
    )
    dt = time.perf_counter() - t0
    print(f"    -> {output_path}  ({dt:.1f}s)")
    print(f"    volume range: [{volumes.min():.2f}, {volumes.max():.2f}]")


# ---------------------------------------------------------------------------
# Pretraining: containment classification
# ---------------------------------------------------------------------------

def generate_pretrain_containment(
    output_path: Path,
    n_samples: int,
    n_test_points: int,
    seed: int,
) -> None:
    """Random zonotopes + test points -> binary containment labels.

    For each zonotope, generates two groups of test points:
      - Half are guaranteed inside (sampled via xi in [-1,1]^p).
      - Half are probed from an expanded bounding box and checked via LP.
    This yields naturally balanced labels with interesting boundary cases.
    """
    print(
        f"\n  [pretrain_containment] Generating {n_samples} zonotopes "
        f"x {n_test_points} points ..."
    )
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()

    centers = np.zeros((n_samples, D_MAX))
    generators = np.zeros((n_samples, D_MAX, P_MAX))
    dims = np.zeros(n_samples, dtype=np.int32)
    n_gens = np.zeros(n_samples, dtype=np.int32)
    points_all = np.zeros((n_samples, n_test_points, D_MAX))
    labels_all = np.zeros((n_samples, n_test_points), dtype=bool)

    n_inside = n_test_points // 2
    n_probe = n_test_points - n_inside

    for i in tqdm(range(n_samples), desc="containment", unit="Z"):
        d = int(rng.integers(2, D_MAX + 1))
        p = int(rng.integers(2, P_MAX + 1))
        Z = random_zonotope(rng, d, p)

        centers[i] = pad_center(Z.c)
        generators[i] = pad_generators(Z.G)
        dims[i] = d
        n_gens[i] = p

        # --- Guaranteed-inside points: x = c + G @ xi, xi in [-1,1]^p ---
        xi = rng.uniform(-1, 1, size=(n_inside, p))
        pts_inside = Z.c[None, :] + xi @ Z.G.T  # (n_inside, d)
        labels_inside = np.ones(n_inside, dtype=bool)

        # --- Probe points: expanded bounding box, check via LP ---
        lo, hi = Z.interval_bounds()
        half_w = (hi - lo) / 2
        pts_probe = rng.uniform(
            lo - 0.5 * half_w, hi + 0.5 * half_w, size=(n_probe, d),
        )
        labels_probe = contains_points_batch(Z, pts_probe)

        # Combine and store (padded to D_MAX)
        pts = np.vstack([pts_inside, pts_probe])
        labels = np.concatenate([labels_inside, labels_probe])
        for j in range(n_test_points):
            points_all[i, j, :d] = pts[j]
        labels_all[i] = labels

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        centers=centers,
        generators=generators,
        dim=dims,
        n_generators=n_gens,
        points=points_all,
        labels=labels_all,
        n_test_points=np.int32(n_test_points),
    )
    dt = time.perf_counter() - t0
    frac_pos = labels_all.mean()
    print(f"    -> {output_path}  ({dt:.1f}s)")
    print(f"    label balance: {frac_pos:.1%} positive")


# ---------------------------------------------------------------------------
# Pretraining: pairwise AABB Jaccard
# ---------------------------------------------------------------------------

def generate_pretrain_pairwise_aabb(
    output_path: Path, n_samples: int, seed: int,
) -> None:
    """Random zonotope pairs -> AABB Jaccard index.

    Teaches the encoder to recognise overlap between two zonotopes.
    Pairs are generated at varying separations to cover the full [0,1]
    Jaccard range.
    """
    print(f"\n  [pretrain_pairwise_aabb] Generating {n_samples} pairs ...")
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()

    centers_1 = np.zeros((n_samples, D_MAX))
    generators_1 = np.zeros((n_samples, D_MAX, P_MAX))
    centers_2 = np.zeros((n_samples, D_MAX))
    generators_2 = np.zeros((n_samples, D_MAX, P_MAX))
    dims = np.zeros(n_samples, dtype=np.int32)
    n_gens_1 = np.zeros(n_samples, dtype=np.int32)
    n_gens_2 = np.zeros(n_samples, dtype=np.int32)
    jaccard = np.zeros(n_samples)

    for i in tqdm(range(n_samples), desc="pairwise_aabb", unit="pair"):
        d = int(rng.integers(2, D_MAX + 1))
        p1 = int(rng.integers(2, P_MAX + 1))
        p2 = int(rng.integers(2, P_MAX + 1))

        Z1 = random_zonotope(rng, d, p1)

        # Vary separation to get good Jaccard coverage:
        #   ~33% close (high overlap), ~34% moderate, ~33% far (low/no overlap)
        r = rng.random()
        if r < 0.33:
            offset_scale = 1.0   # close
        elif r < 0.67:
            offset_scale = 5.0   # moderate
        else:
            offset_scale = 15.0  # far apart
        center_offset = rng.uniform(-offset_scale, offset_scale, size=d)
        Z2 = Zonotope(
            Z1.c + center_offset,
            rng.uniform(-3.0, 3.0, size=(d, p2)),
        )

        centers_1[i] = pad_center(Z1.c)
        generators_1[i] = pad_generators(Z1.G)
        centers_2[i] = pad_center(Z2.c)
        generators_2[i] = pad_generators(Z2.G)
        dims[i] = d
        n_gens_1[i] = p1
        n_gens_2[i] = p2
        jaccard[i] = aabb_jaccard(Z1, Z2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        centers_1=centers_1,
        generators_1=generators_1,
        centers_2=centers_2,
        generators_2=generators_2,
        dim=dims,
        n_generators_1=n_gens_1,
        n_generators_2=n_gens_2,
        aabb_jaccard=jaccard,
    )
    dt = time.perf_counter() - t0
    print(f"    -> {output_path}  ({dt:.1f}s)")
    print(
        f"    Jaccard distribution: "
        f"mean={jaccard.mean():.3f}, "
        f"zeros={np.mean(jaccard == 0):.1%}, "
        f"range=[{jaccard.min():.3f}, {jaccard.max():.3f}]"
    )


# ---------------------------------------------------------------------------
# Pretraining: affine map prediction
# ---------------------------------------------------------------------------

def generate_pretrain_affine_map(
    output_path: Path, n_samples: int, seed: int,
) -> None:
    """Source zonotope + random affine map -> target zonotope.

    Teaches the network how affine propagation (UPR) transforms zonotopes.
    Mix: 50% identity maps (like CONVIDE), 50% near-identity perturbations.
    """
    print(f"\n  [pretrain_affine_map] Generating {n_samples} samples ...")
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()

    src_c = np.zeros((n_samples, D_MAX))
    src_G = np.zeros((n_samples, D_MAX, P_MAX))
    tgt_c = np.zeros((n_samples, D_MAX))
    tgt_G = np.zeros((n_samples, D_MAX, P_MAX))
    F_all = np.zeros((n_samples, D_MAX, D_MAX))
    f_all = np.zeros((n_samples, D_MAX))
    dims = np.zeros(n_samples, dtype=np.int32)
    n_gens = np.zeros(n_samples, dtype=np.int32)

    for i in tqdm(range(n_samples), desc="affine_map", unit="map"):
        d = int(rng.integers(2, D_MAX + 1))
        p = int(rng.integers(2, P_MAX + 1))
        Z_src = random_zonotope(rng, d, p)

        # 50% identity, 50% near-identity perturbation
        if rng.random() < 0.5:
            F = np.eye(d)
            f = np.zeros(d)
        else:
            F = np.eye(d) + rng.normal(0, 0.3, size=(d, d))
            f = rng.uniform(-5.0, 5.0, size=d)

        Z_tgt = affine_map(Z_src, F, f)

        src_c[i] = pad_center(Z_src.c)
        src_G[i] = pad_generators(Z_src.G)
        tgt_c[i] = pad_center(Z_tgt.c)
        tgt_G[i] = pad_generators(Z_tgt.G)
        F_padded = np.zeros((D_MAX, D_MAX))
        F_padded[:d, :d] = F
        F_all[i] = F_padded
        f_all[i, :d] = f
        dims[i] = d
        n_gens[i] = p

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        source_center=src_c,
        source_generators=src_G,
        target_center=tgt_c,
        target_generators=tgt_G,
        F=F_all,
        f=f_all,
        dim=dims,
        n_generators=n_gens,
    )
    dt = time.perf_counter() - t0
    print(f"    -> {output_path}  ({dt:.1f}s)")


# ---------------------------------------------------------------------------
# Training data: per-scenario inconsistency (MC probability + AABB)
# ---------------------------------------------------------------------------

def generate_train_inconsistency(
    scenario: Scenario,
    output_dir: Path,
    n_samples: int,
    mc_samples: int,
    seed: int,
) -> None:
    """Sample theta from PARAM_BOUNDS, compute I(theta) via MC probability.

    Also computes the free AABB Jaccard estimate for each sample (useful
    for multi-fidelity training or as an auxiliary target).

    Stores the post-intervention source zonotope features so the training
    script can build GNN graphs without re-running interventions.
    """
    dir_name = scenario_dir_name(scenario)
    out_dir = output_dir / dir_name
    save_scenario_meta(scenario, out_dir)

    print(f"\n  [train] {scenario.name}  ({n_samples} samples, "
          f"{mc_samples} MC/eval)")

    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()

    bounds = np.array(PARAM_BOUNDS)

    # Sample theta uniformly from parameter bounds
    theta_samples = rng.uniform(
        bounds[:, 0], bounds[:, 1],
        size=(n_samples, len(PARAM_NAMES)),
    )

    # Pre-allocate
    I_mc = np.zeros(n_samples)
    I_mc_se = np.zeros(n_samples)
    I_aabb = np.zeros(n_samples)
    src_centers = np.zeros((n_samples, D_MAX))
    src_generators = np.zeros((n_samples, D_MAX, P_MAX))
    src_n_gens = np.zeros(n_samples, dtype=np.int32)

    mc_seeds = rng.integers(0, 2**31, size=n_samples)

    for i in tqdm(range(n_samples), desc=scenario.name, unit="theta"):
        theta = dict(zip(PARAM_NAMES, theta_samples[i]))
        Z_int = apply_compound_intervention(scenario.source, theta)

        # Store intervened source features (pre-propagation)
        src_centers[i] = pad_center(Z_int.c)
        src_generators[i] = pad_generators(Z_int.G)
        src_n_gens[i] = Z_int.n_generators

        # AABB Jaccard (free, closed-form)
        r_aabb = compute_I_theta_aabb(scenario, source_override=Z_int)
        I_aabb[i] = r_aabb["I_theta"]

        # MC probability (high-fidelity, expensive)
        r_mc = compute_I_theta(
            scenario,
            source_override=Z_int,
            n_samples=mc_samples,
            seed=int(mc_seeds[i]),
        )
        I_mc[i] = r_mc["I_theta"]
        I_mc_se[i] = r_mc["standard_error"]

    out_path = out_dir / "train_inconsistency.npz"
    np.savez_compressed(
        out_path,
        theta=theta_samples,
        param_names=np.array(PARAM_NAMES),
        I_theta_mc=I_mc,
        I_theta_mc_se=I_mc_se,
        I_theta_aabb=I_aabb,
        source_center=src_centers,
        source_generators=src_generators,
        source_n_generators=src_n_gens,
        mc_samples_per_eval=np.int32(mc_samples),
        n_samples=np.int32(n_samples),
    )

    dt = time.perf_counter() - t0

    # Print diagnostics
    rho = (
        float(np.corrcoef(I_mc, I_aabb)[0, 1])
        if np.std(I_mc) > 1e-12 and np.std(I_aabb) > 1e-12
        else 0.0
    )
    print(f"    -> {out_path}  ({dt:.1f}s)")
    print(f"    I_MC   : mean={I_mc.mean():.4f}, std={I_mc.std():.4f}, "
          f"range=[{I_mc.min():.4f}, {I_mc.max():.4f}]")
    print(f"    I_AABB : mean={I_aabb.mean():.4f}, std={I_aabb.std():.4f}")
    print(f"    rho(MC, AABB) = {rho:.4f}")
    print(f"    mean MC SE    = {I_mc_se.mean():.4f}")


# ---------------------------------------------------------------------------
# Task & scenario parsing
# ---------------------------------------------------------------------------

PRETRAIN_TASKS = {
    "pretrain_volume",
    "pretrain_containment",
    "pretrain_pairwise_aabb",
    "pretrain_affine_map",
}
ALL_TASKS = PRETRAIN_TASKS | {"train"}


def parse_tasks(task_args: List[str]) -> Set[str]:
    """Expand task aliases ('all', 'pretrain') into concrete task names."""
    tasks: Set[str] = set()
    for t in task_args:
        if t == "all":
            tasks |= ALL_TASKS
        elif t == "pretrain":
            tasks |= PRETRAIN_TASKS
        elif t in ALL_TASKS:
            tasks.add(t)
        else:
            valid = sorted(ALL_TASKS) + ["all", "pretrain"]
            raise ValueError(f"Unknown task '{t}'. Valid: {valid}")
    return tasks


def parse_scenarios(
    scenario_arg: str, all_scenarios: List[Scenario],
) -> List[Scenario]:
    """Parse --scenario: 'all' or integer index 1-12."""
    if scenario_arg == "all":
        return list(all_scenarios)
    idx = int(scenario_arg)
    if not 1 <= idx <= len(all_scenarios):
        raise ValueError(
            f"Scenario index must be 1-{len(all_scenarios)}, got {idx}"
        )
    return [all_scenarios[idx - 1]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate training data for the learned inconsistency surrogate.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  %(prog)s --tasks all --scenario all
  %(prog)s --tasks train --scenario 4 --n_train 10000
  %(prog)s --tasks pretrain
  %(prog)s --tasks pretrain_volume pretrain_pairwise_aabb
  %(prog)s --tasks train --scenario 8 --n_train 20000 --mc_samples 5000
        """,
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["all"],
        help="Tasks: all | pretrain | train | pretrain_volume | "
             "pretrain_containment | pretrain_pairwise_aabb | "
             "pretrain_affine_map  (can combine multiple)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        help="Scenario index 1-12 or 'all' (default: all). "
             "Only affects 'train' task.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(
            Path(__file__).resolve().parent.parent.parent / "data" / "surrogate"
        ),
        help="Root output directory (default: <project>/data/surrogate/)",
    )
    parser.add_argument(
        "--n_pretrain",
        type=int,
        default=50_000,
        help="Number of samples for each pretraining task (default: 50000)",
    )
    parser.add_argument(
        "--n_train",
        type=int,
        default=5_000,
        help="Training samples per scenario (default: 5000)",
    )
    parser.add_argument(
        "--mc_samples",
        type=int,
        default=2000,
        help="MC samples per I(theta) evaluation (default: 2000)",
    )
    parser.add_argument(
        "--n_containment_points",
        type=int,
        default=50,
        help="Test points per zonotope for containment task (default: 50)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing data files (default: skip)",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    tasks = parse_tasks(args.tasks)
    all_scenarios = create_convide_scenarios()
    selected_scenarios = parse_scenarios(args.scenario, all_scenarios)

    # ── Banner ──
    print(f"\n{'=' * 60}")
    print("SURROGATE TRAINING DATA GENERATION")
    print(f"{'=' * 60}")
    print(f"Output dir     : {output_dir}")
    print(f"Tasks          : {sorted(tasks)}")
    if "train" in tasks:
        print(f"Scenarios      : {[s.name for s in selected_scenarios]}")
        print(f"n_train/scen   : {args.n_train}")
        print(f"MC samples/eval: {args.mc_samples}")
    if tasks & PRETRAIN_TASKS:
        print(f"n_pretrain     : {args.n_pretrain}")
    print(f"Seed           : {args.seed}")
    print(f"{'=' * 60}")

    # ── Save global metadata ──
    output_dir.mkdir(parents=True, exist_ok=True)
    global_meta = {
        "tasks": sorted(tasks),
        "scenarios": [s.name for s in selected_scenarios],
        "n_pretrain": args.n_pretrain,
        "n_train": args.n_train,
        "mc_samples": args.mc_samples,
        "n_containment_points": args.n_containment_points,
        "seed": args.seed,
        "d_max": D_MAX,
        "p_max": P_MAX,
        "param_names": PARAM_NAMES,
        "param_bounds": PARAM_BOUNDS,
    }
    with open(output_dir / "meta.json", "w") as fh:
        json.dump(global_meta, fh, indent=2)

    t_total = time.perf_counter()

    # ── Helper: skip if exists ──
    def should_run(task: str, path: Path) -> bool:
        if task not in tasks:
            return False
        if path.exists() and not args.overwrite:
            print(f"\n  [{task}] Skipping -- {path} exists (use --overwrite)")
            return False
        return True

    # ── Pretraining tasks (scenario-independent) ──

    pretrain_dir = output_dir / "pretrain"

    if should_run("pretrain_volume", pretrain_dir / "volume.npz"):
        generate_pretrain_volume(
            pretrain_dir / "volume.npz",
            args.n_pretrain,
            seed=args.seed,
        )

    if should_run("pretrain_containment", pretrain_dir / "containment.npz"):
        generate_pretrain_containment(
            pretrain_dir / "containment.npz",
            args.n_pretrain,
            args.n_containment_points,
            seed=args.seed + 1,
        )

    if should_run("pretrain_pairwise_aabb", pretrain_dir / "pairwise_aabb.npz"):
        generate_pretrain_pairwise_aabb(
            pretrain_dir / "pairwise_aabb.npz",
            args.n_pretrain,
            seed=args.seed + 2,
        )

    if should_run("pretrain_affine_map", pretrain_dir / "affine_map.npz"):
        generate_pretrain_affine_map(
            pretrain_dir / "affine_map.npz",
            args.n_pretrain,
            seed=args.seed + 3,
        )

    # ── Training tasks (per-scenario) ──

    if "train" in tasks:
        for i, scenario in enumerate(selected_scenarios):
            dir_name = scenario_dir_name(scenario)
            path = output_dir / dir_name / "train_inconsistency.npz"
            if path.exists() and not args.overwrite:
                print(
                    f"\n  [train] Skipping {scenario.name} -- "
                    f"exists (use --overwrite)"
                )
                continue
            generate_train_inconsistency(
                scenario,
                output_dir,
                args.n_train,
                args.mc_samples,
                seed=args.seed + 100 + i,
            )

    # ── Done ──
    dt_total = time.perf_counter() - t_total
    print(f"\n{'=' * 60}")
    print(f"Done.  Total time: {dt_total:.1f}s")
    print(f"Output: {output_dir}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
