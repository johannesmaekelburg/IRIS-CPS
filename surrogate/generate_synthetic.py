"""
Generate synthetic training data for the zonotope overlap surrogate.

Creates scenarios with controlled diversity in:
  - Dimension (2D, 3D, 4D)
  - Generator structure (diagonal, near-diagonal, dense)
  - Generator count (d to ~8)
  - Affine mapping (identity, near-identity, random)
  - Overlap regime (high, partial, low)

Output format matches results_scenario_*.json from the MATLAB pipeline,
so existing dataset loaders work without changes.

Usage:
    python -m surrogate.generate_synthetic                     # defaults (~160 scenarios)
    python -m surrogate.generate_synthetic --n_workers 8       # parallel
    python -m surrogate.generate_synthetic --n_samples 5000    # fewer per scenario
    python -m surrogate.generate_synthetic --dry_run            # just print configs
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.optimize import linprog
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "synthetic_v6"

# Intervention parameter ranges (matching real MATLAB data)
THETA_BOUNDS = {
    "scale_factor": (0.1, 3.0),
    "center_delta": (-1.0, 1.0),
    "correlation_strength": (0.0, 0.95),
}


# ═══════════════════════════════════════════════════════════════════════
# Zonotope operations (self-contained, no cross-package imports)
# ═══════════════════════════════════════════════════════════════════════

def interval_bounds(c, G):
    """AABB of zonotope Z = {c + G·ξ : ξ ∈ [-1,1]^p}."""
    abs_sum = np.abs(G).sum(axis=1)
    return c - abs_sum, c + abs_sum


def aabb_jaccard(c1, G1, c2, G2):
    lo1, hi1 = interval_bounds(c1, G1)
    lo2, hi2 = interval_bounds(c2, G2)
    inter_lo = np.maximum(lo1, lo2)
    inter_hi = np.minimum(hi1, hi2)
    inter_w = np.maximum(0.0, inter_hi - inter_lo)
    union_lo = np.minimum(lo1, lo2)
    union_hi = np.maximum(hi1, hi2)
    union_w = union_hi - union_lo
    if np.any(union_w <= 0):
        return 0.0
    vol_inter = float(np.prod(inter_w))
    vol_union = float(np.prod(union_w))
    return vol_inter / vol_union if vol_union > 0 else 0.0


def contains_point_lp(c, G, x):
    """Check x ∈ Z via LP feasibility."""
    _, p = G.shape
    res = linprog(
        c=np.zeros(p), A_eq=G, b_eq=x - c,
        bounds=[(-1.0, 1.0)] * p,
        method="highs",
        options={"presolve": True, "time_limit": 1.0},
    )
    return res.success and res.status == 0


def mc_inconsistency(c_src, G_src, c_tgt, G_tgt, n_samples, rng):
    """Estimate I(θ) = 1 - P(ξ : src(ξ) ∈ tgt) via Monte Carlo."""
    p = G_src.shape[1]
    xi = rng.uniform(-1, 1, size=(n_samples, p))
    points = c_src[None, :] + xi @ G_src.T

    # Bounding box pre-filter
    lo, hi = interval_bounds(c_tgt, G_tgt)
    inside_box = np.all((points >= lo) & (points <= hi), axis=1)

    n_consistent = 0
    for i in np.where(inside_box)[0]:
        if contains_point_lp(c_tgt, G_tgt, points[i]):
            n_consistent += 1

    p_con = n_consistent / n_samples
    se = np.sqrt(p_con * (1 - p_con) / n_samples) if n_samples > 0 else 0.0
    return 1.0 - p_con, se, p_con


# ═══════════════════════════════════════════════════════════════════════
# Intervention (Givens rotation, matching MATLAB pipeline)
# ═══════════════════════════════════════════════════════════════════════

def apply_intervention(c, G, sf, cd, cs):
    """Apply (scale, shift, Givens rotation) to zonotope.

    Matches the MATLAB two-step engine behavior:
      1. Scale all generators by sf
      2. Shift center proportional to total generator span
      3. Givens-rotate 2nd generator toward 1st by cs
    """
    G_scaled = sf * G
    c_new = c + cd * G_scaled.sum(axis=1)

    G_new = G_scaled.copy()
    if G_new.shape[1] >= 2:
        g0 = G_scaled[:, 0]
        g1 = G_scaled[:, 1]
        ccos = np.sqrt(max(0.0, 1.0 - cs * cs))
        G_new[:, 1] = cs * g0 + ccos * g1

    return c_new, G_new


# ═══════════════════════════════════════════════════════════════════════
# Scenario geometry generation
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ScenarioConfig:
    idx: int
    dim: int
    gen_type: str       # "diagonal", "near_diagonal", "dense"
    n_gens_src: int
    n_gens_tgt: int
    f_type: str         # "identity", "near_identity", "random"
    overlap_regime: str  # "high", "partial", "low"
    seed: int


def create_generators(rng, dim, n_gens, gen_type):
    """Create a generator matrix G of shape (dim, n_gens).

    Calibrated to real CONVIDE 3D/4D data: near-diagonal with
    diag_ratio≈0.93, gen_scale p50≈2.2, p95≈6.5.
    """
    scales = np.exp(rng.uniform(np.log(0.4), np.log(7.0), size=dim))

    if gen_type == "diagonal":
        G = np.zeros((dim, n_gens))
        for i in range(min(dim, n_gens)):
            G[i, i] = scales[i]
        for j in range(dim, n_gens):
            G[rng.integers(dim), j] = np.exp(rng.uniform(np.log(0.4), np.log(7.0)))
    elif gen_type == "near_diagonal":
        G = np.zeros((dim, n_gens))
        for i in range(min(dim, n_gens)):
            G[i, i] = scales[i]
        noise = rng.normal(0, 0.1, size=(dim, n_gens)) * scales[:, None]
        np.fill_diagonal(noise, 0.0)
        G += noise
        for j in range(dim, n_gens):
            G[:, j] = rng.normal(0, 0.15, size=dim) * scales
    else:  # dense
        G = rng.normal(0, 1, size=(dim, n_gens)) * scales[:, None]

    return G


def create_mapping(rng, dim, f_type):
    """Create affine mapping F (dim×dim), f (dim,)."""
    if f_type == "identity":
        return np.eye(dim), np.zeros(dim)
    elif f_type == "near_identity":
        F = np.eye(dim) + rng.normal(0, 0.15, size=(dim, dim))
        f = rng.uniform(-2, 2, size=dim)
        return F, f
    else:  # random
        F = np.eye(dim) + rng.normal(0, 0.5, size=(dim, dim))
        f = rng.uniform(-5, 5, size=dim)
        return F, f


def create_scenario_geometry(rng, cfg: ScenarioConfig):
    """Create source zonotope, target zonotope, and affine mapping.

    Uses binary search on AABB Jaccard to place the target center.
    Per-axis offset ratios don't work because AABB Jaccard is a product
    across dimensions — small per-axis offsets compound to near-zero
    overlap in 3D/4D.

    For "high" overlap, target generators are scaled to match propagated
    source magnitudes per axis, ensuring the max AABB Jaccard (at zero
    offset) is high enough for the binary search to find a solution.
    """
    d = cfg.dim
    center_scale = rng.uniform(10, 100)

    c_src = rng.uniform(-center_scale, center_scale, size=d)
    G_src = create_generators(rng, d, cfg.n_gens_src, cfg.gen_type)

    F, f_vec = create_mapping(rng, d, cfg.f_type)

    # Propagated source at neutral θ
    c_prop = F @ c_src + f_vec
    G_prop = F @ G_src

    # ── Target generators: derived from propagated source ──
    # Real CONVIDE data: target aligned with source, width_ratio mean≈1.5 std≈0.85
    n_tgt = cfg.n_gens_tgt
    G_tgt = G_prop[:, :n_tgt].copy()

    # Per-row scale centered at 1.0 — interventions (mean sf≈1.55) produce
    # the observed width_ratio distribution (mean≈1.5, std≈0.85)
    row_scale = np.exp(rng.normal(0.0, 0.3, size=(d, 1)))
    row_scale = np.clip(row_scale, 0.3, 3.0)
    G_tgt = G_tgt * row_scale
    # Off-diagonal noise matching real diag_ratio≈0.93 (7% off-diagonal energy)
    noise_scale = np.linalg.norm(G_tgt, axis=1, keepdims=True).clip(min=1e-10)
    G_tgt += rng.normal(0, 0.08, size=G_tgt.shape) * noise_scale

    # ── Max AABB Jaccard at zero offset ──
    j_max = aabb_jaccard(c_prop, G_prop, c_prop, G_tgt)

    if j_max < 1e-6:
        c_tgt = c_prop.copy()
        return c_src, G_src, c_tgt, G_tgt, F, f_vec

    # ── Target AABB Jaccard ──
    if cfg.overlap_regime == "high":
        j_target = rng.uniform(0.4 * j_max, 0.9 * j_max)
    elif cfg.overlap_regime == "partial":
        j_target = rng.uniform(0.05 * j_max, 0.4 * j_max)
    else:  # low
        j_target = rng.uniform(0.0, max(0.05 * j_max, 0.005))

    # ── Binary search for target center offset ──
    direction = rng.standard_normal(d)
    direction /= np.linalg.norm(direction) + 1e-10
    total_spread = np.abs(G_prop).sum(axis=1) + np.abs(G_tgt).sum(axis=1)

    lo, hi = 0.0, 3.0
    for _ in range(60):
        mid = (lo + hi) / 2
        c_try = c_prop + mid * direction * total_spread
        j = aabb_jaccard(c_prop, G_prop, c_try, G_tgt)
        if j > j_target:
            lo = mid
        else:
            hi = mid
        if abs(j - j_target) < 0.005:
            break

    c_tgt = c_prop + (lo + hi) / 2 * direction * total_spread
    return c_src, G_src, c_tgt, G_tgt, F, f_vec


# ═══════════════════════════════════════════════════════════════════════
# Worker: generate one scenario
# ═══════════════════════════════════════════════════════════════════════

def _balance_experiments(experiments, rng, n_bins=10):
    """Downsample overrepresented I(θ) bins to flatten the distribution.

    Caps each bin at the median of bins [0, 0.9). This specifically
    targets the [0.9, 1.0) tail that dominates in 3D/4D scenarios.
    """
    i_vals = np.array([e["post_state"]["inconsistency"]["I_theta"]
                       for e in experiments])
    bin_edges = np.linspace(0, 1.0001, n_bins + 1)
    bin_idx = np.digitize(i_vals, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    bin_counts = np.bincount(bin_idx, minlength=n_bins)
    # Cap at median of non-tail bins [0, 0.9)
    non_tail = bin_counts[:n_bins - 1]
    nonempty = non_tail[non_tail > 0]
    if len(nonempty) == 0:
        return experiments

    cap = max(int(np.median(nonempty)), 50)

    keep = []
    for b in range(n_bins):
        members = np.where(bin_idx == b)[0]
        if len(members) <= cap:
            keep.extend(members.tolist())
        else:
            keep.extend(rng.choice(members, size=cap, replace=False).tolist())

    keep.sort()
    return [experiments[i] for i in keep]


def generate_one_scenario(args):
    """Generate all samples for one synthetic scenario. Returns path."""
    cfg, n_samples, mc_samples, output_dir, show_progress, balance = args

    rng = np.random.default_rng(cfg.seed)
    c_src, G_src, c_tgt, G_tgt, F, f_vec = create_scenario_geometry(rng, cfg)

    # Sample intervention parameters.
    # For high-overlap scenarios, 30% of samples use near-neutral θ
    # (sf≈1, cd≈0, cs≈0) to ensure low-I coverage.
    if cfg.overlap_regime == "high":
        n_near = int(0.3 * n_samples)
        n_wide = n_samples - n_near
        thetas_wide = np.column_stack([
            rng.uniform(*THETA_BOUNDS["scale_factor"], size=n_wide),
            rng.uniform(*THETA_BOUNDS["center_delta"], size=n_wide),
            rng.uniform(*THETA_BOUNDS["correlation_strength"], size=n_wide),
        ])
        thetas_near = np.column_stack([
            rng.uniform(0.5, 1.5, size=n_near),
            rng.uniform(-0.1, 0.1, size=n_near),
            rng.uniform(0.0, 0.3, size=n_near),
        ])
        thetas = np.vstack([thetas_near, thetas_wide])
        rng.shuffle(thetas)
    else:
        thetas = np.column_stack([
            rng.uniform(*THETA_BOUNDS["scale_factor"], size=n_samples),
            rng.uniform(*THETA_BOUNDS["center_delta"], size=n_samples),
            rng.uniform(*THETA_BOUNDS["correlation_strength"], size=n_samples),
        ])

    mc_seeds = rng.integers(0, 2**31, size=n_samples)
    experiments = []

    iterator = range(n_samples)
    if show_progress:
        iterator = tqdm(iterator, desc=f"  S{cfg.idx:03d} ({cfg.dim}D {cfg.gen_type})",
                        unit="sample", leave=False)

    for i in iterator:
        sf, cd, cs = thetas[i]

        # Apply intervention to source
        c_int, G_int = apply_intervention(c_src, G_src, sf, cd, cs)

        # Propagate through affine mapping
        c_prop = F @ c_int + f_vec
        G_prop = F @ G_int

        # Compute I(θ) via MC
        sample_rng = np.random.default_rng(int(mc_seeds[i]))
        i_theta, se, p_con = mc_inconsistency(
            c_prop, G_prop, c_tgt, G_tgt, mc_samples, sample_rng)

        # Compute AABB Jaccard
        j_aabb = aabb_jaccard(c_prop, G_prop, c_tgt, G_tgt)

        experiments.append({
            "exp_id": i + 1,
            "scale_factor": float(sf),
            "center_delta": float(cd),
            "correlation_strength": float(cs),
            "post_state": {
                "uncertainty": {
                    "source_center": c_prop.tolist(),
                    "source_generators": G_prop.tolist(),
                    "source_n_generators": int(G_prop.shape[1]),
                    "source_volume": float(np.prod(2 * np.abs(G_prop).sum(axis=1))),
                    "source_radius": float(np.max(np.abs(G_prop).sum(axis=1))),
                    "source_correlation": 0.0,
                    "target_center": c_tgt.tolist(),
                    "target_generators": G_tgt.tolist(),
                    "target_n_generators": int(G_tgt.shape[1]),
                    "target_volume": float(np.prod(2 * np.abs(G_tgt).sum(axis=1))),
                    "target_radius": float(np.max(np.abs(G_tgt).sum(axis=1))),
                },
                "inconsistency": {
                    "I_theta": float(i_theta),
                    "I_theta_p_consistent": float(p_con),
                    "I_theta_standard_error": float(se),
                    "jaccard_index": float(j_aabb),
                    "jaccard_inconsistency": float(1.0 - j_aabb),
                },
            },
        })

    # ── Balance: cap overrepresented I bins (skip 4D — high I is realistic) ──
    if balance and cfg.dim < 4:
        experiments = _balance_experiments(experiments, rng)

    # Save JSON
    out_path = Path(output_dir) / f"results_scenario_{cfg.idx}.json"
    scenario_data = {
        "scenario_name": f"synthetic_{cfg.dim}d_{cfg.idx}",
        "scenario_description": (
            f"Synthetic {cfg.dim}D scenario: {cfg.gen_type} generators, "
            f"{cfg.f_type} mapping, {cfg.overlap_regime} overlap"
        ),
        "synthetic": True,
        "dim": cfg.dim,
        "gen_type": cfg.gen_type,
        "n_gens_src": cfg.n_gens_src,
        "n_gens_tgt": cfg.n_gens_tgt,
        "f_type": cfg.f_type,
        "overlap_regime": cfg.overlap_regime,
        "n_samples": len(experiments),
        "n_samples_before_balance": n_samples,
        "mc_samples": mc_samples,
        "experiments": experiments,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(scenario_data, fh)

    # Quick stats
    i_vals = [e["post_state"]["inconsistency"]["I_theta"] for e in experiments]
    return (cfg.idx, cfg.dim, cfg.gen_type, cfg.overlap_regime,
            len(experiments), np.mean(i_vals), np.std(i_vals), str(out_path))


# ═══════════════════════════════════════════════════════════════════════
# Scenario config generation
# ═══════════════════════════════════════════════════════════════════════

def build_scenario_configs(
    n_2d: int, n_3d: int, n_4d: int, base_seed: int = 1000,
) -> List[ScenarioConfig]:
    """Build a list of scenario configs with controlled diversity."""
    rng = np.random.default_rng(base_seed)
    configs = []
    idx = 1

    # Real CONVIDE 3D/4D: diag_ratio=0.93 → mostly near_diagonal
    gen_types = ["near_diagonal"] * 70 + ["diagonal"] * 25 + ["dense"] * 5
    # Real data: almost all identity UPR
    f_types = ["identity"] * 90 + ["near_identity"] * 10
    overlap_regimes = ["high"] * 50 + ["partial"] * 30 + ["low"] * 20

    # 4D real data has I concentrated in [0.85, 1.0] — more partial/low overlap
    overlap_regimes_4d = ["high"] * 20 + ["partial"] * 40 + ["low"] * 40

    for dim, count in [(2, n_2d), (3, n_3d), (4, n_4d)]:
        for _ in range(count):
            gen_type = rng.choice(gen_types)
            f_type = rng.choice(f_types)
            overlap = rng.choice(overlap_regimes_4d if dim == 4 else overlap_regimes)

            # Real 3D/4D: n_gens == dim always
            n_gens_src = dim
            n_gens_tgt = dim

            configs.append(ScenarioConfig(
                idx=idx, dim=dim, gen_type=gen_type,
                n_gens_src=int(n_gens_src), n_gens_tgt=int(n_gens_tgt),
                f_type=f_type, overlap_regime=overlap,
                seed=base_seed + idx,
            ))
            idx += 1

    return configs


def _global_balance(output_dir, n_bins=10, seed=42, skip_dims=None):
    """Prune all scenario files so the [0.9, 1.0) tail matches the rest.

    Computes a global cap from the median of [0, 0.9) bin counts, then
    randomly prunes experiments from [0.9, 1.0) across all files.
    Files whose "dim" is in skip_dims are left untouched.
    """
    skip_dims = skip_dims or set()
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)
    files = sorted(output_dir.glob("results_scenario_*.json"))

    # Pass 1: collect all labels (only from non-skipped dims)
    entries = []  # (I_theta, file_idx, exp_idx)
    file_data = []
    skipped_files = set()
    for fi, jf in enumerate(files):
        with open(jf) as f:
            data = json.load(f)
        file_data.append((jf, data))
        if data.get("dim", 0) in skip_dims:
            skipped_files.add(fi)
            continue
        for ei, exp in enumerate(data["experiments"]):
            val = exp["post_state"]["inconsistency"]["I_theta"]
            entries.append((val, fi, ei))

    if not entries:
        return

    labels = np.array([x[0] for x in entries])
    bin_edges = np.linspace(0, 1.0001, n_bins + 1)
    counts = np.histogram(labels, bins=bin_edges)[0]

    non_tail = counts[:n_bins - 1]
    nonempty = non_tail[non_tail > 0]
    if len(nonempty) == 0:
        return
    cap = int(np.median(nonempty))

    tail_count = counts[n_bins - 1]
    if tail_count <= cap:
        return

    # Pick which tail entries to keep
    tail_indices = np.where(labels >= 0.9)[0]
    keep_tail = set(rng.choice(tail_indices, size=cap, replace=False).tolist())

    # Build per-file keep sets
    keep_per_file = {}
    for idx, (val, fi, ei) in enumerate(entries):
        if val < 0.9 or idx in keep_tail:
            keep_per_file.setdefault(fi, set()).add(ei)

    # Pass 2: rewrite files (skip files in skip_dims)
    for fi, (jf, data) in enumerate(file_data):
        if fi in skipped_files:
            continue
        keep_set = keep_per_file.get(fi, set())
        orig_n = len(data["experiments"])
        data["experiments"] = [exp for ei, exp in enumerate(data["experiments"])
                               if ei in keep_set]
        data["n_samples"] = len(data["experiments"])
        if len(data["experiments"]) < orig_n:
            with open(jf, "w") as f:
                json.dump(data, f)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic zonotope training data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n_2d", type=int, default=10)
    parser.add_argument("--n_3d", type=int, default=75)
    parser.add_argument("--n_4d", type=int, default=75)
    parser.add_argument("--n_samples", type=int, default=10000,
                        help="Intervention samples per scenario")
    parser.add_argument("--mc_samples", type=int, default=1000,
                        help="MC samples per I(theta) evaluation")
    parser.add_argument("--n_workers", type=int, default=1,
                        help="Parallel workers (1 = sequential with per-sample progress)")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--balance", action="store_true", default=True,
                        help="Cap overrepresented I(theta) bins per scenario (default: on)")
    parser.add_argument("--no_balance", action="store_false", dest="balance",
                        help="Disable I(theta) balancing")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print scenario configs without generating data")
    args = parser.parse_args()

    configs = build_scenario_configs(args.n_2d, args.n_3d, args.n_4d, args.seed)
    output_dir = Path(args.output_dir)

    # ── Banner ──
    total = len(configs)
    total_evals = total * args.n_samples
    print(f"\n{'='*60}")
    print("SYNTHETIC DATA GENERATION")
    print(f"{'='*60}")
    print(f"  Scenarios:    {args.n_2d} × 2D + {args.n_3d} × 3D + {args.n_4d} × 4D = {total}")
    print(f"  Samples/scen: {args.n_samples:,}")
    print(f"  MC/eval:      {args.mc_samples}")
    print(f"  Total evals:  {total_evals:,}")
    print(f"  Workers:      {args.n_workers}")
    print(f"  Balance:      {args.balance}")
    print(f"  Output:       {output_dir}")
    print(f"{'='*60}")

    # ── Dimension / type breakdown ──
    from collections import Counter
    dim_counts = Counter(c.dim for c in configs)
    gen_counts = Counter(c.gen_type for c in configs)
    f_counts = Counter(c.f_type for c in configs)
    ov_counts = Counter(c.overlap_regime for c in configs)
    print(f"\n  Dimensions:  {dict(sorted(dim_counts.items()))}")
    print(f"  Gen types:   {dict(gen_counts)}")
    print(f"  F types:     {dict(f_counts)}")
    print(f"  Overlap:     {dict(ov_counts)}")

    if args.dry_run:
        print(f"\n{'Idx':>4s} {'Dim':>3s} {'GenType':<14s} {'nG_s':>4s} {'nG_t':>4s} "
              f"{'FType':<14s} {'Overlap':<8s}")
        print("-" * 60)
        for c in configs:
            print(f"{c.idx:>4d} {c.dim:>3d} {c.gen_type:<14s} {c.n_gens_src:>4d} "
                  f"{c.n_gens_tgt:>4d} {c.f_type:<14s} {c.overlap_regime:<8s}")
        return

    # ── Skip existing ──
    if not args.overwrite:
        remaining = []
        for c in configs:
            path = output_dir / f"results_scenario_{c.idx}.json"
            if path.exists():
                continue
            remaining.append(c)
        if len(remaining) < len(configs):
            print(f"\n  Skipping {len(configs) - len(remaining)} existing scenarios "
                  f"(use --overwrite to regenerate)")
        configs = remaining

    if not configs:
        print("\n  Nothing to generate.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Generating {len(configs)} scenarios...\n")
    t0 = time.perf_counter()

    # ── Generate ──
    show_progress = args.n_workers <= 1
    worker_args = [
        (cfg, args.n_samples, args.mc_samples, str(output_dir),
         show_progress, args.balance)
        for cfg in configs
    ]

    results = []
    if args.n_workers <= 1:
        for wa in worker_args:
            results.append(generate_one_scenario(wa))
    else:
        with mp.Pool(processes=args.n_workers) as pool:
            for r in tqdm(pool.imap_unordered(generate_one_scenario, worker_args),
                          total=len(worker_args), desc="  scenarios", unit="scen"):
                results.append(r)

    dt = time.perf_counter() - t0

    # ── Global balancing across all scenarios ──
    if args.balance:
        print("\n  Running global I(θ) balancing...")
        total_before_global = sum(r[4] for r in results)
        _global_balance(output_dir, seed=args.seed, skip_dims={4})
        # Recount
        total_after_global = 0
        results_updated = []
        for jf in sorted(output_dir.glob("results_scenario_*.json")):
            with open(jf) as f:
                data = json.load(f)
            exps = data["experiments"]
            i_vals = [e["post_state"]["inconsistency"]["I_theta"] for e in exps]
            idx = data.get("scenario_name", jf.stem).split("_")[-1]
            results_updated.append((
                int(jf.stem.split("_")[-1]), data.get("dim", 0),
                data.get("gen_type", "?"), data.get("overlap_regime", "?"),
                len(exps), np.mean(i_vals), np.std(i_vals), str(jf),
            ))
        results = sorted(results_updated)
        total_after_global = sum(r[4] for r in results)
        print(f"  Global balance: {total_before_global:,} → {total_after_global:,} samples")

    # ── Summary ──
    total_samples = sum(r[4] for r in results)
    print(f"\n{'='*65}")
    print("DONE")
    print(f"{'='*65}")
    print(f"  Generated {len(results)} scenarios, {total_samples:,} total samples "
          f"in {dt:.0f}s ({dt/60:.1f} min)")
    if args.balance:
        print(f"  (balanced from {len(results) * args.n_samples:,} raw samples)")
    print(f"  Output: {output_dir}")

    print(f"\n{'Idx':>4s} {'Dim':>3s} {'GenType':<14s} {'Overlap':<8s} "
          f"{'N':>6s} {'I mean':>7s} {'I std':>7s}")
    print("-" * 65)
    for idx, dim, gt, ov, n, imean, istd, path in sorted(results):
        print(f"{idx:>4d} {dim:>3d} {gt:<14s} {ov:<8s} "
              f"{n:>6d} {imean:>7.3f} {istd:>7.3f}")

    # ── Save generation metadata ──
    meta = {
        "n_2d": args.n_2d, "n_3d": args.n_3d, "n_4d": args.n_4d,
        "n_samples": args.n_samples, "mc_samples": args.mc_samples,
        "seed": args.seed, "total_scenarios": len(results),
        "generation_time_s": dt,
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nTo train with synthetic data, add to train_compare.py DATA_DIRS:")
    print(f'    ROOT / "data" / "synthetic_v6",')


if __name__ == "__main__":
    main()
