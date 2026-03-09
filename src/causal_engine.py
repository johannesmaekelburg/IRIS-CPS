"""
Causal Experiment Engine — Pure Python replacement for the MATLAB pipeline.

Implements the full path from zonotope representation through interventions,
uncertainty propagation, Monte Carlo consistency scoring, and Saltelli-based
Sobol sensitivity analysis of the global inconsistency I(θ).

Matches the semantics of:
  - causal_experiment_engine.m  (interventions, measurement, scenarios)
  - global_inconsistency.m      (I(θ) = 1 − P(consistent))

Mathematical reference (paper §III):
  θ  = [s_u, Δc_u, R_u]  — exogenous intervention vector
  S(θ) = E_ξ[C(ξ,θ)]     — global consistency metric  (Def. 8, Eq. 1)
  I(θ) = 1 − S(θ)         — global inconsistency

Usage:
    python causal_engine.py                     # default demo
    python causal_engine.py --mc_samples 5000   # higher accuracy
    python causal_engine.py --saltelli_N 1024   # more Sobol samples
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.optimize import linprog

try:
    from SALib.analyze import sobol as sobol_analyzer
    try:
        from SALib.sample import sobol as saltelli_sampler
    except ImportError:
        from SALib.sample import saltelli as saltelli_sampler
    SALIB_AVAILABLE = True
except ImportError:
    SALIB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Zonotope representation
# ---------------------------------------------------------------------------

@dataclass
class Zonotope:
    """Zonotope Z = {c + G·ξ : ξ ∈ [-1,1]^p}.

    For constrained polynomial zonotopes with E = I and no constraints
    (A, b, EC all empty), the representation collapses to a standard
    zonotope.  This is the case for every scenario in the current pipeline.
    """

    c: np.ndarray          # center  (n,)
    G: np.ndarray          # generators  (n, p)

    def __post_init__(self):
        self.c = np.asarray(self.c, dtype=float)
        self.G = np.asarray(self.G, dtype=float)
        if self.c.ndim != 1:
            raise ValueError(f"center must be 1-D, got shape {self.c.shape}")
        if self.G.ndim != 2:
            raise ValueError(f"generators must be 2-D, got shape {self.G.shape}")
        if self.c.shape[0] != self.G.shape[0]:
            raise ValueError("dimension mismatch between center and generators")

    @property
    def dim(self) -> int:
        return self.c.shape[0]

    @property
    def n_generators(self) -> int:
        return self.G.shape[1]

    def copy(self) -> Zonotope:
        return Zonotope(self.c.copy(), self.G.copy())

    # ---- geometric helpers used by metrics ----

    def interval_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Axis-aligned bounding box [lo, hi] (interval hull)."""
        abs_sum = np.abs(self.G).sum(axis=1)
        return self.c - abs_sum, self.c + abs_sum

    def volume_interval(self) -> float:
        """Volume of the interval over-approximation."""
        lo, hi = self.interval_bounds()
        return float(np.prod(hi - lo))

    def radius(self) -> float:
        """Half-width of the largest bounding-box dimension."""
        abs_sum = np.abs(self.G).sum(axis=1)
        return float(np.max(abs_sum))

    def generator_correlation(self) -> float:
        """Mean absolute cosine similarity between generator columns."""
        if self.n_generators <= 1:
            return 0.0
        norms = np.linalg.norm(self.G, axis=0, keepdims=True)
        norms = np.where(norms < 1e-15, 1.0, norms)
        G_normed = self.G / norms
        C = G_normed.T @ G_normed
        mask = ~np.eye(C.shape[0], dtype=bool)
        return float(np.mean(np.abs(C[mask])))


# ---------------------------------------------------------------------------
# Zonotope operations
# ---------------------------------------------------------------------------

def affine_map(Z: Zonotope, F: np.ndarray, f: np.ndarray) -> Zonotope:
    """Propagate zonotope through affine map x ↦ Fx + f."""
    return Zonotope(F @ Z.c + f, F @ Z.G)


def contains_point(Z: Zonotope, x: np.ndarray) -> bool:
    """Check x ∈ Z via LP:  find ξ ∈ [-1,1]^p s.t. G·ξ = x − c.

    Uses scipy.optimize.linprog with the HiGHS solver.
    """
    n, p = Z.G.shape
    rhs = x - Z.c

    # linprog minimises c^T ξ subject to A_eq ξ = b_eq, bounds on ξ.
    # We only need feasibility, so c=0.
    res = linprog(
        c=np.zeros(p),
        A_eq=Z.G,
        b_eq=rhs,
        bounds=[(-1.0, 1.0)] * p,
        method="highs",
        options={"presolve": True, "time_limit": 1.0},
    )
    return res.success and res.status == 0


def contains_points_batch(Z: Zonotope, X: np.ndarray) -> np.ndarray:
    """Vectorised containment check for X of shape (n_points, dim).

    Falls back to per-point LP but first applies a fast bounding-box
    pre-filter to skip obvious non-members.
    """
    lo, hi = Z.interval_bounds()
    inside_box = np.all((X >= lo) & (X <= hi), axis=1)

    result = np.zeros(X.shape[0], dtype=bool)
    candidate_idx = np.where(inside_box)[0]
    for i in candidate_idx:
        result[i] = contains_point(Z, X[i])
    return result


# ---------------------------------------------------------------------------
# Interventions  (do-operator on θ)
# ---------------------------------------------------------------------------

def apply_intervention(
    Z: Zonotope,
    intervention_type: str,
    params: dict,
) -> Zonotope:
    """Apply a single do(θ ← v) intervention to zonotope Z.

    Supported types (matching paper §III.D and MATLAB engine):
      widen / shrink : do(s_u ← α)            — scale generators by α
      shift          : do(Δc_u ← δ)           — translate center by δ (scalar → uniform, or vector)
      correlate      : do(R_u ← R(β))         — add dependent generator with strength β
      rotate         : do(R_u ← R_matrix)     — rotate generators by given matrix
    """
    Z_new = Z.copy()

    match intervention_type:
        case "widen" | "shrink":
            alpha = params["scale_factor"]
            Z_new.G = Z.G * alpha
        case "shift":
            if "shift_vector" in params:
                delta = np.asarray(params["shift_vector"], dtype=float)
            else:
                delta_scalar = params["center_delta"]
                delta = np.full(Z.dim, delta_scalar)
            Z_new.c = Z.c + delta
        case "correlate":
            beta = params["correlation_strength"]
            new_gen = (Z.G.sum(axis=1, keepdims=True)) * beta
            Z_new.G = np.hstack([Z.G, new_gen])
        case "rotate":
            R = np.asarray(params["rotation_matrix"], dtype=float)
            Z_new.c = R @ Z.c
            Z_new.G = R @ Z.G
        case _:
            raise ValueError(f"Unknown intervention type: {intervention_type}")

    return Z_new


def apply_compound_intervention(
    Z: Zonotope,
    theta: dict[str, float],
) -> Zonotope:
    """Apply all three paper interventions simultaneously.

    Args:
        Z: base zonotope
        theta: dict with keys  scale_factor, center_delta, correlation_strength
               (any key may be absent → no-op for that intervention)
    """
    Z_new = Z.copy()

    if "scale_factor" in theta:
        alpha = theta["scale_factor"]
        Z_new.G = Z_new.G * alpha

    if "center_delta" in theta:
        delta = theta["center_delta"]
        Z_new.c = Z_new.c + delta

    if "correlation_strength" in theta:
        beta = theta["correlation_strength"]
        if beta > 0:
            new_gen = (Z_new.G.sum(axis=1, keepdims=True)) * beta
            Z_new.G = np.hstack([Z_new.G, new_gen])

    return Z_new


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    """MMS scenario: source zonotope, target zonotope, affine UPR mapping."""

    source: Zonotope
    target: Zonotope
    F: np.ndarray      # mapping matrix
    f: np.ndarray      # mapping offset
    name: str = ""
    dim: int = field(init=False)

    def __post_init__(self):
        self.dim = self.source.dim


def create_convide_scenarios() -> List[Scenario]:
    """Reproduce the 12 CONVIDE scenarios from generate_convide_twostep.m.

    All use identity mapping F=I, f=0 (no distortion baseline).
    Zonotopes have diagonal generators and no polynomial / constraint terms.
    """
    scenarios: list[Scenario] = []

    def _s(sid, name, c_src, g_diag_src, c_tgt, g_diag_tgt):
        d = len(c_src)
        return Scenario(
            source=Zonotope(np.array(c_src), np.diag(g_diag_src)),
            target=Zonotope(np.array(c_tgt), np.diag(g_diag_tgt)),
            F=np.eye(d),
            f=np.zeros(d),
            name=f"S{sid}: {name}",
        )

    # 2D
    scenarios.append(_s(1,  "CAD Export Drift",           [100, 50],     [2.0, 2.0],     [100.3, 50.3],     [2.1, 2.1]))
    scenarios.append(_s(2,  "MBSE Version Mismatch",      [50, 25],      [1.5, 1.5],     [50, 25],          [1.8, 1.8]))
    scenarios.append(_s(3,  "Documentation Sync",         [75, 40],      [2.0, 2.0],     [75.5, 40.5],      [2.0, 2.0]))
    scenarios.append(_s(4,  "Control Design Conflict",    [60, 30],      [1.8, 1.8],     [60.8, 30.8],      [1.5, 1.5]))

    # 3D
    scenarios.append(_s(5,  "Sensor Calibration Drift",   [100, 50, 25], [2.5, 2.5, 1.2], [100.5, 50.5, 25.5], [2.6, 2.6, 1.3]))
    scenarios.append(_s(6,  "Requirements Ambiguity",     [80, 40, 20],  [2.0, 2.0, 1.0], [80, 40, 20],         [2.5, 2.5, 1.2]))
    scenarios.append(_s(7,  "Test Config Mismatch",       [70, 35, 18],  [2.2, 2.2, 1.1], [70.4, 35.4, 18.4],   [2.2, 2.2, 1.1]))
    scenarios.append(_s(8,  "Simulation Numerical Error",  [90, 45, 22], [2.5, 2.5, 1.3], [90.6, 45.6, 22.6],   [2.2, 2.2, 1.1]))

    # 4D
    scenarios.append(_s(9,  "Multi-Physics Coupling",     [100, 50, 25, 12], [3.0, 3.0, 1.5, 0.75], [100.5, 50.5, 25.5, 12.5], [3.1, 3.1, 1.6, 0.8]))
    scenarios.append(_s(10, "Interface Spec Gap",          [85, 42, 21, 10],  [2.5, 2.5, 1.2, 0.6],  [85, 42, 21, 10],          [3.0, 3.0, 1.5, 0.75]))
    scenarios.append(_s(11, "Parameter Estimation Bias",   [90, 45, 22, 11],  [2.8, 2.8, 1.4, 0.7],  [90.5, 45.5, 22.5, 11.5], [2.9, 2.9, 1.5, 0.75]))
    scenarios.append(_s(12, "Cross-Domain Integration",    [95, 47, 24, 12],  [3.0, 3.0, 1.5, 0.75], [95.8, 47.8, 24.8, 12.8], [2.5, 2.5, 1.2, 0.6]))

    return scenarios


# ---------------------------------------------------------------------------
# Global inconsistency  I(θ)
# ---------------------------------------------------------------------------

def mc_consistency_probability(
    models: List[Zonotope],
    n_samples: int = 2000,
    seed: Optional[int] = None,
) -> Dict:
    """Estimate P(consistent) = P(ξ : x(ξ) ∈ ⋂ Z_i) via Monte Carlo.

    Samples ξ ~ U([-1,1]^p) from the first model's generator space,
    computes x = c + G·ξ, and checks containment in all other models.

    Returns dict with p_consistent, I_theta, standard_error, ci95, etc.
    """
    rng = np.random.default_rng(seed)
    Z_source = models[0]

    p = Z_source.n_generators
    xi = rng.uniform(-1.0, 1.0, size=(n_samples, p))
    points = Z_source.c[None, :] + xi @ Z_source.G.T   # (n_samples, dim)

    consistent = np.ones(n_samples, dtype=bool)
    for Z_other in models[1:]:
        consistent &= contains_points_batch(Z_other, points)

    n_consistent = int(consistent.sum())
    p_hat = n_consistent / n_samples

    se = np.sqrt(p_hat * (1 - p_hat) / n_samples) if n_samples > 0 else 0.0
    ci_lo = max(0.0, p_hat - 1.96 * se)
    ci_hi = min(1.0, p_hat + 1.96 * se)

    I_theta = 1.0 - p_hat

    return {
        "p_consistent": p_hat,
        "I_theta": I_theta,
        "n_samples": n_samples,
        "n_consistent": n_consistent,
        "standard_error": se,
        "I_theta_ci95_lower": 1.0 - ci_hi,
        "I_theta_ci95_upper": 1.0 - ci_lo,
    }


def compute_I_theta(
    scenario: Scenario,
    source_override: Optional[Zonotope] = None,
    n_samples: int = 2000,
    seed: Optional[int] = None,
) -> Dict:
    """Compute I(θ) for a scenario.

    Propagates the source through the UPR mapping and scores
    consistency against the target.
    """
    source = source_override if source_override is not None else scenario.source
    Z_prop = affine_map(source, scenario.F, scenario.f)
    return mc_consistency_probability([Z_prop, scenario.target], n_samples, seed)


# ---------------------------------------------------------------------------
# Saltelli sampling  +  Sobol analysis
# ---------------------------------------------------------------------------

PARAM_NAMES = ["scale_factor", "center_delta", "correlation_strength"]
PARAM_BOUNDS = [
    [0.5, 5.0],       # scale_factor   (paper: α ∈ R>0)
    [0.0, 0.2],       # center_delta   (paper: δ ∈ R, here scalar added to each dim)
    [0.0, 0.95],      # correlation_strength (paper: β ∈ [0,1])
]
PARAM_BASELINES = {
    "scale_factor": 1.0,
    "center_delta": 0.0,
    "correlation_strength": 0.0,
}


def make_salib_problem(
    param_names: Optional[List[str]] = None,
    param_bounds: Optional[List[List[float]]] = None,
) -> dict:
    names = param_names or PARAM_NAMES
    bounds = param_bounds or PARAM_BOUNDS
    return {
        "num_vars": len(names),
        "names": names,
        "bounds": bounds,
    }


def generate_saltelli_samples(
    problem: dict,
    N: int = 512,
    calc_second_order: bool = False,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate Saltelli quasi-random samples for Sobol analysis.

    Returns array of shape (N*(2p+2), p)  [or N*(p+2) if not second-order].
    """
    if not SALIB_AVAILABLE:
        raise ImportError("SALib required. Install: pip install SALib")
    # SALib ≥ 1.4 accepts seed kwarg (some versions use 'rng')
    try:
        samples = saltelli_sampler.sample(problem, N,
                                          calc_second_order=calc_second_order,
                                          seed=seed)
    except TypeError:
        samples = saltelli_sampler.sample(problem, N,
                                          calc_second_order=calc_second_order)
    return samples


def evaluate_samples(
    scenario: Scenario,
    samples: np.ndarray,
    param_names: List[str],
    mc_samples: int = 2000,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> np.ndarray:
    """Evaluate I(θ) for each row of the Saltelli sample matrix.

    Args:
        scenario:    base scenario (interventions applied to its source)
        samples:     (N_total, n_params) from generate_saltelli_samples
        param_names: column names matching samples columns
        mc_samples:  MC samples per I(θ) evaluation
        seed:        base seed (incremented per sample for independence)
        verbose:     print progress

    Returns:
        Y: array of shape (N_total,) with I(θ) values
    """
    N_total = samples.shape[0]
    Y = np.empty(N_total)

    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31, size=N_total)

    t0 = time.time()
    for i in range(N_total):
        theta = dict(zip(param_names, samples[i]))
        Z_intervened = apply_compound_intervention(scenario.source, theta)
        result = compute_I_theta(scenario, source_override=Z_intervened,
                                 n_samples=mc_samples, seed=int(seeds[i]))
        Y[i] = result["I_theta"]

        if verbose and (i + 1) % max(1, N_total // 20) == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (N_total - i - 1) / rate
            print(f"  [{i+1:>6d}/{N_total}]  I(θ)={Y[i]:.4f}  "
                  f"({rate:.1f} eval/s, ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    if verbose:
        print(f"  Completed {N_total} evaluations in {elapsed:.1f}s "
              f"({N_total/elapsed:.1f} eval/s)")
    return Y


def run_sobol_analysis(
    scenario: Scenario,
    N: int = 512,
    mc_samples: int = 2000,
    calc_second_order: bool = False,
    seed: int = 42,
    verbose: bool = True,
    param_names: Optional[List[str]] = None,
    param_bounds: Optional[List[List[float]]] = None,
) -> Dict:
    """End-to-end: Saltelli sampling → I(θ) evaluation → Sobol indices.

    No surrogate model — evaluates the true I(θ) at every sample point.
    """
    if not SALIB_AVAILABLE:
        raise ImportError("SALib required. Install: pip install SALib")

    problem = make_salib_problem(param_names, param_bounds)
    p = problem["num_vars"]
    n_eval = N * (2 * p + 2) if calc_second_order else N * (p + 2)

    if verbose:
        print(f"\n{'='*70}")
        print(f"SOBOL SENSITIVITY ANALYSIS  (direct, no surrogate)")
        print(f"{'='*70}")
        print(f"Scenario       : {scenario.name}")
        print(f"Parameters     : {problem['names']}")
        print(f"Bounds         : {problem['bounds']}")
        print(f"Saltelli N     : {N}  →  {n_eval} total evaluations")
        print(f"MC samples/eval: {mc_samples}")
        print(f"Second-order   : {calc_second_order}")
        print(f"{'='*70}\n")

    # 1. Generate samples
    samples = generate_saltelli_samples(problem, N, calc_second_order, seed)
    assert samples.shape == (n_eval, p), f"Expected ({n_eval}, {p}), got {samples.shape}"

    # 2. Evaluate I(θ) at every sample
    Y = evaluate_samples(scenario, samples, problem["names"],
                         mc_samples=mc_samples, seed=seed, verbose=verbose)

    # 3. Sobol decomposition
    Si = sobol_analyzer.analyze(problem, Y, calc_second_order=calc_second_order)

    results = {
        "problem": problem,
        "N": N,
        "n_evaluations": n_eval,
        "mc_samples_per_eval": mc_samples,
        "Y_mean": float(Y.mean()),
        "Y_std": float(Y.std()),
        "Y_min": float(Y.min()),
        "Y_max": float(Y.max()),
        "first_order": dict(zip(problem["names"], Si["S1"].tolist())),
        "first_order_conf": dict(zip(problem["names"], Si["S1_conf"].tolist())),
        "total_order": dict(zip(problem["names"], Si["ST"].tolist())),
        "total_order_conf": dict(zip(problem["names"], Si["ST_conf"].tolist())),
    }

    if verbose:
        print(f"\n{'='*70}")
        print("SOBOL INDICES")
        print(f"{'='*70}")
        print(f"I(θ) distribution: mean={Y.mean():.4f}, std={Y.std():.4f}, "
              f"range=[{Y.min():.4f}, {Y.max():.4f}]")
        print(f"\n{'Parameter':<25s} {'S1':>8s} {'±conf':>8s}  {'ST':>8s} {'±conf':>8s}")
        print("-" * 60)
        for name in problem["names"]:
            s1 = results["first_order"][name]
            s1c = results["first_order_conf"][name]
            st = results["total_order"][name]
            stc = results["total_order_conf"][name]
            print(f"{name:<25s} {s1:>8.4f} {s1c:>8.4f}  {st:>8.4f} {stc:>8.4f}")
        print(f"{'='*70}\n")

    return results


# ---------------------------------------------------------------------------
# Single-intervention sweep (reproduces MATLAB run_intervention_sweep)
# ---------------------------------------------------------------------------

def run_intervention_sweep(
    scenario: Scenario,
    intervention_type: str,
    param_name: str,
    param_values: List[float],
    mc_samples: int = 2000,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> List[Dict]:
    """Sweep a single parameter and record pre/post I(θ)."""
    results = []
    pre = compute_I_theta(scenario, n_samples=mc_samples, seed=seed)

    for val in param_values:
        params = {param_name: val}
        Z_mod = apply_intervention(scenario.source, intervention_type, params)
        post = compute_I_theta(scenario, source_override=Z_mod,
                               n_samples=mc_samples, seed=seed)
        results.append({
            "intervention_type": intervention_type,
            "param_name": param_name,
            "param_value": float(val),
            "I_theta_pre": pre["I_theta"],
            "I_theta_post": post["I_theta"],
            "delta_I_theta": post["I_theta"] - pre["I_theta"],
            "I_theta_se": post["standard_error"],
            "I_theta_ci95_lower": post["I_theta_ci95_lower"],
            "I_theta_ci95_upper": post["I_theta_ci95_upper"],
            "n_mc_samples": mc_samples,
        })
        if verbose:
            print(f"  {intervention_type}({param_name}={val:>8.3f})  "
                  f"I(θ)={post['I_theta']:.4f} ± {post['standard_error']:.4f}  "
                  f"Δ={post['I_theta'] - pre['I_theta']:+.4f}")
    return results


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def export_results(results: dict, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

PARAM_LABELS = {
    "scale_factor": r"Scale $s_u$",
    "center_delta": r"Shift $\Delta c_u$",
    "correlation_strength": r"Correlation $\beta$",
}


def plot_sobol_indices(
    sobol_results: Dict,
    scenario_name: str = "",
    output_path: Optional[str | Path] = None,
    figsize: Tuple[float, float] = (7, 4.5),
) -> plt.Figure:
    """Publication-quality grouped bar chart of first-order and total Sobol indices."""
    names = list(sobol_results["first_order"].keys())
    S1 = np.array([sobol_results["first_order"][n] for n in names])
    ST = np.array([sobol_results["total_order"][n] for n in names])
    S1_conf = np.array([sobol_results["first_order_conf"][n] for n in names])
    ST_conf = np.array([sobol_results["total_order_conf"][n] for n in names])

    labels = [PARAM_LABELS.get(n, n) for n in names]

    x = np.arange(len(names))
    width = 0.32

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    bars1 = ax.bar(x - width / 2, S1, width, yerr=S1_conf, capsize=4,
                   color="#4878CF", edgecolor="white", linewidth=0.6,
                   label=r"$S_i$ (first-order)", zorder=3)
    bars2 = ax.bar(x + width / 2, ST, width, yerr=ST_conf, capsize=4,
                   color="#D65F5F", edgecolor="white", linewidth=0.6,
                   label=r"$S_i^T$ (total-effect)", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Sobol Index", fontsize=12)
    ax.set_ylim(bottom=0, top=max(1.05, (ST + ST_conf).max() * 1.15))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))

    ax.axhline(1.0, color="grey", linewidth=0.6, linestyle="--", zorder=1)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.9)

    title = "Variance-Based Sensitivity (Sobol Indices)"
    if scenario_name:
        title += f"\n{scenario_name}"
    ax.set_title(title, fontsize=13, fontweight="bold")

    # Value annotations on top of bars
    for bar_group, vals in [(bars1, S1), (bars2, ST)]:
        for bar, v in zip(bar_group, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8.5,
                    fontweight="medium")

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {output_path}")

    return fig


def plot_sweeps(
    sweep_results: Dict[str, List[Dict]],
    scenario_name: str = "",
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Four-panel plot of intervention sweeps with I(θ) response curves."""
    sweep_order = ["widen", "shrink", "shift", "correlate"]
    available = [s for s in sweep_order if s in sweep_results]
    n = len(available)
    if n == 0:
        return None

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, stype in zip(axes, available):
        data = sweep_results[stype]
        vals = [d["param_value"] for d in data]
        I_post = [d["I_theta_post"] for d in data]
        se = [d["I_theta_se"] for d in data]
        baseline = data[0]["I_theta_pre"]

        ax.errorbar(vals, I_post, yerr=se, fmt="o-", capsize=3,
                    color="#4878CF", markersize=5, linewidth=1.5, zorder=3)
        ax.axhline(baseline, color="grey", linewidth=0.8, linestyle="--",
                   label=f"baseline={baseline:.3f}", zorder=1)
        ax.set_xlabel(PARAM_LABELS.get(data[0]["param_name"],
                                        data[0]["param_name"]), fontsize=10)
        ax.set_ylabel(r"$I(\theta)$", fontsize=11)
        ax.set_title(stype.capitalize(), fontsize=11, fontweight="bold")
        ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(f"Intervention Sweeps — {scenario_name}" if scenario_name
                 else "Intervention Sweeps", fontsize=13, fontweight="bold")

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {output_path}")

    return fig


# ---------------------------------------------------------------------------
# Main: demo pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Causal Engine — zonotope interventions + Sobol analysis"
    )
    parser.add_argument("--mc_samples", type=int, default=2000,
                        help="MC samples per I(θ) evaluation (default: 2000)")
    parser.add_argument("--saltelli_N", type=int, default=256,
                        help="Saltelli base sample count (default: 256)")
    parser.add_argument("--scenario", type=int, default=4, choices=range(1, 13),
                        help="Scenario index 1-12 (default: 4)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: results/causal_engine_<ts>)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sweep_only", action="store_true",
                        help="Run single-param sweeps only, skip Sobol")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else \
        Path("results") / f"causal_engine_{timestamp}"

    scenarios = create_convide_scenarios()
    scenario = scenarios[args.scenario - 1]

    print(f"\n{'='*70}")
    print(f"CAUSAL ENGINE — Python replacement")
    print(f"{'='*70}")
    print(f"Scenario       : {scenario.name}  (dim={scenario.dim})")
    print(f"Source center   : {scenario.source.c}")
    print(f"Target center   : {scenario.target.c}")
    print(f"MC samples/eval : {args.mc_samples}")
    print(f"Seed            : {args.seed}")
    print(f"Output          : {out_dir}")
    print(f"{'='*70}\n")

    all_results: dict = {
        "timestamp": timestamp,
        "scenario": scenario.name,
        "dim": scenario.dim,
        "mc_samples": args.mc_samples,
        "seed": args.seed,
    }

    # --- Baseline I(θ) ---
    print("Computing baseline I(θ)...")
    baseline = compute_I_theta(scenario, n_samples=args.mc_samples, seed=args.seed)
    print(f"  Baseline I(θ) = {baseline['I_theta']:.4f} "
          f"± {baseline['standard_error']:.4f}\n")
    all_results["baseline"] = baseline

    # --- Single-parameter sweeps ---
    sweeps_config = [
        ("widen",     "scale_factor",          [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]),
        ("shrink",    "scale_factor",          [0.01, 0.1, 0.5, 1.0, 2.0]),
        ("shift",     "center_delta",          [0.0, 0.01, 0.05, 0.1, 0.2]),
        ("correlate", "correlation_strength",  [0.0, 0.3, 0.6, 0.8, 0.9, 0.95]),
    ]

    sweep_results = {}
    for itype, pname, pvals in sweeps_config:
        print(f"\n--- Sweep: {itype} ({pname}) ---")
        sweep_results[itype] = run_intervention_sweep(
            scenario, itype, pname, pvals,
            mc_samples=args.mc_samples, seed=args.seed,
        )
    all_results["sweeps"] = sweep_results

    # --- Sobol analysis ---
    sobol_results = None
    if not args.sweep_only:
        if not SALIB_AVAILABLE:
            print("\nSALib not installed — skipping Sobol analysis.")
        else:
            sobol_results = run_sobol_analysis(
                scenario,
                N=args.saltelli_N,
                mc_samples=args.mc_samples,
                seed=args.seed,
                verbose=True,
            )
            all_results["sobol"] = sobol_results

    # --- Plots ---
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_sweeps(sweep_results, scenario_name=scenario.name,
                output_path=out_dir / "intervention_sweeps.png")

    if sobol_results is not None:
        plot_sobol_indices(sobol_results, scenario_name=scenario.name,
                           output_path=out_dir / "sobol_indices.png")

    plt.close("all")

    # --- Export ---
    export_results(all_results, out_dir / "causal_engine_results.json")

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
