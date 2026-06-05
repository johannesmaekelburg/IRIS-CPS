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
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm
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


def zonotopes_intersect(Z1: Zonotope, Z2: Zonotope) -> bool:
    """Return True iff two zonotopes have a non-empty intersection.

    This solves the feasibility problem

        Z1.c + Z1.G * xi1 = Z2.c + Z2.G * xi2,
        xi1 in [-1, 1]^{p1}, xi2 in [-1, 1]^{p2}

    via a linear program.  It matches the exact intersection-gating semantics
    used by the MATLAB MFMC implementation.
    """
    p1 = Z1.n_generators
    p2 = Z2.n_generators
    A_eq = np.hstack([Z1.G, -Z2.G])
    b_eq = Z2.c - Z1.c
    res = linprog(
        c=np.zeros(p1 + p2),
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=[(-1.0, 1.0)] * (p1 + p2),
        method="highs",
        options={"presolve": True, "time_limit": 1.0},
    )
    return bool(res.success and res.status == 0)


# ---------------------------------------------------------------------------
# Alternative scoring methods  (AABB Jaccard, MC Jaccard)
# ---------------------------------------------------------------------------

def aabb_jaccard(Z1: Zonotope, Z2: Zonotope) -> float:
    """Axis-Aligned Bounding Box Jaccard index between two zonotopes.

    Over-approximates each zonotope by its interval hull and computes the
    closed-form Jaccard index of the resulting boxes:

        J_AABB = ∏_j max(0, min(b1_j,b2_j) - max(a1_j,a2_j))
               / ∏_j (max(b1_j,b2_j) - min(a1_j,a2_j))

    Cost: O(d·γ) — essentially instant.
    Returns value in [0, 1].  Biased upward (over-estimates overlap, hence
    under-estimates inconsistency).
    """
    lo1, hi1 = Z1.interval_bounds()
    lo2, hi2 = Z2.interval_bounds()

    # Intersection box
    inter_lo = np.maximum(lo1, lo2)
    inter_hi = np.minimum(hi1, hi2)
    inter_widths = np.maximum(0.0, inter_hi - inter_lo)

    # Union box
    union_lo = np.minimum(lo1, lo2)
    union_hi = np.maximum(hi1, hi2)
    union_widths = union_hi - union_lo

    if np.any(union_widths <= 0):
        return 0.0

    vol_inter = float(np.prod(inter_widths))
    vol_union = float(np.prod(union_widths))

    return vol_inter / vol_union if vol_union > 0 else 0.0


def mc_jaccard(
    Z1: Zonotope,
    Z2: Zonotope,
    n_samples: int = 2000,
    seed: Optional[int] = None,
) -> Dict:
    """Monte Carlo estimate of the Jaccard index between two zonotopes.

    Samples uniformly from the union bounding box and estimates
    J(Z1, Z2) = vol(Z1 ∩ Z2) / vol(Z1 ∪ Z2).

    Cost: O(N·d·γ) — each point requires membership LP.
    """
    rng = np.random.default_rng(seed)

    lo1, hi1 = Z1.interval_bounds()
    lo2, hi2 = Z2.interval_bounds()
    lo = np.minimum(lo1, lo2)
    hi = np.maximum(hi1, hi2)

    d = len(lo)
    points = rng.uniform(lo, hi, size=(n_samples, d))

    in_Z1 = contains_points_batch(Z1, points)
    in_Z2 = contains_points_batch(Z2, points)

    n_inter = int((in_Z1 & in_Z2).sum())
    n_union = int((in_Z1 | in_Z2).sum())

    jaccard = n_inter / n_union if n_union > 0 else 0.0
    se = np.sqrt(jaccard * (1 - jaccard) / max(1, n_union))

    return {
        "jaccard": jaccard,
        "I_theta": 1.0 - jaccard,
        "n_samples": n_samples,
        "n_intersection": n_inter,
        "n_union": n_union,
        "standard_error": se,
    }


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
    metadata: Dict[str, Any] = field(default_factory=dict)
    dim: int = field(init=False)

    def __post_init__(self):
        self.dim = self.source.dim


def create_engineering_scenarios() -> List[Scenario]:
    """Reproduce the 12 Engineering scenarios from generate_engineering_twostep.m.

    Each scenario has a physically motivated affine UPR mapping (F, f)
    representing sensor calibration, coordinate transforms, or unit conversions.
    Zonotopes have diagonal generators; no polynomial / constraint terms.

    UPR mapping types used:
      - Identity (F=I, f=0): direct constraint check
      - Diagonal scaling (F=diag(s), f=0): per-axis gain factor
      - Diagonal + offset (F=diag(s), f≠0): sensor bias + gain
      - Rotation (2D: 5°; 3D: 5° in xy; 4D: 5° in xy & yz): coordinate transform
    """
    scenarios: list[Scenario] = []

    def _rot2(deg: float) -> np.ndarray:
        θ = np.radians(deg)
        return np.array([[np.cos(θ), -np.sin(θ)],
                         [np.sin(θ),  np.cos(θ)]])

    def _rot3(deg_xy: float) -> np.ndarray:
        θ = np.radians(deg_xy)
        R = np.eye(3)
        R[:2, :2] = [[np.cos(θ), -np.sin(θ)], [np.sin(θ), np.cos(θ)]]
        return R

    def _rot4(deg_xy: float, deg_yz: float) -> np.ndarray:
        R = np.eye(4)
        θ1 = np.radians(deg_xy)
        R[0, 0] = np.cos(θ1); R[0, 1] = -np.sin(θ1)
        R[1, 0] = np.sin(θ1); R[1, 1] =  np.cos(θ1)
        θ2 = np.radians(deg_yz)
        Ry = np.eye(4)
        Ry[1, 1] = np.cos(θ2); Ry[1, 2] = -np.sin(θ2)
        Ry[2, 1] = np.sin(θ2); Ry[2, 2] =  np.cos(θ2)
        return Ry @ R

    def _s(sid, name, c_src, g_diag_src, c_tgt, g_diag_tgt, F=None, f=None,
           upr_type="parametric"):
        d = len(c_src)
        F_ = np.eye(d) if F is None else np.asarray(F, dtype=float)
        f_ = np.zeros(d) if f is None else np.asarray(f, dtype=float)
        return Scenario(
            source=Zonotope(np.array(c_src, dtype=float), np.diag(g_diag_src)),
            target=Zonotope(np.array(c_tgt, dtype=float), np.diag(g_diag_tgt)),
            F=F_,
            f=f_,
            name=f"S{sid}: {name}",
            metadata={"upr_type": upr_type, "dataset_kind": "engineering"},
        )

    # ── 2-D scenarios ─────────────────────────────────────────────────────────
    # S1: identity — direct CAD tolerance check
    scenarios.append(_s(1,  "CAD Export Drift",
        [100, 50], [2.0, 2.0], [100.3, 50.3], [2.1, 2.1],
        upr_type="identity"))

    # S2: parametric — tool/sensor axis gain (x: 0.97, y: 1.04)
    scenarios.append(_s(2,  "MBSE Version Mismatch",
        [50, 25], [1.5, 1.5], [50, 25], [1.8, 1.8],
        F=np.diag([0.97, 1.04]), f=np.zeros(2),
        upr_type="parametric"))

    # S3: structural — measurement bias as sum of tool + systematic offset components
    scenarios.append(_s(3,  "Documentation Sync",
        [75, 40], [2.0, 2.0], [75.5, 40.5], [2.0, 2.0],
        F=np.diag([1.0, 1.0]), f=np.array([0.5, 0.3]),
        upr_type="structural"))

    # S4: parametric — 5° coordinate frame rotation
    scenarios.append(_s(4,  "Control Design Conflict",
        [60, 30], [1.8, 1.8], [60.8, 30.8], [1.5, 1.5],
        F=_rot2(5.0), f=np.zeros(2),
        upr_type="parametric"))

    # ── 3-D scenarios ─────────────────────────────────────────────────────────
    # S5: identity_bidir — sensor couples both model outputs (bidirectional)
    scenarios.append(_s(5,  "Sensor Calibration Drift",
        [100, 50, 25], [2.5, 2.5, 1.2], [100.5, 50.5, 25.5], [2.6, 2.6, 1.3],
        upr_type="identity_bidir"))

    # S6: structural — per-axis sensor gain decomposed into axis-independent components
    scenarios.append(_s(6,  "Requirements Ambiguity",
        [80, 40, 20], [2.0, 2.0, 1.0], [80, 40, 20], [2.5, 2.5, 1.2],
        F=np.diag([0.95, 1.05, 0.98]), f=np.zeros(3),
        upr_type="structural"))

    # S7: guarded — bias offset fires only when configuration guard is active
    scenarios.append(_s(7,  "Test Config Mismatch",
        [70, 35, 18], [2.2, 2.2, 1.1], [70.4, 35.4, 18.4], [2.2, 2.2, 1.1],
        F=np.diag([1.0, 1.0, 1.0]), f=np.array([0.3, -0.2, 0.1]),
        upr_type="guarded"))

    # S8: parametric — 5° attitude rotation
    scenarios.append(_s(8,  "Simulation Numerical Error",
        [90, 45, 22], [2.5, 2.5, 1.3], [90.6, 45.6, 22.6], [2.2, 2.2, 1.1],
        F=_rot3(5.0), f=np.zeros(3),
        upr_type="parametric"))

    # ── 4-D scenarios ─────────────────────────────────────────────────────────
    # S9: identity — multi-physics direct coupling
    scenarios.append(_s(9,  "Multi-Physics Coupling",
        [100, 50, 25, 12], [3.0, 3.0, 1.5, 0.75],
        [100.5, 50.5, 25.5, 12.5], [3.1, 3.1, 1.6, 0.8],
        upr_type="identity"))

    # S10: disambiguation — interface specification has multiple candidate gain versions
    scenarios.append(_s(10, "Interface Spec Gap",
        [85, 42, 21, 10], [2.5, 2.5, 1.2, 0.6],
        [85, 42, 21, 10], [3.0, 3.0, 1.5, 0.75],
        F=np.diag([0.96, 1.03, 0.99, 1.02]), f=np.zeros(4),
        upr_type="disambiguation"))

    # S11: constraint_based — parameter offset represents constraint boundary shift
    scenarios.append(_s(11, "Parameter Estimation Bias",
        [90, 45, 22, 11], [2.8, 2.8, 1.4, 0.7],
        [90.5, 45.5, 22.5, 11.5], [2.9, 2.9, 1.5, 0.75],
        F=np.diag([1.0, 1.0, 1.0, 1.0]), f=np.array([0.4, -0.3, 0.2, -0.1]),
        upr_type="constraint_based"))

    # S12: parametric — 4-D coupled rotation (cross-domain coordinate transform)
    scenarios.append(_s(12, "Cross-Domain Integration",
        [95, 47, 24, 12], [3.0, 3.0, 1.5, 0.75],
        [95.8, 47.8, 24.8, 12.8], [2.5, 2.5, 1.2, 0.6],
        F=_rot4(5.0, 3.0), f=np.zeros(4),
        upr_type="parametric"))

    return scenarios


def _extract_real_scalar_zonotope(element: dict) -> tuple[float, float]:
    """Return (center, scalar generator) from a 1-D real constrained zonotope."""
    zono = element.get("uncertainty", {}).get("zonotope")
    if not isinstance(zono, dict):
        raise ValueError(f"Element '{element.get('id')}' has no zonotope entry.")
    if zono.get("type") != "constrained_zonotope" or zono.get("domain") != "real":
        raise ValueError(
            f"Element '{element.get('id')}' is not a real constrained zonotope."
        )

    center = float(zono["center"])
    generators = zono.get("G", [])
    if not generators:
        raise ValueError(f"Element '{element.get('id')}' has empty generator list.")
    return center, float(generators[0])


def _extract_scalar_zonotope_flexible(element: dict) -> tuple[float, float]:
    """Return (center, |generator|) from a 1-D zonotope element.

    More permissive than _extract_real_scalar_zonotope — accepts any
    zonotope type / domain as long as 'center' and 'G' are present.
    Used for coupled 2-element entries whose domain field is not 'real'.
    """
    zono = element.get("uncertainty", {}).get("zonotope")
    if not isinstance(zono, dict):
        raise ValueError(f"Element '{element.get('id')}' has no zonotope entry.")
    center = float(zono["center"])
    generators = zono.get("G", [])
    if not generators:
        raise ValueError(f"Element '{element.get('id')}' has empty generator list.")
    return center, abs(float(generators[0]))


def _derive_target_from_fixed_value(
    m2_element: dict, src_g: float, operator: str
) -> tuple[float, float]:
    """Synthesise a target zonotope from a fixed_value threshold (1-D only).

    Kept for reference but superseded by _derive_target_nd for multi-dim
    scenarios.  Uses 25 % overlap rule.
    """
    fv = float(m2_element["fixed_value"])
    tgt_g = abs(src_g)
    if operator == ">=":
        tgt_c = fv + 0.25 * tgt_g
    else:
        tgt_c = fv - 0.25 * tgt_g
    return tgt_c, tgt_g


def _derive_target_nd(
    src_c: float, src_g: float, n_dims: int, dim_index: int, operator: str
) -> tuple[float, float]:
    """Synthesise a target zonotope dimension targeting I ≈ 0.5 for an n-D scenario.

    The 25 % overlap rule (designed for 1-D) gives P(consistent per dim) ≈ 0.75,
    so joint P for n=2 is only 0.56 at best — but with real dataset geometries
    the actual joint P collapses to near zero (I ≈ 1.0).

    This rule instead targets P(consistent per dim) = 0.5^(1/n), so the
    joint probability equals 0.5 → I ≈ 0.5 at the base geometry:

      P_per_dim = 0.5^(1/n_dims)          e.g. 0.707 for n=2
      offset    = 2 * src_g * (1 − P)     e.g. 0.586 * src_g for n=2

    The target is shifted by `offset` away from the source centre in the
    direction that creates partial overlap:

      ≤  (source should be below target):  tgt_c = src_c + offset
      ≥  (source should be above target):  tgt_c = src_c − offset

    Physical fixed_value thresholds are intentionally ignored — only source
    geometry is used — because fv and src are often in different units for
    parametric UPR entries.

    Returns (tgt_center, tgt_generator).  tgt_g = src_g (equal width).
    """
    p_per_dim = 0.5 ** (1.0 / max(n_dims, 1))
    offset = 2.0 * abs(src_g) * (1.0 - p_per_dim)
    tgt_g = abs(src_g)
    if operator == ">=":
        tgt_c = src_c - offset
    else:  # <= or anything else
        tgt_c = src_c + offset
    return tgt_c, tgt_g


def _build_coupled_2d_scenario(
    entry: dict,
    domain_label: str,
    json_path: "Path",
    cps_idx: int,
) -> "Optional[Scenario]":
    """Build a 2-D Scenario from a coupled 2-element M1 dataset entry.

    These are the 9th entries in each domain JSON file — physically coupled
    variable pairs (e.g. X/Y positioning errors, temperature+CO2) that
    already represent a 2-D scenario directly, rather than being formed by
    pairing two independent 1-D entries.

    Source zonotope: diagonal 2-D, each element contributes one generator.
    Target zonotope: derived from fixed requirement thresholds via the same
    25%-overlap rule used for the sequential-pair scenarios.

    Handles two target layouts:
      - 2 fixed values + 2 relations: one ≤/≥ constraint per dimension.
      - 1 fixed value  + 1 relation : combined constraint (e.g. radial);
        approximated as an axis-aligned square inscribed in the tolerance.

    Returns None and is silently skipped on any extraction failure.
    """
    try:
        m1 = _find_model(entry, "M1")
        m2 = _find_model(entry, "M2")
    except ValueError:
        return None

    elems_m1 = m1.get("elements", [])
    elems_m2 = m2.get("elements", [])
    crs       = entry.get("consistency_relations", [])

    if len(elems_m1) != 2:
        return None

    # ── Source zonotope ──────────────────────────────────────────────────────
    try:
        src_c1, src_g1 = _extract_scalar_zonotope_flexible(elems_m1[0])
        src_c2, src_g2 = _extract_scalar_zonotope_flexible(elems_m1[1])
    except (ValueError, KeyError, TypeError):
        return None

    source_center     = np.array([src_c1, src_c2], dtype=float)
    source_generators = np.diag([src_g1, src_g2])

    # ── Target zonotope ──────────────────────────────────────────────────────
    # Use the n-D aware overlap rule so that joint P(consistent) ≈ 0.5
    # → I ≈ 0.5 at base geometry, with good variance under interventions.
    # Physical fixed_value thresholds are ignored (they are in different
    # units for parametric UPR entries and cannot be used directly).
    cr0_op = crs[0].get("operator", "<=") if crs else "<="
    cr1_op = crs[1].get("operator", "<=") if len(crs) > 1 else cr0_op

    if len(elems_m1) == 2:
        tc1, tg1 = _derive_target_nd(src_c1, src_g1, n_dims=2, dim_index=0, operator=cr0_op)
        tc2, tg2 = _derive_target_nd(src_c2, src_g2, n_dims=2, dim_index=1, operator=cr1_op)
        target_center     = np.array([tc1, tc2], dtype=float)
        target_generators = np.diag([tg1, tg2])
    else:
        return None  # unsupported layout

    # ── Metadata ─────────────────────────────────────────────────────────────
    name1 = _compact_cps_label(elems_m1[0].get("name", "Dim1"), domain_label)
    name2 = _compact_cps_label(elems_m1[1].get("name", "Dim2"), domain_label)
    scenario_name = f"CPS{cps_idx}: {domain_label} - {name1} & {name2} [coupled]"

    metadata = {
        "dataset_kind":       "cps_uncertainty_dataset",
        "dataset_file":       json_path.name,
        "scenario_id":        entry.get("scenario_id"),
        "scenario_name":      entry.get("name"),
        "coupled":            True,
        "upr_type":           "constraint_based",
        "upr_params":         {"n_relations": len(crs)},
        "domain_label":       domain_label,
        "causality_type":     "constraint_based",
        "relation_types":     [cr.get("upr_type", "constraint_based") for cr in crs],
        "relation_operators": [cr.get("operator", "<=") for cr in crs],
    }

    return Scenario(
        source=Zonotope(source_center, source_generators),
        target=Zonotope(target_center, target_generators),
        F=np.eye(2),
        f=np.zeros(2),
        name=scenario_name,
        metadata=metadata,
    )


def _find_model(scenario: dict, model_id: str) -> dict:
    for model in scenario.get("models", []):
        if model.get("id") == model_id:
            return model
    raise ValueError(f"Scenario '{scenario.get('scenario_id')}' is missing model {model_id}.")


def _compact_cps_label(name: str, domain_label: str) -> str:
    label = name.strip()
    for prefix in (
        f"{domain_label} - ",
        f"{domain_label} ? ",
        f"{domain_label}: ",
    ):
        if label.startswith(prefix):
            return label[len(prefix):].strip()
    return label


def _pair_cps_relation_type(*relations: dict) -> str:
    kinds = [str(rel.get("upr_type", "")).strip() for rel in relations if rel]
    kinds = [kind for kind in kinds if kind]
    if not kinds:
        return "CPS"
    if len(set(kinds)) == 1:
        return kinds[0]
    return "+".join(kinds)


def _load_cps_paired_scenarios_from_dataset() -> List[Scenario]:
    """Build up to 72 2-D CPS scenarios from the raw JSON dataset.

    Per domain (9 domains total), three passes are performed:

    Pass 1 — original 8 single-variable entries (indices 0–7), paired as
      (0,1), (2,3), (4,5), (6,7)  →  4 scenarios per domain = 36 total
      (S13–S48).  M2 zonotopes are pre-computed in the full dataset file.

    Pass 2 — coupled 2-D entry (index 8 in the full file), one per domain
      →  up to 9 more scenarios (S49–S57).  M1 has 2 elements.

    Pass 3 — new single-variable entries (indices 9–14 of the full file),
      paired as (9,10), (11,12), (13,14)  →  3 scenarios per domain = 27
      total (S58–S84).  M2 has only a fixed_value; target zonotope is
      synthesised using the 25 % overlap rule.  Entry 15 is unpaired and
      skipped.

    Index stability:
      The original CPS1–36 (S13–S48) indices are fixed — new passes only
      append to the end of the list.
    """
    project_root = Path(__file__).resolve().parent.parent
    # Pass 1 & 2 use the original dataset (pre-computed M2 zonotopes, stable CPS1–45).
    orig_dir = project_root / "data" / "CPS-uncertainty-dataset"
    files = sorted(orig_dir.glob("*_full.json"))
    if not files:
        raise FileNotFoundError(f"No CPS dataset files found in: {orig_dir}")
    # Pass 3 uses the expanded dataset (new entries 9–14 per domain).
    new_dir = project_root / "data" / "CPS-uncertainty-dataset-full"
    new_files = sorted(new_dir.glob("*_full.json"))

    scenarios: list[Scenario] = []
    cps_idx = 1

    for json_path in files:
        with open(json_path, encoding="utf-8") as fh:
            domain_data = json.load(fh)

        domain_scenarios = []
        for entry in domain_data:
            try:
                m1 = _find_model(entry, "M1")
                m2 = _find_model(entry, "M2")
                if len(m1.get("elements", [])) != 1 or len(m2.get("elements", [])) != 1:
                    continue
                src_c, src_g = _extract_scalar_zonotope_flexible(m1["elements"][0])
            except (ValueError, KeyError, TypeError):
                continue

            m2_elem = m2["elements"][0]
            relation = (entry.get("consistency_relations") or [{}])[0]

            # Use pre-computed M2 zonotope when available; otherwise synthesise
            # from fixed_value using the 25 % overlap rule.
            try:
                tgt_c, tgt_g = _extract_scalar_zonotope_flexible(m2_elem)
            except (ValueError, KeyError, TypeError):
                fv = m2_elem.get("fixed_value")
                if fv is None:
                    continue
                op = relation.get("operator", "<=")
                tgt_c, tgt_g = _derive_target_from_fixed_value(m2_elem, src_g, op)

            domain_scenarios.append({
                "raw": entry,
                "source_center": src_c,
                "source_generator": src_g,
                "target_center": tgt_c,
                "target_generator": tgt_g,
                "relation": relation,
            })

        if len(domain_scenarios) < 8:
            raise ValueError(
                f"Expected at least 8 single-variable scenarios in {json_path.name}, "
                f"found {len(domain_scenarios)}."
            )

        domain_label = str(domain_data[0].get("name", json_path.stem)).split(" - ")[0].strip()
        selected = domain_scenarios[:8]
        for pair_start in range(0, 8, 2):
            left = selected[pair_start]
            right = selected[pair_start + 1]

            source_center = np.array([left["source_center"], right["source_center"]], dtype=float)
            source_generators = np.diag([left["source_generator"], right["source_generator"]])
            target_center = np.array([left["target_center"], right["target_center"]], dtype=float)
            target_generators = np.diag([left["target_generator"], right["target_generator"]])

            # Build block-diagonal F, f and UPR-type metadata from per-relation fields.
            def _mapping_scalars(rel: dict) -> tuple[float, float]:
                m = rel.get("mapping", {})
                return float(m.get("scale", 1.0)), float(m.get("offset", 0.0))

            sl, ol = _mapping_scalars(left["relation"])
            sr, or_ = _mapping_scalars(right["relation"])
            F_2d = np.diag([sl, sr])
            f_2d = np.array([ol, or_])

            # UPR type — both relations in a pair share the same type
            upr_type_l = str(left["relation"].get("upr_type", "parametric")).strip()
            upr_type_r = str(right["relation"].get("upr_type", "parametric")).strip()
            paired_upr_type = (
                upr_type_l if upr_type_l == upr_type_r else f"{upr_type_l}+{upr_type_r}"
            )

            # Type-specific parameters for the 2-D paired scenario
            upr_params: dict = {}

            if "constraint_based" in paired_upr_type:
                # Constraint bounds: [upper_bound_l, upper_bound_r]
                def _constraint_bound(rel: dict) -> float:
                    cp = rel.get("constraint_params", {})
                    return float(cp.get("bound", 1.0))
                upr_params["constraint_bound"] = [
                    _constraint_bound(left["relation"]),
                    _constraint_bound(right["relation"]),
                ]
                upr_params["constraint_direction"] = "leq"

            if "structural" in paired_upr_type:
                upr_params["structural_components_l"] = left["relation"].get("structural_params", {})
                upr_params["structural_components_r"] = right["relation"].get("structural_params", {})

            if "guarded" in paired_upr_type:
                upr_params["guard_params_l"] = left["relation"].get("guard_params", {})
                upr_params["guard_params_r"] = right["relation"].get("guard_params", {})
                upr_params["p_guard"] = float(
                    left["relation"].get("guard_params", {}).get("p_guard", 0.8)
                )

            if "disambiguation" in paired_upr_type:
                upr_params["alternatives_l"] = (
                    left["relation"].get("disambiguation_params", {}).get("alternatives", [])
                )
                upr_params["alternatives_r"] = (
                    right["relation"].get("disambiguation_params", {}).get("alternatives", [])
                )

            if left["relation"].get("bidir") or right["relation"].get("bidir"):
                upr_params["bidir"] = True

            left_name = _compact_cps_label(left["raw"].get("name", "Scenario A"), domain_label)
            right_name = _compact_cps_label(right["raw"].get("name", "Scenario B"), domain_label)
            relation_type = _pair_cps_relation_type(left["relation"], right["relation"])
            scenario_name = f"CPS{cps_idx}: {domain_label} - {left_name} & {right_name}"

            metadata = {
                "dataset_kind": "cps_uncertainty_dataset",
                "dataset_file": json_path.name,
                "paired_scenario_ids": [
                    left["raw"].get("scenario_id"),
                    right["raw"].get("scenario_id"),
                ],
                "paired_scenario_names": [
                    left["raw"].get("name"),
                    right["raw"].get("name"),
                ],
                "consistency_relations": [
                    left["relation"],
                    right["relation"],
                ],
                "relation_types": [
                    left["relation"].get("upr_type"),
                    right["relation"].get("upr_type"),
                ],
                "relation_operators": [
                    left["relation"].get("operator"),
                    right["relation"].get("operator"),
                ],
                "mapping_scales": [sl, sr],
                "mapping_offsets": [ol, or_],
                "domain_label": domain_label,
                "causality_type": relation_type,
                "upr_type": paired_upr_type,
                "upr_params": upr_params,
                "dataset_kind": "cps_uncertainty_dataset",
            }

            scenarios.append(
                Scenario(
                    source=Zonotope(source_center, source_generators),
                    target=Zonotope(target_center, target_generators),
                    F=F_2d,
                    f=f_2d,
                    name=scenario_name,
                    metadata=metadata,
                )
            )
            cps_idx += 1

    # ── Second pass: coupled 2-D entries (index 8 per domain) ────────────────
    # Done after all sequential pairs so that the original CPS1–36 indices
    # (and therefore S13–S48 in create_all_scenarios) remain stable.
    for json_path in files:
        with open(json_path, encoding="utf-8") as fh:
            domain_data = json.load(fh)
        domain_label = str(domain_data[0].get("name", json_path.stem)).split(" - ")[0].strip()

        for entry in domain_data:
            m1_check = next(
                (m for m in entry.get("models", []) if m.get("id") == "M1"), None
            )
            if m1_check is None or len(m1_check.get("elements", [])) != 2:
                continue
            coupled = _build_coupled_2d_scenario(
                entry, domain_label, json_path, cps_idx
            )
            if coupled is not None:
                scenarios.append(coupled)
                cps_idx += 1

    # ── Third pass: new single-variable entries (indices 9–14 per domain) ────
    # The expanded dataset (*_full.json in CPS-uncertainty-dataset-full) contains
    # 7 additional single-variable entries per domain (indices 9–15).  We pair
    # them as (9,10), (11,12), (13,14) → 3 pairs per domain = 27 new scenarios
    # (CPS46–72, S58–S84).  Entry 15 is unpaired and skipped.
    # Target zonotopes are synthesised via _derive_target_from_fixed_value.
    # Skipped entirely if the expanded dataset is not present.
    for json_path in new_files:
        with open(json_path, encoding="utf-8") as fh:
            domain_data = json.load(fh)
        domain_label = str(domain_data[0].get("name", json_path.stem)).split(" - ")[0].strip()

        # Collect only the NEW single-variable entries (indices 9+ in the full
        # dataset).  Entries 0–8 are handled by Pass 1 (old dataset) and Pass 2
        # (coupled entry 8).  We use the list index to skip them unconditionally
        # so that pre-computed M2 zonotopes written by populate_m2_zonotopes.py
        # are correctly read for the new entries without re-processing old ones.
        new_entries: list[dict] = []
        for _entry_idx, entry in enumerate(domain_data):
            if _entry_idx < 9:
                continue  # old entries handled by Pass 1 / Pass 2
            m1 = next((m for m in entry.get("models", []) if m.get("id") == "M1"), None)
            m2 = next((m for m in entry.get("models", []) if m.get("id") == "M2"), None)
            if m1 is None or m2 is None:
                continue
            if len(m1.get("elements", [])) != 1 or len(m2.get("elements", [])) != 1:
                continue
            m2_elem = m2["elements"][0]
            try:
                src_c, src_g = _extract_scalar_zonotope_flexible(m1["elements"][0])
            except (ValueError, KeyError, TypeError):
                continue
            cr = (entry.get("consistency_relations") or [{}])[0]
            op = cr.get("operator", "<=")
            # Use pre-computed M2 zonotope when available (written by
            # populate_m2_zonotopes.py); otherwise synthesise on the fly.
            try:
                tgt_c, tgt_g = _extract_scalar_zonotope_flexible(m2_elem)
            except (ValueError, KeyError, TypeError):
                fv = m2_elem.get("fixed_value")
                if fv is None or not isinstance(fv, (int, float)):
                    continue
                tgt_c, tgt_g = _derive_target_nd(src_c, src_g, n_dims=2, dim_index=0, operator=op)
            new_entries.append({
                "raw": entry,
                "source_center": src_c,
                "source_generator": src_g,
                "target_center": tgt_c,
                "target_generator": tgt_g,
                "relation": cr,
            })

        # Pair sequentially: (0,1), (2,3), (4,5) — skip any trailing unpaired entry.
        for pair_start in range(0, len(new_entries) - 1, 2):
            left = new_entries[pair_start]
            right = new_entries[pair_start + 1]

            source_center = np.array(
                [left["source_center"], right["source_center"]], dtype=float
            )
            source_generators = np.diag(
                [left["source_generator"], right["source_generator"]]
            )
            target_center = np.array(
                [left["target_center"], right["target_center"]], dtype=float
            )
            target_generators = np.diag(
                [left["target_generator"], right["target_generator"]]
            )

            def _mapping_scalars_new(rel: dict) -> tuple[float, float]:
                m = rel.get("mapping", {})
                return float(m.get("scale", 1.0)), float(m.get("offset", 0.0))

            sl, ol = _mapping_scalars_new(left["relation"])
            sr, or_ = _mapping_scalars_new(right["relation"])
            F_2d = np.diag([sl, sr])
            f_2d = np.array([ol, or_])

            upr_type_l = str(left["relation"].get("upr_type", "parametric")).strip()
            upr_type_r = str(right["relation"].get("upr_type", "parametric")).strip()
            paired_upr_type = (
                upr_type_l if upr_type_l == upr_type_r else f"{upr_type_l}+{upr_type_r}"
            )

            left_name = _compact_cps_label(left["raw"].get("name", "Scenario A"), domain_label)
            right_name = _compact_cps_label(right["raw"].get("name", "Scenario B"), domain_label)
            scenario_name = f"CPS{cps_idx}: {domain_label} - {left_name} & {right_name}"

            metadata = {
                "dataset_kind": "cps_uncertainty_dataset",
                "dataset_file": json_path.name,
                "paired_scenario_ids": [
                    left["raw"].get("scenario_id"),
                    right["raw"].get("scenario_id"),
                ],
                "paired_scenario_names": [
                    left["raw"].get("name"),
                    right["raw"].get("name"),
                ],
                "consistency_relations": [
                    left["relation"],
                    right["relation"],
                ],
                "relation_types": [
                    left["relation"].get("upr_type"),
                    right["relation"].get("upr_type"),
                ],
                "relation_operators": [
                    left["relation"].get("operator"),
                    right["relation"].get("operator"),
                ],
                "mapping_scales": [sl, sr],
                "mapping_offsets": [ol, or_],
                "domain_label": domain_label,
                "upr_type": paired_upr_type,
                "upr_params": {},
                "target_derived": True,  # M2 zonotope was synthesised, not pre-computed
            }

            scenarios.append(
                Scenario(
                    source=Zonotope(source_center, source_generators),
                    target=Zonotope(target_center, target_generators),
                    F=F_2d,
                    f=f_2d,
                    name=scenario_name,
                    metadata=metadata,
                )
            )
            cps_idx += 1

    return scenarios


def create_cps_scenarios() -> List[Scenario]:
    """Build up to 72 CPS scenarios: 36 sequential-pair + 9 coupled + 27 new-pair."""
    return _load_cps_paired_scenarios_from_dataset()


def create_all_scenarios() -> List[Scenario]:
    """Return all scenarios: 12 Engineering + up to 72 CPS (up to 84 total)."""
    return create_engineering_scenarios() + create_cps_scenarios()


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


def compute_I_theta_aabb(
    scenario: Scenario,
    source_override: Optional[Zonotope] = None,
) -> Dict:
    """AABB Jaccard inconsistency: I_AABB(θ) = 1 − J_AABB(Z_prop, Z_target).

    Closed-form, no sampling.  Cost: O(d·γ).
    """
    source = source_override if source_override is not None else scenario.source
    Z_prop = affine_map(source, scenario.F, scenario.f)
    j = aabb_jaccard(Z_prop, scenario.target)
    return {
        "I_theta": 1.0 - j,
        "jaccard_aabb": j,
        "standard_error": 0.0,
        "p_consistent": j,
        "n_samples": 0,
    }


def compute_I_theta_mc_jaccard(
    scenario: Scenario,
    source_override: Optional[Zonotope] = None,
    n_samples: int = 2000,
    seed: Optional[int] = None,
) -> Dict:
    """MC Jaccard inconsistency: I_Jac(θ) = 1 − J_MC(Z_prop, Z_target)."""
    source = source_override if source_override is not None else scenario.source
    Z_prop = affine_map(source, scenario.F, scenario.f)
    return mc_jaccard(Z_prop, scenario.target, n_samples, seed)


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
    estimator: str = "mc_probability",
) -> np.ndarray:
    """Evaluate I(θ) for each row of the Saltelli sample matrix.

    Args:
        scenario:    base scenario (interventions applied to its source)
        samples:     (N_total, n_params) from generate_saltelli_samples
        param_names: column names matching samples columns
        mc_samples:  MC samples per I(θ) evaluation
        seed:        base seed (incremented per sample for independence)
        verbose:     print progress
        estimator:   'mc_probability' (default), 'mc_jaccard', or 'aabb_jaccard'

    Returns:
        Y: array of shape (N_total,) with I(θ) values
    """
    N_total = samples.shape[0]
    Y = np.empty(N_total)

    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31, size=N_total)

    desc = f"Saltelli [{estimator}]"
    pbar = tqdm(range(N_total), desc=desc, unit="eval",
                disable=not verbose)
    for i in pbar:
        theta = dict(zip(param_names, samples[i]))
        Z_intervened = apply_compound_intervention(scenario.source, theta)

        if estimator == "mc_probability":
            result = compute_I_theta(scenario, source_override=Z_intervened,
                                     n_samples=mc_samples, seed=int(seeds[i]))
        elif estimator == "mc_jaccard":
            result = compute_I_theta_mc_jaccard(
                scenario, source_override=Z_intervened,
                n_samples=mc_samples, seed=int(seeds[i]))
        elif estimator == "aabb_jaccard":
            result = compute_I_theta_aabb(scenario, source_override=Z_intervened)
        else:
            raise ValueError(f"Unknown estimator: {estimator}")

        Y[i] = result["I_theta"]
        pbar.set_postfix({"I(θ)": f"{Y[i]:.4f}"}, refresh=False)
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
    estimator: str = "mc_probability",
) -> Dict:
    """End-to-end: Saltelli sampling → I(θ) evaluation → Sobol indices.

    No surrogate model — evaluates the true I(θ) at every sample point.

    Args:
        estimator: 'mc_probability' (default), 'mc_jaccard', or 'aabb_jaccard'.
                   For multi-fidelity, use run_multifidelity_sobol_analysis().
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
        print(f"Estimator      : {estimator}")
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
                         mc_samples=mc_samples, seed=seed, verbose=verbose,
                         estimator=estimator)

    # 3. Sobol decomposition
    Si = sobol_analyzer.analyze(problem, Y, calc_second_order=calc_second_order)

    results = {
        "problem": problem,
        "N": N,
        "estimator": estimator,
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
# Multi-fidelity estimator  (AABB low-fidelity + MC high-fidelity)
# ---------------------------------------------------------------------------

def evaluate_samples_multifidelity(
    scenario: Scenario,
    samples: np.ndarray,
    param_names: List[str],
    mc_samples: int = 2000,
    hf_fraction: float = 0.2,
    seed: Optional[int] = None,
    verbose: bool = True,
    hf_estimator: str = "mc_probability",
) -> Dict:
    """Multi-fidelity evaluation: AABB (low-fidelity) + MC (high-fidelity).

    Step 1: Evaluate I_AABB(θ) at ALL sample points — closed-form, free.
    Step 2: Evaluate I_HF(θ) at a pilot subset (n_HF = hf_fraction * N).
    Step 3: Fit linear regression  I_HF ~ I_AABB  from pilot pairs,
            compute correlation ρ and control variate coefficient α*.
    Step 4: Construct regression-corrected Y at all points for Sobol.

    Control variate estimator for the mean:
        I_MF = mean(I_HF) + α* · (mean(I_LF_all) − mean(I_LF_subset))
    with  Var(I_MF) = Var(I_HF) · (1 − ρ²).

    Returns dict with Y_corrected (for Sobol), diagnostics, and pilot data.
    """
    N_total = samples.shape[0]
    rng = np.random.default_rng(seed)

    # --- Step 1: AABB at ALL points (instant) ---
    if verbose:
        print(f"  [MF Step 1] Evaluating AABB Jaccard at all {N_total} points...")
    Y_lf = np.empty(N_total)
    for i in range(N_total):
        theta = dict(zip(param_names, samples[i]))
        Z_int = apply_compound_intervention(scenario.source, theta)
        Y_lf[i] = compute_I_theta_aabb(scenario, source_override=Z_int)["I_theta"]

    # --- Step 2: HF at pilot subset ---
    n_hf = max(10, int(hf_fraction * N_total))
    n_hf = min(n_hf, N_total)
    hf_idx = np.sort(rng.choice(N_total, size=n_hf, replace=False))

    if verbose:
        print(f"  [MF Step 2] Evaluating {hf_estimator} at "
              f"{n_hf}/{N_total} pilot points...")

    hf_seeds = rng.integers(0, 2**31, size=n_hf)
    Y_hf = np.empty(n_hf)

    pbar = tqdm(range(n_hf), desc="HF pilot", unit="eval", disable=not verbose)
    for j in pbar:
        i = hf_idx[j]
        theta = dict(zip(param_names, samples[i]))
        Z_int = apply_compound_intervention(scenario.source, theta)

        if hf_estimator == "mc_probability":
            result = compute_I_theta(scenario, source_override=Z_int,
                                     n_samples=mc_samples, seed=int(hf_seeds[j]))
        elif hf_estimator == "mc_jaccard":
            result = compute_I_theta_mc_jaccard(
                scenario, source_override=Z_int,
                n_samples=mc_samples, seed=int(hf_seeds[j]))
        else:
            raise ValueError(f"Unknown hf_estimator: {hf_estimator}")

        Y_hf[j] = result["I_theta"]
        pbar.set_postfix({"I(θ)": f"{Y_hf[j]:.4f}"}, refresh=False)

    # --- Step 3: Correlation + control variate coefficient ---
    Y_lf_subset = Y_lf[hf_idx]

    if np.std(Y_hf) > 1e-12 and np.std(Y_lf_subset) > 1e-12:
        rho = float(np.corrcoef(Y_hf, Y_lf_subset)[0, 1])
    else:
        rho = 0.0

    cov_hl = np.cov(Y_hf, Y_lf_subset, ddof=1)[0, 1]
    var_lf = np.var(Y_lf_subset, ddof=1)
    alpha_star = -cov_hl / var_lf if var_lf > 1e-15 else 0.0

    var_reduction = 1.0 - rho ** 2

    # --- Step 4: Regression-corrected Y for Sobol ---
    if var_lf > 1e-15:
        b = cov_hl / var_lf          # regression slope  (I_HF ~ I_LF)
        a = np.mean(Y_hf) - b * np.mean(Y_lf_subset)  # intercept
        Y_corrected = a + b * Y_lf
    else:
        a, b = float(np.mean(Y_hf)), 0.0
        Y_corrected = np.full(N_total, np.mean(Y_hf))

    # Control variate estimator for the mean
    mean_hf = float(np.mean(Y_hf))
    mean_lf_all = float(np.mean(Y_lf))
    mean_lf_sub = float(np.mean(Y_lf_subset))
    I_mf = mean_hf + alpha_star * (mean_lf_all - mean_lf_sub)

    if verbose:
        print(f"\n  [MF Step 3] Pilot diagnostics:")
        print(f"    rho(HF, LF)       = {rho:.4f}")
        print(f"    alpha*            = {alpha_star:.4f}")
        print(f"    Var reduction     = {var_reduction:.4f}  (1 - rho^2)")
        if var_reduction > 0.01:
            print(f"    Effective speedup ~ {1/var_reduction:.1f}x")
        else:
            print(f"    Effective speedup ~ 1x (low correlation)")
        print(f"    I_MF (cv mean)    = {I_mf:.4f}")
        print(f"    I_HF (pilot mean) = {mean_hf:.4f}")
        print(f"    I_LF (all mean)   = {mean_lf_all:.4f}")
        print(f"    Regression: I ~ {a:.4f} + {b:.4f} * I_AABB")

    return {
        "Y_corrected": Y_corrected,
        "Y_lf": Y_lf,
        "Y_hf": Y_hf,
        "hf_idx": hf_idx,
        "n_hf": n_hf,
        "n_lf": N_total,
        "rho": rho,
        "alpha_star": alpha_star,
        "variance_reduction": var_reduction,
        "I_mf_mean": I_mf,
        "I_hf_mean": mean_hf,
        "I_lf_mean": mean_lf_all,
        "regression_intercept": float(a),
        "regression_slope": float(b),
    }


def run_multifidelity_sobol_analysis(
    scenario: Scenario,
    N: int = 512,
    mc_samples: int = 2000,
    hf_fraction: float = 0.2,
    calc_second_order: bool = False,
    seed: int = 42,
    verbose: bool = True,
    param_names: Optional[List[str]] = None,
    param_bounds: Optional[List[List[float]]] = None,
    hf_estimator: str = "mc_probability",
) -> Dict:
    """Multi-fidelity Sobol analysis: AABB everywhere + MC at a subset.

    Uses regression-corrected I(θ) values for the Sobol decomposition.
    Reports correlation ρ, variance reduction factor, and comparison to
    pure-LF and pure-HF (pilot) Sobol estimates.
    """
    if not SALIB_AVAILABLE:
        raise ImportError("SALib required. Install: pip install SALib")

    problem = make_salib_problem(param_names, param_bounds)
    p = problem["num_vars"]
    n_eval = N * (2 * p + 2) if calc_second_order else N * (p + 2)

    if verbose:
        print(f"\n{'='*70}")
        print(f"MULTI-FIDELITY SOBOL ANALYSIS")
        print(f"{'='*70}")
        print(f"Scenario       : {scenario.name}")
        print(f"HF estimator   : {hf_estimator}")
        print(f"HF fraction    : {hf_fraction:.0%}  ({int(hf_fraction * n_eval)}/{n_eval})")
        print(f"Parameters     : {problem['names']}")
        print(f"Bounds         : {problem['bounds']}")
        print(f"Saltelli N     : {N}  ->  {n_eval} total points")
        print(f"MC samples/eval: {mc_samples}")
        print(f"{'='*70}\n")

    # 1. Generate Saltelli samples
    samples = generate_saltelli_samples(problem, N, calc_second_order, seed)
    assert samples.shape == (n_eval, p)

    # 2. Multi-fidelity evaluation
    mf = evaluate_samples_multifidelity(
        scenario, samples, problem["names"],
        mc_samples=mc_samples, hf_fraction=hf_fraction,
        seed=seed, verbose=verbose, hf_estimator=hf_estimator,
    )

    # 3. Sobol on regression-corrected values
    Y_mf = mf["Y_corrected"]
    Si_mf = sobol_analyzer.analyze(problem, Y_mf,
                                   calc_second_order=calc_second_order)

    # 4. Sobol on pure LF for comparison
    Y_lf = mf["Y_lf"]
    Si_lf = sobol_analyzer.analyze(problem, Y_lf,
                                   calc_second_order=calc_second_order)

    results = {
        "problem": problem,
        "N": N,
        "estimator": "multi_fidelity",
        "hf_estimator": hf_estimator,
        "n_evaluations": n_eval,
        "n_hf_evaluations": mf["n_hf"],
        "mc_samples_per_eval": mc_samples,
        "hf_fraction": hf_fraction,
        "rho": mf["rho"],
        "alpha_star": mf["alpha_star"],
        "variance_reduction": mf["variance_reduction"],
        "regression_intercept": mf["regression_intercept"],
        "regression_slope": mf["regression_slope"],
        "I_mf_mean": mf["I_mf_mean"],
        "I_hf_mean": mf["I_hf_mean"],
        "I_lf_mean": mf["I_lf_mean"],
        # Multi-fidelity Sobol indices
        "Y_mean": float(Y_mf.mean()),
        "Y_std": float(Y_mf.std()),
        "Y_min": float(Y_mf.min()),
        "Y_max": float(Y_mf.max()),
        "first_order": dict(zip(problem["names"], Si_mf["S1"].tolist())),
        "first_order_conf": dict(zip(problem["names"], Si_mf["S1_conf"].tolist())),
        "total_order": dict(zip(problem["names"], Si_mf["ST"].tolist())),
        "total_order_conf": dict(zip(problem["names"], Si_mf["ST_conf"].tolist())),
        # Pure LF Sobol indices (for comparison)
        "lf_first_order": dict(zip(problem["names"], Si_lf["S1"].tolist())),
        "lf_total_order": dict(zip(problem["names"], Si_lf["ST"].tolist())),
        # Raw arrays (for diagnostics / plotting)
        "_mf_data": mf,
    }

    if verbose:
        print(f"\n{'='*70}")
        print("MULTI-FIDELITY SOBOL INDICES")
        print(f"{'='*70}")
        print(f"Correlation rho          = {mf['rho']:.4f}")
        print(f"Variance reduction       = {mf['variance_reduction']:.4f}")
        if mf["variance_reduction"] > 0.01:
            print(f"Effective speedup        ~ {1/mf['variance_reduction']:.1f}x")
        print(f"I(theta) corrected: mean={Y_mf.mean():.4f}, std={Y_mf.std():.4f}")

        print(f"\n{'Parameter':<25s} {'S1_MF':>8s} {'S1_LF':>8s}  "
              f"{'ST_MF':>8s} {'ST_LF':>8s}")
        print("-" * 70)
        for name in problem["names"]:
            s1m = results["first_order"][name]
            s1l = results["lf_first_order"][name]
            stm = results["total_order"][name]
            stl = results["lf_total_order"][name]
            print(f"{name:<25s} {s1m:>8.4f} {s1l:>8.4f}  "
                  f"{stm:>8.4f} {stl:>8.4f}")
        print(f"{'='*70}\n")

    return results


# ---------------------------------------------------------------------------
# Estimator comparison
# ---------------------------------------------------------------------------

_ESTIMATOR_LABELS = {
    "mc_probability": r"$I_{\mathrm{MC\;prob}}$",
    "mc_jaccard": r"$I_{\mathrm{MC\;Jac}}$",
    "aabb_jaccard": r"$I_{\mathrm{AABB}}$",
    "multi_fidelity": r"$I_{\mathrm{MF}}$",
}

_ESTIMATOR_COLORS = {
    "mc_probability": "#4878CF",
    "mc_jaccard": "#6ACC65",
    "aabb_jaccard": "#D65F5F",
    "multi_fidelity": "#B47CC7",
}


def run_estimator_comparison(
    scenario: Scenario,
    N: int = 256,
    mc_samples: int = 2000,
    mf_hf_fraction: float = 0.2,
    calc_second_order: bool = False,
    seed: int = 42,
    verbose: bool = True,
    param_names: Optional[List[str]] = None,
    param_bounds: Optional[List[List[float]]] = None,
) -> Dict:
    """Run all four estimators on the same Saltelli samples and compare.

    Returns dict with per-estimator Y arrays, Sobol indices, pairwise
    correlations, and timing information.
    """
    import time

    if not SALIB_AVAILABLE:
        raise ImportError("SALib required. Install: pip install SALib")

    problem = make_salib_problem(param_names, param_bounds)
    p = problem["num_vars"]
    n_eval = N * (2 * p + 2) if calc_second_order else N * (p + 2)

    if verbose:
        print(f"\n{'='*70}")
        print("ESTIMATOR COMPARISON")
        print(f"{'='*70}")
        print(f"Scenario       : {scenario.name}  (dim={scenario.dim})")
        print(f"Saltelli N     : {N}  ->  {n_eval} total points")
        print(f"MC samples/eval: {mc_samples}")
        print(f"MF HF fraction : {mf_hf_fraction:.0%}")
        print(f"{'='*70}\n")

    # Generate samples once — shared across all estimators
    samples = generate_saltelli_samples(problem, N, calc_second_order, seed)
    assert samples.shape == (n_eval, p)

    estimators = ["aabb_jaccard", "mc_jaccard", "mc_probability", "multi_fidelity"]
    Y_all: Dict[str, np.ndarray] = {}
    sobol_all: Dict[str, Dict] = {}
    timing: Dict[str, float] = {}

    for est in estimators:
        if verbose:
            print(f"\n--- Evaluating: {est} ---")

        t0 = time.perf_counter()

        if est == "multi_fidelity":
            mf = evaluate_samples_multifidelity(
                scenario, samples, problem["names"],
                mc_samples=mc_samples, hf_fraction=mf_hf_fraction,
                seed=seed, verbose=verbose,
            )
            Y_all[est] = mf["Y_corrected"]
            Y_all["_mf_data"] = mf  # keep for diagnostics
        else:
            Y_all[est] = evaluate_samples(
                scenario, samples, problem["names"],
                mc_samples=mc_samples, seed=seed, verbose=verbose,
                estimator=est,
            )

        timing[est] = time.perf_counter() - t0

        # Sobol on this Y
        Si = sobol_analyzer.analyze(
            problem, Y_all[est], calc_second_order=calc_second_order)
        sobol_all[est] = {
            "first_order": dict(zip(problem["names"], Si["S1"].tolist())),
            "first_order_conf": dict(zip(problem["names"], Si["S1_conf"].tolist())),
            "total_order": dict(zip(problem["names"], Si["ST"].tolist())),
            "total_order_conf": dict(zip(problem["names"], Si["ST_conf"].tolist())),
        }

    # --- Pairwise correlations ---
    est_keys = [e for e in estimators]
    n_est = len(est_keys)
    corr_matrix = np.eye(n_est)
    for i in range(n_est):
        for j in range(i + 1, n_est):
            Yi, Yj = Y_all[est_keys[i]], Y_all[est_keys[j]]
            if np.std(Yi) > 1e-12 and np.std(Yj) > 1e-12:
                rho = float(np.corrcoef(Yi, Yj)[0, 1])
            else:
                rho = 0.0
            corr_matrix[i, j] = rho
            corr_matrix[j, i] = rho

    # --- Print summary ---
    if verbose:
        print(f"\n{'='*70}")
        print("ESTIMATOR COMPARISON — RESULTS")
        print(f"{'='*70}")

        # Timing
        print(f"\n{'Estimator':<20s} {'Time (s)':>10s} {'Mean I':>8s} {'Std I':>8s}")
        print("-" * 50)
        for est in estimators:
            Y = Y_all[est]
            print(f"{est:<20s} {timing[est]:>10.1f} {Y.mean():>8.4f} {Y.std():>8.4f}")

        # Correlations
        print(f"\nPairwise Pearson correlations:")
        header = f"{'':>20s}" + "".join(f" {e:>14s}" for e in est_keys)
        print(header)
        for i, ei in enumerate(est_keys):
            row = f"{ei:>20s}"
            for j in range(n_est):
                row += f" {corr_matrix[i, j]:>14.4f}"
            print(row)

        # Sobol comparison
        print(f"\nSobol first-order indices (S1):")
        header = f"{'Parameter':<25s}" + "".join(
            f" {e:>14s}" for e in est_keys)
        print(header)
        print("-" * (25 + 15 * n_est))
        for name in problem["names"]:
            row = f"{name:<25s}"
            for est in est_keys:
                row += f" {sobol_all[est]['first_order'][name]:>14.4f}"
            print(row)

        print(f"\nSobol total-effect indices (ST):")
        print(header)
        print("-" * (25 + 15 * n_est))
        for name in problem["names"]:
            row = f"{name:<25s}"
            for est in est_keys:
                row += f" {sobol_all[est]['total_order'][name]:>14.4f}"
            print(row)

        print(f"{'='*70}\n")

    return {
        "problem": problem,
        "N": N,
        "n_evaluations": n_eval,
        "mc_samples_per_eval": mc_samples,
        "estimators": estimators,
        "Y": {k: v for k, v in Y_all.items() if k != "_mf_data"},
        "_mf_data": Y_all.get("_mf_data"),
        "sobol": sobol_all,
        "correlations": corr_matrix,
        "timing": timing,
    }


def plot_estimator_comparison(
    comp: Dict,
    scenario_name: str = "",
    output_path: Optional[str | Path] = None,
    figsize: Tuple[float, float] = (16, 12),
) -> plt.Figure:
    """Five-panel comparison plot for all estimators.

    Row 1: Pairwise scatter (MC prob vs each other), correlation heatmap.
    Row 2: Sobol S1 grouped bar chart, Sobol ST grouped bar chart.
    """
    estimators = comp["estimators"]
    Y = comp["Y"]
    sobol = comp["sobol"]
    corr = comp["correlations"]
    problem = comp["problem"]
    names = problem["names"]
    n_est = len(estimators)

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])

    # --- Row 1, left & center: pairwise scatters vs mc_probability ---
    ref = "mc_probability"
    others = [e for e in estimators if e != ref]
    for col_idx, est in enumerate(others[:3]):
        ax = fig.add_subplot(gs[0, col_idx])
        Yr = Y[ref]
        Ye = Y[est]
        ax.scatter(Yr, Ye, s=8, alpha=0.4,
                   color=_ESTIMATOR_COLORS[est], edgecolors="none")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=0.8)

        # Compute rho
        ri = estimators.index(ref)
        ei = estimators.index(est)
        rho = corr[ri, ei]
        ax.set_xlabel(_ESTIMATOR_LABELS[ref], fontsize=11)
        ax.set_ylabel(_ESTIMATOR_LABELS[est], fontsize=11)
        ax.set_title(r"$\rho$ = " + f"{rho:.4f}", fontsize=12,
                     fontweight="bold")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.25)

    # --- Row 2, left: S1 grouped bar ---
    ax_s1 = fig.add_subplot(gs[1, 0:2])
    x = np.arange(len(names))
    total_w = 0.75
    w = total_w / n_est
    for k, est in enumerate(estimators):
        vals = [sobol[est]["first_order"][n] for n in names]
        confs = [sobol[est]["first_order_conf"][n] for n in names]
        offset = -total_w / 2 + w * (k + 0.5)
        ax_s1.bar(x + offset, vals, w, yerr=confs, capsize=2,
                  color=_ESTIMATOR_COLORS[est], edgecolor="white",
                  linewidth=0.5, label=_ESTIMATOR_LABELS[est], zorder=3)
    ax_s1.set_xticks(x)
    ax_s1.set_xticklabels([PARAM_LABELS.get(n, n) for n in names], fontsize=10)
    ax_s1.set_ylabel("Sobol Index", fontsize=11)
    ax_s1.set_title(r"First-order $S_i$", fontsize=12, fontweight="bold")
    ax_s1.legend(fontsize=8, ncol=n_est, loc="upper right")
    ax_s1.grid(axis="y", alpha=0.25)
    ax_s1.set_axisbelow(True)

    # --- Row 2, right: ST grouped bar ---
    ax_st = fig.add_subplot(gs[1, 2])
    for k, est in enumerate(estimators):
        vals = [sobol[est]["total_order"][n] for n in names]
        confs = [sobol[est]["total_order_conf"][n] for n in names]
        offset = -total_w / 2 + w * (k + 0.5)
        ax_st.bar(x + offset, vals, w, yerr=confs, capsize=2,
                  color=_ESTIMATOR_COLORS[est], edgecolor="white",
                  linewidth=0.5, label=_ESTIMATOR_LABELS[est], zorder=3)
    ax_st.set_xticks(x)
    ax_st.set_xticklabels([PARAM_LABELS.get(n, n) for n in names], fontsize=10)
    ax_st.set_ylabel("Sobol Index", fontsize=11)
    ax_st.set_title(r"Total-effect $S_i^T$", fontsize=12, fontweight="bold")
    ax_st.grid(axis="y", alpha=0.25)
    ax_st.set_axisbelow(True)

    title = "Estimator Comparison"
    if scenario_name:
        title += f" — {scenario_name}"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {output_path}")

    return fig


def run_estimator_convergence(
    scenario: Scenario,
    seed: int = 42,
    mc_max: int = 5000,
    n_reps: int = 20,
    mf_hf_fraction: float = 0.2,
    mf_pilot_n: int = 40,
    verbose: bool = True,
) -> Dict:
    """MC convergence study across all four estimators.

    For each MC sample count N, evaluates mc_probability, mc_jaccard,
    aabb_jaccard (constant), and multi_fidelity.

    The multi-fidelity convergence works as follows: at each MC count N,
    run a small pilot of mf_pilot_n evaluations — AABB on all, HF (mc_prob
    with N inner samples) on a fraction — fit the regression, then report
    the control-variate corrected mean.  This shows how the MF estimator's
    accuracy improves as the inner MC count grows.
    """
    mc_counts = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    mc_counts = [c for c in mc_counts if c <= mc_max]

    estimators = ["mc_probability", "mc_jaccard", "aabb_jaccard", "multi_fidelity"]
    results: Dict[str, Dict] = {
        est: {"n_samples": mc_counts, "I_theta_mean": [], "I_theta_std": []}
        for est in estimators
    }

    rng = np.random.default_rng(seed)

    # AABB is deterministic — evaluate once
    aabb_val = compute_I_theta_aabb(scenario)["I_theta"]
    results["aabb_jaccard"]["I_theta_mean"] = [aabb_val] * len(mc_counts)
    results["aabb_jaccard"]["I_theta_std"] = [0.0] * len(mc_counts)

    if verbose:
        print(f"\n{'='*70}")
        print("ESTIMATOR CONVERGENCE TEST")
        print(f"{'='*70}")
        print(f"Scenario: {scenario.name}  (dim={scenario.dim})")
        print(f"MC counts: {mc_counts},  reps: {n_reps}")
        print(f"MF pilot size: {mf_pilot_n},  HF fraction: {mf_hf_fraction:.0%}")
        print(f"AABB (constant): I = {aabb_val:.4f}")
        print(f"{'='*70}\n")

    # --- mc_probability and mc_jaccard ---
    for est in ["mc_probability", "mc_jaccard"]:
        if verbose:
            print(f"--- {est} ---")
        for n in tqdm(mc_counts, desc=est, disable=not verbose):
            reps = []
            for _ in range(n_reps):
                s = int(rng.integers(0, 2**31))
                if est == "mc_probability":
                    r = compute_I_theta(scenario, n_samples=n, seed=s)
                else:
                    r = compute_I_theta_mc_jaccard(scenario, n_samples=n, seed=s)
                reps.append(r["I_theta"])
            results[est]["I_theta_mean"].append(float(np.mean(reps)))
            results[est]["I_theta_std"].append(float(np.std(reps)))
            if verbose:
                print(f"  n={n:>5d}: I = {np.mean(reps):.4f} +/- {np.std(reps):.4f}")

    # --- multi_fidelity ---
    # The MF convergence evaluates at the SAME nominal scenario as the other
    # estimators.  For each MC count N we repeat n_reps times:
    #   1. Draw mf_pilot_n small perturbations around the nominal θ₀
    #      (to build a local AABB↔HF correlation).
    #   2. Evaluate AABB on all pilot points (instant).
    #   3. Evaluate MC probability with N inner samples on a subset.
    #   4. Fit control variate, predict I(θ₀) via the regression.
    #
    # The key insight: θ₀ itself is always included in the pilot so the
    # MF estimate targets I(θ₀) — the same quantity as the other curves.
    if verbose:
        print(f"--- multi_fidelity ---")

    problem = make_salib_problem()
    bounds = np.array(problem["bounds"])
    param_names = problem["names"]

    # Nominal θ₀: identity (no intervention)
    theta_0 = {p: PARAM_BASELINES[p] for p in param_names}
    theta_0_arr = np.array([theta_0[p] for p in param_names])

    # AABB at the nominal point (constant across reps)
    I_aabb_0 = aabb_val  # already computed above

    for n_mc in tqdm(mc_counts, desc="multi_fidelity", disable=not verbose):
        mf_reps = []
        for _ in range(n_reps):
            # Build a small pilot around θ₀ (random perturbations)
            pilot_thetas = rng.uniform(
                bounds[:, 0], bounds[:, 1],
                size=(mf_pilot_n - 1, len(param_names)))
            # Ensure θ₀ is always included (first row)
            pilot_thetas = np.vstack([theta_0_arr, pilot_thetas])

            # AABB on all pilot configs
            Y_lf = np.empty(mf_pilot_n)
            Y_lf[0] = I_aabb_0
            for k in range(1, mf_pilot_n):
                theta = dict(zip(param_names, pilot_thetas[k]))
                Z_int = apply_compound_intervention(scenario.source, theta)
                Y_lf[k] = compute_I_theta_aabb(
                    scenario, source_override=Z_int)["I_theta"]

            # HF on a subset — always include index 0 (θ₀)
            n_hf = max(5, int(mf_hf_fraction * mf_pilot_n))
            other_idx = rng.choice(
                np.arange(1, mf_pilot_n), size=n_hf - 1, replace=False)
            hf_idx = np.concatenate([[0], other_idx])
            Y_hf = np.empty(len(hf_idx))
            for j, idx in enumerate(hf_idx):
                theta = dict(zip(param_names, pilot_thetas[idx]))
                Z_int = apply_compound_intervention(scenario.source, theta)
                r = compute_I_theta(
                    scenario, source_override=Z_int,
                    n_samples=n_mc, seed=int(rng.integers(0, 2**31)))
                Y_hf[j] = r["I_theta"]

            # Fit regression I_HF = a + b * I_AABB from pilot pairs
            Y_lf_sub = Y_lf[hf_idx]
            var_lf = np.var(Y_lf_sub, ddof=1)
            if var_lf > 1e-15:
                cov_hl = np.cov(Y_hf, Y_lf_sub, ddof=1)[0, 1]
                b = cov_hl / var_lf
                a = np.mean(Y_hf) - b * np.mean(Y_lf_sub)
                # Predict I(θ₀) from AABB value at θ₀
                I_mf = a + b * I_aabb_0
            else:
                I_mf = float(np.mean(Y_hf))
            mf_reps.append(float(I_mf))

        results["multi_fidelity"]["I_theta_mean"].append(
            float(np.mean(mf_reps)))
        results["multi_fidelity"]["I_theta_std"].append(
            float(np.std(mf_reps)))
        if verbose:
            print(f"  n={n_mc:>5d}: I_MF = {np.mean(mf_reps):.4f} "
                  f"+/- {np.std(mf_reps):.4f}")

    return results


def plot_estimator_convergence(
    conv: Dict,
    scenario_name: str = "",
    output_path: Optional[str | Path] = None,
    figsize: Tuple[float, float] = (9, 5.5),
) -> plt.Figure:
    """Overlay MC convergence curves for all estimators on a single plot."""
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    for est in ["mc_probability", "mc_jaccard", "aabb_jaccard", "multi_fidelity"]:
        if est not in conv:
            continue
        data = conv[est]
        ns = np.array(data["n_samples"])
        means = np.array(data["I_theta_mean"])
        stds = np.array(data["I_theta_std"])
        color = _ESTIMATOR_COLORS[est]
        label = _ESTIMATOR_LABELS[est]

        if est == "aabb_jaccard":
            ax.axhline(means[0], color=color, linewidth=2, linestyle="--",
                       label=f"{label} = {means[0]:.4f}", zorder=2)
        else:
            ax.fill_between(ns, means - stds, means + stds,
                            alpha=0.15, color=color)
            ax.plot(ns, means, "o-", color=color, markersize=5,
                    linewidth=1.5, label=label, zorder=3)

    ax.set_xscale("log")
    ax.set_xlabel("MC samples per evaluation", fontsize=12)
    ax.set_ylabel(r"$I(\theta)$", fontsize=12)
    title = "Estimator Convergence"
    if scenario_name:
        title += f" — {scenario_name}"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {output_path}")
    return fig


def plot_estimator_landscape(
    scenario: Scenario,
    param1: str = "scale_factor",
    param1_range: Tuple[float, float] = (0.3, 4.0),
    param2: str = "center_delta",
    param2_range: Tuple[float, float] = (0.0, 0.3),
    grid_n: int = 25,
    mc_samples: int = 500,
    mf_hf_fraction: float = 0.2,
    seed: int = 42,
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Side-by-side I(theta) heatmaps — one panel per estimator.

    MC probability | MC Jaccard | AABB Jaccard | Multi-fidelity
    Uses the fast 2D polygon path for mc_probability to keep runtime sane.
    The MF panel uses AABB on the full grid + regression calibrated from a
    pilot subset of HF evaluations.
    """
    from zonotope_plots import _compute_I_theta_2d

    if scenario.dim != 2:
        print("Landscape comparison only supported for 2D scenarios — skipping.")
        return None

    v1 = np.linspace(*param1_range, grid_n)
    v2 = np.linspace(*param2_range, grid_n)
    V1, V2 = np.meshgrid(v1, v2)

    rng = np.random.default_rng(seed)

    estimators = ["mc_probability", "mc_jaccard", "aabb_jaccard"]
    grids: Dict[str, np.ndarray] = {}

    for est in estimators:
        I_grid = np.empty_like(V1)
        desc = f"Landscape [{est}]"
        total = grid_n * grid_n
        pbar = tqdm(total=total, desc=desc, unit="pt")
        for i in range(grid_n):
            for j in range(grid_n):
                theta = {param1: V1[i, j], param2: V2[i, j]}
                if est == "mc_probability":
                    # Use fast 2D polygon containment
                    I_grid[i, j] = _compute_I_theta_2d(
                        scenario, theta, mc_samples, rng)
                elif est == "mc_jaccard":
                    Z_mod = apply_compound_intervention(scenario.source, theta)
                    r = compute_I_theta_mc_jaccard(
                        scenario, source_override=Z_mod,
                        n_samples=mc_samples,
                        seed=int(rng.integers(0, 2**31)))
                    I_grid[i, j] = r["I_theta"]
                else:  # aabb_jaccard
                    Z_mod = apply_compound_intervention(scenario.source, theta)
                    I_grid[i, j] = compute_I_theta_aabb(
                        scenario, source_override=Z_mod)["I_theta"]
                pbar.update(1)
        pbar.close()
        grids[est] = I_grid

    # --- Multi-fidelity panel: AABB everywhere + pilot HF + regression ---
    estimators.append("multi_fidelity")
    aabb_flat = grids["aabb_jaccard"].ravel()
    hf_flat = grids["mc_probability"].ravel()  # reuse existing HF grid
    n_total = len(aabb_flat)
    n_pilot = max(20, int(mf_hf_fraction * n_total))
    pilot_idx = rng.choice(n_total, size=n_pilot, replace=False)
    # Fit linear regression: I_HF ~ a + b * I_AABB on pilot subset
    x_pilot = aabb_flat[pilot_idx]
    y_pilot = hf_flat[pilot_idx]
    b_hat, a_hat = np.polyfit(x_pilot, y_pilot, 1)
    mf_flat = a_hat + b_hat * aabb_flat
    grids["multi_fidelity"] = mf_flat.reshape(V1.shape)
    print(f"  MF landscape: pilot {n_pilot}/{n_total} pts, "
          f"regression I_HF = {a_hat:.4f} + {b_hat:.4f} * I_AABB")

    # Shared colour scale
    vmin = min(g.min() for g in grids.values())
    vmax = max(g.max() for g in grids.values())

    n_est = len(estimators)
    fig, axes = plt.subplots(
        1, n_est, figsize=(4.8 * n_est + 1.5, 4.5), constrained_layout=True)

    for ax, est in zip(axes, estimators):
        I_grid = grids[est]
        cf = ax.contourf(V1, V2, I_grid, levels=20, cmap="RdYlGn_r",
                         vmin=vmin, vmax=vmax)
        ax.contour(V1, V2, I_grid, levels=8, colors="k",
                   linewidths=0.35, alpha=0.4)
        ax.set_xlabel(PARAM_LABELS.get(param1, param1), fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel(PARAM_LABELS.get(param2, param2), fontsize=10)
        ax.set_title(_ESTIMATOR_LABELS[est], fontsize=12, fontweight="bold")

    fig.colorbar(cf, ax=axes.tolist(), label=r"$I(\theta)$", shrink=0.85)
    fig.suptitle(f"Consistency Landscape Comparison — {scenario.name}",
                 fontsize=13, fontweight="bold")

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {output_path}")
    return fig


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


def plot_multifidelity_diagnostics(
    mf_results: Dict,
    scenario_name: str = "",
    output_path: Optional[str | Path] = None,
    figsize: Tuple[float, float] = (14, 10),
) -> plt.Figure:
    """Four-panel diagnostic plot for the multi-fidelity estimator.

    Panel A: HF vs LF scatter with regression line (pilot data).
    Panel B: Sobol index comparison (MF vs pure-LF).
    Panel C: Residual distribution of HF - predicted.
    Panel D: Corrected I(θ) histogram vs LF histogram.
    """
    mf = mf_results.get("_mf_data", mf_results)
    Y_hf = mf["Y_hf"]
    Y_lf_sub = mf["Y_lf"][mf["hf_idx"]]
    Y_lf_all = mf["Y_lf"]
    Y_corr = mf_results.get("Y_corrected", mf.get("Y_corrected"))
    rho = mf_results.get("rho", mf.get("rho", 0.0))
    a = mf_results.get("regression_intercept", mf.get("regression_intercept", 0.0))
    b = mf_results.get("regression_slope", mf.get("regression_slope", 0.0))

    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)

    # --- Panel A: HF vs LF scatter ---
    ax = axes[0, 0]
    ax.scatter(Y_lf_sub, Y_hf, s=20, alpha=0.6, edgecolors="k", linewidth=0.3,
               color="#4878CF", zorder=3)
    lf_range = np.linspace(Y_lf_sub.min(), Y_lf_sub.max(), 100)
    ax.plot(lf_range, a + b * lf_range, "r-", linewidth=2,
            label=f"y = {a:.3f} + {b:.3f}x", zorder=4)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=0.8)
    ax.set_xlabel(r"$I_{\mathrm{AABB}}(\theta)$  (low-fidelity)", fontsize=11)
    ax.set_ylabel(r"$I_{\mathrm{MC}}(\theta)$  (high-fidelity)", fontsize=11)
    ax.set_title(f"Pilot Correlation  "
                 r"$\rho$" + f" = {rho:.4f}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    # --- Panel B: Sobol comparison ---
    ax = axes[0, 1]
    if "first_order" in mf_results and "lf_first_order" in mf_results:
        names = list(mf_results["first_order"].keys())
        labels = [PARAM_LABELS.get(n, n) for n in names]
        S1_mf = [mf_results["first_order"][n] for n in names]
        S1_lf = [mf_results["lf_first_order"][n] for n in names]
        ST_mf = [mf_results["total_order"][n] for n in names]
        ST_lf = [mf_results["lf_total_order"][n] for n in names]

        x = np.arange(len(names))
        w = 0.2
        ax.bar(x - 1.5 * w, S1_mf, w, label=r"$S_i$ MF", color="#4878CF",
               edgecolor="white", linewidth=0.6)
        ax.bar(x - 0.5 * w, S1_lf, w, label=r"$S_i$ LF", color="#4878CF",
               alpha=0.4, edgecolor="white", linewidth=0.6)
        ax.bar(x + 0.5 * w, ST_mf, w, label=r"$S_i^T$ MF", color="#D65F5F",
               edgecolor="white", linewidth=0.6)
        ax.bar(x + 1.5 * w, ST_lf, w, label=r"$S_i^T$ LF", color="#D65F5F",
               alpha=0.4, edgecolor="white", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("Sobol Index", fontsize=11)
        ax.legend(fontsize=8, ncol=2)
    ax.set_title("Sobol: Multi-Fidelity vs Pure LF", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)

    # --- Panel C: Residual distribution ---
    ax = axes[1, 0]
    predicted = a + b * Y_lf_sub
    residuals = Y_hf - predicted
    ax.hist(residuals, bins=25, color="#6ACC65", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Residual  (HF - predicted)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"Regression Residuals  "
                 f"(RMSE = {np.sqrt(np.mean(residuals**2)):.4f})",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.25)

    # --- Panel D: Distribution comparison ---
    ax = axes[1, 1]
    bins = np.linspace(min(Y_lf_all.min(), Y_corr.min()),
                       max(Y_lf_all.max(), Y_corr.max()), 40)
    ax.hist(Y_lf_all, bins=bins, alpha=0.5, color="#4878CF",
            label="AABB (LF)", edgecolor="white")
    ax.hist(Y_corr, bins=bins, alpha=0.5, color="#D65F5F",
            label="Corrected (MF)", edgecolor="white")
    ax.axvline(mf_results.get("I_hf_mean", mf.get("I_hf_mean", 0)),
               color="green", linewidth=2, linestyle="--",
               label=f"HF pilot mean = "
                     f"{mf_results.get('I_hf_mean', mf.get('I_hf_mean', 0)):.3f}")
    ax.set_xlabel(r"$I(\theta)$", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("I(theta) Distribution: LF vs Corrected",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    title = "Multi-Fidelity Estimator Diagnostics"
    if scenario_name:
        title += f"\n{scenario_name}"
    vr = mf_results.get("variance_reduction", mf.get("variance_reduction", 0))
    title += (f"   |   "
              r"$\rho$" + f" = {rho:.3f},  "
              f"Var reduction = {vr:.3f}")
    fig.suptitle(title, fontsize=13, fontweight="bold")

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
# Convergence test
# ---------------------------------------------------------------------------

def run_convergence_test(
    scenario: Scenario,
    seed: int,
    out_dir: Path,
    mc_max: int = 5000,
    n_reps: int = 30,
    sobol_mc: int = 500,
):
    """MC sample-count and Saltelli-N convergence study."""
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("CONVERGENCE TEST")
    print(f"{'='*70}")
    print(f"Scenario: {scenario.name}  (dim={scenario.dim})")
    print(f"{'='*70}\n")

    # --- Part 1: MC convergence of I(θ) ---
    mc_counts = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    mc_counts = [c for c in mc_counts if c <= mc_max]

    print(f"Part 1: MC convergence (n_samples up to {mc_max}, {n_reps} reps each)")
    rng = np.random.default_rng(seed)

    I_all: list[list[float]] = []
    for n in tqdm(mc_counts, desc="MC counts"):
        reps = []
        for _ in range(n_reps):
            s = int(rng.integers(0, 2**31))
            result = compute_I_theta(scenario, n_samples=n, seed=s)
            reps.append(result["I_theta"])
        I_all.append(reps)
        print(f"  n={n:>5d}: I(θ) = {np.mean(reps):.4f} ± {np.std(reps):.4f}")

    mc_results = {
        "n_samples": mc_counts,
        "I_theta_mean": [float(np.mean(r)) for r in I_all],
        "I_theta_std": [float(np.std(r)) for r in I_all],
    }

    # --- Part 2: Sobol convergence ---
    sobol_conv = None
    if SALIB_AVAILABLE:
        N_values = [16, 32, 64, 128, 256, 512]
        print(f"\nPart 2: Sobol convergence "
              f"(Saltelli N in {N_values}, mc_samples={sobol_mc})")

        sobol_conv = {
            "N": N_values, "S1": {}, "ST": {}, "S1_conf": {}, "ST_conf": {},
        }
        for N in tqdm(N_values, desc="Saltelli N"):
            sr = run_sobol_analysis(
                scenario, N=N, mc_samples=sobol_mc, seed=seed, verbose=False)
            for name in sr["first_order"]:
                sobol_conv["S1"].setdefault(name, []).append(
                    sr["first_order"][name])
                sobol_conv["ST"].setdefault(name, []).append(
                    sr["total_order"][name])
                sobol_conv["S1_conf"].setdefault(name, []).append(
                    sr["first_order_conf"][name])
                sobol_conv["ST_conf"].setdefault(name, []).append(
                    sr["total_order_conf"][name])
            print(f"  N={N:>4d}: " + "  ".join(
                f"S1({n})={sr['first_order'][n]:.3f}"
                for n in sr["first_order"]))
    else:
        print("\nSALib not installed — skipping Sobol convergence.")

    # --- Plots ---
    from zonotope_plots import plot_mc_convergence, plot_sobol_convergence

    plot_mc_convergence(
        mc_results, scenario.name,
        output_path=out_dir / "convergence_mc.png",
    )
    if sobol_conv is not None:
        plot_sobol_convergence(
            sobol_conv, scenario.name,
            output_path=out_dir / "convergence_sobol.png",
        )
    plt.close("all")

    # --- Export ---
    all_conv = {"mc_convergence": mc_results}
    if sobol_conv is not None:
        all_conv["sobol_convergence"] = sobol_conv
    export_results(all_conv, out_dir / "convergence_results.json")

    print(f"\n{'='*70}")
    print("CONVERGENCE TEST DONE")
    print(f"{'='*70}")


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
    parser.add_argument("--convergence_test", action="store_true",
                        help="Run MC & Sobol convergence study instead of "
                             "the normal pipeline")
    parser.add_argument("--estimator", type=str, default="mc_probability",
                        choices=["mc_probability", "mc_jaccard",
                                 "aabb_jaccard", "multi_fidelity", "compare"],
                        help="Scoring method for I(theta), or 'compare' to "
                             "run all estimators side-by-side (default: mc_probability)")
    parser.add_argument("--mf_hf_fraction", type=float, default=0.2,
                        help="Fraction of Saltelli points evaluated with HF "
                             "estimator in multi-fidelity mode (default: 0.2)")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else \
        Path("results") / f"causal_engine_{timestamp}"

    scenarios = create_engineering_scenarios()
    scenario = scenarios[args.scenario - 1]

    if args.convergence_test:
        run_convergence_test(scenario, args.seed, out_dir)
        return

    print(f"\n{'='*70}")
    print(f"CAUSAL ENGINE — Python replacement")
    print(f"{'='*70}")
    print(f"Scenario       : {scenario.name}  (dim={scenario.dim})")
    print(f"Source center   : {scenario.source.c}")
    print(f"Target center   : {scenario.target.c}")
    print(f"MC samples/eval : {args.mc_samples}")
    print(f"Estimator       : {args.estimator}")
    if args.estimator == "multi_fidelity":
        print(f"HF fraction     : {args.mf_hf_fraction:.0%}")
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
    baseline_aabb = compute_I_theta_aabb(scenario)
    print(f"  Baseline I_MC(θ)   = {baseline['I_theta']:.4f} "
          f"± {baseline['standard_error']:.4f}")
    print(f"  Baseline I_AABB(θ) = {baseline_aabb['I_theta']:.4f}\n")
    all_results["baseline"] = baseline
    all_results["baseline_aabb"] = baseline_aabb

    # --- 2D zonotope visualizations (Figures 1 & 2) ---
    if scenario.dim == 2:
        from zonotope_plots import plot_zonotope_consistency, plot_intervention_strip
        out_dir.mkdir(parents=True, exist_ok=True)
        print("\nGenerating 2D zonotope visualizations...")

        # Fig 1a: baseline — rectangular source vs target
        plot_zonotope_consistency(
            scenario, n_samples=min(1000, args.mc_samples), seed=args.seed,
            output_path=out_dir / "fig1_zonotope_consistency.png",
        )
        # Fig 1b: correlate intervention — hexagonal source zonotope
        Z_corr = apply_intervention(
            scenario.source, "correlate", {"correlation_strength": 0.6})
        plot_zonotope_consistency(
            scenario, source_override=Z_corr,
            n_samples=min(1000, args.mc_samples), seed=args.seed,
            output_path=out_dir / "fig1_zonotope_consistency_correlated.png",
            title_suffix=r"($\beta=0.6$)",
        )
        # Fig 2: widen strip — rectangles growing
        plot_intervention_strip(
            scenario, "widen", "scale_factor", [0.5, 1.0, 2.0, 5.0],
            n_samples=min(500, args.mc_samples), seed=args.seed,
            output_path=out_dir / "fig2_intervention_strip_widen.png",
        )
        plt.close("all")

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
    mf_results = None
    comp_results = None
    if not args.sweep_only:
        if not SALIB_AVAILABLE:
            print("\nSALib not installed — skipping Sobol analysis.")
        elif args.estimator == "compare":
            comp_results = run_estimator_comparison(
                scenario,
                N=args.saltelli_N,
                mc_samples=args.mc_samples,
                mf_hf_fraction=args.mf_hf_fraction,
                seed=args.seed,
                verbose=True,
            )
            # Use mc_probability Sobol as the primary result
            sobol_results = comp_results["sobol"]["mc_probability"]
            all_results["sobol"] = sobol_results
            all_results["estimator_comparison"] = {
                "estimators": comp_results["estimators"],
                "correlations": comp_results["correlations"].tolist(),
                "timing": comp_results["timing"],
                "sobol": comp_results["sobol"],
            }
        elif args.estimator == "multi_fidelity":
            mf_results = run_multifidelity_sobol_analysis(
                scenario,
                N=args.saltelli_N,
                mc_samples=args.mc_samples,
                hf_fraction=args.mf_hf_fraction,
                seed=args.seed,
                verbose=True,
            )
            # MF results have the same Sobol keys, reuse for plotting
            sobol_results = mf_results
            all_results["sobol"] = {
                k: v for k, v in mf_results.items() if k != "_mf_data"
            }
            all_results["multifidelity"] = {
                "rho": mf_results["rho"],
                "alpha_star": mf_results["alpha_star"],
                "variance_reduction": mf_results["variance_reduction"],
                "regression_intercept": mf_results["regression_intercept"],
                "regression_slope": mf_results["regression_slope"],
                "I_mf_mean": mf_results["I_mf_mean"],
                "I_hf_mean": mf_results["I_hf_mean"],
                "I_lf_mean": mf_results["I_lf_mean"],
                "n_hf_evaluations": mf_results["n_hf_evaluations"],
                "hf_fraction": mf_results["hf_fraction"],
            }
        else:
            sobol_results = run_sobol_analysis(
                scenario,
                N=args.saltelli_N,
                mc_samples=args.mc_samples,
                seed=args.seed,
                verbose=True,
                estimator=args.estimator,
            )
            all_results["sobol"] = sobol_results

    # --- 2D heatmap & membership visualizations (Figure 3) ---
    if scenario.dim == 2:
        from zonotope_plots import (
            plot_inconsistency_heatmap, plot_membership_map,
            plot_inconsistency_slices,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        print("\nGenerating parameter-space and membership visualizations...")

        # Fig 3A: I(θ) landscape over scale × shift
        plot_inconsistency_heatmap(
            scenario, mc_samples=min(500, args.mc_samples), seed=args.seed,
            param2_range=(-0.5, 1.5),
            output_path=out_dir / "fig3a_inconsistency_heatmap.png",
        )
        # Fig 3C: slice panels — scale × shift at multiple correlation values
        plot_inconsistency_slices(
            scenario, mc_samples=min(500, args.mc_samples), seed=args.seed,
            param2_range=(-0.5, 1.5),
            output_path=out_dir / "fig3c_inconsistency_slices.png",
        )
        # Fig 3B: physical-space membership (baseline)
        plot_membership_map(
            scenario,
            output_path=out_dir / "fig3b_membership_map.png",
        )
        plt.close("all")

    # --- Plots ---
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_sweeps(sweep_results, scenario_name=scenario.name,
                output_path=out_dir / "intervention_sweeps.png")

    if sobol_results is not None:
        plot_sobol_indices(sobol_results, scenario_name=scenario.name,
                           output_path=out_dir / "sobol_indices.png")

    if mf_results is not None:
        plot_multifidelity_diagnostics(
            mf_results, scenario_name=scenario.name,
            output_path=out_dir / "multifidelity_diagnostics.png",
        )

    if comp_results is not None:
        plot_estimator_comparison(
            comp_results, scenario_name=scenario.name,
            output_path=out_dir / "estimator_comparison.png",
        )

        # Convergence curves
        print("\nRunning estimator convergence study...")
        conv = run_estimator_convergence(
            scenario, seed=args.seed, mc_max=2000, n_reps=20, verbose=True)
        plot_estimator_convergence(
            conv, scenario_name=scenario.name,
            output_path=out_dir / "estimator_convergence.png",
        )
        all_results["estimator_convergence"] = {
            est: {k: v for k, v in d.items()}
            for est, d in conv.items()
        }

        # Side-by-side consistency landscape (2D only)
        if scenario.dim == 2:
            print("\nGenerating side-by-side consistency landscapes...")
            plot_estimator_landscape(
                scenario,
                mc_samples=min(500, args.mc_samples),
                mf_hf_fraction=args.mf_hf_fraction,
                seed=args.seed,
                output_path=out_dir / "estimator_landscape.png",
            )

    plt.close("all")

    # --- Export ---
    export_results(all_results, out_dir / "causal_engine_results.json")

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
