"""Counterfactual explanation module for MMS inconsistency.

Given an inconsistent θ* with Ĩ(θ*) > γ, finds the nearest configuration
θ' that restores consistency:

    θ' = argmin_θ ||θ - θ*||²   s.t.   Ĩ(θ) ≤ γ

Optimization: projected gradient descent (Adam) with penalty escalation.
    - GNN weights are frozen; only θ is updated.
    - Box constraints enforced via projection (clamp) after each step.
    - Penalty λ is escalated when the constraint is not satisfied.
    - MC verification at the end confirms the fix on the exact simulator.

Evaluation: validity rate, mean proximity, per-parameter Δθ breakdown,
    and comparison against a naive baseline that zeros the highest-Sobol param.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data

# ---------------------------------------------------------------------------
# Internal imports – causal engine + model
# ---------------------------------------------------------------------------
import sys

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from causal_engine import (
    PARAM_BOUNDS,
    PARAM_NAMES,
    Scenario,
    Zonotope,
    apply_compound_intervention,
    affine_map,
    compute_I_theta,
    zonotopes_intersect,
)
from learned_surrogate.dataset import D_MAX, P_MAX, _edge_attr_from_Ff
from learned_surrogate.model import ZonotopeGINE


def _build_model(model_cfg) -> nn.Module:
    """Instantiate the correct model class based on config.

    Checkpoints trained with ``readout="asymmetric"`` use ``ZonotopeGINEv2``
    (from model_new.py); all others use the standard ``ZonotopeGINE``.
    """
    if getattr(model_cfg, "readout", "mean") == "asymmetric":
        try:
            from learned_surrogate.model_new import ZonotopeGINEv2
            return ZonotopeGINEv2(model_cfg)
        except ImportError:
            pass  # fall back to ZonotopeGINE if model_new not available
    return ZonotopeGINE(model_cfg)

# ---------------------------------------------------------------------------
# Parameter box-constraint tensors  (scale_factor, center_delta, corr_strength)
# ---------------------------------------------------------------------------

_THETA_LB = torch.tensor([b[0] for b in PARAM_BOUNDS], dtype=torch.float32)
_THETA_UB = torch.tensor([b[1] for b in PARAM_BOUNDS], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_runtime_device(device: str) -> str:
    """Resolve the requested torch device to one that is available now."""
    from learned_surrogate.config import resolve_device

    resolved = resolve_device(device)
    if resolved != device:
        print(
            f"WARNING: requested device '{device}' is unavailable; "
            f"falling back to '{resolved}'.",
            flush=True,
        )
    return resolved

def load_mfmc_params(data_root: str | Path, scenario_idx: int) -> dict:
    """Compute MFMC control-variate parameters from the training NPZ.

    Uses the full training dataset to estimate:
        α    = Cov(I_MC, I_AABB) / Var(I_AABB)   — optimal coefficient
        μ_AABB = mean(I_AABB)                       — control variate mean

    At verification time the corrected estimate is then free:
        I_MFMC(θ') = I_MC_lowN(θ') + α · (μ_AABB − I_AABB(θ'))

    The covariance fit follows the MATLAB MFMC gate as closely as possible:
    only intersecting source/target pairs are used in the estimate.

    Args:
        data_root:    Path to ``data/surrogate/``.
        scenario_idx: Integer scenario index (1-based).

    Returns:
        dict with keys ``alpha``, ``mu_aabb``, ``rho``, ``variance_reduction``.
        Falls back to ``alpha=0`` (plain MC) if AABB has no variance.
    """
    data_root = Path(data_root)
    matches = sorted(data_root.glob(f"scenario_S{scenario_idx:02d}*"))
    if not matches:
        matches = sorted(data_root.glob(f"scenario_S{scenario_idx}*"))
    if not matches:
        raise FileNotFoundError(
            f"No scenario directory for index {scenario_idx} in {data_root}"
        )
    npz = np.load(matches[0] / "train_inconsistency.npz")
    meta = json.loads((matches[0] / "meta.json").read_text())

    I_mc   = npz["I_theta_mc"].astype(np.float64)
    I_aabb = npz["I_theta_aabb"].astype(np.float64)

    dim = int(meta["dim"])
    tgt_center = np.array(meta["target_center"], dtype=np.float64)[:dim]
    tgt_generators = np.array(meta["target_generators"], dtype=np.float64)[:dim, :]
    target = Zonotope(tgt_center, tgt_generators)
    F = np.array(meta["F"], dtype=np.float64)
    f = np.array(meta["f"], dtype=np.float64)

    n_samples = int(npz["n_samples"])
    src_centers = npz["source_center"].astype(np.float64)
    src_generators = npz["source_generators"].astype(np.float64)
    src_n_gens = npz["source_n_generators"].astype(np.int64)

    has_intersection = np.zeros(n_samples, dtype=bool)
    for i in range(n_samples):
        n_gen = int(src_n_gens[i])
        src = Zonotope(
            src_centers[i][:dim],
            src_generators[i][:dim, :n_gen],
        )
        prop = affine_map(src, F, f)
        has_intersection[i] = zonotopes_intersect(prop, target)

    # Match the MATLAB gate: fit only on intersecting pairs with finite MC/AABB.
    mask = has_intersection & np.isfinite(I_mc) & np.isfinite(I_aabb)
    I_mc, I_aabb = I_mc[mask], I_aabb[mask]

    if len(I_mc) < 3:
        return {"alpha": 0.0, "mu_aabb": float("nan"), "rho": 0.0, "variance_reduction": 0.0}

    mu_aabb  = float(I_aabb.mean())
    var_aabb = float(np.var(I_aabb, ddof=1))

    if var_aabb < 1e-12:
        return {"alpha": 0.0, "mu_aabb": mu_aabb, "rho": 0.0, "variance_reduction": 0.0}

    cov      = float(np.cov(I_mc, I_aabb, ddof=1)[0, 1])
    alpha    = cov / var_aabb
    rho      = cov / (float(np.std(I_mc, ddof=1)) * float(np.std(I_aabb, ddof=1)) + 1e-12)
    var_red  = float(rho ** 2)

    return {
        "alpha":              alpha,
        "mu_aabb":            mu_aabb,
        "rho":                rho,
        "variance_reduction": var_red,
    }


def load_scenario_scale(data_root: str | Path, scenario_idx: int) -> float:
    """Load the per-scenario normalization scale from the training-data NPZ.

    The scale is ``max(|source_center|, |source_generators|,
    |target_center|, |target_generators|)`` over all training samples,
    as computed in ``InconsistencyDataset``.

    Args:
        data_root: Path to ``data/surrogate/`` directory.
        scenario_idx: Integer scenario index (1-based).

    Returns:
        Normalization scale (positive float).
    """
    data_root = Path(data_root)
    matches = sorted(data_root.glob(f"scenario_S{scenario_idx:02d}*"))
    if not matches:
        matches = sorted(data_root.glob(f"scenario_S{scenario_idx}*"))
    if not matches:
        raise FileNotFoundError(
            f"No scenario directory for index {scenario_idx} in {data_root}"
        )
    s_dir = matches[0]
    npz = np.load(s_dir / "train_inconsistency.npz")
    meta = json.loads((s_dir / "meta.json").read_text())

    n = int(npz["n_samples"])
    tgt_c = np.array(meta["target_center"], dtype=np.float32)
    tgt_G = np.array(meta["target_generators"], dtype=np.float32)

    scale = float(max(
        float(np.abs(npz["source_center"][:n]).max()),
        float(np.abs(npz["source_generators"][:n]).max()),
        float(np.abs(tgt_c).max()) if tgt_c.size > 0 else 0.0,
        float(np.abs(tgt_G).max()) if tgt_G.size > 0 else 0.0,
        1e-8,
    ))
    return scale


def _pad_to_dmax(v: np.ndarray) -> np.ndarray:
    """Pad 1-D array to length D_MAX with zeros."""
    out = np.zeros(D_MAX, dtype=np.float32)
    out[: min(len(v), D_MAX)] = v[: D_MAX]
    return out


def _pad_generators_np(G: np.ndarray) -> np.ndarray:
    """(D, P) → (P_MAX, D_MAX), transposed and zero-padded."""
    d, p = G.shape
    p = min(p, P_MAX)
    out = np.zeros((P_MAX, D_MAX), dtype=np.float32)
    out[:p, :d] = G[:d, :p].T
    return out


def _gen_mask_np(n_gen: int) -> np.ndarray:
    mask = np.zeros(P_MAX, dtype=bool)
    mask[: min(n_gen, P_MAX)] = True
    return mask


# ---------------------------------------------------------------------------
# Differentiable intervention
# ---------------------------------------------------------------------------

class DifferentiableIntervention(nn.Module):
    """Apply θ = [s_u, Δc_u, R_u] to a base zonotope in a differentiable way.

    Produces PyG node features for node 0 (the intervened source), compatible
    with ``ZonotopeEncoder``.

    The three interventions exactly mirror ``apply_compound_intervention`` in
    causal_engine.py:

        1. Scale:       G_new = s_u * G_base
        2. Shift:       c_new = c_base + Δc_u   (scalar added to every dim)
        3. Correlate:   g_corr = R_u * G_new.sum(dim=1)   (appended column)

    We always include the correlation generator slot in the features (mask = True)
    so that gradients w.r.t. R_u flow continuously.  When R_u → 0 the generator
    value → 0, which is consistent with training samples that had R_u = 0 and
    no extra generator appended (the zero vector has negligible effect after
    the generator MLP).
    """

    def __init__(
        self,
        base_c: np.ndarray,
        base_G: np.ndarray,
        norm_scale: float,
        d_max: int = D_MAX,
        p_max: int = P_MAX,
    ):
        super().__init__()
        D, P = base_G.shape
        self.D = D
        self.P_base = P
        self.d_max = d_max
        self.p_max = p_max
        self.norm_scale = norm_scale

        # Number of generators after correlation (+1 for the extra column)
        self.n_gen_total = min(P + 1, p_max)

        self.register_buffer(
            "base_c", torch.tensor(base_c, dtype=torch.float32)
        )
        self.register_buffer(
            "base_G", torch.tensor(base_G, dtype=torch.float32)
        )

        # Static mask for source node (always includes the correlation slot)
        mask = _gen_mask_np(self.n_gen_total)
        self.register_buffer("gen_mask", torch.tensor(mask, dtype=torch.bool))

    def forward(
        self, theta: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute node-0 features from θ.

        Args:
            theta: (3,) float32 – [s_u, Δc_u, R_u]

        Returns:
            x_center:         (1, d_max)   – normalized, padded center
            x_generators:     (1, p_max, d_max) – normalized, padded generators
            x_generator_mask: (1, p_max)   – bool mask (no grad)
        """
        s_u = theta[0]
        dc = theta[1]
        R_u = theta[2]

        # 1. Scale generators
        G_scaled = s_u * self.base_G  # (D, P_base)

        # 2. Shift center (uniform across all dims)
        c_int = self.base_c + dc  # (D,)

        # 3. Append correlation generator: sum of scaled columns * R_u
        g_corr = R_u * G_scaled.sum(dim=1, keepdim=True)  # (D, 1)

        # Concatenate and truncate to p_max columns
        G_int = torch.cat([G_scaled, g_corr], dim=1)  # (D, P_base+1)
        if G_int.shape[1] > self.p_max:
            G_int = G_int[:, : self.p_max]

        # Normalize
        c_norm = c_int / self.norm_scale  # (D,)
        G_norm = G_int / self.norm_scale  # (D, P_use)

        # Transpose generators: (D, P) → (P, D)
        G_T = G_norm.T  # (P_use, D)

        # Pad center to d_max
        c_pad = torch.zeros(self.d_max, dtype=theta.dtype, device=theta.device)
        c_pad[: self.D] = c_norm
        x_center = c_pad.unsqueeze(0)  # (1, d_max)

        # Pad generators to (p_max, d_max)
        P_use = G_T.shape[0]
        G_pad = torch.zeros(
            self.p_max, self.d_max, dtype=theta.dtype, device=theta.device
        )
        G_pad[:P_use, : self.D] = G_T
        x_generators = G_pad.unsqueeze(0)  # (1, p_max, d_max)

        x_mask = self.gen_mask.unsqueeze(0)  # (1, p_max)
        return x_center, x_generators, x_mask


# ---------------------------------------------------------------------------
# Single-graph Data construction
# ---------------------------------------------------------------------------

def _build_single_graph(
    intervention: DifferentiableIntervention,
    theta: torch.Tensor,
    tgt_center: torch.Tensor,
    tgt_generators: torch.Tensor,
    tgt_mask: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
) -> Data:
    """Build a differentiable PyG Data object for one MMS graph.

    Node 0 features are computed from theta via DifferentiableIntervention.
    Node 1 (target) features are fixed constants.
    """
    x_c_src, x_gen_src, x_mask_src = intervention(theta)

    x_center = torch.cat([x_c_src, tgt_center], dim=0)          # (2, D_MAX)
    x_generators = torch.cat([x_gen_src, tgt_generators], dim=0) # (2, P_MAX, D_MAX)
    x_generator_mask = torch.cat([x_mask_src, tgt_mask], dim=0)  # (2, P_MAX)

    data = Data(
        x_center=x_center,
        x_generators=x_generators,
        x_generator_mask=x_generator_mask,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=2,
    )
    data.batch = torch.zeros(2, dtype=torch.long, device=theta.device)
    return data


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CounterfactualResult:
    """Output of a single counterfactual query."""

    theta_star: np.ndarray       # Original inconsistent configuration (3,)
    theta_prime: np.ndarray      # Nearest consistent configuration (3,)
    delta_theta: np.ndarray      # theta_prime - theta_star (3,)
    I_surrogate_star: float      # Surrogate at θ*
    I_surrogate_prime: float     # Surrogate at θ'
    I_mc_prime: Optional[float]  # MC verification at θ' (None if not run)
    gamma: float                 # MC/MFMC acceptance threshold
    n_iter: int                  # Optimization iterations used
    converged: bool              # Surrogate reached ≤ γ
    valid: bool                  # MC confirmed: I(θ') ≤ γ
    wall_time_s: float           # Elapsed optimization time
    scenario_idx: Optional[int] = None
    selection_mode: str = "surrogate"
    candidate_count: int = 1
    selection_reason: Optional[str] = None
    gamma_surrogate: Optional[float] = None
    param_names: List[str] = field(default_factory=lambda: list(PARAM_NAMES))

    def dominant_fix_param(self) -> str:
        """Parameter with the largest absolute change."""
        idx = int(np.argmax(np.abs(self.delta_theta)))
        return self.param_names[idx]

    def to_dict(self) -> dict:
        return {
            "scenario_idx": self.scenario_idx,
            "theta_star": self.theta_star.tolist(),
            "theta_prime": self.theta_prime.tolist(),
            "delta_theta": self.delta_theta.tolist(),
            "I_surrogate_star": self.I_surrogate_star,
            "I_surrogate_prime": self.I_surrogate_prime,
            "I_mc_prime": self.I_mc_prime,
            "gamma": self.gamma,
            "gamma_surrogate": self.gamma_surrogate,
            "n_iter": self.n_iter,
            "converged": self.converged,
            "valid": self.valid,
            "wall_time_s": self.wall_time_s,
            "selection_mode": self.selection_mode,
            "candidate_count": self.candidate_count,
            "selection_reason": self.selection_reason,
            "dominant_fix_param": self.dominant_fix_param(),
            "delta_theta_by_param": dict(
                zip(self.param_names, self.delta_theta.tolist())
            ),
        }


# ---------------------------------------------------------------------------
# Core optimizer
# ---------------------------------------------------------------------------

class CounterfactualOptimizer:
    """Projected-gradient optimizer for counterfactual θ' given frozen GNN.

    Minimizes the penalized objective:
        L(θ) = ||θ - θ*||² + λ · max(0, Ĩ(θ) - γ)²

    with λ escalating every ``escalation_freq`` steps when the constraint
    Ĩ(θ) ≤ γ is not yet satisfied.  After each Adam step, θ is projected
    back into the box [θ_lb, θ_ub].

    Args:
        model:           Trained ZonotopeGINE (will be set to eval, frozen).
        scenario:        MMS scenario (source + target + UPR F, f).
        norm_scale:      Per-scenario normalization scale from training data.
        gamma:           Consistency threshold (default 0.3).
        lambda_init:     Initial penalty weight.
        lambda_scale:    Multiplicative escalation factor per step.
        lambda_max:      Upper bound on λ.
        lr:              Adam learning rate for θ.
        max_iter:        Maximum optimization iterations.
        tol:             Convergence tolerance on ||Δθ|| per step.
        patience:        Iterations without surrogate improvement before stop.
        escalation_freq: Penalty escalation every N iterations.
        device:          Torch device string.
    """

    def __init__(
        self,
        model: ZonotopeGINE,
        scenario: Scenario,
        norm_scale: float,
        gamma: float = 0.3,
        gamma_verify: Optional[float] = None,
        lambda_init: float = 10.0,
        lambda_scale: float = 5.0,
        lambda_max: float = 1e5,
        lr: float = 0.01,
        max_iter: int = 500,
        tol: float = 1e-4,
        patience: int = 30,
        escalation_freq: int = 50,
        device: str = "cpu",
        mfmc_params: Optional[dict] = None,
    ):
        self.model = model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.scenario = scenario
        self.norm_scale = norm_scale
        self.gamma = gamma
        self.gamma_verify = gamma if gamma_verify is None else gamma_verify
        self.lambda_init = lambda_init
        self.lambda_scale = lambda_scale
        self.lambda_max = lambda_max
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.patience = patience
        self.escalation_freq = escalation_freq
        self.device = torch.device(device)
        # MFMC control-variate params (None → fall back to plain MC)
        self.mfmc_params = mfmc_params

        # Box-constraint tensors on device
        self.theta_lb = _THETA_LB.to(self.device)
        self.theta_ub = _THETA_UB.to(self.device)
        self.theta_span = (self.theta_ub - self.theta_lb).clamp_min(1e-8)

        # Pre-compute fixed target-node features
        self._tgt_center, self._tgt_gen, self._tgt_mask = (
            self._encode_target()
        )

        # Pre-compute edge attributes (UPR source -> target)
        ea_np = _edge_attr_from_Ff(
            scenario.F, scenario.f,
            upr_type=scenario.metadata.get("upr_type", "parametric"),
        ).reshape(1, -1)
        self._edge_attr = torch.tensor(
            ea_np, dtype=torch.float32, device=self.device
        )
        self._edge_index = torch.tensor(
            [[0], [1]], dtype=torch.long, device=self.device
        )

    def _normalized_sq_distance(
        self,
        theta: torch.Tensor,
        theta_ref: torch.Tensor,
    ) -> torch.Tensor:
        """Squared L2 distance in parameter coordinates normalized to box width."""
        return (((theta - theta_ref) / self.theta_span) ** 2).sum()

    def _proximity_l2(
        self,
        theta_np: np.ndarray,
        theta_star_np: np.ndarray,
        normalized: bool,
    ) -> float:
        delta = np.asarray(theta_np, dtype=np.float32) - np.asarray(
            theta_star_np, dtype=np.float32
        )
        if normalized:
            span = self.theta_span.detach().cpu().numpy().astype(np.float32)
            delta = delta / span
        return float(np.linalg.norm(delta))

    def _proximity_sq(
        self,
        theta_np: np.ndarray,
        theta_star_np: np.ndarray,
        normalized: bool,
    ) -> float:
        """Squared L2 distance in raw or normalized parameter space."""
        delta = np.asarray(theta_np, dtype=np.float32) - np.asarray(
            theta_star_np, dtype=np.float32
        )
        if normalized:
            span = self.theta_span.detach().cpu().numpy().astype(np.float32)
            delta = delta / span
        return float(np.dot(delta, delta))

    def _constraint_rank_key(
        self,
        score: float,
        proximity_sq: float,
        gamma: float,
    ) -> tuple[float, float, float, float, float]:
        """Rank candidates by feasibility, then penalized residual, then closeness."""
        violation = max(0.0, float(score) - float(gamma))
        if violation <= 0.0:
            return (0.0, proximity_sq, violation, proximity_sq, float(score))
        penalized = proximity_sq + self.lambda_init * violation**2
        return (1.0, penalized, violation, proximity_sq, float(score))

    def _baseline_theta(self) -> np.ndarray:
        from causal_engine import PARAM_BASELINES

        return np.array(
            [
                PARAM_BASELINES["scale_factor"],
                PARAM_BASELINES["center_delta"],
                PARAM_BASELINES["correlation_strength"],
            ],
            dtype=np.float32,
        )

    def _candidate_starts(
        self,
        theta_star_np: np.ndarray,
        n_random_starts: int,
        seed: int,
    ) -> List[np.ndarray]:
        """Generate diverse start points for hybrid counterfactual search."""
        rng = np.random.default_rng(seed)
        theta_star_np = np.asarray(theta_star_np, dtype=np.float32)
        baseline = self._baseline_theta()
        starts = [theta_star_np, baseline, 0.5 * (theta_star_np + baseline)]

        for idx in range(len(theta_star_np)):
            reset = theta_star_np.copy()
            reset[idx] = baseline[idx]
            starts.append(reset)

        for _ in range(max(0, n_random_starts)):
            mix = rng.uniform(0.15, 0.85, size=theta_star_np.shape).astype(np.float32)
            starts.append(theta_star_np + mix * (baseline - theta_star_np))

        starts.append(self.theta_lb.detach().cpu().numpy())
        starts.append(self.theta_ub.detach().cpu().numpy())

        unique = []
        seen = set()
        for start in starts:
            clipped = np.clip(
                np.asarray(start, dtype=np.float32),
                self.theta_lb.detach().cpu().numpy(),
                self.theta_ub.detach().cpu().numpy(),
            )
            key = tuple(np.round(clipped.astype(np.float64), 6).tolist())
            if key in seen:
                continue
            seen.add(key)
            unique.append(clipped)
        return unique

    def _select_verified_candidate(
        self,
        candidates: List[CounterfactualResult],
        theta_star_np: np.ndarray,
        normalized: bool,
    ) -> Tuple[CounterfactualResult, str]:
        """Choose the MC/MFMC-verified winner from the reranked candidates."""
        valid_candidates = [c for c in candidates if c.valid]
        if valid_candidates:
            best = min(
                valid_candidates,
                key=lambda c: (
                    self._proximity_l2(c.theta_prime, theta_star_np, normalized),
                    c.I_mc_prime if c.I_mc_prime is not None else float("inf"),
                    c.I_surrogate_prime,
                ),
            )
            return best, "valid_min_proximity"

        best = min(
            candidates,
            key=lambda c: (
                self._constraint_rank_key(
                    c.I_mc_prime if c.I_mc_prime is not None else float("inf"),
                    self._proximity_sq(c.theta_prime, theta_star_np, normalized),
                    self.gamma_verify,
                ),
                c.I_surrogate_prime,
            ),
        )
        return best, "best_penalized_verified_objective"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_target(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode fixed target zonotope as PyG node features (no grad)."""
        Z_t = self.scenario.target
        c = _pad_to_dmax(Z_t.c) / self.norm_scale
        G = _pad_generators_np(Z_t.G) / self.norm_scale
        mask = _gen_mask_np(Z_t.G.shape[1])

        tgt_c = torch.tensor(c, dtype=torch.float32, device=self.device).unsqueeze(0)
        tgt_gen = torch.tensor(G, dtype=torch.float32, device=self.device).unsqueeze(0)
        tgt_mask = torch.tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        return tgt_c, tgt_gen, tgt_mask

    def _build_intervention(self, theta_np: np.ndarray) -> DifferentiableIntervention:
        """Create DifferentiableIntervention from the base zonotope."""
        Z_src = self.scenario.source
        return DifferentiableIntervention(
            base_c=Z_src.c,
            base_G=Z_src.G,
            norm_scale=self.norm_scale,
            d_max=D_MAX,
            p_max=P_MAX,
        ).to(self.device)

    def _forward(
        self,
        theta: torch.Tensor,
        intervention: DifferentiableIntervention,
    ) -> torch.Tensor:
        """θ → Ĩ(θ) via the frozen GNN surrogate."""
        data = _build_single_graph(
            intervention=intervention,
            theta=theta,
            tgt_center=self._tgt_center,
            tgt_generators=self._tgt_gen,
            tgt_mask=self._tgt_mask,
            edge_index=self._edge_index,
            edge_attr=self._edge_attr,
        )
        pred = self.model.forward_inconsistency(data)  # (1, 1)
        return pred.squeeze()

    def _mfmc_verify(self, theta_np: np.ndarray, n_samples: int = 500) -> float:
        """Verify the counterfactual via MFMC (preferred) or plain MC (fallback).

        MFMC formula:
            I_MFMC(θ') = clip(I_MC_lowN(θ') + α·(μ_AABB − I_AABB(θ')), 0, 1)

        α and μ_AABB come from the training dataset (precomputed once).
        With n_samples=500 + free AABB, MFMC matches accuracy of plain MC(2000).
        Falls back to plain MC(n_samples) when MFMC params are unavailable.
        """
        from causal_engine import compute_I_theta_aabb  # noqa

        theta_dict = dict(zip(PARAM_NAMES, theta_np.tolist()))
        Z_int = apply_compound_intervention(self.scenario.source, theta_dict)

        # Low-fidelity: AABB Jaccard (free)
        I_aabb = float(
            compute_I_theta_aabb(self.scenario, source_override=Z_int)["I_theta"]
        )
        # High-fidelity: MC with reduced sample count
        I_mc = float(
            compute_I_theta(self.scenario, source_override=Z_int, n_samples=n_samples)["I_theta"]
        )

        if self.mfmc_params is None or abs(self.mfmc_params.get("alpha", 0.0)) < 1e-12:
            return I_mc  # plain MC fallback

        alpha   = self.mfmc_params["alpha"]
        mu_aabb = self.mfmc_params["mu_aabb"]
        return float(np.clip(I_mc + alpha * (mu_aabb - I_aabb), 0.0, 1.0))

    def _evaluate_theta_np(
        self,
        theta_np: np.ndarray,
        theta_star_np: np.ndarray,
        intervention: DifferentiableIntervention,
        normalized_proximity: bool,
    ) -> tuple[np.ndarray, float, float, float]:
        """Evaluate one candidate using the surrogate penalized objective."""
        theta_clipped = np.clip(
            np.asarray(theta_np, dtype=np.float32),
            self.theta_lb.detach().cpu().numpy(),
            self.theta_ub.detach().cpu().numpy(),
        )
        theta_t = torch.tensor(theta_clipped, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            I_pred = float(self._forward(theta_t, intervention).item())

        delta = theta_clipped - theta_star_np
        if normalized_proximity:
            delta = delta / self.theta_span.detach().cpu().numpy()
        proximity_sq = float(np.dot(delta, delta))
        objective = proximity_sq + self.lambda_init * max(0.0, I_pred - self.gamma) ** 2
        return theta_clipped, I_pred, proximity_sq, objective

    def _candidate_rank_key(
        self,
        I_pred: float,
        proximity_sq: float,
    ) -> tuple[float, float, float, float, float]:
        """Rank candidates by surrogate feasibility first, then closeness."""
        return self._constraint_rank_key(I_pred, proximity_sq, self.gamma)

    def _build_result(
        self,
        theta_star_np: np.ndarray,
        theta_prime_np: np.ndarray,
        I_star: float,
        I_prime: float,
        elapsed: float,
        scenario_idx: Optional[int],
        selection_mode: str,
        n_iter: int,
        verify_mc: bool,
        mc_samples: int,
        candidate_count: int = 1,
        selection_reason: Optional[str] = None,
    ) -> CounterfactualResult:
        """Create a counterfactual result and run optional MC/MFMC verification."""
        I_mc = self._mfmc_verify(theta_prime_np, mc_samples) if verify_mc else None
        return CounterfactualResult(
            theta_star=theta_star_np,
            theta_prime=theta_prime_np,
            delta_theta=theta_prime_np - theta_star_np,
            I_surrogate_star=I_star,
            I_surrogate_prime=I_prime,
            I_mc_prime=I_mc,
            gamma=self.gamma_verify,
            gamma_surrogate=self.gamma,
            n_iter=n_iter,
            converged=I_prime <= self.gamma,
            valid=(I_mc is not None and I_mc <= self.gamma_verify),
            wall_time_s=elapsed,
            scenario_idx=scenario_idx,
            selection_mode=selection_mode,
            candidate_count=candidate_count,
            selection_reason=selection_reason,
        )

    def _optimize_distributional(
        self,
        method: str,
        theta_star: np.ndarray,
        mc_samples: int = 2000,
        verify_mc: bool = True,
        scenario_idx: Optional[int] = None,
        normalized_proximity: bool = True,
        population_size: int = 64,
        elite_frac: float = 0.2,
        temperature: float = 0.05,
        init_sigma_frac: float = 0.2,
        seed: int = 0,
    ) -> CounterfactualResult:
        """Black-box population search using CEM or MPPI-style soft updates."""
        if method not in {"cem", "mppi"}:
            raise ValueError(f"Unsupported distributional method: {method}")

        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        theta_star_np = np.asarray(theta_star, dtype=np.float32)
        intervention = self._build_intervention(theta_star_np)

        theta_star_t = torch.tensor(theta_star_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            I_star = float(self._forward(theta_star_t, intervention).item())

        mean = np.clip(
            theta_star_np.copy(),
            self.theta_lb.detach().cpu().numpy(),
            self.theta_ub.detach().cpu().numpy(),
        )
        span = self.theta_span.detach().cpu().numpy().astype(np.float32)
        sigma = np.maximum(init_sigma_frac * span, 0.02 * span)

        best_theta, best_I, best_prox_sq, _ = self._evaluate_theta_np(
            mean,
            theta_star_np,
            intervention,
            normalized_proximity,
        )
        best_key = self._candidate_rank_key(best_I, best_prox_sq)

        final_iter = self.max_iter
        n_elite = max(2, int(np.ceil(population_size * elite_frac)))

        for it in range(self.max_iter):
            samples = rng.normal(loc=mean, scale=sigma, size=(population_size, 3)).astype(
                np.float32
            )
            samples[0] = mean
            if population_size > 1:
                samples[1] = theta_star_np

            evaluated = [
                self._evaluate_theta_np(s, theta_star_np, intervention, normalized_proximity)
                for s in samples
            ]
            thetas = np.stack([e[0] for e in evaluated])
            I_vals = np.array([e[1] for e in evaluated], dtype=np.float64)
            prox_sq = np.array([e[2] for e in evaluated], dtype=np.float64)
            objectives = np.array([e[3] for e in evaluated], dtype=np.float64)

            for theta_i, I_i, prox_i in zip(thetas, I_vals, prox_sq):
                key = self._candidate_rank_key(float(I_i), float(prox_i))
                if key < best_key:
                    best_key = key
                    best_theta = theta_i.copy()
                    best_I = float(I_i)
                    best_prox_sq = float(prox_i)

            if method == "cem":
                elite_idx = np.argsort(objectives)[:n_elite]
                elite = thetas[elite_idx]
                elite_mean = elite.mean(axis=0)
                elite_std = elite.std(axis=0) + 1e-6
                mean = 0.7 * mean + 0.3 * elite_mean
                sigma = np.maximum(0.7 * sigma + 0.3 * elite_std, 0.01 * span)
            else:
                shifted = objectives - objectives.min()
                weights = np.exp(-shifted / max(temperature, 1e-6))
                weights /= weights.sum() + 1e-12
                mean = np.sum(thetas * weights[:, None], axis=0)
                centered = thetas - mean
                sigma = np.maximum(
                    np.sqrt(np.sum((centered ** 2) * weights[:, None], axis=0)) + 1e-6,
                    0.01 * span,
                )

            mean = np.clip(
                mean.astype(np.float32),
                self.theta_lb.detach().cpu().numpy(),
                self.theta_ub.detach().cpu().numpy(),
            )
            sigma = np.minimum(sigma, span)

            if best_I <= self.gamma and np.sqrt(best_prox_sq) < self.tol:
                final_iter = it + 1
                break
        else:
            final_iter = self.max_iter

        return self._build_result(
            theta_star_np=theta_star_np,
            theta_prime_np=best_theta.astype(np.float32),
            I_star=I_star,
            I_prime=best_I,
            elapsed=time.perf_counter() - t0,
            scenario_idx=scenario_idx,
            selection_mode=method,
            n_iter=final_iter,
            verify_mc=verify_mc,
            mc_samples=mc_samples,
            candidate_count=population_size,
            selection_reason=f"{method}_population_search",
        )

    def optimize_cem(
        self,
        theta_star: np.ndarray,
        mc_samples: int = 2000,
        verify_mc: bool = True,
        scenario_idx: Optional[int] = None,
        normalized_proximity: bool = True,
        population_size: int = 64,
        elite_frac: float = 0.2,
        init_sigma_frac: float = 0.2,
        seed: int = 0,
    ) -> CounterfactualResult:
        return self._optimize_distributional(
            method="cem",
            theta_star=theta_star,
            mc_samples=mc_samples,
            verify_mc=verify_mc,
            scenario_idx=scenario_idx,
            normalized_proximity=normalized_proximity,
            population_size=population_size,
            elite_frac=elite_frac,
            init_sigma_frac=init_sigma_frac,
            seed=seed,
        )

    def optimize_mppi(
        self,
        theta_star: np.ndarray,
        mc_samples: int = 2000,
        verify_mc: bool = True,
        scenario_idx: Optional[int] = None,
        normalized_proximity: bool = True,
        population_size: int = 64,
        temperature: float = 0.05,
        init_sigma_frac: float = 0.2,
        seed: int = 0,
    ) -> CounterfactualResult:
        return self._optimize_distributional(
            method="mppi",
            theta_star=theta_star,
            mc_samples=mc_samples,
            verify_mc=verify_mc,
            scenario_idx=scenario_idx,
            normalized_proximity=normalized_proximity,
            population_size=population_size,
            temperature=temperature,
            init_sigma_frac=init_sigma_frac,
            seed=seed,
        )

    def optimize_spsa(
        self,
        theta_star: np.ndarray,
        mc_samples: int = 2000,
        verify_mc: bool = True,
        scenario_idx: Optional[int] = None,
        normalized_proximity: bool = True,
        perturb_scale: float = 0.1,
        seed: int = 0,
    ) -> CounterfactualResult:
        """Projected SPSA for zero-order counterfactual search."""
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        theta_star_np = np.asarray(theta_star, dtype=np.float32)
        intervention = self._build_intervention(theta_star_np)

        theta_star_t = torch.tensor(theta_star_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            I_star = float(self._forward(theta_star_t, intervention).item())

        current = np.clip(
            theta_star_np.copy(),
            self.theta_lb.detach().cpu().numpy(),
            self.theta_ub.detach().cpu().numpy(),
        )
        best_theta, best_I, best_prox_sq, _ = self._evaluate_theta_np(
            current,
            theta_star_np,
            intervention,
            normalized_proximity,
        )
        best_key = self._candidate_rank_key(best_I, best_prox_sq)
        span = self.theta_span.detach().cpu().numpy().astype(np.float32)

        final_iter = self.max_iter
        for it in range(self.max_iter):
            step = self.lr / ((it + 1) ** 0.602)
            ck = (perturb_scale * span) / ((it + 1) ** 0.101)
            delta = rng.choice([-1.0, 1.0], size=3).astype(np.float32)

            plus = np.clip(current + ck * delta, self.theta_lb.cpu().numpy(), self.theta_ub.cpu().numpy())
            minus = np.clip(current - ck * delta, self.theta_lb.cpu().numpy(), self.theta_ub.cpu().numpy())

            plus_eval = self._evaluate_theta_np(
                plus, theta_star_np, intervention, normalized_proximity
            )
            minus_eval = self._evaluate_theta_np(
                minus, theta_star_np, intervention, normalized_proximity
            )

            grad = (plus_eval[3] - minus_eval[3]) / (2.0 * ck * delta + 1e-8)
            current = np.clip(
                current - step * grad.astype(np.float32),
                self.theta_lb.detach().cpu().numpy(),
                self.theta_ub.detach().cpu().numpy(),
            )

            current_eval = self._evaluate_theta_np(
                current, theta_star_np, intervention, normalized_proximity
            )
            for theta_i, I_i, prox_i in (plus_eval[:3], minus_eval[:3], current_eval[:3]):
                key = self._candidate_rank_key(float(I_i), float(prox_i))
                if key < best_key:
                    best_key = key
                    best_theta = np.asarray(theta_i, dtype=np.float32).copy()
                    best_I = float(I_i)
                    best_prox_sq = float(prox_i)

            if best_I <= self.gamma and np.sqrt(best_prox_sq) < self.tol:
                final_iter = it + 1
                break
        else:
            final_iter = self.max_iter

        return self._build_result(
            theta_star_np=theta_star_np,
            theta_prime_np=best_theta.astype(np.float32),
            I_star=I_star,
            I_prime=best_I,
            elapsed=time.perf_counter() - t0,
            scenario_idx=scenario_idx,
            selection_mode="spsa",
            n_iter=final_iter,
            verify_mc=verify_mc,
            mc_samples=mc_samples,
            candidate_count=3,
            selection_reason="spsa_zero_order",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _optimize_single(
        self,
        theta_star: np.ndarray,
        theta_init: Optional[np.ndarray] = None,
        mc_samples: int = 2000,
        verify_mc: bool = True,
        scenario_idx: Optional[int] = None,
        normalized_proximity: bool = False,
        selection_mode: str = "surrogate",
    ) -> CounterfactualResult:
        """Find the nearest θ' that satisfies Ĩ(θ') ≤ γ.

        Args:
            theta_star:  (3,) query configuration.
            mc_samples:  MC samples for final verification.
            verify_mc:   Whether to run MC verification.
            scenario_idx: Scenario index (stored in result, optional).

        Returns:
            CounterfactualResult with θ', Δθ, surrogate + MC values.
        """
        t0 = time.perf_counter()
        theta_star_np = np.asarray(theta_star, dtype=np.float32)
        progress_prefix = (
            f"  [S{scenario_idx:02d}]" if scenario_idx is not None else "  [counterfactual]"
        )

        # Initial surrogate evaluation
        theta_star_t = torch.tensor(
            theta_star_np, dtype=torch.float32, device=self.device
        ).clamp(self.theta_lb, self.theta_ub)
        theta_init_np = theta_star_np if theta_init is None else np.asarray(
            theta_init, dtype=np.float32
        )
        theta_init_t = torch.tensor(
            theta_init_np, dtype=torch.float32, device=self.device
        ).clamp(self.theta_lb, self.theta_ub)

        intervention = self._build_intervention(theta_star_np)

        with torch.no_grad():
            I_star = self._forward(theta_star_t, intervention).item()

        # Already consistent according to surrogate
        if I_star <= self.gamma:
            I_mc = (
                self._mfmc_verify(theta_star_t.cpu().numpy(), mc_samples)
                if verify_mc
                else None
            )
            return CounterfactualResult(
                theta_star=theta_star_np,
                theta_prime=theta_star_t.cpu().numpy(),
                delta_theta=np.zeros(3, dtype=np.float32),
                I_surrogate_star=I_star,
                I_surrogate_prime=I_star,
                I_mc_prime=I_mc,
                gamma=self.gamma_verify,
                gamma_surrogate=self.gamma,
                n_iter=0,
                converged=True,
                valid=(I_mc is not None and I_mc <= self.gamma_verify),
                wall_time_s=time.perf_counter() - t0,
                scenario_idx=scenario_idx,
                selection_mode=selection_mode,
            )

        # Learnable θ initialized at the requested start point.
        theta = nn.Parameter(theta_init_t.clone())
        theta_star_t = theta_star_t.clone().detach()

        optimizer = torch.optim.Adam([theta], lr=self.lr)

        lam = self.lambda_init
        best_theta = theta.data.clone()
        best_I = I_star
        best_prox_sq = self._proximity_sq(
            best_theta.detach().cpu().numpy(),
            theta_star_np,
            normalized_proximity,
        )
        best_key = self._candidate_rank_key(best_I, best_prox_sq)
        no_improve = 0
        prev_theta = theta.data.clone()
        final_iter = self.max_iter
        report_every = max(25, self.max_iter // 10)

        for it in range(self.max_iter):
            optimizer.zero_grad()

            I_pred = self._forward(theta, intervention)

            # Penalized objective: proximity + constraint violation
            constraint_viol = torch.clamp(I_pred - self.gamma, min=0.0)
            if normalized_proximity:
                proximity = self._normalized_sq_distance(theta, theta_star_t)
            else:
                proximity = (theta - theta_star_t).pow(2).sum()
            loss = proximity + lam * constraint_viol.pow(2)

            loss.backward()
            optimizer.step()

            # Project to box constraints
            with torch.no_grad():
                theta.data.clamp_(self.theta_lb, self.theta_ub)

            with torch.no_grad():
                I_val = self._forward(theta.data, intervention).item()
                prox_sq_val = self._proximity_sq(
                    theta.data.cpu().numpy(),
                    theta_star_np,
                    normalized_proximity,
                )
                cand_key = self._candidate_rank_key(I_val, prox_sq_val)

                # Keep the best point under the same constrained objective the
                # optimizer is using, so infeasible fallbacks stay query-specific.
                if cand_key < best_key:
                    best_key = cand_key
                    best_I = I_val
                    best_theta = theta.data.clone()
                    best_prox_sq = prox_sq_val
                    no_improve = 0
                else:
                    no_improve += 1

                # Check convergence
                theta_change = (theta.data - prev_theta).norm().item()
                prev_theta = theta.data.clone()

                if I_val <= self.gamma and theta_change < self.tol:
                    final_iter = it + 1
                    break

                # Patience-based early stop (surrogate not decreasing)
                if no_improve >= self.patience and I_val <= self.gamma:
                    final_iter = it + 1
                    break

                # Escalate penalty if constraint still violated
                if (it + 1) % self.escalation_freq == 0 and I_val > self.gamma:
                    lam = min(lam * self.lambda_scale, self.lambda_max)

            if it == 0 or (it + 1) % report_every == 0:
                print(
                    f"{progress_prefix} {selection_mode} iter {it + 1}/{self.max_iter} "
                    f"I={I_val:.4f} best={best_I:.4f} lam={lam:.2f} "
                    f"elapsed={time.perf_counter() - t0:.1f}s",
                    flush=True,
                )
        else:
            final_iter = self.max_iter

        theta_prime_np = best_theta.cpu().numpy()
        converged = best_I <= self.gamma

        I_mc = None
        if verify_mc:
            I_mc = self._mfmc_verify(theta_prime_np, mc_samples)

        elapsed = time.perf_counter() - t0
        return CounterfactualResult(
            theta_star=theta_star_np,
            theta_prime=theta_prime_np,
            delta_theta=theta_prime_np - theta_star_np,
            I_surrogate_star=I_star,
            I_surrogate_prime=best_I,
            I_mc_prime=I_mc,
            gamma=self.gamma_verify,
            gamma_surrogate=self.gamma,
            n_iter=final_iter,
            converged=converged,
            valid=(I_mc is not None and I_mc <= self.gamma_verify),
            wall_time_s=elapsed,
            scenario_idx=scenario_idx,
            selection_mode=selection_mode,
        )

    def optimize(
        self,
        theta_star: np.ndarray,
        mc_samples: int = 2000,
        verify_mc: bool = True,
        scenario_idx: Optional[int] = None,
        normalized_proximity: bool = False,
    ) -> CounterfactualResult:
        """Single-start surrogate-driven optimization."""
        return self._optimize_single(
            theta_star=theta_star,
            theta_init=theta_star,
            mc_samples=mc_samples,
            verify_mc=verify_mc,
            scenario_idx=scenario_idx,
            normalized_proximity=normalized_proximity,
            selection_mode="surrogate",
        )

    def optimize_hybrid(
        self,
        theta_star: np.ndarray,
        mc_samples: int = 2000,
        verify_mc: bool = True,
        scenario_idx: Optional[int] = None,
        normalized_proximity: bool = True,
        n_candidate_starts: int = 3,
        rerank_top_k: int = 4,
        seed: int = 0,
    ) -> CounterfactualResult:
        """Hybrid search: GNN proposes candidates, MC/MFMC selects the winner."""
        theta_star_np = np.asarray(theta_star, dtype=np.float32)
        starts = self._candidate_starts(
            theta_star_np=theta_star_np,
            n_random_starts=n_candidate_starts,
            seed=seed,
        )
        progress_prefix = (
            f"  [S{scenario_idx:02d}]" if scenario_idx is not None else "  [counterfactual]"
        )

        surrogate_candidates: List[CounterfactualResult] = []
        seen = set()
        for start_idx, start in enumerate(starts, start=1):
            print(
                f"{progress_prefix} hybrid candidate {start_idx}/{len(starts)} start",
                flush=True,
            )
            candidate = self._optimize_single(
                theta_star=theta_star_np,
                theta_init=start,
                mc_samples=mc_samples,
                verify_mc=False,
                scenario_idx=scenario_idx,
                normalized_proximity=normalized_proximity,
                selection_mode="hybrid_proposal",
            )
            key = tuple(np.round(candidate.theta_prime.astype(np.float64), 6).tolist())
            if key in seen:
                continue
            seen.add(key)
            surrogate_candidates.append(candidate)
            print(
                f"{progress_prefix} hybrid candidate {start_idx}/{len(starts)} "
                f"done: valid={candidate.valid} I={candidate.I_surrogate_prime:.4f} "
                f"iters={candidate.n_iter} prox={np.linalg.norm(candidate.delta_theta):.4f}",
                flush=True,
            )

        if not surrogate_candidates:
            return self.optimize(
                theta_star=theta_star_np,
                mc_samples=mc_samples,
                verify_mc=verify_mc,
                scenario_idx=scenario_idx,
                normalized_proximity=normalized_proximity,
            )

        surrogate_candidates.sort(
            key=lambda c: (
                0 if c.I_surrogate_prime <= self.gamma else 1,
                c.I_surrogate_prime,
                self._proximity_l2(c.theta_prime, theta_star_np, normalized_proximity),
            )
        )

        rerank_count = max(1, min(rerank_top_k, len(surrogate_candidates)))
        verified_candidates: List[CounterfactualResult] = []
        for rank_idx, candidate in enumerate(surrogate_candidates[:rerank_count], start=1):
            print(
                f"{progress_prefix} hybrid rerank {rank_idx}/{rerank_count} "
                f"verify I={candidate.I_surrogate_prime:.4f}",
                flush=True,
            )
            I_mc = (
                self._mfmc_verify(candidate.theta_prime, mc_samples)
                if verify_mc
                else None
            )
            verified_candidates.append(
                CounterfactualResult(
                    theta_star=candidate.theta_star,
                    theta_prime=candidate.theta_prime,
                    delta_theta=candidate.delta_theta,
                    I_surrogate_star=candidate.I_surrogate_star,
                    I_surrogate_prime=candidate.I_surrogate_prime,
                    I_mc_prime=I_mc,
                    gamma=self.gamma_verify,
                    gamma_surrogate=self.gamma,
                    n_iter=candidate.n_iter,
                    converged=candidate.converged,
                    valid=(I_mc is not None and I_mc <= self.gamma_verify),
                    wall_time_s=candidate.wall_time_s,
                    scenario_idx=candidate.scenario_idx,
                    selection_mode="hybrid",
                    candidate_count=rerank_count,
                )
            )

        winner, reason = self._select_verified_candidate(
            verified_candidates,
            theta_star_np=theta_star_np,
            normalized=normalized_proximity,
        )
        winner.selection_reason = reason
        winner.candidate_count = rerank_count
        print(
            f"{progress_prefix} hybrid winner: valid={winner.valid} "
            f"I_mc={winner.I_mc_prime if winner.I_mc_prime is not None else float('nan'):.4f} "
            f"reason={reason}",
            flush=True,
        )
        return winner

    # ------------------------------------------------------------------
    # Multi-fidelity search (Phase 2: MFMC-guided SPSA refinement)
    # ------------------------------------------------------------------

    def _mfmc_spsa_refine(
        self,
        theta_init: np.ndarray,
        theta_star_np: np.ndarray,
        n_iter: int = 40,
        mc_samples_per_eval: int = 50,
        lr: float = 0.005,
        perturb_scale: float = 0.05,
        trust_radius: Optional[float] = None,
        normalized_proximity: bool = True,
        lam_prox: float = 0.5,
        seed: int = 0,
    ) -> Tuple[np.ndarray, float]:
        """Phase-2 MFMC-guided SPSA refinement in a neighbourhood of theta_init.

        Uses the MFMC estimator (cheap, unbiased) as the objective instead of
        the biased surrogate.  Two MFMC evaluations per SPSA iteration
        (theta ± ck*delta) give a noisy gradient estimate that drives theta
        toward the true consistency boundary.

        Args:
            theta_init:         Starting point (surrogate output from Phase 1).
            theta_star_np:      Original query — kept to track proximity.
            n_iter:             SPSA iterations.
            mc_samples_per_eval: MC samples per MFMC call (low: 50 for speed).
            lr:                 SPSA step-size coefficient a_k = lr / (k+1)^0.602.
            perturb_scale:      SPSA perturbation coefficient c_k = c0 / (k+1)^0.161.
            trust_radius:       If set, project theta back into an L∞ ball of this
                                half-width around theta_init after each step.
            normalized_proximity: Use normalised L2 distance for the prox penalty.
            lam_prox:           Weight on the proximity penalty in the SPSA objective.
            seed:               RNG seed for the Bernoulli perturbation vectors.

        Returns:
            (best_theta, best_I_mfmc) — the best θ found and its MFMC value.
        """
        rng = np.random.default_rng(seed)
        lb = self.theta_lb.cpu().numpy()
        ub = self.theta_ub.cpu().numpy()
        span = (ub - lb).clip(min=1e-8)

        theta = theta_init.astype(np.float32).copy()
        best_theta = theta.copy()
        best_I = self._mfmc_verify(theta, mc_samples_per_eval)

        for k in range(n_iter):
            # Robbins–Monro sequences (SPSA standard)
            ck = perturb_scale / (k + 1) ** 0.161
            ak = lr / (k + 1) ** 0.602

            # Bernoulli ±1 perturbation vector
            delta = rng.choice([-1.0, 1.0], size=theta.shape).astype(np.float32)

            theta_p = np.clip(theta + ck * delta, lb, ub)
            theta_m = np.clip(theta - ck * delta, lb, ub)

            # MFMC evaluations at perturbed points
            I_p = self._mfmc_verify(theta_p, mc_samples_per_eval)
            I_m = self._mfmc_verify(theta_m, mc_samples_per_eval)

            # SPSA gradient of the penalised objective: I_MFMC + lam_prox * proximity
            if normalized_proximity:
                prox_p = float(np.sum(((theta_p - theta_star_np) / span) ** 2))
                prox_m = float(np.sum(((theta_m - theta_star_np) / span) ** 2))
            else:
                prox_p = float(np.sum((theta_p - theta_star_np) ** 2))
                prox_m = float(np.sum((theta_m - theta_star_np) ** 2))

            obj_p = I_p + lam_prox * prox_p
            obj_m = I_m + lam_prox * prox_m

            grad_est = (obj_p - obj_m) / (2.0 * ck + 1e-12) * delta

            # Gradient step + box projection
            theta = np.clip(theta - ak * grad_est, lb, ub).astype(np.float32)

            # Optional trust-region (L∞ ball around the Phase-1 solution)
            if trust_radius is not None:
                theta = np.clip(theta, theta_init - trust_radius, theta_init + trust_radius)
                theta = np.clip(theta, lb, ub)

            # Evaluate refined point
            I_val = self._mfmc_verify(theta, mc_samples_per_eval)
            if I_val < best_I or (I_val <= self.gamma_verify and best_I > self.gamma_verify):
                best_I = I_val
                best_theta = theta.copy()

            # Early stop if already consistent
            if best_I <= self.gamma_verify:
                break

        return best_theta, best_I

    def optimize_multifidelity(
        self,
        theta_star: np.ndarray,
        mc_samples: int = 500,
        verify_mc: bool = True,
        scenario_idx: Optional[int] = None,
        normalized_proximity: bool = True,
        n_candidate_starts: int = 3,
        rerank_top_k: int = 4,
        mf_spsa_iter: int = 40,
        mf_mc_per_eval: int = 50,
        seed: int = 0,
    ) -> CounterfactualResult:
        """Multi-fidelity counterfactual search.

        Phase 1 — Hybrid surrogate search (identical to optimize_hybrid):
            * N surrogate-gradient starts → top-K candidates by surrogate I.

        Phase 2 — MFMC-guided SPSA refinement:
            * Each top-K candidate is refined with SPSA using the MFMC
              estimator (unbiased, cheap) as the objective.  This corrects
              the surrogate-MC calibration gap without requiring retraining.

        Phase 3 — Final MC/MFMC verification of the refined candidates.

        The ``search_mode="hybrid"`` path is completely unchanged; this is an
        additive mode that can be selected independently.
        """
        theta_star_np = np.asarray(theta_star, dtype=np.float32)
        t0 = time.perf_counter()

        progress_prefix = (
            f"  [S{scenario_idx:02d}]" if scenario_idx is not None else "  [counterfactual]"
        )

        # ------------------------------------------------------------------
        # Phase 1: surrogate candidates (same as hybrid, no MC verification)
        # ------------------------------------------------------------------
        starts = self._candidate_starts(
            theta_star_np=theta_star_np,
            n_random_starts=n_candidate_starts,
            seed=seed,
        )

        surrogate_candidates: List[CounterfactualResult] = []
        seen: set = set()
        for start_idx, start in enumerate(starts, start=1):
            candidate = self._optimize_single(
                theta_star=theta_star_np,
                theta_init=start,
                mc_samples=mc_samples,
                verify_mc=False,
                scenario_idx=scenario_idx,
                normalized_proximity=normalized_proximity,
                selection_mode="mf_proposal",
            )
            key = tuple(np.round(candidate.theta_prime.astype(np.float64), 6).tolist())
            if key in seen:
                continue
            seen.add(key)
            surrogate_candidates.append(candidate)
            print(
                f"{progress_prefix} mf phase1 {start_idx}/{len(starts)}: "
                f"I_surr={candidate.I_surrogate_prime:.4f} "
                f"prox={np.linalg.norm(candidate.delta_theta):.4f}",
                flush=True,
            )

        if not surrogate_candidates:
            # Fallback: plain single surrogate run + final verification
            return self.optimize(
                theta_star=theta_star_np,
                mc_samples=mc_samples,
                verify_mc=verify_mc,
                scenario_idx=scenario_idx,
                normalized_proximity=normalized_proximity,
            )

        # Sort by surrogate inconsistency (best first)
        surrogate_candidates.sort(
            key=lambda c: (
                0 if c.I_surrogate_prime <= self.gamma else 1,
                c.I_surrogate_prime,
                self._proximity_l2(c.theta_prime, theta_star_np, normalized_proximity),
            )
        )

        # ------------------------------------------------------------------
        # Phase 2: MFMC-SPSA refinement of top-K candidates
        # ------------------------------------------------------------------
        rerank_count = max(1, min(rerank_top_k, len(surrogate_candidates)))
        refined_candidates: List[CounterfactualResult] = []

        for rank_idx, candidate in enumerate(surrogate_candidates[:rerank_count], start=1):
            print(
                f"{progress_prefix} mf phase2 refine {rank_idx}/{rerank_count} "
                f"(I_surr={candidate.I_surrogate_prime:.4f})",
                flush=True,
            )
            refined_theta, I_mfmc_refined = self._mfmc_spsa_refine(
                theta_init=candidate.theta_prime,
                theta_star_np=theta_star_np,
                n_iter=mf_spsa_iter,
                mc_samples_per_eval=mf_mc_per_eval,
                normalized_proximity=normalized_proximity,
                seed=seed + rank_idx,
            )

            # ------------------------------------------------------------------
            # Phase 3: final high-fidelity verification of refined point
            # ------------------------------------------------------------------
            I_mc_final = (
                self._mfmc_verify(refined_theta, mc_samples)
                if verify_mc
                else None
            )
            delta = refined_theta - theta_star_np
            I_mc_final_str = f"{I_mc_final:.4f}" if I_mc_final is not None else "n/a"
            print(
                f"{progress_prefix} mf phase3 verify {rank_idx}/{rerank_count}: "
                f"I_mfmc_refined={I_mfmc_refined:.4f} "
                f"I_mc_final={I_mc_final_str} "
                f"valid={I_mc_final is not None and I_mc_final <= self.gamma_verify}",
                flush=True,
            )
            refined_candidates.append(
                CounterfactualResult(
                    theta_star=theta_star_np,
                    theta_prime=refined_theta,
                    delta_theta=delta,
                    I_surrogate_star=candidate.I_surrogate_star,
                    I_surrogate_prime=candidate.I_surrogate_prime,
                    I_mc_prime=I_mc_final,
                    gamma=self.gamma_verify,
                    gamma_surrogate=self.gamma,
                    n_iter=candidate.n_iter + mf_spsa_iter,
                    converged=I_mfmc_refined <= self.gamma_verify,
                    valid=(I_mc_final is not None and I_mc_final <= self.gamma_verify),
                    wall_time_s=time.perf_counter() - t0,
                    scenario_idx=scenario_idx,
                    selection_mode="multifidelity",
                    candidate_count=rerank_count,
                )
            )

        winner, reason = self._select_verified_candidate(
            refined_candidates,
            theta_star_np=theta_star_np,
            normalized=normalized_proximity,
        )
        winner.selection_reason = reason
        winner.wall_time_s = time.perf_counter() - t0
        print(
            f"{progress_prefix} mf winner: valid={winner.valid} "
            f"I_mc={winner.I_mc_prime if winner.I_mc_prime is not None else float('nan'):.4f} "
            f"reason={reason}",
            flush=True,
        )
        return winner


# ---------------------------------------------------------------------------
# Naive baseline
# ---------------------------------------------------------------------------

def naive_baseline(
    scenario: Scenario,
    theta_star: np.ndarray,
    gamma: float = 0.3,
    highest_sobol_param: str = "center_delta",
    mc_samples: int = 2000,
) -> dict:
    """Set the highest-Sobol parameter to its baseline value and check I(θ').

    The default ``highest_sobol_param = 'center_delta'`` reflects the global
    finding S₁(Δcᵤ) ≈ 0.69 from the sensitivity analysis.  The baseline value
    is taken from ``PARAM_BASELINES`` in causal_engine (Δcᵤ_baseline = 0.0,
    s_u_baseline = 1.0, R_u_baseline = 0.0).

    Args:
        scenario:             MMS scenario.
        theta_star:           (3,) inconsistent query.
        gamma:                Consistency threshold.
        highest_sobol_param:  Which parameter to zero out.
        mc_samples:           MC samples for verification.

    Returns:
        dict with theta_prime, delta_theta, I_mc, valid.
    """
    from causal_engine import PARAM_BASELINES  # noqa: F401

    theta_star_np = np.asarray(theta_star, dtype=np.float32)
    theta_prime = theta_star_np.copy()

    # Locate parameter index
    param_idx = PARAM_NAMES.index(highest_sobol_param)
    baseline_vals = [
        PARAM_BASELINES["scale_factor"],
        PARAM_BASELINES["center_delta"],
        PARAM_BASELINES["correlation_strength"],
    ]
    theta_prime[param_idx] = float(baseline_vals[param_idx])

    # Clamp to box constraints
    lb = np.array([b[0] for b in PARAM_BOUNDS], dtype=np.float32)
    ub = np.array([b[1] for b in PARAM_BOUNDS], dtype=np.float32)
    theta_prime = np.clip(theta_prime, lb, ub)

    # MC verification
    theta_dict = dict(zip(PARAM_NAMES, theta_prime.tolist()))
    Z_int = apply_compound_intervention(scenario.source, theta_dict)
    mc_result = compute_I_theta(scenario, source_override=Z_int, n_samples=mc_samples)
    I_mc = float(mc_result["I_theta"])

    return {
        "theta_star": theta_star_np.tolist(),
        "theta_prime": theta_prime.tolist(),
        "delta_theta": (theta_prime - theta_star_np).tolist(),
        "zeroed_param": highest_sobol_param,
        "I_mc": I_mc,
        "valid": I_mc <= gamma,
        "gamma": gamma,
    }


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

@dataclass
class CounterfactualEvalConfig:
    """Configuration for batch counterfactual evaluation."""

    gamma: float = 0.3
    n_queries: int = 50
    mc_samples_verify: int = 2000
    mc_samples_screen: int = 500   # fast screening to identify inconsistent θ*
    seed: int = 0
    # Optimizer hyperparameters
    lambda_init: float = 10.0
    lambda_scale: float = 5.0
    lambda_max: float = 1e5
    lr: float = 0.01
    max_iter: int = 500
    tol: float = 1e-4
    patience: int = 30
    escalation_freq: int = 50
    # Baseline
    run_baseline: bool = True
    highest_sobol_param: str = "center_delta"
    # Optional measurement-driven query source
    measurement_root: Optional[str] = None
    max_pre_inconsistency: Optional[float] = None
    # Cap starting inconsistency when sampling random queries (no measurement root)
    max_query_inconsistency: Optional[float] = None
    # Search mode
    search_mode: str = "hybrid"
    n_candidate_starts: int = 3
    rerank_top_k: int = 4
    normalize_proximity: bool = True
    gamma_surrogate: Optional[float] = None
    population_size: int = 64
    elite_fraction: float = 0.2
    mppi_temperature: float = 0.05
    init_sigma_frac: float = 0.2
    spsa_perturb_scale: float = 0.1
    # Multi-fidelity (phase-2 SPSA) parameters
    mf_spsa_iter: int = 40
    mf_mc_per_eval: int = 50


def _run_counterfactual_search(
    opt: CounterfactualOptimizer,
    cfg: CounterfactualEvalConfig,
    theta_star: np.ndarray,
    scenario_idx: int,
    query_seed: int,
) -> CounterfactualResult:
    """Dispatch to the selected counterfactual search method."""
    if cfg.search_mode == "hybrid":
        return opt.optimize_hybrid(
            theta_star=theta_star,
            mc_samples=cfg.mc_samples_verify,
            verify_mc=True,
            scenario_idx=scenario_idx,
            normalized_proximity=cfg.normalize_proximity,
            n_candidate_starts=cfg.n_candidate_starts,
            rerank_top_k=cfg.rerank_top_k,
            seed=query_seed,
        )
    if cfg.search_mode == "surrogate":
        return opt.optimize(
            theta_star=theta_star,
            mc_samples=cfg.mc_samples_verify,
            verify_mc=True,
            scenario_idx=scenario_idx,
            normalized_proximity=cfg.normalize_proximity,
        )
    if cfg.search_mode == "cem":
        return opt.optimize_cem(
            theta_star=theta_star,
            mc_samples=cfg.mc_samples_verify,
            verify_mc=True,
            scenario_idx=scenario_idx,
            normalized_proximity=cfg.normalize_proximity,
            population_size=cfg.population_size,
            elite_frac=cfg.elite_fraction,
            init_sigma_frac=cfg.init_sigma_frac,
            seed=query_seed,
        )
    if cfg.search_mode == "mppi":
        return opt.optimize_mppi(
            theta_star=theta_star,
            mc_samples=cfg.mc_samples_verify,
            verify_mc=True,
            scenario_idx=scenario_idx,
            normalized_proximity=cfg.normalize_proximity,
            population_size=cfg.population_size,
            temperature=cfg.mppi_temperature,
            init_sigma_frac=cfg.init_sigma_frac,
            seed=query_seed,
        )
    if cfg.search_mode == "spsa":
        return opt.optimize_spsa(
            theta_star=theta_star,
            mc_samples=cfg.mc_samples_verify,
            verify_mc=True,
            scenario_idx=scenario_idx,
            normalized_proximity=cfg.normalize_proximity,
            perturb_scale=cfg.spsa_perturb_scale,
            seed=query_seed,
        )
    if cfg.search_mode == "multifidelity":
        return opt.optimize_multifidelity(
            theta_star=theta_star,
            mc_samples=cfg.mc_samples_verify,
            verify_mc=True,
            scenario_idx=scenario_idx,
            normalized_proximity=cfg.normalize_proximity,
            n_candidate_starts=cfg.n_candidate_starts,
            rerank_top_k=cfg.rerank_top_k,
            mf_spsa_iter=cfg.mf_spsa_iter,
            mf_mc_per_eval=cfg.mf_mc_per_eval,
            seed=query_seed,
        )
    raise ValueError(f"Unknown search mode: {cfg.search_mode}")


def _sample_inconsistent_queries(
    scenario: Scenario,
    n_queries: int,
    gamma: float,
    mc_samples: int,
    seed: int,
    max_query_inconsistency: Optional[float] = None,
) -> np.ndarray:
    """Sample θ values likely to be inconsistent (I(θ) > γ).

    Uses a quick MC screen to identify inconsistent configurations.
    Returns an (n_found, 3) array of θ* candidates.

    Args:
        max_query_inconsistency: If set, only keep queries with
            γ < I(θ*) ≤ max_query_inconsistency.  Useful for focusing
            on near-boundary queries where the surrogate is better calibrated.
    """
    lb = np.array([b[0] for b in PARAM_BOUNDS], dtype=np.float64)
    ub = np.array([b[1] for b in PARAM_BOUNDS], dtype=np.float64)
    rng = np.random.default_rng(seed)

    candidates = []
    # Oversample to find enough inconsistent ones
    batch = rng.uniform(lb, ub, size=(n_queries * 40, 3))

    for theta_row in batch:
        theta_dict = dict(zip(PARAM_NAMES, theta_row.tolist()))
        Z_int = apply_compound_intervention(scenario.source, theta_dict)
        result = compute_I_theta(
            scenario, source_override=Z_int, n_samples=mc_samples
        )
        I_val = result["I_theta"]
        if I_val > gamma:
            if max_query_inconsistency is None or I_val <= max_query_inconsistency:
                candidates.append(theta_row.astype(np.float32))
        if len(candidates) >= n_queries:
            break

    if not candidates:
        cap_txt = (
            f" and I(θ*) ≤ {max_query_inconsistency}"
            if max_query_inconsistency is not None
            else ""
        )
        raise RuntimeError(
            f"Could not find any inconsistent configurations for scenario "
            f"'{scenario.name}' with γ={gamma}{cap_txt}. "
            f"Try lowering γ, raising --max-query-inconsistency, or increasing n_queries."
        )
    return np.stack(candidates)


def _sample_inconsistent_queries_from_measurements(
    measurement_root: str | Path,
    scenario_idx: int,
    n_queries: int,
    gamma: float,
    seed: int,
    max_pre_inconsistency: Optional[float] = None,
) -> np.ndarray:
    """Load inconsistent theta* directly from stored measurement experiments."""
    measurement_root = Path(measurement_root)
    filename = f"results_scenario_{scenario_idx}.json"
    domain_hint = "CONVIDE" if scenario_idx <= 12 else "CPS"

    candidates = []
    direct_path = measurement_root / filename
    if direct_path.exists():
        candidates.append(direct_path)

    hinted_path = measurement_root / domain_hint / filename
    if hinted_path.exists():
        candidates.append(hinted_path)

    if not candidates and measurement_root.exists():
        recursive = sorted(measurement_root.rglob(filename))
        hinted_recursive = [
            p for p in recursive if p.parent.name.upper() == domain_hint
        ]
        candidates = hinted_recursive or recursive

    if not candidates:
        raise RuntimeError(
            f"No measurement file for scenario {scenario_idx} under {measurement_root}"
        )

    path = candidates[0]

    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    candidates = []
    for exp in data.get("experiments", []):
        if exp.get("intervention_type") != "compound":
            continue

        try:
            theta = np.array(
                [
                    float(exp["scale_factor"]),
                    float(exp["center_delta"]),
                    float(exp["correlation_strength"]),
                ],
                dtype=np.float32,
            )
        except (KeyError, TypeError, ValueError):
            continue

        post = (exp.get("post_state") or {}).get("inconsistency", {})
        pre = (exp.get("pre_state") or {}).get("inconsistency", {})
        post_I = post.get("I_theta")
        pre_I = pre.get("I_theta")
        if isinstance(post_I, list):
            post_I = post_I[0] if post_I else None
        if isinstance(pre_I, list):
            pre_I = pre_I[0] if pre_I else None
        if post_I is None:
            continue
        try:
            post_I = float(post_I)
            pre_I = float(pre_I) if pre_I is not None else None
        except (TypeError, ValueError):
            continue

        if post_I <= gamma:
            continue
        if max_pre_inconsistency is not None and (
            pre_I is None or pre_I > max_pre_inconsistency
        ):
            continue

        candidates.append(theta)

    if not candidates:
        limit_txt = (
            f" and pre_state I(theta) <= {max_pre_inconsistency}"
            if max_pre_inconsistency is not None
            else ""
        )
        raise RuntimeError(
            f"Could not find measured inconsistent configurations for scenario "
            f"{scenario_idx} with post_state I(theta) > {gamma}{limit_txt}."
        )

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(candidates))
    selected = [candidates[i] for i in order[: min(n_queries, len(candidates))]]
    return np.stack(selected)


def evaluate_counterfactuals(
    model: ZonotopeGINE,
    scenarios: List[Scenario],
    scenario_indices: List[int],
    data_root: str | Path,
    cfg: CounterfactualEvalConfig = None,
    device: str = "cpu",
    snapshot_path: Optional[str | Path] = None,
    snapshot_meta: Optional[dict] = None,
) -> dict:
    """Run counterfactual evaluation across a list of scenarios.

    For each scenario:
      1. Sample ``cfg.n_queries`` inconsistent configurations θ*.
      2. Run ``CounterfactualOptimizer`` to find θ'.
      3. MC-verify each θ'.
      4. (Optionally) run the naive baseline on each θ*.

    Args:
        model:            Trained ZonotopeGINE.
        scenarios:        List of Scenario objects aligned with scenario_indices.
        scenario_indices: Integer indices (1-based) for scale loading.
        data_root:        Path to ``data/surrogate/``.
        cfg:              Evaluation configuration.
        device:           Torch device.

    Returns:
        dict with per-scenario results and aggregated metrics.
    """
    if cfg is None:
        cfg = CounterfactualEvalConfig()

    all_results: Dict[int, List[CounterfactualResult]] = {}
    all_baseline: Dict[int, List[dict]] = {}
    summary_rows = []

    for scenario, s_idx in zip(scenarios, scenario_indices):
        print(f"\n[Scenario {s_idx:02d}: {scenario.name}]")

        # Load normalization scale + MFMC params
        try:
            scale = load_scenario_scale(data_root, s_idx)
            mfmc_params = load_mfmc_params(data_root, s_idx)
        except FileNotFoundError:
            print(f"  WARNING: No data for scenario {s_idx}, skipping.")
            continue

        # Sample inconsistent queries
        try:
            if cfg.measurement_root:
                queries = _sample_inconsistent_queries_from_measurements(
                    measurement_root=cfg.measurement_root,
                    scenario_idx=s_idx,
                    n_queries=cfg.n_queries,
                    gamma=cfg.gamma,
                    seed=cfg.seed + s_idx,
                    max_pre_inconsistency=cfg.max_pre_inconsistency,
                )
            else:
                queries = _sample_inconsistent_queries(
                    scenario,
                    n_queries=cfg.n_queries,
                    gamma=cfg.gamma,
                    mc_samples=cfg.mc_samples_screen,
                    seed=cfg.seed + s_idx,
                    max_query_inconsistency=cfg.max_query_inconsistency,
                )
        except RuntimeError as e:
            print(f"  WARNING: {e}")
            continue

        print(f"  Found {len(queries)} inconsistent queries.")

        # Build optimizer
        opt = CounterfactualOptimizer(
            model=model,
            scenario=scenario,
            norm_scale=scale,
            gamma=cfg.gamma if cfg.gamma_surrogate is None else cfg.gamma_surrogate,
            gamma_verify=cfg.gamma,
            lambda_init=cfg.lambda_init,
            lambda_scale=cfg.lambda_scale,
            lambda_max=cfg.lambda_max,
            lr=cfg.lr,
            max_iter=cfg.max_iter,
            tol=cfg.tol,
            patience=cfg.patience,
            escalation_freq=cfg.escalation_freq,
            device=device,
            mfmc_params=mfmc_params,
        )

        scenario_results = []
        scenario_baseline = []

        for q_idx, theta_star in enumerate(queries):
            cf = _run_counterfactual_search(
                opt=opt,
                cfg=cfg,
                theta_star=theta_star,
                scenario_idx=s_idx,
                query_seed=cfg.seed + s_idx * 1000 + q_idx,
            )
            scenario_results.append(cf)

            if cfg.run_baseline:
                bl = naive_baseline(
                    scenario=scenario,
                    theta_star=theta_star,
                    gamma=cfg.gamma,
                    highest_sobol_param=cfg.highest_sobol_param,
                    mc_samples=cfg.mc_samples_verify,
                )
                scenario_baseline.append(bl)

            status = "VALID" if cf.valid else ("SURR" if cf.converged else "FAIL")
            print(
                f"  Q{q_idx+1:03d}: Ĩ(θ*)={cf.I_surrogate_star:.3f} "
                f"→ Ĩ(θ')={cf.I_surrogate_prime:.3f} "
                f"(MC={cf.I_mc_prime:.3f}) "
                f"dom={cf.dominant_fix_param()[:3].upper()} "
                f"[{status}] {cf.n_iter}it {cf.wall_time_s:.1f}s"
            )

        all_results[s_idx] = scenario_results
        all_baseline[s_idx] = scenario_baseline

        # Per-scenario summary
        valid_rate = np.mean([r.valid for r in scenario_results])
        converge_rate = np.mean([r.converged for r in scenario_results])
        mean_prox = float(np.mean([
            np.linalg.norm(r.delta_theta) for r in scenario_results
        ]))
        mean_I_star = float(np.mean([
            r.I_surrogate_star for r in scenario_results
        ]))
        mean_I_surr_prime = float(np.mean([
            r.I_surrogate_prime for r in scenario_results
        ]))
        mean_I_prime = float(np.mean([
            r.I_mc_prime for r in scenario_results if r.I_mc_prime is not None
        ]))
        mean_surrogate_improvement = float(np.mean([
            r.I_surrogate_star - r.I_surrogate_prime for r in scenario_results
        ]))
        mean_mc_improvement = float(np.mean([
            r.I_surrogate_star - r.I_mc_prime
            for r in scenario_results if r.I_mc_prime is not None
        ]))
        surrogate_improvement_rate = float(np.mean([
            r.I_surrogate_prime < r.I_surrogate_star for r in scenario_results
        ]))
        mc_below_040_rate = float(np.mean([
            r.I_mc_prime <= 0.4 for r in scenario_results if r.I_mc_prime is not None
        ]))
        mc_below_050_rate = float(np.mean([
            r.I_mc_prime <= 0.5 for r in scenario_results if r.I_mc_prime is not None
        ]))
        mean_mc_gap_to_gamma = float(np.mean([
            r.I_mc_prime - cfg.gamma for r in scenario_results if r.I_mc_prime is not None
        ]))
        # Δθ breakdown per parameter
        delta_mat = np.stack([r.delta_theta for r in scenario_results])
        abs_delta_mean = np.mean(np.abs(delta_mat), axis=0)

        bl_valid_rate = (
            np.mean([b["valid"] for b in scenario_baseline])
            if scenario_baseline else None
        )

        row = {
            "scenario_idx": s_idx,
            "n_queries": len(scenario_results),
            "validity_rate": float(valid_rate),
            "convergence_rate": float(converge_rate),
            "mean_proximity_l2": mean_prox,
            "mean_I_surrogate_star": mean_I_star,
            "mean_I_surrogate_prime": mean_I_surr_prime,
            "mean_I_mc_prime": mean_I_prime,
            "mean_surrogate_improvement": mean_surrogate_improvement,
            "mean_mc_improvement_vs_surrogate_start": mean_mc_improvement,
            "surrogate_improvement_rate": surrogate_improvement_rate,
            "mc_below_040_rate": mc_below_040_rate,
            "mc_below_050_rate": mc_below_050_rate,
            "mean_mc_gap_to_gamma": mean_mc_gap_to_gamma,
            "mean_abs_delta_theta": {
                name: float(v)
                for name, v in zip(PARAM_NAMES, abs_delta_mean)
            },
            "dominant_param_freq": _dominant_param_freq(scenario_results),
            "baseline_validity_rate": bl_valid_rate,
        }
        summary_rows.append(row)
        print(
            f"  -> start={mean_I_star:.3f}  surr'={mean_I_surr_prime:.3f}"
            f"  mc'={mean_I_prime:.3f}  d_surr={mean_surrogate_improvement:.3f}"
            f"  d_mc={mean_mc_improvement:.3f}"
        )
        print(
            f"  -> improved={surrogate_improvement_rate:.1%}"
            f"  MC<=0.4 {mc_below_040_rate:.1%}"
            f"  MC<=0.5 {mc_below_050_rate:.1%}"
            f"  gap_to_gamma={mean_mc_gap_to_gamma:+.3f}"
        )

        print(
            f"  → validity={valid_rate:.1%}  convergence={converge_rate:.1%}"
            f"  prox={mean_prox:.4f}"
            f"  Δθ[{PARAM_NAMES[0][:3]}]={abs_delta_mean[0]:.4f}"
            f"  Δθ[{PARAM_NAMES[1][:3]}]={abs_delta_mean[1]:.4f}"
            f"  Δθ[{PARAM_NAMES[2][:3]}]={abs_delta_mean[2]:.4f}"
        )
        if bl_valid_rate is not None:
            print(f"  → baseline validity={bl_valid_rate:.1%}")

        if snapshot_path is not None:
            _write_eval_snapshot(
                snapshot_path=snapshot_path,
                all_results=all_results,
                all_baseline=all_baseline,
                summary_rows=summary_rows,
                meta=snapshot_meta,
                status="partial",
            )

    # Global aggregates
    aggregate = _aggregate_summary(summary_rows, all_results)

    result = {
        "per_scenario_results": {
            s_idx: [r.to_dict() for r in rs]
            for s_idx, rs in all_results.items()
        },
        "per_scenario_baseline": all_baseline,
        "per_scenario_summary": summary_rows,
        "aggregate": aggregate,
    }

    if snapshot_path is not None:
        _write_eval_snapshot(
            snapshot_path=snapshot_path,
            all_results=all_results,
            all_baseline=all_baseline,
            summary_rows=summary_rows,
            meta=snapshot_meta,
            status="complete",
        )

    return result


# ---------------------------------------------------------------------------
# Parallel evaluation
# ---------------------------------------------------------------------------

def _worker_run_scenario(args: tuple) -> tuple:
    """Top-level worker function (must be picklable for multiprocessing spawn).

    Loads the model fresh in the worker process to avoid cross-process tensor
    sharing issues on Windows (which uses 'spawn', not 'fork').

    Returns: (s_idx, scenario_results, scenario_baseline, summary_row)
             or (s_idx, None, None, None) on failure.
    """
    checkpoint_path, data_root, scenario, s_idx, cfg, device = args
    device = _resolve_runtime_device(device)

    import sys as _sys
    from pathlib import Path as _Path
    _src = _Path(__file__).resolve().parent.parent
    if str(_src) not in _sys.path:
        _sys.path.insert(0, str(_src))

    import torch as _torch
    from learned_surrogate.config import ModelConfig, _merge_dataclass

    # Load model fresh in this process
    ckpt = _torch.load(checkpoint_path, map_location=device)
    _cfg_dict = ckpt.get("config", {}).get("model", {})
    _cfg_dict = _infer_model_config_from_state_dict(ckpt["model_state_dict"], _cfg_dict)
    model_cfg = _merge_dataclass(ModelConfig, _cfg_dict)
    model = _build_model(model_cfg)
    _load_res = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if _load_res.missing_keys:
        raise RuntimeError(
            f"Checkpoint architecture mismatch — missing keys:\n"
            + "\n".join(f"  {k}" for k in _load_res.missing_keys)
            + f"\nResolved config: readout={model_cfg.readout}, "
              f"gine_hidden_dim={model_cfg.gine_hidden_dim}, "
              f"node_embed_dim={model_cfg.node_embed_dim}, "
              f"head_mlp_dims={model_cfg.head_mlp_dims}"
        )
    print(
        f"[counterfactual] Loaded {type(model).__name__} from checkpoint "
        f"(readout={model_cfg.readout}, hidden={model_cfg.gine_hidden_dim}, "
        f"head={model_cfg.head_mlp_dims})",
        flush=True,
    )
    model.to(device)
    model.eval()

    # Load normalization scale + MFMC params
    try:
        scale = load_scenario_scale(data_root, s_idx)
        mfmc_params = load_mfmc_params(data_root, s_idx)
    except FileNotFoundError:
        print(f"  [S{s_idx:02d}] WARNING: no data, skipping.", flush=True)
        return s_idx, None, None, None

    # Sample inconsistent queries
    try:
        if cfg.measurement_root:
            queries = _sample_inconsistent_queries_from_measurements(
                measurement_root=cfg.measurement_root,
                scenario_idx=s_idx,
                n_queries=cfg.n_queries,
                gamma=cfg.gamma,
                seed=cfg.seed + s_idx,
                max_pre_inconsistency=cfg.max_pre_inconsistency,
            )
        else:
            queries = _sample_inconsistent_queries(
                scenario,
                n_queries=cfg.n_queries,
                gamma=cfg.gamma,
                mc_samples=cfg.mc_samples_screen,
                seed=cfg.seed + s_idx,
                max_query_inconsistency=cfg.max_query_inconsistency,
            )
    except RuntimeError as e:
        print(f"  [S{s_idx:02d}] WARNING: {e}", flush=True)
        return s_idx, None, None, None

    print(f"  [S{s_idx:02d}] {len(queries)} queries found.", flush=True)

    opt = CounterfactualOptimizer(
        model=model,
        scenario=scenario,
        norm_scale=scale,
        gamma=cfg.gamma if cfg.gamma_surrogate is None else cfg.gamma_surrogate,
        gamma_verify=cfg.gamma,
        lambda_init=cfg.lambda_init,
        lambda_scale=cfg.lambda_scale,
        lambda_max=cfg.lambda_max,
        lr=cfg.lr,
        max_iter=cfg.max_iter,
        tol=cfg.tol,
        patience=cfg.patience,
        escalation_freq=cfg.escalation_freq,
        device=device,
        mfmc_params=mfmc_params,
    )

    scenario_results = []
    scenario_baseline = []

    for q_idx, theta_star in enumerate(queries):
        cf = _run_counterfactual_search(
            opt=opt,
            cfg=cfg,
            theta_star=theta_star,
            scenario_idx=s_idx,
            query_seed=cfg.seed + s_idx * 1000 + q_idx,
        )
        scenario_results.append(cf)

        if cfg.run_baseline:
            bl = naive_baseline(
                scenario=scenario,
                theta_star=theta_star,
                gamma=cfg.gamma,
                highest_sobol_param=cfg.highest_sobol_param,
                mc_samples=cfg.mc_samples_verify,
            )
            scenario_baseline.append(bl)

    # Build summary row
    valid_rate = float(np.mean([r.valid for r in scenario_results]))
    converge_rate = float(np.mean([r.converged for r in scenario_results]))
    mean_prox = float(np.mean([np.linalg.norm(r.delta_theta) for r in scenario_results]))
    mean_I_star = float(np.mean([
        r.I_surrogate_star for r in scenario_results
    ]))
    mean_I_surr_prime = float(np.mean([
        r.I_surrogate_prime for r in scenario_results
    ]))
    mean_I_prime = float(np.mean([
        r.I_mc_prime for r in scenario_results if r.I_mc_prime is not None
    ]))
    mean_surrogate_improvement = float(np.mean([
        r.I_surrogate_star - r.I_surrogate_prime for r in scenario_results
    ]))
    mean_mc_improvement = float(np.mean([
        r.I_surrogate_star - r.I_mc_prime
        for r in scenario_results if r.I_mc_prime is not None
    ]))
    surrogate_improvement_rate = float(np.mean([
        r.I_surrogate_prime < r.I_surrogate_star for r in scenario_results
    ]))
    mc_below_040_rate = float(np.mean([
        r.I_mc_prime <= 0.4 for r in scenario_results if r.I_mc_prime is not None
    ]))
    mc_below_050_rate = float(np.mean([
        r.I_mc_prime <= 0.5 for r in scenario_results if r.I_mc_prime is not None
    ]))
    mean_mc_gap_to_gamma = float(np.mean([
        r.I_mc_prime - cfg.gamma for r in scenario_results if r.I_mc_prime is not None
    ]))
    delta_mat = np.stack([r.delta_theta for r in scenario_results])
    abs_delta_mean = np.mean(np.abs(delta_mat), axis=0)
    bl_valid_rate = (
        float(np.mean([b["valid"] for b in scenario_baseline]))
        if scenario_baseline else None
    )

    summary_row = {
        "scenario_idx": s_idx,
        "n_queries": len(scenario_results),
        "validity_rate": valid_rate,
        "convergence_rate": converge_rate,
        "mean_proximity_l2": mean_prox,
        "mean_I_surrogate_star": mean_I_star,
        "mean_I_surrogate_prime": mean_I_surr_prime,
        "mean_I_mc_prime": mean_I_prime,
        "mean_surrogate_improvement": mean_surrogate_improvement,
        "mean_mc_improvement_vs_surrogate_start": mean_mc_improvement,
        "surrogate_improvement_rate": surrogate_improvement_rate,
        "mc_below_040_rate": mc_below_040_rate,
        "mc_below_050_rate": mc_below_050_rate,
        "mean_mc_gap_to_gamma": mean_mc_gap_to_gamma,
        "mean_abs_delta_theta": {
            name: float(v) for name, v in zip(PARAM_NAMES, abs_delta_mean)
        },
        "dominant_param_freq": _dominant_param_freq(scenario_results),
        "baseline_validity_rate": bl_valid_rate,
    }

    print(
        f"  [S{s_idx:02d}] done — validity={valid_rate:.1%}"
        f"  prox={mean_prox:.4f}"
        f"  dom={max(_dominant_param_freq(scenario_results), key=_dominant_param_freq(scenario_results).get)[:3].upper()}",
        flush=True,
    )
    return s_idx, scenario_results, scenario_baseline, summary_row


def evaluate_counterfactuals_parallel(
    checkpoint_path: str | Path,
    scenarios: List[Scenario],
    scenario_indices: List[int],
    data_root: str | Path,
    cfg: CounterfactualEvalConfig = None,
    device: str = "cpu",
    n_workers: Optional[int] = None,
    snapshot_path: Optional[str | Path] = None,
    snapshot_meta: Optional[dict] = None,
) -> dict:
    """Parallel version of :func:`evaluate_counterfactuals`.

    Each scenario is processed in a separate worker process, so MC LP solves
    and GNN optimization run concurrently.  Speedup is roughly linear in
    ``n_workers`` up to the number of scenarios.

    The model is loaded fresh in each worker (avoids multiprocessing tensor
    sharing issues on Windows).

    Args:
        checkpoint_path: Path to ``best_model.pt`` — loaded by each worker.
        scenarios:        List of Scenario objects.
        scenario_indices: Integer indices (1-based) aligned with scenarios.
        data_root:        Path to ``data/surrogate/``.
        cfg:              Evaluation config (default: CounterfactualEvalConfig()).
        device:           Torch device string used in each worker.
        n_workers:        Number of parallel processes.  Defaults to
                          ``min(len(scenarios), os.cpu_count())``.

    Returns:
        Same dict structure as :func:`evaluate_counterfactuals`.
    """
    import multiprocessing as mp
    import os

    if cfg is None:
        cfg = CounterfactualEvalConfig()
    if n_workers is None:
        n_workers = min(len(scenarios), os.cpu_count() or 1)

    n_workers = min(n_workers, len(scenarios))
    checkpoint_path = str(Path(checkpoint_path).resolve())
    data_root = str(Path(data_root).resolve())

    worker_args = [
        (checkpoint_path, data_root, scenario, s_idx, cfg, device)
        for scenario, s_idx in zip(scenarios, scenario_indices)
    ]

    print(
        f"Running {len(scenarios)} scenarios with {n_workers} workers "
        f"({cfg.n_queries} queries each) …"
    )

    all_results: Dict[int, List[CounterfactualResult]] = {}
    all_baseline: Dict[int, List[dict]] = {}
    summary_rows = []

    # 'spawn' is the default on Windows; safe on Linux/macOS too
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        for s_idx, scenario_results, scenario_baseline, summary_row in pool.imap_unordered(
            _worker_run_scenario, worker_args
        ):
            if scenario_results is None:
                continue
            all_results[s_idx] = scenario_results
            all_baseline[s_idx] = scenario_baseline or []
            summary_rows.append(summary_row)
            summary_rows.sort(key=lambda r: r["scenario_idx"])

            if snapshot_path is not None:
                _write_eval_snapshot(
                    snapshot_path=snapshot_path,
                    all_results=all_results,
                    all_baseline=all_baseline,
                    summary_rows=summary_rows,
                    meta=snapshot_meta,
                    status="partial",
                )

    # Sort summary by scenario index
    summary_rows.sort(key=lambda r: r["scenario_idx"])

    aggregate = _aggregate_summary(summary_rows, all_results)

    print(
        f"\nDone. "
        f"Validity={aggregate.get('mean_validity_rate', 0):.1%}  "
        f"Proximity={aggregate.get('mean_proximity_l2', 0):.4f}  "
        f"Dominant={aggregate.get('dominant_param', '?')}"
    )

    result = {
        "per_scenario_results": {
            s_idx: [r.to_dict() for r in rs]
            for s_idx, rs in all_results.items()
        },
        "per_scenario_baseline": all_baseline,
        "per_scenario_summary": summary_rows,
        "aggregate": aggregate,
    }

    if snapshot_path is not None:
        _write_eval_snapshot(
            snapshot_path=snapshot_path,
            all_results=all_results,
            all_baseline=all_baseline,
            summary_rows=summary_rows,
            meta=snapshot_meta,
            status="complete",
        )

    return result


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


def _snapshot_payload(
    all_results: dict,
    all_baseline: dict,
    summary_rows: list,
    meta: Optional[dict],
    status: str,
) -> dict:
    aggregate = _aggregate_summary(summary_rows, all_results)
    payload = {
        "status": status,
        "per_scenario_results": {
            s_idx: [r.to_dict() for r in rs]
            for s_idx, rs in all_results.items()
        },
        "per_scenario_baseline": all_baseline,
        "per_scenario_summary": summary_rows,
        "aggregate": aggregate,
    }
    if meta is not None:
        payload["meta"] = dict(meta)
    return _to_serializable(payload)


def _write_eval_snapshot(
    snapshot_path: str | Path,
    all_results: dict,
    all_baseline: dict,
    summary_rows: list,
    meta: Optional[dict],
    status: str,
) -> None:
    path = Path(snapshot_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _snapshot_payload(
        all_results=all_results,
        all_baseline=all_baseline,
        summary_rows=summary_rows,
        meta=meta,
        status=status,
    )
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _dominant_param_freq(results: List[CounterfactualResult]) -> dict:
    """Fraction of queries where each parameter is the dominant fix lever."""
    counts = {name: 0 for name in PARAM_NAMES}
    for r in results:
        counts[r.dominant_fix_param()] += 1
    n = len(results)
    return {k: v / n for k, v in counts.items()} if n > 0 else counts


def _aggregate_summary(summary_rows: list, all_results: dict) -> dict:
    """Compute global metrics across all scenarios."""
    if not summary_rows:
        return {}

    all_cf: List[CounterfactualResult] = []
    for rs in all_results.values():
        all_cf.extend(rs)

    validity_rates = [r["validity_rate"] for r in summary_rows]
    proximities = [
        np.linalg.norm(r.delta_theta) for r in all_cf
    ]
    surrogate_starts = [r.I_surrogate_star for r in all_cf]
    surrogate_primes = [r.I_surrogate_prime for r in all_cf]
    mc_primes = [r.I_mc_prime for r in all_cf if r.I_mc_prime is not None]
    surrogate_improvements = [
        r.I_surrogate_star - r.I_surrogate_prime for r in all_cf
    ]
    mc_improvements = [
        r.I_surrogate_star - r.I_mc_prime for r in all_cf if r.I_mc_prime is not None
    ]
    surrogate_improvement_rate = float(np.mean([
        r.I_surrogate_prime < r.I_surrogate_star for r in all_cf
    ]))
    mc_below_040_rate = float(np.mean([
        r.I_mc_prime <= 0.4 for r in all_cf if r.I_mc_prime is not None
    ]))
    mc_below_050_rate = float(np.mean([
        r.I_mc_prime <= 0.5 for r in all_cf if r.I_mc_prime is not None
    ]))
    delta_mat = np.stack([r.delta_theta for r in all_cf]) if all_cf else np.zeros((1, 3))
    abs_delta_mean = np.mean(np.abs(delta_mat), axis=0)

    # Dominant fix param across all queries
    dom_freq = _dominant_param_freq(all_cf)

    return {
        "n_scenarios": len(summary_rows),
        "n_queries_total": len(all_cf),
        "mean_validity_rate": float(np.mean(validity_rates)),
        "std_validity_rate": float(np.std(validity_rates)),
        "mean_proximity_l2": float(np.mean(proximities)),
        "mean_I_surrogate_star": float(np.mean(surrogate_starts)),
        "mean_I_surrogate_prime": float(np.mean(surrogate_primes)),
        "mean_I_mc_prime": float(np.mean(mc_primes)) if mc_primes else float("nan"),
        "mean_surrogate_improvement": float(np.mean(surrogate_improvements)),
        "mean_mc_improvement_vs_surrogate_start": (
            float(np.mean(mc_improvements)) if mc_improvements else float("nan")
        ),
        "mean_mc_gap_to_gamma": float(np.mean([
            r.get("mean_mc_gap_to_gamma", float("nan")) for r in summary_rows
        ])),
        "surrogate_improvement_rate": surrogate_improvement_rate,
        "mc_below_040_rate": mc_below_040_rate,
        "mc_below_050_rate": mc_below_050_rate,
        "mean_abs_delta_theta": {
            name: float(v) for name, v in zip(PARAM_NAMES, abs_delta_mean)
        },
        "dominant_param_freq": dom_freq,
        "dominant_param": max(dom_freq, key=dom_freq.get),
    }


# ---------------------------------------------------------------------------
# Checkpoint architecture inference
# ---------------------------------------------------------------------------

def _infer_model_config_from_state_dict(state_dict: dict, cfg_dict: dict) -> dict:
    """Patch cfg_dict with architecture dims inferred from checkpoint weights.

    Handles two common checkpoint variants:
      - DataParallel checkpoints (keys prefixed with ``module.``)
      - BatchNorm layers in the head MLP (1-D ``.weight`` tensors that must
        not be mistaken for Linear layer weights)

    Args:
        state_dict: Raw state dict from ``torch.load``.
        cfg_dict:   Config dict from ``ckpt["config"]["model"]`` (may be empty).

    Returns:
        Updated cfg_dict with corrected ``head_mlp_dims`` and
        ``gine_hidden_dim`` (when detectable).
    """
    patched = dict(cfg_dict)

    # Normalise keys: strip leading 'module.' added by DataParallel / DDP.
    sd: dict = {}
    for k, v in state_dict.items():
        nk = k
        while nk.startswith("module."):
            nk = nk[len("module."):]
        sd[nk] = v

    # --- Infer head_mlp_dims from Linear layers only (2-D weight tensors) ---
    # BatchNorm / LayerNorm also have a `.weight` key but it is 1-D, so we
    # skip any tensor that is not a matrix.
    linear_shapes: List[tuple] = []
    for idx in range(50):          # search up to 50 sequential indices
        w_key = f"inconsistency_head.mlp.{idx}.weight"
        if w_key in sd:
            w = sd[w_key]
            if w.dim() == 2:       # Linear layer: shape (out, in)
                linear_shapes.append((w.shape[1], w.shape[0]))   # (in, out)

    if linear_shapes:
        # head_mlp_dims = [in_0, out_0, out_1, ..., out_n]
        head_dims: List[int] = [linear_shapes[0][0]]
        for _, out_dim in linear_shapes:
            head_dims.append(out_dim)
        patched["head_mlp_dims"] = head_dims

        # For "asymmetric" readout, ZonotopeGINEv2 builds the head with
        # in_dim = 2 * gine_hidden_dim — so head_dims[0] = 2 * gine_hidden_dim,
        # not gine_hidden_dim itself.  Do NOT override gine_hidden_dim in this case;
        # the stored config value is correct.
        if patched.get("readout", "mean") != "asymmetric":
            patched["gine_hidden_dim"] = head_dims[0]

    # --- Fallback: infer gine_hidden_dim from first GINEConv MLP weight ---
    for candidate_key in (
        "convs.0.nn.0.weight",
        "convs.0.nn.mlp.0.weight",
    ):
        if candidate_key in sd and "gine_hidden_dim" not in patched:
            w = sd[candidate_key]
            if w.dim() == 2:
                patched["gine_hidden_dim"] = int(w.shape[0])
            break

    return patched


# ---------------------------------------------------------------------------
# High-level interface: CounterfactualExplainer
# ---------------------------------------------------------------------------

class CounterfactualExplainer:
    """High-level interface: load a trained model and explain inconsistency.

    Usage::

        explainer = CounterfactualExplainer.from_checkpoint(
            checkpoint_path="results/surrogate/best_model.pt",
            data_root="data/surrogate",
            device="cpu",
        )
        result = explainer.explain(scenario, theta_star=np.array([3.0, 0.15, 0.5]))
        print(result.to_dict())

    """

    def __init__(
        self,
        model: ZonotopeGINE,
        data_root: str | Path,
        gamma: float = 0.3,
        optimizer_kwargs: Optional[dict] = None,
        device: str = "cpu",
    ):
        self.model = model
        self.data_root = Path(data_root)
        self.gamma = gamma
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.device = device

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        data_root: str | Path,
        gamma: float = 0.3,
        optimizer_kwargs: Optional[dict] = None,
        device: str = "cpu",
    ) -> "CounterfactualExplainer":
        """Load a trained ZonotopeGINE from a checkpoint file.

        Args:
            checkpoint_path: Path to ``best_model.pt``.
            data_root:       Path to ``data/surrogate/``.
            gamma:           Consistency threshold.
            optimizer_kwargs: Extra kwargs forwarded to CounterfactualOptimizer.
            device:          Torch device.

        Returns:
            Initialized CounterfactualExplainer.
        """
        device = _resolve_runtime_device(device)
        ckpt = torch.load(checkpoint_path, map_location=device)
        from learned_surrogate.config import ModelConfig, _merge_dataclass  # noqa

        cfg_dict = ckpt.get("config", {}).get("model", {})
        cfg_dict = _infer_model_config_from_state_dict(ckpt["model_state_dict"], cfg_dict)
        model_cfg = _merge_dataclass(ModelConfig, cfg_dict)

        model = _build_model(model_cfg)
        _load_res = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if _load_res.missing_keys:
            raise RuntimeError(
                f"Checkpoint architecture mismatch — missing keys:\n"
                + "\n".join(f"  {k}" for k in _load_res.missing_keys)
                + f"\nResolved config: readout={model_cfg.readout}, "
                  f"gine_hidden_dim={model_cfg.gine_hidden_dim}, "
                  f"node_embed_dim={model_cfg.node_embed_dim}, "
                  f"head_mlp_dims={model_cfg.head_mlp_dims}"
            )
        print(
            f"[counterfactual] Loaded {type(model).__name__} from checkpoint "
            f"(readout={model_cfg.readout}, hidden={model_cfg.gine_hidden_dim}, "
            f"head={model_cfg.head_mlp_dims})",
            flush=True,
        )
        model.to(device)
        model.eval()

        return cls(
            model=model,
            data_root=data_root,
            gamma=gamma,
            optimizer_kwargs=optimizer_kwargs,
            device=device,
        )

    def explain(
        self,
        scenario: Scenario,
        theta_star: np.ndarray,
        scenario_idx: Optional[int] = None,
        mc_samples: int = 2000,
        verify_mc: bool = True,
    ) -> CounterfactualResult:
        """Find the minimal fix θ' for an inconsistent θ*.

        Args:
            scenario:     MMS scenario object.
            theta_star:   (3,) inconsistent configuration.
            scenario_idx: Integer index for scale loading (required if
                          ``data_root`` contains per-scenario NPZ files).
            mc_samples:   MC samples for final verification.
            verify_mc:    Whether to run MC verification.

        Returns:
            CounterfactualResult.
        """
        if scenario_idx is None:
            # Try to infer from scenario name (e.g. "S03_...")
            try:
                import re
                m = re.search(r"S(\d+)", scenario.name)
                scenario_idx = int(m.group(1)) if m else None
            except Exception:
                pass

        if scenario_idx is not None:
            scale = load_scenario_scale(self.data_root, scenario_idx)
            try:
                mfmc_params = load_mfmc_params(self.data_root, scenario_idx)
            except FileNotFoundError:
                mfmc_params = None
        else:
            # Fallback: use max abs value of the source zonotope features
            Z = scenario.source
            scale = max(
                float(np.abs(Z.c).max()),
                float(np.abs(Z.G).max()),
                1e-8,
            )
            mfmc_params = None

        opt = CounterfactualOptimizer(
            model=self.model,
            scenario=scenario,
            norm_scale=scale,
            gamma=self.gamma,
            device=self.device,
            mfmc_params=mfmc_params,
            **self.optimizer_kwargs,
        )
        return opt.optimize(
            theta_star=theta_star,
            mc_samples=mc_samples,
            verify_mc=verify_mc,
            scenario_idx=scenario_idx,
        )

    def evaluate(
        self,
        scenarios: List[Scenario],
        scenario_indices: List[int],
        cfg: Optional[CounterfactualEvalConfig] = None,
        n_workers: int = 1,
        checkpoint_path: Optional[str | Path] = None,
        snapshot_path: Optional[str | Path] = None,
        snapshot_meta: Optional[dict] = None,
    ) -> dict:
        """Batch evaluation across multiple scenarios.

        Args:
            scenarios:        List of Scenario objects.
            scenario_indices: Integer indices aligned with scenarios.
            cfg:              Evaluation config.
            n_workers:        Number of parallel processes.  1 = sequential
                              (uses the already-loaded model).  >1 = parallel
                              (requires ``checkpoint_path`` so workers can
                              reload the model independently).
            checkpoint_path:  Required when ``n_workers > 1``.

        Returns:
            Same dict structure as :func:`evaluate_counterfactuals`.
        """
        effective_cfg = cfg or CounterfactualEvalConfig(gamma=self.gamma)

        if n_workers > 1:
            if checkpoint_path is None:
                raise ValueError(
                    "checkpoint_path is required for parallel evaluation "
                    "(each worker needs to reload the model independently)."
                )
            return evaluate_counterfactuals_parallel(
                checkpoint_path=checkpoint_path,
                scenarios=scenarios,
                scenario_indices=scenario_indices,
                data_root=self.data_root,
                cfg=effective_cfg,
                device=self.device,
                n_workers=n_workers,
                snapshot_path=snapshot_path,
                snapshot_meta=snapshot_meta,
            )

        return evaluate_counterfactuals(
            model=self.model,
            scenarios=scenarios,
            scenario_indices=scenario_indices,
            data_root=self.data_root,
            cfg=effective_cfg,
            device=self.device,
            snapshot_path=snapshot_path,
            snapshot_meta=snapshot_meta,
        )
