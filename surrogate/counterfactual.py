"""
surrogate/counterfactual.py
============================
Gradient-based counterfactual optimizer for the DeepSets surrogate.

Given an inconsistent configuration θ* = (scale_factor, center_delta,
correlation_strength) with Î(θ*) > γ, finds the nearest θ' that restores
consistency:

    θ' = argmin_{θ ∈ Θ_box} ||θ - θ*||²_norm   s.t.   Î(θ) ≤ γ

Solved via batched Adam with penalty escalation:

    L(θ) = ||θ - θ*||²_norm + λ · max(0, Î(θ) - γ)²

All queries within a scenario are stacked as (B, 3) and run through a
single forward/backward pass per iteration — no Python loops over queries.
Multi-start restarts are folded into the batch dimension as well.

Parallelism
-----------
- Inner loop  : fully vectorised — (B × restarts, 3) batch through the model
- Across scenarios : call run_counterfactuals_for_scenario() from a
  ThreadPoolExecutor (model inference releases the GIL on CPU; on GPU each
  scenario gets its own CUDA stream automatically).

Intervention (verified against MATLAB ground truth)
-----------------------------------------------------
Given base zonotope (pre_c, pre_G) and θ = (sf, cd, cs):

    G_scaled    = sf * pre_G                              # scale all generators
    post_c      = pre_c + cd * G_scaled.sum(axis=1)      # shift center
    post_G[:,0] = G_scaled[:,0]                           # 1st gen unchanged
    post_G[:,1] = cs * G_scaled[:,0]                      # Givens rotation
                  + sqrt(1-cs²) * G_scaled[:,1]           #   of 2nd gen
    post_G[:,k≥2] = G_scaled[:,k]                         # higher gens scaled

The target zonotope (tgt_c, tgt_G) and UPR are fixed throughout optimisation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from .dataset import N_DIM_FEAT, N_GLOBAL, MAX_DIM, _upr_onehot
from .model import DeepSetsZonotope

# ── Parameter box ─────────────────────────────────────────────────────────────
PARAM_NAMES  = ["scale_factor", "center_delta", "correlation_strength"]
PARAM_BOUNDS = [
    (0.05, 3.10),   # scale_factor
    (-1.05, 1.05),  # center_delta
    (0.0,  1.05),   # correlation_strength
]
_THETA_LB   = torch.tensor([b[0] for b in PARAM_BOUNDS], dtype=torch.float32)
_THETA_UB   = torch.tensor([b[1] for b in PARAM_BOUNDS], dtype=torch.float32)
_THETA_SPAN = _THETA_UB - _THETA_LB


# ═══════════════════════════════════════════════════════════════════════════════
# Differentiable intervention  — batched
# ═══════════════════════════════════════════════════════════════════════════════

class DifferentiableIntervention(nn.Module):
    """Apply a batch of θ = (sf, cd, cs) to a fixed base zonotope.

    Accepts theta of shape (B, 3) or (3,) and returns batched model inputs.

    Args
    ----
    pre_c, pre_G    : pre-intervention source zonotope  (numpy)
    tgt_c, tgt_G    : fixed target zonotope             (numpy)
    upr_scales/offsets : UPR per-dim linear map or None
    upr_onehot      : (N_UPR_TYPES,) one-hot encoding or None
    device          : torch.device
    """

    EPS = 1e-10

    def __init__(
        self,
        pre_c:  np.ndarray,
        pre_G:  np.ndarray,
        tgt_c:  np.ndarray,
        tgt_G:  np.ndarray,
        upr_scales:  Optional[np.ndarray],
        upr_offsets: Optional[np.ndarray],
        upr_onehot:  Optional[np.ndarray],
        device: torch.device,
    ):
        super().__init__()
        d = len(pre_c)
        self.d = d

        def _t(a):
            return torch.tensor(np.asarray(a, dtype=np.float32), device=device)

        self.register_buffer("pre_c", _t(pre_c))   # (d,)
        self.register_buffer("pre_G", _t(pre_G))   # (d, n_gen)
        self.register_buffer("tgt_c", _t(tgt_c))   # (d,)
        self.register_buffer("tgt_G", _t(tgt_G))   # (d, n_gen_t)

        if upr_scales is not None:
            self.register_buffer("upr_scales",  _t(upr_scales[:d]))
            self.register_buffer("upr_offsets", _t(upr_offsets[:d]))
        else:
            self.register_buffer("upr_scales",  torch.ones(d,  device=device))
            self.register_buffer("upr_offsets", torch.zeros(d, device=device))

        self.register_buffer(
            "upr_onehot",
            _t(upr_onehot if upr_onehot is not None else _upr_onehot("unknown")),
        )

        # Pre-compute fixed target row-norms: (d,)
        self.register_buffer(
            "r2_raw",
            torch.linalg.norm(self.tgt_G, dim=1) + self.EPS,
        )

        # Dimension mask padded to MAX_DIM
        mask = torch.zeros(MAX_DIM, device=device)
        mask[:d] = 1.0
        self.register_buffer("mask", mask)

    def forward(self, theta: torch.Tensor):
        """Compute batched model inputs from theta.

        Args
        ----
        theta : (B, 3) or (3,)  —  [scale_factor, center_delta, corr_strength]

        Returns
        -------
        per_dim      : (B, MAX_DIM, N_DIM_FEAT)
        mask         : (B, MAX_DIM)
        global_feats : (B, N_GLOBAL)
        """
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)   # (1, 3)
        B = theta.shape[0]

        sf = theta[:, 0]   # (B,)
        cd = theta[:, 1]   # (B,)
        cs = theta[:, 2]   # (B,)

        # ── Apply intervention ──────────────────────────────────────────────
        # G_scaled: (B, d, n_gen)
        G_scaled = sf.view(B, 1, 1) * self.pre_G.unsqueeze(0)

        # post_c: (B, d)  — center shift proportional to total scaled span
        post_c = self.pre_c.unsqueeze(0) + cd.view(B, 1) * G_scaled.sum(dim=2)

        # Givens rotation on generators 0 and 1
        n_gen = G_scaled.shape[2]
        if n_gen >= 2:
            ccos = torch.sqrt(torch.clamp(1.0 - cs * cs, min=0.0))   # (B,)
            g0  = G_scaled[:, :, 0]                                    # (B, d)
            g1  = G_scaled[:, :, 1]                                    # (B, d)
            g1r = cs.unsqueeze(1) * g0 + ccos.unsqueeze(1) * g1       # (B, d)
            if n_gen == 2:
                post_G = torch.stack([g0, g1r], dim=2)                 # (B, d, 2)
            else:
                rest   = G_scaled[:, :, 2:]
                post_G = torch.cat([g0.unsqueeze(2),
                                    g1r.unsqueeze(2), rest], dim=2)
        else:
            post_G = G_scaled

        # ── UPR propagation ─────────────────────────────────────────────────
        # c1_prop: (B, d)
        c1_prop = self.upr_scales * post_c + self.upr_offsets

        # r1_prop: (B, d) — row-norms of propagated source generators
        r1_prop = torch.linalg.norm(
            self.upr_scales.abs().view(1, self.d, 1) * post_G, dim=2
        )   # (B, d)

        sigma = r1_prop + self.r2_raw.unsqueeze(0)   # (B, d)

        # ── Per-dimension features  [δc, r1, r2, cos] ───────────────────────
        delta_c = (c1_prop - self.tgt_c) / sigma          # (B, d)
        r1_n    = r1_prop / sigma                          # (B, d)
        r2_n    = self.r2_raw / sigma                      # (B, d)

        # Cosine: batched loop over d (d is small: 2–4)
        cos_vals = torch.zeros(B, self.d, dtype=theta.dtype, device=theta.device)
        for i in range(self.d):
            g1_prop_i = self.upr_scales[i] * post_G[:, i, :]   # (B, n_gen)
            n1 = torch.linalg.norm(g1_prop_i, dim=1)            # (B,)
            n2 = self.r2_raw[i]                                  # scalar
            safe = (n1 > self.EPS) & (n2 > self.EPS)
            if safe.any():
                dot = (g1_prop_i[safe] * self.tgt_G[i]).sum(dim=1)
                cos_vals[safe, i] = dot / (n1[safe] * n2)

        # Pad to (B, MAX_DIM, N_DIM_FEAT)
        per_dim = torch.zeros(B, MAX_DIM, N_DIM_FEAT,
                              dtype=theta.dtype, device=theta.device)
        per_dim[:, :self.d, 0] = delta_c
        per_dim[:, :self.d, 1] = r1_n
        per_dim[:, :self.d, 2] = r2_n
        per_dim[:, :self.d, 3] = cos_vals

        # ── Global features  (B, N_GLOBAL) ──────────────────────────────────
        log_vol = torch.log(
            r1_prop.prod(dim=1).clamp(min=self.EPS)
            / (self.r2_raw.prod().clamp(min=self.EPS) + self.EPS)
            + self.EPS
        )   # (B,)
        norm_dist = torch.linalg.norm(
            c1_prop - self.tgt_c, dim=1
        ) / (sigma.mean(dim=1) + self.EPS)   # (B,)

        upr_oh = self.upr_onehot.unsqueeze(0).expand(B, -1)   # (B, N_UPR)
        global_feats = torch.cat(
            [log_vol.unsqueeze(1), norm_dist.unsqueeze(1), upr_oh], dim=1
        )   # (B, N_GLOBAL)

        mask_b = self.mask.unsqueeze(0).expand(B, -1)   # (B, MAX_DIM)

        return per_dim, mask_b, global_feats


# ═══════════════════════════════════════════════════════════════════════════════
# Result dataclass
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CounterfactualResult:
    theta_star:      np.ndarray    # (3,) inconsistent configuration
    theta_prime:     np.ndarray    # (3,) nearest consistent configuration
    delta_theta:     np.ndarray    # theta_prime - theta_star (signed)
    i_surr_star:     float
    i_surr_prime:    float
    dist_normalised: float         # L2 in box-normalised θ-space
    converged:       bool
    n_iter:          int
    wall_time_s:     float
    gamma:           float
    param_names:     List[str] = field(
        default_factory=lambda: list(PARAM_NAMES))

    @property
    def dominant_param(self) -> str:
        spans = np.array([b[1] - b[0] for b in PARAM_BOUNDS])
        idx = int(np.argmax(np.abs(self.delta_theta) / spans))
        return self.param_names[idx]

    def to_dict(self) -> dict:
        return {
            "theta_star":       self.theta_star.tolist(),
            "theta_prime":      self.theta_prime.tolist(),
            "delta_theta":      self.delta_theta.tolist(),
            "i_surr_star":      self.i_surr_star,
            "i_surr_prime":     self.i_surr_prime,
            "dist_normalised":  self.dist_normalised,
            "dominant_param":   self.dominant_param,
            "converged":        self.converged,
            "n_iter":           self.n_iter,
            "wall_time_s":      self.wall_time_s,
            "gamma":            self.gamma,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Batched optimizer
# ═══════════════════════════════════════════════════════════════════════════════

class BatchedCounterfactualOptimizer:
    """Vectorised projected-gradient optimizer for the DeepSets surrogate.

    All queries and all multi-start restarts are fused into a single batch
    of shape (n_queries × n_restarts, 3) and run through one forward/backward
    pass per iteration — no Python loops over queries.

    Objective per sample i:
        L_i(θ) = ||θ - θ*_i||²_norm + λ_i · max(0, Î(θ) - γ)²

    λ_i is escalated independently for each sample every `escalation_freq`
    steps while the constraint is not yet satisfied.

    Args
    ----
    model           : frozen DeepSetsZonotope
    intervention    : DifferentiableIntervention for this scenario
    gamma           : consistency threshold
    lambda_init/scale/max : penalty schedule
    lr              : Adam learning rate
    max_iter        : optimisation iterations
    patience        : early-stop when no improvement for this many steps
    escalation_freq : escalate λ every N iters
    n_restarts      : additional random restarts per query (0 = θ* only)
    device          : torch.device
    """

    def __init__(
        self,
        model:        DeepSetsZonotope,
        intervention: DifferentiableIntervention,
        gamma:        float = 0.5,
        lambda_init:  float = 10.0,
        lambda_scale: float = 5.0,
        lambda_max:   float = 1e5,
        lr:           float = 0.02,
        max_iter:     int   = 400,
        patience:     int   = 50,
        escalation_freq: int = 50,
        n_restarts:   int   = 4,
        device:       torch.device = torch.device("cpu"),
    ):
        self.model        = model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.intervention    = intervention
        self.gamma           = gamma
        self.lambda_init     = lambda_init
        self.lambda_scale    = lambda_scale
        self.lambda_max      = lambda_max
        self.lr              = lr
        self.max_iter        = max_iter
        self.patience        = patience
        self.escalation_freq = escalation_freq
        self.n_restarts      = n_restarts
        self.device          = device

        self.lb   = _THETA_LB.to(device)
        self.ub   = _THETA_UB.to(device)
        self.span = _THETA_SPAN.to(device)

    # ── public API ─────────────────────────────────────────────────────────

    def find_batch(
        self,
        theta_stars: np.ndarray,   # (Q, 3)
    ) -> List[CounterfactualResult]:
        """Find counterfactuals for all Q queries in one batched run.

        Returns a list of Q CounterfactualResult objects.
        """
        t0 = time.perf_counter()
        Q  = len(theta_stars)
        R  = self.n_restarts + 1   # total starts per query (θ* + restarts)
        B  = Q * R                 # total batch size

        ts_t = torch.tensor(theta_stars, dtype=torch.float32, device=self.device)
        # ts_t: (Q, 3)

        # ── Evaluate surrogate at all θ* ──────────────────────────────────
        with torch.no_grad():
            pd0, m0, gf0 = self.intervention(ts_t)
            i_stars = self.model(pd0, m0, gf0).cpu().numpy()  # (Q,)

        # ── Build initial batch: θ* repeated + random restarts ───────────
        # theta_init: (B, 3) = [θ*_0, θ*_0, ...(R times), θ*_1, ..., θ*_{Q-1}]
        theta_rep   = ts_t.repeat_interleave(R, dim=0)   # (B, 3)  starts = θ*
        # Overwrite restart slots (indices 1, 2, ..., R-1 of each block) with uniform noise
        if self.n_restarts > 0:
            restart_mask = torch.zeros(B, dtype=torch.bool, device=self.device)
            for q in range(Q):
                for r in range(1, R):
                    restart_mask[q * R + r] = True
            rand = torch.rand(restart_mask.sum(), 3, device=self.device)
            theta_rep[restart_mask] = self.lb + rand * self.span

        theta = theta_rep.clone().detach().requires_grad_(True)

        # Expand θ* reference: (B, 3)
        ts_ref = ts_t.repeat_interleave(R, dim=0)   # (B, 3)  — fixed

        # ── Per-sample penalty coefficients  (B,) ─────────────────────────
        lam = torch.full((B,), self.lambda_init,
                         dtype=torch.float32, device=self.device)

        opt = torch.optim.Adam([theta], lr=self.lr)

        # Tracking best feasible solution per sample
        best_theta = theta.detach().clone()                    # (B, 3)
        best_dist  = torch.full((B,), float("inf"),
                                device=self.device)
        best_i     = torch.full((B,), float("inf"),
                                device=self.device)
        no_improve = torch.zeros(B, dtype=torch.long, device=self.device)
        active     = torch.ones(B, dtype=torch.bool,  device=self.device)

        for it in range(self.max_iter):
            # Only run active samples (not yet converged + patience not expired)
            if not active.any():
                break

            opt.zero_grad()

            pd, mask, gf = self.intervention(theta)
            i_hat = self.model(pd, mask, gf)   # (B,)

            dist_sq = ((theta - ts_ref) / self.span).pow(2).sum(dim=1)  # (B,)
            penalty = torch.clamp(i_hat - self.gamma, min=0.0).pow(2)   # (B,)
            loss    = (dist_sq + lam * penalty).sum()
            loss.backward()
            opt.step()

            with torch.no_grad():
                theta.clamp_(self.lb, self.ub)

                i_val   = i_hat.detach()       # (B,)
                d_val   = dist_sq.detach().sqrt()  # (B,)

                # Update best feasible (surrogate ≤ γ preferred; else best i)
                feasible = i_val <= self.gamma
                improved_feas = feasible & (d_val < best_dist)
                improved_inf  = ~feasible & ~(best_i <= self.gamma) & (i_val < best_i)
                improved = improved_feas | improved_inf

                best_theta[improved] = theta.detach()[improved]
                best_dist[improved]  = d_val[improved]
                best_i[improved]     = i_val[improved]
                no_improve           = torch.where(improved,
                                                   torch.zeros_like(no_improve),
                                                   no_improve + 1)

                # Penalty escalation for infeasible samples
                if (it + 1) % self.escalation_freq == 0:
                    infeasible = i_val > self.gamma
                    lam = torch.where(infeasible,
                                      (lam * self.lambda_scale).clamp(max=self.lambda_max),
                                      lam)

                # Deactivate samples that have exceeded patience
                active = no_improve < self.patience

        # ── Reduce: per query, pick best across R restarts ────────────────
        best_theta_np = best_theta.detach().cpu().numpy()   # (B, 3)
        best_dist_np  = best_dist.cpu().numpy()             # (B,)
        best_i_np     = best_i.cpu().numpy()                # (B,)

        # Re-evaluate surrogate at best_theta for accurate final i_surr_prime
        with torch.no_grad():
            pd_f, m_f, gf_f = self.intervention(best_theta.detach())
            i_prime_all = self.model(pd_f, m_f, gf_f).cpu().numpy()   # (B,)

        elapsed = time.perf_counter() - t0

        results = []
        for q in range(Q):
            start = q * R
            block = slice(start, start + R)

            # Prefer converged results; among those take smallest distance
            conv_mask = i_prime_all[block] <= self.gamma
            if conv_mask.any():
                dists_block = best_dist_np[block].copy()
                dists_block[~conv_mask] = float("inf")
                best_r = int(np.argmin(dists_block))
            else:
                best_r = int(np.argmin(i_prime_all[block]))

            idx          = start + best_r
            th_prime     = best_theta_np[idx]
            i_prime      = float(i_prime_all[idx])
            dist         = float(best_dist_np[idx])
            converged    = i_prime <= self.gamma

            results.append(CounterfactualResult(
                theta_star      = theta_stars[q].astype(np.float32),
                theta_prime     = th_prime.astype(np.float32),
                delta_theta     = (th_prime - theta_stars[q]).astype(np.float32),
                i_surr_star     = float(i_stars[q]),
                i_surr_prime    = i_prime,
                dist_normalised = dist,
                converged       = converged,
                n_iter          = self.max_iter,
                wall_time_s     = elapsed / Q,
                gamma           = self.gamma,
            ))

        return results

    def find(self, theta_star: np.ndarray) -> CounterfactualResult:
        """Single-query convenience wrapper."""
        return self.find_batch(theta_star[np.newaxis])[0]


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario-level helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_scenario_geometry(json_path: Path) -> dict:
    """Extract base zonotope geometry from a scenario JSON file."""
    import json as _json
    data = _json.loads(Path(json_path).read_text(encoding="utf-8"))
    exp  = data["experiments"][0]

    unc   = exp["pre_state"]["uncertainty"]
    pre_c = np.array(unc["source_center"],     dtype=np.float64)
    pre_G = np.array(unc["source_generators"], dtype=np.float64)
    tgt_c = np.array(unc["target_center"],     dtype=np.float64)
    tgt_G = np.array(unc["target_generators"], dtype=np.float64)

    if pre_G.ndim == 1: pre_G = pre_G[:, None]
    if tgt_G.ndim == 1: tgt_G = tgt_G[:, None]

    upr_scales = upr_offsets = upr_onehot = None
    cr = exp.get("consistency_relations")
    d  = len(pre_c)
    if cr and isinstance(cr, list) and len(cr) >= d:
        upr_scales  = np.array([r["mapping"].get("scale",  1.0) for r in cr[:d]],
                                dtype=np.float64)
        upr_offsets = np.array([r["mapping"].get("offset", 0.0) for r in cr[:d]],
                                dtype=np.float64)
        dominant_type = next(
            (r.get("upr_type", "unknown") for r in cr[:d]
             if r.get("upr_type", "unknown") != "unknown"),
            "unknown",
        )
        from .dataset import _upr_onehot as _oh
        upr_onehot = _oh(dominant_type)

    return dict(pre_c=pre_c, pre_G=pre_G, tgt_c=tgt_c, tgt_G=tgt_G,
                upr_scales=upr_scales, upr_offsets=upr_offsets,
                upr_onehot=upr_onehot, d=d)


def run_counterfactuals_for_scenario(
    rows:       list,
    geom:       dict,
    model:      DeepSetsZonotope,
    gamma:      float = 0.5,
    max_queries: int  = 50,
    device:     torch.device = torch.device("cpu"),
    **opt_kwargs,
) -> List[CounterfactualResult]:
    """Gradient-based counterfactual search for one scenario.

    All inconsistent configurations are processed in a single batched
    optimisation run (one Adam optimiser, one forward/backward pass per iter).

    Args
    ----
    rows        : list of dicts with scale_factor, center_delta,
                  correlation_strength, i_surr
    geom        : from load_scenario_geometry()
    model       : frozen DeepSetsZonotope
    gamma       : consistency threshold
    max_queries : cap on number of inconsistent points to explain
    device      : torch device
    **opt_kwargs: forwarded to BatchedCounterfactualOptimizer
    """
    intervention = DifferentiableIntervention(
        pre_c       = geom["pre_c"],
        pre_G       = geom["pre_G"],
        tgt_c       = geom["tgt_c"],
        tgt_G       = geom["tgt_G"],
        upr_scales  = geom["upr_scales"],
        upr_offsets = geom["upr_offsets"],
        upr_onehot  = geom["upr_onehot"],
        device      = device,
    ).to(device)

    optimizer = BatchedCounterfactualOptimizer(
        model        = model,
        intervention = intervention,
        gamma        = gamma,
        device       = device,
        **opt_kwargs,
    )

    incon_rows = [r for r in rows if r.get("i_surr", 0.0) > gamma]
    if not incon_rows:
        return []

    if len(incon_rows) > max_queries:
        step = len(incon_rows) / max_queries
        incon_rows = [incon_rows[int(i * step)] for i in range(max_queries)]

    theta_stars = np.array(
        [[r["scale_factor"], r["center_delta"], r["correlation_strength"]]
         for r in incon_rows],
        dtype=np.float32,
    )   # (Q, 3)

    return optimizer.find_batch(theta_stars)


def run_counterfactuals_parallel(
    scenarios:   list,
    model:       DeepSetsZonotope,
    gamma:       float = 0.5,
    max_queries: int   = 50,
    device:      torch.device = torch.device("cpu"),
    max_workers: int   = 4,
    **opt_kwargs,
) -> dict:
    """Run counterfactual search for multiple scenarios in parallel threads.

    Uses ThreadPoolExecutor — PyTorch CPU inference releases the GIL so
    threads run concurrently.  On GPU, increase max_workers carefully to
    avoid OOM.

    Args
    ----
    scenarios   : list of dicts with keys 'domain', 'rows', 'json_file'
    model       : frozen DeepSetsZonotope (shared across threads, read-only)
    gamma       : consistency threshold
    max_queries : cap per scenario
    device      : torch device
    max_workers : thread pool size
    **opt_kwargs: forwarded to BatchedCounterfactualOptimizer

    Returns
    -------
    dict mapping scenario index → List[CounterfactualResult]
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _run_one(idx_scenario):
        idx, scenario = idx_scenario
        if not isinstance(scenario, dict):
            return idx, []
        json_file = scenario.get("json_file")
        rows      = scenario.get("rows", [])
        if not json_file or not rows:
            return idx, []
        try:
            geom = load_scenario_geometry(json_file)
            cfs  = run_counterfactuals_for_scenario(
                rows, geom, model,
                gamma=gamma, max_queries=max_queries,
                device=device, **opt_kwargs)
            return idx, cfs
        except Exception as exc:
            print(f"  [warn] CF failed for {json_file}: {exc}")
            return idx, []

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_run_one, (i, sc)): i
            for i, sc in enumerate(scenarios)
        }
        for fut in as_completed(futures):
            idx, cfs = fut.result()
            results[idx] = cfs

    return results
