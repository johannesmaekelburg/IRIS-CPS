"""
surrogate/repair_benchmark.py
=============================
Counterfactual-repair benchmark: gradient-based (surrogate) vs. derivative-free
(CMA-ES, finite-difference GD) search driven by sampling oracles (full MC, MFMC).

The point of the experiment (for the paper's RQ4 / counterfactual section): the
surrogate is differentiable, so repair is projected gradient descent with exact,
noise-free gradients and *zero* expensive oracle calls. Sampling estimators expose
only a noisy, non-differentiable oracle I(theta), forcing derivative-free search
that needs hundreds-to-thousands of expensive evaluations and is corrupted by
estimation noise (finite-differencing noisy MC estimates is especially unstable).

For every repair problem
    theta' = argmin_theta ||theta - theta*||  s.t.  I(theta) <= gamma,
we run five methods:
    surrogate     : Adam on the differentiable surrogate (0 expensive calls)
    cma_mc        : CMA-ES (nevergrad) on a full-MC oracle
    cma_mfmc      : CMA-ES on a multi-fidelity (control-variate) oracle
    fd_mc         : finite-difference GD on a full-MC oracle
    fd_mfmc       : finite-difference GD on the MFMC oracle

ALL final theta' are judged by the SAME high-accuracy MC arbiter (including the
surrogate's), so "success" is ground-truth feasibility, not a model's own
(possibly optimistic) estimate. We report, per method: expensive oracle calls,
LP solves (machine-independent cost), wall-time, success rate, and repair
distance ||theta'-theta*||.

Run (nevergrad injected ephemerally, not added to project deps):
    uv run --with nevergrad python -m surrogate.repair_benchmark --quick
    uv run --with nevergrad python -m surrogate.repair_benchmark \
        --n_problems 8 --budget 400 --out results/repair_benchmark
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch

# Silence nevergrad/cma's noisy optimizer warnings (orphanated injection, bounds-sigma).
try:
    from cma.evolution_strategy import InjectionWarning as _CMAInjWarn
    warnings.filterwarnings("ignore", category=_CMAInjWarn)
except Exception:
    warnings.filterwarnings("ignore", message=".*orphanated injected solution.*")
try:
    from nevergrad.common.errors import NevergradRuntimeWarning as _NgWarn
    warnings.filterwarnings("ignore", category=_NgWarn)
except Exception:
    warnings.filterwarnings("ignore", message=".*Bounds are.*sigma.*")

from .counterfactual import (
    PARAM_BOUNDS, PARAM_NAMES,
    DifferentiableIntervention, BatchedCounterfactualOptimizer,
    load_scenario_geometry,
)
from .generate_synthetic import interval_bounds, contains_point_lp, apply_intervention
from .models_v2 import load_checkpoint

_LB = np.array([b[0] for b in PARAM_BOUNDS], dtype=np.float64)
_UB = np.array([b[1] for b in PARAM_BOUNDS], dtype=np.float64)
_SPAN = _UB - _LB

# Paper evaluation set: HVAC (Fig 15) + the held-out joint_2d3d validation
# scenarios. All are 2D CPS except scenario_7 (the 3D CONVIDE holdout); all route
# to the 2d3d model. Scenarios with no moderate-inconsistency region simply yield
# zero repair problems and are skipped.
EVAL_SCENARIOS_2D3D = [
    "data/measurements_cps_v6/results_scenario_49.json",   # HVAC (paper Fig 15)
    "data/measurements_cps_v6/results_scenario_69.json",   # water chemical process
    "data/measurements_cps_v6/results_scenario_32.json",   # water chemical process
    "data/measurements_cps_v6/results_scenario_36.json",   # wind turbine
    "data/measurements_cps_v6/results_scenario_3.json",    # automotive
    "data/measurements_cps_v6/results_scenario_25.json",   # smart grid
    "data/measurements_cps_v6/results_scenario_42.json",   # satellite aerospace
    "data/measurements_cps_v6/results_scenario_1.json",    # automotive
    "data/measurements_cps_v6/results_scenario_57.json",   # medical device
    "data/measurements_cps_v6/results_scenario_62.json",   # satellite aerospace
    "data/measurements_cps_v6/results_scenario_51.json",   # building HVAC
    "data/measurements_cps_v6/results_scenario_58.json",   # railway
    "data/measurements_v6/results_scenario_7.json",        # CONVIDE 3D holdout
]


# ═══════════════════════════════════════════════════════════════════════════════
# Oracles  — estimate I(theta) for arbitrary theta from a fixed scenario geometry
# ═══════════════════════════════════════════════════════════════════════════════

def _mc_estimate(c_src, G_src, c_tgt, G_tgt, n_samples, rng):
    """Full Monte-Carlo I = 1 - P(src(xi) in tgt). Returns (I, n_lp_solves)."""
    p = G_src.shape[1]
    xi = rng.uniform(-1.0, 1.0, size=(n_samples, p))
    pts = c_src[None, :] + xi @ G_src.T
    lo, hi = interval_bounds(c_tgt, G_tgt)
    inside_box = np.all((pts >= lo) & (pts <= hi), axis=1)
    n_lp = 0
    n_cons = 0
    for i in np.where(inside_box)[0]:
        n_lp += 1
        if contains_point_lp(c_tgt, G_tgt, pts[i]):
            n_cons += 1
    return 1.0 - n_cons / n_samples, n_lp


def _mfmc_estimate(c_src, G_src, c_tgt, G_tgt, n_cheap, n_exp, rng):
    """Multi-fidelity control-variate estimate. Returns (I, n_lp_solves).

    Cheap proxy = membership in the target AABB (vectorised, no LP); its mean is
    estimated from a large cheap sample (low variance). The expensive exact
    containment (LP) is computed only on a small sample, and the control variate
    corrects the small-sample MC estimate using the proxy:
        I_MF = I_MC^small + alpha * (mu_box - I_box^small),
    with alpha = Cov(exact, box)/Var(box). Cost is dominated by the few LP solves
    on the small sample, so MFMC is cheaper per call than full MC at similar accuracy.
    """
    p = G_src.shape[1]
    lo, hi = interval_bounds(c_tgt, G_tgt)

    # Cheap, large sample: box membership only (no LP).
    xi_c = rng.uniform(-1.0, 1.0, size=(n_cheap, p))
    pts_c = c_src[None, :] + xi_c @ G_src.T
    box_c = np.all((pts_c >= lo) & (pts_c <= hi), axis=1).astype(np.float64)
    mu_box = 1.0 - box_c.mean()                      # low-variance proxy inconsistency

    # Expensive, small sample: box + exact (LP) membership.
    xi_e = rng.uniform(-1.0, 1.0, size=(n_exp, p))
    pts_e = c_src[None, :] + xi_e @ G_src.T
    box_e = np.all((pts_e >= lo) & (pts_e <= hi), axis=1)
    exact_e = np.zeros(n_exp, dtype=np.float64)
    n_lp = 0
    for i in np.where(box_e)[0]:
        n_lp += 1
        if contains_point_lp(c_tgt, G_tgt, pts_e[i]):
            exact_e[i] = 1.0
    box_ef = box_e.astype(np.float64)

    I_mc = 1.0 - exact_e.mean()
    I_box_small = 1.0 - box_ef.mean()
    var_box = box_ef.var()
    if var_box > 1e-12:
        alpha = np.cov(exact_e, box_ef)[0, 1] / var_box
    else:
        alpha = 0.0
    I_mf = I_mc + alpha * (mu_box - I_box_small)
    return float(np.clip(I_mf, 0.0, 1.0)), n_lp


class Oracle:
    """Callable I(theta) wrapping a sampling estimator; tracks cost."""

    def __init__(self, geom, kind, seed=0, **kw):
        d = geom["d"]
        self.pre_c = np.asarray(geom["pre_c"], dtype=np.float64)
        self.pre_G = np.asarray(geom["pre_G"], dtype=np.float64)
        self.tgt_c = np.asarray(geom["tgt_c"], dtype=np.float64)
        self.tgt_G = np.asarray(geom["tgt_G"], dtype=np.float64)
        us = geom.get("upr_scales")
        uo = geom.get("upr_offsets")
        self.us = np.ones(d) if us is None else np.asarray(us, dtype=np.float64)[:d]
        self.uo = np.zeros(d) if uo is None else np.asarray(uo, dtype=np.float64)[:d]
        self.kind = kind
        self.kw = kw
        self.rng = np.random.default_rng(seed)
        self.n_calls = 0
        self.n_lp = 0
        self.t = 0.0

    def _propagate(self, theta):
        sf, cd, cs = float(theta[0]), float(theta[1]), float(theta[2])
        c, G = apply_intervention(self.pre_c, self.pre_G, sf, cd, cs)
        return self.us * c + self.uo, self.us[:, None] * G

    def __call__(self, theta):
        self.n_calls += 1
        c_src, G_src = self._propagate(theta)
        t0 = time.perf_counter()
        if self.kind == "mc":
            I, n_lp = _mc_estimate(c_src, G_src, self.tgt_c, self.tgt_G,
                                   self.kw["n_samples"], self.rng)
        else:
            I, n_lp = _mfmc_estimate(c_src, G_src, self.tgt_c, self.tgt_G,
                                     self.kw["n_cheap"], self.kw["n_exp"], self.rng)
        self.t += time.perf_counter() - t0
        self.n_lp += n_lp
        return I


def _make_hybrid_oracle(geom, args, seed):
    """Oracle used by the hybrid's verify+refine. MC by default; MFMC is the cheaper
    control-variate estimator (same accuracy at ~1/3 the LP solves)."""
    if getattr(args, "hybrid_oracle", "mc") == "mfmc":
        return Oracle(geom, "mfmc", seed=seed,
                      n_cheap=args.mfmc_cheap, n_exp=args.mfmc_exp)
    return Oracle(geom, "mc", seed=seed, n_samples=args.mc_samples)


def _make_surr_oracle(geom, args, seed):
    """Oracle that verifies each surrogate restart. MFMC by default (cheap control-
    variate, nearly as accurate as full MC) so the surrogate is costed with the same
    deployable estimator as the hybrid — apples-to-apples wall-time."""
    if getattr(args, "surr_verify", "mfmc") == "mc":
        return Oracle(geom, "mc", seed=seed, n_samples=args.mc_samples)
    return Oracle(geom, "mfmc", seed=seed,
                  n_cheap=args.mfmc_cheap, n_exp=args.mfmc_exp)


def hifi_mc(geom, theta, n_samples, seed):
    """High-accuracy MC arbiter (fixed seed) for validating any theta."""
    d = geom["d"]
    us = geom.get("upr_scales")
    uo = geom.get("upr_offsets")
    us = np.ones(d) if us is None else np.asarray(us, dtype=np.float64)[:d]
    uo = np.zeros(d) if uo is None else np.asarray(uo, dtype=np.float64)[:d]
    c, G = apply_intervention(np.asarray(geom["pre_c"], float),
                              np.asarray(geom["pre_G"], float),
                              float(theta[0]), float(theta[1]), float(theta[2]))
    c_src, G_src = us * c + uo, us[:, None] * G
    I, _ = _mc_estimate(c_src, G_src, np.asarray(geom["tgt_c"], float),
                        np.asarray(geom["tgt_G"], float), n_samples,
                        np.random.default_rng(seed))
    return I


# ═══════════════════════════════════════════════════════════════════════════════
# Repair methods
# ═══════════════════════════════════════════════════════════════════════════════

def _penalty_obj(theta, theta_star, I, gamma, lam):
    dist2 = float(np.sum(((np.asarray(theta) - theta_star) / _SPAN) ** 2))
    return dist2 + lam * max(0.0, I - gamma) ** 2


def repair_surrogate(model, geom, theta_star, gamma, device, margin=0.0,
                     verify_oracle=None, verify_all=False, **opt_kw):
    """Gradient-based repair on the differentiable surrogate (0 expensive calls).

    margin : drive the surrogate below (gamma - margin) to absorb surrogate
             error near the boundary (validated against gamma).
    verify_oracle : optional Oracle; if given, EACH gradient restart's candidate is
             verified with one true oracle call and the best is selected by ground
             truth (lowest true I for consistency; nearest feasible, else lowest true
             I, for repair). These verification calls are counted — the honest
             "surrogate proposes K, oracle verifies K" workflow. Cost is therefore
             (n_restarts + 1) oracle calls, still negligible vs. derivative-free search.
    """
    interv = DifferentiableIntervention(
        geom["pre_c"], geom["pre_G"], geom["tgt_c"], geom["tgt_G"],
        geom.get("upr_scales"), geom.get("upr_offsets"), geom.get("upr_onehot"),
        device, use_v2=True,
    ).to(device)
    opt = BatchedCounterfactualOptimizer(model, interv, gamma=gamma - margin,
                                         device=device, **opt_kw)
    t0 = time.perf_counter()
    res = opt.find(np.asarray(theta_star, dtype=np.float32))
    wall = time.perf_counter() - t0
    info = {"wall": wall, "oracle_calls": 0, "n_lp": 0,
            "model_evals": int(res.n_iter) * (opt_kw.get("n_restarts", 4) + 1)}
    theta_prime = np.asarray(res.theta_prime, dtype=np.float64)

    if verify_oracle is not None:
        if verify_all:
            # oracle-assisted restart selection: verify every restart, keep best by
            # ground truth. Costs (n_restarts+1) calls.
            cands, _ = opt.last_candidates[0]             # (R, 3) restart candidates
            ts = np.asarray(theta_star, dtype=np.float64)
            I_true = np.array([verify_oracle(c) for c in cands])
            if opt.objective == "consistency":
                best = int(np.argmin(I_true))             # most consistent by true MC
            else:
                feas = I_true <= gamma
                if feas.any():
                    d = np.sum(((cands - ts) / _SPAN) ** 2, axis=1)
                    d[~feas] = np.inf
                    best = int(np.argmin(d))              # nearest truly-feasible
                else:
                    best = int(np.argmin(I_true))         # fall back to least inconsistent
            theta_prime = np.asarray(cands[best], dtype=np.float64)
        else:
            # "pure surrogate": verify only the single theta' you would deploy (1 call).
            verify_oracle(theta_prime)
        info["oracle_calls"] = verify_oracle.n_calls
        info["n_lp"] = verify_oracle.n_lp
        info["wall"] += verify_oracle.t
    return theta_prime, info


def repair_cmaes(oracle, theta_star, gamma, budget, lam, margin=0.0, seed=0,
                 objective="repair"):
    """CMA-ES (nevergrad) on a sampling oracle.

    objective="repair"      : minimise dist^2 + lam*max(0,I-(gamma-margin))^2.
    objective="consistency" : minimise I directly (distance ignored).
    margin : operational safety margin for the repair penalty (validation vs gamma).
    """
    try:
        import nevergrad as ng
    except ImportError as e:
        raise ImportError(
            "nevergrad required. Run with:  uv run --with nevergrad python -m "
            "surrogate.repair_benchmark ...") from e
    cons = objective == "consistency"
    g_eff = gamma - margin
    ts = np.asarray(theta_star, dtype=np.float64)
    param = ng.p.Array(init=ts.copy()).set_bounds(_LB, _UB)
    param.random_state.seed(seed)
    optim = ng.optimizers.CMA(parametrization=param, budget=budget)

    # Track the best-incumbent point by objective (nevergrad's recommendation is
    # the distribution mean, which is often infeasible under a noisy objective).
    best = {"obj": float("inf"), "x": ts.copy()}

    def obj(x):
        I = oracle(x)
        val = I if cons else _penalty_obj(x, ts, I, g_eff, lam)
        if val < best["obj"]:
            best["obj"] = val
            best["x"] = np.asarray(x, dtype=np.float64).copy()
        return val

    t0 = time.perf_counter()
    optim.minimize(obj)
    wall = time.perf_counter() - t0
    return best["x"], {
        "wall": wall, "oracle_calls": oracle.n_calls, "n_lp": oracle.n_lp,
    }


def repair_cmaes_surrogate(model, geom, theta_star, gamma, device, margin, budget,
                           lam, objective="repair", verify_oracle=None, seed=0):
    """CMA-ES that uses the SURROGATE as the search model (no MC oracle in the loop).

    Ablation isolating the value of a cheap evaluator from the value of the gradient:
    derivative-free search driven by the surrogate Î costs zero expensive calls during
    search; the final pick is verified once by true MC (counted). If this matches the
    gradient surrogate, the win is "cheap evaluator"; if the gradient is better/faster,
    the differentiability itself adds value.
    """
    try:
        import nevergrad as ng
    except ImportError as e:
        raise ImportError("nevergrad required (run with --with nevergrad).") from e
    cons = objective == "consistency"
    interv = DifferentiableIntervention(
        geom["pre_c"], geom["pre_G"], geom["tgt_c"], geom["tgt_G"],
        geom.get("upr_scales"), geom.get("upr_offsets"), geom.get("upr_onehot"),
        device, use_v2=True,
    ).to(device)
    g_eff = gamma - margin
    ts = np.asarray(theta_star, dtype=np.float64)
    param = ng.p.Array(init=ts.copy()).set_bounds(_LB, _UB)
    param.random_state.seed(seed)
    optim = ng.optimizers.CMA(parametrization=param, budget=budget)
    best = {"obj": float("inf"), "x": ts.copy()}

    def surr_I(x):
        with torch.no_grad():
            xt = torch.tensor(np.asarray(x)[None], dtype=torch.float32, device=device)
            pd, m, gf = interv(xt)
            return float(model(pd, m, gf)[0])

    t0 = time.perf_counter()
    for _ in range(budget):
        cand = optim.ask()
        x = np.asarray(cand.value, dtype=np.float64)
        I = surr_I(x)
        val = I if cons else _penalty_obj(x, ts, I, g_eff, lam)
        optim.tell(cand, val)
        if val < best["obj"]:
            best["obj"] = val
            best["x"] = x.copy()
    wall = time.perf_counter() - t0
    info = {"wall": wall, "oracle_calls": 0, "n_lp": 0, "model_evals": budget}
    tp = best["x"]
    if verify_oracle is not None:
        verify_oracle(tp)                                # one true verification call
        info["oracle_calls"] = verify_oracle.n_calls
        info["n_lp"] = verify_oracle.n_lp
        info["wall"] += verify_oracle.t
    return tp, info


def repair_finite_diff(oracle, theta_star, gamma, budget, lam,
                       lr=0.08, h=0.06, start=None, margin=0.0, objective="repair"):
    """Central-difference gradient descent on a noisy sampling oracle.

    Exposes the noise problem: the gradient differences two noisy oracle
    estimates, so its variance scales as ~sigma^2/h^2.

    objective="repair"      : minimise dist^2 + lam*max(0,I-(gamma-margin))^2.
    objective="consistency" : minimise I directly (distance ignored).
    theta_star : the anchor (distance is measured to it).
    start      : starting point (defaults to theta_star); for the hybrid we
                 warm-start from the surrogate's proposal.
    """
    cons = objective == "consistency"
    g_eff = gamma - margin
    ts = np.asarray(theta_star, dtype=np.float64)
    theta = ts.copy() if start is None else np.asarray(start, dtype=np.float64).copy()

    def objval(x):
        I = oracle(x)
        return I if cons else _penalty_obj(x, ts, I, g_eff, lam)

    t0 = time.perf_counter()
    best = theta.copy()
    best_obj = objval(theta)
    while oracle.n_calls < budget:
        grad = np.zeros(3)
        for j in range(3):
            tp = theta.copy(); tp[j] = np.clip(tp[j] + h * _SPAN[j], _LB[j], _UB[j])
            tm = theta.copy(); tm[j] = np.clip(tm[j] - h * _SPAN[j], _LB[j], _UB[j])
            op = objval(tp); om = objval(tm)
            grad[j] = (op - om) / (2.0 * h * _SPAN[j] + 1e-12)
            if oracle.n_calls >= budget:
                break
        theta = np.clip(theta - lr * grad * _SPAN, _LB, _UB)
        o = objval(theta)
        if o < best_obj:
            best_obj, best = o, theta.copy()
    wall = time.perf_counter() - t0
    return best.astype(np.float64), {
        "wall": wall, "oracle_calls": oracle.n_calls, "n_lp": oracle.n_lp,
    }


def repair_hybrid(model, geom, theta_star, gamma, device, oracle, margin,
                  refine_budget, lam, objective="repair", **opt_kw):
    """Surrogate proposes (free) -> MC verifies (1 call) -> oracle refines.

    repair      : verify the proposal; if already feasible (I<=gamma) stop at 1 call,
                  else run a few oracle finite-diff steps from the warm start to fix
                  the blind-spot. Cost = 1 .. ~refine_budget calls.
    consistency : there is no feasibility gate, so always run refine_budget oracle
                  finite-diff steps from the surrogate warm-start to push I below the
                  level the (saturating) surrogate can reach on its own. This is the
                  cheap bridge between the free surrogate (I~plateau) and cold-start
                  derivative-free search (I~0 at 50-400 calls).
    """
    tp, sinfo = repair_surrogate(model, geom, theta_star, gamma, device,
                                 margin=margin, verify_oracle=None,
                                 objective=objective, **opt_kw)
    I = oracle(tp)                                   # 1 verification call
    if objective != "consistency" and I <= gamma:
        return tp, {"wall": sinfo["wall"] + oracle.t,
                    "oracle_calls": oracle.n_calls, "n_lp": oracle.n_lp}
    # Refine with the true oracle from the surrogate warm-start.
    tp2, _ = repair_finite_diff(oracle, theta_star, gamma,
                                budget=oracle.n_calls + refine_budget,
                                lam=lam, start=tp, margin=margin, objective=objective)
    return tp2, {"wall": sinfo["wall"] + oracle.t,
                 "oracle_calls": oracle.n_calls, "n_lp": oracle.n_lp}


# ═══════════════════════════════════════════════════════════════════════════════
# Problem generation + harness
# ═══════════════════════════════════════════════════════════════════════════════

def make_problems(model, geom, gamma, n_problems, device, rng,
                  val_samples, n_candidates=800, band_lo=0.05, band_hi=0.18,
                  true_lo=None, true_hi=0.75):
    """Sample theta* with a controllable inconsistency band that has genuine nearby
    repairs.

    Difficulty (= distance to feasibility) is set by the TRUE-MC band
    [true_lo, true_hi]; true_lo defaults to gamma (moderate). Pushing true_lo up gives
    FARTHER starts (deeper in the inconsistent region). We still cap true_hi to avoid
    degenerate points whose minimal repair does not exist in-box (every method fails).
    The surrogate-I band [gamma+band_lo, gamma+band_hi] pre-selects candidates cheaply.
    """
    true_lo = gamma if true_lo is None else true_lo
    cand = _LB + rng.random((n_candidates, 3)) * _SPAN
    interv = DifferentiableIntervention(
        geom["pre_c"], geom["pre_G"], geom["tgt_c"], geom["tgt_G"],
        geom.get("upr_scales"), geom.get("upr_offsets"), geom.get("upr_onehot"),
        device, use_v2=True,
    ).to(device)
    with torch.no_grad():
        pd, m, gf = interv(torch.tensor(cand, dtype=torch.float32, device=device))
        I_s = model(pd, m, gf).cpu().numpy()
    band = np.where((I_s > gamma + band_lo) & (I_s < gamma + band_hi))[0]
    rng.shuffle(band)
    problems = []
    for idx in band:
        th = cand[idx]
        i_true = hifi_mc(geom, th, val_samples, seed=12345)
        if true_lo < i_true < true_hi:
            problems.append((th.astype(np.float64), float(I_s[idx])))
        if len(problems) >= n_problems:
            break
    return problems


# Surrogate-repair hyperparameter search space (scored by TRUE MC, not surrogate
# convergence). Sampled by random search since the full grid (optimizer x lr x
# restarts x iters x margin x lambda) is combinatorially large.
HPO_LR       = [0.01, 0.02, 0.05, 0.1]
HPO_RESTARTS = [1, 4, 8, 16]
HPO_ITERS    = [150, 300, 500]
HPO_MARGIN   = [0.0, 0.05, 0.1, 0.15, 0.2]
HPO_OPT      = ["adam", "momentum", "nesterov", "rmsprop"]
HPO_LAM      = [10.0, 30.0, 100.0, 300.0]   # objective balance (consistency vs. distance)


def _sample_hpo_configs(rng, n):
    """Draw n distinct random configs from the search space."""
    seen, configs = set(), []
    attempts = 0
    while len(configs) < n and attempts < 50 * n:
        attempts += 1
        cfg = (
            float(rng.choice(HPO_LR)), int(rng.choice(HPO_RESTARTS)),
            int(rng.choice(HPO_ITERS)), float(rng.choice(HPO_MARGIN)),
            str(rng.choice(HPO_OPT)), float(rng.choice(HPO_LAM)),
        )
        if cfg not in seen:
            seen.add(cfg)
            configs.append(cfg)
    return configs


def _score_hpo_config(cfg, problems, gamma, device, objective, arbiter,
                      val_samples, mfmc_cheap, mfmc_exp):
    """Run the surrogate with one hyperparameter config over all validation problems
    and score the converged points with `arbiter` ('mfmc' = cheap deployable estimator,
    'hifi' = high-accuracy MC). No expensive oracle is used during the surrogate search;
    each converged theta' gets ONE arbiter evaluation, so the offline tuning cost mirrors
    the cheap-verify deployment workflow. lambda is fixed per config (lambda_scale=1) so it
    is a clean consistency-vs-distance knob (inert for the consistency objective)."""
    lr, nr, it = cfg["lr"], cfg["restarts"], cfg["iters"]
    mg, opt, lam = cfg.get("margin", 0.0), cfg["optimizer"], cfg.get("lam", 100.0)
    succ, dists, i_arb = 0, [], []
    for pi, (geom, model, th) in enumerate(problems):
        tp, _ = repair_surrogate(
            model, geom, th, gamma, device, margin=mg,
            verify_oracle=None, lr=lr, n_restarts=nr, max_iter=it,
            patience=it, optimizer=opt, lambda_init=lam, lambda_scale=1.0,
            objective=objective)
        if arbiter == "mfmc":
            # fresh fixed-seed MFMC per problem -> common random numbers across configs
            i_true = Oracle(geom, "mfmc", seed=4242 + pi,
                            n_cheap=mfmc_cheap, n_exp=mfmc_exp)(tp)
        else:
            i_true = hifi_mc(geom, tp, val_samples, seed=999)
        i_arb.append(i_true)
        if i_true <= gamma:
            succ += 1
            dists.append(float(np.sqrt(np.sum(((tp - th) / _SPAN) ** 2))))
    return {
        "optimizer": opt, "lr": float(lr), "restarts": int(nr), "iters": int(it),
        "margin": float(mg), "lam": float(lam),
        "success_rate": succ / len(problems),
        "dist_med": float(np.median(dists)) if dists else float("nan"),
        "i_med": float(np.median(i_arb)),
        "arbiter": arbiter,
    }


def _hpo_loss(m, objective):
    """Scalar nevergrad minimises. consistency -> achieved I; repair -> feasibility
    first (each 10% of success ~= 1.0), distance as the tie-breaker."""
    if objective == "consistency":
        return float(m["i_med"])
    fail = 1.0 - m["success_rate"]
    dist = m["dist_med"] if np.isfinite(m["dist_med"]) else 1.5
    return float(fail + 0.1 * dist)


def _hpo_problems(args, device):
    """Build the shared validation problem set (separate seed from the benchmark)."""
    m2 = load_checkpoint(args.model_2d3d, map_location=args.device)
    m4 = load_checkpoint(args.model_4d, map_location=args.device) if args.scenarios_4d else None
    scenarios = [(f, "2d3d") for f in args.scenarios_2d3d] + \
                [(f, "4d") for f in args.scenarios_4d]
    rng = np.random.default_rng(7)
    problems, per_scenario = [], []
    for sc_path, group in scenarios:
        geom = load_scenario_geometry(Path(sc_path))
        model = m2 if group == "2d3d" else m4
        k0 = len(problems)
        for th, _ in make_problems(model, geom, args.gamma, args.n_problems,
                                   device, rng, args.val_samples):
            problems.append((geom, model, th))
        per_scenario.append((sc_path, geom["d"], len(problems) - k0))
    print("HPO validation problems per scenario:")
    for sc_path, d, k in per_scenario:
        print(f"  {Path(sc_path).name:<28} d={d}  ->  {k} problems")
    return problems


def _run_hpo_nevergrad(args, problems, device):
    """nevergrad (NGOpt) search over the surrogate hyperparameters, scored by the
    `--hpo_arbiter` estimator. Mixed parametrization: Log-scaled lr/lambda, integer
    restarts/iters, categorical optimizer, scalar margin. For the consistency objective
    lambda/margin are inert and dropped from the search space."""
    import nevergrad as ng
    cons = args.objective == "consistency"
    space = dict(
        lr=ng.p.Log(lower=1e-4, upper=5e-1),
        restarts=ng.p.Scalar(lower=0, upper=8).set_integer_casting(),
        iters=ng.p.Scalar(lower=100, upper=1000).set_integer_casting(),
        optimizer=ng.p.Choice(HPO_OPT),
    )
    if not cons:
        space["margin"] = ng.p.Scalar(lower=0.0, upper=0.2)
        space["lam"] = ng.p.Log(lower=10.0, upper=300.0)
    param = ng.p.Instrumentation(**space)
    optimizer = ng.optimizers.NGOpt(parametrization=param, budget=args.hpo_samples)
    print(f"\nnevergrad NGOpt: {args.hpo_samples} evals x {len(problems)} problems "
          f"(arbiter={args.hpo_arbiter}, objective={args.objective})")

    from tqdm import tqdm
    pbar = tqdm(total=args.hpo_samples, unit="config", desc=f"HPO-ng[{args.objective}]")
    rows = []
    for i in range(args.hpo_samples):
        cand = optimizer.ask()
        cfg = dict(cand.kwargs)
        m = _score_hpo_config(cfg, problems, args.gamma, device, args.objective,
                              args.hpo_arbiter, args.val_samples,
                              args.mfmc_cheap, args.mfmc_exp)
        loss = _hpo_loss(m, args.objective)
        optimizer.tell(cand, loss)
        rows.append({**m, "loss": loss})
        best = min(rows, key=lambda r: r["loss"])
        pbar.set_postfix(best=f"{best['loss']:.3f}")
        pbar.update(1)
        tqdm.write(
            f"  [{i+1:>2}/{args.hpo_samples}] opt={m['optimizer']:<9} lr={m['lr']:<7.4f} "
            f"restarts={m['restarts']:<2} iters={m['iters']:<4} margin={m['margin']:<4.2f} "
            f"lam={m['lam']:<6.1f} -> succ={m['success_rate']*100:>4.0f}%  "
            f"I_med={m['i_med']:.3f}  dist={m['dist_med']:.3f}  loss={loss:.3f}")
    pbar.close()
    return rows


def run_hpo(args):
    """Tune surrogate hyperparameters by the TRUE outcome of the proposed repair
    (feasibility + distance / achieved I), not by surrogate convergence.

    --hpo_method nevergrad : NGOpt black-box search over the space (default).
    --hpo_method random    : random search over the discrete grid.
    --hpo_arbiter mfmc     : score each converged point with the cheap MFMC estimator
                             (default; mirrors the deployable cheap-verify workflow);
                             hifi uses high-accuracy MC instead.
    The selected config is then re-validated with HiFi MC so cheap-arbiter selection is
    honestly checked against the true arbiter. (Separate problem seed from the benchmark.)
    """
    device = torch.device(args.device)
    problems = _hpo_problems(args, device)
    cons = args.objective == "consistency"

    if args.hpo_method == "nevergrad":
        rows = _run_hpo_nevergrad(args, problems, device)
    else:
        rng = np.random.default_rng(7)
        configs = _sample_hpo_configs(rng, args.hpo_samples)
        print(f"\nrandom search: {len(configs)} configs x {len(problems)} problems "
              f"(arbiter={args.hpo_arbiter}, objective={args.objective})")
        from tqdm import tqdm
        rows = []
        for ci, (lr, nr, it, mg, opt, lam) in enumerate(
                tqdm(configs, unit="config", desc=f"HPO[{args.objective}]")):
            cfg = dict(lr=lr, restarts=nr, iters=it, margin=mg, optimizer=opt, lam=lam)
            m = _score_hpo_config(cfg, problems, args.gamma, device, args.objective,
                                  args.hpo_arbiter, args.val_samples,
                                  args.mfmc_cheap, args.mfmc_exp)
            rows.append({**m, "loss": _hpo_loss(m, args.objective)})

    rows.sort(key=lambda r: r["loss"])

    # Re-validate the top configs with HiFi MC: cheap-arbiter selection vs. true arbiter.
    revalidated = []
    if args.hpo_arbiter != "hifi":
        print("\nRe-validating top configs with HiFi MC (cheap-MFMC selection -> true arbiter):")
        for r in rows[:min(5, len(rows))]:
            cfg = {k: r[k] for k in ("lr", "restarts", "iters", "margin", "optimizer", "lam")}
            hv = _score_hpo_config(cfg, problems, args.gamma, device, args.objective,
                                   "hifi", args.val_samples, args.mfmc_cheap, args.mfmc_exp)
            r["hifi_success_rate"] = hv["success_rate"]
            r["hifi_i_med"] = hv["i_med"]
            r["hifi_dist_med"] = hv["dist_med"]
            revalidated.append(r)
            print(f"  opt={r['optimizer']:<9} lr={r['lr']:<7.4f} restarts={r['restarts']:<2} "
                  f"iters={r['iters']:<4} margin={r['margin']:<4.2f} lam={r['lam']:<6.1f} | "
                  f"mfmc: succ={r['success_rate']*100:>3.0f}% I={r['i_med']:.3f} "
                  f"| hifi: succ={hv['success_rate']*100:>3.0f}% I={hv['i_med']:.3f} "
                  f"dist={hv['dist_med']:.3f}")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "hpo.json").write_text(json.dumps(rows, indent=2))
    crit = "achieved I" if cons else "feasibility, then distance (loss)"
    print(f"\nBest configs (by {crit}, {args.hpo_method}/{args.hpo_arbiter}):")
    for r in rows[:8]:
        print(f"  opt={r['optimizer']:<9} lr={r['lr']:<7.4f} restarts={r['restarts']:<2} "
              f"iters={r['iters']:<4} margin={r['margin']:<4.2f} lam={r['lam']:<6.1f} "
              f"succ={r['success_rate']*100:.0f}% I_med={r['i_med']:.3f} "
              f"dist={r['dist_med']:.3f} loss={r['loss']:.3f}")
    print(f"\nSaved {out}/hpo.json")


def run_benchmark(args):
    device = torch.device(args.device)
    model_2d3d = load_checkpoint(args.model_2d3d, map_location=args.device)
    model_4d = load_checkpoint(args.model_4d, map_location=args.device)

    scenarios = [(f, "2d3d") for f in args.scenarios_2d3d] + \
                [(f, "4d") for f in args.scenarios_4d]

    cons = args.objective == "consistency"
    surr_kw = dict(max_iter=args.surr_iters, n_restarts=args.surr_restarts,
                   lr=args.surr_lr, patience=args.surr_iters,
                   optimizer=args.surr_optimizer, objective=args.objective,
                   lambda_init=args.surr_lambda_init,
                   lambda_scale=args.surr_lambda_scale)
    records = []
    rng = np.random.default_rng(0)

    # ── Pre-pass: generate all repair problems so the progress bar has a total ──
    tasks = []   # (sc_path, geom, model, qi, theta_star)
    for sc_path, group in scenarios:
        geom = load_scenario_geometry(Path(sc_path))
        model = model_2d3d if group == "2d3d" else model_4d
        problems = make_problems(model, geom, args.gamma, args.n_problems,
                                 device, rng, args.val_samples,
                                 n_candidates=args.n_candidates,
                                 band_lo=args.band_lo, band_hi=args.band_hi,
                                 true_lo=args.true_lo, true_hi=args.true_hi)
        print(f"[{Path(sc_path).name:<28} d={geom['d']}]  ->  {len(problems)} repair problems")
        for qi, (theta_star, _) in enumerate(problems):
            tasks.append((sc_path, geom, model, qi, theta_star))

    # derivative-free methods to sweep; skip the full-MC variants when --mfmc_only
    dfo_methods = [("cma_mfmc", "mfmc", "cma"), ("fd_mfmc", "mfmc", "fd")]
    if not args.mfmc_only:
        dfo_methods = [("cma_mc", "mc", "cma"), ("fd_mc", "mc", "fd")] + dfo_methods

    from tqdm import tqdm
    # base methods (no oracle-budget sweep): surrogate + hybrid + surr_cma
    n_base = 3
    per_problem = n_base + len(args.budgets) * len(dfo_methods)
    pbar = tqdm(total=len(tasks) * per_problem, unit="eval", desc=f"benchmark[{args.objective}]")
    n_ok = 0

    for sc_path, geom, model, qi, theta_star in tasks:
        i_true_star = hifi_mc(geom, theta_star, args.val_samples, seed=999)

        def _val(theta_prime, info, name, budget=None):
            nonlocal n_ok
            i_true = hifi_mc(geom, theta_prime, args.val_samples, seed=999)
            dist = float(np.sqrt(np.sum(((theta_prime - theta_star) / _SPAN) ** 2)))
            rec = {
                "scenario": sc_path, "dim": geom["d"], "query": qi, "method": name,
                "budget": budget, "objective": args.objective,
                "theta_star": theta_star.tolist(), "theta_prime": theta_prime.tolist(),
                "i_true_star": i_true_star, "i_true_prime": i_true,
                "success": bool(i_true <= args.gamma), "dist": dist,
                "oracle_calls": info["oracle_calls"], "n_lp": info["n_lp"],
                "wall": info["wall"],
            }
            records.append(rec)
            n_ok += int(rec["success"])
            pbar.update(1)
            pbar.set_postfix(succ=f"{n_ok}/{len(records)}")

        # 1) surrogate (gradient): propose one candidate per restart, then verify
        #    EACH with a true MC oracle and keep the best by ground truth.
        verify = _make_surr_oracle(geom, args, seed=10_000 + qi)
        tp, info = repair_surrogate(model, geom, theta_star, args.gamma,
                                    device, margin=args.surr_margin,
                                    verify_oracle=verify,
                                    verify_all=args.surr_verify_all, **surr_kw)
        _val(tp, info, "surrogate")

        # 1b) hybrid: surrogate proposes -> verify -> oracle refine
        #     (repair: only if infeasible; consistency: always, to push I lower)
        horacle = _make_hybrid_oracle(geom, args, seed=20_000 + qi)
        tp, info = repair_hybrid(model, geom, theta_star, args.gamma, device,
                                 horacle, margin=args.surr_margin,
                                 refine_budget=args.hybrid_refine, lam=args.lam,
                                 **surr_kw)
        _val(tp, info, "hybrid")

        # 1c) surrogate-CMA: CMA-ES using the surrogate as the search model (0 oracle
        #     calls during search) + one true verify. Isolates evaluator vs. gradient.
        scma_verify = Oracle(geom, "mc", seed=30_000 + qi, n_samples=args.mc_samples)
        tp, info = repair_cmaes_surrogate(
            model, geom, theta_star, args.gamma, device, margin=args.surr_margin,
            budget=args.surr_cma_budget, lam=args.lam, objective=args.objective,
            verify_oracle=scma_verify, seed=qi)
        _val(tp, info, "surr_cma")

        # 2-5) derivative-free with sampling oracles, swept over budget
        for budget in args.budgets:
            for name, kind, opt in dfo_methods:
                if kind == "mc":
                    oracle = Oracle(geom, "mc", seed=qi, n_samples=args.mc_samples)
                else:
                    oracle = Oracle(geom, "mfmc", seed=qi,
                                    n_cheap=args.mfmc_cheap, n_exp=args.mfmc_exp)
                if opt == "cma":
                    tp, info = repair_cmaes(oracle, theta_star, args.gamma,
                                            budget, args.lam, margin=args.surr_margin,
                                            seed=qi, objective=args.objective)
                else:
                    tp, info = repair_finite_diff(oracle, theta_star, args.gamma,
                                                  budget, args.lam,
                                                  margin=args.surr_margin,
                                                  objective=args.objective)
                _val(tp, info, name, budget=budget)
    pbar.close()

    # ── Aggregate + save ──
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "records.json").write_text(json.dumps(records, indent=2))
    _summarize(records, out)
    print(f"\nSaved {len(records)} records to {out}/")


def run_add_method(args, rec_path, method):
    """Compute ONLY `method` (surr_cma or hybrid) over the SAME problems as an
    existing records.json and merge it in (idempotent; backs up first).

    Lets us add a method to an expensive completed run without re-running the hours
    of derivative-free search. theta* and seeds come from the existing records, so the
    new points are directly comparable. The objective (repair/consistency) is read
    per-record, so this works on both the repair and the min-I records files.
    """
    device = torch.device(args.device)
    rec_path = Path(rec_path)
    recs = json.loads(rec_path.read_text())

    probs = {}
    for r in recs:
        probs.setdefault((r["scenario"], r["query"]), r)
    dims = {r["dim"] for r in probs.values()}
    m2 = load_checkpoint(args.model_2d3d, map_location=args.device)
    m4 = load_checkpoint(args.model_4d, map_location=args.device) if 4 in dims else None
    surr_kw = dict(max_iter=args.surr_iters, n_restarts=args.surr_restarts,
                   lr=args.surr_lr, patience=args.surr_iters,
                   optimizer=args.surr_optimizer,
                   lambda_init=args.surr_lambda_init,
                   lambda_scale=args.surr_lambda_scale)

    from tqdm import tqdm
    new = []
    for (scenario, query), r in tqdm(sorted(probs.items()), unit="problem", desc=method):
        geom = load_scenario_geometry(Path(scenario))
        model = m4 if r["dim"] == 4 else m2
        theta_star = np.asarray(r["theta_star"], dtype=np.float64)
        objective = r.get("objective", "repair")
        if method == "surrogate":
            verify = _make_surr_oracle(geom, args, seed=10_000 + query)
            tp, info = repair_surrogate(
                model, geom, theta_star, args.gamma, device, margin=args.surr_margin,
                verify_oracle=verify, verify_all=args.surr_verify_all,
                objective=objective, **surr_kw)
        elif method == "surr_cma":
            verify = Oracle(geom, "mc", seed=30_000 + query, n_samples=args.mc_samples)
            tp, info = repair_cmaes_surrogate(
                model, geom, theta_star, args.gamma, device, margin=args.surr_margin,
                budget=args.surr_cma_budget, lam=args.lam, objective=objective,
                verify_oracle=verify, seed=query)
        else:   # hybrid
            oracle = _make_hybrid_oracle(geom, args, seed=20_000 + query)
            tp, info = repair_hybrid(
                model, geom, theta_star, args.gamma, device, oracle,
                margin=args.surr_margin, refine_budget=args.hybrid_refine,
                lam=args.lam, objective=objective, **surr_kw)
        i_true = hifi_mc(geom, tp, args.val_samples, seed=999)
        dist = float(np.sqrt(np.sum(((tp - theta_star) / _SPAN) ** 2)))
        new.append({
            "scenario": scenario, "dim": r["dim"], "query": query,
            "method": method, "budget": None, "objective": objective,
            "theta_star": theta_star.tolist(), "theta_prime": tp.tolist(),
            "i_true_star": r.get("i_true_star"), "i_true_prime": i_true,
            "success": bool(i_true <= args.gamma), "dist": dist,
            "oracle_calls": info["oracle_calls"], "n_lp": info["n_lp"],
            "wall": info["wall"],
        })

    # Back up, then merge (drop any prior records of this method for idempotency).
    backup = rec_path.with_suffix(".backup.json")
    backup.write_text(json.dumps(recs, indent=2))
    merged = [r for r in recs if r["method"] != method] + new
    rec_path.write_text(json.dumps(merged, indent=2))
    _summarize(merged, rec_path.parent)
    print(f"\nAdded {len(new)} {method} records to {rec_path}  (backup: {backup.name})")


def _summarize(records, out):
    methods = ["surrogate", "hybrid", "surr_cma", "cma_mc", "cma_mfmc", "fd_mc", "fd_mfmc"]
    # group keys: (method, budget) so a budget sweep shows one row per budget
    keys = []
    for m in methods:
        for b in sorted({r.get("budget") for r in records if r["method"] == m},
                        key=lambda x: (x is not None, x)):
            keys.append((m, b))
    print("\n" + "=" * 98)
    print(f"{'method':10s} {'budget':>7s} {'n':>3s} {'succ%':>6s} {'I(med)':>8s} "
          f"{'calls(med)':>11s} {'lp(med)':>9s} {'wall_s(med)':>12s} {'dist(med)':>10s}")
    print("-" * 98)
    summary = {}
    med = lambda key, src: float(np.median([r[key] for r in src])) if src else float("nan")
    for mname, b in keys:
        rs = [r for r in records if r["method"] == mname and r.get("budget") == b]
        if not rs:
            continue
        succ = [r for r in rs if r["success"]]
        row = {
            "budget": b, "n": len(rs), "success_rate": len(succ) / len(rs),
            "i_med": med("i_true_prime", rs),
            "calls_med": med("oracle_calls", rs), "lp_med": med("n_lp", rs),
            "wall_med": med("wall", rs), "dist_med_success": med("dist", succ),
        }
        summary[f"{mname}@{b}"] = row
        btag = "-" if b is None else str(b)
        print(f"{mname:10s} {btag:>7s} {row['n']:>3d} {row['success_rate']*100:>5.0f}% "
              f"{row['i_med']:>8.3f} {row['calls_med']:>11.0f} {row['lp_med']:>9.0f} "
              f"{row['wall_med']:>12.3f} {row['dist_med_success']:>10.3f}")
    print("=" * 98)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_2d3d", default="results/runs/joint_2d3d_exact/product_transformer_exact.pt")
    p.add_argument("--model_4d", default="results/runs/final_4d_real_exact/product_transformer_exact.pt")
    p.add_argument("--scenarios_2d3d", nargs="+", default=EVAL_SCENARIOS_2D3D)
    p.add_argument("--scenarios_4d", nargs="*", default=[])
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--surr_margin", type=float, default=0.1,
                   help="drive surrogate below gamma-margin to absorb model error")
    p.add_argument("--hybrid_refine", type=int, default=42,
                   help="oracle-eval budget for hybrid refinement when verify fails")
    p.add_argument("--hybrid_oracle", choices=["mc", "mfmc"], default="mc",
                   help="estimator the hybrid uses for verify+refine (mfmc is ~3x cheaper)")
    p.add_argument("--n_problems", type=int, default=6)
    p.add_argument("--budget", type=int, default=400, help="oracle-eval budget for DFO methods")
    p.add_argument("--budgets", type=int, nargs="*", default=None,
                   help="sweep DFO over these budgets (overrides --budget) for the Pareto figure")
    p.add_argument("--lam", type=float, default=100.0, help="constraint penalty weight")
    p.add_argument("--mc_samples", type=int, default=400)
    p.add_argument("--mfmc_cheap", type=int, default=4000)
    p.add_argument("--mfmc_exp", type=int, default=150)
    p.add_argument("--val_samples", type=int, default=4000, help="HiFi MC arbiter samples")
    p.add_argument("--surr_iters", type=int, default=400)
    p.add_argument("--surr_restarts", type=int, default=4)
    p.add_argument("--surr_lr", type=float, default=0.005)
    p.add_argument("--surr_optimizer", default="adam")
    p.add_argument("--surr_cma_budget", type=int, default=300,
                   help="surrogate evaluations for the surrogate-as-search-model CMA ablation")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="results/repair_benchmark")
    p.add_argument("--quick", action="store_true",
                   help="tiny smoke run: 1 problem, budget 60, small samples")
    p.add_argument("--hpo", action="store_true",
                   help="hyperparameter search for surrogate repair, scored by true MC")
    p.add_argument("--hpo_samples", type=int, default=40,
                   help="evaluation budget in --hpo mode (configs for random / asks for nevergrad)")
    p.add_argument("--hpo_method", choices=["nevergrad", "random"], default="nevergrad",
                   help="search algorithm over surrogate hyperparameters")
    p.add_argument("--hpo_arbiter", choices=["mfmc", "hifi"], default="mfmc",
                   help="estimator scoring each converged point during HPO "
                        "(mfmc = cheap deployable verify; hifi = high-accuracy MC)")
    p.add_argument("--objective", choices=["repair", "consistency"], default="repair",
                   help="repair: nearest-consistent (min dist s.t. I<=gamma); "
                        "consistency: minimise I ignoring distance (HPO scores by achieved I)")
    p.add_argument("--add_surr_cma", default=None, metavar="RECORDS.JSON",
                   help="compute ONLY surrogate-CMA over an existing records.json's "
                        "problems and merge in (no full re-run)")
    p.add_argument("--add_hybrid", default=None, metavar="RECORDS.JSON",
                   help="compute ONLY hybrid over an existing records.json's problems "
                        "and merge in (no full re-run); works for repair and min-I)")
    p.add_argument("--add_surrogate", default=None, metavar="RECORDS.JSON",
                   help="recompute ONLY the surrogate column (e.g. with --surr_verify mfmc) "
                        "over an existing records.json and merge in (no full re-run)")
    p.add_argument("--surr_verify", choices=["mfmc", "mc"], default="mfmc",
                   help="estimator verifying each surrogate restart "
                        "(mfmc = cheap deployable, matches the hybrid; mc = full MC)")
    p.add_argument("--band_lo", type=float, default=0.05,
                   help="surrogate-I pre-select band low (gamma+band_lo)")
    p.add_argument("--band_hi", type=float, default=0.18,
                   help="surrogate-I pre-select band high (gamma+band_hi)")
    p.add_argument("--true_lo", type=float, default=None,
                   help="min true I* of generated problems (default gamma); raise for FARTHER starts")
    p.add_argument("--true_hi", type=float, default=0.75,
                   help="max true I* (cap so repairs still exist in-box)")
    p.add_argument("--n_candidates", type=int, default=800,
                   help="random theta candidates screened per scenario when sampling problems")
    p.add_argument("--mfmc_only", action="store_true",
                   help="skip the full-MC derivative-free methods; run only the MFMC variants")
    p.add_argument("--surr_lambda_init", type=float, default=10.0,
                   help="surrogate penalty lambda_init (repair); HPO winner uses 42.5")
    p.add_argument("--surr_lambda_scale", type=float, default=5.0,
                   help="surrogate penalty lambda growth/iter; HPO uses 1.0 (constant)")
    p.add_argument("--surr_verify_all", action="store_true",
                   help="verify EVERY restart and keep best by truth (n_restarts+1 calls); "
                        "default verifies only the single deployed theta' (1 call)")
    args = p.parse_args()
    args.budgets = args.budgets if args.budgets else [args.budget]

    if args.quick:
        args.n_problems = 1
        args.budget = 60
        args.mc_samples = 150
        args.mfmc_cheap = 1500
        args.mfmc_exp = 80
        args.val_samples = 1500
        args.surr_iters = 150
        args.scenarios_2d3d = args.scenarios_2d3d[:1]
        args.scenarios_4d = []

    if args.add_surr_cma:
        run_add_method(args, args.add_surr_cma, "surr_cma")
    elif args.add_hybrid:
        run_add_method(args, args.add_hybrid, "hybrid")
    elif args.add_surrogate:
        run_add_method(args, args.add_surrogate, "surrogate")
    elif args.hpo:
        run_hpo(args)
    else:
        run_benchmark(args)


if __name__ == "__main__":
    main()
