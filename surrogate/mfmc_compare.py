"""
surrogate/mfmc_compare.py
=========================
Head-to-head comparison of the two "MFMC" estimators used in the paper, both
applied to the RQ1 task (estimate the per-configuration inconsistency I(theta)
and agree with a gold Monte-Carlo reference).

Everything is computed *from geometry only* (source/target zonotope pairs stored
per experiment). We do NOT read any precomputed estimate field (I_theta,
mc_probability_*, I_MF_*) from the JSON, so the gold reference is independent and
there is no circularity between "reference" and "estimator".

Per configuration theta we compute:
  gold      : MC with a large sample (N_gold) + exact LP containment   -> ground truth
  mc_mod    : MC with a moderate sample (N_mod)                        -> noisy base estimate
  aabb      : analytic AABB volume-Jaccard inconsistency  1 - vol(B_cap)/vol(B_cup)
  mfmc_perT : RQ4 estimator -- per-theta classic MFMC (repair_benchmark._mfmc_estimate):
              cheap = SAMPLED box-membership on the same draws; mu from a large cheap
              sample at the SAME theta; alpha per-theta. (textbook Peherstorfer MFMC)

Per scenario (across its configs) we then compute:
  mfmc_cv   : RQ1 estimator -- scenario-wide control variate (mfmc_correction.py /
              MATLAB compute_and_save_mfmc):
              I_MF[t] = mc_mod[t] + alpha*(mu_aabb - aabb[t]),
              alpha = Cov(mc_mod, aabb)/Var(aabb), mu_aabb = mean(aabb)
              over the *intersecting* configs of the scenario.

For each estimator we report agreement with gold: Spearman rho, R^2 (1-SSres/SStot
with residual = est-gold), MAE, and the operational FPR/FNR at gamma=0.5
(positive class = "inconsistent", i.e. I > gamma). We also save a 1x4 scatter
grid (estimate vs gold) per estimator, pooled across scenarios.

Run:
    uv run python -m surrogate.mfmc_compare \
        --max_configs 120 --n_gold 4000 --n_mod 256 --n_exp 128 --n_cheap 4000 \
        --out results/mfmc_compare
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from .repair_benchmark import _mc_estimate, _mfmc_estimate
from .generate_synthetic import interval_bounds

# Default 4 scenarios spanning easy->hard + a 3D one, drawn from Table I domains.
DEFAULT_SCENARIOS = [
    ("HVAC",        "data/measurements_cps_v6/results_scenario_49.json"),
    ("Medical",     "data/measurements_cps_v6/results_scenario_57.json"),
    ("Wind",        "data/measurements_cps_v6/results_scenario_36.json"),
    ("Engineering", "data/measurements_v6/results_scenario_7.json"),
]

ESTIMATORS = ["aabb", "mc_mod", "mfmc_perT", "mfmc_cv"]
PRETTY = {
    "aabb":      "AABB (analytic)",
    "mc_mod":    f"MC (moderate)",
    "mfmc_perT": "MFMC per-$\\theta$ (RQ4)",
    "mfmc_cv":   "MFMC scenario-CV (RQ1)",
}


# ─────────────────────────────────────────────────────────────────────────────
# geometry helpers
# ─────────────────────────────────────────────────────────────────────────────
def _aabb_inconsistency(c1, G1, c2, G2):
    """Analytic AABB volume-Jaccard inconsistency 1 - vol(inter)/vol(union).

    Returns (I_aabb, intersecting) where intersecting is True iff the interval
    hulls overlap (vol(inter) > 0) -- the control-variate validity condition.
    """
    lo1, hi1 = interval_bounds(c1, G1)
    lo2, hi2 = interval_bounds(c2, G2)
    inter = np.maximum(0.0, np.minimum(hi1, hi2) - np.maximum(lo1, lo2))
    union = np.maximum(hi1, hi2) - np.minimum(lo1, lo2)
    if np.any(union <= 0):
        return 1.0, False
    vol_i = float(np.prod(inter))
    vol_u = float(np.prod(union))
    j = vol_i / vol_u if vol_u > 0 else 0.0
    return 1.0 - j, (vol_i > 0.0)


def _geom(exp):
    unc = exp["post_state"]["uncertainty"]
    c1 = np.asarray(unc["source_center"], dtype=np.float64)
    G1 = np.asarray(unc["source_generators"], dtype=np.float64)
    c2 = np.asarray(unc["target_center"], dtype=np.float64)
    G2 = np.asarray(unc["target_generators"], dtype=np.float64)
    if G1.ndim == 1:
        G1 = G1[:, None]
    if G2.ndim == 1:
        G2 = G2[:, None]
    return c1, G1, c2, G2


# ─────────────────────────────────────────────────────────────────────────────
# metrics
# ─────────────────────────────────────────────────────────────────────────────
def _pearson(x, y):
    x = np.asarray(x); y = np.asarray(y)
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x, y):
    x = np.asarray(x); y = np.asarray(y)
    if len(x) < 2:
        return float("nan")
    try:
        from scipy.stats import rankdata
        rx, ry = rankdata(x), rankdata(y)
    except Exception:  # average-rank not critical for continuous values
        rx = np.argsort(np.argsort(x)).astype(float)
        ry = np.argsort(np.argsort(y)).astype(float)
    return _pearson(rx, ry)


def _r2(est, gold):
    est = np.asarray(est); gold = np.asarray(gold)
    ss_res = float(np.sum((est - gold) ** 2))
    ss_tot = float(np.sum((gold - gold.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")


def _fpr_fnr(est, gold, gamma):
    """positive = inconsistent (I>gamma). FPR among truly-consistent, FNR among
    truly-inconsistent."""
    est = np.asarray(est); gold = np.asarray(gold)
    pred_pos = est > gamma
    true_pos = gold > gamma
    cons = ~true_pos
    inco = true_pos
    fpr = float(np.mean(pred_pos[cons])) if cons.any() else float("nan")
    fnr = float(np.mean(~pred_pos[inco])) if inco.any() else float("nan")
    return fpr, fnr


def _metrics(est, gold, gamma):
    fpr, fnr = _fpr_fnr(est, gold, gamma)
    return dict(rho=_spearman(est, gold), r2=_r2(est, gold),
                mae=float(np.mean(np.abs(np.asarray(est) - np.asarray(gold)))),
                fpr=fpr, fnr=fnr, n=int(len(gold)))


# ─────────────────────────────────────────────────────────────────────────────
# core per-scenario evaluation
# ─────────────────────────────────────────────────────────────────────────────
def eval_scenario(path, max_configs, n_gold, n_mod, n_cheap, n_exp, seed, label=""):
    data = json.loads(Path(path).read_text())
    exps = data["experiments"]
    n = len(exps)
    idx = np.unique(np.linspace(0, n - 1, min(max_configs, n)).astype(int))

    gold, mc_mod, aabb, mfmc_perT, inter = [], [], [], [], []
    n_lp_total = 0
    for k, i in enumerate(tqdm(idx, desc=f"{label:12s}", unit="cfg", leave=True)):
        try:
            c1, G1, c2, G2 = _geom(exps[i])
        except (KeyError, ValueError, TypeError):
            continue
        rng = np.random.default_rng(seed + 1000 * k)
        g, lp1 = _mc_estimate(c1, G1, c2, G2, n_gold, rng)
        m, lp2 = _mc_estimate(c1, G1, c2, G2, n_mod, rng)
        f, lp3 = _mfmc_estimate(c1, G1, c2, G2, n_cheap, n_exp, rng)
        a, ok = _aabb_inconsistency(c1, G1, c2, G2)
        gold.append(g); mc_mod.append(m); mfmc_perT.append(f)
        aabb.append(a); inter.append(ok)
        n_lp_total += lp1 + lp2 + lp3

    gold = np.array(gold); mc_mod = np.array(mc_mod)
    aabb = np.array(aabb); mfmc_perT = np.array(mfmc_perT)
    inter = np.array(inter, dtype=bool)

    # RQ1 scenario-wide control variate, fit on intersecting configs only.
    cv = np.full_like(gold, np.nan)
    if inter.sum() >= 3 and np.var(aabb[inter]) > 1e-12:
        a_in, m_in = aabb[inter], mc_mod[inter]
        mu_aabb = a_in.mean()
        C = np.cov(m_in, a_in, ddof=1)
        alpha = C[0, 1] / C[1, 1]
        cv[inter] = np.clip(mc_mod[inter] + alpha * (mu_aabb - aabb[inter]), 0.0, 1.0)
    else:
        alpha, mu_aabb = float("nan"), float("nan")

    return dict(
        gold=gold, mc_mod=mc_mod, aabb=aabb, mfmc_perT=mfmc_perT, mfmc_cv=cv,
        inter=inter, alpha=float(alpha), mu_aabb=float(mu_aabb),
        n_lp=int(n_lp_total), n_configs=int(len(gold)),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenarios", nargs="*", default=None,
                   help="explicit JSON paths; default = 4 built-in scenarios")
    p.add_argument("--max_configs", type=int, default=120,
                   help="configs subsampled per scenario")
    p.add_argument("--n_gold", type=int, default=4000, help="gold MC sample size")
    p.add_argument("--n_mod", type=int, default=256, help="moderate MC sample size (CV base)")
    p.add_argument("--n_cheap", type=int, default=4000, help="MFMC cheap (box) sample size")
    p.add_argument("--n_exp", type=int, default=128, help="MFMC expensive (LP) sample size")
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/mfmc_compare")
    a = p.parse_args()

    if a.scenarios:
        scen = [(Path(s).stem.replace("results_scenario_", "sc"), s) for s in a.scenarios]
    else:
        scen = DEFAULT_SCENARIOS

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    per_scenario = {}
    pooled = {e: [] for e in ESTIMATORS}
    pooled["gold"] = []
    pooled["scenario"] = []
    for name, path in scen:
        print(f"[{name:12s}] {path}", flush=True)
        r = eval_scenario(path, a.max_configs, a.n_gold, a.n_mod,
                          a.n_cheap, a.n_exp, a.seed, label=name)
        print(f"    {r['n_configs']} configs, {r['n_lp']:,} LP solves, "
              f"alpha={r['alpha']:+.3f}, mu_aabb={r['mu_aabb']:.3f}, "
              f"intersecting={int(r['inter'].sum())}/{r['n_configs']}")
        # per-scenario metrics (CV only over its valid entries; others over all)
        m = {}
        for e in ESTIMATORS:
            est = r[e]
            mask = ~np.isnan(est)
            m[e] = _metrics(est[mask], r["gold"][mask], a.gamma)
        per_scenario[name] = m
        for e in ESTIMATORS:
            mask = ~np.isnan(r[e])
            pooled[e].append(np.where(mask, r[e], np.nan))
        pooled["gold"].append(r["gold"])
        pooled["scenario"].append(np.full(r["n_configs"], name, dtype=object))
        # print per-scenario row
        print(f"    {'estimator':22s} {'rho':>7} {'R2':>7} {'MAE':>7} {'FPR':>7} {'FNR':>7}")
        for e in ESTIMATORS:
            mm = m[e]
            print(f"    {PRETTY[e]:22s} {mm['rho']:7.3f} {mm['r2']:7.3f} "
                  f"{mm['mae']:7.4f} {mm['fpr']:7.3f} {mm['fnr']:7.3f}")

    # pooled arrays
    gold_all = np.concatenate(pooled["gold"])
    scen_all = np.concatenate(pooled["scenario"])
    est_all = {e: np.concatenate(pooled[e]) for e in ESTIMATORS}

    print("\n================ POOLED (all scenarios) ================")
    print(f"{'estimator':22s} {'rho':>7} {'R2':>7} {'MAE':>7} {'FPR':>7} {'FNR':>7} {'n':>6}")
    pooled_metrics = {}
    for e in ESTIMATORS:
        mask = ~np.isnan(est_all[e])
        mm = _metrics(est_all[e][mask], gold_all[mask], a.gamma)
        pooled_metrics[e] = mm
        print(f"{PRETTY[e]:22s} {mm['rho']:7.3f} {mm['r2']:7.3f} "
              f"{mm['mae']:7.4f} {mm['fpr']:7.3f} {mm['fnr']:7.3f} {mm['n']:6d}")

    # ── scatter grid: estimate vs gold, one panel per estimator ──
    colors = plt.get_cmap("tab10")
    names = [s[0] for s in scen]
    cmap = {nm: colors(i) for i, nm in enumerate(names)}
    fig, axes = plt.subplots(1, len(ESTIMATORS), figsize=(3.2 * len(ESTIMATORS), 3.3),
                             sharex=True, sharey=True)
    for ax, e in zip(axes, ESTIMATORS):
        mask = ~np.isnan(est_all[e])
        for nm in names:
            sel = mask & (scen_all == nm)
            ax.scatter(gold_all[sel], est_all[e][sel], s=8, alpha=0.5,
                       color=cmap[nm], label=nm, edgecolors="none")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, zorder=0)
        ax.axhline(a.gamma, color="0.7", lw=0.6, ls=":")
        ax.axvline(a.gamma, color="0.7", lw=0.6, ls=":")
        mm = pooled_metrics[e]
        ax.set_title(PRETTY[e], fontsize=9)
        ax.text(0.04, 0.96,
                f"$\\rho$={mm['rho']:.3f}\n$R^2$={mm['r2']:.3f}\nMAE={mm['mae']:.3f}\n"
                f"FPR={mm['fpr']:.2f} FNR={mm['fnr']:.2f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=7,
                bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.85))
        ax.set_xlabel("gold MC $I(\\theta)$", fontsize=8)
        ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("estimate", fontsize=8)
    axes[-1].legend(fontsize=6, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fp = out / "mfmc_compare_scatter.pdf"
    fig.savefig(fp, bbox_inches="tight"); fig.savefig(fp.with_suffix(".png"), dpi=150)
    print(f"\nsaved {fp}")

    # ── persist metrics ──
    payload = dict(
        config=dict(max_configs=a.max_configs, n_gold=a.n_gold, n_mod=a.n_mod,
                    n_cheap=a.n_cheap, n_exp=a.n_exp, gamma=a.gamma, seed=a.seed),
        scenarios=[s[1] for s in scen],
        per_scenario=per_scenario,
        pooled=pooled_metrics,
    )
    (out / "mfmc_compare_metrics.json").write_text(json.dumps(payload, indent=2))
    print(f"saved {out / 'mfmc_compare_metrics.json'}")


if __name__ == "__main__":
    main()
