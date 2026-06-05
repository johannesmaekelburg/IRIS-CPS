"""
surrogate/plot_repair.py
========================
Figures for the counterfactual-repair benchmark.

  pareto      : cost-quality frontier from a (swept) records.json
                  success-rate and median repair-distance vs. oracle calls;
                  surrogate / hybrid as points at the far left.
  trajectory  : 2-parameter (scale x center, correlation fixed) illustration on
                  HVAC: the TRUE I(theta) surface + consistency boundary, with the
                  gradient (surrogate), CMA-ES, and finite-diff search paths
                  overlaid. Shows WHY gradient repair is efficient.

Usage:
  uv run python -m surrogate.plot_repair pareto --records results/repair_sweep/records.json
  uv run python -m surrogate.plot_repair trajectory --out results/repair_fig
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

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

from .counterfactual import DifferentiableIntervention, load_scenario_geometry
from .models_v2 import load_checkpoint
from .repair_benchmark import Oracle, hifi_mc, _LB, _UB, _SPAN

_PARAM = {"scale": 0, "center": 1, "corr": 2}
_METHOD_STYLE = {
    "surrogate": ("#2E7D32", "*", "Surrogate (gradient)"),
    "hybrid":    ("#1565C0", "P", "Hybrid (propose+verify)"),
    "surr_cma":  ("#00838F", "D", "Surrogate-CMA (surrogate search)"),
    "cma_mc":    ("#C62828", "o", "CMA-ES / MC"),
    "cma_mfmc":  ("#EF6C00", "s", "CMA-ES / MFMC"),
    "fd_mc":     ("#6A1B9A", "^", "Finite-diff / MC"),
    "fd_mfmc":   ("#AD1457", "v", "Finite-diff / MFMC"),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Figure B — cost-quality Pareto from records.json
# ═══════════════════════════════════════════════════════════════════════════════

def _skip(method, mfmc_only, drop=()):
    """Drop full-MC derivative-free variants (mfmc_only) and any explicitly dropped."""
    return (mfmc_only and method.endswith("_mc")) or method in drop


def plot_pareto(records_path, out_dir, xaxis="calls", mfmc_only=False, drop=()):
    recs = json.loads(Path(records_path).read_text())
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    xkey = "wall" if xaxis == "wall" else "oracle_calls"
    xlbl = ("wall-time per repair [s] (median)" if xaxis == "wall"
            else "expensive oracle calls (median)")
    xfloor = 1e-2 if xaxis == "wall" else 1

    # group by (method, budget)
    grp = defaultdict(list)
    for r in recs:
        grp[(r["method"], r["budget"])].append(r)

    def agg(rs):
        succ = [x for x in rs if x["success"]]
        return (float(np.median([x[xkey] for x in rs])),
                len(succ) / len(rs),
                float(np.median([x["dist"] for x in succ])) if succ else np.nan)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for method in _METHOD_STYLE:
        if _skip(method, mfmc_only, drop):
            continue
        pts = sorted([(b, *agg(rs)) for (m, b), rs in grp.items() if m == method],
                     key=lambda t: (t[1]))
        if not pts:
            continue
        color, marker, label = _METHOD_STYLE[method]
        eff = [max(p[1], xfloor) for p in pts]
        succ = [p[2] for p in pts]
        dist = [p[3] for p in pts]
        dfo = method.startswith(("cma", "fd"))
        kw = dict(color=color, marker=marker, label=label,
                  ms=10, lw=1.6 if dfo else 0, ls="-" if dfo else "")
        ax1.plot(eff, succ, **kw)
        ax2.plot(eff, dist, **kw)

    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_xlabel(xlbl)
        ax.grid(alpha=0.3, which="both")
    ax1.set_ylabel("success rate")
    ax1.set_ylim(-0.05, 1.05)
    ax2.set_ylabel("repair distance $\\|\\theta'-\\theta^*\\|$ (median)")
    ax1.legend(fontsize=7, loc="lower right")
    fig.suptitle("Counterfactual repair: cost vs. quality")
    fig.tight_layout()
    p = out / ("repair_pareto_mfmconly.pdf" if mfmc_only else "repair_pareto.pdf")
    fig.savefig(p, bbox_inches="tight"); fig.savefig(p.with_suffix(".png"), dpi=150)
    print(f"saved {p}")


def plot_pareto_combined(records_path, out_dir, xaxis="wall", style="bubble",
                         yaxis="success", mfmc_only=False, drop=()):
    """Single-panel repair cost-quality merging the (quality | distance) panels.

    x = effort (oracle calls or wall-time).
    y = success rate (yaxis="success", higher=better) OR median true I reached
        (yaxis="inconsistency", lower=better).
    The secondary metric, repair distance, is encoded by `style`:
      bubble : marker size (bigger = tighter repair)
      color  : marker colour via a colorbar (resolves small differences best)
      dual   : a second y-axis (distance, dashed) — true linear distance scale
    """
    from matplotlib.lines import Line2D

    recs = json.loads(Path(records_path).read_text())
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    xkey = "wall" if xaxis == "wall" else "oracle_calls"
    xlbl = ("wall-time per repair [s] (median)" if xaxis == "wall"
            else "expensive oracle calls (median)")
    xfloor = 1e-2 if xaxis == "wall" else 1
    incons = yaxis == "inconsistency"
    ylbl = ("median true $I(\\theta')$ reached" if incons
            else "success rate")
    yname = "median $I$" if incons else "success"

    grp = defaultdict(list)
    for r in recs:
        grp[(r["method"], r["budget"])].append(r)

    def agg(rs):
        succ = [x for x in rs if x["success"]]
        y = (float(np.median([x["i_true_prime"] for x in rs])) if incons
             else len(succ) / len(rs))
        dist = float(np.median([x["dist"] for x in succ])) if succ else np.nan
        return (float(np.median([x[xkey] for x in rs])), y, dist)

    # per-method budget-sorted aggregated points: (budget, eff, y, dist)
    series = {}
    all_d = []
    for method in _METHOD_STYLE:
        if _skip(method, mfmc_only, drop):
            continue
        pts = sorted([(b, *agg(rs)) for (m, b), rs in grp.items() if m == method],
                     key=lambda t: t[1])
        if pts:
            series[method] = pts
            all_d += [p[3] for p in pts if np.isfinite(p[3])]

    # ICDM (IEEE single-column): flat aspect + ~scriptsize fonts for the final figures.
    icdm = style in ("dual", "single")
    lblfs, tickfs, legfs = (8.0, 7.0, 6.0) if icdm else (11, 10, 7)
    figsize = (3.5, 2.0) if icdm else (7.6, 5.2)   # 3.5in = IEEE single-column width
    fig, ax = plt.subplots(figsize=figsize)
    if icdm and incons:
        ylbl = "median $I(\\theta')$"   # compact for the flat panel
    ax.set_xscale("log")
    ax.set_xlabel(xlbl, fontsize=lblfs)
    ax.set_ylabel(ylbl, fontsize=lblfs)
    ax.tick_params(labelsize=tickfs)
    if not incons:
        ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3, which="both")
    method_handles = [Line2D([], [], marker=_METHOD_STYLE[m][1], color="0.4", ls="",
                             ms=8, mec="k", label=_METHOD_STYLE[m][2]) for m in series]
    # short method labels + handles for the compact ICDM legends
    _short = {"surrogate": "Surrogate", "hybrid": "Hybrid", "cma_mfmc": "CMA-ES",
              "fd_mfmc": "Finite-diff", "cma_mc": "CMA-ES (MC)",
              "fd_mc": "Finite-diff (MC)"}
    m_handles = [Line2D([], [], marker=_METHOD_STYLE[m][1], color=_METHOD_STYLE[m][0],
                        ls="", ms=6, mec="k", mew=0.4,
                        label=_short.get(m, _METHOD_STYLE[m][2]))
                 for m in series]

    if style == "color":
        norm = plt.Normalize(vmin=min(all_d), vmax=max(all_d))
        cmap = plt.get_cmap("RdYlGn_r")
        for method, pts in series.items():
            _, marker, _ = _METHOD_STYLE[method]
            eff = [max(p[1], xfloor) for p in pts]; yv = [p[2] for p in pts]; dd = [p[3] for p in pts]
            if method.startswith(("cma", "fd")):
                ax.plot(eff, yv, "-", c="0.7", lw=1.0, zorder=1)
            ax.scatter(eff, yv, c=dd, cmap=cmap, norm=norm, marker=marker, s=130,
                       edgecolors="k", linewidths=0.6, zorder=3)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
        fig.colorbar(sm, ax=ax, label="repair distance $\\|\\theta'-\\theta^*\\|$")
        ax.legend(handles=method_handles, fontsize=7, loc="best", title="method",
                  title_fontsize=7, framealpha=0.9)
        ax.set_title(f"Counterfactual repair: cost vs. {yname}\n(colour = repair distance)")

    elif style == "dual":
        ax2 = ax.twinx()
        ax2.set_ylabel("repair distance $\\|\\theta'-\\theta^*\\|$", fontsize=lblfs)
        ax2.tick_params(labelsize=tickfs)
        for method, pts in series.items():
            color, marker, _ = _METHOD_STYLE[method]
            eff = [max(p[1], xfloor) for p in pts]; yv = [p[2] for p in pts]; dd = [p[3] for p in pts]
            dfo = method.startswith(("cma", "fd"))
            ax.plot(eff, yv, color=color, marker=marker, ms=7, mec="k", mew=0.4,
                    lw=1.4 if dfo else 0, ls="-" if dfo else "", zorder=3)
            ax2.plot(eff, dd, color=color, marker=marker, ms=5.5, mfc="none",
                     lw=1.2 if dfo else 0, ls="--" if dfo else "", alpha=0.75, zorder=2)
        # marker glyphs in the encoding legend: filled = primary metric, hollow = distance
        prim_label = "median $I$" if incons else "success rate"
        style_handles = [
            Line2D([], [], color="k", ls="-", marker="o", mfc="k", mec="k", ms=6,
                   label=prim_label),
            Line2D([], [], color="k", ls="--", marker="o", mfc="none", mec="k", ms=6,
                   label="repair distance")]
        leg1 = ax.legend(handles=m_handles, fontsize=legfs, loc="lower right",
                         bbox_to_anchor=(1.0, 0.3), ncol=1, handletextpad=0.3,
                         labelspacing=0.3, borderpad=0.3, framealpha=0.9)
        ax.add_artist(leg1)
        ax.legend(handles=style_handles, fontsize=legfs, loc="center",
                  bbox_to_anchor=(0.30, 0.55), handletextpad=0.3, borderpad=0.3,
                  framealpha=0.9)

    elif style == "single":   # primary metric vs effort only (no distance / 2nd axis)
        for method, pts in series.items():
            color, marker, _ = _METHOD_STYLE[method]
            eff = [max(p[1], xfloor) for p in pts]; yv = [p[2] for p in pts]
            dfo = method.startswith(("cma", "fd"))
            ax.plot(eff, yv, color=color, marker=marker, ms=7, mec="k", mew=0.4,
                    lw=1.4 if dfo else 0, ls="-" if dfo else "", zorder=3)
        ax.legend(handles=m_handles, fontsize=legfs, loc="upper right",
                  ncol=1, handletextpad=0.3, labelspacing=0.3, borderpad=0.3,
                  framealpha=0.9)

    else:   # bubble
        def _size(d):
            return 35.0 if not np.isfinite(d) else 30.0 + 130.0 * (0.2 / max(d, 0.05))
        for method, pts in series.items():
            color, marker, label = _METHOD_STYLE[method]
            eff = [max(p[1], xfloor) for p in pts]; yv = [p[2] for p in pts]; dd = [p[3] for p in pts]
            if method.startswith(("cma", "fd")):
                ax.plot(eff, yv, "-", c=color, lw=1.3, alpha=0.6, zorder=2)
            ax.scatter(eff, yv, s=[_size(d) for d in dd], c=color, marker=marker,
                       edgecolors="k", linewidths=0.5, zorder=3, label=label)
        leg1 = ax.legend(fontsize=7, loc="best", title="method", title_fontsize=7)
        ax.add_artist(leg1)
        size_handles = [ax.scatter([], [], s=_size(d), c="lightgray", edgecolors="k",
                                   linewidths=0.5, label=f"$\\|\\theta'-\\theta^*\\|$ = {d:.2f}")
                        for d in (0.15, 0.30, 0.50)]
        ax.legend(handles=size_handles, fontsize=7, loc="center right",
                  title="repair distance\n(bubble size)", title_fontsize=7,
                  labelspacing=1.6, borderpad=1.0, framealpha=0.9)
        ax.set_title(f"Counterfactual repair: cost vs. {yname}\n(bubble size = repair tightness)")

    fig.tight_layout()
    sfx = "_mfmconly" if mfmc_only else ""
    p = out / f"repair_pareto_{style}_{yaxis}_{xaxis}{sfx}.pdf"
    fig.savefig(p, bbox_inches="tight"); fig.savefig(p.with_suffix(".png"), dpi=150)
    print(f"saved {p}")


def plot_pareto_consistency(records_path, out_dir, mfmc_only=False, drop=()):
    """Min-I Pareto: cost (oracle calls AND wall-time) vs. median true I reached.

    Lower-left is better (cheaper AND more consistent). The surrogate sits at the
    far left (a few verify calls) while the derivative-free methods need many more
    oracle evaluations to reach comparably low I.
    """
    recs = json.loads(Path(records_path).read_text())
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    grp = defaultdict(list)
    for r in recs:
        grp[(r["method"], r["budget"])].append(r)

    def agg(rs):
        return (float(np.median([x["oracle_calls"] for x in rs])),
                float(np.median([x["wall"] for x in rs])),
                float(np.median([x["i_true_prime"] for x in rs])))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for method in _METHOD_STYLE:
        if _skip(method, mfmc_only, drop):
            continue
        pts = [(b, *agg(rs)) for (m, b), rs in grp.items() if m == method]
        if not pts:
            continue
        pts.sort(key=lambda t: t[1])      # by calls
        color, marker, label = _METHOD_STYLE[method]
        calls = [max(p[1], 1) for p in pts]
        walls = [p[2] for p in pts]
        I_avg = [p[3] for p in pts]
        dfo = method.startswith(("cma", "fd"))
        kw = dict(color=color, marker=marker, label=label,
                  ms=10, lw=1.6 if dfo else 0, ls="-" if dfo else "")
        ax1.plot(calls, I_avg, **kw)
        ax2.plot([max(w, 1e-3) for w in walls], I_avg, **kw)

    ax1.set_xlabel("expensive oracle calls (median)")
    ax2.set_xlabel("wall-time per repair [s] (median)")
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_ylabel("median true $I(\\theta')$ reached")
        ax.grid(alpha=0.3, which="both")
    ax1.legend(fontsize=7, loc="upper right")
    fig.suptitle("Min-inconsistency search: cost vs. achieved inconsistency")
    fig.tight_layout()
    p = out / ("repair_pareto_consistency_mfmconly.pdf" if mfmc_only
               else "repair_pareto_consistency.pdf")
    fig.savefig(p, bbox_inches="tight"); fig.savefig(p.with_suffix(".png"), dpi=150)
    print(f"saved {p}")


def plot_distance_curve(records_path, out_dir, bins=(0.50, 0.65, 0.80, 0.93),
                        mfmc_only=True, drop=()):
    """Distance-robustness: repair success vs. how far the start I* is above gamma.

    x = start inconsistency I*(theta*) binned; y = success rate. One line per
    method. Tests whether the surrogate/hybrid degrade as the repair gets harder
    (start deeper in the infeasible region). The expensive-oracle cost per bin is
    printed (it is flat: surrogate/hybrid stay ~1 call, DFO stays at its budget).
    """
    from matplotlib.lines import Line2D

    recs = json.loads(Path(records_path).read_text())
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    edges = list(bins)
    centers = [0.5 * (edges[i] + edges[i + 1]) for i in range(len(edges) - 1)]

    lblfs, tickfs, legfs = 8.0, 7.0, 6.0
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    _short = {"surrogate": "Surrogate", "hybrid": "Hybrid", "cma_mfmc": "CMA-ES",
              "fd_mfmc": "Finite-diff", "surr_cma": "Surrogate-CMA"}
    print(f"\ndistance-curve  (records={records_path})")
    for method in _METHOD_STYLE:
        if _skip(method, mfmc_only, drop):
            continue
        ms = [r for r in recs if r["method"] == method]
        if not ms:
            continue
        color, marker, _ = _METHOD_STYLE[method]
        ys, calls = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sub = [r for r in ms if lo <= r["i_true_star"] < hi]
            ys.append(100 * np.mean([r["success"] for r in sub]) if sub else np.nan)
            calls.append(int(np.median([r["oracle_calls"] for r in sub])) if sub else -1)
        ax.plot(centers, ys, color=color, marker=marker, ms=6, mec="k",
                mew=0.5, lw=1.4, label=_short.get(method, method))
        print(f"  {method:>10}:  succ% {['%3.0f' % v for v in ys]}   "
              f"calls(med) {calls}")
    ns = [sum(lo <= r["i_true_star"] < hi for r in recs if r["method"] == "surrogate")
          for lo, hi in zip(edges[:-1], edges[1:])]
    print(f"  n problems per bin: {ns}")

    ax.set_xlabel("start inconsistency $I(\\theta^*)$", fontsize=lblfs)
    ax.set_ylabel("repair success rate [%]", fontsize=lblfs)
    ax.set_ylim(-3, 103)
    ax.tick_params(labelsize=tickfs)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=legfs, loc="lower left", ncol=1, framealpha=0.9)
    fig.tight_layout()
    p = out / "repair_distance_curve.pdf"
    fig.savefig(p, bbox_inches="tight"); fig.savefig(p.with_suffix(".png"), dpi=150)
    print(f"saved {p}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure A — 2D trajectory over the true cost surface
# ═══════════════════════════════════════════════════════════════════════════════

def _full_theta(xy, corr):
    return np.array([xy[0], xy[1], corr], dtype=np.float64)


def _make_opt(params, name, lr):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    if name in ("momentum", "sgd_momentum"):
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if name in ("nesterov",):
        return torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr)
    raise ValueError(f"unknown optimizer {name}")


def _surrogate_path(model, geom, theta_star, gamma, margin, corr, device,
                    lr=0.03, iters=120, lam=50.0, optimizer="adam",
                    objective="repair", start_xy=None, stop_below=None):
    """Surrogate gradient path over (scale, center) with corr fixed.

    objective="repair"      : dist^2 + lam*relu(I-(gamma-margin))^2  (nearest consistent)
    objective="consistency" : minimise I directly, ignoring distance (most consistent)
    start_xy : optional (scale, center) start for this restart; distance is still
               measured back to theta_star. None -> start at theta_star.
    """
    interv = DifferentiableIntervention(
        geom["pre_c"], geom["pre_G"], geom["tgt_c"], geom["tgt_G"],
        geom.get("upr_scales"), geom.get("upr_offsets"), geom.get("upr_onehot"),
        device, use_v2=True).to(device)
    ts = torch.tensor(theta_star, dtype=torch.float32, device=device)
    span = torch.tensor(_SPAN, dtype=torch.float32, device=device)
    lb = torch.tensor(_LB, dtype=torch.float32, device=device)
    ub = torch.tensor(_UB, dtype=torch.float32, device=device)
    th = ts.clone().detach()
    if start_xy is not None:
        th[0], th[1] = float(start_xy[0]), float(start_xy[1])
    th[2] = corr
    th = th.requires_grad_(True)
    opt = _make_opt([th], optimizer, lr)
    path = [th[:2].detach().cpu().numpy().copy()]
    for _ in range(iters):
        opt.zero_grad()
        pd, m, gf = interv(th.unsqueeze(0))
        I = model(pd, m, gf)[0]
        if stop_below is not None and float(I.detach()) < stop_below:
            break                                  # surrogate predicts feasible -> stop
        if objective == "consistency":
            loss = I
        else:
            dist2 = (((th - ts) / span)[:2] ** 2).sum()
            loss = dist2 + lam * torch.relu(I - (gamma - margin)) ** 2
        loss.backward()
        with torch.no_grad():
            th.grad[2] = 0.0                       # freeze correlation
        opt.step()
        with torch.no_grad():
            th.clamp_(lb, ub); th[2] = corr
        path.append(th[:2].detach().cpu().numpy().copy())
    return np.array(path)


def _surrogate_restart_paths(model, geom, theta_star, gamma, margin, corr, device,
                             n_restarts, objective, val_samples=3000, seed=0, **kw):
    """Run (n_restarts+1) gradient descents: restart 0 from theta_star, the rest
    from uniformly random box points (corr fixed). Each endpoint is verified by true
    MC and the best is selected by ground truth (lowest true I for consistency;
    nearest truly-feasible, else lowest true I, for repair).

    Returns (paths, best_idx, end_I_true) where paths is a list of (T,2) arrays.
    """
    rng = np.random.default_rng(seed)
    starts = [None] + [(_LB[:2] + rng.random(2) * _SPAN[:2]) for _ in range(n_restarts)]
    paths, end_I = [], []
    for s in starts:
        p = _surrogate_path(model, geom, theta_star, gamma, margin, corr, device,
                            objective=objective, start_xy=s, **kw)
        paths.append(p)
        end_I.append(hifi_mc(geom, _full_theta(p[-1], corr), val_samples, seed=999))
    end_I = np.array(end_I)
    if objective == "consistency":
        best = int(np.argmin(end_I))
    else:
        feas = end_I <= gamma
        if feas.any():
            d = np.array([np.sum(((p[-1] - theta_star[:2]) / _SPAN[:2]) ** 2) for p in paths])
            d[~feas] = np.inf
            best = int(np.argmin(d))
        else:
            best = int(np.argmin(end_I))
    return paths, best, end_I


def _logged_oracle(geom, corr, n_samples, seed):
    """MC oracle over 2 params (corr fixed) that logs every evaluated point."""
    orc = Oracle(geom, "mc", seed=seed, n_samples=n_samples)
    log = []

    def f(xy):
        log.append(np.asarray(xy, float).copy())
        return orc(_full_theta(xy, corr))
    return f, log, orc


def _cma_path(geom, theta_star, gamma, margin, corr, device, budget, n_samples,
              lam=50.0, objective="repair"):
    """Returns (incumbent_trajectory, total_calls, calls_to_first_feasible, eval_cloud).

    repair      : incumbent = closest FEASIBLE point so far (objective = dist + penalty).
    consistency : incumbent = lowest-I point so far          (objective = I).
    """
    import nevergrad as ng
    cons = objective == "consistency"
    f, log, orc = _logged_oracle(geom, corr, n_samples, seed=1)
    ts = theta_star[:2]
    lb2, ub2 = _LB[:2], _UB[:2]
    param = ng.p.Array(init=ts.copy()).set_bounds(lb2, ub2)
    param.random_state.seed(1)
    optim = ng.optimizers.CMA(parametrization=param, budget=budget)
    best_m = np.inf   # best metric so far (distance if repair, I if consistency)
    inc, first_feas = [ts.copy()], None
    for _ in range(budget):
        cand = optim.ask()
        xy = np.asarray(cand.value, dtype=np.float64)
        I = f(xy)
        if cons:
            val = I
        else:
            val = float(np.sum(((xy - ts) / _SPAN[:2]) ** 2)) + lam * max(0.0, I - (gamma - margin)) ** 2
        optim.tell(cand, val)
        if I <= gamma and first_feas is None:
            first_feas = orc.n_calls
        # advance the displayed incumbent when it improves the relevant metric
        if cons:
            if I < best_m:
                best_m = I; inc.append(xy.copy())
        elif I <= gamma:
            d = float(np.sum(((xy - ts) / _SPAN[:2]) ** 2))
            if d < best_m:
                best_m = d; inc.append(xy.copy())
    return np.array(inc), orc.n_calls, first_feas, np.array(log)


def _fd_path(geom, theta_star, gamma, margin, corr, device, budget, n_samples,
             lr=0.05, h=0.05, lam=50.0, objective="repair", start=None, seed=2):
    """Returns (incumbent_trajectory, total_calls, calls_to_first_feasible, eval_cloud).

    start : optional (scale, center) warm-start (the hybrid begins the oracle
            finite-diff refine from the surrogate's endpoint). Distance is still
            measured back to theta_star. None -> start at theta_star.
    """
    cons = objective == "consistency"
    f, log, orc = _logged_oracle(geom, corr, n_samples, seed=seed)
    ts = theta_star[:2]
    lb2, ub2, span2 = _LB[:2], _UB[:2], _SPAN[:2]
    xy = ts.copy() if start is None else np.asarray(start, dtype=np.float64).copy()
    best_obj, best_x = np.inf, xy.copy()
    inc, first_feas = [xy.copy()], None

    def obj_and_I(p):
        I = f(p)
        if cons:
            return I, I
        return float(np.sum(((p - ts) / span2) ** 2)) + lam * max(0.0, I - (gamma - margin)) ** 2, I
    while orc.n_calls < budget:
        o, I = obj_and_I(xy)
        if first_feas is None and I <= gamma:
            first_feas = orc.n_calls
        if o < best_obj:
            best_obj, best_x = o, xy.copy()
            inc.append(best_x.copy())
        g = np.zeros(2)
        for j in range(2):
            pp = xy.copy(); pp[j] = np.clip(pp[j] + h * span2[j], lb2[j], ub2[j])
            pm = xy.copy(); pm[j] = np.clip(pm[j] - h * span2[j], lb2[j], ub2[j])
            (op, _), (om, _) = obj_and_I(pp), obj_and_I(pm)
            g[j] = (op - om) / (2 * h * span2[j] + 1e-12)
            if orc.n_calls >= budget:
                break
        xy = np.clip(xy - lr * g * span2, lb2, ub2)
    return np.array(inc), orc.n_calls, first_feas, np.array(log)


def _pick_theta_star(geom, model, dev, gamma, seed):
    """Pick a moderately-inconsistent theta* (corr is then held fixed)."""
    rng = np.random.default_rng(seed)
    interv = DifferentiableIntervention(
        geom["pre_c"], geom["pre_G"], geom["tgt_c"], geom["tgt_G"],
        geom.get("upr_scales"), geom.get("upr_offsets"), geom.get("upr_onehot"),
        dev, use_v2=True).to(dev)
    cand = _LB + rng.random((600, 3)) * _SPAN
    with torch.no_grad():
        pd, m, gf = interv(torch.tensor(cand, dtype=torch.float32, device=dev))
        I_s = model(pd, m, gf).cpu().numpy()
    band = np.where((I_s > gamma + 0.05) & (I_s < gamma + 0.18))[0]
    for idx in band:
        if gamma < hifi_mc(geom, cand[idx], 2000, seed=7) < 0.75:
            return cand[idx].astype(np.float64)
    return cand[int(np.argmax(I_s))].astype(np.float64)


def _load_or_make_surface(geom, model, out, gamma, grid, surf_samples, dev, seed,
                          tag="cache"):
    """Load the cached true-I surface + theta* if present, else compute and cache.

    Cache is keyed per-scenario by `tag` so trajectory and gallery share the SAME
    surface and theta* per scenario (direct comparability) without collisions across
    scenarios.
    """
    out.mkdir(parents=True, exist_ok=True)
    cache = out / f"surface_{tag}.npz"
    if cache.exists():
        z = np.load(cache)
        print(f"loaded cached surface ({cache})")
        return z["s_ax"], z["c_ax"], z["surf"], z["theta_star"].astype(np.float64)
    theta_star = _pick_theta_star(geom, model, dev, gamma, seed)
    corr = float(theta_star[2])
    print(f"theta* = {np.round(theta_star,3)}  (corr fixed at {corr:.3f})")
    s_ax = np.linspace(_LB[0], _UB[0], grid)
    c_ax = np.linspace(_LB[1], _UB[1], grid)
    surf = np.zeros((grid, grid))
    for i, cc in enumerate(c_ax):
        for j, ss in enumerate(s_ax):
            surf[i, j] = hifi_mc(geom, np.array([ss, cc, corr]), surf_samples, seed=11)
        print(f"  surface row {i+1}/{grid}", end="\r")
    print()
    np.savez(cache, s_ax=s_ax, c_ax=c_ax, surf=surf, theta_star=theta_star)
    return s_ax, c_ax, surf, theta_star


def plot_trajectory(scenario, model_path, out_dir, gamma=0.5, margin=0.1,
                    grid=32, surf_samples=600, dfo_budget=120, mc_samples=300,
                    device="cpu", seed=0, surr_lr=0.005, surr_optimizer="adam",
                    surr_iters=120, surr_lam=50.0, refine_budget=42,
                    surr_stop_below=None, start_scale=None, start_center=None,
                    surr_restarts=0, title=None,
                    show_consistency=False, objective="repair"):
    cons = objective == "consistency"
    dev = torch.device(device)
    geom = load_scenario_geometry(Path(scenario))
    model = load_checkpoint(model_path, map_location=device)
    tag = Path(scenario).stem

    out = Path(out_dir)
    s_ax, c_ax, surf, theta_star = _load_or_make_surface(
        geom, model, out, gamma, grid, surf_samples, dev, seed, tag=tag)
    # optional manual start override (corr stays fixed = the surface's corr)
    if start_scale is not None or start_center is not None:
        theta_star = theta_star.copy()
        if start_scale is not None:
            theta_star[0] = start_scale
        if start_center is not None:
            theta_star[1] = start_center
        print(f"start override: theta* = {np.round(theta_star, 3)}")
    corr = float(theta_star[2])

    # ── method trajectories (best-so-far incumbent paths) ──
    # Single surrogate gradient run from theta* (no restarts), matching the benchmark's
    # "pure surrogate" (propose one theta', verify it).
    sur = _surrogate_path(model, geom, theta_star, gamma, margin, corr, dev,
                          lr=surr_lr, iters=surr_iters, lam=surr_lam,
                          optimizer=surr_optimizer, objective=objective,
                          stop_below=surr_stop_below)
    sur_endI = hifi_mc(geom, _full_theta(sur[-1], corr), 3000, seed=999)
    if surr_stop_below is not None:
        print(f"surrogate early-stop @ i_hat<{surr_stop_below}: "
              f"{len(sur)-1} grad steps, endpoint true I={sur_endI:.3f}")
    # Hybrid: warm-start an oracle finite-diff refine from the surrogate's endpoint.
    # (repair: refine when the proposal is infeasible; consistency: always refine.)
    do_refine = cons or (sur_endI > gamma)
    if do_refine:
        hyb, hyb_calls, hyb_feas, _ = _fd_path(
            geom, theta_star, gamma, margin, corr, dev, refine_budget, mc_samples,
            lam=surr_lam, objective=objective, start=sur[-1], seed=3)
    else:
        hyb, hyb_calls, hyb_feas = None, 0, 0
    cma, cma_calls, cma_feas, cma_cloud = _cma_path(
        geom, theta_star, gamma, margin, corr, dev, dfo_budget, mc_samples,
        objective=objective)
    fd, fd_calls, fd_feas, fd_cloud = _fd_path(
        geom, theta_star, gamma, margin, corr, dev, dfo_budget, mc_samples,
        objective=objective)
    print(f"calls-to-first-feasible:  CMA={cma_feas}  finite-diff(cold)={fd_feas}  "
          f"surrogate=1  hybrid-refine={hyb_calls}")

    # ── plot ── (ICDM single-column: fonts match the pareto figures, no title)
    lblfs, tickfs, legfs = 8.0, 7.0, 6.0
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    im = ax.pcolormesh(s_ax, c_ax, surf, shading="auto", cmap="RdYlBu_r",
                       vmin=0, vmax=1)
    cs = ax.contour(s_ax, c_ax, surf, levels=[gamma], colors="k",
                    linewidths=1.4, linestyles="--")
    ax.clabel(cs, fmt={gamma: f"I={gamma:g}"}, fontsize=tickfs)
    cb = fig.colorbar(im, ax=ax, label="true $I(\\theta)$")
    cb.set_label("true $I(\\theta)$", fontsize=lblfs)
    cb.ax.tick_params(labelsize=tickfs)

    def _feas(n):
        return "n/a" if n is None else str(n)

    cc, _, _ = _METHOD_STYLE["cma_mc"]
    fc, _, _ = _METHOD_STYLE["fd_mc"]
    gc, _, _ = _METHOD_STYLE["surrogate"]
    hc, _, _ = _METHOD_STYLE["hybrid"]
    # Faint CMA eval cloud (context: it must sample widely), subordinate to the lines
    ax.scatter(cma_cloud[:, 0], cma_cloud[:, 1], s=4, c=cc, alpha=0.12, zorder=1)
    # Derivative-free incumbents (cold-start from theta*)
    ax.plot(cma[:, 0], cma[:, 1], "-o", ms=2.5, c=cc, lw=1.3, alpha=0.95, zorder=3,
            label=f"CMA-ES ({_feas(cma_feas)} calls)")
    ax.plot(fd[:, 0], fd[:, 1], "-o", ms=2.5, c=fc, lw=1.3, alpha=0.95, zorder=3,
            label=f"Finite-diff, cold ({_feas(fd_feas)})")
    # Surrogate gradient (single run, ~free) — the warm start for the hybrid
    obj_txt = "min $I$" if cons else "nearest cons."
    ax.plot(sur[:, 0], sur[:, 1], "-", c=gc, lw=2.0, zorder=4,
            label=f"Surrogate — {obj_txt} (1 call)")
    endpoints = [(cma, cc), (fd, fc), (sur, gc)]
    # Hybrid refine segment: oracle finite-diff from the surrogate endpoint
    if hyb is not None:
        ax.plot(hyb[:, 0], hyb[:, 1], "-o", ms=2.5, c=hc, lw=1.7, zorder=5,
                label=f"Hybrid refine (+{hyb_calls})")
        # mark the hand-off (surrogate endpoint = refine start)
        ax.scatter([sur[-1, 0]], [sur[-1, 1]], marker="o", s=24, c=gc,
                   edgecolors="k", linewidths=0.7, zorder=6)
        endpoints.append((hyb, hc))
    # Endpoints
    for traj, col in endpoints:
        ax.scatter([traj[-1, 0]], [traj[-1, 1]], marker="X", s=50, c=col,
                   edgecolors="k", linewidths=0.8, zorder=6)
    ax.scatter([theta_star[0]], [theta_star[1]], marker="*", s=150, c="white",
               edgecolors="k", linewidths=1.1, label="$\\theta^*$", zorder=7)

    ax.set_xlabel("scale factor $s_u$", fontsize=lblfs)
    ax.set_ylabel("center shift $\\Delta c_u$", fontsize=lblfs)
    ax.tick_params(labelsize=tickfs)
    if title:
        ax.set_title(title, fontsize=lblfs)
    ax.legend(fontsize=legfs, loc="best", framealpha=0.9)
    fig.tight_layout()
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    suffix = "_consistency" if cons else ""
    p = out / f"repair_trajectory_{tag}{suffix}.pdf"
    fig.savefig(p, bbox_inches="tight"); fig.savefig(p.with_suffix(".png"), dpi=150)
    print(f"saved {p}")


# ═══════════════════════════════════════════════════════════════════════════════
# Gallery — many surrogate trajectories over the surface for different (opt, lr, λ)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_gallery(scenario, model_path, out_dir, gamma=0.5, margin=0.1,
                 grid=32, surf_samples=600, iters=120, device="cpu", seed=0,
                 optimizers=("adam", "momentum", "nesterov", "rmsprop"),
                 lrs=(0.003, 0.01, 0.03, 0.1), lams=(10.0, 30.0, 100.0, 300.0),
                 ncols=None, objective="repair"):
    """Grid of surrogate repair trajectories over the true-I surface, one panel per
    (optimizer, lr, lambda) config. Diagnoses which optimizer/balance gives the
    straightest, shortest path. Reuses the cached trajectory surface + theta*.

    Each panel is annotated with the final true-MC I(theta'), the box-normalised
    repair distance, and the path length (a proxy for straightness).
    """
    dev = torch.device(device)
    geom = load_scenario_geometry(Path(scenario))
    model = load_checkpoint(model_path, map_location=device)
    out = Path(out_dir)
    tag = Path(scenario).stem
    s_ax, c_ax, surf, theta_star = _load_or_make_surface(
        geom, model, out, gamma, grid, surf_samples, dev, seed, tag=tag)
    corr = float(theta_star[2])

    configs = [(o, lr, lam) for lam in lams for o in optimizers for lr in lrs]
    n = len(configs)
    ncols = ncols or len(lrs)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.1 * nrows),
                             squeeze=False)
    sp2 = _SPAN[:2]

    for k, (opt_name, lr, lam) in enumerate(configs):
        ax = axes[k // ncols][k % ncols]
        path = _surrogate_path(model, geom, theta_star, gamma, margin, corr, dev,
                               lr=lr, iters=iters, lam=lam, optimizer=opt_name,
                               objective=objective)
        end = _full_theta(path[-1], corr)
        i_true = hifi_mc(geom, end, 3000, seed=999)
        dist = float(np.sqrt(np.sum(((path[-1] - theta_star[:2]) / sp2) ** 2)))
        plen = float(np.sum(np.sqrt(np.sum((np.diff(path, axis=0) / sp2) ** 2, axis=1))))

        ax.pcolormesh(s_ax, c_ax, surf, shading="auto", cmap="RdBu_r", vmin=0, vmax=1)
        ax.contour(s_ax, c_ax, surf, levels=[gamma], colors="k",
                   linewidths=1.4, linestyles="--")
        ok = i_true <= gamma
        ax.plot(path[:, 0], path[:, 1], "-", c="#2E7D32" if ok else "#B71C1C",
                lw=2.2, zorder=4)
        ax.scatter([path[-1, 0]], [path[-1, 1]], marker="X", s=70,
                   c="#2E7D32" if ok else "#B71C1C", edgecolors="k",
                   linewidths=0.8, zorder=6)
        ax.scatter([theta_star[0]], [theta_star[1]], marker="*", s=150, c="white",
                   edgecolors="k", linewidths=1.0, zorder=7)
        ax.set_title(f"{opt_name} lr={lr:g} λ={lam:g}", fontsize=8)
        ax.text(0.03, 0.03,
                f"I={i_true:.2f} {'OK' if ok else 'x'}\nd={dist:.2f} len={plen:.2f}",
                transform=ax.transAxes, fontsize=7, va="bottom", ha="left",
                bbox=dict(boxstyle="round", fc="white", alpha=0.75, lw=0))
        ax.set_xticks([]); ax.set_yticks([])

    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")

    obj_lbl = "min I (ignore dist)" if objective == "consistency" else "nearest consistent"
    fig.suptitle(f"Surrogate {obj_lbl} trajectories — {Path(scenario).stem} "
                 f"(corr fixed {corr:.2f})\nstar=θ*  X=θ'  green=feasible (true MC)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = out / f"repair_gallery_{tag}_{objective}.pdf"
    fig.savefig(p, bbox_inches="tight"); fig.savefig(p.with_suffix(".png"), dpi=150)
    print(f"saved {p}  ({n} panels)")


def plot_trajectories_all(scenarios, model_path, out_dir, **kw):
    """Generate one trajectory figure per scenario (all 3 methods each)."""
    ok, skipped = 0, []
    for sc in scenarios:
        if not Path(sc).exists():
            skipped.append(sc); continue
        print(f"\n=== trajectory: {Path(sc).stem} ===")
        try:
            plot_trajectory(sc, model_path, out_dir, **kw)
            ok += 1
        except Exception as e:
            print(f"  [skip] {Path(sc).stem}: {e}")
            skipped.append(sc)
    print(f"\ndone: {ok} figures, {len(skipped)} skipped")
    if skipped:
        print("  skipped:", ", ".join(Path(s).stem for s in skipped))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("pareto")
    pp.add_argument("--records", required=True)
    pp.add_argument("--out", default="results/repair_fig")
    pp.add_argument("--xaxis", choices=["calls", "wall"], default="calls")
    pp.add_argument("--mfmc_only", action="store_true", help="drop full-MC cma/fd")
    pp.add_argument("--drop", nargs="*", default=[], help="method names to exclude")

    pco = sub.add_parser("pareto_combined")
    pco.add_argument("--records", required=True)
    pco.add_argument("--out", default="results/repair_fig")
    pco.add_argument("--xaxis", choices=["calls", "wall"], default="wall")
    pco.add_argument("--style", choices=["bubble", "color", "dual", "single"], default="bubble")
    pco.add_argument("--yaxis", choices=["success", "inconsistency"], default="success")
    pco.add_argument("--mfmc_only", action="store_true", help="drop full-MC cma/fd")
    pco.add_argument("--drop", nargs="*", default=[], help="method names to exclude")

    pc = sub.add_parser("pareto_consistency")
    pc.add_argument("--records", required=True)
    pc.add_argument("--out", default="results/repair_fig")
    pc.add_argument("--mfmc_only", action="store_true", help="drop full-MC cma/fd")
    pc.add_argument("--drop", nargs="*", default=[], help="method names to exclude")

    pd = sub.add_parser("distance_curve")
    pd.add_argument("--records", required=True)
    pd.add_argument("--out", default="results/repair_fig")
    pd.add_argument("--bins", type=float, nargs="+", default=[0.50, 0.65, 0.80, 0.93],
                    help="bin edges for start I*")
    pd.add_argument("--mfmc_only", action="store_true", help="drop full-MC cma/fd")
    pd.add_argument("--drop", nargs="*", default=[], help="method names to exclude")

    pt = sub.add_parser("trajectory")
    pt.add_argument("--scenario", default="data/measurements_cps_v6/results_scenario_49.json")
    pt.add_argument("--model", default="results/runs/joint_2d3d_exact/product_transformer_exact.pt")
    pt.add_argument("--out", default="results/repair_fig")
    pt.add_argument("--grid", type=int, default=32)
    pt.add_argument("--surf_samples", type=int, default=600)
    pt.add_argument("--dfo_budget", type=int, default=120)
    pt.add_argument("--margin", type=float, default=0.1)
    pt.add_argument("--surr_lr", type=float, default=0.005)
    pt.add_argument("--surr_iters", type=int, default=120)
    pt.add_argument("--surr_lam", type=float, default=50.0)
    pt.add_argument("--refine_budget", type=int, default=42,
                    help="oracle finite-diff budget for the hybrid refine segment")
    pt.add_argument("--surr_stop_below", type=float, default=None,
                    help="early-stop the surrogate gradient as soon as i_hat < this")
    pt.add_argument("--start_scale", type=float, default=None,
                    help="override theta* scale factor (all methods start here)")
    pt.add_argument("--start_center", type=float, default=None,
                    help="override theta* center shift")
    pt.add_argument("--surr_optimizer", default="adam")
    pt.add_argument("--objective", choices=["repair", "consistency"], default="repair")
    pt.add_argument("--device", default="cpu")

    pta = sub.add_parser("trajectory_all")
    pta.add_argument("--model", default="results/runs/joint_2d3d_exact/product_transformer_exact.pt")
    pta.add_argument("--out", default="results/repair_fig")
    pta.add_argument("--grid", type=int, default=32)
    pta.add_argument("--surf_samples", type=int, default=600)
    pta.add_argument("--dfo_budget", type=int, default=120)
    pta.add_argument("--margin", type=float, default=0.1)
    pta.add_argument("--surr_lr", type=float, default=0.005)
    pta.add_argument("--surr_iters", type=int, default=120)
    pta.add_argument("--surr_lam", type=float, default=50.0)
    pta.add_argument("--refine_budget", type=int, default=42,
                     help="oracle finite-diff budget for the hybrid refine segment")
    pta.add_argument("--surr_optimizer", default="adam")
    pta.add_argument("--objective", choices=["repair", "consistency"], default="repair")
    pta.add_argument("--device", default="cpu")

    pg = sub.add_parser("gallery")
    pg.add_argument("--scenario", default="data/measurements_cps_v6/results_scenario_49.json")
    pg.add_argument("--model", default="results/runs/joint_2d3d_exact/product_transformer_exact.pt")
    pg.add_argument("--out", default="results/repair_fig")
    pg.add_argument("--grid", type=int, default=32)
    pg.add_argument("--surf_samples", type=int, default=600)
    pg.add_argument("--iters", type=int, default=120)
    pg.add_argument("--optimizers", nargs="+",
                    default=["adam", "momentum", "nesterov", "rmsprop"])
    pg.add_argument("--lrs", type=float, nargs="+", default=[0.003, 0.01, 0.03, 0.1])
    pg.add_argument("--lams", type=float, nargs="+", default=[10.0, 30.0, 100.0, 300.0])
    pg.add_argument("--objective", choices=["repair", "consistency"], default="repair")
    pg.add_argument("--device", default="cpu")

    a = p.parse_args()
    if a.cmd == "pareto":
        plot_pareto(a.records, a.out, xaxis=a.xaxis, mfmc_only=a.mfmc_only, drop=set(a.drop))
    elif a.cmd == "pareto_combined":
        plot_pareto_combined(a.records, a.out, xaxis=a.xaxis, style=a.style,
                             yaxis=a.yaxis, mfmc_only=a.mfmc_only, drop=set(a.drop))
    elif a.cmd == "pareto_consistency":
        plot_pareto_consistency(a.records, a.out, mfmc_only=a.mfmc_only, drop=set(a.drop))
    elif a.cmd == "distance_curve":
        plot_distance_curve(a.records, a.out, bins=tuple(a.bins),
                            mfmc_only=a.mfmc_only, drop=set(a.drop))
    elif a.cmd == "gallery":
        plot_gallery(a.scenario, a.model, a.out, grid=a.grid,
                     surf_samples=a.surf_samples, iters=a.iters,
                     optimizers=a.optimizers, lrs=a.lrs, lams=a.lams,
                     objective=a.objective, device=a.device)
    elif a.cmd == "trajectory_all":
        from .repair_benchmark import EVAL_SCENARIOS_2D3D
        plot_trajectories_all(EVAL_SCENARIOS_2D3D, a.model, a.out, grid=a.grid,
                              surf_samples=a.surf_samples, dfo_budget=a.dfo_budget,
                              margin=a.margin, surr_lr=a.surr_lr, surr_iters=a.surr_iters,
                              surr_lam=a.surr_lam, refine_budget=a.refine_budget,
                              surr_optimizer=a.surr_optimizer,
                              objective=a.objective, device=a.device)
    else:
        plot_trajectory(a.scenario, a.model, a.out, grid=a.grid,
                        surf_samples=a.surf_samples, dfo_budget=a.dfo_budget,
                        margin=a.margin, surr_lr=a.surr_lr, surr_iters=a.surr_iters,
                        surr_lam=a.surr_lam, refine_budget=a.refine_budget,
                        surr_stop_below=a.surr_stop_below,
                        start_scale=a.start_scale, start_center=a.start_center,
                        surr_optimizer=a.surr_optimizer,
                        objective=a.objective, device=a.device)


if __name__ == "__main__":
    main()
