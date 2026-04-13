"""2D zonotope visualizations for the paper.

Publication-quality figures showing:
  1. Zonotope overlap with MC consistency samples
  2. Intervention effect strip (multi-panel)
  3A. Parameter-space inconsistency heatmap
  3B. Physical-space membership map
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path as MplPath
from matplotlib.colors import ListedColormap, BoundaryNorm

from causal_engine import (
    Zonotope, Scenario, affine_map,
    apply_intervention, apply_compound_intervention,
    PARAM_LABELS,
)

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

CLR_SOURCE = "#4878CF"
CLR_TARGET = "#E8A838"
CLR_CONSISTENT = "#2CA02C"
CLR_INCONSISTENT = "#D62728"

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def zonotope_vertices_2d(Z: Zonotope) -> np.ndarray:
    """Ordered boundary vertices of a 2D zonotope for polygon plotting.

    Sorts generators by angle then walks the Minkowski-sum boundary.
    Returns shape (2p+1, 2) — closed polygon (first == last).
    """
    if Z.dim != 2:
        raise ValueError("Only 2D zonotopes supported")
    angles = np.arctan2(Z.G[1], Z.G[0])
    order = np.argsort(angles)
    G_sorted = Z.G[:, order]

    v = Z.c - G_sorted.sum(axis=1)
    vertices = [v.copy()]
    for i in range(G_sorted.shape[1]):
        v = v + 2 * G_sorted[:, i]
        vertices.append(v.copy())
    for i in range(G_sorted.shape[1]):
        v = v - 2 * G_sorted[:, i]
        vertices.append(v.copy())
    return np.array(vertices)


def _contains_2d(Z: Zonotope, points: np.ndarray) -> np.ndarray:
    """Fast 2D containment via polygon ray-casting (no LP)."""
    verts = zonotope_vertices_2d(Z)
    path = MplPath(verts)
    return path.contains_points(points)


def _compute_I_theta_2d(
    scenario: Scenario,
    theta: dict,
    n_samples: int,
    rng: np.random.Generator,
) -> float:
    """Fast 2D I(theta) using polygon containment instead of LP."""
    Z_mod = apply_compound_intervention(scenario.source, theta)
    Z_prop = affine_map(Z_mod, scenario.F, scenario.f)
    xi = rng.uniform(-1, 1, size=(n_samples, Z_prop.n_generators))
    points = Z_prop.c[None, :] + xi @ Z_prop.G.T
    inside = _contains_2d(scenario.target, points)
    return 1.0 - inside.sum() / n_samples


# ---------------------------------------------------------------------------
# Figure 1: Zonotope overlap + MC sample scatter
# ---------------------------------------------------------------------------

def plot_zonotope_consistency(
    scenario: Scenario,
    source_override: Optional[Zonotope] = None,
    n_samples: int = 1000,
    seed: Optional[int] = None,
    output_path: Optional[str | Path] = None,
    title_suffix: str = "",
    figsize: Tuple[float, float] = (6, 5.5),
) -> plt.Figure:
    """Source and target zonotopes with MC samples coloured by consistency."""
    if scenario.dim != 2:
        raise ValueError("Only 2D scenarios supported")

    source = source_override if source_override is not None else scenario.source
    Z_prop = affine_map(source, scenario.F, scenario.f)
    Z_target = scenario.target

    rng = np.random.default_rng(seed)
    xi = rng.uniform(-1, 1, size=(n_samples, Z_prop.n_generators))
    points = Z_prop.c[None, :] + xi @ Z_prop.G.T
    consistent = _contains_2d(Z_target, points)

    I_theta = 1.0 - consistent.sum() / n_samples

    verts_src = zonotope_vertices_2d(Z_prop)
    verts_tgt = zonotope_vertices_2d(Z_target)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    ax.add_patch(plt.Polygon(verts_src, alpha=0.20, facecolor=CLR_SOURCE,
                             edgecolor=CLR_SOURCE, linewidth=2,
                             label="Source (propagated)"))
    ax.add_patch(plt.Polygon(verts_tgt, alpha=0.20, facecolor=CLR_TARGET,
                             edgecolor=CLR_TARGET, linewidth=2,
                             label="Target"))

    inc = ~consistent
    ax.scatter(points[inc, 0], points[inc, 1],
               s=8, c=CLR_INCONSISTENT, alpha=0.5, linewidths=0,
               label=f"Inconsistent ({inc.sum()})", zorder=2)
    ax.scatter(points[consistent, 0], points[consistent, 1],
               s=8, c=CLR_CONSISTENT, alpha=0.5, linewidths=0,
               label=f"Consistent ({consistent.sum()})", zorder=3)

    ax.set_xlabel(r"$x_1$", fontsize=12)
    ax.set_ylabel(r"$x_2$", fontsize=12)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)

    title = f"{scenario.name}\n$I(\\theta) = {I_theta:.3f}$"
    if title_suffix:
        title += f"  {title_suffix}"
    ax.set_title(title, fontsize=12, fontweight="bold")

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")
    return fig


# ---------------------------------------------------------------------------
# Figure 2: Intervention strip (multi-panel)
# ---------------------------------------------------------------------------

def plot_intervention_strip(
    scenario: Scenario,
    intervention_type: str,
    param_name: str,
    param_values: List[float],
    n_samples: int = 500,
    seed: Optional[int] = None,
    output_path: Optional[str | Path] = None,
    figsize_per_panel: Tuple[float, float] = (4, 4),
) -> plt.Figure:
    """Multi-panel strip showing how one intervention changes geometry."""
    if scenario.dim != 2:
        raise ValueError("Only 2D scenarios supported")

    n = len(param_values)
    fig, axes = plt.subplots(
        1, n,
        figsize=(figsize_per_panel[0] * n, figsize_per_panel[1]),
        constrained_layout=True,
    )
    if n == 1:
        axes = [axes]

    Z_target = scenario.target
    verts_tgt = zonotope_vertices_2d(Z_target)

    # Shared axis limits across panels
    all_lo, all_hi = [], []
    for val in param_values:
        Z_mod = apply_intervention(scenario.source, intervention_type,
                                   {param_name: val})
        Z_prop = affine_map(Z_mod, scenario.F, scenario.f)
        lo, hi = Z_prop.interval_bounds()
        all_lo.append(lo)
        all_hi.append(hi)
    tgt_lo, tgt_hi = Z_target.interval_bounds()
    all_lo.append(tgt_lo)
    all_hi.append(tgt_hi)
    global_lo = np.min(all_lo, axis=0)
    global_hi = np.max(all_hi, axis=0)
    margin = (global_hi - global_lo) * 0.08
    xlim = (global_lo[0] - margin[0], global_hi[0] + margin[0])
    ylim = (global_lo[1] - margin[1], global_hi[1] + margin[1])

    rng = np.random.default_rng(seed)
    plabel = PARAM_LABELS.get(param_name, param_name)

    for ax, val in zip(axes, param_values):
        Z_mod = apply_intervention(scenario.source, intervention_type,
                                   {param_name: val})
        Z_prop = affine_map(Z_mod, scenario.F, scenario.f)

        xi = rng.uniform(-1, 1, size=(n_samples, Z_prop.n_generators))
        points = Z_prop.c[None, :] + xi @ Z_prop.G.T
        consistent = _contains_2d(Z_target, points)
        I_theta = 1.0 - consistent.sum() / n_samples

        verts_src = zonotope_vertices_2d(Z_prop)

        ax.add_patch(plt.Polygon(verts_tgt, alpha=0.18, facecolor=CLR_TARGET,
                                 edgecolor=CLR_TARGET, linewidth=1.5))
        ax.add_patch(plt.Polygon(verts_src, alpha=0.18, facecolor=CLR_SOURCE,
                                 edgecolor=CLR_SOURCE, linewidth=1.5))

        inc = ~consistent
        ax.scatter(points[inc, 0], points[inc, 1],
                   s=6, c=CLR_INCONSISTENT, alpha=0.45, linewidths=0, zorder=2)
        ax.scatter(points[consistent, 0], points[consistent, 1],
                   s=6, c=CLR_CONSISTENT, alpha=0.45, linewidths=0, zorder=3)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.grid(alpha=0.15)
        ax.set_xlabel(r"$x_1$", fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel(r"$x_2$", fontsize=10)
        ax.set_title(f"{plabel}$={val}$\n$I(\\theta)={I_theta:.3f}$",
                     fontsize=10, fontweight="bold")

    fig.suptitle(f"{intervention_type.capitalize()} — {scenario.name}",
                 fontsize=13, fontweight="bold")

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")
    return fig


# ---------------------------------------------------------------------------
# Figure 3A: Parameter-space inconsistency heatmap
# ---------------------------------------------------------------------------

def plot_inconsistency_heatmap(
    scenario: Scenario,
    param1: str = "scale_factor",
    param1_range: Tuple[float, float] = (0.3, 4.0),
    param2: str = "center_delta",
    param2_range: Tuple[float, float] = (0.0, 0.3),
    grid_n: int = 25,
    mc_samples: int = 500,
    seed: int = 42,
    output_path: Optional[str | Path] = None,
    figsize: Tuple[float, float] = (7, 5.5),
) -> plt.Figure:
    """Heatmap of I(theta) over a 2D grid of two intervention parameters."""
    from tqdm import tqdm

    v1 = np.linspace(*param1_range, grid_n)
    v2 = np.linspace(*param2_range, grid_n)
    V1, V2 = np.meshgrid(v1, v2)
    I_grid = np.empty_like(V1)

    rng = np.random.default_rng(seed)
    total = grid_n * grid_n
    pbar = tqdm(total=total, desc="Heatmap grid", unit="pt")
    for i in range(grid_n):
        for j in range(grid_n):
            theta = {param1: V1[i, j], param2: V2[i, j]}
            I_grid[i, j] = _compute_I_theta_2d(scenario, theta, mc_samples, rng)
            pbar.update(1)
    pbar.close()

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    cf = ax.contourf(V1, V2, I_grid, levels=20, cmap="RdYlGn_r")
    cs = ax.contour(V1, V2, I_grid, levels=8, colors="k",
                    linewidths=0.4, alpha=0.5)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.2f")
    fig.colorbar(cf, ax=ax, label=r"$I(\theta)$", shrink=0.9)

    ax.set_xlabel(PARAM_LABELS.get(param1, param1), fontsize=12)
    ax.set_ylabel(PARAM_LABELS.get(param2, param2), fontsize=12)
    ax.set_title(f"Inconsistency Landscape — {scenario.name}",
                 fontsize=13, fontweight="bold")

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")
    return fig


# ---------------------------------------------------------------------------
# Figure 3B: Physical-space membership map
# ---------------------------------------------------------------------------

def plot_membership_map(
    scenario: Scenario,
    source_override: Optional[Zonotope] = None,
    grid_n: int = 200,
    output_path: Optional[str | Path] = None,
    title_suffix: str = "",
    figsize: Tuple[float, float] = (7, 5.5),
) -> plt.Figure:
    """Rasterised map of source / target / overlap membership regions."""
    if scenario.dim != 2:
        raise ValueError("Only 2D scenarios supported")

    source = source_override if source_override is not None else scenario.source
    Z_prop = affine_map(source, scenario.F, scenario.f)
    Z_target = scenario.target

    lo_s, hi_s = Z_prop.interval_bounds()
    lo_t, hi_t = Z_target.interval_bounds()
    lo = np.minimum(lo_s, lo_t)
    hi = np.maximum(hi_s, hi_t)
    margin = (hi - lo) * 0.12
    lo -= margin
    hi += margin

    x1 = np.linspace(lo[0], hi[0], grid_n)
    x2 = np.linspace(lo[1], hi[1], grid_n)
    X1, X2 = np.meshgrid(x1, x2)
    grid_pts = np.column_stack([X1.ravel(), X2.ravel()])

    in_source = _contains_2d(Z_prop, grid_pts).reshape(grid_n, grid_n)
    in_target = _contains_2d(Z_target, grid_pts).reshape(grid_n, grid_n)

    # 0=neither, 1=source only, 2=target only, 3=both
    membership = in_source.astype(int) + 2 * in_target.astype(int)

    cmap = ListedColormap(["#F5F5F5", CLR_SOURCE, CLR_TARGET, CLR_CONSISTENT])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.pcolormesh(X1, X2, membership, cmap=cmap, norm=norm, shading="auto")

    verts_src = zonotope_vertices_2d(Z_prop)
    verts_tgt = zonotope_vertices_2d(Z_target)
    ax.plot(*verts_src.T, color=CLR_SOURCE, linewidth=1.8)
    ax.plot(*verts_tgt.T, color=CLR_TARGET, linewidth=1.8)

    legend_elements = [
        mpatches.Patch(facecolor=CLR_SOURCE, alpha=0.7, label="Source only"),
        mpatches.Patch(facecolor=CLR_TARGET, alpha=0.7, label="Target only"),
        mpatches.Patch(facecolor=CLR_CONSISTENT, alpha=0.7,
                       label="Overlap (consistent)"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="upper left",
              framealpha=0.9)

    ax.set_xlabel(r"$x_1$", fontsize=12)
    ax.set_ylabel(r"$x_2$", fontsize=12)
    ax.set_aspect("equal")

    title = f"Membership Map — {scenario.name}"
    if title_suffix:
        title += f"  {title_suffix}"
    ax.set_title(title, fontsize=13, fontweight="bold")

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")
    return fig


# ---------------------------------------------------------------------------
# Figure 3C: Slice panels — 2D heatmaps at fixed correlation values
# ---------------------------------------------------------------------------

def plot_inconsistency_slices(
    scenario: Scenario,
    param1: str = "scale_factor",
    param1_range: Tuple[float, float] = (0.3, 4.0),
    param2: str = "center_delta",
    param2_range: Tuple[float, float] = (-0.5, 1.5),
    slice_param: str = "correlation_strength",
    slice_values: Optional[List[float]] = None,
    grid_n: int = 25,
    mc_samples: int = 500,
    seed: int = 42,
    output_path: Optional[str | Path] = None,
    figsize_per_panel: float = 4.5,
) -> plt.Figure:
    """Row of scale x shift heatmaps, one per fixed correlation value."""
    from tqdm import tqdm

    if slice_values is None:
        slice_values = [0.0, 0.3, 0.6, 0.9]

    n_slices = len(slice_values)
    v1 = np.linspace(*param1_range, grid_n)
    v2 = np.linspace(*param2_range, grid_n)
    V1, V2 = np.meshgrid(v1, v2)

    rng = np.random.default_rng(seed)
    grids = []
    total = n_slices * grid_n * grid_n
    pbar = tqdm(total=total, desc="Slice panels", unit="pt")
    for sv in slice_values:
        I_grid = np.empty_like(V1)
        for i in range(grid_n):
            for j in range(grid_n):
                theta = {param1: V1[i, j], param2: V2[i, j], slice_param: sv}
                I_grid[i, j] = _compute_I_theta_2d(
                    scenario, theta, mc_samples, rng)
                pbar.update(1)
        grids.append(I_grid)
    pbar.close()

    vmin = min(g.min() for g in grids)
    vmax = max(g.max() for g in grids)

    fig, axes = plt.subplots(
        1, n_slices,
        figsize=(figsize_per_panel * n_slices + 1.2, figsize_per_panel),
        constrained_layout=True,
    )
    if n_slices == 1:
        axes = [axes]

    slice_label = PARAM_LABELS.get(slice_param, slice_param)

    for ax, sv, I_grid in zip(axes, slice_values, grids):
        cf = ax.contourf(V1, V2, I_grid, levels=20, cmap="RdYlGn_r",
                         vmin=vmin, vmax=vmax)
        ax.contour(V1, V2, I_grid, levels=8, colors="k",
                   linewidths=0.35, alpha=0.4)
        ax.set_xlabel(PARAM_LABELS.get(param1, param1), fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel(PARAM_LABELS.get(param2, param2), fontsize=10)
        ax.set_title(f"{slice_label}$={sv}$", fontsize=10, fontweight="bold")

    fig.colorbar(cf, ax=axes, label=r"$I(\theta)$", shrink=0.85)
    fig.suptitle(f"Inconsistency Landscape — {scenario.name}",
                 fontsize=13, fontweight="bold")

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")
    return fig


# ---------------------------------------------------------------------------
# Convergence diagnostics
# ---------------------------------------------------------------------------

def plot_mc_convergence(
    mc_results: Dict,
    scenario_name: str = "",
    output_path: Optional[str | Path] = None,
    figsize: Tuple[float, float] = (7, 4.5),
) -> plt.Figure:
    """I(theta) estimate vs MC sample count with +/- 1 std band."""
    ns = np.array(mc_results["n_samples"])
    means = np.array(mc_results["I_theta_mean"])
    stds = np.array(mc_results["I_theta_std"])

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    ax.fill_between(ns, means - stds, means + stds,
                    alpha=0.25, color=CLR_SOURCE)
    ax.plot(ns, means, "o-", color=CLR_SOURCE, markersize=5, linewidth=1.5,
            label=r"$\hat{I}(\theta) \pm 1\sigma$")

    ref = means[-1]
    ax.axhline(ref, color="grey", linewidth=0.8, linestyle="--",
               label=f"Converged: {ref:.4f}")

    ax.set_xscale("log")
    ax.set_xlabel("MC samples per evaluation", fontsize=12)
    ax.set_ylabel(r"$I(\theta)$", fontsize=12)
    title = "MC Convergence"
    if scenario_name:
        title += f" — {scenario_name}"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")
    return fig


def plot_sobol_convergence(
    sobol_conv: Dict,
    scenario_name: str = "",
    output_path: Optional[str | Path] = None,
    figsize: Tuple[float, float] = (10, 4.5),
) -> plt.Figure:
    """S1 and ST Sobol indices vs Saltelli N with confidence bands."""
    N_vals = np.array(sobol_conv["N"])
    param_names = list(sobol_conv["S1"].keys())
    colors = ["#4878CF", "#D65F5F", "#6ACC65"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize,
                                   constrained_layout=True)

    for idx, name in enumerate(param_names):
        label = PARAM_LABELS.get(name, name)
        c = colors[idx % len(colors)]

        s1 = np.array(sobol_conv["S1"][name])
        s1c = np.array(sobol_conv["S1_conf"][name])
        ax1.fill_between(N_vals, s1 - s1c, s1 + s1c, alpha=0.15, color=c)
        ax1.plot(N_vals, s1, "o-", color=c, markersize=5, linewidth=1.5,
                 label=label)

        st = np.array(sobol_conv["ST"][name])
        stc = np.array(sobol_conv["ST_conf"][name])
        ax2.fill_between(N_vals, st - stc, st + stc, alpha=0.15, color=c)
        ax2.plot(N_vals, st, "o-", color=c, markersize=5, linewidth=1.5,
                 label=label)

    for ax, title in [(ax1, r"$S_i$ (First-Order)"),
                      (ax2, r"$S_i^T$ (Total-Effect)")]:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Saltelli N", fontsize=12)
        ax.set_ylabel("Sobol Index", fontsize=12)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)
        ax.set_ylim(bottom=-0.1)

    suptitle = "Sobol Convergence"
    if scenario_name:
        suptitle += f" — {scenario_name}"
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")
    return fig
