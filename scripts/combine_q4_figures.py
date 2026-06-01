"""
Combine Q4 counterfactual figures from 2d3d and 4d evaluations into single panels.

From q4_stats.json (always available):
  q4_distances_combined, q4_ternary_combined, q4_repair_decomposition_combined

From q4_repairs_raw.csv (written by run_analysis.py after patching):
  q4_directions_combined, q4_fix_direction_combined
"""
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

BASE = Path("C:/Users/johan_rvnnln/OneDrive/Dokumente/MATLAB/MyCORA/Causality_Uncertainty_Inconsistency/results")
OUT  = BASE / "combined_figures"

# ── Load stats ────────────────────────────────────────────────────────────────
with open(BASE / "eval_2d3d/Q4_counterfactual/q4_stats.json") as f:
    s2d = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
with open(BASE / "eval_4d/Q4_counterfactual/q4_stats.json") as f:
    s4d = {k: v for k, v in json.load(f).items() if not k.startswith("_")}

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.titlesize": 10, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})
C_SCALE  = "#2166ac"
C_CENTER = "#d6604d"
C_CORR   = "#4dac26"
CMAP     = plt.cm.tab10

def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {name}")

# ══════════════════════════════════════════════════════════════════════════════
# 1. q4_distances_combined
# ══════════════════════════════════════════════════════════════════════════════
all_domains = list(s2d.keys()) + list(s4d.keys())
all_stats_d = list(s2d.values()) + list(s4d.values())
n_all = len(all_domains)
xs    = np.arange(n_all)
sep   = len(s2d) - 0.5
colors_d = [CMAP(i % 10) for i in range(len(s2d))] + ["#444444"] * len(s4d)

fig, ax = plt.subplots(figsize=(max(7.5, n_all * 1.2 + 2.5), 3.8))

for i, (d_stats, col) in enumerate(zip(all_stats_d, colors_d)):
    med  = d_stats["median_dist"]
    mean = d_stats["mean_dist"]
    mx   = d_stats["max_dist"]
    ax.plot([xs[i], xs[i]], [0, mx], color=col, lw=1.0, alpha=0.35)
    ax.plot([xs[i]-0.18, xs[i]+0.18], [mx,   mx],   color=col, lw=1.0, alpha=0.40)
    ax.plot([xs[i]-0.18, xs[i]+0.18], [mean, mean], color=col, lw=1.6, alpha=0.60)
    ax.scatter([xs[i]], [med], color=col, s=65, zorder=4,
               edgecolors="#111", linewidths=0.7)
    ax.text(xs[i], mx + 0.028, f"{med:.2f}", ha="center", va="bottom",
            fontsize=8, color="#333", fontweight="semibold")

ax.axvline(sep, color="#888", lw=1.0, ls="--", alpha=0.5)
ax.set_ylim(bottom=0)
ylim_top = ax.get_ylim()[1]
ax.fill_betweenx([0, ylim_top], -0.6, sep,           color="#e8f0f8", alpha=0.25, zorder=0)
ax.fill_betweenx([0, ylim_top], sep,  n_all - 0.4,   color="#f0f0e8", alpha=0.25, zorder=0)
ax.text(sep - 0.15, ylim_top * 0.97, "2d/3d", ha="right", va="top",
        fontsize=9, color="#555", style="italic")
ax.text(sep + 0.15, ylim_top * 0.97, "4d",    ha="left",  va="top",
        fontsize=9, color="#555", style="italic")

ax.set_xticks(xs)
ax.set_xticklabels(all_domains, rotation=30, ha="right")
ax.set_ylabel(r"Normalised repair distance $\|\theta' - \theta^*\|$")
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
ax.set_xlim(-0.6, n_all - 0.4)

h1   = mpatches.Patch(color="#aaccee", label="2d/3d domains")
h2   = mpatches.Patch(color="#aaaaaa", label="4d domain")
hdot = plt.scatter([], [], color="gray", s=55, edgecolors="#111", lw=0.7, label="Median")
hln  = plt.Line2D([0],[0], color="gray", lw=1.6, alpha=0.6, label="Mean  |  bar = max")
ax.legend(handles=[h1, h2, hdot, hln], loc="upper left", framealpha=0.9)

fig.tight_layout()
save(fig, "q4_distances_combined")

# ══════════════════════════════════════════════════════════════════════════════
# 2. q4_ternary_combined
# ══════════════════════════════════════════════════════════════════════════════
_SQRT3_2 = 3.0 ** 0.5 / 2.0
VERTS = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, _SQRT3_2]])

def bary2cart(pts):
    return np.asarray(pts) @ VERTS

fig_t, ax_t = plt.subplots(1, 1, figsize=(8.5, 6.8))

# Triangle
tri = np.vstack([VERTS, VERTS[0]])
ax_t.plot(tri[:,0], tri[:,1], color="#444", lw=1.6, zorder=2)

# Grid lines
for lvl in (0.25, 0.50, 0.75):
    for axis in range(3):
        p = np.zeros((2, 3))
        p[0, axis] = lvl;  p[0, (axis+1)%3] = 1-lvl; p[0, (axis+2)%3] = 0
        p[1, axis] = lvl;  p[1, (axis+1)%3] = 0;      p[1, (axis+2)%3] = 1-lvl
        xy = bary2cart(p)
        ax_t.plot(xy[:,0], xy[:,1], color="#aaa", lw=0.5, alpha=0.45, ls="--", zorder=1)

legend_handles = []

# 2d/3d domains (coloured diamonds)
for i, (d, stats) in enumerate(s2d.items()):
    col  = CMAP(i % 10)
    bary = np.array([stats["scale_pct"], stats["center_pct"], stats["corr_pct"]]) / 100.0
    xy   = bary2cart(bary.reshape(1, 3))[0]
    ax_t.scatter([xy[0]], [xy[1]], s=220, color=col, marker="D",
                 edgecolors="#111", linewidths=1.8, zorder=6)
    ax_t.text(xy[0], xy[1] + 0.030, d, ha="center", va="bottom",
              fontsize=8.5, color=col, fontweight="bold", zorder=7)
    legend_handles.append(mpatches.Patch(color=col, label=d))

# 4d domain (black star)
for d, stats in s4d.items():
    bary = np.array([stats["scale_pct"], stats["center_pct"], stats["corr_pct"]]) / 100.0
    xy   = bary2cart(bary.reshape(1, 3))[0]
    ax_t.scatter([xy[0]], [xy[1]], s=280, color="#111", marker="*",
                 edgecolors="#111", linewidths=1.2, zorder=8)
    ax_t.text(xy[0] + 0.045, xy[1], f"{d} (4d)", ha="left", va="center",
              fontsize=9, color="#111", fontweight="bold", zorder=9)
    legend_handles.append(mpatches.Patch(color="#555", label=f"{d} (4d)"))

# Vertex labels
vlabels  = [r"Scale $s_u$", r"Center $\Delta c_u$", r"Corr. $\rho_u$"]
vcolors  = [C_SCALE, C_CENTER, C_CORR]
voffsets = [(-0.08, -0.07), (0.08, -0.07), (0.0, 0.06)]
for (ox, oy), lbl, col, v in zip(voffsets, vlabels, vcolors, VERTS):
    ax_t.text(v[0]+ox, v[1]+oy, lbl, ha="center", va="center",
              fontsize=11, fontweight="bold", color=col, zorder=8)

dom_lbl = ["Scale-dominated", "Center-dominated", "Corr.-dominated"]
dom_off = [(-0.16, 0.09), (0.18, 0.09), (0.0, 0.11)]
for (ox, oy), lbl, col, v in zip(dom_off, dom_lbl, vcolors, VERTS):
    ax_t.text(v[0]+ox, v[1]+oy, lbl, ha="center", va="center",
              fontsize=8, color=col, style="italic", alpha=0.55, zorder=8)

ax_t.set_xlim(-0.22, 1.22)
ax_t.set_ylim(-0.15, _SQRT3_2 + 0.22)
ax_t.set_aspect("equal", adjustable="box")
ax_t.axis("off")

n_total = len(s2d) + len(s4d)
fig_t.legend(handles=legend_handles,
             title="Domain  (\u25c6 = 2d/3d centroid,  \u2605 = 4d centroid)",
             title_fontsize=9, fontsize=8.5,
             loc="lower center", bbox_to_anchor=(0.5, -0.01),
             ncol=min(n_total, 6), framealpha=0.93, edgecolor="#ccc",
             handlelength=1.2, borderpad=0.5, handletextpad=0.4, columnspacing=1.0)

save(fig_t, "q4_ternary_combined")

# ══════════════════════════════════════════════════════════════════════════════
# 3. q4_repair_decomposition_combined
# ══════════════════════════════════════════════════════════════════════════════
all_doms2  = list(s2d.keys()) + list(s4d.keys())
all_stats2 = list(s2d.values()) + list(s4d.values())
n_d  = len(all_doms2)
x_d  = np.arange(n_d)
sep2 = len(s2d) - 0.5

fig_w = max(8.0, n_d * 1.1 + 2.5)
fig3, axes3 = plt.subplots(2, 1, sharex=True, figsize=(fig_w, 7.5),
                            gridspec_kw={"hspace": 0.10, "height_ratios": [1.0, 0.85]})

p_colors = [C_SCALE, C_CENTER, C_CORR]
p_labels = [r"Scale $s_u$", r"Center $\Delta c_u$", r"Corr. $\rho_u$"]
pcts = [(d["scale_pct"], d["center_pct"], d["corr_pct"]) for d in all_stats2]

# (a) Stacked composition bar
ax_b = axes3[0]
bottoms = np.zeros(n_d)
handles_leg = []
for pi, (col, lbl) in enumerate(zip(p_colors, p_labels)):
    vals = np.array([pcts[i][pi] for i in range(n_d)])
    bars = ax_b.bar(x_d, vals, bottom=bottoms, width=0.55,
                    color=col, alpha=0.86, zorder=3, label=lbl)
    handles_leg.append(bars[0])
    for xi, (bot, h) in enumerate(zip(bottoms, vals)):
        if h > 14:
            ax_b.text(xi, bot + h/2, f"{h:.0f}%", ha="center", va="center",
                      fontsize=8.5, color="white", fontweight="bold", zorder=5)
    bottoms += vals

ax_b.axvline(sep2, color="#888", lw=1.0, ls="--", alpha=0.6, zorder=5)
ax_b.axhline(100, color="#666", lw=0.8, ls="--", alpha=0.35, zorder=2)
ax_b.fill_betweenx([0, 114], -0.6,  sep2,       color="#e8f0f8", alpha=0.18, zorder=0)
ax_b.fill_betweenx([0, 114], sep2,  n_d - 0.4,  color="#f0f0e8", alpha=0.18, zorder=0)
ax_b.text(sep2 - 0.2, 110, "2d/3d", ha="right", va="top",
          fontsize=9, color="#555", style="italic")
ax_b.text(sep2 + 0.2, 110, "4d",    ha="left",  va="top",
          fontsize=9, color="#555", style="italic")
ax_b.set_ylim(0, 114)
ax_b.set_xlim(-0.6, n_d - 0.4)
ax_b.set_ylabel("Repair composition\n(%)")
ax_b.set_title("(a) Absolute repair composition — which parameter dominates",
               fontsize=10, pad=6)
ax_b.yaxis.grid(True, linestyle="--", alpha=0.4)
ax_b.set_axisbelow(True)

# (b) Repair distance (median + mean + max)
ax_c   = axes3[1]
col_c  = [CMAP(i % 10) for i in range(len(s2d))] + ["#444444"] * len(s4d)
meds   = [d["median_dist"] for d in all_stats2]
means  = [d["mean_dist"]   for d in all_stats2]
maxes  = [d["max_dist"]    for d in all_stats2]
g_max  = max(maxes)

for i, (med, mean, mx, col) in enumerate(zip(meds, means, maxes, col_c)):
    ax_c.plot([x_d[i], x_d[i]], [0, mx], color=col, lw=1.0, alpha=0.35)
    ax_c.plot([x_d[i]-0.18, x_d[i]+0.18], [mx,   mx],   color=col, lw=1.0, alpha=0.40)
    ax_c.plot([x_d[i]-0.18, x_d[i]+0.18], [mean, mean], color=col, lw=1.6, alpha=0.60)
    ax_c.scatter([x_d[i]], [med], color=col, s=58, zorder=4,
                 edgecolors="#111", linewidths=0.7)
    ax_c.text(x_d[i], mx + g_max * 0.04, f"{med:.2f}",
              ha="center", va="bottom", fontsize=7.5, color="#333")

ax_c.axvline(sep2, color="#888", lw=1.0, ls="--", alpha=0.6, zorder=5)
ax_c.fill_betweenx([0, g_max*1.45], -0.6,  sep2,      color="#e8f0f8", alpha=0.18, zorder=0)
ax_c.fill_betweenx([0, g_max*1.45], sep2,  n_d - 0.4, color="#f0f0e8", alpha=0.18, zorder=0)
ax_c.set_ylim(0, g_max * 1.45)
ax_c.set_xlim(-0.6, n_d - 0.4)
ax_c.set_xticks(x_d)
ax_c.set_xticklabels(all_doms2, rotation=30, ha="right")
ax_c.set_ylabel(r"Repair distance $\|\theta' - \theta^*\|$")
ax_c.set_title("(b) Repair distance — cost of consistency restoration",
               fontsize=10, pad=6)
ax_c.yaxis.grid(True, linestyle="--", alpha=0.4)
ax_c.set_axisbelow(True)

fig3.legend(handles=handles_leg, labels=p_labels,
            loc="lower center", bbox_to_anchor=(0.5, -0.025),
            ncol=3, fontsize=9.5, framealpha=0.93, edgecolor="#ccc",
            handlelength=1.5, handletextpad=0.6, columnspacing=2.0)
fig3.suptitle("Counterfactual repair decomposition — 2d/3d vs 4d",
              fontsize=11, fontweight="semibold", y=1.01)

save(fig3, "q4_repair_decomposition_combined")

# ══════════════════════════════════════════════════════════════════════════════
# 4 & 5. q4_directions_combined  +  q4_fix_direction_combined
#         Require q4_repairs_raw.csv from both eval folders.
#         These are written by run_analysis.py (patched version).
# ══════════════════════════════════════════════════════════════════════════════
csv_2d3d = BASE / "eval_2d3d/Q4_counterfactual/q4_repairs_raw.csv"
csv_4d   = BASE / "eval_4d/Q4_counterfactual/q4_repairs_raw.csv"

if not csv_2d3d.exists() or not csv_4d.exists():
    missing = [p for p in (csv_2d3d, csv_4d) if not p.exists()]
    print(f"\nSkipping q4_directions_combined and q4_fix_direction_combined —")
    print(f"  missing: {[str(p) for p in missing]}")
    print(f"  Re-run run_analysis.py --plots q4 for each eval set to generate them.")
    print("\nAll done.")
else:
    # Load and tag both CSV files
    def load_repairs(csv_path, eval_tag):
        rows = []
        with open(csv_path, newline="") as fh:
            for r in csv.DictReader(fh):
                rows.append({
                    "domain":       r["domain"],
                    "eval":         eval_tag,
                    "delta_scale":  float(r["delta_scale"]),
                    "delta_center": float(r["delta_center"]),
                    "delta_corr":   float(r["delta_corr"]),
                })
        return rows

    repairs_2d3d = load_repairs(csv_2d3d, "2d/3d")
    repairs_4d   = load_repairs(csv_4d,   "4d")
    all_repairs  = repairs_2d3d + repairs_4d

    # Unique (eval, domain) groups — preserve order: 2d/3d first, then 4d
    seen, groups = set(), []
    for r in all_repairs:
        key = (r["eval"], r["domain"])
        if key not in seen:
            seen.add(key)
            groups.append(key)

    SPANS      = np.array([3.05, 2.10, 1.05])   # parameter span ranges
    PARAM_KEYS = ["delta_scale", "delta_center", "delta_corr"]
    PARAM_XLBLS = [r"Scale $s_u$", r"Center $\Delta c_u$", r"Corr. $\rho_u$"]
    PARAM_COLORS_CF = [C_SCALE, C_CENTER, C_CORR]

    # Build per-group arrays (span-normalised)
    group_par = {}   # key -> (N, 3) signed normalised
    group_tern = {}  # key -> (N, 3) absolute normalised (for ternary)
    for r in all_repairs:
        key = (r["eval"], r["domain"])
        dt  = np.array([r[k] for k in PARAM_KEYS], dtype=float) / (SPANS + 1e-10)
        group_par.setdefault(key, []).append(dt)
        abs_n = np.abs(dt); tot = abs_n.sum() + 1e-10
        group_tern.setdefault(key, []).append(abs_n / tot)

    group_par   = {k: np.array(v) for k, v in group_par.items()}
    group_tern  = {k: np.array(v) for k, v in group_tern.items()}
    group_means = {k: v.mean(axis=0) for k, v in group_par.items()}

    # Colour: same tab10 for 2d/3d domains, dark grey for 4d
    dom_2d3d = sorted({d for e, d in groups if e == "2d/3d"})
    col_map  = {("2d/3d", d): CMAP(i % 10) for i, d in enumerate(dom_2d3d)}
    for e, d in groups:
        if e == "4d":
            col_map[(e, d)] = "#333333"

    def group_label(ev, dm):
        return f"{dm} ({ev})" if ev == "4d" else dm

    # ── Figure 4: q4_directions_combined ─────────────────────────────────────
    fig_d, ax_d = plt.subplots(1, 1, figsize=(11.0, 4.6))

    x_pos = np.array([0, 1, 2])
    y_abs = max(max(abs(v).max() for v in group_par.values()) * 1.18, 0.15)

    ax_d.fill_between([-0.35, 2.35], [0, 0],         [y_abs, y_abs],   color="#ccffcc", alpha=0.08, zorder=0)
    ax_d.fill_between([-0.35, 2.35], [-y_abs, -y_abs], [0, 0],         color="#ffcccc", alpha=0.08, zorder=0)
    ax_d.text(2.30,  y_abs*0.90, "increase", ha="right", va="top",    fontsize=9, color="#448844", style="italic")
    ax_d.text(2.30, -y_abs*0.90, "decrease", ha="right", va="bottom", fontsize=9, color="#884444", style="italic")

    for xi in x_pos:
        ax_d.axvline(xi, color="#999", lw=0.9, alpha=0.50, zorder=1)
    ax_d.axhline(0, color="#444", lw=1.0, ls="--", alpha=0.55, zorder=3)

    # Individual traces (faint)
    for key, arr in group_par.items():
        col = col_map[key]
        lw  = 0.45 if key[0] == "2d/3d" else 0.60
        for row in arr:
            ax_d.plot(x_pos, row, color=col, alpha=0.08, lw=lw, zorder=2)

    # Domain mean lines
    for key in groups:
        mean_row = group_means[key]
        col  = col_map[key]
        ev, dm = key
        ls   = "-" if ev == "2d/3d" else "--"
        lw   = 2.3 if ev == "2d/3d" else 2.6
        mrk  = "o" if ev == "2d/3d" else "^"
        ax_d.plot(x_pos, mean_row, color=col, alpha=0.95, lw=lw, zorder=5,
                  ls=ls, marker=mrk, markersize=5.5,
                  markeredgecolor="white", markeredgewidth=0.9)
        ax_d.text(2.04, mean_row[2], group_label(ev, dm),
                  ha="left", va="center", fontsize=7.5, color=col,
                  fontweight="bold", zorder=7)

    ax_d.set_xlim(-0.35, 2.75)
    ax_d.set_ylim(-y_abs, y_abs)
    ax_d.set_xticks(x_pos)
    ax_d.set_xticklabels(PARAM_XLBLS)
    ax_d.set_ylabel(r"Repair shift (span-normalised $\delta\theta_i / \mathrm{range}_i$)")
    ax_d.spines["bottom"].set_visible(False)
    ax_d.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax_d.set_axisbelow(True)

    # Legend: 2d/3d solid circle, 4d dashed triangle
    h_2d = plt.Line2D([0],[0], color="#555", lw=2.0, ls="-",  marker="o", markersize=5, label="2d/3d domain mean")
    h_4d = plt.Line2D([0],[0], color="#333", lw=2.3, ls="--", marker="^", markersize=5, label="4d domain mean")
    ax_d.legend(handles=[h_2d, h_4d], loc="upper right", framealpha=0.9)

    fig_d.tight_layout()
    save(fig_d, "q4_directions_combined")

    # ── Figure 5: q4_fix_direction_combined ──────────────────────────────────
    n_groups  = len(groups)
    fig_h     = max(4.0, n_groups * 0.48 + 2.0)
    fig_f, axes_f = plt.subplots(1, 3, figsize=(14.0, fig_h), sharey=True,
                                  gridspec_kw={"wspace": 0.10})
    y_pos = np.arange(n_groups, 0, -1)   # top-to-bottom

    for col_idx, (ax, p_key, col_p, pxlbl) in enumerate(
            zip(axes_f, PARAM_KEYS, PARAM_COLORS_CF, PARAM_XLBLS)):

        dom_means = [group_means[g][col_idx] for g in groups]
        arr       = np.array(dom_means, dtype=float)
        sigma     = arr.std() if arr.std() > 1e-9 else 1.0
        pad       = max(abs(arr).max() * 0.28, 0.04)
        x_lo      = min(arr.mean() - 2.8*sigma, -pad)
        x_hi      = max(arr.mean() + 2.8*sigma,  pad)

        for yi, (key, val) in enumerate(zip(groups, dom_means)):
            clipped  = val < x_lo or val > x_hi
            val_draw = float(np.clip(val, x_lo * 0.92, x_hi * 0.92))
            arrow_c  = col_p if val >= 0 else "#cc4444"
            mrk_tip  = ">" if val >= 0 else "<"
            ls_style = "-" if key[0] == "2d/3d" else "--"

            ax.annotate("",
                        xy=(val_draw, y_pos[yi]),
                        xytext=(0.0,  y_pos[yi]),
                        arrowprops=dict(arrowstyle="-|>",
                                        color=arrow_c, lw=2.0,
                                        mutation_scale=11,
                                        linestyle=ls_style))

            gap   = (x_hi - x_lo) * 0.03
            lbl_x = val_draw + (gap if val >= 0 else -gap)
            suffix = "\u22ef" if clipped else ""
            ax.text(lbl_x, y_pos[yi],
                    f"{val:+.3f}{suffix}",
                    ha="left" if val >= 0 else "right",
                    va="center", fontsize=8, color="#222")

        ax.axvline(0, color="#444", lw=1.0, ls="-", alpha=0.70, zorder=2)
        ax.fill_betweenx([y_pos[-1]-0.5, y_pos[0]+0.5], x_lo, 0, color="#ffcccc", alpha=0.10, zorder=0)
        ax.fill_betweenx([y_pos[-1]-0.5, y_pos[0]+0.5], 0, x_hi, color="#ccffcc", alpha=0.10, zorder=0)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_pos[-1]-0.7, y_pos[0]+0.7)
        ax.set_yticks(y_pos)

        if col_idx == 0:
            ylabels = [group_label(ev, dm) for ev, dm in groups]
            ax.set_yticklabels(ylabels, fontsize=9)
        else:
            ax.set_yticklabels([])

        ax.set_xlabel(r"$\leftarrow$ decrease    increase $\rightarrow$", fontsize=8.5)
        ax.set_title(pxlbl, fontsize=10, pad=6, color=col_p)
        ax.yaxis.grid(False)
        ax.xaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes_f[0].set_ylabel("Domain", fontsize=10)
    h_2d2 = plt.Line2D([0],[0], color="#555", lw=2.0, ls="-",  label="2d/3d")
    h_4d2 = plt.Line2D([0],[0], color="#333", lw=2.3, ls="--", label="4d")
    axes_f[1].legend(handles=[h_2d2, h_4d2], loc="upper right", framealpha=0.9, fontsize=9)
    fig_f.suptitle(
        "Mean normalised repair shift required to restore consistency\n"
        r"(positive = increase parameter, negative = decrease)",
        fontsize=10, y=1.03)

    fig_f.tight_layout()
    save(fig_f, "q4_fix_direction_combined")

    print("\nAll done.")
