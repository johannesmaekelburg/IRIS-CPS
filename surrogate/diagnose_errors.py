"""
Diagnose the off-diagonal bands in pred vs true scatter plots.

Retrains deepsets_v2, collects per-sample predictions with scenario metadata,
and analyzes what characterizes the high-error points.

Usage:
    python -m surrogate.diagnose_errors
    python -m surrogate.diagnose_errors --epochs 5 --max_samples 50000  # quick
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset_v2 import ZonotopeDatasetV2, collate_fn, EPS
from .models_v2 import create_model

ROOT = Path(__file__).resolve().parents[1]
DATA_DIRS = [
    ROOT / "data" / "measurements_v6",
    ROOT / "data" / "measurements_cps_v6",
]


class TrackedDataset(ZonotopeDatasetV2):
    """Extends ZonotopeDatasetV2 to track scenario source per sample."""

    def _load(self, json_file, label_key):
        with open(json_file) as f:
            data = json.load(f)

        scenario_name = data.get("scenario_name", json_file.stem)
        parent = json_file.parent.name

        for exp in data["experiments"]:
            try:
                unc = exp["post_state"]["uncertainty"]
                inc = exp["post_state"]["inconsistency"]

                label = inc.get(label_key)
                if not isinstance(label, (int, float)) or not np.isfinite(label):
                    continue

                c1 = np.array(unc["source_center"], dtype=np.float64)
                G1 = np.array(unc["source_generators"], dtype=np.float64)
                c2 = np.array(unc["target_center"], dtype=np.float64)
                G2 = np.array(unc["target_generators"], dtype=np.float64)

                if G1.ndim == 1:
                    G1 = G1[:, None]
                if G2.ndim == 1:
                    G2 = G2[:, None]

                d = len(c1)
                if d > 4:
                    continue

                from .dataset_v2 import _extract_upr, _compute_features
                scales, offsets, upr_onehot = _extract_upr(exp, d)
                feats = _compute_features(c1, G1, c2, G2, scales, offsets, upr_onehot)

                i_theta = float(inc.get("I_theta", float("nan")))
                i_aabb = float(inc.get("jaccard_inconsistency",
                               1.0 - inc.get("jaccard_index", float("nan"))))

                # Intervention parameters
                sf = float(exp.get("scale_factor", 1.0))
                cd = float(exp.get("center_delta", 0.0))
                cs = float(exp.get("correlation_strength", 0.0))

                # UPR type
                cr = exp.get("consistency_relations")
                upr_type = "unknown"
                if cr and isinstance(cr, list) and len(cr) > 0:
                    upr_type = cr[0].get("upr_type", "unknown")

                sample = {
                    **feats,
                    "label": np.float32(label),
                    "i_theta": i_theta,
                    "i_aabb": i_aabb,
                    "i_mfmc": float("nan"),
                    # Tracking metadata
                    "scenario_file": str(json_file.name),
                    "scenario_name": scenario_name,
                    "data_source": parent,
                    "scale_factor": sf,
                    "center_delta": cd,
                    "correlation_strength": cs,
                    "upr_type": upr_type,
                }
                self.samples.append(sample)
            except (KeyError, ValueError, TypeError):
                continue


def tracked_collate_fn(batch):
    """Collate that also passes through metadata lists."""
    from .dataset_v2 import collate_fn as base_collate
    result = base_collate(batch)
    result["scenario_file"] = [s["scenario_file"] for s in batch]
    result["scenario_name"] = [s["scenario_name"] for s in batch]
    result["data_source"] = [s["data_source"] for s in batch]
    result["scale_factor"] = np.array([s["scale_factor"] for s in batch])
    result["center_delta"] = np.array([s["center_delta"] for s in batch])
    result["correlation_strength"] = np.array([s["correlation_strength"] for s in batch])
    result["upr_type"] = [s["upr_type"] for s in batch]
    result["i_aabb_arr"] = np.array([s["i_aabb"] for s in batch])
    return result


def _get_scenario_dim(json_file):
    with open(json_file) as f:
        data = json.load(f)
    return len(data["experiments"][0]["post_state"]["uncertainty"]["source_center"])


@torch.no_grad()
def collect_all(model, loader, device):
    """Collect predictions + all metadata."""
    model.eval()
    records = []
    for batch in tqdm(loader, desc="  collecting", unit="batch"):
        pred = model(
            batch["per_dim_v2"].to(device),
            batch["mask"].to(device),
            batch["global_v2"].to(device),
        ).cpu().numpy()

        labels = batch["label"].numpy()
        dims = batch["mask"].sum(dim=1).numpy().astype(int)
        B = len(labels)

        for i in range(B):
            records.append({
                "pred": float(pred[i]),
                "true": float(labels[i]),
                "error": float(pred[i] - labels[i]),
                "abs_error": float(abs(pred[i] - labels[i])),
                "dim": int(dims[i]),
                "scenario_file": batch["scenario_file"][i],
                "data_source": batch["data_source"][i],
                "scale_factor": float(batch["scale_factor"][i]),
                "center_delta": float(batch["center_delta"][i]),
                "correlation_strength": float(batch["correlation_strength"][i]),
                "upr_type": batch["upr_type"][i],
                "i_aabb": float(batch["i_aabb_arr"][i]),
            })
    return records


def analyze(records, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    preds = np.array([r["pred"] for r in records])
    trues = np.array([r["true"] for r in records])
    errors = np.array([r["error"] for r in records])
    abs_errors = np.array([r["abs_error"] for r in records])
    dims = np.array([r["dim"] for r in records])
    scenarios = np.array([r["scenario_file"] for r in records])
    sources = np.array([r["data_source"] for r in records])
    sfs = np.array([r["scale_factor"] for r in records])
    cds = np.array([r["center_delta"] for r in records])
    css = np.array([r["correlation_strength"] for r in records])
    upr_types = np.array([r["upr_type"] for r in records])
    i_aabbs = np.array([r["i_aabb"] for r in records])

    # ── 1. Per-scenario error breakdown ──
    unique_scenarios = sorted(set(scenarios))
    scenario_stats = []
    for sc in unique_scenarios:
        mask = scenarios == sc
        sc_errors = errors[mask]
        sc_abs = abs_errors[mask]
        scenario_stats.append({
            "scenario": sc,
            "n": int(mask.sum()),
            "dim": int(dims[mask][0]),
            "source": sources[mask][0],
            "mean_error": float(sc_errors.mean()),
            "mean_abs_error": float(sc_abs.mean()),
            "std_error": float(sc_errors.std()),
            "mean_true": float(trues[mask].mean()),
            "mean_pred": float(preds[mask].mean()),
        })
    scenario_stats.sort(key=lambda x: x["mean_abs_error"], reverse=True)

    print(f"\n{'='*90}")
    print("PER-SCENARIO ERROR (top 20 worst)")
    print(f"{'='*90}")
    print(f"{'Scenario':<35s} {'Src':<18s} {'Dim':>3s} {'N':>6s} "
          f"{'MAE':>8s} {'Bias':>8s} {'True':>6s} {'Pred':>6s}")
    print("-" * 90)
    for s in scenario_stats[:20]:
        print(f"{s['scenario']:<35s} {s['source']:<18s} {s['dim']:>3d} "
              f"{s['n']:>6d} {s['mean_abs_error']:>8.4f} "
              f"{s['mean_error']:>+8.4f} {s['mean_true']:>6.3f} "
              f"{s['mean_pred']:>6.3f}")

    # ── 2. Identify band points ──
    threshold = np.percentile(abs_errors, 90)
    high_err = abs_errors > threshold
    over_pred = errors > threshold     # upper band
    under_pred = errors < -threshold   # lower band

    print(f"\n{'='*90}")
    print(f"BAND ANALYSIS (|error| > {threshold:.4f}, top 10%)")
    print(f"{'='*90}")
    print(f"  Upper band (over-predict):  {over_pred.sum():,} points")
    print(f"  Lower band (under-predict): {under_pred.sum():,} points")
    print(f"  On diagonal (good):         {(~high_err).sum():,} points")

    for label, mask in [("UPPER BAND (over-predict)", over_pred),
                        ("LOWER BAND (under-predict)", under_pred)]:
        if not mask.any():
            continue
        print(f"\n  {label}:")
        print(f"    Dim distribution:  "
              f"2D={np.mean(dims[mask]==2):.0%}  "
              f"3D={np.mean(dims[mask]==3):.0%}  "
              f"4D={np.mean(dims[mask]==4):.0%}")
        print(f"    vs overall:        "
              f"2D={np.mean(dims==2):.0%}  "
              f"3D={np.mean(dims==3):.0%}  "
              f"4D={np.mean(dims==4):.0%}")
        print(f"    Data source:       "
              f"v6={np.mean(sources[mask]=='measurements_v6'):.0%}  "
              f"cps={np.mean(sources[mask]=='measurements_cps_v6'):.0%}")
        print(f"    Mean scale_factor: {sfs[mask].mean():.3f} "
              f"(overall: {sfs.mean():.3f})")
        print(f"    Mean center_delta: {cds[mask].mean():.4f} "
              f"(overall: {cds.mean():.4f})")
        print(f"    Mean corr_str:     {css[mask].mean():.3f} "
              f"(overall: {css.mean():.3f})")
        print(f"    Mean true I(θ):    {trues[mask].mean():.3f} "
              f"(overall: {trues.mean():.3f})")
        print(f"    Mean I_AABB:       {i_aabbs[mask].mean():.3f} "
              f"(overall: {i_aabbs.mean():.3f})")
        print(f"    AABB-MC gap:       "
              f"{(i_aabbs[mask] - trues[mask]).mean():.4f} "
              f"(overall: {(i_aabbs - trues).mean():.4f})")

        # Top scenarios in this band
        band_scenarios = defaultdict(int)
        for sc in scenarios[mask]:
            band_scenarios[sc] += 1
        top = sorted(band_scenarios.items(), key=lambda x: -x[1])[:10]
        print(f"    Top scenarios:")
        for sc, cnt in top:
            total = int((scenarios == sc).sum())
            print(f"      {sc:<35s}  {cnt:>5d}/{total} "
                  f"({cnt/total:.0%} of scenario in band)")

    # ── 3. UPR type breakdown ──
    print(f"\n{'='*90}")
    print("ERROR BY UPR TYPE")
    print(f"{'='*90}")
    for utype in sorted(set(upr_types)):
        mask = upr_types == utype
        print(f"  {utype:<25s}  N={mask.sum():>6d}  "
              f"MAE={abs_errors[mask].mean():.4f}  "
              f"bias={errors[mask].mean():>+.4f}")

    # ── 4. Plots ──

    # 4a. Scatter colored by scenario (worst scenarios highlighted)
    worst_scenarios = [s["scenario"] for s in scenario_stats[:5]]
    fig, ax = plt.subplots(figsize=(7, 6))
    other = ~np.isin(scenarios, worst_scenarios)
    ax.scatter(trues[other], preds[other], s=2, alpha=0.1, c="#cccccc",
               label="other", rasterized=True)
    colors_worst = plt.cm.tab10(np.linspace(0, 0.5, len(worst_scenarios)))
    for sc, color in zip(worst_scenarios, colors_worst):
        mask = scenarios == sc
        ax.scatter(trues[mask], preds[mask], s=6, alpha=0.4, color=color,
                   label=sc[:30], rasterized=True)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("True I(θ)")
    ax.set_ylabel("Predicted I(θ)")
    ax.set_title("Worst 5 scenarios highlighted")
    ax.legend(fontsize=6, markerscale=3)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(output_dir / "scatter_worst_scenarios.png", dpi=150)
    plt.close(fig)

    # 4b. Error vs intervention parameters
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, param, vals, pname in [
        (axes[0], sfs, sfs, "scale_factor"),
        (axes[1], cds, cds, "center_delta"),
        (axes[2], css, css, "correlation_strength"),
    ]:
        ax.scatter(vals, errors, s=1, alpha=0.05, rasterized=True)
        ax.axhline(0, color="k", ls="--", lw=0.8)
        ax.set_xlabel(pname)
        ax.set_ylabel("Error (pred - true)")
        ax.set_title(f"Error vs {pname}")
    fig.tight_layout()
    fig.savefig(output_dir / "error_vs_params.png", dpi=150)
    plt.close(fig)

    # 4c. Error vs true I(θ) colored by dim
    fig, ax = plt.subplots(figsize=(7, 4))
    dim_colors = {2: "#4878CF", 3: "#6ACC65", 4: "#D65F5F"}
    for d in sorted(set(dims)):
        mask = dims == d
        ax.scatter(trues[mask], errors[mask], s=2, alpha=0.1,
                   color=dim_colors.get(d, "#333"), label=f"{d}D",
                   rasterized=True)
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("True I(θ)")
    ax.set_ylabel("Error (pred - true)")
    ax.set_title("Error vs True I(θ) by Dimension")
    ax.legend(markerscale=5)
    fig.tight_layout()
    fig.savefig(output_dir / "error_vs_true_by_dim.png", dpi=150)
    plt.close(fig)

    # 4d. Per-scenario MAE bar chart (top 20 worst)
    fig, ax = plt.subplots(figsize=(10, 5))
    top20 = scenario_stats[:20]
    names = [s["scenario"][:28] for s in top20]
    maes = [s["mean_abs_error"] for s in top20]
    colors = ["#D65F5F" if s["dim"] == 4 else
              "#6ACC65" if s["dim"] == 3 else "#4878CF" for s in top20]
    bars = ax.barh(range(len(names)), maes, color=colors, alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("MAE")
    ax.set_title("Top 20 Worst Scenarios by MAE")
    ax.invert_yaxis()
    # Legend for dim colors
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#4878CF", label="2D"),
                       Patch(color="#6ACC65", label="3D"),
                       Patch(color="#D65F5F", label="4D")], fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "worst_scenarios_bar.png", dpi=150)
    plt.close(fig)

    print(f"\n  Diagnostic plots saved to {output_dir}/")

    # Save full stats
    with open(output_dir / "scenario_stats.json", "w") as f:
        json.dump(scenario_stats, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--model", type=str, default="deepsets_v2")
    args = parser.parse_args()

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    # ── Same inductive split as train_compare ──
    all_files = []
    for d in DATA_DIRS:
        all_files.extend(sorted(Path(d).glob("results_scenario_*.json")))

    by_dim = defaultdict(list)
    for jf in all_files:
        by_dim[_get_scenario_dim(jf)].append(jf)

    rng = np.random.default_rng(42)
    train_files, val_files = [], []
    for dim, files in sorted(by_dim.items()):
        files = list(rng.permutation(files))
        n_val = max(1, int(args.val_fraction * len(files)))
        val_files.extend(files[:n_val])
        train_files.extend(files[n_val:])

    print(f"Loading tracked dataset...")
    t0 = time.time()
    train_ds = TrackedDataset(files=train_files, label_key="I_theta",
                              max_samples=args.max_samples)
    val_ds = TrackedDataset(files=val_files, label_key="I_theta")
    print(f"  {len(train_ds):,} train + {len(val_ds):,} val in {time.time()-t0:.1f}s")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=tracked_collate_fn,
                              num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, collate_fn=tracked_collate_fn,
                            num_workers=0)

    # ── Quick train ──
    print(f"\nTraining {args.model} for {args.epochs} epochs...")
    model = create_model(args.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    for epoch in tqdm(range(1, args.epochs + 1), desc="  training", unit="ep"):
        model.train()
        for batch in train_loader:
            pred = model(batch["per_dim_v2"].to(device),
                         batch["mask"].to(device),
                         batch["global_v2"].to(device))
            loss = criterion(pred, batch["label"].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # ── Collect and analyze ──
    print("\nCollecting predictions with metadata...")
    records = collect_all(model, val_loader, device)
    print(f"  {len(records):,} val samples")

    output_dir = ROOT / "results" / "error_diagnosis"
    analyze(records, output_dir)


if __name__ == "__main__":
    main()
