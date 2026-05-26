"""
Evaluate the surrogate against AABB, MC, and MFMC on held-out data.

Reports MAE and Spearman correlation for each method.
Also reports median inference time per sample.

Usage
-----
    python -m surrogate.evaluate
    python -m surrogate.evaluate --max_samples 50000
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from .dataset import ZonotopeDataset, collate_fn
from .model import DeepSetsZonotope

ROOT      = Path(__file__).resolve().parents[1]
DATA_DIRS = [
    ROOT / "data" / "measurements_v6",
    ROOT / "data" / "measurements_cps_v6",
]
SAVE_PATH = ROOT / "surrogate" / "model.pt"


def evaluate(args):
    print("Loading data...")
    ds = ZonotopeDataset(DATA_DIRS, label_key=args.label_key,
                         max_samples=args.max_samples)
    print(f"  {len(ds):,} samples\n")

    loader = DataLoader(ds, batch_size=2048, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(SAVE_PATH, map_location=device)
    saved_args = ckpt["args"]
    model = DeepSetsZonotope(
        phi_hidden=saved_args["phi_hidden"],
        rho_hidden=saved_args["rho_hidden"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    preds, labels = [], []
    i_thetas, i_aabbs, i_mfmcs = [], [], []
    t_total = 0.0

    with torch.no_grad():
        for batch in loader:
            pd = batch["per_dim"].to(device)
            mk = batch["mask"].to(device)
            gf = batch["global_feats"].to(device)
            lb = batch["label"].numpy()

            t0 = time.perf_counter()
            out = model(pd, mk, gf).cpu().numpy()
            t_total += time.perf_counter() - t0

            preds.append(out)
            labels.append(lb)
            i_thetas.extend(batch["i_theta"])
            i_aabbs.extend(batch["i_aabb"])
            i_mfmcs.extend(batch["i_mfmc"])

    preds    = np.concatenate(preds)
    labels   = np.concatenate(labels)
    n        = len(labels)
    i_thetas = np.array(i_thetas)
    i_aabbs  = np.array(i_aabbs)
    i_mfmcs  = np.array(i_mfmcs)

    def metrics(y_hat, y_true, name):
        valid = np.isfinite(y_hat) & np.isfinite(y_true)
        y_h, y_t = y_hat[valid], y_true[valid]
        mae  = np.mean(np.abs(y_h - y_t))
        rho  = spearmanr(y_h, y_t).statistic
        ss_res = np.sum((y_t - y_h) ** 2)
        ss_tot = np.sum((y_t - y_t.mean()) ** 2)
        r2   = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        print(f"  {name:<12}  MAE={mae:.4f}  Spearman={rho:.4f}  R²={r2:.4f}")

    print(f"Results (n={n:,}, ground truth=I_theta):")
    print(f"  {'Method':<12}  {'MAE':>8}  {'Spearman':>10}  {'R²':>8}")
    print("  " + "-" * 46)
    metrics(preds,    i_thetas, "Surrogate")
    metrics(i_aabbs,  i_thetas, "AABB")
    metrics(i_mfmcs,  i_thetas, "MFMC")
    metrics(i_thetas, i_thetas, "MC (I_theta)")

    us_per_sample = t_total / n * 1e6
    print(f"\nSurrogate inference: {us_per_sample:.2f} µs/sample")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--label_key",   type=str, default="I_theta")
    p.add_argument("--max_samples", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
