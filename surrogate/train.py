"""
Train the DeepSets zonotope overlap surrogate.

Usage
-----
    python -m surrogate.train                          # defaults
    python -m surrogate.train --epochs 30 --lr 1e-3
    python -m surrogate.train --max_samples 100000     # quick run
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import ZonotopeDataset, collate_fn
from .model import DeepSetsZonotope

ROOT = Path(__file__).resolve().parents[1]
DATA_DIRS = [
    ROOT / "data" / "measurements_v6",
    ROOT / "data" / "measurements_cps_v6",
]
SAVE_PATH = ROOT / "surrogate" / "model.pt"


def _get_scenario_dim(json_file):
    """Read only the first experiment to determine zonotope dimension."""
    import json
    with open(json_file) as f:
        data = json.load(f)
    c1 = data["experiments"][0]["post_state"]["uncertainty"]["source_center"]
    return len(c1)


def train(args):
    # ── inductive split: stratified by dimension ───────────────────────
    # dim=2: 76 scenarios, dim=3: 4, dim=4: 4
    # A flat random split risks leaving entire dimensions out of val.
    all_files = []
    for d in DATA_DIRS:
        all_files.extend(sorted(Path(d).glob("results_scenario_*.json")))

    # Group files by dimension
    from collections import defaultdict
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
        print(f"  dim={dim}: {len(files)-n_val} train, {n_val} val")

    print(f"Inductive split: {len(train_files)} train scenarios, "
          f"{len(val_files)} val scenarios")

    print("Loading data...")
    t0 = time.time()
    train_ds = ZonotopeDataset(files=train_files, label_key=args.label_key,
                               max_samples=args.max_samples)
    val_ds   = ZonotopeDataset(files=val_files,   label_key=args.label_key)
    print(f"  {len(train_ds)+len(val_ds):,} samples loaded in {time.time()-t0:.1f}s")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn,
                              num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn,
                              num_workers=0)

    n_train = len(train_ds)
    n_val   = len(val_ds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device={device}, train={n_train:,}, val={n_val:,}\n")

    model = DeepSetsZonotope(phi_hidden=args.phi_hidden,
                             rho_hidden=args.rho_hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params} parameters\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )
    criterion = nn.MSELoss()

    best_val, best_epoch = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        # ── train ──────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            pd = batch["per_dim"].to(device)
            mk = batch["mask"].to(device)
            gf = batch["global_feats"].to(device)
            lb = batch["label"].to(device)

            pred = model(pd, mk, gf)
            loss = criterion(pred, lb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(lb)

        train_loss /= n_train

        # ── validate ───────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                pd = batch["per_dim"].to(device)
                mk = batch["mask"].to(device)
                gf = batch["global_feats"].to(device)
                lb = batch["label"].to(device)
                pred = model(pd, mk, gf)
                val_loss += criterion(pred, lb).item() * len(lb)
        val_loss /= n_val

        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            torch.save({"model_state": model.state_dict(),
                        "args": vars(args)}, SAVE_PATH)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{args.epochs}  "
                  f"train={train_loss:.5f}  val={val_loss:.5f}  "
                  f"best={best_val:.5f} (ep {best_epoch})")

    print(f"\nBest model saved to {SAVE_PATH}  (val MSE={best_val:.5f})")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",      type=int,   default=20)
    p.add_argument("--batch_size",  type=int,   default=2048)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--phi_hidden",  type=int,   default=16)
    p.add_argument("--rho_hidden",  type=int,   default=32)
    p.add_argument("--label_key",    type=str,   default="I_theta")
    p.add_argument("--val_fraction", type=float, default=0.15,
                   help="Fraction of scenarios held out for inductive validation")
    p.add_argument("--max_samples",  type=int,   default=None,
                   help="Cap train dataset size for quick experiments")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
