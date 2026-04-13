"""PyTorch Geometric dataset classes for the inconsistency surrogate.

All datasets return ``torch_geometric.data.Data`` objects with structured
node features (``x_center``, ``x_generators``, ``x_generator_mask``) so the
``ZonotopeEncoder`` in model.py can process them without information loss.

The graph structure is **variable-size**: the current CONVIDE scenarios
produce 2-node directed graphs (source -> target), but the architecture
generalises to arbitrary DAGs with N zonotope nodes and E UPR edges.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch_geometric.data import Data, Dataset

from .config import ModelConfig

# Re-export constants for convenience
D_MAX = 4
P_MAX = 12


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pad_center(c: np.ndarray) -> np.ndarray:
    out = np.zeros(D_MAX, dtype=np.float32)
    out[: len(c)] = c
    return out


def _pad_generators(G: np.ndarray) -> np.ndarray:
    """Pad and transpose: (d, p) -> (P_MAX, D_MAX)."""
    d, p = G.shape
    out = np.zeros((P_MAX, D_MAX), dtype=np.float32)
    out[:p, :d] = G.T  # transpose: each row is one generator column
    return out


def _generator_mask(n_gen: int) -> np.ndarray:
    mask = np.zeros(P_MAX, dtype=bool)
    mask[:n_gen] = True
    return mask


def _edge_attr_from_Ff(F: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Flatten F (d,d) and f (d) into a (20,) edge feature vector."""
    F_pad = np.zeros((D_MAX, D_MAX), dtype=np.float32)
    f_pad = np.zeros(D_MAX, dtype=np.float32)
    d = len(f)
    F_pad[:d, :d] = F
    f_pad[:d] = f
    return np.concatenate([F_pad.ravel(), f_pad])  # (20,)


def _scenario_dir_pattern() -> str:
    return "scenario_S*"


def _find_scenario_dirs(data_root: Path) -> Dict[int, Path]:
    """Map scenario index -> directory path."""
    result = {}
    for p in sorted(data_root.glob("scenario_S*")):
        if not p.is_dir():
            continue
        # Extract index from e.g. "scenario_S04_Control_Design_Conflict"
        name = p.name  # "scenario_S04_..."
        idx_str = name.split("_")[1]  # "S04"
        idx = int(idx_str[1:])  # 4
        result[idx] = p
    return result


# ---------------------------------------------------------------------------
# Inconsistency dataset (per-scenario training data)
# ---------------------------------------------------------------------------

class InconsistencyDataset(Dataset):
    """Per-scenario inconsistency training data.

    Builds variable-size graphs from scenario metadata + per-sample data.
    Each CONVIDE sample is a 2-node directed graph (source -> target) but the
    architecture imposes no limit on node count.

    Args:
        data_root: Path to ``data/surrogate/``.
        scenario_indices: List of scenario indices (1-12).
        split: ``'train'``, ``'val'``, or ``'all'``.
        val_fraction: Fraction held out for validation (within each scenario).
        seed: Random seed for the train/val split.
    """

    def __init__(
        self,
        data_root: str | Path,
        scenario_indices: List[int],
        split: str = "train",
        val_fraction: float = 0.15,
        seed: int = 42,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split

        scenario_dirs = _find_scenario_dirs(self.data_root)

        # Accumulate samples across requested scenarios
        self._samples: list[dict] = []

        for s_idx in scenario_indices:
            s_dir = scenario_dirs.get(s_idx)
            if s_dir is None:
                raise FileNotFoundError(
                    f"No data directory for scenario {s_idx} in {self.data_root}"
                )

            with open(s_dir / "meta.json") as f:
                meta = json.load(f)

            npz_path = s_dir / "train_inconsistency.npz"
            if not npz_path.exists():
                raise FileNotFoundError(f"Missing {npz_path}")
            npz = np.load(npz_path)

            n = int(npz["n_samples"])

            # Train/val split
            rng = np.random.default_rng(seed + s_idx)
            perm = rng.permutation(n)
            n_val = int(n * val_fraction)
            if split == "val":
                indices = perm[:n_val]
            elif split == "train":
                indices = perm[n_val:]
            else:  # "all"
                indices = np.arange(n)

            for i in indices:
                self._samples.append({
                    "meta": meta,
                    "source_center": npz["source_center"][i],
                    "source_generators": npz["source_generators"][i],
                    "source_n_generators": int(npz["source_n_generators"][i]),
                    "I_theta_mc": float(npz["I_theta_mc"][i]),
                    "I_theta_aabb": float(npz["I_theta_aabb"][i]),
                    "scenario_idx": s_idx,
                })

    def len(self) -> int:
        return len(self._samples)

    def get(self, idx: int) -> Data:
        s = self._samples[idx]
        meta = s["meta"]
        dim = meta["dim"]

        # --- Node 0: intervened source ---
        src_c = s["source_center"].astype(np.float32)[:D_MAX]
        src_G = s["source_generators"].astype(np.float32)[:D_MAX, :P_MAX]
        src_n_gen = s["source_n_generators"]

        src_c_pad = np.zeros(D_MAX, dtype=np.float32)
        src_c_pad[:D_MAX] = src_c
        src_G_t = np.zeros((P_MAX, D_MAX), dtype=np.float32)
        src_G_t[:src_G.shape[1], :src_G.shape[0]] = src_G.T
        src_mask = _generator_mask(src_n_gen)

        # --- Node 1: target (fixed per scenario) ---
        tgt_c = _pad_center(np.array(meta["target_center"], dtype=np.float32))
        tgt_G = _pad_generators(np.array(meta["target_generators"], dtype=np.float32))
        tgt_mask = _generator_mask(meta["target_n_generators"])

        # --- Stack nodes ---
        x_center = np.stack([src_c_pad, tgt_c])          # (2, D_MAX)
        x_generators = np.stack([src_G_t, tgt_G])         # (2, P_MAX, D_MAX)
        x_generator_mask = np.stack([src_mask, tgt_mask])  # (2, P_MAX)

        # --- Edge: source -> target (UPR) ---
        edge_index = np.array([[0], [1]], dtype=np.int64)  # (2, 1)
        edge_attr = _edge_attr_from_Ff(
            np.array(meta["F"]), np.array(meta["f"]),
        ).reshape(1, -1)  # (1, 20)

        return Data(
            x_center=torch.from_numpy(x_center),
            x_generators=torch.from_numpy(x_generators),
            x_generator_mask=torch.from_numpy(x_generator_mask),
            edge_index=torch.from_numpy(edge_index),
            edge_attr=torch.from_numpy(edge_attr),
            num_nodes=2,
            y=torch.tensor([s["I_theta_mc"]], dtype=torch.float32),
            y_aabb=torch.tensor([s["I_theta_aabb"]], dtype=torch.float32),
            scenario_idx=torch.tensor([s["scenario_idx"]], dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Pretraining: volume
# ---------------------------------------------------------------------------

class VolumeDataset(Dataset):
    """Random zonotopes -> log1p(interval hull volume).

    Each sample is a 1-node graph with no edges.
    """

    def __init__(
        self,
        npz_path: str | Path,
        split: str = "train",
        val_fraction: float = 0.1,
        seed: int = 42,
    ):
        super().__init__()
        npz = np.load(npz_path)
        n = len(npz["volume"])

        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        n_val = int(n * val_fraction)
        if split == "val":
            idx = perm[:n_val]
        elif split == "train":
            idx = perm[n_val:]
        else:
            idx = np.arange(n)

        self.centers = npz["centers"][idx].astype(np.float32)
        self.generators = npz["generators"][idx].astype(np.float32)
        self.dims = npz["dim"][idx]
        self.n_gens = npz["n_generators"][idx]
        self.volumes = npz["volume"][idx].astype(np.float32)

    def len(self) -> int:
        return len(self.volumes)

    def get(self, idx: int) -> Data:
        c = self.centers[idx][:D_MAX]
        G = self.generators[idx][:D_MAX, :P_MAX]  # (D_MAX, P_MAX) from npz
        n_gen = int(self.n_gens[idx])

        # Transpose generators: (D_MAX, P_MAX) -> (P_MAX, D_MAX)
        G_t = np.zeros((P_MAX, D_MAX), dtype=np.float32)
        G_t[:G.shape[1], :G.shape[0]] = G.T

        return Data(
            x_center=torch.from_numpy(c).unsqueeze(0),          # (1, D_MAX)
            x_generators=torch.from_numpy(G_t).unsqueeze(0),    # (1, P_MAX, D_MAX)
            x_generator_mask=torch.from_numpy(
                _generator_mask(n_gen),
            ).unsqueeze(0),                                      # (1, P_MAX)
            edge_index=torch.zeros(2, 0, dtype=torch.long),     # no edges
            edge_attr=torch.zeros(0, 20, dtype=torch.float32),
            num_nodes=1,
            y=torch.tensor([np.log1p(self.volumes[idx])], dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# Pretraining: containment
# ---------------------------------------------------------------------------

class ContainmentDataset(Dataset):
    """Random zonotopes + test points -> binary containment labels.

    Each sample is a 1-node graph with an additional ``points`` tensor.
    """

    def __init__(
        self,
        npz_path: str | Path,
        split: str = "train",
        val_fraction: float = 0.1,
        seed: int = 42,
    ):
        super().__init__()
        npz = np.load(npz_path)
        n = len(npz["dim"])

        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        n_val = int(n * val_fraction)
        if split == "val":
            idx = perm[:n_val]
        elif split == "train":
            idx = perm[n_val:]
        else:
            idx = np.arange(n)

        self.centers = npz["centers"][idx].astype(np.float32)
        self.generators = npz["generators"][idx].astype(np.float32)
        self.dims = npz["dim"][idx]
        self.n_gens = npz["n_generators"][idx]
        self.points = npz["points"][idx].astype(np.float32)
        self.labels = npz["labels"][idx]

    def len(self) -> int:
        return len(self.dims)

    def get(self, idx: int) -> Data:
        c = self.centers[idx][:D_MAX]
        G = self.generators[idx][:D_MAX, :P_MAX]
        n_gen = int(self.n_gens[idx])

        G_t = np.zeros((P_MAX, D_MAX), dtype=np.float32)
        G_t[:G.shape[1], :G.shape[0]] = G.T

        return Data(
            x_center=torch.from_numpy(c).unsqueeze(0),
            x_generators=torch.from_numpy(G_t).unsqueeze(0),
            x_generator_mask=torch.from_numpy(
                _generator_mask(n_gen),
            ).unsqueeze(0),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            edge_attr=torch.zeros(0, 20, dtype=torch.float32),
            num_nodes=1,
            points=torch.from_numpy(self.points[idx][:, :D_MAX]).unsqueeze(0),  # (1, K, D_MAX)
            labels=torch.from_numpy(self.labels[idx].astype(np.float32)).unsqueeze(0),  # (1, K)
        )


# ---------------------------------------------------------------------------
# Pretraining: pairwise AABB Jaccard
# ---------------------------------------------------------------------------

class PairwiseAABBDataset(Dataset):
    """Random zonotope pairs -> AABB Jaccard index.

    Each sample is a 2-node bidirectional graph (no UPR — identity edges).
    """

    def __init__(
        self,
        npz_path: str | Path,
        split: str = "train",
        val_fraction: float = 0.1,
        seed: int = 42,
    ):
        super().__init__()
        npz = np.load(npz_path)
        n = len(npz["aabb_jaccard"])

        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        n_val = int(n * val_fraction)
        if split == "val":
            idx = perm[:n_val]
        elif split == "train":
            idx = perm[n_val:]
        else:
            idx = np.arange(n)

        self.c1 = npz["centers_1"][idx].astype(np.float32)
        self.g1 = npz["generators_1"][idx].astype(np.float32)
        self.c2 = npz["centers_2"][idx].astype(np.float32)
        self.g2 = npz["generators_2"][idx].astype(np.float32)
        self.dims = npz["dim"][idx]
        self.ng1 = npz["n_generators_1"][idx]
        self.ng2 = npz["n_generators_2"][idx]
        self.jaccard = npz["aabb_jaccard"][idx].astype(np.float32)

    def len(self) -> int:
        return len(self.jaccard)

    def get(self, idx: int) -> Data:
        def _encode(c, G, ng):
            c_pad = c[:D_MAX]
            G_raw = G[:D_MAX, :P_MAX]
            G_t = np.zeros((P_MAX, D_MAX), dtype=np.float32)
            G_t[:G_raw.shape[1], :G_raw.shape[0]] = G_raw.T
            return c_pad, G_t, _generator_mask(int(ng))

        c1, g1, m1 = _encode(self.c1[idx], self.g1[idx], self.ng1[idx])
        c2, g2, m2 = _encode(self.c2[idx], self.g2[idx], self.ng2[idx])

        # Identity edge features (no UPR — just linking the two zonotopes)
        identity_edge = np.zeros(20, dtype=np.float32)
        identity_edge[:D_MAX * D_MAX] = np.eye(D_MAX, dtype=np.float32).ravel()

        # Bidirectional edges
        edge_index = np.array([[0, 1], [1, 0]], dtype=np.int64)
        edge_attr = np.stack([identity_edge, identity_edge])

        return Data(
            x_center=torch.from_numpy(np.stack([c1, c2])),
            x_generators=torch.from_numpy(np.stack([g1, g2])),
            x_generator_mask=torch.from_numpy(np.stack([m1, m2])),
            edge_index=torch.from_numpy(edge_index),
            edge_attr=torch.from_numpy(edge_attr),
            num_nodes=2,
            y=torch.tensor([self.jaccard[idx]], dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# Pretraining: affine map
# ---------------------------------------------------------------------------

class AffineMapDataset(Dataset):
    """Source zonotope + (F, f) -> reconstruct target zonotope.

    Each sample is a 2-node directed graph. Node 0 is the source (with real
    features); node 1 is initialised to zeros (the network must reconstruct
    target geometry from the source + edge information).
    """

    def __init__(
        self,
        npz_path: str | Path,
        split: str = "train",
        val_fraction: float = 0.1,
        seed: int = 42,
    ):
        super().__init__()
        npz = np.load(npz_path)
        n = len(npz["dim"])

        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        n_val = int(n * val_fraction)
        if split == "val":
            idx = perm[:n_val]
        elif split == "train":
            idx = perm[n_val:]
        else:
            idx = np.arange(n)

        self.src_c = npz["source_center"][idx].astype(np.float32)
        self.src_G = npz["source_generators"][idx].astype(np.float32)
        self.tgt_c = npz["target_center"][idx].astype(np.float32)
        self.tgt_G = npz["target_generators"][idx].astype(np.float32)
        self.F = npz["F"][idx].astype(np.float32)
        self.f = npz["f"][idx].astype(np.float32)
        self.dims = npz["dim"][idx]
        self.n_gens = npz["n_generators"][idx]

    def len(self) -> int:
        return len(self.dims)

    def get(self, idx: int) -> Data:
        n_gen = int(self.n_gens[idx])

        # Source node
        sc = self.src_c[idx][:D_MAX]
        sG = self.src_G[idx][:D_MAX, :P_MAX]
        sG_t = np.zeros((P_MAX, D_MAX), dtype=np.float32)
        sG_t[:sG.shape[1], :sG.shape[0]] = sG.T

        # Target node (zero-initialised — network must reconstruct)
        zero_c = np.zeros(D_MAX, dtype=np.float32)
        zero_G = np.zeros((P_MAX, D_MAX), dtype=np.float32)

        # Edge features from F and f
        ea = _edge_attr_from_Ff(self.F[idx], self.f[idx]).reshape(1, -1)

        # Reconstruction targets
        tc = self.tgt_c[idx][:D_MAX]
        tG = self.tgt_G[idx][:D_MAX, :P_MAX]
        tG_t = np.zeros((P_MAX, D_MAX), dtype=np.float32)
        tG_t[:tG.shape[1], :tG.shape[0]] = tG.T

        return Data(
            x_center=torch.from_numpy(np.stack([sc, zero_c])),
            x_generators=torch.from_numpy(np.stack([sG_t, zero_G])),
            x_generator_mask=torch.from_numpy(np.stack([
                _generator_mask(n_gen),
                _generator_mask(0),
            ])),
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            edge_attr=torch.from_numpy(ea),
            num_nodes=2,
            # Graph-level targets need leading dim so PyG batches to (B, ...)
            y_center=torch.from_numpy(tc).unsqueeze(0),         # (1, D_MAX)
            y_generators=torch.from_numpy(tG_t).unsqueeze(0),   # (1, P_MAX, D_MAX)
            y_generator_mask=torch.from_numpy(
                _generator_mask(n_gen),
            ).unsqueeze(0),                                      # (1, P_MAX)
        )
