"""
Unified dataset for all model variants.

Provides both:
  - Enhanced per-dim features (5 features) + global (3) for DeepSetsV2 / FlatMLP / SetTransformer
  - Raw propagated zonotope geometry for SiameseDeepSets
  - Original v1 features (4 per-dim + 10 global) for baseline comparison

Reads the same results_scenario_*.json files as the original dataset.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

MAX_DIM = 4
P_MAX = 12
EPS = 1e-10

N_DIM_FEAT_V1 = 4
N_GLOBAL_V1 = 10
N_DIM_FEAT_V2 = 5
N_GLOBAL_V2 = 5

UPR_TYPES = [
    "identity", "identity_bidir", "parametric", "structural",
    "guarded", "constraint_based", "disambiguation", "unknown",
]
_UPR_INDEX = {t: i for i, t in enumerate(UPR_TYPES)}


def _upr_onehot(upr_type_str: str) -> np.ndarray:
    oh = np.zeros(len(UPR_TYPES), dtype=np.float32)
    first = (upr_type_str or "unknown").split("+")[0].strip().lower()
    oh[_UPR_INDEX.get(first, _UPR_INDEX["unknown"])] = 1.0
    return oh


def _extract_upr(exp, d):
    """Extract UPR scales, offsets, and one-hot from an experiment record."""
    cr = exp.get("consistency_relations")
    if cr and isinstance(cr, list) and len(cr) >= d:
        scales = np.array([r["mapping"].get("scale", 1.0) for r in cr[:d]])
        offsets = np.array([r["mapping"].get("offset", 0.0) for r in cr[:d]])
        dominant = "unknown"
        for r in cr[:d]:
            t = r.get("upr_type", "unknown")
            if t != "unknown":
                dominant = t
                break
        return scales, offsets, _upr_onehot(dominant)
    return np.ones(d), np.zeros(d), _upr_onehot("unknown")


def _compute_features(c1, G1, c2, G2, scales, offsets, upr_onehot):
    """Compute both v1 and v2 feature sets + raw propagated geometry.

    Returns dict with all data needed by every model variant.
    """
    d = len(c1)

    # ── Propagate source through UPR ──
    c1_prop = scales * c1 + offsets
    G1_prop = scales[:, None] * G1                   # (d, p1) — propagated generators
    r1_prop = np.linalg.norm(G1_prop, axis=1)        # (d,)
    r2_raw = np.linalg.norm(G2, axis=1)              # (d,)
    sigma = r1_prop + r2_raw + EPS

    # ── Per-dim features (shared computation) ──
    delta_c = (c1_prop - c2) / sigma
    r1_n = r1_prop / sigma
    r2_n = r2_raw / sigma

    # Cosine: pad shorter row with zeros for unequal generator counts
    p1, p2 = G1.shape[1], G2.shape[1]
    p_max_local = max(p1, p2)
    cos = np.zeros(d, dtype=np.float64)
    for i in range(d):
        g1_row = np.zeros(p_max_local)
        g2_row = np.zeros(p_max_local)
        g1_row[:p1] = scales[i] * G1[i]
        g2_row[:p2] = G2[i]
        n1 = np.linalg.norm(g1_row)
        n2 = np.linalg.norm(g2_row)
        if n1 > EPS and n2 > EPS:
            cos[i] = np.dot(g1_row, g2_row) / (n1 * n2)

    # Width ratio (v2 only)
    width_ratio = r1_prop / (r2_raw + EPS)

    # ── V1 per-dim (4 features): delta_c, r1_n, r2_n, cos ──
    per_dim_v1 = np.stack([delta_c, r1_n, r2_n, cos], axis=1).astype(np.float32)

    # ── V2 per-dim (5 features): + width_ratio ──
    per_dim_v2 = np.stack([delta_c, r1_n, r2_n, cos, width_ratio],
                          axis=1).astype(np.float32)

    # ── Global features ──
    log_vol = np.log(np.prod(r1_prop + EPS) / (np.prod(r2_raw + EPS) + EPS) + EPS)
    offset = c1_prop - c2
    off_mag = np.linalg.norm(offset)
    norm_dist = off_mag / (np.mean(sigma) + EPS)

    # ── Support-function features (offset alignment with shape) ──
    # Captures WHY inconsistency is high: not just center distance, but how
    # the offset relates to the zonotope extents along that direction.
    # Distinguishes "close centers but target thin along offset → high I"
    # from "close centers, well overlapped → low I".
    if off_mag > EPS:
        off_hat = offset / off_mag
        tgt_support = float(np.sum(np.abs(G2.T @ off_hat)))
        src_support = float(np.sum(np.abs(G1_prop.T @ off_hat)))
        sep_ratio = off_mag / (src_support + tgt_support + EPS)
        off_over_tgt = min(off_mag / (tgt_support + EPS), 10.0)
    else:
        sep_ratio = 0.0
        off_over_tgt = 0.0

    global_v1 = np.concatenate([
        np.array([log_vol, norm_dist], dtype=np.float32),
        upr_onehot,
    ])  # (10,)

    global_v2 = np.array([log_vol, norm_dist, float(d), sep_ratio, off_over_tgt],
                         dtype=np.float32)  # (5,)

    # ── Raw geometry for Siamese (propagated source, raw target) ──
    # Normalize by a single scalar so magnitudes are comparable across scenarios
    norm_factor = float(np.max(sigma)) + EPS

    return {
        "per_dim_v1": per_dim_v1,       # (d, 4)
        "global_v1": global_v1,          # (10,)
        "per_dim_v2": per_dim_v2,       # (d, 5)
        "global_v2": global_v2,          # (5,)
        "src_center_prop": (c1_prop / norm_factor).astype(np.float32),  # (d,)
        "src_gens_prop": (G1_prop / norm_factor).astype(np.float32),    # (d, p1)
        "src_n_gen": G1.shape[1],
        "tgt_center": (c2 / norm_factor).astype(np.float32),            # (d,)
        "tgt_gens": (G2 / norm_factor).astype(np.float32),              # (d, p2)
        "tgt_n_gen": G2.shape[1],
        "dim": d,
    }


class ZonotopeDatasetV2(Dataset):
    """Loads scenario JSON files and extracts features for all model variants.

    Parameters
    ----------
    data_dirs : list of str/Path
        Directories containing results_scenario_*.json.
    label_key : str
        Key in post_state.inconsistency to use as target.
    max_samples : int or None
        Cap total samples.
    files : list of Path or None
        Explicit file list (overrides data_dirs). For inductive splits.
    """

    def __init__(self, data_dirs=None, label_key="I_theta",
                 max_samples=None, files=None):
        self.samples = []
        if files is not None:
            file_iter = files
        else:
            file_iter = []
            for d in (data_dirs or []):
                file_iter.extend(sorted(Path(d).glob("results_scenario_*.json")))

        for jf in file_iter:
            self._load(jf, label_key)
            if max_samples and len(self.samples) >= max_samples:
                break

        if max_samples:
            self.samples = self.samples[:max_samples]

    def _load(self, json_file, label_key):
        with open(json_file) as f:
            data = json.load(f)

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
                if d > MAX_DIM:
                    continue

                scales, offsets, upr_onehot = _extract_upr(exp, d)

                feats = _compute_features(c1, G1, c2, G2, scales, offsets, upr_onehot)

                # Competitor scores
                i_theta = float(inc.get("I_theta", float("nan")))
                i_aabb = float(inc.get("jaccard_inconsistency",
                               1.0 - inc.get("jaccard_index", float("nan"))))
                i_mfmc = 1.0 - float(inc.get("I_MF_adaptive_sobol", float("nan")))

                sample = {
                    **feats,
                    "label": np.float32(label),
                    "i_theta": i_theta,
                    "i_aabb": i_aabb,
                    "i_mfmc": i_mfmc,
                }
                self.samples.append(sample)
            except (KeyError, ValueError, TypeError):
                continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _pad_center(c, d_max=MAX_DIM):
    out = np.zeros(d_max, dtype=np.float32)
    out[:len(c)] = c
    return out


def _pad_generators_colmajor(G, d_max=MAX_DIM, p_max=P_MAX):
    """Pad generator matrix to (P_MAX, D_MAX) in column-major layout.

    Input G is (d, p).  Output is (P_MAX, D_MAX) where each row is one
    generator column padded to D_MAX.
    """
    d, p = G.shape
    out = np.zeros((p_max, d_max), dtype=np.float32)
    out[:p, :d] = G.T
    return out


def _gen_mask(n_gen, p_max=P_MAX):
    mask = np.zeros(p_max, dtype=np.float32)
    mask[:n_gen] = 1.0
    return mask


def collate_fn(batch):
    """Collate batch into tensors for all model variants."""
    B = len(batch)

    # ── V1 features (original DeepSets baseline) ──
    pd_v1 = np.zeros((B, MAX_DIM, N_DIM_FEAT_V1), dtype=np.float32)
    gl_v1 = np.zeros((B, N_GLOBAL_V1), dtype=np.float32)

    # ── V2 features (DeepSetsV2, FlatMLP, SetTransformer) ──
    pd_v2 = np.zeros((B, MAX_DIM, N_DIM_FEAT_V2), dtype=np.float32)
    gl_v2 = np.zeros((B, N_GLOBAL_V2), dtype=np.float32)

    # ── Dimension mask ──
    masks = np.zeros((B, MAX_DIM), dtype=np.float32)

    # ── Siamese raw geometry ──
    src_c = np.zeros((B, MAX_DIM), dtype=np.float32)
    src_g = np.zeros((B, P_MAX, MAX_DIM), dtype=np.float32)
    src_gm = np.zeros((B, P_MAX), dtype=np.float32)
    tgt_c = np.zeros((B, MAX_DIM), dtype=np.float32)
    tgt_g = np.zeros((B, P_MAX, MAX_DIM), dtype=np.float32)
    tgt_gm = np.zeros((B, P_MAX), dtype=np.float32)

    # ── Labels ──
    labels = np.zeros(B, dtype=np.float32)

    for i, s in enumerate(batch):
        d = s["dim"]

        pd_v1[i, :d] = s["per_dim_v1"]
        gl_v1[i] = s["global_v1"]

        pd_v2[i, :d] = s["per_dim_v2"]
        gl_v2[i] = s["global_v2"]

        masks[i, :d] = 1.0

        src_c[i] = _pad_center(s["src_center_prop"])
        src_g[i] = _pad_generators_colmajor(s["src_gens_prop"])
        src_gm[i] = _gen_mask(s["src_n_gen"])

        tgt_c[i] = _pad_center(s["tgt_center"])
        tgt_g[i] = _pad_generators_colmajor(s["tgt_gens"])
        tgt_gm[i] = _gen_mask(s["tgt_n_gen"])

        labels[i] = s["label"]

    return {
        "per_dim_v1": torch.from_numpy(pd_v1),
        "global_v1": torch.from_numpy(gl_v1),
        "per_dim_v2": torch.from_numpy(pd_v2),
        "global_v2": torch.from_numpy(gl_v2),
        "mask": torch.from_numpy(masks),
        "src_center": torch.from_numpy(src_c),
        "src_generators": torch.from_numpy(src_g),
        "src_gen_mask": torch.from_numpy(src_gm),
        "tgt_center": torch.from_numpy(tgt_c),
        "tgt_generators": torch.from_numpy(tgt_g),
        "tgt_gen_mask": torch.from_numpy(tgt_gm),
        "label": torch.from_numpy(labels),
        "i_theta": [s["i_theta"] for s in batch],
        "i_aabb": [s["i_aabb"] for s in batch],
        "i_mfmc": [s["i_mfmc"] for s in batch],
    }
