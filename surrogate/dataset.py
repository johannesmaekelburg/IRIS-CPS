"""
Data loading for the zonotope overlap surrogate.

Reads results_scenario_*.json files produced by the MATLAB two-step pipeline.
Each experiment record contains post-intervention zonotope geometry, the UPR
(Uncertainty Propagation Relation) connecting source and target, and the
pre-computed inconsistency scores used as training labels.

Feature extraction
------------------
The UPR maps source zonotope Z1 to the target space via y = scale_i * x_i + offset_i
per dimension.  All distance / spread features are computed in the *propagated*
source frame so that the surrogate sees a physically meaningful comparison.

For each dimension i of the (propagated) source and the target zonotope (c2, G2):
    c1_prop_i  = scale_i * c1_i + offset_i
    r1_prop_i  = |scale_i| * ||G1[i,:]||    (scaled generator row-norm)
    sigma_i    = r1_prop_i + ||G2[i,:]||    (joint scale, avoids div-by-zero)
    delta_c_i  = (c1_prop_i - c2_i) / sigma_i
    r1_i       = r1_prop_i / sigma_i
    r2_i       = ||G2[i,:]|| / sigma_i
    cos_i      = (scale_i * G1[i,:]) . G2[i,:] / (r1_prop_i * ||G2[i,:]||)

Global (10 = 2 geometric + 8 UPR-type one-hot):
    log_vol_ratio     = log(prod(r1_prop) / prod(r2_raw))
    norm_center_dist  = ||c1_prop - c2|| / mean(sigma)
    upr_type_onehot   = one-hot over 8 UPR classes (see UPR_TYPES)

UPR type classes (8)
--------------------
    0 identity
    1 identity_bidir
    2 parametric
    3 structural
    4 guarded
    5 constraint_based
    6 disambiguation
    7 unknown          (CONVIDE scenarios or missing consistency_relations)
"""

import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset

from .model import MAX_DIM, N_DIM_FEAT, N_GLOBAL

EPS = 1e-10

# Canonical UPR type ordering — must stay in sync with N_GLOBAL = 2 + len(UPR_TYPES)
UPR_TYPES = [
    "identity",
    "identity_bidir",
    "parametric",
    "structural",
    "guarded",
    "constraint_based",
    "disambiguation",
    "unknown",
]
N_UPR_TYPES = len(UPR_TYPES)   # 8
_UPR_INDEX  = {t: i for i, t in enumerate(UPR_TYPES)}

assert N_GLOBAL == 2 + N_UPR_TYPES, (
    f"N_GLOBAL={N_GLOBAL} but 2 geometric + {N_UPR_TYPES} UPR classes = {2+N_UPR_TYPES}"
)


def _upr_type_to_idx(upr_type_str: str) -> int:
    """Map a (possibly compound) UPR type string to a class index.

    Compound types like 'parametric+structural' use only the first token.
    Unknown tokens fall back to the 'unknown' class.
    """
    if not upr_type_str:
        return _UPR_INDEX["unknown"]
    first = upr_type_str.split("+")[0].strip().lower()
    return _UPR_INDEX.get(first, _UPR_INDEX["unknown"])


def _upr_onehot(upr_type_str: str) -> np.ndarray:
    oh = np.zeros(N_UPR_TYPES, dtype=np.float32)
    oh[_upr_type_to_idx(upr_type_str)] = 1.0
    return oh


def _make_features(c1, G1, c2, G2,
                   upr_scales=None, upr_offsets=None, upr_type_onehot=None):
    """Return (per_dim, global_feats) as float32 numpy arrays.

    Parameters
    ----------
    c1, G1 : source center (d,) and generator matrix (d, ng1)
    c2, G2 : target center (d,) and generator matrix (d, ng2)
    upr_scales  : per-dimension scale  (d,) or None → defaults to ones
    upr_offsets : per-dimension offset (d,) or None → defaults to zeros
    upr_type_onehot : (N_UPR_TYPES,) float32 one-hot or None → unknown class
    """
    d = len(c1)

    if upr_scales is not None:
        scales  = np.asarray(upr_scales[:d],  dtype=np.float64)
        offsets = np.asarray(upr_offsets[:d], dtype=np.float64)
    else:
        scales  = np.ones(d,  dtype=np.float64)
        offsets = np.zeros(d, dtype=np.float64)

    # Propagate source through UPR
    c1_prop = scales * c1 + offsets
    # Propagated generator row-norms: |scale_i| * ||G1[i,:]||
    r1_prop = np.abs(scales) * np.linalg.norm(G1, axis=1)  # (d,)
    r2_raw  = np.linalg.norm(G2, axis=1)                   # (d,)
    sigma   = r1_prop + r2_raw + EPS

    delta_c = (c1_prop - c2) / sigma
    r1_n    = r1_prop / sigma
    r2_n    = r2_raw  / sigma

    cos = np.zeros(d, dtype=np.float64)
    for i in range(d):
        g1_prop_i = scales[i] * G1[i]          # propagated generator row
        n1 = np.linalg.norm(g1_prop_i)
        n2 = np.linalg.norm(G2[i])
        if n1 > EPS and n2 > EPS:
            cos[i] = np.dot(g1_prop_i, G2[i]) / (n1 * n2)

    per_dim = np.stack([delta_c, r1_n, r2_n, cos], axis=1).astype(np.float32)  # (d, 4)

    # Global geometric features (in propagated frame)
    log_vol   = np.log(np.prod(r1_prop + EPS) / (np.prod(r2_raw + EPS) + EPS) + EPS)
    norm_dist = np.linalg.norm(c1_prop - c2) / (np.mean(sigma) + EPS)

    if upr_type_onehot is None:
        upr_type_onehot = _upr_onehot("unknown")

    global_feats = np.concatenate(
        [np.array([log_vol, norm_dist], dtype=np.float32), upr_type_onehot]
    )  # (N_GLOBAL,) = (10,)

    return per_dim, global_feats


class ZonotopeDataset(Dataset):
    """
    Parameters
    ----------
    data_dirs : list of str/Path
        Directories containing results_scenario_*.json files.
        Ignored when `files` is provided.
    label_key : str
        Key inside post_state.inconsistency to use as the regression target.
    max_samples : int or None
        Cap total samples loaded (useful for quick tests).
    files : list of Path or None
        Explicit list of JSON files to load. When set, data_dirs is ignored.
        Use this for inductive splits (pass train files / val files separately).
    """

    def __init__(self, data_dirs=None, label_key="I_MF_sobol",
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

                c1 = np.array(unc["source_center"],     dtype=np.float64)
                G1 = np.array(unc["source_generators"], dtype=np.float64)
                c2 = np.array(unc["target_center"],     dtype=np.float64)
                G2 = np.array(unc["target_generators"], dtype=np.float64)

                if G1.ndim == 1:
                    G1 = G1[:, None]
                if G2.ndim == 1:
                    G2 = G2[:, None]

                d = len(c1)
                if d > MAX_DIM:
                    continue

                # --- UPR extraction -------------------------------------------
                upr_scales  = None
                upr_offsets = None
                upr_onehot  = None

                cr = exp.get("consistency_relations")
                if cr and isinstance(cr, list) and len(cr) >= d:
                    upr_scales  = []
                    upr_offsets = []
                    dominant_type = "unknown"   # first non-identity type wins
                    for rel in cr[:d]:
                        m = rel.get("mapping", {})
                        upr_scales.append(float(m.get("scale",  1.0)))
                        upr_offsets.append(float(m.get("offset", 0.0)))
                        if dominant_type == "unknown":
                            t = rel.get("upr_type", "unknown")
                            dominant_type = t
                    upr_onehot = _upr_onehot(dominant_type)
                # else: CONVIDE or missing → all defaults (scale=1, offset=0, unknown)
                # --------------------------------------------------------------

                per_dim, global_feats = _make_features(
                    c1, G1, c2, G2,
                    upr_scales=upr_scales,
                    upr_offsets=upr_offsets,
                    upr_type_onehot=upr_onehot,
                )

                # Competitor scores — all stored as inconsistency ∈ [0,1]
                i_theta = float(inc.get("I_theta",             float("nan")))
                i_aabb  = float(inc.get("jaccard_inconsistency",
                                        1.0 - inc.get("jaccard_index", float("nan"))))
                i_mfmc  = 1.0 - float(inc.get("I_MF_adaptive_sobol", float("nan")))

                self.samples.append(
                    dict(
                        per_dim=per_dim,           # (d, 4)
                        global_feats=global_feats, # (N_GLOBAL,) = (10,)
                        label=np.float32(label),
                        dim=d,
                        i_theta=i_theta,
                        i_aabb=i_aabb,
                        i_mfmc=i_mfmc,
                    )
                )
            except (KeyError, ValueError, TypeError):
                continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """Pad per_dim to MAX_DIM and build a boolean mask for real dimensions."""
    per_dims, masks, globals_, labels = [], [], [], []

    for s in batch:
        d = s["dim"]
        pd = np.zeros((MAX_DIM, N_DIM_FEAT), dtype=np.float32)
        pd[:d] = s["per_dim"]
        per_dims.append(pd)

        mask = np.zeros(MAX_DIM, dtype=np.float32)
        mask[:d] = 1.0
        masks.append(mask)

        globals_.append(s["global_feats"])
        labels.append(s["label"])

    return dict(
        per_dim=torch.tensor(np.stack(per_dims)),      # (B, MAX_DIM, 4)
        mask=torch.tensor(np.stack(masks)),             # (B, MAX_DIM)
        global_feats=torch.tensor(np.stack(globals_)), # (B, N_GLOBAL)
        label=torch.tensor(np.array(labels)),          # (B,)
        i_theta=[s["i_theta"] for s in batch],
        i_aabb=[s["i_aabb"]  for s in batch],
        i_mfmc=[s["i_mfmc"]  for s in batch],
    )
