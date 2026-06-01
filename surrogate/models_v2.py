"""
Model variants for zonotope overlap estimation.

Four architectures, all predicting I(theta) in [0, 1]:
  1. DeepSetsV2     — sum+max dual pooling over dimensions, larger MLPs
  2. FlatMLP        — simple MLP on flattened per-dim features
  3. SiameseDeepSets — encode each zonotope as set of generator columns, compare
  4. SmallSetTransformer — self-attention over dimensions

All models use the same enhanced feature set (5 per-dim, 3 global) except
SiameseDeepSets which operates on raw (propagated) zonotope geometry.
"""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_DIM = 4
P_MAX = 12
N_DIM_FEAT = 5   # delta_c, r1_norm, r2_norm, cos, width_ratio
N_GLOBAL = 5     # log_vol_ratio, norm_center_dist, dim, sep_ratio, off_over_tgt


# ═══════════════════════════════════════════════════════════════════════
# 1. DeepSets V2  — sum + max dual pooling
# ═══════════════════════════════════════════════════════════════════════

class DeepSetsV2(nn.Module):
    """Enhanced DeepSets with dual sum+max pooling and deeper MLPs.

    phi encodes each dimension independently, then both sum-pool (captures
    total overlap volume) and max-pool (captures bottleneck dimension) are
    concatenated and fed to rho.
    """

    def __init__(self, phi_hidden: int = 32, rho_hidden: int = 64):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(N_DIM_FEAT, phi_hidden),
            nn.ReLU(),
            nn.Linear(phi_hidden, phi_hidden),
            nn.ReLU(),
            nn.Linear(phi_hidden, phi_hidden),
        )
        rho_input = 2 * phi_hidden + N_GLOBAL
        self.rho = nn.Sequential(
            nn.Linear(rho_input, rho_hidden),
            nn.ReLU(),
            nn.Linear(rho_hidden, rho_hidden // 2),
            nn.ReLU(),
            nn.Linear(rho_hidden // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, per_dim, mask, global_feats):
        """
        per_dim:      (B, MAX_DIM, N_DIM_FEAT)
        mask:         (B, MAX_DIM)  1=real, 0=padded
        global_feats: (B, N_GLOBAL)
        """
        B, D, _ = per_dim.shape
        encoded = self.phi(per_dim.reshape(B * D, -1)).reshape(B, D, -1)

        mask_exp = mask.unsqueeze(-1)                         # (B, D, 1)
        sum_pool = (encoded * mask_exp).sum(dim=1)            # (B, H)
        masked_max = encoded + (1.0 - mask_exp) * (-1e9)
        max_pool = masked_max.max(dim=1).values               # (B, H)

        agg = torch.cat([sum_pool, max_pool, global_feats], dim=1)
        return self.rho(agg).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════
# 2. Flat MLP
# ═══════════════════════════════════════════════════════════════════════

class FlatMLP(nn.Module):
    """Flattened per-dim features + global → MLP.

    No permutation invariance — treats each axis slot as a distinct input.
    Relies on zero-padding for d < MAX_DIM.
    """

    def __init__(self, hidden: int = 64):
        super().__init__()
        input_dim = MAX_DIM * N_DIM_FEAT + N_GLOBAL
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, per_dim, mask, global_feats):
        B = per_dim.shape[0]
        flat = per_dim.reshape(B, -1)                         # (B, MAX_DIM * 5)
        x = torch.cat([flat, global_feats], dim=1)
        return self.mlp(x).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════
# 3. Siamese DeepSets
# ═══════════════════════════════════════════════════════════════════════

class SiameseDeepSets(nn.Module):
    """Encode each zonotope as a set of generator columns, then compare.

    Each zonotope is represented by its center (D_MAX,) and generator
    columns (P_MAX, D_MAX).  A shared encoder processes each zonotope
    into a fixed-size embedding.  The two embeddings are combined via
    concatenation, difference, and element-wise product, then fed to
    a comparison head.

    This architecture respects the permutation invariance of generators
    and sees the full geometry (not just per-row summaries).
    """

    def __init__(self, gen_hidden: int = 32, embed_dim: int = 32):
        super().__init__()
        self.gen_phi = nn.Sequential(
            nn.Linear(MAX_DIM, gen_hidden),
            nn.ReLU(),
            nn.Linear(gen_hidden, gen_hidden),
        )
        self.center_mlp = nn.Sequential(
            nn.Linear(MAX_DIM, gen_hidden),
            nn.ReLU(),
        )
        self.combine = nn.Sequential(
            nn.Linear(2 * gen_hidden, embed_dim),
            nn.ReLU(),
        )
        head_input = 4 * embed_dim + N_GLOBAL
        self.head = nn.Sequential(
            nn.Linear(head_input, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def encode_zonotope(self, center, generators, gen_mask):
        """
        center:     (B, MAX_DIM)
        generators: (B, P_MAX, MAX_DIM)  — each row is one generator column
        gen_mask:   (B, P_MAX)  1=real, 0=padded
        """
        c_emb = self.center_mlp(center)                       # (B, H)

        B, P, D = generators.shape
        g_emb = self.gen_phi(generators.reshape(B * P, D))
        g_emb = g_emb.reshape(B, P, -1)                       # (B, P, H)
        g_pool = (g_emb * gen_mask.unsqueeze(-1)).sum(dim=1)   # (B, H)

        return self.combine(torch.cat([c_emb, g_pool], dim=-1))

    def forward(self, src_center, src_gens, src_gen_mask,
                tgt_center, tgt_gens, tgt_gen_mask, global_feats):
        e1 = self.encode_zonotope(src_center, src_gens, src_gen_mask)
        e2 = self.encode_zonotope(tgt_center, tgt_gens, tgt_gen_mask)

        combined = torch.cat([e1, e2, e1 - e2, e1 * e2, global_feats], dim=-1)
        return self.head(combined).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════
# 4. Small Set Transformer
# ═══════════════════════════════════════════════════════════════════════

class _MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)        # (3, B, H, N, d)
        q, k, v = qkv.unbind(0)

        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale  # (B, H, N, N)
        if mask is not None:
            attn = attn.masked_fill(
                ~mask.unsqueeze(1).unsqueeze(2).bool(), float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = torch.nan_to_num(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)


class _SetAttentionBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, ff_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _MultiHeadAttention(dim, n_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, dim),
        )

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ff(self.norm2(x))
        return x


class SmallSetTransformer(nn.Module):
    """Self-attention over spatial dimensions, then pool + MLP.

    Unlike DeepSets, the attention layers let dimensions interact during
    encoding — capturing cross-dimensional effects that sum/max pooling miss.
    """

    def __init__(self, d_model: int = 32, n_heads: int = 2,
                 ff_dim: int = 64, n_layers: int = 1):
        super().__init__()
        self.project = nn.Linear(N_DIM_FEAT, d_model)
        self.layers = nn.ModuleList([
            _SetAttentionBlock(d_model, n_heads, ff_dim)
            for _ in range(n_layers)
        ])
        rho_input = d_model + N_GLOBAL
        self.rho = nn.Sequential(
            nn.Linear(rho_input, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
            nn.Sigmoid(),
        )

    def forward(self, per_dim, mask, global_feats):
        x = self.project(per_dim)                              # (B, D, d_model)
        for layer in self.layers:
            x = layer(x, mask)

        mask_exp = mask.unsqueeze(-1)
        pooled = (x * mask_exp).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)

        return self.rho(torch.cat([pooled, global_feats], dim=1)).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════
# 5. Product-of-Experts Set Transformer
# ═══════════════════════════════════════════════════════════════════════

class ProductSetTransformer(nn.Module):
    """Self-attention over dimensions with multiplicative aggregation.

    Inconsistency is a product across dimensions: a point must be inside
    the target in ALL dimensions. This model outputs per-dimension
    log-containment estimates after attention, sums them (= log of product),
    and adds a global bias before the final sigmoid.
    """

    def __init__(self, d_model: int = 32, n_heads: int = 2,
                 ff_dim: int = 64, n_layers: int = 1):
        super().__init__()
        self.project = nn.Linear(N_DIM_FEAT, d_model)
        self.layers = nn.ModuleList([
            _SetAttentionBlock(d_model, n_heads, ff_dim)
            for _ in range(n_layers)
        ])
        head_hidden = d_model // 2
        # Per-dim head: outputs a scalar log-containment per dimension
        self.dim_head = nn.Sequential(
            nn.Linear(d_model, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )
        # Global bias: adjusts the product based on global features
        self.global_bias = nn.Sequential(
            nn.Linear(N_GLOBAL, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, per_dim, mask, global_feats):
        x = self.project(per_dim)                              # (B, D, d_model)
        for layer in self.layers:
            x = layer(x, mask)

        log_p = self.dim_head(x).squeeze(-1)                   # (B, D)
        # Masked sum of per-dim log-containment
        log_p = (log_p * mask).sum(dim=1)                      # (B,)
        bias = self.global_bias(global_feats).squeeze(-1)      # (B,)

        return torch.sigmoid(log_p + bias)


class ProductSetTransformerExact(ProductSetTransformer):
    """Exact noisy-AND head: I = 1 - prod_i p_i * p_global.

    Identical to ProductSetTransformer except the output is a true product of
    per-dimension consistency probabilities. Each per-dim head output is read
    as a logit; p_i = sigmoid(l_i) in (0,1) is the probability that a source
    realization is contained in the target along dimension i. The global term
    is one additional containment factor p_g = sigmoid(b(g)). Consistency is
    their product (a point must be contained in ALL dimensions), so

        P(consistent) = prod_i p_i * p_g,   I = 1 - P(consistent).

    Computed in log-space for stability: log P = sum_i logsigmoid(l_i) +
    logsigmoid(b), then I = 1 - exp(log P) = -expm1(log P). This directly
    encodes the conjunction structure (no logistic approximation), with
    I in [0, 1) by construction.
    """

    def forward(self, per_dim, mask, global_feats):
        x = self.project(per_dim)                              # (B, D, d_model)
        for layer in self.layers:
            x = layer(x, mask)

        logit = self.dim_head(x).squeeze(-1)                   # (B, D)
        log_p = F.logsigmoid(logit)                            # (B, D), <= 0
        log_p = (log_p * mask).sum(dim=1)                      # (B,) masked sum
        bias = self.global_bias(global_feats).squeeze(-1)      # (B,)
        log_consistent = log_p + F.logsigmoid(bias)            # (B,), <= 0

        return (-torch.expm1(log_consistent)).clamp(0.0, 1.0)  # 1 - exp(.)


# ═══════════════════════════════════════════════════════════════════════
# 6. Direction-Probe Support Network
# ═══════════════════════════════════════════════════════════════════════

class DirectionProbeNet(nn.Module):
    """Support-function separation model over a learned set of probe directions.

    Geometric basis. Two zonotopes overlap iff for every direction v,
        <o, v> <= h1(v) + h2(v),   o = c1 - c2,   h_k(v) = sum_j |g_kj . v|.
    "Contained along all directions" is a conjunction, so inconsistency is a
    product over directions of per-direction containment probabilities. This
    model instantiates that directly:

      - Probe directions V = { offset hat(o) } u { axes e_i (real dims) }
        u { L learned directions }. The offset and axes recover the existing
        per-dim + support-function features as special cases; the learned
        directions are differentiable analogues of attention queries.
      - For each v, supports are computed exactly from the RAW generators by
        sum-pooling |g . v| over the generator set (handles arbitrary
        generator counts; permutation- and sign-invariant). This is the
        ReLU<->zonotope support-function duality (Froese et al.): a sum of
        |g . v| is a 2-layer ReLU network evaluating the support function.
      - A shared per-direction MLP maps (h1, h2, |o.v|) to a containment
        logit; the masked sum of log-sigmoids is the log-product, and
        I = 1 - prod_v p_v * p_dim is the exact noisy-AND (as in
        ProductSetTransformerExact), with a per-dimensionality calibration
        factor p_dim.

    Crucially, it is NOT given the hand-engineered support globals
    (sep_ratio, off_over_tgt) - it must compute support from raw geometry.
    That is the whole point of the comparison.
    """

    def __init__(self, n_learned: int = 6, mlp_hidden: int = 32):
        super().__init__()
        self.n_learned = n_learned
        # Learned probe directions (in the padded MAX_DIM space).
        self.learned_dirs = nn.Parameter(torch.randn(n_learned, MAX_DIM) * 0.5)
        # Shared per-direction containment head on (h1, h2, |o.v|).
        self.dir_mlp = nn.Sequential(
            nn.Linear(3, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
        )
        # Start near p_v ~ 1 (small weights + positive bias) so the initial
        # log-product is moderate rather than saturating I near 1 (which would
        # vanish the gradient through the 1 - exp(.) head).
        nn.init.normal_(self.dir_mlp[-1].weight, std=0.01)
        nn.init.constant_(self.dir_mlp[-1].bias, 3.0)
        # Per-dimensionality calibration bias (indexed by d-1), starts ~ p=0.95.
        self.dim_bias = nn.Parameter(torch.full((MAX_DIM,), 3.0))

    def forward(self, src_center, src_generators, src_gen_mask,
                tgt_center, tgt_generators, tgt_gen_mask, dim_mask):
        """
        src_center/tgt_center:        (B, MAX_DIM)
        src_generators/tgt_generators:(B, P_MAX, MAX_DIM)  row = one generator
        src_gen_mask/tgt_gen_mask:    (B, P_MAX)  1=real generator
        dim_mask:                     (B, MAX_DIM) 1=real spatial dimension
        """
        B = src_center.shape[0]
        eps = 1e-10
        o = src_center - tgt_center                              # (B, D)
        off_mag = o.norm(dim=1, keepdim=True)                    # (B, 1)
        off_hat = o / (off_mag + eps)                            # (B, D)

        # ── Assemble probe directions (B, n_dir, D) and validity (B, n_dir) ──
        offset_dir = off_hat.unsqueeze(1)                        # (B, 1, D)
        offset_valid = (off_mag > eps).float()                  # (B, 1)

        axes = torch.eye(MAX_DIM, device=o.device).unsqueeze(0).expand(B, -1, -1)
        axes_valid = dim_mask                                    # (B, D)

        q = self.learned_dirs.unsqueeze(0).expand(B, -1, -1)     # (B, L, D)
        q = q * dim_mask.unsqueeze(1)                            # live in real subspace
        q = q / (q.norm(dim=2, keepdim=True) + eps)              # unit directions
        q_valid = torch.ones(B, self.n_learned, device=o.device)

        V = torch.cat([offset_dir, axes, q], dim=1)             # (B, n_dir, D)
        vmask = torch.cat([offset_valid, axes_valid, q_valid], dim=1)  # (B, n_dir)

        # ── Exact supports via sum-pool of |g . v| over generator set ──
        proj_s = torch.einsum("bpd,bnd->bpn", src_generators, V).abs()
        h1 = (proj_s * src_gen_mask.unsqueeze(-1)).sum(dim=1)    # (B, n_dir)
        proj_t = torch.einsum("bpd,bnd->bpn", tgt_generators, V).abs()
        h2 = (proj_t * tgt_gen_mask.unsqueeze(-1)).sum(dim=1)    # (B, n_dir)
        proj_o = torch.einsum("bd,bnd->bn", o, V).abs()          # (B, n_dir)

        feat = torch.stack([h1, h2, proj_o], dim=-1)            # (B, n_dir, 3)
        logit = self.dir_mlp(feat).squeeze(-1)                  # (B, n_dir)

        log_p = F.logsigmoid(logit)                             # (B, n_dir) <= 0
        log_p = (log_p * vmask).sum(dim=1)                      # (B,) masked log-product

        d_idx = (dim_mask.sum(dim=1).long() - 1).clamp(0, MAX_DIM - 1)
        log_consistent = log_p + F.logsigmoid(self.dim_bias[d_idx])

        return (-torch.expm1(log_consistent)).clamp(0.0, 1.0)   # 1 - prod p


# ═══════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════

MODEL_REGISTRY = {
    "deepsets_v2": DeepSetsV2,
    "flat_mlp": FlatMLP,
    "siamese": SiameseDeepSets,
    "set_transformer": SmallSetTransformer,
    "product_transformer": ProductSetTransformer,
    "product_transformer_exact": ProductSetTransformerExact,
    "direction_probe": DirectionProbeNet,
    # Size-matched to product_transformer_exact (~9.4k params) for a
    # capacity-controlled comparison: wider per-direction MLP + more probes.
    "direction_probe_large": partial(
        DirectionProbeNet, n_learned=12, mlp_hidden=94),
    "product_transformer_large": partial(
        ProductSetTransformer, d_model=64, n_heads=4, ff_dim=128, n_layers=2),
}


def create_model(name: str, **kwargs) -> nn.Module:
    cls = MODEL_REGISTRY[name]
    return cls(**kwargs)


def load_checkpoint(path, map_location="cpu") -> nn.Module:
    """Load a model checkpoint saved by train_compare.py.

    Returns an eval-mode model ready for inference on v2 features
    (per_dim: (B, MAX_DIM, 5), mask: (B, MAX_DIM), global: (B, 5)).
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model = create_model(ckpt["model_name"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
