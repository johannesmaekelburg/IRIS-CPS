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


# ═══════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════

class ProductSetTransformerExact(ProductSetTransformer):
    """Exact noisy-AND head: I = 1 - prod_i p_i * p_global.

    Identical architecture to ProductSetTransformer but the forward pass
    computes the product of per-dimension containment probabilities in
    log-space for numerical stability:

        log P(consistent) = sum_i logsigmoid(l_i) + logsigmoid(b)
        I = 1 - exp(log P)  = -expm1(log P),  in [0, 1).
    """

    def forward(self, per_dim, mask, global_feats):
        x = self.project(per_dim)
        for layer in self.layers:
            x = layer(x, mask)

        logit = self.dim_head(x).squeeze(-1)               # (B, D)
        log_p = F.logsigmoid(logit)                        # (B, D), <= 0
        log_p = (log_p * mask).sum(dim=1)                  # (B,) masked sum
        bias = self.global_bias(global_feats).squeeze(-1)  # (B,)
        log_consistent = log_p + F.logsigmoid(bias)        # (B,), <= 0

        return (-torch.expm1(log_consistent)).clamp(0.0, 1.0)


MODEL_REGISTRY = {
    "deepsets_v2": DeepSetsV2,
    "flat_mlp": FlatMLP,
    "siamese": SiameseDeepSets,
    "set_transformer": SmallSetTransformer,
    "product_transformer": ProductSetTransformer,
    "product_transformer_exact": ProductSetTransformerExact,
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
