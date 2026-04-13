"""GINE-based MPNN for zonotope inconsistency prediction.

Architecture overview
=====================
ZonotopeEncoder:  (center, generators, mask) -> node embedding  (DeepSets-style)
EdgeEncoder:      (F_flat, f) -> edge embedding
GINEBackbone:     stack of GINEConv layers with residual connections
InconsistencyHead: global_mean_pool -> MLP -> sigmoid  (variable-size graphs)

The model makes **no assumption** on the number of nodes per graph,
so it generalises from the current 2-node CONVIDE scenarios to arbitrary
multi-model DAGs.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool

from .config import ModelConfig


# ---------------------------------------------------------------------------
# Utility: MLP builder
# ---------------------------------------------------------------------------

def build_mlp(
    dims: List[int],
    batch_norm: bool = True,
    final_activation: bool = False,
) -> nn.Sequential:
    """Build a sequential MLP from a list of layer dimensions.

    Applies ``Linear -> [BatchNorm1d ->] ReLU`` between each pair of
    dimensions.  The final activation is omitted unless *final_activation*
    is set.
    """
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        is_last = i == len(dims) - 2
        if not is_last or final_activation:
            if batch_norm and dims[i + 1] > 1:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Zonotope encoder (DeepSets / PointNet-style)
# ---------------------------------------------------------------------------

class ZonotopeEncoder(nn.Module):
    """Encode (center, generators, generator_mask) -> node embedding.

    Center branch:    MLP(D_MAX -> h_c)
    Generator branch: shared MLP per column (D_MAX -> h_g), masked sum/mean pool
    Output:           Linear(h_c + h_g -> node_embed_dim) + BN + ReLU
    """

    def __init__(
        self,
        center_mlp_dims: List[int],
        generator_mlp_dims: List[int],
        pool: str = "sum",
        node_embed_dim: int = 128,
        batch_norm: bool = True,
    ):
        super().__init__()
        self.center_mlp = build_mlp(center_mlp_dims, batch_norm=batch_norm, final_activation=True)
        self.gen_mlp = build_mlp(generator_mlp_dims, batch_norm=batch_norm, final_activation=True)
        self.pool = pool
        concat_dim = center_mlp_dims[-1] + generator_mlp_dims[-1]
        self.projection = nn.Linear(concat_dim, node_embed_dim)
        self.norm = nn.BatchNorm1d(node_embed_dim) if batch_norm else nn.Identity()

    def forward(
        self,
        center: torch.Tensor,
        generators: torch.Tensor,
        generator_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            center:         (N, D_MAX)
            generators:     (N, P_MAX, D_MAX)
            generator_mask: (N, P_MAX)  bool — True for real generators.

        Returns:
            (N, node_embed_dim)
        """
        # Center branch
        c_emb = self.center_mlp(center)  # (N, h_c)

        # Generator branch: apply shared MLP per column
        N, P, D = generators.shape
        g_flat = generators.reshape(N * P, D)
        g_emb = self.gen_mlp(g_flat)      # (N*P, h_g)
        g_emb = g_emb.reshape(N, P, -1)   # (N, P, h_g)

        # Masked pooling
        mask = generator_mask.unsqueeze(-1).float()  # (N, P, 1)
        g_emb = g_emb * mask
        if self.pool == "sum":
            g_pool = g_emb.sum(dim=1)  # (N, h_g)
        else:
            count = mask.sum(dim=1).clamp(min=1)
            g_pool = g_emb.sum(dim=1) / count

        combined = torch.cat([c_emb, g_pool], dim=-1)
        out = self.projection(combined)
        out = self.norm(F.relu(out))
        return out


# ---------------------------------------------------------------------------
# Edge encoder
# ---------------------------------------------------------------------------

class EdgeEncoder(nn.Module):
    """Encode UPR edge attributes (F_flat + f = 20-dim) -> edge embedding."""

    def __init__(self, edge_mlp_dims: List[int], batch_norm: bool = True):
        super().__init__()
        self.mlp = build_mlp(edge_mlp_dims, batch_norm=batch_norm, final_activation=True)

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """(E, edge_input_dim) -> (E, edge_embed_dim)"""
        return self.mlp(edge_attr)


# ---------------------------------------------------------------------------
# GINE backbone
# ---------------------------------------------------------------------------

class GINEBackbone(nn.Module):
    """Stack of GINEConv layers with residual connections.

    Each layer: GINEConv (with inner MLP + edge_dim) -> BatchNorm -> ReLU -> Dropout + residual.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_layers: int,
        mlp_layers: int = 2,
        eps_learnable: bool = True,
        dropout: float = 0.1,
        batch_norm: bool = True,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(n_layers):
            inner_dims = [hidden_dim] * (mlp_layers + 1)
            inner_mlp = build_mlp(inner_dims, batch_norm=batch_norm, final_activation=False)
            conv = GINEConv(nn=inner_mlp, train_eps=eps_learnable, edge_dim=hidden_dim)
            self.convs.append(conv)
            self.norms.append(nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity())

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:          (N_total, hidden_dim)
            edge_index: (2, E_total)
            edge_attr:  (E_total, hidden_dim)

        Returns:
            (N_total, hidden_dim)
        """
        for conv, norm in zip(self.convs, self.norms):
            x_res = x
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = x + x_res
        return x


# ---------------------------------------------------------------------------
# Task heads
# ---------------------------------------------------------------------------

class InconsistencyHead(nn.Module):
    """Graph-level readout -> MLP -> sigmoid for I(theta) in [0, 1].

    Uses ``global_mean_pool`` or ``global_add_pool`` — works for any
    number of nodes per graph.
    """

    def __init__(
        self,
        readout: str,
        hidden_dim: int,
        head_mlp_dims: List[int],
        batch_norm: bool = True,
    ):
        super().__init__()
        self.readout = readout
        actual_dims = [hidden_dim] + head_mlp_dims[1:]
        self.mlp = build_mlp(actual_dims, batch_norm=batch_norm)

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     (N_total, hidden_dim) node embeddings after GINE.
            batch: (N_total,) graph membership.

        Returns:
            (B, 1) predictions in [0, 1].
        """
        if self.readout == "sum":
            graph_emb = global_add_pool(x, batch)
        else:
            graph_emb = global_mean_pool(x, batch)
        return torch.sigmoid(self.mlp(graph_emb))


class VolumeHead(nn.Module):
    """Single-node embedding -> log1p(volume) regression."""

    def __init__(self, hidden_dim: int, batch_norm: bool = True):
        super().__init__()
        self.mlp = build_mlp([hidden_dim, 64, 1], batch_norm=batch_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, hidden_dim) -> (N, 1)"""
        return self.mlp(x)


class ContainmentHead(nn.Module):
    """Node embedding + point -> binary containment probability.

    Concatenates the (broadcast) node embedding with each point and
    applies an MLP with sigmoid output.
    """

    def __init__(self, hidden_dim: int, point_dim: int = 4, batch_norm: bool = True):
        super().__init__()
        self.mlp = build_mlp([hidden_dim + point_dim, 64, 1], batch_norm=batch_norm)

    def forward(
        self, node_emb: torch.Tensor, points: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            node_emb: (B, hidden_dim) — one embedding per zonotope.
            points:   (B, K, D_MAX) — K test points per zonotope.

        Returns:
            (B, K) probabilities.
        """
        B, K, D = points.shape
        emb_exp = node_emb.unsqueeze(1).expand(-1, K, -1)  # (B, K, h)
        combined = torch.cat([emb_exp, points], dim=-1)     # (B, K, h+D)
        combined_flat = combined.reshape(B * K, -1)
        out = torch.sigmoid(self.mlp(combined_flat))        # (B*K, 1)
        return out.reshape(B, K)


class AffineMapHead(nn.Module):
    """Predict target center + generators from node embedding."""

    def __init__(
        self,
        hidden_dim: int,
        d_max: int = 4,
        p_max: int = 12,
        batch_norm: bool = True,
    ):
        super().__init__()
        self.d_max = d_max
        self.p_max = p_max
        out_dim = d_max + p_max * d_max
        self.mlp = build_mlp([hidden_dim, 128, out_dim], batch_norm=batch_norm)

    def forward(self, x: torch.Tensor):
        """x: (B, hidden_dim) -> (center: (B, D), generators: (B, P, D))"""
        out = self.mlp(x)
        center = out[:, : self.d_max]
        generators = out[:, self.d_max :].reshape(-1, self.p_max, self.d_max)
        return center, generators


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class ZonotopeGINE(nn.Module):
    """Full GINE-based model for zonotope inconsistency prediction.

    Shared backbone (encoder + GINE) with swappable task heads.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Shared backbone
        self.zonotope_encoder = ZonotopeEncoder(
            center_mlp_dims=config.center_mlp_dims,
            generator_mlp_dims=config.generator_mlp_dims,
            pool=config.generator_pool,
            node_embed_dim=config.node_embed_dim,
            batch_norm=config.batch_norm,
        )
        self.edge_encoder = EdgeEncoder(
            edge_mlp_dims=config.edge_mlp_dims,
            batch_norm=config.batch_norm,
        )
        self.gine_backbone = GINEBackbone(
            hidden_dim=config.gine_hidden_dim,
            n_layers=config.n_gine_layers,
            mlp_layers=config.gine_mlp_layers,
            eps_learnable=True,
            dropout=config.dropout,
            batch_norm=config.batch_norm,
        )

        # Main task head
        self.inconsistency_head = InconsistencyHead(
            readout=config.readout,
            hidden_dim=config.gine_hidden_dim,
            head_mlp_dims=config.head_mlp_dims,
            batch_norm=config.batch_norm,
        )

        # Pretraining heads (lazily initialised)
        self.volume_head: Optional[VolumeHead] = None
        self.containment_head: Optional[ContainmentHead] = None
        self.pairwise_aabb_head: Optional[InconsistencyHead] = None
        self.affine_map_head: Optional[AffineMapHead] = None

    def init_pretrain_heads(self) -> None:
        """Create pretraining task heads (called only when pretraining)."""
        cfg = self.config
        self.volume_head = VolumeHead(cfg.gine_hidden_dim, cfg.batch_norm)
        self.containment_head = ContainmentHead(
            cfg.gine_hidden_dim, cfg.d_max, cfg.batch_norm,
        )
        self.pairwise_aabb_head = InconsistencyHead(
            readout=cfg.readout,
            hidden_dim=cfg.gine_hidden_dim,
            head_mlp_dims=cfg.head_mlp_dims,
            batch_norm=cfg.batch_norm,
        )
        self.affine_map_head = AffineMapHead(
            cfg.gine_hidden_dim, cfg.d_max, cfg.p_max, cfg.batch_norm,
        )

    # --- Shared forward ---

    def encode_graph(self, data) -> torch.Tensor:
        """Encode nodes and edges, run GINE backbone.

        Returns updated node embeddings (N_total, hidden_dim).
        """
        x = self.zonotope_encoder(
            data.x_center, data.x_generators, data.x_generator_mask,
        )

        if data.edge_attr.shape[0] > 0:
            edge_emb = self.edge_encoder(data.edge_attr)
            x = self.gine_backbone(x, data.edge_index, edge_emb)

        return x

    # --- Task-specific forwards ---

    def forward_inconsistency(self, data) -> torch.Tensor:
        x = self.encode_graph(data)
        return self.inconsistency_head(x, data.batch)

    def forward_volume(self, data) -> torch.Tensor:
        x = self.encode_graph(data)
        return self.volume_head(x)

    def forward_pairwise_aabb(self, data) -> torch.Tensor:
        x = self.encode_graph(data)
        return self.pairwise_aabb_head(x, data.batch)

    def forward_containment(self, data) -> torch.Tensor:
        x = self.encode_graph(data)
        # 1-node graphs: each node IS a graph, so x shape is (B, h)
        return self.containment_head(x, data.points)

    def forward_affine_map(self, data):
        x = self.encode_graph(data)
        # Extract target node (node 1 in each 2-node graph)
        B = int(data.batch.max().item()) + 1
        # Node indices for node-1 in each graph: 1, 3, 5, ...
        node1_idx = torch.arange(B, device=x.device) * 2 + 1
        x_target = x[node1_idx]
        return self.affine_map_head(x_target)

    # --- Parameter groups for differential LR ---

    def backbone_parameters(self):
        return (
            list(self.zonotope_encoder.parameters())
            + list(self.edge_encoder.parameters())
            + list(self.gine_backbone.parameters())
        )

    def head_parameters(self, task: str = "inconsistency"):
        heads = {
            "inconsistency": self.inconsistency_head,
            "volume": self.volume_head,
            "containment": self.containment_head,
            "pairwise_aabb": self.pairwise_aabb_head,
            "affine_map": self.affine_map_head,
        }
        head = heads[task]
        return list(head.parameters()) if head is not None else []
