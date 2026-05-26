"""
DeepSets surrogate for zonotope overlap estimation.

Input : two constrained zonotopes (c1, G1) and (c2, G2) with d in {2, 3, 4}
Output: overlap score in [0, 1]  (same quantity as MFMC I_MF_sobol)

Architecture
------------
For each dimension i:
    feats_i = [delta_c_i/sigma_i, ||G1_i||/sigma_i, ||G2_i||/sigma_i, cos(G1_i, G2_i)]
    e_i     = phi(feats_i)          # shared MLP, same weights every dim

aggregate = sum_i(e_i * mask_i)     # zero out padded dims
output    = sigmoid(rho([aggregate, global_feats]))
"""

import torch
import torch.nn as nn


MAX_DIM    = 4   # largest scenario dimension
N_DIM_FEAT = 4   # features per dimension
N_GLOBAL   = 10  # global features: 2 geometric + 8 UPR-type one-hot


class DeepSetsZonotope(nn.Module):
    def __init__(self, phi_hidden: int = 16, rho_hidden: int = 32):
        super().__init__()

        self.phi = nn.Sequential(
            nn.Linear(N_DIM_FEAT, phi_hidden),
            nn.ReLU(),
            nn.Linear(phi_hidden, phi_hidden),
        )

        self.rho = nn.Sequential(
            nn.Linear(phi_hidden + N_GLOBAL, rho_hidden),
            nn.ReLU(),
            nn.Linear(rho_hidden, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        per_dim: torch.Tensor,   # (B, MAX_DIM, N_DIM_FEAT)
        mask: torch.Tensor,      # (B, MAX_DIM)  1=real, 0=padded
        global_feats: torch.Tensor,  # (B, N_GLOBAL)
    ) -> torch.Tensor:           # (B,)
        B, D, _ = per_dim.shape

        # encode each dimension with shared weights
        encoded = self.phi(per_dim.view(B * D, N_DIM_FEAT))  # (B*D, H)
        encoded = encoded.view(B, D, -1)                       # (B, D, H)

        # zero-out padded dimensions then sum-pool
        aggregated = (encoded * mask.unsqueeze(-1)).sum(dim=1)  # (B, H)

        out = self.rho(torch.cat([aggregated, global_feats], dim=1))  # (B, 1)
        return out.squeeze(-1)                                          # (B,)
