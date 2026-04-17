"""YAML config loading with dataclass defaults for the surrogate training pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml


# ---------------------------------------------------------------------------
# Dataclass hierarchy
# ---------------------------------------------------------------------------

@dataclass
class PretrainConfig:
    enabled: bool = True
    tasks: List[str] = field(default_factory=lambda: [
        "volume", "containment", "pairwise_aabb", "affine_map",
    ])
    epochs: int = 500
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5
    scheduler: str = "cosine"
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        "volume": 1.0, "containment": 1.0,
        "pairwise_aabb": 1.0, "affine_map": 1.0,
    })
    val_fraction: float = 0.1
    patience: int = 10


@dataclass
class TrainConfig:
    scenarios_train: List[int] = field(default_factory=lambda: [1,2,3,4,6,7,8,9,10,11,12])
    scenarios_test: List[int] = field(default_factory=lambda: [5])
    val_fraction: float = 0.15
    epochs: int = 100
    batch_size: int = 128
    lr: float = 5e-4
    weight_decay: float = 1e-5
    scheduler: str = "cosine"
    loss: str = "huber"
    huber_delta: float = 0.1
    mse_weight: float = 0.5        # only used when loss == "combined"
    patience: int = 15
    freeze_backbone_epochs: int = 0
    backbone_lr_factor: float = 0.1
    auxiliary_aabb_weight: float = 0.0


@dataclass
class ModelConfig:
    d_max: int = 4
    p_max: int = 12
    center_mlp_dims: List[int] = field(default_factory=lambda: [4, 64])
    generator_mlp_dims: List[int] = field(default_factory=lambda: [4, 64, 64])
    generator_pool: str = "sum"
    node_embed_dim: int = 128
    edge_input_dim: int = 20
    edge_mlp_dims: List[int] = field(default_factory=lambda: [20, 64, 128])
    n_gine_layers: int = 3
    gine_hidden_dim: int = 128
    gine_mlp_layers: int = 2
    dropout: float = 0.1
    batch_norm: bool = True
    readout: str = "mean"
    head_mlp_dims: List[int] = field(default_factory=lambda: [128, 64, 1])


@dataclass
class SurrogateConfig:
    data_root: str = "data/surrogate"
    output_dir: str = "results/surrogate"
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    seed: int = 42
    device: str = "auto"
    num_workers: int = 4


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_NESTED = {
    "pretrain": PretrainConfig,
    "train": TrainConfig,
    "model": ModelConfig,
}


def _merge_dataclass(dc_cls, overrides: dict):
    """Create a dataclass instance, overriding defaults with *overrides*."""
    defaults = dc_cls()
    merged = {}
    for f in dc_cls.__dataclass_fields__:
        if f in overrides:
            merged[f] = overrides[f]
        else:
            merged[f] = getattr(defaults, f)
    return dc_cls(**merged)


def load_config(path: Optional[Union[str, Path]] = None) -> SurrogateConfig:
    """Load config from YAML, falling back to dataclass defaults.

    Args:
        path: Path to YAML file.  If *None*, returns pure defaults.
    """
    if path is None:
        return SurrogateConfig()

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    kwargs = {}
    for key in SurrogateConfig.__dataclass_fields__:
        if key in _NESTED:
            sub = raw.get(key, {})
            kwargs[key] = _merge_dataclass(_NESTED[key], sub if sub else {})
        elif key in raw:
            kwargs[key] = raw[key]

    return _merge_dataclass(SurrogateConfig, kwargs)


def resolve_device(device_str: str) -> str:
    """Resolve ``'auto'`` to ``'cuda:0'`` if available, else ``'cpu'``."""
    if device_str == "auto":
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return device_str
