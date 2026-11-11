from __future__ import annotations

from typing import Any, Mapping

from omegaconf import OmegaConf

from .base import DataDistribution
from .registry import DATASETS, register_dataset

from . import toy2d
from . import cifar10


def build_dataset(cfg: Mapping[str, Any]) -> DataDistribution:
    if OmegaConf.is_config(cfg):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = dict(cfg)
    name = cfg.pop("name")
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset '{name}'. Registered: {sorted(DATASETS)}")
    return DATASETS[name](**cfg)


__all__ = ["DataDistribution", "DATASETS", "register_dataset", "build_dataset"]
