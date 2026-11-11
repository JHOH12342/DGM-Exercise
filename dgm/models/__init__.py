from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from omegaconf import OmegaConf

from . import (
    ddpm,
    flow,
    gan,
    gaussian,
    ncsn,
    vae,
)
from .base import GenerativeModel
from .registry import MODELS, register_model


def build_model(cfg: Mapping[str, Any], data_shape: tuple[int, ...]) -> GenerativeModel:
    if not OmegaConf.is_config(cfg):
        cfg = OmegaConf.create(dict(cfg))
    name = cfg.name
    if name not in MODELS:
        raise KeyError(f"Unknown model '{name}'. Registered: {sorted(MODELS)}")
    return MODELS[name](tuple(data_shape), cfg)


__all__ = ["MODELS", "GenerativeModel", "build_model", "register_model"]
