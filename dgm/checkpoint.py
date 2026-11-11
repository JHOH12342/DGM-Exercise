from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from .data import build_dataset
from .models import build_model


def save_checkpoint(path: str | Path, cfg: DictConfig, model, step: int, ema=None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "cfg": OmegaConf.to_container(cfg, resolve=True),
        "step": int(step),
        "model": model.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "optimizers": {k: o.state_dict() for k, o in getattr(model, "optimizers", {}).items()},
        "lr_schedulers": {
            k: s.state_dict() for k, s in getattr(model, "lr_schedulers", {}).items()
        },
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(ckpt, tmp)
    tmp.replace(path)
    return path


def load_checkpoint(path: str | Path, map_location="cpu") -> dict:
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def build_from_checkpoint(
    path: str | Path, device: torch.device | str | None = None, use_ema: bool = True
):
    ckpt = load_checkpoint(path)
    cfg = OmegaConf.create(ckpt["cfg"])
    OmegaConf.set_struct(cfg, False)
    dataset = build_dataset(cfg.data)
    model = build_model(cfg.model, dataset.shape)
    if use_ema and ckpt.get("ema") is not None:
        model.load_state_dict(ckpt["ema"]["module"])
    else:
        model.load_state_dict(ckpt["model"])
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return cfg, dataset, model, ckpt["step"]
