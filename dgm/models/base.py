from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn


class GenerativeModel(nn.Module, ABC):
    name: str = "base"

    def __init__(self, data_shape: tuple[int, ...], cfg: DictConfig | dict | None = None):
        super().__init__()
        self.data_shape = tuple(int(s) for s in data_shape)
        if cfg is None:
            cfg = {}
        self.cfg: DictConfig = cfg if OmegaConf.is_config(cfg) else OmegaConf.create(dict(cfg))
        self.optimizers: dict[str, torch.optim.Optimizer] = {}
        self.lr_schedulers: dict[str, Any] = {}

    @abstractmethod
    def loss(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]: ...

    @abstractmethod
    def sample(
        self, n: int, return_trajectory: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def vector_field(self, x: torch.Tensor, t: torch.Tensor | float) -> torch.Tensor:
        raise NotImplementedError

    def configure_optimizers(self) -> dict[str, torch.optim.Optimizer]:
        return {
            "main": torch.optim.Adam(
                self.parameters(),
                lr=float(self.cfg.get("lr", 1e-3)),
                betas=tuple(self.cfg.get("betas", (0.9, 0.999))),
                weight_decay=float(self.cfg.get("weight_decay", 0.0)),
            )
        }

    def configure_lr_schedulers(self) -> dict[str, Any]:
        return {}

    def train_step(self, x: torch.Tensor, step: int) -> dict[str, float]:
        loss, logs = self.loss(x)
        grad_norm = self.optimizer_step(loss, "main")
        return {"loss": float(loss.detach()), "grad_norm": grad_norm, **logs}

    def setup_optimizers(self) -> None:
        self.optimizers = self.configure_optimizers()
        self.lr_schedulers = self.configure_lr_schedulers()

    def optimizer_step(
        self,
        loss: torch.Tensor,
        name: str = "main",
        clip_grad: float | None = None,
        retain_graph: bool = False,
    ) -> float:
        opt = self.optimizers[name]
        opt.zero_grad(set_to_none=True)
        loss.backward(retain_graph=retain_graph)
        params = [p for g in opt.param_groups for p in g["params"] if p.grad is not None]
        clip = self.cfg.get("clip_grad", None) if clip_grad is None else clip_grad
        if clip:
            gn = torch.nn.utils.clip_grad_norm_(params, float(clip))
        else:
            gn = (
                torch.norm(torch.stack([p.grad.norm() for p in params]))
                if params
                else torch.tensor(0.0)
            )
        opt.step()
        return float(gn)

    def noise(self, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        return torch.randn(n, *self.data_shape, device=self.device, generator=generator)

    @property
    def dim(self) -> int:
        return int(math.prod(self.data_shape))

    @property
    def is_image(self) -> bool:
        return len(self.data_shape) == 3

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @property
    def has_log_prob(self) -> bool:
        return type(self).log_prob is not GenerativeModel.log_prob

    @property
    def has_vector_field(self) -> bool:
        return type(self).vector_field is not GenerativeModel.vector_field

    def extra_repr(self) -> str:
        return f"data_shape={self.data_shape}"
