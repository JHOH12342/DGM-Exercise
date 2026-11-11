from __future__ import annotations

import torch
from torch import nn

from .base import GenerativeModel
from .registry import register_model


@register_model("gaussian")
class DiagonalGaussian(GenerativeModel):
    def __init__(self, data_shape, cfg=None):
        super().__init__(data_shape, cfg)
        self.mean = nn.Parameter(torch.zeros(*self.data_shape))
        self.log_std = nn.Parameter(torch.zeros(*self.data_shape))

    @property
    def std(self) -> torch.Tensor:
        return self.log_std.exp()

    def _dist(self) -> torch.distributions.Independent:
        return torch.distributions.Independent(
            torch.distributions.Normal(self.mean, self.std), len(self.data_shape)
        )

    def loss(self, x):
        nll = -self.log_prob(x).mean()
        return nll, {"nll_per_dim": float(nll.detach()) / self.dim}

    @torch.no_grad()
    def sample(self, n, return_trajectory=False):
        eps = self.noise(n)
        x = self.mean + self.std * eps
        if return_trajectory:
            return x, torch.stack([eps, x], dim=0)
        return x

    def log_prob(self, x):
        return self._dist().log_prob(x)

    def vector_field(self, x, t):
        return -(x - self.mean) / self.std**2
