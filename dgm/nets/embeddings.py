from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int, time_scale: float = 1000.0, max_period: float = 10_000.0):
        super().__init__()
        if dim % 2:
            raise ValueError("dim must be even")
        self.dim, self.time_scale, self.max_period = dim, time_scale, max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float().reshape(-1) * self.time_scale
        half = self.dim // 2
        freqs = torch.exp(-math.log(self.max_period) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int = 128, out_dim: int | None = None, time_scale: float = 1000.0):
        super().__init__()
        out_dim = out_dim or dim
        self.sin = SinusoidalEmbedding(dim, time_scale=time_scale)
        self.mlp = nn.Sequential(nn.Linear(dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sin(t))
