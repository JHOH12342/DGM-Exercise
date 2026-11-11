from __future__ import annotations

import math

import torch
from torch import nn

from .embeddings import TimeEmbedding

_ACTS = {
    "silu": nn.SiLU,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "lrelu": lambda: nn.LeakyReLU(0.2),
}


def get_act(name: str) -> nn.Module:
    return _ACTS[name]()


class MLP(nn.Module):
    def __init__(
        self,
        in_shape: int | tuple[int, ...],
        out_shape: int | tuple[int, ...],
        hidden: int = 256,
        n_layers: int = 3,
        act: str = "silu",
        out_act: str | None = None,
    ):
        super().__init__()
        self.in_shape = (in_shape,) if isinstance(in_shape, int) else tuple(in_shape)
        self.out_shape = (out_shape,) if isinstance(out_shape, int) else tuple(out_shape)
        in_dim, out_dim = math.prod(self.in_shape), math.prod(self.out_shape)
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), get_act(act)]
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        if out_act is not None:
            layers.append(get_act(out_act))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        return self.net(x.reshape(b, -1)).reshape(b, *self.out_shape)


class TimeConditionedMLP(nn.Module):
    def __init__(
        self,
        in_shape: int | tuple[int, ...],
        out_shape: int | tuple[int, ...] | None = None,
        hidden: int = 256,
        n_layers: int = 4,
        time_dim: int = 128,
        time_scale: float = 1000.0,
        act: str = "silu",
    ):
        super().__init__()
        self.in_shape = (in_shape,) if isinstance(in_shape, int) else tuple(in_shape)
        out_shape = self.in_shape if out_shape is None else out_shape
        self.out_shape = (out_shape,) if isinstance(out_shape, int) else tuple(out_shape)
        in_dim, out_dim = math.prod(self.in_shape), math.prod(self.out_shape)

        self.time_emb = TimeEmbedding(time_dim, hidden, time_scale=time_scale)
        self.inp = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        self.t_proj = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        self.act = get_act(act)
        self.out = nn.Linear(hidden, out_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        t = t.reshape(-1).float()
        if t.numel() == 1 and b > 1:
            t = t.expand(b)
        temb = self.time_emb(t)
        h = self.act(self.inp(x.reshape(b, -1)))
        for lin, tp in zip(self.blocks, self.t_proj):
            h = self.act(lin(h) + tp(temb)) + h
        return self.out(h).reshape(b, *self.out_shape)
