from __future__ import annotations

import math

import torch
from torch import nn

from .mlp import get_act


class ConvEncoder(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        out_dim: int = 128,
        ch: int = 64,
        resolution: int = 32,
        act: str = "lrelu",
        norm: str | None = "group",
        out_act: str | None = None,
    ):
        super().__init__()
        n_down = int(math.log2(resolution // 4))
        layers: list[nn.Module] = []
        c_in, c = in_ch, ch
        for i in range(n_down):
            layers.append(nn.Conv2d(c_in, c, 4, stride=2, padding=1))
            if norm == "group" and i > 0:
                layers.append(nn.GroupNorm(8, c))
            elif norm == "batch" and i > 0:
                layers.append(nn.BatchNorm2d(c))
            layers.append(get_act(act))
            c_in, c = c, min(c * 2, 512)
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(c_in * 4 * 4, out_dim)
        self.out_act = get_act(out_act) if out_act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        return self.out_act(self.fc(h.flatten(1)))


class ConvDecoder(nn.Module):
    def __init__(
        self,
        in_dim: int = 128,
        out_ch: int = 3,
        ch: int = 64,
        resolution: int = 32,
        act: str = "relu",
        norm: str | None = "group",
        out_act: str | None = "tanh",
    ):
        super().__init__()
        n_up = int(math.log2(resolution // 4))
        chans = [min(ch * 2**i, 512) for i in range(n_up)][::-1]
        self.c0 = chans[0]
        self.fc = nn.Linear(in_dim, self.c0 * 4 * 4)
        layers: list[nn.Module] = []
        c_in = self.c0
        for i in range(n_up):
            c_out = chans[i + 1] if i + 1 < n_up else out_ch
            layers.append(nn.ConvTranspose2d(c_in, c_out, 4, stride=2, padding=1))
            if i + 1 < n_up:
                if norm == "group":
                    layers.append(nn.GroupNorm(8, c_out))
                elif norm == "batch":
                    layers.append(nn.BatchNorm2d(c_out))
                layers.append(get_act(act))
            c_in = c_out
        self.deconv = nn.Sequential(*layers)
        self.out_act = get_act(out_act) if out_act else nn.Identity()
        self.out_shape = (out_ch, resolution, resolution)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).reshape(z.shape[0], self.c0, 4, 4)
        return self.out_act(self.deconv(h))
