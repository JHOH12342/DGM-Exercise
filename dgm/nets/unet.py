from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .embeddings import TimeEmbedding


def _gn(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(32, ch // 4) if ch >= 8 else 1, ch)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, temb_dim: int, dropout: float):
        super().__init__()
        self.norm1, self.conv1 = _gn(in_ch), nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.temb = nn.Linear(temb_dim, out_ch)
        self.norm2, self.conv2 = _gn(out_ch), nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb(F.silu(temb))[:, :, None, None]
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    def __init__(self, ch: int, n_heads: int = 4):
        super().__init__()
        self.norm = _gn(ch)
        self.qkv = nn.Conv2d(ch, 3 * ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.n_heads = n_heads
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = (
            self.qkv(self.norm(x)).reshape(b, 3, self.n_heads, c // self.n_heads, h * w).unbind(1)
        )
        out = F.scaled_dot_product_attention(
            q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2)
        )
        out = out.transpose(-1, -2).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x, temb=None):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x, temb=None):
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class UNet(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int | None = None,
        ch: int = 64,
        ch_mult: tuple[int, ...] = (1, 2, 2, 2),
        num_res_blocks: int = 2,
        attn_resolutions: tuple[int, ...] = (16,),
        dropout: float = 0.1,
        resolution: int = 32,
        time_scale: float = 1000.0,
    ):
        super().__init__()
        out_ch = out_ch or in_ch
        temb_dim = ch * 4
        self.time_emb = TimeEmbedding(ch, temb_dim, time_scale=time_scale)
        self.inp = nn.Conv2d(in_ch, ch, 3, padding=1)

        self.down = nn.ModuleList()
        chans = [ch]
        cur, res = ch, resolution
        for lvl, m in enumerate(ch_mult):
            for _ in range(num_res_blocks):
                blk = nn.ModuleList([ResBlock(cur, ch * m, temb_dim, dropout)])
                cur = ch * m
                if res in attn_resolutions:
                    blk.append(AttnBlock(cur))
                self.down.append(blk)
                chans.append(cur)
            if lvl != len(ch_mult) - 1:
                self.down.append(nn.ModuleList([Downsample(cur)]))
                chans.append(cur)
                res //= 2

        self.mid = nn.ModuleList(
            [
                ResBlock(cur, cur, temb_dim, dropout),
                AttnBlock(cur),
                ResBlock(cur, cur, temb_dim, dropout),
            ]
        )

        self.up = nn.ModuleList()
        for lvl, m in reversed(list(enumerate(ch_mult))):
            for _ in range(num_res_blocks + 1):
                blk = nn.ModuleList([ResBlock(cur + chans.pop(), ch * m, temb_dim, dropout)])
                cur = ch * m
                if res in attn_resolutions:
                    blk.append(AttnBlock(cur))
                self.up.append(blk)
            if lvl != 0:
                self.up.append(nn.ModuleList([Upsample(cur)]))
                res *= 2

        self.out_norm = _gn(cur)
        self.out = nn.Conv2d(cur, out_ch, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    @staticmethod
    def _run(blocks: nn.ModuleList, h: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        for m in blocks:
            h = m(h, temb) if isinstance(m, (ResBlock, Downsample, Upsample)) else m(h)
        return h

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = t.reshape(-1).float()
        if t.numel() == 1 and x.shape[0] > 1:
            t = t.expand(x.shape[0])
        temb = self.time_emb(t)
        h = self.inp(x)
        hs = [h]
        for blk in self.down:
            h = self._run(blk, h, temb)
            hs.append(h)
        h = self._run(self.mid, h, temb)
        for blk in self.up:
            if isinstance(blk[0], ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = self._run(blk, h, temb)
        return self.out(F.silu(self.out_norm(h)))
