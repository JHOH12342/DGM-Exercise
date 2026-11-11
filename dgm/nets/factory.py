from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from omegaconf import OmegaConf
from torch import nn

from .conv import ConvDecoder, ConvEncoder
from .mlp import MLP, TimeConditionedMLP
from .unet import UNet


def _kw(net_cfg: Mapping[str, Any] | None, key: str) -> dict:
    if net_cfg is None:
        return {}
    if OmegaConf.is_config(net_cfg):
        net_cfg = OmegaConf.to_container(net_cfg, resolve=True)
    sub = net_cfg.get(key) or {}
    return {k: (tuple(v) if isinstance(v, list) else v) for k, v in dict(sub).items()}


def is_image_shape(data_shape: tuple[int, ...]) -> bool:
    return len(data_shape) == 3


def build_time_conditioned_net(
    data_shape: tuple[int, ...], net_cfg=None, out_shape=None, **overrides
) -> nn.Module:
    data_shape = tuple(data_shape)
    if is_image_shape(data_shape):
        c, h, w = data_shape
        out_ch = data_shape[0] if out_shape is None else out_shape[0]
        kw = {**_kw(net_cfg, "unet"), **overrides}
        kw.setdefault("resolution", h)
        return UNet(in_ch=c, out_ch=out_ch, **kw)
    kw = {**_kw(net_cfg, "mlp"), **overrides}
    return TimeConditionedMLP(data_shape, out_shape, **kw)


def build_encoder(
    data_shape: tuple[int, ...], out_dim: int, net_cfg=None, **overrides
) -> nn.Module:
    data_shape = tuple(data_shape)
    if is_image_shape(data_shape):
        kw = {**_kw(net_cfg, "conv"), **overrides}
        kw.setdefault("resolution", data_shape[1])
        return ConvEncoder(in_ch=data_shape[0], out_dim=out_dim, **kw)
    kw = {**_kw(net_cfg, "mlp"), **overrides}
    kw.pop("time_dim", None)
    kw.pop("time_scale", None)
    return MLP(data_shape, out_dim, **kw)


def build_decoder(
    data_shape: tuple[int, ...], in_dim: int, net_cfg=None, out_act: str | None = None, **overrides
) -> nn.Module:
    data_shape = tuple(data_shape)
    if is_image_shape(data_shape):
        kw = {**_kw(net_cfg, "conv"), **overrides}
        kw.setdefault("resolution", data_shape[1])
        return ConvDecoder(in_dim=in_dim, out_ch=data_shape[0], out_act=out_act, **kw)
    kw = {**_kw(net_cfg, "mlp"), **overrides}
    kw.pop("time_dim", None)
    kw.pop("time_scale", None)
    return MLP(in_dim, data_shape, out_act=out_act, **kw)
