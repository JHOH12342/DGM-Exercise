from __future__ import annotations

from .base import GenerativeModel
from .registry import register_model


@register_model("flow")
class FlowMatching(GenerativeModel):
    def __init__(self, data_shape, cfg=None):
        super().__init__(data_shape, cfg)
        raise NotImplementedError

    def loss(self, x):
        raise NotImplementedError

    def sample(self, n, return_trajectory=False):
        raise NotImplementedError
