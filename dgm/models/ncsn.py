from __future__ import annotations

from .base import GenerativeModel
from .registry import register_model
from ..nets.unet import UNet
from ..nets.mlp import TimeConditionedMLP

import torch

@register_model("ncsn")
class NCSN(GenerativeModel):
    def __init__(self, data_shape, cfg=None):
        super().__init__(data_shape, cfg)
        self.sigma_seq = torch.logspace(start=0, end=-2, steps=10, base=10)
        self.T = 100
        self.e = 0.0002
        self.L = 10
        self.data_shape = data_shape
        if len(data_shape) > 1:
            self.network = UNet(
                in_ch=data_shape[0],
                out_ch=data_shape[0],
                resolution=data_shape[1]
            )
        self.mlp = TimeConditionedMLP(
            in_shape=self.data_shape,
            out_shape=self.data_shape
        )
    def loss(self, x):
        sigma_seq = self.sigma_seq
        L = self.L
        t = torch.randint(1, L+1, (x.shape[0],))
        sigma = sigma_seq[t-1].reshape(x.shape[0], *([1] * (x.ndim - 1)))
        epsilon = torch.randn(x.shape)
        x_bar = x+sigma*epsilon

        score_pred = self.mlp(x_bar, sigma)
        squared_error = (sigma * score_pred + epsilon).pow(2)

        loss = 0.5 * squared_error.flatten(1).sum(dim=1).mean()

        return loss, {"loss": loss.detach().item()}
    @torch.no_grad()
    def sample(self, n, return_trajectory=False):
        x = torch.randn(n, *self.data_shape)
        trajectory = torch.zeros(1001, n, *self.data_shape)
        trajectory[0] = x
        for i in range(self.L):
            sigma = self.sigma_seq[i]
            sigma_L = self.sigma_seq[-1]
            alpha = self.e * sigma.pow(2) / sigma_L.pow(2)
            for t in range(self.T):
                z_t = torch.randn(x.shape)
                score = self.mlp(x, sigma)
                x = x + alpha/2*(score) + torch.sqrt(alpha)*z_t
                trajectory[t+1] = x
            trajectory[0] = x
        if return_trajectory:
            return trajectory[0], trajectory
        else:
            return trajectory[0]
