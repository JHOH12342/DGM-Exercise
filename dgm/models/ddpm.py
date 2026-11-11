from __future__ import annotations

from .base import GenerativeModel
from .registry import register_model
from ..nets.unet import UNet
from ..nets.mlp import TimeConditionedMLP

import numpy as np
import torch
import torch.nn as nn


@register_model("ddpm")
class DDPM(GenerativeModel):
    def __init__(self, data_shape, cfg=None):
        super().__init__(data_shape, cfg)
        self.data_shape = tuple(int(s) for s in data_shape)
        self.betas = torch.linspace(1e-4, 0.02, 1000)
        self.alphas = 1- self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        print(self.data_shape)
        if len(data_shape) > 1:
            self.net = UNet(
                in_ch=self.data_shape[0],
                out_ch=self.data_shape[0],
                resolution=self.data_shape[1]
            )
        self.mlp = TimeConditionedMLP(
            in_shape=self.data_shape,
            out_shape=self.data_shape
        )
        self.batch_size = 128

    def loss(self, x):
        alpha_bar = self.alpha_bar
        data_shape = self.data_shape
        batch_size = self.batch_size
        t = torch.randint(0, 1000, (x.shape[0],))

        alpha_bar_t = alpha_bar[t].reshape(-1, 1)
        epsilon = torch.randn(x.shape)
        x_t = torch.sqrt(alpha_bar_t)*x + torch.sqrt(1 - alpha_bar_t)*epsilon
        eps_pred = self.mlp(x_t, t)

        criterion = nn.MSELoss()
        loss = criterion(eps_pred, epsilon)

        return loss, {"some_metric": 0.0}

    @torch.no_grad()
    def sample(self, n, return_trajectory=False):
        data_shape = self.data_shape
        alpha_bar = self.alpha_bar
        alpha = self.alphas
        betas = self.betas
        x_T = torch.randn(n, *data_shape)

        trajectory = torch.zeros(1001, n, *data_shape)
        trajectory[0] = x_T

        for t in range(1000, 0, -1):

            eps = torch.zeros(n, *data_shape)
            if t > 1:
                eps = torch.randn(n, *data_shape)

            alpha_bar_t = alpha_bar[t-1]
            alpha_t = alpha[t-1]
            beta_t = betas[t-1]
            x_t = trajectory[1000-t]

            eps_pred = self.mlp(x_t, torch.tensor(t-1))
            x_t_1 = (x_t - (1-alpha_t) / (torch.sqrt(1-alpha_bar_t)) * eps_pred) / torch.sqrt(alpha_t) + torch.sqrt(beta_t)*eps
            
            trajectory[1001-t] = x_t_1
        if return_trajectory:
            return trajectory[-1], trajectory
        else:
            return trajectory[-1]
