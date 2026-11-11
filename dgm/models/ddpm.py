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
        betas = torch.linspace(1e-4, 0.02, 1000)
        alphas = 1- betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)

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
        batch_size = x.shape[0]

        t = torch.randint(
            0,
            len(self.betas),
            (batch_size,),
            device=x.device
        )

        alpha_bar_t = self.alpha_bar[t].view(
            batch_size, *([1] * (x.ndim - 1))
        )

        epsilon = torch.randn_like(x)

        x_t = (
            torch.sqrt(alpha_bar_t) * x
            + torch.sqrt(1 - alpha_bar_t) * epsilon
        )

        eps_pred = self.net(x_t, t)

        loss = torch.nn.functional.mse_loss(eps_pred, epsilon)

        return loss, {"some_metric": 0.0}

    @torch.no_grad()
    def sample(self, n, return_trajectory=False):
        device = next(self.parameters()).device

        alpha_bar = self.alpha_bar
        alpha = self.alphas
        betas = self.betas

        # Initial Gaussian noise
        x_t = torch.randn(n, *self.data_shape, device=device)

        if return_trajectory:
            trajectory = [x_t.detach().cpu().clone()]

        for t in range(999, -1, -1):

            # Time indices: (batch_size,)
            t_batch = torch.full(
                (n,),
                t,
                device=device,
                dtype=torch.long
            )

            alpha_bar_t = alpha_bar[t]
            alpha_t = alpha[t]
            beta_t = betas[t]

            # Predict noise
            eps_pred = self.mlp(x_t, t_batch)

            # Reverse diffusion mean
            mean = (
                x_t
                - (1 - alpha_t)
                / torch.sqrt(1 - alpha_bar_t)
                * eps_pred
            ) / torch.sqrt(alpha_t)

            # Stochastic noise
            if t > 0:
                z = torch.randn_like(x_t)
                x_t = mean + torch.sqrt(beta_t) * z
            else:
                x_t = mean

            if return_trajectory:
                trajectory.append(x_t.detach().cpu().clone())

        if return_trajectory:
            trajectory = torch.stack(trajectory, dim=0)
            return x_t, trajectory

        return x_t
