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
        if len(self.data_shape) == 3:
            self.net = UNet(
                in_ch=self.data_shape[0],
                out_ch=self.data_shape[0],
                ch=128,
                ch_mult=(1, 2, 2, 2),
                num_res_blocks=2,
                attn_resolutions=(16,),
                dropout=0.1,
                resolution=self.data_shape[1],
                time_scale=1.0
            )
        else:
            self.net = TimeConditionedMLP(
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
        device = self.betas.device
        T = len(self.betas)

        alpha_bar = self.alpha_bar
        alphas = self.alphas
        betas = self.betas

        was_training = self.training
        self.eval()

        try:
            x_t = torch.randn(
                n, *self.data_shape,
                device=device
            )

            if return_trajectory:
                trajectory = [x_t.cpu().clone()]

            for t in range(T - 1, -1, -1):
                # Batch-wise timestep
                t_batch = torch.full(
                    (n,),
                    t,
                    device=device,
                    dtype=torch.long
                )

                alpha_bar_t = alpha_bar[t]
                alpha_t = alphas[t]
                beta_t = betas[t]

                # Predict noise
                eps_pred = self.net(x_t, t_batch)

                # Reverse process mean
                mean = (
                    x_t
                    - beta_t
                    / torch.sqrt(1 - alpha_bar_t)
                    * eps_pred
                ) / torch.sqrt(alpha_t)

                # Reverse diffusion
                if t > 0:
                    eps = torch.randn_like(x_t)
                    x_t = mean + torch.sqrt(beta_t) * eps
                else:
                    x_t = mean

                if return_trajectory:
                    trajectory.append(x_t.cpu().clone())

            if return_trajectory:
                return x_t, torch.stack(trajectory, dim=0)

            return x_t

        finally:
            if was_training:
                self.train()

