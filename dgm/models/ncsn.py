
from __future__ import annotations

import math
import torch

from .base import GenerativeModel
from .registry import register_model
from ..nets.factory import build_time_conditioned_net


@register_model("ncsn")
class NCSN(GenerativeModel):
    def __init__(self, data_shape, cfg=None):
        super().__init__(data_shape, cfg)

        # Noise schedule: sigma_max -> sigma_min
        sigma_max = float(self.cfg.get("sigma_max", 1.0))
        sigma_min = float(self.cfg.get("sigma_min", 0.01))

        self.L = int(self.cfg.get("n_sigmas", 10))
        self.T = int(self.cfg.get("n_steps_each", 100))
        self.e = float(self.cfg.get("step_lr", 2e-5))

        self.register_buffer(
            "sigma_seq",
            torch.logspace(
                math.log10(sigma_max),
                math.log10(sigma_min),
                steps=self.L,
            )
        )
        self.network = build_time_conditioned_net(
            self.data_shape,
            self.cfg.get("net", None),
        )
    def loss(self, x):
        B = x.shape[0]
        t = torch.randint(
            0, self.L, (B,), device=x.device
        )
        sigma = self.sigma_seq[t]
        sigma_b = sigma.reshape(
            B, *([1] * (x.ndim - 1))
        )
        epsilon = torch.randn_like(x)
        x_bar = x + sigma_b * epsilon
        score_pred = self.network(x_bar, sigma)
        squared_error = (
            sigma_b * score_pred + epsilon
        ).pow(2)
        loss = 0.5 * squared_error.flatten(1).sum(dim=1).mean()

        return loss, {"loss": loss.detach().item()}

    @torch.no_grad()
    def sample(self, n, return_trajectory=False):
        # Initialize samples on the model device
        x = self.noise(n)
        trajectory = (
            [x.detach().cpu().clone()]
            if return_trajectory else None
        )
        sigma_L = self.sigma_seq[-1]
        for i in range(self.L):
            sigma = self.sigma_seq[i]

            alpha = self.e * (
                sigma / sigma_L
            ).pow(2)

            # Network conditioning: [batch_size]
            sigma_batch = sigma.expand(n)

            for _ in range(self.T):
                z = torch.randn_like(x)
                score = self.network(
                    x, sigma_batch
                )
                x = (
                    x
                    + alpha / 2 * score
                    + torch.sqrt(alpha) * z
                )
            if return_trajectory:
                trajectory.append(
                    x.detach().cpu().clone()
                )

        if return_trajectory:
            return x, torch.stack(trajectory, dim=0)

        return x
