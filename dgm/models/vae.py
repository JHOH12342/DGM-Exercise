from __future__ import annotations

from .base import GenerativeModel
from .registry import register_model
import torch
from ..nets.mlp import MLP
from ..nets.conv import ConvEncoder, ConvDecoder

@register_model("vae")
class VAE(GenerativeModel):
    def __init__(self, data_shape, cfg=None):
        super().__init__(data_shape, cfg)
        self.batch_data = 100
        self.mlp = MLP(
            in_shape=data_shape,
            out_shape=data_shape
        )
        self.latent_dim = 2

        self.encoder = MLP(
            in_shape=data_shape,
            out_shape=2 * self.latent_dim,
            hidden=256,
            n_layers=3
        )

        self.decoder = MLP(
            in_shape=self.latent_dim,
            out_shape=data_shape,
            hidden=256,
            n_layers=3
        )

    def loss(self, x):
        encoder = self.encoder
        decoder = self.decoder
        params = encoder(x)
        mu, logvar = params.chunk(2, dim=1)
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mu + std * epsilon

        kl_loss = -0.5* (
            1 + logvar - mu.pow(2) - logvar.exp()
        ).sum(dim=1).mean()

        x_recon = decoder(z)

        recon_loss = (
            (x - x_recon).pow(2) / 2
        ).flatten(1).sum(dim=1).mean()

        loss = kl_loss+recon_loss
        return loss, {"vae_loss": loss}
    @torch.no_grad()
    def sample(self, n, return_trajectory=False):
        device = next(self.parameters()).device

        z = torch.randn(n, self.latent_dim, device=device)
        x = self.decoder(z)

        return x
