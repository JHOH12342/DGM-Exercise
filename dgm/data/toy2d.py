from __future__ import annotations

import math
from itertools import product
from typing import Iterator

import numpy as np
import torch

from .base import DataDistribution
from .registry import register_dataset

_NORM_FIT_N = 200_000


class Toy2D(DataDistribution):
    shape = (2,)
    is_image = False

    def __init__(self, n_train: int | None = None, seed: int = 0, normalize: bool = True):
        self.n_train = n_train
        self.seed = seed
        self._mean = torch.zeros(2)
        self._scale = 1.0
        if normalize:
            gen = torch.Generator().manual_seed(seed + 777)
            raw = self._sample_raw(_NORM_FIT_N, gen)
            self._mean = raw.mean(0)
            self._scale = float(raw.std(0).pow(2).mean().sqrt())
        with torch.no_grad():
            z = self.sample(20_000, torch.Generator().manual_seed(seed + 778))
            b = float(z.abs().max()) * 1.1
        self.bounds = (-math.ceil(b * 2) / 2, math.ceil(b * 2) / 2)
        self._train_data: torch.Tensor | None = None

    def _sample_raw(self, n: int, generator: torch.Generator) -> torch.Tensor:
        raise NotImplementedError

    def _log_prob_raw(self, x_raw: torch.Tensor) -> torch.Tensor | None:
        return None

    def normalize(self, x_raw: torch.Tensor) -> torch.Tensor:
        return (x_raw - self._mean.to(x_raw)) / self._scale

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self._scale + self._mean.to(x)

    def sample(self, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        if generator is None:
            generator = torch.Generator().manual_seed(int(torch.randint(0, 2**31 - 1, (1,))))
        return self.normalize(self._sample_raw(n, generator)).float()

    def log_prob(self, x: torch.Tensor) -> torch.Tensor | None:
        lp = self._log_prob_raw(self.unnormalize(x))
        if lp is None:
            return None
        return lp + 2.0 * math.log(self._scale)

    @property
    def has_log_prob(self) -> bool:
        return type(self)._log_prob_raw is not Toy2D._log_prob_raw

    def train_iterator(self, batch_size: int, **_) -> Iterator[torch.Tensor]:
        gen = torch.Generator().manual_seed(self.seed + 1)
        if self.n_train is None:
            while True:
                yield self.sample(batch_size, gen)
        if self._train_data is None:
            self._train_data = self.sample(self.n_train, gen)
        data = self._train_data
        n = data.shape[0]
        if batch_size > n:
            raise ValueError(f"batch_size {batch_size} > n_train {n}")
        while True:
            perm = torch.randperm(n, generator=gen)
            for i in range(0, n - batch_size + 1, batch_size):
                yield data[perm[i : i + batch_size]]

    @staticmethod
    def _np_seed(generator: torch.Generator) -> int:
        return int(torch.randint(0, 2**31 - 1, (1,), generator=generator))


@register_dataset("gaussian")
class Gaussian2D(Toy2D):

    def __init__(self, std: tuple[float, float] = (1.0, 0.3), rotation_deg: float = 30.0, **kw):
        th = math.radians(rotation_deg)
        R = torch.tensor([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        self._cov_sqrt = R @ torch.diag(torch.tensor(std, dtype=torch.float32))
        self._cov = self._cov_sqrt @ self._cov_sqrt.T
        super().__init__(**kw)

    def _sample_raw(self, n, generator):
        return torch.randn(n, 2, generator=generator) @ self._cov_sqrt.T

    def _log_prob_raw(self, x_raw):
        dist = torch.distributions.MultivariateNormal(torch.zeros(2, device=x_raw.device), self._cov.to(x_raw))
        return dist.log_prob(x_raw)


@register_dataset("gmm")
class GridGMM(Toy2D):

    def __init__(self, grid_size: int = 5, spacing: float = 2.0, std: float = 0.05, **kw):
        g = grid_size
        axis = (torch.arange(g, dtype=torch.float32) - (g - 1) / 2) * spacing
        self._centers_raw = torch.tensor(list(product(axis.tolist(), axis.tolist())), dtype=torch.float32)
        self._std_raw = float(std)
        super().__init__(**kw)
        self.centers = self.normalize(self._centers_raw)
        self.std = self._std_raw / self._scale
        self.n_modes = self.centers.shape[0]

    def _sample_raw(self, n, generator):
        idx = torch.randint(0, self._centers_raw.shape[0], (n,), generator=generator)
        return self._centers_raw[idx] + self._std_raw * torch.randn(n, 2, generator=generator)

    def _log_prob_raw(self, x_raw):
        c = self._centers_raw.to(x_raw)
        d2 = torch.cdist(x_raw, c).pow(2)
        log_comp = -0.5 * d2 / self._std_raw**2 - math.log(2 * math.pi * self._std_raw**2)
        return torch.logsumexp(log_comp, dim=1) - math.log(c.shape[0])


@register_dataset("halfmoon")
class HalfMoon(Toy2D):
    def __init__(self, noise: float = 0.1, **kw):
        self.noise = noise
        super().__init__(**kw)

    def _sample_raw(self, n, generator):
        from sklearn.datasets import make_moons

        x, _ = make_moons(n, noise=self.noise, random_state=self._np_seed(generator))
        return torch.from_numpy(x.astype(np.float32))


@register_dataset("swissroll")
class SwissRoll(Toy2D):
    def __init__(self, noise: float = 0.5, **kw):
        self.noise = noise
        super().__init__(**kw)

    def _sample_raw(self, n, generator):
        from sklearn.datasets import make_swiss_roll

        x, _ = make_swiss_roll(n, noise=self.noise, random_state=self._np_seed(generator))
        return torch.from_numpy(x[:, [0, 2]].astype(np.float32))


@register_dataset("circles")
class Circles(Toy2D):
    def __init__(self, noise: float = 0.05, factor: float = 0.5, **kw):
        self.noise, self.factor = noise, factor
        super().__init__(**kw)

    def _sample_raw(self, n, generator):
        from sklearn.datasets import make_circles

        x, _ = make_circles(n, noise=self.noise, factor=self.factor, random_state=self._np_seed(generator))
        return torch.from_numpy(x.astype(np.float32))


@register_dataset("checkerboard")
class Checkerboard(Toy2D):

    def _sample_raw(self, n, generator):
        x1 = torch.rand(n, generator=generator) * 4 - 2
        x2_ = torch.rand(n, generator=generator) - torch.randint(0, 2, (n,), generator=generator) * 2
        x2 = x2_ + (torch.floor(x1) % 2)
        return torch.stack([x1, x2], dim=1)

    def _log_prob_raw(self, x_raw):
        x1, x2 = x_raw[:, 0], x_raw[:, 1]
        inside = (x1.abs() <= 2) & (x2.abs() <= 2)
        parity = ((torch.floor(x1) + torch.floor(x2)) % 2) == 0
        on = inside & parity
        return torch.where(on, torch.full_like(x1, -math.log(8.0)), torch.full_like(x1, -1e4))


@register_dataset("spiral")
class TwoSpirals(Toy2D):
    def __init__(self, noise: float = 0.1, n_turns: float = 1.5, **kw):
        self.noise, self.n_turns = noise, n_turns
        super().__init__(**kw)

    def _sample_raw(self, n, generator):
        t = torch.sqrt(torch.rand(n, generator=generator)) * 2 * math.pi * self.n_turns
        sign = torch.randint(0, 2, (n,), generator=generator).float() * 2 - 1
        r = t
        x = torch.stack([r * torch.cos(t), r * torch.sin(t)], dim=1) * sign[:, None]
        return x + self.noise * torch.randn(n, 2, generator=generator)
