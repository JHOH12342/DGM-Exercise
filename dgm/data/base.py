from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Iterator

import torch


class DataDistribution(ABC):

    name: str = "base"
    shape: tuple[int, ...] = ()
    is_image: bool = False
    bounds: tuple[float, float] | None = None

    @property
    def dim(self) -> int:
        return int(math.prod(self.shape))

    @abstractmethod
    def sample(self, n: int, generator: torch.Generator | None = None) -> torch.Tensor: ...

    @abstractmethod
    def train_iterator(self, batch_size: int, **kwargs) -> Iterator[torch.Tensor]: ...

    def test_set(self, n: int) -> torch.Tensor:
        gen = torch.Generator().manual_seed(2**31 - 1)
        return self.sample(n, generator=gen)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor | None:
        return None

    @property
    def has_log_prob(self) -> bool:
        return type(self).log_prob is not DataDistribution.log_prob

    def to_display(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, shape={self.shape})"
