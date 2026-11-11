from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .base import DataDistribution
from .registry import register_dataset


def _lenient_cifar10_cls():
    import torchvision.datasets as tvd

    class _LenientCIFAR10(tvd.CIFAR10):
        train_list = [[f, None] for f, _ in tvd.CIFAR10.train_list]
        test_list = [[f, None] for f, _ in tvd.CIFAR10.test_list]
        meta = {**tvd.CIFAR10.meta, "md5": None}

    return _LenientCIFAR10


class _XOnly(Dataset):
    def __init__(self, ds: Dataset):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        return self.ds[i][0]


@register_dataset("cifar10")
class CIFAR10(DataDistribution):
    shape = (3, 32, 32)
    is_image = True

    def __init__(
        self,
        root: str = "./data",
        download: bool = True,
        hflip: bool = True,
        classes: Sequence[int] | None = None,
        num_workers: int = 4,
        seed: int = 0,
    ):
        self.root = str(Path(root).expanduser())
        self.download = download
        self.hflip = hflip
        self.classes = list(classes) if classes is not None else None
        self.num_workers = num_workers
        self.seed = seed
        self._train = self._train_aug = self._test = None
        self._test_cache: torch.Tensor | None = None

    def _tf(self, aug: bool):
        import torchvision.transforms as T

        ops = [T.RandomHorizontalFlip()] if (aug and self.hflip) else []
        ops += [T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)]
        return T.Compose(ops)

    def _make(self, train: bool, aug: bool) -> Dataset:
        ds = _lenient_cifar10_cls()(self.root, train=train, download=self.download, transform=self._tf(aug))
        if self.classes is not None:
            keep = set(self.classes)
            idx = [i for i, y in enumerate(ds.targets) if y in keep]
            ds = Subset(ds, idx)
        return _XOnly(ds)

    @property
    def train_plain(self) -> Dataset:
        if self._train is None:
            self._train = self._make(train=True, aug=False)
        return self._train

    @property
    def train_aug(self) -> Dataset:
        if self._train_aug is None:
            self._train_aug = self._make(train=True, aug=True)
        return self._train_aug

    @property
    def test(self) -> Dataset:
        if self._test is None:
            self._test = self._make(train=False, aug=False)
        return self._test

    def sample(self, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        ds = self.train_plain
        idx = torch.randperm(len(ds), generator=generator)[:n].tolist()
        return torch.stack([ds[i] for i in idx])

    def test_set(self, n: int) -> torch.Tensor:
        n = min(n, len(self.test))
        if self._test_cache is None or self._test_cache.shape[0] < n:
            loader = DataLoader(self.test, batch_size=500, shuffle=False, num_workers=min(4, self.num_workers))
            xs, got = [], 0
            for x in loader:
                xs.append(x)
                got += x.shape[0]
                if got >= n:
                    break
            self._test_cache = torch.cat(xs)[:n]
        return self._test_cache[:n]

    def train_iterator(self, batch_size: int, num_workers: int | None = None, **_) -> Iterator[torch.Tensor]:
        nw = self.num_workers if num_workers is None else num_workers
        gen = torch.Generator().manual_seed(self.seed + 1)
        loader = DataLoader(
            self.train_aug,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=nw,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=nw > 0,
            generator=gen,
        )
        while True:
            yield from loader

    def to_display(self, x: torch.Tensor) -> torch.Tensor:
        return (x.clamp(-1, 1) + 1) / 2
