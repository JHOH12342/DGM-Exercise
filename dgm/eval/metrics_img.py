from __future__ import annotations

import warnings

import torch


class ImageMetrics:
    def __init__(
        self,
        dataset,
        n_ref: int = 10_000,
        device: torch.device | str = "cpu",
        feature: int = 2048,
        inception_score: bool = True,
        batch_size: int = 250,
    ):
        self.dataset = dataset
        self.n_ref = n_ref
        self.device = torch.device(device)
        self.batch_size = batch_size
        self._ref_done = False
        self.available = False
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance
            from torchmetrics.image.inception import InceptionScore
        except ImportError:
            warnings.warn(
                "torchmetrics[image] not installed -> FID/IS disabled (pip install -e '.[image]')"
            )
            self.fid = self.inc = None
            return
        self.fid = FrechetInceptionDistance(
            feature=feature, normalize=False, reset_real_features=False
        ).to(self.device)
        self.inc = InceptionScore(normalize=False).to(self.device) if inception_score else None
        self.available = True

    def _to_uint8(self, x_model_space: torch.Tensor) -> torch.Tensor:
        x = self.dataset.to_display(x_model_space.float())
        return (x * 255).round().clamp(0, 255).to(torch.uint8)

    @torch.no_grad()
    def _ensure_ref(self) -> None:
        if self._ref_done:
            return
        ref = self.dataset.test_set(self.n_ref)
        for i in range(0, ref.shape[0], self.batch_size):
            self.fid.update(self._to_uint8(ref[i : i + self.batch_size]).to(self.device), real=True)
        self._ref_done = True

    @torch.no_grad()
    def compute(self, x_gen: torch.Tensor) -> dict[str, float]:
        if not self.available:
            return {}
        self._ensure_ref()
        self.fid.reset()
        if self.inc is not None:
            self.inc.reset()
        for i in range(0, x_gen.shape[0], self.batch_size):
            xb = self._to_uint8(x_gen[i : i + self.batch_size]).to(self.device)
            self.fid.update(xb, real=False)
            if self.inc is not None:
                self.inc.update(xb)
        out = {"fid": float(self.fid.compute())}
        if self.inc is not None:
            m, s = self.inc.compute()
            out["is_mean"], out["is_std"] = float(m), float(s)
        return out
