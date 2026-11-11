from __future__ import annotations

import copy

import torch
from torch import nn


class EMA:
    def __init__(
        self, model: nn.Module, decay: float = 0.999, start_step: int = 0, warmup: bool = True
    ):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.start_step = start_step
        self.warmup = warmup
        self.n_updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        if step < self.start_step:
            self._copy(model)
            return
        d = self.decay
        if self.warmup:
            d = min(d, (1 + self.n_updates) / (10 + self.n_updates))
        self.n_updates += 1
        for e, p in zip(self.module.parameters(), model.parameters()):
            e.lerp_(p.detach(), 1.0 - d)
        for eb, b in zip(self.module.buffers(), model.buffers()):
            eb.copy_(b)

    @torch.no_grad()
    def _copy(self, model: nn.Module) -> None:
        for e, p in zip(self.module.parameters(), model.parameters()):
            e.copy_(p.detach())
        for eb, b in zip(self.module.buffers(), model.buffers()):
            eb.copy_(b)

    def state_dict(self) -> dict:
        return {"module": self.module.state_dict(), "n_updates": self.n_updates}

    def load_state_dict(self, sd: dict) -> None:
        self.module.load_state_dict(sd["module"])
        self.n_updates = sd.get("n_updates", 0)
