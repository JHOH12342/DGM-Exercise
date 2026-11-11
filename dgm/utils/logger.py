from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np


class MetricLogger:
    def __init__(
        self,
        out_dir: str | Path,
        tensorboard: bool = True,
        wandb_cfg: dict | None = None,
        run_cfg: dict | None = None,
        run_name: str | None = None,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl = open(self.out_dir / "metrics.jsonl", "a", buffering=1)
        self._t0 = time.time()

        self._tb = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tb = SummaryWriter(log_dir=str(self.out_dir / "tb"))
            except ImportError:
                print("[logger] tensorboard not installed; skipping TB logging")

        self._wandb = None
        wandb_cfg = wandb_cfg or {}
        if wandb_cfg.get("enabled", False):
            try:
                import wandb

                self._wandb = wandb.init(
                    project=wandb_cfg.get("project", "dgm"),
                    entity=wandb_cfg.get("entity"),
                    name=run_name,
                    config=run_cfg,
                    dir=str(self.out_dir),
                    mode=wandb_cfg.get("mode", "online"),
                )
            except ImportError:
                print("[logger] wandb not installed; skipping wandb logging")

    def log_scalars(self, metrics: dict[str, Any], step: int, prefix: str = "") -> None:
        clean: dict[str, float] = {}
        for k, v in metrics.items():
            if v is None:
                continue
            if hasattr(v, "item"):
                v = v.item()
            if isinstance(v, (int, float, np.floating, np.integer)) and np.isfinite(v):
                clean[f"{prefix}{k}"] = float(v)
        if not clean:
            return
        rec = {"step": int(step), "time": round(time.time() - self._t0, 2), **clean}
        self._jsonl.write(json.dumps(rec) + "\n")
        if self._tb is not None:
            for k, v in clean.items():
                self._tb.add_scalar(k, v, step)
        if self._wandb is not None:
            self._wandb.log(clean, step=step)

    def log_image_file(self, tag: str, path: str | Path, step: int) -> None:
        path = Path(path)
        if not path.exists() or path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            return
        if self._tb is None and self._wandb is None:
            return
        from PIL import Image

        img = np.asarray(Image.open(path).convert("RGB"))
        if self._tb is not None:
            self._tb.add_image(tag, img, step, dataformats="HWC")
        if self._wandb is not None:
            import wandb

            self._wandb.log({tag: wandb.Image(img)}, step=step)

    def close(self) -> None:
        self._jsonl.close()
        if self._tb is not None:
            self._tb.close()
        if self._wandb is not None:
            self._wandb.finish()


def format_scalars(metrics: dict[str, Any]) -> str:
    parts = []
    for k, v in metrics.items():
        if hasattr(v, "item"):
            v = v.item()
        if isinstance(v, float):
            parts.append(f"{k}={v:.4g}")
        else:
            parts.append(f"{k}={v}")
    return " ".join(parts)
