from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from ._mpl import plt


def read_metrics(jsonl_path: str | Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            step = rec.get("step", 0)
            for k, v in rec.items():
                if k in ("step", "time") or not isinstance(v, (int, float)):
                    continue
                series[k].append((step, v))
    return {
        k: (np.array([s for s, _ in v]), np.array([x for _, x in v])) for k, v in series.items()
    }


def _smooth(y: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(y) < w:
        return y
    k = np.ones(w) / w
    return np.convolve(y, k, mode="valid")


_DEFAULT_SKIP = (
    "eval/modes_total",
    "eval/sample_time_s",
    "eval/nonfinite_frac",
    "train/steps_per_s",
)


def plot_training_curves(
    jsonl_path: str | Path,
    out_path: str | Path,
    keys: list[str] | None = None,
    smooth: int = 1,
    ncols: int = 4,
) -> Path | None:
    series = read_metrics(jsonl_path)
    if keys is None:
        keys = [
            k for k in sorted(series) if k not in _DEFAULT_SKIP and not k.startswith("train/lr_")
        ]
    keys = [k for k in keys if k in series]
    if not keys:
        return None
    nrows = math.ceil(len(keys) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.6 * nrows), squeeze=False)
    for ax, k in zip(axes.ravel(), keys):
        s, y = series[k]
        if k.startswith("train/") and smooth > 1:
            ys = _smooth(y, smooth)
            ax.plot(s[len(s) - len(ys) :], ys, lw=1.2)
            ax.plot(s, y, lw=0.5, alpha=0.3)
        else:
            ax.plot(s, y, lw=1.2, marker="." if len(s) < 50 else None)
        ax.set_title(k, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(keys) :]:
        ax.axis("off")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
