from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torchvision.utils import make_grid, save_image

from ._mpl import plt


def save_grid(
    x_display: torch.Tensor, out_path: str | Path, nrow: int = 8, padding: int = 2
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(x_display.detach().float().cpu().clamp(0, 1), out_path, nrow=nrow, padding=padding)
    return out_path


def _grid_np(x: torch.Tensor, nrow: int) -> np.ndarray:
    g = make_grid(x.detach().float().cpu().clamp(0, 1), nrow=nrow, padding=2, pad_value=1.0)
    return g.permute(1, 2, 0).numpy()


def plot_real_vs_gen(
    real_display: torch.Tensor,
    gen_display: torch.Tensor,
    out_path: str | Path,
    nrow: int = 8,
    title: str | None = None,
) -> Path:
    n = min(real_display.shape[0], gen_display.shape[0], nrow * nrow)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.8))
    for ax, x, name in zip(axes, (real_display[:n], gen_display[:n]), ("real", "generated")):
        ax.imshow(_grid_np(x, nrow))
        ax.set_title(name, fontsize=10)
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=11)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_trajectory_strip(
    traj_display: torch.Tensor, out_path: str | Path, n_rows: int = 4, n_cols: int = 10
) -> Path:
    T1, n = traj_display.shape[:2]
    rows = min(n_rows, n)
    t_idx = np.unique(np.linspace(0, T1 - 1, min(n_cols, T1)).round().astype(int))
    tiles = traj_display[t_idx][:, :rows]
    tiles = tiles.permute(1, 0, 2, 3, 4).reshape(-1, *traj_display.shape[2:])
    fig, ax = plt.subplots(figsize=(1.0 * len(t_idx), 1.0 * rows + 0.4))
    ax.imshow(_grid_np(tiles, nrow=len(t_idx)))
    ax.set_title(f"sampling trajectory  (t = {t_idx[0]} ... {t_idx[-1]} of {T1 - 1})", fontsize=9)
    ax.axis("off")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
