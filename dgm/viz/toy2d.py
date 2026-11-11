from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch

from ._mpl import plt

_PT = dict(s=3, alpha=0.5, linewidths=0)


def _np(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()


def _grid(bounds: tuple[float, float], n: int) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    lo, hi = bounds
    xs = np.linspace(lo, hi, n)
    X, Y = np.meshgrid(xs, xs)
    pts = torch.from_numpy(np.stack([X.ravel(), Y.ravel()], 1)).float()
    return X, Y, pts


def _style(ax, bounds, title: str | None = None):
    ax.set_xlim(*bounds)
    ax.set_ylim(*bounds)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)


@torch.no_grad()
def _density_panel(
    ax, log_prob_fn: Callable[[torch.Tensor], torch.Tensor], bounds, title, grid_n=200, device="cpu"
):
    X, Y, pts = _grid(bounds, grid_n)
    lp = log_prob_fn(pts.to(device)).float().cpu().reshape(grid_n, grid_n).numpy()
    dens = np.exp(lp - np.nanmax(lp))
    ax.imshow(dens, origin="lower", extent=(*bounds, *bounds), cmap="viridis", aspect="equal")
    _style(ax, bounds, title)


def plot_samples(
    real: torch.Tensor,
    gen: torch.Tensor,
    bounds: tuple[float, float],
    out_path: str | Path,
    title: str | None = None,
    dataset=None,
    model=None,
    grid_n: int = 200,
) -> Path:
    real, gen = _np(real), _np(gen)
    panels: list[Callable] = []
    panels.append(
        lambda ax: (
            ax.scatter(real[:, 0], real[:, 1], c="k", **_PT),
            _style(ax, bounds, f"real (n={len(real)})"),
        )
    )
    panels.append(
        lambda ax: (
            ax.scatter(gen[:, 0], gen[:, 1], c="C3", **_PT),
            _style(ax, bounds, f"generated (n={len(gen)})"),
        )
    )

    def hist(ax):
        ax.hist2d(
            gen[:, 0], gen[:, 1], bins=100, range=[list(bounds), list(bounds)], cmap="viridis"
        )
        _style(ax, bounds, "generated density")

    panels.append(hist)
    if dataset is not None and getattr(dataset, "has_log_prob", False):
        panels.append(
            lambda ax: _density_panel(ax, dataset.log_prob, bounds, "true density  p(x)", grid_n)
        )
    if model is not None and getattr(model, "has_log_prob", False):
        panels.append(
            lambda ax: _density_panel(
                ax, model.log_prob, bounds, "model density  p_θ(x)", grid_n, model.device
            )
        )

    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.4))
    for ax, fn in zip(np.atleast_1d(axes), panels):
        fn(ax)
    if title:
        fig.suptitle(title, fontsize=11)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_trajectories(
    traj: torch.Tensor,
    bounds: tuple[float, float],
    out_path: str | Path,
    n_show: int = 64,
    final: torch.Tensor | None = None,
    title: str | None = None,
) -> Path:
    tr = _np(traj[:, :n_show])
    T1, k, _ = tr.shape
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    if final is not None:
        f = _np(final)
        ax.scatter(f[:, 0], f[:, 1], c="lightgray", s=2, alpha=0.6, linewidths=0, zorder=0)
    cmap = plt.get_cmap("plasma")
    for i in range(k):
        pts = tr[:, i]
        for s in range(T1 - 1):
            ax.plot(
                pts[s : s + 2, 0],
                pts[s : s + 2, 1],
                color=cmap(s / max(T1 - 2, 1)),
                lw=0.8,
                alpha=0.8,
            )
    ax.scatter(tr[0, :, 0], tr[0, :, 1], c="gray", s=10, marker="o", label="x_T (noise)", zorder=3)
    ax.scatter(tr[-1, :, 0], tr[-1, :, 1], c="C3", s=12, marker="*", label="x_0 (sample)", zorder=4)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    _style(ax, bounds, title or f"sampling trajectories ({T1 - 1} steps, {k} shown)")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


@torch.no_grad()
def plot_vector_field(
    field_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    bounds: tuple[float, float],
    out_path: str | Path,
    times: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    grid_n: int = 24,
    samples: torch.Tensor | None = None,
    device="cpu",
    title: str | None = None,
) -> Path:
    X, Y, pts = _grid(bounds, grid_n)
    fig, axes = plt.subplots(1, len(times), figsize=(3.2 * len(times), 3.4))
    for ax, t in zip(np.atleast_1d(axes), times):
        tt = torch.full((pts.shape[0],), float(t), device=device)
        v = field_fn(pts.to(device), tt).float().cpu().numpy().reshape(grid_n, grid_n, 2)
        mag = np.linalg.norm(v, axis=-1)
        vn = v / (mag[..., None] + 1e-8)
        if samples is not None:
            s = _np(samples)
            ax.scatter(s[:, 0], s[:, 1], c="lightgray", s=2, alpha=0.5, linewidths=0)
        ax.quiver(
            X,
            Y,
            vn[..., 0],
            vn[..., 1],
            np.log10(mag + 1e-8),
            cmap="viridis",
            pivot="mid",
            scale=grid_n * 1.2,
        )
        _style(ax, bounds, f"t={t:.2f}  (|v| log10: {np.log10(mag.mean() + 1e-8):.1f})")
    if title:
        fig.suptitle(title, fontsize=11)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def animate_trajectory(
    traj: torch.Tensor,
    bounds: tuple[float, float],
    out_path: str | Path,
    real: torch.Tensor | None = None,
    fps: int = 20,
    max_frames: int = 100,
    n_show: int = 2000,
) -> Path:
    from matplotlib.animation import FuncAnimation, PillowWriter

    tr = _np(traj[:, :n_show])
    T1 = tr.shape[0]
    idx = np.unique(np.linspace(0, T1 - 1, min(max_frames, T1)).round().astype(int))
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    if real is not None:
        r = _np(real)
        ax.scatter(r[:, 0], r[:, 1], c="lightgray", s=2, alpha=0.5, linewidths=0)
    sc = ax.scatter(tr[0, :, 0], tr[0, :, 1], c="C3", s=4, alpha=0.7, linewidths=0)
    _style(ax, bounds)
    ttl = ax.set_title("")

    def update(i):
        sc.set_offsets(tr[i])
        ttl.set_text(f"step {i}/{T1 - 1}")
        return sc, ttl

    anim = FuncAnimation(fig, update, frames=idx, blit=False)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path


def plot_dataset(dataset, out_path: str | Path, n: int = 5000, grid_n: int = 200) -> Path:
    x = _np(dataset.sample(n, torch.Generator().manual_seed(0)))
    has_lp = getattr(dataset, "has_log_prob", False)
    fig, axes = plt.subplots(1, 2 if has_lp else 1, figsize=(6.6 if has_lp else 3.4, 3.4))
    axes = np.atleast_1d(axes)
    axes[0].scatter(x[:, 0], x[:, 1], c="k", **_PT)
    _style(axes[0], dataset.bounds, f"{dataset.name}  samples (n={n})")
    if has_lp:
        _density_panel(axes[1], dataset.log_prob, dataset.bounds, "true density", grid_n)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
