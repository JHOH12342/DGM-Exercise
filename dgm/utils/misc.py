from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only))


@torch.no_grad()
def sample_in_batches(
    sample_fn: Callable[..., torch.Tensor | tuple[torch.Tensor, torch.Tensor]],
    n: int,
    batch_size: int,
    return_trajectory: bool = False,
    trajectory_max: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    xs, trajs = [], []
    kept = 0
    remaining = n
    while remaining > 0:
        b = min(batch_size, remaining)
        want_traj = return_trajectory and (trajectory_max is None or kept < trajectory_max)
        out = sample_fn(b, return_trajectory=want_traj) if want_traj else sample_fn(b)
        if isinstance(out, tuple):
            x, traj = out
            if want_traj and traj is not None:
                take = (
                    traj.shape[1]
                    if trajectory_max is None
                    else min(traj.shape[1], trajectory_max - kept)
                )
                trajs.append(traj[:, :take].detach().cpu())
                kept += take
        else:
            x = out
        xs.append(x.detach())
        remaining -= b
    x_all = torch.cat(xs, dim=0)
    traj_all = torch.cat(trajs, dim=1) if (return_trajectory and trajs) else None
    return x_all, traj_all
