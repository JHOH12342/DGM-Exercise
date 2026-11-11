from __future__ import annotations

import math

import torch


def _flat(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1).float()


def _subsample(x: torch.Tensor, max_n: int | None, seed: int = 0) -> torch.Tensor:
    if max_n is None or x.shape[0] <= max_n:
        return x
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randperm(x.shape[0], generator=g)[:max_n].to(x.device)
    return x[idx]


def _finite(v: torch.Tensor | float) -> float:
    v = float(v)
    return v if math.isfinite(v) else float("nan")


@torch.no_grad()
def mmd_rbf(
    x: torch.Tensor,
    y: torch.Tensor,
    bandwidths: tuple[float, ...] = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0),
    max_n: int = 5000,
) -> float:
    x, y = _flat(_subsample(x, max_n)), _flat(_subsample(y, max_n)).to(x.device)
    n, m = x.shape[0], y.shape[0]

    def K(a, b):
        d2 = torch.cdist(a, b).pow(2)
        return sum(torch.exp(-d2 / (2 * h**2)) for h in bandwidths)

    kxx, kyy, kxy = K(x, x), K(y, y), K(x, y)
    mmd2 = (
        (kxx.sum() - kxx.diagonal().sum()) / (n * (n - 1))
        + (kyy.sum() - kyy.diagonal().sum()) / (m * (m - 1))
        - 2 * kxy.mean()
    )
    return _finite(mmd2)


@torch.no_grad()
def sliced_wasserstein(
    x: torch.Tensor, y: torch.Tensor, n_proj: int = 512, p: int = 2, seed: int = 0
) -> float:
    x, y = _flat(x), _flat(y).to(x.device)
    n = min(x.shape[0], y.shape[0])
    x, y = _subsample(x, n, seed), _subsample(y, n, seed + 1)
    g = torch.Generator(device="cpu").manual_seed(seed)
    dirs = torch.randn(x.shape[1], n_proj, generator=g).to(x)
    dirs = dirs / dirs.norm(dim=0, keepdim=True)
    px, py = (x @ dirs).sort(dim=0).values, (y @ dirs).sort(dim=0).values
    return _finite((px - py).abs().pow(p).mean().pow(1.0 / p))


@torch.no_grad()
def energy_distance(x: torch.Tensor, y: torch.Tensor, max_n: int = 5000) -> float:
    x, y = _flat(_subsample(x, max_n)), _flat(_subsample(y, max_n)).to(x.device)
    exy = torch.cdist(x, y).mean()
    exx = torch.cdist(x, x).sum() / (x.shape[0] * (x.shape[0] - 1))
    eyy = torch.cdist(y, y).sum() / (y.shape[0] * (y.shape[0] - 1))
    return _finite(2 * exy - exx - eyy)


@torch.no_grad()
def wasserstein2_exact(x: torch.Tensor, y: torch.Tensor, max_n: int = 2000, seed: int = 0) -> float:
    from scipy.optimize import linear_sum_assignment

    x, y = _flat(x), _flat(y).to(x.device)
    n = min(x.shape[0], y.shape[0], max_n)
    x, y = _subsample(x, n, seed), _subsample(y, n, seed + 1)
    cost = torch.cdist(x, y).pow(2).cpu().numpy()
    r, c = linear_sum_assignment(cost)
    return _finite(math.sqrt(cost[r, c].mean()))


@torch.no_grad()
def gmm_mode_metrics(
    x: torch.Tensor,
    centers: torch.Tensor,
    std: float,
    thresh_std: float = 3.0,
    min_frac: float = 0.1,
) -> dict[str, float]:
    x = _flat(x)
    centers = _flat(centers).to(x)
    K = centers.shape[0]
    d = torch.cdist(x, centers)
    dmin, assign = d.min(dim=1)
    hq = dmin < thresh_std * std
    n_hq = int(hq.sum())
    counts = torch.bincount(assign[hq], minlength=K).float()
    expected = x.shape[0] / K
    covered = int((counts >= max(1.0, min_frac * expected)).sum())
    p = counts / max(n_hq, 1)
    kl = float((p[p > 0] * (p[p > 0] * K).log()).sum()) if n_hq > 0 else float("nan")
    return {
        "high_quality_frac": n_hq / x.shape[0],
        "modes_covered": covered,
        "modes_total": K,
        "mode_kl": kl,
    }


@torch.no_grad()
def data_log_prob(x_gen: torch.Tensor, dataset, clamp_min: float = -1e3) -> float | None:
    if not getattr(dataset, "has_log_prob", False):
        return None
    lp = dataset.log_prob(x_gen.cpu())
    return None if lp is None else _finite(lp.clamp(min=clamp_min).mean())


@torch.no_grad()
def model_nll(model, x_real: torch.Tensor, batch_size: int = 2048) -> float | None:
    if not getattr(model, "has_log_prob", False):
        return None
    lps = []
    for i in range(0, x_real.shape[0], batch_size):
        lps.append(model.log_prob(x_real[i : i + batch_size].to(model.device)).float().cpu())
    return _finite(-torch.cat(lps).mean())


@torch.no_grad()
def compute_2d_metrics(
    x_gen: torch.Tensor,
    x_real: torch.Tensor,
    dataset=None,
    model=None,
    which: tuple[str, ...] | list[str] | None = None,
) -> dict[str, float]:
    which = set(which or ("mmd", "swd", "energy", "w2", "mode_coverage", "data_logp", "model_nll"))
    x_gen = x_gen.detach().float().cpu()
    x_real = x_real.detach().float().cpu()
    out: dict[str, float] = {}
    if not torch.isfinite(x_gen).all():
        out["nonfinite_frac"] = float((~torch.isfinite(_flat(x_gen)).all(dim=1)).float().mean())
        x_gen = x_gen[torch.isfinite(_flat(x_gen)).all(dim=1)]
        if x_gen.shape[0] < 2:
            return out
    if "mmd" in which:
        out["mmd2"] = mmd_rbf(x_gen, x_real)
    if "swd" in which:
        out["swd"] = sliced_wasserstein(x_gen, x_real)
    if "energy" in which:
        out["energy"] = energy_distance(x_gen, x_real)
    if "w2" in which:
        out["w2"] = wasserstein2_exact(x_gen, x_real)
    if "mode_coverage" in which and dataset is not None and hasattr(dataset, "centers"):
        out.update(gmm_mode_metrics(x_gen, dataset.centers, float(dataset.std)))
    if "data_logp" in which and dataset is not None:
        v = data_log_prob(x_gen, dataset)
        if v is not None:
            out["data_logp"] = v
    if "model_nll" in which and model is not None:
        v = model_nll(model, x_real)
        if v is not None:
            out["model_nll"] = v
    return out
