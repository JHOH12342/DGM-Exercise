import torch

from dgm.data import build_dataset
from dgm.eval import (
    compute_2d_metrics,
    energy_distance,
    gmm_mode_metrics,
    mmd_rbf,
    sliced_wasserstein,
    wasserstein2_exact,
)


def _two():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1000, 2, generator=g)
    y = torch.randn(1000, 2, generator=g)
    return x, y


def test_distances_order():
    x, y = _two()
    z = y + 2.0
    for fn in (mmd_rbf, sliced_wasserstein, energy_distance, wasserstein2_exact):
        same, diff = fn(x, y), fn(x, z)
        assert same < diff, fn.__name__
        assert abs(same) < (0.35 if fn is wasserstein2_exact else 0.2), (fn.__name__, same)


def test_mode_coverage():
    ds = build_dataset({"name": "gmm", "grid_size": 5})
    perfect = ds.sample(5000)
    m = gmm_mode_metrics(perfect, ds.centers, ds.std)
    assert m["modes_covered"] == 25 and m["high_quality_frac"] > 0.97 and m["mode_kl"] < 0.05
    collapsed = ds.centers[:1] + 0.2 * ds.std * torch.randn(5000, 2)
    m2 = gmm_mode_metrics(collapsed, ds.centers, ds.std)
    assert m2["modes_covered"] == 1 and m2["high_quality_frac"] > 0.97
    junk = torch.randn(5000, 2) * 3
    assert gmm_mode_metrics(junk, ds.centers, ds.std)["high_quality_frac"] < 0.2


def test_aggregator_keys():
    ds = build_dataset({"name": "gmm", "grid_size": 5})
    x, ref = ds.sample(500), ds.test_set(500)
    m = compute_2d_metrics(x, ref, dataset=ds)
    for k in ("mmd2", "swd", "energy", "w2", "modes_covered", "data_logp"):
        assert k in m
    assert "model_nll" not in m
    x[0] = float("nan")
    m2 = compute_2d_metrics(x, ref, dataset=ds, which=["mmd"])
    assert m2["nonfinite_frac"] > 0 and "mmd2" in m2
