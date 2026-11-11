import pytest
import torch

from dgm.data import DATASETS, build_dataset
from dgm.utils.config import CONFIG_DIR, load_group

TOYS = [p.stem for p in sorted((CONFIG_DIR / "data").glob("*.yaml")) if p.stem != "cifar10"]


@pytest.mark.parametrize("stem", TOYS)
def test_toy_basic(stem):
    ds = build_dataset(load_group("data", stem))
    x = ds.sample(1000)
    assert x.shape == (1000, 2) and x.dtype == torch.float32 and torch.isfinite(x).all()
    assert ds.bounds is not None and ds.bounds[0] < 0 < ds.bounds[1]
    assert (x.abs() <= ds.bounds[1]).float().mean() > 0.99
    assert 0.6 < x.std() < 1.4
    assert torch.equal(ds.test_set(100), ds.test_set(100))
    it = ds.train_iterator(64)
    b1, b2 = next(it), next(it)
    assert b1.shape == (64, 2) and not torch.equal(b1, b2)


def test_fixed_train_set_cycles():
    ds = build_dataset({"name": "halfmoon", "n_train": 256})
    it = ds.train_iterator(128)
    a, b, c = next(it), next(it), next(it)
    allv = torch.cat([a, b])
    assert allv.shape[0] == 256 and len({tuple(r) for r in allv.tolist()}) == 256
    assert all(tuple(r) in {tuple(q) for q in allv.tolist()} for r in c.tolist())


@pytest.mark.parametrize("stem", ["gmm25", "gmm9", "gaussian", "checkerboard"])
def test_log_prob_integrates_to_one(stem):
    ds = build_dataset(load_group("data", stem))
    assert ds.has_log_prob
    lo, hi = ds.bounds
    n = 800
    xs = torch.linspace(lo, hi, n)
    X, Y = torch.meshgrid(xs, xs, indexing="xy")
    pts = torch.stack([X.ravel(), Y.ravel()], 1)
    lp = ds.log_prob(pts)
    mass = float(torch.exp(lp).sum() * ((hi - lo) / (n - 1)) ** 2)
    assert abs(mass - 1.0) < 0.08, mass


def test_gmm_centers_and_modes():
    ds = build_dataset(load_group("data", "gmm25"))
    assert ds.centers.shape == (25, 2) and ds.n_modes == 25
    x = ds.sample(5000)
    d = torch.cdist(x, ds.centers).min(1).values
    assert (d < 3 * ds.std).float().mean() > 0.98
    gap = (ds.centers[0] + ds.centers[1]) / 2
    assert ds.log_prob(ds.centers[:1]).item() > ds.log_prob(gap[None]).item() + 10


def test_halfmoon_no_log_prob():
    ds = build_dataset({"name": "halfmoon"})
    assert not ds.has_log_prob and ds.log_prob(torch.zeros(3, 2)) is None


def test_registry_has_cifar():
    assert "cifar10" in DATASETS
    ds = DATASETS["cifar10"](root="/nonexistent", download=False)
    assert ds.shape == (3, 32, 32) and ds.is_image
    x = torch.tensor([-1.0, 0.0, 1.0])
    assert torch.allclose(ds.to_display(x), torch.tensor([0.0, 0.5, 1.0]))
