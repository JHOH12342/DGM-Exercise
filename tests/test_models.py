import pytest
import torch

from dgm.data import build_dataset
from dgm.models import MODELS, build_model

ALGOS = ["vae", "gan", "ncsn", "ddpm", "flow"]


def test_registry():
    assert set(ALGOS) | {"gaussian"} <= set(MODELS)


@pytest.mark.parametrize("name", ALGOS)
def test_stubs_raise(name):
    with pytest.raises(NotImplementedError):
        build_model({"name": name}, (2,))


def test_gaussian_reference_learns_mean():
    ds = build_dataset(
        {"name": "gaussian", "std": [1.0, 0.3], "rotation_deg": 0.0, "normalize": False}
    )
    model = build_model({"name": "gaussian", "lr": 0.05}, ds.shape)
    model.setup_optimizers()
    it = ds.train_iterator(512)
    first = None
    for step in range(200):
        logs = model.train_step(next(it), step)
        first = logs["loss"] if first is None else first
    assert logs["loss"] < first
    assert torch.allclose(model.mean.detach(), torch.zeros(2), atol=0.15)
    assert torch.allclose(model.std.detach(), torch.tensor([1.0, 0.3]), atol=0.15)
    x, traj = model.sample(10, return_trajectory=True)
    assert x.shape == (10, 2) and traj.shape == (2, 10, 2)
    assert model.log_prob(x).shape == (10,)
    assert model.vector_field(x, torch.zeros(10)).shape == (10, 2)
    assert model.has_log_prob and model.has_vector_field


def test_gaussian_on_image_shape():
    model = build_model({"name": "gaussian"}, (3, 8, 8))
    model.setup_optimizers()
    x = torch.rand(4, 3, 8, 8) * 2 - 1
    logs = model.train_step(x, 0)
    assert "loss" in logs and model.sample(2).shape == (2, 3, 8, 8)
