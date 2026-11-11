import json

import torch
from omegaconf import OmegaConf

from dgm.checkpoint import build_from_checkpoint
from dgm.data import DataDistribution, register_dataset
from dgm.models import GenerativeModel, register_model
from dgm.train import Trainer


def test_trainer_end_to_end(tiny_cfg):
    tr = Trainer(tiny_cfg)
    tr.fit()
    tr.close()
    out = tr.out_dir
    assert (out / "config.yaml").exists() and (out / "metrics.jsonl").exists()
    recs = [json.loads(line) for line in open(out / "metrics.jsonl")]
    assert any("train/loss" in r for r in recs) and any("eval/mmd2" in r for r in recs)
    assert (out / "viz" / "samples_step_0000030.png").exists()
    assert (out / "viz" / "traj_step_0000030.png").exists()
    assert (out / "viz" / "field_step_0000030.png").exists()
    assert (out / "viz" / "traj_step_0000030.gif").exists()
    assert not (out / "viz" / "traj_step_0000015.gif").exists()
    assert (out / "eval" / "metrics_step_0000030.json").exists()
    assert (out / "ckpt" / "last.pt").exists() and (out / "curves.png").exists()
    assert len(list((out / "ckpt").glob("step_*.pt"))) == 1

    cfg, ds, model, step = build_from_checkpoint(out / "ckpt" / "last.pt", device="cpu")
    assert step == 30 and model.sample(5).shape == (5, 2)


def test_resume(tiny_cfg):
    tr = Trainer(tiny_cfg)
    tr.fit()
    tr.close()
    cfg2 = OmegaConf.create(OmegaConf.to_container(tiny_cfg, resolve=True))
    OmegaConf.set_struct(cfg2, False)
    cfg2.train.resume = str(tr.out_dir / "ckpt" / "last.pt")
    cfg2.train.steps = 45
    cfg2.run_name = "test_run_resumed"
    tr2 = Trainer(cfg2)
    assert tr2.step == 30
    tr2.fit()
    tr2.close()
    assert tr2.step == 45


@register_model("_test_two_opt")
class _TwoOpt(GenerativeModel):
    def __init__(self, data_shape, cfg=None):
        super().__init__(data_shape, cfg)
        self.a = torch.nn.Parameter(torch.zeros(*data_shape))
        self.b = torch.nn.Parameter(torch.ones(*data_shape))

    def configure_optimizers(self):
        return {"g": torch.optim.SGD([self.a], lr=0.1), "d": torch.optim.SGD([self.b], lr=0.1)}

    def train_step(self, x, step):
        la = ((x - self.a) ** 2).mean()
        self.optimizer_step(la, "g")
        lb = ((x.std(0) - self.b) ** 2).mean()
        self.optimizer_step(lb, "d")
        return {"g_loss": float(la.detach()), "d_loss": float(lb.detach())}

    def loss(self, x):
        raise RuntimeError("not used")

    def sample(self, n, return_trajectory=False):
        return self.a + self.b * self.noise(n)


def test_two_optimizer_model(tiny_cfg):
    tiny_cfg.model = OmegaConf.create({"name": "_test_two_opt"})
    tiny_cfg.run_name = "two_opt"
    tr = Trainer(tiny_cfg)
    tr.fit()
    tr.close()
    recs = [json.loads(line) for line in open(tr.out_dir / "metrics.jsonl")]
    assert any("train/g_loss" in r and "train/d_loss" in r and "train/lr_g" in r for r in recs)
    ck = torch.load(tr.out_dir / "ckpt" / "last.pt", weights_only=False)
    assert set(ck["optimizers"]) == {"g", "d"}


@register_dataset("_fake_images")
class _FakeImages(DataDistribution):
    shape = (3, 8, 8)
    is_image = True

    def __init__(self, seed=0):
        self.seed = seed

    def sample(self, n, generator=None):
        return torch.rand(n, *self.shape, generator=generator) * 2 - 1

    def train_iterator(self, batch_size, **_):
        g = torch.Generator().manual_seed(self.seed)
        while True:
            yield self.sample(batch_size, g)

    def to_display(self, x):
        return (x.clamp(-1, 1) + 1) / 2


def test_image_pipeline_without_fid(tiny_cfg):
    tiny_cfg.data = OmegaConf.create({"name": "_fake_images"})
    tiny_cfg.eval.fid.enabled = False
    tiny_cfg.eval.n_samples = 64
    tiny_cfg.eval.sample_batch_size = 32
    tiny_cfg.run_name = "fake_img"
    tr = Trainer(tiny_cfg)
    tr.fit()
    tr.close()
    out = tr.out_dir
    assert (out / "viz" / "samples_step_0000030.png").exists()
    assert (out / "viz" / "real_vs_gen_step_0000030.png").exists()
    assert (out / "viz" / "traj_step_0000030.png").exists()
    m = json.load(open(out / "eval" / "metrics_step_0000030.json"))
    assert "std_ratio" in m and "fid" not in m


def test_sample_in_batches_shapes():
    from dgm.models import build_model
    from dgm.utils import sample_in_batches

    model = build_model({"name": "gaussian"}, (2,))
    x, traj = sample_in_batches(model.sample, 25, 10, return_trajectory=False)
    assert x.shape == (25, 2) and traj is None
    x, traj = sample_in_batches(model.sample, 25, 10, return_trajectory=True, trajectory_max=15)
    assert x.shape == (25, 2) and traj.shape == (2, 15, 2)
    x, traj = sample_in_batches(model.sample, 25, 10, return_trajectory=True)
    assert traj.shape == (2, 25, 2)
