import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch
from omegaconf import OmegaConf

from dgm.utils import load_config


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.fixture
def tiny_cfg(tmp_path):
    cfg = load_config(
        [
            "data=gmm25",
            "model=gaussian",
            "device=cpu",
            f"out_root={tmp_path}",
            "run_name=test_run",
            "train.steps=30",
            "train.batch_size=128",
            "train.log_every=10",
            "train.eval_every=15",
            "train.ckpt_every=15",
            "train.keep_ckpts=1",
            "eval.n_samples=400",
            "eval.sample_batch_size=200",
            "viz.trajectory_max=50",
            "viz.n_trajectories=8",
            "viz.gif=final",
            "logging.tensorboard=false",
        ]
    )
    OmegaConf.set_struct(cfg, False)
    return cfg
