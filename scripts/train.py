"""Train a generative model.

    python -m scripts.train data=gmm25 model=gaussian train.steps=3000
    python -m scripts.train data=halfmoon model=ddpm run_name=ddpm_hm
    python -m scripts.train config=configs/exp/cifar10_ddpm.yaml

Outputs go to ``outputs/<run_name>/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dgm.train import Trainer
from dgm.utils import load_config


def main(argv=None):
    cfg = load_config(argv)
    trainer = Trainer(cfg)
    try:
        trainer.fit()
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
