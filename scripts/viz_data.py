"""Plot the target distributions themselves.

python -m scripts.viz_data                      # every 2-D toy -> outputs/data_viz/
python -m scripts.viz_data --datasets gmm25 cifar10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from dgm.data import build_dataset
from dgm.utils.config import CONFIG_DIR, load_group
from dgm.viz import plot_dataset, save_grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="config stems under configs/data (default: all 2-D toys)",
    )
    ap.add_argument("--out", default="outputs/data_viz")
    ap.add_argument("--n", type=int, default=5000)
    args = ap.parse_args()

    names = args.datasets or [
        p.stem for p in sorted((CONFIG_DIR / "data").glob("*.yaml")) if p.stem != "cifar10"
    ]
    out = Path(args.out)
    for name in names:
        ds = build_dataset(load_group("data", name))
        if ds.is_image:
            x = ds.sample(64, torch.Generator().manual_seed(0))
            p = save_grid(ds.to_display(x), out / f"{name}.png")
        else:
            p = plot_dataset(ds, out / f"{name}.png", n=args.n)
        print(f"{name:14s} shape={ds.shape} bounds={ds.bounds} log_prob={ds.has_log_prob} -> {p}")


if __name__ == "__main__":
    main()
