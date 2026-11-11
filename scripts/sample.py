"""Sample from a checkpoint, compute metrics and save figures.

python -m scripts.sample outputs/<run>/ckpt/last.pt --n 10000 --gif
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from dgm.checkpoint import build_from_checkpoint
from dgm.evaluator import Evaluator
from dgm.utils import seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument(
        "--n", type=int, default=None, help="number of samples (default: cfg.eval.n_samples)"
    )
    ap.add_argument("--out", default=None, help="output dir (default: <run>/samples)")
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--gif", action="store_true", help="also write a trajectory GIF (2-D only)")
    ap.add_argument("--save-tensor", action="store_true", help="save samples as samples.pt")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seed_everything(args.seed)
    cfg, dataset, model, step = build_from_checkpoint(
        args.ckpt, device=args.device, use_ema=not args.no_ema
    )
    if args.n:
        cfg.eval.n_samples = args.n
        cfg.eval.sample_batch_size = min(int(cfg.eval.sample_batch_size), args.n)
    if args.gif:
        cfg.viz.gif = True
    out_dir = Path(args.out) if args.out else Path(args.ckpt).resolve().parents[1] / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    ev = Evaluator(cfg, dataset, model.device)
    tag = f"step_{step:07d}_{'ema' if (not args.no_ema) else 'raw'}"
    metrics = ev.run(model, out_dir, step, final=True, tag=tag)
    if args.save_tensor:
        x, _ = ev.sample(model, with_trajectory=False)
        torch.save(x, out_dir / f"samples_{tag}.pt")
    print(json.dumps({"ckpt": args.ckpt, "step": step, **metrics}, indent=2))
    print(f"figures -> {out_dir / 'viz'}")


if __name__ == "__main__":
    main()
