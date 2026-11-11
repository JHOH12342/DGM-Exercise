"""Metrics-only evaluation of one or more checkpoints.

python -m scripts.evaluate outputs/*/ckpt/last.pt --n 10000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dgm.checkpoint import build_from_checkpoint
from dgm.evaluator import Evaluator
from dgm.utils import seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None, help="write results here")
    args = ap.parse_args()

    rows = []
    for ck in args.ckpts:
        seed_everything(args.seed)
        cfg, dataset, model, step = build_from_checkpoint(
            ck, device=args.device, use_ema=not args.no_ema
        )
        if args.n:
            cfg.eval.n_samples = args.n
            cfg.eval.sample_batch_size = min(int(cfg.eval.sample_batch_size), args.n)
        ev = Evaluator(cfg, dataset, model.device)
        x, _ = ev.sample(model, with_trajectory=False)
        m = ev.metrics(model, x)
        rows.append(
            {
                "run": cfg.run_name,
                "model": cfg.model.name,
                "data": cfg.defaults.data,
                "step": step,
                **m,
            }
        )

    keys = sorted({k for r in rows for k in r} - {"run", "model", "data", "step"})
    hdr = ["run", "model", "data", "step"] + keys
    print("\t".join(hdr))
    for r in rows:
        print(
            "\t".join(
                f"{r.get(k, ''):.4g}" if isinstance(r.get(k), float) else str(r.get(k, ""))
                for k in hdr
            )
        )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
