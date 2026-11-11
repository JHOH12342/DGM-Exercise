"""Re-plot training curves for one or more runs.

python -m scripts.plot_curves outputs/<run> [--keys train/loss eval/mmd2]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dgm.viz import plot_training_curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--keys", nargs="*", default=None)
    ap.add_argument("--smooth", type=int, default=5)
    args = ap.parse_args()
    for r in args.runs:
        r = Path(r)
        p = plot_training_curves(
            r / "metrics.jsonl", r / "curves.png", keys=args.keys, smooth=args.smooth
        )
        print(f"{r} -> {p}")


if __name__ == "__main__":
    main()
