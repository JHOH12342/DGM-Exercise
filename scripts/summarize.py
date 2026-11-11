"""Compare finished runs: metrics table, sample/density/trajectory grid, learning curves.

python -m scripts.summarize outputs/<sweep_dir> --out reports/<name>
python -m scripts.summarize outputs/run_a outputs/run_b --out reports/cmp --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import torch

from dgm.checkpoint import build_from_checkpoint
from dgm.evaluator import Evaluator
from dgm.utils import seed_everything
from dgm.viz._mpl import plt
from dgm.viz.curves import read_metrics
from dgm.viz.toy2d import _PT, _density_panel, _np, _style

TABLE_KEYS = [
    "swd",
    "mmd2",
    "energy",
    "w2",
    "modes_covered",
    "high_quality_frac",
    "mode_kl",
    "data_logp",
    "model_nll",
    "std_ratio",
    "fid",
    "is_mean",
]


def find_runs(paths: list[str]) -> list[Path]:
    runs: list[Path] = []
    for p in map(Path, paths):
        if (p / "ckpt" / "last.pt").exists():
            runs.append(p)
        else:
            runs += sorted(d for d in p.iterdir() if (d / "ckpt" / "last.pt").exists())
    return runs


def fmt(v) -> str:
    if v is None or v == "":
        return ""
    if isinstance(v, float):
        return f"{v:.3g}" if abs(v) < 1e4 else f"{v:.2e}"
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid-n", type=int, default=150, help="resolution of the density heat-maps")
    ap.add_argument(
        "--no-floor",
        action="store_true",
        help="skip the 'real:<data>' rows (metrics of a fresh real sample vs the reference = finite-sample noise floor)",
    )
    args = ap.parse_args()

    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    runs = find_runs(args.paths)
    if not runs:
        sys.exit("no runs with ckpt/last.pt found")

    rows, panels, curves, floors_done = [], [], {}, set()
    for run in runs:
        seed_everything(args.seed)
        cfg, ds, model, step = build_from_checkpoint(run / "ckpt" / "last.pt", device=args.device)
        cfg.eval.n_samples = args.n
        cfg.eval.sample_batch_size = min(int(cfg.eval.sample_batch_size), args.n)
        cfg.viz.trajectory_max = 64
        ev = Evaluator(cfg, ds, model.device)
        x, traj = ev.sample(model, with_trajectory=True)
        m = ev.metrics(model, x)
        series = read_metrics(run / "metrics.jsonl") if (run / "metrics.jsonl").exists() else {}
        final_loss = float(series["train/loss"][1][-1]) if "train/loss" in series else None
        label = run.name
        rows.append(
            {
                "run": label,
                "model": cfg.model.name,
                "data": cfg.defaults.data,
                "step": step,
                "train_loss": final_loss,
                **m,
            }
        )
        curves[label] = series
        if not args.no_floor and not ds.is_image and cfg.defaults.data not in floors_done:
            real2 = ds.sample(args.n, torch.Generator().manual_seed(args.seed + 1))
            mf = ev.metrics(None, real2)
            mf.pop("model_nll", None)
            rows.append(
                {
                    "run": f"real:{cfg.defaults.data}",
                    "model": "(real sample)",
                    "data": cfg.defaults.data,
                    "step": "",
                    **mf,
                }
            )
            floors_done.add(cfg.defaults.data)
        if not ds.is_image:
            panels.append((label, ds, model, ev.x_ref, x, traj))
        for p in sorted((run / "viz").glob(f"*step_{step:07d}*")):
            shutil.copyfile(p, out / "figures" / f"{label}_{p.name}")
        print(f"[{label}] step={step} " + " ".join(f"{k}={fmt(v)}" for k, v in m.items()))

    keys = ["run", "model", "data", "step", "train_loss"] + [
        k for k in TABLE_KEYS if any(k in r for r in rows)
    ]
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(k)) for k in keys) + " |")
    table_md = "\n".join(lines)
    with open(out / "summary.json", "w") as f:
        json.dump(rows, f, indent=2)

    grid_path = None
    if panels:
        row_names = [
            "real",
            "generated",
            "generated density",
            "model density  p_θ(x)",
            "trajectories",
        ]
        n_col = len(panels)
        fig, axes = plt.subplots(
            len(row_names), n_col, figsize=(2.3 * n_col, 2.3 * len(row_names) + 0.3), squeeze=False
        )
        cmap = plt.get_cmap("plasma")
        for j, (label, ds, model, real, gen, traj) in enumerate(panels):
            b = ds.bounds
            r, g = _np(real), _np(gen)
            axes[0, j].scatter(r[:, 0], r[:, 1], c="k", **_PT)
            _style(axes[0, j], b, label)
            axes[1, j].scatter(g[:, 0], g[:, 1], c="C3", **_PT)
            _style(axes[1, j], b)
            axes[2, j].hist2d(g[:, 0], g[:, 1], bins=100, range=[list(b), list(b)], cmap="viridis")
            _style(axes[2, j], b)
            if getattr(model, "has_log_prob", False):
                _density_panel(axes[3, j], model.log_prob, b, None, args.grid_n, model.device)
            else:
                axes[3, j].axis("off")
            ax = axes[4, j]
            ax.scatter(g[:, 0], g[:, 1], c="lightgray", s=1, alpha=0.5, linewidths=0)
            if traj is not None:
                tr = _np(traj[:, :48])
                T1 = tr.shape[0]
                for i in range(tr.shape[1]):
                    for s in range(T1 - 1):
                        ax.plot(
                            tr[s : s + 2, i, 0],
                            tr[s : s + 2, i, 1],
                            color=cmap(s / max(T1 - 2, 1)),
                            lw=0.6,
                            alpha=0.8,
                        )
                ax.scatter(tr[-1, :, 0], tr[-1, :, 1], c="C3", s=6, marker="*", zorder=4)
            _style(ax, b)
        for i, name in enumerate(row_names):
            axes[i, 0].set_ylabel(name, fontsize=9)
            axes[i, 0].yaxis.set_visible(True)
            axes[i, 0].set_yticks([])
        fig.tight_layout()
        grid_path = out / "summary_grid.png"
        fig.savefig(grid_path)
        plt.close(fig)

    curve_keys = [
        k
        for k in ("train/loss", "eval/swd", "eval/mmd2", "eval/model_nll", "eval/modes_covered")
        if any(k in s for s in curves.values())
    ]
    curves_path = None
    if curve_keys:
        fig, axes = plt.subplots(
            1, len(curve_keys), figsize=(3.6 * len(curve_keys), 3.0), squeeze=False
        )
        for ax, k in zip(axes[0], curve_keys):
            for label, s in curves.items():
                if k in s:
                    st, y = s[k]
                    ax.plot(st, y, lw=1.1, label=label, marker="." if len(st) < 30 else None)
            ax.set_title(k, fontsize=9)
            ax.tick_params(labelsize=7)
            if k == "train/loss":
                ax.set_yscale("log")
        axes[0, 0].legend(fontsize=6, frameon=False)
        fig.tight_layout()
        curves_path = out / "summary_curves.png"
        fig.savefig(curves_path)
        plt.close(fig)

    md = [f"# Summary: {', '.join(r['run'] for r in rows)}", "", table_md, ""]
    if grid_path:
        md += [f"![grid]({grid_path.name})", ""]
    if curves_path:
        md += [f"![curves]({curves_path.name})", ""]
    md += ["Per-run final figures: `figures/`", ""]
    (out / "summary.md").write_text("\n".join(md))
    print("\n" + table_md)
    print(f"\n-> {out}/summary.md, summary_grid.png, summary_curves.png, figures/")


if __name__ == "__main__":
    main()
