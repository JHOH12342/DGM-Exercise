from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from omegaconf import DictConfig

from .eval import ImageMetrics, compute_2d_metrics
from .utils.misc import sample_in_batches
from .viz import (
    animate_trajectory,
    plot_real_vs_gen,
    plot_samples,
    plot_trajectories,
    plot_trajectory_strip,
    plot_vector_field,
    save_grid,
)


class Evaluator:
    def __init__(self, cfg: DictConfig, dataset, device: torch.device | str = "cpu"):
        self.cfg = cfg
        self.dataset = dataset
        self.device = torch.device(device)
        self.x_ref = dataset.test_set(int(cfg.eval.n_samples))
        self.image_metrics = None
        if dataset.is_image and cfg.eval.fid.get("enabled", True):
            fcfg = cfg.eval.fid
            self.image_metrics = ImageMetrics(
                dataset,
                n_ref=int(fcfg.get("n_ref", 10_000)),
                device=self.device,
                feature=int(fcfg.get("feature", 2048)),
                inception_score=bool(fcfg.get("inception_score", True)),
            )

    @torch.no_grad()
    def sample(self, model, n: int | None = None, with_trajectory: bool = True):
        n = int(self.cfg.eval.n_samples if n is None else n)
        model.eval()
        x, traj = sample_in_batches(
            model.sample,
            n,
            int(self.cfg.eval.sample_batch_size),
            return_trajectory=with_trajectory,
            trajectory_max=int(self.cfg.viz.trajectory_max),
        )
        return x.float().cpu(), (None if traj is None else traj.float().cpu())

    @torch.no_grad()
    def metrics(self, model, x: torch.Tensor) -> dict[str, float]:
        out: dict[str, float] = {}
        flat = x.reshape(x.shape[0], -1)
        finite = torch.isfinite(flat).all(dim=1)
        out["nonfinite_frac"] = float((~finite).float().mean())
        x = x[finite]
        if x.shape[0] < 2:
            return out
        ref = self.x_ref.reshape(self.x_ref.shape[0], -1)
        out["std_ratio"] = float(x.reshape(x.shape[0], -1).std(0).mean() / ref.std(0).mean())
        if self.dataset.is_image:
            if self.image_metrics is not None:
                out.update(self.image_metrics.compute(x))
        else:
            out.update(
                compute_2d_metrics(
                    x, self.x_ref, self.dataset, model, which=list(self.cfg.eval.metrics2d)
                )
            )
        return out

    @torch.no_grad()
    def visualize(
        self,
        model,
        x: torch.Tensor,
        traj: torch.Tensor | None,
        viz_dir: Path,
        tag: str,
        step: int,
        final: bool = False,
    ) -> dict[str, Path]:
        vcfg = self.cfg.viz
        ds = self.dataset
        figs: dict[str, Path] = {}
        title = f"{getattr(model, 'name', type(model).__name__)} on {ds.name} @ step {step}"
        if ds.is_image:
            disp = ds.to_display
            figs["samples"] = save_grid(disp(x[:64]), viz_dir / f"samples_{tag}.png")
            figs["real_vs_gen"] = plot_real_vs_gen(
                disp(self.x_ref[:64]), disp(x[:64]), viz_dir / f"real_vs_gen_{tag}.png", title=title
            )
            if traj is not None and traj.shape[0] >= 2:
                figs["trajectory"] = plot_trajectory_strip(
                    disp(traj[:, :8]), viz_dir / f"traj_{tag}.png"
                )
        else:
            figs["samples"] = plot_samples(
                self.x_ref,
                x,
                ds.bounds,
                viz_dir / f"samples_{tag}.png",
                title=title,
                dataset=ds,
                model=model,
            )
            if traj is not None and traj.shape[0] >= 2:
                figs["trajectory"] = plot_trajectories(
                    traj,
                    ds.bounds,
                    viz_dir / f"traj_{tag}.png",
                    n_show=int(vcfg.n_trajectories),
                    final=x,
                )
                gif = vcfg.get("gif", False)
                if gif is True or (gif == "final" and final):
                    figs["gif"] = animate_trajectory(
                        traj, ds.bounds, viz_dir / f"traj_{tag}.gif", real=self.x_ref
                    )
            if vcfg.get("vector_field", True) and getattr(model, "has_vector_field", False):
                figs["vector_field"] = plot_vector_field(
                    model.vector_field,
                    ds.bounds,
                    viz_dir / f"field_{tag}.png",
                    times=list(vcfg.vector_field_times),
                    samples=self.x_ref[:1000],
                    device=model.device,
                    title=title,
                )
        return figs

    @torch.no_grad()
    def run(
        self,
        model,
        out_dir: str | Path,
        step: int,
        logger=None,
        final: bool = False,
        tag: str | None = None,
    ) -> dict[str, float]:
        out_dir = Path(out_dir)
        tag = tag or f"step_{step:07d}"
        t0 = time.time()
        x, traj = self.sample(model, with_trajectory=bool(self.cfg.viz.enabled))
        metrics = {"sample_time_s": time.time() - t0}
        metrics.update(self.metrics(model, x))
        if self.cfg.viz.enabled:
            figs = self.visualize(model, x, traj, out_dir / "viz", tag, step, final=final)
            if logger is not None:
                for k, p in figs.items():
                    logger.log_image_file(f"viz/{k}", p, step)
        (out_dir / "eval").mkdir(parents=True, exist_ok=True)
        with open(out_dir / "eval" / f"metrics_{tag}.json", "w") as f:
            json.dump({"step": step, **metrics}, f, indent=2)
        if logger is not None:
            logger.log_scalars(metrics, step, prefix="eval/")
        return metrics
