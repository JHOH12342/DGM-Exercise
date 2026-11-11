from __future__ import annotations

import math
import shutil
import time
from collections import defaultdict
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from .checkpoint import load_checkpoint, save_checkpoint
from .data import build_dataset
from .evaluator import Evaluator
from .models import build_model
from .utils import EMA, MetricLogger, count_parameters, resolve_device, save_config, seed_everything
from .utils.config import finalize
from .utils.logger import format_scalars
from .viz import plot_training_curves


class Trainer:
    def __init__(self, cfg: DictConfig, dataset=None, model=None):
        self.cfg = cfg
        finalize(cfg)
        seed_everything(int(cfg.seed))
        self.device = resolve_device(cfg.get("device", "auto"))
        self.out_dir = Path(cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        save_config(cfg, self.out_dir / "config.yaml")

        self.dataset = dataset if dataset is not None else build_dataset(cfg.data)
        self.model = (
            model if model is not None else build_model(cfg.model, self.dataset.shape)
        ).to(self.device)
        self.model.setup_optimizers()

        ecfg = cfg.train.ema
        self.ema = (
            EMA(self.model, float(ecfg.decay), int(ecfg.get("start_step", 0)))
            if ecfg.enabled
            else None
        )

        lcfg = cfg.logging
        self.logger = MetricLogger(
            self.out_dir,
            tensorboard=bool(lcfg.get("tensorboard", True)),
            wandb_cfg=OmegaConf.to_container(lcfg.wandb, resolve=True) if "wandb" in lcfg else None,
            run_cfg=OmegaConf.to_container(cfg, resolve=True),
            run_name=cfg.run_name,
        )
        self.evaluator = Evaluator(cfg, self.dataset, self.device)
        self.step = 0
        if cfg.train.get("resume"):
            self.load(cfg.train.resume)

        print(
            f"[trainer] run={cfg.run_name}\n"
            f"[trainer] data={self.dataset!r}\n"
            f"[trainer] model={type(self.model).__name__}(name={self.model.name!r}) "
            f"params={count_parameters(self.model):,} device={self.device}\n"
            f"[trainer] out_dir={self.out_dir}  start_step={self.step}"
        )

    @property
    def sampling_model(self):
        return (
            self.ema.module
            if (self.ema is not None and self.cfg.eval.get("use_ema", True))
            else self.model
        )

    def fit(self) -> None:
        tcfg = self.cfg.train
        total = int(tcfg.steps)
        log_every, eval_every, ckpt_every = (
            int(tcfg.log_every),
            int(tcfg.get("eval_every", 0)),
            int(tcfg.get("ckpt_every", 0)),
        )
        it = self.dataset.train_iterator(
            int(tcfg.batch_size), num_workers=int(tcfg.get("num_workers", 0))
        )
        agg: dict[str, list[float]] = defaultdict(list)
        t_last, last_eval_step = time.time(), -1

        pbar = tqdm(
            range(self.step, total),
            initial=self.step,
            total=total,
            dynamic_ncols=True,
            desc="train",
        )
        for _ in pbar:
            x = next(it).to(self.device, non_blocking=True)
            self.model.train()
            logs = self.model.train_step(x, self.step)
            for sch in self.model.lr_schedulers.values():
                sch.step()
            self.step += 1
            if self.ema is not None:
                self.ema.update(self.model, self.step)
            for k, v in logs.items():
                agg[k].append(float(v))

            if self.step % log_every == 0:
                mean_logs = {k: sum(v) / len(v) for k, v in agg.items()}
                for name, opt in self.model.optimizers.items():
                    mean_logs[f"lr_{name}"] = opt.param_groups[0]["lr"]
                now = time.time()
                mean_logs["steps_per_s"] = log_every / max(now - t_last, 1e-8)
                t_last = now
                self.logger.log_scalars(mean_logs, self.step, prefix="train/")
                pbar.set_postfix_str(
                    format_scalars(
                        {k: v for k, v in mean_logs.items() if "loss" in k or k == "steps_per_s"}
                    )
                )
                agg.clear()
                bad = [k for k, v in mean_logs.items() if "loss" in k and not math.isfinite(v)]
                if bad:
                    self.save("nan")
                    raise FloatingPointError(f"non-finite training loss {bad} at step {self.step}")

            if eval_every and self.step % eval_every == 0:
                self.evaluate(final=self.step >= total)
                last_eval_step = self.step
            if ckpt_every and self.step % ckpt_every == 0:
                self.save()

        if last_eval_step != self.step:
            self.evaluate(final=True)
        self.save()
        plot_training_curves(self.out_dir / "metrics.jsonl", self.out_dir / "curves.png", smooth=5)
        print(f"[trainer] done. outputs in {self.out_dir}")

    def evaluate(self, final: bool = False) -> dict[str, float]:
        m = self.evaluator.run(
            self.sampling_model, self.out_dir, self.step, logger=self.logger, final=final
        )
        tqdm.write(f"[eval @ {self.step}] {format_scalars(m)}")
        return m

    def save(self, name: str | None = None) -> Path:
        ckpt_dir = self.out_dir / "ckpt"
        path = ckpt_dir / (f"{name}.pt" if name else f"step_{self.step:07d}.pt")
        save_checkpoint(path, self.cfg, self.model, self.step, self.ema)
        shutil.copyfile(path, ckpt_dir / "last.pt")
        keep = int(self.cfg.train.get("keep_ckpts", 0) or 0)
        if keep > 0:
            steps = sorted(ckpt_dir.glob("step_*.pt"))
            for old in steps[:-keep]:
                old.unlink()
        return path

    def load(self, path: str | Path) -> None:
        ckpt = load_checkpoint(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        for k, sd in ckpt.get("optimizers", {}).items():
            if k in self.model.optimizers:
                self.model.optimizers[k].load_state_dict(sd)
        for k, sd in ckpt.get("lr_schedulers", {}).items():
            if k in self.model.lr_schedulers:
                self.model.lr_schedulers[k].load_state_dict(sd)
        if self.ema is not None and ckpt.get("ema") is not None:
            self.ema.load_state_dict(ckpt["ema"])
        self.step = int(ckpt["step"])
        print(f"[trainer] resumed from {path} @ step {self.step}")

    def close(self) -> None:
        self.logger.close()
