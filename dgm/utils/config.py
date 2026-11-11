from __future__ import annotations

import sys
import time
from collections.abc import Iterable
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"
GROUPS = ("data", "model")


def _available(group: str) -> list[str]:
    return sorted(p.stem for p in (CONFIG_DIR / group).glob("*.yaml"))


def load_group(group: str, name: str) -> DictConfig:
    path = CONFIG_DIR / group / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"No {group} config '{name}'. Available: {_available(group)} (dir: {CONFIG_DIR / group})"
        )
    cfg = OmegaConf.load(path)
    cfg.setdefault("name", name)
    return cfg


def load_config(
    argv: Iterable[str] | None = None, default_path: str | Path | None = None
) -> DictConfig:
    argv = list(sys.argv[1:] if argv is None else argv)
    default_path = Path(default_path or CONFIG_DIR / "default.yaml")
    base = OmegaConf.load(default_path)

    group_names: dict[str, str | None] = {g: None for g in GROUPS}
    extra_paths: list[str] = []
    dotlist: list[str] = []
    for tok in argv:
        if "=" not in tok:
            raise ValueError(f"Expected key=value, got {tok!r}")
        k, v = tok.split("=", 1)
        if k in group_names:
            group_names[k] = v
        elif k == "config":
            extra_paths.append(v)
        else:
            dotlist.append(tok)

    extras = [OmegaConf.load(p) for p in extra_paths]

    merged_for_defaults = OmegaConf.merge(base, *extras) if extras else base
    for g in GROUPS:
        if group_names[g] is None:
            group_names[g] = merged_for_defaults.defaults[g]

    cfg = OmegaConf.create(OmegaConf.to_container(base, resolve=False))
    for g in GROUPS:
        cfg.defaults[g] = group_names[g]
        cfg[g] = load_group(g, group_names[g])
    if extras:
        cfg = OmegaConf.merge(cfg, *extras)
    if dotlist:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))

    OmegaConf.set_struct(cfg, False)
    finalize(cfg)
    return cfg


def finalize(cfg: DictConfig) -> DictConfig:
    if cfg.get("run_name") in (None, "", "null"):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        cfg.run_name = f"{cfg.defaults.model}_{cfg.defaults.data}_{stamp}"
    cfg.out_dir = str(Path(cfg.out_root) / cfg.run_name)
    return cfg


def save_config(cfg: DictConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path)


def resolve_device(spec: str | None) -> torch.device:
    if spec in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)
