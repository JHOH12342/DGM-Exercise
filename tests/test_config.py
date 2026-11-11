from dgm.utils import load_config
from dgm.utils.config import CONFIG_DIR


def test_defaults_and_groups():
    cfg = load_config([])
    assert cfg.defaults.data == "gmm25" and cfg.data.name == "gmm"
    assert cfg.defaults.model == "gaussian" and cfg.model.name == "gaussian"
    assert cfg.run_name.startswith("gaussian_gmm25_")
    assert cfg.out_dir.endswith(cfg.run_name)


def test_group_override_and_dotlist():
    cfg = load_config(["data=halfmoon", "model=ddpm", "train.steps=7", "model.T=50", "run_name=x"])
    assert cfg.data.name == "halfmoon" and cfg.model.name == "ddpm"
    assert cfg.train.steps == 7 and cfg.model.T == 50
    assert cfg.run_name == "x"


def test_exp_config_sets_defaults():
    cfg = load_config([f"config={CONFIG_DIR / 'exp' / 'cifar10_ddpm.yaml'}", "train.steps=3"])
    assert cfg.defaults.data == "cifar10" and cfg.data.name == "cifar10"
    assert cfg.model.name == "ddpm" and cfg.model.net.unet.ch == 128
    assert cfg.train.steps == 3


def test_every_group_yaml_loads():
    for g in ("data", "model"):
        for p in (CONFIG_DIR / g).glob("*.yaml"):
            cfg = load_config([f"{g}={p.stem}"])
            assert cfg[g].name
