# DGM — Deep Generative Model Playground

A small codebase for learning generative models by implementing them.
Everything *around* the algorithm is provided: target distributions p(x) (2-D toys and CIFAR-10),
network backbones, a step-based training loop, metrics, visualization and a config system.
You implement the algorithms themselves in `dgm/models/`.

| model | status |
|---|---|
| `gaussian` | implemented — a diagonal Gaussian fitted by maximum likelihood; shows what every method of the model contract returns |
| `vae`, `gan`, `ncsn`, `ddpm`, `flow` | **to implement** — stubs that raise `NotImplementedError` |

Pick a stub, fill in `loss()` and `sample()`, and the whole pipeline (training, evaluation, plots) runs.

```
DGM/
├── configs/            # default.yaml + data/<name>.yaml + model/<name>.yaml + exp/<experiment>.yaml
├── dgm/
│   ├── data/           # p(x): toy2d.py (gmm / halfmoon / swissroll / checkerboard / circles / spiral / gaussian), cifar10.py
│   ├── nets/           # backbones: MLP, TimeConditionedMLP, UNet (32x32), ConvEncoder / ConvDecoder + factory
│   ├── models/         # ★ algorithms. base.py = contract, gaussian.py = reference, the rest are yours
│   ├── eval/           # metrics2d.py (MMD / SWD / energy / W2 / mode coverage / log-likelihood), metrics_img.py (FID / IS)
│   ├── viz/            # toy2d.py (scatter / density / trajectory / quiver / gif), image.py (grids), curves.py
│   ├── train.py        # Trainer: step loop, EMA, checkpoint / resume, logging, periodic eval + viz
│   ├── evaluator.py    # sample -> metrics -> figures (shared by the Trainer and scripts/sample.py)
│   └── checkpoint.py
├── scripts/            # train / sample / evaluate / summarize / viz_data / plot_curves / fetch_cifar10 / sweep_2d.sh
├── tests/              # pytest (CPU, ~1 min)
└── outputs/            # run outputs (gitignored)
```

---

## 1. Setup

```bash
git clone <this repo> DGM && cd DGM
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -e ".[dev,image]"   # image: FID / IS (torchmetrics + torch-fidelity)
source .venv/bin/activate

pytest                      # everything should pass before you start
python -m scripts.viz_data  # plots of every 2-D dataset -> outputs/data_viz/
```

Plain `pip install -e ".[dev,image]"` in any Python 3.10–3.12 environment works too.

## 2. Running

```bash
# syntax: python -m scripts.train data=<configs/data stem> model=<configs/model stem> key.sub=value ...
python -m scripts.train data=gmm25 model=gaussian train.steps=1000        # reference model, always works
python -m scripts.train data=gmm25 model=ddpm
python -m scripts.train data=halfmoon model=vae train.steps=50000 model.lr=5e-4 run_name=vae_hm
python -m scripts.train config=configs/exp/cifar10_ddpm.yaml               # full experiment file
CUDA_VISIBLE_DEVICES=1 python -m scripts.train data=spiral model=ncsn      # or device=cuda:1

# sample from a checkpoint: metrics + figures (+ GIF)
python -m scripts.sample outputs/<run>/ckpt/last.pt --n 10000 --gif
# metric table over several runs
python -m scripts.evaluate outputs/*/ckpt/last.pt --n 10000
# one model on every 2-D target (8 runs concurrently on one GPU), then a comparison report
MODEL=ddpm GPU=0 bash scripts/sweep_2d.sh
python -m scripts.summarize outputs/ddpm_2d --out reports/ddpm_2d
# re-plot learning curves / TensorBoard
python -m scripts.plot_curves outputs/<run>
tensorboard --logdir outputs
```

Output layout `outputs/<run_name>/`:

| path | content |
|---|---|
| `config.yaml` | the fully composed config |
| `metrics.jsonl` | `train/*`, `eval/*` scalars per step |
| `viz/samples_step_*.png` | real / generated scatter, generated density, true and model density when available |
| `viz/traj_step_*.png`, `viz/traj_*.gif` | sampling trajectories (when `sample` returns a trajectory) |
| `viz/field_step_*.png` | quiver plot of `vector_field(x, t)` (when implemented) |
| `viz/real_vs_gen_*.png`, `viz/samples_*.png` | images: grids and denoising strips |
| `eval/metrics_step_*.json` | metrics at that step |
| `ckpt/step_*.pt`, `ckpt/last.pt` | model + EMA + optimizer + config |
| `curves.png`, `tb/` | learning curves, TensorBoard |

---

## 3. What you implement — the model contract (`dgm/models/base.py`)

```python
from dgm.models import GenerativeModel, register_model
from dgm.nets import build_time_conditioned_net, build_encoder, build_decoder

@register_model("ddpm")                      # must match `name` in configs/model/ddpm.yaml
class DDPM(GenerativeModel):
    def __init__(self, data_shape, cfg):     # data_shape: (2,) or (3, 32, 32); cfg = the `model:` block of the yaml
        super().__init__(data_shape, cfg)
        ...

    def loss(self, x):                       # x: [B, *data_shape], already on self.device
        ...
        return loss, {"some_metric": 0.0}    # (scalar tensor, dict of floats to log)

    @torch.no_grad()
    def sample(self, n, return_trajectory=False):
        ...
        return x                             # [n, *data_shape]
        # iterative samplers may return (x, traj) with traj: [T+1, n, *data_shape], traj[0] = noise, traj[-1] = x
```

| method | required | notes |
|---|---|---|
| `loss(x) -> (loss, logs)` | yes | the default `train_step` backpropagates this through `optimizers["main"]` |
| `sample(n, return_trajectory)` | yes | returning a trajectory enables trajectory plots, GIFs and denoising strips |
| `log_prob(x) -> [B]` | optional | enables the `model_nll` metric and the model-density heat-map |
| `vector_field(x, t)` | optional | `t ∈ [0, 1]` with **t = 0 data, t = 1 noise**; enables quiver plots |
| `configure_optimizers() -> {name: Optimizer}` | optional | default: one Adam over all parameters (`cfg.lr / betas / weight_decay`) |
| `configure_lr_schedulers() -> {name: scheduler}` | optional | the Trainer calls `.step()` once per training step |
| `train_step(x, step) -> dict[str, float]` | optional | override when one `loss()` is not enough (e.g. several optimizers); use `self.optimizer_step(loss, name)` |

Notes:

- `self.cfg` is the `model:` block of the yaml (OmegaConf). Add whatever hyper-parameters you need to `configs/model/<name>.yaml`.
- Data is roughly unit scale (images are in `[-1, 1]`).
- The Trainer calls `train_step` in `model.train()` mode and `sample` / `log_prob` / `vector_field` under `torch.no_grad()` + `model.eval()`, on the EMA copy when EMA is enabled.
- `self.noise(n)`, `self.dim`, `self.is_image`, `self.device` are available helpers.
- A new model is: a file in `dgm/models/`, `@register_model("<name>")`, one import line in `dgm/models/__init__.py`, and `configs/model/<name>.yaml`.

### Backbones (`dgm/nets`)

| factory | 2-D data | image data | typical use |
|---|---|---|---|
| `build_time_conditioned_net(shape, cfg.net)` | `TimeConditionedMLP` | `UNet` | any network `f(x, t)` |
| `build_encoder(shape, out_dim, cfg.net)` | `MLP` | `ConvEncoder` | `x -> [B, out_dim]` |
| `build_decoder(shape, in_dim, cfg.net, out_act)` | `MLP` | `ConvDecoder` | `z: [B, in_dim] -> x` |

Every time-conditioned network takes `t` as a float tensor of shape `[B]`. The sinusoidal embedding multiplies `t` by
`time_scale` (default 1000), so continuous `t ∈ [0, 1]` is spread out like 1000 discrete steps; pass integer steps with `time_scale: 1`.

---

## 4. Data (`dgm/data`)

| config | registry `name` | log_prob | notes |
|---|---|---|---|
| `gmm25`, `gmm9` | `gmm` | yes | grid mixture of Gaussians; exposes `centers` / `std` for mode-coverage metrics |
| `halfmoon`, `swissroll`, `circles`, `spiral` | same | no | scikit-learn based |
| `checkerboard` | `checkerboard` | yes | uniform on the dark squares of a 4x4 board |
| `gaussian` | `gaussian` | yes | single anisotropic Gaussian (sanity target) |
| `cifar10` | `cifar10` | no | `[-1, 1]`, horizontal-flip augmentation; `classes: [0]` for a single class; location `data.root` (default `./data`, env `DGM_DATA_ROOT`) |

Common interface: `shape`, `sample(n)`, `train_iterator(bs)` (infinite), `test_set(n)` (fixed reference), `to_display(x)`, `bounds`.
Toy datasets draw fresh samples every step (`n_train: null`); set `data.n_train=10000` for a fixed training set.
A new toy dataset subclasses `Toy2D` and implements `_sample_raw` (and optionally `_log_prob_raw`).

CIFAR-10 is downloaded on first use. If the default torchvision download is slow, fetch it from the HuggingFace mirror first:

```bash
python -m scripts.fetch_cifar10 --root ./data
python -m scripts.train config=configs/exp/cifar10_quick.yaml   # image-pipeline sanity run (gaussian model, one class, cheap FID)
```

## 5. Metrics

| name | data | better | meaning |
|---|---|---|---|
| `mmd2` | 2-D | lower | unbiased MMD² with a sum of RBF kernels |
| `swd` | 2-D | lower | sliced Wasserstein-2 |
| `energy` | 2-D | lower | energy distance |
| `w2` | 2-D | lower | exact empirical W2 (subsampled to 2000 points) |
| `modes_covered`, `high_quality_frac`, `mode_kl` | GMM | higher / higher / lower | mode coverage (within 3σ of a centre) and balance across modes |
| `data_logp` | datasets with log_prob | higher | true log-density of generated samples |
| `model_nll` | models with `log_prob` | lower | model NLL of held-out real data |
| `std_ratio` | all | ≈ 1 | generated std / real std |
| `fid`, `is_mean` | images | lower / higher | torchmetrics; `eval.fid.feature=64` for a fast approximation |

## 6. Config system

`configs/default.yaml` → `configs/data/<x>.yaml` (into `cfg.data`) + `configs/model/<y>.yaml` (into `cfg.model`) → `config=<extra.yaml>` → CLI `key=value` overrides.
`cfg.defaults.{data,model}` hold the yaml stems, `cfg.{data,model}.name` the registry keys. The default `run_name` is `{model}_{data}_{timestamp}`.

Model yamls hold toy-scale defaults; CIFAR-scale settings go in `configs/exp/*.yaml` or on the command line (see `configs/exp/cifar10_ddpm.yaml`).
Select a GPU with `CUDA_VISIBLE_DEVICES=N` or `device=cuda:N`.
