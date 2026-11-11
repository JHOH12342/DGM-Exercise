"""Fetch CIFAR-10 from the HuggingFace parquet mirror into torchvision's ``cifar-10-batches-py`` layout.

    python -m scripts.fetch_cifar10 [--root ./data]

Use this when the default torchvision download is slow.
"""

from __future__ import annotations

import argparse
import io
import os
import pickle
import sys
import urllib.request
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

HF = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text"
URLS = {"train": f"{HF}/train-00000-of-00001.parquet", "test": f"{HF}/test-00000-of-00001.parquet"}
LABEL_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def download(url: str, dst: Path) -> Path:
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part")
    with tqdm(unit="B", unit_scale=True, desc=dst.name) as bar:

        def hook(n, bs, total):
            if total > 0:
                bar.total = total
            bar.update(n * bs - bar.n)

        urllib.request.urlretrieve(url, tmp, reporthook=hook)
    tmp.replace(dst)
    return dst


def parquet_to_arrays(path: Path) -> tuple[np.ndarray, list[int]]:
    import pyarrow.parquet as pq
    from PIL import Image

    table = pq.read_table(path)
    imgs = table.column("img").to_pylist()
    labels = [int(v) for v in table.column("label").to_pylist()]
    data = np.empty((len(imgs), 3 * 32 * 32), dtype=np.uint8)
    for i, d in enumerate(tqdm(imgs, desc=f"decode {path.name}", unit="img")):
        arr = np.asarray(Image.open(io.BytesIO(d["bytes"])).convert("RGB"))
        data[i] = arr.transpose(2, 0, 1).reshape(-1)
    return data, labels


def write_batches(
    root: Path, train: tuple[np.ndarray, list[int]], test: tuple[np.ndarray, list[int]]
) -> Path:
    out = root / "cifar-10-batches-py"
    out.mkdir(parents=True, exist_ok=True)
    xtr, ytr = train
    assert xtr.shape == (50000, 3072), xtr.shape
    for i in range(5):
        sl = slice(i * 10000, (i + 1) * 10000)
        with open(out / f"data_batch_{i + 1}", "wb") as f:
            pickle.dump(
                {
                    "batch_label": f"training batch {i + 1} of 5",
                    "data": xtr[sl],
                    "labels": ytr[sl],
                    "filenames": [f"hf_{j}.png" for j in range(sl.start, sl.stop)],
                },
                f,
                protocol=2,
            )
    xte, yte = test
    assert xte.shape == (10000, 3072), xte.shape
    with open(out / "test_batch", "wb") as f:
        pickle.dump(
            {
                "batch_label": "testing batch 1 of 1",
                "data": xte,
                "labels": yte,
                "filenames": [f"hf_test_{j}.png" for j in range(10000)],
            },
            f,
            protocol=2,
        )
    with open(out / "batches.meta", "wb") as f:
        pickle.dump(
            {"label_names": LABEL_NAMES, "num_cases_per_batch": 10000, "num_vis": 3072},
            f,
            protocol=2,
        )
    (out / "SOURCE.txt").write_text(
        "written by scripts/fetch_cifar10.py from https://huggingface.co/datasets/uoft-cs/cifar10\n"
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("DGM_DATA_ROOT", "./data"))
    ap.add_argument("--keep-parquet", action="store_true")
    args = ap.parse_args()
    root = Path(args.root).expanduser()
    out = root / "cifar-10-batches-py"
    if all(
        (out / f).exists()
        for f in [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch", "batches.meta"]
    ):
        print(f"already present: {out}")
        return
    pq_dir = root / "_hf_parquet"
    files = {k: download(u, pq_dir / Path(u).name) for k, u in URLS.items()}
    train, test = parquet_to_arrays(files["train"]), parquet_to_arrays(files["test"])
    out = write_batches(root, train, test)
    if not args.keep_parquet:
        for p in files.values():
            p.unlink()
        pq_dir.rmdir()
    from dgm.data import build_dataset

    ds = build_dataset({"name": "cifar10", "root": str(root), "download": False, "num_workers": 0})
    x = ds.sample(4)
    print(
        f"ok: {out}  train={len(ds.train_plain)} test={len(ds.test)} sample={tuple(x.shape)} range=[{x.min():.2f},{x.max():.2f}]"
    )


if __name__ == "__main__":
    main()
