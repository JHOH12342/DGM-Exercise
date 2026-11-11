#!/usr/bin/env bash
# Train one model on every 2-D toy target, all runs concurrently on one GPU.
#   MODEL=ddpm GPU=0 bash scripts/sweep_2d.sh                 -> outputs/ddpm_2d/<dataset>/
#   MODEL=vae STEPS=50000 bash scripts/sweep_2d.sh model.lr=5e-4
# Compare afterwards: python -m scripts.summarize outputs/<NAME> --out reports/<NAME>
set -euo pipefail
cd "$(dirname "$0")/.."
GPU=${GPU:-0}
MODEL=${MODEL:-gaussian}
NAME=${NAME:-${MODEL}_2d}
STEPS=${STEPS:-20000}
DATASETS=(${DATASETS:-gmm25 gmm9 halfmoon swissroll checkerboard circles spiral gaussian})
mkdir -p "outputs/$NAME"
for d in "${DATASETS[@]}"; do
  CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4 TQDM_DISABLE=1 PYTHONUNBUFFERED=1 \
    ${PYTHON:-python} -m scripts.train data=$d model=$MODEL run_name=$NAME/$d \
      train.steps=$STEPS logging.tensorboard=false "$@" > "outputs/$NAME/$d.log" 2>&1 &
  echo "launched $MODEL on $d (pid $!)"
done
wait
echo "ALL_DONE"
