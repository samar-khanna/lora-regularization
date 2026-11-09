#!/bin/bash
# Pretrain the post's two base models and evaluate them.
#
#   standard  12-layer nanochat model (286M parameters, 110M non-embedding), 2.2B tokens of FineWeb-Edu
#   droppath  the same model and data, pretrained with drop-path rising linearly with depth from 0 to 0.1
#
# Steps: download 50 FineWeb-Edu train shards plus the validation shard, train a 32K BPE tokenizer, pretrain each
# base for 4,200 steps of 524,288 tokens, then measure validation bits per byte and the CORE score.
# The post ran this on one NVIDIA DGX Spark (single GB10 GPU, 128 GB unified memory).
#
# Usage:
#   bash runs/pretrain.sh                       # both bases
#   bash runs/pretrain.sh standard              # one base
#   WANDB_RUN=1 bash runs/pretrain.sh           # also log to wandb
#
# Everything is written under $NANOCHAT_BASE_DIR (default: ./data_and_checkpoints): base_data/, tokenizer/,
# base_checkpoints/<tag>/, and the training log pretrain_<tag>.log (read by `python -m scripts.results collect`
# into results/pretraining.csv, Figure A1).
set -euo pipefail
cd "$(dirname "$0")/.."
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$PWD/data_and_checkpoints}"
export OMP_NUM_THREADS=1  # PyArrow, tokenizer and OMP threads otherwise contend for the CPU cores
PYTHON="${PYTHON:-python}"
mkdir -p "$NANOCHAT_BASE_DIR"

# Data and tokenizer (skipped if already present)
$PYTHON -m nanochat.dataset -n 50 -w 8
if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ]; then
    $PYTHON -m scripts.tok_train   # 32,768-token BPE on 2B characters
fi

declare -A DROP_PATH=([standard]=0.0 [droppath]=0.1)
TAGS=("$@")
[ ${#TAGS[@]} -eq 0 ] && TAGS=(standard droppath)
for tag in "${TAGS[@]}"; do
    rate="${DROP_PATH[$tag]}"
    run="dummy"; [ -n "${WANDB_RUN:-}" ] && run="pretrain_$tag"
    $PYTHON -m nanochat.report reset
    # --target-param-data-ratio=20: 20 tokens per non-embedding parameter (2.2B tokens)
    # --window-pattern=L: full causal attention in every layer
    # --drop-path-rate: rate at the deepest block; per-block rates rise linearly from 0
    $PYTHON -m scripts.base_train \
        --depth=12 \
        --target-param-data-ratio=20 \
        --window-pattern=L \
        --drop-path-rate="$rate" \
        --device-batch-size=32 \
        --save-every=1000 \
        --eval-every=100 \
        --core-metric-every=1000 \
        --model-tag="$tag" \
        --run="$run" 2>&1 | tee "$NANOCHAT_BASE_DIR/pretrain_$tag.log"
    $PYTHON -m scripts.base_eval --model-tag="$tag" --device-batch-size=32
done
echo "Base checkpoints: $NANOCHAT_BASE_DIR/base_checkpoints/"
