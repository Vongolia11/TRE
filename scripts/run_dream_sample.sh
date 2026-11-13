#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

python scripts/generate_dream_trajectories.py \
  --dataset triviaqa \
  --model-path Dream-org/Dream-v0-Instruct-7B \
  --output-dir outputs/dream_triviaqa_sample \
  --subsample 16 \
  --diffusion-steps 128 \
  --gen-length 128 \
  --batch-size 1 \
  --temperature 0.0 \
  --top-p none \
  --alg origin

python scripts/compute_tre.py \
  --manifest outputs/dream_triviaqa_sample/manifest.jsonl \
  --output-dir outputs/dream_triviaqa_sample/tre_scores
