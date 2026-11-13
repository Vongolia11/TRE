#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

python scripts/generate_llada_trajectories.py \
  --dataset triviaqa \
  --model-path GSAI-ML/LLaDA-8B-Instruct \
  --output-dir outputs/llada_triviaqa_sample \
  --subsample 16 \
  --diffusion-steps 128 \
  --gen-length 128 \
  --batch-size 1 \
  --temperature 0.0

python scripts/compute_tre.py \
  --manifest outputs/llada_triviaqa_sample/manifest.jsonl \
  --output-dir outputs/llada_triviaqa_sample/tre_scores
