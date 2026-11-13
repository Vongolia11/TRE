# TRE: Training-Free Hallucination Detection for Diffusion Language Models

**Accepted at EACL 2027 (Main Conference)** · [OpenReview submission](https://openreview.net/forum?id=ziiqj7oIj5)

**Authors:** Pengcheng Weng, Yanyu Qian, Yue Tan, Yixin Liu.

This repository contains the **core training-free detector and trajectory-generation code** supplied with the TRE paper. It supports LLaDA-8B-Instruct and Dream-v0-Instruct-7B, three QA benchmarks, and scoring saved reveal/entropy traces. It is a **minimal public implementation**, **not** a complete reproduction bundle of every baseline or rebuttal experiment.

## Overview

TRE uses the uncertainty of *newly revealed tokens* at every denoising step, with a greater weight on later steps:

\[
\operatorname{TRE}(x) = \sum_{t=1}^{T} \frac{t}{T}
\sum_{j\in G_t} H_{t,j}.
\]

Here `G_t` is the set of positions first revealed at step `t`, and `H_{t,j}` is their token-level entropy. Larger TRE values indicate higher estimated hallucination risk. Entropies of tokenizer special tokens are zeroed when trajectories are recorded; the scorer uses the recorded per-reveal values if available.

## Repository structure

```text
config/                       Example model/dataset settings
scripts/
  generate_llada_trajectories.py
  generate_dream_trajectories.py
  compute_tre.py
  run_llada_sample.sh
  run_dream_sample.sh
src/tre_dllm/
  data/                        TriviaQA / HotpotQA / CommonsenseQA prompts
  generation/                  LLaDA and Dream tracing
  utils/                       File I/O
  scoring.py                   TRE, AUROC, AUPRC, FPR95
tests/                         CPU-only regression tests (synthetic traces)
```

## Installation

Requires Python 3.10 and a GPU with enough VRAM for the selected D-LLM for **trajectory generation**. Scoring existing `.npz` traces is CPU-only.

```bash
conda env create -f environment.yml
conda activate tre-dllm
pip install -e .
```

Or (with an existing appropriate PyTorch/CUDA installation):

```bash
pip install -r requirements.txt
pip install -e .
```

The pinned dependencies reflect the provided project environment and have not been tested against all newer Transformers/model revisions. Hugging Face `trust_remote_code=True` is required by the model loaders: review and trust the model repository before executing downloaded Python code.

## Quick start

Run from the repository root, after installation. The shell helpers generate 16 unlabeled trajectories and then score them:

```bash
bash scripts/run_llada_sample.sh
# or
bash scripts/run_dream_sample.sh
```

Or generate a larger evaluation manifest:

```bash
python scripts/generate_llada_trajectories.py \
  --dataset triviaqa \
  --model-path GSAI-ML/LLaDA-8B-Instruct \
  --output-dir outputs/llada_triviaqa \
  --subsample 400 --seed 42 \
  --diffusion-steps 128 --gen-length 128 \
  --batch-size 1 --temperature 0.0

python scripts/compute_tre.py \
  --manifest outputs/llada_triviaqa/manifest.jsonl \
  --output-dir outputs/llada_triviaqa/tre_scores
```

Dream:

```bash
python scripts/generate_dream_trajectories.py \
  --dataset triviaqa \
  --model-path Dream-org/Dream-v0-Instruct-7B \
  --output-dir outputs/dream_triviaqa \
  --subsample 400 --seed 42 \
  --diffusion-steps 128 --gen-length 128 \
  --batch-size 1 --temperature 0.0 --top-p none --alg origin

python scripts/compute_tre.py \
  --manifest outputs/dream_triviaqa/manifest.jsonl \
  --output-dir outputs/dream_triviaqa/tre_scores
```

To use other datasets, pass `--dataset hotpotqa` or `--dataset commonsenseqa`. These scripts build prompts, generate responses, and record entropy/reveal steps, but **do not produce hallucination labels**. You must add independently evaluated labels before interpreting AUROC/AUPRC/FPR95. A sample manifest format is documented in [`docs/TRACE_FORMAT.md`](docs/TRACE_FORMAT.md).

## Outputs and evaluation

Trajectory generation saves `outputs/<run>/manifest.jsonl`, `run_config.json`, and `entropy/*.npz`. TRE scoring saves `tre_scores.csv` and `tre_summary.json`.

If your manifest contains either `"hallucination": {"label": "hallucination"}` or `"hallucination": {"label": "non_hallucination"}`, the scorer additionally reports AUROC, non-interpolated AUPRC (average precision), and FPR95. **All numeric metrics in JSON are in [0,1], not percent.** Positive class: hallucination. The FPR95 implementation chooses the smallest observed FPR among thresholds achieving at least 95% TPR. Metrics requiring both classes are JSON `null` when a class is absent.

## Test the scoring code without GPUs or models

```bash
pip install -e '.[test]' 
PYTHONPATH=src pytest -q
```

The tests use synthetic entropy traces and a fake Dream hook, **not** actual LLaDA/Dream checkpoints. No 7B/8B end-to-end GPU reproduction was performed as part of this packaging.

## Reproducibility scope and important caveats

- This release does **not** ship benchmark samples, model checkpoints, precomputed entropy traces, hallucination labels, judge API integrations, detector baselines, training scripts, or original experiment logs. The published paper's exact tables cannot be regenerated from this archive alone.
- For consistent comparisons, use the paper's prompt and labeling protocols, dataset splits, generation settings, and model checkpoints. The example runner defaults are not a substitute for all reported experimental configurations.
- `scripts/generate_dream_trajectories.py` uses Dream's built-in `diffusion_generate` hooks. The upstream Dream code already aligns (shifts) logits before invoking the hook; this version computes entropy on **that** tensor without shifting twice. Compare/revalidate outputs if upgrading from earlier local versions that double-shifted logits.
- `src/tre_dllm/scoring.py` uses the **recorded per-reveal entropy** to respect the special-token treatment in the generator. Legacy NPZ traces without this field use the raw entropy matrix instead. The bundled regression tests explicitly cover the two paths.
- For readability and release safety, this repository ignores generated output directories, model weights, large arrays, and local credentials. Keep these files outside the public repository.

## Third-party assets

Model weights and benchmark datasets are downloaded from their original hosts and remain subject to their respective model cards, licenses, and dataset terms:

- [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct)
- [Dream-v0-Instruct-7B](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B)
- [TriviaQA](https://huggingface.co/datasets/mandarjoshi/trivia_qa)
- [HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa)
- [CommonsenseQA](https://huggingface.co/datasets/tau/commonsense_qa)

The code in this archive retains its MIT license. Please check the copyright attribution with all contributors before publication.

## Citation

Please cite the accepted EACL 2027 paper **“TRE: Training-Free Hallucination Detection for Diffusion Language Models”** by Pengcheng Weng, Yanyu Qian, Yue Tan, and Yixin Liu. An ACL Anthology BibTeX entry can be added once the final proceedings metadata are available; no provisional DOI or proceedings pages are asserted here.
