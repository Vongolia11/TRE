# Trace and label format

## Manifest (JSON Lines)

Each line is one generated response. Example (labels are illustrative only):

```json
{"index": 0, "question_id": "sample-0", "question": "...", "generation": "...", "ground_truth": "...", "entropy_file": "entropy/sample_000000_sample-0.npz", "hallucination": {"label": "non_hallucination"}}
{"index": 1, "question_id": "sample-1", "question": "...", "generation": "...", "ground_truth": "...", "entropy_file": "entropy/sample_000001_sample-1.npz", "hallucination": {"label": "hallucination"}}
```

`hallucination` is `null` in freshly generated manifests. Supplying labels is the researcher's responsibility: TRE is a **detector** and does not label its own test examples. Labels can be string values or nested objects as shown. Other labels are ignored.

## Per-response NPZ

- `entropy`: `[steps, prompt_len + gen_length]` matrix of token entropies just before each reveal step.
- `prompt_len`: prompt length including any batch padding (single sample or batch padding).
- `gen_length`: number of generated token positions.
- `revealed_counts`: `[steps]` number of newly revealed tokens at each step.
- `revealed_positions`: `[steps, max_count]` token positions, padded with `-1`.
- `position_mode`: `absolute` (new files), referring to the full prompt+generation sequence. Older files may omit this field.
- `revealed_token_ids`: `[steps, max_count]` token IDs (with `-1` padding).
- `revealed_token_entropies`: `[steps, max_count]` entropy at the reveal event; special-token contributions are zeroed.
- `revealed_token_entropy_mean`, `remaining_entropy_before_mean`, `remaining_entropy_before_sum`: auxiliary per-step statistics.

The scorer sums only **first-reveal** positions for each trajectory, weighted by `t/T`. It uses `revealed_token_entropies` where available; `entropy` is the legacy fallback. This means special-token masking is consistently applied to newly generated NPZ files.

## Metrics

`tre_summary.json` is empty when the manifest contains no usable labeled traces. With labels it includes `n`, class counts, `auroc`, `auprc`, and `fpr95`; all non-null metric values use the `[0,1]` scale (`null` indicates an undefined metric). Scores and labels are emitted in `tre_scores.csv`.
