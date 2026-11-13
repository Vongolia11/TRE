import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tre_dllm.scoring import (
    Trace,
    auroc,
    average_precision,
    compute_tre_from_trace,
    fpr_at_95_tpr,
    load_trace,
    metric_bundle,
    parse_label,
    score_manifest,
)
from tre_dllm.utils.io import save_entropy_trace


def test_metric_perfect_ranking():
    y = np.array([1, 1, 0, 0])
    s = np.array([0.9, 0.8, 0.2, 0.1])
    assert auroc(y, s) == 1.0
    assert average_precision(y, s) == 1.0
    assert fpr_at_95_tpr(y, s) == 0.0
    assert metric_bundle(y.tolist(), s.tolist())['auprc'] == 1.0


def test_metric_inverted_ranking_and_ties():
    y = np.array([1, 0, 1, 0])
    s = np.array([0.1, 0.9, 0.2, 0.8])
    assert auroc(y, s) == 0.0
    assert average_precision(y, s) == pytest.approx(0.4166666667)
    assert fpr_at_95_tpr(y, s) == 1.0
    assert auroc(np.array([0, 1]), np.array([0.3, 0.3])) == 0.5
    assert average_precision(np.array([0, 1]), np.array([0.3, 0.3])) == 0.5


def test_scoring_absolute_reveal_special_token_zero():
    # prompt occupies [0,1), generated tokens are positions 1,2.
    entropy = np.array([[50., 2., 3.], [50., 4., 5.]])
    trace = Trace(
        entropy=entropy, prompt_len=1, gen_length=2,
        revealed_counts=np.array([1, 1]),
        revealed_positions=np.array([[1], [2]]),
        revealed_token_entropies=np.array([[0.], [5.]]),
        position_mode='absolute',
    )
    # step1 special token gets zero, step2 gets 5 * 1.0
    assert compute_tre_from_trace(trace) == 5.0
    # Raw-matrix fallback for old traces (special token is not explicitly zeroed).
    legacy = Trace(entropy=entropy, prompt_len=1, gen_length=2,
                   revealed_counts=np.array([1, 1]),
                   revealed_positions=np.array([[1], [2]]), position_mode='absolute')
    assert compute_tre_from_trace(legacy) == 6.0


def test_scoring_relative_positions_and_duplicate():
    trace = Trace(entropy=np.array([[10., 3., 4.], [10., 5., 6.]]),
                  prompt_len=1, gen_length=2,
                  revealed_counts=np.array([1, 2]),
                  revealed_positions=np.array([[0, -1], [0, 1]]),
                  revealed_token_entropies=np.array([[3., np.nan], [5., 6.]]),
                  position_mode='relative')
    # Position 0 on step 2 was already revealed: 3/2 + 6 = 7.5
    assert compute_tre_from_trace(trace) == 7.5


def test_npz_round_trip_and_manifest(tmp_path):
    step_trace = {
        'revealed_counts': np.array([[1], [1]], dtype=np.int32),
        'revealed_positions': [[np.array([1])], [np.array([2])]],
        'revealed_token_ids': [[np.array([11])], [np.array([12])]],
        'revealed_token_entropies': [[np.array([0.])], [np.array([5.])]],
        'revealed_token_entropy_mean': np.array([[0.], [5.]], dtype=np.float32),
        'remaining_entropy_before_mean': np.array([[2.], [4.]], dtype=np.float32),
        'remaining_entropy_before_sum': np.array([[5.], [9.]], dtype=np.float32),
    }
    trace_dir = tmp_path / 'entropy'
    trace_dir.mkdir()
    for idx, value in enumerate((5., 2.)):
        step_trace['revealed_token_entropies'][1][0] = np.array([value])
        filename = f'sample{idx}.npz'
        save_entropy_trace(trace_dir / filename,
                           entropy=np.array([[0., 2., 3.], [0., 4., 5.]]),
                           meta={'prompt_len': 1, 'gen_length': 2, 'total_steps': 2},
                           step_trace=step_trace, batch_index=0)
        assert load_trace(trace_dir / filename).position_mode == 'absolute'
    manifest = tmp_path / 'manifest.jsonl'
    rows = [
        {'index': 0, 'question_id': 'q0', 'entropy_file': 'entropy/sample0.npz',
         'hallucination': {'label': 'hallucination'}},
        {'index': 1, 'question_id': 'q1', 'entropy_file': 'entropy/sample1.npz',
         'hallucination': {'label': 'non_hallucination'}},
    ]
    manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    scored, metrics = score_manifest(manifest)
    assert [r['tre_score'] for r in scored] == [5.0, 2.0]
    assert metrics['auroc'] == 1.0
    assert metrics['auprc'] == 1.0
    assert metrics['fpr95'] == 0.0


def test_label_parsing():
    assert parse_label({'hallucination': {'label': 'hallucination'}}) == 1
    assert parse_label({'hallucination': {'label': 'non-hallucination'}}) == 0
    assert parse_label({'hallucination': None}) is None
    assert parse_label({'label': 'unknown'}) is None


def test_dream_logits_hook_does_not_shift_again():
    torch = pytest.importorskip('torch')
    from tre_dllm.generation.dream import generate_dream_with_entropy

    class FakeDream:
        config = SimpleNamespace(mask_token_id=99, eos_token_id=2)

        def diffusion_generate(self, prompt, *, generation_logits_hook_func,
                               generation_tokens_hook_func, **kwargs):
            x = torch.cat([prompt, torch.full((1, 1), 99, dtype=torch.long)], dim=1)
            # Upstream Dream passes pre-shifted, position-aligned logits to the hook.
            logits = torch.tensor([[[0., 0., 0.], [12., 0., 0.]]])
            generation_tokens_hook_func(None, x, None)
            logits = generation_logits_hook_func(0, x, logits)
            x[0, 1] = 0
            generation_tokens_hook_func(0, x, logits)
            assert kwargs['output_history'] is False
            return SimpleNamespace(sequences=x)

    prompt = torch.tensor([[1]], dtype=torch.long)
    generated, entropies, meta, trace = generate_dream_with_entropy(
        FakeDream(), prompt, steps=1, gen_length=1, mask_id=99)
    # Entropy of logits at generated position 1 must be near zero, not log(3).
    assert entropies[0][0, 1] < 0.001
    assert trace['revealed_counts'].tolist() == [[1]]


def test_single_class_metrics_are_valid_json():
    metrics = metric_bundle([1, 1], [0.3, 0.8])
    assert metrics['auroc'] is None
    assert metrics['fpr95'] is None
    assert metrics['auprc'] == 1.0
    assert json.loads(json.dumps(metrics, allow_nan=False)) == metrics


def test_bad_entropy_shape_is_rejected():
    trace = Trace(entropy=np.zeros((2, 2)), prompt_len=2, gen_length=3,
                  revealed_counts=np.array([0, 0]),
                  revealed_positions=np.full((2, 1), -1),
                  position_mode='absolute')
    with pytest.raises(ValueError, match='Entropy width'):
        compute_tre_from_trace(trace)
