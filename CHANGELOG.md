# Release notes: EACL 2027 public-code packaging

Based exclusively on the user-provided `TRE_NeurIPS_2026(2).zip`, not on hidden or inaccessible experiments.

- Removed anonymous-review wording and updated the README for the accepted paper.
- Added focused CPU-only regression tests and trace documentation.
- Aligned scoring with recorded per-reveal entropies so zeroed special-token contributions are not inadvertently restored.
- Added explicit absolute/relative position metadata in newly generated traces.
- Fixed Dream entropy alignment: upstream `diffusion_generate` already shifts logits before the hook. Prevent a second shift.
- Removed the process-wide `torch.multinomial` monkeypatch from Dream integration and disabled unnecessary generation history output.
- Added AUPRC (average precision) and FPR95 to the evaluation summary, with explicit conventions.
- Hardened NPZ loading (`allow_pickle=False`) and trace shape validation.
- Extended `.gitignore` to prevent committing model weights, generated traces, local credentials, and other large artifacts.

**Important:** Code-level smoke/regression tests do not demonstrate the original paper's quantitative performance. Rerun a small model-backed sample and, if possible, a retained labeled benchmark before announcing full reproducibility.
