# Stage 2 Candidate Audit: stage2_transformer_topk20_beta010_centeredrms010_seed1221

- Generated at: `2026-08-15T02:17:01`
- Audit pass: `True`
- Environment mode: `transformer`
- Environment beta / top-k ratio: `0.1 / 0.2`
- Environment output postprocess: `centered_rms`
- Environment output ratio / init std: `0.1 / 0.01`
- Manifest/provenance pass: `True`
- Warmup froze A+B+LoRA: `True`
- Joint selector changed: `True`
- Joint environment changed or parameter-free: `True`
- Selector checkpoints are FP32: `True`
- Learned environment checkpoints are FP32: `True`
- Final diagnostics are finite: `True`

## Metrics

| Model | Overall Acc | Overall AUC | Final Acc | Final AUC |
|---|---:|---:|---:|---:|
| baseline | 69.22 | 74.95 | 61.17 | 75.32 |
| A-module reference | 70.03 | 75.38 | 66.80 | 77.26 |
| historical Stage 2 candidate | 68.06 | 75.25 | 67.18 | 74.73 |

## Parameter Changes

- bootstrap_to_warmup `adapter` changed tensors: `0`
- bootstrap_to_warmup `selector` changed tensors: `0`
- bootstrap_to_warmup `environment` changed tensors: `0`
- warmup_to_joint `adapter` changed tensors: `392`
- warmup_to_joint `selector` changed tensors: `7`
- warmup_to_joint `environment` changed tensors: `18`
- warmup_to_joint `calibrator` changed tensors: `1`
