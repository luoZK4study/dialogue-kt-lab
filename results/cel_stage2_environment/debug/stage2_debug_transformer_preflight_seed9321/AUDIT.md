# Stage 2 Candidate Audit: stage2_debug_transformer_preflight_seed9321

- Generated at: `2026-08-14T15:49:36`
- Audit pass: `True`
- Environment mode: `transformer`
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
| stage1_v26 | 70.03 | 75.38 | 66.80 | 77.26 |
| stage2 | 48.28 | 64.76 | 22.22 | 78.57 |

## Parameter Changes

- bootstrap_to_warmup `adapter` changed tensors: `0`
- bootstrap_to_warmup `selector` changed tensors: `0`
- bootstrap_to_warmup `environment` changed tensors: `0`
- warmup_to_joint `adapter` changed tensors: `392`
- warmup_to_joint `selector` changed tensors: `7`
- warmup_to_joint `environment` changed tensors: `18`
- warmup_to_joint `calibrator` changed tensors: `1`
