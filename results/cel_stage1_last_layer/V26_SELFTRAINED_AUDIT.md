# V26 Self-Trained Audit

> Historical artifact name retained for provenance. The audited method is referred to as the **A module**.

- Generated at: `2026-08-12T23:10:12`
- Audit pass: `True`
- Provenance pass: `True`
- Intermediate stages skipped test: `True`
- Final joint test completed: `True`
- Warmup kept LoRA/selector frozen: `True`
- Joint phase changed at least one trainable component: `True`

## Initialization Chain

- Bootstrap: Qwen3 base -> `cel_task_conditioned_lastlayer_v26_bootstrap_qwen3_1.7b`
- Calibrator warmup: `cel_task_conditioned_lastlayer_v26_bootstrap_qwen3_1.7b` -> `cel_task_conditioned_lastlayer_v26_calibrator_warmup_qwen3_1.7b`
- Strict joint: `cel_task_conditioned_lastlayer_v26_calibrator_warmup_qwen3_1.7b` -> `cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b`
- Historical baseline and v1 checkpoints are comparison-only and were not loaded.

## Metric Comparison

| Model | Overall Acc | Overall AUC | Final Acc | Final AUC |
|---|---:|---:|---:|---:|
| baseline | 69.22 | 74.95 | 61.17 | 75.32 |
| A-module joint | 70.03 | 75.38 | 66.80 | 77.26 |
| delta | +0.81 | +0.43 | +5.63 | +1.94 |

- Beats baseline Overall Acc + AUC gate: `True`
- Beats baseline Final Acc + AUC gate: `True`
- Beats baseline on all four metrics: `True`

## Parameter Audit

- Bootstrap -> warmup adapter equal: `True`
- Bootstrap -> warmup selector equal: `True`
- Warmup -> joint adapter changed tensors: `392`
- Warmup -> joint selector changed tensors: `0`
- Warmup -> joint calibrator changed tensors: `1`
