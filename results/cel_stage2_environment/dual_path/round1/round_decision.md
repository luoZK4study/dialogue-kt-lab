# round1 Validation-driven Decision

## Evidence

| Metric | Validation value |
|---|---:|
| Mixed Overall AUC | 72.5095 |
| Evidence Overall AUC | 72.5223 |
| Reversal Overall AUC | 46.9442 |
| JS | 0.000005 |
| mean |p_r-p_m| | 0.001801 |
| h_nb RMS | 0.938104 |
| beta*h_nb / h_r RMS | 0.001409 |
| a saturation fraction | 0.963781 |
| complement raw max error | 7.18240741 |
| complement relative RMS error | 0.00235714 |
| weight complement max error | 0.00195312 |

- Validation audit pass: `true`.
- This decision uses validation metrics and diagnostics only; no test result or control experiment informed it.

## Decision For round2

B contribution was weak relative to h_r, so Round 2 increases the mixed-path contribution while retaining a consistency safeguard. Selector saturation also lowers bootstrap and joint selector learning rates.

- Candidate: `stage2_dual_round2_contextual_transformer_seed1221`
- Initialization: raw Qwen3-1.7B base, never another candidate checkpoint.
- Model/data seed: `1221`.
- B remains the contextual Transformer; no architecture-reducing ablation is introduced.

## Exact Configuration

- `STAGE2_A_BOOTSTRAP_LR`: `0.0001`
- `STAGE2_BETA_START_RATIO`: `0.2`
- `STAGE2_CALIBRATOR_LR`: `0.0001`
- `STAGE2_CONSISTENCY_RAMP`: `0.25`
- `STAGE2_ENVIRONMENT_LR`: `0.0001`
- `STAGE2_ENV_BETA`: `0.15`
- `STAGE2_ENV_OUTPUT_RATIO`: `1.25`
- `STAGE2_LAMBDA_CONS`: `0.15`
- `STAGE2_SELECTOR_LR`: `0.000005`

Generated launcher environment: `next_round.env` in `round2`.
