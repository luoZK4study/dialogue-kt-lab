# round2 Validation-driven Decision

## Evidence

| Metric | Validation value |
|---|---:|
| Mixed Overall AUC | 72.5915 |
| Evidence Overall AUC | 72.5298 |
| Reversal Overall AUC | 47.4017 |
| JS | 0.000017 |
| mean |p_r-p_m| | 0.002631 |
| h_nb RMS | 4.737066 |
| beta*h_nb / h_r RMS | 0.011099 |
| a saturation fraction | 0.000008 |
| complement raw max error | 7.67022792 |
| complement relative RMS error | 0.00203108 |
| weight complement max error | 0.00195312 |

- Validation audit pass: `true`.
- This decision uses validation metrics and diagnostics only; no test result or control experiment informed it.

## Decision For round3

Round 2 still showed a weak B contribution, so Round 3 further raises only the bounded mixed-path amplitude and consistency weight.

- Candidate: `stage2_dual_round3_contextual_transformer_seed1221`
- Initialization: raw Qwen3-1.7B base, never another candidate checkpoint.
- Model/data seed: `1221`.
- B remains the contextual Transformer; no architecture-reducing ablation is introduced.

## Exact Configuration

- `STAGE2_A_BOOTSTRAP_LR`: `0.0001`
- `STAGE2_BETA_START_RATIO`: `0.2`
- `STAGE2_CALIBRATOR_LR`: `0.0001`
- `STAGE2_CONSISTENCY_RAMP`: `0.25`
- `STAGE2_ENVIRONMENT_LR`: `0.0001`
- `STAGE2_ENV_BETA`: `0.2`
- `STAGE2_ENV_OUTPUT_RATIO`: `1.5`
- `STAGE2_LAMBDA_CONS`: `0.18`
- `STAGE2_SELECTOR_LR`: `0.000005`

Generated launcher environment: `next_round.env` in `round3`.
