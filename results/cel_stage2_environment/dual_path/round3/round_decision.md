# Round 3 Validation-driven Freeze Decision

## Evidence

| Metric | Validation value |
|---|---:|
| Mixed Overall AUC | 68.6671 |
| Evidence Overall AUC | 68.6406 |
| Reversal Overall AUC | 47.5138 |
| JS | 0.000026 |
| mean |p_r-p_m| | 0.003902 |
| h_nb RMS | 19.011387 |
| beta*h_nb / h_r RMS | 0.068843 |
| a saturation fraction | 0.000000 |
| complement raw max error | 8.00000000 |
| complement relative RMS error | 0.00199535 |
| weight complement max error | 0.00195312 |

- Validation audit pass: `true`.
- This decision uses validation metrics and diagnostics only; no test result or control experiment informed it.

## Decision

Round 3 is the third and final configured candidate. Its configuration is frozen; the next permitted action is final test and final audit for all three completed rounds, not a fourth training run.

## Frozen Configuration

- `STAGE2_A_BOOTSTRAP_LR`: `0.0001`
- `STAGE2_BETA_START_RATIO`: `0.2`
- `STAGE2_CALIBRATOR_LR`: `0.0001`
- `STAGE2_CONSISTENCY_RAMP`: `0.25`
- `STAGE2_ENVIRONMENT_LR`: `0.0001`
- `STAGE2_ENV_BETA`: `0.2`
- `STAGE2_ENV_OUTPUT_RATIO`: `1.5`
- `STAGE2_LAMBDA_CONS`: `0.18`
- `STAGE2_SELECTOR_LR`: `0.000005`
