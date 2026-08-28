# Stage 2 Dual-path A+B Report

- Generated at: `2026-08-17T22:48:53`
- Task: MathDial ATC Dialogue-KT, Qwen3-1.7B + LoRA.
- Method: complementary `h_r/h_n`, contextual Transformer B, shared `P(h_r)`/`P(h_m)` and reversal `P(h_n)` diagnostics.
- Experimental scope: three configuration rounds, all using model/data seed `1221`; this is not a multi-seed robustness claim.
- A-module reference Overall AUC: `75.38`; paper target Overall AUC: `76.71`.
- Tensor-contract closeout: `pass` (`results/cel_stage2_environment/dual_path/tensor_contracts.final.stdout.log`).

## Completion State

| Round | Training/audit | Validation | Final test |
|---|---|---|---|
| `round1` | pass | available | available |
| `round2` | pass | available | available |
| `round3` | pass | available | available |

## Validation Metrics

Metrics use `Acc/AUC`. The primary path is mixed; evidence and reversal are diagnostic paths.

| Round | Candidate | Mixed Overall | Mixed Final | Evidence Overall | Evidence Final | Reversal Overall | Reversal Final |
|---|---|---:|---:|---:|---:|---:|---:|
| `round1` | `stage2_dual_round1_contextual_transformer_seed1221` | 66.78/72.51 | 61.20/73.46 | 66.84/72.52 | 61.45/73.51 | 52.71/46.94 | 25.78/40.80 |
| `round2` | `stage2_dual_round2_contextual_transformer_seed1221` | 65.93/72.59 | 64.82/73.01 | 66.10/72.53 | 64.58/72.89 | 52.65/47.40 | 25.54/41.18 |
| `round3` | `stage2_dual_round3_contextual_transformer_seed1221` | 63.82/68.67 | 57.35/68.88 | 63.25/68.64 | 56.87/68.86 | 52.65/47.51 | 25.54/42.12 |

## Validation Diagnostics

| Round | JS | mean `|p_r-p_m|` | `h_nb` RMS | `beta*h_nb` RMS | Complement raw max | Complement relative RMS | Weight complement max |
|---|---:|---:|---:|---:|---:|---:|---:|
| `round1` | 0.000005 | 0.001801 | 0.938104 | 0.093810 | 7.18240741 | 0.00235714 | 0.00195312 |
| `round2` | 0.000017 | 0.002631 | 4.737066 | 0.710560 | 7.67022792 | 0.00203108 | 0.00195312 |
| `round3` | 0.000026 | 0.003902 | 19.011387 | 3.802277 | 8.00000000 | 0.00199535 | 0.00195312 |

## Test Metrics

Metrics use `Acc/AUC`. The primary path is mixed; evidence and reversal are diagnostic paths.

| Round | Candidate | Mixed Overall | Mixed Final | Evidence Overall | Evidence Final | Reversal Overall | Reversal Final |
|---|---|---:|---:|---:|---:|---:|---:|
| `round1` | `stage2_dual_round1_contextual_transformer_seed1221` | 68.61/73.50 | 62.72/72.66 | 68.61/73.51 | 62.91/72.67 | 52.44/44.12 | 26.80/36.48 |
| `round2` | `stage2_dual_round2_contextual_transformer_seed1221` | 67.91/73.30 | 66.02/73.42 | 67.91/73.28 | 65.83/73.36 | 52.44/43.91 | 26.80/36.36 |
| `round3` | `stage2_dual_round3_contextual_transformer_seed1221` | 67.15/71.59 | 59.61/70.84 | 66.85/71.54 | 59.22/70.88 | 52.44/44.02 | 26.80/35.83 |

## Configuration Iterations

| Round | B / objective configuration | Change from prior round |
|---|---|---|
| `round1` | B=`contextual_transformer` (1024h/4L/8heads/ffn=4096); beta=`0.1`; ratio=`1.0`; lambda_cons=`0.1` | initial configuration |
| `round2` | B=`contextual_transformer` (1024h/4L/8heads/ffn=4096); beta=`0.15`; ratio=`1.25`; lambda_cons=`0.15` | env_beta: `0.1` -> `0.15`; env_output_ratio: `1.0` -> `1.25`; lambda_cons: `0.1` -> `0.15`; BOOTSTRAP_LR: `0.00020` -> `0.0001`; SELECTOR_LR: `0.00001` -> `0.000005`; ENVIRONMENT_LR: `0.00010` -> `0.0001`; CALIBRATOR_LR: `0.00010` -> `0.0001` |
| `round3` | B=`contextual_transformer` (1024h/4L/8heads/ffn=4096); beta=`0.2`; ratio=`1.5`; lambda_cons=`0.18` | env_beta: `0.15` -> `0.2`; env_output_ratio: `1.25` -> `1.5`; lambda_cons: `0.15` -> `0.18` |

## Audit State

| Round | audit_pass | identity_ok | warmup_freeze_ok | b_warmup_freeze_ok | a_joint_update_ok | joint_update_ok | selector_fp32_ok | environment_fp32_ok | complement_contract_ok |
|---|---|---|---|---|---|---|---|---|---|
| `round1` | pass | pass | pass | pass | pass | pass | pass | pass | pass |
| `round2` | pass | pass | pass | pass | pass | pass | pass | pass | pass |
| `round3` | pass | pass | pass | pass | pass | pass | pass | pass | pass |

## Parameter Update Detail

Each strict joint phase must update every trainable component listed below; warmup freeze checks must keep the preceding components unchanged.

| Round | A joint component updates | Final joint component updates |
|---|---|---|
| `round1` | adapter=pass, calibrator=pass, selector=pass | adapter=pass, calibrator=pass, environment=pass, selector=pass |
| `round2` | adapter=pass, calibrator=pass, selector=pass | adapter=pass, calibrator=pass, environment=pass, selector=pass |
| `round3` | adapter=pass, calibrator=pass, selector=pass | adapter=pass, calibrator=pass, environment=pass, selector=pass |

## Observed Problems and Engineering Adjustments

- The initial long-batch preflight did not scan the complete training split and missed the actual worst collated geometry (`sample_index=6802`, 4 rows, 2009 tokens, attention risk `16144324`).
- A first Round 1 joint attempt reached an out-of-memory failure near the former high-risk region. The archived failure logs are retained as recovery evidence and are not counted as formal candidate results.
- The formal path now scans the complete split, uses expanded execution below the attention-risk threshold, and uses serial evidence/mixed execution with exact gradient splitting for high-risk batches. The real-Qwen joint and B-only worst-batch preflights then passed.
- Subsequent rounds remain validation-driven: B amplitude, output ratio, consistency weight/ramp, and selector learning rates may change only from validation diagnostics; no test result, control run, or additional seed drives a configuration change.

## Per-round Decisions

### round1

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

### round2

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

### round3

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

## Stage 2 Closeout Gate

All three rounds require final-test reports, passing audits, a passing tensor-contract closeout, and documented validation-driven decisions. The final interpretation must distinguish configuration observations from multi-seed evidence.

- Technical closeout: `eligible`.
- Best mixed test Overall AUC: `73.50` from `round1`.
- Delta vs A-module reference: `-1.88`.
- Delta vs paper target: `-3.21`.
- Research conclusion: the best audited Stage 2 candidate does not improve on the A-module reference; the requested three-round cycle can close as a negative configuration study, but the Stage 2 mechanism is not validated as a performance improvement and Stage 3 should not start from this result.
- Scope conclusion: this is a three-round, seed-1221 configuration study, not a multi-seed robustness claim.
