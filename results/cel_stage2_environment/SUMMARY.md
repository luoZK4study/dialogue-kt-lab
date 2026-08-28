# CEL Stage 2 First-Round Historical Results

- Generated at: `2026-08-15T16:01:32`
- Audited candidates: `3`
- A-module reference Overall AUC: `75.38`
- Paper target Overall AUC: `76.71`
- Interpretation: legacy single-path environment-residual pilots; not the current dual-path A+B method.

## Current Dual-Path Execution Status

The default method rule is unified end-to-end training: all enabled modules,
including an optional calibrator and future modules, share one objective and
optimizer from epoch 1. Fix `max_epochs`, validation metric, `patience`, and
`min_delta`; run train/validation each epoch, restore the validation-best
checkpoint after early stopping, then run final test and audits. The training
CLI now implements this contract; `--epochs` remains a compatibility alias
and historical staged flags remain available for provenance. Existing
artifacts are not retroactively treated as compliant with this target.

> Status refresh: `2026-08-17 18:45 CST`. This is a moving execution snapshot, not a replacement for a fresh SSH check. The detailed handoff is `.claude/STAGE2_HANDOFF.md`.

- Round 1 and Round 2 completed training, validation, and validation audit using only seed `1221`; final tests remain frozen until Round 3 validation is complete.
- Round 1 Mixed Overall Acc/AUC: `66.78 / 72.51`; Round 2: `65.93 / 72.59`. Both are below the A-module reference `70.03 / 75.38` and paper target `76.71`.
- The unique three-round supervisor was active at the snapshot. Round 3 completed `a_bootstrap`, `calibrator_warmup`, and `a_joint`; `b_warmup` training completed `8425/8425`, and its phase validation was at `1112/2202`.
- The automatic remainder is Round 3 final `joint`, Round 3 validation/audit, all three final tests/audits, tensor-contract closeout, and `STAGE2_DUAL_PATH_REPORT.md` generation.
- Do not start another supervisor, add another seed, run controls/ablations, or start Stage 3. A new conversation must first perform a read-only check of SSH alias `3090` because the PID and progress snapshot will become stale.

| Candidate | Historical direction | Output | Audit | Overall Acc | Overall AUC | Final Acc | Final AUC | Delta AUC vs A module |
|---|---|---|---|---:|---:|---:|---:|---:|
| `stage2_mlp_topk20_beta005_seed1221` | `mlp` | `none` | `True` | 68.46 | 75.47 | 63.88 | 76.29 | +0.09 |
| `stage2_shuffle_topk20_beta010_seed1221` | `shuffle` | `none` | `True` | 68.72 | 74.14 | 62.52 | 72.79 | -1.24 |
| `stage2_transformer_topk20_beta010_centeredrms010_seed1221` | `transformer` | `centered_rms@0.1` | `True` | 68.06 | 75.25 | 67.18 | 74.73 | -0.13 |

## Historical Branch Diagnostics

The stored diagnostic keys predate the canonical terminology. `gate_mean` is displayed as the mean of `a`; the legacy selected/non-selected fields are displayed as historical `|a|` separation.

| Candidate | a Mean | Historical selected-|a| gap | Raw B-output L2 | Scaled B-output L2 | B/Input | B/A-evidence Cos | B/Input Cos |
|---|---:|---:|---:|---:|---:|---:|---:|
| `stage2_mlp_topk20_beta005_seed1221` | -0.8905 | 0.0896 | 1.41 | 0.07 | 0.0005 | -0.0050 | 0.0037 |
| `stage2_shuffle_topk20_beta010_seed1221` | 0.9560 | 0.0209 | 3118.57 | 311.86 | 1.0000 | 0.9997 | 0.7859 |
| `stage2_transformer_topk20_beta010_centeredrms010_seed1221` | -0.9858 | 0.0088 | 18.47 | 24.42 | 0.0813 | 0.0002 | -0.0076 |

## Paired Dialogue-Cluster Bootstrap

Resamples / seed: `10000 / 1221`. Intervals are paired percentage-point deltas versus the A-module reference.

| Candidate | Overall AUC delta | 95% CI | P(>0) | Final AUC delta | 95% CI | P(>0) | Non-final AUC delta | 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `stage2_mlp_topk20_beta005_seed1221` | +0.10 | [-1.13, +1.32] | 0.570 | -0.98 | [-3.52, +1.56] | 0.227 | +1.11 | [-0.24, +2.49] |
| `stage2_shuffle_topk20_beta010_seed1221` | -1.23 | [-2.77, +0.27] | 0.056 | -4.47 | [-7.64, -1.45] | 0.002 | +0.20 | [-1.55, +1.97] |
| `stage2_transformer_topk20_beta010_centeredrms010_seed1221` | -0.13 | [-1.45, +1.18] | 0.432 | -2.54 | [-5.53, +0.44] | 0.050 | +0.25 | [-1.23, +1.70] |

## Historical Interpretation and Current Scope

- All recorded first-round candidates remain historical pilots; no Stage 2 winner is declared.
- Their metrics do not test complementary `h_r/h_n`, separate `p_r/p_m`, or prediction consistency.
- The dual-path implementation, tensor contracts, and audit support for `a -> h_r/h_n/h_nb/h_m` are complete.
- The current main B is a 4-layer, width-1024 contextual Transformer; Shuffle, token-wise MLP, and small Transformer variants remain historical pilots.
- The active scope is three validation-driven configuration rounds using only seed `1221`. Controls, ablations, additional seeds, and Stage 3 are explicitly out of scope for this run.
- Evidence-weight saturation and B-output scale remain diagnostics; add a regularizer only if full-module training shows the corresponding collapse.
- Do not start Stage 3 before the target A+B mechanism is validated.
