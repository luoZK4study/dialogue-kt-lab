# CEL Stage 2 Results and Interpretation

This directory stores the completed first-round Stage 2 artifacts: metrics,
per-turn predictions, logs, manifests, paired bootstrap reports, and provenance
audits.

## Canonical terminology

- `H`: Qwen intermediate hidden state.
- `a`: A-module token-level selection strength in approximately `[-1, 1]`.
- `h_r = ((a + 1) / 2) ⊙ H`: evidence representation.
- `h_n = ((1 - a) / 2) ⊙ H`: non-evidence representation.
- `h_nb = B(h_n)`: B-transformed non-evidence representation.
- `h_m = h_r + β · h_nb`: mixed representation.

The complementary weights apply only to evidence-eligible context tokens.
Other valid prompt tokens remain unchanged in `h_r/h_m` and are zero in
`h_n`.

Legacy source and diagnostic fields such as `signal` and `gate_mean` remain in
historical artifacts for compatibility. Their method-level interpretation is
the selection strength `a`.

## First-round status

All three first-round candidates completed their self-contained training chain
and passed audit:

| Historical direction | Candidate | Overall Acc/AUC | Final Acc/AUC | AUC delta vs A-module reference |
|---|---|---:|---:|---:|
| Shuffle | `stage2_shuffle_topk20_beta010_seed1221` | `68.72 / 74.14` | `62.52 / 72.79` | `-1.24` |
| Token-wise MLP | `stage2_mlp_topk20_beta005_seed1221` | `68.46 / 75.47` | `63.88 / 76.29` | `+0.09` |
| Small Transformer | `stage2_transformer_topk20_beta010_centeredrms010_seed1221` | `68.06 / 75.25` | `67.18 / 74.73` | `-0.13` |

The A-module reference has Overall Acc/AUC `70.03 / 75.38` and Final Acc/AUC
`66.80 / 77.26`. Its historical checkpoint name is retained in audit files
for provenance only.

The MLP point estimate is highest, but its paired Overall AUC delta is `+0.10`
with 95% CI `[-1.13, +1.32]`. No first-round candidate establishes a stable
improvement.

## Interpretation boundary

The first-round candidates used a legacy single-path residual flow. They did
not implement the current target method because they did not:

- construct complementary `h_r` and `h_n` from `(a + 1) / 2` and
  `(1 - a) / 2`;
- run the same downstream Qwen network separately on `h_r` and `h_m`;
- compute evidence-path and mixed-path task losses;
- constrain the two predictions with a consistency loss.

Consequently, this directory records historical pilots and engineering
diagnostics, not a formal validation of the target A+B mechanism. No Stage 2
winner is declared.

## Default training protocol

The target protocol is a single end-to-end train/validation loop. A, B, LoRA,
and any enabled calibrator or future module share one optimizer and the full
objective from epoch 1. Fix `max_epochs`, validation metric, `patience`, and
`min_delta` before training; save the validation-best checkpoint, early-stop
after patience, restore the best checkpoint, then run final test and audits.
Different parameter-group learning rates are allowed, but default freezing,
alternation, warmup, per-batch loss ramps, and calibrator-only fitting are not.

The calibrator is optional and only calibrates output probabilities after the
shared prediction head. If enabled, use the same instance for evidence and
mixed paths (and reversal at evaluation) and train it jointly from epoch 1.
Pure monotonic post-processing does not change AUC, so raw and calibrated
probabilities must be reported separately. The training CLI now exposes
`--max_epochs`, `--patience`, and `--min_delta` (`--epochs` remains a
compatibility alias), while historical phase flags remain available for
provenance. Existing artifacts are not retroactively treated as compliant
with the unified protocol.

The historical `pre_block` tensor contract remains useful: it proves that a
context intervention must occur before a causal block that can propagate the
change to the prediction token. It does not by itself establish evidence
sufficiency or prediction invariance.

## Current next work

1. Migrate the training loop and launcher from historical phase flags to the
   unified protocol, including `patience`/`min_delta` and best-checkpoint restore.
2. Verify shared-downstream dual-path prediction and the three-term objective in
   `dialogue_kt/training.py` under the unified loop.
3. Maintain complementarity, gradient, causal-propagation, finite-value, and
   dual-prediction tensor contracts.
4. Design a capacity-adequate B module. Lightweight B modules remain sanity
   checks or ablations only.
5. Register A-only, β=0, Frozen-B, No-consistency, capacity, and multi-seed
   controls.

Record evidence-weight saturation and B-output scale as diagnostics. Add a
corresponding regularizer only if the full-module training exhibits that
specific collapse.

Do not start a new formal candidate or Stage 3 until these contracts are
implemented and verified.
