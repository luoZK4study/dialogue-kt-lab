# Repository Guidelines

## Scope

This workspace contains the MathDial ATC Dialogue-KT baseline and CEL modules
on Qwen3-1.7B + LoRA. The current research scope is A and A+B; future modules
are named C, D, and so on. The `data/` and `literature/` directories are
protected and are not part of code backup or cleanup operations.

Certified reference values:

- baseline: Overall Acc/AUC `69.22 / 74.95`, Final Acc/AUC `61.17 / 75.32`;
- historical A-module reference: Overall Acc/AUC `70.03 / 75.38`, Final
  Acc/AUC `66.80 / 77.26`;
- paper target: Overall AUC `76.71`.

The historical A checkpoint identifier
`cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b`
is provenance only. Use **A 模块** in method prose. A and A+B will be retrained
from the raw Qwen base after the refactor; no old result is a formal winner.

## Canonical Method Contract

- `H`: Qwen intermediate hidden state.
- `A`: selector; `a` is its tanh-bounded token-level selection strength.
- `B`: non-evidence representation transformer.
- `h_r = ((a + 1) / 2) ⊙ H` and `h_n = ((1 - a) / 2) ⊙ H` on evidence-eligible context tokens.
- `h_nb = B(h_n)` and `h_m = h_r + β · h_nb`.
- Shared downstream Qwen continuation and head produce `p_r = P(h_r)` and
  `p_m = P(h_m)`; report reversal `P(h_n)` as a diagnostic.
- Core objective: `λ_r BCE(p_r,y) + λ_m BCE(p_m,y) + λ_cons JS(p_r,p_m)`.

Do not use `score`, `signal`, `gate`, or `delta` as method-level synonyms for
`a`. Legacy field names may remain in compatibility code and audit schemas.

## Global Module Design Standard

The design standard applies to A, B, C, D, and every future module. Start from
the research hypothesis and causal mechanism, then specify the module's input
and output representations, information flow, downstream use, complete loss,
optimization schedule, and diagnostics. Capacity must match the intended
transformation. Lightweight modules are for interface checks, sanity checks,
and ablations only; they are not evidence for or replacements of the core
method.

## Unified Training Protocol

Unless a special mechanism is explicitly justified and audited, all enabled
modules (A, B, LoRA, optional calibrator, C/D, ...) are trained as one model:

1. Start each formal candidate from raw Qwen; never load baseline, historical
   anchors, or another candidate.
2. Put every enabled module in the same forward graph, complete objective, and
   optimizer from epoch 1.
3. Run train and validation every epoch. Fix `max_epochs`, validation metric,
   `patience`, and `min_delta` before training; save and restore the
   validation-best whole-model checkpoint before final test and audits.
4. Parameter-group learning rates are allowed, but do not turn training into
   independent, alternating, frozen, bootstrap, warmup, or calibrator-only
   stages.
5. Epoch N+1 continues from the previous epoch's whole-model state. A special
   staged protocol must be declared in its manifest and reported separately.

`L_budget` and `L_scale` are not core terms. First record selection and B-output
scale diagnostics; add a regularizer only after an observed collapse. A
calibrator is optional probability post-processing and, when enabled, must be
jointly trained from epoch 1 on both paths.

## Repository Layout

- `dialogue_kt/`: main Python package; keep CEL logic in `cel_methods.py` and
  training behavior in `training.py`.
- `scripts/cel/`: canonical A and A+B unified launchers.
- `scripts/cel_stage2_environment/`: reusable dual-path tensor-contract,
  evaluation, audit, and SSH synchronization tools.
- `results/baseline/`: certified baseline.
- `results/a/`, `results/a_b/`: new training outputs.
- `results/method_design/`: canonical method documents.

Use `/home/luo/miniconda3/envs/diagkt/bin/python` for local smoke tests. Formal
training uses SSH alias `3090`, remote root `/home/user4/dialogue-kt`, and the
existing `luo_2` environment. Do not change SSH target, port, identity, or
permissions. Use `apply_patch` for manual edits and inspect `git status` before
any deletion.
