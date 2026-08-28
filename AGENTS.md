# Repository Guidelines

## Current Focus

This workspace is intentionally narrowed to baseline + CEL A/B on MathDial ATC with Qwen3-1.7B + LoRA. The paper target remains `76.71` Overall AUC.

Current state:

- Certified baseline: Overall Acc/AUC `69.22 / 74.95`, Final Acc/AUC `61.17 / 75.32`.
- Audited A-module reference: Overall Acc/AUC `70.03 / 75.38`, Final Acc/AUC `66.80 / 77.26`.
- The A-module artifact is historically stored as `cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b`. Keep this identifier for provenance only; method prose must call it the **A module**.
- Stage 2 first-round Shuffle, token-wise MLP, and small Transformer runs are complete and audited. They used the legacy single-path environment-residual flow and are historical pilots, not a validation of the current target method. No Stage 2 winner is declared.
- Current work is to implement the explicit `h_r/h_m` dual-path objective. Do not start a new formal run or Stage 3 until the code, tensor contracts, and audit path are updated.
- The latest recorded authoritative remote refresh for the A-module artifacts completed at `2026-08-12 23:17:24`. `fresh_remote` refers to that event only.

## Canonical Terminology

- `H`: Qwen intermediate-layer hidden state.
- `A`: selector.
- `a`: A's token-level selection strength. The current A applies `tanh`, so `a` is approximately in `[-1, 1]`.
- Current A additive evidence enhancement: `H' = H + γ(a ⊙ H)`.
- `h_r = ((a + 1) / 2) ⊙ H`: evidence representation.
- `h_n = ((1 - a) / 2) ⊙ H`: non-evidence representation.
- `h_nb = B(h_n)`: B-transformed non-evidence representation.
- `h_m = h_r + β · h_nb`: mixed representation.

The complementary weights apply only to evidence-eligible context tokens. Other valid prompt tokens remain unchanged in `h_r/h_m` and are zero in `h_n`.

Do not use `score`, `signal`, `gate`, or `delta` as method-level synonyms for `a`. Legacy source identifiers and diagnostic keys may remain for compatibility, but their prose interpretation must use `a`.

## Target Objective

Use the same downstream Qwen continuation and prediction head for both paths:

```text
p_r = P(h_r)
p_m = P(h_m)
```

The core Stage 2 objective is:

```text
L_total = λ_r BCE(p_r, y)
        + λ_m BCE(p_m, y)
        + λ_cons JS([p_r, 1-p_r], [p_m, 1-p_m])
```

`L_budget` and `L_scale` are not part of the current core objective. Record the selection distribution and B-output scale first; add either regularizer only if the corresponding collapse is observed in full-module training.

Also report reversal performance from `P(h_n)`.

## Method Design Principle

Method design must first serve the research hypothesis and target causal mechanism. Minimal changes, minimal parameter count, or the lowest single-run cost are not primary objectives for a formal design. Derive a complete module from the required representation contract, information flow, downstream use, training objective, optimization schedule, and diagnostics. Module capacity must match the complexity of the intended transformation. Lightweight modules are valid for interface checks, sanity checks, and ablations, but not as the main evidence for or against the research direction. Do not save formal training cost by weakening the core module. Optimize parameter efficiency only after the mechanism is adequately validated.

## Read Order

1. `.claude/STAGE2_HANDOFF.md`
2. `.claude/CLAUDE.md`
3. `.claude/MEMORY.md`
4. `.claude/STRUCTURE.md`
5. `results/method_design/CEL_Modular_Flow.md`
6. `results/method_design/CEL_Stage1_A_Module_Implementation.md`
7. `results/method_design/CEL_Stage2_Environment_Generator_Implementation.md`
8. `results/cel_stage2_environment/SUMMARY.md`

## Project Structure

`dialogue_kt/` is the main package. `main.py` defines the CLI, `training.py` handles train/test flows, `kt_data_loading.py` and `prompting.py` build the LLMKT input pipeline, and `cel_methods.py` contains A/B and hook logic.

- `results/baseline/`: certified baseline artifacts.
- `results/cel_stage1_last_layer/`: A-module history, metrics, logs, and audits.
- `results/cel_stage2_environment/`: legacy first-round Stage 2 results and audits.
- `results/method_design/`: canonical method documents.
- `scripts/cel_stage1_last_layer/`: A-module historical training and audit tools.
- `scripts/cel_stage2_environment/`: Stage 2 launcher, tensor-contract, audit, and report tools.

## Environment

- Local root: `/home/luo/work/dialogue-kt_migration_20260806`
- Local test environment: `/home/luo/miniconda3/envs/diagkt`
- Local GPU: NVIDIA GeForce RTX 4050 Laptop GPU, 6GB
- Formal training: SSH alias `3090`, remote root `/home/user4/dialogue-kt`, dual RTX 3090
- Local `saved_models/` contains no formal checkpoints.

Keep `/home/luo/.ssh` at mode `700` and SSH config/private keys at mode `600`. Do not change the preserved SSH target, port, identity, or remote repository settings.

## Formal Experiment Rules

- Each formal A+B candidate starts from the raw Qwen3-1.7B base.
- A candidate may load only its own immediately preceding phase checkpoint.
- Do not initialize from baseline, historical anchors, or another candidate.
- A result is formal only after train, validation, final test, provenance audit, parameter audit, and closeout.
- Best configurations require at least three model/data seeds.
- Do not launch formal training until the dual-path code and tensor contracts are complete.

## Default Training Protocol

- Unless a module explicitly declares and justifies a special mechanism, train all enabled modules as one end-to-end model from epoch 1. A, B, LoRA, an optional calibrator, and future modules such as C/D must share the forward graph, complete objective, optimizer, train/validation loop, and early-stopping decision.
- Fix `max_epochs`, validation metric, `patience`, and `min_delta` before training. Run train and validation every epoch, keep the validation-best checkpoint, stop after the configured patience, restore that checkpoint, then run final test and audits.
- Parameter-group learning rates are allowed, but do not treat them as independent training. Bootstrap, freezing, alternating updates, warmup, per-batch loss ramps, and calibrator-only fitting are special protocols only; record their motivation, scope, schedule, enabling condition, and audit criteria in the method document and manifest.
- The calibrator is optional. It only performs monotonic probability calibration after the shared prediction head; it does not learn A/B representations and is not required for the dual-path objective. If enabled, initialize it explicitly and train it jointly from epoch 1 on both paths. Report raw and calibrated probabilities separately; pure monotonic post-processing does not change AUC.

## Coding Style

Use 4-space indentation and snake_case. Keep CEL logic in `cel_methods.py` or focused helpers. Preserve historical artifact names in files and audit schemas; use canonical method terminology in prose and new user-facing labels.

## Worktree Safety

The worktree already contains many user modifications and deletions. Preserve them. Do not reset, checkout, or clean unrelated changes. Use `apply_patch` for manual edits.

## Commit Guidance

Use concise prefixes such as `feat:`, `fix:`, and `docs:`. Summaries should state the method motivation, changed commands or contracts, affected output paths, and `.claude/` status updates.
