# Strict Full-Train Report

> Terminology note: historical candidate identifiers are preserved for closeout and audit. Method documents refer to the audited successful design as the **A module**.

- Scope: only SSH-side complete `train + val + test` candidates are eligible.
- Excluded from winner judgment: calibrator-only, frozen retraining, fixed-bias eval, validation-fit bias, and any other post-hoc diagnostic flow.

## Current State

- Stage: `task_conditioned_tuning_completed_with_winner`
- Next action: `done`
- Winner found: `True`
- Current formal decision: `winner_or_done`
- Current formal rerun target: `none`
- Current formal recorded non-winner: `v25`
- Current formal recorded non-winner queue: `v21, v22, v23, v24, v25`
- Task-conditioned sync freshness: `fresh_remote`
- Latest authoritative task-conditioned remote refresh event: `[2026-08-12 23:17:24] task_conditioned refresh complete :: sync=ok sync_exit=0 stage=task_conditioned_tuning_completed_with_winner next_action=done live_phase=unknown round3=done,done,done,done,done,done winner=true duration=87s`
- Latest task-conditioned refresh attempt: `[2026-08-12 23:17:24] task_conditioned refresh complete :: sync=ok sync_exit=0 stage=task_conditioned_tuning_completed_with_winner next_action=done live_phase=unknown round3=done,done,done,done,done,done winner=true duration=87s`
- Task-conditioned controller pid/alive: `none / no`
- Task-conditioned controller command: `none`
- Task-conditioned controller child command: `none`
- Task-conditioned controller next wake estimate: `none`
- Current local report is backed by a fresh SSH sync.
- Recommended local poll interval: `not_applicable` (not applicable while no active strict full-train monitoring is required)
- Formal experiment audit summary: `recorded=6`
- Baseline: `lmkt_qwen3_1.7b_recert_20260620`
- Baseline Overall Acc / AUC: **69.22 / 74.95**
- Baseline Final Acc / AUC: **61.17 / 75.32**
- Historical selector/ranking comparison anchor: `cel_task_conditioned_lastlayer_v1_qwen3_1.7b`
- Comparison-anchor Overall Acc / AUC: **68.87 / 75.29**

## Formal Round 3 Candidates

| Candidate | Start Point | Method | Status | Overall Acc | Delta Overall Acc | Overall AUC | Delta Overall AUC | Final Acc | Delta Final Acc | Final AUC | Delta Final AUC | Overall Gate | Final Pair vs Baseline |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `v21` / `cel_task_conditioned_lastlayer_v21_joint_bias_fulltrain_qwen3_1.7b` | baseline checkpoint | joint bias calibrator full training | done | 68.61 | -0.61 | 75.26 | +0.31 | 56.70 | -4.47 | 74.60 | -0.72 | partial | no |
| `v22` / `cel_task_conditioned_lastlayer_v22_fullv1_joint_bias_tinylr_qwen3_1.7b` | full `v1` checkpoint | tiny-lr joint full fine-tune | done | 68.26 | -0.96 | 74.93 | -0.02 | 64.27 | +3.10 | 72.86 | -2.46 | no | partial |
| `v23` / `cel_task_conditioned_lastlayer_v23_joint_bias_posinit_qwen3_1.7b` | baseline checkpoint | keep the v21 joint-bias full-train setup but initialize the calibrator with a positive bias | done | 68.41 | -0.81 | 74.29 | -0.66 | 56.12 | -5.05 | 74.59 | -0.73 | no | no |
| `v24` / `cel_task_conditioned_lastlayer_v24_selectorinit_joint_bias_fulltrain_qwen3_1.7b` | baseline checkpoint | keep the v21 joint-bias full-train setup but initialize the selector from the historical selector/ranking v1 anchor | done | 68.16 | -1.06 | 74.44 | -0.51 | 56.89 | -4.28 | 73.49 | -1.83 | no | no |
| `v25` / `cel_task_conditioned_lastlayer_v25_joint_bias_calibratorlr10x_qwen3_1.7b` | baseline checkpoint | keep the v21 joint-bias full-train setup but assign the calibrator a higher dedicated learning rate | done | 66.90 | -2.32 | 74.98 | +0.03 | 52.04 | -9.13 | 74.27 | -1.05 | partial | no |
| `v26` / `cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b` | Qwen3-1.7B base weights with newly initialized A-module LoRA, selector, and calibrator | self-contained A-module chain: bootstrap representation training, calibrator-only warmup, then strict tiny-LR joint training | done | 70.03 | +0.81 | 75.38 | +0.43 | 66.80 | +5.63 | 77.26 | +1.94 | yes | yes |

## Decision

- A-module reference recorded from historical checkpoint `cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b` with Overall Acc / AUC **70.03 / 75.38**.
- Next step: keep the A-module result frozen, then implement and verify the explicit `h_r/h_m` dual-path objective, tensor contracts, and audit path before registering a formal Stage 2 candidate.

## Completed Candidate Analysis

### `v21` / `cel_task_conditioned_lastlayer_v21_joint_bias_fulltrain_qwen3_1.7b`

- Synced Overall Acc / AUC: **68.61 / 75.26**; Final Acc / AUC: **56.70 / 74.60**.
- `Pred True` at threshold `0.5`: **32.59%**.
- Best-threshold diagnostic: best Acc **69.52** at threshold **0.393**.
- Delta vs baseline Overall Acc / AUC: **-0.61 / +0.31**.
- Delta vs baseline Final Acc / AUC: **-4.47 / -0.72**.
- Delta vs anchor `v1` Overall Acc / AUC: **-0.26 / -0.03**.

### `v22` / `cel_task_conditioned_lastlayer_v22_fullv1_joint_bias_tinylr_qwen3_1.7b`

- Synced Overall Acc / AUC: **68.26 / 74.93**; Final Acc / AUC: **64.27 / 72.86**.
- `Pred True` at threshold `0.5`: **47.76%**.
- Best-threshold diagnostic: best Acc **69.07** at threshold **0.406**.
- Delta vs baseline Overall Acc / AUC: **-0.96 / -0.02**.
- Delta vs baseline Final Acc / AUC: **+3.10 / -2.46**.
- Delta vs anchor `v1` Overall Acc / AUC: **-0.61 / -0.36**.

### `v23` / `cel_task_conditioned_lastlayer_v23_joint_bias_posinit_qwen3_1.7b`

- Synced Overall Acc / AUC: **68.41 / 74.29**; Final Acc / AUC: **56.12 / 74.59**.
- `Pred True` at threshold `0.5`: **33.50%**.
- Best-threshold diagnostic: best Acc **68.87** at threshold **0.385**.
- Delta vs baseline Overall Acc / AUC: **-0.81 / -0.66**.
- Delta vs baseline Final Acc / AUC: **-5.05 / -0.73**.
- Delta vs anchor `v1` Overall Acc / AUC: **-0.46 / -1.00**.

### `v24` / `cel_task_conditioned_lastlayer_v24_selectorinit_joint_bias_fulltrain_qwen3_1.7b`

- Synced Overall Acc / AUC: **68.16 / 74.44**; Final Acc / AUC: **56.89 / 73.49**.
- `Pred True` at threshold `0.5`: **36.22%**.
- Best-threshold diagnostic: best Acc **68.87** at threshold **0.430**.
- Delta vs baseline Overall Acc / AUC: **-1.06 / -0.51**.
- Delta vs baseline Final Acc / AUC: **-4.28 / -1.83**.
- Delta vs anchor `v1` Overall Acc / AUC: **-0.71 / -0.85**.

### `v25` / `cel_task_conditioned_lastlayer_v25_joint_bias_calibratorlr10x_qwen3_1.7b`

- Synced Overall Acc / AUC: **66.90 / 74.98**; Final Acc / AUC: **52.04 / 74.27**.
- `Pred True` at threshold `0.5`: **30.18%**.
- Best-threshold diagnostic: best Acc **69.87** at threshold **0.392**.
- Delta vs baseline Overall Acc / AUC: **-2.32 / +0.03**.
- Delta vs baseline Final Acc / AUC: **-9.13 / -1.05**.
- Delta vs anchor `v1` Overall Acc / AUC: **-1.97 / -0.31**.

### `v26` / `cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b`

- Synced Overall Acc / AUC: **70.03 / 75.38**; Final Acc / AUC: **66.80 / 77.26**.
- `Pred True` at threshold `0.5`: **43.98%**.
- Best-threshold diagnostic: best Acc **70.53** at threshold **0.520**.
- Delta vs baseline Overall Acc / AUC: **+0.81 / +0.43**.
- Delta vs baseline Final Acc / AUC: **+5.63 / +1.94**.
- Delta vs anchor `v1` Overall Acc / AUC: **+1.16 / +0.09**.


## Operator Loop

1. Launch or rerun only through the SSH strict full-train path; local runs do not count as formal evidence.
2. Monitor with the unified monitor entrypoint and adjust cadence based on live phase.
3. After each landed SSH run, use the unified finalize entrypoint to sync, rebuild, and verify closeout readiness.
4. Check `FORMAL_EXPERIMENT_AUDIT.md` before declaring a result recorded or successful.
5. Review the candidate from the rebuilt authoritative surfaces before judging winner status.
6. If the method still misses baseline on either metric, record the result and change only one main variable in the next formal design.

Recommended commands:

```bash
bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090
bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090 --dry-run
bash scripts/cel_stage1_last_layer/monitor_formal_candidate.sh once
bash scripts/cel_stage1_last_layer/monitor_formal_candidate_via_ssh_alias.sh 3090 once
bash scripts/cel_stage1_last_layer/finalize_current_formal_candidate.sh
bash scripts/cel_stage1_last_layer/finalize_current_formal_candidate_via_ssh_alias.sh 3090
bash scripts/cel_stage1_last_layer/review_current_formal_candidate.sh
bash scripts/cel_stage1_last_layer/review_current_formal_candidate_via_ssh_alias.sh 3090
bash scripts/cel_stage1_last_layer/finalize_formal_candidate_via_ssh_alias.sh 3090 v26
bash scripts/cel_stage1_last_layer/review_formal_candidate_via_ssh_alias.sh 3090 v26
```

- Current recommended local monitoring cadence: `not_applicable` (not applicable while no active strict full-train monitoring is required).

## Rule

- A new method only counts as successful when one strict full-train candidate finishes on SSH and achieves both `Overall Acc > baseline` and `Overall AUC > baseline`.
- Every completed candidate must also report Final Acc / AUC and their two deltas versus baseline; Final-turn comparison is a required diagnostic and audit surface, not a replacement for the Overall winner gate.
