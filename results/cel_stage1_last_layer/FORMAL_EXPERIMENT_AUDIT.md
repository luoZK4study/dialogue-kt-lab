# Formal Experiment Audit

- Scope: only strict SSH-side full `train + val + test` candidates are audited here.
- Purpose: verify that every completed formal candidate has synced artifacts and has been reflected across the required markdown surfaces before any winner claim.

## Audit Rule

1. A formal candidate is only completion-ready after synced `metrics`, `qual`, and `stdout log` all exist, and its metrics contain Overall Acc/AUC plus Final Acc/AUC.
2. A completed formal candidate must also be visible in `STATUS.md`, `STRICT_FULL_TRAIN_REPORT.md`, `CEL_Stage1_LastLayer_DialogueKT_实验记录.md`, `TASK_CONDITIONED_TUNING_LOOP.md`, `COMPARISON.md`, `DETAILED_ANALYSIS.md`, `FORMAL_CANDIDATE_BRIEFS.md`, and `FORMAL_CANDIDATE_CONFIG_DIFFS.md`.
   For completed formal candidates, the core markdown surfaces must carry Overall Acc / AUC and Final Acc / AUC. The strict report, Stage 1 record, tuning loop, `COMPARISON.md`, `DETAILED_ANALYSIS.md`, and candidate briefs must explicitly expose both Overall and Final deltas versus baseline. The strict report and candidate briefs must also carry `Pred True`, the best-threshold diagnostic, and anchor deltas when available.
3. Winner judgment is only allowed after the above audit is clean and the candidate beats baseline on both Overall Acc and Overall AUC.
4. Failed strict full-train candidates must still be reflected in `STATUS.md`, the strict report, the Stage 1 record, the tuning loop, and the candidate briefs before they count as properly analyzed rerun targets.

## Current Gate

- Baseline gate: `Overall Acc > 69.22` and `Overall AUC > 74.95`.
- Current next action: `done`
- Winner found: `True`
- Task-conditioned sync freshness: `fresh_remote`
- Latest authoritative task-conditioned remote refresh event: `[2026-08-12 23:17:24] task_conditioned refresh complete :: sync=ok sync_exit=0 stage=task_conditioned_tuning_completed_with_winner next_action=done live_phase=unknown round3=done,done,done,done,done,done winner=true duration=87s`
- Latest task-conditioned refresh attempt: `[2026-08-12 23:17:24] task_conditioned refresh complete :: sync=ok sync_exit=0 stage=task_conditioned_tuning_completed_with_winner next_action=done live_phase=unknown round3=done,done,done,done,done,done winner=true duration=87s`
- Current audit is backed by a fresh SSH sync.

## Candidate Audit

| Candidate | Status | Metrics+Qual | Metric Set | Stdout Log | Markdown Coverage | Beats Baseline? | Audit State |
|---|---|---|---|---|---|---|---|
| `v21` / `cel_task_conditioned_lastlayer_v21_joint_bias_fulltrain_qwen3_1.7b` | done | ok | complete | ok | ok | no | `recorded` |
| `v22` / `cel_task_conditioned_lastlayer_v22_fullv1_joint_bias_tinylr_qwen3_1.7b` | done | ok | complete | ok | ok | no | `recorded` |
| `v23` / `cel_task_conditioned_lastlayer_v23_joint_bias_posinit_qwen3_1.7b` | done | ok | complete | ok | ok | no | `recorded` |
| `v24` / `cel_task_conditioned_lastlayer_v24_selectorinit_joint_bias_fulltrain_qwen3_1.7b` | done | ok | complete | ok | ok | no | `recorded` |
| `v25` / `cel_task_conditioned_lastlayer_v25_joint_bias_calibratorlr10x_qwen3_1.7b` | done | ok | complete | ok | ok | no | `recorded` |
| `v26` / `cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b` | done | ok | complete | ok | ok | yes | `recorded` |

## Notes

- `v21`: Full-train artifacts are present and the candidate's result is reflected across all required markdown surfaces.
- `v21` synced Overall Acc / AUC: **68.61 / 75.26**.
- `v21` synced Final Acc / AUC: **56.70 / 74.60**.
- `v21` synced `Pred True`: **32.59%**.
- `v21` best-threshold diagnostic: best Acc **69.52** at threshold **0.393**.
- `v21` delta vs baseline Overall Acc / AUC: **-0.61 / +0.31**.
- `v21` delta vs baseline Final Acc / AUC: **-4.47 / -0.72**.
- `v21` delta vs anchor `v1` Overall Acc / AUC: **-0.26 / -0.03**.
- `v22`: Full-train artifacts are present and the candidate's result is reflected across all required markdown surfaces.
- `v22` synced Overall Acc / AUC: **68.26 / 74.93**.
- `v22` synced Final Acc / AUC: **64.27 / 72.86**.
- `v22` synced `Pred True`: **47.76%**.
- `v22` best-threshold diagnostic: best Acc **69.07** at threshold **0.406**.
- `v22` delta vs baseline Overall Acc / AUC: **-0.96 / -0.02**.
- `v22` delta vs baseline Final Acc / AUC: **+3.10 / -2.46**.
- `v22` delta vs anchor `v1` Overall Acc / AUC: **-0.61 / -0.36**.
- `v23`: Full-train artifacts are present and the candidate's result is reflected across all required markdown surfaces.
- `v23` synced Overall Acc / AUC: **68.41 / 74.29**.
- `v23` synced Final Acc / AUC: **56.12 / 74.59**.
- `v23` synced `Pred True`: **33.50%**.
- `v23` best-threshold diagnostic: best Acc **68.87** at threshold **0.385**.
- `v23` delta vs baseline Overall Acc / AUC: **-0.81 / -0.66**.
- `v23` delta vs baseline Final Acc / AUC: **-5.05 / -0.73**.
- `v23` delta vs anchor `v1` Overall Acc / AUC: **-0.46 / -1.00**.
- `v24`: Full-train artifacts are present and the candidate's result is reflected across all required markdown surfaces.
- `v24` synced Overall Acc / AUC: **68.16 / 74.44**.
- `v24` synced Final Acc / AUC: **56.89 / 73.49**.
- `v24` synced `Pred True`: **36.22%**.
- `v24` best-threshold diagnostic: best Acc **68.87** at threshold **0.430**.
- `v24` delta vs baseline Overall Acc / AUC: **-1.06 / -0.51**.
- `v24` delta vs baseline Final Acc / AUC: **-4.28 / -1.83**.
- `v24` delta vs anchor `v1` Overall Acc / AUC: **-0.71 / -0.85**.
- `v25`: Full-train artifacts are present and the candidate's result is reflected across all required markdown surfaces.
- `v25` synced Overall Acc / AUC: **66.90 / 74.98**.
- `v25` synced Final Acc / AUC: **52.04 / 74.27**.
- `v25` synced `Pred True`: **30.18%**.
- `v25` best-threshold diagnostic: best Acc **69.87** at threshold **0.392**.
- `v25` delta vs baseline Overall Acc / AUC: **-2.32 / +0.03**.
- `v25` delta vs baseline Final Acc / AUC: **-9.13 / -1.05**.
- `v25` delta vs anchor `v1` Overall Acc / AUC: **-1.97 / -0.31**.
- `v26`: Full-train artifacts are present and the candidate's result is reflected across all required markdown surfaces.
- `v26` synced Overall Acc / AUC: **70.03 / 75.38**.
- `v26` synced Final Acc / AUC: **66.80 / 77.26**.
- `v26` synced `Pred True`: **43.98%**.
- `v26` best-threshold diagnostic: best Acc **70.53** at threshold **0.520**.
- `v26` delta vs baseline Overall Acc / AUC: **+0.81 / +0.43**.
- `v26` delta vs baseline Final Acc / AUC: **+5.63 / +1.94**.
- `v26` delta vs anchor `v1` Overall Acc / AUC: **+1.16 / +0.09**.

## Current Read

- Every completed formal candidate currently has the required artifacts and markdown coverage.
- Keep using this audit together with `STRICT_FULL_TRAIN_REPORT.md` before declaring any new Stage 1 formal winner.
