# STATUS

> Terminology note: model and candidate identifiers below retain their historical names for provenance. The audited current method is referred to as the **A module** in method documents. Stage 2 first-round records use a legacy single-path environment-residual flow and are historical pilots, not the current `h_r/h_m` dual-path target method.

- Generated at: `2026-08-15T15:37:02`
- Stage: `task_conditioned_tuning_completed_with_winner`
- Latest loop event: `none`
- Latest follow-up event: `none`
- Task-conditioned sync freshness: `fresh_remote`
- Latest authoritative task-conditioned remote refresh event: `[2026-08-12 23:17:24] task_conditioned refresh complete :: sync=ok sync_exit=0 stage=task_conditioned_tuning_completed_with_winner next_action=done live_phase=unknown round3=done,done,done,done,done,done winner=true duration=87s`
- Latest task-conditioned sync event: `[2026-08-12 23:17:24] task_conditioned refresh complete :: sync=ok sync_exit=0 stage=task_conditioned_tuning_completed_with_winner next_action=done live_phase=unknown round3=done,done,done,done,done,done winner=true duration=87s`
- Latest task-conditioned refresh attempt: `[2026-08-12 23:17:24] task_conditioned refresh complete :: sync=ok sync_exit=0 stage=task_conditioned_tuning_completed_with_winner next_action=done live_phase=unknown round3=done,done,done,done,done,done winner=true duration=87s`
- Latest task-conditioned watcher event: `none`
- Latest task-conditioned controller event: `none`
- Task-conditioned controller pid/alive: `none / no`
- Task-conditioned controller command: `none`
- Task-conditioned controller child command: `none`
- Task-conditioned controller next wake estimate: `none`
- Snapshot provenance: `fresh_remote` means the latest recorded refresh succeeded; it does not prove that the copied checkout has revalidated the remote state on the report-generation date.
- Authoritative task-conditioned snapshot should be read from the latest successful remote refresh above; watcher/controller events are process-log breadcrumbs and may lag behind that recorded snapshot.
- Task-conditioned next action: `done`
- Task-conditioned winner found: `True`
- Formal experiment audit summary: `recorded=6`
- Recommended task-conditioned poll interval: `not_applicable` (not applicable while no active strict full-train monitoring is required)
- Current formal decision: `winner_or_done`
- Current formal rerun target: `none`
- Current formal recorded non-winner: `v25`
- Current formal recorded non-winner queue: `v21, v22, v23, v24, v25`

## Baseline
- Model: `lmkt_qwen3_1.7b_recert_20260620`
- Loss: `0.5928`
- Overall Acc: `69.22`
- Overall AUC: `74.95`
- Final Acc: `61.17`
- Final AUC: `75.32`

## CEL Stage1 Last-Layer

## Task-Conditioned Round 3
- `cel_task_conditioned_lastlayer_v21_joint_bias_fulltrain_qwen3_1.7b`: `done`, `testing`, epoch `1`, `1985/1985 (100%)`, Overall Acc/AUC `68.61/75.26`, Final Acc/AUC `56.70/74.60`
- `cel_task_conditioned_lastlayer_v22_fullv1_joint_bias_tinylr_qwen3_1.7b`: `done`, `testing`, epoch `1`, `1985/1985 (100%)`, Overall Acc/AUC `68.26/74.93`, Final Acc/AUC `64.27/72.86`
- `cel_task_conditioned_lastlayer_v23_joint_bias_posinit_qwen3_1.7b`: `done`, `testing`, epoch `1`, `1985/1985 (100%)`, Overall Acc/AUC `68.41/74.29`, Final Acc/AUC `56.12/74.59`
- `cel_task_conditioned_lastlayer_v24_selectorinit_joint_bias_fulltrain_qwen3_1.7b`: `done`, `testing`, `1985/1985 (100%)`, Overall Acc/AUC `68.16/74.44`, Final Acc/AUC `56.89/73.49`
- `cel_task_conditioned_lastlayer_v25_joint_bias_calibratorlr10x_qwen3_1.7b`: `done`, `testing`, epoch `1`, `1985/1985 (100%)`, Overall Acc/AUC `66.90/74.98`, Final Acc/AUC `52.04/74.27`
- `cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b`: `done`, `testing`, `1985/1985 (100%)`, Overall Acc/AUC `70.03/75.38`, Final Acc/AUC `66.80/77.26`

## Current Formal Workflow
- Current priority: preserve the A-module reference from historical checkpoint `cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b`; do not launch another Stage 1 candidate.
- Current local state helper: `python3 scripts/cel_stage1_last_layer/print_current_formal_next_action.py`
- Verified local-snapshot review in this WSL checkout: `TASK_CONDITIONED_REVIEW_SKIP_REFRESH=1 bash scripts/cel_stage1_last_layer/review_formal_candidate.sh v26`
- Stage 2 first-round legacy single-path experiments are complete and audited, but remain historical pilots; the next work is the explicit `h_r/h_m` dual-path implementation, tensor contracts, and audit path.
- Copied-workspace note: the active Stage 1 orchestration resolves the local repository root from each script location; SSH target and remote repository settings remain unchanged.

## Formal Loop Summary
- Formal method results only count after one SSH-side complete `train + val + test` run.
- Any calibrator-only, frozen retraining, fixed-bias eval, validation-fit bias, or other post-hoc ckpt adjustment stays diagnostic-only.
- Every completed formal SSH run must also pass the `finalize -> audit -> review` closeout path before it is treated as recorded and analyzed.
- If the current environment cannot open SSH itself, print the strict manual launch fallback and run the same registered candidate from another reachable terminal.
- A recorded strict formal winner already exists in the audit-ready queue; keep all conclusion surfaces frozen, stop launching new formal candidates, and only reopen the loop for an explicit strict rerun.
- Current recommended local monitoring cadence: `not_applicable` (not applicable while no active strict full-train monitoring is required). Early long training can stay slower, stable mid-run stays moderate, and near-finish or `Validation / Testing` tightens automatically.
- After each completed SSH run, process results in this order: sync `metrics / qual / stdout log` -> rebuild `task_conditioned_status.json` and markdown surfaces -> check `FORMAL_EXPERIMENT_AUDIT.md` -> only then judge baseline win/loss.
- If the candidate still fails to beat baseline on both Overall Acc and Overall AUC, record the result and change only one main design variable for the next formal candidate.

## Synced CEL Metrics
| Model | Type | Loss | Overall Acc | Overall AUC | Final Acc | Final AUC | Diagnostics |
|---|---|---:|---:|---:|---:|---:|---|
| `cel_adapter_lastlayer_v1_qwen3_1.7b` | formal | 0.6149 | 68.21 | 75.04 | 65.63 | 72.66 | gate_mean: 0.0000, gate_abs_mean: 0.0000, gate_pos_frac: 0.0000, gate_neg_frac: 0.0000, gate_max: 0.0000, gate_min: 0.0000, injected_mean: 1.0821 |
| `cel_mlp_lastlayer_v1_qwen3_1.7b` | formal | 0.6906 | 65.74 | 74.81 | 69.32 | 72.73 | gate_mean: 0.0000, gate_abs_mean: 0.0000, gate_pos_frac: 0.0000, gate_neg_frac: 0.0000, gate_max: 0.0000, gate_min: 0.0000, injected_mean: 1.0770 |
| `cel_task_conditioned_lastlayer_v11_cal_bias_only_qwen3_1.7b` | diagnostic | 0.6733 | 65.04 | 73.83 | 49.90 | 73.84 | gate_mean: 0.0000, gate_abs_mean: 0.0000, gate_pos_frac: 0.0000, gate_neg_frac: 0.0000, gate_max: 0.0000, gate_min: 0.0000, injected_mean: 1.0836, cal_bias: 0.0945 |
| `cel_task_conditioned_lastlayer_v12_cal_affine_only_qwen3_1.7b` | diagnostic | 0.6294 | 65.49 | 73.83 | 50.49 | 73.84 | gate_mean: 0.0000, gate_abs_mean: 0.0000, gate_pos_frac: 0.0000, gate_neg_frac: 0.0000, gate_max: 0.0000, gate_min: 0.0000, injected_mean: 1.0836, cal_bias: 0.0890, cal_scale: 0.6794 |
| `cel_task_conditioned_lastlayer_v15_fullv1_cal_bias_only_qwen3_1.7b` | diagnostic | 0.6078 | 69.27 | 75.29 | 61.94 | 75.27 | gate_mean: 0.0000, gate_abs_mean: 0.0000, gate_pos_frac: 0.0000, gate_neg_frac: 0.0000, gate_max: 0.0000, gate_min: 0.0000, injected_mean: 1.1436, cal_bias: 0.3802 |
| `cel_task_conditioned_lastlayer_v1_qwen3_1.7b` | formal | 0.6300 | 68.87 | 75.29 | 56.31 | 75.27 | gate_mean: 0.0000, gate_abs_mean: 0.0000, gate_pos_frac: 0.0000, gate_neg_frac: 0.0000, gate_max: 0.0000, gate_min: 0.0000, injected_mean: 1.1436 |
| `cel_task_conditioned_lastlayer_v21_joint_bias_fulltrain_qwen3_1.7b` | formal | 0.6340 | 68.61 | 75.26 | 56.70 | 74.60 | gate_mean: 0.0000, gate_abs_mean: 0.0000, gate_pos_frac: 0.0000, gate_neg_frac: 0.0000, gate_max: 0.0000, gate_min: 0.0000, injected_mean: 1.1551, cal_bias: -0.0001, post_prob_invalid_frac: 0.0000, post_prob_label_invalid_frac: 0.0000, post_prob_label_out_of_range_frac: 0.0000, post_prob_label_raw_max: 0.4736, post_prob_label_raw_min: 0.4736, post_prob_label_safe_max: 0.4736, post_prob_label_safe_min: 0.4736, post_prob_label_sanitized_frac: 0.0000, post_prob_out_of_range_frac: 0.0000, post_prob_raw_max: 0.3806, post_prob_raw_min: 0.3806, post_prob_safe_max: 0.3806, post_prob_safe_min: 0.3806, post_prob_sanitized_frac: 0.0000, pre_prob_invalid_frac: 0.0000, pre_prob_out_of_range_frac: 0.0000, pre_prob_raw_max: 0.3806, pre_prob_raw_min: 0.3806, pre_prob_safe_max: 0.3806, pre_prob_safe_min: 0.3806, pre_prob_sanitized_frac: 0.0000 |
| `cel_task_conditioned_lastlayer_v22_fullv1_joint_bias_tinylr_qwen3_1.7b` | formal | 0.8027 | 68.26 | 74.93 | 64.27 | 72.86 | gate_mean: 0.0000, gate_abs_mean: 0.0000, gate_pos_frac: 0.0000, gate_neg_frac: 0.0000, gate_max: 0.0000, gate_min: 0.0000, injected_mean: 1.1202, cal_bias: 0.0001, post_prob_invalid_frac: 0.0000, post_prob_out_of_range_frac: 0.0000, post_prob_raw_max: 0.4834, post_prob_raw_min: 0.4834, post_prob_safe_max: 0.4834, post_prob_safe_min: 0.4834, post_prob_sanitized_frac: 0.0272, pre_prob_invalid_frac: 0.0000, pre_prob_out_of_range_frac: 0.0000, pre_prob_raw_max: 0.4833, pre_prob_raw_min: 0.4833, pre_prob_safe_max: 0.4833, pre_prob_safe_min: 0.4833, pre_prob_sanitized_frac: 0.0272 |
| `cel_task_conditioned_lastlayer_v23_joint_bias_posinit_qwen3_1.7b` | formal | 0.6301 | 68.41 | 74.29 | 56.12 | 74.59 | gate_mean: 0.0000, gate_abs_mean: 0.0000, gate_pos_frac: 0.0000, gate_neg_frac: 0.0000, gate_max: 0.0000, gate_min: 0.0000, injected_mean: 1.1407, cal_bias: 0.3302, post_prob_invalid_frac: 0.0000, post_prob_label_invalid_frac: 0.0000, post_prob_label_out_of_range_frac: 0.0000, post_prob_label_raw_max: 0.4736, post_prob_label_raw_min: 0.4736, post_prob_label_safe_max: 0.4736, post_prob_label_safe_min: 0.4736, post_prob_label_sanitized_frac: 0.0000, post_prob_out_of_range_frac: 0.0000, post_prob_raw_max: 0.3921, post_prob_raw_min: 0.3921, post_prob_safe_max: 0.3921, post_prob_safe_min: 0.3921, post_prob_sanitized_frac: 0.0000, pre_prob_invalid_frac: 0.0000, pre_prob_out_of_range_frac: 0.0000, pre_prob_raw_max: 0.3413, pre_prob_raw_min: 0.3413, pre_prob_safe_max: 0.3413, pre_prob_safe_min: 0.3413, pre_prob_sanitized_frac: 0.0000 |
| `cel_task_conditioned_lastlayer_v24_selectorinit_joint_bias_fulltrain_qwen3_1.7b` | formal | 0.6326 | 68.16 | 74.44 | 56.89 | 73.49 | gate_mean: 0.0000, gate_abs_mean: 0.0000, gate_pos_frac: 0.0000, gate_neg_frac: 0.0000, gate_max: 0.0000, gate_min: 0.0000, injected_mean: 1.1219, cal_bias: -0.0002, post_prob_invalid_frac: 0.0000, post_prob_label_invalid_frac: 0.0000, post_prob_label_out_of_range_frac: 0.0000, post_prob_label_raw_max: 0.4736, post_prob_label_raw_min: 0.4736, post_prob_label_safe_max: 0.4736, post_prob_label_safe_min: 0.4736, post_prob_label_sanitized_frac: 0.0000, post_prob_out_of_range_frac: 0.0000, post_prob_raw_max: 0.3952, post_prob_raw_min: 0.3952, post_prob_safe_max: 0.3952, post_prob_safe_min: 0.3952, post_prob_sanitized_frac: 0.0000, pre_prob_invalid_frac: 0.0000, pre_prob_out_of_range_frac: 0.0000, pre_prob_raw_max: 0.3953, pre_prob_raw_min: 0.3953, pre_prob_safe_max: 0.3953, pre_prob_safe_min: 0.3953, pre_prob_sanitized_frac: 0.0000 |
| `cel_task_conditioned_lastlayer_v25_joint_bias_calibratorlr10x_qwen3_1.7b` | formal | 0.6363 | 66.90 | 74.98 | 52.04 | 74.27 | gate_mean: 0.0000, gate_abs_mean: 0.0000, gate_pos_frac: 0.0000, gate_neg_frac: 0.0000, gate_max: 0.0000, gate_min: 0.0000, injected_mean: 1.1560, cal_bias: -0.0011, post_prob_invalid_frac: 0.0000, post_prob_label_invalid_frac: 0.0000, post_prob_label_out_of_range_frac: 0.0000, post_prob_label_raw_max: 0.4736, post_prob_label_raw_min: 0.4736, post_prob_label_safe_max: 0.4736, post_prob_label_safe_min: 0.4736, post_prob_label_sanitized_frac: 0.0000, post_prob_out_of_range_frac: 0.0000, post_prob_raw_max: 0.3700, post_prob_raw_min: 0.3700, post_prob_safe_max: 0.3700, post_prob_safe_min: 0.3700, post_prob_sanitized_frac: 0.0000, pre_prob_invalid_frac: 0.0000, pre_prob_out_of_range_frac: 0.0000, pre_prob_raw_max: 0.3701, pre_prob_raw_min: 0.3701, pre_prob_safe_max: 0.3701, pre_prob_safe_min: 0.3701, pre_prob_sanitized_frac: 0.0000 |
| `cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b` | formal | 0.6657 | 70.03 | 75.38 | 66.80 | 77.26 | gate_mean: 0.0000, gate_abs_mean: 0.0000, gate_pos_frac: 0.0000, gate_neg_frac: 0.0000, gate_max: 0.0000, gate_min: 0.0000, injected_mean: 0.2813, cal_bias: 0.3220, post_prob_invalid_frac: 0.0000, post_prob_label_invalid_frac: 0.0000, post_prob_label_out_of_range_frac: 0.0000, post_prob_label_raw_max: 0.4736, post_prob_label_raw_min: 0.4736, post_prob_label_safe_max: 0.4736, post_prob_label_safe_min: 0.4736, post_prob_label_sanitized_frac: 0.0000, post_prob_out_of_range_frac: 0.0000, post_prob_raw_max: 0.4687, post_prob_raw_min: 0.4687, post_prob_safe_max: 0.4687, post_prob_safe_min: 0.4687, post_prob_sanitized_frac: 0.0000, pre_prob_invalid_frac: 0.0000, pre_prob_out_of_range_frac: 0.0000, pre_prob_raw_max: 0.4277, pre_prob_raw_min: 0.4277, pre_prob_safe_max: 0.4277, pre_prob_safe_min: 0.4277, pre_prob_sanitized_frac: 0.0000 |

## Best So Far
- A-module reference (historical checkpoint): `cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b` with Overall Acc `70.03` and Overall AUC `75.38`, delta Acc `+0.81`, delta AUC `+0.43`
- Diagnostic-only best: `cel_task_conditioned_lastlayer_v15_fullv1_cal_bias_only_qwen3_1.7b` with Overall Acc `69.27` and Overall AUC `75.29`

## Key Files
- Formal loop: `results/cel_stage1_last_layer/FORMAL_EXPERIMENT_LOOP.md`
- Formal method protocol: `results/cel_stage1_last_layer/FORMAL_METHOD_PROTOCOL.md`
- Baseline record: `results/baseline/Baseline_DialogueKT_实验记录.md`
- Stage1 last-layer record: `results/cel_stage1_last_layer/CEL_Stage1_LastLayer_DialogueKT_实验记录.md`
- Strict full-train report: `results/cel_stage1_last_layer/STRICT_FULL_TRAIN_REPORT.md`
- Formal experiment audit: `results/cel_stage1_last_layer/FORMAL_EXPERIMENT_AUDIT.md`
- Formal candidate briefs: `results/cel_stage1_last_layer/FORMAL_CANDIDATE_BRIEFS.md`
- Formal candidate config diffs: `results/cel_stage1_last_layer/FORMAL_CANDIDATE_CONFIG_DIFFS.md`
- Strict full-train runbook: `results/cel_stage1_last_layer/STRICT_FULL_TRAIN_RUNBOOK.md`
- Formal next-action helper: `scripts/cel_stage1_last_layer/print_current_formal_next_action.py`
- Formal start/monitor/finalize/review scripts: `scripts/cel_stage1_last_layer/`
- Comparison report: `results/cel_stage1_last_layer/COMPARISON.md`
- Retained candidate stdout, refresh log, status JSON, and audit files: `results/cel_stage1_last_layer/`
- Historical controller/watcher process logs, pid/lock files, and temporary scaffolds were removed after Stage 1 closeout
- Step logs: `results/cel_stage1_last_layer/step_logs`
