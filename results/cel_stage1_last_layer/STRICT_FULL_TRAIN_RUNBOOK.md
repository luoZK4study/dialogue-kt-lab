# Strict Full-Train Runbook

> Terminology note: historical candidate keys, checkpoint names, and command arguments remain for auditability. The recorded successful method is called the **A module**.

## Purpose

- This runbook only governs **strict formal** `task_conditioned` candidates.
- A result is formal only if it comes from full `train + val + test` on the SSH server.
- Any calibrator-only, frozen retraining, fixed-bias eval, validation-fit bias, or other post-hoc checkpoint adjustment remains diagnostic only.
- The higher-level decision policy now lives in `results/cel_stage1_last_layer/FORMAL_EXPERIMENT_LOOP.md`; this runbook focuses on execution commands and readiness checks.

## Current Formal Goal

- Baseline to beat: `lmkt_qwen3_1.7b_recert_20260620`
- Baseline Overall Acc / AUC: **69.22 / 74.95**
- Historical selector/ranking comparison anchor: `cel_task_conditioned_lastlayer_v1_qwen3_1.7b`
- Comparison-anchor Overall Acc / AUC: **68.87 / 75.29**
- Formal success condition:
  - `Overall Acc > 69.22`
  - `Overall AUC > 74.95`

## Completed Formal Candidate Queue

1. `cel_task_conditioned_lastlayer_v21_joint_bias_fulltrain_qwen3_1.7b`
   - start point: baseline checkpoint
   - method: joint bias calibrator full training
2. `cel_task_conditioned_lastlayer_v22_fullv1_joint_bias_tinylr_qwen3_1.7b`
   - start point: full `v1` checkpoint
   - method: tiny-lr joint full fine-tune
3. `cel_task_conditioned_lastlayer_v23_joint_bias_posinit_qwen3_1.7b`
   - start point: baseline checkpoint
   - method: keep the `v21` joint-bias full-train setup but initialize the calibrator with a positive bias
4. `cel_task_conditioned_lastlayer_v24_selectorinit_joint_bias_fulltrain_qwen3_1.7b`
   - start point: baseline checkpoint
   - method: keep the `v21` joint-bias full-train setup but warm-start only the selector from historical selector/ranking anchor `v1`
5. `cel_task_conditioned_lastlayer_v25_joint_bias_calibratorlr10x_qwen3_1.7b`
   - start point: baseline checkpoint
   - method: keep the `v21` joint-bias full-train setup but assign the calibrator a higher dedicated learning rate
6. `cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b`
   - start point: Qwen3-1.7B base with newly initialized A-module LoRA, selector, and calibrator
   - method: self-contained A-module bootstrap -> calibrator warmup -> strict tiny-lr joint chain; no baseline, v1, or withdrawn historical checkpoint is loaded

## Current Priority

Current priority (2026-08-12):

1. `v21`, `v22`, `v23`, `v24`, and `v25` are already recorded full-train non-winners at Overall Acc / AUC **68.61 / 75.26**, **68.26 / 74.93**, **68.41 / 74.29**, **68.16 / 74.44**, and **66.90 / 74.98**.
2. Old `v26_fullv1` was withdrawn and deleted; it is not a recorded result.
3. The A-module historical checkpoint completed strict full-train, final test, and provenance audit at Overall Acc/AUC **70.03 / 75.38** and Final Acc/AUC **66.80 / 77.26**.
4. Do not rerun baseline or v1; use them only as comparison references.
5. Stage 2 first-round single-path runs are historical pilots. Implement and verify the explicit `h_r/h_m` dual-path method before registering another formal candidate.

## Authoritative Files

- Live status: `results/cel_stage1_last_layer/STATUS.md`
- Formal loop policy: `results/cel_stage1_last_layer/FORMAL_EXPERIMENT_LOOP.md`
- Formal-only scoreboard: `results/cel_stage1_last_layer/STRICT_FULL_TRAIN_REPORT.md`
- Formal experiment audit: `results/cel_stage1_last_layer/FORMAL_EXPERIMENT_AUDIT.md`
- Formal candidate briefs: `results/cel_stage1_last_layer/FORMAL_CANDIDATE_BRIEFS.md`
- Formal candidate config diffs: `results/cel_stage1_last_layer/FORMAL_CANDIDATE_CONFIG_DIFFS.md`
- Formal candidate scaffold: `scripts/cel_stage1_last_layer/scaffold_formal_candidate.py`
- Stage 1 record: `results/cel_stage1_last_layer/CEL_Stage1_LastLayer_DialogueKT_实验记录.md`
- Tuning loop: `results/cel_stage1_last_layer/TASK_CONDITIONED_TUNING_LOOP.md`
- Local task state: `results/cel_stage1_last_layer/task_conditioned_status.json`
- Refresh log: `results/cel_stage1_last_layer/task_conditioned_refresh.log`
- Historical watcher log: removed after Stage 1 closeout; current authoritative state is `STATUS.md` plus `task_conditioned_status.json` and `task_conditioned_refresh.log`.

## Standard Loop

Quick current-state helper:

```bash
python3 scripts/cel_stage1_last_layer/print_current_formal_next_action.py
bash scripts/cel_stage1_last_layer/show_current_formal_runtime_health.sh 3090
bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090 --dry-run
bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090
```

Background-controller visibility:

- 若需要确认“后台 controller 现在到底在执行什么”，优先看 `STATUS.md` 里的 `Task-conditioned controller pid/alive`、`Task-conditioned controller command`、`Task-conditioned controller child command`。
- 需要更完整的当前快照时，直接运行 `bash scripts/cel_stage1_last_layer/show_current_formal_runtime_health.sh 3090`。

The alias-based current-state runner resolves the authoritative local strict-formal decision first, then dispatches the matching SSH-side action in one step:

- `monitor_active_full_train` -> `monitor_formal_candidate_via_ssh_alias.sh 3090 once`
- `launch_next_candidate` -> `start_current_formal_candidate_via_ssh_alias.sh 3090`
- `review_recorded_or_completed_formal_candidate` / `winner_or_done` -> finalize + review via the current-state alias wrappers
- `freeze_recorded_then_rerun` -> snapshot-finalize the recorded non-winner first, then launch the current rerun target

1. Confirm only strict full-train candidates are in scope.
2. Read the candidate's design brief and confirm the single main variable / hypothesis you are about to test.
3. Confirm the candidate's CLI-level config diff still matches its declared one-variable story.
4. Start one formal candidate through the unified start entrypoint.
5. Monitor it through the unified monitor entrypoint.
6. Finalize the landed result through the unified closeout entrypoint.
7. Refresh and review the landed result through the unified review entrypoint.
8. Check `FORMAL_EXPERIMENT_AUDIT.md` to confirm artifact completeness and markdown coverage for any completed full-train candidate.
9. Only judge winner status after full metrics are present and the audit is clean.
10. If no winner exists, record the failure mode or metric gap before changing method design.

When you do need a brand-new formal candidate after the current queue is cleared:

1. Scaffold it locally through `scripts/cel_stage1_last_layer/scaffold_formal_candidate.py`.
2. Declare the exact `config_diff_keys` up front.
3. Keep the generated run script's real diff keys aligned with that declaration.
4. Only then register it into the formal candidate registry and rerun preflight.

Queue guard:

- `scaffold_formal_candidate.py` now refuses to create a new formal candidate while any current formal candidate is still `needs_rerun`.
- Use `--allow-pending-rerun` only for offline draft scaffolding; it does not authorize changing the real formal queue before the rerun obligation is cleared.

## Launch

Current state note:

- The active formal loop is **not** closed: independent `v26_selftrained` is currently running on SSH.
- Do not launch another candidate, rerun baseline, or rerun `v1`; the commands below are only for monitoring the active v26 and, after it completes, running its closeout.

Preferred strict full-train local operator flow:

```bash
bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090 --dry-run
bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090
bash scripts/cel_stage1_last_layer/print_recommended_wsl_ssh_alias.sh
bash scripts/cel_stage1_last_layer/start_current_formal_candidate_via_ssh_alias.sh 3090
bash scripts/cel_stage1_last_layer/start_formal_candidate_via_ssh_alias.sh 3090 <candidate_key>
bash scripts/cel_stage1_last_layer/print_current_formal_manual_launch_steps.sh
```

Recommended local WSL alias bootstrap:

```bash
bash scripts/cel_stage1_last_layer/print_recommended_wsl_ssh_alias.sh
bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090 --dry-run
bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090
bash scripts/cel_stage1_last_layer/review_current_formal_candidate_via_ssh_alias.sh 3090
bash scripts/cel_stage1_last_layer/review_formal_candidate_via_ssh_alias.sh 3090 v26
bash scripts/cel_stage1_last_layer/print_current_formal_manual_launch_steps.sh
```

When the local state already points at a running SSH-side formal candidate, the current-state alias runner is preferred over manually guessing whether you should launch, monitor, or close out.

Equivalent direct-start wrapper, if you do not want the alias helper:

```bash
bash scripts/cel_stage1_last_layer/start_formal_candidate.sh <candidate_key>
```

The repository now treats the alias path as the preferred local operator flow:

- `check_ssh_formal_config.sh` validates `PreferredAuthentications publickey` and `IdentitiesOnly yes` behavior for alias-based launches.
- `preflight_strict_full_train.sh` now validates the direct/default SSH path first, then also validates alias mode automatically when alias `3090` resolves in the current shell.
- `start_current_formal_candidate_via_ssh_alias.sh` resolves the current launchable strict candidate from authoritative local state before delegating to the candidate-level alias wrapper.
- Automated SSH / rsync commands add `BatchMode=yes` and `ConnectTimeout=15` so a broken key setup fails fast instead of hanging on password prompts.

Launch guard:

- `start_formal_candidate.sh` and `launch_round3_candidate_on_server.sh` now both enforce the current strict queue through `formal_queue_guard.py`.
- A candidate can only launch if it is itself the current rerun-first target, or if no formal candidate remains in `needs_rerun`.
- Under the current queue, use `print_current_formal_next_action.py` as the authority; when the local state is already `winner_or_done`, no extra launch action is allowed.

Before launch, use `results/cel_stage1_last_layer/FORMAL_CANDIDATE_BRIEFS.md` to confirm:

- the single main variable
- the hypothesis for why it should improve both Acc and AUC
- the expected signs to watch after the SSH run lands
- the one-variable interpretation rule if the candidate still misses the baseline gate

Also use `results/cel_stage1_last_layer/FORMAL_CANDIDATE_CONFIG_DIFFS.md` to confirm:

- the candidate's reference script
- the exact CLI keys that are allowed to differ
- that the current config diff check still reports `ok`

The strict start wrapper now prints this candidate context directly before preflight and SSH launch, including:

- current formal queue summary
- the candidate's single main variable
- the hypothesis for why it should improve both Acc and AUC
- the expected signals to watch after the SSH run lands
- the candidate-specific launch-readiness gate for the current strict full-train scope

Equivalent lower-level sequence when you need to debug the launch step-by-step:

```bash
bash scripts/cel_stage1_last_layer/preflight_strict_full_train.sh
bash scripts/cel_stage1_last_layer/launch_round3_candidate_on_server.sh <candidate_key>
```

The lower-level launch wrapper now also prints the same formal candidate context as `start_formal_candidate.sh`, so direct launch debugging and controller-triggered launch logs still expose:

- the current formal queue summary
- the candidate's single main variable
- the hypothesis for why it should improve both Acc and AUC
- the expected post-run signals to watch

These launch wrappers sync current local code to SSH before launch by default and avoid silently relaunching over an already-synced completed run unless `TASK_CONDITIONED_FORCE_RERUN=1` is set.

Optional local controller:

```bash
TASK_CONDITIONED_CONTROLLER_SLEEP_SECS=600 bash scripts/cel_stage1_last_layer/run_task_conditioned_controller.sh
TASK_CONDITIONED_CONTROLLER_MAX_CYCLES=1 bash scripts/cel_stage1_last_layer/run_task_conditioned_controller_via_ssh_alias.sh 3090
bash scripts/cel_stage1_last_layer/start_task_conditioned_controller_background.sh 3090
```

Controller rule:

- It reads the authoritative local `current_formal_decision` first, then dispatches the matching strict action.
- It only auto-launches through the guarded strict launch path.
- When a full-train run is already active, it only refreshes/monitors and follows the generated `Suggested next monitor after` or `Recommended poll interval`.
- When a candidate has landed, it automatically executes current-state `finalize + review` before the loop advances.
- For a long-lived local loop, prefer the background entrypoint so the controller is not tied to one foreground shell.
- It must not auto-launch legacy diagnostic rounds.
- Its sleep cadence now follows the generated `STATUS.md` recommended poll interval when present; `TASK_CONDITIONED_CONTROLLER_SLEEP_SECS` is only the fallback.
- Manual monitor, controller refresh, finalize, and review now share one refresh lock, so adjacent requests serialize instead of starting duplicate sync/rebuild cycles in parallel.
- The controller now reads the local authoritative state before refreshing; if the current state is still `fresh_remote` and the next suggested monitor is not due, it sleeps through that window instead of issuing an early refresh.
- `v26` 是三个独立进程串行执行的链：bootstrap `--epochs 2`，随后 calibrator warmup `--epochs 1`，最后 strict joint `--epochs 1`。每个阶段的 epoch 计数都会重新从 1 开始；bootstrap -> warmup -> joint 的切换是预期流程，不是服务器重启或 launcher 漂移，且只有 joint 执行正式 test。

## Sync And Refresh

Preferred one-shot monitor/refresh entrypoint:

```bash
bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090 --dry-run
bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090
bash scripts/cel_stage1_last_layer/monitor_formal_candidate.sh once
bash scripts/cel_stage1_last_layer/monitor_formal_candidate_via_ssh_alias.sh 3090 once
```

Under the current authoritative state, the current-state alias runner resolves to one monitor cycle because `v26` is already in progress on SSH.

`once` should complete one refresh/rebuild cycle and then print the same formal snapshot summary used by the local-snapshot path:

- current stage / next action / sync freshness
- local or fallback poll-interval guidance
- formal audit summary
- rerun queue and recorded non-winner queue
- current strict decision summary

Preferred watcher-style monitor entrypoint:

```bash
TASK_CONDITIONED_MONITOR_MAX_CYCLES=1 bash scripts/cel_stage1_last_layer/monitor_formal_candidate.sh watch
TASK_CONDITIONED_MONITOR_MAX_CYCLES=1 bash scripts/cel_stage1_last_layer/monitor_formal_candidate_via_ssh_alias.sh 3090 watch
```

Preferred post-run closeout entrypoint:

```bash
bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090 --dry-run
bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090
bash scripts/cel_stage1_last_layer/finalize_current_formal_candidate.sh
bash scripts/cel_stage1_last_layer/finalize_current_formal_candidate_via_ssh_alias.sh 3090
bash scripts/cel_stage1_last_layer/review_current_formal_candidate.sh
bash scripts/cel_stage1_last_layer/review_current_formal_candidate_via_ssh_alias.sh 3090
bash scripts/cel_stage1_last_layer/finalize_formal_candidate.sh v26
bash scripts/cel_stage1_last_layer/finalize_formal_candidate_via_ssh_alias.sh 3090 v26
bash scripts/cel_stage1_last_layer/review_formal_candidate_via_ssh_alias.sh 3090 v26
```

Closeout gate:

- `finalize_formal_candidate.sh` is not just a summary helper; it is the strict closeout gate.
- By default it now fails when the candidate is still `pending`, or when synced artifacts / markdown coverage are incomplete.
- The normal success states are:
  - `recorded`: completed full-train artifacts plus required markdown coverage are all present
  - `needs_rerun`: failed strict full-train candidate has already been reflected across the required surfaces and is ready to remain in the rerun queue
- Use `TASK_CONDITIONED_FINALIZE_ALLOW_INCOMPLETE=1` only when you intentionally need to inspect an incomplete local snapshot without treating it as a completed closeout.

Equivalent lower-level refresh chain:

```bash
TASK_CONDITIONED_REFRESH_SYNC_TIMEOUT_SECS=240 bash scripts/cel_stage1_last_layer/refresh_task_conditioned_round3_once.sh
bash scripts/cel_stage1_last_layer/rebuild_task_conditioned_reports.sh
```

Refresh order invariant:

1. Sync remote artifacts first.
2. Rebuild `task_conditioned_status.json`.
3. Regenerate markdown outputs.
4. Keep the full refresh chain sequential.
5. `STRICT_FULL_TRAIN_REPORT.md` and `CEL_Stage1_LastLayer_DialogueKT_实验记录.md` now read stage/freshness directly from `task_conditioned_status.json` and `task_conditioned_refresh.log`, but the full chain should still stay ordered so all surfaces share one snapshot.
6. `FORMAL_EXPERIMENT_AUDIT.md` must be regenerated in the same chain; use it to verify that every completed formal SSH run has artifacts plus markdown coverage before any winner claim.
7. `FORMAL_CANDIDATE_BRIEFS.md` must be refreshed in the same chain so each completed candidate's recorded metrics and next-step interpretation stay aligned with the latest strict full-train snapshot.
8. `FORMAL_CANDIDATE_CONFIG_DIFFS.md` must be refreshed in the same chain so future candidate creation and current candidate interpretation continue to share the same one-variable config contract.
9. `finalize_formal_candidate.sh` is the preferred wrapper for this chain once a run lands, because it prints the candidate-specific closeout state plus the recorded post-run analysis fields from the rebuilt authoritative surfaces.

## Monitoring Cadence

- Shared adaptive polling policy:
  - `900s` for early long full-train phases
  - `600s` for steady mid-run training
  - `300s` for `Validation / Testing`, `>=95%` progress, or short remaining ETA
  - `120s` for very short remaining ETA such as `<=5m`

Recommended commands:

```bash
TASK_CONDITIONED_CONTROLLER_SLEEP_SECS=600 bash scripts/cel_stage1_last_layer/run_task_conditioned_controller.sh
TASK_CONDITIONED_MONITOR_SLEEP_SECS=600 TASK_CONDITIONED_MONITOR_FAST_SLEEP_SECS=300 bash scripts/cel_stage1_last_layer/monitor_formal_candidate.sh watch
TASK_CONDITIONED_MONITOR_MAX_CYCLES=1 TASK_CONDITIONED_MONITOR_SYNC_TIMEOUT_SECS=240 bash scripts/cel_stage1_last_layer/monitor_formal_candidate.sh watch
```

Live recommendation source of truth:

- `STATUS.md`
- `STRICT_FULL_TRAIN_REPORT.md`

## Sync Freshness Rules

- `Latest authoritative task-conditioned refresh event` is the only authoritative remote-sync signal.
- `Latest task-conditioned watcher event` is only a local monitoring event and may reflect a rebuild from cached artifacts.
- If the latest authoritative refresh summary contains `sync=ok`, the local reports reflect remote state at that event's timestamp; a copied snapshot may no longer be current.
- If the latest authoritative refresh summary contains `sync=failed` or `sync=timeout`, the local reports are rebuilt from the last synced snapshot only.
- Do not treat progress movement under `sync!=ok` as new remote truth.

## Result Readiness Checklist

Before formal analysis, confirm all of the following for a candidate:

1. `metrics_<model>.txt` exists
2. `qual_<model>.csv` exists
3. stdout log exists and is synced
4. the run completed full `train + val + test`
5. metrics include all four values: Overall Acc/AUC and Final Acc/AUC
6. Overall and Final deltas versus baseline are present in the generated reports
7. the result is visible in `STRICT_FULL_TRAIN_REPORT.md`
8. the result is visible in `FORMAL_CANDIDATE_BRIEFS.md`
9. the candidate's config diff contract is still documented in `FORMAL_CANDIDATE_CONFIG_DIFFS.md`
10. the result is marked clean in `FORMAL_EXPERIMENT_AUDIT.md`

If any item is missing, the candidate is not yet ready for formal winner judgment.

## Winner Judgment

Declare a new formal winner only when:

1. The candidate is ready under the checklist above.
2. The candidate beat baseline on both:
   - Overall Acc
   - Overall AUC
   Final Acc/AUC and their deltas must also be reported and audited, but they are not an additional winner gate.
3. The result is recorded in:
   - `STRICT_FULL_TRAIN_REPORT.md`
   - `FORMAL_EXPERIMENT_AUDIT.md`
   - `FORMAL_CANDIDATE_BRIEFS.md`
   - `FORMAL_CANDIDATE_CONFIG_DIFFS.md`
   - `CEL_Stage1_LastLayer_DialogueKT_实验记录.md`
   - `TASK_CONDITIONED_TUNING_LOOP.md`
   - `COMPARISON.md`
   - `DETAILED_ANALYSIS.md`

## Failure Handling

If a full-train candidate fails:

1. Sync the latest stdout log immediately.
2. Identify the actual failure type:
   - oom
   - traceback
   - stalled or incomplete run
3. Record the failure in the Stage 1 record and tuning loop.
4. Rerun only after the failure mode is understood and the same full-train scope is preserved.

Current `v21` note:

- Latest synced `v21` failure hit a CUDA-side BCE range assert during epoch 1 training near `5376/8425 (64%)`.
- The current local fix is in `dialogue_kt/training.py`: sanitize probability / label tensors and compute the probability BCE path via `BCEWithLogitsLoss(torch.logit(safe_probs), safe_labels)`.
- Before rerunning `v21`, sync the updated code to SSH:

```bash
bash scripts/cel_stage1_last_layer/sync_code_to_server.sh
bash scripts/cel_stage1_last_layer/sync_code_to_server_via_ssh_alias.sh 3090
```

- Local preflight now also checks that `dialogue_kt/training.py` still uses the sanitized probability path with `BCEWithLogitsLoss(torch.logit(safe_probs), safe_labels)` instead of direct probability-space `BCELoss`.
- Local preflight now also checks that `sync_code_to_server.sh` still covers the strict formal-loop sync surface, including report generators, registry checks, and every registry-declared formal run script.

- Or use the local strict-start wrapper directly, which performs preflight plus that sync automatically before launch:

```bash
bash scripts/cel_stage1_last_layer/print_recommended_wsl_ssh_alias.sh
bash scripts/cel_stage1_last_layer/start_current_formal_candidate_via_ssh_alias.sh 3090
bash scripts/cel_stage1_last_layer/start_formal_candidate_via_ssh_alias.sh 3090 v21
bash scripts/cel_stage1_last_layer/start_formal_candidate.sh v21
```

- That strict-start wrapper now also prints a candidate-specific launch-readiness summary before launch, including whether the queue allows this candidate, whether the candidate is currently the rerun target or otherwise launchable under the queue, and whether the local BCE-safety / sync-manifest guards are currently passing.

- Then rerun the same strict full-train scope on SSH; do not convert it into calibrator-only, checkpoint-only, or any other post-hoc diagnostic flow.

## Post-Run Analysis

After a candidate finishes and passes readiness checks:

Preferred formal review entrypoint:

```bash
bash scripts/cel_stage1_last_layer/finalize_current_formal_candidate.sh
bash scripts/cel_stage1_last_layer/review_current_formal_candidate.sh
bash scripts/cel_stage1_last_layer/finalize_formal_candidate.sh v26
bash scripts/cel_stage1_last_layer/review_formal_candidate.sh v26
bash scripts/cel_stage1_last_layer/finalize_formal_candidate_via_ssh_alias.sh 3090 v26
bash scripts/cel_stage1_last_layer/review_formal_candidate_via_ssh_alias.sh 3090 v26
```

If the current environment cannot SSH, read the current local snapshot only:

```bash
bash scripts/cel_stage1_last_layer/monitor_formal_candidate.sh skip-sync
bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090 --dry-run
```

If you specifically need an offline snapshot review example for an already recorded non-winner, `v24` remains the safe example because it has already completed strict closeout:

```bash
TASK_CONDITIONED_FINALIZE_SKIP_REFRESH=1 bash scripts/cel_stage1_last_layer/finalize_formal_candidate.sh v24
TASK_CONDITIONED_REVIEW_SKIP_REFRESH=1 bash scripts/cel_stage1_last_layer/review_formal_candidate.sh v24
TASK_CONDITIONED_FINALIZE_SKIP_REFRESH=1 bash scripts/cel_stage1_last_layer/finalize_formal_candidate_via_ssh_alias.sh 3090 v24
TASK_CONDITIONED_REVIEW_SKIP_REFRESH=1 bash scripts/cel_stage1_last_layer/review_formal_candidate_via_ssh_alias.sh 3090 v24
```

If the current environment cannot open SSH itself but you still have another terminal that can reach the server, print the strict manual fallback launch sequence:

```bash
bash scripts/cel_stage1_last_layer/print_manual_formal_launch_steps.sh v26
bash scripts/cel_stage1_last_layer/print_current_formal_manual_launch_steps.sh
```

That fallback preserves the same strict scope:

- local preflight
- local code sync
- SSH login
- remote duplicate-process / existing-metrics guard
- remote log / metrics backup before intentional rerun
- remote `nohup` launch of the registered formal run script
- local monitor / finalize / review closeout

For monitor-only local snapshots, `bash scripts/cel_stage1_last_layer/monitor_formal_candidate.sh skip-sync` should expose:

- current stage / next action / sync freshness
- local or fallback poll-interval guidance
- formal audit summary
- current rerun queue and recorded non-winner queue
- the current strict decision summary from `STRICT_FULL_TRAIN_REPORT.md`

1. Compare Overall Acc / AUC vs baseline; this is the only formal winner gate.
2. Compare Final Acc / AUC vs baseline and report both Final deltas; these are mandatory audit metrics, but do not replace the Overall winner gate.
3. Compare Overall Acc / AUC vs `v1` anchor.
4. Record:
   - model name
   - Overall Acc / AUC
   - Final Acc / AUC
   - `Pred True`
   - best-threshold diagnostic (`best_acc`, `best_threshold`)
   - Overall Acc / AUC and Final Acc / AUC deltas vs baseline
   - delta vs `v1` anchor
   - whether both metrics beat baseline
5. If not a winner, write the single main reason for the next design change.
6. If the next candidate will be new, scaffold it locally first and keep the real CLI diff keys aligned with the declared one-variable plan.
7. Verify `FORMAL_EXPERIMENT_AUDIT.md` shows the completed candidate as fully recorded before ending the loop for that run.

## Constraints

- All formal experiments run on SSH only.
- All completed formal experiments must be recorded and analyzed in markdown.
- Stage 2 remains locked until the active self-trained `v26` finishes strict full training, final testing, provenance audit, and closeout.
