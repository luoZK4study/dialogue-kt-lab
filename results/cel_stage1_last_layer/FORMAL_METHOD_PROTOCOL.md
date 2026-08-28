# Formal Method Protocol

> Terminology note: this historical protocol retains candidate identifiers for auditability. The audited current method is called the **A module** in method documents; candidate numbers are not method names.

- Generated at: `2026-08-15T15:37:02`
- Scope: `results/cel_stage1_last_layer/` 当前 strict formal `task_conditioned` loop。
- 目的：把“什么算正式方法、当前该跑什么、如何监控、如何落盘、每轮实验后必须记录什么”集中到一个摘要入口。

## Core Rules

- 只有 SSH 服务器上完整 `train + val + test` 的结果才算正式方法结果。
- 任何 calibrator-only、frozen retraining、fixed-bias eval、validation-fit bias、以及从已训练 ckpt 出发的 post-hoc 调整都只算诊断证据。
- formal winner 只能在 `Overall Acc > baseline` 且 `Overall AUC > baseline` 时成立，而且必须先通过 `finalize -> audit -> review`。
- 若 formal queue 中仍存在 `needs_rerun` 候选，禁止跳过去直接设计新方法。
- 每轮新的 formal candidate 只允许改一个主变量，并且必须能解释为什么有希望同时改善 Acc 与 AUC。

## Method Admissibility Checklist

在把某个新方案登记为正式方法前，先逐条确认：

1. 该候选会在 SSH 上独立完成一轮新的 `train + val + test`，而不是只基于已有 ckpt 做补训、后处理或校准。
2. 如果使用 warm-start，它必须被明确写进候选方法定义本身，并且该候选仍需完成自己的 SSH-side full-train；单纯复用旧 ckpt 的后续结果不计入正式方法。
3. 候选只改一个主变量，其余训练口径、比较基线和结果解读规则保持稳定。
4. 候选在启动前已经写清：模型名、起点、唯一主改动、为什么有机会同时改善 Acc 与 AUC、预期观察信号、以及失败后下一轮只改一个变量的解释。
5. 候选脚本、formal registry、`FORMAL_CANDIDATE_BRIEFS.md`、`FORMAL_CANDIDATE_CONFIG_DIFFS.md`、`preflight_strict_full_train.sh` 必须一致；若 config diff 未声明就发生漂移，该候选不应启动。

## Current Gate

- Baseline gate: `Overall Acc > 69.22` and `Overall AUC > 74.95`.
- Historical selector/ranking comparison anchor: `cel_task_conditioned_lastlayer_v1_qwen3_1.7b` with Overall Acc / AUC **68.87 / 75.29**.
- Current next action: `done`
- Winner found: `True`
- Current formal decision: `winner_or_done`
- Current formal rerun target: `none`
- Current formal recorded non-winner: `v25`
- Current formal recorded non-winner queue: `v21, v22, v23, v24, v25`
- Task-conditioned sync freshness: `fresh_remote`
- Latest authoritative remote refresh event: `[2026-08-12 23:17:24] task_conditioned refresh complete :: sync=ok sync_exit=0 stage=task_conditioned_tuning_completed_with_winner next_action=done live_phase=unknown round3=done,done,done,done,done,done winner=true duration=87s`
- Latest refresh attempt: `[2026-08-12 23:17:24] task_conditioned refresh complete :: sync=ok sync_exit=0 stage=task_conditioned_tuning_completed_with_winner next_action=done live_phase=unknown round3=done,done,done,done,done,done winner=true duration=87s`
- Formal audit summary: `recorded=6`
- Recommended local monitoring cadence: `not_applicable` (当前没有 active strict full-train 需要监控)

## Current Queue

| Candidate | Start Point | Status | Audit | Current Result | Next Use |
|---|---|---|---|---|---|
| `v21` | baseline checkpoint | done | `recorded` | 68.61/75.26 | design next one-variable candidate only after review |
| `v22` | full `v1` checkpoint | done | `recorded` | 68.26/74.93 | design next one-variable candidate only after review |
| `v23` | baseline checkpoint | done | `recorded` | 68.41/74.29 | design next one-variable candidate only after review |
| `v24` | baseline checkpoint | done | `recorded` | 68.16/74.44 | design next one-variable candidate only after review |
| `v25` | baseline checkpoint | done | `recorded` | 66.90/74.98 | design next one-variable candidate only after review |
| `v26` | Qwen3-1.7B base weights with newly initialized A-module LoRA, selector, and calibrator | done | `recorded` | 70.03/75.38 | freeze winner |

## Current Required Action

- 当前 A 模块参考结果已固定（历史 candidate key：`v26`），formal queue 已进入 done 状态，不再设计新的 Stage 1 candidate。
- 如需复核结论，优先使用 `review_formal_candidate.sh v26`、对应 alias 包装器，或统一 `finalize -> review` 快照入口。
- 当前工作重点是实现 `h_r/h_m` 双路径前向、完整训练目标、tensor contracts 与 audit；在这些工作完成前不注册新的正式 Stage 2 candidate。

## Runtime Source Priority

运行期判断一律按下面顺序取权威来源，不要从 `.claude/` 或手工笔记里读取易过期的 live 数字：

1. `python3 scripts/cel_stage1_last_layer/print_current_formal_next_action.py`
2. `bash scripts/cel_stage1_last_layer/show_current_formal_runtime_health.sh 3090`
3. `results/cel_stage1_last_layer/STATUS.md`
4. `results/cel_stage1_last_layer/STRICT_FULL_TRAIN_REPORT.md`
5. `results/cel_stage1_last_layer/task_conditioned_refresh.log`

补充规则：

- 若最新 refresh 不是 `sync=ok`，只能把本地 surface 当作旧快照重建结果，不能当成新的远端真值。
- 若 `current_formal_decision=monitor_active_full_train`，默认只继续 refresh / monitor，不得并行启动新候选。
- 若当前候选是多 epoch full-train，`epoch 1 validation -> epoch 2 training` 的切换应视为正常流程，不表示 run 重启。

## Recommended WSL SSH Alias

- 推荐先在 WSL 下固定一个本地 alias，例如 `3090`，再通过 alias wrapper 启动 strict formal run。
- 仓库内置模板脚本：`bash scripts/cel_stage1_last_layer/print_recommended_wsl_ssh_alias.sh`。
- 推荐 `~/.ssh/config` 配置块：

```sshconfig
Host 3090
    HostName 119.29.183.125
    Port 22049
    User user4
    PreferredAuthentications publickey
    IdentityFile /home/luo/.ssh/id_rsa_u4
    IdentitiesOnly yes
```

- 推荐权限：`chmod 700 ~/.ssh`、`chmod 600 ~/.ssh/config`、`chmod 600 ~/.ssh/id_rsa_u4`。
- 自动化 SSH / rsync 命令会额外附加 `BatchMode=yes` 与 `ConnectTimeout=15`，避免 formal loop 卡在密码提示或长时间无响应的连接上。

## Formal Loop

1. 先读 `STATUS.md`、`STRICT_FULL_TRAIN_REPORT.md`、`FORMAL_EXPERIMENT_AUDIT.md`，确认当前是 rerun、继续监控，还是已经允许设计下一轮 formal candidate。
2. 若仍有 `needs_rerun`，先保持方法范围不变，修复代码后按同一 strict full-train 口径在 SSH 上重跑。
3. 若 rerun 队列已清空且当前结果仍未双超 baseline，使用 `scaffold_formal_candidate.py` 生成下一轮脚手架，并确保只改一个主变量。
4. 本地只做代码实现、文档更新、`py_compile`、shell 语法检查、`preflight_strict_full_train.sh`；本地训练结果不计入正式结论。
5. 所有正式实验都必须通过 `start_current_formal_candidate_via_ssh_alias.sh`、`start_formal_candidate.sh` 或其 manual SSH fallback 走 SSH full-train 启动。
6. SSH run 落地后，必须先执行 `finalize_formal_candidate.sh`（或对应 alias wrapper），再检查 `FORMAL_EXPERIMENT_AUDIT.md`，最后才允许 `review_formal_candidate.sh`（或对应 alias wrapper）判断 winner / non-winner / rerun。
7. 只有在 audit 干净、结果完整落盘、且双超 baseline 时，才允许把某个候选升级为新的正式方法结论。

## Monitoring Cadence

- 共享自适应轮询规则：训练早期长时 full-train 用 `900s`，稳定训练中期通常 `600s`，`Validation / Testing`、`>=95%` 进度或短剩余 ETA 时通常收紧到 `300s`，而 `<=5m` 的极短剩余 ETA 会进一步收紧到 `120s`。
- 当前 cadence 的解释性来源以 `STATUS.md` 和 `STRICT_FULL_TRAIN_REPORT.md` 里的 recommended poll interval 为准。
- 若最新 authoritative refresh 不是 `sync=ok`，说明本地只是旧快照 rebuild；此时继续看日志可以，但不能把本地状态变化当成新的远端结论。
- 手动 monitor、后台 controller、finalize、review 现在共享同一 refresh 锁；相邻请求会串行化执行，避免同一时间窗口重复 sync/rebuild。
- controller 现在会先读取本地 authoritative surface；若当前仍是 `fresh_remote` 且未到 `Suggested next local monitor after`，它会直接继续休眠，而不是先做一次多余 refresh。
- 若需要在本地长期挂起完整 formal loop，优先使用后台 controller：`bash scripts/cel_stage1_last_layer/start_task_conditioned_controller_background.sh 3090`。
- 若需要确认“后台 controller 现在到底在执行什么”，优先看 `STATUS.md` 里的 `Task-conditioned controller pid/alive`、`Task-conditioned controller command`、`Task-conditioned controller child command`，或直接运行 `bash scripts/cel_stage1_last_layer/show_current_formal_runtime_health.sh 3090`。
- 推荐监控命令：`bash scripts/cel_stage1_last_layer/monitor_formal_candidate.sh once`、`TASK_CONDITIONED_MONITOR_MAX_CYCLES=1 bash scripts/cel_stage1_last_layer/monitor_formal_candidate.sh watch`。
- 若当前在 WSL 里依赖 SSH alias，推荐命令：`bash scripts/cel_stage1_last_layer/monitor_formal_candidate_via_ssh_alias.sh 3090 once`、`TASK_CONDITIONED_MONITOR_MAX_CYCLES=1 bash scripts/cel_stage1_last_layer/monitor_formal_candidate_via_ssh_alias.sh 3090 watch`。

### Cadence Decision Checklist

每次决定要不要加密监控前，先看这四件事：

1. `Live phase` 是 `training`、`validating` 还是 `testing`。
2. `Live epoch / step / ETA` 是否持续推进。
3. 最新 authoritative refresh 是否是 `sync=ok`。
4. 当前报告给出的 `Recommended poll interval` 与 `Suggested next local monitor after`。

执行规则：

- 训练早期且 ETA 仍长时，优先接受较慢 cadence，不必人为缩短。
- 中期稳定训练默认接受 `600s` 左右的 cadence，避免无意义高频 SSH 同步。
- 一旦转入 `Validation / Testing`、`95%+` 进度或 ETA 明显缩短，主动收紧 cadence。
- 若 ETA 已短到约 `<=5m`，再进一步收紧到 `120s`，以便尽快接住 closeout。

## Closeout And Recording

1. 同步远端 `metrics / qual / stdout log`。
2. 重建 `task_conditioned_status.json` 与所有 markdown surfaces。
3. 检查 `FORMAL_EXPERIMENT_AUDIT.md`，确认 artifacts 与 markdown coverage 都完整。
4. 再通过 `review_formal_candidate.sh` 或对应 alias wrapper 读取 authoritative surfaces 做 formal judgment。

每个完成的 formal candidate 必须记录：

- 模型名
- 起点
- 方法主改动
- Overall Acc / AUC
- Final Acc / AUC
- `Pred True`
- Overall Acc / AUC 与 Final Acc / AUC 相对 baseline 的四项 delta
- 是否双超 baseline
- 若失败：失败类型、失败位置或关键报错、以及为什么修复后仍保持 strict full-train 口径
- 若未双超：下一轮为什么只改一个新的主变量

### Per-Candidate Recording Checklist

每个 SSH full-train 候选结束后，至少补齐下面这些字段，缺一项都不算正式闭环完成：

1. `Overall Acc / AUC`
2. `Final Acc / AUC`
3. `Pred True`
4. best-threshold diagnostic
5. Overall Acc / AUC 与 Final Acc / AUC 的四项 delta vs baseline
6. delta vs historical selector/ranking comparison anchor `v1`
7. 当前审计状态：`recorded`、`pending` 或 `needs_rerun`
8. 若失败，写清失败类型、失败位置、修复动作、以及为什么修复后仍保持 strict full-train 口径
9. 若未双超，写清下一轮只改一个变量的理由

必须覆盖的 markdown surfaces：

- `results/cel_stage1_last_layer/STATUS.md`
- `results/cel_stage1_last_layer/STRICT_FULL_TRAIN_REPORT.md`
- `results/cel_stage1_last_layer/FORMAL_EXPERIMENT_AUDIT.md`
- `results/cel_stage1_last_layer/FORMAL_CANDIDATE_BRIEFS.md`
- `results/cel_stage1_last_layer/FORMAL_CANDIDATE_CONFIG_DIFFS.md`
- `results/cel_stage1_last_layer/TASK_CONDITIONED_TUNING_LOOP.md`
- `results/cel_stage1_last_layer/CEL_Stage1_LastLayer_DialogueKT_实验记录.md`
- `results/cel_stage1_last_layer/COMPARISON.md`
- `results/cel_stage1_last_layer/DETAILED_ANALYSIS.md`

## Current Candidate Notes

### `v21` / `cel_task_conditioned_lastlayer_v21_joint_bias_fulltrain_qwen3_1.7b`

- 起点：从 baseline checkpoint 起步
- 方法：把 bias calibrator 作为方法一部分，从训练开始就联合训练全部可训练参数
- 当前状态：`done`
- Audit 状态：`recorded`
- 当前日志进度：epoch 1 testing 约 `1985/1985 (100%)`
- 当前已同步 Overall Acc / AUC：**68.61 / 75.26**；Final Acc / AUC：**56.70 / 74.60**。
- `Pred True`：**32.59%**。
- best-threshold 诊断：best Acc **69.52**，threshold **0.393**。
- 相对 baseline 的 Overall Acc / AUC 变化：**-0.61 / +0.31**。
- 相对 baseline 的 Final Acc / AUC 变化：**-4.47 / -0.72**。
- 当前结论：该 full-train 候选的 artifacts 与 markdown coverage 已完整，但结果未双超 baseline；当前应冻结为 recorded non-winner。
- 当前本地修复：`dialogue_kt/training.py` 已把概率 BCE 路径统一改为先 sanitize，再用 `BCEWithLogitsLoss(torch.logit(safe_probs), safe_labels)` 计算，并补充 label sanitize，避免 joint calibrator 路径再次把非法值直接送入 CUDA `BCELoss`。
- 下一次 `v21` 重跑要求：先同步更新后的 `dialogue_kt/training.py` 到 SSH，再保持原来的 strict full-train 配置完整重跑；不能改成 checkpoint-only / post-hoc 校准。
- 若未双超时的下一轮解释：如果 AUC 仍有竞争力但 Acc 还是不过线，下一轮正式候选只应围绕校准强度、初始化或学习率再改一个变量；如果 AUC 也回落，说明联合 bias 训练已经伤害排序，下一轮应改成一个更受控的训练期主变量，而不是继续叠加后处理补丁。

### `v22` / `cel_task_conditioned_lastlayer_v22_fullv1_joint_bias_tinylr_qwen3_1.7b`

- 起点：从完整 `task_conditioned v1` checkpoint 起步
- 方法：保持全部相关参数可训练，用极小学习率做完整联合微调
- 当前状态：`done`
- Audit 状态：`recorded`
- 当前日志进度：epoch 1 testing 约 `1985/1985 (100%)`
- 当前已同步 Overall Acc / AUC：**68.26 / 74.93**；Final Acc / AUC：**64.27 / 72.86**。
- `Pred True`：**47.76%**。
- best-threshold 诊断：best Acc **69.07**，threshold **0.406**。
- 相对 baseline 的 Overall Acc / AUC 变化：**-0.96 / -0.02**。
- 相对 baseline 的 Final Acc / AUC 变化：**+3.10 / -2.46**。
- 当前结论：该 full-train 候选的 artifacts 与 markdown coverage 已完整，但结果未双超 baseline；当前应冻结为 recorded non-winner。
- 若未双超时的下一轮解释：如果 Acc 和 AUC 一起回落，或 calibrator 几乎没动，说明 tiny-lr warm-start 既没有纠正保守概率，也没有稳住排序；下一轮正式候选应改一个更直接影响训练期校准的主变量，而不是继续重复 warm-start + 后处理式思路。

### `v23` / `cel_task_conditioned_lastlayer_v23_joint_bias_posinit_qwen3_1.7b`

- 起点：从 baseline checkpoint 起步
- 方法：保持 v21 的 joint bias full-train 训练口径，只把 calibrator 初始化改成正偏置起点
- 当前状态：`done`
- Audit 状态：`recorded`
- 当前日志进度：epoch 1 testing 约 `1985/1985 (100%)`
- 当前已同步 Overall Acc / AUC：**68.41 / 74.29**；Final Acc / AUC：**56.12 / 74.59**。
- `Pred True`：**33.50%**。
- best-threshold 诊断：best Acc **68.87**，threshold **0.385**。
- 相对 baseline 的 Overall Acc / AUC 变化：**-0.81 / -0.66**。
- 相对 baseline 的 Final Acc / AUC 变化：**-5.05 / -0.73**。
- 当前结论：该 full-train 候选的 artifacts 与 markdown coverage 已完整，但结果未双超 baseline；当前应冻结为 recorded non-winner。
- 若未双超时的下一轮解释：如果 Pred True 仍然明显偏低且 calibrator bias 又回到接近 0，说明单纯正初始化不足以驱动联训校准；下一轮只应再改一个训练动力学变量，例如 calibrator 学习率或 warmup。若 Pred True 过冲而 AUC 回落，则应减小初始化幅度，而不是叠加更多变量。

### `v24` / `cel_task_conditioned_lastlayer_v24_selectorinit_joint_bias_fulltrain_qwen3_1.7b`

- 起点：从 baseline checkpoint 起步
- 方法：保持 v21 的 joint bias full-train 训练口径，只把 selector 初始化改成来自历史 selector/ranking 锚点 v1
- 实现口径核对：代码口径核对：相对 v21 额外只加载 `cel_selector.pt` 到 `_cel_selector`，不会同时 warm-start calibrator，也不会把 backbone / LoRA 一起从 `v1` 继续加载。
- 当前状态：`done`
- Audit 状态：`recorded`
- 当前日志进度：testing 约 `1985/1985 (100%)`
- 多 epoch 监控说明：该候选脚本配置 `--epochs 2`；epoch 间出现 `validation -> training` 的回切属于正常的多 epoch full-train 流程，不表示 run 重启。
- 当前已同步 Overall Acc / AUC：**68.16 / 74.44**；Final Acc / AUC：**56.89 / 73.49**。
- `Pred True`：**36.22%**。
- best-threshold 诊断：best Acc **68.87**，threshold **0.430**。
- 相对 baseline 的 Overall Acc / AUC 变化：**-1.06 / -0.51**。
- 相对 baseline 的 Final Acc / AUC 变化：**-4.28 / -1.83**。
- 当前结论：该 full-train 候选的 artifacts 与 markdown coverage 已完整，但结果未双超 baseline；当前应冻结为 recorded non-winner。
- 若未双超时的下一轮解释：如果 AUC 仍然保不住，说明 selector warm-start 也不足以稳定 joint full-train；下一轮应只再改一个训练动力学变量，而不是继续叠加更多 warm-start 部件。若 AUC 保住但 Acc 仍不过线，则下一轮只应继续围绕训练期校准强度或学习率改一个变量。

### `v25` / `cel_task_conditioned_lastlayer_v25_joint_bias_calibratorlr10x_qwen3_1.7b`

- 起点：从 baseline checkpoint 起步
- 方法：保持 v21 的 joint bias full-train 训练口径，只给 calibrator 单独更高的学习率
- 当前状态：`done`
- Audit 状态：`recorded`
- 当前日志进度：epoch 1 testing 约 `1985/1985 (100%)`
- 当前已同步 Overall Acc / AUC：**66.90 / 74.98**；Final Acc / AUC：**52.04 / 74.27**。
- `Pred True`：**30.18%**。
- best-threshold 诊断：best Acc **69.87**，threshold **0.392**。
- 相对 baseline 的 Overall Acc / AUC 变化：**-2.32 / +0.03**。
- 相对 baseline 的 Final Acc / AUC 变化：**-9.13 / -1.05**。
- 当前结论：该 full-train 候选的 artifacts 与 markdown coverage 已完整，但结果未双超 baseline；当前应冻结为 recorded non-winner。
- 若未双超时的下一轮解释：如果 calibrator bias 终于动起来但 AUC 明显回落，说明更高 calibrator 学习率正在过度扰动联合训练，下一轮应只减小该学习率或改调度；如果 bias 仍然几乎不动，说明问题不在步幅本身，下一轮应只改另一个训练动力学变量，例如 warmup 或阶段性冻结。

### `v26` / `cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b`

- 起点：从 Qwen3-1.7B 原始基座开始，新建当前 A 模块训练链自己的 LoRA、A 和 calibrator
- 方法：A 模块自包含三阶段链：表征 bootstrap、calibrator-only warmup、strict tiny-LR joint training
- 当前状态：`done`
- Audit 状态：`recorded`
- 当前日志进度：testing 约 `1985/1985 (100%)`
- 多 epoch 监控说明：该候选是独立进程串行的多阶段链：`bootstrap (2 epochs) -> calibrator_warmup (1 epoch) -> joint (1 epoch)`；每个阶段的 epoch 计数会重新从 1 开始。bootstrap -> calibrator warmup -> joint 的阶段切换不是 run 重启；只有 joint 执行正式 test。
- 当前已同步 Overall Acc / AUC：**70.03 / 75.38**；Final Acc / AUC：**66.80 / 77.26**。
- `Pred True`：**43.98%**。
- best-threshold 诊断：best Acc **70.53**，threshold **0.520**。
- 相对 baseline 的 Overall Acc / AUC 变化：**+0.81 / +0.43**。
- 相对 baseline 的 Final Acc / AUC 变化：**+5.63 / +1.94**。
- 当前结论：该 full-train 候选的 artifacts 与 markdown coverage 已完整，且结果已双超 baseline；可以进入 winner 固化流程。
- 若未双超时的下一轮解释：如果来源或参数变化审计失败，无论指标如何都判为无效；若审计通过但指标未过 baseline gate，则先把独立训练的 A 模块 candidate 记录为 non-winner，再只调整一个训练变量。

## Key References

- `results/cel_stage1_last_layer/FORMAL_EXPERIMENT_LOOP.md`
- `results/cel_stage1_last_layer/STRICT_FULL_TRAIN_RUNBOOK.md`
- `results/cel_stage1_last_layer/FORMAL_EXPERIMENT_AUDIT.md`
- `results/cel_stage1_last_layer/STRICT_FULL_TRAIN_REPORT.md`
- `results/cel_stage1_last_layer/FORMAL_CANDIDATE_BRIEFS.md`
- `scripts/cel_stage1_last_layer/finalize_current_formal_candidate.sh`
- `scripts/cel_stage1_last_layer/finalize_current_formal_candidate_via_ssh_alias.sh`
- `scripts/cel_stage1_last_layer/finalize_formal_candidate_via_ssh_alias.sh`
- `scripts/cel_stage1_last_layer/print_current_formal_manual_launch_steps.sh`
- `scripts/cel_stage1_last_layer/review_current_formal_candidate.sh`
- `scripts/cel_stage1_last_layer/review_current_formal_candidate_via_ssh_alias.sh`
- `scripts/cel_stage1_last_layer/review_formal_candidate_via_ssh_alias.sh`
- `scripts/cel_stage1_last_layer/show_current_formal_runtime_health.sh`
- `scripts/cel_stage1_last_layer/start_current_formal_candidate_via_ssh_alias.sh`
- `scripts/cel_stage1_last_layer/start_formal_candidate.sh`
- `scripts/cel_stage1_last_layer/start_task_conditioned_controller_background.sh`
- `scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh`
- `scripts/cel_stage1_last_layer/sync_code_to_server_via_ssh_alias.sh`
- `scripts/cel_stage1_last_layer/print_current_formal_next_action.py`
- `scripts/cel_stage1_last_layer/monitor_formal_candidate.sh`
- `scripts/cel_stage1_last_layer/monitor_formal_candidate_via_ssh_alias.sh`
- `scripts/cel_stage1_last_layer/finalize_formal_candidate.sh`
- `scripts/cel_stage1_last_layer/review_formal_candidate.sh`
- `scripts/cel_stage1_last_layer/sync_results_from_server_via_ssh_alias.sh`
