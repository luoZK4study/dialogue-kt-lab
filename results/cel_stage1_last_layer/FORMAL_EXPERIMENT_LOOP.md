# Formal Experiment Loop

> 术语说明：本文保留历史 candidate、checkpoint 和命令参数用于审计追溯；当前已验证的方法统一称为 **A 模块**，候选编号不再作为方法名称。

## 目标

- 当前只推进 `Stage 1 last-layer` 下的 `task_conditioned` 正式方法。
- 正式成功条件只有一个：
  - `Overall Acc > 69.22`
  - `Overall AUC > 74.95`
- 只有 **SSH 服务器上完整 `train + val + test`** 得到的结果，才允许当作新方法结论。

## 非正式结果一律降级

以下路线只允许作为诊断证据，不能当作正式 winner：

- calibrator-only
- frozen retraining / partial retraining
- fixed-bias eval
- validation-fit bias
- 任何从已训练 ckpt 出发再做的 post-hoc 调整

如果某个结果不是完整 full-train 落地，它最多只能帮助下一轮设计，不允许直接更新正式结论。

## 当前正式状态

当前状态（2026-08-12）：

- baseline：`lmkt_qwen3_1.7b_recert_20260620`
  - Overall Acc / AUC：**69.22 / 74.95**
- 历史 selector/ranking 比较锚点：`cel_task_conditioned_lastlayer_v1_qwen3_1.7b`
  - Overall Acc / AUC：**68.87 / 75.29**
- `v21`：`cel_task_conditioned_lastlayer_v21_joint_bias_fulltrain_qwen3_1.7b`
  - 已完成 full-train
  - Overall Acc / AUC：**68.61 / 75.26**
  - 结论：已记录，但未双超 baseline
- `v22`：`cel_task_conditioned_lastlayer_v22_fullv1_joint_bias_tinylr_qwen3_1.7b`
  - 已完成 full-train
  - Overall Acc / AUC：**68.26 / 74.93**
  - 结论：已记录，但未双超 baseline
- `v23`：`cel_task_conditioned_lastlayer_v23_joint_bias_posinit_qwen3_1.7b`
  - 已完成 full-train
  - Overall Acc / AUC：**68.41 / 74.29**
  - 结论：已记录，但未双超 baseline
- `v24`：`cel_task_conditioned_lastlayer_v24_selectorinit_joint_bias_fulltrain_qwen3_1.7b`
  - 已完成 full-train
  - Overall Acc / AUC：**68.16 / 74.44**
  - 结论：已记录，但未双超 baseline
- `v25`：`cel_task_conditioned_lastlayer_v25_joint_bias_calibratorlr10x_qwen3_1.7b`
  - 已完成 full-train
  - Overall Acc / AUC：**66.90 / 74.98**
  - 结论：已记录，但未双超 baseline
- 旧 `v26_fullv1`：已撤回；其 checkpoint 和结果记录已删除，禁止再作为有效 winner 或初始化来源。
- A 模块参考结果：历史 checkpoint 为 `cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b`
  - 已在 SSH 上完成；起点为 Qwen3-1.7B 原始基座
  - 训练链为 `bootstrap -> calibrator warmup -> strict joint -> final test`
  - Overall Acc / AUC：**70.03 / 75.38**
  - Final Acc / AUC：**66.80 / 77.26**
  - 来源、参数变化、artifact coverage 与 closeout 审计全部通过，当前为正式 winner

当前 recorded non-winner 队列：

- `v21`
- `v22`
- `v23`
- `v24`
- `v25`

当前权威 formal state：

- `current_formal_decision=winner_or_done`
- `current_active_target=none`
- `current_recorded_non_winner=v25`
- `rerun_queue=none`

当前强制优先级：

1. 冻结 A 模块参考结果及其 Overall/Final 四项指标。
2. 保留 `V26_SELFTRAINED_AUDIT`、metrics、qual、kcs 和日志作为来源与结果证据。
3. 不再启动新的 Stage 1 formal candidate；Stage 2 首轮三方向仅作为旧单路径历史 pilot，下一步先实现 `h_r/h_m` 双路径前向、`L_r + L_m + L_cons` 三项核心目标与 tensor contracts。
4. baseline 和 v1 不重跑；anchor 仅用于比较，不是 A 模块训练链的初始化来源。

## 正式实验主循环

### 1. 先判断当前该做什么

推荐先运行统一 helper，直接读取本地权威状态给出的正式下一步：

```bash
python3 scripts/cel_stage1_last_layer/print_current_formal_next_action.py
bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090 --dry-run
bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090
```

其中 `run_current_formal_next_action_via_ssh_alias.sh` 会先解析本地权威状态，再自动分派到当前真正该执行的 SSH 动作：

- 当前有 active full-train 时，执行一次 alias monitor
- 当前需要 launch 时，执行 current-formal alias launch
- 当前 run 已完成待落盘时，执行 finalize + review alias closeout
- 当前既有 recorded non-winner 又有 rerun-first target 时，先 snapshot-finalize，再 launch 当前目标

每次开始前先看：

1. `results/cel_stage1_last_layer/STATUS.md`
2. `results/cel_stage1_last_layer/STRICT_FULL_TRAIN_REPORT.md`
3. `results/cel_stage1_last_layer/FORMAL_EXPERIMENT_AUDIT.md`
4. `results/cel_stage1_last_layer/TASK_CONDITIONED_TUNING_LOOP.md`
5. `results/cel_stage1_last_layer/CEL_Stage1_LastLayer_DialogueKT_实验记录.md`

决策规则：

- 若存在 `needs_rerun` 的 formal candidate，先重跑它，不允许跳过去直接设计新方法。
- 若已完成 formal candidate 全部只是 `recorded` 且仍未双超 baseline，才进入下一轮新方法设计。
- 若 `winner_found=true`，保持所有 markdown surface 处于冻结一致状态，停止新增 formal candidate；没有新的明确 strict rerun 指令时，不再自动进入新方法设计。

### 2. 新方法设计规则

每次新的 formal candidate 只允许改 **一个主变量**，其余口径尽量保持不变。

每轮新方法设计前，必须明确写清：

1. 候选模型名
2. 起点
3. 唯一主改动
4. 为什么这个改动有希望同时改善 Acc 与 AUC
5. 预计会影响的关键现象
   - 例如 `Pred True`
   - 或阈值附近概率是否过于保守
6. 为什么它仍然属于完整方法，而不是 post-hoc 补丁

本地实现时再额外遵守两条：

- 先用 `scripts/cel_stage1_last_layer/scaffold_formal_candidate.py` 生成新候选脚手架，再改实际脚本，避免直接手工复制导致漂移。
- 新候选相对参考脚本的实际 CLI 差异，必须和 `FORMAL_CANDIDATE_CONFIG_DIFFS.md` 中声明的 `config_diff_keys` 一致；若不一致，preflight 应失败。

推荐本地创建顺序：

1. 运行 `python3 scripts/cel_stage1_last_layer/scaffold_formal_candidate.py ...`
2. 在生成的脚手架里只改声明过的 CLI 差异
3. 把 Python / shell registry stub 同步补进正式 registry
4. 重建 `FORMAL_CANDIDATE_BRIEFS.md / FORMAL_CANDIDATE_CONFIG_DIFFS.md`
5. 跑 `preflight_strict_full_train.sh`

附加约束：

- 当 `FORMAL_EXPERIMENT_AUDIT.md` 里仍存在 `needs_rerun` formal candidate 时，`scaffold_formal_candidate.py` 默认拒绝生成新正式候选脚手架。
- 只有当前 formal rerun 队列清空后，才允许把脚手架当成下一轮真实正式候选的起点。
- 若只是离线起草思路，可显式使用 `--allow-pending-rerun` 生成临时草稿，但这不能绕过正式 loop 的“先 rerun、后换方法”规则。

若一轮失败或未双超 baseline，下一轮只允许围绕一个新的主原因做修改，不要同时换多项变量。

### 3. 本地改代码，但只做本地校验

本地允许做的事情：

- 修改 `dialogue_kt/` 与 `scripts/cel_stage1_last_layer/`
- 更新 runbook / loop / record / analysis 文档
- 做 `py_compile`
- 做 shell 语法检查
- 跑 `preflight_strict_full_train.sh`

本地不允许把训练结果当正式结论。

### 4. 所有正式训练都只在 SSH 上启动

推荐顺序：

```bash
bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090 --dry-run
bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090
bash scripts/cel_stage1_last_layer/print_recommended_wsl_ssh_alias.sh
bash scripts/cel_stage1_last_layer/start_current_formal_candidate_via_ssh_alias.sh 3090
bash scripts/cel_stage1_last_layer/start_formal_candidate_via_ssh_alias.sh 3090 v26
bash scripts/cel_stage1_last_layer/start_formal_candidate.sh v26
bash scripts/cel_stage1_last_layer/print_current_formal_manual_launch_steps.sh
bash scripts/cel_stage1_last_layer/print_manual_formal_launch_steps.sh v26
```

若未来新增 formal candidate，也必须沿用同样的 SSH full-train 口径。

若当前执行环境本身不能打开 SSH，但你有另一个可联网终端，则使用：

```bash
bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090 --dry-run
bash scripts/cel_stage1_last_layer/print_current_formal_manual_launch_steps.sh
bash scripts/cel_stage1_last_layer/print_manual_formal_launch_steps.sh v26
```

它会引导到同一个共享远端启动入口：

```bash
bash scripts/cel_stage1_last_layer/launch_round3_candidate_in_remote_repo.sh v26
```

底层 fallback：

```bash
bash scripts/cel_stage1_last_layer/preflight_strict_full_train.sh
bash scripts/cel_stage1_last_layer/launch_round3_candidate_on_server.sh v26
```

### 5. 训练监控节奏

监控间隔不是固定死值，但必须有统一规则。当前默认规则：

- 长时 full-train 的训练早期：可放宽到 `900s`
- 稳定训练中期：通常使用 `600s`
- 若已进入 `Validation / Testing`：通常使用 `300s`
- 若同步到的最新进度达到约 `95%+`，或预计剩余时间已较短：通常使用 `300s`
- 若剩余 ETA 已缩短到约 `<=5m`：进一步收紧到 `120s`

实践规则：

- 以 `STATUS.md` 与 `STRICT_FULL_TRAIN_REPORT.md` 里的 recommended poll interval 为准。
- 若本轮训练特别长，但仍处于稳定早期训练阶段，可保持较慢轮询；不要为了“看起来在动”而过密刷新。
- 若已经接近结束，或正在验证/测试阶段，应主动收紧轮询，方便尽快同步正式结果。
- 若轻量远端直查已经明确看到 `training -> validating/testing` 的相位切换，允许在原定 monitor 时间之前主动补做一次 authoritative refresh，把新的 live phase 与 cadence 建议写回本地 surfaces；不要等旧训练态的建议时间机械到点。
- `v26` 是三个独立进程串行执行的链：bootstrap `--epochs 2`，随后 calibrator warmup `--epochs 1`，最后 strict joint `--epochs 1`。每个阶段的 epoch 计数都会从 1 重新开始；bootstrap -> warmup -> joint 切换不是 run 重启或结果回退，且只有 joint 执行正式 test。
- `active_timing_summary` 只对应当前 live phase 的局部计时；多 epoch run 在 `training / validating / testing` 之间切换时，剩余 ETA 可能重新估计，不应把某一阶段的 ETA 直接当作整轮 full-train 的总剩余时间。

推荐命令：

```bash
bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090 --dry-run
bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090
bash scripts/cel_stage1_last_layer/monitor_formal_candidate.sh once
TASK_CONDITIONED_MONITOR_MAX_CYCLES=1 bash scripts/cel_stage1_last_layer/monitor_formal_candidate.sh watch
```

当某个 formal candidate 的 SSH run 已经完成后，必须先执行统一 closeout：

```bash
bash scripts/cel_stage1_last_layer/finalize_current_formal_candidate.sh
bash scripts/cel_stage1_last_layer/review_current_formal_candidate.sh
bash scripts/cel_stage1_last_layer/finalize_formal_candidate.sh v26
```

若当前执行环境不能 SSH，可先只读本地快照：

```bash
bash scripts/cel_stage1_last_layer/monitor_formal_candidate.sh skip-sync
```

### 6. 结果完成后的强制落盘顺序

每个 formal experiment 完成后，必须按下面顺序处理：

1. 同步远端 `metrics / qual / stdout log`
2. 重建 `task_conditioned_status.json`
3. 重建 markdown surfaces
4. 检查 `FORMAL_EXPERIMENT_AUDIT.md`
5. 再判断是否双超 baseline

不能跳过 audit 直接宣布结果。

推荐回看入口：

```bash
bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090 --dry-run
bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090
bash scripts/cel_stage1_last_layer/finalize_current_formal_candidate.sh
bash scripts/cel_stage1_last_layer/review_current_formal_candidate.sh
bash scripts/cel_stage1_last_layer/finalize_formal_candidate.sh v26
bash scripts/cel_stage1_last_layer/review_formal_candidate.sh v26
```

若当前执行环境不能 SSH，可先读本地快照：

```bash
TASK_CONDITIONED_FINALIZE_SKIP_REFRESH=1 bash scripts/cel_stage1_last_layer/finalize_formal_candidate.sh v26
TASK_CONDITIONED_REVIEW_SKIP_REFRESH=1 bash scripts/cel_stage1_last_layer/review_formal_candidate.sh v26
```

## 每轮实验完成后必须记录什么

每个完成的 formal candidate 都必须写清：

1. 模型名
2. 起点
3. 方法主改动
4. Overall Acc / AUC
5. Final Acc / AUC
6. `Pred True`
7. 相对 baseline 的 delta
8. 是否双超 baseline
9. 若失败：
   - 失败类型
   - 失败位置或关键报错
   - 修复后为何还能保持 strict full-train 口径
10. 若未双超：
   - 下一轮只改一个主变量的原因

## 必须更新的 markdown surfaces

每个已完成 formal SSH run 都必须回写到：

- `STATUS.md`
- `STRICT_FULL_TRAIN_REPORT.md`
- `FORMAL_EXPERIMENT_AUDIT.md`
- `FORMAL_CANDIDATE_BRIEFS.md`
- `FORMAL_CANDIDATE_CONFIG_DIFFS.md`
- `TASK_CONDITIONED_TUNING_LOOP.md`
- `COMPARISON.md`
- `DETAILED_ANALYSIS.md`
- `CEL_Stage1_LastLayer_DialogueKT_实验记录.md`

如果这些 surface 没更新完整，该实验还不能当作正式闭环完成。

## 当前具体执行队列

Stage 1 当前已经完成。执行队列固定为：

1. 保持 `v21 / v22 / v23 / v24 / v25` 为已记录 non-winner。
2. 保持 A 模块历史 checkpoint 为审计通过的 recorded winner，不再继续监控或重跑。
3. 不启动新的 Stage 1 完整联合训练候选；先完成目标 A+B 双路径实现与本地 contract 验证，再注册新的正式 Stage 2 candidate。

## 一句话原则

- **先完整训练，再下结论。**
- **先完成记录与分析，再进入下一轮。**
- **先验证目标机制与 tensor contract，再投入正式训练。**
