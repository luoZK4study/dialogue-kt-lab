# Formal Candidate Briefs

- 目的：把当前 strict full-train 正式候选的“唯一主改动 / 目标假设 / 期望观察 / 失败后如何解释”集中记录在一处。
- 口径：只有 SSH 上完整 `train + val + test` 的结果才允许进入正式方法判断；任何 ckpt 后处理或冻结补训都只算诊断。

## Current Gate

- Formal winner gate: `Overall Acc > 69.22` and `Overall AUC > 74.95`.
- Historical selector/ranking comparison anchor: `cel_task_conditioned_lastlayer_v1_qwen3_1.7b` with Overall Acc / AUC **68.87 / 75.29**.
- Current loop next action: `done`.
- Rule: if any formal candidate is still `needs_rerun`, do not design a new method yet.

## Current Queue

| Candidate | Status | Audit | Current Result | Next Use |
|---|---|---|---|---|
| `v21` | done | `recorded` | 68.61/75.26 | freeze / analyze |
| `v22` | done | `recorded` | 68.26/74.93 | freeze / analyze |
| `v23` | done | `recorded` | 68.41/74.29 | freeze / analyze |
| `v24` | done | `recorded` | 68.16/74.44 | freeze / analyze |
| `v25` | done | `recorded` | 66.90/74.98 | freeze / analyze |
| `v26` | done | `recorded` | 70.03/75.38 | freeze / analyze |

## Candidate Briefs

### `v21` / `cel_task_conditioned_lastlayer_v21_joint_bias_fulltrain_qwen3_1.7b`

- 当前状态：`done`
- Audit 状态：`recorded`
- 起点：从 baseline checkpoint 起步
- 方法：把 bias calibrator 作为方法一部分，从训练开始就联合训练全部可训练参数
- 唯一主改动：把 bias calibrator 从后处理调节前移为训练期联合优化的一部分
- 目标假设：如果 `v1` 的主要缺口来自概率过于保守，那么把 bias calibrator 从训练开始就纳入联合优化，应能在不破坏排序能力的前提下把 0.5 阈值下的 Acc 拉回去。
- 期望观察：
  - `Pred True` 应从 `v1` 的保守区间往 baseline 水平回升，而不是继续明显偏低。
  - Overall Acc 应相对 `v1` 回升并争取超过 baseline，同时 Overall AUC 至少不要明显低于历史 selector/ranking 比较锚点。
  - 若该方法成立，收益必须体现在一次完整 SSH `train + val + test` 里，而不是只在后处理校准里出现。
- 当前已同步结果：Overall Acc / AUC **68.61 / 75.26**，Final Acc / AUC **56.70 / 74.60**。
- 当前 `Pred True`：**32.59%**。
- 当前 best-threshold 诊断：best Acc **69.52**，threshold **0.393**。
- 相对 baseline 的 Overall Acc / AUC 变化：**-0.61 / +0.31**。
- 相对 baseline 的 Final Acc / AUC 变化：**-4.47 / -0.72**。
- 相对历史 selector/ranking 比较锚点 `v1` 的 Overall Acc / AUC 变化：**-0.26 / -0.03**。
- 当前日志进度：epoch 1 testing 约 `1985/1985 (100%)`
- 若未双超时的下一轮解释：如果 AUC 仍有竞争力但 Acc 还是不过线，下一轮正式候选只应围绕校准强度、初始化或学习率再改一个变量；如果 AUC 也回落，说明联合 bias 训练已经伤害排序，下一轮应改成一个更受控的训练期主变量，而不是继续叠加后处理补丁。
- 当前失败修复说明：当前本地修复：`dialogue_kt/training.py` 已把概率 BCE 路径统一改为先 sanitize，再用 `BCEWithLogitsLoss(torch.logit(safe_probs), safe_labels)` 计算，并补充 label sanitize，避免 joint calibrator 路径再次把非法值直接送入 CUDA `BCELoss`。
- 当前失败修复说明：下一次 `v21` 重跑要求：先同步更新后的 `dialogue_kt/training.py` 到 SSH，再保持原来的 strict full-train 配置完整重跑；不能改成 checkpoint-only / post-hoc 校准。
- 当前结论：该候选已完成并已记录，但当前只能冻结为 non-winner，不应继续把它当成新的正式方法结论。
- 推荐命令：
  - 当前 formal queue 一键 alias 动作：`bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090`
  - 当前 formal queue 一键 alias 动作 dry-run：`bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090 --dry-run`
  - 本地快照落盘：`TASK_CONDITIONED_FINALIZE_SKIP_REFRESH=1 bash scripts/cel_stage1_last_layer/finalize_formal_candidate.sh v21`
  - 本地快照回看：`TASK_CONDITIONED_REVIEW_SKIP_REFRESH=1 bash scripts/cel_stage1_last_layer/review_formal_candidate.sh v21`
  - alias 本地快照落盘：`TASK_CONDITIONED_FINALIZE_SKIP_REFRESH=1 bash scripts/cel_stage1_last_layer/finalize_formal_candidate_via_ssh_alias.sh 3090 v21`
  - alias 本地快照回看：`TASK_CONDITIONED_REVIEW_SKIP_REFRESH=1 bash scripts/cel_stage1_last_layer/review_formal_candidate_via_ssh_alias.sh 3090 v21`
  - 本地快照总览：`bash scripts/cel_stage1_last_layer/monitor_formal_candidate.sh skip-sync`
  - 当前候选已记录为 non-winner；除非明确要做同口径 rerun，否则不应重新启动 `v21`。

### `v22` / `cel_task_conditioned_lastlayer_v22_fullv1_joint_bias_tinylr_qwen3_1.7b`

- 当前状态：`done`
- Audit 状态：`recorded`
- 起点：从完整 `task_conditioned v1` checkpoint 起步
- 方法：保持全部相关参数可训练，用极小学习率做完整联合微调
- 唯一主改动：把主变量改成从完整 `v1` checkpoint 出发、只用极小学习率做联合微调
- 目标假设：如果 `v1` 已经具备更好的排序，主要问题只是默认阈值下过于保守，那么从 `v1` warm-start 并用极小学习率联调，可能在尽量不破坏 AUC 的前提下把概率尺度推回更有利于 Acc 的区间。
- 期望观察：
  - Overall AUC 应尽量贴近 `v1`，而不是把当前锚点的排序优势洗掉。
  - `Pred True` 应温和上升并靠近 baseline，而不是直接漂移到过多 True 的区间。
  - 若 warm-start 校准假设成立，Overall Acc 应向 baseline 靠近并最好越过 baseline。
- 当前已同步结果：Overall Acc / AUC **68.26 / 74.93**，Final Acc / AUC **64.27 / 72.86**。
- 当前 `Pred True`：**47.76%**。
- 当前 best-threshold 诊断：best Acc **69.07**，threshold **0.406**。
- 相对 baseline 的 Overall Acc / AUC 变化：**-0.96 / -0.02**。
- 相对 baseline 的 Final Acc / AUC 变化：**+3.10 / -2.46**。
- 相对历史 selector/ranking 比较锚点 `v1` 的 Overall Acc / AUC 变化：**-0.61 / -0.36**。
- 当前日志进度：epoch 1 testing 约 `1985/1985 (100%)`
- 若未双超时的下一轮解释：如果 Acc 和 AUC 一起回落，或 calibrator 几乎没动，说明 tiny-lr warm-start 既没有纠正保守概率，也没有稳住排序；下一轮正式候选应改一个更直接影响训练期校准的主变量，而不是继续重复 warm-start + 后处理式思路。
- 当前结论：该候选已完成并已记录，但当前只能冻结为 non-winner，不应继续把它当成新的正式方法结论。
- 推荐命令：
  - 当前 formal queue 一键 alias 动作：`bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090`
  - 当前 formal queue 一键 alias 动作 dry-run：`bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090 --dry-run`
  - 本地快照落盘：`TASK_CONDITIONED_FINALIZE_SKIP_REFRESH=1 bash scripts/cel_stage1_last_layer/finalize_formal_candidate.sh v22`
  - 本地快照回看：`TASK_CONDITIONED_REVIEW_SKIP_REFRESH=1 bash scripts/cel_stage1_last_layer/review_formal_candidate.sh v22`
  - alias 本地快照落盘：`TASK_CONDITIONED_FINALIZE_SKIP_REFRESH=1 bash scripts/cel_stage1_last_layer/finalize_formal_candidate_via_ssh_alias.sh 3090 v22`
  - alias 本地快照回看：`TASK_CONDITIONED_REVIEW_SKIP_REFRESH=1 bash scripts/cel_stage1_last_layer/review_formal_candidate_via_ssh_alias.sh 3090 v22`
  - 本地快照总览：`bash scripts/cel_stage1_last_layer/monitor_formal_candidate.sh skip-sync`
  - 当前候选已记录为 non-winner；除非明确要做同口径 rerun，否则不应重新启动 `v22`。

### `v23` / `cel_task_conditioned_lastlayer_v23_joint_bias_posinit_qwen3_1.7b`

- 当前状态：`done`
- Audit 状态：`recorded`
- 起点：从 baseline checkpoint 起步
- 方法：保持 v21 的 joint bias full-train 训练口径，只把 calibrator 初始化改成正偏置起点
- 唯一主改动：把 joint bias full-train 的 calibrator 初始化从零改成基于诊断证据的正偏置起点
- 目标假设：如果 v21 失败的主要原因是 joint bias 在零初始化下几乎没有被推离 no-op 状态，那么把初始 bias 设为 v15 已验证过的正偏置起点，应能在保住 AUC 的同时把 Pred True 和默认阈值 Acc 往 baseline 区间拉回。
- 期望观察：
  - Pred True 应从 v21 的 32.59% 明显回升，并尽量靠近 baseline 40.8%，而不是继续卡在保守区间。
  - Overall Acc 应至少超过 v21 的 68.61，并争取越过 baseline 69.22，同时 Overall AUC 不应明显低于 v21/v1 的 75.26/75.29。
  - 训练结束后 calibrator bias 应保持显著正值，而不是像 v21 一样几乎回到 0。
- 当前已同步结果：Overall Acc / AUC **68.41 / 74.29**，Final Acc / AUC **56.12 / 74.59**。
- 当前 `Pred True`：**33.50%**。
- 当前 best-threshold 诊断：best Acc **68.87**，threshold **0.385**。
- 相对 baseline 的 Overall Acc / AUC 变化：**-0.81 / -0.66**。
- 相对 baseline 的 Final Acc / AUC 变化：**-5.05 / -0.73**。
- 相对历史 selector/ranking 比较锚点 `v1` 的 Overall Acc / AUC 变化：**-0.46 / -1.00**。
- 当前日志进度：epoch 1 testing 约 `1985/1985 (100%)`
- 若未双超时的下一轮解释：如果 Pred True 仍然明显偏低且 calibrator bias 又回到接近 0，说明单纯正初始化不足以驱动联训校准；下一轮只应再改一个训练动力学变量，例如 calibrator 学习率或 warmup。若 Pred True 过冲而 AUC 回落，则应减小初始化幅度，而不是叠加更多变量。
- 当前结论：该候选已完成并已记录，但当前只能冻结为 non-winner，不应继续把它当成新的正式方法结论。
- 推荐命令：
  - 当前 formal queue 一键 alias 动作：`bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090`
  - 当前 formal queue 一键 alias 动作 dry-run：`bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090 --dry-run`
  - 本地快照落盘：`TASK_CONDITIONED_FINALIZE_SKIP_REFRESH=1 bash scripts/cel_stage1_last_layer/finalize_formal_candidate.sh v23`
  - 本地快照回看：`TASK_CONDITIONED_REVIEW_SKIP_REFRESH=1 bash scripts/cel_stage1_last_layer/review_formal_candidate.sh v23`
  - alias 本地快照落盘：`TASK_CONDITIONED_FINALIZE_SKIP_REFRESH=1 bash scripts/cel_stage1_last_layer/finalize_formal_candidate_via_ssh_alias.sh 3090 v23`
  - alias 本地快照回看：`TASK_CONDITIONED_REVIEW_SKIP_REFRESH=1 bash scripts/cel_stage1_last_layer/review_formal_candidate_via_ssh_alias.sh 3090 v23`
  - 本地快照总览：`bash scripts/cel_stage1_last_layer/monitor_formal_candidate.sh skip-sync`
  - 当前候选已记录为 non-winner；除非明确要做同口径 rerun，否则不应重新启动 `v23`。

### `v24` / `cel_task_conditioned_lastlayer_v24_selectorinit_joint_bias_fulltrain_qwen3_1.7b`

- 当前状态：`done`
- Audit 状态：`recorded`
- 起点：从 baseline checkpoint 起步
- 方法：保持 v21 的 joint bias full-train 训练口径，只把 selector 初始化改成来自历史 selector/ranking 锚点 v1
- 唯一主改动：在 v21 的 joint bias full-train 基础上，仅把 selector 初始化改为来自历史 selector/ranking 锚点 v1
- 目标假设：如果 v21 的问题主要不是联训 bias 本身，而是从 baseline 冷启动时丢失了 v1 已学到的排序信号，那么仅用 v1 warm-start selector 应能在不破坏训练期校准的前提下保住 AUC 并把 Acc 拉回去。
- 实现口径核对：代码口径核对：相对 v21 额外只加载 `cel_selector.pt` 到 `_cel_selector`，不会同时 warm-start calibrator，也不会把 backbone / LoRA 一起从 `v1` 继续加载。
- 期望观察：
  - Overall AUC 应至少贴近 v21/v1 的 75.26/75.29，而不是像全量 warm-start 的 v22 那样回落。
  - Pred True 应比 v21 的 32.59% 更接近 baseline 区间，同时不要漂移到 v22 的过高 True 区间。
  - Overall Acc 应至少超过 v21 的 68.61，并争取越过 baseline 69.22。
- 当前已同步结果：Overall Acc / AUC **68.16 / 74.44**，Final Acc / AUC **56.89 / 73.49**。
- 当前 `Pred True`：**36.22%**。
- 当前 best-threshold 诊断：best Acc **68.87**，threshold **0.430**。
- 相对 baseline 的 Overall Acc / AUC 变化：**-1.06 / -0.51**。
- 相对 baseline 的 Final Acc / AUC 变化：**-4.28 / -1.83**。
- 相对历史 selector/ranking 比较锚点 `v1` 的 Overall Acc / AUC 变化：**-0.71 / -0.85**。
- 当前日志进度：testing 约 `1985/1985 (100%)`
- 多 epoch 监控说明：该候选脚本配置 `--epochs 2`；epoch 间出现 `validation -> training` 的回切属于正常的多 epoch full-train 流程，不表示 run 重启。
- 若未双超时的下一轮解释：如果 AUC 仍然保不住，说明 selector warm-start 也不足以稳定 joint full-train；下一轮应只再改一个训练动力学变量，而不是继续叠加更多 warm-start 部件。若 AUC 保住但 Acc 仍不过线，则下一轮只应继续围绕训练期校准强度或学习率改一个变量。
- 当前结论：该候选已完成并已记录，但当前只能冻结为 non-winner，不应继续把它当成新的正式方法结论。
- 推荐命令：
  - 当前 formal queue 一键 alias 动作：`bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090`
  - 当前 formal queue 一键 alias 动作 dry-run：`bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090 --dry-run`
  - 本地快照落盘：`TASK_CONDITIONED_FINALIZE_SKIP_REFRESH=1 bash scripts/cel_stage1_last_layer/finalize_formal_candidate.sh v24`
  - 本地快照回看：`TASK_CONDITIONED_REVIEW_SKIP_REFRESH=1 bash scripts/cel_stage1_last_layer/review_formal_candidate.sh v24`
  - alias 本地快照落盘：`TASK_CONDITIONED_FINALIZE_SKIP_REFRESH=1 bash scripts/cel_stage1_last_layer/finalize_formal_candidate_via_ssh_alias.sh 3090 v24`
  - alias 本地快照回看：`TASK_CONDITIONED_REVIEW_SKIP_REFRESH=1 bash scripts/cel_stage1_last_layer/review_formal_candidate_via_ssh_alias.sh 3090 v24`
  - 本地快照总览：`bash scripts/cel_stage1_last_layer/monitor_formal_candidate.sh skip-sync`
  - 当前候选已记录为 non-winner；除非明确要做同口径 rerun，否则不应重新启动 `v24`。

### `v25` / `cel_task_conditioned_lastlayer_v25_joint_bias_calibratorlr10x_qwen3_1.7b`

- 当前状态：`done`
- Audit 状态：`recorded`
- 起点：从 baseline checkpoint 起步
- 方法：保持 v21 的 joint bias full-train 训练口径，只给 calibrator 单独更高的学习率
- 唯一主改动：在 v21 的 joint bias full-train 基础上，仅给 calibrator 单独更高的学习率
- 目标假设：如果 v21 和 v23 失败的主要原因是 calibrator 在共享学习率下几乎无法摆脱 no-op 动力学，那么只提高 calibrator 的训练步幅，应能把 Pred True 和默认阈值 Acc 往 baseline 区间拉回，同时尽量保住 v21 的排序能力。
- 期望观察：
  - 训练结束后 calibrator bias 应明显偏离 v21 的近零终点，而不是继续停在 no-op 附近。
  - Pred True 应从 v21 的 32.59% 往 baseline 40.8% 区间回升，但不要像 v22 那样过冲到明显过高的 True 率。
  - Overall AUC 应尽量贴近 v21 的 75.26，同时 Overall Acc 至少要超过 v21 的 68.61 并争取越过 baseline 69.22。
- 当前已同步结果：Overall Acc / AUC **66.90 / 74.98**，Final Acc / AUC **52.04 / 74.27**。
- 当前 `Pred True`：**30.18%**。
- 当前 best-threshold 诊断：best Acc **69.87**，threshold **0.392**。
- 相对 baseline 的 Overall Acc / AUC 变化：**-2.32 / +0.03**。
- 相对 baseline 的 Final Acc / AUC 变化：**-9.13 / -1.05**。
- 相对历史 selector/ranking 比较锚点 `v1` 的 Overall Acc / AUC 变化：**-1.97 / -0.31**。
- 当前日志进度：epoch 1 testing 约 `1985/1985 (100%)`
- 若未双超时的下一轮解释：如果 calibrator bias 终于动起来但 AUC 明显回落，说明更高 calibrator 学习率正在过度扰动联合训练，下一轮应只减小该学习率或改调度；如果 bias 仍然几乎不动，说明问题不在步幅本身，下一轮应只改另一个训练动力学变量，例如 warmup 或阶段性冻结。
- 当前结论：该候选已完成并已记录，但当前只能冻结为 non-winner，不应继续把它当成新的正式方法结论。
- 推荐命令：
  - 当前 formal queue 一键 alias 动作：`bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090`
  - 当前 formal queue 一键 alias 动作 dry-run：`bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090 --dry-run`
  - 本地快照落盘：`TASK_CONDITIONED_FINALIZE_SKIP_REFRESH=1 bash scripts/cel_stage1_last_layer/finalize_formal_candidate.sh v25`
  - 本地快照回看：`TASK_CONDITIONED_REVIEW_SKIP_REFRESH=1 bash scripts/cel_stage1_last_layer/review_formal_candidate.sh v25`
  - alias 本地快照落盘：`TASK_CONDITIONED_FINALIZE_SKIP_REFRESH=1 bash scripts/cel_stage1_last_layer/finalize_formal_candidate_via_ssh_alias.sh 3090 v25`
  - alias 本地快照回看：`TASK_CONDITIONED_REVIEW_SKIP_REFRESH=1 bash scripts/cel_stage1_last_layer/review_formal_candidate_via_ssh_alias.sh 3090 v25`
  - 本地快照总览：`bash scripts/cel_stage1_last_layer/monitor_formal_candidate.sh skip-sync`
  - 当前候选已记录为 non-winner；除非明确要做同口径 rerun，否则不应重新启动 `v25`。

### `v26` / `cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b`

- 当前状态：`done`
- Audit 状态：`recorded`
- 起点：从 Qwen3-1.7B 原始基座开始，新建当前 A 模块训练链自己的 LoRA、A 和 calibrator
- 方法：A 模块自包含三阶段链：表征 bootstrap、calibrator-only warmup、strict tiny-LR joint training
- 唯一主改动：移除 baseline/v1 checkpoint 初始化，每个阶段只加载同一 A 模块 candidate 上一阶段生成的 checkpoint
- 目标假设：从 Qwen3 原始基座独立训练的 A 模块可以自行学习 KT LoRA 和 task-conditioned selector，在完成概率校准后通过 strict joint 阶段保留有效参数更新，而无需继承 baseline 或 v1 权重。
- 期望观察：
  - bootstrap 日志必须显示新建 LoRA，且不能出现 baseline、v1 或旧 v26 checkpoint 路径。
  - calibrator warmup 必须保持同一 candidate 的 LoRA/A 不变，strict joint 阶段必须让至少一个 A 模块可训练组件发生变化。
  - 最终报告必须同时比较 Overall Acc/AUC 与 Final Acc/AUC。
- 当前已同步结果：Overall Acc / AUC **70.03 / 75.38**，Final Acc / AUC **66.80 / 77.26**。
- 当前 `Pred True`：**43.98%**。
- 当前 best-threshold 诊断：best Acc **70.53**，threshold **0.520**。
- 相对 baseline 的 Overall Acc / AUC 变化：**+0.81 / +0.43**。
- 相对 baseline 的 Final Acc / AUC 变化：**+5.63 / +1.94**。
- 相对历史 selector/ranking 比较锚点 `v1` 的 Overall Acc / AUC 变化：**+1.16 / +0.09**。
- 当前日志进度：testing 约 `1985/1985 (100%)`
- 多 epoch 监控说明：该候选是独立进程串行的多阶段链：`bootstrap (2 epochs) -> calibrator_warmup (1 epoch) -> joint (1 epoch)`；每个阶段的 epoch 计数会重新从 1 开始。bootstrap -> calibrator warmup -> joint 的阶段切换不是 run 重启；只有 joint 执行正式 test。
- 若未双超时的下一轮解释：如果来源或参数变化审计失败，无论指标如何都判为无效；若审计通过但指标未过 baseline gate，则先把独立训练的 A 模块 candidate 记录为 non-winner，再只调整一个训练变量。
- 当前结论：该候选在已同步结果中满足双超 baseline，且已完成 recorded closeout；当前应保持 strict winner 结论冻结，而不是继续启动新 run。
- 推荐命令：
  - 当前 formal queue 一键 alias 动作：`bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090`
  - 当前 formal queue 一键 alias 动作 dry-run：`bash scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh 3090 --dry-run`
  - 本地快照落盘：`TASK_CONDITIONED_FINALIZE_SKIP_REFRESH=1 bash scripts/cel_stage1_last_layer/finalize_formal_candidate.sh v26`
  - 本地快照回看：`TASK_CONDITIONED_REVIEW_SKIP_REFRESH=1 bash scripts/cel_stage1_last_layer/review_formal_candidate.sh v26`
  - alias 本地快照落盘：`TASK_CONDITIONED_FINALIZE_SKIP_REFRESH=1 bash scripts/cel_stage1_last_layer/finalize_formal_candidate_via_ssh_alias.sh 3090 v26`
  - alias 本地快照回看：`TASK_CONDITIONED_REVIEW_SKIP_REFRESH=1 bash scripts/cel_stage1_last_layer/review_formal_candidate_via_ssh_alias.sh 3090 v26`
  - 本地快照总览：`bash scripts/cel_stage1_last_layer/monitor_formal_candidate.sh skip-sync`
  - 当前候选已是 recorded strict winner；优先保持结论面冻结状态，而不是重新启动 `v26`。

