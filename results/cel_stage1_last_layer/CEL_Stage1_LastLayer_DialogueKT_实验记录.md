# CEL Stage1 Last-Layer 实验记录

> 术语说明：本文保留历史 candidate 与 checkpoint 名称用于实验追溯；当前方法统一称为 **A 模块**，不再以历史候选编号作为方法名称。

## 当前任务定义

- 当前阶段仍属于 `Stage 1 last-layer`，不是 Stage 2。
- 当前正式评价口径：只有完整训练完成后得到的指标才算正式结果。
- 冻结补训、calibrator-only、fixed-bias eval、validation-fit bias 全部只记为诊断证据。

## 当前状态

- 当前阶段：`task_conditioned_tuning_completed_with_winner`
- Baseline：`lmkt_qwen3_1.7b_recert_20260620`
- Baseline Overall Acc / AUC：**69.22 / 74.95**
- Baseline Final Acc / AUC：**61.17 / 75.32**
- Stage 1 历史 selector/ranking 比较锚点：`cel_task_conditioned_lastlayer_v1_qwen3_1.7b`，Overall Acc / AUC **68.87 / 75.29**
- Stage 1 历史 selector/ranking 比较锚点 Final Acc / AUC：**56.31 / 75.27**
- 当前 A 模块参考结果：历史 checkpoint `cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b`，Overall Acc / AUC **70.03 / 75.38**
- task-conditioned sync freshness：`fresh_remote`
- 最新 authoritative task-conditioned 远端真值摘要：`[2026-08-12 23:17:24] task_conditioned refresh complete :: sync=ok sync_exit=0 stage=task_conditioned_tuning_completed_with_winner next_action=done live_phase=unknown round3=done,done,done,done,done,done winner=true duration=87s`
- 最新 task-conditioned 刷新尝试：`[2026-08-12 23:17:24] task_conditioned refresh complete :: sync=ok sync_exit=0 stage=task_conditioned_tuning_completed_with_winner next_action=done live_phase=unknown round3=done,done,done,done,done,done winner=true duration=87s`
- 当前本地汇总基于上方最近一次 `sync=ok` 事件；其时效性以事件时间戳为准，不能仅凭 `fresh_remote` 判断为当前实时远端状态。
- 当前正式下一步：A 模块历史 checkpoint `cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b` 已完成严格 closeout；停止新增 Stage 1 candidate，转入目标 A+B 双路径实现与 contract 验证。
- Round 3 `cel_task_conditioned_lastlayer_v21_joint_bias_fulltrain_qwen3_1.7b`：`done`，当前 `testing`，epoch 1，进度 `1985/1985 (100%)`
- Round 3 `cel_task_conditioned_lastlayer_v22_fullv1_joint_bias_tinylr_qwen3_1.7b`：`done`，当前 `testing`，epoch 1，进度 `1985/1985 (100%)`
- Round 3 `cel_task_conditioned_lastlayer_v23_joint_bias_posinit_qwen3_1.7b`：`done`，当前 `testing`，epoch 1，进度 `1985/1985 (100%)`
- Round 3 `cel_task_conditioned_lastlayer_v24_selectorinit_joint_bias_fulltrain_qwen3_1.7b`：`done`，当前 `testing`，进度 `1985/1985 (100%)`
- Round 3 `cel_task_conditioned_lastlayer_v25_joint_bias_calibratorlr10x_qwen3_1.7b`：`done`，当前 `testing`，epoch 1，进度 `1985/1985 (100%)`
- Round 3 `cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b`：`done`，当前 `testing`，进度 `1985/1985 (100%)`

## 当前严格正式候选

- 当前 Round 3 strict formal queue 已全部完成，A 模块历史 checkpoint `cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b` 已通过 audit；controller / watcher 不再自动启动任何新候选，除非后续明确需要 strict full-train 复验。
- formal winner 的唯一判定标准：SSH 上完整 `train + val + test` 结束后，Overall Acc 和 Overall AUC 都严格高于 baseline。
- Final Acc / AUC 不改变上述 winner gate，但必须与 baseline 的两项 Final delta 一起完整落盘并通过 audit。

| Model | Start Point | Method | Scope | Status | Success Gate |
|---|---|---|---|---|---|
| `cel_task_conditioned_lastlayer_v21_joint_bias_fulltrain_qwen3_1.7b` | baseline checkpoint | joint bias calibrator full training | SSH train + val + test | done; testing, epoch 1; 1985/1985 (100%) | > 69.22 Acc and > 74.95 AUC |
| `cel_task_conditioned_lastlayer_v22_fullv1_joint_bias_tinylr_qwen3_1.7b` | full `v1` checkpoint | tiny-lr joint full fine-tune | SSH train + val + test | done; testing, epoch 1; 1985/1985 (100%) | > 69.22 Acc and > 74.95 AUC |
| `cel_task_conditioned_lastlayer_v23_joint_bias_posinit_qwen3_1.7b` | baseline checkpoint | keep the v21 joint-bias full-train setup but initialize the calibrator with a positive bias | SSH train + val + test | done; testing, epoch 1; 1985/1985 (100%) | > 69.22 Acc and > 74.95 AUC |
| `cel_task_conditioned_lastlayer_v24_selectorinit_joint_bias_fulltrain_qwen3_1.7b` | baseline checkpoint | keep the v21 joint-bias full-train setup but initialize the selector from the historical selector/ranking v1 anchor | SSH train + val + test | done; testing; 1985/1985 (100%) | > 69.22 Acc and > 74.95 AUC |
| `cel_task_conditioned_lastlayer_v25_joint_bias_calibratorlr10x_qwen3_1.7b` | baseline checkpoint | keep the v21 joint-bias full-train setup but assign the calibrator a higher dedicated learning rate | SSH train + val + test | done; testing, epoch 1; 1985/1985 (100%) | > 69.22 Acc and > 74.95 AUC |
| `cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b` | Qwen3-1.7B base weights with newly initialized A-module LoRA, selector, and calibrator | self-contained A-module chain: bootstrap representation training, calibrator-only warmup, then strict tiny-LR joint training | SSH train + val + test | done; testing; 1985/1985 (100%) | > 69.22 Acc and > 74.95 AUC |

## 正式候选已完成结果分析

### `cel_task_conditioned_lastlayer_v21_joint_bias_fulltrain_qwen3_1.7b`

- Audit 状态：`recorded`
- 是否双超 baseline：`no`
- 起点：从 baseline checkpoint 起步
- 方法主改动：把 bias calibrator 从后处理调节前移为训练期联合优化的一部分
- Overall Acc / AUC：**68.61 / 75.26**；Final Acc / AUC：**56.70 / 74.60**。
- `Pred True`：**32.59%**。
- best-threshold 诊断：best Acc **69.52**，threshold **0.393**。
- 相对 baseline 的 Overall Acc / AUC 变化：**-0.61 / +0.31**。
- 相对 baseline 的 Final Acc / AUC 变化：**-4.47 / -0.72**。
- 相对历史 selector/ranking 比较锚点 `v1` 的 Overall Acc / AUC 变化：**-0.26 / -0.03**。
- 当前结论：该候选已完成并已记录，但未双超 baseline；当前应冻结为 recorded non-winner，而不是继续当作新的正式方法结论。
- 若未双超时的下一轮解释：如果 AUC 仍有竞争力但 Acc 还是不过线，下一轮正式候选只应围绕校准强度、初始化或学习率再改一个变量；如果 AUC 也回落，说明联合 bias 训练已经伤害排序，下一轮应改成一个更受控的训练期主变量，而不是继续叠加后处理补丁。

### `cel_task_conditioned_lastlayer_v22_fullv1_joint_bias_tinylr_qwen3_1.7b`

- Audit 状态：`recorded`
- 是否双超 baseline：`no`
- 起点：从完整 `task_conditioned v1` checkpoint 起步
- 方法主改动：把主变量改成从完整 `v1` checkpoint 出发、只用极小学习率做联合微调
- Overall Acc / AUC：**68.26 / 74.93**；Final Acc / AUC：**64.27 / 72.86**。
- `Pred True`：**47.76%**。
- best-threshold 诊断：best Acc **69.07**，threshold **0.406**。
- 相对 baseline 的 Overall Acc / AUC 变化：**-0.96 / -0.02**。
- 相对 baseline 的 Final Acc / AUC 变化：**+3.10 / -2.46**。
- 相对历史 selector/ranking 比较锚点 `v1` 的 Overall Acc / AUC 变化：**-0.61 / -0.36**。
- 当前结论：该候选已完成并已记录，但未双超 baseline；当前应冻结为 recorded non-winner，而不是继续当作新的正式方法结论。
- 若未双超时的下一轮解释：如果 Acc 和 AUC 一起回落，或 calibrator 几乎没动，说明 tiny-lr warm-start 既没有纠正保守概率，也没有稳住排序；下一轮正式候选应改一个更直接影响训练期校准的主变量，而不是继续重复 warm-start + 后处理式思路。

### `cel_task_conditioned_lastlayer_v23_joint_bias_posinit_qwen3_1.7b`

- Audit 状态：`recorded`
- 是否双超 baseline：`no`
- 起点：从 baseline checkpoint 起步
- 方法主改动：把 joint bias full-train 的 calibrator 初始化从零改成基于诊断证据的正偏置起点
- Overall Acc / AUC：**68.41 / 74.29**；Final Acc / AUC：**56.12 / 74.59**。
- `Pred True`：**33.50%**。
- best-threshold 诊断：best Acc **68.87**，threshold **0.385**。
- 相对 baseline 的 Overall Acc / AUC 变化：**-0.81 / -0.66**。
- 相对 baseline 的 Final Acc / AUC 变化：**-5.05 / -0.73**。
- 相对历史 selector/ranking 比较锚点 `v1` 的 Overall Acc / AUC 变化：**-0.46 / -1.00**。
- 当前结论：该候选已完成并已记录，但未双超 baseline；当前应冻结为 recorded non-winner，而不是继续当作新的正式方法结论。
- 若未双超时的下一轮解释：如果 Pred True 仍然明显偏低且 calibrator bias 又回到接近 0，说明单纯正初始化不足以驱动联训校准；下一轮只应再改一个训练动力学变量，例如 calibrator 学习率或 warmup。若 Pred True 过冲而 AUC 回落，则应减小初始化幅度，而不是叠加更多变量。

### `cel_task_conditioned_lastlayer_v24_selectorinit_joint_bias_fulltrain_qwen3_1.7b`

- Audit 状态：`recorded`
- 是否双超 baseline：`no`
- 起点：从 baseline checkpoint 起步
- 方法主改动：在 v21 的 joint bias full-train 基础上，仅把 selector 初始化改为来自历史 selector/ranking 锚点 v1
- 实现口径核对：代码口径核对：相对 v21 额外只加载 `cel_selector.pt` 到 `_cel_selector`，不会同时 warm-start calibrator，也不会把 backbone / LoRA 一起从 `v1` 继续加载。
- Overall Acc / AUC：**68.16 / 74.44**；Final Acc / AUC：**56.89 / 73.49**。
- `Pred True`：**36.22%**。
- best-threshold 诊断：best Acc **68.87**，threshold **0.430**。
- 相对 baseline 的 Overall Acc / AUC 变化：**-1.06 / -0.51**。
- 相对 baseline 的 Final Acc / AUC 变化：**-4.28 / -1.83**。
- 相对历史 selector/ranking 比较锚点 `v1` 的 Overall Acc / AUC 变化：**-0.71 / -0.85**。
- 当前结论：该候选已完成并已记录，但未双超 baseline；当前应冻结为 recorded non-winner，而不是继续当作新的正式方法结论。
- 若未双超时的下一轮解释：如果 AUC 仍然保不住，说明 selector warm-start 也不足以稳定 joint full-train；下一轮应只再改一个训练动力学变量，而不是继续叠加更多 warm-start 部件。若 AUC 保住但 Acc 仍不过线，则下一轮只应继续围绕训练期校准强度或学习率改一个变量。

### `cel_task_conditioned_lastlayer_v25_joint_bias_calibratorlr10x_qwen3_1.7b`

- Audit 状态：`recorded`
- 是否双超 baseline：`no`
- 起点：从 baseline checkpoint 起步
- 方法主改动：在 v21 的 joint bias full-train 基础上，仅给 calibrator 单独更高的学习率
- Overall Acc / AUC：**66.90 / 74.98**；Final Acc / AUC：**52.04 / 74.27**。
- `Pred True`：**30.18%**。
- best-threshold 诊断：best Acc **69.87**，threshold **0.392**。
- 相对 baseline 的 Overall Acc / AUC 变化：**-2.32 / +0.03**。
- 相对 baseline 的 Final Acc / AUC 变化：**-9.13 / -1.05**。
- 相对历史 selector/ranking 比较锚点 `v1` 的 Overall Acc / AUC 变化：**-1.97 / -0.31**。
- 当前结论：该候选已完成并已记录，但未双超 baseline；当前应冻结为 recorded non-winner，而不是继续当作新的正式方法结论。
- 若未双超时的下一轮解释：如果 calibrator bias 终于动起来但 AUC 明显回落，说明更高 calibrator 学习率正在过度扰动联合训练，下一轮应只减小该学习率或改调度；如果 bias 仍然几乎不动，说明问题不在步幅本身，下一轮应只改另一个训练动力学变量，例如 warmup 或阶段性冻结。

### `cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b`

- Audit 状态：`recorded`
- 是否双超 baseline：`yes`
- 起点：从 Qwen3-1.7B 原始基座开始，新建当前 A 模块训练链自己的 LoRA、A 和 calibrator
- 方法主改动：移除 baseline/v1 checkpoint 初始化，每个阶段只加载同一 A 模块 candidate 上一阶段生成的 checkpoint
- Overall Acc / AUC：**70.03 / 75.38**；Final Acc / AUC：**66.80 / 77.26**。
- `Pred True`：**43.98%**。
- best-threshold 诊断：best Acc **70.53**，threshold **0.520**。
- 相对 baseline 的 Overall Acc / AUC 变化：**+0.81 / +0.43**。
- 相对 baseline 的 Final Acc / AUC 变化：**+5.63 / +1.94**。
- 相对历史 selector/ranking 比较锚点 `v1` 的 Overall Acc / AUC 变化：**+1.16 / +0.09**。
- 当前结论：该候选已完成完整 SSH full-train，且当前记录口径下已满足双超 baseline；当前应冻结为严格 winner，并作为 Stage 1 的正式方法结论。


## 已同步结果

| Model | Type | Loss | Overall Acc | Overall AUC | Final Acc | Final AUC | Diagnostics |
|---|---|---:|---:|---:|---:|---:|---|
| `cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b` | formal | 0.6657 | 70.03 | 75.38 | 66.80 | 77.26 | gate_mean: 0.0000, gate_abs_mean: 0.0000, gate_pos_frac: 0.0000, gate_neg_frac: 0.0000, gate_max: 0.0000, gate_min: 0.0000, injected_mean: 0.2813, cal_bias: 0.3220, post_prob_invalid_frac: 0.0000, post_prob_label_invalid_frac: 0.0000, post_prob_label_out_of_range_frac: 0.0000, post_prob_label_raw_max: 0.4736, post_prob_label_raw_min: 0.4736, post_prob_label_safe_max: 0.4736, post_prob_label_safe_min: 0.4736, post_prob_label_sanitized_frac: 0.0000, post_prob_out_of_range_frac: 0.0000, post_prob_raw_max: 0.4687, post_prob_raw_min: 0.4687, post_prob_safe_max: 0.4687, post_prob_safe_min: 0.4687, post_prob_sanitized_frac: 0.0000, pre_prob_invalid_frac: 0.0000, pre_prob_out_of_range_frac: 0.0000, pre_prob_raw_max: 0.4277, pre_prob_raw_min: 0.4277, pre_prob_safe_max: 0.4277, pre_prob_safe_min: 0.4277, pre_prob_sanitized_frac: 0.0000 |
| `cel_task_conditioned_lastlayer_v1_qwen3_1.7b` | formal | 0.6300 | 68.87 | 75.29 | 56.31 | 75.27 | gate_mean: 0.0000, gate_abs_mean: 0.0000, gate_pos_frac: 0.0000, gate_neg_frac: 0.0000, gate_max: 0.0000, gate_min: 0.0000, injected_mean: 1.1436 |
| `cel_task_conditioned_lastlayer_v21_joint_bias_fulltrain_qwen3_1.7b` | formal | 0.6340 | 68.61 | 75.26 | 56.70 | 74.60 | gate_mean: 0.0000, gate_abs_mean: 0.0000, gate_pos_frac: 0.0000, gate_neg_frac: 0.0000, gate_max: 0.0000, gate_min: 0.0000, injected_mean: 1.1551, cal_bias: -0.0001, post_prob_invalid_frac: 0.0000, post_prob_label_invalid_frac: 0.0000, post_prob_label_out_of_range_frac: 0.0000, post_prob_label_raw_max: 0.4736, post_prob_label_raw_min: 0.4736, post_prob_label_safe_max: 0.4736, post_prob_label_safe_min: 0.4736, post_prob_label_sanitized_frac: 0.0000, post_prob_out_of_range_frac: 0.0000, post_prob_raw_max: 0.3806, post_prob_raw_min: 0.3806, post_prob_safe_max: 0.3806, post_prob_safe_min: 0.3806, post_prob_sanitized_frac: 0.0000, pre_prob_invalid_frac: 0.0000, pre_prob_out_of_range_frac: 0.0000, pre_prob_raw_max: 0.3806, pre_prob_raw_min: 0.3806, pre_prob_safe_max: 0.3806, pre_prob_safe_min: 0.3806, pre_prob_sanitized_frac: 0.0000 |
| `cel_adapter_lastlayer_v1_qwen3_1.7b` | formal | 0.6149 | 68.21 | 75.04 | 65.63 | 72.66 | gate_mean: 0.0000, gate_abs_mean: 0.0000, gate_pos_frac: 0.0000, gate_neg_frac: 0.0000, gate_max: 0.0000, gate_min: 0.0000, injected_mean: 1.0821 |
| `cel_task_conditioned_lastlayer_v25_joint_bias_calibratorlr10x_qwen3_1.7b` | formal | 0.6363 | 66.90 | 74.98 | 52.04 | 74.27 | gate_mean: 0.0000, gate_abs_mean: 0.0000, gate_pos_frac: 0.0000, gate_neg_frac: 0.0000, gate_max: 0.0000, gate_min: 0.0000, injected_mean: 1.1560, cal_bias: -0.0011, post_prob_invalid_frac: 0.0000, post_prob_label_invalid_frac: 0.0000, post_prob_label_out_of_range_frac: 0.0000, post_prob_label_raw_max: 0.4736, post_prob_label_raw_min: 0.4736, post_prob_label_safe_max: 0.4736, post_prob_label_safe_min: 0.4736, post_prob_label_sanitized_frac: 0.0000, post_prob_out_of_range_frac: 0.0000, post_prob_raw_max: 0.3700, post_prob_raw_min: 0.3700, post_prob_safe_max: 0.3700, post_prob_safe_min: 0.3700, post_prob_sanitized_frac: 0.0000, pre_prob_invalid_frac: 0.0000, pre_prob_out_of_range_frac: 0.0000, pre_prob_raw_max: 0.3701, pre_prob_raw_min: 0.3701, pre_prob_safe_max: 0.3701, pre_prob_safe_min: 0.3701, pre_prob_sanitized_frac: 0.0000 |
| `cel_task_conditioned_lastlayer_v22_fullv1_joint_bias_tinylr_qwen3_1.7b` | formal | 0.8027 | 68.26 | 74.93 | 64.27 | 72.86 | gate_mean: 0.0000, gate_abs_mean: 0.0000, gate_pos_frac: 0.0000, gate_neg_frac: 0.0000, gate_max: 0.0000, gate_min: 0.0000, injected_mean: 1.1202, cal_bias: 0.0001, post_prob_invalid_frac: 0.0000, post_prob_out_of_range_frac: 0.0000, post_prob_raw_max: 0.4834, post_prob_raw_min: 0.4834, post_prob_safe_max: 0.4834, post_prob_safe_min: 0.4834, post_prob_sanitized_frac: 0.0272, pre_prob_invalid_frac: 0.0000, pre_prob_out_of_range_frac: 0.0000, pre_prob_raw_max: 0.4833, pre_prob_raw_min: 0.4833, pre_prob_safe_max: 0.4833, pre_prob_safe_min: 0.4833, pre_prob_sanitized_frac: 0.0272 |
| `cel_mlp_lastlayer_v1_qwen3_1.7b` | formal | 0.6906 | 65.74 | 74.81 | 69.32 | 72.73 | gate_mean: 0.0000, gate_abs_mean: 0.0000, gate_pos_frac: 0.0000, gate_neg_frac: 0.0000, gate_max: 0.0000, gate_min: 0.0000, injected_mean: 1.0770 |
| `cel_task_conditioned_lastlayer_v24_selectorinit_joint_bias_fulltrain_qwen3_1.7b` | formal | 0.6326 | 68.16 | 74.44 | 56.89 | 73.49 | gate_mean: 0.0000, gate_abs_mean: 0.0000, gate_pos_frac: 0.0000, gate_neg_frac: 0.0000, gate_max: 0.0000, gate_min: 0.0000, injected_mean: 1.1219, cal_bias: -0.0002, post_prob_invalid_frac: 0.0000, post_prob_label_invalid_frac: 0.0000, post_prob_label_out_of_range_frac: 0.0000, post_prob_label_raw_max: 0.4736, post_prob_label_raw_min: 0.4736, post_prob_label_safe_max: 0.4736, post_prob_label_safe_min: 0.4736, post_prob_label_sanitized_frac: 0.0000, post_prob_out_of_range_frac: 0.0000, post_prob_raw_max: 0.3952, post_prob_raw_min: 0.3952, post_prob_safe_max: 0.3952, post_prob_safe_min: 0.3952, post_prob_sanitized_frac: 0.0000, pre_prob_invalid_frac: 0.0000, pre_prob_out_of_range_frac: 0.0000, pre_prob_raw_max: 0.3953, pre_prob_raw_min: 0.3953, pre_prob_safe_max: 0.3953, pre_prob_safe_min: 0.3953, pre_prob_sanitized_frac: 0.0000 |
| `cel_task_conditioned_lastlayer_v23_joint_bias_posinit_qwen3_1.7b` | formal | 0.6301 | 68.41 | 74.29 | 56.12 | 74.59 | gate_mean: 0.0000, gate_abs_mean: 0.0000, gate_pos_frac: 0.0000, gate_neg_frac: 0.0000, gate_max: 0.0000, gate_min: 0.0000, injected_mean: 1.1407, cal_bias: 0.3302, post_prob_invalid_frac: 0.0000, post_prob_label_invalid_frac: 0.0000, post_prob_label_out_of_range_frac: 0.0000, post_prob_label_raw_max: 0.4736, post_prob_label_raw_min: 0.4736, post_prob_label_safe_max: 0.4736, post_prob_label_safe_min: 0.4736, post_prob_label_sanitized_frac: 0.0000, post_prob_out_of_range_frac: 0.0000, post_prob_raw_max: 0.3921, post_prob_raw_min: 0.3921, post_prob_safe_max: 0.3921, post_prob_safe_min: 0.3921, post_prob_sanitized_frac: 0.0000, pre_prob_invalid_frac: 0.0000, pre_prob_out_of_range_frac: 0.0000, pre_prob_raw_max: 0.3413, pre_prob_raw_min: 0.3413, pre_prob_safe_max: 0.3413, pre_prob_safe_min: 0.3413, pre_prob_sanitized_frac: 0.0000 |
| `cel_task_conditioned_lastlayer_v15_fullv1_cal_bias_only_qwen3_1.7b` | diagnostic | 0.6078 | 69.27 | 75.29 | 61.94 | 75.27 | gate_mean: 0.0000, gate_abs_mean: 0.0000, gate_pos_frac: 0.0000, gate_neg_frac: 0.0000, gate_max: 0.0000, gate_min: 0.0000, injected_mean: 1.1436, cal_bias: 0.3802 |
| `cel_task_conditioned_lastlayer_v12_cal_affine_only_qwen3_1.7b` | diagnostic | 0.6294 | 65.49 | 73.83 | 50.49 | 73.84 | gate_mean: 0.0000, gate_abs_mean: 0.0000, gate_pos_frac: 0.0000, gate_neg_frac: 0.0000, gate_max: 0.0000, gate_min: 0.0000, injected_mean: 1.0836, cal_bias: 0.0890, cal_scale: 0.6794 |
| `cel_task_conditioned_lastlayer_v11_cal_bias_only_qwen3_1.7b` | diagnostic | 0.6733 | 65.04 | 73.83 | 49.90 | 73.84 | gate_mean: 0.0000, gate_abs_mean: 0.0000, gate_pos_frac: 0.0000, gate_neg_frac: 0.0000, gate_max: 0.0000, gate_min: 0.0000, injected_mean: 1.0836, cal_bias: 0.0945 |

## 当前结论

- 当前 A 模块参考结果来自历史 checkpoint `cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b`，Overall Acc / AUC **70.03 / 75.38**，相对 baseline 为 **+0.81 / +0.43**。
- 当前诊断最强结果是 `cel_task_conditioned_lastlayer_v15_fullv1_cal_bias_only_qwen3_1.7b`，Overall Acc / AUC **69.27 / 75.29**，但它不计入正式 winner。
- 当前不能把 `v15` 或 `v20` 当作正式方法分数。
- 当前 A 模块参考结果已冻结，历史 checkpoint 为 `cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b`；诊断轮次仍不计入正式方法，若未来需要复验必须重新走完整 SSH strict full-train workflow。

## 下一步

1. 冻结 A 模块历史 checkpoint `cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b` 的 strict winner 结论，不再启动新的 formal candidate。
2. 复核 `STATUS.md / STRICT_FULL_TRAIN_REPORT.md / FORMAL_EXPERIMENT_AUDIT.md / TASK_CONDITIONED_TUNING_LOOP.md / CEL_Stage1_LastLayer_DialogueKT_实验记录.md` 的 winner 文案是否一致。
3. 将 Stage 2 首轮旧单路径结果保留为历史 pilot，不作为当前 A+B 机制的正式验证。
4. 实现 `a -> h_r/h_n/h_nb/h_m`、共享后续网络双路径前向、`L_r + L_m + L_cons` 三项核心损失与 tensor contracts。
5. 在代码与 audit 路径验证完成前，不注册新的正式 Stage 2 candidate，也不启动 Stage 3。
