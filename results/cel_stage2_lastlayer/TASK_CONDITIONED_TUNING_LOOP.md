# Task-Conditioned Tuning Loop

## 目标

基于当前固定方案 `cel_task_conditioned_lastlayer_v1_qwen3_1.7b`，继续调优，目标是在同一模型族中同时超过当前 baseline：

- Baseline Overall AUC: **74.95**
- Baseline Overall Acc: **69.22**

当前起点：

- `cel_task_conditioned_lastlayer_v1_qwen3_1.7b`
- Overall AUC: **75.29**
- Overall Acc: **68.87**

## 当前分析

当前 `v1` 的关键现象不是“排序不够”，而是“概率偏保守”：

- `Pred True`: **33.8%**
- baseline `Pred True`: **40.8%**
- `v1` 的最佳 Acc 阈值约为 **0.418**
- baseline 的最佳 Acc 阈值约为 **0.488**

这说明：

- 当前 `task_conditioned v1` 的排序能力已经优于 baseline
- 但输出分数整体偏低，导致固定 `0.5` 阈值下 Recall/Acc 被压住

因此本轮先不激进修改 selector 或 hook，而先验证最小改动：

## Round 1: Output Calibration

候选：

1. `cel_task_conditioned_lastlayer_v11_cal_bias_only_qwen3_1.7b`
   - 固定 `task_conditioned v1` selector 行为
   - 只训练一个 logit bias 校准项
2. `cel_task_conditioned_lastlayer_v12_cal_affine_only_qwen3_1.7b`
   - 固定 `task_conditioned v1` selector 行为
   - 只训练一个正向 affine logit 校准器

设计意图：

- 尽量保留 `v1` 已有的 AUC 优势
- 优先把 `Pred True` 从 33.8% 往 baseline 40.8% 附近拉回
- 如果 Acc 能超过 **69.22**，就证明当前短板主要是 calibration，不是 ranking

## 记录要求

每轮结束后补充：

- 模型名
- Overall Acc / AUC
- Final Acc / AUC
- `Pred True`
- 和 baseline 的差值
- 是否同时超过 baseline 的 Acc 和 AUC
- 若未超过，写明下一轮调整原因
