# Baseline 实验记录

## 目标

重新训练并认证 `Qwen3-1.7B + LoRA` 在当前代码与当前数据切分下的 baseline 性能，作为新一轮 CEL 设计的唯一比较基准。

## 计划运行

- 模型名：`lmkt_qwen3_1.7b_recert_20260620`
- 结果目录：`results/baseline/`
- 训练脚本：baseline 认证使用的历史脚本已清理；后续 A/A+B 重训入口见 `scripts/cel/`。

## 状态

- 2026-06-20：baseline 已完成当前认证，指标已同步到本地。

## 结果

- Loss: **0.5928**
- Overall: Acc **69.22**, AUC **74.95**, Prec **71.12**, Rec **58.94**, F1 **64.46**
- Final Turn: Acc **61.17**, AUC **75.32**, Prec **89.64**, Rec **52.93**, F1 **66.56**
