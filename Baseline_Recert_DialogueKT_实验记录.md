# Baseline Recert 实验记录

## 目标

重新训练并认证 `Qwen3-1.7B + LoRA` 在当前代码与当前数据切分下的 baseline 性能，作为新一轮 CEL 设计的唯一比较基准。

## 计划运行

- 模型名：`lmkt_qwen3_1.7b_recert_20260620`
- 结果目录：`results/baseline_recert/`
- 训练脚本：`scripts/cel_stage2_lastlayer/run_baseline_recert.sh`

## 状态

- 2026-06-20：脚本已创建，等待服务器正式运行结果。
- 2026-06-20：当前运行状态与最新 loop 事件可先看 `results/cel_stage2_lastlayer/STATUS.md`。
- 2026-06-20：若主 loop 完成后 CEL 仍未超过 baseline，会自动进入 `results/cel_stage2_lastlayer/followup.log` 对应的后续实验链。
