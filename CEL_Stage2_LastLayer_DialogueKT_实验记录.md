# CEL Stage2 Last-Layer 实验记录

## 目标

在 `baseline recert` 之后，基于“最后一层注入 + baseline checkpoint 初始化 + 零偏移残差 gate”重新验证：

1. `mlp`
2. `adapter`
3. `task_conditioned`

目标是三者中至少一个超过重跑后的 baseline。

## 设计变更

- 注入层改为最后一层：`--cel_layer_idx -1`
- selector 输出从 `sigmoid gate` 改为 `tanh delta`
- selector 末层零初始化，训练起点与 baseline 等价
- 默认关闭额外 norm：`--cel_use_norm 0`
- `task_conditioned` 强制走 unpacked prompt，避免多 KC 平均 gate
- CEL 训练从 baseline checkpoint 开始：`--pt_model_name lmkt_qwen3_1.7b_recert_20260620`

## 首轮计划

- `cel_mlp_lastlayer_v1_qwen3_1.7b`
- `cel_adapter_lastlayer_v1_qwen3_1.7b`
- `cel_task_conditioned_lastlayer_v1_qwen3_1.7b`

## 条件化第二轮

如果首轮最佳 AUC 仍未超过 baseline，则自动追加：

- `cel_adapter_lastlayer_v2_qwen3_1.7b`

## 自动 follow-up 设计

如果主 loop（baseline + round1 + `adapter_v2`）结束后仍未超过 baseline，则继续：

1. `selector-only` 三方法：
   - `cel_mlp_lastlayer_v2_selector_only_qwen3_1.7b`
   - `cel_adapter_lastlayer_v3_selector_only_qwen3_1.7b`
   - `cel_task_conditioned_lastlayer_v2_selector_only_qwen3_1.7b`
2. 若仍未超过 baseline，则读取最佳模型的 `CEL Diagnostics`：
   - `gate_abs_mean` 偏大：跑 `cel_adapter_lastlayer_v4_selector_only_lowgamma_qwen3_1.7b`
   - 否则：跑 `cel_adapter_lastlayer_v5_selector_only_highgamma_qwen3_1.7b`

## 状态

- 2026-06-20：本地实现与远端 loop 脚本已创建，等待服务器正式运行结果。
- 2026-06-20：新增 `results/cel_stage2_lastlayer/STATUS.md`，用于汇总 baseline/CEL 当前阶段、已同步指标、最佳模型与最新 loop 事件。
- 2026-06-20：新增 `selector-only` 与 diagnostics-guided follow-up loop，避免主 loop 结束后实验停住。
