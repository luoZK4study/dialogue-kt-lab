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

## 状态

- 2026-06-20：本地实现与远端 loop 脚本已创建，等待服务器正式运行结果。
