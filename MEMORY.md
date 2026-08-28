# MEMORY.md — 当前研究状态

> 当前工作树只保留 **baseline + CEL**。  
> 当前已完成的是 **Stage 1 last-layer**；但当前优先任务不是直接做 Stage 2，而是继续把 `task_conditioned` 调到 **AUC/Acc 同时超过 baseline**。  
> 如果开新对话，先确认 `results/cel_stage1_last_layer/TASK_CONDITIONED_TUNING_LOOP.md` 与 `STATUS.md` 是否最新，再按本文件的“默认动作”继续。

## 1. 当前阶段快照

- baseline 当前认证目录：`results/baseline/`
- baseline 模型：`lmkt_qwen3_1.7b_recert_20260620`
- baseline 指标：Overall Acc **69.22**, Overall AUC **74.95**, Final Turn AUC **75.32**
- 当前完成目录：`results/cel_stage1_last_layer/`
- 当前固定的 Stage 1 暂定方案：`cel_task_conditioned_lastlayer_v1_qwen3_1.7b`
- 该方案指标：Overall Acc **68.87**, Overall AUC **75.29**, Final Turn AUC **75.27**
- 相对当前 baseline 增益：**+0.34 Overall AUC**
- 当前优先任务：**task_conditioned tuning loop**
- 当前 tuning 目标：**Overall AUC > 74.95 且 Overall Acc > 69.22**

## 2. Stage 1 现有结论

历史 `results/cel_stage1/` 是较早的 Stage 1 A 模块初筛。

当前更应参考的是 `results/cel_stage1_last_layer/`，因为它是基于当前 baseline 重跑后的最后一层验证。三种首轮结果如下：

| Model | CEL Mode | Overall AUC | Final AUC | 结论 |
|---|---|---:|---:|---|
| `cel_mlp_lastlayer_v1_qwen3_1.7b` | `mlp` | 74.81 | 72.73 | 低于 baseline |
| `cel_adapter_lastlayer_v1_qwen3_1.7b` | `adapter` | 75.04 | 72.66 | 略高于 baseline |
| `cel_task_conditioned_lastlayer_v1_qwen3_1.7b` | `task_conditioned` | 75.29 | 75.27 | 当前最佳，固定为 Stage 1 暂定方案 |

当前判断：

- 最后一层表示上是有可用信号的，不是“完全无效”。
- `task_conditioned` 已证明当前 A 模块可以超过 baseline，因此后续 Stage 2 不应重新定义这轮实验。
- Stage 2 的变量应集中在 **B 模块 environment generator**，而不是继续把这轮 last-layer 试验叫成 Stage 2。

## 3. 固定的 Stage 1 暂定方案

固定脚本：`scripts/cel_stage1_last_layer/run_task_conditioned_v1.sh`

当前固定参数：

- `--pt_model_name lmkt_qwen3_1.7b_recert_20260620`
- `--model_name cel_task_conditioned_lastlayer_v1_qwen3_1.7b`
- `--result_subdir cel_stage1_last_layer`
- `--cel_mode task_conditioned`
- `--cel_layer_idx -1`
- `--cel_selector_hidden_dim 512`
- `--cel_gamma 0.30`
- `--cel_use_norm 0`

当前默认的其余 CEL 关键行为来自 CLI 默认值：

- `--cel_hook_site last_block`
- `--cel_injection_variant scalar_gate`
- `--cel_application_mode token_residual`
- `--cel_train_selector_only 0`

Stage 2 默认应以此方案为固定 A 模块起点。若要继续训练 Stage 2，优先考虑：

- `--pt_model_name lmkt_qwen3_1.7b_recert_20260620`
- `--cel_selector_init_model_name cel_task_conditioned_lastlayer_v1_qwen3_1.7b`

原因：这样可以保留当前已验证的 selector 权重，同时避免把 Stage 2 和重新搜索 Stage 1 混在一起。

## 4. 当前要做什么

当前默认任务不是继续命名整理，也不是直接实现 Stage 2，而是：

1. 把 `task_conditioned last-layer v1` 视为固定的 Stage 1 暂定方案。
2. 继续围绕它做 `task_conditioned` 同族调优，目标是 **AUC/Acc 同时超过 baseline**。
3. 所有训练与验证都必须在 SSH 服务器完成。
4. 每轮训练结束后必须同步远端结果，并更新本地分析记录。
5. 只有当 `task_conditioned` 调优目标达成后，才继续 Stage 2。

当前已启动的 Round 1：

- `cel_task_conditioned_lastlayer_v11_cal_bias_only_qwen3_1.7b`
- `cel_task_conditioned_lastlayer_v12_cal_affine_only_qwen3_1.7b`

若 Round 1 未达标，已准备的下一轮最小改动候选：

- `cel_task_conditioned_lastlayer_v13_selector_cal_bias_qwen3_1.7b`
- `cel_task_conditioned_lastlayer_v14_selector_cal_affine_qwen3_1.7b`

推荐的 Stage 2 结果目录命名：

- `results/cel_stage2_environment/`
- `scripts/cel_stage2_environment/`
- `results/cel_stage2_environment/CEL_Stage2_Environment_Generator_DialogueKT_实验记录.md`

## 5. 新对话默认动作

如果以后打开新对话，默认按下面顺序做：

1. 先读 `results/cel_stage1_last_layer/STATUS.md`
2. 再读 `results/baseline/Baseline_DialogueKT_实验记录.md`
3. 再读 `results/cel_stage1_last_layer/CEL_Stage1_LastLayer_DialogueKT_实验记录.md`
4. 再读 `results/cel_stage1_last_layer/TASK_CONDITIONED_TUNING_LOOP.md`
5. 如果要看误差结构，再读 `results/cel_stage1_last_layer/DETAILED_ANALYSIS.md`
6. 看 `git status --short`
7. 检查 SSH 上当前训练状态
8. 然后继续当前 tuning loop，而不是默认切去 Stage 2

如果 `STATUS.md` 落后，执行：

```bash
bash scripts/cel_stage1_last_layer/sync_results_from_server.sh
python scripts/cel_stage1_last_layer/generate_status_summary.py
python scripts/cel_stage1_last_layer/generate_comparison_report.py
python scripts/cel_stage1_last_layer/generate_detailed_analysis.py
python scripts/cel_stage1_last_layer/update_baseline_record.py
python scripts/cel_stage1_last_layer/update_stage1_last_layer_record.py
python scripts/cel_stage1_last_layer/update_task_conditioned_tuning_loop.py
```

## 6. 关键文件

- 操作说明：`.claude/CLAUDE.md`
- 当前状态：`.claude/MEMORY.md`
- 目录说明：`.claude/STRUCTURE.md`
- baseline 记录：`results/baseline/Baseline_DialogueKT_实验记录.md`
- 当前 Stage 1 last-layer 记录：`results/cel_stage1_last_layer/CEL_Stage1_LastLayer_DialogueKT_实验记录.md`
- 当前 Stage 1 状态摘要：`results/cel_stage1_last_layer/STATUS.md`
- 当前 task_conditioned 调优闭环：`results/cel_stage1_last_layer/TASK_CONDITIONED_TUNING_LOOP.md`
- 当前 Stage 1 对比摘要：`results/cel_stage1_last_layer/COMPARISON.md`
- 当前 Stage 1 细粒度分析：`results/cel_stage1_last_layer/DETAILED_ANALYSIS.md`
- Stage 2 设计方案：`results/method_design/CEL_Stage2_Environment_Generator_Implementation.md`

## 7. 已确认经验

- Qwen3-1.7B baseline 在当前代码下依然很强，Stage 2 不能假定“只要多加模块就会涨分”。
- 当前最直接的短板是 `task_conditioned v1` 的 calibration 偏保守，而不是 ranking 完全失败。
- 因此当前最优先路径是先完成 `task_conditioned` 调优闭环，而不是跳去实现 Stage 2。
- 当前最稳妥的推进方式，是固定 `task_conditioned v1`，把新增变量限制在 B 模块。
- Stage 2 若一开始同时改 selector、hook 位点、environment 生成器和损失项，实验不可解释。
- 当前目录命名已统一为 `baseline` 与 `cel_stage1_last_layer`；以后不要再把当前 last-layer 实验写成 `stage2 lastlayer`。
