# Results 目录说明

当前 `results/` 主要保留五类内容：baseline 认证、历史 Stage 1 结果、已完成并审计通过的 A 模块记录、Stage 2 首轮旧单路径实验，以及当前 A+B 双路径方法设计文档。

当前 Stage 1 权威文件最后一次成功远端刷新时间是 `2026-08-12 23:17:24`。当前目录中的 `metrics / qual / kcs / logs / audit` 可以互相审计；模型 checkpoint 没有包含在本地 `results/` 中，仍保存在正式 SSH 训练环境。

新 Stage 2 对话的最小入口是 `.claude/STAGE2_HANDOFF.md`。历史 candidate stdout、metrics、qual、audit 和刷新记录保留用于追溯；方法正文统一使用 A 模块、B 模块和 `H/a/h_r/h_n/h_nb/h_m`，不再以历史候选编号命名方法。

## `baseline/`

- `metrics/`：当前 baseline 指标文件
- `kcs/`：baseline 的 KC 输出
- `qual/`：baseline 的逐轮预测
- `Baseline_DialogueKT_实验记录.md`：baseline 记录
- 当前认证结果：`lmkt_qwen3_1.7b_recert_20260620`，Overall Acc/AUC `69.22 / 74.95`

## `cel_stage1/`

- `metrics/`：较早一轮 Stage 1 的 `metrics_*.txt`
- `kcs/`：KC 级输出 `kcs_*.json`
- `qual/`：逐轮预测 `qual_*.csv`
- `CEL_Stage1_DialogueKT_实验记录.md`：较早一轮主实验记录
- `final_metrics_table.md`：三种 selector 的汇总表

## `cel_stage1_last_layer/`

- `metrics/`：当前应优先参考的 last-layer 指标
- `kcs/`：KC 级输出
- `qual/`：逐轮预测
- `CEL_Stage1_LastLayer_DialogueKT_实验记录.md`：当前 last-layer 设计与实验记录
- `STATUS.md`：当前 Stage 1 last-layer 所处状态、最佳模型、关键文件
- `COMPARISON.md`：baseline 与本轮 CEL 的自动对比表
- `DETAILED_ANALYSIS.md`：误差、偏置、final/non-final 分化分析
- `TASK_CONDITIONED_TUNING_LOOP.md`：当前 `task_conditioned` 调优目标、各轮结果与下一轮决策
- `step_logs/`：各子实验独立训练日志

该目录当前状态：

- 旧 `v26_fullv1` 结果已经撤回，相关 checkpoint 和结果记录已删除，不再作为 formal winner。
- A 模块已在 SSH 上从 Qwen3 原始基座独立训练并通过审计。
- A 模块参考工件的历史训练链：`Qwen3 base -> A bootstrap -> calibrator warmup -> A strict joint -> final test`；该链仅用于 provenance，不是后续多模块的默认协议。
- baseline 与历史 anchor 不重跑，也不作为 A 模块正式训练的初始化来源。
- A 模块 Overall Acc/AUC 为 `70.03 / 75.38`，Final Acc/AUC 为 `66.80 / 77.26`；相对 baseline 四项 delta 为 `+0.81 / +0.43 / +5.63 / +1.94`。
- 历史 checkpoint 文件名 `cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b` 仅用于工件和 audit 追溯。
- `v15 / v20` 等后训练校准或后处理结果只属于诊断口径。

## `method_design/`

- `CEL_Modular_Flow.md`：统一术语、A+B forward、整体损失与方法设计原则。
- `CEL_Stage1_A_Module_Implementation.md`：A 模块加性证据增强、训练链与 Stage 2 接口。
- `CEL_Stage2_Environment_Generator_Implementation.md`：`h_r/h_m` 双路径、B 模块、训练目标和控制实验。
- `CEL_Stage2_实施流程与思路.md`：A 接入后的整体流程、统一端到端训练规范、历史五阶段链和三轮实验口径。

## `cel_stage2_environment/`

- `README.md`：Stage 2 实现契约、首轮结果与解释边界
- `SUMMARY.md`：三方向 audit 汇总和下一轮建议
- `PAIRED_BOOTSTRAP.md/json`：按 dialogue cluster 的配对不确定性分析
- `metrics/`, `qual/`, `kcs/`：三方向 final test 输出
- `<candidate>/AUDIT.md/json`：provenance、冻结状态、参数变化、dtype 和 diagnostics 审计
- `stages/` 与 stdout：bootstrap、calibrator warmup、strict joint 的可追溯日志

首轮三个旧 candidate 均从 Qwen3-1.7B 原始基座独立初始化并通过 audit，但采用旧单路径 environment residual 流程，只保留为历史 pilot。当前双路径 `a -> h_r/h_n/h_nb/h_m`、共享后续网络预测和 `L_r + L_m + L_cons` 已完成；后续正式训练默认从第一个 epoch 起将所有模块（可选 calibrator 在内）放入同一 optimizer、完整损失和统一 train/validation 循环，并用固定 `max_epochs`、`patience`、`min_delta` 早停。冻结、warmup、交替训练、calibrator-only 和按 batch ramp 仅在明确声明的特殊协议中使用。证据比例与 B 输出尺度先作为诊断记录，是否加入额外约束由正式训练现象决定。

训练 CLI 已实现 `--max_epochs`、`--patience` 和 `--min_delta` 早停参数，`--epochs` 仅作兼容别名；现有日志和 checkpoint 仍按其历史固定 epoch/分阶段协议归档。
