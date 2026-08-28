# STRUCTURE.md - 当前目录说明

## 1. 总体原则

当前仓库是 baseline + CEL A/B 工作区。A 模块已经完成正式训练、测试与审计；Stage 2 首轮旧单路径实验已经完成。当前 `h_r/h_m` 双路径预测一致性实现、tensor contracts 与审计链已完成，正在按固定 seed `1221` 执行三轮配置实验，不启动 Stage 3。

历史 checkpoint、candidate 名称和 audit key 可以保留原始编号用于溯源；方法正文统一使用 A 模块、B 模块和 `H/a/h_r/h_n/h_nb/h_m`。

## 2. 顶层目录

```text
dialogue-kt_migration_20260806/
|-- .claude/
|-- data/
|-- dialogue_kt/
|-- literature/
|-- results/
|-- saved_models/
|-- scripts/
|-- AGENTS.md
|-- README.md
`-- requirements.txt
```

- `.claude/`：当前状态、交接和操作规则。
- `data/annotated/`：当前训练和测试数据。
- `dialogue_kt/`：模型、数据和训练代码。
- `literature/`：Faith-DARE、DARE 等参考论文分析和代码。
- `results/`：baseline、A 模块、Stage 2 历史结果和方法设计。
- `saved_models/`：checkpoint 约定目录；当前本地为空。
- `scripts/`：训练、同步、测试、审计和报告脚本。

## 3. 当前核心代码

```text
dialogue_kt/
|-- main.py
|-- training.py
|-- cel_methods.py
|-- kt_data_loading.py
|-- prompting.py
`-- models/
```

- `cel_methods.py`
  - A selector、目标双路径 `a -> h_r/h_n/h_nb/h_m`、B contextual Transformer 和 hook。
  - 历史单路径 B 保留用于兼容和诊断，不作为当前正式定义。
- `training.py`
  - baseline/CEL 训练测试、五阶段 checkpoint 链、双路径 `p_r/p_m/p_n`、calibrator 和 diagnostics。
- `main.py`
  - CLI 参数。
- `kt_data_loading.py`, `prompting.py`
  - LLMKT 输入与 batch 构造。

## 4. 当前结果目录

```text
results/
|-- baseline/
|-- cel_stage1/
|-- cel_stage1_last_layer/
|-- cel_stage2_environment/
`-- method_design/
```

### `results/baseline/`

认证 baseline：Overall Acc/AUC `69.22 / 74.95`，Final Acc/AUC `61.17 / 75.32`。

### `results/cel_stage1_last_layer/`

保存 A 模块历史训练链、metrics、qual、logs、audit 和整体方法说明。

A 模块参考结果：Overall Acc/AUC `70.03 / 75.38`，Final Acc/AUC `66.80 / 77.26`。

历史 checkpoint 名称中的 candidate 编号只用于溯源，不再作为方法名称。

### `results/cel_stage2_environment/`

保存 Stage 2 首轮三个旧单路径 candidate 的 metrics、qual、logs、paired bootstrap 和 audit。这些结果属于历史 pilot，不是当前双路径目标方法的正式验证。

### `results/method_design/`

- `CEL_Modular_Flow.md`：唯一方法定义入口与设计原则。
- `CEL_Stage1_A_Module_Implementation.md`：A 模块实现与训练链。
- `CEL_Stage2_Environment_Generator_Implementation.md`：B 模块目标 forward、loss、训练和 controls。
- `CEL_Stage2_实施流程与思路.md`：A 接入后的整体流程、技术选择和三轮实验口径。

## 5. 当前脚本目录

- `scripts/cel_stage1/`：早期 Stage 1 脚本。
- `scripts/cel_stage1_last_layer/`：A 模块历史训练、同步、finalize 和 audit。
- `scripts/cel_stage2_environment/`：Stage 2 历史 launcher、audit、tensor contract、summary 和 paired bootstrap。

`scripts/cel_stage2_environment/` 已包含双路径 tensor contracts、真实 Qwen preflight、candidate launcher、validation/test evaluator 和 provenance/parameter audit；旧单路径 launcher 只用于历史结果追溯。

## 6. 新对话默认读取顺序

1. `.claude/STAGE2_HANDOFF.md`
2. `.claude/CLAUDE.md`
3. `.claude/MEMORY.md`
4. `.claude/STRUCTURE.md`
5. `results/method_design/CEL_Modular_Flow.md`
6. `results/method_design/CEL_Stage1_A_Module_Implementation.md`
7. `results/method_design/CEL_Stage2_Environment_Generator_Implementation.md`
8. `results/cel_stage2_environment/SUMMARY.md`

## 7. 路径与环境

- 本地根目录：`/home/luo/work/dialogue-kt_migration_20260806`
- 本地测试解释器：`/home/luo/miniconda3/envs/diagkt/bin/python`
- 正式训练：SSH alias `3090`，远端 `/home/user4/dialogue-kt`
- 最近记录的 A 模块远端刷新：`2026-08-12 23:17:24`

## 8. 编辑约束

- 保留工作树中的既有用户修改和删除。
- 不执行 destructive git 操作。
- 不修改历史 metrics、qual、checkpoint 名称或 audit 数据来伪造新方法结果。
- 更新生成式报告时同步修改其生成脚本，避免下次重建恢复旧表述。
