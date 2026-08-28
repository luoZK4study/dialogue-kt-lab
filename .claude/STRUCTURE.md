# 当前目录说明

```text
dialogue-kt_migration_20260806/
|-- .claude/                         # 持久化规范与研究状态
|-- data/                            # 数据，不参与本次代码备份
|-- dialogue_kt/                     # 主 Python 包
|-- literature/                      # 文献，不参与本次代码备份
|-- results/
|   |-- baseline/                    # 认证 baseline
|   |-- a/                           # A 模块重训输出
|   |-- a_b/                         # A+B 模块重训输出
|   |-- method_design/               # A/B 及未来模块设计文档
|   `-- README.md
|-- scripts/
|   `-- cel/                         # A/B 训练、同步和审计入口
|-- AGENTS.md
|-- README.md
`-- requirements.txt
```

## 核心代码

- `dialogue_kt/main.py`：训练、测试和诊断 CLI。
- `dialogue_kt/training.py`：统一 train/validation、validation-best checkpoint、A/B 路径和损失。
- `dialogue_kt/cel_methods.py`：A selector、`h_r/h_n` 分解、B 变换和 hook。
- `dialogue_kt/kt_data_loading.py`、`dialogue_kt/prompting.py`：数据和 prompt 构造。

## 设计文档

- `results/method_design/CEL_Modular_Flow.md`：唯一的总体方法入口，包含适用于 A/B/C/D 的设计规范、表示契约和统一训练协议。
- `results/method_design/CEL_Stage1_A_Module_Implementation.md`：A 的实现与诊断。
- `results/method_design/CEL_Stage2_Environment_Generator_Implementation.md`：B 的实现、双路径目标和审计要求。

## 命名与结果

方法正文统一称为 A 模块、B 模块、C 模块……；不要用旧 candidate 编号代替模块名称。历史 checkpoint 和 audit key 可以原样保留在 provenance 中。新结果按 `results/a/`、`results/a_b/` 组织，训练脚本不得默认生成分阶段目录。

`data/`、`literature/` 和远端 SSH 配置是受保护范围。整理时只修改代码、规范、脚本和明确列出的结果目录。
