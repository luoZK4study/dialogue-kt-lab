# STRUCTURE.md — 当前目录说明

## 1. 总体原则

当前仓库不是“多方法并行试验场”，而是 **baseline + CEL** 工作区。当前已经完成的是 **Stage 1 last-layer**，但当前优先推进的是 **task_conditioned tuning loop**，目标是先把 Overall AUC 与 Overall Acc 同时拉过 baseline。

## 2. 顶层目录说明

```text
dialogue-kt/
├── .claude/
├── data/
├── dialogue_kt/
├── literature/
├── results/
├── saved_models/
├── scripts/
├── AGENTS.md
├── README.md
└── requirements.txt
```

各目录作用：

- `.claude/`
  - 持久化项目状态与操作规则。
  - 新对话优先读取：`CLAUDE.md`、`MEMORY.md`、`STRUCTURE.md`。
- `data/`
  - `annotated/`：当前训练/测试使用的标注数据。
  - `src/`：原始数据源。
- `dialogue_kt/`
  - 核心 Python 包。
  - 当前重点文件：`main.py`、`training.py`、`kt_data_loading.py`、`prompting.py`、`cel_methods.py`。
  - `models/` 保留 baseline 模型与 LLM loader。
- `literature/`
  - 论文分析、外部参考代码、历史阅读材料。
- `results/`
  - `baseline/`：当前 baseline 认证目录。
  - `cel_stage1/`：较早一轮 Stage 1 A 模块结果。
  - `cel_stage1_last_layer/`：当前应优先参考的 Stage 1 last-layer 结果目录。
  - `method_design/`：CEL 设计文档与后续 Stage 2 实施方案。
- `saved_models/`
  - 训练 checkpoint 存放位置。
- `scripts/`
  - `scripts/cel_stage1/`：历史 Stage 1 脚本。
  - `scripts/cel_stage1_last_layer/`：当前活跃脚本。

## 3. 当前活跃代码结构

```text
dialogue_kt/
├── annotate.py
├── cel_methods.py
├── data_loading.py
├── human_eval.py
├── kt_data_loading.py
├── main.py
├── prompting.py
├── training.py
├── utils.py
├── visualize.py
└── models/
    ├── dkt_multi_kc.py
    ├── dkt_sem.py
    ├── lm.py
    └── simplekt.py
```

说明：

- `cel_methods.py`：CEL selector、hook 注入逻辑，以及后续 Stage 2 B 模块的首选扩展位置。
- `training.py`：baseline 与 CEL 的训练/测试主流程。
- `kt_data_loading.py`：LMKT baseline/CEL 所需 dataset 与 collator。
- `data_loading.py`：数据集切分、KC 字典、结果路径。
- `prompting.py`：annotation prompt、KT prompt、MathDial context 拼接。

## 4. 当前活跃结果结构

```text
results/
├── baseline/
│   └── Baseline_DialogueKT_实验记录.md
├── cel_stage1/
│   ├── metrics/
│   ├── kcs/
│   ├── qual/
│   ├── CEL_Stage1_DialogueKT_实验记录.md
│   └── final_metrics_table.md
├── cel_stage1_last_layer/
│   ├── metrics/
│   ├── kcs/
│   ├── qual/
│   ├── step_logs/
│   ├── STATUS.md
│   ├── COMPARISON.md
│   ├── DETAILED_ANALYSIS.md
│   ├── TASK_CONDITIONED_TUNING_LOOP.md
│   └── CEL_Stage1_LastLayer_DialogueKT_实验记录.md
└── method_design/
    ├── CEL_Modular_Flow.md
    ├── CEL_Stage1_A_Module_Implementation.md
    └── CEL_Stage2_Environment_Generator_Implementation.md
```

## 5. 命名约定

- 当前 baseline 目录：`results/baseline/`
- 当前 last-layer 目录：`results/cel_stage1_last_layer/`
- 未来 Stage 2 环境生成器目录：`results/cel_stage2_environment/`
- 当前 Stage 1 最优模型名保留历史产物名：
  - `cel_task_conditioned_lastlayer_v1_qwen3_1.7b`
- metrics：`results/<result_subdir>/metrics/metrics_<model_name>.txt`
- KC 输出：`results/<result_subdir>/kcs/kcs_<model_name>.json`
- qual：`results/<result_subdir>/qual/qual_<model_name>.csv`

## 6. 新对话默认动作

如果未来打开新对话并需要继续工作，默认按这个顺序：

1. 看 `.claude/MEMORY.md`
2. 看 `results/cel_stage1_last_layer/STATUS.md`
3. 看 `results/cel_stage1_last_layer/TASK_CONDITIONED_TUNING_LOOP.md`
4. 看 `git status --short`
5. 看 `results/baseline/Baseline_DialogueKT_实验记录.md`
6. 看 `results/cel_stage1_last_layer/CEL_Stage1_LastLayer_DialogueKT_实验记录.md`
7. 检查 SSH 上当前训练状态与结果是否已同步
8. 若当前 tuning 目标尚未达成，再决定下一轮 `task_conditioned` 要改的代码入口，优先是 `dialogue_kt/training.py` 与 `dialogue_kt/cel_methods.py`

除非明确要回看历史，否则不要再把 `cel_stage1_last_layer/` 误写成 `stage2 lastlayer`。
