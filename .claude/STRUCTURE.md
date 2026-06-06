# STRUCTURE.md — 项目结构规范

## 目录组织原则

项目按论文方法名进行命名空间隔离：通用代码保留在顶层 `dialogue_kt/`，方法特有脚本、结果、模型按 `<paper_name>` 分目录管理。

`STRUCTURE.md` 只记录**文件怎么放**；研究状态和实验结论见 `.claude/MEMORY.md`，详细实验记录见 `results/<paper_name>/`。

## 标准目录结构

```
dialogue-kt/
├── literature/                         # 论文PDF，按主题分类
│   ├── dialogue-kt.pdf                 # 框架论文
│   ├── evidence_rationale_extraction/
│   ├── information_bottleneck/
│   └── representation_information_compression/
├── dialogue_kt/                        # 核心代码包
│   ├── main.py                         # CLI入口
│   ├── training.py                     # 训练/评估核心
│   ├── kt_data_loading.py              # 数据集与collator
│   ├── prompting.py                    # Prompt模板
│   ├── selfelicit.py                   # SELFELICIT模块
│   ├── bayesian_epistemology.py        # Bayesian Epistemology模块
│   ├── informativeness_search.py       # Informativeness Search模块
│   ├── models/                         # 模型定义/加载
│   └── utils.py                        # 通用工具
├── scripts/                            # 方法特定分析/评估脚本
│   ├── <paper_name>/
│   └── honest_ensemble/                # validation-selected ensemble评估脚本
├── saved_models/                       # 模型checkpoint
│   └── <paper_name>/
├── results/                            # 实验结果
│   └── <paper_name>/
│       ├── metrics/                    # metrics_*.txt
│       ├── kcs/                        # kcs_*.json
│       ├── qual/                       # qual_*.csv
│       └── <PaperName>_DialogueKT_实验记录.md
│   └── honest_ensemble/                # ensemble结果、validation预测与实验记录
├── data/annotated/                     # 标注数据与缓存
├── .claude/
│   ├── CLAUDE.md                       # 操作指南
│   ├── MEMORY.md                       # 研究状态/结论索引
│   └── STRUCTURE.md                    # 本文件
└── requirements.txt
```

## 新论文方法添加步骤

1. 将论文 PDF 放入 `literature/<category>/`。
2. 在 `dialogue_kt/` 新增方法模块，或在已有通用模块中通过参数开关接入。
3. 在 `main.py` 添加必要 CLI 参数。
4. 在 `scripts/<paper_name>/` 放置分析、预计算、ensemble 或诊断脚本。
5. 在 `results/<paper_name>/` 创建：
   - `metrics/`
   - `kcs/`
   - `qual/`
   - `<PaperName>_DialogueKT_实验记录.md`
6. 训练 checkpoint 保存到 `saved_models/<paper_name>/`，或在模型名中包含方法前缀。
7. 更新 `.claude/MEMORY.md` 中的论文状态表、方法摘要和结果总表。
8. 如目录规范发生变化，再更新本文件。

## 命名规范

| 类型 | 规范 | 示例 |
|---|---|---|
| 方法目录 | 小写 snake_case | `bayesian_epistemology`, `informativeness_search` |
| 代码模块 | 小写 snake_case | `dialogue_kt/informativeness_search.py` |
| 模型名 | 方法名前缀 + 模型 | `sip_novelty_qwen3_1.7b` |
| metrics | `metrics_<model_name>.txt` | `metrics_sip_novelty_qwen3_1.7b.txt` |
| KC预测 | `kcs_<model_name>.json` | `kcs_sip_novelty_qwen3_1.7b.json` |
| 逐轮预测 | `qual_<model_name>.csv` | `qual_sip_novelty_qwen3_1.7b.csv` |
| 实验记录 | `<PaperName>_DialogueKT_实验记录.md` | `InformativenessSearch_DialogueKT_实验记录.md` |

## 原则

- **模块化**：方法特定代码优先独立成文件，公共训练/数据逻辑通过参数开关调用。
- **可复现**：每个方法必须保留训练命令、关键参数、metrics、qual、kcs 和实验记录。
- **不覆盖**：新实验使用新 `model_name`，不要覆盖已有结果或 checkpoint。
- **文档分工**：
  - `.claude/CLAUDE.md`：怎么操作项目。
  - `.claude/MEMORY.md`：当前研究状态和关键结论。
  - `.claude/STRUCTURE.md`：目录和命名规范。
  - `results/<paper_name>/`：完整实验细节。
