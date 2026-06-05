# MEMORY.md — 项目总览

> dialogue-kt 作为知识追踪实验框架，与多篇论文方法结合。  
> 本文件为总览索引 + 标准工作流，**详细记录、具体实现、完整结论**保存在 `results/<paper_name>/` 目录下。

---

## 〇、标准工作流

### 0. 论文准入检查
```
0.1 从 literature/ 选取一篇未处理的论文PDF
0.2 检查论文是否给出代码链接
    → 有 → 进入阶段1
    → 无 → 搜索 GitHub 是否有同名/同作者仓库
        → 有 → 进入阶段1
        → 无 → 跳过，在下方"论文状态表"标注"无代码，已跳过"
0.3 检查该论文是否已在"论文状态表"中标记为"已验证"
    → 是 → 跳过，取下一篇
    → 否 → 继续
```

### 1. 论文理解
```
1.1 使用 PDF 工具提取论文文本（find-skills 或 pymupdf）
1.2 阅读并记录：核心方法（1-2句）、技术细节、代码仓库URL
1.3 Clone 代码仓库，阅读关键文件理解实现
1.4 分析：论文方法中哪些思想/组件可与 dialogue-kt 结合
```

### 2. 方法设计
```
2.1 提出 ≥3 种结合方案，每种描述核心思想、改动范围、预期效果
2.2 评估可行性，确定实施优先级
```

### 3. 代码实现
```
3.1 在 dialogue_kt/ 中实现核心模块（新增文件模块化，修改文件通过参数开关）
3.2 在 main.py 添加新参数
3.3 在 scripts/<paper_name>/ 存放分析/评估脚本
```

### 4. 训练验证
```
4.1 rsync 代码到服务器
4.2 运行基线训练（若无已有基线）
4.3 每种方法训练 2 epochs（Epoch 2 过拟合则停，否则最多3）
4.4 记录 Train Loss、Val Loss、过拟合判断
4.5 收集 Overall AUC、Final Turn AUC
4.6 若方法失败，分析原因并记录
```

### 5. 结果整理
```
5.1 建立 results/<paper_name>/ 结构：
    ├── metrics/   # *.txt
    ├── kcs/       # *.json
    ├── qual/      # *.csv
    └── <PaperName>_实验记录.md
5.2 实验记录MD包含：论文简介、方法描述、结果表、代码变更、关键发现
5.3 模型保存到 saved_models/<paper_name>/
```

### 6. 文档更新
```
6.1 更新本文件"论文状态表"和"已结合方法"
6.2 更新 .claude/CLAUDE.md "已有结合"列表
6.3 更新 .claude/STRUCTURE.md 如新增目录
6.4 清理 results/ 冗余文件
```

### 7. 交叉结合检查
```
7.1 检查当前方法是否可与之前已验证的方法组合
    → 两个方法作用于不同环节（如改prompt + 改loss）
    → 两个方法输出/输入可互补
7.2 若有可行组合 → 回到阶段2，命名为"<PaperA>+<PaperB>"
7.3 若无 → 回到阶段0，取下一篇论文
```

---

## 一、论文状态表

| 论文 | 分类目录 | 代码 | 状态 |
|------|---------|:--:|:----:|
| Liu_2025_SelfElicit | evidence_rationale_extraction | ✅ GitHub | ✅ 已验证 |
| Wang_2025_Stepwise_Informativeness_Search | evidence_rationale_extraction | — | 🔄 进行中 |
| Kim_2025_From_Evidence_to_Belief | evidence_rationale_extraction | — | 🔄 进行中 |
| Wu_2025_Expert_Heads | evidence_rationale_extraction | — | 📋 待处理 |
| Vasu_2025_HypER | evidence_rationale_extraction | — | 📋 待处理 |
| Evidence-R1_2025 | evidence_rationale_extraction | — | 📋 待处理 |
| Bian_2025_IBCircuit | information_bottleneck | — | 📋 待处理 |
| Conklin_2026_Learning_is_Forgetting | information_bottleneck | — | 📋 待处理 |
| Oh_2025_Vittle | information_bottleneck | — | 📋 待处理 |
| Wang_2025_QUITO-X | information_bottleneck | — | 📋 待处理 |
| Dai_2025_Pretraining_Context_Compressor | representation_information_compression | — | 📋 待处理 |
| Kim_2025_KVzip | representation_information_compression | — | 📋 待处理 |
| Li_2025_500xCompressor | representation_information_compression | — | 📋 待处理 |
| Zhang_2026_LLM2Comp | representation_information_compression | — | 📋 待处理 |
| Zhao_2025_DAC | representation_information_compression | — | 📋 待处理 |

> 状态说明：📋待处理 → 🔄进行中 → ✅已验证 → ⏭️已跳过(无代码)

---

## 二、框架概述

**核心任务**: 在导师-学生对话中预测学生对知识点（KC）的掌握程度（True/False 二分类）。

**基座模型**: Qwen3-1.7B + LoRA (r=16, alpha=16)

**数据集**: MathDial (ATC标准标注, 149个KC, 1802训练/451验证/595测试对话)

**评估指标**: Overall AUC（排除每对话第一个标签）、Final Turn AUC

**论文基线**: LLMKT (Llama-3.1-8B) MathDial Overall AUC = **76.71%** (LAK 2025 Table 2)

---

## 三、已结合的论文方法

### 2.1 SELFELICIT (ACL 2025)

- **论文**: Liu et al., "SelfElicit: Your Language Model Secretly Knows Where is the Relevant Evidence", ACL 2025
- **核心思想**: 利用模型深层 Transformer 注意力分数识别上下文中的证据句子，标记后重新推理
- **结合方式**: 将对话轮次视为"句子"，提取注意力分数→轮次级证据分数→标记高分数轮次→训练/推理时引导模型关注证据轮次
- **提出方法**: 7种（5种启发式证据标记 + 1种真实注意力证据 + 1种集成）
- **最佳结果**: 等权集成 AUC **76.24%**（超越1.7B基线75.99%，论文8B为76.71%）
- **关键发现**: 启发式证据标记（最近N轮）意外优于真实注意力；小模型注意力分数极均匀导致阈值选择困难

> 📁 **详细记录**: `results/selfelicit/SELFELICIT_DialogueKT_实验完整记录.md`  
> 📁 **代码**: `dialogue_kt/selfelicit.py`, `scripts/selfelicit/`  
> 📁 **模型**: `saved_models/selfelicit/`

### 3.2 Bayesian Epistemology 系列 (进行中)

- **论文**: Kim_2025_From_Evidence_to_Belief 等
- **状态**: 🔄 进行中
- **详细记录**: `results/bayesian_epistemology/`

### 3.3 Informativeness Search 系列 (进行中)

- **论文**: Wang_2025_Stepwise_Informativeness_Search
- **状态**: 🔄 进行中
- **详细记录**: `results/informativeness_search/`

---

## 四、各模型训练记录

| 模型 | 参数 | 基座 | Overall AUC | 备注 |
|------|:---:|------|:---:|------|
| 论文 LLMKT | 8B | Llama-3.1-8B | **76.71%** | 论文基准 |
| 基线 | 1.7B | Qwen3-1.7B | 75.99% | Epoch 1 即最佳 |
| SE-Adaptive | 1.7B | Qwen3-1.7B | 75.79% | 最佳单一SE方法 |
| SE-Combined | 1.7B | Qwen3-1.7B | 75.32% | 最佳Final Turn (76.66%) |
| 集成 (等权6模型) | 1.7B×6 | Qwen3-1.7B | 76.24% | 最佳集成 |

---

## 五、项目文件组织规范

**核心原则**: 按论文方法名进行命名空间隔离。通用代码在顶层，方法特有代码独立目录。

```
dialogue-kt/
├── literature/                      # 论文PDF（分类子目录）
├── dialogue_kt/                     # 核心代码包（通用）
│   ├── models/lm.py                 # LLM基座加载
│   ├── training.py                  # 训练/评估
│   └── ...                          # 其他通用模块
├── scripts/<paper_name>/            # 分析脚本，按论文隔离
├── saved_models/<paper_name>/       # 模型检查点
├── results/<paper_name>/            # 实验结果
│   ├── metrics/                     # .txt 评测指标
│   ├── kcs/                         # .json KC预测
│   ├── qual/                        # .csv 逐轮预测
│   └── <PaperName>_实验记录.md      # 完整实验记录
└── .claude/STRUCTURE.md             # 结构说明
```

### 添加新论文方法的标准步骤
1. `literature/` 放入论文PDF
2. 代码模块（如需要）加入 `dialogue_kt/` 或子模块
3. `scripts/<paper>/` 放入分析脚本
4. `results/<paper>/` 创建 `{metrics,kcs,qual}/` + 实验记录MD
5. `saved_models/<paper>/` 保存模型
6. 更新本文件第三节

---

## 六、实践经验

### 模型下载：ModelScope > hf-mirror
```python
from modelscope import snapshot_download
snapshot_download('qwen/Qwen3-1.7B', cache_dir='/home/user4/.cache/huggingface/hub/')
```

### `--quantize` 参数陷阱
`bool_type(x) = x != "0"`：`--quantize 0` → False ✅；`--quantize False` → True ❌

### 训练规律
1. **1.7B 是最佳性价比**: AUC 76% 接近论文 8B 的 76.7%，速度快 3.5 倍
2. **Epoch 1 即最佳**: 之后快速过拟合，考虑降低 lr 或增大 dropout
3. **debug 模式 1 epoch 足够判断方向**
4. 所有方法使用 AdamW, wd=1e-2, lr=2e-4, r=16, lora_alpha=16

### SELFELICIT 实验教训
1. 真实注意力需要 `attn_implementation="eager"`，sdpa不输出attention
2. 集成时不能在测试集上调权重（数据泄露），等权平均为诚实数字
3. 小模型注意力分数极均匀(0.002-0.008)，启发式有时比注意力更有效
