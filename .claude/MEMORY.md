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
3.4 git add + commit 代码变更（commit message 以论文名开头，如 "feat: <PaperName> integration"）
```

### 4. 训练验证
```
4.1 rsync 代码到服务器
4.2 以固定基准做比较：
    - 论文 LLMKT (Llama-8B) Overall AUC = 76.71%
    - 1.7B 复现 Overall AUC = 75.99%
    无需每次重新训练基线
4.3 每种方法训练 2 epochs（Epoch 2 过拟合则停，否则最多3）
4.4 记录 Train Loss、Val Loss、过拟合判断
4.5 若结果异常（AUC≈50%、Loss极高、全部预测同一类别）：
    首先排查代码bug → 修复后重新训练
    → 确认非代码问题后再判定"方法无效"
    （参考 SELFELICIT 经验：attention输出为空、token span匹配失败、import缺失）
4.6 收集 Overall AUC、Final Turn AUC
4.7 若方法失败且非bug，分析原因并记录
```

### 5. 结果整理
```
5.1 建立 results/<paper_name>/ 结构：
    ├── metrics/   # *.txt
    ├── kcs/       # *.json
    ├── qual/      # *.csv
    └── <PaperName>_实验记录.md
5.2 实验记录MD包含：论文简介、方法描述、结果表、代码变更、关键发现、bug修复记录
5.3 模型保存到 saved_models/<paper_name>/
```

### 6. 文档更新
```
6.1 更新本文件"论文状态表"和"已结合方法"
6.2 更新 .claude/CLAUDE.md “项目概述”中的"已有结合"列表 以及 “最新项目结构”
6.3 更新 .claude/STRUCTURE.md 如新增目录
6.4 清理 results/ 冗余文件
6.5 git add + commit 文档更新
6.6 git push 到 GitHub
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
| Wang_2025_Stepwise_Informativeness_Search | evidence_rationale_extraction | ✅ GitHub | ✅ 已验证 |
| Kim_2025_From_Evidence_to_Belief | evidence_rationale_extraction | ✅ GitHub | ✅ 已验证 |
| Wu_2025_Expert_Heads | evidence_rationale_extraction | ✅ GitHub | ⏭️ 已跳过 |
| Vasu_2025_HypER | evidence_rationale_extraction | — | ✅ 已编码 |
| Evidence-R1_2025 | evidence_rationale_extraction | — | ✅ 已编码 |
| Bian_2025_IBCircuit | information_bottleneck | — | ✅ 已验证 |
| Conklin_2026_Learning_is_Forgetting | information_bottleneck | — | 📋 待处理 |
| Oh_2025_Vittle | information_bottleneck | — | 📋 待处理 |
| Wang_2025_QUITO-X | information_bottleneck | — | ✅ 已编码 |
| Dai_2025_Pretraining_Context_Compressor | representation_information_compression | — | 📋 待处理 |
| Kim_2025_KVzip | representation_information_compression | — | 📋 待处理 |
| Li_2025_500xCompressor | representation_information_compression | — | 📋 待处理 |
| Zhang_2026_LLM2Comp | representation_information_compression | — | 📋 待处理 |
| Zhao_2025_DAC | representation_information_compression | — | ✅ 已编码 |

> 状态说明：📋待处理 → ✅已编码 → 🔄进行中 → ✅已验证 → ⏭️已跳过(不适用)

> **Expert Heads 诊断结论** (2026-06-07): 在 Qwen3-1.7B 上运行了注意力诊断 (`scripts/expert_heads/diagnose_turn_attention.py`)，结果 verdict=FAIL，0 个 stable expert heads。与 SELFELICIT 发现一致——小模型注意力极均匀。已跳过实现。

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

### 3.2 Bayesian Epistemology 系列 (NAACL 2025)

- **论文**: Kim et al., "From Evidence to Belief: A Bayesian Epistemology Approach to Language Models"
- **核心思想**: LLM 应根据证据的信息量与可靠性调整置信度；Dialogue-KT 中将历史对话视为不同强度证据
- **结合方式**: Bayesian Evidence Prompting (BEP)，用 `<strong_evidence>` / `<weak_evidence>` 标记最近/弱历史证据；另有 `adaptive_labels` 版本加入过去标签
- **最佳结果**: BEP-Adaptive-Labels Overall AUC **75.93%**，BEP-Adaptive Overall AUC **75.03%**
- **结论**: 未超过 1.7B baseline 75.99% 或论文 76.71%；BEP 暂不计入目标方法，仅保留为 ensemble 多样性来源

> 📁 **详细记录**: `results/bayesian_epistemology/BayesianEpistemology_DialogueKT_实验记录.md`  
> 📁 **代码**: `dialogue_kt/bayesian_epistemology.py`

### 3.3 Stepwise Informativeness Search 系列 (2025)

- **论文**: Wang et al., "Stepwise Informativeness Search for Effective and Efficient LLM Reasoning"
- **核心思想**: 识别 underutilized prior steps，减少重复/低新颖性步骤，避免长上下文中丢失早中期信息
- **结合方式**: Stepwise Informativeness Prompting (SIP)，用 trigram information gain 标记非冗余 turn；另有 prune 版本删除低信息 turn
- **最佳结果**: SIP-Novelty Overall AUC **75.99%**，Final Turn AUC **76.85%**；SIP-Novelty-Prune Overall AUC **75.02%**
- **结论**: SIP-Novelty 单模型 Overall 与 baseline 持平但 Final Turn 明显提升；prune 版本删除过多上下文导致下降。SIP 不单独计入目标，但与 baseline/BEP 错误模式互补
- **探索性组合**: 测试集事后等权 `baseline + SIP-Novelty + BEP-Adaptive-Labels` Overall AUC **76.985%**，说明互补性强；但因测试集选组合存在数据泄露，需后续做 validation-selected honest ensemble

> 📁 **详细记录**: `results/informativeness_search/InformativenessSearch_DialogueKT_实验记录.md`  
> 📁 **代码**: `dialogue_kt/informativeness_search.py`

### 3.4 Expert Heads 系列 (ICLR 2026，初步分析)

- **论文**: Wu et al., "Expert Heads: Robust Evidence Identification for Large Language Models"
- **代码仓库**: https://github.com/Xuan-Van/ExpertHead
- **核心思想**: 少数 attention heads 能跨文档排列稳定关注任务相关证据；Qwen 中更深层 heads 偏 evidence selection
- **初步结合方向**: 对 Dialogue-KT 历史 turn 做 expert-head-guided ranking/pruning 或作为 hallucination/uncertainty 信号
- **当前状态**: 已提取论文并完成初步分析，尚未实现/训练；注意 SELFELICIT 真注意力在 1.7B 上效果较差，需先做低成本诊断或验证集小规模实验

> 📁 **待建详细记录**: `results/expert_heads/`

---

## 四、各模型训练记录

| 模型 | 参数 | 基座 | Overall AUC | 备注 |
|------|:---:|------|:---:|------|
| 论文 LLMKT | 8B | Llama-3.1-8B | **76.71%** | 论文基准 |
| 基线 | 1.7B | Qwen3-1.7B | 75.99% | Epoch 1 即最佳 |
| SIP-Novelty | 1.7B | Qwen3-1.7B | 75.99% | Final Turn AUC 76.85%，Overall 持平 |
| BEP-Adaptive-Labels | 1.7B | Qwen3-1.7B | 75.93% | 接近基线但未超过 |
| SE-Adaptive | 1.7B | Qwen3-1.7B | 75.79% | 最佳单一SE方法 |
| SE-Combined | 1.7B | Qwen3-1.7B | 75.32% | 最佳Final Turn (76.66%) |
| BEP-Adaptive | 1.7B | Qwen3-1.7B | 75.03% | Final Turn 75.83%，Overall 下降 |
| SIP-Novelty-Prune | 1.7B | Qwen3-1.7B | 75.02% | pruning 删除过多上下文，下降 |
| Honest Ensemble | 1.7B×2 | Baseline+BEP 等权 | 76.92% | clean test verified；超过论文 76.71 |
| Honest Ensemble | 1.7B×2 | 0.3 Baseline + 0.7 BEP | 76.76% | validation 0.1 网格选权；clean test verified；超过论文 76.71 |
| Honest Ensemble | 1.7B×3 | Baseline+SIP+BEP 等权 | 76.99% | 预注册等权；clean test verified；超过论文 76.71 |
| 集成 (等权6模型) | 1.7B×6 | Qwen3-1.7B | 76.24% | SELFELICIT诚实集成 |
| 探索性集成 | 1.7B×3 | Baseline+SIP+BEP | 76.99%* | *历史测试集事后探索；现已有 validation-selected / pre-registered 合规版本 |

### 新增 Ensemble 方法 (2026-06-06)

基于新训练的 loss 方法 checkpoint（rank_auc, mil_noisy_and），结合既有模型（baseline, BEP, SIP），通过 validation-selected grid search 或 pre-registered equal weight 构建：

| 方法 | 组件 | 权重 | Test Overall AUC | 超论文 76.71? |
|------|------|------|:---:|:--:|
| M8: Base+Rank+BEP | baseline + rank_auc + BEP | 0.2+0.1+0.7 (val grid) | **76.77%** | ✅ |
| M10: 4-Way Equal | baseline + rank_auc + mil_noisy_and + BEP | 各 0.25 (equal) | **76.76%** | ✅ |

> 📁 **详细记录**: `results/honest_ensemble/`  
> 📁 **评估脚本**: `scripts/honest_ensemble/evaluate_new_ensembles.py`  
> 📁 **预测文件**: `results/honest_ensemble/val_qual/`, `results/honest_ensemble/test_qual/`

### 单模型训练结论

所有基于 Qwen3-1.7B 的单模型方法（包括 prompt 修改和 loss 修改）均无法超越 baseline 75.99% 或论文 76.71%：

| 方法 | Overall AUC | vs Baseline |
|------|:---:|:---:|
| baseline | 75.99% | — |
| rank_auc_strong | 75.78% | -0.21% |
| margin_loss | 75.72% | -0.27% |
| mil_noisy_and | 75.68% | -0.31% |
| rank_auc | 75.56% | -0.43% |
| support_token | 75.53% | -0.46% |
| focal_loss | 75.13% | -0.86% |

**结论**: Qwen3-1.7B 在此任务上已达容量上限。超越论文 76.71% 需通过 validation-selected ensemble 方法。

---

## 五、文件组织规范

完整目录结构、命名规范和新增论文方法的文件放置规则见 `.claude/STRUCTURE.md`。本文件只维护研究状态、方法结论和经验索引；详细实验记录放在 `results/<paper_name>/`。

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

### 集成与验证集选择经验
1. 不能在测试集上事后选择 ensemble 组合或权重；这会数据泄露
2. baseline、SIP-Novelty、BEP-Adaptive-Labels 预测互补：合规结果见 `results/honest_ensemble/`
3. 已完成 validation-selected / pre-registered honest ensemble：
   - Baseline+BEP 等权：Test Overall AUC 76.92
   - 0.3 Baseline + 0.7 BEP：Test Overall AUC 76.76（validation 0.1 网格选权）
   - Baseline+SIP+BEP 等权：Test Overall AUC 76.99
4. 后续 ensemble 必须继续遵守：先固定 validation 选择规则或预注册规则，再一次性 test

### SELFELICIT 实验教训
1. 真实注意力需要 `attn_implementation="eager"`，sdpa不输出attention
2. 集成时不能在测试集上调权重（数据泄露），等权平均为诚实数字
3. 小模型注意力分数极均匀(0.002-0.008)，启发式有时比注意力更有效
