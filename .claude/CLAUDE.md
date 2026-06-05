# CLAUDE.md — 项目指南

## 项目概述

以 [dialogue-kt](https://github.com/umass-ml4ed/dialogue-kt)（LAK 2025）代码为基础主干，将 **Qwen3-1.7B** 作为默认训练模型，**MathDial** 作为默认训练数据集，构建对话知识追踪实验框架。

核心工作流：
1. 对 `literature/` 目录中的论文逐篇阅读分析，结合论文给出的代码仓库理解其方法
2. 将论文方法或其核心思想与 dialogue-kt 代码结合，提出新的结合方法
3. 编写对应代码实现，在 SSH 服务器上训练验证（2-3 epochs）
4. 按规范记录方法描述、代码变更、实验结果，分类保存

- **原始论文**: Exploring Knowledge Tracing in Tutor-Student Dialogues using LLMs (LAK 2025)
- **arXiv**: https://arxiv.org/abs/2409.16490
- **任务**: True/False 二分类，基于 LLM logit 比较
- **默认模型**: Qwen3-1.7B (LoRA, r=16)
- **默认数据集**: MathDial (ATC 标注)
- **已有结合**: SELFELICIT ✅ | Bayesian Epistemology 🔄 | Informativeness Search 🔄 详见 `results/`
- **组织规范**: 按论文方法名进行命名空间隔离，详见 `MEMORY.md` 及 `STRUCTURE.md`

## 本地环境

| 项目 | 详情 |
|------|------|
| **路径** | `/home/lzk/code/dialogue-kt/` |
| **OS** | WSL2 (Ubuntu) |
| **GPU** | RTX 5060 (8 GB VRAM) |

### 本地网络

- Windows 宿主机代理: `http://127.0.0.1:7890`（WSL2 localhost 转发，无需显式 IP）
- 遇到网络问题时设置代理:
  ```bash
  export http_proxy=http://127.0.0.1:7890
  export https_proxy=http://127.0.0.1:7890
  ```

## 服务器环境

| 项目 | 详情 |
|------|------|
| **路径** | `/home/user4/dialogue-kt/` |
| **GPU** | 2x RTX 3090 (24 GB VRAM, Ampere) |
| **CUDA** | 12.2 (驱动 535), 11.7 (toolkit) |
| **Conda** | `/opt/anaconda3/`, env: `luo_2` |
| **Python** | 3.10.18 |
| **torch** | 2.4.1+cu118 |
| **transformers** | 5.7.0 |
| **peft** | 0.19.1 |
| **磁盘** | 7.2T (1.6T 可用) |

### SSH 连接

```bash
ssh -i ~/.ssh/id_rsa_u4 -p 22049 user4@119.29.183.125
```

### 网络

- HuggingFace 不可直连 → 模型用 **ModelScope** 下载
- PyPI 直连可用，清华镜像 `https://pypi.tuna.tsinghua.edu.cn/simple` 更快

### 环境激活

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate luo_2
cd /home/user4/dialogue-kt
export WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

### 同步代码到服务器
```bash
rsync -avz -e "ssh -i ~/.ssh/id_rsa_u4 -p 22049" /home/lzk/code/dialogue-kt/dialogue_kt/ user4@119.29.183.125:/home/user4/dialogue-kt/dialogue_kt/ --exclude '__pycache__' --exclude '*.pyc'
```

### 同步结果/脚本
```bash
scp -i ~/.ssh/id_rsa_u4 -P 22049 /home/lzk/code/dialogue-kt/scripts/selfelicit/*.py user4@119.29.183.125:/home/user4/dialogue-kt/scripts/selfelicit/
scp -i ~/.ssh/id_rsa_u4 -P 22049 /home/lzk/code/dialogue-kt/results/selfelicit/*.md user4@119.29.183.125:/home/user4/dialogue-kt/results/selfelicit/
```

---

## 各组件说明

### `main.py` — CLI 入口

5 个子命令：`train`, `test`, `annotate`, `human-eval`, `visualize`

关键参数：
| 参数 | 默认值 | 说明 |
|------|------|------|
| `--dataset` | `comta` | `mathdial` / `comta` |
| `--model_type` | `lmkt` | `lmkt` / `dkt-sem` / `dkt` / `akt` / `dkvmn` / `saint` / `simplekt` / `bkt` |
| `--base_model` | `Qwen/Qwen3-1.7B` | HuggingFace 基座模型 |
| `--epochs` | 2 | 训练轮数 |
| `--quantize` | `True` | 4-bit 量化（服务器用 `--quantize 0` 禁用） |
| `--debug` | `False` | 数据子集（100 训练/25 验证） |
| `--pack_kcs` | `True` | 多 KC 打包到单个 prompt |
| `--agg` | `mean-ar` | KC 聚合方式（`prod` / `mean-ar` / `mean-geo`） |

> **注意**: `--quantize` 用 `bool_type(x != "0")` 解析，所以 `--quantize 0` 禁用，`--quantize False` 反而是 True！

### `training.py` — 训练核心

- 损失函数：BCE Loss（预测正确性概率 vs 真实标签）
- LLMKT 默认超参：lr=2e-4, r=16, lora_alpha=16, batch_size=1, grad_accum=32
- 评估：Accuracy, AUC, Precision, Recall, F1（Overall + Final Turn）

### `kt_data_loading.py` — 数据集

- **LMKTDatasetPacked**（默认）：所有 KC 打包到一个 prompt，通过 3D attention mask 隔离
- **LMKTDatasetUnpacked**：每个 KC 单独一个 prompt

### `models/lm.py` — 模型加载

- `get_base_model()`：加载基座模型（bfloat16 或 4-bit 量化）
- `get_model()`：添加 LoRA adapter（target_modules 为所有线性投影层）
- 启用 gradient checkpointing（`use_reentrant=True`）

### `prompting.py` — Prompt 模板

- `KT_SYSTEM_PROMPT`：LLM 扮演数学教师
- `kt_user_prompt()`：问题上下文 + 对话历史 + 待评估 KC
- `get_true_false_tokens()`：Qwen True=2514, False=4049

### `utils.py`

- `bool_type(x)`: `x != "0"` → `--quantize 0` 才禁用
- `initialize_seeds(221)`: 固定随机种子，启用确定性算法

---

## 当前默认训练命令

```bash
CUDA_VISIBLE_DEVICES=1 python -m dialogue_kt.main train \
    --dataset mathdial --model_type lmkt \
    --base_model /home/user4/.cache/huggingface/hub/qwen/Qwen3-1.7B \
    --model_name lmkt_qwen3_1.7b --epochs 2 --quantize 0
```

---

## Skill 使用

必要时使用 `find-skills` 技能寻找合适的工具辅助阅读论文、分析代码、提出新方法。

### 调用方式
```
/find-skills
```
或直接 `Skill` 工具调用 `find-skills`

### 常用场景
- PDF论文文本提取（如 pip install pymupdf）
- 调试 tokenization 问题
- 搜索开源社区解决方案

---

## 最新项目结构

```
dialogue-kt/
├── literature/                      # 论文PDF
│   ├── dialogue-kt.pdf              # 框架论文（直接放置）
│   ├── evidence_rationale_extraction/
│   ├── information_bottleneck/
│   └── representation_information_compression/
├── dialogue_kt/                     # 核心代码
│   ├── selfelicit.py                # SELFELICIT注意力模块
│   ├── training.py                  # 训练/评估（含--selfelicit参数）
│   ├── kt_data_loading.py           # 数据集（含LMKTDatasetPackedSE）
│   ├── prompting.py                 # 提示模板（含SE模板）
│   └── models/lm.py                 # 模型加载（含eager_attention参数）
├── scripts/
│   ├── selfelicit/                  # SELFELICIT分析脚本
│   ├── bayesian_epistemology/
│   └── informativeness_search/
├── saved_models/                    # 模型（仅1.7B）
├── results/
│   ├── selfelicit/                  # ✅ 已验证
│   ├── bayesian_epistemology/       # 🔄 进行中
│   └── informativeness_search/      # 🔄 进行中
├── .claude/
│   ├── CLAUDE.md                    # 本文件：指南
│   ├── MEMORY.md                    # 总览 + 工作流 + 论文状态
│   └── STRUCTURE.md                 # 目录结构规范
└── requirements.txt
```

---

## 工作流

完整标准工作流见 **`.claude/MEMORY.md` 第〇节**，核心步骤：
0. 论文准入（检查代码、跳过已验证）→ 1. 论文理解 → 2. 方法设计 → 3. 代码实现 → 4. 训练验证 → 5. 结果整理 → 6. 文档更新 → 7. 交叉结合检查

---
