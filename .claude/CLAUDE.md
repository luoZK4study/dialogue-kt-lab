# CLAUDE.md — 项目操作指南

## 项目定位

本项目以 [dialogue-kt](https://github.com/umass-ml4ed/dialogue-kt)（LAK 2025）为主干，使用 **Qwen3-1.7B + LoRA** 和 **MathDial ATC** 构建论文方法结合实验框架。

- **任务**: 对话知识追踪 True/False 二分类，基于 LLM 对 `True` / `False` token 的 logit 比较。
- **默认模型**: Qwen3-1.7B，LoRA `r=16`, `lora_alpha=16`。
- **默认数据集**: MathDial ATC 标注。
- **目标指标**: Overall AUC；Dialogue-KT 论文 LLMKT 汇报值为 **76.71%**。
- **dialogue-kt论文分析**：见 `literature/DialogueKT_Paper_Code_Analysis.md`
- **详细研究状态**: 见 `.claude/MEMORY.md`。
- **目录组织规范**: 见 `.claude/STRUCTURE.md`。

## 标准工作流入口

完整标准工作流见 `.claude/MEMORY.md` 第 〇 节。摘要：

1. 从 `literature/` 选未处理论文，检查/搜索代码仓库。
2. 阅读论文与代码，提出 Dialogue-KT 结合方案。
3. 代码实现，按参数开关接入。
4. 服务器训练 2–3 epochs，记录 Overall / Final Turn AUC。
5. 结果写入 `results/<paper_name>/`。
6. 更新 `.claude/MEMORY.md` 与必要的 `.claude/STRUCTURE.md`。
7. 检查是否能与既有方法做 validation-selected honest ensemble。

## 当前研究状态速览

| 方法/论文 | 状态 | 关键结论 |
|---|---|---|
| SELFELICIT | ✅ 已验证 | 等权集成 Overall AUC 76.24%；未超论文 76.71 |
| Bayesian Epistemology / BEP | ✅ 已验证 | 最佳 BEP-Adaptive-Labels Overall AUC 75.93；未超 baseline |
| Stepwise Informativeness Search / SIP | ✅ 已验证 | SIP-Novelty Overall AUC 75.99，Final Turn AUC 76.85；prune 版本下降 |
| Expert Heads | 🔄 初步分析中 | 已读论文与仓库；建议先做 1.7B attention 诊断再训练 |
| Honest Ensemble | ✅ 已验证 | 3个 clean-test 合规方法超论文：Base+BEP 等权 76.92；0.3Base+0.7BEP 76.76；Base+SIP+BEP 等权 76.99 |

> 详细结果、代码变更、实验记录都在 `results/<paper_name>/` 下维护；本文件只保留操作必需信息。

## 本地环境

| 项目 | 详情 |
|------|------|
| **路径** | `/home/lzk/code/dialogue-kt/` |
| **OS** | WSL2 (Ubuntu) |
| **GPU** | RTX 5060 (8 GB VRAM) |

### 本地网络

Windows 宿主机代理：`http://127.0.0.1:7890`。遇到网络问题时：

```bash
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
```

## SSH 服务器环境

| 项目 | 详情 |
|------|------|
| **路径** | `/home/user4/dialogue-kt/` |
| **GPU** | 2x RTX 3090 (24 GB VRAM) |
| **CUDA** | 12.2 driver / 11.7 toolkit |
| **Conda** | `/opt/anaconda3/`, env: `luo_2` |
| **Python** | 3.10.18 |
| **torch** | 2.4.1+cu118 |
| **transformers** | 5.7.0 |
| **peft** | 0.19.1 |

### 连接

```bash
ssh -i ~/.ssh/id_rsa_u4 -p 22049 user4@119.29.183.125
```

### 环境激活

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate luo_2
cd /home/user4/dialogue-kt
export WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

HuggingFace 不可直连时优先用 ModelScope 下载模型；PyPI 可用清华镜像。

## 常用同步命令

### 同步核心代码

```bash
rsync -avz -e "ssh -i ~/.ssh/id_rsa_u4 -p 22049" \
  /home/lzk/code/dialogue-kt/dialogue_kt/ \
  user4@119.29.183.125:/home/user4/dialogue-kt/dialogue_kt/ \
  --exclude '__pycache__' --exclude '*.pyc'
```

### 同步 `.claude` 项目文档

```bash
rsync -avz -e "ssh -i ~/.ssh/id_rsa_u4 -p 22049" \
  /home/lzk/code/dialogue-kt/.claude/ \
  user4@119.29.183.125:/home/user4/dialogue-kt/.claude/
```

### 同步某个方法的结果目录

```bash
rsync -avz -e "ssh -i ~/.ssh/id_rsa_u4 -p 22049" \
  /home/lzk/code/dialogue-kt/results/<paper_name>/ \
  user4@119.29.183.125:/home/user4/dialogue-kt/results/<paper_name>/
```

## 默认训练命令

```bash
CUDA_VISIBLE_DEVICES=1 python -m dialogue_kt.main train \
  --dataset mathdial --model_type lmkt \
  --base_model /home/user4/.cache/huggingface/hub/qwen/Qwen3-1.7B \
  --model_name lmkt_qwen3_1.7b --epochs 2 --quantize 0
```

重要参数：

- `--quantize 0` 才是禁用量化；`--quantize False` 会被解析成 True。
- `--debug` 用 100 train / 25 val 快速检查链路。
- `--selfelicit ...`、`--bayes_evidence ...`、`--informativeness ...` 为已加入的方法开关。



## Skill 使用

必要时使用 `find-skills` 查找工具，常见场景：

- PDF 论文文本提取。
- 分析外部代码仓库。
- 调试 tokenization / attention / 长上下文问题。

当前已安装项目 skill：`pdf-extraction`。
