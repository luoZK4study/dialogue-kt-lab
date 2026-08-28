# Stepwise Informativeness Search × Dialogue-KT 实验记录

## 论文信息

- **论文**: Wang et al., *Stepwise Informativeness Search for Effective and Efficient LLM Reasoning* (2025)
- **本地PDF**: `/home/lzk/code/literature/evidence_rationale_extraction/Wang_2025_Stepwise_Informativeness_Search.pdf`
- **代码仓库**: https://github.com/SiyuanWangw/Informativeness-Search

## 论文方法分析

论文指出 LLM 在长上下文/多步推理中容易忽略中间或早期有用步骤，并生成冗余重复的中间结论。其核心框架包括：

1. **Grounding-guided selection**：识别 underutilized prior steps，并用 attention 判断候选新步骤是否有效 ground 到这些步骤。
2. **Novelty-guided selection**：用 trigram overlap 判断新结论是否与已有结论重复，过滤低新颖性步骤。
3. **Self-grounding strategy**：要求每一步显式写成 `[Step-i] From <source>, <deduction>`，让模型明确引用相关前提。

官方实现中的关键工具在 `Src/utils/select_steps.py`：
- `compute_max_infogain`: `1 - trigram overlap`
- `get_all_step_infogain`: 根据 information gain 阈值找 underutilized steps
- `get_new_conclusion`: 根据结论 trigram overlap 计算 novelty

## 与 Dialogue-KT 的结合思路

Dialogue-KT 的历史对话可以视作一串 reasoning/evidence steps。当前 KC 预测时，模型可能：

- 过度关注最近 turn，忽略早期关键学生错误/教师纠正；
- 重复利用语义相似但信息量低的 turn；
- 在长对话中丢失中间证据。

因此提出 **Stepwise Informativeness Prompting (SIP)**：用 trigram information gain 在 prompt 构造阶段标记非冗余、可能 underutilized 的 turn，并在 system prompt 中要求模型显式优先利用这些 informative steps。

## 已实现方法

### SIP-Novelty

- CLI: `--informativeness novelty`
- 代码：
  - `dialogue_kt/informativeness_search.py`
  - `dialogue_kt/prompting.py`
  - `dialogue_kt/kt_data_loading.py`
  - `dialogue_kt/main.py`
- 策略：
  - 当前 turn 标记为 `<informative_step>`。
  - 对历史可见 turn，计算该 turn 学生回答与后续 turn 的最大 trigram overlap。
  - 若 `1 - max_overlap >= 0.7`，认为该 turn 非冗余/未被充分利用，标记为 `<informative_step>`。
  - 最近上一 turn 保留为 `<recent_step>`。

### SIP-Novelty-Recent

- CLI: `--informativeness novelty_recent`
- 在 SIP-Novelty 基础上，将最近上一 turn 也提升为 `<informative_step>`。

### SIP-Recent-Grounded

- CLI: `--informativeness recent_grounded`
- 消融版本：最近 3 个 turn 标记为 informative，不做 novelty 计算。

## 训练命令

```bash
CUDA_VISIBLE_DEVICES=1 python -m dialogue_kt.main train \
    --dataset mathdial --model_type lmkt \
    --base_model /home/user4/.cache/huggingface/hub/qwen/Qwen3-1.7B \
    --model_name sip_novelty_qwen3_1.7b --epochs 2 --quantize 0 \
    --informativeness novelty
```

## 实验结果

待服务器训练完成后填写。

| 方法 | Overall AUC | Final Turn AUC | vs 1.7B baseline 75.99 | vs paper 76.71 | 备注 |
|------|:-----------:|:--------------:|:----------------------:|:--------------:|------|
| SIP-Novelty | — | — | — | — | 待运行 |

## 当前结论

SIP 是第二篇论文的轻量 prompt-side adaptation，优点是不需要额外 LLM 推理或 attention 输出，训练成本接近标准 LLMKT。它比 BEP 更直接针对“长对话中早期/中间证据被忽略”和“重复证据干扰”的问题，可能更适合 Dialogue-KT。
