# From Evidence to Belief × Dialogue-KT 实验记录

## 论文信息

- **论文**: Kim et al., *From Evidence to Belief: A Bayesian Epistemology Approach to Language Models* (NAACL 2025)
- **本地PDF**: `/home/lzk/code/literature/evidence_rationale_extraction/Kim_2025_From_Evidence_to_Belief.pdf`
- **代码仓库**: https://github.com/MS0117/BayesianEpistemology
- **License**: MIT

## 论文方法分析

论文从贝叶斯认识论角度研究 LLM 如何根据证据的信息量和可靠性调整回答与置信度。核心实验构造不同证据类型：

1. **Golden / original evidence**：正确且支持答案的证据。
2. **Conflicting evidence**：替换为冲突/误导证据。
3. **Incomplete evidence**：只保留一部分原始证据。
4. **Contradictory evidence**：正确证据中混入否定/冲突证据。
5. **Irrelevant evidence**：替换为其他样本的无关证据。
6. **Coincidental evidence**：通过不可靠或偶然推理得到正确答案。
7. **Strength-of-evidence**：从来源可信度、细节程度、时效性、实验可靠性等角度区分强/弱证据。

论文发现：LLM 对 golden evidence 通常会提高置信度，但对冲突、无关、偶然证据的置信度调整不完全符合贝叶斯假设；模型存在偏向“看似标准/黄金证据”的倾向。

## 与 Dialogue-KT 的结合思路

Dialogue-KT 本质是在当前教师问题处，根据历史对话和问题上下文判断学生是否掌握某个 KC。这里可以把历史对话视作证据集合：

- 当前教师问题和最近学生回答：高信息量、高相关证据。
- 更早的历史对话：弱证据或背景证据。
- 过去 correctness/KC 标签（仅训练/提示可控时）：高可靠观测证据。
- 远距离、无关或偶然上下文：不应过度影响当前判断。

因此提出 **Bayesian Evidence Prompting (BEP)**：在 prompt 中显式标记不同证据强度，并在 system prompt 中要求模型像贝叶斯证据更新一样调整对学生掌握状态的信念。

## 已实现方法

### BEP-Adaptive

- CLI: `--bayes_evidence adaptive`
- 代码：
  - `dialogue_kt/bayesian_epistemology.py`
  - `dialogue_kt/prompting.py`
  - `dialogue_kt/kt_data_loading.py`
  - `dialogue_kt/main.py`
- 策略：
  - 当前 turn 与最近上一 turn 标记为 `<strong_evidence>`。
  - 更早的近邻历史标记为 `<weak_evidence>`。
  - 其他内容不标记，作为背景/可能无关证据。
- 不引入额外模型推理，成本与标准 LLMKT 基本一致。

### BEP-Recent2 / BEP-Recent3

- CLI: `--bayes_evidence recent2` / `recent3`
- 更简单的消融：只把最近 2 或 3 个 turn 标记为强证据。

### BEP-Adaptive-Labels

- CLI: `--bayes_evidence adaptive_labels`
- 在 prompt 中额外包含过去 turn 的 correctness/KC 标签，作为高可靠观测证据。
- 风险：需要确认不会泄漏当前 turn 标签；实现中只在当前教师问题处截断，因此不会包含当前学生回答和当前标签。

## 训练命令

```bash
CUDA_VISIBLE_DEVICES=0 python -m dialogue_kt.main train \
    --dataset mathdial --model_type lmkt \
    --base_model /home/user4/.cache/huggingface/hub/qwen/Qwen3-1.7B \
    --model_name bayes_adaptive_qwen3_1.7b --epochs 2 --quantize 0 \
    --bayes_evidence adaptive
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m dialogue_kt.main train \
    --dataset mathdial --model_type lmkt \
    --base_model /home/user4/.cache/huggingface/hub/qwen/Qwen3-1.7B \
    --model_name bayes_adaptive_labels_qwen3_1.7b --epochs 2 --quantize 0 \
    --bayes_evidence adaptive_labels
```

## 实验结果

BEP-Adaptive 与 BEP-Adaptive-Labels 均已完成。

| 方法 | Overall AUC | Final Turn AUC | vs 1.7B baseline 75.99 | vs paper 76.71 | 备注 |
|------|:-----------:|:--------------:|:----------------------:|:--------------:|------|
| BEP-Adaptive | 75.03 | 75.83 | -0.96 | -1.68 | 2 epochs；Final Turn 略优于 baseline，但 Overall 明显下降 |
| BEP-Adaptive-Labels | 75.93 | 75.27 | -0.06 | -0.78 | 2 epochs；未超过基线/论文 |

## 当前结论

本方法是论文思想的轻量 prompt-side adaptation，利用“证据强度/可靠性”指导 LLMKT 不要平均吸收所有历史上下文。完整实验显示：BEP-Adaptive-Labels Overall AUC 75.93，接近但低于 1.7B baseline；BEP-Adaptive Overall AUC 75.03，Final Turn AUC 75.83 有小幅收益但整体下降。因此 Bayesian Evidence Prompting 暂不计入目标方法，后续只作为可能的 ensemble 多样性来源。
