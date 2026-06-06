# Expert Heads × Dialogue-KT 初步记录

## 论文信息

- **论文**: Wu et al., *Expert Heads: Robust Evidence Identification for Large Language Models* (ICLR 2026)
- **本地PDF**: `/home/lzk/code/literature/evidence_rationale_extraction/Wu_2025_Expert_Heads.pdf`
- **代码仓库**: https://github.com/Xuan-Van/ExpertHead

## 论文方法初步分析

论文发现 LLM 在多文档证据识别中存在明显位置敏感性，许多 attention heads 过度关注序列边界而忽略中间内容。作者通过对 gold documents 在不同位置的排列进行 attention 分析，定义：

- **Activated Head**：某个 head 对所有 gold documents 的注意力都高于任意 distractor documents。
- **Sensitive Head**：在至少一个 permutation 中满足 activation frequency 与 average attention score 阈值。
- **Expert Head**：在所有 permutations 中稳定满足阈值的 sensitive head。

关键发现：

1. 少量 Expert Heads 能稳定关注任务相关证据，且比普通 top-attention head 更稳健。
2. Qwen-2.5-7B 中更深层 heads 更偏向 evidence selection；LLaMA/Mistral 中层更偏 semantic integration。
3. Expert Head 激活强度与答案正确性相关，可作为 hallucination / uncertainty 诊断信号。
4. Expert Head voting 可用于 document ranking/context pruning。

## 与 Dialogue-KT 的潜在结合

Dialogue-KT 的历史 dialogue turns 可视作候选 documents。Expert Heads 思路可映射为：

1. **Turn ranking**：用 Qwen 深层 attention heads 对各历史 turn 打分，选择最相关 turns。
2. **Context pruning**：删除低 expert-head attention 的 turn，降低长上下文噪声。
3. **Uncertainty signal**：若 expert-head attention 分散，说明证据不足，预测可能不可靠。
4. **Ensemble/reranking**：与 SIP/BEP 的 prompt-side 方法组合，形成 validation-selected ensemble。

## 风险与注意事项

- SELFELICIT 真注意力证据在 Qwen3-1.7B 上 Overall AUC 只有 74.90%，说明小模型 attention 分布可能过于均匀。
- Expert Heads 论文主要在 7B 级模型做分析；1.7B 上是否存在稳定 expert heads 不确定。
- 若实现，应先做小规模 validation/debug attention 诊断，不宜直接大规模训练。

## 当前状态

已完成论文初步阅读与方向分析；尚未实现代码和训练。建议下一步先写一个 `scripts/expert_heads/diagnose_turn_attention.py`，在 validation 子集上抽样计算 Qwen3-1.7B 深层 heads 对 turn 的注意力集中度，再决定是否进入完整训练。
