# CEL Stage 2 B 模块统一实现方案

## 1. 阶段目标

Stage 2 的目标是：

> 在 A 已经产生 token-level 选择强度 `a` 的基础上，显式构造证据表征与非证据表征，用 B 变换非证据表征，并要求混合表征的预测与证据表征的预测保持一致。

Stage 2 当前只讨论 A+B，不启动 Stage 3。

## 2. 统一术语

- `H`：Qwen 中间层 hidden state。
- `A`：selector。
- `a`：A 输出的 token-level 选择强度，经过 `tanh`，范围约为 `[-1, 1]`。
- `h_r`：证据表征。
- `h_n`：非证据表征。
- `B`：非证据表征变换模块。
- `h_nb`：B 变换后的非证据表征。
- `h_m`：混合表征。

方法正文不再使用 `score`、`signal`、`gate`、`delta`、`H_env` 或 `R_stage2` 指代上述对象。历史源码字段和已落盘 diagnostics 可以保留原键名，但解释时必须映射到统一术语。

## 3. 目标前向流程

### 3.1 A 输出选择强度

```text
a = tanh(A(H, task condition))
```

### 3.2 构造互补表征

```text
w_r = (a + 1) / 2
w_n = (1 - a) / 2

h_r = w_r ⊙ H
h_n = w_n ⊙ H
```

因此：

```text
w_r + w_n = 1
h_r + h_n = H
```

这一步取代旧实现中基于 `abs(a)` 的 top-k split。top-k 可以保留为离散化消融，但不再作为目标流程的主定义。

上述公式只在可选择证据的 context token 上应用。正式 mask contract 为：

```text
w_r = context_mask * ((a + 1) / 2) + (1 - context_mask)
w_n = context_mask * ((1 - a) / 2)
```

因此非 context 的有效 prompt token 在 `h_r/h_m` 中保持原始 `H`，在 `h_n` 中为零。

### 3.3 B 变换非证据表征

```text
h_nb = B(h_n)
```

B 只能读取 `h_n` 和必要的有效 token mask。它不直接读取标签，不负责生成主证据，也不能依赖 prediction token 的未来信息。

### 3.4 构造混合表征

```text
h_m = h_r + β · h_nb
```

`β` 控制非证据环境变换对混合表征的影响。B 输出尺度必须与 `h_r` 明确对齐，避免 `h_nb` 数值消失或淹没证据表征。

## 4. 双路径预测

设 `P` 为共享的 Qwen 后续网络、最终 normalization、`lm_head` 与输出校准器。

```text
z_r = P(h_r)
p_r = softmax_true_false(z_r)

z_m = P(h_m)
p_m = softmax_true_false(z_m)
```

证据路径和混合路径共享同一个 `P`。正式实现需要在同一 batch 内完成两次后续前向，不能只将 `h_nb` 加到原始 `H` 后进行一次预测。

旧的单路径形式：

```text
H + γ(a ⊙ H + β * environment)
```

只属于历史 Stage 2 首轮实现。它保留完整 `H`，没有 evidence-only prediction，也没有 `p_r/p_m` 一致性约束，因此不再作为目标方法定义。

## 5. 整体训练目标

### 5.1 两条路径的任务损失

```text
L_r = BCE(p_r, y)
L_m = BCE(p_m, y)
```

- `L_r` 保证 `h_r` 本身具有任务充分性。
- `L_m` 保证加入变换后的非证据环境后仍能正确预测。

### 5.2 预测一致性

```text
q_r = [p_r, 1 - p_r]
q_m = [p_m, 1 - p_m]
q_bar = (q_r + q_m) / 2

L_cons = 0.5 * KL(q_r || q_bar)
       + 0.5 * KL(q_m || q_bar)
```

该式是在二分类 Bernoulli 分布上计算 Jensen-Shannon divergence，直接落实：

```text
P(y | h_r) ≈ P(y | h_m)
```

实现 contract：

- `p_r` 和 `p_m` 是经过相同 KC 聚合与共享 calibrator 后的最终样本级正确概率。
- 构造分布前将概率限制到 `[ε, 1-ε]`，避免 `log(0)`。
- 对 batch 中有效样本求平均。
- 两条路径同时反向传播，不使用 teacher/student detach。
- 同时记录 `mean(abs(p_r - p_m))`、最大概率差和校准前后的 logit 差作为诊断。

Faith-DARE 原代码分别监督 rationale-only 与 mixed/counterfactual prediction，并使用 InfoNCE 做表征对齐；它没有直接使用上述 JS。这里的 `L_cons` 是针对当前 KT 流程新增的输出一致性目标。

### 5.3 总损失

```text
L_total = λ_r L_r
        + λ_m L_m
        + λ_cons L_cons
```

### 5.4 暂不加入的候选约束

当前不加入证据预算或 B 输出尺度损失。首轮目标流程训练先完整记录：

- `a`、`(a + 1) / 2` 的均值、分位数与饱和比例。
- `h_n`、`h_nb`、`βh_nb` 的 RMS 和相对尺度。
- B 参数梯度、参数变化及 `h_m - h_r` 的实际幅度。

若实际出现 A 将几乎全部 context 置为证据，再评估证据预算正则；若 B 的输出和梯度长期接近零，再评估输出尺度或结构约束。二者目前不是核心目标。

## 6. 统一训练协议（默认）

除非模块明确声明特有训练机制，A、B、LoRA、calibrator 及未来新增模块默认从训练开始即作为一个整体进行端到端训练：

```text
raw Qwen + LoRA + A + B + optional calibrator (+ C/D/...)
-> one optimizer, one L_total, same forward graph from epoch 1
-> train + validation each epoch
-> validation-best checkpoint, patience/min_delta early stopping
-> restore best checkpoint -> final test and audit
```

训练开始前固定 `max_epochs`、validation 指标、`patience` 和 `min_delta`。不同模块可以使用 parameter-group learning rate，但这仍属于同一优化任务，不得默认冻结、交替或拆分为独立训练。

当前实现状态：训练 CLI 已实现 `--max_epochs`、`--patience`、`--min_delta` 与 validation-best checkpoint 早停；`--epochs` 仅为兼容别名。本节描述的统一协议不对历史五阶段运行作回溯性认证。

### 6.1 Calibrator 的必要性与边界

Calibrator 是共享预测头之后的可选概率校准层，bias 模式为 `sigmoid(logit(p)+b)`，affine 模式为 `sigmoid(s*logit(p)+b)` 且 `s>0`。它只调整概率偏移、尺度和阈值行为，不学习 A 的选择或 B 的非证据变换，因此不是双路径表示学习或 `L_total` 的必要条件。

机制验证优先使用 `none` 或恒等初始化的共享 calibrator。启用时，它必须从 epoch 1 起同时作用于 `p_r` 与 `p_m`（评估时含 `p_n`），并与所有模块使用同一 optimizer 和完整 `L_total`。不得把 calibrator-only warmup 或训练后单独校准作为默认流程；后处理/消融结果必须单独标记。单调 bias/positive-affine 后处理不改变 AUC，应分别报告 raw/calibrated 概率、ECE/Brier 和阈值指标。

### 6.2 历史 staged protocol（仅用于兼容与溯源）

每个正式 A+B candidate 使用独立、自包含的来源链：

```text
Qwen3-1.7B raw base
-> fresh LoRA + A + calibrator
-> A bootstrap with additive evidence enhancement
-> same-candidate calibrator warmup
-> same-candidate A strict joint
-> initialize B
-> B warmup with A/downstream controlled
-> A+B+LoRA+calibrator strict joint on L_total
-> final dual-path test
-> provenance / parameter / tensor-contract audit
```

禁止加载其他 A+B candidate、baseline 或历史 anchor 的参数。Stage 2 可以读取同一 candidate 刚完成的 A 阶段 checkpoint，因为它属于当前 candidate 自身的顺序训练链。

这套 `A bootstrap -> calibrator warmup -> A strict joint -> B warmup -> joint` 是现有代码和已运行工件的历史阶段链，不是新默认训练规范。它可以用于解释旧日志和复现实验，但不能作为统一端到端结果的证据。

历史协议下的建议优化顺序：

1. 先固定 A 与共享后续网络，使用 `L_m + λ_cons L_cons` 训练 B，使 `h_nb` 形成可测的有效输出；此时 `L_r` 只作监控，因为它对 B 没有梯度。
2. 再以完整的 `λ_r L_r + λ_m L_m + λ_cons L_cons` 和分组学习率联合训练 A、B、LoRA 与 calibrator。
3. A 的学习率应小于 B，避免 B 尚未建立时 A 迅速改变分解边界。
4. 两条预测路径必须共享 batch、标签、后续网络和校准器。

## 7. B 模块的正式设计要求

正式 B 不以“参数最少”为设计目标，而应从 `h_n -> h_nb` 所需的上下文变换反推结构。至少满足：

- 能处理 sequence-level `h_n`，而不是只做彼此独立的 token 映射，除非 token-wise 版本被明确标记为消融。
- 具备充分的隐藏维度、层数和注意力容量，以表达跨 turn、跨 token 的非证据环境关系。
- 使用与 Qwen hidden state 匹配的 normalization、mask 和输出投影。
- 输出具有明确的幅度控制，并报告 `h_nb/h_n` 与 `h_nb/h_r` 的尺度比例。
- 对 padding、空 non-evidence 区域和混合精度保持数值稳定。
- 不通过标签或未来 token 泄漏预测信息。

当前正式主方向是重新设计一个 sequence-level contextual Transformer B：使用多层 pre-norm attention/FFN block、与 Qwen hidden state 匹配或充分宽的内部维度、严格的有效 token mask 和完整输出投影。现有首轮的一层 `512` 维 bottleneck Transformer 只保留为历史实现与容量消融，不直接作为新主候选，也不复用其 checkpoint。

轻量 MLP、Shuffle 或 tiny Transformer 只能用于：

- 接口和梯度检查；
- `β=0`、冻结 B、容量消融等对照；
- 小规模 pilot。

它们不能作为 B 方向的充分实现，也不能因其失败而否定目标机制。

## 8. 后续控制设计与当前必需诊断

### 8.1 后续控制实验（当前不执行）

以下 controls 属于更完整的机制研究设计，但用户已明确将它们排除在当前三轮实现与验证范围之外。当前不得启动这些实验，也不得把它们或额外 seed 作为本次 Stage 2 技术关闭的前置条件。

- **A-only**：只使用 `h_r` 路径，确定证据路径性能。
- **β=0**：保留共同训练轨迹，关闭 B 对 `h_m` 的直接贡献。
- **Frozen-B**：B 不更新，区分结构存在与学习所得作用。
- **No-consistency**：移除 `L_cons`，验证预测一致性目标的贡献。
- **Capacity ablation**：比较充分容量 B 与轻量 B，但只将充分容量版本作为主候选。
- **Multi-seed**：未来若要形成跨 seed 稳健性结论，最佳配置至少进行三个 model/data seed 复验；当前只使用 seed `1221`。

### 8.2 表征和预测诊断

- `a`、`w_r`、`w_n` 的分布与饱和情况。
- `h_r`、`h_n`、`h_nb`、`h_m` 的 L2/RMS 与有限值检查。
- `p_r`、`p_m` 的 AUC、Acc、概率差、logit 差与 Jensen-Shannon divergence。
- `P(h_n)` 的 reversal performance。
- A、B、LoRA、calibrator 的梯度和参数变化。
- 注入位置对 prediction token 的因果影响。

## 9. 首轮历史结果的重新定位

已完成的三个 Stage 2 首轮 candidate 均通过来源和参数审计：

| 历史方向 | Overall Acc/AUC | Final Acc/AUC | 相对 A 模块参考的 Overall AUC |
|---|---:|---:|---:|
| Shuffle | `68.72 / 74.14` | `62.52 / 72.79` | `-1.24` |
| Token-wise MLP | `68.46 / 75.47` | `63.88 / 76.29` | `+0.09` |
| Small Transformer | `68.06 / 75.25` | `67.18 / 74.73` | `-0.13` |

这些结果来自旧的单路径 residual 流程，只能作为历史 pilot 与工程诊断：

- 它们没有构造 `h_r/h_n` 的互补分解。
- 它们没有分别计算 `p_r` 与 `p_m`。
- 它们没有使用 `L_cons`。
- 它们不能验证证据充分性或环境不变性。

因此当前没有 Stage 2 winner。下一轮正式工作应先实现本文定义的双路径流程，再设计充分容量的 B 主候选。

## 10. 当前实现状态与代码边界

双路径实现已集中落在：

- `dialogue_kt/cel_methods.py`
  - 从 `a` 构造 `h_r/h_n`。
  - 实现 `h_nb` 与 `h_m`。
  - 支持两条后续前向所需的注入接口。
- `dialogue_kt/training.py`
  - 同一 batch 计算 `p_r/p_m`。
  - 实现 `L_r/L_m/L_cons`。
  - 增加 B warmup、分组学习率与双路径 diagnostics。
- `dialogue_kt/main.py`
  - 增加双路径训练、一致性权重和 B 容量参数。
- `scripts/cel_stage2_environment/`
  - 增加双路径 tensor contracts、audit 与结果汇总。

长序列 joint 训练按 `rows * sequence_length^2` 评估注意力风险：低于 `8000000` 时将 evidence/mixed 沿 batch 维展开；达到或超过该阈值时串行执行两条路径，并对共享损失做精确梯度拆分。该策略不改变 `L_total`；自动 tensor contract 在 CPU/CUDA 上测得的最大参数梯度差分别为 `2.98e-08` 和 `1.19e-07`。完整训练集最坏 batch 的真实 Qwen joint 与 B-only 预检均已通过。

当前允许的正式工作是固定 seed `1221` 的三轮 validation-driven 配置实验、冻结后的统一 final test、来源/参数/互补/tensor-contract 审计和最终报告；不启动 controls、额外 seed 或 Stage 3。

## 11. 方法设计原则

方法设计应首先服从研究假设和目标因果机制，而不应以“最小改动、最少参数或单次训练成本最低”作为正式方案的首要目标。对于每一个阶段，应从目标机制反向设计完整且功能闭合的模块，明确其输入输出表征、模块之间的信息流、下游使用方式、训练目标、优化日程以及可验证的诊断指标。模块容量应与所要建模的变换复杂度相匹配，不能用过小或过于简化的模块替代核心方法，再据此判断研究方向是否有效。轻量模块可以用于接口检查、sanity check 或消融实验，但不应作为核心方法的主要证据。不应通过削弱核心模块的表达能力来节省正式训练。参数效率应在机制得到充分验证之后再进行优化。
