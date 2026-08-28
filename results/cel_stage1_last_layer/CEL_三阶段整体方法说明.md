# CEL 整体方法说明

## 1. 当前范围

CEL 当前聚焦 A 模块与 B 模块：

- A 从 Qwen 中间层 hidden state 中产生 token-level 选择强度。
- A 将 hidden state 分解为证据表征和非证据表征。
- B 只变换非证据表征。
- 证据表征与混合表征分别进入共享后续网络。
- 训练目标要求两条路径都能完成 KT 预测，并保持预测一致。

Stage 3 暂不讨论，也不启动相关实验。

## 2. 任务形式

任务仍采用 Dialogue-KT / LLMKT 的 True/False token prediction：

```text
dialogue history + current Knowledge Component
-> Qwen3-1.7B + LoRA
-> True/False logits
-> mastery probability
```

设选定中间层的 hidden state 为：

```text
H ∈ R^(B × S × d)
```

CEL 不替换最终语言模型预测头，而是改变进入后续 Qwen 网络的中间表征。

## 3. 统一术语

- `H`：Qwen 中间层 hidden state。
- `A`：selector。
- `a`：A 输出的 token-level 选择强度。
- 当前 A 对 `a` 使用 `tanh`，因此其范围约为 `[-1, 1]`。
- `h_r`：证据表征。
- `h_n`：非证据表征。
- `B`：非证据表征变换模块。
- `h_nb`：B 变换后的非证据表征。
- `h_m`：混合表征。

方法正文不再混用 `score`、`signal`、`gate` 或 `delta` 指代 `a`。历史源码变量、CLI 名称和 diagnostics 字段可暂时保留，以保证已有 checkpoint 与结果兼容。

## 4. A 模块

### 4.1 选择强度

A 根据中间层表示、当前 prediction 位置和当前 KC 产生：

```text
a = tanh(A(H, h_answer, h_kc, context_mask))
```

`a` 表示每个有效 context token 对当前知识判断的选择强度。

### 4.2 当前已验证的加性证据增强

A 生成的证据表征为：

```text
a ⊙ H
```

当前 Stage 1 的实际注入为：

```text
H' = H + γ(a ⊙ H)
```

由于原始 `H` 被保留，这一做法统一称为 **加性证据增强**。它用于先验证 A 产生的证据表征能否真实影响最终 KT 预测。

### 4.3 A 模块训练目标

```text
p_A = P(H')
L_A = BCE(p_A, y)
```

`P` 表示注入点之后的 Qwen 网络、最终 normalization、`lm_head` 与输出校准器。

### 4.4 A 模块训练流程（历史参考）

当前 A 模块参考工件使用过以下已经验证的历史训练机制：

```text
Qwen3-1.7B raw base
-> fresh LoRA + A bootstrap
-> same-candidate calibrator warmup
-> same-candidate strict joint
-> final test and audit
```

该链仅用于 A 模块工件的 provenance 和复现，不是未来 A+B 或新增模块的默认训练协议。方法正文统一称其为 **A 模块历史参考链**，不再使用历史候选编号作为方法名称。

后续默认协议是固定 `max_epochs`、validation 指标、`patience` 和 `min_delta`，从第一个 epoch 起将所有启用模块放入同一计算图、完整损失、optimizer 和 train/validation 循环；calibrator 为可选概率校准层，启用时也必须联合训练，禁止默认 calibrator-only warmup 或训练后单独校准。

## 5. A 产生互补证据分解

为了将有符号 `a` 转换为 `[0, 1]` 范围的互补权重，定义：

```text
w_r = (a + 1) / 2
w_n = (1 - a) / 2
```

随后构造：

```text
h_r = w_r ⊙ H = ((a + 1) / 2) ⊙ H
h_n = w_n ⊙ H = ((1 - a) / 2) ⊙ H
```

其中：

- `h_r` 是证据表征。
- `h_n` 是非证据表征。
- `w_r + w_n = 1`。
- `h_r + h_n = H`。

该互补分解是目标 A+B 方法的正式接口。旧 Stage 2 使用的 top-k split 只保留为历史消融，不再承担主流程定义。

这些公式只在可选证据的 context token 上应用。非 context 的有效 prompt token 在 `h_r/h_m` 中保持原始 `H`，在 `h_n` 中为零；padding 仍由 attention mask 排除。

## 6. B 模块与混合表征

B 只接收非证据表征：

```text
h_nb = B(h_n)
```

然后构造：

```text
h_m = h_r + β · h_nb
```

`h_m` 是混合表征。其核心含义是：在保留证据表征的基础上，改变非证据环境，并检验预测是否仍然稳定。

B 不应直接承担稳定标签证据的生成任务。正式 B 必须具备与 sequence-level 环境变换相匹配的容量，并显式控制输出尺度、mask、数值稳定性和标签泄漏风险。

## 7. 双路径预测

证据路径：

```text
z_r = P(h_r)
p_r = softmax_true_false(z_r)
```

混合路径：

```text
z_m = P(h_m)
p_m = softmax_true_false(z_m)
```

两条路径共享同一个后续网络 `P`。这样可以直接检验：

```text
P(y | h_r) ≈ P(y | h_m)
```

这与旧的单路径 residual 注入存在本质区别。旧实现只产生一个最终预测，无法证明证据路径充分，也无法测量混合环境是否保持预测不变。

## 8. 整体训练目标

证据路径任务损失：

```text
L_r = BCE(p_r, y)
```

混合路径任务损失：

```text
L_m = BCE(p_m, y)
```

预测一致性损失：

```text
q_r = [p_r, 1 - p_r]
q_m = [p_m, 1 - p_m]
q_bar = (q_r + q_m) / 2

L_cons = 0.5 * KL(q_r || q_bar)
       + 0.5 * KL(q_m || q_bar)
```

即在 `[p, 1-p]` 上计算 Jensen-Shannon divergence。`p_r/p_m` 采用相同 KC 聚合与共享 calibrator 后的最终样本级概率；概率先限制到 `[ε,1-ε]`，两条路径均参与反向传播，不使用 `detach`。该项是当前 KT 流程新增的输出一致性目标，不是 Faith-DARE 原始损失。

Stage 2 核心总损失为：

```text
L_total = λ_r L_r
        + λ_m L_m
        + λ_cons L_cons
```

当前训练目标同时要求：

1. `h_r` 足以预测标签。
2. `h_m` 能正确预测标签。
3. `p_r` 与 `p_m` 保持一致。

证据比例和 B 输出尺度当前只作为诊断。若正式训练实际出现全证据坍缩或 B 输出长期失效，再决定是否增加针对性正则。

此外，`P(h_n)` 的 reversal performance 必须作为正式诊断。目标状态是证据表征保留主要预测信息，非证据表征不应单独保持同等预测能力。

## 9. A+B 训练顺序（历史与默认协议）

历史 candidate 使用过以下自包含训练链：

```text
raw Qwen3 base
-> 建立当前 candidate 的 A 模块
-> calibrator warmup
-> A strict joint
-> 初始化并 warm up B
-> A+B+LoRA+calibrator 使用 L_total 联合训练
-> 双路径 final test
-> provenance / parameter / tensor-contract audit
```

正式训练不加载其他 candidate 的参数。上述 staged 链只用于解释旧工件。新的 A+B（以及未来 A+B+C 等）默认从 raw Qwen 开始统一端到端训练；同一 optimizer 可使用不同模块的 parameter-group learning rate，但不得默认阶段冻结、交替、warmup 或按 batch 调整损失。早停后恢复 validation 最优 checkpoint，再执行 final test 和审计。

## 10. 方法设计原则

方法设计应首先服从研究假设和目标因果机制，而不应以“最小改动、最少参数或单次训练成本最低”作为正式方案的首要目标。对于每一个阶段，应从目标机制反向设计完整且功能闭合的模块，明确其输入输出表征、模块之间的信息流、下游使用方式、训练目标、优化日程以及可验证的诊断指标。模块容量应与所要建模的变换复杂度相匹配，不能用过小或过于简化的模块替代核心方法，再据此判断研究方向是否有效。轻量模块可以用于接口检查、sanity check 或消融实验，但不应作为核心方法的主要证据。不应通过削弱核心模块的表达能力来节省正式训练。参数效率应在机制得到充分验证之后再进行优化。

该原则要求正式候选优先做到：表示定义完整、forward 闭合、损失闭合、容量充分和诊断可验证。它不等于无条件增加参数。

## 11. 当前实验状态

### 11.1 Baseline

- Overall Acc/AUC：`69.22 / 74.95`
- Final Acc/AUC：`61.17 / 75.32`

### 11.2 A 模块参考结果

- Overall Acc/AUC：`70.03 / 75.38`
- Final Acc/AUC：`66.80 / 77.26`

其历史 checkpoint 文件名为 `cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b`。该名称仅用于工件、日志和审计追溯；方法正文统一称为 **A 模块**。

### 11.3 Stage 2 首轮历史结果

首轮 Shuffle、token-wise MLP 和 small Transformer 均完成正式训练与审计，但它们采用旧的单路径 environment residual 设计，没有实现本文定义的 `h_r/h_m` 双路径预测和一致性损失。

因此：

- 三个结果保留为历史 pilot 和工程诊断。
- 当前没有 Stage 2 winner。
- 不应继续围绕“tiny 模块是否涨分”判断 B 方向是否成立。
- 下一轮应先完成目标流程的代码、loss、tensor contract 和 audit，再训练充分容量的 B 主候选。

## 12. 当前文件关系

- `results/method_design/CEL_Modular_Flow.md`
  - 统一方法定义与设计原则。
- `results/method_design/CEL_Stage1_A_Module_Implementation.md`
  - A 模块当前实现、训练链和接口。
- `results/method_design/CEL_Stage2_Environment_Generator_Implementation.md`
  - B 模块目标实现、损失、训练顺序和控制实验。
- `.claude/STAGE2_HANDOFF.md`
  - 当前状态与下一步执行入口。
- `results/cel_stage2_environment/`
  - 首轮历史 pilot 的结果、诊断和审计记录。
