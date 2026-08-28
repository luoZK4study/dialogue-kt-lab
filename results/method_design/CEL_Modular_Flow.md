# CEL A-B 核心方法统一流程

## 1. 文档范围

本文是 CEL 当前方法定义的统一入口，只讨论 A 模块与 B 模块。Stage 3 暂不展开。

当前研究目标不是简单地向 Qwen 隐藏状态增加一个额外残差，而是建立下面这条完整链路：

```text
H
-> A 输出 a
-> 构造互补的证据表征 h_r 与非证据表征 h_n
-> B 变换 h_n 得到 h_nb
-> 构造混合表征 h_m
-> h_r 与 h_m 分别经过共享后续网络
-> 约束两条路径都能完成 KT 预测，且预测保持一致
```

## 2. 统一术语与符号

### 2.1 基本定义

- `H`：Qwen 某个中间层的 hidden state，形状为 `B x S x d`。
- `A`：token-level selector。
- `a`：A 输出的 token-level 选择强度。
- 当前 A 对输出使用 `tanh`，因此 `a` 为有符号强度，取值范围约为 `[-1, 1]`。
- `B`：非证据表征变换模块。

方法文档中不再使用 `score`、`signal`、`gate` 或 `delta` 指代 `a`。源码中的历史类名、变量名和诊断字段可以暂时保留，以避免破坏 checkpoint 与结果兼容性；解释这些字段时统一映射为 `a`。

### 2.2 当前 A 模块的加性证据增强

当前已经验证的 A 模块先构造：

```text
a ⊙ H
```

其中 `⊙` 表示逐 token 广播乘法。`a ⊙ H` 统一称为 **A 生成的证据表征**。

当前 Stage 1 的实际注入为：

```text
H' = H + γ(a ⊙ H)
```

因为原始 `H` 被完整保留，这一形式属于 **加性证据增强**，而不是证据/非证据的显式互补分解。

### 2.3 目标流程中的互补分解

为得到 `[0, 1]` 范围的互补权重，定义：

```text
w_r = (a + 1) / 2
w_n = (1 - a) / 2
```

于是：

```text
w_r + w_n = 1
h_r = w_r ⊙ H = ((a + 1) / 2) ⊙ H
h_n = w_n ⊙ H = ((1 - a) / 2) ⊙ H
```

- `h_r`：证据表征。
- `h_n`：非证据表征。

该定义满足：

```text
h_r + h_n = H
```

因此，`h_r` 与 `h_n` 是由同一个 `a` 得到的互补分解，不再依赖另一套 top-k 划分信号。

上述公式是 context 区域的简写。正式实现中：

```text
w_r = context_mask * ((a + 1) / 2) + (1 - context_mask)
w_n = context_mask * ((1 - a) / 2)
```

因此，证据候选 context 之外的 KC、prediction prompt 等有效 token 在 `h_r` 和 `h_m` 中保持原始 `H`，在 `h_n` 中为零；padding 继续由 attention mask 排除。

### 2.4 B 模块与混合表征

B 只接收非证据表征：

```text
h_nb = B(h_n)
```

- `h_nb`：B 模块变换后的非证据表征。

混合表征定义为：

```text
h_m = h_r + β · h_nb
```

- `h_m`：混合表征。
- `β`：B 分支的混合强度。

B 的职责是改变非证据环境，而不是重新生成稳定标签证据。`h_m` 应保留 `h_r` 的任务信息，同时对非证据环境变化保持预测稳定。

## 3. 双路径后续预测

设 `P` 表示共享的后续网络，包括注入层之后的 Qwen blocks、最终 normalization、`lm_head` 和当前使用的输出校准器。

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

两条路径必须共享 `P` 的参数。这样，`p_r` 与 `p_m` 的差异主要来自输入表征，而不是来自两个不同预测头的容量差异。

当前单路径实现 `H + γ(a ⊙ H + β * environment)` 不等价于上述双路径流程，因为它始终保留完整原始 `H`，也没有产生可比较的 `p_r` 与 `p_m`。

## 4. 整体训练目标

### 4.1 Stage 1：A 模块目标

Stage 1 保留当前已经验证的加性证据增强：

```text
H' = H + γ(a ⊙ H)
p_A = P(H')
L_A = BCE(p_A, y)
```

这一阶段用于建立可影响 KT 预测的 A 模块。已审计的 A 参考结果包含 calibrator warmup 与严格联合训练，但这属于历史协议；后续默认训练按第 5 节统一端到端执行。

### 4.2 Stage 2：A+B 核心目标

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

即在二分类 Bernoulli 分布上计算 Jensen-Shannon divergence。`p_r` 与 `p_m` 使用同一 KC 聚合和同一 calibrator 后的最终样本级预测概率；计算前限制到 `[ε, 1-ε]`，按 batch 取均值。两条路径都保留梯度，不对任一路径执行 `detach`。

该项是针对当前 KT 目标中“证据预测与混合预测应接近”的要求增加的输出一致性约束。Faith-DARE 原实现使用两条任务预测损失并通过 InfoNCE 对齐表征，并未直接使用该 JS 损失。

Stage 2 的核心总损失为：

```text
L_total = λ_r L_r
        + λ_m L_m
        + λ_cons L_cons
```

三项含义分别是：

- `L_r`：要求证据表征本身足以完成 KT 预测。
- `L_m`：要求混合表征仍能正确预测。
- `L_cons`：要求非证据环境经过 B 变换并混入后，不改变证据路径的预测结论。

当前不把证据预算和 B 输出尺度约束加入 `L_total`。正式训练必须先记录 `a`/`w_r` 的分布与饱和情况、`h_nb` 的 RMS、`βh_nb` 对 `h_m` 的实际贡献以及 B 的梯度；只有观察到全证据坍缩或 B 输出失效，才讨论加入对应正则。

`P(h_n)` 的预测能力必须作为 reversal diagnostic 单独报告。只有当 `h_r` 具备预测能力、`h_n` 的独立预测能力下降、且 `p_r` 与 `p_m` 保持一致时，才能支持“证据充分、非证据环境不主导预测”的解释。

### 4.3 Calibrator 的必要性与训练边界

Calibrator 是位于共享预测头输出之后的可学习概率变换层：bias 模式执行 `sigmoid(logit(p) + b)`，affine 模式执行 `sigmoid(s * logit(p) + b)`，其中 `s > 0`。它只能调整概率的偏移、尺度和阈值行为，不学习 A 的证据选择，也不学习 B 的非证据变换。

因此，calibrator **不是 A/B 方法成立或 `L_total` 可优化的必要条件**。关闭 calibrator 时，`P` 直接输出经过 KC 聚合的概率，双路径目标仍然完整。calibrator 的价值仅在于需要改善概率校准、类别偏置或固定阈值下的决策表现时提供额外自由度；正的单调变换通常不改变排序，因此不能把 AUC 提升简单解释为表示学习收益。

统一训练协议下的默认规则是：

- 机制验证优先使用 `none`，或使用以恒等变换初始化的共享 calibrator 作为明确的可选组件；
- 一旦启用，calibrator 必须同时作用于 `p_r` 与 `p_m`（以及评估时的 `p_n`），从第一个 epoch 起纳入同一个 optimizer 和完整 `L_total`；
- 不得默认执行 calibrator-only 阶段、在 A/B 训练后单独调校，或让 calibrator 替代表示学习；这些只能作为标注清楚的后处理/消融；
- 报告 raw probability 与 calibrated probability，并记录 calibrator 参数、校准数据 split 和是否影响 AUC/Acc；
- 若校准器的单独训练改变了最终指标，该结果不能作为 A/B 核心机制的直接证据。

当前代码中存在的 `calibrator_warmup` 和“Stage 2 非首阶段必须启用 calibrator”约束属于历史 staged 实现，不是新的默认训练规范。历史工件保留用于溯源，但不得据此宣称完成了统一端到端训练。

## 5. 统一训练协议（默认）

除非某个模块明确声明并论证特殊训练机制，所有模块都遵循传统深度学习式的统一训练与验证：

1. 训练开始前固定 `max_epochs`、优化器、参数组学习率、验证指标、`patience` 和 `min_delta`。
2. A、B、LoRA、calibrator 以及后续新增的 C、D 等模块，从第一个 epoch 起作为同一个模型和同一个优化任务参与 forward、反向传播和 optimizer 更新。
3. 每个 epoch 完成统一的 train 与 validation；依据 validation 指标保存最佳 checkpoint，并在触发 `patience` 后早停，最后恢复最佳 checkpoint 再执行 final test。
4. 允许为不同模块设置不同 parameter-group learning rate，但这不构成独立训练、交替训练或阶段性冻结。
5. 默认不使用 `a_bootstrap`、`b_warmup`、calibrator-only、按 batch 的 loss ramp 或其他人为阶段切换。若确有模块特有机制，必须在方法文档和 manifest 中写明动机、冻结范围、损失/日程、启用条件及审计判据，并将其结果标记为特殊协议。

该统一协议是后续新增模块的继承规则：新增模块默认加入现有完整计算图、优化器和验证流程，只有用户或方法设计明确指定的特有训练方式才可例外。

实现状态：训练 CLI 现已使用 `--max_epochs`、`--patience` 和 `--min_delta`（`--epochs` 为兼容别名），并按 validation loss 保存最佳 checkpoint、支持 patience 早停；旧阶段运行仍只能按历史协议记录。

## 6. 训练顺序（历史/特殊协议说明）

历史 A+B candidate 曾从 Qwen3-1.7B 原始基座开始，并按以下阶段形成自包含训练链：

1. 初始化当前 candidate 的 LoRA、A 和 calibrator。
2. 使用加性证据增强路径训练 A。
3. 进行 calibrator warmup。
4. 对 A、LoRA 与 calibrator 做严格联合训练，得到当前 candidate 自身的 A 阶段 checkpoint。
5. 初始化 B，并使用 `h_r/h_m` 双路径目标训练 B 与后续网络。
6. 以分组学习率联合微调 A、B、LoRA 与 calibrator。
7. 完成 final test、双路径诊断、来源审计和参数变化审计。

这套分阶段链仅用于解释既有代码和工件，不能作为新的默认协议。除非另有明确的机制假设，今后的正式 A+B（及 A+B+C 等）应直接采用上一节的统一端到端训练。

## 7. B 模块设计要求

B 的正式设计必须满足：

- 输入和输出均保持 `B x S x d`，与 `h_n`、`h_r` 可直接组合。
- 只处理有效 context 中的 `h_n`，不得读取答案标签或 prediction token 的未来信息。
- 具备与任务所需变换相匹配的表达能力，能够建模必要的 token 间关系和上下文依赖。
- 输出经过稳定的 normalization 与投影，并持续报告其相对尺度；是否增加显式尺度正则由训练诊断决定。
- 能报告参数变化、梯度、输出尺度、有效 token 覆盖率以及 `p_r/p_m` 一致性。

Shuffle、极小 MLP 或 tiny Transformer 可以作为接口检查和消融对照，但不能作为该方向的充分实现，并据此否定 B 模块的研究假设。

当前正式 B 主方向是有充分容量的 sequence-level contextual Transformer。首轮的一层 bottleneck Transformer 仅作为历史 pilot 和容量消融，不作为新主候选的结构定稿或参数来源。

## 8. 方法设计原则

方法设计应首先服从研究假设和目标因果机制，而不应以“最小改动、最少参数或单次训练成本最低”作为正式方案的首要目标。对于每一个阶段，应从目标机制反向设计完整且功能闭合的模块，明确其输入输出表征、模块之间的信息流、下游使用方式、训练目标、优化日程以及可验证的诊断指标。模块容量应与所要建模的变换复杂度相匹配，不能用过小或过于简化的模块替代核心方法，再据此判断研究方向是否有效。轻量模块可以用于接口检查、sanity check 或消融实验，但不应作为核心方法的主要证据。不应通过削弱核心模块的表达能力来节省正式训练。参数效率应在机制得到充分验证之后再进行优化。

该原则不等于盲目增加参数。正式模块应追求容量充分、结构合理和目标闭合；参数规模本身不是目标。

## 9. 当前实验解释边界

当前已审计的 A 模块参考结果为 Overall Acc/AUC `70.03 / 75.38`，Final Acc/AUC `66.80 / 77.26`。其历史 checkpoint 名称中包含 `v26_selftrained`，该名称仅用于工件溯源；方法正文统一称为 **A 模块**。

已完成的 Stage 2 首轮三方向采用的是旧的单路径 environment residual 注入流程。它们能够用于分析 hook timing、B 输出尺度和训练稳定性，但没有实现 `h_r/h_m` 双路径预测与一致性目标，因此不能作为本文目标流程的正式验证，也不产生 Stage 2 winner。
