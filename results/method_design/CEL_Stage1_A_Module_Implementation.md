# CEL Stage 1 A 模块统一实现说明

## 1. 模块定位

A 是面向当前 Knowledge Component 的 token-level selector。它接收 Qwen 中间层 hidden state `H`，并为有效对话 token 输出选择强度 `a`。

统一定义：

- `H`：Qwen 中间层 hidden state。
- `A`：selector。
- `a`：A 输出的 token-level 选择强度。
- `a` 经过 `tanh`，范围约为 `[-1, 1]`。

方法说明中不再使用 `score`、`signal`、`gate` 或 `delta` 指代 `a`。源码中的历史命名暂不强制修改，以保持 checkpoint、CLI 与结果字段兼容。

## 2. A 的输入与输出

A 使用三类信息：

```text
H
当前 prediction 位置表示
当前 Knowledge Component 表示
```

其抽象计算为：

```text
a = tanh(A(H, h_answer, h_kc, context_mask))
```

`context_mask` 只允许对话上下文 token 参与选择，避免将答案 continuation 或无效 padding 当作证据。

当一个样本对应多个 KC 时，A 分别获得各 KC 条件下的 `a`，再在样本内进行聚合。聚合后的结果仍统一记为 `a`。

## 3. 当前 A 模块的实际前向

A 生成的证据表征为：

```text
a ⊙ H
```

当前 Stage 1 的实际注入为：

```text
H' = H + γ(a ⊙ H)
```

随后：

```text
H'
-> Qwen 后续网络
-> lm_head
-> True/False logits
-> mastery probability p_A
```

因为原始 `H` 被保留，这一机制统一称为 **加性证据增强**。它验证的是 A 生成的证据表征能否通过中间层干预改善 KT 预测，而不是显式删除非证据 token。

当前有效注入必须发生在 prediction token 仍可接收 context 信息的位置。历史 `post_block + token_residual` 路径对最后一个 prediction token 不产生有效影响；后续正式实现应使用具备因果作用的注入位置，并通过 prediction delta 与 A 梯度合约验证。

## 4. Stage 1 训练目标

Stage 1 的主目标为：

```text
p_A = P(H')
L_A = BCE(p_A, y)
```

其中 `P` 表示注入点之后的共享 Qwen 网络、`lm_head` 与当前输出校准器。

Stage 1 不使用 B，也不在该阶段引入 `h_r/h_m` 一致性损失。它只回答：

> A 是否能产生可影响最终 KT 预测的证据表征，并形成可复用的选择强度 `a`？

## 5. 训练日程

当前 A 参考结果使用过以下历史分阶段训练做法，但它只用于工件溯源，不是后续新增模块的默认协议：

1. **A bootstrap**
   - 从 Qwen3-1.7B 原始基座开始。
   - 新建当前 candidate 的 LoRA 与 A。
   - 使用加性证据增强路径训练。
2. **Calibrator warmup**
   - 加载同一 candidate 的 A bootstrap checkpoint。
   - 冻结 A、LoRA 与其他主体参数，只训练输出校准器。
3. **Strict joint**
   - 加载同一 candidate 的 warmup checkpoint。
   - 以分组学习率联合训练 A、LoRA 与 calibrator。
4. **Final test and audit**
   - 完成 test、来源审计、冻结审计和参数变化审计。

禁止将 baseline、历史 A anchor 或其他 candidate checkpoint 作为正式 A 模块的参数来源。

### 5.1 Calibrator 与统一训练规则

输出 calibrator 是共享预测头后的可选概率校准层，不是 A 的证据选择学习所必需。关闭时，`P` 直接输出任务概率；启用时，calibrator 只做 bias 或正 affine 的单调概率变换。

后续正式训练默认使用统一的 `max_epochs`、validation、`patience` 和 `min_delta` 流程。A、LoRA、calibrator 以及任何新增模块从第一个 epoch 起进入同一计算图、同一 optimizer 和完整损失。不得默认执行 calibrator-only warmup 或训练后单独校准；若作为校准消融使用，必须与 A 的表示学习结果分开报告 raw/calibrated 概率、校准指标和验证 split。

## 6. 与目标 A+B 流程的接口

Stage 2 不再另建一套选择变量，而是直接将 `a` 线性映射为互补权重：

```text
w_r = (a + 1) / 2
w_n = (1 - a) / 2
h_r = w_r ⊙ H
h_n = w_n ⊙ H
```

其中：

- `h_r`：证据表征。
- `h_n`：非证据表征。
- `h_r + h_n = H`。

互补权重只作用于可选证据的 context token。非 context 的有效 prompt token 在 `h_r` 中保持原始 `H`，在 `h_n` 中为零。

当前 A 的加性证据增强与目标互补分解承担不同职责：

- `a ⊙ H` 用于 Stage 1 已验证的加性增强。
- `h_r/h_n` 用于 Stage 2 的显式证据/非证据双视图训练。

二者都由同一个 `a` 产生，不能再混写为两套 selector 输出。

## 7. 必须记录的诊断

正式 A 模块至少记录：

- `a` 的 masked mean、absolute mean、正负比例与极值。
- `a` 在有效 context 中的饱和比例。
- 证据权重 `w_r` 的均值、分布和预算偏差。
- A 参数是否获得非零梯度并在训练阶段发生变化。
- 注入前后 prediction token 的 logits/probability delta。
- Overall、Final 与 non-final 的 Acc/AUC。
- 校准器参数及校准前后的概率分布。

历史日志中的 `gate_*` 字段应解释为 `a` 的统计量；字段名仅为兼容性保留。

## 8. 当前参考结果

当前已审计的 A 模块参考结果：

| Split | Acc | AUC |
|---|---:|---:|
| Overall | `70.03` | `75.38` |
| Final | `66.80` | `77.26` |

历史 checkpoint 的正式文件名为 `cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b`。该标识仅用于结果和来源追溯；当前方法统一称为 **A 模块**。

## 9. 实现位置

- `dialogue_kt/cel_methods.py`
  - A 的网络结构、context mask、注入与 hook。
- `dialogue_kt/training.py`
  - A、LoRA、calibrator 的初始化、冻结、训练、保存和审计信息。
- `dialogue_kt/main.py`
  - 相关 CLI 参数。
- `scripts/cel_stage1_last_layer/`
  - 历史训练链、同步、finalize 与 audit 工具。

## 10. 方法设计原则

方法设计应首先服从研究假设和目标因果机制，而不应以“最小改动、最少参数或单次训练成本最低”作为正式方案的首要目标。对于每一个阶段，应从目标机制反向设计完整且功能闭合的模块，明确其输入输出表征、模块之间的信息流、下游使用方式、训练目标、优化日程以及可验证的诊断指标。模块容量应与所要建模的变换复杂度相匹配，不能用过小或过于简化的模块替代核心方法，再据此判断研究方向是否有效。轻量模块可以用于接口检查、sanity check 或消融实验，但不应作为核心方法的主要证据。不应通过削弱核心模块的表达能力来节省正式训练。参数效率应在机制得到充分验证之后再进行优化。

对 A 而言，这意味着正式设计必须先满足 task-conditioned evidence selection 的因果路径、表示接口和可训练性，再讨论参数压缩。低容量 A 的失败不能直接否定 A 模块方向。
