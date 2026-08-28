# CEL Stage1 A 模块实现方案

本文只覆盖 `CEL_Modular_Flow.md` 里的 **Stage 1 / A 模块** 三种 selector：`A1 MLP gate`、`A2 Adapter gate`、`A3 Task-conditioned selector`。这一阶段必须做 `h_r` 注入，否则只能验证 gate 分数，不能验证 selector 是否真正改变了模型推理。

## 0. 先回答两个关键问题

- **必须注入 `h_r`**：Stage1 的目标不是“看 selector 输出是否像样”，而是检验 selector 是否真的会改变下游 KT 预测。若不把 `h_r` 注入回主干，最终预测仍然是原始 LLMKT 路径，等于没有测试方法有效性。
- **预测仍然用原始 `lm_head`**：注入后的 hidden state 继续走后续 Transformer 层，最后仍由原模型的 `lm_head` 输出 `True/False` logits。也就是说，方法只改中间表示，不换分类头。
- **先只用 `BCELoss`**：Stage1 先固定为 `L_total = BCELoss(corr_probs, labels)`，不加 coverage / entropy / alignment / sparsity 等额外项。等三种方法跑通并观察到训练现象后，再二轮加正则。

## 1. Stage 1 的统一目标

Stage 1 统一采用下面的链路：

```text
input prompt
→ Qwen forward 到第 k 层
→ H_k
→ A(H_k, optional conditions)
→ α_sel
→ H_r = α_sel ⊙ H_k
→ H_k^r = Norm(H_k + γ · H_r)
→ 从第 k+1 层继续前向
→ final hidden state
→ lm_head
→ True/False logits
```

也就是说，**预测必须在注入后的模型上完成**。如果不做这一步，就无法判断 A 模块是否真的改善/破坏了 KT 推理。

## 2. 当前代码下需要改的文件

| 文件 | 修改内容 |
|---|---|
| `dialogue_kt/main.py` | 新增 `--cel_mode`、`--cel_layer_idx`、`--cel_gamma`、`--cel_selector_hidden_dim`、`--cel_query_mode`、`--cel_kc_pooling` 等参数。 |
| `dialogue_kt/models/lm.py` | 增加支持“到第 k 层截断并继续前向”的 Qwen wrapper，或提供分层 forward helper。 |
| `dialogue_kt/training.py` | 新增 CEL 训练/测试路由，计算 `BCELoss`，保存 selector 和 LoRA checkpoint。 |
| `dialogue_kt/kt_data_loading.py` | 补充 `kc_token_ranges`、`dialogue_token_ranges`、`context_token_mask`，供 A3 使用。 |
| `dialogue_kt/cel_methods.py` | 新增 selector 模块、pooling、注入函数、KC/query 构造工具。 |
| `dialogue_kt/prompting.py` | Stage 1 不改 prompt。保持原始 LLMKT 输入，避免混杂变量。 |

## 3. 共用实现方式

### 3.1 中间层截取与注入

当前 `get_model()` 直接返回 `AutoModelForCausalLM + LoRA`，还不能天然“停在第 k 层再改 hidden state”。因此需要在 `models/lm.py` 里加一个 helper，做两种之一：

1. 返回 `transformer` / `model.model.layers` 级别的 wrapper；
2. 通过 `forward(..., output_hidden_states=True)` 先拿到 `hidden_states`，然后用一个分层 replay helper 从第 `k+1` 层继续跑。

Stage 1 推荐优先实现 **分层 replay helper**，改动更小，便于先验证效果。

### 3.2 只保留一个主损失

这一阶段先只用：

```text
L_total = BCELoss(corr_probs, labels)
```

不先加 coverage / entropy / alignment / adapter regularization。等三种方法都跑通后，再看训练是否出现：

- gate 全 0 / 全 1
- loss 降但 AUC 不升
- 过拟合过快

再决定是否补正则项。

## 4. A1: MLP gate

### 实现

在 `dialogue_kt/cel_methods.py` 新增 `MLPGateSelector`：

```python
α_sel = sigmoid(MLP(H_k))
```

建议只对 `context_token_mask` 覆盖到的 token 产出 gate，KC continuation token 默认置 0，避免标签泄漏。

### 预测方式

用注入后的 `H_k^r` 继续前向，最终在原始 `lmkt` 的 `True/False` token 上取 logits 做预测。


### 训练注意事项

- 这是最适合做 baseline 的 A 方法。
- 先用 `--debug` 确认 replay 后的 logits、loss、AUC 都正常。
- 如果注入后梯度不稳定，先把 `γ` 设小一些，例如 `0.1~0.3`。

## 5. A2: Adapter gate

### 实现

在 `dialogue_kt/cel_methods.py` 新增 `AdapterGateSelector`：

```python
Z = H_k + Adapter(H_k)
α_sel = sigmoid(Linear(Z))
```

或者 `Adapter(H_k)` 后接一个 gate head。Adapter 只用于生成 selector，不直接承担最终分类。

### 预测方式

和 A1 一样，先构造 `H_r`，再注入并 replay 后续层，最后用 `lm_head` 预测。

### 损失

只用 `BCELoss`。

### 训练注意事项

- 这是最可能比 A1 更强的版本，但也更容易过拟合。
- 如果后续发现 selector 输出塌缩，再补 `entropy` 或 `coverage` 正则，不要一开始就加。

## 6. A3: Task-conditioned selector

### 实现

在 `dialogue_kt/kt_data_loading.py` 里补出两类信息：

- 当前 KC 的 token span
- 只允许证据来自 dialogue history 的 context mask

在 `dialogue_kt/cel_methods.py` 新增 `TaskConditionedSelector`：

```python
q = f([h_answer; h_kc])
score_t = q^T W H_k,t
α_sel = sigmoid/softmax(score)
```

其中：

- `h_kc` 建议对 KC span mean pooling
- `h_answer` 取当前预测位置 hidden state
- `context_mask` 只在 dialogue 区域内生效

### 预测方式

先用 query-conditioned gate 得到 `H_r`，再注入并 replay 后续层，最后预测 `True/False`。

### 损失

只用 `BCELoss`。

### 训练注意事项

- 这版最贴近任务语义，建议在 A1 跑通后上。
- 一定检查 KC span 对齐，否则 query 会被错误 token 污染。

## 7. `models/lm.py` 需要新增的能力

为了支持三种方法都“真的注入后再预测”，建议在 `dialogue_kt/models/lm.py` 里补一个明确的分层执行接口，例如：

```python
hidden_k = forward_to_layer(input_ids, attention_mask, position_ids, layer_idx=k)
hidden_k_r = inject_rationale(hidden_k, alpha_sel, gamma)
logits = forward_from_layer(hidden_k_r, start_layer=k+1, ...)
```

如果不方便拆成两个函数，至少要有一个 CEL 专用 wrapper，能接受外部传入的 `hidden_states_override`。

## 8. 训练流程建议

1. 先只做 A1，确认 replay chain 正常。
2. 再做 A3，确认 query-conditioning 带来可解释变化。
3. 最后做 A2，对比 adapter 是否优于简单 MLP。

统一训练设置建议保持当前 LLMKT 基线：

- `Qwen3-1.7B + LoRA`
- `epochs=2`
- `pack_kcs=True`
- `quantize=0`
- 先 `--debug` 再正式训练

## 9. 结果记录

建议结果目录放在 `results/cel_stage1/`，至少保存：

- `metrics/`
- `kcs/`
- `qual/`
- 训练日志
- selector checkpoint

阶段 1 先只比较：

- Overall AUC
- Final Turn AUC

等三种方法的注入链路稳定后，再补 gate 稀疏度、熵、覆盖率等分析指标。
