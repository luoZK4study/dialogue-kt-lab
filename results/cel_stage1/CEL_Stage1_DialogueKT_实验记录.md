# CEL Stage1 A模块实验记录

## 0. 实验目标

验证 CEL（Counterfactual Evidence Learning）Stage1 的三种 rationale selector 在 Dialogue-KT 任务上的有效性。  
Stage1 统一使用 `h_r` 注入 + `BCELoss`，暂不加额外正则项。

**核心问题**：selector 能否通过选择性增强 dialogue token hidden state 来改善 KC 预测？

## 1. 方法定义

三种 selector 均工作在第 k 层（默认 layer_idx=12），从 `H_k` 生成逐 token gate `α_sel`，然后注入回主干：

```
H_r = α_sel ⊙ H_k
H_k' = Norm(H_k + γ · H_r)
→ 继续前向至 lm_head → True/False logits
```

### A1. MLP Selector

```python
z = GELU(Linear(H_k))
gate = sigmoid(Linear(z))
```

参数量较大，直接从 token hidden state 学习 gate。

### A2. Adapter Selector

```python
z = H_k + up(dropout(GELU(down(H_k))))
gate = sigmoid(Linear(z))
```

Bottleneck 结构，参数效率更高。

### A3. Task-conditioned Selector

```python
q = query_proj(concat[h_answer, h_kc])
z = tanh(q · token_proj(H_k))
gate = sigmoid(score(z))
```

逐 token 与 task query 交互，更贴合"evidence 应与当前 KC 相关"的直觉。

## 2. 实验配置

- **模型**: Qwen3-1.7B + LoRA (r=16, alpha=16)
- **数据集**: MathDial ATC
- **训练**: 2 epochs, lr=2e-4, wd=1e-2, batch_size=1, grad_accum=32
- **CEL 参数**: `layer_idx=12`, `gamma=1.0`, `cel_selector_hidden_dim=512`, `cel_adapter_dim=256` (仅 adapter)
- **Loss**: `BCELoss` only, no auxiliary regularization
- **Baseline**: Qwen3-1.7B Overall AUC **75.99%**
- **样本规模**: train `1782` 对话 / `8425` 样本点；val `447` 对话 / `2202` 样本点
- **保存机制**: 仅在每个 epoch 完整训练并跑完 validation 后，若 `val loss` 更优才保存 checkpoint 和自动进入 test；因此正式 `saved_models/metrics/qual/kcs` 不会在 epoch 中途出现

## 2.1 本轮实现与工程变更

- 已在 `dialogue_kt/cel_methods.py` 中实现三种 selector，并通过 layer forward hook 完成真实 `h_r` 注入。
- 预测链保持为“注入后的 hidden state → 后续 transformer 层 → 原始 `lm_head` → `True/False` logits”，没有额外分类头。
- 已在 `dialogue_kt/training.py` 中加入 `get_lmkt_loss_cel()` 路由，并保存/加载 `cel_selector.pt`。
- 已在 `dialogue_kt/kt_data_loading.py` 中补充 `dialogue_token_ranges`、`kc_token_ranges`、`prompt_kcs` 等字段，支撑 A3 query 构造。
- 已添加远端训练辅助脚本：
  - `scripts/cel_stage1/run_mlp_full.sh`
  - `scripts/cel_stage1/run_adapter_full.sh`
  - `scripts/cel_stage1/run_task_conditioned_full.sh`
  - `scripts/cel_stage1/queue_task_conditioned_full.sh`
  - `scripts/cel_stage1/sync_results_from_server.sh`
  - `scripts/cel_stage1/summarize_metrics.py`
- 当前仓库已进一步清理为 **baseline + CEL** 单线工作区：
  - 删除了 `results/` 下与 CEL 无关的历史方法结果
  - 删除了其他方法对应的代码与脚本分支
  - 将 CEL 输出路径统一到 `results/cel_stage1/{metrics,qual,kcs}`

## 3. Debug 结果（100 train / 25 val）

| Method | Overall AUC | Final AUC | Acc | F1 | Loss |
|---|---|---|---|---|---|
| MLP | 59.76 | 100.00 | 55.17 | 64.86 | 0.7409 |
| Adapter | 74.52 | 71.43 | 58.62 | 40.00 | 0.6744 |
| Task-conditioned | 63.81 | 85.71 | 51.72 | 36.36 | 0.7445 |

**Debug 阶段观察**：
- 三种方法的训练/测试/selector 加载闭环均已打通
- Debug 数据量太小，AUC 波动大，不具备结论性
- Adapter 在 debug 子集上明显优于另外两种方法，说明 bottleneck selector 值得优先做正式训练
- 正式训练将使用全量数据（~8425 train / ~2202 val）

## 4. 正式训练结果

| Method | Loss | Overall Acc | Overall AUC | Overall Prec | Overall Rec | Overall F1 | Final Acc | Final AUC | Final Prec | Final Rec | Final F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MLP | 0.6664 | 61.11 | 67.08 | 58.09 | 64.15 | 60.97 | 59.03 | 74.12 | 88.37 | 50.53 | 64.30 |
| Adapter | 0.6432 | 64.43 | 69.15 | 63.70 | 57.87 | 60.65 | 54.17 | 77.34 | 92.17 | 40.69 | 56.46 |
| Task-conditioned | 0.6403 | 64.28 | 68.29 | 71.92 | 40.32 | 51.67 | 41.36 | 76.15 | 94.05 | 21.01 | 34.35 |

三种方法的正式训练命令与配置已固定在 `scripts/cel_stage1/` 下，结果文件已归档到：

- `results/cel_stage1/metrics/`
- `results/cel_stage1/qual/`
- `results/cel_stage1/kcs/`

## 5. 结论

- 最佳 Overall AUC 来自 **Adapter**，为 **69.15**，较 Qwen3-1.7B baseline `75.99` 下降 **6.84**。
- 该结果未超过论文 baseline `76.71`，仍相差 **7.56**。
- 最佳 Final Turn AUC 来自 **Adapter**，为 **77.34**。
- 按 Overall AUC 排序：Adapter (69.15), Task-conditioned (68.29), MLP (67.08)。

## 6. 相关文件

- 实现方案: `results/method_design/CEL_Stage1_A_Module_Implementation.md`
- CEL 完整设计: `results/method_design/CEL_Modular_Flow.md`
- 代码模块: `dialogue_kt/cel_methods.py`
- 训练路由: `dialogue_kt/training.py` (L251-329, L1170+)
- CLI: `dialogue_kt/main.py` (L70-77)
