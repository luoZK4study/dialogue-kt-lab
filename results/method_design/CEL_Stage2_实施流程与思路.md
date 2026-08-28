# CEL Stage 2 实施流程与思路

本文档说明在已经完成 A 模块的基础上，如何构造和训练 Stage 2 的 A+B 方法。文档只描述表示契约、信息流、训练技术和实验口径，不给出具体实现代码。

## 1. 方法定位

当前任务是 MathDial ATC Dialogue-KT，主模型为 Qwen3-1.7B，参数高效训练采用 LoRA。A 模块已经验证了 task-conditioned token-level evidence selection 的可行性；Stage 2 的问题不是重新训练一个 selector，而是利用 A 产生的选择强度 `a`，把中间表示显式拆成证据视图和非证据视图，再学习一个非证据环境变换 B。

Stage 2 要验证的因果假设是：

1. 证据表示本身足以完成知识追踪预测；
2. 对非证据环境进行 B 变换后，混合表示仍应保持同一预测；
3. 只使用非证据表示时，预测应表现为 reversal，而不应携带完整的证据决策。

因此，旧的“把 A、B residual 一起加回完整 `H`，只算一次预测”的单路径流程只作为历史 pilot，不是当前 Stage 2 定义。

## 2. 接上 A 后的整体流程

### 2.1 输入与任务条件

每个样本先按现有 MathDial prompt 流程构造 Qwen 输入，并保留：

- 对话 context token 的范围；
- prediction token 的位置；
- 当前样本对应的 Knowledge Component 条件；
- True/False 标签及同一 dialogue 内的 KC 聚合关系。

task-conditioned A 对每个 KC 条件使用相同的中间层 hidden state `H`，最后在样本内聚合得到 token-level 选择强度 `a`。只允许 context token 参与选择；答案 continuation、padding 和其他非证据位置不参与分解。

### 2.2 A 模块的已完成路径

A 的输出范围由 `tanh` 约束在 `[-1, 1]`。Stage 1 已验证的加性证据增强为：

`H' = H + γ(a ⊙ H)`

这条路径继续用于 A bootstrap 和 A joint，使 A 先学会对当前 KC 有效的证据选择。随后仍使用 Qwen 后续网络、True/False prediction head 和输出 calibrator 得到 A 阶段概率。

### 2.3 Stage 2 的互补表示

Stage 2 不另建 selector，而是把同一个 `a` 线性映射成互补权重：

`w_r = (a + 1) / 2`

`w_n = (1 - a) / 2`

`h_r = w_r ⊙ H`

`h_n = w_n ⊙ H`

在可选择的 context token 上，`w_r + w_n = 1`，所以 `h_r + h_n = H`。其他有效 prompt token 在 `h_r` 中保持原始 `H`，在 `h_n` 中置零。这样可以区分“证据视图”与“被移出的非证据视图”，而不是用 top-k 产生不连续的二值切分。

### 2.4 B 环境变换与混合表示

B 只接收 `h_n` 和有效 context mask，输出：

`h_nb = B(h_n)`

再用一个显式的幅度系数构造混合视图：

`h_m = h_r + β · h_nb`

B 不读取标签，也不读取 prediction token 的未来信息。B 的输出在进入混合前做 mask 和数值检查，并记录相对于 `h_n`、`h_r` 的 RMS 比例；正式主候选使用 centered-RMS 对齐，避免 B 输出消失或淹没证据表示。

### 2.5 共享 Qwen 后缀和三条评估路径

在最后一个 Qwen block 的输入处同时建立三条分支：

- evidence：输入 `h_r`，得到 `p_r = P(h_r)`；
- mixed：输入 `h_m`，得到 `p_m = P(h_m)`；
- reversal：评估时输入 `h_n`，得到 `p_n = P(h_n)`。

三条路径共享同一个 Qwen 后缀、True/False head、KC 聚合方式和 calibrator。训练时只需要 evidence 与 mixed 两条分支；reversal 只在 validation/test 中计算，以免它改变主目标。低风险 batch 沿 batch 维展开 evidence/mixed 的输入和 attention/position 参数；高风险 batch 则串行执行两条共享后缀路径并精确拆分梯度。两种执行方式都不复制 Qwen 参数，也不会产生不同预测头。

## 3. 各步骤采用的技术

| 步骤 | 技术选择 | 目的与约束 |
|---|---|---|
| 中间表示获取 | Qwen3-1.7B 最后一个 Transformer block 的 pre-block hook | 保证修改仍能经过 Qwen 的自注意力和后续输出层；避免在已完成预测之后注入而失去因果作用 |
| A selector | task-conditioned MLP selector，输入 `H`、prediction 位置表示和 KC 表示，输出 token 标量 `a`，末端 `tanh` | 复用已验证的 A；保持 KC-specific evidence selection，不引入第二套选择变量 |
| 互补分解 | 连续线性互补权重 `(a+1)/2` 与 `(1-a)/2` | 保证表示守恒、可微和可审计；只在 context mask 内生效 |
| B 模块 | sequence-level contextual Transformer；4 个 pre-norm attention/FFN 层，内部宽度 1024，8 heads，FFN 4096 | 建模跨 token、跨 turn 的非证据环境关系；容量足以表达目标变换，tiny MLP/Transformer 只作接口或消融 |
| B 的数值接口 | RMSNorm、有效 token mask、FP32 B 参数、输出投影回 Qwen hidden size、centered-RMS 输出对齐、训练时 gradient checkpoint | 处理空非证据区域、padding、混合精度、尺度坍缩和长序列显存 |
| 双路径 forward | 以 `rows * sequence_length^2` 评估注意力风险；低于 `8000000` 时展开 evidence/mixed，高风险时串行执行并精确拆分共享梯度；共享 Qwen suffix 和 lm_head | 让 `p_r`、`p_m` 来自同一预测器，同时使 24GB GPU 能覆盖完整训练集的最坏长序列 batch |
| 任务损失 | 两条路径各自使用 BCE；二分类分布上使用 Jensen-Shannon divergence | 保证证据路径可预测、混合路径可预测，并显式约束输出一致 |
| 输出校准 | 可选的、与两条路径共享的 monotonic bias/affine calibrator | 只做概率偏移/尺度校准，不改变路径相对顺序；启用时从 epoch 1 与其他模块联合训练 |
| 训练优化 | 默认由 LoRA、A、B 及已启用的 calibrator 使用一个 optimizer 统一端到端训练；允许 parameter-group learning rate | 保持完整机制在同一训练目标中协同优化，避免默认人为拆分训练 |

## 4. Stage 2 训练目标

对同一批样本和同一标签 `y`，先得到经过共享 KC 聚合和 calibrator 的 `p_r`、`p_m`。核心目标为：

`L_r = BCE(p_r, y)`

`L_m = BCE(p_m, y)`

`L_cons = JS([p_r, 1-p_r], [p_m, 1-p_m])`

`L_total = λ_r L_r + λ_m L_m + λ_cons L_cons`

当前首轮不加入 `L_budget` 或 `L_scale`。先记录 `a`/`w_r` 分布、`h_nb` 输出尺度和 `βh_nb` 的实际贡献；只有出现全 context 被判为证据或 B 输出/梯度长期为零时，才针对性加入正则。这样可以先判断核心机制本身，而不是让额外正则掩盖问题。

## 5. 默认统一训练与历史阶段链

### 5.1 默认统一训练协议

除非新增模块明确声明特有训练机制，所有模块默认从第一个 epoch 起参与同一 forward、同一 `L_total`、同一 optimizer 和同一 validation 流程。训练开始前固定 `max_epochs`、validation 指标、`patience` 和 `min_delta`；每个 epoch 完成 train/validation，保存最佳 checkpoint，触发 patience 后早停并恢复最佳 checkpoint，再执行 final test 和 audit。

当前训练 CLI 已实现 `--max_epochs`、`--patience`、`--min_delta` 与 validation-best checkpoint 早停，`--epochs` 仅为兼容别名；现有五阶段结果仍必须按历史 staged protocol 解释。

允许为 A、B、LoRA、calibrator 或未来 C/D 等模块设置不同 parameter-group learning rate，但这不构成独立训练。默认不使用 bootstrap、warmup、冻结、交替更新或按 batch 的 loss ramp。

Calibrator 在该协议中是可选组件：它不参与 A/B 表示的形成，不是双路径目标的必要条件；启用时必须共享于 `p_r/p_m`（评估时含 `p_n`）并从 epoch 1 联合训练。训练后单独校准只能作为明确标记的后处理/消融，且单调校准不会改变 AUC。

### 5.2 同一 candidate 的历史阶段链（仅作溯源）

历史 candidate 必须从 raw Qwen3-1.7B 独立开始，只能读取自己上一阶段的 checkpoint：

1. **A bootstrap**：新建 LoRA 和 A，使用 Stage 1 加性证据增强训练。
2. **Calibrator warmup**：冻结 Qwen、LoRA、A，只训练共享 bias calibrator。
3. **A joint**：加载同一 candidate 的上一阶段，使用分组学习率联合训练 A、LoRA 和 calibrator；此时仍是 A 的加性路径。
4. **B warmup**：新建 B，冻结 A、LoRA、Qwen 和 calibrator，只训练 B；逐步把 `β` 从目标值的 0.2 提升到目标值，并逐步加入一致性项。
5. **A+B joint**：从 B warmup checkpoint 加载 A、B、LoRA、calibrator，使用完整 `L_total` 联合训练。
6. **Evaluation/Audit**：保存 validation 最优的 joint checkpoint，分别评估 `p_r`、`p_m` 和 `p_n`，再做来源、参数、tensor-contract 审计。

历史阶段链的 B 配置为 4 层、1024 hidden、8 heads、4096 FFN、FP32；对应默认 `β=0.10`，`λ_r=1.0`，`λ_m=1.0`，`λ_cons=0.10`。这些 warmup 和学习率关系只描述历史运行，不覆盖 5.1 的默认协议。

## 6. 三轮实验口径

本轮执行按已经确认的口径进行：

- 三轮都使用 `model seed=data seed=1221`，不额外运行其他 seed；因此三轮是配置探索轮次，不宣称 multi-seed 方差结论。
- 历史 Round 1 使用上述默认容量、尺度、`β`、损失权重和 warmup 日程；该记录不代表已采用 5.1 的统一协议。
- Round 1 完成后只看 validation 和训练诊断：A 饱和、互补误差、B 输出 RMS、`p_r/p_m` 差异、JS、梯度与 reversal。根据这些证据调整 Round 2。
- Round 2 从 raw Qwen 独立重新开始，不继承 Round 1 权重；仍只用 validation 决定是否调整 `β`、B 输出比例、`λ_cons`、学习率或 warmup。
- Round 3 同样从 raw Qwen 独立开始。配置冻结后，不再根据任何 test 数值回调参数。
- Round 3 配置冻结后，统一对 Round 1、Round 2、Round 3 的最终 joint checkpoint 执行 final test，再生成三轮对照、审计和 Stage 2 结束结论。

因此，validation 用于选择配置，final test 只用于最终报告；历史 Shuffle、token-wise MLP 和 small Transformer 结果不混入这三轮比较，因为它们属于旧的单路径 pilot。

## 7. 必须观察的诊断

### 7.1 表示与选择

- `a` 的 masked mean、正负比例、绝对值均值和饱和比例；
- `w_r`、`w_n` 的分位数，以及 `h_r + h_n - H` 的绝对最大误差、相对 RMS 误差和权重互补误差；其中 BF16 下绝对最大误差会被 hidden-state 极值放大，审计以尺度无关的相对 RMS 与权重误差为门槛；
- `h_r`、`h_n`、`h_nb`、`βh_nb`、`h_m` 的 RMS、有限值比例和余弦关系；
- A、B、LoRA、calibrator 是否获得非零梯度并实际更新。

### 7.2 预测与因果路径

- `p_r`、`p_m` 的 Overall/Final/non-final Acc/AUC；
- `|p_r-p_m|`、logit 差和 JS 是否下降；
- `p_n` 的 reversal performance 是否明显低于 evidence/mixed；
- hook 前后的 prediction-token 输出变化，确认修改真正传播到任务位置。

### 7.3 诊断驱动的调整规则

- 若 `a` 迅速饱和且 `w_n` 接近零，优先检查 A 学习率和 warmup，再考虑 evidence budget；
- 若 `h_nb`/`βh_nb` 接近零且 B 梯度也接近零，优先检查输出初始化、centered-RMS 比例和 B 学习率，再考虑尺度正则；
- 若 `p_r` 与 `p_m` 很一致但同时错误，不能把一致性当作成功，应优先提高两条路径的任务损失有效性；
- 若 `p_m` 正确但与 `p_r` 差距大，说明 B 改变了决策边界，应调整 `β` 或一致性权重；
- 若 reversal 与 evidence 几乎相同，说明非证据视图仍携带过多决策信息，应检查 mask、B 输入和 prediction-token 处理。

## 8. 完成判据

Stage 2 只有在以下条件都满足后才算完成：

1. 三轮 candidate 的五阶段来源链和参数冻结关系通过 audit；
2. validation/test 均能产出 evidence、mixed、reversal 三条路径报告；
3. tensor contracts、有限值、梯度传播和 prediction-token 因果检查通过；
4. final test 结果与 A 模块参考值 `75.38` 及论文目标 `76.71` 的差异被明确报告；
5. 在没有额外 seed 的当前执行口径下，结论只表述为配置轮次观察，不夸大为跨 seed 稳健性结论。

在上述闭环完成前，不启动 Stage 3，也不把旧单路径 pilot 宣布为 Stage 2 winner。
