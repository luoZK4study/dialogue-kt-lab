# Stage 2 A+B 方法交接

> 本文件是当前 Stage 2 的最小权威入口。目标 A+B 双路径 forward、训练目标、tensor contracts 和审计链已完成；当前按 seed=1221 执行三轮配置实验，不启动 Stage 3。Round 1/2 已完成 validation 与 audit，Round 3 正在运行。

## 1. 当前结果状态

- 任务：MathDial ATC Dialogue-KT，比较 True/False token logits。
- 模型：Qwen3-1.7B + LoRA。
- 论文目标：Overall AUC `76.71`。
- Baseline：Overall Acc/AUC `69.22 / 74.95`，Final Acc/AUC `61.17 / 75.32`。
- A 模块参考结果：Overall Acc/AUC `70.03 / 75.38`，Final Acc/AUC `66.80 / 77.26`。

A 模块参考结果对应的历史 checkpoint 名称为 `cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b`。该名称仅用于工件与审计溯源；方法正文统一称为 **A 模块**。

### 当前双路径实验快照

> 以下运行进度是 `2026-08-17 18:45 CST` 的远端只读快照。训练链会继续推进；新对话接管时必须先重新检查 SSH alias `3090`，不得仅依据 PID 或百分比判断当前阶段。

- Goal 状态：`active`，不是 `blocked`，不需要执行 `/goal resume`。
- 唯一 supervisor：PID `3641318`，命令为 `scripts/cel_stage2_environment/run_dual_path_three_round_chain.sh`。
- Round 3 launcher：PID `3733118`；当前 `b_warmup` 进程：PID `3789645`。
- Round 1、Round 2 的训练、validation 和 validation audit 均已完成；三轮 final test 按冻结协议统一延后执行。
- Round 3 的 `a_bootstrap`、`calibrator_warmup` 和 `a_joint` 已完成。`b_warmup` 训练已完成 `8425/8425`，耗时约 `1:31:38`；快照时阶段内 validation 为 `1112/2202`（约 `50.5%`）。
- 快照检查未发现 OOM、Traceback、RuntimeError 或 NaN/Inf；不要启动第二个 supervisor 或 Round 3 launcher。

已完成的双路径 validation 指标如下；主路径为 mixed：

| Round | 路径 | Overall Acc/AUC | Final Acc/AUC |
|---|---|---:|---:|
| 1 | Mixed | `66.78 / 72.51` | `61.20 / 73.46` |
| 1 | Evidence | `66.84 / 72.52` | `61.45 / 73.51` |
| 1 | Reversal `P(h_n)` | `52.71 / 46.94` | `25.78 / 40.80` |
| 2 | Mixed | `65.93 / 72.59` | `64.82 / 73.01` |
| 2 | Evidence | `66.10 / 72.53` | `64.58 / 72.89` |
| 2 | Reversal `P(h_n)` | `52.65 / 47.40` | `25.54 / 41.18` |

Round 1/2 的 Mixed Overall AUC 均低于 A 模块参考 `75.38` 和论文目标 `76.71`。当前只能确认工程链与审计链正常，不能宣布 Stage 2 研究成功或关闭。

Stage 2 首轮三个历史 candidate 均通过 audit：

| 历史方向 | Overall Acc/AUC | Final Acc/AUC | Overall AUC 相对 A 模块参考 |
|---|---:|---:|---:|
| Shuffle | `68.72 / 74.14` | `62.52 / 72.79` | `-1.24` |
| Token-wise MLP | `68.46 / 75.47` | `63.88 / 76.29` | `+0.09` |
| Small Transformer | `68.06 / 75.25` | `67.18 / 74.73` | `-0.13` |

这些结果来自旧的单路径 environment residual 流程，没有实现双路径预测一致性，因此只保留为历史 pilot，不产生 Stage 2 winner。

- `2026-08-15` 已在远端 GPU1 通过真实 Qwen 双路径 preflight：从前 64 个 MathDial 对话中选取最长的 unpacked batch（2 rows、1564 tokens）完成了训练反传、A/B 非零梯度、三条路径评估和有限值检查，峰值显存为 `8545.4 MiB`。该检查不保存 checkpoint 或正式结果，不能替代任何 candidate 的训练、validation 或 audit。
- `2026-08-16` 的恢复预检改为按 `num_kcs * max_tokens^2` 扫描完整训练集并选择最坏 collated batch。4 rows、2009 tokens、attention risk `16144324` 的 B-only 路径和自适应联合路径均通过；B-only 的峰值为 `10729.2 MiB`，联合路径为 `11975.2 MiB`，且 B 梯度非零、A 在 B-only 中的梯度为零。高风险联合 batch 使用串行双路径和精确梯度拆分，低风险 batch 保持展开执行。两项预检仍不保存 checkpoint 或正式结果。
- 远端 GPU1 的 tensor-contract closeout 预检已通过；它现在同时验证表示互补、hook/梯度传播、checkpoint 重算以及实际双路径损失的展开/串行梯度等价，CUDA 最大参数梯度差为 `1.19e-07`。
- 双路径参数审计要求 A strict joint 的 LoRA、selector、calibrator 全部更新，以及最终 joint 的 LoRA、selector、B、calibrator 全部更新；不再用“任意一个组件变化”作为通过条件。
- B-only 的两次历史 OOM 已保留在 `results/cel_stage2_environment/dual_path/round1/stages/`。修复为关闭该冻结阶段的 Qwen 前缀输入梯度：B 仍通过 `h_m` 和冻结后缀获得梯度，但不会为冻结前缀保留激活。不要恢复旧的输入梯度设置。

## 2. 统一术语

- `H`：Qwen 中间层 hidden state。
- `A`：selector。
- `a`：A 输出的 token-level 选择强度，经过 `tanh`，范围约为 `[-1, 1]`。
- `h_r`：证据表征。
- `h_n`：非证据表征。
- `h_nb`：B 变换后的非证据表征。
- `h_m`：混合表征。

正文不再使用 `score`、`signal`、`gate` 或 `delta` 指代 `a`。历史源码和 diagnostics 中的旧字段名只作为兼容键保留。

## 3. A 模块保留的做法

当前 A 模块生成的证据表征为：

```text
a ⊙ H
```

Stage 1 的实际注入为：

```text
H' = H + γ(a ⊙ H)
```

因为原始 `H` 被保留，该路径定义为加性证据增强。A 模块参考结果继续保留 raw-base bootstrap、calibrator warmup、strict joint、final test 和来源/参数审计的历史链，供 provenance 和复现实验使用；这不构成今后 A+B 或更多模块的默认训练方式。

Calibrator 是共享预测头后的可选概率校准层，只调整概率偏移/尺度，不学习 A 或 B 的表示。它不是双路径目标的必要条件。若启用，必须从第一个 epoch 起与所有模块使用同一 optimizer 和完整损失，并共享于 evidence/mixed（评估时含 reversal）；不得默认执行 calibrator-only warmup 或训练后单独调校。bias/positive-affine 校准是单调变换，纯后处理不会改变 AUC，应分开报告 raw/calibrated 概率与 ECE、Brier、阈值指标。

## 4. Stage 2 目标流程

```text
w_r = (a + 1) / 2
w_n = (1 - a) / 2

h_r = w_r ⊙ H
h_n = w_n ⊙ H
h_nb = B(h_n)
h_m = h_r + β · h_nb
```

互补权重只作用于可选证据的 context token；非 context 的有效 prompt token 在 `h_r/h_m` 中保持原始 `H`，在 `h_n` 中为零。

随后用共享后续网络 `P` 分别预测：

```text
p_r = P(h_r)
p_m = P(h_m)
```

核心要求为：

```text
P(y | h_r) ≈ P(y | h_m)
```

旧实现 `H + γ(A residual + B residual)` 不再作为目标定义，因为它保留完整 `H`，只有一个预测路径。

## 5. 整体训练目标

```text
L_r = BCE(p_r, y)
L_m = BCE(p_m, y)
L_cons = JS([p_r, 1-p_r], [p_m, 1-p_m])

L_total = λ_r L_r
        + λ_m L_m
        + λ_cons L_cons
```

- `L_r`：保证证据表征足以预测。
- `L_m`：保证混合表征正确预测。
- `L_cons`：保证变换非证据环境后预测不变。

当前不把证据预算或 B 输出尺度约束加入核心损失。先记录 `a`/证据权重分布、`h_nb` 的 RMS、`βh_nb` 对 `h_m` 的实际贡献和相应梯度；只有正式训练出现明确退化时，才增加针对性的正则。

`P(h_n)` 的 reversal performance 必须作为正式诊断。

## 6. 目标训练链

### 6.1 新的默认训练协议

每个正式 A+B candidate（以及后续加入 C、D 等模块的 candidate）默认从 raw Qwen 开始，将所有模块放入同一计算图、同一 optimizer 和完整目标，从 epoch 1 统一训练和验证：

```text
raw Qwen + LoRA + A + B + optional calibrator (+ C/D/...)
-> unified train/validation for max_epochs
-> validation-best checkpoint + patience early stopping
-> restore best checkpoint -> final test and audit
```

不同模块可以使用 parameter-group learning rate，但不因此拆成独立训练。`max_epochs`、`patience`、`min_delta` 和 validation 指标须在训练开始前固定。

冻结、交替训练、warmup、按 batch loss ramp 和 calibrator-only 只能在模块明确提出特有机制、给出动机并写入 manifest/audit 后使用；此类结果必须标记为特殊协议。

实现状态：训练 CLI 已实现 `--max_epochs`、`--patience`、`--min_delta` 和 validation-best checkpoint 早停，`--epochs` 仅是兼容别名；本协议不能回溯性地套用到正在运行或已完成的五阶段工件，历史工件仍按其原始 staged protocol 解释。

### 6.2 历史 staged 链（仅用于解释现有工件）

每个正式 A+B candidate 使用自包含链：

```text
Qwen3 raw base
-> current-candidate A bootstrap
-> calibrator warmup
-> A strict joint
-> initialize and warm up B
-> A+B+LoRA+calibrator strict joint with L_total
-> dual-path final test
-> provenance / parameter / tensor-contract audit
```

不得加载其他 candidate、baseline 或历史 anchor 的参数。同一 candidate 可以加载自身上一阶段 checkpoint。

上述五阶段链是当前代码和既有运行的历史 staged protocol，仅为工件解释与 provenance 保留。后续正式训练应采用 6.1 的统一端到端协议，不应再默认执行这些阶段。

## 7. 当前执行顺序

1. Round 1 已从 raw Qwen、seed `1221` 完成五阶段训练、validation 和 validation audit。验证显示 A 的选择强度接近饱和且 B 对 `h_m` 的贡献较弱，因此 Round 2 仅依据 validation 将 `beta` 从 `0.10` 提高到 `0.15`、B output ratio 从 `1.00` 提高到 `1.25`、`lambda_cons` 从 `0.10` 提高到 `0.15`，并降低 A bootstrap/selector 学习率。
2. Round 2 已从 raw Qwen、seed `1221` 独立完成五阶段训练、validation 和 validation audit。B 的相对贡献仍弱，因此 Round 3 仅依据 validation 将 `beta` 提高到 `0.20`、B output ratio 提高到 `1.50`、`lambda_cons` 提高到 `0.18`；没有使用 test、control 或额外 seed 作决策。
3. Round 3 同样从 raw Qwen、seed `1221` 独立开始。已完成：
   - `a_bootstrap`：两轮 train/validation loss 约为 `0.6920/0.6498`、`0.6115/0.6540`，选择 epoch 1 checkpoint。
   - `calibrator_warmup`：train/validation loss 约为 `0.6205/0.6628`。
   - `a_joint`：train/validation loss 约为 `0.5928/0.6609`；A 精确继承 `7/7` tensors，calibrator 继承已验证。
   - `b_warmup`：训练完成，阶段内 validation 仍在运行（见上方带时间戳快照）。
4. supervisor 的剩余自动链为：完成 Round 3 `b_warmup` -> 最终 `joint` -> Round 3 validation 与 validation audit -> 冻结 Round 3 决策 -> 对 Round 1/2/3 执行 final test -> 三轮 final audit -> tensor-contract closeout -> 生成 `STAGE2_DUAL_PATH_REPORT.md`。
5. 完成后必须分别判断：三轮研究流程是否技术闭环、Stage 2 是否优于 A 模块、是否达到论文目标，以及现有证据是否支持进入 Stage 3。技术完成不等同于研究成功。
6. 当前不额外运行其他 seed，不运行第五点 control/ablation，不把历史单路径 pilot 或 preflight checkpoint 计入正式结果；在三轮闭环完成前不启动 Stage 3。

新对话接管时先进行只读检查：确认 supervisor、当前 candidate 进程、最新日志和产物。如果原进程仍存活，继续监控；只有确认控制链已经退出且产物明确不完整时，才分析恢复方案，不能直接重启。

## 8. 方法设计原则

方法设计应首先服从研究假设和目标因果机制，而不应以“最小改动、最少参数或单次训练成本最低”作为正式方案的首要目标。对于每一个阶段，应从目标机制反向设计完整且功能闭合的模块，明确其输入输出表征、模块之间的信息流、下游使用方式、训练目标、优化日程以及可验证的诊断指标。模块容量应与所要建模的变换复杂度相匹配，不能用过小或过于简化的模块替代核心方法，再据此判断研究方向是否有效。轻量模块可以用于接口检查、sanity check 或消融实验，但不应作为核心方法的主要证据。不应通过削弱核心模块的表达能力来节省正式训练。参数效率应在机制得到充分验证之后再进行优化。

## 9. 关键文件

- `results/method_design/CEL_Modular_Flow.md`
- `results/method_design/CEL_Stage1_A_Module_Implementation.md`
- `results/method_design/CEL_Stage2_Environment_Generator_Implementation.md`
- `results/method_design/CEL_Stage2_实施流程与思路.md`
- `results/cel_stage1_last_layer/CEL_三阶段整体方法说明.md`
- `results/cel_stage2_environment/SUMMARY.md`
- `results/cel_stage2_environment/PAIRED_BOOTSTRAP.md`
- `dialogue_kt/cel_methods.py`
- `dialogue_kt/training.py`
- `scripts/cel_stage2_environment/check_stage2_real_qwen_preflight.py`
- `scripts/cel_stage2_environment/run_validation_after_dual_path_chain.sh`
- `scripts/cel_stage2_environment/generate_dual_path_stage2_report.py`
