# MEMORY.md - 当前研究状态

## 1. 当前结论

- 当前仓库只保留 baseline + CEL 主线。
- 任务：MathDial ATC Dialogue-KT。
- 模型：Qwen3-1.7B + LoRA。
- 论文目标：Overall AUC `76.71`。
- Baseline：Overall Acc/AUC `69.22 / 74.95`，Final Acc/AUC `61.17 / 75.32`。
- A 模块参考结果：Overall Acc/AUC `70.03 / 75.38`，Final Acc/AUC `66.80 / 77.26`。
- Stage 2 首轮三个历史 candidate 均已完成并通过 audit，但采用旧的单路径 environment residual 流程，当前没有 Stage 2 winner。
- 双路径目标实现、合同测试和审计链已完成；当前按固定 seed=1221 执行三轮配置实验，不启动 Stage 3。
- 当前三轮状态：Round 1/2 已完成训练、validation 和 validation audit；Round 3 正在执行。Round 1/2 Mixed Overall Acc/AUC 分别为 `66.78 / 72.51`、`65.93 / 72.59`，均低于 A 模块参考 `70.03 / 75.38`。
- `2026-08-17 18:45 CST` 快照：唯一 supervisor PID `3641318` 仍存活；Round 3 `b_warmup` 训练已完成 `8425/8425`，阶段内 validation 为 `1112/2202`。该快照会过时，新对话必须先通过 SSH alias `3090` 重新只读核对。

A 模块参考结果对应的历史 checkpoint 名称为 `cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b`。该名称只用于工件与审计追溯；方法正文统一称为 **A 模块**。

## 2. 统一术语

- `H`：Qwen 中间层 hidden state。
- `A`：selector。
- `a`：A 输出的 token-level 选择强度，经过 `tanh`，范围约为 `[-1, 1]`。
- `h_r`：证据表征。
- `h_n`：非证据表征。
- `h_nb`：B 变换后的非证据表征。
- `h_m`：混合表征。

方法正文不再使用 `score`、`signal`、`gate` 或 `delta` 指代 `a`。历史代码和 diagnostics 的旧字段名只作为兼容键保留。

## 3. A 模块

当前 A 模块生成的证据表征为：

```text
a ⊙ H
```

当前 Stage 1 注入为：

```text
H' = H + γ(a ⊙ H)
```

该路径保留原始 `H`，因此定义为加性证据增强。

A 模块参考结果保留已经验证的历史训练链：

```text
Qwen3 raw base
-> fresh LoRA + A bootstrap
-> same-candidate calibrator warmup
-> same-candidate strict joint
-> final test and audit
```

这不是新增模块的默认训练规范。对于统一的 A+B（以及未来 A+B+C 等）任务，所有模块应从第一个 epoch 起在同一计算图、同一 optimizer 和完整目标中端到端训练；历史 warmup 仅作为 provenance 保留。

Calibrator 只是共享预测头后的可选概率校准层，不是 A/B 表示学习的必要组件。启用时必须与所有模块从 epoch 1 联合训练，并同时作用于 evidence/mixed；不得默认使用 calibrator-only warmup 或训练后校准。关闭 calibrator 不影响双路径目标的定义；纯单调后处理不改变 AUC，raw/calibrated 概率和校准指标应分开报告。

## 4. Stage 2 目标方法

```text
w_r = (a + 1) / 2
w_n = (1 - a) / 2

h_r = w_r ⊙ H
h_n = w_n ⊙ H
h_nb = B(h_n)
h_m = h_r + β · h_nb
```

互补权重只作用于可选证据的 context token；其他有效 prompt token 在 `h_r/h_m` 中保持原始 `H`，在 `h_n` 中为零。

共享后续网络分别计算：

```text
p_r = P(h_r)
p_m = P(h_m)
```

核心训练目标：

```text
L_r = BCE(p_r, y)
L_m = BCE(p_m, y)
L_cons = JS([p_r, 1-p_r], [p_m, 1-p_m])

L_total = λ_r L_r
        + λ_m L_m
        + λ_cons L_cons
```

目标解释：证据表征应足以预测；变换后的非证据环境混入后仍应正确预测；两条预测应保持一致。

当前不加入证据预算或 B 输出尺度正则。先记录 `a`/证据权重分布、`h_nb` 的输出尺度和 `βh_nb` 的实际贡献，再根据正式训练是否出现对应退化决定。

`P(h_n)` 的 reversal performance 是必需诊断。

## 5. Stage 2 首轮历史结果

| 历史方向 | Overall Acc/AUC | Final Acc/AUC | Overall AUC 相对 A 模块参考 |
|---|---:|---:|---:|
| Shuffle | `68.72 / 74.14` | `62.52 / 72.79` | `-1.24` |
| Token-wise MLP | `68.46 / 75.47` | `63.88 / 76.29` | `+0.09` |
| Small Transformer | `68.06 / 75.25` | `67.18 / 74.73` | `-0.13` |

解释边界：

- 三个 candidate 都通过来源与参数审计。
- 它们没有构造互补的 `h_r/h_n`。
- 它们没有分别计算 `p_r/p_m`。
- 它们没有使用预测一致性损失。
- 结果只作为历史 pilot 和工程诊断，不能验证当前目标方法。

## 6. 当前执行状态

1. Round 1 已完成并通过 validation audit。Mixed/Evidence/Reversal 的 Overall Acc/AUC 分别为 `66.78 / 72.51`、`66.84 / 72.52`、`52.71 / 46.94`；相应 Final Acc/AUC 为 `61.20 / 73.46`、`61.45 / 73.51`、`25.78 / 40.80`。
2. Round 2 已完成并通过 validation audit。Mixed/Evidence/Reversal 的 Overall Acc/AUC 分别为 `65.93 / 72.59`、`66.10 / 72.53`、`52.65 / 47.40`；相应 Final Acc/AUC 为 `64.82 / 73.01`、`64.58 / 72.89`、`25.54 / 41.18`。
3. Round 1 -> 2 的 validation-only 调整为 `beta 0.10 -> 0.15`、B output ratio `1.00 -> 1.25`、`lambda_cons 0.10 -> 0.15`，并因 A 饱和降低 bootstrap/selector 学习率。Round 2 -> 3 调整为 `beta 0.15 -> 0.20`、B output ratio `1.25 -> 1.50`、`lambda_cons 0.15 -> 0.18`。没有使用 test、control 或额外 seed 作决策。
4. Round 3 的 `a_bootstrap`、`calibrator_warmup`、`a_joint` 已完成；A 精确继承 `7/7` tensors，calibrator 继承已验证。`b_warmup` 训练已完成，快照时正在阶段内 validation。
5. 唯一 supervisor 为 `scripts/cel_stage2_environment/run_dual_path_three_round_chain.sh`。剩余自动链：Round 3 最终 `joint`、validation/audit、三轮 final test/final audit、tensor-contract closeout、`STAGE2_DUAL_PATH_REPORT.md`。活跃时不得启动第二个 supervisor 或 candidate launcher。
6. 不额外运行其他 seed；不运行第五点 control/ablation；历史单路径 pilot、preflight checkpoint 和未完成链不计入正式结果；三轮闭环前不启动 Stage 3。
7. `2026-08-16` 已按完整训练集的 `rows * sequence_length^2` 几何找到最坏 batch：4 rows、2009 tokens、attention risk `16144324`。自适应 joint 预检峰值 `11975.2 MiB`，B-only 峰值 `10729.2 MiB`；两者均通过，且 joint 的 A/B 梯度非零、B-only 的 A 梯度为零。
8. 高风险 joint batch 使用串行 evidence/mixed 执行和精确梯度拆分，低风险 batch 使用展开执行；自动 tensor contract 在 CPU/CUDA 上的最大梯度差分别为 `2.98e-08` 和 `1.19e-07`。
9. 当前 Goal 应保持 `active`，不是 `blocked`。新对话先读取交接文件并重新只读检查 SSH；不要依赖旧 PID/百分比，也不要在未确认原链退出前执行恢复或重启。

## 7. 方法设计原则

方法设计应首先服从研究假设和目标因果机制，而不应以“最小改动、最少参数或单次训练成本最低”作为正式方案的首要目标。对于每一个阶段，应从目标机制反向设计完整且功能闭合的模块，明确其输入输出表征、模块之间的信息流、下游使用方式、训练目标、优化日程以及可验证的诊断指标。模块容量应与所要建模的变换复杂度相匹配，不能用过小或过于简化的模块替代核心方法，再据此判断研究方向是否有效。轻量模块可以用于接口检查、sanity check 或消融实验，但不应作为核心方法的主要证据。不应通过削弱核心模块的表达能力来节省正式训练。参数效率应在机制得到充分验证之后再进行优化。

默认训练规范补充：训练前固定 `max_epochs`、validation 指标、`patience`、`min_delta`；每个 epoch 统一 train/validation，保存最佳 checkpoint，按 patience 早停并恢复最佳 checkpoint。新增模块默认直接加入整体 forward、完整损失和 optimizer，只有明确声明的模块特有机制才可冻结、交替或 warmup，并须记录动机、范围和审计判据。

实现状态：训练 CLI 现已提供 `--max_epochs`、`--patience` 和 `--min_delta`，并按 validation loss 保存最佳 checkpoint、支持 patience 早停；`--epochs` 仅为兼容别名。现有历史阶段链和运行记录不应被解释为已满足新的统一协议。

## 8. 环境与约束

- 本地根目录：`/home/luo/work/dialogue-kt_migration_20260806`。
- 本地测试环境：`/home/luo/miniconda3/envs/diagkt`。
- 本地 GPU：NVIDIA GeForce RTX 4050 Laptop GPU 6GB。
- 正式训练环境：SSH alias `3090`，远端 `/home/user4/dialogue-kt`，双 RTX 3090。
- 本地 `saved_models/` 没有正式 checkpoint。
- 最近记录的 Stage 1 权威远端刷新时间：`2026-08-12 23:17:24`。
- 工作树原有大量用户修改与删除，不得 reset 或清理。

## 9. 新对话读取顺序

1. `.claude/STAGE2_HANDOFF.md`
2. `.claude/CLAUDE.md`
3. `.claude/MEMORY.md`
4. `results/method_design/CEL_Modular_Flow.md`
5. `results/method_design/CEL_Stage1_A_Module_Implementation.md`
6. `results/method_design/CEL_Stage2_Environment_Generator_Implementation.md`
7. `results/cel_stage2_environment/SUMMARY.md`

## 10. 关键文件

- 当前方法定义：`results/method_design/CEL_Modular_Flow.md`
- A 模块实现：`results/method_design/CEL_Stage1_A_Module_Implementation.md`
- B 模块目标实现：`results/method_design/CEL_Stage2_Environment_Generator_Implementation.md`
- Stage 2 中文流程：`results/method_design/CEL_Stage2_实施流程与思路.md`
- 整体说明：`results/cel_stage1_last_layer/CEL_三阶段整体方法说明.md`
- Stage 2 交接：`.claude/STAGE2_HANDOFF.md`
- 历史结果：`results/cel_stage2_environment/`
