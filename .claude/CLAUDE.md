# CLAUDE.md - 项目操作指南

## 1. 当前定位

- 任务：MathDial ATC Dialogue-KT，比较 True/False token logits。
- 模型：Qwen3-1.7B + LoRA。
- 当前范围：baseline + CEL A/B。
- 论文目标：Overall AUC `76.71`。
- Baseline：Overall Acc/AUC `69.22 / 74.95`，Final Acc/AUC `61.17 / 75.32`。
- A 模块参考结果：Overall Acc/AUC `70.03 / 75.38`，Final Acc/AUC `66.80 / 77.26`。
- Stage 2 首轮历史实验已完成并通过审计，但采用旧单路径 residual 流程，当前没有 Stage 2 winner。
- 目标 A+B 双路径方法、合同测试和审计链已完成；当前执行三轮配置实验（固定 seed=1221），不启动 Stage 3。

A 模块参考结果的历史 checkpoint 名称为 `cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b`。该名称只用于文件、日志和 audit 追溯；方法说明统一称为 **A 模块**。

## 2. 方法术语

- `H`：Qwen 中间层 hidden state。
- `A`：selector。
- `a`：A 输出的 token-level 选择强度，经过 `tanh`，范围约为 `[-1, 1]`。
- `h_r`：证据表征。
- `h_n`：非证据表征。
- `h_nb`：B 变换后的非证据表征。
- `h_m`：混合表征。

正文不再混用 `score`、`signal`、`gate` 或 `delta`。源码和旧结果中的历史字段名可暂时保留，以避免破坏兼容性。

## 3. 当前 A 与目标 A+B

当前 A 模块的加性证据增强：

```text
H' = H + γ(a ⊙ H)
```

目标 A+B：

```text
w_r = (a + 1) / 2
w_n = (1 - a) / 2
h_r = w_r ⊙ H
h_n = w_n ⊙ H
h_nb = B(h_n)
h_m = h_r + β · h_nb
```

互补权重只作用于可选证据的 context token；其他有效 prompt token 在 `h_r/h_m` 中保持原始 `H`，在 `h_n` 中为零。

共享后续网络分别计算 `p_r = P(h_r)` 和 `p_m = P(h_m)`。

核心损失：

```text
L_total = λ_r BCE(p_r, y)
        + λ_m BCE(p_m, y)
        + λ_cons JS([p_r, 1-p_r], [p_m, 1-p_m])
```

当前核心目标只包含以上三项。证据比例和 B 输出尺度必须记录为诊断；仅在正式训练实际出现全证据坍缩或 B 输出失效时，再讨论是否加入对应正则。

完整定义见 `results/method_design/CEL_Modular_Flow.md`。

### 3.1 Calibrator 的定位

`calibrator` 是共享预测头之后的可学习概率校准层，只调整概率的偏移、尺度和阈值行为，不负责学习 A 的选择或 B 的表征变换。它不是 A+B 表示学习和 `L_total` 优化的必要条件；机制验证可以使用 `none`，也可以使用以恒等变换初始化的可选 calibrator。

若启用 calibrator，必须从第一个 epoch 起与 A、B、LoRA 使用同一 optimizer、同一 batch 和同一 `L_total` 联合训练，并对 evidence/mixed（评估时含 reversal）使用同一个实例。默认不得安排 calibrator-only warmup 或训练后单独校准；后者只能作为明确标注的后处理/消融，且不能充当 A/B 机制证据。由于 bias/positive-affine 变换是单调的，纯后处理校准不会改变 AUC，应同时报告 raw 与 calibrated 概率及校准指标。

## 4. 当前执行顺序

1. Round 1 使用 4-layer/1024-width contextual Transformer B，从 raw Qwen、seed=1221 开始，只依据 validation 选择 checkpoint。
2. Round 2、Round 3 都从 raw Qwen 独立开始，只根据上一轮 validation/diagnostics 调整配置。
3. Round 3 配置冻结后，对三轮 candidate 统一执行 final test，再完成 provenance、parameter 和 tensor-contract audit。
4. 不额外运行其他 seed；历史单路径 pilot 和 preflight 工件不计入正式结果。

## 5. 正式实验约束

- 每个正式 candidate 从 Qwen3-1.7B 原始基座开始。
- 同一 candidate 可以加载自身上一阶段 checkpoint。
- 禁止加载 baseline、历史 anchor 或其他 candidate 参数作为正式初始化来源。
- 只有完整 train + validation + final test + provenance audit + parameter audit 才计入正式结果。
- 后处理、冻结补训和纯校准结果只算诊断。
- 本次用户确认的执行口径固定为 `model/data seed=1221`，不额外扩展 seed；因此结论不外推为 multi-seed 稳健性。
- 在目标 A+B 方法稳定前不启动 Stage 3。

### 5.1 默认统一训练协议

- 正式训练开始前固定 `max_epochs`、validation 指标、`patience` 和 `min_delta`；每个 epoch 都执行统一 train/validation，保存 validation 最优 checkpoint，触发 patience 后早停并恢复最佳 checkpoint。
- 所有模块默认作为一个整体参与同一计算图、同一损失和同一 optimizer。从第一个 epoch 起，A、B、LoRA、calibrator 以及未来新增的 C/D 等模块都必须可训练并接受联合验证。
- 允许模块使用不同 parameter-group learning rate，但不能因此变成独立训练、交替训练或默认冻结阶段。
- `a_bootstrap`、`b_warmup`、calibrator-only、按 batch loss ramp 等阶段机制仅能作为明确声明、论证并审计的特殊协议；其结果不得冒充默认统一训练结果。

当前代码和正在运行的历史 Stage 2 链仍包含上述阶段参数。它们用于兼容和追溯，后续实现应迁移到本节协议；本文件的默认规范优先于旧 launcher 的阶段命名。

实现状态说明：训练 CLI 现已提供 `--max_epochs`、`--patience` 和 `--min_delta`，`--epochs` 仅作为兼容别名；训练循环按 validation loss 保存最佳 checkpoint，并在 patience 触发后停止。历史训练记录仍必须按其原始固定 epoch/分阶段行为解释。

## 6. 方法设计原则

方法设计应首先服从研究假设和目标因果机制，而不应以“最小改动、最少参数或单次训练成本最低”作为正式方案的首要目标。对于每一个阶段，应从目标机制反向设计完整且功能闭合的模块，明确其输入输出表征、模块之间的信息流、下游使用方式、训练目标、优化日程以及可验证的诊断指标。模块容量应与所要建模的变换复杂度相匹配，不能用过小或过于简化的模块替代核心方法，再据此判断研究方向是否有效。轻量模块可以用于接口检查、sanity check 或消融实验，但不应作为核心方法的主要证据。不应通过削弱核心模块的表达能力来节省正式训练。参数效率应在机制得到充分验证之后再进行优化。

## 7. 新对话读取顺序

1. `.claude/STAGE2_HANDOFF.md`
2. `.claude/CLAUDE.md`
3. `.claude/MEMORY.md`
4. `.claude/STRUCTURE.md`
5. `results/method_design/CEL_Modular_Flow.md`
6. `results/method_design/CEL_Stage1_A_Module_Implementation.md`
7. `results/method_design/CEL_Stage2_Environment_Generator_Implementation.md`
8. `results/cel_stage2_environment/SUMMARY.md`

## 8. 本地环境

- 根目录：`/home/luo/work/dialogue-kt_migration_20260806`
- OS：WSL2 Ubuntu 22.04.5
- 系统 Python：`/usr/bin/python3`，不作为项目测试解释器
- 测试环境：`/home/luo/miniconda3/envs/diagkt`
- Python：3.10.20
- Torch：2.4.1+cu118
- Transformers：5.7.0
- PEFT：0.19.1
- 本地 GPU：NVIDIA GeForce RTX 4050 Laptop GPU 6GB
- 本地 `saved_models/` 没有正式 checkpoint

激活：

```bash
conda activate diagkt
```

## 9. 正式训练环境

- SSH alias：`3090`
- 远端：`user4@119.29.183.125:22049`
- 根目录：`/home/user4/dialogue-kt`
- Conda 环境：`luo_2`
- GPU：2 x RTX 3090 24GB

SSH 配置、端口、密钥和远端根目录保持不变。`/home/luo/.ssh` 权限保持 `700`，配置和私钥保持 `600`。

## 10. 常用只读与验证命令

```bash
git status --short
bash scripts/cel_stage2_environment/sync_results_from_server.sh 3090
/home/luo/miniconda3/envs/diagkt/bin/python scripts/cel_stage2_environment/check_stage2_tensor_contracts.py --cuda
/home/luo/miniconda3/envs/diagkt/bin/python scripts/cel_stage2_environment/analyze_stage2_paired_bootstrap.py
/home/luo/miniconda3/envs/diagkt/bin/python scripts/cel_stage2_environment/generate_stage2_summary.py
/home/luo/miniconda3/envs/diagkt/bin/python scripts/cel_stage2_environment/generate_dual_path_stage2_report.py --require-validation --require-test --require-contracts
```

正式训练 launcher 仅使用 `scripts/cel_stage2_environment/run_dual_path_candidate.sh`；旧单路径 launcher 不再用于当前目标方法。

## 11. 工作树规则

- 当前工作树已有大量用户修改和删除，不得 reset、checkout 或清理。
- 只修改任务直接涉及的文件。
- 手工编辑使用 `apply_patch`。
- 本地测试使用 `diagkt`，不要用依赖为空的系统 Python 解释项目结果。
