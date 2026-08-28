# 项目操作指南

## 当前状态

本工作区是 MathDial ATC Dialogue-KT 的 baseline + CEL 模块研究代码。当前保留 A、B 的实现与方法设计，旧的分阶段训练记录和无关实验结果已清理；A 模块与 A+B 模块等待按新协议重新训练。

- 底座：Qwen3-1.7B + LoRA。
- 认证 baseline：Overall Acc/AUC `69.22 / 74.95`，Final Acc/AUC `61.17 / 75.32`。
- A 模块历史参考：Overall Acc/AUC `70.03 / 75.38`，Final Acc/AUC `66.80 / 77.26`。
- 历史 checkpoint 名称 `cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b` 仅用于 provenance，不作为新的方法名。
- 论文目标：Overall AUC `76.71`。

新对话按以下顺序读取：

1. `AGENTS.md`
2. `.claude/CLAUDE.md`
3. `.claude/MEMORY.md`
4. `.claude/STRUCTURE.md`
5. `results/method_design/CEL_Modular_Flow.md`
6. `results/method_design/CEL_Stage1_A_Module_Implementation.md`
7. `results/method_design/CEL_Stage2_Environment_Generator_Implementation.md`

## 统一术语

- `H`：Qwen 中间层 hidden state。
- `A`：selector 模块；`a` 是经 `tanh` 的 token-level 选择强度，约在 `[-1, 1]`。
- `B`：非证据表征变换模块。
- `C`、`D`、……：未来新增模块，沿用同一命名规则。
- `h_r`：证据表征；`h_n`：非证据表征；`h_nb`：B 变换后的非证据表征；`h_m`：混合表征。

方法正文不要用 `score`、`signal`、`gate`、`delta` 作为 `a` 的同义词。旧源码字段和审计键可为兼容性保留，但解释时统一映射为 `a`。

## 方法与训练规范

设计规范对 A、B、C、D 等所有模块生效。正式模块必须从研究假设和目标因果机制出发，明确输入/输出表征、信息流、下游使用、训练目标、优化日程和诊断；容量要足以表达目标变换。轻量模块只能用于接口检查、sanity check 或消融，不能用来代表核心方法或替核心方法节省正式训练成本。

目标 A+B 表示契约为：

```text
w_r = ((a + 1) / 2)      on evidence-eligible context tokens
w_n = ((1 - a) / 2)
h_r = w_r ⊙ H
h_n = w_n ⊙ H
h_nb = B(h_n)
h_m = h_r + β · h_nb
p_r = P(h_r)
p_m = P(h_m)
```

`P` 是两条路径共享的后续 Qwen 网络和预测头；评估时还要报告 `P(h_n)` 的 reversal performance。核心损失为：

```text
L_total = λ_r BCE(p_r, y)
        + λ_m BCE(p_m, y)
        + λ_cons JS([p_r, 1-p_r], [p_m, 1-p_m])
```

默认训练协议：

1. A、B、LoRA、可选 calibrator 以及未来 C/D 等模块从 epoch 1 起进入同一 forward、同一完整损失和同一 optimizer。
2. 可以使用 parameter-group learning rate，但不同学习率不等于独立训练、交替训练或冻结训练。
3. 每个 epoch 都进行 train/validation；固定 `max_epochs`、validation 指标、`patience`、`min_delta`，保存 validation-best checkpoint，早停后恢复最佳 checkpoint，再执行 final test 和 audit。
4. 后续 epoch 必须从前一 epoch 的整体模型状态继续；任何恢复都只能读取同一 candidate 的整体 checkpoint。
5. bootstrap、warmup、按 batch loss ramp、calibrator-only 和分阶段冻结不是默认流程。若模块确有特有机制，必须在方法文档、manifest 和 audit 中声明动机、范围、日程及判据，并单独标记结果。

`L_budget` 与 `L_scale` 当前不属于核心目标。先记录选择分布、B 输出尺度、梯度和参数变化，只有观察到明确坍缩后才增加对应正则。calibrator 只是共享预测头后的可选概率校准层，不学习 A/B 表征；启用时也必须从 epoch 1 联合训练。

## 代码与实验边界

- 核心代码：`dialogue_kt/main.py`、`dialogue_kt/training.py`、`dialogue_kt/cel_methods.py`、`dialogue_kt/kt_data_loading.py`、`dialogue_kt/prompting.py`。
- A/B 训练与审计入口：`scripts/cel/`。
- 方法文档：`results/method_design/`。
- 新训练结果：`results/a/` 和 `results/a_b/`；baseline 保存在 `results/baseline/`。
- `data/` 与 `literature/` 不纳入代码整理和 GitHub 代码备份。

正式训练环境为 SSH alias `3090`（远端 `/home/user4/dialogue-kt`，conda 环境 `luo_2`）。本地 smoke test 使用 `/home/luo/miniconda3/envs/diagkt/bin/python`。保持现有 SSH 目标、端口、身份和私钥权限不变。

不要恢复、覆盖或重命名 `data/`、`literature/`，也不要用历史 checkpoint 初始化新的 formal candidate。手工编辑使用 `apply_patch`；删除必须使用明确路径并先检查 Git 状态。
