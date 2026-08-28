# Task-Conditioned Tuning Loop

## 目标

本记录保留历史 task-conditioned 调优结果；独立初始化的 A 模块历史 candidate 已经完成 strict full-train、正式测试、来源/参数审计与 closeout。

- Baseline Overall AUC: **74.95**
- Baseline Overall Acc: **69.22**
- Baseline Final AUC: **75.32**
- Baseline Final Acc: **61.17**

历史 selector/ranking anchor（comparison-only，不是 A 模块训练链起点）：

- `cel_task_conditioned_lastlayer_v1_qwen3_1.7b`
- Overall AUC: **75.29**
- Overall Acc: **68.87**
- Final AUC: **75.27**
- Final Acc: **56.31**

当前正式状态：

- 旧 `v26 fullv1` 结果已撤回，原因是最终测试 checkpoint 来自 calibrator-only warmup。
- 当前 A 模块参考结果对应的历史 checkpoint：`cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b`。
- Overall Acc / AUC：**70.03 / 75.38**；Final Acc / AUC：**66.80 / 77.26**。
- A 模块从 Qwen3-1.7B 原始基座建立自己的 bootstrap，后续阶段只加载同一 candidate 上一阶段的 checkpoint；baseline 与 v1 仅用于比较。

## 当前分析

当前 `v1` 的关键现象不是“排序不够”，而是“概率偏保守”：

- `Pred True`: **33.75%**
- baseline `Pred True`: **40.76%**
- `v1` 的最佳 Acc 阈值约为 **0.418**
- `v1` 的最佳理论 Acc 约为 **69.67**
- baseline 的最佳 Acc 阈值约为 **0.488**
- 离线 logit 扫描表明：如果只做保守 bias 平移，`v1` 的 `0.5` 阈值下 Acc 理论上可到约 **69.67**，而 affine 没有额外增益

这说明：

- 当前 `task_conditioned v1` 的排序能力已经优于 baseline
- 但输出分数整体偏低，导致固定 `0.5` 阈值下 Recall/Acc 被压住
- 因此历史上先做了一批 calibration / frozen 微调诊断，但这些结果不再计入正式方法胜负判断

## Round 1: Diagnostic Post-hoc Calibration

这一轮只用于诊断 `v1` 的概率偏移问题，不计入“完整方法最终指标”。

1. `cel_task_conditioned_lastlayer_v11_cal_bias_only_qwen3_1.7b`
   - baseline LoRA + `v1` selector
   - 冻结主体，只训练 bias calibrator
2. `cel_task_conditioned_lastlayer_v12_cal_affine_only_qwen3_1.7b`
   - baseline LoRA + `v1` selector
   - 冻结主体，只训练 affine calibrator
3. `cel_task_conditioned_lastlayer_v15_fullv1_cal_bias_only_qwen3_1.7b`
   - 从完整 `task_conditioned v1` checkpoint 起步
   - 冻结主体，只训练 bias calibrator
4. `cel_task_conditioned_lastlayer_v16_fullv1_cal_affine_only_qwen3_1.7b`
   - 从完整 `task_conditioned v1` checkpoint 起步
   - 冻结主体，只训练 affine calibrator

设计意图：

- 判断 `v1` 的主要短板是否真的是概率偏保守
- 估计“只调输出刻度”最多能带来多少 Acc 收益
- 这些结果只作为诊断/上限，不作为最终正式方法分数

### Round 1: Diagnostic Post-hoc Calibration Results

| Model | Status | Overall Acc | Overall AUC | Final Acc | Final AUC | Pred True | Best Acc | Best Threshold | Delta Overall Acc | Delta Overall AUC | Delta Final Acc | Delta Final AUC | Overall Gate | Final Pair |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `cel_task_conditioned_lastlayer_v11_cal_bias_only_qwen3_1.7b` | diagnostic | 65.04 | 73.83 | 49.90 | 73.84 | 23.48% | 68.82 | 0.371 | -4.18 | -1.12 | -11.27 | -1.48 | no | no |
| `cel_task_conditioned_lastlayer_v12_cal_affine_only_qwen3_1.7b` | diagnostic | 65.49 | 73.83 | 50.49 | 73.84 | 24.63% | 68.82 | 0.417 | -3.73 | -1.12 | -10.68 | -1.48 | no | no |
| `cel_task_conditioned_lastlayer_v15_fullv1_cal_bias_only_qwen3_1.7b` | diagnostic | 69.27 | 75.29 | 61.94 | 75.27 | 40.60% | 69.67 | 0.512 | +0.05 | +0.34 | +0.77 | -0.05 | yes | partial |
| `cel_task_conditioned_lastlayer_v16_fullv1_cal_affine_only_qwen3_1.7b` | failed (traceback) | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |

### Round 1 Decision

- 这一轮只提供诊断证据，不计入正式 end-to-end 方法结果。
- `v15` 说明：如果只做后续 bias 校准，`v1` 的 Acc 确实可以被拉回 baseline 之上；但这种做法不属于当前严格口径下的最终方法。

## Round 2: Diagnostic Frozen / Partial Retraining

这一轮仍不属于严格 end-to-end 完整方法训练，只作为补充诊断。

1. `cel_task_conditioned_lastlayer_v13_selector_cal_bias_qwen3_1.7b`
   - 从完整 `task_conditioned v1` checkpoint 起步
   - 冻结 backbone，仅联训 selector + bias calibrator
2. `cel_task_conditioned_lastlayer_v14_selector_cal_affine_qwen3_1.7b`
   - 从完整 `task_conditioned v1` checkpoint 起步
   - 冻结 backbone，仅联训 selector + affine calibrator
3. `cel_task_conditioned_lastlayer_v17_selector_cal_bias_memfix_qwen3_1.7b`
   - `v13` 的显存修正版
4. `cel_task_conditioned_lastlayer_v18_fullv1_fixed_bias_eval_qwen3_1.7b`
   - 固定 bias 的纯诊断评估，不训练

设计意图：

- 判断冻结 backbone 后，selector/calibrator 微调是否还能稳定改进
- 这些结果也不计入最终正式方法分数

### Round 2: Diagnostic Frozen / Partial Retraining Results

| Model | Status | Overall Acc | Overall AUC | Final Acc | Final AUC | Pred True | Best Acc | Best Threshold | Delta Overall Acc | Delta Overall AUC | Delta Final Acc | Delta Final AUC | Overall Gate | Final Pair |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `cel_task_conditioned_lastlayer_v13_selector_cal_bias_qwen3_1.7b` | failed (oom) | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| `cel_task_conditioned_lastlayer_v14_selector_cal_affine_qwen3_1.7b` | pending | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| `cel_task_conditioned_lastlayer_v17_selector_cal_bias_memfix_qwen3_1.7b` | pending | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| `cel_task_conditioned_lastlayer_v18_fullv1_fixed_bias_eval_qwen3_1.7b` | pending | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |

### Round 2 Decision

- 这一轮也只提供诊断证据，不计入正式 end-to-end 方法结果。
- 当前最重要的信息是：冻结/部分微调路线没有形成可直接汇报的完整方法闭环。

## Round 3: Valid End-to-End Full Training

候选：

1. `cel_task_conditioned_lastlayer_v21_joint_bias_fulltrain_qwen3_1.7b`
   - 从 baseline checkpoint 起步
   - 把 bias calibrator 作为方法一部分，从训练开始就联合训练全部可训练参数
   - 唯一主改动：把 bias calibrator 从后处理调节前移为训练期联合优化的一部分
   - 目标假设：如果 `v1` 的主要缺口来自概率过于保守，那么把 bias calibrator 从训练开始就纳入联合优化，应能在不破坏排序能力的前提下把 0.5 阈值下的 Acc 拉回去。
   - 期望观察 1：`Pred True` 应从 `v1` 的保守区间往 baseline 水平回升，而不是继续明显偏低。
   - 期望观察 2：Overall Acc 应相对 `v1` 回升并争取超过 baseline，同时 Overall AUC 至少不要明显低于历史 selector/ranking 比较锚点。
   - 期望观察 3：若该方法成立，收益必须体现在一次完整 SSH `train + val + test` 里，而不是只在后处理校准里出现。
   - 若未双超时的下一轮解释：如果 AUC 仍有竞争力但 Acc 还是不过线，下一轮正式候选只应围绕校准强度、初始化或学习率再改一个变量；如果 AUC 也回落，说明联合 bias 训练已经伤害排序，下一轮应改成一个更受控的训练期主变量，而不是继续叠加后处理补丁。
2. `cel_task_conditioned_lastlayer_v22_fullv1_joint_bias_tinylr_qwen3_1.7b`
   - 从完整 `task_conditioned v1` checkpoint 起步
   - 保持全部相关参数可训练，用极小学习率做完整联合微调
   - 唯一主改动：把主变量改成从完整 `v1` checkpoint 出发、只用极小学习率做联合微调
   - 目标假设：如果 `v1` 已经具备更好的排序，主要问题只是默认阈值下过于保守，那么从 `v1` warm-start 并用极小学习率联调，可能在尽量不破坏 AUC 的前提下把概率尺度推回更有利于 Acc 的区间。
   - 期望观察 1：Overall AUC 应尽量贴近 `v1`，而不是把当前锚点的排序优势洗掉。
   - 期望观察 2：`Pred True` 应温和上升并靠近 baseline，而不是直接漂移到过多 True 的区间。
   - 期望观察 3：若 warm-start 校准假设成立，Overall Acc 应向 baseline 靠近并最好越过 baseline。
   - 若未双超时的下一轮解释：如果 Acc 和 AUC 一起回落，或 calibrator 几乎没动，说明 tiny-lr warm-start 既没有纠正保守概率，也没有稳住排序；下一轮正式候选应改一个更直接影响训练期校准的主变量，而不是继续重复 warm-start + 后处理式思路。
3. `cel_task_conditioned_lastlayer_v23_joint_bias_posinit_qwen3_1.7b`
   - 从 baseline checkpoint 起步
   - 保持 v21 的 joint bias full-train 训练口径，只把 calibrator 初始化改成正偏置起点
   - 唯一主改动：把 joint bias full-train 的 calibrator 初始化从零改成基于诊断证据的正偏置起点
   - 目标假设：如果 v21 失败的主要原因是 joint bias 在零初始化下几乎没有被推离 no-op 状态，那么把初始 bias 设为 v15 已验证过的正偏置起点，应能在保住 AUC 的同时把 Pred True 和默认阈值 Acc 往 baseline 区间拉回。
   - 期望观察 1：Pred True 应从 v21 的 32.59% 明显回升，并尽量靠近 baseline 40.8%，而不是继续卡在保守区间。
   - 期望观察 2：Overall Acc 应至少超过 v21 的 68.61，并争取越过 baseline 69.22，同时 Overall AUC 不应明显低于 v21/v1 的 75.26/75.29。
   - 期望观察 3：训练结束后 calibrator bias 应保持显著正值，而不是像 v21 一样几乎回到 0。
   - 若未双超时的下一轮解释：如果 Pred True 仍然明显偏低且 calibrator bias 又回到接近 0，说明单纯正初始化不足以驱动联训校准；下一轮只应再改一个训练动力学变量，例如 calibrator 学习率或 warmup。若 Pred True 过冲而 AUC 回落，则应减小初始化幅度，而不是叠加更多变量。
4. `cel_task_conditioned_lastlayer_v24_selectorinit_joint_bias_fulltrain_qwen3_1.7b`
   - 从 baseline checkpoint 起步
   - 保持 v21 的 joint bias full-train 训练口径，只把 selector 初始化改成来自历史 selector/ranking 锚点 v1
   - 唯一主改动：在 v21 的 joint bias full-train 基础上，仅把 selector 初始化改为来自历史 selector/ranking 锚点 v1
   - 目标假设：如果 v21 的问题主要不是联训 bias 本身，而是从 baseline 冷启动时丢失了 v1 已学到的排序信号，那么仅用 v1 warm-start selector 应能在不破坏训练期校准的前提下保住 AUC 并把 Acc 拉回去。
   - 实现口径核对：代码口径核对：相对 v21 额外只加载 `cel_selector.pt` 到 `_cel_selector`，不会同时 warm-start calibrator，也不会把 backbone / LoRA 一起从 `v1` 继续加载。
   - 期望观察 1：Overall AUC 应至少贴近 v21/v1 的 75.26/75.29，而不是像全量 warm-start 的 v22 那样回落。
   - 期望观察 2：Pred True 应比 v21 的 32.59% 更接近 baseline 区间，同时不要漂移到 v22 的过高 True 区间。
   - 期望观察 3：Overall Acc 应至少超过 v21 的 68.61，并争取越过 baseline 69.22。
   - 若未双超时的下一轮解释：如果 AUC 仍然保不住，说明 selector warm-start 也不足以稳定 joint full-train；下一轮应只再改一个训练动力学变量，而不是继续叠加更多 warm-start 部件。若 AUC 保住但 Acc 仍不过线，则下一轮只应继续围绕训练期校准强度或学习率改一个变量。
5. `cel_task_conditioned_lastlayer_v25_joint_bias_calibratorlr10x_qwen3_1.7b`
   - 从 baseline checkpoint 起步
   - 保持 v21 的 joint bias full-train 训练口径，只给 calibrator 单独更高的学习率
   - 唯一主改动：在 v21 的 joint bias full-train 基础上，仅给 calibrator 单独更高的学习率
   - 目标假设：如果 v21 和 v23 失败的主要原因是 calibrator 在共享学习率下几乎无法摆脱 no-op 动力学，那么只提高 calibrator 的训练步幅，应能把 Pred True 和默认阈值 Acc 往 baseline 区间拉回，同时尽量保住 v21 的排序能力。
   - 期望观察 1：训练结束后 calibrator bias 应明显偏离 v21 的近零终点，而不是继续停在 no-op 附近。
   - 期望观察 2：Pred True 应从 v21 的 32.59% 往 baseline 40.8% 区间回升，但不要像 v22 那样过冲到明显过高的 True 率。
   - 期望观察 3：Overall AUC 应尽量贴近 v21 的 75.26，同时 Overall Acc 至少要超过 v21 的 68.61 并争取越过 baseline 69.22。
   - 若未双超时的下一轮解释：如果 calibrator bias 终于动起来但 AUC 明显回落，说明更高 calibrator 学习率正在过度扰动联合训练，下一轮应只减小该学习率或改调度；如果 bias 仍然几乎不动，说明问题不在步幅本身，下一轮应只改另一个训练动力学变量，例如 warmup 或阶段性冻结。
6. `cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b`
   - 从 Qwen3-1.7B 原始基座开始，新建当前 A 模块训练链自己的 LoRA、A 和 calibrator
   - A 模块自包含三阶段链：表征 bootstrap、calibrator-only warmup、strict tiny-LR joint training
   - 唯一主改动：移除 baseline/v1 checkpoint 初始化，每个阶段只加载同一 A 模块 candidate 上一阶段生成的 checkpoint
   - 目标假设：从 Qwen3 原始基座独立训练的 A 模块可以自行学习 KT LoRA 和 task-conditioned selector，在完成概率校准后通过 strict joint 阶段保留有效参数更新，而无需继承 baseline 或 v1 权重。
   - 期望观察 1：bootstrap 日志必须显示新建 LoRA，且不能出现 baseline、v1 或旧 v26 checkpoint 路径。
   - 期望观察 2：calibrator warmup 必须保持同一 candidate 的 LoRA/A 不变，strict joint 阶段必须让至少一个 A 模块可训练组件发生变化。
   - 期望观察 3：最终报告必须同时比较 Overall Acc/AUC 与 Final Acc/AUC。
   - 若未双超时的下一轮解释：如果来源或参数变化审计失败，无论指标如何都判为无效；若审计通过但指标未过 baseline gate，则先把独立训练的 A 模块 candidate 记录为 non-winner，再只调整一个训练变量。

设计意图：

- 满足“完整方法训练完成后再评测”的严格口径
- 不再使用冻结补训、固定 bias 评估或 validation-fit 后处理作为最终分数

### Round 3: Valid End-to-End Full Training Results

| Model | Status | Overall Acc | Overall AUC | Final Acc | Final AUC | Pred True | Best Acc | Best Threshold | Delta Overall Acc | Delta Overall AUC | Delta Final Acc | Delta Final AUC | Overall Gate | Final Pair |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `cel_task_conditioned_lastlayer_v21_joint_bias_fulltrain_qwen3_1.7b` | done | 68.61 | 75.26 | 56.70 | 74.60 | 32.59% | 69.52 | 0.393 | -0.61 | +0.31 | -4.47 | -0.72 | no | no |
| `cel_task_conditioned_lastlayer_v22_fullv1_joint_bias_tinylr_qwen3_1.7b` | done | 68.26 | 74.93 | 64.27 | 72.86 | 47.76% | 69.07 | 0.406 | -0.96 | -0.02 | +3.10 | -2.46 | no | partial |
| `cel_task_conditioned_lastlayer_v23_joint_bias_posinit_qwen3_1.7b` | done | 68.41 | 74.29 | 56.12 | 74.59 | 33.50% | 68.87 | 0.385 | -0.81 | -0.66 | -5.05 | -0.73 | no | no |
| `cel_task_conditioned_lastlayer_v24_selectorinit_joint_bias_fulltrain_qwen3_1.7b` | done | 68.16 | 74.44 | 56.89 | 73.49 | 36.22% | 68.87 | 0.430 | -1.06 | -0.51 | -4.28 | -1.83 | no | no |
| `cel_task_conditioned_lastlayer_v25_joint_bias_calibratorlr10x_qwen3_1.7b` | done | 66.90 | 74.98 | 52.04 | 74.27 | 30.18% | 69.87 | 0.392 | -2.32 | +0.03 | -9.13 | -1.05 | no | no |
| `cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b` | done | 70.03 | 75.38 | 66.80 | 77.26 | 43.98% | 70.53 | 0.520 | +0.81 | +0.43 | +5.63 | +1.94 | yes | yes |

### Round 3 Decision

- Round 3 已经找到符合严格口径的 end-to-end 候选：`cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b` (Acc **70.03**, AUC **75.38**)。

## Global Decision

- 当前已有符合严格口径的模型同时超过 baseline 的 Acc/AUC：`cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b` (Acc **70.03**, AUC **75.38**)。
- `task_conditioned` Stage 1 调优目标已完成；停止新增 formal candidate，后续只做结果总结、误差分析和必要的 strict 复验。

## Winner Closeout

- 历史上的 `v11 / v12 / v15 / v16 / v19 / v20` 全部降级为诊断或后处理结果，不再作为正式 winner 判定依据。
- 当前 A 模块参考结果已固定，历史 checkpoint 为 `cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b`；当前 `task_conditioned` 调优停止新增 formal candidate，也不再自动重启任何旧候选。
- 如需回看当前 winner，可直接使用：`scripts/cel_stage1_last_layer/review_formal_candidate.sh v26`。
- 若当前在 WSL alias 路径下回看，也可用：`scripts/cel_stage1_last_layer/review_formal_candidate_via_ssh_alias.sh 3090 v26`。
- 如需再次确认 winner closeout，可使用：`scripts/cel_stage1_last_layer/finalize_formal_candidate_via_ssh_alias.sh 3090 v26`。
- Stage 1 调优目标已完成；Stage 2 首轮三方向虽已完成训练与审计，但采用旧单路径流程，只保留为历史 pilot。下一步先实现 `h_r/h_m` 双路径前向、`L_r + L_m + L_cons` 三项核心损失、tensor contracts 和 audit，再注册容量充分的 B 主候选。
- 已准备好的底层候选脚本：
  1. `scripts/cel_stage1_last_layer/run_task_conditioned_v21_joint_bias_fulltrain.sh`
  2. `scripts/cel_stage1_last_layer/run_task_conditioned_v22_fullv1_joint_bias_tinylr.sh`
  3. `scripts/cel_stage1_last_layer/run_task_conditioned_v23_joint_bias_posinit.sh`
  4. `scripts/cel_stage1_last_layer/run_task_conditioned_v24_selectorinit_joint_bias_fulltrain.sh`
  5. `scripts/cel_stage1_last_layer/run_task_conditioned_v25_joint_bias_calibratorlr10x.sh`
  6. `scripts/cel_stage1_last_layer/run_task_conditioned_v26_selftrained.sh`
- 诊断脚本仍保留，但仅作参考，不参与正式 winner 判定。
- 设计意图：直接验证“把 bias calibrator 当成方法一部分、完整训练后是否能双超 baseline”。

## Strict Full-Train Execution Loop

1. 只有当代码改动、启动脚本和记录口径都同步完成后，才允许启动新的 `task_conditioned` 正式候选。
2. 正式候选必须在 SSH 服务器上完成完整 `train + val + test`；冻结补训、fixed-bias eval、validation-fit bias 都只算诊断。
3. 训练运行后，优先使用 `scripts/cel_stage1_last_layer/finalize_current_formal_candidate.sh` 做统一 closeout：sync `metrics / qual / stdout log`、刷新 `STATUS.md / STRICT_FULL_TRAIN_REPORT.md / FORMAL_EXPERIMENT_AUDIT.md / COMPARISON.md / DETAILED_ANALYSIS.md / TASK_CONDITIONED_TUNING_LOOP.md`。
4. closeout 完成后先检查 `FORMAL_EXPERIMENT_AUDIT.md` 是否已经对已完成 formal candidate 给出完整 artifact + markdown coverage。
5. 只有完整结果全部落盘且 audit 干净后，才通过 `review_current_formal_candidate.sh` 判断是否双超 baseline；中途的 val 感觉更好、threshold 更好，都不能提前宣布 winner。
6. 若本轮失败或未双超，记录 Overall/Final 四项指标、相对 baseline 的四项 delta、失败类型、`Pred True`，以及下一轮只改一个主要变量的原因。

## Monitoring Cadence

- 当前没有 active strict full-train run；winner 已完成 closeout，本地 controller / watcher 无需持续轮询，推荐监控间隔为 `not_applicable`。
- 如需核对 winner 工件或文档一致性，优先使用一次性的 `review` / `finalize` / `sync` 命令，而不是重新启动 controller。
- 只有未来明确发起新的 strict SSH full-train 复验时，才恢复共享自适应监控策略：训练早期 **900s**，稳定中期 **600s**，进入 `Validation / Testing`、达到 **95%+** 进度或剩余 ETA 很短时收紧到 **300s**，而 `<=5m` 时进一步收紧到 **120s**。

## 记录要求

- 每轮结束后补充：
  - 模型名
  - Overall Acc / AUC
  - Final Acc / AUC
  - `Pred True`
  - Overall Acc / AUC 与 Final Acc / AUC 相对 baseline 的四项差值
  - 是否同时超过 baseline 的 Overall Acc 和 Overall AUC
  - Final Acc / AUC 相对 baseline 的差值，以及是否同时超过 baseline
  - 若未超过，写明下一轮调整原因
- 对已完成 formal SSH run，额外确认 `FORMAL_EXPERIMENT_AUDIT.md` 已显示 artifact 完整且主要 markdown surfaces 已覆盖。
