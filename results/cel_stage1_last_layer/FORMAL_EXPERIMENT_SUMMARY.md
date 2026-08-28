# Formal Experiment Summary

> Terminology note: candidate identifiers in this historical table are retained for provenance. The recorded successful design is referred to as the **A module** outside the experiment ledger.

- Generated at: `2026-08-15T15:35:37`
- Scope: strict formal `task_conditioned` candidates only; diagnostic or post-hoc calibration runs are excluded from method judgment.
- Current next action: `done`
- Winner found: `True`
- Active formal candidate(s): `none`
- Baseline gate: `Overall Acc > 69.22` and `Overall AUC > 74.95`
- Historical selector/ranking comparison anchor: `cel_task_conditioned_lastlayer_v1_qwen3_1.7b` with Overall Acc / AUC `68.87 / 75.29`
- Audit counts: `recorded=6`

## Candidate Table

| Candidate | Start Point | Single Variable | Status | Audit | Overall Acc | Overall AUC | Final Acc | Final AUC | Delta Overall Acc | Delta Overall AUC | Delta Final Acc | Delta Final AUC | Pred True | Beats Baseline |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `v21` / `cel_task_conditioned_lastlayer_v21_joint_bias_fulltrain_qwen3_1.7b` | 从 baseline checkpoint 起步 | 把 bias calibrator 从后处理调节前移为训练期联合优化的一部分 | done | `recorded` | 68.61 | 75.26 | 56.70 | 74.60 | -0.61 | 0.31 | -4.47 | -0.72 | 32.59% | no |
| `v22` / `cel_task_conditioned_lastlayer_v22_fullv1_joint_bias_tinylr_qwen3_1.7b` | 从完整 `task_conditioned v1` checkpoint 起步 | 把主变量改成从完整 `v1` checkpoint 出发、只用极小学习率做联合微调 | done | `recorded` | 68.26 | 74.93 | 64.27 | 72.86 | -0.96 | -0.02 | 3.10 | -2.46 | 47.76% | no |
| `v23` / `cel_task_conditioned_lastlayer_v23_joint_bias_posinit_qwen3_1.7b` | 从 baseline checkpoint 起步 | 把 joint bias full-train 的 calibrator 初始化从零改成基于诊断证据的正偏置起点 | done | `recorded` | 68.41 | 74.29 | 56.12 | 74.59 | -0.81 | -0.66 | -5.05 | -0.73 | 33.50% | no |
| `v24` / `cel_task_conditioned_lastlayer_v24_selectorinit_joint_bias_fulltrain_qwen3_1.7b` | 从 baseline checkpoint 起步 | 在 v21 的 joint bias full-train 基础上，仅把 selector 初始化改为来自历史 selector/ranking 锚点 v1 | done | `recorded` | 68.16 | 74.44 | 56.89 | 73.49 | -1.06 | -0.51 | -4.28 | -1.83 | 36.22% | no |
| `v25` / `cel_task_conditioned_lastlayer_v25_joint_bias_calibratorlr10x_qwen3_1.7b` | 从 baseline checkpoint 起步 | 在 v21 的 joint bias full-train 基础上，仅给 calibrator 单独更高的学习率 | done | `recorded` | 66.90 | 74.98 | 52.04 | 74.27 | -2.32 | 0.03 | -9.13 | -1.05 | 30.18% | no |
| `v26` / `cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b` | 从 Qwen3-1.7B 原始基座开始，新建当前 A 模块训练链自己的 LoRA、A 和 calibrator | 移除 baseline/v1 checkpoint 初始化，每个阶段只加载同一 A 模块 candidate 上一阶段生成的 checkpoint | done | `recorded` | 70.03 | 75.38 | 66.80 | 77.26 | 0.81 | 0.43 | 5.63 | 1.94 | 43.98% | yes |

## Per-Candidate Analysis

### `v21` / `cel_task_conditioned_lastlayer_v21_joint_bias_fulltrain_qwen3_1.7b`

- 起点：从 baseline checkpoint 起步
- 方法：把 bias calibrator 作为方法一部分，从训练开始就联合训练全部可训练参数
- 单变量：把 bias calibrator 从后处理调节前移为训练期联合优化的一部分
- 假设：如果 `v1` 的主要缺口来自概率过于保守，那么把 bias calibrator 从训练开始就纳入联合优化，应能在不破坏排序能力的前提下把 0.5 阈值下的 Acc 拉回去。
- 当前状态：`done` / audit=`recorded` / beats_baseline=`no`
- 当前结果：Overall Acc / AUC `68.61 / 75.26`；Final Acc / AUC `56.70 / 74.60`；`Pred True` `32.59%`；best Acc `69.52` @ threshold `0.3926`
- 相对 baseline：delta Acc `-0.61`；delta AUC `0.31`；delta Final Acc `-4.47`；delta Final AUC `-0.72`
- 相对 anchor `v1`：delta Acc `-0.26`；delta AUC `-0.03`
- 分析：AUC 仍高于或接近 baseline，但固定阈值下的 Acc 仍未过线；相对历史 selector/ranking 比较锚点 `v1`，排序优势也没有稳住；`Pred True` 为 32.59%。
- Closeout note: Full-train artifacts are present and the candidate's result is reflected across all required markdown surfaces.

### `v22` / `cel_task_conditioned_lastlayer_v22_fullv1_joint_bias_tinylr_qwen3_1.7b`

- 起点：从完整 `task_conditioned v1` checkpoint 起步
- 方法：保持全部相关参数可训练，用极小学习率做完整联合微调
- 单变量：把主变量改成从完整 `v1` checkpoint 出发、只用极小学习率做联合微调
- 假设：如果 `v1` 已经具备更好的排序，主要问题只是默认阈值下过于保守，那么从 `v1` warm-start 并用极小学习率联调，可能在尽量不破坏 AUC 的前提下把概率尺度推回更有利于 Acc 的区间。
- 当前状态：`done` / audit=`recorded` / beats_baseline=`no`
- 当前结果：Overall Acc / AUC `68.26 / 74.93`；Final Acc / AUC `64.27 / 72.86`；`Pred True` `47.76%`；best Acc `69.07` @ threshold `0.4063`
- 相对 baseline：delta Acc `-0.96`；delta AUC `-0.02`；delta Final Acc `3.10`；delta Final AUC `-2.46`
- 相对 anchor `v1`：delta Acc `-0.61`；delta AUC `-0.36`
- 分析：AUC 与 Acc 都未能同时保住，说明排序与阈值表现都没有完成 formal gate；相对历史 selector/ranking 比较锚点 `v1`，排序优势也没有稳住；`Pred True` 为 47.76%。
- Closeout note: Full-train artifacts are present and the candidate's result is reflected across all required markdown surfaces.

### `v23` / `cel_task_conditioned_lastlayer_v23_joint_bias_posinit_qwen3_1.7b`

- 起点：从 baseline checkpoint 起步
- 方法：保持 v21 的 joint bias full-train 训练口径，只把 calibrator 初始化改成正偏置起点
- 单变量：把 joint bias full-train 的 calibrator 初始化从零改成基于诊断证据的正偏置起点
- 假设：如果 v21 失败的主要原因是 joint bias 在零初始化下几乎没有被推离 no-op 状态，那么把初始 bias 设为 v15 已验证过的正偏置起点，应能在保住 AUC 的同时把 Pred True 和默认阈值 Acc 往 baseline 区间拉回。
- 当前状态：`done` / audit=`recorded` / beats_baseline=`no`
- 当前结果：Overall Acc / AUC `68.41 / 74.29`；Final Acc / AUC `56.12 / 74.59`；`Pred True` `33.50%`；best Acc `68.87` @ threshold `0.3852`
- 相对 baseline：delta Acc `-0.81`；delta AUC `-0.66`；delta Final Acc `-5.05`；delta Final AUC `-0.73`
- 相对 anchor `v1`：delta Acc `-0.46`；delta AUC `-1.00`
- 分析：AUC 与 Acc 都未能同时保住，说明排序与阈值表现都没有完成 formal gate；相对历史 selector/ranking 比较锚点 `v1`，排序优势也没有稳住；`Pred True` 为 33.50%。
- Closeout note: Full-train artifacts are present and the candidate's result is reflected across all required markdown surfaces.

### `v24` / `cel_task_conditioned_lastlayer_v24_selectorinit_joint_bias_fulltrain_qwen3_1.7b`

- 起点：从 baseline checkpoint 起步
- 方法：保持 v21 的 joint bias full-train 训练口径，只把 selector 初始化改成来自历史 selector/ranking 锚点 v1
- 单变量：在 v21 的 joint bias full-train 基础上，仅把 selector 初始化改为来自历史 selector/ranking 锚点 v1
- 假设：如果 v21 的问题主要不是联训 bias 本身，而是从 baseline 冷启动时丢失了 v1 已学到的排序信号，那么仅用 v1 warm-start selector 应能在不破坏训练期校准的前提下保住 AUC 并把 Acc 拉回去。
- 当前状态：`done` / audit=`recorded` / beats_baseline=`no`
- 当前结果：Overall Acc / AUC `68.16 / 74.44`；Final Acc / AUC `56.89 / 73.49`；`Pred True` `36.22%`；best Acc `68.87` @ threshold `0.4296`
- 相对 baseline：delta Acc `-1.06`；delta AUC `-0.51`；delta Final Acc `-4.28`；delta Final AUC `-1.83`
- 相对 anchor `v1`：delta Acc `-0.71`；delta AUC `-0.85`
- 分析：AUC 与 Acc 都未能同时保住，说明排序与阈值表现都没有完成 formal gate；相对历史 selector/ranking 比较锚点 `v1`，排序优势也没有稳住；`Pred True` 为 36.22%。
- Closeout note: Full-train artifacts are present and the candidate's result is reflected across all required markdown surfaces.

### `v25` / `cel_task_conditioned_lastlayer_v25_joint_bias_calibratorlr10x_qwen3_1.7b`

- 起点：从 baseline checkpoint 起步
- 方法：保持 v21 的 joint bias full-train 训练口径，只给 calibrator 单独更高的学习率
- 单变量：在 v21 的 joint bias full-train 基础上，仅给 calibrator 单独更高的学习率
- 假设：如果 v21 和 v23 失败的主要原因是 calibrator 在共享学习率下几乎无法摆脱 no-op 动力学，那么只提高 calibrator 的训练步幅，应能把 Pred True 和默认阈值 Acc 往 baseline 区间拉回，同时尽量保住 v21 的排序能力。
- 当前状态：`done` / audit=`recorded` / beats_baseline=`no`
- 当前结果：Overall Acc / AUC `66.90 / 74.98`；Final Acc / AUC `52.04 / 74.27`；`Pred True` `30.18%`；best Acc `69.87` @ threshold `0.3923`
- 相对 baseline：delta Acc `-2.32`；delta AUC `0.03`；delta Final Acc `-9.13`；delta Final AUC `-1.05`
- 相对 anchor `v1`：delta Acc `-1.97`；delta AUC `-0.31`
- 分析：AUC 仍高于或接近 baseline，但固定阈值下的 Acc 仍未过线；相对历史 selector/ranking 比较锚点 `v1`，排序优势也没有稳住；`Pred True` 为 30.18%。
- Closeout note: Full-train artifacts are present and the candidate's result is reflected across all required markdown surfaces.

### `v26` / `cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b`

- 起点：从 Qwen3-1.7B 原始基座开始，新建当前 A 模块训练链自己的 LoRA、A 和 calibrator
- 方法：A 模块自包含三阶段链：表征 bootstrap、calibrator-only warmup、strict tiny-LR joint training
- 单变量：移除 baseline/v1 checkpoint 初始化，每个阶段只加载同一 A 模块 candidate 上一阶段生成的 checkpoint
- 假设：从 Qwen3 原始基座独立训练的 A 模块可以自行学习 KT LoRA 和 task-conditioned selector，在完成概率校准后通过 strict joint 阶段保留有效参数更新，而无需继承 baseline 或 v1 权重。
- 当前状态：`done` / audit=`recorded` / beats_baseline=`yes`
- 当前结果：Overall Acc / AUC `70.03 / 75.38`；Final Acc / AUC `66.80 / 77.26`；`Pred True` `43.98%`；best Acc `70.53` @ threshold `0.5196`
- 相对 baseline：delta Acc `0.81`；delta AUC `0.43`；delta Final Acc `5.63`；delta Final AUC `1.94`
- 相对 anchor `v1`：delta Acc `1.16`；delta AUC `0.09`
- 分析：该候选已经在完整 SSH full-train 口径下同时超过 baseline 的 Overall Acc 和 Overall AUC，可视为 formal winner。
- Closeout note: Full-train artifacts are present and the candidate's result is reflected across all required markdown surfaces.
