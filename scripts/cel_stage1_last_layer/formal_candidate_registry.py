#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path


FORMAL_CANDIDATE_ORDER = [
    "cel_task_conditioned_lastlayer_v21_joint_bias_fulltrain_qwen3_1.7b",
    "cel_task_conditioned_lastlayer_v22_fullv1_joint_bias_tinylr_qwen3_1.7b",
    "cel_task_conditioned_lastlayer_v23_joint_bias_posinit_qwen3_1.7b",
    "cel_task_conditioned_lastlayer_v24_selectorinit_joint_bias_fulltrain_qwen3_1.7b",
    "cel_task_conditioned_lastlayer_v25_joint_bias_calibratorlr10x_qwen3_1.7b",
    "cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b",
]

FORMAL_CANDIDATES = {
    "cel_task_conditioned_lastlayer_v21_joint_bias_fulltrain_qwen3_1.7b": {
        "launch_key": "v21",
        "label": "v21",
        "model_pattern": "task_conditioned_lastlayer_v21",
        "start_point_en": "baseline checkpoint",
        "start_point_cn": "从 baseline checkpoint 起步",
        "method_en": "joint bias calibrator full training",
        "method_cn": "把 bias calibrator 作为方法一部分，从训练开始就联合训练全部可训练参数",
        "single_variable_en": "move bias calibration into train-time joint optimization instead of post-hoc adjustment",
        "single_variable_cn": "把 bias calibrator 从后处理调节前移为训练期联合优化的一部分",
        "config_reference_rel": "scripts/cel_stage1_last_layer/run_task_conditioned_v1.sh",
        "config_reference_label_cn": "task_conditioned v1 完整训练脚本",
        "config_diff_keys": [
            "model_name",
            "lr",
            "wd",
            "cel_output_calibration",
            "cel_calibrator_init_bias",
        ],
        "config_diff_rationale_cn": "这些配置差异共同实现同一个语义变量：把原本只在诊断轮里出现的 bias calibrator 前移到完整训练流程中，并为该联训路径显式指定优化超参数。",
        "hypothesis_en": "If the main weakness of `v1` is conservative probability calibration, training the bias calibrator jointly from the start should recover thresholded accuracy without giving up ranking quality.",
        "hypothesis_cn": "如果 `v1` 的主要缺口来自概率过于保守，那么把 bias calibrator 从训练开始就纳入联合优化，应能在不破坏排序能力的前提下把 0.5 阈值下的 Acc 拉回去。",
        "expected_signals_en": [
            "`Pred True` should move upward from the conservative `v1` regime toward the baseline level instead of staying suppressed.",
            "Overall Acc should recover toward or above baseline while Overall AUC stays at least near the historical selector/ranking comparison anchor.",
            "If the method is valid, the gain should appear in one complete SSH-side train + val + test run rather than in post-hoc calibration only.",
        ],
        "expected_signals_cn": [
            "`Pred True` 应从 `v1` 的保守区间往 baseline 水平回升，而不是继续明显偏低。",
            "Overall Acc 应相对 `v1` 回升并争取超过 baseline，同时 Overall AUC 至少不要明显低于历史 selector/ranking 比较锚点。",
            "若该方法成立，收益必须体现在一次完整 SSH `train + val + test` 里，而不是只在后处理校准里出现。",
        ],
        "if_fail_en": "If AUC stays competitive but Acc still misses the gate, the next formal candidate should change only one calibration-related variable such as initialization, strength, or learning rate. If AUC also falls, joint bias training is damaging ranking and the next candidate should switch to one more controlled train-time variable instead of stacking more post-hoc fixes.",
        "if_fail_cn": "如果 AUC 仍有竞争力但 Acc 还是不过线，下一轮正式候选只应围绕校准强度、初始化或学习率再改一个变量；如果 AUC 也回落，说明联合 bias 训练已经伤害排序，下一轮应改成一个更受控的训练期主变量，而不是继续叠加后处理补丁。",
        "scope_en": "SSH train + val + test",
        "stdout_log_name": "task_conditioned_v21_joint_bias_fulltrain.stdout.log",
        "stdout_log_rel": "results/cel_stage1_last_layer/task_conditioned_v21_joint_bias_fulltrain.stdout.log",
        "metrics_rel": "results/cel_stage1_last_layer/metrics/metrics_cel_task_conditioned_lastlayer_v21_joint_bias_fulltrain_qwen3_1.7b.txt",
        "run_script_name": "run_task_conditioned_v21_joint_bias_fulltrain.sh",
        "run_script_rel": "scripts/cel_stage1_last_layer/run_task_conditioned_v21_joint_bias_fulltrain.sh",
        "default_gpu_id": "0",
        "planned_epochs": 2,
        "prior_failure_progress": {
            "epoch": 1,
            "phase": "training",
            "step": 5376,
            "total": 8425,
            "pct": 64,
        },
    },
    "cel_task_conditioned_lastlayer_v22_fullv1_joint_bias_tinylr_qwen3_1.7b": {
        "launch_key": "v22",
        "label": "v22",
        "model_pattern": "task_conditioned_lastlayer_v22",
        "start_point_en": "full `v1` checkpoint",
        "start_point_cn": "从完整 `task_conditioned v1` checkpoint 起步",
        "method_en": "tiny-lr joint full fine-tune",
        "method_cn": "保持全部相关参数可训练，用极小学习率做完整联合微调",
        "single_variable_en": "switch from fresh full training to tiny-lr warm-start joint fine-tuning from the full `v1` checkpoint",
        "single_variable_cn": "把主变量改成从完整 `v1` checkpoint 出发、只用极小学习率做联合微调",
        "config_reference_rel": "scripts/cel_stage1_last_layer/run_task_conditioned_v21_joint_bias_fulltrain.sh",
        "config_reference_label_cn": "v21 联合 bias full-train 脚本",
        "config_diff_keys": [
            "pt_model_name",
            "cel_selector_init_model_name",
            "model_name",
            "epochs",
            "lr",
            "wd",
        ],
        "config_diff_rationale_cn": "这些配置差异共同实现同一个语义变量：把联训 bias 的起点从 baseline 完整训练切换为 `v1` warm-start，并把训练节奏降到 tiny-lr 微调。",
        "hypothesis_en": "If `v1` already has the stronger ranking and mainly lacks thresholded accuracy, a tiny-lr warm-start should preserve AUC while nudging the probability scale back toward a better default threshold.",
        "hypothesis_cn": "如果 `v1` 已经具备更好的排序，主要问题只是默认阈值下过于保守，那么从 `v1` warm-start 并用极小学习率联调，可能在尽量不破坏 AUC 的前提下把概率尺度推回更有利于 Acc 的区间。",
        "expected_signals_en": [
            "Overall AUC should stay close to `v1` rather than washing out the anchor's ranking advantage.",
            "`Pred True` should rise moderately toward the baseline region instead of drifting to an over-True regime.",
            "Overall Acc should move toward or above baseline if the warm-start calibration hypothesis is correct.",
        ],
        "expected_signals_cn": [
            "Overall AUC 应尽量贴近 `v1`，而不是把当前锚点的排序优势洗掉。",
            "`Pred True` 应温和上升并靠近 baseline，而不是直接漂移到过多 True 的区间。",
            "若 warm-start 校准假设成立，Overall Acc 应向 baseline 靠近并最好越过 baseline。",
        ],
        "if_fail_en": "If both Acc and AUC fall or the calibrator barely moves, tiny-lr warm-start is not solving the conservative-probability issue. The next formal candidate should change one more direct train-time calibration variable instead of repeating warm-start plus post-hoc style adjustments.",
        "if_fail_cn": "如果 Acc 和 AUC 一起回落，或 calibrator 几乎没动，说明 tiny-lr warm-start 既没有纠正保守概率，也没有稳住排序；下一轮正式候选应改一个更直接影响训练期校准的主变量，而不是继续重复 warm-start + 后处理式思路。",
        "scope_en": "SSH train + val + test",
        "stdout_log_name": "task_conditioned_v22_fullv1_joint_bias_tinylr.stdout.log",
        "stdout_log_rel": "results/cel_stage1_last_layer/task_conditioned_v22_fullv1_joint_bias_tinylr.stdout.log",
        "metrics_rel": "results/cel_stage1_last_layer/metrics/metrics_cel_task_conditioned_lastlayer_v22_fullv1_joint_bias_tinylr_qwen3_1.7b.txt",
        "run_script_name": "run_task_conditioned_v22_fullv1_joint_bias_tinylr.sh",
        "run_script_rel": "scripts/cel_stage1_last_layer/run_task_conditioned_v22_fullv1_joint_bias_tinylr.sh",
        "default_gpu_id": "1",
        "planned_epochs": 1,
    },
    "cel_task_conditioned_lastlayer_v23_joint_bias_posinit_qwen3_1.7b": {
        "launch_key": "v23",
        "label": "v23",
        "model_pattern": "task_conditioned_lastlayer_v23_joint_bias_posinit",
        "start_point_en": "baseline checkpoint",
        "start_point_cn": "从 baseline checkpoint 起步",
        "method_en": "keep the v21 joint-bias full-train setup but initialize the calibrator with a positive bias",
        "method_cn": "保持 v21 的 joint bias full-train 训练口径，只把 calibrator 初始化改成正偏置起点",
        "single_variable_en": "change the joint-bias full-train calibrator initialization from zero to a positive bias informed by prior diagnostic evidence",
        "single_variable_cn": "把 joint bias full-train 的 calibrator 初始化从零改成基于诊断证据的正偏置起点",
        "config_reference_rel": "scripts/cel_stage1_last_layer/run_task_conditioned_v21_joint_bias_fulltrain.sh",
        "config_reference_label_cn": "v21 联合 bias full-train 脚本",
        "config_diff_keys": [
            "model_name",
            "cel_calibrator_init_bias",
        ],
        "config_diff_rationale_cn": "相对 v21 只改一个语义变量：保留 joint bias full-train 训练口径不变，只把 calibrator 的初始 bias 从 0.0 改成一个有诊断依据的正起点。",
        "hypothesis_en": "If v21 failed mainly because the joint bias calibrator stayed near a no-op zero initialization, seeding it with the positive bias starting point already validated by v15 should pull Pred True and default-threshold accuracy back toward the baseline region while preserving AUC.",
        "hypothesis_cn": "如果 v21 失败的主要原因是 joint bias 在零初始化下几乎没有被推离 no-op 状态，那么把初始 bias 设为 v15 已验证过的正偏置起点，应能在保住 AUC 的同时把 Pred True 和默认阈值 Acc 往 baseline 区间拉回。",
        "expected_signals_en": [
            "Pred True should rise clearly from the v21 level of 32.59 percent and move closer to the baseline region of 40.8 percent instead of staying suppressed.",
            "Overall Acc should at least beat the v21 value of 68.61 and ideally clear the 69.22 baseline while Overall AUC stays near the v21 and v1 range of 75.26 and 75.29.",
            "The learned calibrator bias should remain meaningfully positive after training instead of collapsing back near zero as in v21.",
        ],
        "expected_signals_cn": [
            "Pred True 应从 v21 的 32.59% 明显回升，并尽量靠近 baseline 40.8%，而不是继续卡在保守区间。",
            "Overall Acc 应至少超过 v21 的 68.61，并争取越过 baseline 69.22，同时 Overall AUC 不应明显低于 v21/v1 的 75.26/75.29。",
            "训练结束后 calibrator bias 应保持显著正值，而不是像 v21 一样几乎回到 0。",
        ],
        "if_fail_en": "If Pred True remains clearly too low and the calibrator bias collapses back near zero, positive initialization alone is not enough and the next candidate should change exactly one training-dynamics variable such as calibrator learning rate or warmup. If Pred True overshoots and AUC falls, reduce the initialization magnitude rather than stacking more variables.",
        "if_fail_cn": "如果 Pred True 仍然明显偏低且 calibrator bias 又回到接近 0，说明单纯正初始化不足以驱动联训校准；下一轮只应再改一个训练动力学变量，例如 calibrator 学习率或 warmup。若 Pred True 过冲而 AUC 回落，则应减小初始化幅度，而不是叠加更多变量。",
        "scope_en": "SSH train + val + test",
        "stdout_log_name": "task_conditioned_v23_joint_bias_posinit.stdout.log",
        "stdout_log_rel": "results/cel_stage1_last_layer/task_conditioned_v23_joint_bias_posinit.stdout.log",
        "metrics_rel": "results/cel_stage1_last_layer/metrics/metrics_cel_task_conditioned_lastlayer_v23_joint_bias_posinit_qwen3_1.7b.txt",
        "run_script_name": "run_task_conditioned_v23_joint_bias_posinit.sh",
        "run_script_rel": "scripts/cel_stage1_last_layer/run_task_conditioned_v23_joint_bias_posinit.sh",
        "default_gpu_id": "0",
        "planned_epochs": 2,
    },
    "cel_task_conditioned_lastlayer_v24_selectorinit_joint_bias_fulltrain_qwen3_1.7b": {
        "launch_key": "v24",
        "label": "v24",
        "model_pattern": "task_conditioned_lastlayer_v24_selectorinit_joint_bias_fulltrain",
        "start_point_en": "baseline checkpoint",
        "start_point_cn": "从 baseline checkpoint 起步",
        "method_en": "keep the v21 joint-bias full-train setup but initialize the selector from the historical selector/ranking v1 anchor",
        "method_cn": "保持 v21 的 joint bias full-train 训练口径，只把 selector 初始化改成来自历史 selector/ranking 锚点 v1",
        "single_variable_en": "keep the v21 joint-bias full-train setup but initialize only the selector from the historical selector/ranking v1 anchor",
        "single_variable_cn": "在 v21 的 joint bias full-train 基础上，仅把 selector 初始化改为来自历史 selector/ranking 锚点 v1",
        "config_reference_rel": "scripts/cel_stage1_last_layer/run_task_conditioned_v21_joint_bias_fulltrain.sh",
        "config_reference_label_cn": "v21 联合 bias full-train 脚本",
        "config_diff_keys": [
            "model_name",
            "cel_selector_init_model_name",
        ],
        "config_diff_rationale_cn": "相对 v21 只改一个语义变量：保持 baseline checkpoint 起步和 joint bias full-train 训练口径不变，只把 selector 初始化改成来自历史 selector/ranking 锚点 v1。",
        "implementation_guard_cn": "代码口径核对：相对 v21 额外只加载 `cel_selector.pt` 到 `_cel_selector`，不会同时 warm-start calibrator，也不会把 backbone / LoRA 一起从 `v1` 继续加载。",
        "hypothesis_en": "If the main issue in v21 is not joint bias training itself but losing the ranking signal that v1 already learned from a cold baseline start, warm-starting only the selector from v1 should preserve AUC while keeping train-time calibration in the loop for accuracy recovery.",
        "hypothesis_cn": "如果 v21 的问题主要不是联训 bias 本身，而是从 baseline 冷启动时丢失了 v1 已学到的排序信号，那么仅用 v1 warm-start selector 应能在不破坏训练期校准的前提下保住 AUC 并把 Acc 拉回去。",
        "expected_signals_en": [
            "Overall AUC should stay at least near the v21/v1 range of 75.26/75.29 instead of falling off as in the full warm-start v22 run.",
            "Pred True should move closer to the baseline region than v21 did without drifting into the over-True regime seen in v22.",
            "Overall Acc should at least beat the v21 value of 68.61 and ideally clear the 69.22 baseline.",
        ],
        "expected_signals_cn": [
            "Overall AUC 应至少贴近 v21/v1 的 75.26/75.29，而不是像全量 warm-start 的 v22 那样回落。",
            "Pred True 应比 v21 的 32.59% 更接近 baseline 区间，同时不要漂移到 v22 的过高 True 区间。",
            "Overall Acc 应至少超过 v21 的 68.61，并争取越过 baseline 69.22。",
        ],
        "if_fail_en": "If AUC still does not hold, selector warm-start is also insufficient to stabilize joint full-train and the next candidate should change exactly one training-dynamics variable rather than stacking more warm-start components. If AUC holds but Acc still misses, the next candidate should change only one train-time calibration strength or learning-rate variable.",
        "if_fail_cn": "如果 AUC 仍然保不住，说明 selector warm-start 也不足以稳定 joint full-train；下一轮应只再改一个训练动力学变量，而不是继续叠加更多 warm-start 部件。若 AUC 保住但 Acc 仍不过线，则下一轮只应继续围绕训练期校准强度或学习率改一个变量。",
        "scope_en": "SSH train + val + test",
        "stdout_log_name": "task_conditioned_v24_selectorinit_joint_bias_fulltrain.stdout.log",
        "stdout_log_rel": "results/cel_stage1_last_layer/task_conditioned_v24_selectorinit_joint_bias_fulltrain.stdout.log",
        "metrics_rel": "results/cel_stage1_last_layer/metrics/metrics_cel_task_conditioned_lastlayer_v24_selectorinit_joint_bias_fulltrain_qwen3_1.7b.txt",
        "run_script_name": "run_task_conditioned_v24_selectorinit_joint_bias_fulltrain.sh",
        "run_script_rel": "scripts/cel_stage1_last_layer/run_task_conditioned_v24_selectorinit_joint_bias_fulltrain.sh",
        "default_gpu_id": "0",
        "planned_epochs": 2,
    },
    "cel_task_conditioned_lastlayer_v25_joint_bias_calibratorlr10x_qwen3_1.7b": {
        "launch_key": "v25",
        "label": "v25",
        "model_pattern": "task_conditioned_lastlayer_v25_joint_bias_calibratorlr10x",
        "start_point_en": "baseline checkpoint",
        "start_point_cn": "从 baseline checkpoint 起步",
        "method_en": "keep the v21 joint-bias full-train setup but assign the calibrator a higher dedicated learning rate",
        "method_cn": "保持 v21 的 joint bias full-train 训练口径，只给 calibrator 单独更高的学习率",
        "single_variable_en": "keep the v21 joint-bias full-train setup but give the calibrator a higher dedicated learning rate",
        "single_variable_cn": "在 v21 的 joint bias full-train 基础上，仅给 calibrator 单独更高的学习率",
        "config_reference_rel": "scripts/cel_stage1_last_layer/run_task_conditioned_v21_joint_bias_fulltrain.sh",
        "config_reference_label_cn": "v21 联合 bias full-train 脚本",
        "config_diff_keys": [
            "model_name",
            "cel_calibrator_lr",
        ],
        "config_diff_rationale_cn": "相对 v21 只改一个语义变量：保持 baseline checkpoint 起步和 joint bias full-train 训练口径不变，只把 calibrator 从共享学习率切到更高的独立学习率。",
        "hypothesis_en": "If v21 and v23 mainly failed because the calibrator barely moved under the shared learning rate, increasing only the calibrator step size should pull Pred True and default-threshold accuracy back toward the baseline region while preserving the ranking quality seen in v21.",
        "hypothesis_cn": "如果 v21 和 v23 失败的主要原因是 calibrator 在共享学习率下几乎无法摆脱 no-op 动力学，那么只提高 calibrator 的训练步幅，应能把 Pred True 和默认阈值 Acc 往 baseline 区间拉回，同时尽量保住 v21 的排序能力。",
        "expected_signals_en": [
            "The learned calibrator bias should finish meaningfully farther from the near-zero v21 endpoint instead of staying trapped near a no-op solution.",
            "Pred True should rise from the v21 level of 32.59 percent toward the baseline region of 40.8 percent without overshooting into the clearly over-True regime seen in v22.",
            "Overall AUC should stay close to the v21 value of 75.26 while Overall Acc at least beats the v21 value of 68.61 and ideally clears the 69.22 baseline.",
        ],
        "expected_signals_cn": [
            "训练结束后 calibrator bias 应明显偏离 v21 的近零终点，而不是继续停在 no-op 附近。",
            "Pred True 应从 v21 的 32.59% 往 baseline 40.8% 区间回升，但不要像 v22 那样过冲到明显过高的 True 率。",
            "Overall AUC 应尽量贴近 v21 的 75.26，同时 Overall Acc 至少要超过 v21 的 68.61 并争取越过 baseline 69.22。",
        ],
        "if_fail_en": "If the calibrator bias finally moves but AUC drops clearly, the higher calibrator learning rate is overdriving joint training and the next candidate should only reduce that rate or change its schedule. If the bias still barely moves, the issue is not step size itself and the next candidate should change only one other training-dynamics variable such as warmup or staged freezing.",
        "if_fail_cn": "如果 calibrator bias 终于动起来但 AUC 明显回落，说明更高 calibrator 学习率正在过度扰动联合训练，下一轮应只减小该学习率或改调度；如果 bias 仍然几乎不动，说明问题不在步幅本身，下一轮应只改另一个训练动力学变量，例如 warmup 或阶段性冻结。",
        "scope_en": "SSH train + val + test",
        "stdout_log_name": "task_conditioned_v25_joint_bias_calibratorlr10x.stdout.log",
        "stdout_log_rel": "results/cel_stage1_last_layer/task_conditioned_v25_joint_bias_calibratorlr10x.stdout.log",
        "metrics_rel": "results/cel_stage1_last_layer/metrics/metrics_cel_task_conditioned_lastlayer_v25_joint_bias_calibratorlr10x_qwen3_1.7b.txt",
        "run_script_name": "run_task_conditioned_v25_joint_bias_calibratorlr10x.sh",
        "run_script_rel": "scripts/cel_stage1_last_layer/run_task_conditioned_v25_joint_bias_calibratorlr10x.sh",
        "default_gpu_id": "0",
        "planned_epochs": 2,
    },
    "cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b": {
        "launch_key": "v26",
        "label": "v26",
        "model_pattern": "run_task_conditioned_v26_selftrained|task_conditioned_lastlayer_v26_",
        "start_point_en": "Qwen3-1.7B base weights with newly initialized A-module LoRA, selector, and calibrator",
        "start_point_cn": "从 Qwen3-1.7B 原始基座开始，新建当前 A 模块训练链自己的 LoRA、A 和 calibrator",
        "method_en": "self-contained A-module chain: bootstrap representation training, calibrator-only warmup, then strict tiny-LR joint training",
        "method_cn": "A 模块自包含三阶段链：表征 bootstrap、calibrator-only warmup、strict tiny-LR joint training",
        "single_variable_en": "remove all baseline/v1 checkpoint initialization and allow each stage to load only the checkpoint produced by the preceding stage of the same A-module candidate",
        "single_variable_cn": "移除 baseline/v1 checkpoint 初始化，每个阶段只加载同一 A 模块 candidate 上一阶段生成的 checkpoint",
        "config_reference_rel": "scripts/cel_stage1_last_layer/run_task_conditioned_v26_selftrained.sh",
        "config_reference_label_cn": "A 模块独立三阶段总控脚本（历史 key：v26）",
        "config_diff_keys": [
            "model_name",
            "pt_model_name",
            "cel_selector_init_model_name",
            "cel_calibrator_init_model_name",
            "cel_calibrator_init_bias",
            "lr",
            "wd",
        ],
        "config_diff_rationale_cn": "这些配置共同保证当前 A 模块 candidate 不读取任何历史实验权重，并通过 bootstrap -> warmup -> joint 的同 candidate checkpoint 链完成训练；正式测试只读取 joint 阶段产物。",
        "multistage_contract": [
            {
                "label": "bootstrap",
                "run_script_rel": "scripts/cel_stage1_last_layer/run_task_conditioned_v26_bootstrap.sh",
                "expected_args": {
                    "model_name": "cel_task_conditioned_lastlayer_v26_bootstrap_qwen3_1.7b",
                    "epochs": "2",
                    "lr": "0.0002",
                    "wd": "0.01",
                    "skip_test_after_train": "1",
                    "cel_mode": "task_conditioned",
                },
                "forbidden_args": [
                    "pt_model_name",
                    "cel_selector_init_model_name",
                    "cel_calibrator_init_model_name",
                ],
            },
            {
                "label": "calibrator_warmup",
                "run_script_rel": "scripts/cel_stage1_last_layer/run_task_conditioned_v26_calibrator_warmup.sh",
                "expected_args": {
                    "pt_model_name": "cel_task_conditioned_lastlayer_v26_bootstrap_qwen3_1.7b",
                    "cel_selector_init_model_name": "cel_task_conditioned_lastlayer_v26_bootstrap_qwen3_1.7b",
                    "cel_require_exact_selector_init": "1",
                    "model_name": "cel_task_conditioned_lastlayer_v26_calibrator_warmup_qwen3_1.7b",
                    "epochs": "1",
                    "lr": "0.0002",
                    "wd": "0.0",
                    "skip_test_after_train": "1",
                    "cel_train_calibrator_only": "1",
                    "cel_output_calibration": "bias",
                },
                "forbidden_args": ["cel_calibrator_init_model_name"],
            },
            {
                "label": "joint",
                "run_script_rel": "scripts/cel_stage1_last_layer/run_task_conditioned_v26_joint.sh",
                "expected_args": {
                    "pt_model_name": "cel_task_conditioned_lastlayer_v26_calibrator_warmup_qwen3_1.7b",
                    "cel_selector_init_model_name": "cel_task_conditioned_lastlayer_v26_calibrator_warmup_qwen3_1.7b",
                    "cel_calibrator_init_model_name": "cel_task_conditioned_lastlayer_v26_calibrator_warmup_qwen3_1.7b",
                    "cel_require_exact_selector_init": "1",
                    "cel_require_complete_checkpoint": "1",
                    "model_name": "cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b",
                    "epochs": "1",
                    "lr": "0.00001",
                    "wd": "0.0",
                    "cel_output_calibration": "bias",
                },
                "forbidden_args": ["skip_test_after_train"],
            },
        ],
        "forbidden_initializer_names": [
            "lmkt_qwen3_1.7b_recert_20260620",
            "cel_task_conditioned_lastlayer_v1_qwen3_1.7b",
            "cel_task_conditioned_lastlayer_v26_fullv1_biaswarmup_joint_tinylr_qwen3_1.7b",
        ],
        "hypothesis_en": "A self-contained A module trained from Qwen3 base weights can learn its own KT LoRA and task-conditioned selector, calibrate its probability boundary, and retain useful updates after a strict joint phase without inheriting baseline or v1 parameters.",
        "hypothesis_cn": "从 Qwen3 原始基座独立训练的 A 模块可以自行学习 KT LoRA 和 task-conditioned selector，在完成概率校准后通过 strict joint 阶段保留有效参数更新，而无需继承 baseline 或 v1 权重。",
        "expected_signals_en": [
            "Bootstrap logs must show new LoRA initialization and contain no baseline, v1, or historical v26 checkpoint path.",
            "The calibrator warmup must leave the same-candidate LoRA and selector unchanged, while the strict joint phase must change at least one trainable A-module component.",
            "The final report must compare Overall Acc/AUC and Final Acc/AUC against the certified baseline.",
        ],
        "expected_signals_cn": [
            "bootstrap 日志必须显示新建 LoRA，且不能出现 baseline、v1 或旧 v26 checkpoint 路径。",
            "calibrator warmup 必须保持同一 candidate 的 LoRA/A 不变，strict joint 阶段必须让至少一个 A 模块可训练组件发生变化。",
            "最终报告必须同时比较 Overall Acc/AUC 与 Final Acc/AUC。",
        ],
        "if_fail_en": "If provenance or parameter-change audit fails, the run is invalid regardless of metrics. If provenance passes but metrics miss the baseline gate, record the independently trained A-module candidate as a non-winner before changing one training variable.",
        "if_fail_cn": "如果来源或参数变化审计失败，无论指标如何都判为无效；若审计通过但指标未过 baseline gate，则先把独立训练的 A 模块 candidate 记录为 non-winner，再只调整一个训练变量。",
        "scope_en": "SSH train + val + test",
        "stdout_log_name": "task_conditioned_v26_selftrained.stdout.log",
        "stdout_log_rel": "results/cel_stage1_last_layer/task_conditioned_v26_selftrained.stdout.log",
        "metrics_rel": "results/cel_stage1_last_layer/metrics/metrics_cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b.txt",
        "run_script_name": "run_task_conditioned_v26_selftrained.sh",
        "run_script_rel": "scripts/cel_stage1_last_layer/run_task_conditioned_v26_selftrained.sh",
        "default_gpu_id": "0",
        "planned_epochs": 4,
    },
}

FORMAL_CANDIDATE_MODELS = tuple(FORMAL_CANDIDATE_ORDER)
FORMAL_CANDIDATE_LAUNCH_KEYS = {
    meta["launch_key"]: model_name for model_name, meta in FORMAL_CANDIDATES.items()
}
FORMAL_CANDIDATE_LOG_FILES = {
    model_name: meta["stdout_log_name"] for model_name, meta in FORMAL_CANDIDATES.items()
}

ROUND3_FAILURE_FOLLOWUPS_EN = {
    "cel_task_conditioned_lastlayer_v21_joint_bias_fulltrain_qwen3_1.7b": [
        "Current local fix: `dialogue_kt/training.py` now sanitizes labels and computes the probability BCE path as `BCEWithLogitsLoss(torch.logit(safe_probs), safe_labels)` instead of direct probability-space `BCELoss`.",
        "Required rerun rule: sync the updated `dialogue_kt/training.py` to SSH and rerun the same strict full-train scope; do not replace it with checkpoint-only or post-hoc calibration.",
    ],
}

ROUND3_FAILURE_FOLLOWUPS_CN = {
    "cel_task_conditioned_lastlayer_v21_joint_bias_fulltrain_qwen3_1.7b": [
        "当前本地修复：`dialogue_kt/training.py` 已把概率 BCE 路径统一改为先 sanitize，再用 `BCEWithLogitsLoss(torch.logit(safe_probs), safe_labels)` 计算，并补充 label sanitize，避免 joint calibrator 路径再次把非法值直接送入 CUDA `BCELoss`。",
        "下一次 `v21` 重跑要求：先同步更新后的 `dialogue_kt/training.py` 到 SSH，再保持原来的 strict full-train 配置完整重跑；不能改成 checkpoint-only / post-hoc 校准。",
    ],
}

ROUND3_SHARED_DESCRIPTION_CN = [
    "",
    "设计意图：",
    "",
    "- 满足“完整方法训练完成后再评测”的严格口径",
    "- 不再使用冻结补训、固定 bias 评估或 validation-fit 后处理作为最终分数",
]


def formal_candidate(model_name: str) -> dict:
    return FORMAL_CANDIDATES[model_name]


def formal_candidate_from_launch_key(launch_key: str) -> dict:
    return FORMAL_CANDIDATES[FORMAL_CANDIDATE_LAUNCH_KEYS[launch_key]]


def formal_candidate_log_path(root_dir: Path, model_name: str) -> Path:
    return root_dir / FORMAL_CANDIDATES[model_name]["stdout_log_rel"]


def build_formal_log_path_map(root_dir: Path) -> dict[str, Path]:
    return {
        model_name: formal_candidate_log_path(root_dir, model_name)
        for model_name in FORMAL_CANDIDATE_ORDER
    }


def build_round3_description_cn() -> list[str]:
    lines = ["候选：", ""]
    for idx, model_name in enumerate(FORMAL_CANDIDATE_ORDER, start=1):
        meta = FORMAL_CANDIDATES[model_name]
        lines.extend(
            [
                f"{idx}. `{model_name}`",
                f"   - {meta['start_point_cn']}",
                f"   - {meta['method_cn']}",
                f"   - 唯一主改动：{meta['single_variable_cn']}",
                f"   - 目标假设：{meta['hypothesis_cn']}",
            ]
        )
        if meta.get("implementation_guard_cn"):
            lines.append(f"   - 实现口径核对：{meta['implementation_guard_cn']}")
        for signal_idx, signal in enumerate(meta["expected_signals_cn"], start=1):
            lines.append(f"   - 期望观察 {signal_idx}：{signal}")
        lines.append(f"   - 若未双超时的下一轮解释：{meta['if_fail_cn']}")
    lines.extend(ROUND3_SHARED_DESCRIPTION_CN)
    return lines


def build_formal_candidate_brief_cn(model_name: str) -> list[str]:
    meta = FORMAL_CANDIDATES[model_name]
    lines = [
        f"### `{meta['label']}` / `{model_name}`",
        "",
        f"- 起点：{meta['start_point_cn']}",
        f"- 方法：{meta['method_cn']}",
        f"- 唯一主改动：{meta['single_variable_cn']}",
        f"- 目标假设：{meta['hypothesis_cn']}",
    ]
    if meta.get("implementation_guard_cn"):
        lines.append(f"- 实现口径核对：{meta['implementation_guard_cn']}")
    lines.append("- 期望观察：")
    for signal in meta["expected_signals_cn"]:
        lines.append(f"  - {signal}")
    lines.append(f"- 若未双超时的下一轮解释：{meta['if_fail_cn']}")
    return lines


def build_formal_launch_script_lines() -> list[str]:
    lines: list[str] = []
    for idx, model_name in enumerate(FORMAL_CANDIDATE_ORDER, start=1):
        meta = FORMAL_CANDIDATES[model_name]
        lines.append(f"  {idx}. `scripts/cel_stage1_last_layer/{meta['run_script_name']}`")
    return lines


def _validate_registry() -> None:
    if set(FORMAL_CANDIDATE_ORDER) != set(FORMAL_CANDIDATES):
        raise ValueError("FORMAL_CANDIDATE_ORDER must match FORMAL_CANDIDATES keys exactly")

    launch_keys = [meta["launch_key"] for meta in FORMAL_CANDIDATES.values()]
    if len(launch_keys) != len(set(launch_keys)):
        raise ValueError("formal candidate launch keys must be unique")

    labels = [meta["label"] for meta in FORMAL_CANDIDATES.values()]
    if len(labels) != len(set(labels)):
        raise ValueError("formal candidate labels must be unique")


_validate_registry()
