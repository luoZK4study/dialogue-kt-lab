# Formal Candidate Config Diffs

- 目的：把每个 strict full-train 正式候选相对其参考脚本的 CLI 配置差异显式列出，并防止未来新增候选时出现未声明的配置漂移。
- 规则：`config_diff_keys` 必须完整覆盖候选脚本相对参考脚本的实际参数差异；若出现未声明差异或声明项与实际不符，preflight 应失败。

## `v21` / `cel_task_conditioned_lastlayer_v21_joint_bias_fulltrain_qwen3_1.7b`

- 参考脚本：`scripts/cel_stage1_last_layer/run_task_conditioned_v1.sh` (task_conditioned v1 完整训练脚本)
- 候选脚本：`scripts/cel_stage1_last_layer/run_task_conditioned_v21_joint_bias_fulltrain.sh`
- 语义主变量：把 bias calibrator 从后处理调节前移为训练期联合优化的一部分
- 差异解释：这些配置差异共同实现同一个语义变量：把原本只在诊断轮里出现的 bias calibrator 前移到完整训练流程中，并为该联训路径显式指定优化超参数。
- 声明差异键：`model_name, lr, wd, cel_output_calibration, cel_calibrator_init_bias`
- 实际差异键：`cel_calibrator_init_bias, cel_output_calibration, lr, model_name, wd`
- 配置校验：`ok`

| Arg | Reference | Candidate |
|---|---|---|
| `cel_calibrator_init_bias` | `<unset>` | `0.0` |
| `cel_output_calibration` | `<unset>` | `bias` |
| `lr` | `<unset>` | `0.0002` |
| `model_name` | `cel_task_conditioned_lastlayer_v1_qwen3_1.7b` | `cel_task_conditioned_lastlayer_v21_joint_bias_fulltrain_qwen3_1.7b` |
| `wd` | `<unset>` | `0.01` |

## `v22` / `cel_task_conditioned_lastlayer_v22_fullv1_joint_bias_tinylr_qwen3_1.7b`

- 参考脚本：`scripts/cel_stage1_last_layer/run_task_conditioned_v21_joint_bias_fulltrain.sh` (v21 联合 bias full-train 脚本)
- 候选脚本：`scripts/cel_stage1_last_layer/run_task_conditioned_v22_fullv1_joint_bias_tinylr.sh`
- 语义主变量：把主变量改成从完整 `v1` checkpoint 出发、只用极小学习率做联合微调
- 差异解释：这些配置差异共同实现同一个语义变量：把联训 bias 的起点从 baseline 完整训练切换为 `v1` warm-start，并把训练节奏降到 tiny-lr 微调。
- 声明差异键：`pt_model_name, cel_selector_init_model_name, model_name, epochs, lr, wd`
- 实际差异键：`cel_selector_init_model_name, epochs, lr, model_name, pt_model_name, wd`
- 配置校验：`ok`

| Arg | Reference | Candidate |
|---|---|---|
| `cel_selector_init_model_name` | `<unset>` | `cel_task_conditioned_lastlayer_v1_qwen3_1.7b` |
| `epochs` | `2` | `1` |
| `lr` | `0.0002` | `0.00001` |
| `model_name` | `cel_task_conditioned_lastlayer_v21_joint_bias_fulltrain_qwen3_1.7b` | `cel_task_conditioned_lastlayer_v22_fullv1_joint_bias_tinylr_qwen3_1.7b` |
| `pt_model_name` | `lmkt_qwen3_1.7b_recert_20260620` | `cel_task_conditioned_lastlayer_v1_qwen3_1.7b` |
| `wd` | `0.01` | `0.0` |

## `v23` / `cel_task_conditioned_lastlayer_v23_joint_bias_posinit_qwen3_1.7b`

- 参考脚本：`scripts/cel_stage1_last_layer/run_task_conditioned_v21_joint_bias_fulltrain.sh` (v21 联合 bias full-train 脚本)
- 候选脚本：`scripts/cel_stage1_last_layer/run_task_conditioned_v23_joint_bias_posinit.sh`
- 语义主变量：把 joint bias full-train 的 calibrator 初始化从零改成基于诊断证据的正偏置起点
- 差异解释：相对 v21 只改一个语义变量：保留 joint bias full-train 训练口径不变，只把 calibrator 的初始 bias 从 0.0 改成一个有诊断依据的正起点。
- 声明差异键：`model_name, cel_calibrator_init_bias`
- 实际差异键：`cel_calibrator_init_bias, model_name`
- 配置校验：`ok`

| Arg | Reference | Candidate |
|---|---|---|
| `cel_calibrator_init_bias` | `0.0` | `0.331` |
| `model_name` | `cel_task_conditioned_lastlayer_v21_joint_bias_fulltrain_qwen3_1.7b` | `cel_task_conditioned_lastlayer_v23_joint_bias_posinit_qwen3_1.7b` |

## `v24` / `cel_task_conditioned_lastlayer_v24_selectorinit_joint_bias_fulltrain_qwen3_1.7b`

- 参考脚本：`scripts/cel_stage1_last_layer/run_task_conditioned_v21_joint_bias_fulltrain.sh` (v21 联合 bias full-train 脚本)
- 候选脚本：`scripts/cel_stage1_last_layer/run_task_conditioned_v24_selectorinit_joint_bias_fulltrain.sh`
- 语义主变量：在 v21 的 joint bias full-train 基础上，仅把 selector 初始化改为来自历史 selector/ranking 锚点 v1
- 差异解释：相对 v21 只改一个语义变量：保持 baseline checkpoint 起步和 joint bias full-train 训练口径不变，只把 selector 初始化改成来自历史 selector/ranking 锚点 v1。
- 声明差异键：`model_name, cel_selector_init_model_name`
- 实际差异键：`cel_selector_init_model_name, model_name`
- 配置校验：`ok`
- 实现口径核对：代码口径核对：相对 v21 额外只加载 `cel_selector.pt` 到 `_cel_selector`，不会同时 warm-start calibrator，也不会把 backbone / LoRA 一起从 `v1` 继续加载。

| Arg | Reference | Candidate |
|---|---|---|
| `cel_selector_init_model_name` | `<unset>` | `cel_task_conditioned_lastlayer_v1_qwen3_1.7b` |
| `model_name` | `cel_task_conditioned_lastlayer_v21_joint_bias_fulltrain_qwen3_1.7b` | `cel_task_conditioned_lastlayer_v24_selectorinit_joint_bias_fulltrain_qwen3_1.7b` |

## `v25` / `cel_task_conditioned_lastlayer_v25_joint_bias_calibratorlr10x_qwen3_1.7b`

- 参考脚本：`scripts/cel_stage1_last_layer/run_task_conditioned_v21_joint_bias_fulltrain.sh` (v21 联合 bias full-train 脚本)
- 候选脚本：`scripts/cel_stage1_last_layer/run_task_conditioned_v25_joint_bias_calibratorlr10x.sh`
- 语义主变量：在 v21 的 joint bias full-train 基础上，仅给 calibrator 单独更高的学习率
- 差异解释：相对 v21 只改一个语义变量：保持 baseline checkpoint 起步和 joint bias full-train 训练口径不变，只把 calibrator 从共享学习率切到更高的独立学习率。
- 声明差异键：`model_name, cel_calibrator_lr`
- 实际差异键：`cel_calibrator_lr, model_name`
- 配置校验：`ok`

| Arg | Reference | Candidate |
|---|---|---|
| `cel_calibrator_lr` | `<unset>` | `0.002` |
| `model_name` | `cel_task_conditioned_lastlayer_v21_joint_bias_fulltrain_qwen3_1.7b` | `cel_task_conditioned_lastlayer_v25_joint_bias_calibratorlr10x_qwen3_1.7b` |

## `v26` / `cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b`

- 多阶段总控：`scripts/cel_stage1_last_layer/run_task_conditioned_v26_selftrained.sh`
- 语义主变量：移除 baseline/v1 checkpoint 初始化，每个阶段只加载同一 A 模块 candidate 上一阶段生成的 checkpoint
- 差异解释：这些配置共同保证当前 A 模块 candidate 不读取任何历史实验权重，并通过 bootstrap -> warmup -> joint 的同 candidate checkpoint 链完成训练；正式测试只读取 joint 阶段产物。
- 阶段顺序：`bootstrap -> calibrator_warmup -> joint`
- 计划总 epoch：`4`；实际阶段合计：`4`
- 来源与配置校验：`ok`

| Stage | Script | Model | Epochs | LR | Input LoRA | Input Selector | Input Calibrator | Test |
|---|---|---|---:|---:|---|---|---|---|
| `bootstrap` | `scripts/cel_stage1_last_layer/run_task_conditioned_v26_bootstrap.sh` | `cel_task_conditioned_lastlayer_v26_bootstrap_qwen3_1.7b` | `2` | `0.0002` | `<unset>` | `<unset>` | `<unset>` | `skip` |
| `calibrator_warmup` | `scripts/cel_stage1_last_layer/run_task_conditioned_v26_calibrator_warmup.sh` | `cel_task_conditioned_lastlayer_v26_calibrator_warmup_qwen3_1.7b` | `1` | `0.0002` | `cel_task_conditioned_lastlayer_v26_bootstrap_qwen3_1.7b` | `cel_task_conditioned_lastlayer_v26_bootstrap_qwen3_1.7b` | `<unset>` | `skip` |
| `joint` | `scripts/cel_stage1_last_layer/run_task_conditioned_v26_joint.sh` | `cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b` | `1` | `0.00001` | `cel_task_conditioned_lastlayer_v26_calibrator_warmup_qwen3_1.7b` | `cel_task_conditioned_lastlayer_v26_calibrator_warmup_qwen3_1.7b` | `cel_task_conditioned_lastlayer_v26_calibrator_warmup_qwen3_1.7b` | `formal` |

- 禁止初始化来源：`lmkt_qwen3_1.7b_recert_20260620`, `cel_task_conditioned_lastlayer_v1_qwen3_1.7b`, `cel_task_conditioned_lastlayer_v26_fullv1_biaswarmup_joint_tinylr_qwen3_1.7b`
- 只有最终 joint 阶段执行正式 test；中间阶段只产生本次 v26 链自己的 checkpoint。

