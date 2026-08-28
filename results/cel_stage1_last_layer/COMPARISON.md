# CEL Stage1 Last-Layer Comparison

> Terminology note: the model identifier below is a historical artifact name. The current method name is **A module**.

- Current stage: `task_conditioned_tuning_completed_with_winner`

## Baseline

- Model: `lmkt_qwen3_1.7b_recert_20260620`
- Loss: **0.5928**
- Overall Acc: **69.22**
- Overall AUC: **74.95**
- Final Acc: **61.17**
- Final AUC: **75.32**

## Round 3 Formal State

- Winner found: `True`
- Current strict formal winner: `cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b` with Overall Acc **70.03** and Overall AUC **75.38**.


## Formal Winner Gate

- A formal winner must satisfy both `Overall Acc > 69.22` and `Overall AUC > 74.95`.
- Current strict formal winner in synced artifacts: `cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b` with Overall Acc / AUC **70.03 / 75.38**.
- Best diagnostic-only result is `cel_task_conditioned_lastlayer_v15_fullv1_cal_bias_only_qwen3_1.7b` at Overall Acc / AUC **69.27 / 75.29**, but it remains excluded from formal winner judgment.

## Best Formal Result By Family

- `mlp`: `cel_mlp_lastlayer_v1_qwen3_1.7b` with Overall Acc / AUC **65.74 / 74.81**, delta Acc / AUC **-3.48 / -0.14**, does not clear strict gate
- `adapter`: `cel_adapter_lastlayer_v1_qwen3_1.7b` with Overall Acc / AUC **68.21 / 75.04**, delta Acc / AUC **-1.01 / +0.09**, does not clear strict gate
- `task_conditioned`: `cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b` with Overall Acc / AUC **70.03 / 75.38**, delta Acc / AUC **+0.81 / +0.43**, clears strict gate

## Best Diagnostic Result

- `cel_task_conditioned_lastlayer_v15_fullv1_cal_bias_only_qwen3_1.7b` with Overall Acc **69.27** and Overall AUC **75.29**.
- 该结果只用于说明后处理/冻结补训上限，不计入正式方法比较。

## All CEL Results

| Model | Family | Type | Loss | Overall Acc | Overall AUC | Final Acc | Final AUC | Delta Overall Acc | Delta Overall AUC | Delta Final Acc | Delta Final AUC | Formal Gate | Diagnostics |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b` | task_conditioned | formal | 0.6657 | 70.03 | 75.38 | 66.80 | 77.26 | +0.81 | +0.43 | +5.63 | +1.94 | yes | gate_mean: 0.0000, gate_abs_mean: 0.0000, gate_pos_frac: 0.0000, gate_neg_frac: 0.0000, gate_max: 0.0000, gate_min: 0.0000, injected_mean: 0.2813, cal_bias: 0.3220, post_prob_invalid_frac: 0.0000, post_prob_label_invalid_frac: 0.0000, post_prob_label_out_of_range_frac: 0.0000, post_prob_label_raw_max: 0.4736, post_prob_label_raw_min: 0.4736, post_prob_label_safe_max: 0.4736, post_prob_label_safe_min: 0.4736, post_prob_label_sanitized_frac: 0.0000, post_prob_out_of_range_frac: 0.0000, post_prob_raw_max: 0.4687, post_prob_raw_min: 0.4687, post_prob_safe_max: 0.4687, post_prob_safe_min: 0.4687, post_prob_sanitized_frac: 0.0000, pre_prob_invalid_frac: 0.0000, pre_prob_out_of_range_frac: 0.0000, pre_prob_raw_max: 0.4277, pre_prob_raw_min: 0.4277, pre_prob_safe_max: 0.4277, pre_prob_safe_min: 0.4277, pre_prob_sanitized_frac: 0.0000 |
| `cel_task_conditioned_lastlayer_v1_qwen3_1.7b` | task_conditioned | formal | 0.6300 | 68.87 | 75.29 | 56.31 | 75.27 | -0.35 | +0.34 | -4.86 | -0.05 | no | gate_mean: 0.0000, gate_abs_mean: 0.0000, gate_pos_frac: 0.0000, gate_neg_frac: 0.0000, gate_max: 0.0000, gate_min: 0.0000, injected_mean: 1.1436 |
| `cel_task_conditioned_lastlayer_v21_joint_bias_fulltrain_qwen3_1.7b` | task_conditioned | formal | 0.6340 | 68.61 | 75.26 | 56.70 | 74.60 | -0.61 | +0.31 | -4.47 | -0.72 | no | gate_mean: 0.0000, gate_abs_mean: 0.0000, gate_pos_frac: 0.0000, gate_neg_frac: 0.0000, gate_max: 0.0000, gate_min: 0.0000, injected_mean: 1.1551, cal_bias: -0.0001, post_prob_invalid_frac: 0.0000, post_prob_label_invalid_frac: 0.0000, post_prob_label_out_of_range_frac: 0.0000, post_prob_label_raw_max: 0.4736, post_prob_label_raw_min: 0.4736, post_prob_label_safe_max: 0.4736, post_prob_label_safe_min: 0.4736, post_prob_label_sanitized_frac: 0.0000, post_prob_out_of_range_frac: 0.0000, post_prob_raw_max: 0.3806, post_prob_raw_min: 0.3806, post_prob_safe_max: 0.3806, post_prob_safe_min: 0.3806, post_prob_sanitized_frac: 0.0000, pre_prob_invalid_frac: 0.0000, pre_prob_out_of_range_frac: 0.0000, pre_prob_raw_max: 0.3806, pre_prob_raw_min: 0.3806, pre_prob_safe_max: 0.3806, pre_prob_safe_min: 0.3806, pre_prob_sanitized_frac: 0.0000 |
| `cel_adapter_lastlayer_v1_qwen3_1.7b` | adapter | formal | 0.6149 | 68.21 | 75.04 | 65.63 | 72.66 | -1.01 | +0.09 | +4.46 | -2.66 | no | gate_mean: 0.0000, gate_abs_mean: 0.0000, gate_pos_frac: 0.0000, gate_neg_frac: 0.0000, gate_max: 0.0000, gate_min: 0.0000, injected_mean: 1.0821 |
| `cel_task_conditioned_lastlayer_v25_joint_bias_calibratorlr10x_qwen3_1.7b` | task_conditioned | formal | 0.6363 | 66.90 | 74.98 | 52.04 | 74.27 | -2.32 | +0.03 | -9.13 | -1.05 | no | gate_mean: 0.0000, gate_abs_mean: 0.0000, gate_pos_frac: 0.0000, gate_neg_frac: 0.0000, gate_max: 0.0000, gate_min: 0.0000, injected_mean: 1.1560, cal_bias: -0.0011, post_prob_invalid_frac: 0.0000, post_prob_label_invalid_frac: 0.0000, post_prob_label_out_of_range_frac: 0.0000, post_prob_label_raw_max: 0.4736, post_prob_label_raw_min: 0.4736, post_prob_label_safe_max: 0.4736, post_prob_label_safe_min: 0.4736, post_prob_label_sanitized_frac: 0.0000, post_prob_out_of_range_frac: 0.0000, post_prob_raw_max: 0.3700, post_prob_raw_min: 0.3700, post_prob_safe_max: 0.3700, post_prob_safe_min: 0.3700, post_prob_sanitized_frac: 0.0000, pre_prob_invalid_frac: 0.0000, pre_prob_out_of_range_frac: 0.0000, pre_prob_raw_max: 0.3701, pre_prob_raw_min: 0.3701, pre_prob_safe_max: 0.3701, pre_prob_safe_min: 0.3701, pre_prob_sanitized_frac: 0.0000 |
| `cel_task_conditioned_lastlayer_v22_fullv1_joint_bias_tinylr_qwen3_1.7b` | task_conditioned | formal | 0.8027 | 68.26 | 74.93 | 64.27 | 72.86 | -0.96 | -0.02 | +3.10 | -2.46 | no | gate_mean: 0.0000, gate_abs_mean: 0.0000, gate_pos_frac: 0.0000, gate_neg_frac: 0.0000, gate_max: 0.0000, gate_min: 0.0000, injected_mean: 1.1202, cal_bias: 0.0001, post_prob_invalid_frac: 0.0000, post_prob_out_of_range_frac: 0.0000, post_prob_raw_max: 0.4834, post_prob_raw_min: 0.4834, post_prob_safe_max: 0.4834, post_prob_safe_min: 0.4834, post_prob_sanitized_frac: 0.0272, pre_prob_invalid_frac: 0.0000, pre_prob_out_of_range_frac: 0.0000, pre_prob_raw_max: 0.4833, pre_prob_raw_min: 0.4833, pre_prob_safe_max: 0.4833, pre_prob_safe_min: 0.4833, pre_prob_sanitized_frac: 0.0272 |
| `cel_mlp_lastlayer_v1_qwen3_1.7b` | mlp | formal | 0.6906 | 65.74 | 74.81 | 69.32 | 72.73 | -3.48 | -0.14 | +8.15 | -2.59 | no | gate_mean: 0.0000, gate_abs_mean: 0.0000, gate_pos_frac: 0.0000, gate_neg_frac: 0.0000, gate_max: 0.0000, gate_min: 0.0000, injected_mean: 1.0770 |
| `cel_task_conditioned_lastlayer_v24_selectorinit_joint_bias_fulltrain_qwen3_1.7b` | task_conditioned | formal | 0.6326 | 68.16 | 74.44 | 56.89 | 73.49 | -1.06 | -0.51 | -4.28 | -1.83 | no | gate_mean: 0.0000, gate_abs_mean: 0.0000, gate_pos_frac: 0.0000, gate_neg_frac: 0.0000, gate_max: 0.0000, gate_min: 0.0000, injected_mean: 1.1219, cal_bias: -0.0002, post_prob_invalid_frac: 0.0000, post_prob_label_invalid_frac: 0.0000, post_prob_label_out_of_range_frac: 0.0000, post_prob_label_raw_max: 0.4736, post_prob_label_raw_min: 0.4736, post_prob_label_safe_max: 0.4736, post_prob_label_safe_min: 0.4736, post_prob_label_sanitized_frac: 0.0000, post_prob_out_of_range_frac: 0.0000, post_prob_raw_max: 0.3952, post_prob_raw_min: 0.3952, post_prob_safe_max: 0.3952, post_prob_safe_min: 0.3952, post_prob_sanitized_frac: 0.0000, pre_prob_invalid_frac: 0.0000, pre_prob_out_of_range_frac: 0.0000, pre_prob_raw_max: 0.3953, pre_prob_raw_min: 0.3953, pre_prob_safe_max: 0.3953, pre_prob_safe_min: 0.3953, pre_prob_sanitized_frac: 0.0000 |
| `cel_task_conditioned_lastlayer_v23_joint_bias_posinit_qwen3_1.7b` | task_conditioned | formal | 0.6301 | 68.41 | 74.29 | 56.12 | 74.59 | -0.81 | -0.66 | -5.05 | -0.73 | no | gate_mean: 0.0000, gate_abs_mean: 0.0000, gate_pos_frac: 0.0000, gate_neg_frac: 0.0000, gate_max: 0.0000, gate_min: 0.0000, injected_mean: 1.1407, cal_bias: 0.3302, post_prob_invalid_frac: 0.0000, post_prob_label_invalid_frac: 0.0000, post_prob_label_out_of_range_frac: 0.0000, post_prob_label_raw_max: 0.4736, post_prob_label_raw_min: 0.4736, post_prob_label_safe_max: 0.4736, post_prob_label_safe_min: 0.4736, post_prob_label_sanitized_frac: 0.0000, post_prob_out_of_range_frac: 0.0000, post_prob_raw_max: 0.3921, post_prob_raw_min: 0.3921, post_prob_safe_max: 0.3921, post_prob_safe_min: 0.3921, post_prob_sanitized_frac: 0.0000, pre_prob_invalid_frac: 0.0000, pre_prob_out_of_range_frac: 0.0000, pre_prob_raw_max: 0.3413, pre_prob_raw_min: 0.3413, pre_prob_safe_max: 0.3413, pre_prob_safe_min: 0.3413, pre_prob_sanitized_frac: 0.0000 |
| `cel_task_conditioned_lastlayer_v15_fullv1_cal_bias_only_qwen3_1.7b` | task_conditioned | diagnostic | 0.6078 | 69.27 | 75.29 | 61.94 | 75.27 | +0.05 | +0.34 | +0.77 | -0.05 | diagnostic | gate_mean: 0.0000, gate_abs_mean: 0.0000, gate_pos_frac: 0.0000, gate_neg_frac: 0.0000, gate_max: 0.0000, gate_min: 0.0000, injected_mean: 1.1436, cal_bias: 0.3802 |
| `cel_task_conditioned_lastlayer_v12_cal_affine_only_qwen3_1.7b` | task_conditioned | diagnostic | 0.6294 | 65.49 | 73.83 | 50.49 | 73.84 | -3.73 | -1.12 | -10.68 | -1.48 | diagnostic | gate_mean: 0.0000, gate_abs_mean: 0.0000, gate_pos_frac: 0.0000, gate_neg_frac: 0.0000, gate_max: 0.0000, gate_min: 0.0000, injected_mean: 1.0836, cal_bias: 0.0890, cal_scale: 0.6794 |
| `cel_task_conditioned_lastlayer_v11_cal_bias_only_qwen3_1.7b` | task_conditioned | diagnostic | 0.6733 | 65.04 | 73.83 | 49.90 | 73.84 | -4.18 | -1.12 | -11.27 | -1.48 | diagnostic | gate_mean: 0.0000, gate_abs_mean: 0.0000, gate_pos_frac: 0.0000, gate_neg_frac: 0.0000, gate_max: 0.0000, gate_min: 0.0000, injected_mean: 1.0836, cal_bias: 0.0945 |
