# CEL Stage1 Last-Layer Detailed Analysis

> Terminology note: historical model identifiers remain unchanged for prediction-file alignment. The audited method is called the **A module** in current method documents.

- Generated at: `2026-08-12T23:35:53`

## Baseline Probability Profile

- Model: `lmkt_qwen3_1.7b_recert_20260620`
- Overall Acc / AUC: **69.22 / 74.95**
- Final Acc / AUC: **61.17 / 75.32**
- Overall: AUC **74.95**, pred-True **40.8%**, pos/neg mean **0.5767/0.3561**, gap **0.2206**, std **0.2526**
- Final: AUC **75.32**, pred-True **44.3%**, pos/neg mean **0.5300/0.3239**, gap **0.2062**
- Non-final AUC: **76.89**

## Current Formal State

- Winner found: `True`
- Current strict formal winner: `cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b` with Overall Acc **70.03** and Overall AUC **75.38**.
- Strict formal winner gate: `Overall Acc > 69.22` and `Overall AUC > 74.95`.

## CEL vs Baseline

| Model | Type | Overall Acc | Overall AUC | Delta Overall Acc | Delta Overall AUC | Formal Gate | Non-final AUC | Final Acc | Delta Final Acc | Final AUC | Delta Final AUC | Pred True | Gap | Final Gap | Better/Worse | Notes |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b` | formal | 70.03 | 75.38 | +0.81 | +0.43 | yes | 76.05 | 66.80 | +5.63 | 77.26 | +1.94 | 44.0% | 0.3027 | 0.3179 | 1269/716/0 | helps final, hurts non-final; near-identity diagnostics |
| `cel_task_conditioned_lastlayer_v1_qwen3_1.7b` | formal | 68.87 | 75.29 | -0.35 | +0.34 | no | 77.03 | 56.31 | -4.86 | 75.27 | -0.05 | 33.8% | 0.2573 | 0.2349 | 1213/743/29 | near-identity diagnostics |
| `cel_task_conditioned_lastlayer_v21_joint_bias_fulltrain_qwen3_1.7b` | formal | 68.61 | 75.26 | -0.61 | +0.31 | no | 76.81 | 56.70 | -4.47 | 74.60 | -0.72 | 32.6% | 0.2559 | 0.2325 | 1202/752/31 | under-predicts True; near-identity diagnostics |
| `cel_adapter_lastlayer_v1_qwen3_1.7b` | formal | 68.21 | 75.04 | -1.01 | +0.09 | no | 77.54 | 65.63 | +4.46 | 72.66 | -2.66 | 52.5% | 0.2270 | 0.1985 | 955/969/61 | over-predicts True; near-identity diagnostics |
| `cel_task_conditioned_lastlayer_v25_joint_bias_calibratorlr10x_qwen3_1.7b` | formal | 66.90 | 74.98 | -2.32 | +0.03 | no | 77.07 | 52.04 | -9.13 | 74.27 | -1.05 | 30.2% | 0.2424 | 0.2171 | 1177/806/2 | under-predicts True; near-identity diagnostics |
| `cel_task_conditioned_lastlayer_v22_fullv1_joint_bias_tinylr_qwen3_1.7b` | formal | 68.26 | 74.93 | -0.96 | -0.02 | no | 76.87 | 64.27 | +3.10 | 72.86 | -2.46 | 47.8% | 0.3248 | 0.2861 | 1310/668/7 | better mean separation, weaker ranking; more rows improve, ranking still below baseline; near-identity diagnostics |
| `cel_mlp_lastlayer_v1_qwen3_1.7b` | formal | 65.74 | 74.81 | -3.48 | -0.14 | no | 77.74 | 69.32 | +8.15 | 72.73 | -2.59 | 61.8% | 0.2434 | 0.2265 | 1061/893/31 | better mean separation, weaker ranking; over-predicts True; more rows improve, ranking still below baseline; near-identity diagnostics |
| `cel_task_conditioned_lastlayer_v24_selectorinit_joint_bias_fulltrain_qwen3_1.7b` | formal | 68.16 | 74.44 | -1.06 | -0.51 | no | 76.29 | 56.89 | -4.28 | 73.49 | -1.83 | 36.2% | 0.2400 | 0.2174 | 1103/859/23 | better mean separation, weaker ranking; more rows improve, ranking still below baseline; near-identity diagnostics |
| `cel_task_conditioned_lastlayer_v23_joint_bias_posinit_qwen3_1.7b` | formal | 68.41 | 74.29 | -0.81 | -0.66 | no | 75.59 | 56.12 | -5.05 | 74.59 | -0.73 | 33.5% | 0.2398 | 0.2213 | 1171/814/0 | better mean separation, weaker ranking; more rows improve, ranking still below baseline; near-identity diagnostics |
| `cel_task_conditioned_lastlayer_v15_fullv1_cal_bias_only_qwen3_1.7b` | diagnostic | 69.27 | 75.29 | +0.05 | +0.34 | diagnostic | 77.03 | 61.94 | +0.77 | 75.27 | -0.05 | 40.6% | 0.2606 | 0.2467 | 1242/740/3 | near-identity diagnostics |
| `cel_task_conditioned_lastlayer_v11_cal_bias_only_qwen3_1.7b` | diagnostic | 65.04 | 73.83 | -4.18 | -1.12 | diagnostic | 74.91 | 49.90 | -11.27 | 73.84 | -1.48 | 23.5% | 0.1922 | 0.1915 | 1051/934/0 | under-predicts True; more rows improve, ranking still below baseline; near-identity diagnostics; strong positive-class suppression |
| `cel_task_conditioned_lastlayer_v12_cal_affine_only_qwen3_1.7b` | diagnostic | 65.49 | 73.83 | -3.73 | -1.12 | diagnostic | 74.91 | 50.49 | -10.68 | 73.84 | -1.48 | 24.6% | 0.1507 | 0.1534 | 958/1023/4 | under-predicts True; near-identity diagnostics; strong positive-class suppression |

## Current Read

- Current strict formal winner in synced artifacts: `cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b` with Overall Acc / AUC **70.03 / 75.38**.
- Best diagnostic-only result so far: `cel_task_conditioned_lastlayer_v15_fullv1_cal_bias_only_qwen3_1.7b` at Overall Acc / AUC **69.27 / 75.29**.
- That diagnostic result remains excluded from formal winner judgment because it is not a complete strict full-train method result.
- Largest overall probability gap so far: `cel_task_conditioned_lastlayer_v22_fullv1_joint_bias_tinylr_qwen3_1.7b` at **0.3248**.
- The model with the widest positive/negative mean separation is not currently the best-AUC model, so ranking quality and class bias still matter more than average probability spread alone.
