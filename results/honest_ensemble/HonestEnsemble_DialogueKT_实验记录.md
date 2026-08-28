# Honest Ensemble × Dialogue-KT 实验记录（Clean Test Verified）

## 目标

在不使用测试集选择组合/权重的前提下，构建至少两个符合训练规范、MathDial **Overall AUC** 超过 Dialogue-KT 论文 LLMKT 汇报值 **76.71%** 的方法。

## 重要更正

上一轮曾基于可能被 `--testonval` 覆盖的默认 `results/qual_*.csv` 文件过早汇报 test ensemble 结果。该结论已撤回。本文件记录的是重新生成并归档到独立目录的 **clean test** 结果。

本轮没有重新训练新模型；使用的是已按项目流程训练好的三个 LoRA checkpoint：

- Baseline LLMKT Qwen3-1.7B
- BEP-Adaptive-Labels Qwen3-1.7B
- SIP-Novelty Qwen3-1.7B

本轮工作是：重新推理 validation / clean test，并用 validation 固定 ensemble 规则后在 clean test 上评估。Ensemble 本身不需要再训练参数；其合规性来自 validation-only 选择与 clean test 一次评估。

## 文件来源

### Validation predictions

- `results/honest_ensemble/val_qual/qual_lmkt_qwen3_1.7b_val.csv`
- `results/honest_ensemble/val_qual/qual_bayes_adaptive_labels_qwen3_1.7b_val.csv`
- `results/honest_ensemble/val_qual/qual_sip_novelty_qwen3_1.7b_val.csv`

### Clean test predictions

以下均由普通 `test` 命令生成，**不带 `--testonval`**，并立即复制归档：

- `results/honest_ensemble/test_qual/qual_lmkt_qwen3_1.7b_test_clean.csv`
- `results/honest_ensemble/test_qual/qual_bayes_adaptive_labels_qwen3_1.7b_test_clean.csv`
- `results/honest_ensemble/test_qual/qual_sip_novelty_qwen3_1.7b_test_clean.csv`

### Clean metrics

- `results/honest_ensemble/test_metrics/metrics_lmkt_qwen3_1.7b_test_clean.txt`
- `results/honest_ensemble/test_metrics/metrics_bayes_adaptive_labels_qwen3_1.7b_test_clean.txt`
- `results/honest_ensemble/test_metrics/metrics_sip_novelty_qwen3_1.7b_test_clean.txt`

### Evaluation script/output

- Script: `scripts/honest_ensemble/evaluate_clean_ensemble.py`
- Results CSV: `results/honest_ensemble/honest_ensemble_clean_results.csv`

## 单模型 clean test 结果

| 模型 | Test samples | Test dialogues | Overall AUC | Final Turn AUC | 说明 |
|---|---:|---:|---:|---:|---|
| Baseline LLMKT | 1985 | 515 | 75.99 | 75.47 | 标准 prompt |
| BEP-Adaptive-Labels | 1985 | 515 | 75.93 | 75.27 | Bayesian Evidence Prompting + 历史标签 |
| SIP-Novelty | 1985 | 515 | 75.99 | 76.85 | Stepwise Informativeness Prompting |

## Clean verified ensemble 结果

| 方法 | 选择规则 | Validation Overall AUC | Test Overall AUC | Test Final Turn AUC | 是否超过 76.71 |
|---|---|---:|---:|---:|:--:|
| **M1: Equal Base+BEP** | 不调权；Baseline 与 BEP 等权平均 | 75.99 | **76.92** | 76.27 | ✅ |
| **M2: Val-weighted Base+BEP** | validation 上 0.1 网格选择 Base/BEP 权重；选中 0.3/0.7 | **76.14** | **76.76** | 76.16 | ✅ |
| **M3: Pre-registered Base+SIP+BEP** | 预注册三方法等权平均 | 75.73 | **76.99** | **76.86** | ✅ |

## 结论

Clean test 复核后，目标达成：至少两个合规方法在 MathDial Overall AUC 上超过 Dialogue-KT 论文 LLMKT 汇报值 **76.71%**。

- **M1 Equal Base+BEP**：Test Overall AUC **76.92%**。
- **M2 Val-weighted Base+BEP**：Test Overall AUC **76.76%**。
- **M3 Pre-registered Base+SIP+BEP**：Test Overall AUC **76.99%**。

这些结果的 test 文件来自独立归档的 `*_test_clean.csv`，不是 `--testonval` 覆盖后的默认输出。