# 当前研究状态

## 结论

- 当前主线为 MathDial ATC、Qwen3-1.7B + LoRA 的 baseline + CEL A/B。
- A、B 的代码和目标双路径设计保留；旧分阶段训练方式不再作为默认协议。
- 旧训练记录、旧 candidate 结果和无关方法结果已清理；A 模块和 A+B 模块需要从 raw Qwen 按统一协议重训。
- 论文目标为 Overall AUC `76.71`。baseline 认证值为 Overall Acc/AUC `69.22 / 74.95`，Final Acc/AUC `61.17 / 75.32`。
- A 模块历史参考值为 Overall Acc/AUC `70.03 / 75.38`，Final Acc/AUC `66.80 / 77.26`。历史 checkpoint 名称只用于 provenance。

## 方法定义

```text
a = A(H, task context)
h_r = ((a + 1) / 2) ⊙ H
h_n = ((1 - a) / 2) ⊙ H
h_nb = B(h_n)
h_m = h_r + β · h_nb
p_r = P(h_r)
p_m = P(h_m)
```

context 之外的有效 prompt token 在 `h_r/h_m` 中保持原始 `H`，在 `h_n` 中为零。两条路径共享后续网络 `P`。训练使用 `λ_r BCE + λ_m BCE + λ_cons JS`，并报告 `P(h_n)` reversal performance。

## 统一训练口径

A、B、LoRA、可选 calibrator 及未来 C/D 模块从 epoch 1 起作为一个整体训练：同一 forward、同一完整损失、同一 optimizer、每 epoch train/validation、validation-best checkpoint 和 patience/min_delta 早停。模块可用不同 parameter-group learning rate，但不得默认拆成 bootstrap、warmup、冻结或独立训练。任何特殊协议必须显式记录并单独解释。

“epoch 2 读取 epoch 1 权重”在这里指整个模型状态自然延续，不是只加载某个模块的权重；恢复训练时也只能读取同一 candidate 的完整 checkpoint。

## 下一步

1. 在远端同步最新代码后，从 raw Qwen 独立重训 A 模块。
2. 从 raw Qwen 独立重训 A+B 模块，启用双路径目标和充分容量的 B。
3. 每个 candidate 完成 validation-best 恢复、final test、provenance、parameter、tensor-contract 和诊断审计后再比较。
4. 只有在机制闭环和结果证据充分后，才考虑 C 模块或额外正则。
