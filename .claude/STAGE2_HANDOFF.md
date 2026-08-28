# A+B 模块工作交接

本文件用于接管当前 A+B 研究工作。旧的 Stage 2 分阶段链和历史结果已经清理，不应继续作为默认训练入口；它们只能从 Git 历史中追溯。

## 目标

在 Dialogue-KT 的 Qwen3-1.7B + LoRA 基础上联合训练 A、B：

```text
H -> A -> a -> h_r, h_n -> B(h_n) = h_nb -> h_m
  -> shared P(h_r), shared P(h_m)
```

```text
h_r = ((a + 1) / 2) ⊙ H
h_n = ((1 - a) / 2) ⊙ H
h_m = h_r + β · h_nb
L_total = λ_r BCE(p_r, y) + λ_m BCE(p_m, y)
        + λ_cons JS([p_r, 1-p_r], [p_m, 1-p_m])
```

同时评估 `P(h_n)` 的 reversal performance，并记录 `a`、互补误差、B 输出尺度、梯度、参数更新和 prediction-token 因果影响。

## 默认训练协议

A、B、LoRA、可选 calibrator 以及未来 C/D 从 epoch 1 起进入同一计算图、同一完整损失和同一 optimizer。每个 epoch 训练和验证，保存 validation-best 整体 checkpoint，依据固定 `patience`/`min_delta` 早停，恢复最佳状态后再进行 test 和 audit。parameter-group learning rate 可以不同，但不能把训练拆成独立阶段。任何 warmup、冻结或交替协议都必须显式声明为特殊协议。

## 重训顺序

- A 模块：raw Qwen + LoRA + A（可选恒等 calibrator），输出写入 `results/a/`。
- A+B 模块：raw Qwen + LoRA + A + B（可选恒等 calibrator），输出写入 `results/a_b/`。
- 两个 candidate 彼此独立，不加载 baseline、旧 A anchor 或其他 candidate 的参数。

实现入口见 `dialogue_kt/cel_methods.py`、`dialogue_kt/training.py` 和 `scripts/cel/`；开始正式训练前先运行 tensor-contract smoke test，并在 SSH alias `3090` 上确认环境和 GPU 状态。
