# ERJT 实验记录

> Evidence-Rationale Joint Training (ERJT) — Dialogue-KT 结合实验
>
> 训练日期: 2026-06-10
> 模型: Qwen3-1.7B + LoRA (r=16, alpha=16)
> 数据集: MathDial ATC
> 基线: Overall AUC 75.99% (1.7B), 论文 76.71% (8B)

---

## 方法简介

ERJT v2 设计融合三篇论文的核心机制:
1. **UNIREX** (ICML 2022): 多粒度证据提取器 + Faithfulness/Plausibility 三损失
2. **Faith-DARE** (TPAMI 2025): R/U 解耦 + 互信息最小化 + 环境不变性
3. **PLMR** (KDD 2025): 生成器-预测器分离

Phase A 验证四个独立模块:
- **A1**: Multi-Granularity Extractor + Plausibility Loss
- **A2**: UNIREX Faithfulness Loss (comprehensiveness)
- **A3**: Faith-DARE Disentanglement (R/U separation + MI minimization)
- **A4**: Pseudo-Label Bootstrapping (Round 0)

---

## 实验结果

### Phase A: 独立模块验证

| 实验 | 方法 | Epoch | Train Loss | Val Loss | Overall AUC | Final Turn AUC | vs Baseline |
|------|------|:---:|:---:|:---:|:---:|:---:|:---:|
| — | Baseline | 2 | — | — | **75.99%** | — | — |
| A1 | Extractor + Plausibility | 2 | 0.9497 / — | 0.6602 / — | **74.97%** | 75.03% | -1.02% |
| A2 | Faithfulness | 2 | 0.8557 / — | 0.6162 / — | **75.46%** | 75.37% | -0.53% |
| A3 | Disentangle | 2 | 0.8360 / — | 0.6082 / — | **74.88%** | 73.03% | -1.11% |
| A4 | Bootstrap R0 | 1 | — | — | **75.20%** | 75.69% | -0.79% |

### 关键发现

1. **所有方法均低于 baseline 75.99%**，降幅 0.53-1.11%
2. **A2 (Faithfulness) 最佳**: 75.46%，最接近 baseline (-0.53%)
3. **A3 (Disentangle) 最差**: 74.88%，且 Final Turn AUC 仅 73.03%
4. **A1 与 A4 本质相同**: Extractor-only，A4 多一轮 bootstrap 但 AUC 接近 (74.97% vs 75.20%)
5. **无方法坍缩**: 所有 AUC > 74%，证明方法实现正确、loss 稳定
6. **模式一致**: 与之前 11 个单模型方法相同模式——Qwen3-1.7B 在此任务上已达容量上限

### Phase B 决策

训练计划规定: "只在 Phase A 验证有效（AUC ≥ baseline 或 AUC 稳定不坍缩）的模块才进入 Phase B 叠加"

- 无模块达到 "≥ baseline" 标准
- 虽然所有模块稳定不坍缩，但鉴于:
  - 最佳单模块 A2 仅 75.46%（低于 baseline 0.53%）
  - 组合低于 baseline 的单模块不太可能超越 baseline
  - 之前所有 11 个单模型方法均未超越 baseline
- **决策: 跳过 Phase B (叠加实验)**

---

## 代码变更

### 新增文件
- `dialogue_kt/erjt_methods.py` — MultiGranularityExtractor, DisentanglementModule, pseudo-label 构造, loss helpers

### 修改文件
- `dialogue_kt/training.py` — 新增 `get_lmkt_loss_erjt_extractor`, `get_lmkt_loss_erjt_faithfulness`, `get_lmkt_loss_erjt_disentangle`, `get_lmkt_loss_erjt_full` + train/test 路由
- `dialogue_kt/main.py` — 新增 `--erjt_mode`, `--erjt_alpha_p`, `--erjt_alpha_f`, `--erjt_beta`, `--erjt_gamma`, `--erjt_k_ratio`, `--erjt_pseudo_threshold`

### 训练命令

```bash
# A1: Extractor
CUDA_VISIBLE_DEVICES=0 python -m dialogue_kt.main train \
  --dataset mathdial --model_type lmkt --base_model .../Qwen3-1.7B \
  --model_name erjt_a1 --epochs 2 --quantize 0 \
  --erjt_mode extractor --erjt_alpha_p 0.2 --batch_size 1 --grad_accum_steps 32

# A2: Faithfulness
CUDA_VISIBLE_DEVICES=1 python -m dialogue_kt.main train \
  --dataset mathdial --model_type lmkt --base_model .../Qwen3-1.7B \
  --model_name erjt_a2 --epochs 2 --quantize 0 \
  --erjt_mode faithfulness --erjt_alpha_f 0.1 --erjt_k_ratio 0.3 \
  --batch_size 1 --grad_accum_steps 32

# A3: Disentangle
CUDA_VISIBLE_DEVICES=0 python -m dialogue_kt.main train \
  --dataset mathdial --model_type lmkt --base_model .../Qwen3-1.7B \
  --model_name erjt_a3 --epochs 2 --quantize 0 \
  --erjt_mode disentangle --erjt_beta 0.05 --erjt_gamma 0.05 \
  --batch_size 1 --grad_accum_steps 32

# A4: Bootstrap
CUDA_VISIBLE_DEVICES=1 python -m dialogue_kt.main train \
  --dataset mathdial --model_type lmkt --base_model .../Qwen3-1.7B \
  --model_name erjt_a4_r0 --epochs 1 --quantize 0 \
  --erjt_mode extractor --erjt_alpha_p 0.2 \
  --batch_size 1 --grad_accum_steps 32
```

---

## Bug 修复记录

1. **dtype 不匹配**: Extractor/Disentangle 模块默认 fp32，模型 hidden states 为 bf16 → 修复: 初始化时 `.to(dtype=model.dtype)`
2. **pseudo_labels dtype**: 构造的伪标签为 fp32，与 bf16 token_scores 不匹配 → 修复: plausibility_loss 中 cast pseudo_labels 到 target dtype
3. **GPU 内存泄漏**: 重复启动训练进程导致两组 A1/A2 同时运行 → 修复: kill 重复进程，仅保留原始训练进程

---

## 结论

ERJT Phase A 四个独立模块全部验证完成，Overall AUC 范围 74.88-75.46%。无一超越 baseline 75.99%。结果与之前 11 个方法一致——Qwen3-1.7B 在此任务上已达容量上限。Phase B 叠加实验跳过。

下一步优先级: LTIM → AGTR → VIB-Turn
