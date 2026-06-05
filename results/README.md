# Results 目录结构

## selfelicit/ — SELFELICIT + Dialogue-KT 实验
- `metrics/` — 模型评测指标 (.txt)
- `kcs/` — KC级别预测详情 (.json)
- `qual/` — 逐轮次预测结果 (.csv)
- `SELFELICIT_实验记录.md` — 完整实验记录

## archive/ — 历史实验
- `0.5B_baseline/` — Qwen2.5-0.5B 基线实验

## 新增论文方法时
创建新的子目录 `results/<paper_name>/`，按相同结构组织：
```
results/
├── selfelicit/          # 当前SELFELICIT实验
│   ├── metrics/
│   ├── kcs/
│   ├── qual/
│   └── 实验记录.md
└── <new_paper>/         # 未来新论文实验
    ├── metrics/
    ├── kcs/
    ├── qual/
    └── 实验记录.md
```
