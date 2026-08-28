#!/bin/bash
# Reorganize dialogue-kt for modular paper-method integration
# Each paper gets its own namespace: selfelicit/, future papers follow same pattern

set -e
BASE=/home/user4/dialogue-kt
cd $BASE

echo "=== 1. Clean pycache ==="
rm -rf dialogue_kt/__pycache__ dialogue_kt/models/__pycache__

echo "=== 2. Organize papers/ ==="
mkdir -p papers
mv -f Liu_2025_SelfElicit.pdf papers/ 2>/dev/null || true
mv -f Exploring_Knowledge_Tracing_in_Tutor-Student_Dialogues_using_LLMs.pdf papers/dialogue-kt.pdf 2>/dev/null || true

echo "=== 3. Organize root scripts into scripts/selfelicit/ ==="
mkdir -p scripts/selfelicit
for f in ensemble_eval.py final_ensemble.py full_ensemble_eval.py optimize_ensemble.py grid6_ensemble.py proper_ensemble.py debug_attention.py test_token_span.py; do
    [ -f "$f" ] && mv "$f" scripts/selfelicit/ && echo "  moved $f"
done

echo "=== 4. Move precompute_evidence.py into scripts/selfelicit/ ==="
[ -f dialogue_kt/precompute_evidence.py ] && mv dialogue_kt/precompute_evidence.py scripts/selfelicit/precompute_evidence.py && echo "  moved precompute_evidence.py"

echo "=== 5. Organize results/ by paper ==="
cd results

# Create subdirectories
mkdir -p selfelicit/metrics selfelicit/kcs selfelicit/qual
mkdir -p archive/0.5B_baseline archive/qwen05b

# Move SELFELICIT experiment results
for prefix in lmkt_qwen3_1.7b lmkt_se_prompt_1.7b lmkt_se_combined_1.7b lmkt_se_adaptive_1.7b lmkt_se_conservative_1.7b lmkt_se_true_attn_1.7b lmkt_se_weighted_1.7b; do
    for ext in txt; do
        [ -f "metrics_${prefix}.${ext}" ] && mv "metrics_${prefix}.${ext}" selfelicit/metrics/ && echo "  moved metrics_${prefix}.${ext}"
    done
    [ -f "kcs_${prefix}.json" ] && mv "kcs_${prefix}.json" selfelicit/kcs/ && echo "  moved kcs_${prefix}.json"
    [ -f "qual_${prefix}.csv" ] && mv "qual_${prefix}.csv" selfelicit/qual/ && echo "  moved qual_${prefix}.csv"
done

# Move MD files
for f in SELFELICIT_DialogueKT_实验完整记录.md selfelicit_experiments.md; do
    [ -f "$f" ] && mv "$f" selfelicit/ && echo "  moved $f"
done

# Move old 0.5B and full baseline files to archive
for prefix in lmkt_mathdial_full lmkt_mathdial_qwen05b lmkt_mathdial_qwen05b_final; do
    [ -f "metrics_${prefix}.txt" ] && mv "metrics_${prefix}.txt" archive/0.5B_baseline/ && echo "  archived metrics_${prefix}.txt"
    [ -f "kcs_${prefix}.json" ] && mv "kcs_${prefix}.json" archive/0.5B_baseline/ && echo "  archived kcs_${prefix}.json"
    [ -f "qual_${prefix}.csv" ] && mv "qual_${prefix}.csv" archive/0.5B_baseline/ && echo "  archived qual_${prefix}.csv"
done

# Create README for results structure
cat > README.md << 'EOF'
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
EOF

cd $BASE

echo "=== 6. Organize saved_models/ by paper ==="
mkdir -p saved_models/selfelicit
# Move SE experiment models
for m in lmkt_se_prompt_1.7b lmkt_se_combined_1.7b lmkt_se_adaptive_1.7b lmkt_se_conservative_1.7b lmkt_se_true_attn_1.7b lmkt_se_weighted_1.7b; do
    [ -d "saved_models/$m" ] && mv "saved_models/$m" "saved_models/selfelicit/" && echo "  moved $m"
done

# Rename baseline for clarity
[ -d "saved_models/lmkt_qwen3_1.7b" ] && mv "saved_models/lmkt_qwen3_1.7b" "saved_models/baseline_qwen3_1.7b" && echo "  renamed baseline"

# Archive old models
mkdir -p saved_models/archive
for m in debug_1.7b debug_4b lmkt_mathdial_full lmkt_mathdial_qwen05b lmkt_mathdial_qwen05b_final lmkt_qwen3_8b_server; do
    [ -d "saved_models/$m" ] && mv "saved_models/$m" "saved_models/archive/" && echo "  archived $m"
done

echo "=== 7. Organize data/annotated/ ==="
cd data/annotated
mkdir -p evidence_cache
for f in mathdial_train_atc_evidence.json mathdial_val_atc_evidence.json mathdial_test_atc_evidence.json; do
    [ -f "$f" ] && mv "$f" evidence_cache/ && echo "  moved $f to evidence_cache/"
done
cd $BASE

echo "=== 8. Create README for project root ==="
cat > STRUCTURE.md << 'EOF'
# 项目结构说明

## 目录组织原则

按论文方法名进行命名空间隔离。每个新论文的方法代码、实验数据、模型独立管理。

```
dialogue-kt/
├── papers/                    # 论文PDF
│   ├── dialogue-kt.pdf
│   └── selfelicit.pdf
├── dialogue_kt/               # 核心代码包
│   ├── models/                # 模型定义
│   │   ├── lm.py              # LLM基座加载（通用）
│   │   └── selfelicit/        # SELFELICIT相关模块（按论文隔离）
│   │       ├── __init__.py
│   │       ├── core.py        #   注意力提取、证据识别
│   │       ├── prompting.py   #   SE提示模板
│   │       └── dataset.py     #   SE数据集类
│   ├── main.py                # CLI入口
│   ├── training.py            # 训练/评估核心
│   ├── prompting.py           # 基础提示模板（通用）
│   ├── kt_data_loading.py     # 基础数据集（通用）
│   └── utils.py               # 工具函数
├── scripts/                   # 分析脚本
│   └── selfelicit/            # SELFELICIT相关脚本
│       ├── ensemble.py
│       ├── precompute_evidence.py
│       └── ...
├── saved_models/              # 模型检查点
│   ├── baseline_qwen3_1.7b/   # 基线模型
│   ├── selfelicit/            # SE实验模型
│   └── archive/               # 旧模型存档
├── results/                   # 实验结果
│   ├── selfelicit/            # SE实验结果
│   │   ├── metrics/
│   │   ├── kcs/
│   │   ├── qual/
│   │   └── 实验记录.md
│   ├── archive/               # 历史实验
│   └── README.md              # 结果目录说明
├── data/annotated/            # 标注数据
│   └── evidence_cache/        # 证据预计算缓存
└── STRUCTURE.md               # 本文件
```

## 添加新论文方法的步骤

1. 在 `papers/` 放入论文PDF
2. 在 `dialogue_kt/models/<paper_name>/` 创建方法模块（`__init__.py`, `core.py`等）
3. 在 `scripts/<paper_name>/` 放入分析脚本
4. 在 `results/<paper_name>/` 创建结果目录结构
5. 在 `saved_models/<paper_name>/` 保存训练模型
6. 更新本文件记录

## 原则
- **模块化**: 每个方法的代码独立目录
- **可扩展**: 通用代码保持在顶层，方法特有代码隔离
- **可复现**: 结果、模型、脚本按方法分类保存
EOF

echo ""
echo "=== Done! Final structure ==="
find . -maxdepth 3 -type d | grep -v '.git\|__pycache__\|.claude' | sort
