# 项目结构说明

## 目录组织原则

按论文方法名进行命名空间隔离。每个新论文的方法代码、实验数据、模型独立管理。

```
dialogue-kt/
├── literature/                  # 论文PDF
│   ├── dialogue-kt.pdf          # 框架论文（直接放置）
│   ├── evidence_rationale_extraction/
│   ├── information_bottleneck/
│   └── representation_information_compression/
├── dialogue_kt/                 # 核心代码包
│   ├── selfelicit.py            # SELFELICIT模块（示例）
│   ├── main.py                  # CLI入口
│   ├── training.py              # 训练/评估核心
│   ├── prompting.py             # 提示模板（通用+SE）
│   ├── kt_data_loading.py       # 数据集（通用+SE）
│   ├── models/lm.py             # LLM基座加载
│   └── utils.py                 # 工具函数
├── scripts/<paper_name>/        # 分析脚本，按论文隔离
├── saved_models/<paper_name>/   # 模型检查点
├── results/<paper_name>/        # 实验结果
│   ├── metrics/                 # .txt 评测指标
│   ├── kcs/                     # .json KC预测
│   ├── qual/                    # .csv 逐轮预测
│   └── <PaperName>_实验记录.md   # 完整实验记录
├── data/annotated/              # 标注数据
│   └── evidence_cache/          # 证据预计算缓存
└── .claude/                     # 项目文档
    ├── CLAUDE.md                # 操作指南
    ├── MEMORY.md                # 总览+工作流+论文状态
    └── STRUCTURE.md             # 本文件
```

## 添加新论文方法的标准步骤

1. 在 `literature/` 对应分类目录放入论文PDF
2. 代码模块加入 `dialogue_kt/`（新增文件或修改现有文件通过参数开关）
3. 在 `scripts/<paper_name>/` 放入分析脚本
4. 在 `results/<paper_name>/` 创建 `{metrics,kcs,qual}/` + 实验记录MD
5. 在 `saved_models/<paper_name>/` 保存训练模型
6. 更新 `.claude/MEMORY.md` 论文状态表和已结合方法
7. 更新本文件

## 原则
- **模块化**: 每个方法的代码独立，新增文件模块化，修改通过参数开关
- **可扩展**: 通用代码保持在顶层，方法特有代码按 `<paper_name>` 隔离
- **可复现**: 结果、模型、脚本按方法分类保存
- **文档同步**: 每次迭代后更新 `.claude/` 下三个文件
