# Results 目录说明

`results/` 只保存可复用的基线、方法设计和新一轮正式训练输出。旧 Stage 1/Stage 2 candidate、分阶段日志、无关方法结果和同步快照已从工作树移除；需要追溯时可从 Git 历史检查点恢复。

## 当前结构

```text
results/
|-- baseline/       # 认证 baseline 的记录与必要输出
|-- a/              # A 模块统一端到端重训输出
|-- a_b/            # A+B 模块统一端到端重训输出
|-- method_design/  # 总体设计、A 实现、B 实现
`-- README.md
```

新训练输出建议按以下约定保存：

```text
results/<a|a_b>/
|-- metrics/
|-- qual/
|-- kcs/
`-- audit/
```

模型名和目录名使用 `a`、`a_b` 等模块组合；历史 checkpoint 的长名称只能出现在 provenance 和 audit 中。不要再创建 `stage1_last_layer`、`stage2_environment` 或按 bootstrap/warmup/joint 拆开的默认结果目录。

正式结果必须记录 raw Qwen 初始化来源、完整模块清单、parameter-group learning rate、`max_epochs`、validation 指标、`patience`、`min_delta`、validation-best 恢复情况、final test 和审计结论。A、B、C 等模块必须在同一模型和同一训练闭环中解释。
