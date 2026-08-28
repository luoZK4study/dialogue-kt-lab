# CLAUDE.md — 项目操作指南

## 1. 当前定位

本仓库已经从“多方法试验场”收敛成 **baseline + CEL** 工作区。

- 任务：Dialogue-KT 二分类，比较 `True` / `False` token logits
- 数据集：MathDial ATC
- 默认底座：Qwen3-1.7B + LoRA (`r=16`, `lora_alpha=16`)
- 论文目标：Overall AUC **76.71**
- 当前 baseline：`lmkt_qwen3_1.7b_recert_20260620`，Overall AUC **74.95**, Overall Acc **69.22**
- 当前固定的 Stage 1 暂定方案：`cel_task_conditioned_lastlayer_v1_qwen3_1.7b`，Overall AUC **75.29**, Overall Acc **68.87**
- 当前优先任务：**继续调优 `task_conditioned`，直到 Overall AUC 与 Overall Acc 同时超过当前 baseline**

开启新对话时，先读：

1. `results/cel_stage1_last_layer/STATUS.md`
2. `.claude/MEMORY.md`
3. `.claude/STRUCTURE.md`
4. `results/baseline/Baseline_DialogueKT_实验记录.md`
5. `results/cel_stage1_last_layer/CEL_Stage1_LastLayer_DialogueKT_实验记录.md`
6. `results/cel_stage1_last_layer/TASK_CONDITIONED_TUNING_LOOP.md`
7. `results/method_design/CEL_Stage2_Environment_Generator_Implementation.md`

如果这些文件之间有冲突，以 `.claude/MEMORY.md` 的阶段结论为准；它明确记录“当前该做什么”和“当前停在哪个阶段”。

如果 `results/cel_stage1_last_layer/STATUS.md` 的 `Generated at` 明显落后当前时间，并且需要重新确认远端实验产物，先执行：

```bash
bash scripts/cel_stage1_last_layer/sync_results_from_server.sh
python scripts/cel_stage1_last_layer/generate_status_summary.py
python scripts/cel_stage1_last_layer/generate_comparison_report.py
python scripts/cel_stage1_last_layer/generate_detailed_analysis.py
python scripts/cel_stage1_last_layer/update_baseline_record.py
python scripts/cel_stage1_last_layer/update_stage1_last_layer_record.py
python scripts/cel_stage1_last_layer/update_task_conditioned_tuning_loop.py
```

## 2. 本地环境

| 项目 | 详情 |
|------|------|
| 路径 | `/home/lzk/code/dialogue-kt/` |
| OS | WSL2 (Ubuntu) |
| GPU | RTX 5060 8GB |

Windows 宿主机代理：

```bash
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
```

## 3. 服务器环境

| 项目 | 详情 |
|------|------|
| 路径 | `/home/user4/dialogue-kt/` |
| GPU | 2x RTX 3090 24GB |
| CUDA | driver 12.2 / toolkit 11.7 |
| Conda | `/opt/anaconda3/` |
| Env | `luo_2` |
| Python | 3.10.18 |
| torch | 2.4.1+cu118 |
| transformers | 5.7.0 |
| peft | 0.19.1 |

连接：

```bash
ssh -i ~/.ssh/id_rsa_u4 -p 22049 user4@119.29.183.125
```

激活：

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate luo_2
cd /home/user4/dialogue-kt
export WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

## 4. 当前目录约定

- 代码入口：`dialogue_kt/main.py`
- CEL 核心：`dialogue_kt/cel_methods.py`
- 训练与测试：`dialogue_kt/training.py`
- 数据与 prompt：`dialogue_kt/kt_data_loading.py`, `dialogue_kt/prompting.py`
- 当前基线结果：`results/baseline/`
- 历史 Stage 1 A 结果：`results/cel_stage1/`
- 当前完成的 Stage 1 last-layer 结果：`results/cel_stage1_last_layer/`
- 设计说明：`results/method_design/`
- 当前活跃脚本：`scripts/cel_stage1_last_layer/`

当前 `results/cel_stage1_last_layer/` 中重点看：

- `STATUS.md`
- `COMPARISON.md`
- `DETAILED_ANALYSIS.md`
- `CEL_Stage1_LastLayer_DialogueKT_实验记录.md`

Stage 2 正式实现时，目录命名固定为：

- `scripts/cel_stage2_environment/`
- `results/cel_stage2_environment/`

## 5. 常用同步命令

同步核心代码：

```bash
rsync -avz -e "ssh -i ~/.ssh/id_rsa_u4 -p 22049" \
  /home/lzk/code/dialogue-kt/dialogue_kt/ \
  user4@119.29.183.125:/home/user4/dialogue-kt/dialogue_kt/ \
  --exclude '__pycache__' --exclude '*.pyc'
```

同步当前脚本与文档：

```bash
rsync -avz -e "ssh -i ~/.ssh/id_rsa_u4 -p 22049" \
  /home/lzk/code/dialogue-kt/scripts/cel_stage1_last_layer/ \
  user4@119.29.183.125:/home/user4/dialogue-kt/scripts/cel_stage1_last_layer/

rsync -avz -e "ssh -i ~/.ssh/id_rsa_u4 -p 22049" \
  /home/lzk/code/dialogue-kt/.claude/ \
  user4@119.29.183.125:/home/user4/dialogue-kt/.claude/
```

## 6. 当前默认闭环

当前默认不是直接做 Stage 2，而是围绕 `task_conditioned` 执行下面这个 loop：

1. 检查 SSH 服务器上当前训练状态。
2. 同步远端 `metrics / qual / logs` 回本地。
3. 更新 `STATUS.md`、`DETAILED_ANALYSIS.md`、`TASK_CONDITIONED_TUNING_LOOP.md`。
4. 判断当前轮次是否已经同时超过 baseline：
   - Overall AUC `> 74.95`
   - Overall Acc `> 69.22`
5. 若未达标，只做最小必要改动，然后继续下一轮远端训练。

Stage 2 只在这项目标达成后再继续推进。

## 7. 常用命令

复现 baseline：

```bash
bash scripts/cel_stage1_last_layer/run_baseline.sh
```

复现当前 Stage 1 暂定方案：

```bash
bash scripts/cel_stage1_last_layer/run_task_conditioned_v1.sh
```

刷新本地汇总：

```bash
python scripts/cel_stage1_last_layer/generate_status_summary.py
python scripts/cel_stage1_last_layer/generate_comparison_report.py
python scripts/cel_stage1_last_layer/generate_detailed_analysis.py
```

## 8. 注意事项

- `--quantize 0` 才是关闭量化；`--quantize False` 会被解析成 True。
- `--debug` 当前固定使用约 `100 train / 25 val` 做链路检查。
- `--cel_selector_init_model_name` 只会 warm-start selector，不会覆盖 `--pt_model_name` 加载的 baseline LoRA。
- 当前 task_conditioned 调优状态文件是 `results/cel_stage1_last_layer/TASK_CONDITIONED_TUNING_LOOP.md`。
- `v11` / `v12` 这类并行候选需要都完成后再做比较，不要只看到一个 metrics 文件就提前收尾。
- 当前 Stage 1 暂定方案的有效性来自最后一层 `task_conditioned` 实验，Stage 2 不应再把这轮 last-layer 误记为 environment generator。
- 如果开始实现 Stage 2，先固定 Stage 1 `task_conditioned v1` 方案，再新增 B 模块；不要一开始同时改 selector、hook 位点和 environment 生成方式。
