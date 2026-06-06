# Dialogue-KT 论文与代码深度分析

## 1. 论文核心思想

论文：Scarlatos et al., *Exploring Knowledge Tracing in Tutor-Student Dialogues using LLMs*，LAK 2025。
本地文件：`literature/dialogue-kt.pdf`。

论文提出 **dialogueKT**：把开放式 tutor-student 对话转换成知识追踪任务。核心流程分三步：

1. **对话标注**：用 GPT-4o 对每个 tutor/student turn pair 标注：
   - 学生回答 correctness：`true/false/na`；
   - 该 turn 涉及的 Knowledge Components (KCs)，使用 Common Core / Achieve the Core standards。
2. **KT 建模**：把一个 tutor 问题 + 学生回答 pair 看作 KT 的一个 time step；历史对话作为 student knowledge state 的证据。
3. **LLMKT**：用 Llama-3.1-8B-Instruct + LoRA 微调，把当前 dialogue history 和 KC 文本输入 LLM，让模型在 `True` / `False` token 上输出 KC mastery 概率，再将多个 KC mastery 聚合成学生回答正确概率。

论文关键公式：

- 多 KC 的回答正确概率使用 compensatory mean：

  `P(y_j = 1) = (1/K) * sum_k P(z_jk = 1)`

- LLMKT 的 KC mastery：

  `z_hat_jk = softmax(logits(True), logits(False))`

- 损失：

  对 turn-level correctness 使用 binary cross entropy。

## 2. 论文实验设置与汇报结果

论文使用两个数据集：

| 数据集 | 对话数 | labels | unique KCs | 设置 |
|---|---:|---:|---:|---|
| CoMTA | 153 | 623 | 164 | 5-fold CV |
| MathDial | 2823 | 13200 | 145 | train/test split |

关键指标：Accuracy、AUC、F1。KT 文献中 AUC 是主要指标。论文评估时排除每个 dialogue 的第一个 correctness label，因为传统 KT 方法没有历史信息。

论文 Table 2 的 MathDial 结果：

| 方法 | MathDial Overall AUC |
|---|---:|
| BKT | 64.19 |
| DKT | 63.22 |
| DKVMN | 60.42 |
| AKT | 63.31 |
| SAINT | 60.10 |
| simpleKT | 63.83 |
| DKT-Sem | 66.18 |
| **LLMKT** | **76.71** |

本项目目标以论文 LLMKT MathDial Overall AUC **76.71** 为对照。

## 3. 对应代码分析

### 3.1 CLI 入口：`dialogue_kt/main.py`

主要子命令：

- `annotate`：执行 correctness/KC 自动标注；
- `train`：训练 LLMKT 或 baseline KT 模型；
- `test`：加载 checkpoint 并输出 metrics / qual / KC predictions；
- `visualize`：学习曲线分析。

关键参数：

- `--dataset mathdial/comta`
- `--model_type lmkt`
- `--base_model`
- `--quantize 0`：禁用 4-bit 量化；注意项目中 `bool_type(x) = x != "0"`。
- `--testonval`：把 validation split 当作 test_df 评估。该参数会写默认 `results/qual_<model>.csv`，因此必须立即归档，避免覆盖 test 文件。
- `--debug` | `False` | 数据子集（100 训练/25 验证） |
- `--pack_kcs` | `True` | 多 KC 打包到单个 prompt |
- `--agg` | `mean-ar` | KC 聚合方式（`prod` / `mean-ar` / `mean-geo`）
- `--epochs` | 2 | 训练轮数 |

### 3.2 数据处理：`dialogue_kt/data_loading.py` 与 `dialogue_kt/kt_data_loading.py`

`load_annotated_data(args)`：

- MathDial：
  - 读取 `data/annotated/mathdial_train_atc.csv` 和 `mathdial_test_atc.csv`；
  - train CSV 随机打乱后按 80/20 分 train/validation；
  - test CSV 作为最终 test。

`apply_annotations(sample)`：

- 把 annotation 中的 correctness 和 KCs 写入 dialogue turn；
- 对 MathDial final turn，使用 `self_correctness` 修正最终 turn correctness。

`LMKTDatasetPacked`：

- 构造 packed prompt；
- 同一个 turn 的多个 KC 被拼接在同一 prompt 后；
- collator 使用 3D attention mask 隔离不同 KC continuation，避免 KC 之间互相看见。

`LMKTDatasetUnpacked`:

-每个 KC 单独一个 prompt

### 3.3 Prompt：`dialogue_kt/prompting.py`

LLMKT system prompt 要求模型扮演数学教师，根据当前对话历史判断学生是否掌握某个 KC，并只输出 `True` 或 `False`。

MathDial prompt 包含：

- problem；
- correct solution；
- initial incorrect student solution；
- dialogue history，当前 turn 只包含 teacher utterance，不包含当前 student response，以避免泄漏 correctness label。

- `KT_SYSTEM_PROMPT`：LLM 扮演数学教师
- `kt_user_prompt()`：问题上下文 + 对话历史 + 待评估 KC
- `get_true_false_tokens()`：Qwen True=2514, False=4049

### 3.4 训练/评估：`dialogue_kt/training.py`

`get_lmkt_loss_packed()`：

1. 从模型输出 logits 中取每个 KC continuation 的最后 token 位置；
2. 取 `True` / `False` token logits；
3. softmax 得到 KC mastery probability；
4. 对多个 KC 使用 `mean-ar` 聚合成 correctness probability；
5. 用 BCE loss 训练。

`test_lmkt()`：

- 加载 LoRA checkpoint；
- 构造 test dataset；
- 收集 `all_labels` / `all_preds`；
- 计算 Overall metrics 和 Final Turn metrics；
- 评估 Accuracy, AUC, Precision, Recall, F1（Overall + Final Turn）
- 写出：
  - `results/metrics_<model>.txt`
  - `results/qual_<model>.csv`
  - `results/kcs_<model>.json`

重要发现：`--testonval` 和普通 test 使用同一默认文件名，因此 clean test 验证必须把输出复制到独立目录。

### `utils.py`

- `bool_type(x)`: `x != "0"` → `--quantize 0` 才禁用
- `initialize_seeds(221)`: 固定随机种子，启用确定性算法

