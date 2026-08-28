# Dialogue Knowledge Tracing

## Current Workspace Status

This repository is a research workspace derived from the upstream Dialogue-KT
codebase. The active scope is narrower than the upstream README below:

- task: MathDial ATC dialogue knowledge tracing with `True` / `False` token
  probabilities;
- model: Qwen3-1.7B with LoRA;
- methods in scope: the recertified LLMKT baseline and CEL A/B;
- certified baseline: `lmkt_qwen3_1.7b_recert_20260620`, Overall Acc/AUC
  `69.22 / 74.95`;
- audited A-module reference: Overall Acc/AUC `70.03 / 75.38` and Final
  Acc/AUC `66.80 / 77.26`;
- the historical A-module checkpoint name is
  `cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b`;
  this identifier is retained for provenance only, while method prose calls it
  the A module;
- paper reference target: Overall AUC `76.71`, which has not yet been reached;
- Stage 2 first-round Shuffle, token-wise MLP, and small Transformer candidates
  completed training and audit, but used a legacy single-path environment
  residual flow. They are historical pilots and do not validate the current
  target method; no Stage 2 winner is declared.
- the current target constructs `h_r = ((a + 1) / 2) ⊙ H`,
  `h_n = ((1 - a) / 2) ⊙ H`, `h_nb = B(h_n)`, and
  `h_m = h_r + β · h_nb`, then compares evidence-only and mixed predictions
  through the same downstream Qwen network;
- the dual-path forward, three-term supervised/consistency objective, tensor
  contracts, and capacity-adequate B module are implemented;
- the default protocol is unified end-to-end training: all enabled modules
  (including an optional calibrator) share one optimizer and objective from
  epoch 1, with fixed max epochs, validation-based best-checkpoint selection,
  patience, and min_delta early stopping; staged warmups are historical or
  explicitly declared exceptions;
- the training CLI exposes `--max_epochs`, `--patience`, and `--min_delta`;
  `--epochs` remains a compatibility alias, and historical runs are not
  retroactively claimed to have used the new early-stopping protocol;
- Round 1 of the three-round configuration study is running from raw Qwen
  with `model/data seed=1221`; no extra seed is being run and Stage 3 is not
  started.

The current authoritative Stage 1 result was successfully refreshed from the
training server on `2026-08-12 23:17:24`. It contains metrics, per-turn
predictions, KC outputs, and logs, but `saved_models/` contains no checkpoints.
Use `/home/luo/miniconda3/envs/diagkt` for local source and smoke tests; formal
checkpoint-based training and evaluation remain on the SSH server.

For a new Stage 2 conversation, start with [.claude/STAGE2_HANDOFF.md](.claude/STAGE2_HANDOFF.md),
[AGENTS.md](AGENTS.md), and [results/method_design/CEL_Stage2_实施流程与思路.md](results/method_design/CEL_Stage2_实施流程与思路.md).
The original upstream project instructions continue below for reference.

## Upstream Project README

This is the official repo for the paper <a href="https://arxiv.org/abs/2409.16490">Exploring Knowledge Tracing in Tutor-Student Dialogues using LLMs</a>. The primary contributions here include 1) LLMKT and DKT-Sem, our language model-based KT models, 2) code to train and evaluate KT models, including the DKT family and BKT, on the dialogue knowledge tracing task, and 3) code to automatically annotate dialogues with knowledge component and correctness labels using the OpenAI API.

If you use our code or find this work useful in your research then please cite us!
```
@inproceedings{scarlatos2024exploringknowledgetracingtutorstudent,
      title={Exploring Knowledge Tracing in Tutor-Student Dialogues using LLMs},
      author={Alexander Scarlatos and Ryan S. Baker and Andrew Lan},
      year={2025},
      booktitle={Proceedings of the 15th Learning Analytics and Knowledge Conference, {LAK} 2025, Dublin, Ireland, March 3-7, 2025},
      publisher={{ACM}},
}
```

## Annotated Data

Annotated versions of the CoMTA and MathDial datasets (i.e. including per-turn knowledge component and correctness labels) are available in `data/annotated`, and can be loaded as-is during knowledge tracing training.

These versions of the datasets are subject to their original licenses. The license for CoMTA is available in `data/annotated/COMTA_LICENSE.txt` and MathDial is licensed under <a href="https://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.

## Setup

### Download Data
This step is not necessary to reproduce our knowledge tracing results since we release the annotated data in `data/annotated`. However, you can follow the steps below to replicate our workflow or to experiment with custom data annotation.

<b>Achieve the Core (ATC)</b>: Download the <a href="https://huggingface.co/datasets/allenai/achieve-the-core">ATC HuggingFace dataset</a> and put `standards.jsonl` and `domain_groups.json` under `data/src/ATC/`. At the time of releasing this code, the data was not accessible via HuggingFace due to a bug. If the data is still not accessible then you can contact us or the authors of <a href="https://arxiv.org/pdf/2408.04226">the paper</a> to send you a copy.

<b>CoMTA</b>: Download the <a href="https://github.com/Khan/tutoring-accuracy-dataset/blob/main/CoMTA_dataset.json">CoMTA data file</a> and put it under `data/src`.

<b>MathDial</b>: Clone the <a href="https://github.com/eth-nlped/mathdial/tree/main">MathDial repo</a> and put the root under `data/src`.

### Environment
We used Python 3.10.12 in the development of this work. Run the following to set up a Python environment:
```
python -m venv dk
source dk/bin/activate
pip install -r requirements.txt
```

Also add the following to your environment:
```
export OPENAI_API_KEY=<your key here> # For automated annotation via OpenAI
export CUBLAS_WORKSPACE_CONFIG=:4096:8 # For enabling deterministic operations
```

## Prepare Dialogues for KT (Run Annotation with OpenAI)
This step is not necessary to reproduce our results because we release the annotated datasets, but is here for reference.

Dialogue KT requires each dialogue turn to be annotated with correctness and knowledge component (KC) labels. We automate this process with LLM prompting via the OpenAI API. You can run the following to tag correctness and ATC standard KCs on the two datasets:
```
python -m dialogue_kt.main annotate --mode collect --openai_model gpt-4o --dataset comta
python -m dialogue_kt.main annotate --mode collect --openai_model gpt-4o --dataset mathdial
```

To see statistics on the resulting labels, run:
```
python -m dialogue_kt.main annotate --mode analyze --dataset comta
python -m dialogue_kt.main annotate --mode analyze --dataset mathdial
```

## Train and Evaluate KT Methods
Each of the following runs a train/test cross-validation on the CoMTA data for a different model:
```
python -m dialogue_kt.main train --dataset comta --crossval --model_type lmkt --model_name lmkt_comta         # LLMKT
python -m dialogue_kt.main train --dataset comta --crossval --model_type dkt-sem --model_name dkt-sem_comta   # DKT-Sem
python -m dialogue_kt.main train --dataset comta --crossval --model_type dkt --model_name dkt_comta           # DKT
python -m dialogue_kt.main train --dataset comta --crossval --model_type dkvmn --model_name dkvmn_comta       # DKVMN
python -m dialogue_kt.main train --dataset comta --crossval --model_type akt --model_name akt_comta           # AKT
python -m dialogue_kt.main train --dataset comta --crossval --model_type saint --model_name saint_comta       # SAINT
python -m dialogue_kt.main train --dataset comta --crossval --model_type simplekt --model_name simplekt_comta # simpleKT
python -m dialogue_kt.main train --dataset comta --crossval --model_type bkt                                  # BKT
```

Check the `results` folder for metric summaries and turn-level predictions for analysis.

To see all training options, run:
```
python -m dialogue_kt.main train --help
```

### Hyperparameter Sweep
We run a grid search to find the optimal hyperparameters for the DKT family models. For example, to run a search for DKT on CoMTA, run the following (crossval is inferred and model_name is set automatically):
```
python -m dialogue_kt.main train --dataset comta --hyperparam_sweep --model_type dkt
```

The output will indicate the model that achieved the highest validation AUC. To get its performance on the test folds, run:
```
python -m dialogue_kt.main test --dataset comta --crossval --model_type dkt --model_name <copy from output> --emb_size <get from model_name>
```

#### Best Hyperparameters Found

CoMTA:
- DKT-Sem: lr=2e-4, emb_size=256
- DKT: lr=1e-3, emb_size=32
- DKVMN: lr=1e-4, emb_size=16
- AKT: lr=5e-3, emb_size=32
- SAINT: lr=1e-3, emb_size=32
- simpleKT: lr=2e-4, emb_size=16

MathDial:
- DKT-Sem: lr=2e-3, emb_size=512
- DKT: lr=5e-3, emb_size=256
- DKVMN: lr=1e-3, emb_size=128
- AKT: lr=2e-4, emb_size=64
- SAINT: lr=2e-4, emb_size=64
- simpleKT: lr=5e-4, emb_size=256

## Visualize Learning Curves
To generate the learning curve graphs, run the following (they will be placed in `results`):
```
python -m dialogue_kt.main visualize --dataset comta --model_name <trained model to visualize predictions for>
```
