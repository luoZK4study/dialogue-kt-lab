# Dialogue Knowledge Tracing

## Current Workspace Status

This repository is a compact research workspace derived from Dialogue-KT. The
active scope is MathDial ATC with Qwen3-1.7B + LoRA and CEL modules:

- A module: token-level task-conditioned evidence selection;
- B module: contextual transformation of the complementary non-evidence view;
- future modules use the same A/B/C naming convention;
- certified baseline: Overall Acc/AUC `69.22 / 74.95`, Final Acc/AUC
  `61.17 / 75.32`;
- A module historical reference: Overall Acc/AUC `70.03 / 75.38`, Final
  Acc/AUC `66.80 / 77.26`;
- paper target: Overall AUC `76.71`;
- A and A+B will be retrained from raw Qwen after the refactor. Old staged
  records and unrelated experiment results are intentionally not part of the
  current workspace.

The target A+B path is:

```text
H -> A(a) -> h_r, h_n -> B(h_n) = h_nb -> h_m
  -> shared P(h_r), shared P(h_m)
```

with `L_total = λ_r BCE(p_r, y) + λ_m BCE(p_m, y) +
λ_cons JS([p_r, 1-p_r], [p_m, 1-p_m])`. All enabled modules are trained as one
end-to-end model from epoch 1. Parameter-group learning rates are allowed, but
not independent or alternating module training. Validation-best checkpoint,
fixed early-stopping settings, final test, and audits are required.

See [AGENTS.md](AGENTS.md), [.claude/STAGE2_HANDOFF.md](.claude/STAGE2_HANDOFF.md),
and [results/method_design/CEL_Modular_Flow.md](results/method_design/CEL_Modular_Flow.md).

Canonical rerun entry points (execute in the configured local or SSH
environment) are:

```bash
bash scripts/cel/run_a_unified.sh
bash scripts/cel/run_a_b_unified.sh
```

Both commands start from raw Qwen and use one end-to-end training loop. Set
`DIALOGUE_KT_ROOT`, `BASE_MODEL`, `PYTHON_BIN`, or the documented learning-rate
environment variables when running on the training server.
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
