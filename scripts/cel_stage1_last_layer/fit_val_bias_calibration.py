#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from types import SimpleNamespace

import torch
from tqdm import tqdm

from dialogue_kt.data_loading import get_default_fold, load_annotated_data
from dialogue_kt.kt_data_loading import (
    LMKTCollatorPacked,
    LMKTCollatorUnpacked,
    LMKTDatasetPacked,
    LMKTDatasetUnpacked,
    get_dataloader,
)
from dialogue_kt.models.lm import get_model
from dialogue_kt.prompting import get_true_false_tokens
from dialogue_kt.training import (
    CELProbabilityCalibrator,
    _init_cel_calibrator,
    _init_cel_selector,
    apply_defaults,
    get_lmkt_loss_cel,
    test,
)
from dialogue_kt.utils import get_checkpoint_path, initialize_seeds


def build_args(source_model_name: str, target_model_name: str, base_model: str, result_subdir: str) -> SimpleNamespace:
    args = SimpleNamespace(
        command="test",
        dataset="mathdial",
        split_by_subject=False,
        typical_cutoff=1,
        tag_src="atc",
        debug=False,
        model_type="lmkt",
        model_name=target_model_name,
        base_model=base_model,
        inc_first_label=False,
        batch_size=None,
        crossval=False,
        testonval=False,
        agg="mean-ar",
        result_subdir=result_subdir,
        pack_kcs=True,
        quantize=False,
        prompt_inc_labels=False,
        emb_size=None,
        cel_mode="task_conditioned",
        cel_layer_idx=-1,
        cel_hook_site="last_block",
        cel_gamma=0.30,
        cel_selector_hidden_dim=512,
        cel_adapter_dim=None,
        cel_drop=0.1,
        cel_use_norm=False,
        cel_injection_variant="scalar_gate",
        cel_application_mode="token_residual",
        cel_train_selector_only=False,
        cel_output_calibration="bias",
        cel_calibrator_init_bias=0.0,
        cel_calibrator_init_scale=1.0,
        cel_train_calibrator_only=False,
        cel_train_selector_and_calibrator_only=False,
        epochs=None,
        lr=None,
        wd=None,
        gc=None,
        grad_accum_steps=None,
        r=None,
        lora_alpha=None,
        optim="adamw",
        pt_model_name=source_model_name,
        cel_selector_init_model_name=source_model_name,
        hyperparam_sweep=False,
    )
    apply_defaults(args)
    return args


def load_eval_model(args: SimpleNamespace, source_model_name: str):
    model, tokenizer = get_model(
        args.base_model,
        True,
        model_name=source_model_name,
        quantize=args.quantize,
        use_gradient_checkpointing=False,
        eager_attention=False,
    )
    model.eval()
    model._cel_selector = _init_cel_selector(model, args)
    selector_path = Path(get_checkpoint_path(source_model_name)) / "cel_selector.pt"
    if not selector_path.exists():
        raise FileNotFoundError(f"Missing selector checkpoint: {selector_path}")
    model._cel_selector.load_state_dict(torch.load(selector_path, map_location="cpu"))
    model._cel_calibrator = _init_cel_calibrator(SimpleNamespace(**{**args.__dict__, "cel_output_calibration": "none"}))
    model._cel_metrics = {}
    return model, tokenizer


def collect_split_probs(args: SimpleNamespace, source_model_name: str) -> tuple[list[int], list[float]]:
    model, tokenizer = load_eval_model(args, source_model_name)
    fold = get_default_fold(args)
    _, val_df, _ = load_annotated_data(args, fold)

    effective_pack_kcs = args.pack_kcs and args.cel_mode != "task_conditioned"
    dataset_cls = LMKTDatasetPacked if effective_pack_kcs else LMKTDatasetUnpacked
    collator_cls = LMKTCollatorPacked if effective_pack_kcs else LMKTCollatorUnpacked
    dataset = dataset_cls(val_df, tokenizer, args, skip_first_turn=not args.inc_first_label)
    dataloader = get_dataloader(dataset, collator_cls(tokenizer), args.batch_size, False)
    true_token, false_token = get_true_false_tokens(tokenizer)

    get_loss = lambda cur_model, batch, tt, ft, cur_args: get_lmkt_loss_cel(
        cur_model, batch, tt, ft, cur_args, tokenizer=tokenizer
    )

    all_labels: list[int] = []
    all_preds: list[float] = []
    for batch in tqdm(dataloader, desc="Fitting validation bias"):
        with torch.no_grad():
            _, _, corr_probs = get_loss(model, batch, true_token, false_token, args)
        all_labels.extend(batch["labels"].tolist())
        all_preds.extend(corr_probs.tolist())
    return all_labels, all_preds


def accuracy_at_threshold(labels: list[int], preds: list[float], threshold: float) -> float:
    hard = [1 if pred >= threshold else 0 for pred in preds]
    return 100.0 * sum(int(label == pred) for label, pred in zip(labels, hard)) / len(labels)


def best_threshold(labels: list[int], preds: list[float]) -> tuple[float, float]:
    thresholds = sorted(set(preds))
    best_acc = -1.0
    best_thr = 0.5
    for thr in thresholds:
        acc = accuracy_at_threshold(labels, preds, thr)
        if acc > best_acc:
            best_acc = acc
            best_thr = thr
    return best_acc, best_thr


def threshold_to_bias(threshold: float) -> float:
    threshold = min(max(float(threshold), 1e-6), 1 - 1e-6)
    return -math.log(threshold / (1.0 - threshold))


def pred_true_pct(preds: list[float], threshold: float = 0.5) -> float:
    return 100.0 * sum(pred >= threshold for pred in preds) / len(preds)


def ensure_target_checkpoint(source_model_name: str, target_model_name: str) -> Path:
    src = Path(get_checkpoint_path(source_model_name))
    dst = Path(get_checkpoint_path(target_model_name))
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        target = dst / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=True)
        else:
            shutil.copy2(child, target)
    return dst


def save_bias_artifacts(target_ckpt_dir: Path, bias: float, val_acc: float, val_threshold: float, val_pred_true: float) -> None:
    calibrator = CELProbabilityCalibrator("bias", init_bias=bias)
    torch.save(calibrator.state_dict(), target_ckpt_dir / "cel_calibrator.pt")
    (target_ckpt_dir / "val_bias_fit.json").write_text(
        json.dumps(
            {
                "fit_split": "validation",
                "objective": "accuracy_at_threshold_0.5_after_logit_bias",
                "best_validation_accuracy": val_acc,
                "best_validation_threshold": val_threshold,
                "derived_logit_bias": bias,
                "validation_pred_true_at_0.5": val_pred_true,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_model_name", required=True)
    parser.add_argument("--target_model_name", required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--result_subdir", default="cel_stage1_last_layer")
    args = parser.parse_args()

    initialize_seeds(221)
    eval_args = build_args(args.source_model_name, args.target_model_name, args.base_model, args.result_subdir)
    labels, preds = collect_split_probs(eval_args, args.source_model_name)
    val_acc, val_thr = best_threshold(labels, preds)
    bias = threshold_to_bias(val_thr)
    val_pred_true = pred_true_pct(preds)

    target_ckpt_dir = ensure_target_checkpoint(args.source_model_name, args.target_model_name)
    save_bias_artifacts(target_ckpt_dir, bias, val_acc, val_thr, val_pred_true)

    print(
        json.dumps(
            {
                "source_model": args.source_model_name,
                "target_model": args.target_model_name,
                "validation_best_acc": round(val_acc, 4),
                "validation_best_threshold": round(val_thr, 6),
                "derived_logit_bias": round(bias, 6),
                "validation_pred_true_at_0.5": round(val_pred_true, 4),
            },
            ensure_ascii=False,
        )
    )

    test(eval_args)


if __name__ == "__main__":
    main()
