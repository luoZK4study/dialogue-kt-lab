import json
import math
import os
from tqdm import tqdm
import torch
import transformers
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
from pykt.models.dkt import DKT
from pykt.models.akt import AKT
from pykt.models.dkvmn import DKVMN
from pykt.models.saint import SAINT
from pyBKT.models import Model as BKT
from sentence_transformers import SentenceTransformer

from dialogue_kt.models.lm import get_model
from dialogue_kt.models.dkt_multi_kc import DKTMultiKC
from dialogue_kt.models.dkt_sem import DKTSem
from dialogue_kt.models.simplekt import simpleKT
from dialogue_kt.data_loading import (load_annotated_data, get_kc_result_filename, get_qual_result_filename, get_default_fold, load_kc_dict,
                          correct_to_str, standards_to_str, get_model_file_suffix, get_results_dir, COMTA_SUBJECTS)
from dialogue_kt.kt_data_loading import (LMKTDatasetUnpacked, LMKTCollatorUnpacked, LMKTDatasetPacked, LMKTCollatorPacked,
                             DKTDataset, DKTCollator, get_dataloader, apply_annotations)
from dialogue_kt.prompting import get_true_false_tokens
from dialogue_kt.utils import device, get_checkpoint_path, initialize_seeds
from dialogue_kt.cel_methods import (
    MLPGateSelector,
    MLPShiftSelector,
    AdapterGateSelector,
    AdapterShiftSelector,
    TaskConditionedSelector,
    TaskConditionedShiftSelector,
    build_environment_generator,
    register_cel_dual_path_hook,
    register_cel_injection_hook,
)


PROB_EPS = 1e-6
# The expanded two-branch suffix is safe below this conservative attention
# risk on the formal 24 GiB GPU. Longer batches use the exact serial gradient
# split so one outlier cannot terminate an otherwise valid epoch.
STAGE2_SERIAL_RISK_THRESHOLD = 8_000_000
CEL_SELECTOR_FILENAME = "cel_selector.pt"
CEL_CALIBRATOR_FILENAME = "cel_calibrator.pt"
CEL_ENVIRONMENT_FILENAME = "cel_environment.pt"
CEL_STAGE2_MANIFEST_FILENAME = "cel_stage2_manifest.json"
STAGE2_PHASES = (
    "a_bootstrap",
    "calibrator_warmup",
    "a_joint",
    "b_warmup",
    "joint",
)
HISTORICAL_STAGE2_PHASES = ("a_bootstrap", "calibrator_warmup", "a_joint", "b_warmup")


def _softplus_inverse_with_floor(target: float, floor: float = 1e-4) -> float:
    shifted = max(float(target) - floor, 1e-8)
    return math.log(math.expm1(shifted))


class CELProbabilityCalibrator(torch.nn.Module):
    def __init__(self, mode: str, init_bias: float = 0.0, init_scale: float = 1.0):
        super().__init__()
        self.mode = mode
        if mode == "none":
            self.logit_scale_raw = None
            self.logit_bias = None
        elif mode == "bias":
            self.logit_scale_raw = None
            self.logit_bias = torch.nn.Parameter(torch.tensor([init_bias], dtype=torch.float32))
        elif mode == "affine":
            scale_raw = _softplus_inverse_with_floor(init_scale)
            self.logit_scale_raw = torch.nn.Parameter(torch.tensor([scale_raw], dtype=torch.float32))
            self.logit_bias = torch.nn.Parameter(torch.tensor([init_bias], dtype=torch.float32))
        else:
            raise ValueError(f"Unsupported CEL output calibration mode: {mode}")

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return probs
        probs = probs.clamp(min=PROB_EPS, max=1 - PROB_EPS)
        logits = torch.logit(probs)
        if self.mode == "bias":
            logits = logits + self.logit_bias
        elif self.mode == "affine":
            scale = torch.nn.functional.softplus(self.logit_scale_raw) + 1e-4
            logits = logits * scale + self.logit_bias
        return torch.sigmoid(logits)


def _sanitize_binary_label_tensor(labels: torch.Tensor):
    labels = labels.float()
    if labels.numel() == 0:
        zero = torch.tensor(0.0, device=labels.device)
        nan = torch.tensor(float("nan"), device=labels.device)
        return labels, {
            "invalid_frac": zero,
            "out_of_range_frac": zero,
            "sanitized_frac": zero,
            "raw_min": nan,
            "raw_max": nan,
            "safe_min": nan,
            "safe_max": nan,
        }

    finite_mask = torch.isfinite(labels)
    out_of_range_mask = finite_mask & ((labels < 0.0) | (labels > 1.0))
    sanitized_mask = (~finite_mask) | out_of_range_mask

    if finite_mask.any():
        finite_vals = labels[finite_mask]
        raw_min = finite_vals.min().detach()
        raw_max = finite_vals.max().detach()
    else:
        raw_min = torch.tensor(float("nan"), device=labels.device)
        raw_max = torch.tensor(float("nan"), device=labels.device)

    safe_labels = torch.nan_to_num(labels, nan=0.0, posinf=1.0, neginf=0.0)
    safe_labels = safe_labels.clamp(min=0.0, max=1.0)

    stats = {
        "invalid_frac": (~finite_mask).to(dtype=labels.dtype).mean().detach(),
        "out_of_range_frac": out_of_range_mask.to(dtype=labels.dtype).mean().detach(),
        "sanitized_frac": sanitized_mask.to(dtype=labels.dtype).mean().detach(),
        "raw_min": raw_min,
        "raw_max": raw_max,
        "safe_min": safe_labels.min().detach(),
        "safe_max": safe_labels.max().detach(),
    }
    return safe_labels, stats


def _sanitize_probability_tensor(probs: torch.Tensor, eps: float = PROB_EPS):
    probs = probs.float()
    if probs.numel() == 0:
        zero = torch.tensor(0.0, device=probs.device)
        nan = torch.tensor(float("nan"), device=probs.device)
        return probs, {
            "invalid_frac": zero,
            "out_of_range_frac": zero,
            "sanitized_frac": zero,
            "raw_min": nan,
            "raw_max": nan,
            "safe_min": nan,
            "safe_max": nan,
        }

    finite_mask = torch.isfinite(probs)
    out_of_range_mask = finite_mask & ((probs < 0.0) | (probs > 1.0))
    boundary_mask = finite_mask & ((probs <= eps) | (probs >= 1.0 - eps))
    sanitized_mask = (~finite_mask) | boundary_mask

    if finite_mask.any():
        finite_vals = probs[finite_mask]
        raw_min = finite_vals.min().detach()
        raw_max = finite_vals.max().detach()
    else:
        raw_min = torch.tensor(float("nan"), device=probs.device)
        raw_max = torch.tensor(float("nan"), device=probs.device)

    safe_probs = torch.nan_to_num(probs, nan=0.5, posinf=1.0 - eps, neginf=eps)
    safe_probs = safe_probs.clamp(min=eps, max=1.0 - eps)

    stats = {
        "invalid_frac": (~finite_mask).to(dtype=probs.dtype).mean().detach(),
        "out_of_range_frac": out_of_range_mask.to(dtype=probs.dtype).mean().detach(),
        "sanitized_frac": sanitized_mask.to(dtype=probs.dtype).mean().detach(),
        "raw_min": raw_min,
        "raw_max": raw_max,
        "safe_min": safe_probs.min().detach(),
        "safe_max": safe_probs.max().detach(),
    }
    return safe_probs, stats


def _compute_bce_loss_from_probs(corr_probs: torch.Tensor, labels: torch.Tensor):
    corr_probs, prob_stats = _sanitize_probability_tensor(corr_probs)
    safe_labels, label_stats = _sanitize_binary_label_tensor(labels)
    # BCEWithLogitsLoss avoids CUDA-side range asserts while remaining
    # equivalent to BCELoss on sanitized probabilities.
    corr_logits = torch.logit(corr_probs)
    loss = torch.nn.BCEWithLogitsLoss()(corr_logits, safe_labels)
    prob_stats.update(_prefix_metric_keys("label", label_stats))
    return loss, corr_probs, prob_stats


def _prefix_metric_keys(prefix: str, metrics_dict):
    return {f"{prefix}_{key}": value for key, value in metrics_dict.items()}

# ===== Common Functions =====

def apply_defaults(args):
    requested_max_epochs = getattr(args, "max_epochs", None)
    requested_epochs = getattr(args, "epochs", None)
    if args.model_type == "lmkt":
        defaults = {
            "epochs": 3,
            "lr": 2e-4,
            "wd": 1e-2,
            "gc": 1.0,
            "batch_size": 1,
            "grad_accum_steps": 32,
            "r": 16,
            "lora_alpha": 16
        }
    else:
        defaults = {
            "epochs": 100,
            "lr": 1e-3,
            "wd": 1e-2,
            "gc": 0,
            "batch_size": 64,
            "grad_accum_steps": 1,
            "emb_size": 64
        }
    for key, val in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, val)
    # ``--epochs`` remains a compatibility alias for existing launchers, but
    # the public training contract is an explicit max-epoch budget.  Normalize
    # both names to one value so every training loop uses the same schedule.
    if requested_max_epochs is not None and requested_epochs is not None:
        if int(requested_max_epochs) != int(requested_epochs):
            raise ValueError(
                "--max_epochs and --epochs specify different values; use one "
                "shared max-epoch budget"
            )
    max_epochs = requested_max_epochs if requested_max_epochs is not None else requested_epochs
    if max_epochs is None:
        max_epochs = defaults.get("epochs", 1)
    max_epochs = int(max_epochs)
    if max_epochs <= 0:
        raise ValueError("max_epochs must be a positive integer")
    args.max_epochs = max_epochs
    # Internal and historical helpers still read ``epochs``.  Keeping this
    # alias avoids changing old stage launchers while enforcing one budget.
    args.epochs = max_epochs

    patience = int(getattr(args, "patience", 0) or 0)
    min_delta = float(getattr(args, "min_delta", 0.0) or 0.0)
    if patience < 0:
        raise ValueError("patience must be >= 0 (0 disables early stopping)")
    if min_delta < 0:
        raise ValueError("min_delta must be >= 0")
    args.patience = patience
    args.min_delta = min_delta
    print("Arguments:", args)


def _validation_improved(current: float, best: float | None, min_delta: float) -> bool:
    """Return whether a lower validation loss is a meaningful improvement."""
    return best is None or float(current) < float(best) - float(min_delta)


def _early_stopping_update(best: float | None, stale_epochs: int, current: float, args):
    """Update validation state and report whether the epoch budget should stop.

    ``patience=0`` intentionally disables early stopping.  Otherwise patience
    counts consecutive epochs without an improvement of at least ``min_delta``.
    """
    if _validation_improved(current, best, getattr(args, "min_delta", 0.0)):
        return float(current), 0, False, True
    stale_epochs += 1
    patience = int(getattr(args, "patience", 0) or 0)
    should_stop = patience > 0 and stale_epochs >= patience
    return best, stale_epochs, should_stop, False

def hyperparam_sweep(args):
    apply_defaults(args)
    args.testonval = True
    args.crossval = args.dataset == "comta"
    model_names = []
    results = []
    if args.model_type == "lmkt":
        for lr in [5e-5, 1e-4, 2e-4, 3e-4]:
            for r in [4, 8, 16, 32]:
                args.model_name = f"hpsweep_{args.dataset}_{args.tag_src}_lmkt_agg{args.agg}_lr{lr}_r{r}"
                args.lr = lr
                args.r = r
                args.lora_alpha = r
                model_names.append(args.model_name)
                results.append(train(args))
    else:
        for lr in [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3]:
            for emb_size in [8, 16, 32, 64, 128, 256, 512]:
                args.model_name = f"hpsweep_{args.dataset}_{args.tag_src}_{args.model_type}_agg{args.agg}_lr{lr}_es{emb_size}"
                args.lr = lr
                args.emb_size = emb_size
                model_names.append(args.model_name)
                results.append(train(args))
    aucs = np.array([metrics.mean(0)[2] if args.crossval else metrics[2] for metrics in results])
    best_model_idx = aucs.argmax()
    result_str = "\n".join([f"{model_name}: {auc:.2f}" for model_name, auc in zip(model_names, aucs)])
    result_str += f"\nBest: {model_names[best_model_idx]}: {aucs[best_model_idx]:.2f}"
    print(result_str)
    results_dir = get_results_dir(args, "metrics")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f"metrics_hpsweep_{args.dataset}_{args.tag_src}_{args.model_type}_agg{args.agg}.txt"), "w") as file:
        file.write(result_str + "\n")

def crossval(args, fn):
    # Train/test models across folds
    metrics_agg = []
    folds = COMTA_SUBJECTS if args.split_by_subject else range(1, 6)
    for fold in folds:
        print(f"Fold {fold}...")
        metrics = fn(args, fold)
        metrics_agg.append(metrics)
    # Aggregate and report metrics across folds
    metrics_np = np.stack(metrics_agg, axis=0)
    avg = metrics_np.mean(axis=0)
    std = metrics_np.std(axis=0)
    metric_names = ["Loss", "Acc", "AUC", "Prec", "Rec", "F1"]
    if len(avg) > 6:
        metric_names += ["Acc (Final)", "AUC (Final)", "Prec (Final)", "Rec (Final)", "F1 (Final)"]
    results = [
        f"{metric}: ${avg[idx]:.2f}_{{\\pm {std[idx]:.2f}}}$" for idx, metric in
        enumerate(metric_names)
    ]
    result_str = "\n".join(results)
    print(result_str)
    results_dir = get_results_dir(args, "metrics")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f"metrics_crossval{'_subj' if args.split_by_subject else ''}_{get_model_file_suffix(args)}.txt"), "w") as out_file:
        out_file.writelines([
            str(metrics_agg) + "\n",
            result_str + "\n"
        ])
    # Aggregate and save qual analysis files across folds
    if args.model_type == "lmkt":
        dfs = [pd.read_csv(get_qual_result_filename(args, fold)) for fold in folds]
        pd.concat(dfs).to_csv(get_qual_result_filename(args), index=False)
    return metrics_np

def train(args):
    if args.hyperparam_sweep:
        args.hyperparam_sweep = False
        return hyperparam_sweep(args)

    assert args.model_name or args.model_type == "bkt"
    apply_defaults(args)
    fn = train_lmkt if args.model_type == "lmkt" else train_test_bkt if args.model_type == "bkt" else train_baseline
    if args.crossval:
        return crossval(args, fn)
    else:
        return fn(args, get_default_fold(args))

def test(args):
    apply_defaults(args)
    fn = test_lmkt if args.model_type == "lmkt" else test_baseline
    if args.crossval:
        return crossval(args, fn)
    else:
        return fn(args, get_default_fold(args))

def compute_metrics(labels, preds):
    hard_preds = np.round(preds)
    acc = accuracy_score(labels, hard_preds)
    auc = roc_auc_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, hard_preds, average="binary")
    return acc * 100, auc * 100, prec * 100, rec * 100, f1 * 100

def count_parameters(model):
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return trainable, total

def summarize_metric_dicts(metrics_list):
    if not metrics_list:
        return {}
    keys = sorted({key for metrics in metrics_list for key in metrics})
    return {
        key: float(np.mean([metrics[key] for metrics in metrics_list if key in metrics]))
        for key in keys
    }

def format_named_metrics(title, metrics_dict):
    if not metrics_dict:
        return ""
    ordered_keys = [
        "gate_mean",
        "gate_abs_mean",
        "gate_pos_frac",
        "gate_neg_frac",
        "shift_abs_mean",
        "shift_l2_mean",
        "pred_shift_l2_mean",
        "gate_max",
        "gate_min",
        "shift_max",
        "shift_global_max",
        "injected_mean",
        "env_split_frac",
        "env_non_rationale_frac",
        "env_selected_score",
        "env_non_selected_score",
        "rationale_l2_mean",
        "env_l2_mean",
        "env_scaled_l2_mean",
        "env_input_l2_mean",
        "env_to_rationale_l2_ratio",
        "env_rationale_cosine",
        "env_input_cosine",
        "env_shuffle_change_frac",
    ]
    keys = [key for key in ordered_keys if key in metrics_dict]
    keys.extend(key for key in sorted(metrics_dict) if key not in keys)
    def format_value(value):
        value = float(value)
        if value != 0.0 and abs(value) < 1e-4:
            return f"{value:.6e}"
        return f"{value:.4f}"

    metrics_str = ", ".join(f"{key}: {format_value(metrics_dict[key])}" for key in keys)
    return f"{title}: {metrics_str}\n"


def _build_lmkt_optimizer(model, args, calibrator_lr_override=None):
    module_specs = _cel_aux_module_specs(model, args)
    selector = getattr(model, "_cel_selector", None)
    environment = getattr(model, "_cel_environment", None)
    calibrator = getattr(model, "_cel_calibrator", None)
    calibrator_lr = calibrator_lr_override
    if calibrator_lr is None:
        calibrator_lr = getattr(args, "cel_calibrator_lr", None)
    if calibrator is not None:
        module_specs = tuple(
            (name, module, calibrator_lr if name == "calibrator" else learning_rate)
            for name, module, learning_rate in module_specs
        )
    for name, _, learning_rate in module_specs:
        if learning_rate is not None and learning_rate <= 0:
            raise ValueError(f"cel_{name}_lr must be positive when provided")

    custom_param_ids = {
        id(param)
        for _, module, _ in module_specs
        if module is not None
        for param in module.parameters()
    }
    optimizer_params = []
    group_summaries = []
    base_params = [
        param
        for param in model.parameters()
        if param.requires_grad and id(param) not in custom_param_ids
    ]
    if base_params:
        optimizer_params.append({
            "params": base_params,
            "lr": args.lr,
            "weight_decay": args.wd,
        })
        group_summaries.append(("lora", args.lr, sum(param.numel() for param in base_params)))

    for name, module, learning_rate in module_specs:
        if module is None:
            continue
        params = [param for param in module.parameters() if param.requires_grad]
        if not params:
            continue
        active_lr = args.lr if learning_rate is None else learning_rate
        optimizer_params.append({
            "params": params,
            "lr": active_lr,
            "weight_decay": args.wd,
        })
        group_summaries.append((name, active_lr, sum(param.numel() for param in params)))

    if not optimizer_params:
        raise ValueError("No trainable parameters are available for the optimizer")
    print(
        "CEL optimizer groups: "
        + ", ".join(
            f"{name}(lr={learning_rate}, params={parameter_count:,})"
            for name, learning_rate, parameter_count in group_summaries
        )
    )

    if args.optim == "adamw":
        return torch.optim.AdamW(optimizer_params, lr=args.lr, weight_decay=args.wd)
    return transformers.Adafactor(optimizer_params, lr=args.lr, weight_decay=args.wd, relative_step=False)


def _cel_aux_module_specs(model, args):
    """Return every attached CEL module so future modules join joint training.

    A module is opt-in by being attached as ``model._cel_<name>``.  Known
    modules retain their dedicated learning-rate flags; new modules use
    ``--cel_<name>_lr`` when supplied and otherwise the base learning rate.
    """
    known = {
        "selector": getattr(args, "cel_selector_lr", None),
        "environment": getattr(args, "cel_environment_lr", None),
        "calibrator": getattr(args, "cel_calibrator_lr", None),
    }
    specs = []
    seen = set()
    for name, learning_rate in known.items():
        module = getattr(model, f"_cel_{name}", None)
        if isinstance(module, torch.nn.Module):
            specs.append((name, module, learning_rate))
            seen.add(id(module))
    attached_modules = getattr(model, "_modules", {})
    for attr_name, module in attached_modules.items():
        if not attr_name.startswith("_cel_") or not isinstance(module, torch.nn.Module):
            continue
        if id(module) in seen:
            continue
        name = attr_name[len("_cel_"):]
        specs.append((name, module, getattr(args, f"cel_{name}_lr", None)))
        seen.add(id(module))
    return tuple(specs)

def compute_all_metrics(loss, all_labels, all_preds, final_turn_labels, final_turn_preds, args, fold, extra_metrics=None):
    result_str = f"Loss: {loss:.4f}\n"
    result_str += f"Overall ({len(all_labels)} samples):\n"
    result_str += f"GT - True: {sum(all_labels)}, False: {len(all_labels) - sum(all_labels)}; "
    result_str += f"Pred - True: {sum(np.round(all_preds))}, False: {len(all_preds) - sum(np.round(all_preds))}\n"
    all_metrics = compute_metrics(all_labels, all_preds)
    result_str += "Acc: {:.2f}, AUC: {:.2f}, Prec: {:.2f}, Rec: {:.2f}, F1: {:.2f}\n".format(*all_metrics)
    if final_turn_labels is not None:
        result_str += f"Final Turn ({len(final_turn_labels)} samples):\n"
        result_str += f"GT - True: {sum(final_turn_labels)}, False: {len(final_turn_labels) - sum(final_turn_labels)}; "
        result_str += f"Pred - True: {sum(np.round(final_turn_preds))}, False: {len(final_turn_preds) - sum(np.round(final_turn_preds))}\n"
        final_metrics = compute_metrics(final_turn_labels, final_turn_preds)
        result_str += "Acc: {:.2f}, AUC: {:.2f}, Prec: {:.2f}, Rec: {:.2f}, F1: {:.2f}\n".format(*final_metrics)
    else:
        final_metrics = []
    if extra_metrics:
        result_str += format_named_metrics("CEL Diagnostics", extra_metrics)
    print(result_str)
    results_dir = get_results_dir(args, "metrics")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f"metrics_{get_model_file_suffix(args, fold)}.txt"), "w") as out_file:
        out_file.write(result_str)
    return all_metrics, final_metrics

def _prepare_attention_mask(batch, suffix="", model_dtype=None, invert=True):
    attention_mask = batch[f"attention_mask{suffix}"].clone()
    if model_dtype is None:
        model_dtype = attention_mask.dtype
    if invert:
        min_dtype = torch.finfo(model_dtype).min
        attention_mask[attention_mask == 0] = min_dtype
        attention_mask[attention_mask == 1] = 0
    return attention_mask.type(model_dtype)


def _forward_lmkt(model, batch, suffix="", output_hidden_states=False, output_attentions=False):
    invert = batch[f"attention_mask{suffix}"].dim() > 2
    kwargs = {
        "input_ids": batch[f"input_ids{suffix}"],
        "attention_mask": _prepare_attention_mask(batch, suffix, model.dtype, invert=invert),
    }
    if f"position_ids{suffix}" in batch:
        kwargs["position_ids"] = batch[f"position_ids{suffix}"]
    if output_hidden_states:
        kwargs["output_hidden_states"] = True
    if output_attentions:
        kwargs["output_attentions"] = True
    return model(**kwargs)


def get_lmkt_probs_unpacked(model, batch, true_token, false_token, args):
    model_output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    batch_size = model_output.logits.shape[0]
    logits = model_output.logits[torch.arange(batch_size), batch["last_idxs"]]
    logits = torch.stack([logits[:, true_token], logits[:, false_token]], dim=1)
    kc_probs = torch.softmax(logits, dim=1)[torch.arange(batch_size), 0]
    num_kc_counter = 0
    kc_probs_grouped = []
    corr_probs = []
    for num_kcs in batch["num_kcs"]:
        kc_probs_grouped.append(kc_probs[num_kc_counter : num_kc_counter + num_kcs].tolist())
        if args.agg == "prod":
            prob = kc_probs[num_kc_counter : num_kc_counter + num_kcs].prod()
        elif args.agg == "mean-ar":
            prob = kc_probs[num_kc_counter : num_kc_counter + num_kcs].mean()
        else:
            prob = kc_probs[num_kc_counter : num_kc_counter + num_kcs].prod() ** (1 / num_kcs)
        corr_probs.append(prob)
        num_kc_counter += num_kcs
    corr_probs = torch.stack(corr_probs)
    return kc_probs, kc_probs_grouped, corr_probs


def get_lmkt_loss_unpacked(model, batch, true_token, false_token, args):
    kc_probs, kc_probs_grouped, corr_probs = get_lmkt_probs_unpacked(model, batch, true_token, false_token, args)
    loss, corr_probs, _ = _compute_bce_loss_from_probs(corr_probs, batch["labels"])
    return loss, kc_probs_grouped, corr_probs


def _init_cel_selector(model, args):
    hidden_dim = model.config.hidden_size
    selector_dim = getattr(args, "cel_selector_hidden_dim", None) or hidden_dim
    mode = getattr(args, "cel_mode", None)
    variant = getattr(args, "cel_injection_variant", "scalar_gate")
    if mode == "mlp":
        if variant == "vector_shift":
            selector = MLPShiftSelector(hidden_dim=hidden_dim, gate_hidden_dim=selector_dim, dropout=getattr(args, "cel_drop", 0.1))
        else:
            selector = MLPGateSelector(hidden_dim=hidden_dim, gate_hidden_dim=selector_dim, dropout=getattr(args, "cel_drop", 0.1))
    elif mode == "adapter":
        if variant == "vector_shift":
            selector = AdapterShiftSelector(hidden_dim=hidden_dim, adapter_dim=getattr(args, "cel_adapter_dim", None), dropout=getattr(args, "cel_drop", 0.1))
        else:
            selector = AdapterGateSelector(hidden_dim=hidden_dim, adapter_dim=getattr(args, "cel_adapter_dim", None), dropout=getattr(args, "cel_drop", 0.1))
    elif mode == "task_conditioned":
        if variant == "vector_shift":
            selector = TaskConditionedShiftSelector(hidden_dim=hidden_dim, gate_hidden_dim=selector_dim, dropout=getattr(args, "cel_drop", 0.1))
        else:
            selector = TaskConditionedSelector(hidden_dim=hidden_dim, gate_hidden_dim=selector_dim, dropout=getattr(args, "cel_drop", 0.1))
    else:
        raise ValueError(f"Unsupported cel_mode: {mode}")
    # Keep the Stage 2 A module in FP32. The Qwen backbone may run in bf16, but
    # the strict joint phase uses a tiny learning rate whose updates must remain
    # representable in the selector checkpoint. Stage 1 keeps its legacy dtype.
    selector_dtype = torch.float32 if _stage2_enabled(args) else model.dtype
    return selector.to(device=device, dtype=selector_dtype)


def _stage2_enabled(args) -> bool:
    return bool(getattr(args, "cel_stage2_enabled", False))


def _stage2_phase(args) -> str | None:
    return getattr(args, "cel_stage2_phase", None)


def _stage2_uses_dual_path(args) -> bool:
    # A missing phase is the current default: train the complete A+B candidate
    # end to end from epoch 1.  The named warmup phases remain compatibility
    # modes for historical provenance only.
    return _stage2_enabled(args) and _stage2_phase(args) in (None, "b_warmup", "joint")


def _stage2_requires_environment(args) -> bool:
    return _stage2_uses_dual_path(args)


def _stage2_manifest_from_args(args) -> dict:
    phase = getattr(args, "cel_stage2_phase", None)
    return {
        "schema_version": 2,
        "objective": "dual_path_hr_hm_js",
        "training_protocol": (
            "unified_end_to_end"
            if phase is None
            else "historical_or_special_staged_protocol"
        ),
        "max_epochs": int(getattr(args, "max_epochs", getattr(args, "epochs", 0))),
        "early_stopping": {
            "patience": int(getattr(args, "patience", 0) or 0),
            "min_delta": float(getattr(args, "min_delta", 0.0) or 0.0),
            "monitor": "validation_loss",
            "restore_best": True,
        },
        "candidate_id": getattr(args, "cel_stage2_candidate_id", None),
        "phase": phase,
        "parent_model_name": getattr(args, "cel_stage2_parent_model_name", None),
        "model_name": getattr(args, "model_name", None),
        "fresh_init_required": bool(getattr(args, "cel_stage2_fresh_init", False)),
        "base_model": getattr(args, "base_model", None),
        "env_mode": getattr(args, "cel_env_mode", None),
        "env_beta": float(getattr(args, "cel_env_beta", 0.1)),
        "env_split_mode": getattr(args, "cel_env_split_mode", "complementary"),
        "env_topk_ratio": float(getattr(args, "cel_env_topk_ratio", 0.1)),
        "env_sigmoid_temperature": float(getattr(args, "cel_env_sigmoid_temperature", 5.0)),
        "env_hidden_dim": int(getattr(args, "cel_env_hidden_dim", 1024)),
        "env_num_layers": int(getattr(args, "cel_env_num_layers", 4)),
        "env_num_heads": int(getattr(args, "cel_env_num_heads", 8)),
        "env_ffn_dim": getattr(args, "cel_env_ffn_dim", None),
        "env_output_postprocess": getattr(args, "cel_env_output_postprocess", "centered_rms"),
        "env_output_ratio": float(getattr(args, "cel_env_output_ratio", 1.0)),
        "env_output_init_std": float(getattr(args, "cel_env_output_init_std", 0.01)),
        "lambda_r": float(getattr(args, "cel_stage2_lambda_r", 1.0)),
        "lambda_m": float(getattr(args, "cel_stage2_lambda_m", 1.0)),
        "lambda_cons": float(getattr(args, "cel_stage2_lambda_cons", 0.1)),
        "beta_start_ratio": float(getattr(args, "cel_stage2_beta_start_ratio", 0.2)),
        "consistency_ramp_fraction": float(getattr(args, "cel_stage2_consistency_ramp_fraction", 0.25)),
        "selector_lr": getattr(args, "cel_selector_lr", None),
        "environment_lr": getattr(args, "cel_environment_lr", None),
        "calibrator_lr": getattr(args, "cel_calibrator_lr", None),
        "cel_mode": getattr(args, "cel_mode", None),
        "cel_selector_hidden_dim": getattr(args, "cel_selector_hidden_dim", None),
        "cel_drop": float(getattr(args, "cel_drop", 0.1)),
        "cel_layer_idx": int(getattr(args, "cel_layer_idx", -1)),
        "cel_hook_site": getattr(args, "cel_hook_site", "last_block"),
        "cel_hook_timing": getattr(args, "cel_hook_timing", "post_block"),
        "cel_gamma": float(getattr(args, "cel_gamma", 0.3)),
        "cel_injection_variant": getattr(args, "cel_injection_variant", "scalar_gate"),
        "cel_application_mode": getattr(args, "cel_application_mode", "token_residual"),
        "cel_output_calibration": getattr(args, "cel_output_calibration", "none"),
        "cel_calibrator_init_bias": float(getattr(args, "cel_calibrator_init_bias", 0.0)),
        "cel_env_drop": float(getattr(args, "cel_env_drop", 0.1)),
        "cel_env_shuffle_seed": int(getattr(args, "cel_env_shuffle_seed", 221)),
        "model_init_seed": int(getattr(args, "model_init_seed", 221)),
    }


def _load_stage2_manifest(model_name: str) -> dict:
    manifest_path = os.path.join(get_checkpoint_path(model_name), CEL_STAGE2_MANIFEST_FILENAME)
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Stage 2 manifest not found at {manifest_path}")
    with open(manifest_path, encoding="utf-8") as manifest_file:
        return json.load(manifest_file)


def _validate_stage2_args(args, training: bool) -> None:
    if not _stage2_enabled(args):
        return
    if getattr(args, "cel_mode", None) != "task_conditioned":
        raise ValueError("Dual-path Stage 2 requires --cel_mode task_conditioned")
    if getattr(args, "cel_injection_variant", "scalar_gate") != "scalar_gate":
        raise ValueError("Dual-path Stage 2 requires token-scalar a")
    if getattr(args, "cel_application_mode", "token_residual") != "token_residual":
        raise ValueError("Dual-path Stage 2 requires --cel_application_mode token_residual")
    if getattr(args, "cel_hook_site", "last_block") != "last_block":
        raise ValueError("Dual-path Stage 2 requires --cel_hook_site last_block")
    if getattr(args, "cel_hook_timing", "post_block") != "pre_block":
        raise ValueError("Dual-path Stage 2 requires --cel_hook_timing pre_block")
    if getattr(args, "cel_layer_idx", -1) != -1:
        raise ValueError("Dual-path Stage 2 currently branches only before the final Qwen block")
    if getattr(args, "cel_env_mode", None) != "contextual_transformer":
        raise ValueError("Formal dual-path Stage 2 requires --cel_env_mode contextual_transformer")
    if getattr(args, "cel_env_split_mode", "complementary") != "complementary":
        raise ValueError("Dual-path Stage 2 requires the continuous complementary split")
    if float(getattr(args, "cel_env_beta", 0.0)) <= 0:
        raise ValueError("cel_env_beta must be positive")
    output_postprocess = getattr(args, "cel_env_output_postprocess", "none")
    if output_postprocess not in ("none", "centered_rms"):
        raise ValueError(f"Unsupported Stage 2 environment output postprocess: {output_postprocess}")
    if float(getattr(args, "cel_env_output_ratio", 0.0)) <= 0:
        raise ValueError("cel_env_output_ratio must be positive")
    if float(getattr(args, "cel_env_output_init_std", 0.0)) <= 0:
        raise ValueError("cel_env_output_init_std must be positive")
    for name in ("cel_stage2_lambda_r", "cel_stage2_lambda_m", "cel_stage2_lambda_cons"):
        if float(getattr(args, name, 0.0)) < 0:
            raise ValueError(f"{name} must be non-negative")
    if float(getattr(args, "cel_stage2_lambda_m", 0.0)) <= 0:
        raise ValueError("cel_stage2_lambda_m must be positive")
    beta_start_ratio = float(getattr(args, "cel_stage2_beta_start_ratio", 0.2))
    if not 0 < beta_start_ratio <= 1:
        raise ValueError("cel_stage2_beta_start_ratio must be in (0, 1]")
    ramp_fraction = float(getattr(args, "cel_stage2_consistency_ramp_fraction", 0.25))
    if not 0 <= ramp_fraction <= 1:
        raise ValueError("cel_stage2_consistency_ramp_fraction must be in [0, 1]")

    candidate_id = getattr(args, "cel_stage2_candidate_id", None)
    phase = getattr(args, "cel_stage2_phase", None)
    parent_model_name = getattr(args, "cel_stage2_parent_model_name", None)
    if phase is None:
        if training and not candidate_id:
            raise ValueError("Unified Stage 2 training requires --cel_stage2_candidate_id for provenance")
        if training and not getattr(args, "cel_stage2_fresh_init", False):
            raise ValueError(
                "Unified Stage 2 training must start from the raw base model; "
                "set --cel_stage2_fresh_init 1"
            )
        if training and any(
            getattr(args, field, None)
            for field in (
                "pt_model_name",
                "cel_selector_init_model_name",
                "cel_calibrator_init_model_name",
                "cel_environment_init_model_name",
                "cel_stage2_parent_model_name",
            )
        ):
            raise ValueError("Unified Stage 2 training does not load a previous candidate checkpoint")
        special_flags = (
            "cel_train_calibrator_only",
            "cel_train_selector_only",
            "cel_train_selector_and_calibrator_only",
            "cel_train_environment_only",
        )
        if training and any(getattr(args, flag, False) for flag in special_flags):
            raise ValueError(
                "Unified Stage 2 training does not permit frozen or module-only "
                "updates; use an explicitly documented historical/special protocol"
            )
        if training and int(getattr(args, "cel_calibrator_warmup_epochs", 0) or 0) != 0:
            raise ValueError(
                "Unified Stage 2 training does not permit calibrator warmup; "
                "train the optional calibrator jointly from epoch 1"
            )
        return
    if not candidate_id:
        raise ValueError("Named Stage 2 phases require --cel_stage2_candidate_id")
    if phase not in STAGE2_PHASES:
        raise ValueError(f"Unsupported Stage 2 phase: {phase}")
    if not getattr(args, "cel_stage2_fresh_init", False):
        raise ValueError("Formal Stage 2 candidates require --cel_stage2_fresh_init 1")
    if phase != "a_bootstrap" and getattr(args, "cel_output_calibration", "none") == "none":
        raise ValueError(f"Stage 2 {phase} requires a shared output calibrator")
    if phase == "calibrator_warmup" and not getattr(args, "cel_train_calibrator_only", False):
        raise ValueError("Stage 2 calibrator_warmup must train only the shared calibrator")
    if phase == "b_warmup" and not getattr(args, "cel_train_environment_only", False):
        raise ValueError("Stage 2 b_warmup must train only B")
    if phase in ("a_bootstrap", "a_joint", "joint") and any(
        getattr(args, flag, False)
        for flag in (
            "cel_train_calibrator_only",
            "cel_train_selector_only",
            "cel_train_selector_and_calibrator_only",
            "cel_train_environment_only",
        )
    ):
        raise ValueError(f"Stage 2 {phase} does not accept an exclusive trainability flag")

    if not training:
        return
    warm_start_fields = {
        "pt_model_name": getattr(args, "pt_model_name", None),
        "cel_selector_init_model_name": getattr(args, "cel_selector_init_model_name", None),
        "cel_calibrator_init_model_name": getattr(args, "cel_calibrator_init_model_name", None),
        "cel_environment_init_model_name": getattr(args, "cel_environment_init_model_name", None),
    }
    if phase == "a_bootstrap":
        populated = {key: value for key, value in warm_start_fields.items() if value}
        if populated or parent_model_name:
            raise ValueError(f"Stage 2 A bootstrap must start from the raw base model; found {populated}")
        return

    if not parent_model_name:
        raise ValueError(f"Stage 2 {phase} requires --cel_stage2_parent_model_name")
    required_fields = ["pt_model_name", "cel_selector_init_model_name"]
    if phase in ("a_joint", "b_warmup", "joint"):
        required_fields.append("cel_calibrator_init_model_name")
    if phase == "joint":
        required_fields.append("cel_environment_init_model_name")
    mismatches = {
        field: warm_start_fields[field]
        for field in required_fields
        if warm_start_fields[field] != parent_model_name
    }
    if mismatches:
        raise ValueError(
            f"Stage 2 {phase} must load every component from the declared same-candidate parent "
            f"{parent_model_name}; mismatches={mismatches}"
        )
    if phase == "calibrator_warmup" and warm_start_fields["cel_calibrator_init_model_name"]:
        raise ValueError("Stage 2 calibrator warmup must create a new calibrator, not load one")
    if phase in ("calibrator_warmup", "a_joint") and warm_start_fields["cel_environment_init_model_name"]:
        raise ValueError(f"Stage 2 {phase} must not load B before B warmup")
    if phase == "b_warmup" and warm_start_fields["cel_environment_init_model_name"]:
        raise ValueError("Stage 2 B warmup must initialize a fresh B module")

    parent_manifest = _load_stage2_manifest(parent_model_name)
    expected_parent_phase = {
        "calibrator_warmup": "a_bootstrap",
        "a_joint": "calibrator_warmup",
        "b_warmup": "a_joint",
        "joint": "b_warmup",
    }[phase]
    checks = {
        "candidate_id": candidate_id,
        "phase": expected_parent_phase,
        "objective": "dual_path_hr_hm_js",
        "env_mode": getattr(args, "cel_env_mode", None),
        "env_beta": float(getattr(args, "cel_env_beta", 0.1)),
        "env_split_mode": getattr(args, "cel_env_split_mode", "complementary"),
        "env_hidden_dim": int(getattr(args, "cel_env_hidden_dim", 512)),
        "env_num_layers": int(getattr(args, "cel_env_num_layers", 1)),
        "env_num_heads": int(getattr(args, "cel_env_num_heads", 4)),
        "env_ffn_dim": getattr(args, "cel_env_ffn_dim", None),
        "env_output_postprocess": getattr(args, "cel_env_output_postprocess", "none"),
        "env_output_ratio": float(getattr(args, "cel_env_output_ratio", 0.1)),
        "env_output_init_std": float(getattr(args, "cel_env_output_init_std", 0.001)),
        "lambda_r": float(getattr(args, "cel_stage2_lambda_r", 1.0)),
        "lambda_m": float(getattr(args, "cel_stage2_lambda_m", 1.0)),
        "lambda_cons": float(getattr(args, "cel_stage2_lambda_cons", 0.1)),
        "beta_start_ratio": float(getattr(args, "cel_stage2_beta_start_ratio", 0.2)),
        "consistency_ramp_fraction": float(getattr(args, "cel_stage2_consistency_ramp_fraction", 0.25)),
        "cel_mode": getattr(args, "cel_mode", None),
        "cel_selector_hidden_dim": getattr(args, "cel_selector_hidden_dim", None),
        "cel_drop": float(getattr(args, "cel_drop", 0.1)),
        "cel_layer_idx": int(getattr(args, "cel_layer_idx", -1)),
        "cel_hook_site": getattr(args, "cel_hook_site", "last_block"),
        "cel_hook_timing": getattr(args, "cel_hook_timing", "post_block"),
        "cel_gamma": float(getattr(args, "cel_gamma", 0.3)),
        "cel_injection_variant": getattr(args, "cel_injection_variant", "scalar_gate"),
        "cel_application_mode": getattr(args, "cel_application_mode", "token_residual"),
        "cel_env_drop": float(getattr(args, "cel_env_drop", 0.1)),
        "cel_env_shuffle_seed": int(getattr(args, "cel_env_shuffle_seed", 221)),
        "model_init_seed": int(getattr(args, "model_init_seed", 221)),
    }
    manifest_mismatches = {
        key: {"expected": value, "actual": parent_manifest.get(key)}
        for key, value in checks.items()
        if parent_manifest.get(key) != value
    }
    if manifest_mismatches:
        raise ValueError(f"Stage 2 parent manifest mismatch: {manifest_mismatches}")


def _init_cel_environment(model, args):
    if not _stage2_requires_environment(args):
        return None
    environment = build_environment_generator(
        mode=getattr(args, "cel_env_mode", None),
        hidden_dim=model.config.hidden_size,
        env_hidden_dim=getattr(args, "cel_env_hidden_dim", 512),
        num_layers=getattr(args, "cel_env_num_layers", 1),
        num_heads=getattr(args, "cel_env_num_heads", 4),
        ffn_dim=getattr(args, "cel_env_ffn_dim", None),
        dropout=getattr(args, "cel_env_drop", 0.1),
        output_init_std=getattr(args, "cel_env_output_init_std", 0.001),
        shuffle_seed=getattr(args, "cel_env_shuffle_seed", 221),
    )
    # Learned B modules follow the same FP32 rule as the selector.  Shuffle is
    # parameter-free, so this is harmless and keeps the interface uniform.
    return environment.to(device=device, dtype=torch.float32)


def _maybe_load_cel_environment_init(model, args):
    environment = getattr(model, "_cel_environment", None)
    init_model_name = getattr(args, "cel_environment_init_model_name", None)
    if environment is None or not init_model_name:
        return
    env_path = os.path.join(get_checkpoint_path(init_model_name), CEL_ENVIRONMENT_FILENAME)
    if not os.path.isfile(env_path):
        raise FileNotFoundError(f"Stage 2 environment warm-start file not found at {env_path}")
    source_state = torch.load(env_path, map_location="cpu", weights_only=True)
    try:
        environment.load_state_dict(source_state, strict=True)
    except RuntimeError:
        if getattr(args, "cel_require_exact_environment_init", False):
            raise
        environment.load_state_dict(source_state, strict=False)
    print(f"Stage 2 environment warm-start loaded from {env_path}")


def _maybe_load_cel_selector_init(model, args):
    selector = getattr(model, "_cel_selector", None)
    init_model_name = getattr(args, "cel_selector_init_model_name", None)
    if selector is None or not init_model_name:
        return

    cel_path = os.path.join(get_checkpoint_path(init_model_name), "cel_selector.pt")
    if not os.path.exists(cel_path):
        if getattr(args, "cel_require_exact_selector_init", False):
            raise FileNotFoundError(f"CEL selector warm-start file not found at {cel_path}")
        print(f"WARNING: CEL selector warm-start file not found at {cel_path}; using fresh selector init")
        return

    source_state = torch.load(cel_path, map_location="cpu")
    target_state = selector.state_dict()
    matched_state = {}
    skipped_keys = []
    for key, value in source_state.items():
        if key in target_state and target_state[key].shape == value.shape:
            matched_state[key] = value
        else:
            skipped_keys.append(key)

    if not matched_state:
        if getattr(args, "cel_require_exact_selector_init", False):
            raise ValueError(f"CEL selector warm-start from {cel_path} had no compatible tensors")
        print(f"WARNING: CEL selector warm-start from {cel_path} had no compatible tensors; using fresh selector init")
        return

    if getattr(args, "cel_require_exact_selector_init", False):
        source_keys = set(source_state)
        target_keys = set(target_state)
        mismatched_shapes = [
            key
            for key in sorted(source_keys & target_keys)
            if source_state[key].shape != target_state[key].shape
        ]
        if source_keys != target_keys or mismatched_shapes:
            raise ValueError(
                "CEL selector warm-start must exactly match the target selector: "
                f"missing={sorted(target_keys - source_keys)}, "
                f"unexpected={sorted(source_keys - target_keys)}, "
                f"shape_mismatch={mismatched_shapes}"
            )

    merged_state = dict(target_state)
    merged_state.update(matched_state)
    selector.load_state_dict(merged_state)

    matched_params = sum(value.numel() for value in matched_state.values())
    total_params = sum(value.numel() for value in target_state.values())
    print(
        "CEL selector warm-start loaded "
        f"{len(matched_state)}/{len(target_state)} tensors "
        f"({matched_params:,}/{total_params:,} params) from {cel_path}"
    )
    if skipped_keys:
        print(
            "CEL selector warm-start skipped "
            f"{len(skipped_keys)} tensors due to missing keys or shape mismatch"
        )


def _init_cel_calibrator(args):
    mode = getattr(args, "cel_output_calibration", "none")
    init_bias = getattr(args, "cel_calibrator_init_bias", 0.0)
    init_scale = getattr(args, "cel_calibrator_init_scale", 1.0)
    return CELProbabilityCalibrator(mode, init_bias=init_bias, init_scale=init_scale).to(device=device)


def _maybe_load_cel_calibrator(model, model_name: str, required: bool = False):
    calibrator = getattr(model, "_cel_calibrator", None)
    if calibrator is None or calibrator.mode == "none":
        return
    cal_path = os.path.join(get_checkpoint_path(model_name), "cel_calibrator.pt")
    if not os.path.exists(cal_path):
        if required:
            raise FileNotFoundError(f"CEL calibrator not found at {cal_path}")
        print(f"WARNING: CEL calibrator not found at {cal_path}")
        return
    calibrator.load_state_dict(torch.load(cal_path, map_location=device))
    print(f"CEL calibrator loaded from {cal_path}")


def _maybe_load_cel_calibrator_init(model, args):
    calibrator = getattr(model, "_cel_calibrator", None)
    init_model_name = getattr(args, "cel_calibrator_init_model_name", None)
    if calibrator is None or calibrator.mode == "none" or not init_model_name:
        return

    cal_path = os.path.join(get_checkpoint_path(init_model_name), "cel_calibrator.pt")
    if not os.path.exists(cal_path):
        raise FileNotFoundError(f"CEL calibrator warm-start file not found at {cal_path}")
    calibrator.load_state_dict(torch.load(cal_path, map_location=device))
    print(f"CEL calibrator warm-start loaded from {cal_path}")


def _capture_cel_joint_trainable_param_ids(model):
    joint_ids = getattr(model, "_cel_joint_trainable_param_ids", None)
    if joint_ids is None:
        joint_ids = {id(param) for param in model.parameters() if param.requires_grad}
        model._cel_joint_trainable_param_ids = joint_ids
    return joint_ids


def _restore_cel_joint_trainability(model):
    joint_ids = _capture_cel_joint_trainable_param_ids(model)
    for param in model.parameters():
        param.requires_grad = id(param) in joint_ids


def _resolve_cel_train_mode(args, mode_override=None):
    if mode_override is not None:
        return mode_override
    if getattr(args, "cel_train_calibrator_only", False):
        return "calibrator_only"
    if getattr(args, "cel_train_selector_and_calibrator_only", False):
        return "selector_and_calibrator_only"
    if getattr(args, "cel_train_selector_only", False):
        return "selector_only"
    if getattr(args, "cel_train_environment_only", False):
        return "environment_only"
    return "joint"


def _configure_cel_trainability(model, args, mode_override=None):
    selector = getattr(model, "_cel_selector", None)
    calibrator = getattr(model, "_cel_calibrator", None)
    environment = getattr(model, "_cel_environment", None)
    if selector is None:
        return
    mode = _resolve_cel_train_mode(args, mode_override)
    model._cel_active_train_mode = mode
    _restore_cel_joint_trainability(model)
    if mode == "calibrator_only":
        for param in model.parameters():
            param.requires_grad = False
        for param in selector.parameters():
            param.requires_grad = False
        if environment is not None:
            for param in environment.parameters():
                param.requires_grad = False
        if calibrator is not None:
            for param in calibrator.parameters():
                param.requires_grad = True
        trainable, total = count_parameters(model)
        print(f"CEL calibrator-only training enabled: {trainable:,} / {total:,} parameters trainable")
    elif mode == "selector_and_calibrator_only":
        for param in model.parameters():
            param.requires_grad = False
        for param in selector.parameters():
            param.requires_grad = True
        if environment is not None:
            for param in environment.parameters():
                param.requires_grad = False
        if calibrator is not None:
            for param in calibrator.parameters():
                param.requires_grad = True
        trainable, total = count_parameters(model)
        print(f"CEL selector+calibrator-only training enabled: {trainable:,} / {total:,} parameters trainable")
    elif mode == "selector_only":
        for param in model.parameters():
            param.requires_grad = False
        for param in selector.parameters():
            param.requires_grad = True
        if environment is not None:
            for param in environment.parameters():
                param.requires_grad = False
        if calibrator is not None:
            for param in calibrator.parameters():
                param.requires_grad = False
        trainable, total = count_parameters(model)
        print(f"CEL selector-only training enabled: {trainable:,} / {total:,} parameters trainable")
    elif mode == "environment_only":
        if environment is None:
            raise ValueError("CEL environment-only training requires an initialized B module")
        for param in model.parameters():
            param.requires_grad = False
        for param in environment.parameters():
            param.requires_grad = True
        trainable, total = count_parameters(model)
        print(f"CEL B-only training enabled: {trainable:,} / {total:,} parameters trainable")
    else:
        trainable, total = count_parameters(model)
        print(f"CEL joint training enabled: {trainable:,} / {total:,} parameters trainable")


def _cel_can_skip_backbone_grad_path(args) -> bool:
    """Return whether no phase needs gradients through the Qwen prefix.

    B warmup still differentiates through the frozen suffix after ``h_m`` to
    update B, but it does not need gradients into Qwen's input or preceding
    blocks: the trainable mixed representation begins at the final-block hook.
    """
    return bool(
        getattr(args, "cel_train_calibrator_only", False)
        or getattr(args, "cel_train_environment_only", False)
    )


def _configure_cel_frozen_backbone_memory(model, args):
    if not _cel_can_skip_backbone_grad_path(args):
        return

    # These phases do not differentiate through Qwen's prefix, so PEFT input
    # gradients and checkpointing only retain unnecessary activations.
    for module in (model, getattr(model, "base_model", None)):
        if module is None:
            continue
        if hasattr(module, "gradient_checkpointing_disable"):
            module.gradient_checkpointing_disable()
        if hasattr(module, "disable_input_require_grads"):
            module.disable_input_require_grads()


def _configure_cel_module_modes(model, args, mode_override=None):
    selector = getattr(model, "_cel_selector", None)
    calibrator = getattr(model, "_cel_calibrator", None)
    environment = getattr(model, "_cel_environment", None)
    mode = _resolve_cel_train_mode(args, mode_override)
    if mode == "calibrator_only":
        model.eval()
        if selector is not None:
            selector.eval()
        if environment is not None:
            environment.eval()
        if calibrator is not None and calibrator.mode != "none":
            calibrator.train()
    elif mode == "selector_only":
        model.eval()
        if selector is not None:
            selector.train()
        if environment is not None:
            environment.eval()
        if calibrator is not None and calibrator.mode != "none":
            calibrator.eval()
    elif mode == "selector_and_calibrator_only":
        model.eval()
        if selector is not None:
            selector.train()
        if environment is not None:
            environment.eval()
        if calibrator is not None and calibrator.mode != "none":
            calibrator.train()
    elif mode == "environment_only":
        model.eval()
        if selector is not None:
            selector.eval()
        if environment is not None:
            environment.train()
        if calibrator is not None and calibrator.mode != "none":
            calibrator.eval()
    else:
        if selector is not None:
            selector.train()
        if environment is not None:
            environment.train()
        if calibrator is not None and calibrator.mode != "none":
            calibrator.train()


def _validate_cel_calibrator_warmup(args):
    warmup_epochs = int(getattr(args, "cel_calibrator_warmup_epochs", 0) or 0)
    warmup_lr = getattr(args, "cel_calibrator_warmup_lr", None)

    if warmup_epochs < 0:
        raise ValueError("cel_calibrator_warmup_epochs must be >= 0")
    if warmup_lr is not None and warmup_lr <= 0:
        raise ValueError("cel_calibrator_warmup_lr must be positive when provided")
    if warmup_epochs == 0:
        return 0

    if getattr(args, "cel_mode", None) is None:
        raise ValueError("cel_calibrator_warmup_epochs requires CEL mode")
    if getattr(args, "cel_output_calibration", "none") == "none":
        raise ValueError("cel_calibrator_warmup_epochs requires an active CEL output calibrator")
    if any(
        getattr(args, flag, False)
        for flag in (
            "cel_train_calibrator_only",
            "cel_train_selector_only",
            "cel_train_selector_and_calibrator_only",
            "cel_train_environment_only",
        )
    ):
        raise ValueError("cel_calibrator_warmup_epochs is only supported for joint CEL training schedules")
    if warmup_epochs >= args.epochs:
        raise ValueError("cel_calibrator_warmup_epochs must leave at least one later epoch for non-warmup training")
    return warmup_epochs


def _apply_cel_calibrator(model, corr_probs: torch.Tensor) -> torch.Tensor:
    corr_probs, input_prob_stats = _sanitize_probability_tensor(corr_probs)
    calibrator = getattr(model, "_cel_calibrator", None)
    if calibrator is not None:
        return calibrator(corr_probs), input_prob_stats
    return corr_probs, input_prob_stats


def _clear_pending_cel_hooks(model) -> None:
    handles = getattr(model, "_cel_pending_hook_handles", [])
    while handles:
        handles.pop().remove()
    model._cel_pending_hook_handles = []


def _extract_cel_probabilities(logits, batch, true_token, false_token, args):
    batch_size = logits.shape[0]
    last_idxs = batch["last_idxs"]
    if last_idxs.dim() == 1:
        selected_logits = logits[torch.arange(batch_size, device=logits.device), last_idxs]
        binary_logits = torch.stack([
            selected_logits[:, true_token],
            selected_logits[:, false_token],
        ], dim=1)
        kc_probs = torch.softmax(binary_logits, dim=1)[:, 0]
        num_kc_counter = 0
        kc_probs_grouped = []
        corr_probs = []
        for num_kcs in batch["num_kcs"]:
            current = kc_probs[num_kc_counter:num_kc_counter + num_kcs]
            kc_probs_grouped.append(current.tolist())
            if args.agg == "prod":
                prob = current.prod()
            elif args.agg == "mean-ar":
                prob = current.mean()
            else:
                prob = current.prod() ** (1 / num_kcs)
            corr_probs.append(prob)
            num_kc_counter += num_kcs
        return torch.stack(corr_probs), kc_probs_grouped

    selected_logits = logits[
        torch.arange(batch_size, device=logits.device).unsqueeze(1),
        last_idxs,
    ]
    binary_logits = torch.stack([
        selected_logits[:, :, true_token],
        selected_logits[:, :, false_token],
    ], dim=2)
    kc_probs = torch.softmax(binary_logits, dim=2)[:, :, 0]
    kc_probs_grouped = [
        probs[:num_kcs].tolist()
        for probs, num_kcs in zip(kc_probs, batch["num_kcs"])
    ]
    return aggregate_kc_probs(kc_probs, batch, args), kc_probs_grouped


def _bernoulli_js_divergence(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left, _ = _sanitize_probability_tensor(left)
    right, _ = _sanitize_probability_tensor(right)
    left_dist = torch.stack([left, 1.0 - left], dim=-1)
    right_dist = torch.stack([right, 1.0 - right], dim=-1)
    midpoint = 0.5 * (left_dist + right_dist)
    left_kl = (left_dist * (left_dist.log() - midpoint.log())).sum(dim=-1)
    right_kl = (right_dist * (right_dist.log() - midpoint.log())).sum(dim=-1)
    return 0.5 * (left_kl + right_kl).mean()


def _stage2_schedule_values(model, args) -> tuple[float, float]:
    target_beta = float(getattr(args, "cel_env_beta", 0.1))
    target_consistency = float(getattr(args, "cel_stage2_lambda_cons", 0.1))
    if _stage2_phase(args) != "b_warmup":
        return target_beta, target_consistency

    progress = float(getattr(model, "_cel_stage2_training_progress", 1.0))
    progress = max(0.0, min(progress, 1.0))
    beta_start_ratio = float(getattr(args, "cel_stage2_beta_start_ratio", 0.2))
    effective_beta = target_beta * (
        beta_start_ratio + (1.0 - beta_start_ratio) * progress
    )
    ramp_fraction = float(getattr(args, "cel_stage2_consistency_ramp_fraction", 0.25))
    if ramp_fraction == 0:
        consistency_scale = 1.0
    else:
        consistency_scale = min(progress / ramp_fraction, 1.0)
    return effective_beta, target_consistency * consistency_scale


def _stage2_serial_path_required(model, batch) -> bool:
    if not getattr(model, "_cel_serial_joint_backward", False):
        return False
    input_ids = batch.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or input_ids.dim() < 2:
        return True
    rows, sequence_length = input_ids.shape[0], input_ids.shape[1]
    attention_risk = int(rows) * int(sequence_length) * int(sequence_length)
    return attention_risk >= STAGE2_SERIAL_RISK_THRESHOLD


def _cel_forward_dual_path(
    model,
    batch,
    true_token,
    false_token,
    args,
    selector,
    tokenizer=None,
    branch_names=None,
):
    # A previous training batch may have kept its hook alive for checkpoint
    # recomputation.  Remove it before registering the next forward hook.
    _clear_pending_cel_hooks(model)
    environment = getattr(model, "_cel_environment", None)
    if environment is None:
        raise ValueError("Dual-path Stage 2 requires an initialized B module")
    effective_beta, effective_consistency = _stage2_schedule_values(model, args)
    include_reversal = not torch.is_grad_enabled()
    handle, captured = register_cel_dual_path_hook(
        model=model,
        batch=batch,
        selector=selector,
        environment_generator=environment,
        mode=getattr(args, "cel_mode", "task_conditioned"),
        layer_idx=getattr(args, "cel_layer_idx", -1),
        beta=effective_beta,
        tokenizer=tokenizer,
        include_reversal=include_reversal,
        branch_names=branch_names,
        environment_output_postprocess=getattr(args, "cel_env_output_postprocess", "centered_rms"),
        environment_output_ratio=getattr(args, "cel_env_output_ratio", 1.0),
    )
    try:
        model_output = _forward_lmkt(
            model,
            batch,
            output_hidden_states=False,
            output_attentions=False,
        )
    except Exception:
        handle.remove()
        raise

    original_batch_size = batch["input_ids"].shape[0]
    branch_names = captured.get("branch_names", ())
    expected_batch_size = original_batch_size * len(branch_names)
    if model_output.logits.shape[0] != expected_batch_size:
        handle.remove()
        raise ValueError(
            "Dual-path Qwen suffix returned an unexpected batch size: "
            f"expected={expected_batch_size}, actual={model_output.logits.shape[0]}"
        )

    branch_outputs = {}
    for branch_idx, branch_name in enumerate(branch_names):
        start = branch_idx * original_batch_size
        end = start + original_batch_size
        corr_probs, kc_probs_grouped = _extract_cel_probabilities(
            model_output.logits[start:end],
            batch,
            true_token,
            false_token,
            args,
        )
        branch_outputs[branch_name] = {
            "corr_probs": corr_probs,
            "kc_probs_grouped": kc_probs_grouped,
        }
    return (
        branch_outputs,
        model_output,
        captured,
        handle,
        effective_beta,
        effective_consistency,
    )


def _cel_forward(model, batch, true_token, false_token, args, selector, tokenizer=None, track_base_grad: bool = True):
    layer_idx = getattr(args, "cel_layer_idx", -1)
    handle, captured = register_cel_injection_hook(
        model=model,
        batch=batch,
        selector=selector,
        mode=getattr(args, "cel_mode", "mlp"),
        layer_idx=layer_idx,
        hook_site=getattr(args, "cel_hook_site", "last_block"),
        hook_timing=getattr(args, "cel_hook_timing", "post_block"),
        gamma=getattr(args, "cel_gamma", 1.0),
        use_norm=getattr(args, "cel_use_norm", True),
        tokenizer=tokenizer,
        application_mode=getattr(args, "cel_application_mode", "token_residual"),
        environment_generator=getattr(model, "_cel_environment", None),
        environment_beta=getattr(args, "cel_env_beta", 0.1),
        environment_split_mode=getattr(args, "cel_env_split_mode", "topk_abs"),
        environment_topk_ratio=getattr(args, "cel_env_topk_ratio", 0.1),
        environment_sigmoid_temperature=getattr(args, "cel_env_sigmoid_temperature", 5.0),
        environment_output_postprocess=getattr(args, "cel_env_output_postprocess", "none"),
        environment_output_ratio=getattr(args, "cel_env_output_ratio", 0.1),
    )
    hook_active = True
    try:
        if track_base_grad:
            model_output = _forward_lmkt(model, batch, output_hidden_states=False, output_attentions=False)
        else:
            with torch.no_grad():
                model_output = _forward_lmkt(model, batch, output_hidden_states=False, output_attentions=False)
        batch_size = model_output.logits.shape[0]
        last_idxs = batch["last_idxs"]
        if last_idxs.dim() == 1:
            logits = model_output.logits[torch.arange(batch_size), last_idxs]
            logits = torch.stack([logits[:, true_token], logits[:, false_token]], dim=1)
            kc_probs = torch.softmax(logits, dim=1)[torch.arange(batch_size), 0]
            num_kc_counter = 0
            kc_probs_grouped = []
            corr_probs = []
            for num_kcs in batch["num_kcs"]:
                kc_probs_grouped.append(kc_probs[num_kc_counter : num_kc_counter + num_kcs].tolist())
                if args.agg == "prod":
                    prob = kc_probs[num_kc_counter : num_kc_counter + num_kcs].prod()
                elif args.agg == "mean-ar":
                    prob = kc_probs[num_kc_counter : num_kc_counter + num_kcs].mean()
                else:
                    prob = kc_probs[num_kc_counter : num_kc_counter + num_kcs].prod() ** (1 / num_kcs)
                corr_probs.append(prob)
                num_kc_counter += num_kcs
            corr_probs = torch.stack(corr_probs)
        else:
            logits = model_output.logits[torch.arange(batch_size).unsqueeze(1), last_idxs]
            logits = torch.stack([logits[:, :, true_token], logits[:, :, false_token]], dim=2)
            kc_probs = torch.softmax(logits, dim=2)[:, :, 0]
            kc_probs_grouped = [probs[:num_kcs].tolist() for probs, num_kcs in zip(kc_probs, batch["num_kcs"])]
            corr_probs = aggregate_kc_probs(kc_probs, batch, args)
    except Exception:
        # The training loop owns the handle after a successful forward.  On a
        # forward failure, remove it here so the next batch cannot accumulate
        # stale CEL hooks.
        handle.remove()
        hook_active = False
        raise
    return corr_probs, kc_probs_grouped, model_output, captured, handle if hook_active else None


def _get_lmkt_loss_cel_dual_path(
    model,
    batch,
    true_token,
    false_token,
    args,
    selector,
    tokenizer=None,
):
    active_train_mode = getattr(model, "_cel_active_train_mode", _resolve_cel_train_mode(args))
    if (
        torch.is_grad_enabled()
        and active_train_mode != "environment_only"
        and _stage2_serial_path_required(model, batch)
    ):
        return _get_lmkt_loss_cel_dual_path_serial(
            model,
            batch,
            true_token,
            false_token,
            args,
            selector,
            tokenizer=tokenizer,
        )
    environment_only_training = (
        torch.is_grad_enabled() and active_train_mode == "environment_only"
    )
    if environment_only_training:
        # During B warmup, A, LoRA, and the calibrator are frozen. The
        # evidence path consequently has no trainable dependency, so compute
        # it without a graph and reserve the suffix backward graph for the
        # mixed path. This preserves the B gradient of L_m + L_cons while
        # avoiding a 2x long-sequence suffix batch.
        evidence_handle = None
        evidence_model_output = None
        try:
            with torch.no_grad():
                (
                    evidence_outputs,
                    evidence_model_output,
                    _,
                    evidence_handle,
                    _,
                    _,
                ) = _cel_forward_dual_path(
                    model,
                    batch,
                    true_token,
                    false_token,
                    args,
                    selector,
                    tokenizer=tokenizer,
                    branch_names=("evidence",),
                )
        finally:
            if evidence_handle is not None:
                evidence_handle.remove()
            if evidence_model_output is not None:
                del evidence_model_output

        (
            mixed_outputs,
            model_output,
            captured,
            hook_handle,
            effective_beta,
            effective_consistency,
        ) = _cel_forward_dual_path(
            model,
            batch,
            true_token,
            false_token,
            args,
            selector,
            tokenizer=tokenizer,
            branch_names=("mixed",),
        )
        branch_outputs = {**evidence_outputs, **mixed_outputs}
    else:
        (
            branch_outputs,
            model_output,
            captured,
            hook_handle,
            effective_beta,
            effective_consistency,
        ) = _cel_forward_dual_path(
            model,
            batch,
            true_token,
            false_token,
            args,
            selector,
            tokenizer=tokenizer,
        )
    defer_hook_cleanup = bool(torch.is_grad_enabled())
    try:
        calibrated = {}
        calibrator_stats = {}
        for branch_name, output in branch_outputs.items():
            calibrated_probs, pre_stats = _apply_cel_calibrator(
                model,
                output["corr_probs"],
            )
            calibrated[branch_name] = calibrated_probs
            calibrator_stats[branch_name] = pre_stats

        loss_r, p_r, post_r_stats = _compute_bce_loss_from_probs(
            calibrated["evidence"],
            batch["labels"],
        )
        loss_m, p_m, post_m_stats = _compute_bce_loss_from_probs(
            calibrated["mixed"],
            batch["labels"],
        )
        consistency_loss = _bernoulli_js_divergence(p_r, p_m)
        lambda_r = (
            0.0
            if _stage2_phase(args) == "b_warmup"
            else float(getattr(args, "cel_stage2_lambda_r", 1.0))
        )
        lambda_m = float(getattr(args, "cel_stage2_lambda_m", 1.0))
        total_loss = (
            lambda_r * loss_r
            + lambda_m * loss_m
            + effective_consistency * consistency_loss
        )

        p_n = None
        if "non_evidence" in calibrated:
            p_n, _ = _sanitize_probability_tensor(calibrated["non_evidence"])

        probability_gap = (p_r - p_m).abs()
        logit_gap = (
            torch.logit(p_r.clamp(PROB_EPS, 1.0 - PROB_EPS))
            - torch.logit(p_m.clamp(PROB_EPS, 1.0 - PROB_EPS))
        ).abs()
        metrics = {
            "stage2_loss_r": float(loss_r.detach().item()),
            "stage2_loss_m": float(loss_m.detach().item()),
            "stage2_loss_cons": float(consistency_loss.detach().item()),
            "stage2_lambda_r_effective": lambda_r,
            "stage2_lambda_m_effective": lambda_m,
            "stage2_lambda_cons_effective": effective_consistency,
            "stage2_effective_beta": effective_beta,
            "p_r_mean": float(p_r.detach().mean().item()),
            "p_m_mean": float(p_m.detach().mean().item()),
            "p_r_p_m_abs_mean": float(probability_gap.detach().mean().item()),
            "p_r_p_m_abs_max": float(probability_gap.detach().max().item()),
            "z_r_z_m_abs_mean": float(logit_gap.detach().mean().item()),
        }
        if p_n is not None:
            metrics["p_n_mean"] = float(p_n.detach().mean().item())
        metrics.update({
            key: float(value.item()) if isinstance(value, torch.Tensor) else float(value)
            for key, value in captured.get("stage2_stats", {}).items()
        })
        metrics.update({
            key: float(value.item()) if isinstance(value, torch.Tensor) else float(value)
            for key, value in _prefix_metric_keys(
                "p_r_pre_prob",
                calibrator_stats["evidence"],
            ).items()
        })
        metrics.update({
            key: float(value.item()) if isinstance(value, torch.Tensor) else float(value)
            for key, value in _prefix_metric_keys("p_r_post_prob", post_r_stats).items()
        })
        metrics.update({
            key: float(value.item()) if isinstance(value, torch.Tensor) else float(value)
            for key, value in _prefix_metric_keys(
                "p_m_pre_prob",
                calibrator_stats["mixed"],
            ).items()
        })
        metrics.update({
            key: float(value.item()) if isinstance(value, torch.Tensor) else float(value)
            for key, value in _prefix_metric_keys("p_m_post_prob", post_m_stats).items()
        })
        calibrator = getattr(model, "_cel_calibrator", None)
        if calibrator is not None and calibrator.mode != "none":
            if calibrator.logit_bias is not None:
                metrics["cal_bias"] = float(calibrator.logit_bias.detach().item())
            if calibrator.logit_scale_raw is not None:
                metrics["cal_scale"] = float((
                    torch.nn.functional.softplus(calibrator.logit_scale_raw.detach()) + 1e-4
                ).item())
        model._cel_metrics = metrics
        model._cel_last_outputs = {
            "evidence": p_r.detach(),
            "mixed": p_m.detach(),
            "non_evidence": p_n.detach() if p_n is not None else None,
        }

        if defer_hook_cleanup:
            model._cel_pending_hook_handles = [hook_handle]
        else:
            hook_handle.remove()
        return total_loss, branch_outputs["mixed"]["kc_probs_grouped"], p_m
    except Exception:
        hook_handle.remove()
        raise
    finally:
        del model_output


def _get_lmkt_loss_cel_dual_path_serial(
    model,
    batch,
    true_token,
    false_token,
    args,
    selector,
    tokenizer=None,
):
    """Train the two prediction paths without retaining a doubled suffix graph.

    The mixed path receives the partial derivative of ``L_m + L_cons`` while
    the evidence path is recomputed and receives ``L_r + L_cons``.  Since the
    two path graphs are independent branches of the same parameterized model,
    summing these partial derivatives is exactly the gradient of the summed
    objective.  Recomputing the evidence branch keeps peak memory bounded by a
    single long-sequence Qwen suffix.
    """
    model._cel_serial_batch_count = getattr(model, "_cel_serial_batch_count", 0) + 1
    effective_beta, effective_consistency = _stage2_schedule_values(model, args)
    lambda_r = float(getattr(args, "cel_stage2_lambda_r", 1.0))
    lambda_m = float(getattr(args, "cel_stage2_lambda_m", 1.0))
    backward_scale = float(getattr(model, "_cel_serial_backward_scale", 1.0))

    # First obtain a detached evidence target for the mixed-path consistency
    # derivative.  B is not needed for this pass.
    evidence_target_outputs = None
    evidence_target_model_output = None
    evidence_target_handle = None
    try:
        with torch.no_grad():
            (
                evidence_target_outputs,
                evidence_target_model_output,
                _,
                evidence_target_handle,
                _,
                _,
            ) = _cel_forward_dual_path(
                model,
                batch,
                true_token,
                false_token,
                args,
                selector,
                tokenizer=tokenizer,
                branch_names=("evidence",),
            )
            evidence_target_probs, _ = _apply_cel_calibrator(
                model,
                evidence_target_outputs["evidence"]["corr_probs"],
            )
            evidence_target_probs = evidence_target_probs.detach()
    finally:
        if evidence_target_handle is not None:
            evidence_target_handle.remove()
        if evidence_target_model_output is not None:
            del evidence_target_model_output
        if evidence_target_outputs is not None:
            del evidence_target_outputs

    mixed_outputs = None
    mixed_handle = None
    evidence_handle = None
    mixed_model_output = None
    evidence_model_output = None
    try:
        (
            mixed_outputs,
            mixed_model_output,
            mixed_captured,
            mixed_handle,
            mixed_beta,
            mixed_consistency,
        ) = _cel_forward_dual_path(
            model,
            batch,
            true_token,
            false_token,
            args,
            selector,
            tokenizer=tokenizer,
            branch_names=("mixed",),
        )
        mixed_probs, mixed_pre_stats = _apply_cel_calibrator(
            model,
            mixed_outputs["mixed"]["corr_probs"],
        )
        loss_m, mixed_probs, mixed_post_stats = _compute_bce_loss_from_probs(
            mixed_probs,
            batch["labels"],
        )
        mixed_consistency_loss = _bernoulli_js_divergence(
            evidence_target_probs,
            mixed_probs,
        )
        mixed_objective = (
            lambda_m * loss_m + mixed_consistency * mixed_consistency_loss
        )
        (backward_scale * mixed_objective).backward()
        mixed_value = float(loss_m.detach().item())
        mixed_consistency_value = float(mixed_consistency_loss.detach().item())
        mixed_probs_target = mixed_probs.detach()
        mixed_kc_probs_grouped = mixed_outputs["mixed"]["kc_probs_grouped"]
    finally:
        if mixed_handle is not None:
            mixed_handle.remove()
        if mixed_model_output is not None:
            del mixed_model_output
        if mixed_outputs is not None:
            del mixed_outputs

    # Recompute the evidence branch with gradients so the consistency term
    # contributes its evidence-side partial derivative as well.
    evidence_outputs = None
    try:
        (
            evidence_outputs,
            evidence_model_output,
            evidence_captured,
            evidence_handle,
            evidence_beta,
            evidence_consistency,
        ) = _cel_forward_dual_path(
            model,
            batch,
            true_token,
            false_token,
            args,
            selector,
            tokenizer=tokenizer,
            branch_names=("evidence",),
        )
        evidence_probs, evidence_pre_stats = _apply_cel_calibrator(
            model,
            evidence_outputs["evidence"]["corr_probs"],
        )
        loss_r, evidence_probs, evidence_post_stats = _compute_bce_loss_from_probs(
            evidence_probs,
            batch["labels"],
        )
        evidence_consistency_loss = _bernoulli_js_divergence(
            evidence_probs,
            mixed_probs_target,
        )
        evidence_objective = (
            lambda_r * loss_r + evidence_consistency * evidence_consistency_loss
        )
        (backward_scale * evidence_objective).backward()
        evidence_value = float(loss_r.detach().item())
        evidence_consistency_value = float(evidence_consistency_loss.detach().item())
        evidence_probs_target = evidence_probs.detach()
    finally:
        if evidence_handle is not None:
            evidence_handle.remove()
        if evidence_model_output is not None:
            del evidence_model_output
        if evidence_outputs is not None:
            del evidence_outputs

    # The two branch schedules are identical; retain a defensive check so a
    # future schedule change cannot silently produce asymmetric objectives.
    if abs(mixed_beta - effective_beta) > 1e-12 or abs(evidence_beta - effective_beta) > 1e-12:
        raise RuntimeError("serial dual-path branches used different beta schedules")
    if abs(mixed_consistency - effective_consistency) > 1e-12 or abs(
        evidence_consistency - effective_consistency
    ) > 1e-12:
        raise RuntimeError("serial dual-path branches used different consistency schedules")

    probability_gap = (evidence_probs_target - mixed_probs_target).abs()
    logit_gap = (
        torch.logit(evidence_probs_target.clamp(PROB_EPS, 1.0 - PROB_EPS))
        - torch.logit(mixed_probs_target.clamp(PROB_EPS, 1.0 - PROB_EPS))
    ).abs()
    metrics = {
        "stage2_loss_r": evidence_value,
        "stage2_loss_m": mixed_value,
        "stage2_loss_cons": 0.5 * (evidence_consistency_value + mixed_consistency_value),
        "stage2_lambda_r_effective": lambda_r,
        "stage2_lambda_m_effective": lambda_m,
        "stage2_lambda_cons_effective": effective_consistency,
        "stage2_effective_beta": effective_beta,
        "p_r_mean": float(evidence_probs_target.mean().item()),
        "p_m_mean": float(mixed_probs_target.mean().item()),
        "p_r_p_m_abs_mean": float(probability_gap.mean().item()),
        "p_r_p_m_abs_max": float(probability_gap.max().item()),
        "z_r_z_m_abs_mean": float(logit_gap.mean().item()),
    }
    metrics.update({
        key: float(value.item()) if isinstance(value, torch.Tensor) else float(value)
        for key, value in mixed_captured.get("stage2_stats", {}).items()
    })
    metrics.update({
        key: float(value.item()) if isinstance(value, torch.Tensor) else float(value)
        for key, value in _prefix_metric_keys("p_r_pre_prob", evidence_pre_stats).items()
    })
    metrics.update({
        key: float(value.item()) if isinstance(value, torch.Tensor) else float(value)
        for key, value in _prefix_metric_keys("p_r_post_prob", evidence_post_stats).items()
    })
    metrics.update({
        key: float(value.item()) if isinstance(value, torch.Tensor) else float(value)
        for key, value in _prefix_metric_keys("p_m_pre_prob", mixed_pre_stats).items()
    })
    metrics.update({
        key: float(value.item()) if isinstance(value, torch.Tensor) else float(value)
        for key, value in _prefix_metric_keys("p_m_post_prob", mixed_post_stats).items()
    })
    calibrator = getattr(model, "_cel_calibrator", None)
    if calibrator is not None and calibrator.mode != "none":
        if calibrator.logit_bias is not None:
            metrics["cal_bias"] = float(calibrator.logit_bias.detach().item())
        if calibrator.logit_scale_raw is not None:
            metrics["cal_scale"] = float((
                torch.nn.functional.softplus(calibrator.logit_scale_raw.detach()) + 1e-4
            ).item())
    model._cel_metrics = metrics
    model._cel_last_outputs = {
        "evidence": evidence_probs_target.detach(),
        "mixed": mixed_probs_target.detach(),
        "non_evidence": None,
    }
    model._cel_loss_backward_done = True
    total_value = (
        lambda_r * evidence_value
        + lambda_m * mixed_value
        + effective_consistency * metrics["stage2_loss_cons"]
    )
    detached_loss = evidence_probs_target.new_tensor(total_value).detach()
    return detached_loss, mixed_kc_probs_grouped, mixed_probs_target.detach()


def get_lmkt_loss_cel(model, batch, true_token, false_token, args, tokenizer=None):
    selector = getattr(model, "_cel_selector", None)
    if selector is None:
        raise ValueError("CEL selector is not initialized")
    if _stage2_uses_dual_path(args):
        return _get_lmkt_loss_cel_dual_path(
            model,
            batch,
            true_token,
            false_token,
            args,
            selector,
            tokenizer=tokenizer,
        )
    active_train_mode = getattr(model, "_cel_active_train_mode", _resolve_cel_train_mode(args))
    track_base_grad = active_train_mode != "calibrator_only"
    corr_probs, kc_probs_grouped, model_output, captured, hook_handle = _cel_forward(
        model,
        batch,
        true_token,
        false_token,
        args,
        selector,
        tokenizer=tokenizer,
        track_base_grad=track_base_grad,
    )
    defer_hook_cleanup = bool(track_base_grad and torch.is_grad_enabled())
    try:
        if track_base_grad:
            corr_probs, pre_prob_stats = _apply_cel_calibrator(model, corr_probs)
        else:
            corr_probs, pre_prob_stats = _apply_cel_calibrator(model, corr_probs.detach())
        loss, corr_probs, post_prob_stats = _compute_bce_loss_from_probs(corr_probs, batch["labels"])
        gate = captured.get("gate")
        gate_stats = captured.get("gate_stats", {})
        if gate is not None:
            metrics = {
                key: float(val.item()) if isinstance(val, torch.Tensor) else float(val)
                for key, val in gate_stats.items()
            }
            metrics.update({
                key: float(val.item()) if isinstance(val, torch.Tensor) else float(val)
                for key, val in _prefix_metric_keys("pre_prob", pre_prob_stats).items()
            })
            metrics.update({
                key: float(val.item()) if isinstance(val, torch.Tensor) else float(val)
                for key, val in _prefix_metric_keys("post_prob", post_prob_stats).items()
            })
            if gate.dim() == 2:
                metrics.update({
                    "gate_max": gate.max().item(),
                    "gate_min": gate.min().item(),
                })
            else:
                metrics.update({
                    "shift_global_max": gate.abs().max().item(),
                })
            metrics["injected_mean"] = float(captured.get("injected_mean", torch.tensor(0.0)).item())
            metrics.update({
                key: float(val.item()) if isinstance(val, torch.Tensor) else float(val)
                for key, val in captured.get("stage2_stats", {}).items()
            })
            calibrator = getattr(model, "_cel_calibrator", None)
            if calibrator is not None and calibrator.mode != "none":
                if calibrator.logit_bias is not None:
                    metrics["cal_bias"] = float(calibrator.logit_bias.detach().item())
                if calibrator.logit_scale_raw is not None:
                    metrics["cal_scale"] = float((torch.nn.functional.softplus(calibrator.logit_scale_raw.detach()) + 1e-4).item())
            model._cel_metrics = metrics
        if defer_hook_cleanup:
            model._cel_pending_hook_handles = [hook_handle]
        else:
            hook_handle.remove()
        return loss, kc_probs_grouped, corr_probs
    except Exception:
        hook_handle.remove()
        raise
    finally:
        del model_output

def get_lmkt_probs_packed(model, batch, true_token, false_token, args, suffix=""):
    # Invert attention mask
    attention_mask = batch[f"attention_mask{suffix}"]
    min_dtype = torch.finfo(model.dtype).min
    attention_mask[attention_mask == 0] = min_dtype
    attention_mask[attention_mask == 1] = 0
    attention_mask = attention_mask.type(model.dtype)
    # Get logits at last token of each sequence
    model_output = model(input_ids=batch[f"input_ids{suffix}"], attention_mask=attention_mask, position_ids=batch[f"position_ids{suffix}"])
    batch_size = model_output.logits.shape[0]
    logits = model_output.logits[torch.arange(batch_size).unsqueeze(1), batch[f"last_idxs{suffix}"]]
    # Return probability of True token over False token for each sequence
    logits = torch.stack([logits[:, :, true_token], logits[:, :, false_token]], dim=2)
    kc_probs = torch.softmax(logits, dim=2)[:, :, 0]
    kc_probs_grouped = [probs[:num_kcs].tolist() for probs, num_kcs in zip(kc_probs, batch["num_kcs"])] # 如果有多个batch，那么取出每个batch对应的kc个数得分
    return kc_probs, kc_probs_grouped


def aggregate_kc_probs(kc_probs, batch, args, suffix=""):
    padding_val = 0 if args.agg == "mean-ar" else 1
    kc_probs = torch.masked_scatter(kc_probs, batch[f"last_idxs{suffix}"].to(device) == 0, torch.full_like(kc_probs, padding_val).to(device))
    # 连乘/ 算术平均/ 几何平均
    if args.agg == "prod":
        corr_probs = kc_probs.prod(dim=1)
    elif args.agg == "mean-ar":
        corr_probs = kc_probs.sum(dim=1) / batch["num_kcs"]
    elif args.agg == "mean-geo":
        corr_probs = kc_probs.prod(dim=1) ** (1 / batch["num_kcs"])
    return corr_probs


def get_lmkt_loss_packed(model, batch, true_token, false_token, args):
    kc_probs, kc_probs_grouped = get_lmkt_probs_packed(model, batch, true_token, false_token, args)
    corr_probs = aggregate_kc_probs(kc_probs, batch, args)
    loss, corr_probs, _ = _compute_bce_loss_from_probs(corr_probs, batch["labels"])
    return loss, kc_probs_grouped, corr_probs


def train_lmkt(args, fold):
    cel_mode = getattr(args, "cel_mode", None)
    _validate_stage2_args(args, training=True)
    initialize_seeds(int(getattr(args, "model_init_seed", 221)))
    model, tokenizer = get_model(
        args.base_model,
        False,
        pt_model_name=args.pt_model_name,
        r=args.r,
        lora_alpha=args.lora_alpha,
        quantize=args.quantize,
        use_gradient_checkpointing=not _cel_can_skip_backbone_grad_path(args),
        eager_attention=False,
    )
    model.print_trainable_parameters()

    effective_pack_kcs = args.pack_kcs and cel_mode != "task_conditioned"
    if cel_mode == "task_conditioned" and args.pack_kcs:
        print("CEL task_conditioned: forcing unpacked prompts to preserve KC-specific gating")

    KTDataset = LMKTDatasetPacked if effective_pack_kcs else LMKTDatasetUnpacked
    KTCollator = LMKTCollatorPacked if effective_pack_kcs else LMKTCollatorUnpacked
    get_loss = get_lmkt_loss_packed if effective_pack_kcs else get_lmkt_loss_unpacked
    if cel_mode:
        model._cel_selector = _init_cel_selector(model, args)
        model._cel_calibrator = _init_cel_calibrator(args)
        model._cel_environment = _init_cel_environment(model, args)
        model._cel_metrics = {}
        model._cel_last_outputs = {}
        _maybe_load_cel_selector_init(model, args)
        _maybe_load_cel_calibrator_init(model, args)
        _maybe_load_cel_environment_init(model, args)
        _capture_cel_joint_trainable_param_ids(model)
        warmup_epochs = _validate_cel_calibrator_warmup(args)
        initial_cel_mode = "calibrator_only" if warmup_epochs > 0 else None
        _configure_cel_trainability(model, args, mode_override=initial_cel_mode)
        _configure_cel_frozen_backbone_memory(model, args)
        model._cel_serial_joint_backward = bool(
            _stage2_uses_dual_path(args)
            and _resolve_cel_train_mode(args) == "joint"
        )
        model._cel_serial_batch_count = 0
        model._cel_loss_backward_done = False
        get_loss = lambda model, batch, true_token, false_token, args: get_lmkt_loss_cel(model, batch, true_token, false_token, args, tokenizer=tokenizer)
        print(
            f"CEL enabled: mode={cel_mode}, hook_site={getattr(args, 'cel_hook_site', 'last_block')}, "
            f"layer_idx={getattr(args, 'cel_layer_idx', -1)}, gamma={getattr(args, 'cel_gamma', 1.0)}"
        )
        print(f"CEL application mode: {getattr(args, 'cel_application_mode', 'token_residual')}")
        print(f"CEL output calibration: {getattr(args, 'cel_output_calibration', 'none')}")
        if _stage2_enabled(args):
            print(
                "CEL Stage 2 enabled: "
                f"candidate={getattr(args, 'cel_stage2_candidate_id', None)}, "
                f"phase={getattr(args, 'cel_stage2_phase', None)}, "
                f"env_mode={getattr(args, 'cel_env_mode', None)}, "
                f"split={getattr(args, 'cel_env_split_mode', 'complementary')}, "
                f"beta={getattr(args, 'cel_env_beta', 0.1)}, "
                f"lambda_r={getattr(args, 'cel_stage2_lambda_r', 1.0)}, "
                f"lambda_m={getattr(args, 'cel_stage2_lambda_m', 1.0)}, "
                f"lambda_cons={getattr(args, 'cel_stage2_lambda_cons', 0.1)}, "
                f"output_postprocess={getattr(args, 'cel_env_output_postprocess', 'none')}, "
                f"output_ratio={getattr(args, 'cel_env_output_ratio', 0.1)}"
            )
        if warmup_epochs > 0:
            warmup_lr = getattr(args, "cel_calibrator_warmup_lr", None)
            lr_text = warmup_lr if warmup_lr is not None else getattr(args, "cel_calibrator_lr", None) or args.lr
            print(
                "CEL calibrator warmup enabled: "
                f"warmup_epochs={warmup_epochs}, warmup_lr={lr_text}, "
                f"joint_lr={args.lr}, joint_calibrator_lr={getattr(args, 'cel_calibrator_lr', None) or args.lr}"
            )
    else:
        warmup_epochs = 0

    train_df, val_df, _ = load_annotated_data(args, fold)
    if args.debug:
        train_df = train_df[:100]
        val_df = val_df[:25]
        print(train_df.iloc[0])
        print(val_df.iloc[0])

    train_dataset = KTDataset(train_df, tokenizer, args)
    val_dataset = KTDataset(val_df, tokenizer, args)
    collator = KTCollator(tokenizer)
    train_dataloader = get_dataloader(train_dataset, collator, args.batch_size, True)
    val_dataloader = get_dataloader(val_dataset, collator, args.batch_size, False)

    # For finding logits for loss
    true_token, false_token = get_true_false_tokens(tokenizer)

    # Do training loop
    warmup_lr_override = None
    if cel_mode and warmup_epochs > 0:
        warmup_lr_override = getattr(args, "cel_calibrator_warmup_lr", None)
    optimizer = _build_lmkt_optimizer(model, args, calibrator_lr_override=warmup_lr_override)
    best_val_loss = None
    stale_epochs = 0
    for epoch in range(args.epochs):
        if cel_mode and warmup_epochs > 0 and epoch == warmup_epochs:
            print("Switching from CEL calibrator warmup to the configured joint CEL train mode")
            _configure_cel_trainability(model, args)
            optimizer = _build_lmkt_optimizer(model, args)

        in_warmup_epoch = bool(cel_mode and warmup_epochs > 0 and epoch < warmup_epochs)
        if in_warmup_epoch:
            print(f"Epoch {epoch + 1} (CEL calibrator warmup)")
        else:
            print(f"Epoch {epoch + 1}")
        total_train_loss = 0
        total_val_loss = 0
        train_cel_metrics = []
        val_cel_metrics = []

        model.train()
        if cel_mode:
            epoch_mode_override = "calibrator_only" if in_warmup_epoch else None
            _configure_cel_module_modes(model, args, mode_override=epoch_mode_override)
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            if cel_mode:
                _clear_pending_cel_hooks(model)
                model._cel_loss_backward_done = False
                if getattr(model, "_cel_serial_joint_backward", False):
                    model._cel_serial_backward_scale = 1.0 / args.grad_accum_steps
            if _stage2_uses_dual_path(args):
                total_steps = max(args.epochs * len(train_dataloader) - 1, 1)
                current_step = epoch * len(train_dataloader) + batch_idx
                model._cel_stage2_training_progress = current_step / total_steps
            try:
                loss, _, _ = get_loss(model, batch, true_token, false_token, args)
                total_train_loss += loss.item()
                if cel_mode and getattr(model, "_cel_metrics", None):
                    train_cel_metrics.append(dict(model._cel_metrics))
                if not getattr(model, "_cel_loss_backward_done", False):
                    loss = loss / args.grad_accum_steps
                    loss.backward()
            finally:
                if cel_mode:
                    _clear_pending_cel_hooks(model)
            if (batch_idx + 1) % args.grad_accum_steps == 0 or batch_idx == len(train_dataloader) - 1:
                if args.gc:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.gc)
                optimizer.step()
                optimizer.zero_grad()

        with torch.no_grad():
            model.eval()
            if _stage2_uses_dual_path(args):
                model._cel_stage2_training_progress = 1.0
            for batch in tqdm(val_dataloader, desc="Validating"):
                loss, _, _ = get_loss(model, batch, true_token, false_token, args)
                total_val_loss += loss.item()
                if cel_mode and getattr(model, "_cel_metrics", None):
                    val_cel_metrics.append(dict(model._cel_metrics))

        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        if cel_mode:
            train_cel_summary = summarize_metric_dicts(train_cel_metrics)
            val_cel_summary = summarize_metric_dicts(val_cel_metrics)
            if getattr(model, "_cel_serial_joint_backward", False):
                print(
                    "CEL serial high-risk batches: "
                    f"{getattr(model, '_cel_serial_batch_count', 0)}"
                )
            if train_cel_summary:
                print(format_named_metrics("CEL Train", train_cel_summary).strip())
            if val_cel_summary:
                print(format_named_metrics("CEL Val", val_cel_summary).strip())
        best_val_loss, stale_epochs, should_stop, improved = _early_stopping_update(
            best_val_loss,
            stale_epochs,
            avg_val_loss,
            args,
        )
        if improved:
            print("Best! Saving model...")
            model_name = args.model_name + (f"_{fold}" if fold else "")
            model.save_pretrained(get_checkpoint_path(model_name))
            if hasattr(model, "_cel_selector") and model._cel_selector is not None:
                cel_path = os.path.join(get_checkpoint_path(model_name), CEL_SELECTOR_FILENAME)
                torch.save(model._cel_selector.state_dict(), cel_path)
                print(f"CEL selector saved to {cel_path}")
            if hasattr(model, "_cel_calibrator") and model._cel_calibrator is not None and model._cel_calibrator.mode != "none":
                cal_path = os.path.join(get_checkpoint_path(model_name), CEL_CALIBRATOR_FILENAME)
                torch.save(model._cel_calibrator.state_dict(), cal_path)
                print(f"CEL calibrator saved to {cal_path}")
            if hasattr(model, "_cel_environment") and model._cel_environment is not None:
                env_path = os.path.join(get_checkpoint_path(model_name), CEL_ENVIRONMENT_FILENAME)
                torch.save(model._cel_environment.state_dict(), env_path)
                print(f"Stage 2 environment saved to {env_path}")
            if _stage2_enabled(args):
                manifest_path = os.path.join(get_checkpoint_path(model_name), CEL_STAGE2_MANIFEST_FILENAME)
                with open(manifest_path, "w", encoding="utf-8") as manifest_file:
                    json.dump(_stage2_manifest_from_args(args), manifest_file, indent=2, sort_keys=True)
                    manifest_file.write("\n")
                print(f"Stage 2 manifest saved to {manifest_path}")
        elif getattr(args, "patience", 0):
            print(
                f"No meaningful validation improvement for {stale_epochs} epoch(s) "
                f"(patience={args.patience}, min_delta={args.min_delta})"
            )
        if should_stop:
            print(f"Early stopping after epoch {epoch + 1}; restoring best checkpoint on next test")
            break

    if getattr(args, "skip_test_after_train", False):
        print("Skipping test after training as requested; checkpoint selection is complete")
        return None
    return test_lmkt(args, fold)

def test_lmkt(args, fold):
    model_name = args.model_name and args.model_name + (f"_{fold}" if fold else "")
    cel_mode = getattr(args, "cel_mode", None)
    _validate_stage2_args(args, training=False)
    model, tokenizer = get_model(
        args.base_model,
        True,
        model_name=model_name,
        quantize=args.quantize,
        use_gradient_checkpointing=False,
        eager_attention=False,
    )
    model.eval()

    if cel_mode:
        model._cel_selector = _init_cel_selector(model, args)
        cel_path = os.path.join(get_checkpoint_path(model_name), CEL_SELECTOR_FILENAME)
        if os.path.exists(cel_path):
            model._cel_selector.load_state_dict(torch.load(cel_path, map_location=device))
            print(f"CEL selector loaded from {cel_path}")
        else:
            if getattr(args, "cel_require_complete_checkpoint", False):
                raise FileNotFoundError(f"CEL selector not found at {cel_path}")
            print(f"WARNING: CEL selector not found at {cel_path}")
        model._cel_calibrator = _init_cel_calibrator(args)
        _maybe_load_cel_calibrator(
            model,
            model_name,
            required=getattr(args, "cel_require_complete_checkpoint", False),
        )
        model._cel_environment = _init_cel_environment(model, args)
        if model._cel_environment is not None:
            env_path = os.path.join(get_checkpoint_path(model_name), CEL_ENVIRONMENT_FILENAME)
            if not os.path.isfile(env_path):
                raise FileNotFoundError(f"Stage 2 environment checkpoint not found at {env_path}")
            model._cel_environment.load_state_dict(torch.load(env_path, map_location=device, weights_only=True))
            _load_stage2_manifest(model_name)
            print(f"Stage 2 environment loaded from {env_path}")
        model._cel_metrics = {}
        model._cel_last_outputs = {}

    effective_pack_kcs = args.pack_kcs and cel_mode != "task_conditioned"
    if cel_mode == "task_conditioned" and args.pack_kcs:
        print("CEL task_conditioned: forcing unpacked prompts to preserve KC-specific gating")

    KTDataset = LMKTDatasetPacked if effective_pack_kcs else LMKTDatasetUnpacked
    KTCollator = LMKTCollatorPacked if effective_pack_kcs else LMKTCollatorUnpacked
    get_loss = get_lmkt_loss_packed if effective_pack_kcs else get_lmkt_loss_unpacked
    if cel_mode:
        get_loss = lambda model, batch, true_token, false_token, args: get_lmkt_loss_cel(model, batch, true_token, false_token, args, tokenizer=tokenizer)

    _, val_df, test_df = load_annotated_data(args, fold)
    if args.testonval:
        test_df = val_df
    if args.debug:
        test_df = test_df[:10]
        print(test_df.iloc[0])

    test_dataset = KTDataset(test_df, tokenizer, args, skip_first_turn=not args.inc_first_label)
    collator = KTCollator(tokenizer)
    test_dataloader = get_dataloader(test_dataset, collator, args.batch_size, False)
    true_token, false_token = get_true_false_tokens(tokenizer)

    dialogue_idx_to_sample_idxs = {}
    all_labels = []
    all_preds = []
    all_kc_probs = []
    all_kcs = []
    sample_dialogue_ids = []
    stage2_path_preds = {
        "evidence": [],
        "mixed": [],
        "non_evidence": [],
    } if _stage2_uses_dual_path(args) else None
    total_loss = 0
    test_cel_metrics = []
    for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Testing")):
        sample_offset = len(all_labels)
        for sample_idx, sample in enumerate(batch["meta_data"]):
            dialogue_idx_to_sample_idxs.setdefault(sample["dialogue_idx"], []).append(sample_offset + sample_idx)
            sample_dialogue_ids.append(sample["dialogue_idx"])
        with torch.no_grad():
            loss, kc_probs, corr_probs = get_loss(model, batch, true_token, false_token, args)
        total_loss += loss.item()
        if cel_mode and getattr(model, "_cel_metrics", None):
            test_cel_metrics.append(dict(model._cel_metrics))
        all_labels.extend(batch["labels"].tolist())
        all_preds.extend(corr_probs.tolist())
        all_kc_probs.extend(kc_probs)
        all_kcs.extend([sample["kcs"] for sample in batch["meta_data"]])
        if stage2_path_preds is not None:
            last_outputs = getattr(model, "_cel_last_outputs", {})
            for path_name in stage2_path_preds:
                path_probs = last_outputs.get(path_name)
                if path_probs is None:
                    raise ValueError(f"Missing Stage 2 {path_name} predictions during evaluation")
                stage2_path_preds[path_name].extend(path_probs.tolist())

    loss = total_loss / len(test_dataloader)
    final_turn_labels = [all_labels[idxs[-1]] for idxs in dialogue_idx_to_sample_idxs.values()]
    final_turn_preds = [all_preds[idxs[-1]] for idxs in dialogue_idx_to_sample_idxs.values()]
    test_cel_summary = summarize_metric_dicts(test_cel_metrics)
    all_metrics, final_metrics = compute_all_metrics(
        loss,
        all_labels,
        all_preds,
        final_turn_labels,
        final_turn_preds,
        args,
        fold,
        extra_metrics=test_cel_summary,
    )

    if stage2_path_preds is not None:
        final_indices = {idxs[-1] for idxs in dialogue_idx_to_sample_idxs.values()}
        non_final_indices = [idx for idx in range(len(all_labels)) if idx not in final_indices]
        ordered_final_indices = sorted(final_indices)
        path_report = {
            "schema_version": 1,
            "evaluation_split": "validation" if args.testonval else "test",
            "primary_path": "mixed",
            "sample_count": len(all_labels),
            "paths": {},
            "diagnostics": test_cel_summary,
        }
        for path_name, predictions in stage2_path_preds.items():
            overall = compute_metrics(all_labels, predictions)
            final = compute_metrics(
                [all_labels[idx] for idx in ordered_final_indices],
                [predictions[idx] for idx in ordered_final_indices],
            )
            non_final = compute_metrics(
                [all_labels[idx] for idx in non_final_indices],
                [predictions[idx] for idx in non_final_indices],
            )
            path_report["paths"][path_name] = {
                "overall": dict(zip(("acc", "auc", "precision", "recall", "f1"), overall)),
                "final": dict(zip(("acc", "auc", "precision", "recall", "f1"), final)),
                "non_final": dict(zip(("acc", "auc", "precision", "recall", "f1"), non_final)),
            }
            print(
                f"Stage 2 {path_name}: Overall Acc/AUC {overall[0]:.2f}/{overall[1]:.2f}; "
                f"Final {final[0]:.2f}/{final[1]:.2f}; "
                f"Non-final {non_final[0]:.2f}/{non_final[1]:.2f}"
            )

        metrics_dir = get_results_dir(args, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        suffix = get_model_file_suffix(args, fold)
        path_report_path = os.path.join(metrics_dir, f"stage2_paths_{suffix}.json")
        with open(path_report_path, "w", encoding="utf-8") as report_file:
            json.dump(path_report, report_file, indent=2, sort_keys=True)
            report_file.write("\n")
        prediction_rows = []
        for sample_idx, label in enumerate(all_labels):
            prediction_rows.append({
                "sample_idx": sample_idx,
                "dialogue_idx": sample_dialogue_ids[sample_idx],
                "is_final": sample_idx in final_indices,
                "label": label,
                "p_r": stage2_path_preds["evidence"][sample_idx],
                "p_m": stage2_path_preds["mixed"][sample_idx],
                "p_n": stage2_path_preds["non_evidence"][sample_idx],
            })
        prediction_dir = get_results_dir(args, "qual")
        os.makedirs(prediction_dir, exist_ok=True)
        pd.DataFrame(prediction_rows).to_csv(
            os.path.join(prediction_dir, f"stage2_paths_{suffix}.csv"),
            index=False,
        )

    kc_results = {
        dialogue_idx: [
            {
                kc: kc_prob
                for kc, kc_prob in zip(all_kcs[sample_idx], all_kc_probs[sample_idx])
            }
            for sample_idx in sample_idxs
        ]
        for dialogue_idx, sample_idxs in dialogue_idx_to_sample_idxs.items()
    }
    with open(get_kc_result_filename(args, fold), "w") as out_file:
        json.dump(kc_results, out_file, indent=2)

    qual_data = []
    for dia_idx, sample in test_df.iterrows():
        dialogue = apply_annotations(sample)
        if dia_idx not in dialogue_idx_to_sample_idxs:
            continue
        dia_preds = [all_preds[idx] for idx in dialogue_idx_to_sample_idxs[dia_idx]]
        dia_labels = [all_labels[idx] for idx in dialogue_idx_to_sample_idxs[dia_idx]]
        dia_acc = f"{(np.round(dia_preds) == dia_labels).mean():.4f}"
        first_turn = not args.inc_first_label
        label_counter = 0
        for turn in dialogue:
            if turn["correct"] is not None and not first_turn:
                label_idx = dialogue_idx_to_sample_idxs[dia_idx][label_counter]
                prob = f"{all_preds[label_idx]:.4f}"
                kc_probs = ", ".join([f"{kc_prob:.4f}" for kc_prob in all_kc_probs[label_idx]])
                label_counter += 1
            else:
                prob = "--"
                kc_probs = "--"
            if turn["correct"] is not None and first_turn:
                first_turn = False
            qual_data.append({
                "Dialogue ID": dia_idx,
                "Turn": turn["turn"],
                "Teacher": turn["teacher"] or "--",
                "Student": turn["student"],
                "Correct": correct_to_str(turn["correct"]),
                "Prob": prob,
                "KC Probs": kc_probs,
                "Dialogue Acc.": dia_acc,
                "KCs": standards_to_str(turn["kcs"], "\n"),
                "Notes": ""
            })
        qual_data.append({key: "" for key in qual_data[0]})
    pd.DataFrame(qual_data).to_csv(get_qual_result_filename(args, fold), index=False)

    return np.array([loss, *all_metrics, *final_metrics])


# ===== Baselines =====

BASELINE_MODELS = ["dkt-multi", "dkt-sem", "dkt", "akt", "dkvmn", "saint", "simplekt"]
NON_FLAT_KC_ARCH = ["dkt-multi", "dkt-sem"]

def select_flat_baseline_out_vectors(y: torch.Tensor, batch, shift_turn_end_idxs: bool):
    if shift_turn_end_idxs:
        # Predict KCs with output from first KC of turn for models where correctness is only visible in previous idxs
        # Clip at end to prevent out of bounds, no effect since last pred unused
        batch["turn_end_idxs"] = torch.clip(batch["turn_end_idxs"] + 1, max=batch["turn_end_idxs"].max())
    # Get output vectors at index of last KC per turn (to predict next turn's KCs)
    turn_end_idxs = batch["turn_end_idxs"].unsqueeze(2).repeat(1, 1, y.shape[2])
    return torch.gather(y, 1, turn_end_idxs)

def get_baseline_loss(y: torch.Tensor, batch, args):
    # Aggregate KC probs from outputs, one output per question
    batch_size, max_seq_len, max_num_kcs = batch["kc_ids"].shape
    kc_pad_mask = torch.arange(max_num_kcs).repeat(batch_size, max_seq_len, 1).to(device) >= batch["num_kcs"].unsqueeze(2)
    y = y[:, :-1].contiguous() # Last item in sequence doesn't predict anything
    kc_probs = torch.gather(y, 2, batch["kc_ids"][:, 1:]) # Collect KC predictions for next question, B x L x K
    # Set probs on padded indices
    padding_val = 0 if args.agg == "mean-ar" else 1
    kc_probs = torch.masked_scatter(kc_probs, kc_pad_mask[:, 1:], torch.full_like(kc_probs, padding_val).to(device))
    # Calculate correct probabilities (B x L)
    if args.agg == "prod":
        corr_probs = kc_probs.prod(dim=2)
    elif args.agg == "mean-ar":
        corr_probs = kc_probs.sum(dim=2) / batch["num_kcs"][:, 1:]
    elif args.agg == "mean-geo":
        corr_probs = kc_probs.prod(dim=2) ** (1 / batch["num_kcs"][:, 1:])

    # Compute BCE loss
    corr_probs, _ = _sanitize_probability_tensor(corr_probs)
    labels_flat = batch["labels"][:, 1:].contiguous().view(-1)
    loss_mask = labels_flat != -100
    labels_flat = labels_flat[loss_mask].type(torch.float)
    corr_probs_flat = corr_probs.view(-1)[loss_mask]
    loss, _, _ = _compute_bce_loss_from_probs(corr_probs_flat, labels_flat)
    return loss, corr_probs

def get_baseline_model(kc_dict: dict, kc_emb_matrix: torch.Tensor, args):
    num_kcs = len(kc_dict)
    emb_size = args.emb_size
    n_blocks = 4 # For layered models
    if args.model_type == "dkt-multi":
        return DKTMultiKC(num_kcs, emb_size).to(device)
    if args.model_type == "dkt-sem":
        return DKTSem(emb_size, kc_emb_matrix).to(device)
    if args.model_type == "dkt":
        return DKT(num_kcs, emb_size).to(device)
    if args.model_type == "akt":
        model = AKT(num_kcs, num_kcs, emb_size, n_blocks, 0.05, emb_size, final_fc_dim=emb_size)
        model.out[3] = torch.nn.Linear(emb_size, emb_size) # Reduce from 256 to emb_size to avoid overparameterization
        model.out[6] = torch.nn.Linear(emb_size, num_kcs) # Predict all KCs instead of just current question
        return model.to(device)
    if args.model_type == "dkvmn":
        model = DKVMN(num_kcs, emb_size, 50)
        model.p_layer = torch.nn.Linear(emb_size, num_kcs) # Predict all KCs instead of just current question
        return model.to(device)
    if args.model_type == "saint":
        model = SAINT(num_kcs, num_kcs, 256, emb_size, 8, 0.2, n_blocks)
        model.out = torch.nn.Linear(emb_size, num_kcs) # Predict all KCs instead of just current question
        return model.to(device)
    if args.model_type == "simplekt":
        model = simpleKT(num_kcs, num_kcs, emb_size, n_blocks, 0.2, d_ff=emb_size, final_fc_dim=emb_size, final_fc_dim2=emb_size)
        model.out[6] = torch.nn.Linear(emb_size, num_kcs) # Predict all KCs instead of just current question
        return model.to(device)
    raise Exception(f"Model {args.model_type} not supported")

def compute_baseline_loss(model, batch, args):
    if args.model_type == "dkt-multi":
        y = model(batch)
        return get_baseline_loss(y, batch, args)
    elif args.model_type == "dkt-sem":
        y = model(batch)
        return get_baseline_loss(y, batch, args)
    elif args.model_type == "dkt":
        y = model(batch["kc_ids_flat"], batch["labels_flat"])
        y = select_flat_baseline_out_vectors(y, batch, False)
        return get_baseline_loss(y, batch, args)
    elif args.model_type == "akt":
        y, rasch_loss = model(batch["kc_ids_flat"], batch["labels_flat"], batch["kc_ids_flat"])
        y = select_flat_baseline_out_vectors(y, batch, True)
        loss, corr_probs = get_baseline_loss(y, batch, args)
        loss += rasch_loss
        return loss, corr_probs
    elif args.model_type == "dkvmn":
        y = model(batch["kc_ids_flat"], batch["labels_flat"])
        y = select_flat_baseline_out_vectors(y, batch, True)
        return get_baseline_loss(y, batch, args)
    elif args.model_type == "saint":
        y = model(batch["kc_ids_flat"], batch["kc_ids_flat"], batch["labels_flat"][:, :-1])
        y = select_flat_baseline_out_vectors(y, batch, True)
        return get_baseline_loss(y, batch, args)
    elif args.model_type == "simplekt":
        y = model({
            "qseqs": batch["kc_ids_flat"][:, :-1],
            "cseqs": batch["kc_ids_flat"][:, :-1],
            "rseqs": batch["labels_flat"][:, :-1],
            "shft_qseqs": batch["kc_ids_flat"][:, 1:],
            "shft_cseqs": batch["kc_ids_flat"][:, 1:],
            "shft_rseqs": batch["labels_flat"][:, 1:]
        })
        y = select_flat_baseline_out_vectors(y, batch, True)
        return get_baseline_loss(y, batch, args)
    raise Exception(f"Model {args.model_type} not supported")

def compute_kc_emb_matrix(sbert_model: SentenceTransformer, kc_dict: dict):
    print("Computing SBERT embeddings...")
    kcs = [kv[0] for kv in sorted(kc_dict.items(), key=lambda kv: kv[1])]
    kc_emb_matrix = sbert_model.encode(kcs, convert_to_tensor=True)
    return kc_emb_matrix

def train_baseline(args, fold):
    assert args.model_type in BASELINE_MODELS

    # Load KC dictionary and optionally text embeddings
    kc_dict = load_kc_dict(args)
    if args.model_type == "dkt-sem":
        sbert_model = SentenceTransformer("all-mpnet-base-v2")
        kc_emb_matrix = compute_kc_emb_matrix(sbert_model, kc_dict)
    else:
        sbert_model = None
        kc_emb_matrix = None

    # Create model
    model = get_baseline_model(kc_dict, kc_emb_matrix, args)

    # Load and split dataset, annotated with correctness and KCs
    train_df, val_df, _ = load_annotated_data(args, fold)
    if args.debug:
        train_df = train_df[:2]
        val_df = val_df[:2]
        print(train_df.iloc[0])
        print(val_df.iloc[0])
    flatten_kcs = args.model_type not in NON_FLAT_KC_ARCH # Flatten KCs in sequence for architectures that don't support multi-KCs
    train_dataset = DKTDataset(train_df, kc_dict, kc_emb_matrix, sbert_model)
    val_dataset = DKTDataset(val_df, kc_dict, kc_emb_matrix, sbert_model)
    collator = DKTCollator(flatten_kcs)
    train_dataloader = get_dataloader(train_dataset, collator, args.batch_size, True)
    val_dataloader = get_dataloader(val_dataset, collator, args.batch_size, False)

    # Do training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best_val_loss = None
    stale_epochs = 0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}")
        total_train_loss = 0
        total_val_loss = 0

        model.train()
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            loss, _ = compute_baseline_loss(model, batch, args)
            total_train_loss += loss.item()
            loss = loss / args.grad_accum_steps
            loss.backward()
            if (batch_idx + 1) % args.grad_accum_steps == 0 or batch_idx == len(train_dataloader) - 1:
                if args.gc:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.gc)
                optimizer.step()
                optimizer.zero_grad()

        with torch.no_grad():
            model.eval()
            for batch in tqdm(val_dataloader, desc="Validating"):
                loss, _ = compute_baseline_loss(model, batch, args)
                total_val_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        best_val_loss, stale_epochs, should_stop, improved = _early_stopping_update(
            best_val_loss,
            stale_epochs,
            avg_val_loss,
            args,
        )
        if improved:
            print("Best! Saving model...")
            model_name = args.model_name + (f"_{fold}" if fold else "") + ".pt"
            torch.save(model.state_dict(), get_checkpoint_path(model_name))
        elif getattr(args, "patience", 0):
            print(
                f"No meaningful validation improvement for {stale_epochs} epoch(s) "
                f"(patience={args.patience}, min_delta={args.min_delta})"
            )
        if should_stop:
            print(f"Early stopping after epoch {epoch + 1}; best checkpoint retained")
            break

    return test_baseline(args, fold)

def test_baseline(args, fold):
    # Load KC dictionary and optionally text embeddings
    kc_dict = load_kc_dict(args)
    if args.model_type == "dkt-sem":
        sbert_model = SentenceTransformer("all-mpnet-base-v2")
        kc_emb_matrix = compute_kc_emb_matrix(sbert_model, kc_dict)
    else:
        sbert_model = None
        kc_emb_matrix = None

    # Load trained model
    if args.model_type in BASELINE_MODELS:
        model = get_baseline_model(kc_dict, kc_emb_matrix, args)
        model_name = args.model_name + (f"_{fold}" if fold else "") + ".pt"
        model.load_state_dict(torch.load(get_checkpoint_path(model_name), map_location=device))
        model.eval()
    else:
        model = None

    # Load annotated data
    _, val_df, test_df = load_annotated_data(args, fold)
    if args.testonval:
        test_df = val_df
    if args.debug:
        test_df = test_df[:10]
        print(test_df.iloc[0])
    flatten_kcs = args.model_type not in NON_FLAT_KC_ARCH # Flatten KCs in sequence for architectures that don't support multi-KCs
    test_dataset = DKTDataset(test_df, kc_dict, kc_emb_matrix, sbert_model)
    collator = DKTCollator(flatten_kcs)
    test_dataloader = get_dataloader(test_dataset, collator, args.batch_size, False)

    # Collect meta data and predicted KC/correctness probabilities for test set
    all_labels = []
    all_preds = []
    final_turn_labels = []
    final_turn_preds = []
    total_loss = 0
    for batch in tqdm(test_dataloader):
        labels = batch["labels"][:, 1:]
        if model is not None:
            with torch.no_grad():
                loss, corr_probs = compute_baseline_loss(model, batch, args)
        elif args.model_type == "random":
            corr_probs = torch.zeros_like(labels).random_(0, 2)
            loss = torch.tensor(0)
        elif args.model_type == "majority":
            corr_probs = torch.full_like(labels, fill_value=test_dataset.majority_class)
            loss = torch.tensor(0)
        total_loss += loss.item()
        mask = labels != -100
        all_labels.extend(labels[mask].tolist())
        all_preds.extend(corr_probs[mask].tolist())
        final_idxs = mask.sum(dim=1) - 1
        final_turn_labels.extend(labels[torch.arange(mask.shape[0]), final_idxs].tolist())
        final_turn_preds.extend(corr_probs[torch.arange(mask.shape[0]), final_idxs].tolist())

    # Compute quantitative metrics across all turns and only on final turns
    loss = total_loss / len(test_dataloader)
    all_metrics, final_metrics = compute_all_metrics(loss, all_labels, all_preds, final_turn_labels, final_turn_preds, args, fold)

    return np.array([loss, *all_metrics, *final_metrics])

def bkt_prep_data(df: pd.DataFrame, kc_dict: dict):
    dataset = DKTDataset(df, kc_dict, None, None)
    results = []
    order_id = 0
    for _, sample in enumerate(dataset.data):
        for kc, label in zip(sample["kc_ids_flat"], sample["labels_flat"]):
            results.append({"user_id": sample["dialogue_idx"], "skill_name": str(kc), "correct": label, "order_id": order_id})
            order_id += 1
    return pd.DataFrame(results), dataset

def train_test_bkt(args, fold):
    # Load and reformat data
    kc_dict = load_kc_dict(args)
    train_df, val_df, test_df = load_annotated_data(args, fold)
    train_df, _ = bkt_prep_data(pd.concat([train_df, val_df]), kc_dict)
    test_df, test_dataset = bkt_prep_data(test_df, kc_dict)

    # Train model
    model = BKT(seed=221, num_fits=1)
    model.fit(data=train_df)

    # Test model
    print("Train Acc./AUC:", model.evaluate(data=train_df, metric=["accuracy", "auc"]))
    print("Test Acc./AUC:", model.evaluate(data=test_df, metric=["accuracy", "auc"]))
    pred_df: pd.DataFrame = model.predict(data=test_df)
    pred_df = pred_df.sort_values(["order_id"])
    all_labels = []
    all_preds = []
    for sample, (_, user) in zip(test_dataset, pred_df.groupby("user_id", sort=False)):
        preds_flat = user["correct_predictions"]
        labels = sample["labels"]
        preds = []
        prev_idx = 0
        for turn_end_idx in sample["turn_end_idxs"]:
            if args.agg == "prod":
                preds.append(np.prod(preds_flat[prev_idx : turn_end_idx + 1]))
            elif args.agg == "mean-ar":
                preds.append(np.mean(preds_flat[prev_idx : turn_end_idx + 1]))
            elif args.agg == "mean-geo":
                preds.append(np.prod(preds_flat[prev_idx : turn_end_idx + 1]) ** (1 / (turn_end_idx - prev_idx + 1)))
            prev_idx = turn_end_idx + 1
        all_labels.extend(labels[1:])
        all_preds.extend(preds[1:])
    metrics, _ = compute_all_metrics(0, all_labels, all_preds, None, None, args, fold)
    return [0, *metrics]
