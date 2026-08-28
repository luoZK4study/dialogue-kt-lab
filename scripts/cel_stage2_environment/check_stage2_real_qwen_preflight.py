#!/usr/bin/env python3

"""Run one real-Qwen dual-path forward/backward without saving artifacts."""

from __future__ import annotations

import argparse
import os
import sys
from argparse import Namespace
from pathlib import Path

# Match the formal launcher before importing torch, because initialize_seeds
# enables deterministic operations.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# Data-path helpers in the training pipeline are repository-relative.
os.chdir(ROOT)

from dialogue_kt.data_loading import load_annotated_data
from dialogue_kt.kt_data_loading import LMKTCollatorUnpacked, LMKTDatasetUnpacked
from dialogue_kt.models.lm import get_model
from dialogue_kt.prompting import get_true_false_tokens
from dialogue_kt.training import (
    _clear_pending_cel_hooks,
    _cel_can_skip_backbone_grad_path,
    _configure_cel_frozen_backbone_memory,
    _configure_cel_module_modes,
    _configure_cel_trainability,
    _init_cel_calibrator,
    _init_cel_environment,
    _init_cel_selector,
    get_lmkt_loss_cel,
)
from dialogue_kt.utils import initialize_seeds


def build_args(cli_args: argparse.Namespace) -> Namespace:
    return Namespace(
        dataset="mathdial",
        tag_src="atc",
        typical_cutoff=1,
        split_by_subject=False,
        prompt_inc_labels=False,
        inc_first_label=False,
        pack_kcs=True,
        agg="mean-ar",
        cel_mode="task_conditioned",
        cel_layer_idx=-1,
        cel_hook_site="last_block",
        cel_hook_timing="pre_block",
        cel_gamma=0.30,
        cel_selector_hidden_dim=512,
        cel_adapter_dim=None,
        cel_drop=0.10,
        cel_use_norm=False,
        cel_injection_variant="scalar_gate",
        cel_application_mode="token_residual",
        cel_output_calibration="bias",
        cel_calibrator_init_bias=0.331,
        cel_calibrator_init_scale=1.0,
        cel_stage2_enabled=True,
        cel_stage2_phase="b_warmup" if cli_args.environment_only else "joint",
        cel_env_mode="contextual_transformer",
        cel_env_beta=0.10,
        cel_env_split_mode="complementary",
        cel_env_topk_ratio=0.1,
        cel_env_sigmoid_temperature=5.0,
        cel_env_hidden_dim=1024,
        cel_env_num_layers=4,
        cel_env_num_heads=8,
        cel_env_ffn_dim=4096,
        cel_env_drop=0.10,
        cel_env_output_postprocess="centered_rms",
        cel_env_output_ratio=1.0,
        cel_env_output_init_std=0.01,
        cel_env_shuffle_seed=221,
        cel_stage2_lambda_r=1.0,
        cel_stage2_lambda_m=1.0,
        cel_stage2_lambda_cons=0.10,
        cel_stage2_beta_start_ratio=0.20,
        cel_stage2_consistency_ramp_fraction=0.25,
        cel_train_environment_only=cli_args.environment_only,
        cel_train_calibrator_only=False,
        cel_train_selector_only=False,
        cel_train_selector_and_calibrator_only=False,
    )


def gradient_norm(module: torch.nn.Module) -> float:
    values = [
        parameter.grad.detach().float().square().sum()
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    if not values:
        return 0.0
    return float(torch.stack(values).sum().sqrt().item())


def select_longest_sample(dataset, tokenizer):
    """Select by the same row/padding geometry used by the formal collator."""
    candidates = []
    for sample in dataset.data:
        tokenized = tokenizer(sample["prompts"], padding=True)
        row_count = len(tokenized["input_ids"])
        max_tokens = len(tokenized["input_ids"][0])
        # The unpacked collator turns one turn's KCs into independent Qwen
        # rows. Attention memory therefore scales with both rows and S^2.
        attention_risk = row_count * max_tokens * max_tokens
        candidates.append((attention_risk, max_tokens, row_count))
    sample_index = max(range(len(dataset.data)), key=lambda idx: candidates[idx])
    attention_risk, max_tokens, row_count = candidates[sample_index]
    return sample_index, max_tokens, row_count, attention_risk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-model",
        default="/home/user4/.cache/huggingface/hub/qwen/Qwen3-1.7B",
    )
    parser.add_argument("--seed", type=int, default=1221)
    parser.add_argument(
        "--dialogues",
        type=int,
        default=None,
        help="optional number of training dialogues to inspect; default is the full training split",
    )
    parser.add_argument(
        "--environment-only",
        action="store_true",
        help="exercise the B-only warmup gradient and memory path",
    )
    args = parser.parse_args()
    if args.dialogues is not None and args.dialogues < 1:
        raise ValueError("--dialogues must be positive")

    config = build_args(args)
    initialize_seeds(args.seed)
    model, tokenizer = get_model(
        args.base_model,
        False,
        r=16,
        lora_alpha=16,
        quantize=False,
        use_gradient_checkpointing=not _cel_can_skip_backbone_grad_path(config),
        eager_attention=False,
    )
    model._cel_selector = _init_cel_selector(model, config)
    model._cel_calibrator = _init_cel_calibrator(config)
    model._cel_environment = _init_cel_environment(model, config)
    model._cel_metrics = {}
    model._cel_last_outputs = {}
    model._cel_pending_hook_handles = []
    model._cel_serial_joint_backward = not args.environment_only
    model._cel_loss_backward_done = False
    model._cel_serial_backward_scale = 1.0
    _configure_cel_trainability(model, config)
    _configure_cel_frozen_backbone_memory(model, config)

    train_df, _, _ = load_annotated_data(config, None)
    source_df = train_df if args.dialogues is None else train_df.iloc[: args.dialogues]
    dataset = LMKTDatasetUnpacked(source_df, tokenizer, config)
    if not dataset.data:
        raise RuntimeError("the selected preflight dialogues produced no valid LLMKT example")
    (
        sample_index,
        selected_token_length,
        selected_row_count,
        selected_attention_risk,
    ) = select_longest_sample(dataset, tokenizer)
    batch = LMKTCollatorUnpacked(tokenizer)([dataset[sample_index]])
    true_token, false_token = get_true_false_tokens(tokenizer)

    torch.cuda.reset_peak_memory_stats()
    model.train()
    _configure_cel_module_modes(model, config)
    loss, _, _ = get_lmkt_loss_cel(model, batch, true_token, false_token, config, tokenizer=tokenizer)
    if not torch.isfinite(loss):
        raise RuntimeError("dual-path training loss is not finite")
    if not getattr(model, "_cel_loss_backward_done", False):
        loss.backward()
    _clear_pending_cel_hooks(model)
    selector_grad = gradient_norm(model._cel_selector)
    environment_grad = gradient_norm(model._cel_environment)
    if environment_grad <= 0:
        raise RuntimeError(
            f"missing dual-path B gradient: selector={selector_grad}, B={environment_grad}"
        )
    if not args.environment_only and selector_grad <= 0:
        raise RuntimeError(
            f"missing dual-path A gradient: selector={selector_grad}, B={environment_grad}"
        )
    if args.environment_only and selector_grad != 0:
        raise RuntimeError(
            f"B-only preflight unexpectedly produced selector gradients: {selector_grad}"
        )

    model.zero_grad(set_to_none=True)
    model.eval()
    model._cel_selector.eval()
    model._cel_environment.eval()
    model._cel_calibrator.eval()
    with torch.no_grad():
        eval_loss, _, _ = get_lmkt_loss_cel(model, batch, true_token, false_token, config, tokenizer=tokenizer)
    _clear_pending_cel_hooks(model)
    paths = model._cel_last_outputs
    if not torch.isfinite(eval_loss):
        raise RuntimeError("dual-path evaluation loss is not finite")
    if any(paths.get(name) is None for name in ("evidence", "mixed", "non_evidence")):
        raise RuntimeError("evaluation did not produce evidence, mixed, and reversal probabilities")
    if not all(torch.isfinite(paths[name]).all() for name in paths):
        raise RuntimeError("non-finite dual-path probability")

    print("Real-Qwen dual-path preflight passed")
    print(f"selected_sample_index={sample_index}")
    print(f"selected_max_tokens={selected_token_length}")
    print(f"selected_rows={selected_row_count}")
    print(f"selected_attention_risk={selected_attention_risk}")
    print(f"train_mode={'environment_only' if args.environment_only else 'joint'}")
    print(f"input_rows={batch['input_ids'].shape[0]}")
    print(f"sequence_length={batch['input_ids'].shape[1]}")
    print(f"train_loss={loss.item():.6f}")
    print(f"eval_loss={eval_loss.item():.6f}")
    print(f"selector_grad_norm={selector_grad:.6f}")
    print(f"b_grad_norm={environment_grad:.6f}")
    print(f"peak_memory_mib={torch.cuda.max_memory_allocated() / 1024 / 1024:.1f}")


if __name__ == "__main__":
    main()
