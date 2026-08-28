"""
Pre-compute SELFELICIT evidence annotations for dialogue KT data.

This script runs the base model (without LoRA) on each sample to identify
which dialogue turns are most important evidence for KC prediction.
The computed evidence masks are cached to JSON for use during training/testing.

Usage:
    python -m dialogue_kt.precompute_evidence --dataset mathdial
"""

import argparse
import json
import torch
import numpy as np
from tqdm import tqdm

from dialogue_kt.utils import initialize_seeds, device
from dialogue_kt.data_loading import load_annotated_data, get_default_fold
from dialogue_kt.kt_data_loading import apply_annotations, save_evidence_cache
from dialogue_kt.prompting import kt_system_prompt, kt_user_prompt
from dialogue_kt.selfelicit import (
    compute_evidence_for_sample,
    get_evidence_layer_range,
    DEFAULT_LAYER_SPAN, DEFAULT_ALPHA,
    MARKER_IMPSTART, MARKER_IMPEND
)
from dialogue_kt.models.lm import get_base_model
from transformers import AutoTokenizer


def compute_evidence_for_split(model, tokenizer, data, args, split_name="train"):
    """
    Compute SELFELICIT evidence annotations for all samples in a data split.

    Args:
        model: Base LLM (no LoRA)
        tokenizer: HF tokenizer
        data: DataFrame with annotated dialogue samples
        args: Training args
        split_name: Name of split for progress display

    Returns:
        evidence_dict: {dialogue_idx_turn_idx: {evidence_mask, turn_scores}}
    """
    from dialogue_kt.kt_data_loading import apply_annotations

    model.eval()
    n_layers = model.config.num_hidden_layers
    evidence_dict = {}

    for idx, sample in tqdm(list(data.iterrows()), desc=f"Computing evidence ({split_name})"):
        dialogue = apply_annotations(sample)
        if not dialogue:
            continue

        for turn in dialogue:
            if turn["correct"] is None:
                continue

            # Build prompt for this turn
            system_content = kt_system_prompt(args)
            user_content = kt_user_prompt(sample, dialogue, turn["turn"], None, args)

            prompt = tokenizer.apply_chat_template([
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ], tokenize=False)

            # Compute evidence
            try:
                marked_prompt, turn_scores, evidence_mask = compute_evidence_for_sample(
                    model, tokenizer, prompt, dialogue,
                    n_layers, DEFAULT_LAYER_SPAN, DEFAULT_ALPHA
                )

                key = f"{idx}_{turn['turn']}"
                evidence_dict[key] = {
                    "turn_scores": turn_scores.tolist(),
                    "evidence_mask": evidence_mask.tolist(),
                }
            except Exception as e:
                print(f"  Warning: Failed for dialogue {idx}, turn {turn['turn']}: {e}")

    return evidence_dict


def main():
    initialize_seeds(221)

    parser = argparse.ArgumentParser(description="Pre-compute SELFELICIT evidence annotations")
    parser.add_argument("--dataset", type=str, choices=["comta", "mathdial"], default="mathdial")
    parser.add_argument("--tag_src", type=str, choices=["base", "atc"], default="atc")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--quantize", type=lambda x: x != "0", default=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--split", type=str, choices=["train", "test", "both"], default="both")
    parser.add_argument("--typical_cutoff", type=int, default=1)
    parser.add_argument("--prompt_inc_labels", type=lambda x: x != "0", default=False)
    args = parser.parse_args()

    # Load tokenizer and base model
    print(f"Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, padding_side="right", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.bos_token if tokenizer.bos_token is not None else tokenizer.eos_token

    model = get_base_model(args.base_model, tokenizer, args.quantize, eager_attention=True)
    model.eval()
    print(f"Model loaded. Layers: {model.config.num_hidden_layers}")

    if args.split in ("train", "both"):
        train_df, val_df, _ = load_annotated_data(args, get_default_fold(args))
        if args.debug:
            train_df = train_df[:10]
            val_df = val_df[:5]

        print(f"Computing evidence for training set ({len(train_df)} samples)...")
        train_evidence = compute_evidence_for_split(model, tokenizer, train_df, args, "train")
        save_evidence_cache(train_evidence, args, "train")

        print(f"Computing evidence for validation set ({len(val_df)} samples)...")
        # Use 'train' split name since val is part of training data loading
        val_evidence = compute_evidence_for_split(model, tokenizer, val_df, args, "val")
        save_evidence_cache(val_evidence, args, "val")

    if args.split in ("test", "both"):
        _, val_df, test_df = load_annotated_data(args, get_default_fold(args))
        if args.debug:
            test_df = test_df[:10]

        print(f"Computing evidence for test set ({len(test_df)} samples)...")
        test_evidence = compute_evidence_for_split(model, tokenizer, test_df, args, "test")
        save_evidence_cache(test_evidence, args, "test")

    print("Evidence pre-computation complete!")


if __name__ == "__main__":
    main()
