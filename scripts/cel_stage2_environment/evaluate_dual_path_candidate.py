#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SAVED_MODELS = ROOT / "saved_models"
EXPECTED_BASE_MODEL = "/home/user4/.cache/huggingface/hub/qwen/Qwen3-1.7B"
EXPECTED_MODEL_SEED = 1221


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("round_id")
    parser.add_argument("candidate_id")
    parser.add_argument("split", choices=("validation", "test"))
    parser.add_argument("--result_subdir", default=None)
    parser.add_argument("--cuda", default=None)
    args = parser.parse_args()

    final_model = f"cel_{args.candidate_id}_joint_qwen3_1.7b"
    manifest_path = SAVED_MODELS / final_model / "cel_stage2_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing final Stage 2 manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_candidate_id = (
        f"stage2_dual_{args.round_id}_contextual_transformer_seed1221"
    )
    if args.candidate_id != expected_candidate_id:
        raise ValueError(
            f"candidate id does not match round {args.round_id}: "
            f"expected {expected_candidate_id}, got {args.candidate_id}"
        )
    if manifest.get("phase") != "joint":
        raise ValueError(f"final manifest is not a joint phase: {manifest_path}")
    if manifest.get("candidate_id") != args.candidate_id:
        raise ValueError("candidate id does not match the final manifest")
    if manifest.get("model_name") != final_model:
        raise ValueError("model name does not match the final manifest")
    if manifest.get("base_model") != EXPECTED_BASE_MODEL:
        raise ValueError("final manifest does not use the formal raw Qwen base")
    if manifest.get("model_init_seed") != EXPECTED_MODEL_SEED:
        raise ValueError("final manifest does not use model/data seed 1221")
    if manifest.get("objective") != "dual_path_hr_hm_js":
        raise ValueError("final manifest does not use the dual-path Stage 2 objective")
    if manifest.get("fresh_init_required") is not True:
        raise ValueError("final manifest is not marked as a fresh raw-base candidate")

    result_subdir = args.result_subdir or (
        f"cel_stage2_environment/dual_path/{args.round_id}/evaluations/{args.split}"
    )
    command = [
        sys.executable,
        "-m",
        "dialogue_kt.main",
        "test",
        "--dataset",
        "mathdial",
        "--model_type",
        "lmkt",
        "--base_model",
        str(manifest["base_model"]),
        "--model_name",
        final_model,
        "--result_subdir",
        result_subdir,
        "--quantize",
        "0",
        "--pack_kcs",
        "1",
        "--cel_mode",
        str(manifest["cel_mode"]),
        "--cel_layer_idx",
        str(manifest["cel_layer_idx"]),
        "--cel_hook_site",
        str(manifest["cel_hook_site"]),
        "--cel_hook_timing",
        str(manifest["cel_hook_timing"]),
        "--cel_gamma",
        str(manifest["cel_gamma"]),
        "--cel_drop",
        str(manifest.get("cel_drop", 0.1)),
        "--cel_injection_variant",
        str(manifest["cel_injection_variant"]),
        "--cel_application_mode",
        str(manifest["cel_application_mode"]),
        "--cel_stage2_enabled",
        "1",
        "--cel_stage2_fresh_init",
        "1",
        "--cel_stage2_candidate_id",
        args.candidate_id,
        "--cel_stage2_phase",
        "joint",
        "--cel_env_mode",
        str(manifest["env_mode"]),
        "--cel_env_split_mode",
        str(manifest["env_split_mode"]),
        "--cel_env_beta",
        str(manifest["env_beta"]),
        "--cel_env_hidden_dim",
        str(manifest["env_hidden_dim"]),
        "--cel_env_num_layers",
        str(manifest["env_num_layers"]),
        "--cel_env_num_heads",
        str(manifest["env_num_heads"]),
        "--cel_env_drop",
        str(manifest.get("cel_env_drop", 0.1)),
        "--cel_env_output_postprocess",
        str(manifest["env_output_postprocess"]),
        "--cel_env_output_ratio",
        str(manifest["env_output_ratio"]),
        "--cel_env_output_init_std",
        str(manifest["env_output_init_std"]),
        "--cel_env_shuffle_seed",
        str(manifest.get("cel_env_shuffle_seed", 221)),
        "--cel_stage2_lambda_r",
        str(manifest["lambda_r"]),
        "--cel_stage2_lambda_m",
        str(manifest["lambda_m"]),
        "--cel_stage2_lambda_cons",
        str(manifest["lambda_cons"]),
        "--cel_stage2_beta_start_ratio",
        str(manifest["beta_start_ratio"]),
        "--cel_stage2_consistency_ramp_fraction",
        str(manifest["consistency_ramp_fraction"]),
        "--cel_output_calibration",
        str(manifest.get("cel_output_calibration", "bias")),
        "--cel_calibrator_init_bias",
        str(manifest.get("cel_calibrator_init_bias", 0.0)),
        "--cel_require_complete_checkpoint",
        "1",
    ]
    if manifest.get("cel_selector_hidden_dim") is not None:
        command.extend(["--cel_selector_hidden_dim", str(manifest["cel_selector_hidden_dim"])])
    if manifest.get("env_ffn_dim") is not None:
        command.extend(["--cel_env_ffn_dim", str(manifest["env_ffn_dim"])])
    if args.split == "validation":
        command.append("--testonval")

    environment = None
    if args.cuda is not None:
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = args.cuda
    print("Executing:", " ".join(command))
    completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
