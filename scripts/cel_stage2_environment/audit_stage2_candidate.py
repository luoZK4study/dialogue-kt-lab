#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path

import torch
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[2]
SAVED_MODELS = ROOT / "saved_models"
RESULTS = ROOT / "results/cel_stage2_environment"
BASELINE_METRIC_CANDIDATES = (
    ROOT / "results/baseline/metrics/metrics_lmkt_qwen3_1.7b_recert_20260620.txt",
    ROOT / "results/baseline_recert/metrics/metrics_lmkt_qwen3_1.7b_recert_20260620.txt",
)
STAGE1_METRIC_CANDIDATES = (
    ROOT / "results/cel_stage1_last_layer/metrics/metrics_cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b.txt",
)

LOSS_RE = re.compile(r"^Loss:\s+([0-9.]+)$", re.MULTILINE)
OVERALL_RE = re.compile(r"Overall.*?\n.*?\nAcc:\s+([0-9.]+),\s+AUC:\s+([0-9.]+)", re.DOTALL)
FINAL_RE = re.compile(r"Final Turn.*?\n.*?\nAcc:\s+([0-9.]+),\s+AUC:\s+([0-9.]+)", re.DOTALL)
DIAGNOSTICS_RE = re.compile(r"^CEL Diagnostics:\s+(.*)$", re.MULTILINE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_metrics(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    loss = LOSS_RE.search(text)
    overall = OVERALL_RE.search(text)
    final = FINAL_RE.search(text)
    if not loss or not overall or not final:
        raise ValueError(f"incomplete metrics file: {path}")
    return {
        "loss": float(loss.group(1)),
        "overall_acc": float(overall.group(1)),
        "overall_auc": float(overall.group(2)),
        "final_acc": float(final.group(1)),
        "final_auc": float(final.group(2)),
    }


def parse_diagnostics(path: Path) -> dict[str, float]:
    match = DIAGNOSTICS_RE.search(path.read_text(encoding="utf-8"))
    if not match:
        raise ValueError(f"missing CEL diagnostics: {path}")
    diagnostics = {}
    for item in match.group(1).split(", "):
        key, separator, value = item.partition(": ")
        if separator:
            diagnostics[key] = float(value)
    if not diagnostics:
        raise ValueError(f"empty CEL diagnostics: {path}")
    return diagnostics


def first_existing(paths: tuple[Path, ...]) -> Path:
    for path in paths:
        if path.is_file():
            return path
    return paths[0]


def tensor_diff(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> dict:
    common = sorted(set(left) & set(right))
    changed = []
    max_abs_diff = 0.0
    for key in common:
        if left[key].shape != right[key].shape:
            changed.append(key)
            continue
        diff = (left[key].detach().cpu().float() - right[key].detach().cpu().float()).abs().max().item()
        max_abs_diff = max(max_abs_diff, diff)
        if diff > 0:
            changed.append(key)
    return {
        "left_tensors": len(left),
        "right_tensors": len(right),
        "common_tensors": len(common),
        "changed_tensors": len(changed),
        "max_abs_diff": max_abs_diff,
        "all_tensors_equal": set(left) == set(right) and not changed,
    }


def load_state(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        return load_file(path)
    return torch.load(path, map_location="cpu", weights_only=True)


def floating_state_dtypes(state: dict[str, torch.Tensor]) -> list[str]:
    return sorted({str(value.dtype) for value in state.values() if value.is_floating_point()})


def require_files(paths: list[Path]) -> None:
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing Stage 2 artifacts: " + ", ".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_id")
    parser.add_argument("--debug-layout", action="store_true")
    args = parser.parse_args()
    candidate_id = args.candidate_id
    baseline_metrics = first_existing(BASELINE_METRIC_CANDIDATES)
    stage1_metrics = first_existing(STAGE1_METRIC_CANDIDATES)

    bootstrap_name = f"cel_{candidate_id}_bootstrap_qwen3_1.7b"
    warmup_name = f"cel_{candidate_id}_calibrator_warmup_qwen3_1.7b"
    final_name = f"cel_{candidate_id}_joint_qwen3_1.7b"
    model_names = {
        "bootstrap": bootstrap_name,
        "calibrator_warmup": warmup_name,
        "joint": final_name,
    }
    checkpoint_paths = {
        stage: {
            "adapter": SAVED_MODELS / model_name / "adapter_model.safetensors",
            "selector": SAVED_MODELS / model_name / "cel_selector.pt",
            "environment": SAVED_MODELS / model_name / "cel_environment.pt",
            "manifest": SAVED_MODELS / model_name / "cel_stage2_manifest.json",
            "calibrator": SAVED_MODELS / model_name / "cel_calibrator.pt",
        }
        for stage, model_name in model_names.items()
    }
    if args.debug_layout:
        result_root = RESULTS / "debug" / candidate_id
        logs = {
            "bootstrap": result_root / "stages/bootstrap.stdout.log",
            "calibrator_warmup": result_root / "stages/calibrator_warmup.stdout.log",
            "joint": result_root / "joint.stdout.log",
        }
        metrics_path = result_root / "joint/metrics" / f"metrics_{final_name}.txt"
        qual_path = result_root / "joint/qual" / f"qual_{final_name}.csv"
        kcs_path = result_root / "joint/kcs" / f"kcs_{final_name}.json"
        audit_dir = result_root
    else:
        logs = {
            "bootstrap": RESULTS / candidate_id / "stages/bootstrap.stdout.log",
            "calibrator_warmup": RESULTS / candidate_id / "stages/calibrator_warmup.stdout.log",
            "joint": RESULTS / f"{candidate_id}.stdout.log",
        }
        metrics_path = RESULTS / "metrics" / f"metrics_{final_name}.txt"
        qual_path = RESULTS / "qual" / f"qual_{final_name}.csv"
        kcs_path = RESULTS / "kcs" / f"kcs_{final_name}.json"
        audit_dir = RESULTS / candidate_id

    required = [baseline_metrics, stage1_metrics, metrics_path, qual_path, kcs_path, *logs.values()]
    for stage, paths in checkpoint_paths.items():
        required.extend([paths["adapter"], paths["selector"], paths["environment"], paths["manifest"]])
        if stage != "bootstrap":
            required.append(paths["calibrator"])
    require_files(required)

    manifests = {
        stage: json.loads(paths["manifest"].read_text(encoding="utf-8"))
        for stage, paths in checkpoint_paths.items()
    }
    log_text = {stage: path.read_text(encoding="utf-8", errors="replace") for stage, path in logs.items()}
    forbidden_fragments = (
        "lmkt_qwen3_1.7b_recert_20260620",
        "cel_task_conditioned_lastlayer_v1_qwen3_1.7b",
        "cel_task_conditioned_lastlayer_v26_",
    )
    forbidden_hits = {}
    for stage, text in log_text.items():
        allowed_names = set(model_names.values())
        forbidden_hits[stage] = [
            fragment
            for fragment in forbidden_fragments
            if fragment in text
        ]

    expected_parent = {
        "bootstrap": None,
        "calibrator_warmup": bootstrap_name,
        "joint": warmup_name,
    }
    manifest_chain_ok = all(
        manifest.get("candidate_id") == candidate_id
        and manifest.get("phase") == stage
        and manifest.get("parent_model_name") == expected_parent[stage]
        and manifest.get("fresh_init_required") is True
        and manifest.get("cel_hook_timing") == "pre_block"
        for stage, manifest in manifests.items()
    )
    config_keys = (
        "candidate_id", "env_mode", "env_beta", "env_split_mode", "env_topk_ratio",
        "env_hidden_dim", "env_num_layers", "env_num_heads", "env_ffn_dim",
        "env_output_postprocess", "env_output_ratio", "env_output_init_std",
        "cel_mode", "cel_layer_idx", "cel_hook_site", "cel_hook_timing", "cel_gamma",
        "cel_injection_variant", "cel_application_mode", "model_init_seed",
    )
    config_consistency_ok = all(
        manifests[stage].get(key) == manifests["bootstrap"].get(key)
        for stage in ("calibrator_warmup", "joint")
        for key in config_keys
    )
    provenance_ok = (
        "Initializing trainable model with new LoRA adapters" in log_text["bootstrap"]
        and bootstrap_name in log_text["calibrator_warmup"]
        and warmup_name in log_text["joint"]
        and all("Testing:" not in log_text[stage] for stage in ("bootstrap", "calibrator_warmup"))
        and "Skipping test after training as requested" in log_text["bootstrap"]
        and "Skipping test after training as requested" in log_text["calibrator_warmup"]
        and "Testing:" in log_text["joint"]
        and not any(forbidden_hits.values())
        and manifest_chain_ok
        and config_consistency_ok
    )

    component_diffs = {}
    for transition, left_stage, right_stage in (
        ("bootstrap_to_warmup", "bootstrap", "calibrator_warmup"),
        ("warmup_to_joint", "calibrator_warmup", "joint"),
    ):
        component_diffs[transition] = {
            component: tensor_diff(
                load_state(checkpoint_paths[left_stage][component]),
                load_state(checkpoint_paths[right_stage][component]),
            )
            for component in ("adapter", "selector", "environment")
        }
        if transition == "warmup_to_joint":
            component_diffs[transition]["calibrator"] = tensor_diff(
                load_state(checkpoint_paths[left_stage]["calibrator"]),
                load_state(checkpoint_paths[right_stage]["calibrator"]),
            )

    warmup_freeze_ok = all(
        diff["all_tensors_equal"]
        for diff in component_diffs["bootstrap_to_warmup"].values()
    )
    joint_selector_changed = not component_diffs["warmup_to_joint"]["selector"]["all_tensors_equal"]
    joint_manifest = manifests["joint"]
    env_mode = joint_manifest["env_mode"]
    joint_environment_changed = (
        True if env_mode == "shuffle"
        else not component_diffs["warmup_to_joint"]["environment"]["all_tensors_equal"]
    )
    joint_update_ok = (
        not component_diffs["warmup_to_joint"]["adapter"]["all_tensors_equal"]
        and joint_selector_changed
        and joint_environment_changed
        and not component_diffs["warmup_to_joint"]["calibrator"]["all_tensors_equal"]
    )

    state_dtypes = {
        stage: {
            component: floating_state_dtypes(load_state(paths[component]))
            for component in ("selector", "environment")
        }
        for stage, paths in checkpoint_paths.items()
    }
    selector_fp32_ok = all(
        dtypes == ["torch.float32"]
        for stage in state_dtypes.values()
        for component, dtypes in stage.items()
        if component == "selector"
    )
    environment_fp32_ok = env_mode == "shuffle" or all(
        stage["environment"] == ["torch.float32"]
        for stage in state_dtypes.values()
    )

    metrics = {
        "baseline": parse_metrics(baseline_metrics),
        "stage1_v26": parse_metrics(stage1_metrics),
        "stage2": parse_metrics(metrics_path),
    }
    diagnostics = parse_diagnostics(metrics_path)
    diagnostics_finite = all(math.isfinite(value) for value in diagnostics.values())
    deltas = {
        reference: {
            key: metrics["stage2"][key] - metrics[reference][key]
            for key in ("overall_acc", "overall_auc", "final_acc", "final_auc")
        }
        for reference in ("baseline", "stage1_v26")
    }
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "candidate_id": candidate_id,
        "models": model_names,
        "manifests": manifests,
        "provenance": {
            "manifest_chain_ok": manifest_chain_ok,
            "config_consistency_ok": config_consistency_ok,
            "forbidden_initializer_hits": forbidden_hits,
            "provenance_ok": provenance_ok,
        },
        "parameter_differences": component_diffs,
        "warmup_freeze_ok": warmup_freeze_ok,
        "joint_selector_changed": joint_selector_changed,
        "joint_environment_changed": joint_environment_changed,
        "joint_update_ok": joint_update_ok,
        "state_dtypes": state_dtypes,
        "selector_fp32_ok": selector_fp32_ok,
        "environment_fp32_ok": environment_fp32_ok,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "diagnostics_finite": diagnostics_finite,
        "deltas": deltas,
        "beats_stage1_overall_auc": deltas["stage1_v26"]["overall_auc"] > 0,
        "audit_pass": (
            provenance_ok
            and warmup_freeze_ok
            and joint_update_ok
            and selector_fp32_ok
            and environment_fp32_ok
            and diagnostics_finite
        ),
        "artifact_hashes": {
            stage: {
                component: sha256(path)
                for component, path in paths.items()
                if path.is_file()
            }
            for stage, paths in checkpoint_paths.items()
        },
    }

    audit_dir.mkdir(parents=True, exist_ok=True)
    json_path = audit_dir / "AUDIT.json"
    md_path = audit_dir / "AUDIT.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    current = metrics["stage2"]
    lines = [
        f"# Stage 2 Candidate Audit: {candidate_id}", "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Audit pass: `{report['audit_pass']}`",
        f"- Environment mode: `{env_mode}`",
        f"- Environment beta / top-k ratio: `{joint_manifest['env_beta']} / {joint_manifest['env_topk_ratio']}`",
        f"- Environment output postprocess: `{joint_manifest.get('env_output_postprocess', 'none')}`",
        f"- Environment output ratio / init std: "
        f"`{joint_manifest.get('env_output_ratio', 'not_recorded')} / "
        f"{joint_manifest.get('env_output_init_std', 'not_recorded')}`",
        f"- Manifest/provenance pass: `{provenance_ok}`",
        f"- Warmup froze A+B+LoRA: `{warmup_freeze_ok}`",
        f"- Joint selector changed: `{joint_selector_changed}`",
        f"- Joint environment changed or parameter-free: `{joint_environment_changed}`",
        f"- Selector checkpoints are FP32: `{selector_fp32_ok}`",
        f"- Learned environment checkpoints are FP32: `{environment_fp32_ok}`",
        f"- Final diagnostics are finite: `{diagnostics_finite}`",
        "", "## Metrics", "",
        "| Model | Overall Acc | Overall AUC | Final Acc | Final AUC |",
        "|---|---:|---:|---:|---:|",
    ]
    display_labels = {
        "baseline": "baseline",
        "stage1_v26": "A-module reference",
        "stage2": "historical Stage 2 candidate",
    }
    for label in ("baseline", "stage1_v26", "stage2"):
        row = metrics[label]
        lines.append(
            f"| {display_labels[label]} | {row['overall_acc']:.2f} | {row['overall_auc']:.2f} | "
            f"{row['final_acc']:.2f} | {row['final_auc']:.2f} |"
        )
    lines.extend(["", "## Parameter Changes", ""])
    for transition, components in component_diffs.items():
        for component, diff in components.items():
            lines.append(f"- {transition} `{component}` changed tensors: `{diff['changed_tensors']}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path.read_text(encoding="utf-8"))
    if not report["audit_pass"]:
        raise SystemExit("Stage 2 candidate audit failed")


if __name__ == "__main__":
    main()
