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
RESULTS = ROOT / "results/cel_stage2_environment/dual_path"
EXPECTED_BASE_MODEL = "/home/user4/.cache/huggingface/hub/qwen/Qwen3-1.7B"
EXPECTED_MODEL_SEED = 1221
COMPLEMENT_RELATIVE_RMS_LIMIT = 0.01
WEIGHT_COMPLEMENT_MAX_ABS_LIMIT = 0.01


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_state(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        return load_file(path)
    return torch.load(path, map_location="cpu", weights_only=True)


def tensor_diff(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> dict:
    common = sorted(set(left) & set(right))
    changed = []
    max_abs_diff = 0.0
    for key in common:
        if left[key].shape != right[key].shape:
            changed.append(key)
            continue
        difference = (left[key].float() - right[key].float()).abs().max().item()
        max_abs_diff = max(max_abs_diff, difference)
        if difference > 0:
            changed.append(key)
    return {
        "left_tensors": len(left),
        "right_tensors": len(right),
        "common_tensors": len(common),
        "changed_tensors": len(changed),
        "max_abs_diff": max_abs_diff,
        "all_tensors_equal": set(left) == set(right) and not changed,
    }


def floating_dtypes(state: dict[str, torch.Tensor]) -> list[str]:
    return sorted({str(value.dtype) for value in state.values() if value.is_floating_point()})


def parse_path_report(path: Path, expected_split: str) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema_version") != 1:
        raise ValueError(f"unexpected Stage 2 path-report schema in {path}")
    if report.get("evaluation_split") != expected_split:
        raise ValueError(
            f"unexpected evaluation split in {path}: "
            f"{report.get('evaluation_split')!r}"
        )
    if report.get("primary_path") != "mixed":
        raise ValueError(f"unexpected primary path in {path}")
    sample_count = report.get("sample_count")
    if not isinstance(sample_count, int) or sample_count <= 0:
        raise ValueError(f"invalid sample count in {path}: {sample_count!r}")
    required_diagnostics = (
        "stage2_loss_r",
        "stage2_loss_m",
        "stage2_loss_cons",
        "stage2_lambda_r_effective",
        "stage2_lambda_m_effective",
        "stage2_lambda_cons_effective",
        "stage2_effective_beta",
        "p_r_p_m_abs_mean",
        "a_saturation_frac",
        "w_r_mean",
        "w_n_mean",
        "h_r_rms",
        "h_n_rms",
        "h_nb_rms",
        "beta_h_nb_rms",
        "dual_path_finite_frac",
        "complement_max_abs_error",
        "complement_relative_rms_error",
        "weight_complement_max_abs_error",
    )
    diagnostics = report.get("diagnostics", {})
    missing_diagnostics = [key for key in required_diagnostics if key not in diagnostics]
    if missing_diagnostics:
        raise ValueError(
            f"missing Stage 2 diagnostics in {path}: {', '.join(missing_diagnostics)}"
        )
    if not all(math.isfinite(float(diagnostics[key])) for key in required_diagnostics):
        raise ValueError(f"non-finite Stage 2 diagnostic in {path}")
    if float(diagnostics["dual_path_finite_frac"]) < 0.999:
        raise ValueError(f"insufficient finite dual-path values in {path}")
    for path_name in ("evidence", "mixed", "non_evidence"):
        if path_name not in report.get("paths", {}):
            raise ValueError(f"missing {path_name} path in {path}")
        for split_name in ("overall", "final", "non_final"):
            metrics = report["paths"][path_name].get(split_name, {})
            if not metrics:
                raise ValueError(f"missing {path_name}/{split_name} metrics in {path}")
            if not all(math.isfinite(float(value)) for value in metrics.values()):
                raise ValueError(f"non-finite {path_name}/{split_name} metric in {path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("round_id")
    parser.add_argument("candidate_id")
    parser.add_argument("--require-validation", action="store_true")
    parser.add_argument("--require-test", action="store_true")
    args = parser.parse_args()

    phase_names = ("a_bootstrap", "calibrator_warmup", "a_joint", "b_warmup", "joint")
    model_names = {
        phase: f"cel_{args.candidate_id}_{phase}_qwen3_1.7b"
        for phase in phase_names
    }
    paths = {
        phase: {
            "adapter": SAVED_MODELS / model_name / "adapter_model.safetensors",
            "selector": SAVED_MODELS / model_name / "cel_selector.pt",
            "calibrator": SAVED_MODELS / model_name / "cel_calibrator.pt",
            "environment": SAVED_MODELS / model_name / "cel_environment.pt",
            "manifest": SAVED_MODELS / model_name / "cel_stage2_manifest.json",
        }
        for phase, model_name in model_names.items()
    }
    round_root = RESULTS / args.round_id
    logs = {
        "a_bootstrap": round_root / "stages/a_bootstrap.stdout.log",
        "calibrator_warmup": round_root / "stages/calibrator_warmup.stdout.log",
        "a_joint": round_root / "stages/a_joint.stdout.log",
        "b_warmup": round_root / "stages/b_warmup.stdout.log",
        "joint": round_root / "joint.stdout.log",
    }
    missing = [
        str(path.relative_to(ROOT))
        for phase in phase_names
        for key, path in paths[phase].items()
        if key in ("adapter", "selector", "manifest") and not path.is_file()
    ]
    for phase in ("calibrator_warmup", "a_joint", "b_warmup", "joint"):
        if not paths[phase]["calibrator"].is_file():
            missing.append(str(paths[phase]["calibrator"].relative_to(ROOT)))
    for phase in ("b_warmup", "joint"):
        if not paths[phase]["environment"].is_file():
            missing.append(str(paths[phase]["environment"].relative_to(ROOT)))
    missing.extend(
        str(path.relative_to(ROOT)) for path in logs.values() if not path.is_file()
    )
    if missing:
        raise FileNotFoundError("missing dual-path artifacts: " + ", ".join(missing))

    manifests = {
        phase: json.loads(paths[phase]["manifest"].read_text(encoding="utf-8"))
        for phase in phase_names
    }
    expected_parent = {
        "a_bootstrap": None,
        "calibrator_warmup": model_names["a_bootstrap"],
        "a_joint": model_names["calibrator_warmup"],
        "b_warmup": model_names["a_joint"],
        "joint": model_names["b_warmup"],
    }
    manifest_chain_ok = all(
        manifests[phase].get("candidate_id") == args.candidate_id
        and manifests[phase].get("phase") == phase
        and manifests[phase].get("model_name") == model_names[phase]
        and manifests[phase].get("parent_model_name") == expected_parent[phase]
        and manifests[phase].get("fresh_init_required") is True
        and manifests[phase].get("objective") == "dual_path_hr_hm_js"
        and manifests[phase].get("cel_hook_timing") == "pre_block"
        for phase in phase_names
    )
    identity_ok = all(
        manifests[phase].get("base_model") == EXPECTED_BASE_MODEL
        and manifests[phase].get("model_init_seed") == EXPECTED_MODEL_SEED
        for phase in phase_names
    )
    config_keys = (
        "candidate_id", "objective", "env_mode", "env_beta", "env_split_mode",
        "env_hidden_dim", "env_num_layers", "env_num_heads", "env_ffn_dim",
        "env_output_postprocess", "env_output_ratio", "env_output_init_std",
        "lambda_r", "lambda_m", "lambda_cons", "beta_start_ratio",
        "consistency_ramp_fraction", "cel_mode", "cel_selector_hidden_dim",
        "cel_drop", "cel_layer_idx", "cel_hook_site", "cel_hook_timing",
        "cel_gamma", "cel_injection_variant", "cel_application_mode",
        "cel_env_drop", "cel_env_shuffle_seed", "model_init_seed",
    )
    config_consistency_ok = all(
        manifests[phase].get(key) == manifests["a_bootstrap"].get(key)
        for phase in phase_names[1:]
        for key in config_keys
    )

    log_text = {phase: logs[phase].read_text(encoding="utf-8", errors="replace") for phase in phase_names}
    forbidden_fragments = (
        "lmkt_qwen3_1.7b_recert_20260620",
        "cel_task_conditioned_lastlayer_v1_qwen3_1.7b",
        "cel_task_conditioned_lastlayer_v26_",
    )
    forbidden_hits = {
        phase: [fragment for fragment in forbidden_fragments if fragment in text]
        for phase, text in log_text.items()
    }
    provenance_ok = (
        "Initializing trainable model with new LoRA adapters" in log_text["a_bootstrap"]
        and all("Testing:" not in text for text in log_text.values())
        and all("Skipping test after training" in text for text in log_text.values())
        and not any(forbidden_hits.values())
        and manifest_chain_ok
        and identity_ok
        and config_consistency_ok
    )

    component_diffs = {}
    transitions = (
        ("a_bootstrap_to_calibrator_warmup", "a_bootstrap", "calibrator_warmup"),
        ("calibrator_warmup_to_a_joint", "calibrator_warmup", "a_joint"),
        ("a_joint_to_b_warmup", "a_joint", "b_warmup"),
        ("b_warmup_to_joint", "b_warmup", "joint"),
    )
    for transition, left_phase, right_phase in transitions:
        components = ("adapter", "selector", "calibrator")
        if right_phase in ("b_warmup", "joint"):
            components += ("environment",)
        component_diffs[transition] = {}
        for component in components:
            left_path = paths[left_phase][component]
            right_path = paths[right_phase][component]
            if not left_path.is_file() or not right_path.is_file():
                component_diffs[transition][component] = {"not_applicable": True}
            else:
                component_diffs[transition][component] = tensor_diff(
                    load_state(left_path), load_state(right_path)
                )

    warmup_freeze_ok = all(
        component_diffs["a_bootstrap_to_calibrator_warmup"][component]["all_tensors_equal"]
        for component in ("adapter", "selector")
    )
    b_warmup_freeze_ok = all(
        component_diffs["a_joint_to_b_warmup"][component]["all_tensors_equal"]
        for component in ("adapter", "selector", "calibrator")
    )
    a_joint_updates = component_diffs["calibrator_warmup_to_a_joint"]
    a_joint_component_updates = {
        component: not a_joint_updates[component]["all_tensors_equal"]
        for component in ("adapter", "selector", "calibrator")
    }
    a_joint_update_ok = all(a_joint_component_updates.values())
    b_updates = component_diffs["b_warmup_to_joint"]
    joint_component_updates = {
        component: not b_updates[component]["all_tensors_equal"]
        for component in ("adapter", "selector", "environment", "calibrator")
    }
    joint_update_ok = all(joint_component_updates.values())

    selector_fp32_ok = all(
        floating_dtypes(load_state(paths[phase]["selector"])) == ["torch.float32"]
        for phase in phase_names
    )
    environment_fp32_ok = all(
        floating_dtypes(load_state(paths[phase]["environment"])) == ["torch.float32"]
        for phase in ("b_warmup", "joint")
    )

    evaluation_reports = {}
    for split, required in (("validation", args.require_validation), ("test", args.require_test)):
        report_path = round_root / f"evaluations/{split}/metrics/stage2_paths_{model_names['joint']}.json"
        if required and not report_path.is_file():
            raise FileNotFoundError(f"missing {split} path report: {report_path}")
        if report_path.is_file():
            evaluation_reports[split] = parse_path_report(report_path, split)

    complement_contract_ok = all(
        float(evaluation["diagnostics"]["complement_relative_rms_error"])
        <= COMPLEMENT_RELATIVE_RMS_LIMIT
        and float(evaluation["diagnostics"]["weight_complement_max_abs_error"])
        <= WEIGHT_COMPLEMENT_MAX_ABS_LIMIT
        for evaluation in evaluation_reports.values()
    )

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "round_id": args.round_id,
        "candidate_id": args.candidate_id,
        "models": model_names,
        "manifests": manifests,
        "provenance": {
            "manifest_chain_ok": manifest_chain_ok,
            "identity_ok": identity_ok,
            "config_consistency_ok": config_consistency_ok,
            "forbidden_initializer_hits": forbidden_hits,
            "provenance_ok": provenance_ok,
        },
        "identity_ok": identity_ok,
        "parameter_differences": component_diffs,
        "warmup_freeze_ok": warmup_freeze_ok,
        "b_warmup_freeze_ok": b_warmup_freeze_ok,
        "a_joint_component_updates": a_joint_component_updates,
        "a_joint_update_ok": a_joint_update_ok,
        "joint_component_updates": joint_component_updates,
        "joint_update_ok": joint_update_ok,
        "selector_fp32_ok": selector_fp32_ok,
        "environment_fp32_ok": environment_fp32_ok,
        "complement_contract_ok": complement_contract_ok,
        "evaluation_reports": evaluation_reports,
        "artifact_hashes": {
            phase: {
                component: sha256(path)
                for component, path in paths[phase].items()
                if path.is_file()
            }
            for phase in phase_names
        },
    }
    report["audit_pass"] = all((
        provenance_ok,
        identity_ok,
        warmup_freeze_ok,
        b_warmup_freeze_ok,
        a_joint_update_ok,
        joint_update_ok,
        selector_fp32_ok,
        environment_fp32_ok,
        complement_contract_ok,
    ))

    audit_dir = round_root / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "AUDIT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"# Dual-path Stage 2 audit: {args.round_id}",
        "",
        f"- Candidate: `{args.candidate_id}`",
        f"- Audit pass: `{report['audit_pass']}`",
        f"- Provenance pass: `{provenance_ok}`",
        f"- Raw-base/seed identity: `{identity_ok}`",
        f"- A warmup freeze: `{warmup_freeze_ok}`",
        f"- B warmup freeze: `{b_warmup_freeze_ok}`",
        f"- A joint update: `{a_joint_update_ok}`",
        "- A joint component updates: "
        + ", ".join(
            f"`{component}={changed}`"
            for component, changed in a_joint_component_updates.items()
        ),
        f"- Final joint update: `{joint_update_ok}`",
        "- Final joint component updates: "
        + ", ".join(
            f"`{component}={changed}`"
            for component, changed in joint_component_updates.items()
        ),
        f"- Selector FP32: `{selector_fp32_ok}`",
        f"- B FP32: `{environment_fp32_ok}`",
        f"- Complement contract: `{complement_contract_ok}`",
        "",
        "## Evaluations",
        "",
    ]
    for split, evaluation in evaluation_reports.items():
        lines.append(f"### {split}")
        mixed = evaluation["paths"]["mixed"]
        evidence = evaluation["paths"]["evidence"]
        reversal = evaluation["paths"]["non_evidence"]
        lines.append(
            f"- mixed Overall Acc/AUC: `{mixed['overall']['acc']:.2f}/{mixed['overall']['auc']:.2f}`"
        )
        lines.append(
            f"- evidence Overall Acc/AUC: `{evidence['overall']['acc']:.2f}/{evidence['overall']['auc']:.2f}`"
        )
        lines.append(
            f"- reversal Overall Acc/AUC: `{reversal['overall']['acc']:.2f}/{reversal['overall']['auc']:.2f}`"
        )
        lines.append("")
    (audit_dir / "AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print((audit_dir / "AUDIT.md").read_text(encoding="utf-8"))
    if not report["audit_pass"]:
        raise SystemExit("dual-path Stage 2 audit failed")


if __name__ == "__main__":
    main()
