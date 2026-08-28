#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

import torch
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[2]
SAVED_MODELS = ROOT / "saved_models"
RESULTS = ROOT / "results/cel_stage1_last_layer"

BOOTSTRAP_MODEL = "cel_task_conditioned_lastlayer_v26_bootstrap_qwen3_1.7b"
WARMUP_MODEL = "cel_task_conditioned_lastlayer_v26_calibrator_warmup_qwen3_1.7b"
FINAL_MODEL = "cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b"
BASELINE_MODEL = "lmkt_qwen3_1.7b_recert_20260620"

LOGS = {
    "bootstrap": RESULTS / "task_conditioned_v26_bootstrap.stdout.log",
    "calibrator_warmup": RESULTS / "task_conditioned_v26_calibrator_warmup.stdout.log",
    "joint": RESULTS / "task_conditioned_v26_selftrained_joint.stdout.log",
}

BASELINE_METRIC_CANDIDATES = (
    ROOT / f"results/baseline/metrics/metrics_{BASELINE_MODEL}.txt",
    ROOT / f"results/baseline_recert/metrics/metrics_{BASELINE_MODEL}.txt",
)


def resolve_existing_path(candidates: tuple[Path, ...]) -> Path:
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


METRICS = {
    "baseline": resolve_existing_path(BASELINE_METRIC_CANDIDATES),
    "joint": RESULTS / "metrics" / f"metrics_{FINAL_MODEL}.txt",
}

FORBIDDEN_INITIALIZERS = (
    "lmkt_qwen3_1.7b_recert_20260620",
    "cel_task_conditioned_lastlayer_v1_qwen3_1.7b",
    "cel_task_conditioned_lastlayer_v26_fullv1_biaswarmup_joint_tinylr_qwen3_1.7b",
)

LOSS_RE = re.compile(r"^Loss:\s+([0-9.]+)$", re.MULTILINE)
OVERALL_RE = re.compile(
    r"Overall.*?\n.*?\nAcc:\s+([0-9.]+),\s+AUC:\s+([0-9.]+)",
    re.DOTALL,
)
FINAL_RE = re.compile(
    r"Final Turn.*?\n.*?\nAcc:\s+([0-9.]+),\s+AUC:\s+([0-9.]+)",
    re.DOTALL,
)


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


def load_torch_state(path: Path) -> dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu", weights_only=True)


def checkpoint_files(model_name: str) -> dict[str, Path]:
    model_dir = SAVED_MODELS / model_name
    return {
        "adapter": model_dir / "adapter_model.safetensors",
        "selector": model_dir / "cel_selector.pt",
        "calibrator": model_dir / "cel_calibrator.pt",
    }


def require_files(paths: list[Path]) -> None:
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required v26 artifacts: {', '.join(missing)}")


def compare_metrics(current: dict, baseline: dict) -> dict:
    deltas = {
        key: current[key] - baseline[key]
        for key in ("overall_acc", "overall_auc", "final_acc", "final_auc")
    }
    return {
        "deltas_vs_baseline": deltas,
        "beats_baseline_overall_gate": deltas["overall_acc"] > 0 and deltas["overall_auc"] > 0,
        "beats_baseline_final_gate": deltas["final_acc"] > 0 and deltas["final_auc"] > 0,
        "beats_baseline_all_four": all(value > 0 for value in deltas.values()),
    }


def format_delta(value: float) -> str:
    return f"{value:+.2f}"


def main() -> None:
    bootstrap = checkpoint_files(BOOTSTRAP_MODEL)
    warmup = checkpoint_files(WARMUP_MODEL)
    joint = checkpoint_files(FINAL_MODEL)
    required = [
        bootstrap["adapter"], bootstrap["selector"],
        warmup["adapter"], warmup["selector"], warmup["calibrator"],
        joint["adapter"], joint["selector"], joint["calibrator"],
        *LOGS.values(), *METRICS.values(),
    ]
    require_files(required)

    log_text = {name: path.read_text(encoding="utf-8", errors="replace") for name, path in LOGS.items()}
    forbidden_hits = {
        stage: [name for name in FORBIDDEN_INITIALIZERS if name in text]
        for stage, text in log_text.items()
    }
    bootstrap_from_base = "Initializing trainable model with new LoRA adapters" in log_text["bootstrap"]
    warmup_from_bootstrap = BOOTSTRAP_MODEL in log_text["calibrator_warmup"]
    joint_from_warmup = (
        log_text["joint"].count(WARMUP_MODEL) >= 3
        and "CEL calibrator warm-start loaded" in log_text["joint"]
    )
    intermediate_test_isolation_ok = all(
        "Testing:" not in log_text[stage]
        and "Skipping test after training as requested" in log_text[stage]
        for stage in ("bootstrap", "calibrator_warmup")
    )
    final_test_completed = "Testing:" in log_text["joint"]

    bootstrap_to_warmup = {
        "adapter": tensor_diff(load_file(bootstrap["adapter"]), load_file(warmup["adapter"])),
        "selector": tensor_diff(load_torch_state(bootstrap["selector"]), load_torch_state(warmup["selector"])),
    }
    warmup_to_joint = {
        "adapter": tensor_diff(load_file(warmup["adapter"]), load_file(joint["adapter"])),
        "selector": tensor_diff(load_torch_state(warmup["selector"]), load_torch_state(joint["selector"])),
        "calibrator": tensor_diff(load_torch_state(warmup["calibrator"]), load_torch_state(joint["calibrator"])),
    }

    parsed_metrics = {name: parse_metrics(path) for name, path in METRICS.items()}
    comparison = compare_metrics(parsed_metrics["joint"], parsed_metrics["baseline"])
    provenance_ok = (
        bootstrap_from_base
        and warmup_from_bootstrap
        and joint_from_warmup
        and intermediate_test_isolation_ok
        and final_test_completed
        and not any(forbidden_hits.values())
    )
    warmup_freeze_ok = (
        bootstrap_to_warmup["adapter"]["all_tensors_equal"]
        and bootstrap_to_warmup["selector"]["all_tensors_equal"]
    )
    joint_update_ok = any(
        not values["all_tensors_equal"]
        for values in warmup_to_joint.values()
    )

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "models": {
            "bootstrap": BOOTSTRAP_MODEL,
            "calibrator_warmup": WARMUP_MODEL,
            "joint": FINAL_MODEL,
            "baseline_comparison_only": BASELINE_MODEL,
        },
        "provenance": {
            "bootstrap_from_qwen_base": bootstrap_from_base,
            "warmup_from_v26_bootstrap": warmup_from_bootstrap,
            "joint_from_v26_warmup": joint_from_warmup,
            "intermediate_stages_did_not_test": intermediate_test_isolation_ok,
            "final_joint_test_completed": final_test_completed,
            "forbidden_initializer_hits": forbidden_hits,
            "provenance_ok": provenance_ok,
        },
        "checkpoint_hashes": {
            stage: {kind: sha256(path) for kind, path in files.items() if path.is_file()}
            for stage, files in (("bootstrap", bootstrap), ("calibrator_warmup", warmup), ("joint", joint))
        },
        "parameter_differences": {
            "bootstrap_to_calibrator_warmup": bootstrap_to_warmup,
            "calibrator_warmup_to_joint": warmup_to_joint,
            "warmup_freeze_ok": warmup_freeze_ok,
            "joint_update_ok": joint_update_ok,
        },
        "metrics": parsed_metrics,
        "comparison": comparison,
        "audit_pass": provenance_ok and warmup_freeze_ok and joint_update_ok,
    }

    json_path = RESULTS / "V26_SELFTRAINED_AUDIT.json"
    md_path = RESULTS / "V26_SELFTRAINED_AUDIT.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    current = parsed_metrics["joint"]
    baseline = parsed_metrics["baseline"]
    deltas = comparison["deltas_vs_baseline"]
    lines = [
        "# V26 Self-Trained Audit",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Audit pass: `{report['audit_pass']}`",
        f"- Provenance pass: `{provenance_ok}`",
        f"- Intermediate stages skipped test: `{intermediate_test_isolation_ok}`",
        f"- Final joint test completed: `{final_test_completed}`",
        f"- Warmup kept LoRA/selector frozen: `{warmup_freeze_ok}`",
        f"- Joint phase changed at least one trainable component: `{joint_update_ok}`",
        "",
        "## Initialization Chain",
        "",
        f"- Bootstrap: Qwen3 base -> `{BOOTSTRAP_MODEL}`",
        f"- Calibrator warmup: `{BOOTSTRAP_MODEL}` -> `{WARMUP_MODEL}`",
        f"- Strict joint: `{WARMUP_MODEL}` -> `{FINAL_MODEL}`",
        "- Historical baseline and v1 checkpoints are comparison-only and were not loaded.",
        "",
        "## Metric Comparison",
        "",
        "| Model | Overall Acc | Overall AUC | Final Acc | Final AUC |",
        "|---|---:|---:|---:|---:|",
        f"| baseline | {baseline['overall_acc']:.2f} | {baseline['overall_auc']:.2f} | {baseline['final_acc']:.2f} | {baseline['final_auc']:.2f} |",
        f"| A-module joint | {current['overall_acc']:.2f} | {current['overall_auc']:.2f} | {current['final_acc']:.2f} | {current['final_auc']:.2f} |",
        f"| delta | {format_delta(deltas['overall_acc'])} | {format_delta(deltas['overall_auc'])} | {format_delta(deltas['final_acc'])} | {format_delta(deltas['final_auc'])} |",
        "",
        f"- Beats baseline Overall Acc + AUC gate: `{comparison['beats_baseline_overall_gate']}`",
        f"- Beats baseline Final Acc + AUC gate: `{comparison['beats_baseline_final_gate']}`",
        f"- Beats baseline on all four metrics: `{comparison['beats_baseline_all_four']}`",
        "",
        "## Parameter Audit",
        "",
        f"- Bootstrap -> warmup adapter equal: `{bootstrap_to_warmup['adapter']['all_tensors_equal']}`",
        f"- Bootstrap -> warmup selector equal: `{bootstrap_to_warmup['selector']['all_tensors_equal']}`",
        f"- Warmup -> joint adapter changed tensors: `{warmup_to_joint['adapter']['changed_tensors']}`",
        f"- Warmup -> joint selector changed tensors: `{warmup_to_joint['selector']['changed_tensors']}`",
        f"- Warmup -> joint calibrator changed tensors: `{warmup_to_joint['calibrator']['changed_tensors']}`",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path.read_text(encoding="utf-8"))
    if not report["audit_pass"]:
        raise SystemExit("v26 self-trained artifact audit failed")


if __name__ == "__main__":
    main()
