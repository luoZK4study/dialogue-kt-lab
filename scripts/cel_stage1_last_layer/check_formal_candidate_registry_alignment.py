#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

from formal_candidate_registry import FORMAL_CANDIDATE_LAUNCH_KEYS, FORMAL_CANDIDATES


def parse_shell_registry(path: Path) -> dict[str, dict[str, str]]:
    text = path.read_text(encoding="utf-8")

    keys_match = re.search(r"FORMAL_CANDIDATE_KEYS=\(([^)]*)\)", text, re.S)
    if not keys_match:
        raise ValueError("FORMAL_CANDIDATE_KEYS not found in shell registry")
    shell_keys = [token for token in keys_match.group(1).split() if token.strip()]

    maps = {
        "FORMAL_MODEL_NAME": "model_name",
        "FORMAL_MODEL_PATTERN": "model_pattern",
        "FORMAL_RUN_SCRIPT_NAME": "run_script_name",
        "FORMAL_RUN_SCRIPT_REL": "run_script_rel",
        "FORMAL_STDOUT_LOG_NAME": "stdout_log_name",
        "FORMAL_STDOUT_LOG_REL": "stdout_log_rel",
        "FORMAL_METRICS_REL": "metrics_rel",
        "FORMAL_DEFAULT_GPU_ID": "default_gpu_id",
    }

    parsed: dict[str, dict[str, str]] = {key: {} for key in shell_keys}
    for assoc_name, field_name in maps.items():
        pattern = rf"declare -A {assoc_name}=\((.*?)\n\)"
        match = re.search(pattern, text, re.S)
        if not match:
            raise ValueError(f"{assoc_name} not found in shell registry")
        block = match.group(1)
        entries = re.findall(r'\[(.+?)\]="(.+?)"', block)
        for key, value in entries:
            if key not in parsed:
                parsed[key] = {}
            parsed[key][field_name] = value

    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shell-registry",
        default="scripts/cel_stage1_last_layer/formal_candidate_registry.sh",
    )
    args = parser.parse_args()

    shell_registry = parse_shell_registry(Path(args.shell_registry))

    expected_keys = set(FORMAL_CANDIDATE_LAUNCH_KEYS)
    actual_keys = set(shell_registry)
    if expected_keys != actual_keys:
        raise SystemExit(
            f"launch_key mismatch between python/shell registries: python={sorted(expected_keys)} shell={sorted(actual_keys)}"
        )

    mismatches: list[str] = []
    for launch_key, model_name in FORMAL_CANDIDATE_LAUNCH_KEYS.items():
        py_meta = FORMAL_CANDIDATES[model_name]
        sh_meta = shell_registry[launch_key]

        expected_pairs = {
            "model_name": model_name,
            "model_pattern": py_meta["model_pattern"],
            "run_script_name": py_meta["run_script_name"],
            "run_script_rel": py_meta["run_script_rel"],
            "stdout_log_name": py_meta["stdout_log_name"],
            "stdout_log_rel": py_meta["stdout_log_rel"],
            "metrics_rel": py_meta["metrics_rel"],
            "default_gpu_id": py_meta["default_gpu_id"],
        }
        for field_name, expected_value in expected_pairs.items():
            actual_value = sh_meta.get(field_name)
            if actual_value != expected_value:
                mismatches.append(
                    f"{launch_key}.{field_name}: python={expected_value!r} shell={actual_value!r}"
                )

    if mismatches:
        raise SystemExit("formal candidate registry mismatch:\n" + "\n".join(mismatches))

    print("formal candidate registries are aligned")


if __name__ == "__main__":
    main()
