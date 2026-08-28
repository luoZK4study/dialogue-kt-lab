#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

from formal_candidate_registry import FORMAL_CANDIDATES, FORMAL_CANDIDATE_LAUNCH_KEYS


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts" / "cel_stage1_last_layer"
FORMAL_AUDIT_JSON = ROOT / "results" / "cel_stage1_last_layer" / "formal_experiment_audit.json"


def ensure_unused(launch_key: str, model_name: str, run_script_name: str) -> None:
    if launch_key in FORMAL_CANDIDATE_LAUNCH_KEYS:
        raise SystemExit(f"launch_key already exists in formal registry: {launch_key}")
    if model_name in FORMAL_CANDIDATES:
        raise SystemExit(f"model_name already exists in formal registry: {model_name}")
    if (SCRIPTS_DIR / run_script_name).exists():
        raise SystemExit(f"run script already exists: {run_script_name}")


def load_formal_audit(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"failed to parse formal audit JSON at {path}: {exc}") from exc


def enforce_queue_clear_for_new_candidate(audit: dict | None, allow_override: bool) -> None:
    if allow_override or audit is None:
        return
    rows = audit.get("rows") or []
    needs_rerun = [
        str(row.get("label") or row.get("model"))
        for row in rows
        if row.get("audit") == "needs_rerun"
    ]
    if needs_rerun:
        joined = ",".join(needs_rerun)
        raise SystemExit(
            "cannot scaffold a new formal candidate while strict rerun targets remain: "
            f"{joined}. Rerun the existing formal queue first, or pass --allow-pending-rerun only if you are intentionally doing offline draft scaffolding."
        )


def extract_train_args(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    command_lines: list[str] = []
    capturing = False

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not capturing and "python -m dialogue_kt.main train" in line:
            capturing = True
            line = line[line.index("python -m dialogue_kt.main train") :]
        if not capturing:
            continue

        if line.endswith("\\"):
            command_lines.append(line[:-1].strip())
            continue
        command_lines.append(line.strip())
        break

    if not command_lines:
        raise SystemExit(f"train command not found in {path}")

    tokens = shlex.split(" ".join(command_lines))
    if tokens[:4] != ["python", "-m", "dialogue_kt.main", "train"]:
        raise SystemExit(f"unexpected train command prefix in {path}: {tokens[:4]}")
    return tokens


def tokens_to_arg_map(tokens: list[str]) -> dict[str, str]:
    args: dict[str, str] = {}
    idx = 4
    while idx < len(tokens):
        key = tokens[idx]
        if not key.startswith("--"):
            raise SystemExit(f"unexpected token while parsing args: {key}")
        if idx + 1 < len(tokens) and not tokens[idx + 1].startswith("--"):
            args[key[2:]] = tokens[idx + 1]
            idx += 2
        else:
            args[key[2:]] = "true"
            idx += 1
    return args


def arg_map_to_tokens(args: dict[str, str]) -> list[str]:
    tokens = ["python", "-m", "dialogue_kt.main", "train"]
    for key, value in args.items():
        tokens.append(f"--{key}")
        if value != "true":
            tokens.append(value)
    return tokens


def update_arg_map(
    args: dict[str, str],
    model_name: str,
    sets: list[tuple[str, str]],
    unsets: list[str],
) -> dict[str, str]:
    updated = dict(args)
    updated["model_name"] = model_name
    for key, value in sets:
        updated[key] = value
    for key in unsets:
        updated.pop(key, None)
    return updated


def compute_diff_keys(reference_args: dict[str, str], candidate_args: dict[str, str]) -> list[str]:
    keys = sorted(set(reference_args) | set(candidate_args))
    return [key for key in keys if reference_args.get(key) != candidate_args.get(key)]


def render_script(tokens: list[str], gpu_id: str) -> str:
    arg_lines = []
    for idx in range(4, len(tokens), 2):
        key = tokens[idx]
        if idx + 1 < len(tokens) and not tokens[idx + 1].startswith("--"):
            value = tokens[idx + 1]
            arg_lines.append(f"  {key} {value} \\")
        else:
            arg_lines.append(f"  {key} \\")

    if arg_lines:
        arg_lines[-1] = arg_lines[-1].removesuffix(" \\")

    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            'ROOT_DIR="/home/user4/dialogue-kt"',
            "",
            'cd "$ROOT_DIR"',
            "source /opt/anaconda3/etc/profile.d/conda.sh",
            "conda activate luo_2",
            "export WANDB_MODE=disabled",
            "export CUBLAS_WORKSPACE_CONFIG=:4096:8",
            "",
            f'CUDA_VISIBLE_DEVICES="${{CUDA_VISIBLE_DEVICES:-{gpu_id}}}" python -m dialogue_kt.main train \\',
            *arg_lines,
            "",
        ]
    )


def parse_set_arg(items: list[str]) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise SystemExit(f"empty KEY in --set item: {item}")
        parsed.append((key, value))
    return parsed


def registry_block(
    *,
    launch_key: str,
    label: str,
    model_name: str,
    model_pattern: str,
    run_script_name: str,
    default_gpu_id: str,
    reference_rel: str,
    reference_label_cn: str,
    single_variable_cn: str,
    single_variable_en: str,
    config_diff_keys: list[str],
    config_diff_rationale_cn: str,
    start_point_cn: str,
    start_point_en: str,
    method_cn: str,
    method_en: str,
    hypothesis_cn: str,
    hypothesis_en: str,
    expected_signals_cn: list[str],
    expected_signals_en: list[str],
    if_fail_cn: str,
    if_fail_en: str,
) -> str:
    expected_cn = "\n".join(f'            "{line}",' for line in expected_signals_cn)
    expected_en = "\n".join(f'            "{line}",' for line in expected_signals_en)
    diff_keys = "\n".join(f'            "{key}",' for key in config_diff_keys)
    log_stem = model_name
    if log_stem.startswith("cel_task_conditioned_lastlayer_"):
        log_stem = log_stem.removeprefix("cel_task_conditioned_lastlayer_")
    if log_stem.endswith("_qwen3_1.7b"):
        log_stem = log_stem.removesuffix("_qwen3_1.7b")
    stdout_name = f"task_conditioned_{log_stem}.stdout.log"
    return f"""    "{model_name}": {{
        "launch_key": "{launch_key}",
        "label": "{label}",
        "model_pattern": "{model_pattern}",
        "start_point_en": "{start_point_en}",
        "start_point_cn": "{start_point_cn}",
        "method_en": "{method_en}",
        "method_cn": "{method_cn}",
        "single_variable_en": "{single_variable_en}",
        "single_variable_cn": "{single_variable_cn}",
        "config_reference_rel": "{reference_rel}",
        "config_reference_label_cn": "{reference_label_cn}",
        "config_diff_keys": [
{diff_keys}
        ],
        "config_diff_rationale_cn": "{config_diff_rationale_cn}",
        "hypothesis_en": "{hypothesis_en}",
        "hypothesis_cn": "{hypothesis_cn}",
        "expected_signals_en": [
{expected_en}
        ],
        "expected_signals_cn": [
{expected_cn}
        ],
        "if_fail_en": "{if_fail_en}",
        "if_fail_cn": "{if_fail_cn}",
        "scope_en": "SSH train + val + test",
        "stdout_log_name": "{stdout_name}",
        "stdout_log_rel": "results/cel_stage1_last_layer/{stdout_name}",
        "metrics_rel": "results/cel_stage1_last_layer/metrics/metrics_{model_name}.txt",
        "run_script_name": "{run_script_name}",
        "run_script_rel": "scripts/cel_stage1_last_layer/{run_script_name}",
        "default_gpu_id": "{default_gpu_id}",
    }},"""


def shell_registry_stub(
    *,
    launch_key: str,
    model_name: str,
    model_pattern: str,
    run_script_name: str,
    default_gpu_id: str,
) -> str:
    log_stem = model_name
    if log_stem.startswith("cel_task_conditioned_lastlayer_"):
        log_stem = log_stem.removeprefix("cel_task_conditioned_lastlayer_")
    if log_stem.endswith("_qwen3_1.7b"):
        log_stem = log_stem.removesuffix("_qwen3_1.7b")
    stdout_name = f"task_conditioned_{log_stem}.stdout.log"
    return "\n".join(
        [
            f"# Add `{launch_key}` to FORMAL_CANDIDATE_KEYS and the associative arrays below.",
            f'[{launch_key}]="{model_name}"  # FORMAL_MODEL_NAME',
            f'[{launch_key}]="{model_pattern}"  # FORMAL_MODEL_PATTERN',
            f'[{launch_key}]="{run_script_name}"  # FORMAL_RUN_SCRIPT_NAME',
            f'[{launch_key}]="scripts/cel_stage1_last_layer/{run_script_name}"  # FORMAL_RUN_SCRIPT_REL',
            f'[{launch_key}]="{stdout_name}"  # FORMAL_STDOUT_LOG_NAME',
            f'[{launch_key}]="results/cel_stage1_last_layer/{stdout_name}"  # FORMAL_STDOUT_LOG_REL',
            f'[{launch_key}]="results/cel_stage1_last_layer/metrics/metrics_{model_name}.txt"  # FORMAL_METRICS_REL',
            f'[{launch_key}]="{default_gpu_id}"  # FORMAL_DEFAULT_GPU_ID',
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch-key", required=True, help="New formal candidate key, e.g. v23")
    parser.add_argument("--label", help="Display label; defaults to launch key")
    parser.add_argument("--model-name", required=True, help="Full model name for the new formal candidate")
    parser.add_argument("--reference-run-script", required=True, help="Reference formal run script under scripts/cel_stage1_last_layer/")
    parser.add_argument("--run-script-name", required=True, help="New run script filename under scripts/cel_stage1_last_layer/")
    parser.add_argument("--default-gpu-id", default="0", help="Default GPU id for the new candidate")
    parser.add_argument("--single-variable-cn", required=True, help="Single main variable description in Chinese")
    parser.add_argument("--single-variable-en", required=True, help="Single main variable description in English")
    parser.add_argument("--config-diff-keys", required=True, help="Comma-separated list of CLI keys allowed to differ from the reference script")
    parser.add_argument("--config-diff-rationale-cn", required=True, help="Why those CLI diffs still implement one semantic variable")
    parser.add_argument("--start-point-cn", required=True)
    parser.add_argument("--start-point-en", required=True)
    parser.add_argument("--method-cn", required=True)
    parser.add_argument("--method-en", required=True)
    parser.add_argument("--hypothesis-cn", required=True)
    parser.add_argument("--hypothesis-en", required=True)
    parser.add_argument("--expected-signal-cn", action="append", required=True, help="Repeatable expected observation in Chinese")
    parser.add_argument("--expected-signal-en", action="append", required=True, help="Repeatable expected observation in English")
    parser.add_argument("--if-fail-cn", required=True)
    parser.add_argument("--if-fail-en", required=True)
    parser.add_argument("--set", action="append", default=[], help="CLI override to apply to the generated run script, format KEY=VALUE")
    parser.add_argument("--unset", action="append", default=[], help="CLI key to remove from the generated run script")
    parser.add_argument("--out-dir", default="tmp/formal_candidate_scaffold", help="Relative output directory for generated helper files")
    parser.add_argument("--allow-pending-rerun", action="store_true", help="Bypass the strict queue gate and scaffold only for offline draft work")
    args = parser.parse_args()

    enforce_queue_clear_for_new_candidate(load_formal_audit(FORMAL_AUDIT_JSON), args.allow_pending_rerun)

    label = args.label or args.launch_key
    reference_rel = f"scripts/cel_stage1_last_layer/{args.reference_run_script}"
    reference_path = ROOT / reference_rel
    run_script_name = args.run_script_name
    model_pattern = args.model_name.replace("cel_", "").replace("_qwen3_1.7b", "")
    ensure_unused(args.launch_key, args.model_name, run_script_name)

    tokens = extract_train_args(reference_path)
    reference_args = tokens_to_arg_map(tokens)
    set_items = parse_set_arg(args.set)
    updated_args = update_arg_map(reference_args, args.model_name, set_items, args.unset)
    actual_diff_keys = compute_diff_keys(reference_args, updated_args)
    declared_diff_keys = [item.strip() for item in args.config_diff_keys.split(",") if item.strip()]
    if set(actual_diff_keys) != set(declared_diff_keys):
        raise SystemExit(
            "declared config_diff_keys do not match actual generated script diffs:\n"
            f"declared={declared_diff_keys}\nactual={actual_diff_keys}"
        )
    updated_tokens = arg_map_to_tokens(updated_args)

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    run_script_path = out_dir / run_script_name
    run_script_path.write_text(render_script(updated_tokens, args.default_gpu_id), encoding="utf-8")

    declared_keys = declared_diff_keys
    python_registry_path = out_dir / f"{args.launch_key}_formal_registry_block.txt"
    python_registry_path.write_text(
        registry_block(
            launch_key=args.launch_key,
            label=label,
            model_name=args.model_name,
            model_pattern=model_pattern,
            run_script_name=run_script_name,
            default_gpu_id=args.default_gpu_id,
            reference_rel=reference_rel,
            reference_label_cn=f"参考脚本 {args.reference_run_script}",
            single_variable_cn=args.single_variable_cn,
            single_variable_en=args.single_variable_en,
            config_diff_keys=declared_keys,
            config_diff_rationale_cn=args.config_diff_rationale_cn,
            start_point_cn=args.start_point_cn,
            start_point_en=args.start_point_en,
            method_cn=args.method_cn,
            method_en=args.method_en,
            hypothesis_cn=args.hypothesis_cn,
            hypothesis_en=args.hypothesis_en,
            expected_signals_cn=args.expected_signal_cn,
            expected_signals_en=args.expected_signal_en,
            if_fail_cn=args.if_fail_cn,
            if_fail_en=args.if_fail_en,
        )
        + "\n",
        encoding="utf-8",
    )

    shell_registry_path = out_dir / f"{args.launch_key}_formal_shell_registry_stub.txt"
    shell_registry_path.write_text(
        shell_registry_stub(
            launch_key=args.launch_key,
            model_name=args.model_name,
            model_pattern=model_pattern,
            run_script_name=run_script_name,
            default_gpu_id=args.default_gpu_id,
        ),
        encoding="utf-8",
    )

    notes_path = out_dir / f"{args.launch_key}_README.txt"
    notes_path.write_text(
        "\n".join(
            [
                f"Scaffolded formal candidate: {args.launch_key}",
                f"Reference run script: {reference_rel}",
                f"Generated run script stub: {run_script_path.relative_to(ROOT)}",
                f"Generated Python registry block stub: {python_registry_path.relative_to(ROOT)}",
                f"Generated shell registry stub: {shell_registry_path.relative_to(ROOT)}",
                "",
                "Next steps:",
                "1. Inspect the generated run script and confirm its actual diff keys match the declared config_diff_keys.",
                "2. Paste and adapt the generated Python registry block into formal_candidate_registry.py.",
                "3. Paste the generated shell registry lines into formal_candidate_registry.sh.",
                "4. Keep the Python and shell registries aligned.",
                "5. Add the new launch key/model/run script to the rest of the sync/launch flow if needed.",
                "6. Regenerate FORMAL_CANDIDATE_BRIEFS.md and FORMAL_CANDIDATE_CONFIG_DIFFS.md.",
                "7. Run preflight_strict_full_train.sh and confirm the config diff report stays `ok`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"generated_run_script={run_script_path}")
    print(f"generated_python_registry_block={python_registry_path}")
    print(f"generated_shell_registry_stub={shell_registry_path}")
    print(f"generated_notes={notes_path}")


if __name__ == "__main__":
    main()
