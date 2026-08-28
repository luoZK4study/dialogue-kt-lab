#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

from formal_candidate_registry import FORMAL_CANDIDATE_ORDER, FORMAL_CANDIDATES


ROOT = Path(__file__).resolve().parents[2]


def extract_train_args(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    command_lines: list[str] = []
    capturing = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not capturing and "python -m dialogue_kt.main train" in line:
            capturing = True
            line = line[line.index("python -m dialogue_kt.main train") :]
        if not capturing:
            continue

        if line.endswith("\\"):
            command_lines.append(line[:-1].strip())
            continue
        command_lines.append(line)
        break

    if not command_lines:
        raise ValueError(f"train command not found in {path}")

    tokens = shlex.split(" ".join(command_lines))
    if tokens[:4] != ["python", "-m", "dialogue_kt.main", "train"]:
        raise ValueError(f"unexpected train command prefix in {path}: {tokens[:4]}")

    args: dict[str, str] = {}
    idx = 4
    while idx < len(tokens):
        token = tokens[idx]
        if not token.startswith("--"):
            raise ValueError(f"unexpected token in {path}: {token}")
        key = token[2:]
        if idx + 1 >= len(tokens) or tokens[idx + 1].startswith("--"):
            args[key] = "true"
            idx += 1
            continue
        args[key] = tokens[idx + 1]
        idx += 2
    return args


def arg_diff(reference_args: dict[str, str], candidate_args: dict[str, str]) -> dict[str, tuple[str | None, str | None]]:
    diff: dict[str, tuple[str | None, str | None]] = {}
    for key in sorted(set(reference_args) | set(candidate_args)):
        ref_value = reference_args.get(key)
        cand_value = candidate_args.get(key)
        if ref_value != cand_value:
            diff[key] = (ref_value, cand_value)
    return diff


def fmt_value(value: str | None) -> str:
    return "<unset>" if value is None else value


def build_multistage_section(model_name: str, meta: dict) -> tuple[list[str], list[str]]:
    mismatches: list[str] = []
    contract = list(meta.get("multistage_contract") or [])
    controller_rel = str(meta["run_script_rel"])
    controller_path = ROOT / controller_rel
    controller_text = controller_path.read_text(encoding="utf-8")
    forbidden_names = list(meta.get("forbidden_initializer_names") or [])
    parsed_stages: list[tuple[dict, dict[str, str]]] = []
    call_positions: list[int] = []

    for stage in contract:
        stage_rel = str(stage["run_script_rel"])
        stage_path = ROOT / stage_rel
        stage_text = stage_path.read_text(encoding="utf-8")
        stage_args = extract_train_args(stage_path)
        parsed_stages.append((stage, stage_args))

        stage_name = str(stage["label"])
        for key, expected in (stage.get("expected_args") or {}).items():
            actual = stage_args.get(key)
            if actual != expected:
                mismatches.append(
                    f"{meta['label']}.{stage_name}.{key}: expected={expected!r} actual={actual!r}"
                )
        for key in stage.get("forbidden_args") or []:
            if key in stage_args:
                mismatches.append(
                    f"{meta['label']}.{stage_name}: forbidden argument --{key}={stage_args[key]!r}"
                )
        for forbidden_name in forbidden_names:
            if forbidden_name in stage_text:
                mismatches.append(
                    f"{meta['label']}.{stage_name}: forbidden historical initializer name found: {forbidden_name}"
                )

        script_name = stage_path.name
        position = controller_text.find(script_name)
        if position < 0:
            mismatches.append(
                f"{meta['label']}: controller {controller_rel} does not invoke {script_name}"
            )
        call_positions.append(position)

    valid_positions = [position for position in call_positions if position >= 0]
    if len(valid_positions) == len(call_positions) and valid_positions != sorted(valid_positions):
        mismatches.append(f"{meta['label']}: multistage controller invocation order does not match registry contract")

    planned_epochs = meta.get("planned_epochs")
    actual_epochs = sum(int(args.get("epochs", "0")) for _stage, args in parsed_stages)
    if actual_epochs != planned_epochs:
        mismatches.append(
            f"{meta['label']}: planned_epochs={planned_epochs!r} but stage scripts total {actual_epochs}"
        )

    final_args = parsed_stages[-1][1] if parsed_stages else {}
    if final_args.get("model_name") != model_name:
        mismatches.append(
            f"{meta['label']}: final stage model_name mismatch, expected {model_name!r} got {final_args.get('model_name')!r}"
        )

    stage_status = "ok" if not mismatches else "mismatch"
    lines = [
        f"## `{meta['label']}` / `{model_name}`",
        "",
        f"- 多阶段总控：`{controller_rel}`",
        f"- 语义主变量：{meta['single_variable_cn']}",
        f"- 差异解释：{meta['config_diff_rationale_cn']}",
        f"- 阶段顺序：`{' -> '.join(str(stage['label']) for stage in contract)}`",
        f"- 计划总 epoch：`{planned_epochs}`；实际阶段合计：`{actual_epochs}`",
        f"- 来源与配置校验：`{stage_status}`",
        "",
        "| Stage | Script | Model | Epochs | LR | Input LoRA | Input Selector | Input Calibrator | Test |",
        "|---|---|---|---:|---:|---|---|---|---|",
    ]
    for stage, args in parsed_stages:
        test_mode = "skip" if args.get("skip_test_after_train") == "1" else "formal"
        lines.append(
            f"| `{stage['label']}` | `{stage['run_script_rel']}` | `{fmt_value(args.get('model_name'))}` | "
            f"`{fmt_value(args.get('epochs'))}` | `{fmt_value(args.get('lr'))}` | "
            f"`{fmt_value(args.get('pt_model_name'))}` | `{fmt_value(args.get('cel_selector_init_model_name'))}` | "
            f"`{fmt_value(args.get('cel_calibrator_init_model_name'))}` | `{test_mode}` |"
        )
    lines.extend(
        [
            "",
            "- 禁止初始化来源：`" + "`, `".join(forbidden_names) + "`",
            "- 只有最终 joint 阶段执行正式 test；中间阶段只产生本次 v26 链自己的 checkpoint。",
            "",
        ]
    )
    return lines, mismatches


def build_report() -> tuple[str, list[str]]:
    mismatches: list[str] = []
    lines = [
        "# Formal Candidate Config Diffs",
        "",
        "- 目的：把每个 strict full-train 正式候选相对其参考脚本的 CLI 配置差异显式列出，并防止未来新增候选时出现未声明的配置漂移。",
        "- 规则：`config_diff_keys` 必须完整覆盖候选脚本相对参考脚本的实际参数差异；若出现未声明差异或声明项与实际不符，preflight 应失败。",
        "",
    ]

    for model_name in FORMAL_CANDIDATE_ORDER:
        meta = FORMAL_CANDIDATES[model_name]
        if meta.get("multistage_contract"):
            section, section_mismatches = build_multistage_section(model_name, meta)
            lines.extend(section)
            mismatches.extend(section_mismatches)
            continue

        ref_rel = meta["config_reference_rel"]
        cand_rel = meta["run_script_rel"]
        ref_path = ROOT / ref_rel
        cand_path = ROOT / cand_rel

        ref_args = extract_train_args(ref_path)
        cand_args = extract_train_args(cand_path)
        diff = arg_diff(ref_args, cand_args)
        actual_keys = list(diff.keys())
        declared_keys = list(meta["config_diff_keys"])

        declared_set = set(declared_keys)
        actual_set = set(actual_keys)
        extra_keys = sorted(actual_set - declared_set)
        missing_keys = sorted(declared_set - actual_set)
        status = "ok" if not extra_keys and not missing_keys else "mismatch"

        if status != "ok":
            mismatches.append(
                f"{meta['label']}: declared={declared_keys} actual={actual_keys} extra={extra_keys} missing={missing_keys}"
            )

        if cand_args.get("model_name") != model_name:
            mismatches.append(
                f"{meta['label']}: run script model_name mismatch, expected {model_name!r} got {cand_args.get('model_name')!r}"
            )

        lines.extend(
            [
                f"## `{meta['label']}` / `{model_name}`",
                "",
                f"- 参考脚本：`{ref_rel}` ({meta['config_reference_label_cn']})",
                f"- 候选脚本：`{cand_rel}`",
                f"- 语义主变量：{meta['single_variable_cn']}",
                f"- 差异解释：{meta['config_diff_rationale_cn']}",
                f"- 声明差异键：`{', '.join(declared_keys)}`",
                f"- 实际差异键：`{', '.join(actual_keys) if actual_keys else 'none'}`",
                f"- 配置校验：`{status}`",
                "",
                "| Arg | Reference | Candidate |",
                "|---|---|---|",
            ]
        )
        if meta.get("implementation_guard_cn"):
            lines.insert(len(lines) - 3, f"- 实现口径核对：{meta['implementation_guard_cn']}")

        for key in actual_keys:
            ref_value, cand_value = diff[key]
            lines.append(f"| `{key}` | `{fmt_value(ref_value)}` | `{fmt_value(cand_value)}` |")

        if not actual_keys:
            lines.append("| `_none_` | `_same_` | `_same_` |")

        if extra_keys:
            lines.append("")
            lines.append(f"- 未声明但实际发生的差异：`{', '.join(extra_keys)}`")
        if missing_keys:
            lines.append("")
            lines.append(f"- 已声明但当前脚本中未发生的差异：`{', '.join(missing_keys)}`")
        lines.append("")

    return "\n".join(lines) + "\n", mismatches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/cel_stage1_last_layer/FORMAL_CANDIDATE_CONFIG_DIFFS.md")
    args = parser.parse_args()

    report, mismatches = build_report()
    out_path = ROOT / args.out
    out_path.write_text(report, encoding="utf-8")

    if mismatches:
        raise SystemExit("formal candidate config diff mismatch:\n" + "\n".join(mismatches))


if __name__ == "__main__":
    main()
