#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path


REQUIRED_STATIC_SYNC_FILES = (
    "dialogue_kt/main.py",
    "dialogue_kt/training.py",
    "dialogue_kt/cel_methods.py",
    "scripts/cel_stage1_last_layer/check_formal_candidate_registry_alignment.py",
    "scripts/cel_stage1_last_layer/check_formal_surface_consistency.py",
    "scripts/cel_stage1_last_layer/check_ssh_formal_config.sh",
    "scripts/cel_stage1_last_layer/current_formal_state.py",
    "scripts/cel_stage1_last_layer/print_current_formal_manual_launch_steps.sh",
    "scripts/cel_stage1_last_layer/print_current_formal_next_action.py",
    "scripts/cel_stage1_last_layer/check_probability_bce_safety.py",
    "scripts/cel_stage1_last_layer/check_sync_manifest_coverage.py",
    "scripts/cel_stage1_last_layer/finalize_current_formal_candidate.sh",
    "scripts/cel_stage1_last_layer/finalize_current_formal_candidate_via_ssh_alias.sh",
    "scripts/cel_stage1_last_layer/finalize_formal_candidate.sh",
    "scripts/cel_stage1_last_layer/finalize_formal_candidate_via_ssh_alias.sh",
    "scripts/cel_stage1_last_layer/formal_candidate_registry.py",
    "scripts/cel_stage1_last_layer/formal_candidate_registry.sh",
    "scripts/cel_stage1_last_layer/formal_queue_state.py",
    "scripts/cel_stage1_last_layer/generate_comparison_report.py",
    "scripts/cel_stage1_last_layer/generate_detailed_analysis.py",
    "scripts/cel_stage1_last_layer/generate_formal_candidate_briefs.py",
    "scripts/cel_stage1_last_layer/generate_formal_candidate_config_diffs.py",
    "scripts/cel_stage1_last_layer/generate_formal_experiment_audit.py",
    "scripts/cel_stage1_last_layer/generate_formal_method_protocol.py",
    "scripts/cel_stage1_last_layer/generate_formal_experiment_summary.py",
    "scripts/cel_stage1_last_layer/generate_status_summary.py",
    "scripts/cel_stage1_last_layer/generate_strict_full_train_report.py",
    "scripts/cel_stage1_last_layer/check_formal_candidate_rerun_readiness.py",
    "scripts/cel_stage1_last_layer/launch_round3_candidate_in_remote_repo.sh",
    "scripts/cel_stage1_last_layer/launch_round3_candidate_on_server.sh",
    "scripts/cel_stage1_last_layer/monitor_formal_candidate.sh",
    "scripts/cel_stage1_last_layer/monitor_formal_candidate_via_ssh_alias.sh",
    "scripts/cel_stage1_last_layer/print_formal_candidate_context.py",
    "scripts/cel_stage1_last_layer/print_manual_formal_launch_steps.sh",
    "scripts/cel_stage1_last_layer/print_recommended_wsl_ssh_alias.sh",
    "scripts/cel_stage1_last_layer/preflight_strict_full_train.sh",
    "scripts/cel_stage1_last_layer/rebuild_task_conditioned_reports.sh",
    "scripts/cel_stage1_last_layer/refresh_task_conditioned_round3_once.sh",
    "scripts/cel_stage1_last_layer/run_current_formal_next_action_via_ssh_alias.sh",
    "scripts/cel_stage1_last_layer/run_task_conditioned_controller_via_ssh_alias.sh",
    "scripts/cel_stage1_last_layer/review_current_formal_candidate.sh",
    "scripts/cel_stage1_last_layer/review_current_formal_candidate_via_ssh_alias.sh",
    "scripts/cel_stage1_last_layer/review_formal_candidate.sh",
    "scripts/cel_stage1_last_layer/review_formal_candidate_via_ssh_alias.sh",
    "scripts/cel_stage1_last_layer/run_task_conditioned_controller.sh",
    "scripts/cel_stage1_last_layer/scaffold_formal_candidate.py",
    "scripts/cel_stage1_last_layer/show_current_formal_runtime_health.sh",
    "scripts/cel_stage1_last_layer/ssh_config.sh",
    "scripts/cel_stage1_last_layer/start_task_conditioned_controller_background.sh",
    "scripts/cel_stage1_last_layer/start_current_formal_candidate_via_ssh_alias.sh",
    "scripts/cel_stage1_last_layer/start_formal_candidate.sh",
    "scripts/cel_stage1_last_layer/start_formal_candidate_via_ssh_alias.sh",
    "scripts/cel_stage1_last_layer/sync_code_to_server_via_ssh_alias.sh",
    "scripts/cel_stage1_last_layer/sync_code_to_server.sh",
    "scripts/cel_stage1_last_layer/sync_results_from_server_via_ssh_alias.sh",
    "scripts/cel_stage1_last_layer/sync_results_from_server.sh",
    "scripts/cel_stage1_last_layer/task_conditioned_failure_utils.py",
    "scripts/cel_stage1_last_layer/update_baseline_record.py",
    "scripts/cel_stage1_last_layer/update_stage1_last_layer_record.py",
    "scripts/cel_stage1_last_layer/update_task_conditioned_tuning_loop.py",
    "scripts/cel_stage1_last_layer/watch_task_conditioned_tuning.sh",
    "scripts/cel_stage1_last_layer/run_task_conditioned_v26_bootstrap.sh",
    "scripts/cel_stage1_last_layer/run_task_conditioned_v26_calibrator_warmup.sh",
    "scripts/cel_stage1_last_layer/run_task_conditioned_v26_joint.sh",
    "scripts/cel_stage1_last_layer/audit_task_conditioned_v26_selftrained.py",
)

REQUIRED_SYNC_SOURCE_LINE = (
    'source "$ROOT_DIR/scripts/cel_stage1_last_layer/formal_candidate_registry.sh"'
)
REQUIRED_DYNAMIC_LOOP_SNIPPETS = (
    'for key in "${FORMAL_CANDIDATE_KEYS[@]}"; do',
    'FILES_TO_SYNC+=("${FORMAL_RUN_SCRIPT_REL[$key]}")',
)
REQUIRED_RESULT_SYNC_DYNAMIC_SNIPPETS = (
    'for key in "${FORMAL_CANDIDATE_KEYS[@]}"; do',
    'LOG_FILES+=("${FORMAL_STDOUT_LOG_NAME[$key]}")',
)


def parse_array_block(text: str, array_name: str) -> list[str]:
    match = re.search(rf"{re.escape(array_name)}=\((.*?)\n\)", text, re.S)
    if not match:
        raise ValueError(f"{array_name} block not found in script")
    block = match.group(1)
    return re.findall(r'"([^"]+)"', block)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sync-script",
        default="scripts/cel_stage1_last_layer/sync_code_to_server.sh",
    )
    parser.add_argument(
        "--result-sync-script",
        default="scripts/cel_stage1_last_layer/sync_results_from_server.sh",
    )
    args = parser.parse_args()

    sync_script_path = Path(args.sync_script)
    sync_text = sync_script_path.read_text(encoding="utf-8")
    static_entries = set(parse_array_block(sync_text, "FILES_TO_SYNC"))

    missing_static = [
        rel_path
        for rel_path in REQUIRED_STATIC_SYNC_FILES
        if rel_path not in static_entries
    ]
    if missing_static:
        raise SystemExit(
            "strict formal sync manifest is missing required files:\n"
            + "\n".join(missing_static)
        )

    if REQUIRED_SYNC_SOURCE_LINE not in sync_text:
        raise SystemExit(
            "sync manifest must source formal_candidate_registry.sh before adding registered run scripts"
        )

    missing_loop_snippets = [
        snippet for snippet in REQUIRED_DYNAMIC_LOOP_SNIPPETS if snippet not in sync_text
    ]
    if missing_loop_snippets:
        raise SystemExit(
            "sync manifest must append FORMAL_RUN_SCRIPT_REL for every registered formal candidate:\n"
            + "\n".join(missing_loop_snippets)
        )

    result_sync_script_path = Path(args.result_sync_script)
    result_sync_text = result_sync_script_path.read_text(encoding="utf-8")
    parse_array_block(result_sync_text, "LOG_FILES")

    if REQUIRED_SYNC_SOURCE_LINE not in result_sync_text:
        raise SystemExit(
            "result sync manifest must source formal_candidate_registry.sh before adding registered formal stdout logs"
        )

    missing_result_sync_snippets = [
        snippet
        for snippet in REQUIRED_RESULT_SYNC_DYNAMIC_SNIPPETS
        if snippet not in result_sync_text
    ]
    if missing_result_sync_snippets:
        raise SystemExit(
            "result sync manifest must append FORMAL_STDOUT_LOG_NAME for every registered formal candidate:\n"
            + "\n".join(missing_result_sync_snippets)
        )

    print(
        "strict formal sync manifests cover required static files, registry-driven run scripts, and registry-driven formal stdout logs"
    )


if __name__ == "__main__":
    main()
