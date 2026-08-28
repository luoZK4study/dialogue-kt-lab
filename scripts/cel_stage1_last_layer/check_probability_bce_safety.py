#!/usr/bin/env python3

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAINING_PATH = ROOT / "dialogue_kt" / "training.py"

REQUIRED_FUNCTIONS = {
    "_sanitize_binary_label_tensor",
    "_sanitize_probability_tensor",
    "_compute_bce_loss_from_probs",
    "_prefix_metric_keys",
}


def load_tree() -> ast.Module:
    return ast.parse(TRAINING_PATH.read_text(encoding="utf-8"), filename=str(TRAINING_PATH))


def function_map(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    funcs: dict[str, ast.FunctionDef] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            funcs[node.name] = node
    return funcs


def attr_chain(node: ast.AST) -> str | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return None


def collect_call_targets(node: ast.AST) -> set[str]:
    targets: set[str] = set()
    for inner in ast.walk(node):
        if not isinstance(inner, ast.Call):
            continue
        func = inner.func
        if isinstance(func, ast.Name):
            targets.add(func.id)
        else:
            chain = attr_chain(func)
            if chain is not None:
                targets.add(chain)
    return targets


def has_bce_loss_call(tree: ast.Module) -> bool:
    return "torch.nn.BCELoss" in collect_call_targets(tree)


def assert_required_structure(funcs: dict[str, ast.FunctionDef]) -> None:
    missing = sorted(REQUIRED_FUNCTIONS.difference(funcs))
    if missing:
        raise SystemExit(
            "probability BCE safety check failed; missing required training.py functions:\n"
            + "\n".join(missing)
        )

    compute_node = funcs["_compute_bce_loss_from_probs"]
    compute_targets = collect_call_targets(compute_node)
    required_calls = {
        "_sanitize_probability_tensor",
        "_sanitize_binary_label_tensor",
        "_prefix_metric_keys",
        "torch.logit",
        "torch.nn.BCEWithLogitsLoss",
    }
    missing_calls = sorted(required_calls.difference(compute_targets))
    if missing_calls:
        raise SystemExit(
            "probability BCE safety check failed; `_compute_bce_loss_from_probs` is missing required calls:\n"
            + "\n".join(missing_calls)
        )

def assert_required_returns(funcs: dict[str, ast.FunctionDef]) -> None:
    compute_node = funcs["_compute_bce_loss_from_probs"]
    tuple_return_ok = False
    for node in ast.walk(compute_node):
        if not isinstance(node, ast.Return):
            continue
        value = node.value
        if not isinstance(value, ast.Tuple) or len(value.elts) != 3:
            continue
        names = []
        for elt in value.elts:
            if isinstance(elt, ast.Name):
                names.append(elt.id)
            else:
                names.append("")
        if names == ["loss", "corr_probs", "prob_stats"]:
            tuple_return_ok = True
            break
    if not tuple_return_ok:
        raise SystemExit(
            "probability BCE safety check failed; `_compute_bce_loss_from_probs` must return `(loss, corr_probs, prob_stats)`"
        )


def run_optional_runtime_checks(tree: ast.Module, funcs: dict[str, ast.FunctionDef]) -> str:
    try:
        import torch  # type: ignore
    except ModuleNotFoundError:
        return "torch unavailable; runtime tensor checks skipped"

    selected_nodes: list[ast.AST] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "PROB_EPS":
                    selected_nodes.append(node)
                    break
        elif isinstance(node, ast.FunctionDef) and node.name in REQUIRED_FUNCTIONS:
            selected_nodes.append(node)

    mini_module = ast.Module(body=selected_nodes, type_ignores=[])
    ast.fix_missing_locations(mini_module)
    namespace = {"torch": torch}
    exec(compile(mini_module, filename=str(TRAINING_PATH), mode="exec"), namespace)

    sanitize_probs = namespace["_sanitize_probability_tensor"]
    sanitize_labels = namespace["_sanitize_binary_label_tensor"]
    compute_loss = namespace["_compute_bce_loss_from_probs"]
    eps = float(namespace["PROB_EPS"])

    probs = torch.tensor([float("nan"), float("inf"), float("-inf"), -0.2, 0.0, 1.0, 1.2, 0.25, 0.75], dtype=torch.float32)
    labels = torch.tensor([float("nan"), float("inf"), float("-inf"), -1.0, 0.0, 1.0, 1.5, 0.25, 0.75], dtype=torch.float32)

    safe_probs, prob_stats = sanitize_probs(probs)
    safe_labels, label_stats = sanitize_labels(labels)
    loss, returned_probs, combined_stats = compute_loss(probs, labels)

    if not torch.isfinite(safe_probs).all():
        raise SystemExit("probability BCE safety check failed; sanitized probabilities still contain non-finite values")
    if not torch.isfinite(safe_labels).all():
        raise SystemExit("probability BCE safety check failed; sanitized labels still contain non-finite values")
    if torch.any(safe_probs < eps) or torch.any(safe_probs > 1.0 - eps):
        raise SystemExit("probability BCE safety check failed; sanitized probabilities are not clamped into the open interval")
    if torch.any(safe_labels < 0.0) or torch.any(safe_labels > 1.0):
        raise SystemExit("probability BCE safety check failed; sanitized labels are not clamped into [0, 1]")
    if not torch.isfinite(loss):
        raise SystemExit("probability BCE safety check failed; sanitized BCE loss is still non-finite")
    if not torch.allclose(returned_probs, safe_probs):
        raise SystemExit("probability BCE safety check failed; `_compute_bce_loss_from_probs` does not return the sanitized probabilities")

    expected_loss = torch.nn.BCEWithLogitsLoss()(torch.logit(safe_probs), safe_labels)
    if not torch.allclose(loss, expected_loss):
        raise SystemExit("probability BCE safety check failed; sanitized BCE loss is not computed from logit-space probabilities as expected")

    required_prob_stat_keys = {"invalid_frac", "out_of_range_frac", "sanitized_frac", "raw_min", "raw_max", "safe_min", "safe_max"}
    if set(prob_stats) != required_prob_stat_keys:
        raise SystemExit("probability BCE safety check failed; probability sanitizer stats keys drifted unexpectedly")
    if set(label_stats) != required_prob_stat_keys:
        raise SystemExit("probability BCE safety check failed; label sanitizer stats keys drifted unexpectedly")
    required_combined_keys = required_prob_stat_keys.union({f"label_{key}" for key in required_prob_stat_keys})
    if set(combined_stats) != required_combined_keys:
        raise SystemExit("probability BCE safety check failed; combined loss stats keys drifted unexpectedly")

    return "runtime tensor checks passed"


def main() -> None:
    tree = load_tree()
    funcs = function_map(tree)

    if has_bce_loss_call(tree):
        raise SystemExit("probability BCE safety check failed; found forbidden `torch.nn.BCELoss` call in training.py")

    assert_required_structure(funcs)
    assert_required_returns(funcs)
    runtime_note = run_optional_runtime_checks(tree, funcs)
    print(f"probability BCE safety check passed ({runtime_note})")


if __name__ == "__main__":
    main()
