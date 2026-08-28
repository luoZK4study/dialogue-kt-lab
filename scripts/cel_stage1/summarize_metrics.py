#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


LOSS_RE = re.compile(r"^Loss:\s+([0-9.]+)$", re.MULTILINE)
OVERALL_RE = re.compile(
    r"Overall.*?\n.*?\nAcc:\s+([0-9.]+),\s+AUC:\s+([0-9.]+),\s+Prec:\s+([0-9.]+),\s+Rec:\s+([0-9.]+),\s+F1:\s+([0-9.]+)",
    re.DOTALL,
)
FINAL_RE = re.compile(
    r"Final Turn.*?\n.*?\nAcc:\s+([0-9.]+),\s+AUC:\s+([0-9.]+),\s+Prec:\s+([0-9.]+),\s+Rec:\s+([0-9.]+),\s+F1:\s+([0-9.]+)",
    re.DOTALL,
)


def parse_metrics_file(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    loss_match = LOSS_RE.search(text)
    overall_match = OVERALL_RE.search(text)
    final_match = FINAL_RE.search(text)
    if not loss_match or not overall_match:
        raise ValueError(f"Cannot parse metrics file: {path}")
    result = {
        "model": path.stem.removeprefix("metrics_"),
        "loss": float(loss_match.group(1)),
        "overall_acc": float(overall_match.group(1)),
        "overall_auc": float(overall_match.group(2)),
        "overall_prec": float(overall_match.group(3)),
        "overall_rec": float(overall_match.group(4)),
        "overall_f1": float(overall_match.group(5)),
        "final_acc": None,
        "final_auc": None,
        "final_prec": None,
        "final_rec": None,
        "final_f1": None,
    }
    if final_match:
        result.update(
            {
                "final_acc": float(final_match.group(1)),
                "final_auc": float(final_match.group(2)),
                "final_prec": float(final_match.group(3)),
                "final_rec": float(final_match.group(4)),
                "final_f1": float(final_match.group(5)),
            }
        )
    return result


def to_markdown(rows: list[dict]) -> str:
    header = "| Model | Loss | Overall Acc | Overall AUC | Overall Prec | Overall Rec | Overall F1 | Final Acc | Final AUC | Final Prec | Final Rec | Final F1 |"
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for row in rows:
        lines.append(
            "| {model} | {loss:.4f} | {overall_acc:.2f} | {overall_auc:.2f} | {overall_prec:.2f} | {overall_rec:.2f} | {overall_f1:.2f} | {final_acc} | {final_auc} | {final_prec} | {final_rec} | {final_f1} |".format(
                model=row["model"],
                loss=row["loss"],
                overall_acc=row["overall_acc"],
                overall_auc=row["overall_auc"],
                overall_prec=row["overall_prec"],
                overall_rec=row["overall_rec"],
                overall_f1=row["overall_f1"],
                final_acc=f"{row['final_acc']:.2f}" if row["final_acc"] is not None else "-",
                final_auc=f"{row['final_auc']:.2f}" if row["final_auc"] is not None else "-",
                final_prec=f"{row['final_prec']:.2f}" if row["final_prec"] is not None else "-",
                final_rec=f"{row['final_rec']:.2f}" if row["final_rec"] is not None else "-",
                final_f1=f"{row['final_f1']:.2f}" if row["final_f1"] is not None else "-",
            )
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        nargs="*",
        default=["results/cel_stage1/metrics/metrics_cel_*.txt"],
        help="Metrics files or glob patterns",
    )
    args = parser.parse_args()

    files: list[Path] = []
    for pattern in args.paths:
        matched = sorted(Path().glob(pattern))
        if matched:
            files.extend(matched)
        else:
            path = Path(pattern)
            if path.exists():
                files.append(path)
    if not files:
        raise SystemExit("No metrics files found")

    rows = [parse_metrics_file(path) for path in files]
    print(to_markdown(rows))


if __name__ == "__main__":
    main()
