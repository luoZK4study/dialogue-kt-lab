#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/cel_stage2_environment"
REFERENCE_QUAL = (
    ROOT
    / "results/cel_stage1_last_layer/qual/"
    "qual_cel_task_conditioned_lastlayer_v26_selftrained_biaswarmup_joint_tinylr_qwen3_1.7b.csv"
)
REFERENCE_NAME = "a_module_reference"


def load_predictions(path: Path, probability_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"Dialogue ID", "Turn", "Correct", "Prob"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    frame = frame.copy()
    frame[probability_name] = pd.to_numeric(frame["Prob"], errors="coerce")
    frame = frame.loc[frame[probability_name].notna()].copy()
    frame["label"] = (
        frame["Correct"].astype(str).str.strip().str.lower().map({"false": 0, "true": 1})
    )
    if frame["label"].isna().any():
        bad = sorted(frame.loc[frame["label"].isna(), "Correct"].astype(str).unique())
        raise ValueError(f"{path} has unsupported labels: {bad}")

    columns = ["Dialogue ID", "Turn", "label", probability_name]
    frame = frame[columns]
    if frame.duplicated(["Dialogue ID", "Turn"]).any():
        raise ValueError(f"{path} has duplicate Dialogue ID + Turn prediction rows")
    return frame


def align_predictions(reference_path: Path, candidate_path: Path) -> pd.DataFrame:
    reference = load_predictions(reference_path, "reference_prob")
    candidate = load_predictions(candidate_path, "candidate_prob")
    aligned = reference.merge(
        candidate,
        on=["Dialogue ID", "Turn"],
        how="outer",
        suffixes=("_reference", "_candidate"),
        indicator=True,
        validate="one_to_one",
    )
    if not aligned["_merge"].eq("both").all():
        counts = aligned["_merge"].value_counts().to_dict()
        raise ValueError(f"prediction keys do not align: {counts}")
    if not aligned["label_reference"].eq(aligned["label_candidate"]).all():
        raise ValueError("reference and candidate labels differ")

    aligned = aligned.rename(columns={"label_reference": "label"})
    aligned = aligned.drop(columns=["label_candidate", "_merge"])
    aligned = aligned.sort_values(["Dialogue ID", "Turn"]).reset_index(drop=True)
    max_turn = aligned.groupby("Dialogue ID")["Turn"].transform("max")
    aligned["is_final"] = aligned["Turn"].eq(max_turn)
    group_codes, group_values = pd.factorize(aligned["Dialogue ID"], sort=True)
    aligned["group_code"] = group_codes
    aligned.attrs["dialogue_ids"] = group_values.tolist()
    return aligned


def observed_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "acc": float(np.mean((probabilities >= 0.5) == labels) * 100.0),
        "auc": float(roc_auc_score(labels, probabilities) * 100.0),
    }


def bootstrap_accuracy(
    labels: np.ndarray,
    probabilities: np.ndarray,
    group_codes: np.ndarray,
    bootstrap_counts: np.ndarray,
) -> np.ndarray:
    cluster_count = bootstrap_counts.shape[1]
    totals = np.bincount(group_codes, minlength=cluster_count).astype(np.float64)
    correct = np.bincount(
        group_codes,
        weights=((probabilities >= 0.5) == labels).astype(np.float64),
        minlength=cluster_count,
    )
    denominator = bootstrap_counts @ totals
    return (bootstrap_counts @ correct) / denominator * 100.0


def bootstrap_auc(
    labels: np.ndarray,
    probabilities: np.ndarray,
    group_codes: np.ndarray,
    bootstrap_counts: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    cluster_count = bootstrap_counts.shape[1]
    _, tie_codes = np.unique(probabilities, return_inverse=True)
    tie_count = int(tie_codes.max()) + 1
    positives = np.zeros((tie_count, cluster_count), dtype=np.int32)
    negatives = np.zeros((tie_count, cluster_count), dtype=np.int32)
    positive_rows = labels == 1
    negative_rows = ~positive_rows
    np.add.at(positives, (tie_codes[positive_rows], group_codes[positive_rows]), 1)
    np.add.at(negatives, (tie_codes[negative_rows], group_codes[negative_rows]), 1)

    aucs = np.empty(len(bootstrap_counts), dtype=np.float64)
    for start in range(0, len(bootstrap_counts), chunk_size):
        stop = min(start + chunk_size, len(bootstrap_counts))
        counts = bootstrap_counts[start:stop]
        positive_weights = counts @ positives.T
        negative_weights = counts @ negatives.T
        negative_before = np.cumsum(negative_weights, axis=1) - negative_weights
        concordant = np.sum(
            positive_weights * (negative_before + 0.5 * negative_weights),
            axis=1,
            dtype=np.float64,
        )
        total_positive = positive_weights.sum(axis=1)
        total_negative = negative_weights.sum(axis=1)
        denominator = total_positive * total_negative
        aucs[start:stop] = np.divide(
            concordant,
            denominator,
            out=np.full(stop - start, np.nan, dtype=np.float64),
            where=denominator > 0,
        ) * 100.0
    return aucs


def summarize_delta(delta: np.ndarray) -> dict[str, float | int]:
    finite = delta[np.isfinite(delta)]
    if not len(finite):
        raise ValueError("bootstrap produced no finite metric deltas")
    low, median, high = np.quantile(finite, [0.025, 0.5, 0.975])
    return {
        "samples": int(len(finite)),
        "mean": float(finite.mean()),
        "median": float(median),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "probability_positive": float(np.mean(finite > 0)),
    }


def analyze_subset(
    frame: pd.DataFrame,
    mask: np.ndarray,
    bootstrap_counts: np.ndarray,
    chunk_size: int,
    reference_cache: dict[tuple[str, str], np.ndarray],
    subset_name: str,
) -> dict:
    subset = frame.loc[mask]
    labels = subset["label"].to_numpy(dtype=np.int8)
    groups = subset["group_code"].to_numpy(dtype=np.int32)
    reference_prob = subset["reference_prob"].to_numpy(dtype=np.float64)
    candidate_prob = subset["candidate_prob"].to_numpy(dtype=np.float64)
    reference_observed = observed_metrics(labels, reference_prob)
    candidate_observed = observed_metrics(labels, candidate_prob)

    reference_acc_key = (subset_name, "acc")
    reference_auc_key = (subset_name, "auc")
    if reference_acc_key not in reference_cache:
        reference_cache[reference_acc_key] = bootstrap_accuracy(
            labels, reference_prob, groups, bootstrap_counts
        )
    if reference_auc_key not in reference_cache:
        reference_cache[reference_auc_key] = bootstrap_auc(
            labels, reference_prob, groups, bootstrap_counts, chunk_size
        )

    candidate_acc = bootstrap_accuracy(labels, candidate_prob, groups, bootstrap_counts)
    candidate_auc = bootstrap_auc(
        labels, candidate_prob, groups, bootstrap_counts, chunk_size
    )
    return {
        "rows": int(len(subset)),
        "positive_rows": int(labels.sum()),
        "reference": reference_observed,
        "candidate": candidate_observed,
        "observed_delta": {
            metric: candidate_observed[metric] - reference_observed[metric]
            for metric in ("acc", "auc")
        },
        "bootstrap_delta": {
            "acc": summarize_delta(candidate_acc - reference_cache[reference_acc_key]),
            "auc": summarize_delta(candidate_auc - reference_cache[reference_auc_key]),
        },
    }


def fmt_delta(value: float) -> str:
    return f"{value:+.2f}"


def fmt_ci(summary: dict) -> str:
    return f"[{summary['ci95_low']:+.2f}, {summary['ci95_high']:+.2f}]"


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# Stage 2 Paired Dialogue-Cluster Bootstrap",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Reference: `{report['reference_name']}`",
        f"- Resamples / seed: `{report['bootstrap_samples']} / {report['seed']}`",
        f"- Dialogue clusters: `{report['dialogue_clusters']}`",
        "- Resampling unit: complete dialogue; predictions are paired on `Dialogue ID + Turn`.",
        "- Values below are percentage-point deltas (candidate minus the A-module reference).",
        "",
        "| Candidate | Split | Rows | AUC delta | AUC 95% CI | P(AUC > 0) | Acc delta | Acc 95% CI | P(Acc > 0) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate_id, candidate in report["candidates"].items():
        for subset_name in ("overall", "final", "non_final"):
            subset = candidate["subsets"][subset_name]
            auc_bootstrap = subset["bootstrap_delta"]["auc"]
            acc_bootstrap = subset["bootstrap_delta"]["acc"]
            lines.append(
                f"| `{candidate_id}` | `{subset_name}` | {subset['rows']} | "
                f"{fmt_delta(subset['observed_delta']['auc'])} | {fmt_ci(auc_bootstrap)} | "
                f"{auc_bootstrap['probability_positive']:.3f} | "
                f"{fmt_delta(subset['observed_delta']['acc'])} | {fmt_ci(acc_bootstrap)} | "
                f"{acc_bootstrap['probability_positive']:.3f} |"
            )

    lines.extend(["", "## Interpretation", ""])
    for candidate_id, candidate in report["candidates"].items():
        overall_auc = candidate["subsets"]["overall"]["bootstrap_delta"]["auc"]
        if overall_auc["ci95_low"] > 0:
            conclusion = "positive at the paired 95% interval"
        elif overall_auc["ci95_high"] < 0:
            conclusion = "negative at the paired 95% interval"
        else:
            conclusion = "statistically indistinguishable from the A-module reference at the paired 95% interval"
        lines.append(f"- `{candidate_id}` is {conclusion} on Overall AUC.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1221)
    parser.add_argument("--chunk-size", type=int, default=256)
    args = parser.parse_args()
    if args.samples <= 0 or args.chunk_size <= 0:
        parser.error("--samples and --chunk-size must be positive")
    if not REFERENCE_QUAL.is_file():
        raise FileNotFoundError(REFERENCE_QUAL)

    audits = []
    for audit_path in sorted(RESULTS.glob("*/AUDIT.json")):
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("audit_pass") is True:
            audits.append(audit)
    if not audits:
        raise RuntimeError("no audited Stage 2 candidates found")

    first_model = audits[0]["models"]["joint"]
    first_qual = RESULTS / "qual" / f"qual_{first_model}.csv"
    first_aligned = align_predictions(REFERENCE_QUAL, first_qual)
    dialogue_ids = first_aligned.attrs["dialogue_ids"]
    cluster_count = len(dialogue_ids)
    rng = np.random.default_rng(args.seed)
    bootstrap_counts = rng.multinomial(
        cluster_count,
        np.full(cluster_count, 1.0 / cluster_count),
        size=args.samples,
    )

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "reference_name": REFERENCE_NAME,
        "reference_qual": str(REFERENCE_QUAL.relative_to(ROOT)),
        "bootstrap_samples": args.samples,
        "seed": args.seed,
        "dialogue_clusters": cluster_count,
        "candidates": {},
    }
    reference_cache: dict[tuple[str, str], np.ndarray] = {}
    for audit in audits:
        candidate_id = audit["candidate_id"]
        final_model = audit["models"]["joint"]
        candidate_qual = RESULTS / "qual" / f"qual_{final_model}.csv"
        aligned = align_predictions(REFERENCE_QUAL, candidate_qual)
        if aligned.attrs["dialogue_ids"] != dialogue_ids:
            raise ValueError(f"{candidate_id} dialogue clusters differ from the reference alignment")
        masks = {
            "overall": np.ones(len(aligned), dtype=bool),
            "final": aligned["is_final"].to_numpy(dtype=bool),
            "non_final": ~aligned["is_final"].to_numpy(dtype=bool),
        }
        subsets = {
            subset_name: analyze_subset(
                aligned,
                mask,
                bootstrap_counts,
                args.chunk_size,
                reference_cache,
                subset_name,
            )
            for subset_name, mask in masks.items()
        }
        report["candidates"][candidate_id] = {
            "direction": audit["manifests"]["joint"]["env_mode"],
            "qual": str(candidate_qual.relative_to(ROOT)),
            "subsets": subsets,
        }

    json_path = RESULTS / "PAIRED_BOOTSTRAP.json"
    markdown_path = RESULTS / "PAIRED_BOOTSTRAP.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, markdown_path)
    print(markdown_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
