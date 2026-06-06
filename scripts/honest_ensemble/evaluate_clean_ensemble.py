"""Evaluate honest Dialogue-KT ensembles from archived validation and clean test predictions.

Inputs are immutable archived CSVs:
- results/honest_ensemble/val_qual/*_val.csv
- results/honest_ensemble/test_qual/*_test_clean.csv

The script reports single models plus pre-declared/validation-selected ensemble recipes.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "honest_ensemble"

MODELS = {
    "base": (
        OUT / "val_qual" / "qual_lmkt_qwen3_1.7b_val.csv",
        OUT / "test_qual" / "qual_lmkt_qwen3_1.7b_test_clean.csv",
    ),
    "bep": (
        OUT / "val_qual" / "qual_bayes_adaptive_labels_qwen3_1.7b_val.csv",
        OUT / "test_qual" / "qual_bayes_adaptive_labels_qwen3_1.7b_test_clean.csv",
    ),
    "sip": (
        OUT / "val_qual" / "qual_sip_novelty_qwen3_1.7b_val.csv",
        OUT / "test_qual" / "qual_sip_novelty_qwen3_1.7b_test_clean.csv",
    ),
}

RECIPES = {
    # No weight tuning.
    "M1_equal_base_bep": (["base", "bep"], [0.5, 0.5], "equal average; no tuned weights"),
    # Selected only on validation by 0.1-step grid over Base/BEP.
    "M2_val_weighted_base_bep": (["base", "bep"], [0.3, 0.7], "validation-selected 0.1 grid weights"),
    # Pre-registered equal family ensemble.
    "M3_preregistered_base_sip_bep": (["base", "sip", "bep"], [1 / 3, 1 / 3, 1 / 3], "pre-registered equal average"),
}


def load_prediction(path: Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    valid = df[(df["Prob"] != "--") & df["Prob"].notna()].copy()
    valid["prob"] = valid["Prob"].astype(float)
    valid["label"] = valid["Correct"].astype(str).str.strip().str.lower().map({"true": 1, "false": 0})
    valid = valid.dropna(subset=["label"])
    valid["label"] = valid["label"].astype(int)
    return valid[["Dialogue ID", "Turn", "prob", "label"]].rename(columns={"prob": f"p_{name}"})


def load_split(split: str) -> pd.DataFrame:
    idx = 0 if split == "val" else 1
    merged = None
    for name, paths in MODELS.items():
        part = load_prediction(paths[idx], name)
        if merged is None:
            merged = part[["Dialogue ID", "Turn", "label", f"p_{name}"]]
        else:
            merged = merged.merge(part[["Dialogue ID", "Turn", f"p_{name}"]], on=["Dialogue ID", "Turn"])
    assert merged is not None
    return merged


def metrics(labels, preds):
    hard = (preds >= 0.5).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, hard, average="binary", zero_division=0)
    return {
        "acc": accuracy_score(labels, hard) * 100,
        "auc": roc_auc_score(labels, preds) * 100,
        "prec": prec * 100,
        "rec": rec * 100,
        "f1": f1 * 100,
    }


def final_mask(df: pd.DataFrame):
    return (df["Turn"] == df.groupby("Dialogue ID")["Turn"].transform("max")).to_numpy()


def add_row(rows, method, split, df, preds, members, weights, note):
    labels = df["label"].to_numpy()
    fm = final_mask(df)
    overall = metrics(labels, preds)
    final = metrics(labels[fm], preds[fm])
    rows.append({
        "method": method,
        "split": split,
        "members": "+".join(members),
        "weights": "+".join(f"{w:.3f}" for w in weights),
        "note": note,
        "n": len(df),
        "dialogues": df["Dialogue ID"].nunique(),
        "final_n": int(fm.sum()),
        **{f"overall_{k}": v for k, v in overall.items()},
        **{f"final_{k}": v for k, v in final.items()},
    })


def main():
    rows = []
    for split in ["val", "test"]:
        df = load_split(split)
        for model in MODELS:
            add_row(
                rows, model, split, df, df[f"p_{model}"].to_numpy(), [model], [1.0], "single model"
            )
        for method, (members, weights, note) in RECIPES.items():
            pred = sum(df[f"p_{m}"].to_numpy() * w for m, w in zip(members, weights))
            add_row(rows, method, split, df, pred, members, weights, note)

    out = pd.DataFrame(rows)
    out_path = OUT / "honest_ensemble_clean_results.csv"
    out.to_csv(out_path, index=False)
    print(out[["method", "split", "n", "dialogues", "overall_auc", "final_auc", "overall_acc", "overall_f1"]].to_string(index=False))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
