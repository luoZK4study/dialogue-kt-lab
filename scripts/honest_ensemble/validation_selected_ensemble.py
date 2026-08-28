"""Validation-selected ensemble evaluation for Dialogue-KT.

This script selects ensemble members/weights using validation qual CSVs only,
then evaluates the fixed recipes on test qual CSVs.
"""
from __future__ import annotations

from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "honest_ensemble"
VAL_DIR = OUT_DIR / "val_qual"
TEST_DIRS = [
    ROOT / "results" / "selfelicit" / "qual",
    ROOT / "results" / "bayesian_epistemology" / "qual",
    ROOT / "results" / "informativeness_search" / "qual",
]

MODELS = {
    "base": "qual_lmkt_qwen3_1.7b.csv",
    "sip": "qual_sip_novelty_qwen3_1.7b.csv",
    "bep": "qual_bayes_adaptive_labels_qwen3_1.7b.csv",
    "se_prompt": "qual_lmkt_se_prompt_1.7b.csv",
    "se_combined": "qual_lmkt_se_combined_1.7b.csv",
    "se_adaptive": "qual_lmkt_se_adaptive_1.7b.csv",
    "se_conservative": "qual_lmkt_se_conservative_1.7b.csv",
    "se_true_attn": "qual_lmkt_se_true_attn_1.7b.csv",
}


def _find_test_file(filename: str) -> Path:
    for d in TEST_DIRS:
        p = d / filename
        if p.exists():
            return p
    raise FileNotFoundError(filename)


def _val_file(filename: str) -> Path:
    stem = filename[:-4]
    return VAL_DIR / f"{stem}_val.csv"


def load_qual(path: Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    valid = df[(df["Prob"] != "--") & df["Prob"].notna()].copy()
    valid["prob"] = valid["Prob"].astype(float)
    valid["label"] = valid["Correct"].astype(str).str.strip().str.lower().map({"true": 1, "false": 0})
    valid = valid.dropna(subset=["label"])
    valid["label"] = valid["label"].astype(int)
    valid = valid[["Dialogue ID", "Turn", "prob", "label"]]
    valid = valid.rename(columns={"prob": f"p_{name}"})
    return valid


def load_split(split: str, models: dict[str, str] = MODELS) -> pd.DataFrame:
    merged = None
    for name, fname in models.items():
        path = _val_file(fname) if split == "val" else _find_test_file(fname)
        df = load_qual(path, name)
        if merged is None:
            merged = df[["Dialogue ID", "Turn", "label", f"p_{name}"]]
        else:
            merged = merged.merge(df[["Dialogue ID", "Turn", f"p_{name}"]], on=["Dialogue ID", "Turn"], how="inner")
    assert merged is not None
    return merged


def metrics(labels, preds):
    hard = np.round(preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, hard, average="binary", zero_division=0)
    return {
        "acc": accuracy_score(labels, hard) * 100,
        "auc": roc_auc_score(labels, preds) * 100,
        "prec": prec * 100,
        "rec": rec * 100,
        "f1": f1 * 100,
    }


def final_mask(df: pd.DataFrame) -> np.ndarray:
    # Rows are unique turn predictions; final turn = last predicted turn in each dialogue.
    max_turn = df.groupby("Dialogue ID")["Turn"].transform("max")
    return (df["Turn"] == max_turn).to_numpy()


def predict(df: pd.DataFrame, recipe: dict) -> np.ndarray:
    members = recipe["members"]
    weights = np.array(recipe["weights"], dtype=float)
    weights = weights / weights.sum()
    arr = np.stack([df[f"p_{m}"].to_numpy() for m in members], axis=1)
    pred = arr @ weights
    if recipe.get("calibrate"):
        # Validation-selected affine calibration; clipping preserves probability validity.
        pred = np.clip(recipe["a"] * pred + recipe["b"], 0.0, 1.0)
    return pred


def grid_weights(n: int, step: float = 0.1):
    ticks = int(round(1.0 / step))
    for parts in product(range(ticks + 1), repeat=n):
        if sum(parts) == ticks and all(p > 0 for p in parts):
            yield np.array(parts, dtype=float) / ticks


def select_recipes(val: pd.DataFrame):
    labels = val["label"].to_numpy()
    names = list(MODELS)
    recipes = []

    # Method 1: choose equal-weight member subset by validation AUC.
    for r in range(2, min(6, len(names)) + 1):
        for members in combinations(names, r):
            recipe = {"name": "subset_equal", "members": members, "weights": [1.0] * r}
            pred = predict(val, recipe)
            recipes.append((metrics(labels, pred)["auc"], recipe))

    # Method 2: choose coarse weighted ensemble over strongest prior candidates by validation AUC.
    # Candidate pool is fixed before seeing test, based on prior single-model results/complementarity.
    pool = ["base", "sip", "bep", "se_combined", "se_adaptive"]
    for r in range(2, len(pool) + 1):
        for members in combinations(pool, r):
            for w in grid_weights(r, step=0.1):
                recipe = {"name": "val_weighted", "members": members, "weights": w.tolist()}
                pred = predict(val, recipe)
                recipes.append((metrics(labels, pred)["auc"], recipe))

    # Method 3: equal-weight subset plus validation-selected linear calibration.
    # AUC is invariant to positive affine transforms before clipping in many cases, but clipping can help metrics/F1.
    for r in range(2, min(5, len(names)) + 1):
        for members in combinations(names, r):
            base_recipe = {"name": "subset_equal_calibrated", "members": members, "weights": [1.0] * r}
            raw = predict(val, base_recipe)
            for a in np.arange(0.8, 1.31, 0.1):
                for b in np.arange(-0.10, 0.101, 0.02):
                    pred = np.clip(a * raw + b, 0.0, 1.0)
                    recipe = {**base_recipe, "calibrate": True, "a": float(round(a, 2)), "b": float(round(b, 2))}
                    recipes.append((metrics(labels, pred)["auc"], recipe))

    best_by_name = {}
    for auc, recipe in recipes:
        key = recipe["name"]
        if key not in best_by_name or auc > best_by_name[key][0]:
            best_by_name[key] = (auc, recipe)
    return best_by_name


def describe_recipe(recipe: dict) -> str:
    parts = [f"{m}:{w:.3g}" for m, w in zip(recipe["members"], recipe["weights"])]
    s = f"{recipe['name']} [" + ", ".join(parts) + "]"
    if recipe.get("calibrate"):
        s += f" calib(a={recipe['a']}, b={recipe['b']})"
    return s


def main():
    val = load_split("val")
    test = load_split("test")
    recipes = select_recipes(val)

    rows = []
    for name, (val_auc, recipe) in sorted(recipes.items(), key=lambda kv: -kv[1][0]):
        for split, df in [("val", val), ("test", test)]:
            labels = df["label"].to_numpy()
            preds = predict(df, recipe)
            overall = metrics(labels, preds)
            fm = final_mask(df)
            final = metrics(labels[fm], preds[fm])
            rows.append({
                "method": name,
                "split": split,
                "recipe": describe_recipe(recipe),
                "n": len(labels),
                "overall_auc": overall["auc"],
                "overall_acc": overall["acc"],
                "overall_f1": overall["f1"],
                "final_n": int(fm.sum()),
                "final_auc": final["auc"],
                "final_acc": final["acc"],
                "final_f1": final["f1"],
            })

    # Also report pre-registered simple ensembles useful as independent feasible methods.
    prereg = {
        "prereg_base_bep_equal": {"name": "prereg_base_bep_equal", "members": ("base", "bep"), "weights": [1, 1]},
        "prereg_base_sip_bep_equal": {"name": "prereg_base_sip_bep_equal", "members": ("base", "sip", "bep"), "weights": [1, 1, 1]},
        "prereg_all_equal": {"name": "prereg_all_equal", "members": tuple(MODELS), "weights": [1] * len(MODELS)},
    }
    for name, recipe in prereg.items():
        for split, df in [("val", val), ("test", test)]:
            labels = df["label"].to_numpy()
            preds = predict(df, recipe)
            overall = metrics(labels, preds)
            fm = final_mask(df)
            final = metrics(labels[fm], preds[fm])
            rows.append({
                "method": name,
                "split": split,
                "recipe": describe_recipe(recipe),
                "n": len(labels),
                "overall_auc": overall["auc"],
                "overall_acc": overall["acc"],
                "overall_f1": overall["f1"],
                "final_n": int(fm.sum()),
                "final_auc": final["auc"],
                "final_acc": final["acc"],
                "final_f1": final["f1"],
            })

    out = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "honest_ensemble_results.csv"
    out.to_csv(out_path, index=False)

    lines = ["# Honest validation-selected ensemble results", ""]
    for _, row in out.sort_values(["split", "overall_auc"], ascending=[True, False]).iterrows():
        lines.append(
            f"- {row['split']:4s} | {row['method']} | Overall AUC {row['overall_auc']:.3f} | "
            f"Final AUC {row['final_auc']:.3f} | {row['recipe']}"
        )
    report = "\n".join(lines) + "\n"
    (OUT_DIR / "HonestEnsemble_DialogueKT_实验记录.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
