"""
Evaluate new ensemble methods using newly trained checkpoints.

Uses the same validation-selected / pre-registered protocol as the
existing honest ensemble evaluation, but with new model components.

New methods:
  M4: baseline + rank_auc (val grid-selected weights)
  M5: baseline + mil_noisy_and (val grid-selected weights)
  M6: baseline + rank_auc + mil_noisy_and (val grid-selected weights)
  M7: rank_auc + mil_noisy_and (equal weight)
  M8: baseline + BEP + rank_auc (val grid-selected weights)
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


def generate_predictions(model_name, checkpoint, args_dict, split="test"):
    """Generate prediction CSV for a trained model."""
    import subprocess
    cmd = [
        sys.executable, "-m", "dialogue_kt.main", "test",
        "--dataset", "mathdial", "--model_type", "lmkt",
        "--base_model", args_dict.get("base_model", "/home/user4/.cache/huggingface/hub/qwen/Qwen3-1.7B"),
        "--model_name", checkpoint,
        "--quantize", "0",
    ]
    if split == "val":
        cmd.append("--testonval")

    # Add method-specific args
    for k, v in args_dict.items():
        if k != "base_model" and v is not None:
            cmd.extend([f"--{k}", str(v)])

    subprocess.run(cmd, check=True)

    # Move output to ensemble directory
    src_qual = RESULTS / f"qual_{checkpoint}.csv"
    dst_dir = RESULTS / "honest_ensemble" / f"{split}_qual"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"qual_{checkpoint}_{split}.csv"
    if src_qual.exists():
        src_qual.rename(dst)
        print(f"  Saved {dst}")
    return dst


def load_predictions(csv_path: Path, name: str) -> pd.DataFrame:
    """Load prediction CSV into standardized format."""
    df = pd.read_csv(csv_path)
    valid = df[(df["Prob"] != "--") & df["Prob"].notna()].copy()
    valid["prob"] = valid["Prob"].astype(float)
    valid["label"] = valid["Correct"].astype(str).str.strip().str.lower().map({"true": 1, "false": 0})
    valid = valid.dropna(subset=["label"])
    valid["label"] = valid["label"].astype(int)
    return valid[["Dialogue ID", "Turn", "prob", "label"]].rename(columns={"prob": f"p_{name}"})


def merge_predictions(model_paths: dict, split: str) -> pd.DataFrame:
    """Merge predictions from multiple models."""
    merged = None
    for name, path in model_paths.items():
        part = load_predictions(Path(path), name)
        if merged is None:
            merged = part
        else:
            merged = merged.merge(part[["Dialogue ID", "Turn", f"p_{name}"]], on=["Dialogue ID", "Turn"])
    return merged


def final_turn_mask(df: pd.DataFrame) -> np.ndarray:
    return (df["Turn"] == df.groupby("Dialogue ID")["Turn"].transform("max")).to_numpy()


def grid_search_weights(val_df, model_names, step=0.1):
    """Grid search ensemble weights on validation set."""
    best_auc = 0
    best_weights = None

    n = len(model_names)
    if n == 2:
        for w in np.arange(0, 1.01, step):
            weights = [w, 1 - w]
            pred = sum(val_df[f"p_{m}"].to_numpy() * weights[i] for i, m in enumerate(model_names))
            auc = roc_auc_score(val_df["label"], pred) * 100
            if auc > best_auc:
                best_auc = auc
                best_weights = weights
    elif n == 3:
        for w1 in np.arange(0, 1.01, step):
            for w2 in np.arange(0, 1.01 - w1, step):
                w3 = 1 - w1 - w2
                weights = [w1, w2, w3]
                pred = sum(val_df[f"p_{m}"].to_numpy() * weights[i] for i, m in enumerate(model_names))
                auc = roc_auc_score(val_df["label"], pred) * 100
                if auc > best_auc:
                    best_auc = auc
                    best_weights = weights

    return list(best_weights) if best_weights else [1.0 / n] * n


def evaluate_ensemble(test_df, model_names, weights, method_name):
    """Evaluate ensemble on test set and return metrics."""
    pred = sum(test_df[f"p_{m}"].to_numpy() * weights[i] for i, m in enumerate(model_names))
    labels = test_df["label"].to_numpy()
    fm = final_turn_mask(test_df)

    overall_auc = roc_auc_score(labels, pred) * 100
    final_auc = roc_auc_score(labels[fm], pred[fm]) * 100

    return {
        "method": method_name,
        "members": "+".join(model_names),
        "weights": [round(w, 3) for w in weights],
        "overall_auc": round(overall_auc, 2),
        "final_auc": round(final_auc, 2),
        "n_samples": len(test_df),
        "n_dialogues": test_df["Dialogue ID"].nunique(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true", help="Generate prediction CSVs from checkpoints")
    parser.add_argument("--base_model", default="/home/user4/.cache/huggingface/hub/qwen/Qwen3-1.7B")
    args = parser.parse_args()

    # Define new model checkpoints (name: (checkpoint, extra_args))
    NEW_MODELS = {
        "base": ("lmkt_qwen3_1.7b", {}),
        "bep": ("bayes_adaptive_labels_qwen3_1.7b", {"bayes_evidence": "adaptive_labels"}),
        "sip": ("sip_novelty_qwen3_1.7b", {"informativeness": "novelty"}),
        "rank_auc": ("lmkt_rank_auc_qwen3_1.7b", {"kt_method": "rank_auc", "rank_loss_weight": 0.1, "rank_margin": 0.05}),
        "mil_noisy_and": ("lmkt_mil_noisy_and_qwen3_1.7b", {"kt_method": "mil_noisy_and", "aux_loss_weight": 0.2}),
        "focal": ("lmkt_focal_qwen3_1.7b", {"kt_method": "focal_loss", "focal_gamma": 2.0}),
        "support": ("lmkt_support_token_qwen3_1.7b", {"kt_method": "support_token"}),
    }

    # Generate predictions if requested
    if args.generate:
        for name, (ckpt, extra) in NEW_MODELS.items():
            for split in ["val", "test"]:
                print(f"Generating {split} predictions for {name} ({ckpt})...")
                try:
                    generate_predictions(name, ckpt, {"base_model": args.base_model, **extra}, split)
                except Exception as e:
                    print(f"  Failed: {e}")

    # Collect available prediction files
    model_paths = {}
    for split in ["val", "test"]:
        qual_dir = RESULTS / "honest_ensemble" / f"{split}_qual"
        if not qual_dir.exists():
            continue
        for csv_file in qual_dir.glob("*.csv"):
            # Extract model name from filename
            fname = csv_file.stem.replace("qual_", "").replace(f"_{split}", "")
            for model_name in NEW_MODELS:
                if fname.startswith(model_name) or model_name in fname:
                    model_paths.setdefault(model_name, {})[split] = str(csv_file)
                    break

    print(f"\nFound predictions for: {list(model_paths.keys())}")
    if len(model_paths) < 2:
        print("Need at least 2 models. Run with --generate first.")
        return

    # Load all available predictions
    val_data = {}
    test_data = {}
    for name, paths in model_paths.items():
        if "val" in paths and "test" in paths:
            val_data[name] = paths["val"]
            test_data[name] = paths["test"]

    val_df = merge_predictions(val_data, "val")
    test_df = merge_predictions(test_data, "test")

    model_names = list(val_data.keys())
    print(f"Models for ensemble: {model_names}")

    # Define ensemble recipes
    recipes = []

    # M4: baseline + rank_auc (grid-selected)
    if "base" in model_names and "rank_auc" in model_names:
        members = ["base", "rank_auc"]
        weights = grid_search_weights(val_df, members)
        recipes.append(("M4_base_rank_auc", members, weights, "val grid-selected"))

    # M5: baseline + mil_noisy_and (grid-selected)
    if "base" in model_names and "mil_noisy_and" in model_names:
        members = ["base", "mil_noisy_and"]
        weights = grid_search_weights(val_df, members)
        recipes.append(("M5_base_mil_noisy_and", members, weights, "val grid-selected"))

    # M6: baseline + rank_auc + mil_noisy_and (grid-selected)
    if all(m in model_names for m in ["base", "rank_auc", "mil_noisy_and"]):
        members = ["base", "rank_auc", "mil_noisy_and"]
        weights = grid_search_weights(val_df, members)
        recipes.append(("M6_base_rank_mil", members, weights, "val grid-selected"))

    # M7: rank_auc + mil_noisy_and (equal weight)
    if "rank_auc" in model_names and "mil_noisy_and" in model_names:
        members = ["rank_auc", "mil_noisy_and"]
        weights = [0.5, 0.5]
        recipes.append(("M7_rank_mil_equal", members, weights, "equal weight"))

    # M8: baseline + BEP + rank_auc (grid-selected)
    if all(m in model_names for m in ["base", "bep", "rank_auc"]):
        members = ["base", "bep", "rank_auc"]
        weights = grid_search_weights(val_df, members)
        recipes.append(("M8_base_bep_rank", members, weights, "val grid-selected"))

    # Evaluate all recipes
    results = []
    # Single model baselines
    for name in model_names:
        r = evaluate_ensemble(test_df, [name], [1.0], f"single_{name}")
        results.append(r)

    # Ensembles
    for method, members, weights, note in recipes:
        r = evaluate_ensemble(test_df, members, weights, method)
        r["note"] = note
        results.append(r)

    # Print results
    print("\n" + "=" * 80)
    print(f"{'Method':<35} {'Overall AUC':>12} {'Final AUC':>10} {'Members':>30}")
    print("-" * 80)
    for r in sorted(results, key=lambda x: x["overall_auc"], reverse=True):
        exceeds = "✅" if r["overall_auc"] > 76.71 else ""
        print(f"{r['method']:<35} {r['overall_auc']:>10.2f}% {r['final_auc']:>8.2f}% {str(r.get('members','')):>30} {exceeds}")

    # Save
    out_path = RESULTS / "honest_ensemble" / "new_ensemble_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Count methods above 76.71%
    above = [r for r in results if r["overall_auc"] > 76.71 and not r["method"].startswith("single_")]
    print(f"\n{len(above)} ensemble methods surpass 76.71%:")
    for r in above:
        print(f"  {r['method']}: {r['overall_auc']:.2f}%")


if __name__ == "__main__":
    main()
