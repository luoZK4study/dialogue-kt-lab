"""Compute ensemble AUC from baseline + SE model predictions."""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import json

# Load predictions from both models
base_qual = pd.read_csv("results/qual_lmkt_qwen3_1.7b.csv")
se_qual = pd.read_csv("results/qual_lmkt_se_combined_1.7b.csv")

# Filter valid predictions
base_valid = base_qual[(base_qual["Prob"] != "--") & (base_qual["Prob"].notna())].copy()
base_valid["prob"] = base_valid["Prob"].astype(float)
# Handle different capitalization
base_valid["label"] = base_valid["Correct"].str.strip().str.lower().map({"true": 1, "false": 0})
base_valid = base_valid.dropna(subset=["label"])
base_valid["label"] = base_valid["label"].astype(int)
print(f"Baseline valid: {len(base_valid)}, True: {base_valid['label'].sum()}, False: {len(base_valid)-base_valid['label'].sum()}")

se_valid = se_qual[(se_qual["Prob"] != "--") & (se_qual["Prob"].notna())].copy()
se_valid["prob"] = se_valid["Prob"].astype(float)
se_valid = se_valid[["Dialogue ID", "Turn", "prob"]]
print(f"SE valid: {len(se_valid)}")

# Merge
merged = base_valid.merge(se_valid, on=["Dialogue ID", "Turn"], suffixes=("_base", "_se"))
print(f"Aligned: {len(merged)}, True: {merged['label'].sum()}, False: {len(merged)-merged['label'].sum()}")

# Ensemble
merged["prob_ensemble"] = (merged["prob_base"] + merged["prob_se"]) / 2
labels = merged["label"].values

print(f"\n=== Overall AUC ===")
print(f"Baseline:       {roc_auc_score(labels, merged['prob_base'].values)*100:.2f}%")
print(f"SE-Combined:    {roc_auc_score(labels, merged['prob_se'].values)*100:.2f}%")
print(f"Ensemble:       {roc_auc_score(labels, merged['prob_ensemble'].values)*100:.2f}%")

# Final Turn
final = merged.groupby("Dialogue ID").last().reset_index()
fl = final["label"].values
print(f"\n=== Final Turn AUC ===")
print(f"Baseline:       {roc_auc_score(fl, final['prob_base'].values)*100:.2f}%")
print(f"SE-Combined:    {roc_auc_score(fl, final['prob_se'].values)*100:.2f}%")
print(f"Ensemble:       {roc_auc_score(fl, final['prob_ensemble'].values)*100:.2f}%")
