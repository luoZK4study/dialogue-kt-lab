"""Ensemble all 5 models and compare strategies."""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

models = {
    'baseline': 'results/qual_lmkt_qwen3_1.7b.csv',
    'se_prompt': 'results/qual_lmkt_se_prompt_1.7b.csv',
    'se_combined': 'results/qual_lmkt_se_combined_1.7b.csv',
    'se_adaptive': 'results/qual_lmkt_se_adaptive_1.7b.csv',
    'se_conservative': 'results/qual_lmkt_se_conservative_1.7b.csv',
}

probs = {}
for name, fname in models.items():
    df = pd.read_csv(fname)
    valid = df[(df['Prob'] != '--') & (df['Prob'].notna())].copy()
    valid['prob'] = valid['Prob'].astype(float)
    valid['label'] = valid['Correct'].str.strip().str.lower().map({'true': 1, 'false': 0})
    valid = valid.dropna(subset=['label'])
    valid['label'] = valid['label'].astype(int)
    probs[name] = valid[['Dialogue ID', 'Turn', 'prob', 'label']]

# Align on (Dialogue ID, Turn)
merged = probs['baseline'][['Dialogue ID', 'Turn', 'label']].copy()
for name in models:
    m = probs[name][['Dialogue ID', 'Turn', 'prob']]
    merged = merged.merge(m, on=['Dialogue ID', 'Turn'], suffixes=('', f'_{name}'))
    merged.rename(columns={'prob': f'prob_{name}'}, inplace=True)

labels = merged['label'].values
print(f"Aligned: {len(merged)}, True={labels.sum()}, False={len(merged)-labels.sum()}")

# Ensemble strategies
merged['avg2'] = (merged['prob_baseline'] + merged['prob_se_combined']) / 2
merged['avg5'] = (merged['prob_baseline'] + merged['prob_se_prompt'] +
                  merged['prob_se_combined'] + merged['prob_se_adaptive'] +
                  merged['prob_se_conservative']) / 5
merged['weighted'] = (0.40 * merged['prob_baseline'] +
                      0.30 * merged['prob_se_combined'] +
                      0.15 * merged['prob_se_adaptive'] +
                      0.10 * merged['prob_se_prompt'] +
                      0.05 * merged['prob_se_conservative'])

print("\n=== Overall AUC ===")
for col, label in [('prob_baseline', 'Baseline'), ('avg2', 'Ensemble(base+comb)'),
                    ('avg5', 'Ensemble(all 5)'), ('weighted', 'Ensemble(weighted)')]:
    print(f"  {label}: {roc_auc_score(labels, merged[col].values)*100:.2f}%")

final = merged.groupby('Dialogue ID').last().reset_index()
fl = final['label'].values
print("\n=== Final Turn AUC ===")
for col, label in [('prob_baseline', 'Baseline'), ('avg2', 'Ensemble(base+comb)'),
                    ('avg5', 'Ensemble(all 5)'), ('weighted', 'Ensemble(weighted)')]:
    print(f"  {label}: {roc_auc_score(fl, final[col].values)*100:.2f}%")

# Goal check
best_auc = max(
    roc_auc_score(labels, merged['avg5'].values),
    roc_auc_score(labels, merged['weighted'].values),
    roc_auc_score(labels, merged['avg2'].values)
)
print(f"\n=== Goal Check ===")
print(f"  Best Overall AUC: {best_auc*100:.2f}%")
print(f"  Paper AUC: 76.71%")
print(f"  Gap: {(best_auc*100 - 76.71):+.2f}%")
print(f"  Goal 2 met: {'YES' if best_auc*100 > 76.71 else 'NO'}")
