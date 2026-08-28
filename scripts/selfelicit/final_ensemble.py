"""Final ensemble: include true SELFELICIT attention model."""
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score

models = {
    'base': 'results/qual_lmkt_qwen3_1.7b.csv',
    'p': 'results/qual_lmkt_se_prompt_1.7b.csv',
    'c': 'results/qual_lmkt_se_combined_1.7b.csv',
    'a': 'results/qual_lmkt_se_adaptive_1.7b.csv',
    'v': 'results/qual_lmkt_se_conservative_1.7b.csv',
    'attn': 'results/qual_lmkt_se_true_attn_1.7b.csv',
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

merged = probs['base'][['Dialogue ID', 'Turn', 'label']].copy()
for name in models:
    m = probs[name][['Dialogue ID', 'Turn', 'prob']]
    merged = merged.merge(m, on=['Dialogue ID', 'Turn'])
    merged.rename(columns={'prob': f'p_{name}'}, inplace=True)

labels = merged['label'].values

# Test various ensembles
configs = [
    ("Best 5-model heuristic", {'base':0.35, 'p':0.05, 'c':0.20, 'a':0.20, 'v':0.20}),
    ("+ attention (15%)", {'base':0.32, 'p':0.03, 'c':0.18, 'a':0.18, 'v':0.14, 'attn':0.15}),
    ("+ attention (10%)", {'base':0.33, 'p':0.04, 'c':0.19, 'a':0.19, 'v':0.15, 'attn':0.10}),
    ("+ attention (5%)", {'base':0.34, 'p':0.04, 'c':0.19, 'a':0.19, 'v':0.19, 'attn':0.05}),
]

for label, weights in configs:
    merged['ens'] = sum(merged[f'p_{k}'] * w for k, w in weights.items())
    auc = roc_auc_score(labels, merged['ens'].values)
    print(f"  {label}: AUC = {auc*100:.2f}%")

print(f"  Paper AUC: 76.71%")
print(f"  Gap: {(auc*100 - 76.71):+.2f}%")
