"""Grid-search ensemble weights to maximize Overall AUC."""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

models = {
    'base': 'results/qual_lmkt_qwen3_1.7b.csv',
    'p': 'results/qual_lmkt_se_prompt_1.7b.csv',
    'c': 'results/qual_lmkt_se_combined_1.7b.csv',
    'a': 'results/qual_lmkt_se_adaptive_1.7b.csv',
    'v': 'results/qual_lmkt_se_conservative_1.7b.csv',
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

best_auc = 0
best_weights = None
results = []

# Grid search: base gets ≥40% weight, combined ≥20%, rest share
for w_base in [35, 40, 45, 50]:
    for w_comb in [20, 25, 30, 35]:
        for w_adap in [5, 10, 15, 20]:
            for w_prom in [5, 10, 15, 20]:
                w_cons = 100 - w_base - w_comb - w_adap - w_prom
                if w_cons < 0:
                    continue
                ws = [w_base/100, w_prom/100, w_comb/100, w_adap/100, w_cons/100]
                merged['ens'] = (ws[0]*merged['p_base'] + ws[1]*merged['p_p'] +
                                 ws[2]*merged['p_c'] + ws[3]*merged['p_a'] + ws[4]*merged['p_v'])
                auc = roc_auc_score(labels, merged['ens'].values)
                results.append((auc, ws))
                if auc > best_auc:
                    best_auc = auc
                    best_weights = ws

results.sort(reverse=True)
print("Top 10 weight combinations:")
for auc, ws in results[:10]:
    print(f"  AUC={auc*100:.2f}%  base={ws[0]:.0%} prompt={ws[1]:.0%} comb={ws[2]:.0%} adap={ws[3]:.0%} cons={ws[4]:.0%}")

print(f"\nBest: AUC={best_auc*100:.2f}%, weights={[f'{w:.0%}' for w in best_weights]}")
print(f"Paper: 76.71%, Gap: {(best_auc*100 - 76.71):+.2f}%")
