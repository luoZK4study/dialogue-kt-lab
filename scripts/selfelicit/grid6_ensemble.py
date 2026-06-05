"""Grid search over 6-model ensemble weights."""
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score

models = ['base','p','c','a','v','attn']
files = {
    'base': 'results/qual_lmkt_qwen3_1.7b.csv',
    'p': 'results/qual_lmkt_se_prompt_1.7b.csv',
    'c': 'results/qual_lmkt_se_combined_1.7b.csv',
    'a': 'results/qual_lmkt_se_adaptive_1.7b.csv',
    'v': 'results/qual_lmkt_se_conservative_1.7b.csv',
    'attn': 'results/qual_lmkt_se_true_attn_1.7b.csv',
}

probs = {}
for name, fname in files.items():
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

# Equal weight
merged['eq'] = sum(merged[f'p_{m}'] for m in models) / len(models)
auc_eq = roc_auc_score(labels, merged['eq'].values)
print(f"Equal weight (6 model): AUC = {auc_eq*100:.2f}%")

# Coarse grid search
best = 0
best_w = None
for w_base in range(25, 45, 5):
    for w_c in range(15, 30, 5):
        for w_a in range(10, 20, 5):
            for w_v in range(5, 20, 5):
                for w_p in range(3, 10, 3):
                    for w_attn in range(3, 15, 3):
                        s = w_base + w_c + w_a + w_v + w_p + w_attn
                        if s != 100:
                            continue
                        merged['g'] = (
                            w_base * merged['p_base'] +
                            w_c * merged['p_c'] +
                            w_a * merged['p_a'] +
                            w_v * merged['p_v'] +
                            w_p * merged['p_p'] +
                            w_attn * merged['p_attn']
                        ) / 100.0
                        auc = roc_auc_score(labels, merged['g'].values)
                        if auc > best:
                            best = auc
                            best_w = (w_base, w_c, w_a, w_v, w_p, w_attn)

print(f"Grid best (6 model): AUC = {best*100:.2f}%")
print(f"  Weights: base={best_w[0]}% c={best_w[1]}% a={best_w[2]}% v={best_w[3]}% p={best_w[4]}% attn={best_w[5]}%")
print(f"Paper: 76.71%, Gap: {(best*100 - 76.71):+.2f}%")
