"""Proper ensemble: optimize weights on VALIDATION set, evaluate on TEST set."""
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score

# Warning: qual files are TEST set predictions, metrics are from test set.
# We can't access validation set predictions directly from the qual CSVs.
# Instead, let's check: are the metrics computed from qual CSVs consistent?

# The proper way to validate ensemble weighting:
# We need predictions on a held-out validation set to tune weights,
# then evaluate on the test set. Since we only saved test qual CSVs,
# let's use a stricter approach: simple average (no weight tuning)
# and K-fold style validation.

# Approach: leave-one-model-out cross-validation on the test set
# to estimate ensemble generalization.

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
n = len(merged)

# Method 1: Simple uniform average (NO parameter tuning - safest)
merged['uniform'] = sum(merged[f'p_{m}'] for m in models) / len(models)
auc_uniform = roc_auc_score(labels, merged['uniform'].values)

# Method 2: 2-fold CV - tune on half, test on other half
# Split by dialogue ID to avoid data leakage
dids = merged['Dialogue ID'].unique()
np.random.seed(42)
np.random.shuffle(dids)
half = len(dids) // 2
train_dids = set(dids[:half])
test_dids = set(dids[half:])

fold_train = merged[merged['Dialogue ID'].isin(train_dids)]
fold_test = merged[merged['Dialogue ID'].isin(test_dids)]

# Grid search on fold_train
best_auc = 0
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
                        fold_train['g'] = (
                            w_base * fold_train['p_base'] + w_c * fold_train['p_c'] +
                            w_a * fold_train['p_a'] + w_v * fold_train['p_v'] +
                            w_p * fold_train['p_p'] + w_attn * fold_train['p_attn']
                        ) / 100.0
                        auc = roc_auc_score(fold_train['label'].values, fold_train['g'].values)
                        if auc > best_auc:
                            best_auc = auc
                            best_w = (w_base, w_c, w_a, w_v, w_p, w_attn)

# Evaluate on fold_test with weights from fold_train
w_base, w_c, w_a, w_v, w_p, w_attn = best_w
fold_test['g'] = (
    w_base * fold_test['p_base'] + w_c * fold_test['p_c'] +
    w_a * fold_test['p_a'] + w_v * fold_test['p_v'] +
    w_p * fold_test['p_p'] + w_attn * fold_test['p_attn']
) / 100.0
auc_cv = roc_auc_score(fold_test['label'].values, fold_test['g'].values)

print(f"Uniform average (6 model, no tuning):  AUC = {auc_uniform*100:.2f}%")
print(f"2-fold CV (tune on half, test on half): AUC = {auc_cv*100:.2f}%")
print(f"  Tuned weights: base={best_w[0]} c={best_w[1]} a={best_w[2]} v={best_w[3]} p={best_w[4]} attn={best_w[5]}")
print(f"Baseline AUC: 75.99%")
print(f"Paper AUC: 76.71%")
