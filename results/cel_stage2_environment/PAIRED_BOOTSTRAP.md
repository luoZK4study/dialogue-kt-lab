# Stage 2 Paired Dialogue-Cluster Bootstrap

- Generated at: `2026-08-15T15:38:17`
- Reference: `a_module_reference`
- Resamples / seed: `10000 / 1221`
- Dialogue clusters: `515`
- Resampling unit: complete dialogue; predictions are paired on `Dialogue ID + Turn`.
- Values below are percentage-point deltas (candidate minus the A-module reference).

| Candidate | Split | Rows | AUC delta | AUC 95% CI | P(AUC > 0) | Acc delta | Acc 95% CI | P(Acc > 0) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `stage2_mlp_topk20_beta005_seed1221` | `overall` | 1985 | +0.10 | [-1.13, +1.32] | 0.570 | -1.56 | [-3.23, +0.15] | 0.032 |
| `stage2_mlp_topk20_beta005_seed1221` | `final` | 515 | -0.98 | [-3.52, +1.56] | 0.227 | -2.91 | [-6.41, +0.58] | 0.043 |
| `stage2_mlp_topk20_beta005_seed1221` | `non_final` | 1470 | +1.11 | [-0.24, +2.49] | 0.949 | -1.09 | [-2.92, +0.75] | 0.117 |
| `stage2_shuffle_topk20_beta010_seed1221` | `overall` | 1985 | -1.23 | [-2.77, +0.27] | 0.056 | -1.31 | [-3.20, +0.62] | 0.088 |
| `stage2_shuffle_topk20_beta010_seed1221` | `final` | 515 | -4.47 | [-7.64, -1.45] | 0.002 | -4.27 | [-7.96, -0.58] | 0.010 |
| `stage2_shuffle_topk20_beta010_seed1221` | `non_final` | 1470 | +0.20 | [-1.55, +1.97] | 0.603 | -0.27 | [-2.44, +1.97] | 0.404 |
| `stage2_transformer_topk20_beta010_centeredrms010_seed1221` | `overall` | 1985 | -0.13 | [-1.45, +1.18] | 0.432 | -1.96 | [-3.80, -0.10] | 0.017 |
| `stage2_transformer_topk20_beta010_centeredrms010_seed1221` | `final` | 515 | -2.54 | [-5.53, +0.44] | 0.050 | +0.39 | [-3.11, +3.89] | 0.566 |
| `stage2_transformer_topk20_beta010_centeredrms010_seed1221` | `non_final` | 1470 | +0.25 | [-1.23, +1.70] | 0.633 | -2.79 | [-4.89, -0.74] | 0.003 |

## Interpretation

- `stage2_mlp_topk20_beta005_seed1221` is statistically indistinguishable from the A-module reference at the paired 95% interval on Overall AUC.
- `stage2_shuffle_topk20_beta010_seed1221` is statistically indistinguishable from the A-module reference at the paired 95% interval on Overall AUC.
- `stage2_transformer_topk20_beta010_centeredrms010_seed1221` is statistically indistinguishable from the A-module reference at the paired 95% interval on Overall AUC.
