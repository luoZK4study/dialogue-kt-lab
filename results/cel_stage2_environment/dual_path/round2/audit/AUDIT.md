# Dual-path Stage 2 audit: round2

- Candidate: `stage2_dual_round2_contextual_transformer_seed1221`
- Audit pass: `True`
- Provenance pass: `True`
- Raw-base/seed identity: `True`
- A warmup freeze: `True`
- B warmup freeze: `True`
- A joint update: `True`
- A joint component updates: `adapter=True`, `selector=True`, `calibrator=True`
- Final joint update: `True`
- Final joint component updates: `adapter=True`, `selector=True`, `environment=True`, `calibrator=True`
- Selector FP32: `True`
- B FP32: `True`
- Complement contract: `True`

## Evaluations

### validation
- mixed Overall Acc/AUC: `65.93/72.59`
- evidence Overall Acc/AUC: `66.10/72.53`
- reversal Overall Acc/AUC: `52.65/47.40`
