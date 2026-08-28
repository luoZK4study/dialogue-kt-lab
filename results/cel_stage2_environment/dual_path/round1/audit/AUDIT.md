# Dual-path Stage 2 audit: round1

- Candidate: `stage2_dual_round1_contextual_transformer_seed1221`
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
- mixed Overall Acc/AUC: `66.78/72.51`
- evidence Overall Acc/AUC: `66.84/72.52`
- reversal Overall Acc/AUC: `52.71/46.94`
