# Dual-path Stage 2 audit: round3

- Candidate: `stage2_dual_round3_contextual_transformer_seed1221`
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
- mixed Overall Acc/AUC: `63.82/68.67`
- evidence Overall Acc/AUC: `63.25/68.64`
- reversal Overall Acc/AUC: `52.65/47.51`

### test
- mixed Overall Acc/AUC: `67.15/71.59`
- evidence Overall Acc/AUC: `66.85/71.54`
- reversal Overall Acc/AUC: `52.44/44.02`
