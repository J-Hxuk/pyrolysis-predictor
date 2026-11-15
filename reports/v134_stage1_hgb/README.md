# v134 — default Stage 1 10-feature Poisson HGB
- The unified application now builds BET and TPV with the same estimator factory as the OOF evaluation. Raw nonnegative target (no log transform). Training source remains 417 candidate records; BET n=361, TPV n=231.
- Four optional activation features added; blank numeric values stay missing and are imputed within the model pipeline. Non-activation ignores activation-specific values. The separate composition BET RF path remains optional and is not covered by the HGB figure.
- Stage 2 v114 binary and inference logic unchanged. The linked Stage 1 → Stage 2 assessment has not been re-run.
- Reproduce: `python -m scripts.evaluate_stage1_hgb_v134`. 5-fold DOI-grouped OOF, no group overlap. Excluding the largest 25 BET and 23 TPV residuals is post-hoc descriptive analysis only; do not claim it is independent model performance. All observations remain in deployment training.
- Validation on reconstructed equality-coded DOI/source snapshot, with original categorical labels restored, scikit-learn 1.8.0: full BET MAE 234.668423, R² 0.599104; full TPV MAE 0.164332, R² 0.534483. Trimmed BET n=336, MAE 181.269225, R² 0.679168; trimmed TPV n=208, MAE 0.122610, R² 0.630816. Re-run with repository source and pinned environment to verify.
- No external validation; original 417-row staging data contains provisional candidate values.
