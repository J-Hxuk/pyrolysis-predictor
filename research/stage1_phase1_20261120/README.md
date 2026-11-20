# Stage 1 Phase 1/2 — research-only diagnostics and feature ablation (2026-11-20)

**Not deployment.** This branch is deliberately separate from `main`. It does **not** replace Stage 1 v134, Stage 2 v139 (KNN / Extra Trees 50:50), the Streamlit UI, the manufacturing ranking model, or published poster metrics.

## Data and scope

- Uses `포스터용 최종 압축(6).zip` (2026-11-19 user-supplied archival source), specifically `data/imports/Biochar_Stage1_v60/Biochar_Stage1_v60_Final/`.
- `01_Stage1_Master_v60_256.csv`, archived DOI-LODO `02_BET_AllData_FixedArchitecture_LODO.csv` (254 rows / 30 DOI) and `02_TPV_AllData_FixedArchitecture_LODO.csv` (191 rows / 27 DOI).
- **This cohort is not the final Stage 1 v134 Full OOF** (BET 361/49 DOI, TPV 231/35 DOI). Archived v60 predictions are for diagnosis only; do not compare their MAE directly with the *new* 5-fold HGB model as if they used the same CV protocol.
- Formerly exposed test papers occur in the development data. The 5-fold results are exploratory, **not** a sealed external challenge.

## Tested, predefined candidate models

All run with the same `GroupKFold(5)` grouped by DOI, with within-fold preprocessing:

1. `HGB10`: 10 primary Stage 1 manufacturing/feedstock fields.
2. `HGB10_plus_process`: primary fields + detailed process and activation fields already recorded in the Stage 1 archival master.
3. `HGB10_plus_basis_safe_composition`: primary fields + separately named source-basis raw feedstock composition columns. Only `daf` and `moisture-free feedstock, paper p4` accepted into different feature columns. Unverified composition is not imputed from another feedstock.

Algorithm: scikit-learn `HistGradientBoostingRegressor(loss="poisson",max_iter=120,learning_rate=.06,max_leaf_nodes=15,min_samples_leaf=8,l2_regularization=5,early_stopping=False,random_state=20261120)`. No tuning against outer held-out groups.

## Test results (research-only, NOT v134)

| Target | Variant | MAE | R2 | DOI-macro MAE |
|---|---|---:|---:|---:|
| BET (m²/g) | HGB10 | 322.810 | 0.323 | 374.702 |
| BET | + process | **300.470** | **0.386** | **350.939** |
| BET | + basis-safe composition | 322.810 | 0.323 | 374.702 |
| TPV (cm³/g) | HGB10 | **0.19724** | **0.271** | 0.19788 |
| TPV | + process | 0.20469 | 0.250 | 0.19780 |
| TPV | + basis-safe composition | 0.19724 | 0.271 | 0.19788 |

Paired DOI bootstrap CI for DOI-macro MAE improvement from **adding process conditions** (positive favors extra inputs):

- BET: **+23.763 m²/g, 95% CI [−21.222, +72.755]**. Worst DOI MAE worsened from 777.978 to 1035.091 m²/g, and 9 DOIs worsened.
- TPV: **+0.000086 cm³/g, 95% CI [−0.019910, +0.021626]**. Pooled TPV MAE worsened.

Neither interval excludes zero. Basis-safe composition made identical predictions in this constrained data coverage (BET 28 rows / 2 DOI; TPV 11 rows / 1 DOI), which **does not** imply that raw composition is generally unhelpful.

## Archived v60 error characteristics (no retraining)

| Route | BET MAE (m²/g) | TPV MAE (cm³/g) |
|---|---:|---:|
| Non-activated | 66.772 | 0.06171 |
| Physical activation | 286.121 | 0.20877 |
| Chemical activation | **420.870** | **0.27327** |

These routes have different target ranges and DOI composition; do not treat association as a causal effect.

## How to run

The large poster ZIP is intentionally not committed. Supply the locally held archive (or the chat-produced input subset containing the three exact original paths).

```bash
python research/stage1_phase1_20261120/run_experiment.py \
  --archive "/path/to/포스터용 최종 압축(6).zip" \
  --out research_outputs/stage1_phase1_20261120
```

Outputs: paired OOF, fold audit, DOI-level MAE tables and JSON metrics. Python dependencies: pandas, numpy, scikit-learn >=1.5.

## Next gates before any model promotion

1. Obtain and validate **final v134 Stage1 361/231 row-level OOF**, not just a summary screenshot.
2. Validate feedstock chemistry measurement basis, target BET/TPV definitions, and process data for high-error DOI.
3. Stage 1→2 error transfer requires a paired **same 178 linked rows / 36 DOI** comparison on final Stage 2 v139 50:50 using measured vs genuinely DOI-cross-fitted predicted properties. Both stages must exclude held-out DOIs; neither training target nor held-out label can enter preprocessing.
4. Manufacturing Top-1/regret requires same-source multiple manufacturing recipes with matched adsorption conditions and actual measured uptake outcomes.
5. Preserve production artifacts and poster metrics until a prospective, sealed external test supports the change.

Report and full local artifact package are separate from this branch; they include additional visualizations, frozen input hashes, and labeled research-only fitted prototype models.
