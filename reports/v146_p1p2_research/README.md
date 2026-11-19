# P1/P2 research evaluation — 2026-11-19

**RESEARCH ONLY · DO NOT DEPLOY · Stage 1 v134 / Stage 2 v139 remain active.**

The attached diagnostic requested a frozen evaluation contract before any model
change. The following are exploratory comparisons from a fixed 4,795-row
snapshot and 56-material-group five-fold split, not an original-paper DOI
independent test and not strict equilibrium q_e validation.

## Stage 1 source composition coverage

| Candidate | Total target rows / DOI | CHON + Ash/VM/FC rows / DOI |
| --- | ---: | ---: |
| BET | 361 / 49 | 41 / 4 |
| TPV | 231 / 35 | 8 / 2 |

The complete rows still mix raw feedstock analysis bases (dry/daf,
unspecified). A fair A/B/C comparison using verified matched rows is not
supported; no composition predictor was fit or promoted.

## Stage 2 research results

All variants used fixed 56-material-group five-folds, the same reported-
capacity target (fraction times C0/dose), the same 4,795 held-out observations,
and the existing KNN k=10, weighted ET 100-tree settings.

| Variant | MAE mg/g | R² |
| --- | ---: | ---: |
| Original full 18 inputs, 50:50 | 17.282 | 0.7180 |
| Drop TPV and log(TPV) | 21.384 | 0.6707 |
| Drop BET and log(BET) | 27.620 | 0.1111 |
| Drop all BET/TPV inputs | 51.051 | -0.7800 |
| ET only, full 18 | 20.193 | 0.4784 |
| ET only, BET inputs | 17.848 | 0.7459 |
| KNN only, full 18 | 16.891 | 0.7653 |
| **75% KNN + 25% ET**, full 18 | **16.556** | **0.7657** |
| 25% KNN + 75% ET, full 18 | 18.498 | 0.6222 |

The v146 blend is an **exploratory candidate, not a confirmed improvement**:
relative to the 50:50 reference, delta MAE = -0.725 mg/g, but a
material-group paired bootstrap 95% interval is **[-3.604, +0.968]**, crossing
zero. Core97 MAE *worsened* from 21.513 to 21.995 mg/g.

The original v139 reference lists MAE 17.273 / R² 0.718386, whereas our
re-run on the matching SHA-256 CSV produced 17.282 / 0.718011. These scores
are close, **not byte-for-byte parity**. Historical fold implementation or
environment differences still require reconciliation.

**Source snapshot SHA-256:**
`de2dd067e8a02c481466875fa927c42b93049734f1b255739a7587bc1338388c`

The candidate class in `pyrolysis/biochar_stage2/research_blend_v146.py`
inherits v139 training exactly, modifies *only* the KNN/ET blend weights,
and is never imported by the production engine. No training data, app code,
or deployment pointer has changed.

**Promotion gate:** separate source DOI traceability, independent source-out
evaluation, frozen post-selection validation, adverse group analysis and
environment parity checks. No strict-qe claims are supported.
