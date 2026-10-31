"""Read-only v28 diagnosis of whether manufacturing rankings are identifiable.

Requires v27 held-out DOI predictions. Does NOT retrain, select models, or touch
any deployment artifacts. The holdout cohort is the old 98-row Phase8 set.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.compare_phase8_v23 import stage2_targets
from scripts.evaluate_v27_recommendation import (
    GROUP_FIELDS, MANUFACTURING_INPUTS, extract_comparable_groups, tie_aware_accuracy,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports' / 'v28_feature_identifiability'
V27 = ROOT / 'reports' / 'v27_recommendation_audit'


def audit_identifiability(
    core: pd.DataFrame, heldout_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    groups = extract_comparable_groups(core)
    if groups.empty:
        raise ValueError('No manufacturing contrast cohorts')
    if not heldout_predictions.observation_id.is_unique:
        raise ValueError('Repeated OOF prediction ID')
    if set(groups.observation_id) != set(heldout_predictions.observation_id):
        raise ValueError('Evaluation records do not match frozen v27 cohort')
    if 'pred_b_raw' not in heldout_predictions:
        raise ValueError('Missing raw OOF predictions')

    groups = groups.merge(
        heldout_predictions[['observation_id', 'pred_b_raw']],
        on='observation_id', validate='one_to_one', how='left',
    )
    if groups.pred_b_raw.isna().any():
        raise ValueError('Missing predictions')
    details = []
    pairs = []
    for cid, sub in groups.groupby('comparison_id', sort=True):
        sub = sub.reset_index(drop=True)
        y = sub.target_value.astype(float).to_numpy()
        predictions = sub.pred_b_raw.astype(float).to_numpy()
        temp = pd.to_numeric(sub.pyrolysis_temperature_C, errors='coerce').to_numpy()
        feature_signatures = [
            tuple('<NR>' if pd.isna(val) else str(val) for val in row)
            for row in sub[MANUFACTURING_INPUTS].itertuples(index=False,name=None)
        ]
        varying = []
        for field in MANUFACTURING_INPUTS:
            if sub[field].fillna('<NR>').astype(str).nunique() > 1:
                varying.append(field)
        observed_temps = sorted(set(temp[np.isfinite(temp)]))
        temp_contrast = len(observed_temps)>1 and np.isfinite(temp).all()
        n_identical_pairs = 0
        n_valid_observed_pairs = 0
        n_identical_with_target_gap = 0
        for i,j in combinations(range(len(sub)),2):
            if abs(y[i]-y[j]) < 1e-9:
                continue
            n_valid_observed_pairs += 1
            same = feature_signatures[i] == feature_signatures[j]
            n_identical_pairs += int(same)
            n_identical_with_target_gap += int(same and abs(y[i]-y[j])>1.)
            pred_equal = np.isclose(predictions[i],predictions[j],atol=1e-9,rtol=0)
            t_comparable = np.isfinite(temp[i]) and np.isfinite(temp[j])
            pairs.append({
                'comparison_id':cid, 'doi':sub.doi.iloc[0],
                'observation_1':sub.observation_id.iloc[i], 'observation_2':sub.observation_id.iloc[j],
                'observed_gap_mg_g':float(abs(y[i]-y[j])),
                'same_model_manufacturing_inputs':bool(same),
                'predicted_tie':bool(pred_equal),
                'temperature_pair_comparable':bool(t_comparable),
                'temperature_diff_C':float(temp[i]-temp[j]) if t_comparable else None,
            })
        pairwise, n_pairs, pred_ties = tie_aware_accuracy(y,predictions)
        details.append({
            'comparison_id':cid, 'doi':sub.doi.iloc[0],
            'feedstock':str(sub.feedstock_group_for_novelty_check.iloc[0]),
            'pollutant':str(sub.pollutant.iloc[0]),
            'candidate_rows':len(sub),
            'unique_full_manufacturing_vectors':len(set(feature_signatures)),
            'varying_manufacturing_fields':';'.join(varying),
            'pyrolysis_temperature_values_reported':len(observed_temps),
            'temperature_contrast_testable':bool(temp_contrast),
            'all_temperature_missing':bool(np.isnan(temp).all()),
            'observed_pairs':n_pairs,
            'pairs_with_indistinguishable_model_inputs':n_identical_pairs,
            'indistinguishable_pairs_gap_over_1mg':n_identical_with_target_gap,
            'v27_B_pairwise':pairwise,
            'v27_predicted_tie_pairs':pred_ties,
        })
    detail = pd.DataFrame(details)
    pair_df = pd.DataFrame(pairs)
    evaluable = detail.loc[detail.observed_pairs.gt(0)]
    summary = {
        'source':'v27 frozen OOF data; 98-row/26-DOI cohort; zero training in v28',
        'comparison_groups':int(len(detail)),
        'evaluable_groups':int(len(evaluable)),
        'evaluable_DOIs':int(evaluable.doi.nunique()),
        'groups_with_verified_temperature_contrast':int(detail.temperature_contrast_testable.sum()),
        'groups_with_identical_temperature_in_all_reported_candidates':int((detail.pyrolysis_temperature_values_reported==1).sum()),
        'groups_with_all_temperature_unreported':int(detail.all_temperature_missing.sum()),
        'evaluable_groups_with_collapsed_material_inputs':int(((evaluable.pairs_with_indistinguishable_model_inputs>0)).sum()),
        'pairwise_comparisons':int(pair_df.shape[0]),
        'pairs_with_identical_manufacturing_input_signature':int(pair_df.same_model_manufacturing_inputs.sum()),
        'pairs_with_predicted_tie':int(pair_df.predicted_tie.sum()),
        'model_B_pairwise_macro_from_v27':float(evaluable.v27_B_pairwise.mean()),
        'random_tie_baseline_pairwise':0.5,
        'limitations':[
            'No held-out comparison group varies pyrolysis temperature; do not claim temperature optimization.',
            'A material pair with identical model inputs cannot be distinguished even when labels differ.',
            'Performance is post-hoc, group results within same DOI not independent.',
            'Use raw OOF predictions, never clipping, when auditing model reliability.',
            'No new DOI, no new trial, no model training or automatic activation in this study.',
        ],
    }
    return detail, pair_df, summary


def run(root:Path=ROOT, out:Path=REPORT):
    root=Path(root); out=Path(out)
    _, core=stage2_targets()
    heldout=pd.read_csv(root/'reports/v27_recommendation_audit/stage2_heldout_doi_predictions.csv')
    detail, pair_df, summary=audit_identifiability(core, heldout)
    out.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out/'groups_feature_identifiability.csv',index=False)
    pair_df.to_csv(out/'pairwise_input_collisions.csv',index=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    return summary


if __name__=='__main__':
    print(json.dumps(run(),ensure_ascii=False,indent=2))
