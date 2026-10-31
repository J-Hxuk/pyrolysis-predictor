"""v27 DOI-heldout manufacturing ranking and stage1 reliability regression tests."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_v27_recommendation import (
    ROOT, REPORT, GROUP_FIELDS, MANUFACTURING_INPUTS,
    extract_comparable_groups, tie_aware_accuracy, expected_top1_and_regret,
    score_group, stage1_strata_audit, doi_uncertainty,
)
from scripts.compare_phase8_v23 import stage2_targets


def _out():
    return json.loads((REPORT/'summary.json').read_text(encoding='utf-8'))


def test_true_ties_are_unrankable_and_predicted_ties_receive_half_credit():
    assert tie_aware_accuracy([5,5],[1,2]) == (None,0,0)
    assert tie_aware_accuracy([1,3],[5,5]) == (.5,1,1)
    assert tie_aware_accuracy([1,3],[5,6]) == (1.,1,0)
    assert tie_aware_accuracy([1,3],[6,5]) == (0.,1,0)
    assert tie_aware_accuracy([1,1.5],[5,6],min_difference=1.) == (None,0,0)


def test_top1_tie_expectation_is_not_arbitrary_choice():
    hit,regret,n = expected_top1_and_regret([10,12,5],[9,9,3])
    assert hit==.5 and regret==1.0 and n==2
    hit,regret,n = expected_top1_and_regret([10,12,5],[9,9,9])
    assert hit==pytest.approx(1/3) and regret==pytest.approx(3.) and n==3


def test_groups_do_not_mix_adsorption_conditions_or_doi():
    _,core=stage2_targets()
    grouped=extract_comparable_groups(core)
    assert grouped.comparison_id.nunique()==12
    assert grouped.observation_id.is_unique
    for group,chunk in grouped.groupby('comparison_id'):
        assert len(chunk[GROUP_FIELDS].drop_duplicates())==1
        assert len(chunk[MANUFACTURING_INPUTS].drop_duplicates())>=2
        assert chunk.doi.nunique()==1
        assert chunk.observation_id.is_unique


def test_predicted_heldouts_match_frozen_data_and_training_excludes_doi():
    _,core=stage2_targets()
    predictions=pd.read_csv(REPORT/'stage2_heldout_doi_predictions.csv')
    assert len(predictions)==33 and predictions.observation_id.is_unique
    assert predictions.doi.nunique()==8
    assert set(predictions.observation_id).issubset(set(core.observation_id))
    assert (predictions.training_dois == 25).all()
    assert np.isfinite(predictions.pred_b_raw).all()


def test_physical_gate_is_not_hidden_by_clipping():
    groups=pd.read_csv(REPORT/'stage2_strict_comparison_groups.csv')
    assert len(groups)==12
    assert groups.n_raw_violations.sum()==17
    assert (groups.n_raw_violations+groups.n_physical_predictions == groups.n).all()
    assert (groups.n_pairs==0).sum()==1  # observed ties cannot validate ranking
    assert (groups.n_raw_violations.eq(0) & groups.n_pairs.gt(0)).sum()==6


def test_doi_is_unit_of_uncertainty_not_group_count():
    summ=_out()['stage2_ranking']
    assert summ['found']['independent_DOIs']==8
    assert summ['all_evaluable_groups']['n_dois']==7
    assert summ['doi_uncertainty']['doi_count']==7
    assert summ['all_evaluable_groups']['macro_pairwise_tie_half'] < 1


def test_stage1_doi_robustness_matches_existing_oof():
    s,d,a=stage1_strata_audit()
    assert a['BET']['doi_improved']==19 and a['BET']['doi_worsened']==13
    assert a['TPV']['doi_improved']==14 and a['TPV']['doi_worsened']==15
    assert len(d[d.target=='BET'])==32 and len(d[d.target=='TPV'])==29
    assert len(s)==8


def test_bootstrap_is_fixed_seed_reproducible():
    g=pd.read_csv(REPORT/'stage2_strict_comparison_groups.csv')
    out=doi_uncertainty(g[g.n_pairs>0],n_boot=300)
    out2=doi_uncertainty(g[g.n_pairs>0],n_boot=300)
    assert out==out2
    assert out['doi_count']==7
    assert 0 <= out['signflip_two_sided_p_exploratory'] <= 1


def test_fixed_data_contract_and_no_external_contamination():
    summ=_out()['stage2_ranking']
    assert '98 rows / 26 DOI' in summ['contract']
    assert 'v24 69 rows excluded' in summ['contract']
    assert summ['found']['model_B_raw_violations_in_group_rows']==17
    assert 'stage1_robustness' in _out()


def test_stage2_screening_ui_connects_conservative_gate_before_scoring():
    # Streamlit is not available in this build environment; static syntax is validated separately.
    # The guard is also unit tested by v26 tests.
    p=ROOT/'pyrolysis/biochar_ui/stage2_pages.py'
    text=p.read_text(encoding='utf-8')
    begin=text.index('def page_screening():')
    end=text.index('def page_compare():')
    screen=text[begin:end]
    assert 'scope = assess_scope(core, matched.matched, ads["pollutant"], ads)' in screen
    assert screen.index('scope = assess_scope(') < screen.index('score_observed_candidates(')
    assert 'if not scope.eligible_for_exploratory_ranking:' in screen
    assert 'return' in screen[screen.index('if not scope.eligible_for_exploratory_ranking:'):screen.index('score_observed_candidates(')]
