"""v28: read-only identifiability diagnosis and sidebar navigation regression."""
from pathlib import Path
import ast
import json

import pandas as pd
import pytest

from scripts.evaluate_v28_feature_identifiability import (
    ROOT, REPORT, V27, audit_identifiability,
)
from scripts.compare_phase8_v23 import stage2_targets


def test_frozen_evaluation_matches_existing_v27_predictions():
    _,core=stage2_targets()
    oof=pd.read_csv(V27/'stage2_heldout_doi_predictions.csv')
    a,p,s=audit_identifiability(core,oof)
    assert s['comparison_groups']==12
    assert s['evaluable_groups']==11 and s['evaluable_DOIs']==7
    assert a.observed_pairs.sum()==30
    assert len(p)==30
    assert s['model_B_pairwise_macro_from_v27']==pytest.approx(.87121212)


def test_no_temperature_contrast_and_collapsed_recipes():
    s=json.loads((REPORT/'summary.json').read_text(encoding='utf-8'))
    assert s['groups_with_verified_temperature_contrast']==0
    assert s['groups_with_identical_temperature_in_all_reported_candidates']==10
    assert s['groups_with_all_temperature_unreported']==2
    assert s['evaluable_groups_with_collapsed_material_inputs']==4
    assert s['pairs_with_identical_manufacturing_input_signature']==6
    assert s['pairs_with_predicted_tie']==10


def test_collapsed_pair_is_a_bona_fide_label_difference():
    p=pd.read_csv(REPORT/'pairwise_input_collisions.csv')
    coll=p[p.same_model_manufacturing_inputs]
    assert len(coll)==6
    assert (coll.observed_gap_mg_g>0).all()
    # Feature-identical predictions are tied in deterministic Ridge pipeline.
    assert coll.predicted_tie.all()


def test_no_false_temperature_credit_from_unreported_values():
    df=pd.read_csv(REPORT/'groups_feature_identifiability.csv')
    assert df.temperature_contrast_testable.sum()==0
    # Unknown temperature is not counted as an observed temperature difference.
    assert (df.pyrolysis_temperature_values_reported.isin([0,1])).all()


def test_sidebar_four_menu_contract_and_old_functions_preserved():
    path=ROOT/'pyrolysis/biochar_ui/stage2_pages.py'
    txt=path.read_text(encoding='utf-8')
    ast.parse(txt)
    pages=txt[txt.index('PAGES = {'):txt.index('def render_stage2_app():')]
    assert all(k in pages for k in ['흡착성능 예측','제조조건 탐색','BET 물성 예측','데이터·모델 검증'])
    assert pages.count(': page_')+pages.count(': render_') == 4
    assert 'def page_compare():' in txt
    assert 'def page_validation():' in txt
    assert 'def page_screening():' in txt
    assert '"문헌 후보 비교", "연구용 온도 민감도"' in txt
    assert '"모델 개요", "예측 검증", "Phase 8 원자료"' in txt


def test_v27_gate_preserved_in_restructured_app():
    txt=(ROOT/'pyrolysis/biochar_ui/stage2_pages.py').read_text(encoding='utf8')
    screen=txt[txt.index('def page_screening():'):txt.index('def page_compare():')]
    assert screen.index('scope = assess_scope(') < screen.index('score_observed_candidates(')
    assert 'if not scope.eligible_for_exploratory_ranking:' in screen
