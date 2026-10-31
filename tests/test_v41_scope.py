"""No Streamlit / model-loading required: pure applicability contract tests."""
import math
import numpy as np
import pandas as pd
import pytest
from pyrolysis.biochar_stage2.v41_scope import inspect_active_scope, scope_to_table


def sample():
    a = []
    # two independent rice-husk DOI and one unique corn-straw source
    for doi, name, temp, ph in [
        ('10.1/alpha', 'rice husk', 400, 5),
        ('10.1/beta', 'rice husk', 600, np.nan),
        ('10.1/gamma', 'maize straw', 500, 6),
    ]:
        a.append(dict(doi=doi, pollutant='Cd(II)', C0_mg_L=100.,
                      initial_pH=ph, dose_g_L=2.0, adsorption_T_C=25.,
                      contact_time_min=120., feedstock_label_normalized=name,
                      manufacturing_route='SINGLE_STAGE_PYROLYSIS',
                      process_step_count=1, pyrolysis_temperature_C=temp,
                      pyrolysis_hold_min=60., heating_rate_C_min=np.nan,
                      pyrolysis_atmosphere=np.nan,
                      activation_or_modification=np.nan,
                      secondary_temperature_C=np.nan,
                      secondary_hold_min=np.nan))
    return pd.DataFrame(a)


def row(**kw):
    d=sample().iloc[0].to_dict()
    d.update(kw)
    return d


def test_active_a_ignores_new_feedstock_names():
    s = inspect_active_scope(sample(), row(feedstock_label_normalized='never_seen'), 'A')
    assert s.n_cohort_rows == 3
    assert not any('feedstock' in x for x in s.warnings)
    assert s.status == 'RANGE_ONLY'


def test_unknown_feedstock_b_fails_range_and_cannot_rank():
    s = inspect_active_scope(sample(), row(feedstock_label_normalized='never_seen'), 'B')
    assert s.status == 'OUTSIDE_RANGE'
    assert not s.can_rank
    assert s.n_cohort_rows == 0


def test_single_doi_pair_cannot_rank_even_when_global_ranges_pass():
    s = inspect_active_scope(sample(), row(feedstock_label_normalized='maize straw', pyrolysis_temperature_C=500), 'B')
    assert s.status == 'LIMITED_EVIDENCE'
    assert s.n_cohort_doi == 1
    assert not s.can_rank


def test_two_doi_without_controlled_temperature_contrast_still_cannot_rank():
    s = inspect_active_scope(sample(), row(), 'B')
    assert not s.can_rank
    assert s.status == 'LIMITED_EVIDENCE'
    assert s.n_controlled_temperature_panels == 0


def test_controlled_temperature_panel_is_only_necessary_not_sufficient():
    df = sample()
    additional = df.iloc[[0]].copy()
    additional['pyrolysis_temperature_C'] = 500
    df = pd.concat([df, additional],ignore_index=True)
    s = inspect_active_scope(df, row(), 'B')
    assert s.can_rank  # only an *eligibility precheck*, never validated optimality
    assert s.status == 'RANGE_ONLY'
    assert s.n_controlled_temperature_panels == 1
    assert '보장하지 않습니다' in s.note


def test_invalid_physical_dose_blocks_and_no_imputation():
    a = inspect_active_scope(sample(), row(dose_g_L=0), 'A')
    assert a.status == 'INVALID'
    assert a.blockers
    b = inspect_active_scope(sample(), row(pyrolysis_temperature_C=np.nan), 'B')
    assert not b.can_rank
    assert any('미보고' in x for x in b.warnings)


def test_oob_numerical_and_unknown_pollutant():
    assert inspect_active_scope(sample(), row(C0_mg_L=200), 'A').status == 'OUTSIDE_RANGE'
    assert inspect_active_scope(sample(), row(pollutant='Hg(II)'), 'A').status == 'OUTSIDE_RANGE'


def test_missingness_report_and_no_qe_data_leakage():
    s = inspect_active_scope(sample(), row(), 'B')
    d = scope_to_table(s)
    assert 'target_value' not in d['모델 입력'].values
    assert d.loc[d['모델 입력']=='heating_rate_C_min','보고율 (%)'].iloc[0] == 0.0
    assert s.n_reference_rows == 3


def test_nonfinite_numeric_rejected():
    assert inspect_active_scope(sample(), row(C0_mg_L=float('inf')), 'A').status == 'INVALID'
    assert inspect_active_scope(sample(), row(initial_pH='wrong'), 'A').status == 'INVALID'


def test_fail_closed_without_required_data_fields():
    with pytest.raises(ValueError,match='필수 열'):
        inspect_active_scope(sample().drop(columns=['doi']), row(), 'B')
    with pytest.raises(ValueError):
        inspect_active_scope(sample(), row(), 'C')
