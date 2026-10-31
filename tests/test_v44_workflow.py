"""v44 workflow smoke tests; no Streamlit or archived training ZIP required."""
import math
import pytest
from pyrolysis.biochar_stage1.v44_workflow import (
    bet_temperature_sweep, bet_result_csv, build_bet_record, handoff_to_stage2,
)


def model():
    return {'models': {'proximate': {
        'features': ['temperature_C', 'time_min', 'heating_rate_C_min',
                     'VM_pct', 'Ash_pct', 'FC_pct'],
        'feature_range': {'temperature_C': [300, 700]},
        'trees': [{'left': [-1], 'right': [-1], 'feature': [-2],
                   'threshold': [-2], 'value': [35.0]}],
    }}}


def check(*, valid=True):
    return {'variant': 'proximate', 'predictable': valid,
            'can_explore_composition_ood': not valid,
            'values': {'temperature_C': 500.0, 'time_min': 60.0,
                       'heating_rate_C_min': 10.0, 'VM_pct': 70.0,
                       'Ash_pct': 10.0, 'FC_pct': 20.0}}


def process():
    return {'pyrolysis_temperature_C': 500.0,
            'pyrolysis_hold_min': 60.0, 'heating_rate_C_min': 10.0}


def test_stage1_curve_scope_and_no_fabricated_targets():
    f = bet_temperature_sweep(model(), check(), n_points=7)
    assert len(f) == 7
    assert f['열분해온도 (°C)'].min() == 300
    assert f['열분해온도 (°C)'].max() == 700
    assert (f['예측 BET (m²/g)'] == 35).all()
    assert not {'qe', 'TPV'}.intersection(' '.join(f.columns).split())


def test_explicit_process_handoff_never_copies_bet():
    r = build_bet_record('rice husk', process(), check(), 35.0)
    dst = handoff_to_stage2(r)
    assert dst['s2_bio'] == 'rice husk'
    assert dst['s2_enable_B']
    assert dst['s2_pt'] == 500 and dst['s2_hold'] == 60 and dst['s2_hr'] == 10
    assert not any('bet' in k.lower() for k in dst)


def test_unknown_feedstock_is_not_replaced():
    r = build_bet_record('unseen biomass', process(), check(), 35)
    assert handoff_to_stage2(r)['s2_bio'] == 'unseen biomass'


def test_invalid_inputs_cannot_handoff_or_export():
    with pytest.raises(ValueError):
        build_bet_record('', process(), check(), 35)
    with pytest.raises(ValueError):
        build_bet_record('rice husk', process(), check(), math.nan)
    with pytest.raises(ValueError):
        build_bet_record('rice husk', {**process(), 'pyrolysis_hold_min': 0}, check(), 35)
    with pytest.raises(ValueError):
        handoff_to_stage2({**build_bet_record('x', process(), check(), 2), 'activated': True})


def test_ood_requires_explicit_consent_and_labels_provisional():
    with pytest.raises(ValueError):
        bet_temperature_sweep(model(), check(valid=False))
    with pytest.raises(ValueError):
        build_bet_record('x', process(), check(valid=False), 35)
    f = bet_temperature_sweep(model(), check(valid=False), allow_composition_ood=True)
    r = build_bet_record('x', process(), check(valid=False), 35, allow_composition_ood=True)
    assert set(f['input_scope']) == {'COMPOSITION_OOD_PROVISIONAL'}
    assert r['input_scope'] == 'COMPOSITION_OOD_PROVISIONAL'


def test_csv_is_excel_friendly_and_records_scope():
    r = build_bet_record('rice husk', process(), check(), 35)
    b = bet_result_csv(r, bet_temperature_sweep(model(), check()))
    assert b.startswith(b'\xef\xbb\xbf')
    s = b.decode('utf-8-sig')
    assert 'rice husk' in s and 'MARGINAL_RANGE_EXPLORATORY' in s
    assert 'predicted_BET_m2_g' in s


def test_temperature_grid_invalid_size_fails():
    with pytest.raises(ValueError):
        bet_temperature_sweep(model(), check(), n_points=1000)
