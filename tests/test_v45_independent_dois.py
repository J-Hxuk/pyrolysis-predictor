"""v45 guarded ingestion tests; no active deployment changes."""
from __future__ import annotations
import tempfile
from pathlib import Path
import pandas as pd
from scripts.build_v45_independent_dois import NEW_DOI,material_rows,uptake_rows,hold_out,sakhiya_sensitivity,run


def test_four_distinct_dois():
    assert len(set(NEW_DOI.values()))==4

def test_new_materials_23_unique():
    df=material_rows()
    assert len(df)==23 and df.material_id.is_unique
    assert df.doi.nunique()==4

def test_new_numeric_BET_12_and_TPV_5():
    f=material_rows()
    assert f.BET_m2_g_training_candidate.notna().sum()==12
    assert f.TPV_cm3_g_training_candidate.notna().sum()==5

def test_2025_figure_bet_not_fake_exact_numbers():
    f=material_rows()
    bean=f[f.doi.eq(NEW_DOI['beans'])]
    assert len(bean)==9 and bean.BET_m2_g_training_candidate.isna().all()

def test_2021_kinetic_fit_separated_from_qe():
    d=hold_out(); f=d[d.doi.eq(NEW_DOI['kitchen'])]
    assert len(f)==4
    assert set(f.decision)=={'SEPARATE_DO_NOT_TRAIN_EXPERIMENTAL_QE'}
    assert not uptake_rows().doi.eq(NEW_DOI['kitchen']).any()

def test_digitized_research_qe_and_provenance():
    f=uptake_rows()
    assert len(f)==10 and f.doi.nunique()==2
    assert f.target_name.eq('qe_reported_mg_g').all()
    assert f.source_target_status.eq('RESEARCH_ONLY_DIGITIZED_APPROX').all()
    assert f.initial_pH.isna().all()

def test_figure_qe_caps_and_methods():
    f=uptake_rows()
    assert (f.target_value>=0).all()
    assert (f.target_value<=f.C0_mg_L/f.dose_g_L).all()
    assert set(f.C0_mg_L)=={47.44,1000}
    assert set(f.contact_time_min)=={240,1440}

def test_sakhiya_volumes_explicitly_inferred():
    f=sakhiya_sensitivity()
    assert len(f)==6
    assert f.Volume_reported_mL.eq(5).all()
    assert f.Volume_assumed_corrected_mL.eq(50).all()
    assert f.Correction_status.eq('inferred_typo_not_author_verified').all()
    assert f.Main_training.eq('EXCLUDED').all()
    assert (f.qe_reported_mg_g>f.physical_cap_reported_mg_g).all()

def test_2025_isotherm_graph_not_silently_trained():
    f=hold_out();p=f[f.doi.eq(NEW_DOI['beans'])]
    assert len(p)==18
    assert p.value_mg_g.isna().all()

def test_repeated_build_is_identical(tmp_path):
    """Create only the necessary source inputs for a clean reproducibility smoke test."""
    here=Path(__file__).resolve().parents[1]
    for name in ('Stage1_Master_381_RESEARCH_STAGING.csv','Stage2_reported_uptake_128_RESEARCH_CONDITIONAL.csv'):
        x=(here/'reports/v40_new9'/name)
        assert x.exists(), 'Install v40 first: '+str(x)
        target=tmp_path/'reports/v40_new9'/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(x.read_bytes())
    a=run(tmp_path);b=run(tmp_path)
    assert a==b
    assert a['materials_total']==404 and a['stage2_total']==138
    assert a['BET_total']==348 and a['TPV_total']==218
