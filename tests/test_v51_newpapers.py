from pathlib import Path
import csv
from scripts.audit_v51_newpapers import audit, BASE, normalize_doi

ROOT=Path(__file__).resolve().parents[1]


def test_v51_reviewer_intake_is_not_auto_approved():
    report=audit(ROOT)
    assert report['candidate_observations']==8
    assert report['cu_reported_candidates']==4
    assert report['cd_approx_candidates']==4
    assert report['strict_qe_approved']==0
    assert report['stage2_training_rows_added']==0
    assert not report['training_rerun']
    assert not report['active_models_changed']
    assert not report['external_holdout_certified']


def test_v51_no_imputed_dose_or_contact_time():
    path=ROOT/BASE/'Additional_Observed_Adsorption_Candidates.csv'
    with path.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    assert all(x['dose_g_L']=='NR' for x in rows)
    assert all(x['contact_time_min']=='NR' for x in rows)
    assert all(x['strict_qe_status']=='REVIEW' for x in rows)


def test_v51_legacy_exposure_cannot_be_independent_holdout():
    r=audit(ROOT)
    assert '10.1016/j.biortech.2013.03.186' in r['historically_exposed_doi']


def test_v51_snapshot_doi_uniqueness_and_base_unchanged():
    r=audit(ROOT)
    for name,rec in r['baseline_snapshot'].items():
        assert rec['exists'],f'기존 v46 통합 CSV 누락: {name}'
        assert not set(r['review_dois']) & set(rec['new_paper_dois_already_in_snapshot'])
    assert r['baseline_snapshot']['Stage1_417_EXPLORATORY_UNAPPROVED.csv']['rows']==417
    assert r['baseline_snapshot']['Stage2_140_SOURCE_VERIFIED_AUGMENTED_RESEARCH.csv']['rows']==140
