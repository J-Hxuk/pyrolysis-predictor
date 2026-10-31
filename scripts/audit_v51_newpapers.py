"""Read-only validation of v51 new literature review intake.

No model fitting and no modification of v46/v50 model datasets.
Run: python -m scripts.audit_v51_newpapers --root .
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

DOIS={
    "10.3390/toxics10120787",
    "10.1016/j.biortech.2013.03.186",
    "10.2166/wst.2020.309",
}
BASE=Path('data/imports/v51_newpaper_review')
STAGE1=Path('data/imports/v46_phase8/Stage1_417_EXPLORATORY_UNAPPROVED.csv')
STAGE2=Path('data/imports/v46_phase8/Stage2_140_SOURCE_VERIFIED_AUGMENTED_RESEARCH.csv')


def read_csv(path:Path):
    with path.open('r',encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def normalize_doi(value):
    return str(value or '').strip().lower().replace('https://doi.org/','').replace('http://doi.org/','')


def audit(root:Path) -> dict:
    root=root.resolve()
    base=root/BASE
    observed=read_csv(base/'Additional_Observed_Adsorption_Candidates.csv')
    series=read_csv(base/'Additional_Comparison_Series_Readiness.csv')
    if len(observed)!=8 or len(series)!=4:
        raise ValueError('예상 데이터 수 불일치: 8개 후보 관측, 4개 비교 시리즈 필요')
    if set(map(lambda row:normalize_doi(row['doi']),series))!=DOIS:
        raise ValueError('비교 시리즈 DOI 목록이 원문 검토자료와 일치하지 않습니다')
    if set(map(lambda row:normalize_doi(row['doi']),observed))!={"10.3390/toxics10120787"}:
        raise ValueError('대나무 후보 관측 DOI가 잘못됐습니다')
    if any((row.get('strict_qe_status') or '').strip().upper()!='REVIEW' for row in observed):
        raise ValueError('REVIEW 데이터가 승인 대상으로 잘못 표시되었습니다')
    if any(str(row.get('dose_g_L','')).strip() not in ('','NR') for row in observed):
        raise ValueError('수치가 없는 대나무 dose를 임의로 채웠습니다')
    if any(str(row.get('contact_time_min','')).strip() not in ('','NR') for row in observed):
        raise ValueError('수치가 없는 대나무 contact time을 임의로 채웠습니다')
    if any(str(row.get('ranking_series_eligible','')).strip()=='YES' for row in observed):
        raise ValueError('승인되지 않은 ranking series가 있습니다')
    id_pairs=[(normalize_doi(x['doi']),x['original_sample_id'],x['pollutant']) for x in observed]
    if len(id_pairs)!=len(set(id_pairs)):
        raise ValueError('후보 데이터 내부 관측키 중복')
    baseline_checks={}
    for filename in (STAGE1,STAGE2):
        path=root/filename
        baseline_checks[filename.name]={'exists':path.is_file()}
        if path.is_file():
            blob=path.read_bytes()
            historical=read_csv(path)
            seen=set(normalize_doi(x.get('doi')) for x in historical)
            baseline_checks[filename.name].update({
                'rows':len(historical),
                'sha256':hashlib.sha256(blob).hexdigest(),
                'doi_count':len(seen-{''}),
                'new_paper_dois_already_in_snapshot':sorted(seen&DOIS),
            })
    material=baseline_checks.get(STAGE1.name,{})
    aqueous=baseline_checks.get(STAGE2.name,{})
    # No inference about historical holdout from current-snapshot absence:
    # Miscanthus occurs in legacy stage-1 active-model history.
    report={
        'review_dois':sorted(DOIS),
        'candidate_observations':len(observed),
        'cu_reported_candidates':sum(x['pollutant']=='Cu(II)' for x in observed),
        'cd_approx_candidates':sum(x['pollutant']=='Cd(II)' for x in observed),
        'series_rows':len(series),
        'strict_qe_approved':0,
        'new_stage1_approved':0,
        'stage2_training_rows_added':0,
        'training_rerun':False,
        'active_models_changed':False,
        'external_holdout_certified':False,
        'historically_exposed_doi':['10.1016/j.biortech.2013.03.186'],
        'baseline_snapshot':baseline_checks,
        'missing_bamboo_fields':['dose_g_L','contact_time_min','equilibrium_evidence','Cd_figure_to_text_mapping'],
        'source_files':{
            p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(base.iterdir()) if p.is_file()
        },
    }
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('.'))
    parser.add_argument('--out',type=Path,default=None)
    args=parser.parse_args()
    report=audit(args.root)
    if args.out:
        target=args.out if args.out.is_absolute() else args.root/args.out
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
