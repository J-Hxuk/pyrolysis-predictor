"""Build strictly separate Phase8 read-only research cohorts and gap audit.

Run after installation from project root:
  python -m scripts.build_phase8_v22_reports

Never overwrites v15/v16 model files or baseline data. No fit() is called.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
from pyrolysis.biochar_stage2.phase8_evidence import audit_phase8, build_stage2_core98_candidate


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError('cannot write empty source')
    with path.open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]),extrasaction='raise')
        writer.writeheader(); writer.writerows(rows)


def build(base: Path) -> dict:
    base=Path(base)
    report=audit_phase8(base)
    candidate=build_stage2_core98_candidate(base)
    path=base/'reports/phase8_v22'
    path.mkdir(parents=True,exist_ok=True)
    _write_csv(path/'stage2_98_RESEARCH_CANDIDATE_NOT_TRAINED.csv', candidate)
    _write_csv(path/'stage1_new8_observations.csv',report['stage1_new'])
    _write_csv(path/'stage2_new1_observation.csv',report['stage2_new'])
    coverage=[]
    for target,block in report['composition_coverage'].items():
        for name,data in block['fields'].items():
            coverage.append({'target':target,'feature':name,'observed_rows':data['n'],
                             'total_rows':block['total'],'observed_pct':round(100*data['n']/block['total'],1),
                             'independent_DOI':data['doi']})
    _write_csv(path/'stage1_raw_composition_coverage.csv',coverage)
    (path/'audit_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report

if __name__ == '__main__':
    base=Path(__file__).resolve().parents[1]
    result=build(base)
    print('Phase8 research snapshot ready: ',result['stage1']['integrated'],result['stage2']['integrated'])
    print('Active models unchanged; no model retraining.')
