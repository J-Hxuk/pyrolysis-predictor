"""v47 integrated installation checks; never retrains or changes active models.
Usage: python -m scripts.verify_v47 [--full-checksums]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import hashlib

ROOT=Path(__file__).resolve().parents[1]


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as fh:
        for chunk in iter(lambda:fh.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-checksums',action='store_true',help='verify all v47 packaging file hashes')
    args=parser.parse_args()
    problems=[]
    try:
        from pyrolysis.biochar_stage1.extract import REQUIRED_FILES,IMPORT_SUBDIR
        from pyrolysis.biochar_stage2.v15_extract import _extracted_state, IMPORT_DIR,PACKAGE_ROOT
    except Exception as exc:
        print('Failed to import data loaders:',repr(exc));return 2
    s1missing=[x for x in REQUIRED_FILES if not (ROOT/IMPORT_SUBDIR/x).is_file()]
    if s1missing: problems.append('Stage1 v41 extracted files missing: '+str(s1missing))
    invalid, manifest_valid=_extracted_state(ROOT/IMPORT_DIR/PACKAGE_ROOT)
    if invalid or not manifest_valid: problems.append('Stage2 v15 extracted SHA256 failed: '+str(invalid))
    critical=[
        'biochar_stage2_app.py','biochar_app.py',
        'pyrolysis/biochar_ui/stage2_pages.py',
        'pyrolysis/biochar_ui/stage1_bet_v20_page.py',
        'pyrolysis/biochar_stage2/v43_scenarios.py',
        'pyrolysis/biochar_stage1/v44_workflow.py',
        'data/imports/v46_phase8/Stage2_140_SOURCE_VERIFIED_AUGMENTED_RESEARCH.csv',
        'data/imports/v46_phase8/Stage1_417_EXPLORATORY_UNAPPROVED.csv',
        'deployment/stage1_models/active.json','deployment/stage2_models/active.json',
    ]
    for rel in critical:
        if not (ROOT/rel).is_file(): problems.append('Required file missing: '+rel)
    if args.full_checksums:
        mf=ROOT/'V47_MANIFEST.json'
        if not mf.is_file():problems.append('V47_MANIFEST.json missing')
        else:
            entries=json.loads(mf.read_text(encoding='utf-8'))['file_hashes']
            for rel,expected in entries.items():
                path=ROOT/rel
                if not path.is_file() or sha256(path)!=expected:
                    problems.append('SHA256 mismatch: '+rel)
    print('v47 integrated baseline verification')
    print('Stage1 extracted inputs:',len(REQUIRED_FILES)-len(s1missing),'/',len(REQUIRED_FILES))
    print('Stage2 v15 source SHA256:', 'PASS' if not invalid and manifest_valid else 'FAIL')
    print('Active models: unchanged by v47 consolidation')
    print('v46 expanded research data: 417 Stage1 / 140 Stage2 records (not active)')
    if problems:
        for p in problems: print('ERROR:',p)
        return 1
    print('PASS: on-disk data & code readiness (not browser rendering or predictive accuracy)')
    return 0

if __name__=='__main__': sys.exit(main())
