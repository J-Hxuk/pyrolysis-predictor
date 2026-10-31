"""v47 baseline: installation integrity, cross-version sources, and immutable legacy data."""
from pathlib import Path
from scripts.verify_v47 import main, ROOT


def test_v47_integrated_files_present():
    for relative in [
        'biochar_stage2_app.py',
        'pyrolysis/biochar_ui/stage2_pages.py',
        'pyrolysis/biochar_ui/stage1_bet_v20_page.py',
        'pyrolysis/biochar_stage2/v43_scenarios.py',
        'pyrolysis/biochar_stage1/v44_workflow.py',
        'data/imports/v46_phase8/Stage2_140_SOURCE_VERIFIED_AUGMENTED_RESEARCH.csv',
        'deployment/stage2_models/active.json',
    ]:
        assert (ROOT / relative).is_file(),relative


def test_v47_extracted_legacy_archives_no_zip_required():
    from pyrolysis.biochar_stage1.extract import REQUIRED_FILES, IMPORT_SUBDIR
    from pyrolysis.biochar_stage2.v15_extract import _extracted_state, IMPORT_DIR, PACKAGE_ROOT
    assert all((ROOT/IMPORT_SUBDIR/x).is_file() for x in REQUIRED_FILES)
    assert _extracted_state(ROOT/IMPORT_DIR/PACKAGE_ROOT)==([],True)


def test_v47_both_apps_remain_entrypoints():
    assert 'render_stage2_app' in (ROOT/'biochar_stage2_app.py').read_text(encoding='utf-8')
    assert 'render_app' in (ROOT/'biochar_app.py').read_text(encoding='utf-8')


def test_v47_research_and_deployed_models_separate():
    import json
    stage2=json.loads((ROOT/'deployment/stage2_models/active.json').read_text(encoding='utf-8'))
    assert stage2
    research=list((ROOT/'deployment/research_v46_NOT_ACTIVE').glob('*.joblib'))
    assert len(research)==4
    assert all(x.resolve().parent!=(ROOT/'deployment/stage2_models').resolve() for x in research)
