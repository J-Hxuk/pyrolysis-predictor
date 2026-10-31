"""v41 archive not shipped: extracted source data should still be usable."""

import zipfile

import pytest

from pyrolysis.biochar_stage1 import extract as ex


def _make_all_extracted(root):
    for rel in ex.REQUIRED_FILES:
        p = root / ex.IMPORT_SUBDIR / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("source-data-placeholder", encoding="utf-8")


def test_complete_extracted_import_works_without_backup_zip(tmp_path):
    _make_all_extracted(tmp_path)
    outputs = ex.extract_required_files(tmp_path)
    assert len(outputs) == len(ex.REQUIRED_FILES)
    assert all((tmp_path / ex.IMPORT_SUBDIR / key).is_file() for key in outputs)


def test_incomplete_import_without_zip_fails_instead_of_inventing_data(tmp_path):
    _make_all_extracted(tmp_path)
    (tmp_path / ex.IMPORT_SUBDIR / ex.REQUIRED_FILES[0]).unlink()
    with pytest.raises(ex.ExtractError, match="누락 1개"):
        ex.extract_required_files(tmp_path)


def test_partial_import_recovers_only_missing_file_from_zip(tmp_path):
    _make_all_extracted(tmp_path)
    missing = ex.REQUIRED_FILES[0]
    (tmp_path / ex.IMPORT_SUBDIR / missing).unlink()
    sentinel = tmp_path / ex.IMPORT_SUBDIR / ex.REQUIRED_FILES[1]
    with zipfile.ZipFile(tmp_path / ex.ZIP_NAME, "w") as z:
        z.writestr(f"{ex.ZIP_ROOT}/{missing}", "recovered")
    result = ex.extract_required_files(tmp_path)
    assert len(result) == len(ex.REQUIRED_FILES)
    assert (tmp_path / ex.IMPORT_SUBDIR / missing).read_text() == "recovered"
    assert sentinel.read_text() == "source-data-placeholder"
