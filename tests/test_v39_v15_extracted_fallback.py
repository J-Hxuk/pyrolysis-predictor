"""Legacy Stage2 v15 archive relocation regression tests."""
from __future__ import annotations

import csv
import hashlib
import os
import shutil
import zipfile
from pathlib import Path

import pytest

from pyrolysis.biochar_stage2 import v15_extract as ext


def _sample(base: Path) -> tuple[Path, bytes]:
    """Make a miniature, hash-verified v15 package without a 60 MB backup."""
    pkg = base / ext.IMPORT_DIR / ext.PACKAGE_ROOT
    pkg.mkdir(parents=True)
    rows = []
    for name in ext.REQUIRED_FILES:
        if name == "14_SHA256_Manifest.csv":
            continue
        data = f"ORIGINAL_TEST_DATA_{name}".encode()
        (pkg / name).write_bytes(data)
        rows.append((name, len(data), hashlib.sha256(data).hexdigest()))
    with (pkg / "14_SHA256_Manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "size_bytes", "SHA256"])
        writer.writerows(rows)
    return pkg, (pkg / "14_SHA256_Manifest.csv").read_bytes()


def test_extract_returns_existing_verified_data_without_zip(tmp_path):
    pkg, _ = _sample(tmp_path)
    assert not (tmp_path / "data/imports" / ext.ZIP_NAME).exists()
    assert ext.extract_v15_files(tmp_path) == pkg
    assert ext.resolve_v15_dir(tmp_path) == pkg


def test_missing_extracted_file_and_zip_does_not_make_up_data(tmp_path):
    pkg, _ = _sample(tmp_path)
    (pkg / "03_Stage2_QE_Core_97.csv").unlink()
    with pytest.raises(ext.V15ExtractError, match="03_Stage2_QE_Core_97.csv"):
        ext.extract_v15_files(tmp_path)


def test_sha_mismatch_is_not_silently_accepted(tmp_path):
    pkg, _ = _sample(tmp_path)
    (pkg / "03_Stage2_QE_Core_97.csv").write_text("TAMPERED", encoding="utf-8")
    with pytest.raises(ext.V15ExtractError, match="03_Stage2_QE_Core_97.csv"):
        ext.resolve_v15_dir(tmp_path)


def test_partial_extracted_files_repaired_using_moved_zip(tmp_path, monkeypatch):
    pkg, _ = _sample(tmp_path)
    moved_zip = tmp_path / "elsewhere" / ext.ZIP_NAME
    moved_zip.parent.mkdir(parents=True)
    original = (pkg / "03_Stage2_QE_Core_97.csv").read_bytes()
    with zipfile.ZipFile(moved_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in ext.REQUIRED_FILES:
            zf.write(pkg / name, arcname=f"{ext.PACKAGE_ROOT}/{name}")
    (pkg / "03_Stage2_QE_Core_97.csv").unlink()
    monkeypatch.setenv("BIOCHAR_V15_ZIP_PATH", str(moved_zip))
    assert ext.extract_v15_files(tmp_path) == pkg
    assert (pkg / "03_Stage2_QE_Core_97.csv").read_bytes() == original
    assert ext.resolve_v15_dir(tmp_path) == pkg


def test_overwrite_requires_original_zip(tmp_path):
    _sample(tmp_path)
    with pytest.raises(ext.V15ExtractError, match="ZIP"):
        ext.extract_v15_files(tmp_path, overwrite=True)


def test_stored_real_core97_unchanged_after_relocation(tmp_path):
    """Use the real, provided v15 fixture if available in the working checkout."""
    source = Path(__file__).resolve().parents[1] / ext.IMPORT_DIR / ext.PACKAGE_ROOT
    if not (source / "14_SHA256_Manifest.csv").exists():
        pytest.skip("Original v15 extracted fixture not mounted")
    dest = tmp_path / ext.IMPORT_DIR / ext.PACKAGE_ROOT
    shutil.copytree(source, dest)
    assert ext.extract_v15_files(tmp_path) == dest
    import pandas as pd
    frame = pd.read_csv(dest / "03_Stage2_QE_Core_97.csv")
    assert len(frame) == 97
    assert frame["doi"].nunique() == 25
