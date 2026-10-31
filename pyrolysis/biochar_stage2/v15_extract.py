"""Stage2 v15 archive/extracted-data resolver (legacy 97-row reproducibility).

The immutable v15 package was historically distributed as a large archive.  When
its explicitly extracted, SHA256-verified input files remain available, no ZIP
is necessary.  This module DOES NOT recreate missing research data or mix v15
with later versions' qe, qt, removal, or qmax datasets.
"""
from __future__ import annotations

import csv
import hashlib
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

__all__ = [
    "ZIP_NAME", "PACKAGE_ROOT", "IMPORT_DIR", "REQUIRED_FILES", "BACKUP_EXCLUDED",
    "extract_v15_files", "resolve_v15_dir", "V15ExtractError",
]

ZIP_NAME = "Stage1_Stage2_Model_Ready_v15_20261029.zip"
PACKAGE_ROOT = "Stage1_Stage2_Model_Ready_v15_20261029"
ZIP_SUBDIR = Path("data/imports")
IMPORT_DIR = Path("data/imports/Stage1_Stage2_Model_Ready_v15")
# Original v15 source files, without the 60 MB baseline backup inside the archive.
REQUIRED_FILES = [
    "00_README_START_HERE.md",
    "01_Manufacturing_And_Feedstock_Materials_441.csv",
    "02_Adsorption_Target_Ledger_151.csv",
    "03_Stage2_QE_Core_97.csv",
    "04_QEXP_Kinetic_Optional_6.csv",
    "05_QT_120min_Separate_32.csv",
    "06_Removal_Separate_16.csv",
    "07_QE_Derived_From_Removal_16_DO_NOT_DOUBLE_COUNT.csv",
    "08_Langmuir_Qmax_Fitted_Separate_32.csv",
    "09_Manufacturing_Recommendation_Benchmark_Chen16.csv",
    "10_Primary_Source_DOI_Registry_61.csv",
    "11_Feature_Contract_A_B_C_D.csv",
    "12_QA_Model_Gates.csv",
    "13_Evaluation_And_Handoff_Protocol.md",
    "14_SHA256_Manifest.csv",
    "Stage1_Stage2_Model_Ready_v15.sqlite",
]
BACKUP_EXCLUDED = ["BASELINE_v14_DO_NOT_EDIT.zip"]
_MANIFEST = "14_SHA256_Manifest.csv"


class V15ExtractError(Exception):
    """Cannot safely load or restore original v15 files."""


def _zip_candidates(base: Path) -> tuple[Path, ...]:
    paths = [base / ZIP_SUBDIR / ZIP_NAME, base / ZIP_NAME]
    explicit = os.environ.get("BIOCHAR_V15_ZIP_PATH", "").strip()
    if explicit:
        paths.insert(0, Path(explicit).expanduser())
    return tuple(paths)


def _zip_path(base: Path) -> Path:
    """Find known original ZIP location, or an explicitly supplied moved path."""
    candidates = _zip_candidates(base)
    for path in candidates:
        if path.is_file():
            return path
    raise V15ExtractError(
        "v15 원본 ZIP을 찾을 수 없습니다. 확인 경로: "
        + "; ".join(str(x) for x in candidates)
        + ". ZIP을 이동했다면 BIOCHAR_V15_ZIP_PATH 환경변수에 원본 ZIP 경로를 지정할 수 있습니다."
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def _manifest_rows(data: bytes) -> dict[str, tuple[int, str]]:
    try:
        text = data.decode("utf-8-sig")
        rows = {}
        for row in csv.DictReader(text.splitlines()):
            name = row["filename"]
            rows[name] = (int(row["size_bytes"]), row["SHA256"].lower())
        for rel in REQUIRED_FILES:
            if rel != _MANIFEST and rel not in rows:
                raise ValueError(f"SHA256 목록에 필수 항목 누락: {rel}")
        return rows
    except (KeyError, TypeError, ValueError, UnicodeError) as exc:
        raise V15ExtractError(f"v15 SHA256 목록이 유효하지 않습니다: {exc}") from exc


def _invalid_files(pkg_dir: Path, rows: dict[str, tuple[int, str]]) -> list[str]:
    invalid = []
    for name in REQUIRED_FILES:
        path = pkg_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            invalid.append(name)
        elif name != _MANIFEST:
            size, sha = rows[name]
            if path.stat().st_size != size or _sha256(path) != sha:
                invalid.append(name)
    return invalid


def _extracted_state(pkg_dir: Path) -> tuple[list[str], bool]:
    """Return missing/modified entries and whether a valid manifest exists."""
    manifest = pkg_dir / _MANIFEST
    if not manifest.is_file() or manifest.stat().st_size == 0:
        return [x for x in REQUIRED_FILES if not (pkg_dir / x).is_file()] + [_MANIFEST], False
    try:
        rows = _manifest_rows(manifest.read_bytes())
    except V15ExtractError:
        return [_MANIFEST], False
    return _invalid_files(pkg_dir, rows), True


def _atomic_extract(zf: zipfile.ZipFile, member: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temp:
            tmp_path = Path(temp.name)
            with zf.open(member) as src:
                shutil.copyfileobj(src, temp)
        tmp_path.replace(destination)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def extract_v15_files(base: Path | None = None, *, overwrite: bool = False) -> Path:
    """Return intact original extracted package or restore it from the original ZIP.

    Without overwrite, a complete package verified against the original embedded
    SHA256 manifest is sufficient even if the large ZIP was moved/deleted.  If
    missing/modified, restore from ZIP when available; otherwise FAIL LOUDLY.
    """
    base = Path(base) if base is not None else Path.cwd()
    pkg_dir = base / IMPORT_DIR / PACKAGE_ROOT
    invalid, manifest_ok = _extracted_state(pkg_dir)
    if not overwrite and manifest_ok and not invalid:
        return pkg_dir

    try:
        zip_path = _zip_path(base)
    except V15ExtractError as exc:
        raise V15ExtractError(
            f"v15 원본 파일 복구 불가. 추출 위치: {pkg_dir}. "
            f"누락/변경 파일: {', '.join(dict.fromkeys(invalid))}. {exc} "
            "원본과 다른 최신 데이터로 대체하지 않습니다."
        ) from exc

    try:
        with zipfile.ZipFile(zip_path) as zf:
            names = set(zf.namelist())
            all_members = [f"{PACKAGE_ROOT}/{rel}" for rel in REQUIRED_FILES]
            missing_in_zip = [rel for rel, member in zip(REQUIRED_FILES, all_members) if member not in names]
            if missing_in_zip:
                raise V15ExtractError(f"v15 ZIP 내부 필수 파일 누락: {missing_in_zip}")
            # The authoritative manifest must be taken from the original archive.
            original_manifest = zf.read(f"{PACKAGE_ROOT}/{_MANIFEST}")
            rows = _manifest_rows(original_manifest)
            pkg_dir.mkdir(parents=True, exist_ok=True)
            if overwrite or not manifest_ok or (pkg_dir / _MANIFEST).read_bytes() != original_manifest:
                _atomic_extract(zf, f"{PACKAGE_ROOT}/{_MANIFEST}", pkg_dir / _MANIFEST)
            # Re-evaluate integrity against authoritative manifest, not a stale one.
            to_restore = [x for x in _invalid_files(pkg_dir, rows) if x != _MANIFEST]
            if overwrite:
                to_restore = [x for x in REQUIRED_FILES if x != _MANIFEST]
            for rel in to_restore:
                _atomic_extract(zf, f"{PACKAGE_ROOT}/{rel}", pkg_dir / rel)
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise V15ExtractError(f"v15 ZIP 복구 중 오류: {zip_path}: {exc}") from exc

    invalid, manifest_ok = _extracted_state(pkg_dir)
    if not manifest_ok or invalid:
        raise V15ExtractError(f"v15 추출 파일 SHA256 검증 실패: {invalid}")
    return pkg_dir


def resolve_v15_dir(base: Path | None = None) -> Path:
    """Always validate extracted files, including legacy core-97 test paths."""
    return extract_v15_files(base)
