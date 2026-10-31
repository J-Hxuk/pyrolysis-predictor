"""
extract.py — v60 Final Package 선택 추출(원본 zip 미변경).

패키지 zip 을 data/imports 아래로 추출해 모델·master·예측기를 사용할 수 있게 한다.
원본 zip 과 파일 내용은 변경하지 않는다. 필요한 파일만 쓰는 선택 추출은 아니며(패키지가
서로 의존하는 소수 파일 묶음이므로 전체를 추출), 다른 프로젝트 자료와 섞지 않게 전용
디렉터리로 분리한다.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

__all__ = [
    "ZIP_NAME",
    "PACKAGE_ROOT",
    "IMPORT_DIR",
    "FINAL_SUBDIR",
    "extract_v60_package",
    "resolve_v60_final_dir",
    "V60ExtractError",
]

# 전달된 zip 위치 (data/ 아래). 원본은 변경하지 않는다.
ZIP_NAME = "Biochar_Stage1_v60_Final_Package.zip"
# zip 내부 최상위 폴더
PACKAGE_ROOT = "Biochar_Stage1_v60_Final"
# 추출 위치 (다른 import 자료와 분리)
IMPORT_DIR = Path("data/imports/Biochar_Stage1_v60")
FINAL_SUBDIR = PACKAGE_ROOT

# 모델 예측에 반드시 필요한 핵심 파일 (추출 후 존재 확인용)
REQUIRED_FILES = [
    f"{PACKAGE_ROOT}/01_Stage1_Master_v60_256.csv",
    f"{PACKAGE_ROOT}/MODEL_STAGE1_V60_BET_XGB_Eng59.joblib",
    f"{PACKAGE_ROOT}/MODEL_STAGE1_V60_TPV_XGB_Eng59.joblib",
    f"{PACKAGE_ROOT}/predict_stage1_v60.py",
    f"{PACKAGE_ROOT}/02_BET_AllData_FixedArchitecture_LODO.csv",
    f"{PACKAGE_ROOT}/02_TPV_AllData_FixedArchitecture_LODO.csv",
    f"{PACKAGE_ROOT}/example_input.json",
    f"{PACKAGE_ROOT}/example_output.json",
    f"{PACKAGE_ROOT}/FINAL_STATUS.json",
]


class V60ExtractError(Exception):
    """v60 패키지 추출 실패."""


def _zip_path(base: Path) -> Path:
    # data/ 아래 또는 루트에 있을 수 있다.
    for cand in (base / "data" / ZIP_NAME, base / ZIP_NAME):
        if cand.is_file():
            return cand
    raise V60ExtractError(
        f"v60 패키지 zip 을 찾을 수 없습니다: {base / 'data' / ZIP_NAME}"
    )


def extract_v60_package(base: Path | None = None, *, overwrite: bool = False) -> Path:
    """v60 패키지를 data/imports/Biochar_Stage1_v60 아래로 추출한다.

    Returns: 추출된 Final 디렉터리 경로.
    Raises: V60ExtractError — zip 없음 또는 핵심 파일 누락.
    """
    base = Path(base) if base is not None else Path.cwd()
    zip_path = _zip_path(base)
    out_root = base / IMPORT_DIR
    final_dir = out_root / FINAL_SUBDIR

    need_extract = overwrite or not (final_dir / "predict_stage1_v60.py").is_file()
    if need_extract:
        out_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            names = set(zf.namelist())
            missing = [f for f in REQUIRED_FILES if f not in names]
            if missing:
                raise V60ExtractError(
                    "v60 zip 에 핵심 파일이 없습니다: " + ", ".join(missing)
                )
            # Final 폴더와 Independent_Challenge 폴더 모두 추출(검증 기록 보존)
            zf.extractall(out_root)

    if not (final_dir / "predict_stage1_v60.py").is_file():
        raise V60ExtractError(f"추출 후에도 예측기를 찾을 수 없습니다: {final_dir}")
    return final_dir


def resolve_v60_final_dir(base: Path | None = None) -> Path:
    """추출된 Final 디렉터리를 반환한다. 없으면 자동 추출한다."""
    base = Path(base) if base is not None else Path.cwd()
    final_dir = base / IMPORT_DIR / FINAL_SUBDIR
    if not (final_dir / "predict_stage1_v60.py").is_file():
        final_dir = extract_v60_package(base)
    return final_dir
