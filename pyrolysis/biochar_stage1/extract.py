"""
extract.py — v41 ZIP 에서 Stage1 평가·학습에 필요한 파일만 선택 추출.

지시서 §1:
  - 519개 항목을 무조건 학습에 쓰지 않는다. 필요한 파일만 선택 추출한다.
  - 원본(ZIP)은 변경하지 않는다. 추출본은 data/imports 아래 별도 폴더에 둔다.
  - 데이터 불러오기와 모델 학습은 별개다 (이 모듈은 추출만 한다).

추출 대상 (지시서에서 명시한 중요 파일):
  merge30/ : Cumulative_observations_v30.csv, BET_core.csv,
             BET_default_train_pool.csv, TPV_default_train_pool.csv,
             Excluded_or_separate.csv, README_KO.md
  eval37/  : base.csv, extra_leng.csv, INPUT_CONTRACT_KO.md, run.py,
             README_KO.md, summary.csv, plan.json, versions.json,
             metrics.csv, splits.csv, predictions.csv
  eval38/  : README_KO.md  (외부 검증 보류 현황)
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

__all__ = [
    "ZIP_NAME",
    "IMPORT_SUBDIR",
    "REQUIRED_FILES",
    "extract_required_files",
    "resolve_eval37_dir",
    "resolve_merge30_dir",
]

ZIP_NAME = "Biochar_Complete_Project_v41.zip"
ZIP_ROOT = "Biochar_Complete_Project_v41"
IMPORT_SUBDIR = Path("data/imports/Biochar_Complete_Project_v41")

# ZIP 내부 상대 경로 (ZIP_ROOT 아래). 필요한 파일만 나열한다.
REQUIRED_FILES: list[str] = [
    "merge30/Cumulative_observations_v30.csv",
    "merge30/BET_core.csv",
    "merge30/BET_default_train_pool.csv",
    "merge30/TPV_default_train_pool.csv",
    "merge30/Excluded_or_separate.csv",
    "merge30/README_KO.md",
    "eval37/base.csv",
    "eval37/extra_leng.csv",
    "eval37/INPUT_CONTRACT_KO.md",
    "eval37/run.py",
    "eval37/README_KO.md",
    "eval37/summary.csv",
    "eval37/plan.json",
    "eval37/versions.json",
    "eval37/metrics.csv",
    "eval37/splits.csv",
    "eval37/predictions.csv",
    "eval38/README_KO.md",
]


class ExtractError(Exception):
    """ZIP 추출 실패."""


def extract_required_files(
    base: Path | None = None, *, overwrite: bool = False
) -> dict[str, str]:
    """필요한 파일만 data/imports 아래로 추출한다. 원본 ZIP 은 읽기만 한다.

    Args:
        base: 워크스페이스 루트. None 이면 현재 작업 디렉터리.
        overwrite: 이미 추출된 파일을 다시 쓸지 여부.

    Returns:
        {zip 내부 상대경로: 추출된 파일 절대경로}.

    Raises:
        ExtractError: ZIP 과 기존 추출본 모두에서 필요한 파일을 찾을 수 없을 때.
    """
    base = Path(base) if base is not None else Path.cwd()
    out_root = base / IMPORT_SUBDIR
    zip_path = base / ZIP_NAME

    # 전달용 프로젝트에는 v41 백업 ZIP이 없을 수 있다. 정규 위치에 이미
    # 추출된 원본이 모두 존재하면 다시 압축 해제할 필요가 없다.
    # 단, 일부 파일이 빠졌다면 '있는 척'하지 않고 ZIP에서 복구하거나 실패한다.
    extracted = {
        rel: str(out_root / rel)
        for rel in REQUIRED_FILES
        if (out_root / rel).is_file() and (out_root / rel).stat().st_size > 0
    }
    if not overwrite and len(extracted) == len(REQUIRED_FILES):
        return extracted

    if not zip_path.is_file():
        missing = [rel for rel in REQUIRED_FILES if rel not in extracted]
        raise ExtractError(
            f"Stage1 v41 원본 복구 불가: 백업 ZIP 없음 ({zip_path}); "
            f"기존 추출본 누락 {len(missing)}개: {missing}. "
            "원본 ZIP을 프로젝트 루트에 복사하거나, 확인된 추출본을 "
            f"{out_root}에 복원해야 합니다. 학습 데이터는 임의 생성하지 않습니다."
        )

    # 원본 ZIP이 있으면 없는 파일만 복구. 전체 항목을 먼저 검증하여
    # 손상/잘못된 ZIP에서 부분 추출이 시작되지 않도록 한다.
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        needed = [rel for rel in REQUIRED_FILES if overwrite or rel not in extracted]
        absent = [rel for rel in needed if f"{ZIP_ROOT}/{rel}" not in names]
        if absent:
            raise ExtractError(f"ZIP 내부에 필요한 항목이 없습니다: {absent}")
        for rel in needed:
            target = out_root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(f"{ZIP_ROOT}/{rel}") as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted[rel] = str(target)
    return extracted


def resolve_eval37_dir(base: Path | None = None) -> Path:
    """추출된 eval37 디렉터리를 반환한다. 없으면 자동 추출한다."""
    base = Path(base) if base is not None else Path.cwd()
    d = base / IMPORT_SUBDIR / "eval37"
    if not (d / "base.csv").is_file():
        extract_required_files(base)
    if not (d / "base.csv").is_file():
        raise ExtractError(f"eval37/base.csv 를 찾을 수 없습니다: {d}")
    return d


def resolve_merge30_dir(base: Path | None = None) -> Path:
    """추출된 merge30 디렉터리를 반환한다. 없으면 자동 추출한다."""
    base = Path(base) if base is not None else Path.cwd()
    d = base / IMPORT_SUBDIR / "merge30"
    if not (d / "README_KO.md").is_file():
        extract_required_files(base)
    return d
