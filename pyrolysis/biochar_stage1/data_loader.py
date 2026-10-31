"""
data_loader.py — Stage1 BET 데이터 불러오기 (학습과 분리).

지시서 §1: "데이터 불러오기와 모델 학습은 별개 작업으로 구현하라."
이 모듈은 읽기만 한다. 값을 생성하거나 정규화하지 않는다.

데이터 출처 (v41 ZIP, eval37):
  - base.csv       : BET 기준 평가 데이터 (182행, 10개 입력 + value + provenance_group)
  - extra_leng.csv : BET 추가 학습 후보 (99행, Leng2022)

데이터셋 버전 해시를 계산해 모델 메타데이터에 기록할 수 있게 한다.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pyrolysis.biochar_stage1.extract import resolve_eval37_dir
from pyrolysis.biochar_stage1.features import (
    BET_GROUP_COL,
    BET_INPUT_ORDER,
    BET_TARGET_COL,
)

__all__ = [
    "Stage1Dataset",
    "load_bet_base",
    "load_bet_extra",
    "load_stage1_dataset",
    "file_sha256",
]


def file_sha256(path: Path) -> str:
    """파일 내용의 SHA-256 (데이터셋 버전 식별용)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_bet_base(base: Path | None = None) -> pd.DataFrame:
    """eval37/base.csv (182행 기준 평가 데이터)를 읽는다.

    필수 열(10개 입력 + value + provenance_group)이 모두 있는지 확인한다.
    값을 변경하거나 결측을 채우지 않는다.
    """
    d = resolve_eval37_dir(base)
    df = pd.read_csv(d / "base.csv")
    required = BET_INPUT_ORDER + [BET_TARGET_COL, BET_GROUP_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"base.csv 에 필수 열이 없습니다: {missing}")
    return df


def load_bet_extra(base: Path | None = None) -> pd.DataFrame:
    """eval37/extra_leng.csv (99행 추가 후보)를 읽는다.

    추가 후보는 평가 폴드에 무조건 병합하지 않는다 (evaluation 에서 누설 방지 로직 적용).
    이 로더는 원본 열을 그대로 반환한다.
    """
    d = resolve_eval37_dir(base)
    df = pd.read_csv(d / "extra_leng.csv")
    # extra_leng 는 provenance_group 과 10개 입력, value 를 포함한다.
    required = BET_INPUT_ORDER + [BET_TARGET_COL, BET_GROUP_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"extra_leng.csv 에 필수 열이 없습니다: {missing}")
    return df


@dataclass
class Stage1Dataset:
    """Stage1 BET 데이터 묶음과 버전 식별 정보."""

    base: pd.DataFrame            # 182행 기준 평가 데이터
    extra: pd.DataFrame           # 99행 추가 후보
    base_sha256: str
    extra_sha256: str
    base_path: str
    extra_path: str

    @property
    def dataset_version_hash(self) -> str:
        """base+extra 해시를 결합한 데이터셋 버전 식별자."""
        combined = (self.base_sha256 + ":" + self.extra_sha256).encode()
        return hashlib.sha256(combined).hexdigest()[:16]

    def version_info(self) -> dict:
        return {
            "dataset_source": "Biochar_Complete_Project_v41 / eval37",
            "base_file": "eval37/base.csv",
            "extra_file": "eval37/extra_leng.csv",
            "base_rows": int(len(self.base)),
            "extra_rows": int(len(self.extra)),
            "base_sha256": self.base_sha256,
            "extra_sha256": self.extra_sha256,
            "dataset_version_hash": self.dataset_version_hash,
        }


def load_stage1_dataset(base: Path | None = None) -> Stage1Dataset:
    """base + extra 를 함께 읽고 버전 해시를 계산한다 (학습 아님)."""
    d = resolve_eval37_dir(base)
    base_df = load_bet_base(base)
    extra_df = load_bet_extra(base)
    return Stage1Dataset(
        base=base_df,
        extra=extra_df,
        base_sha256=file_sha256(d / "base.csv"),
        extra_sha256=file_sha256(d / "extra_leng.csv"),
        base_path=str(d / "base.csv"),
        extra_path=str(d / "extra_leng.csv"),
    )
