"""
v15_loader.py — Stage2 v15 데이터 로더·검증 (학습과 분리된 읽기 전용).

핵심 학습 데이터: 03_Stage2_QE_Core_97.csv (97행 / 25 DOI, 탐색적 reported uptake qe).

지시 사항:
  - 전체 97행·25 DOI, observation_id 고유성, DOI·material_uid·experiment_series_id 관계,
    단위·결측 현황, 물질수지 상한, 제조공정 분류·활성화 유효성을 검사한다.
  - 원본 결측을 평균·대표값으로 덮어쓰지 않는다(대체는 학습 fold 내부에서만).
  - 미보고 활성화제를 임의 none 처리하지 않는다.
  - qe / qt / removal / qmax 를 단일 회귀 타깃으로 혼합하지 않는다 → Core 97(qe)만 로드.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pyrolysis.biochar_stage2.v15_extract import resolve_v15_dir
from pyrolysis.biochar_stage2 import v15_features as fc

__all__ = [
    "Stage2CoreDataset",
    "load_core_qe",
    "load_chen16",
    "validate_core",
    "TargetMixError",
]

CORE_FILE = "03_Stage2_QE_Core_97.csv"
CHEN16_FILE = "09_Manufacturing_Recommendation_Benchmark_Chen16.csv"

# 핵심 qe 에 '절대 합치지 않는' 다른 타깃 파일 (지시 §2)
SEPARATE_TARGET_FILES = {
    "qexp_kinetic": "04_QEXP_Kinetic_Optional_6.csv",
    "qt_120min": "05_QT_120min_Separate_32.csv",
    "removal": "06_Removal_Separate_16.csv",
    "qe_derived_from_removal": "07_QE_Derived_From_Removal_16_DO_NOT_DOUBLE_COUNT.csv",
    "langmuir_qmax": "08_Langmuir_Qmax_Fitted_Separate_32.csv",
}

EXPECTED_ROWS = 97
EXPECTED_DOI = 25
# Core 97 의 타깃은 reported experimental uptake 하나여야 한다.
EXPECTED_TARGET_NAME = "qe_reported_mg_g"


class TargetMixError(Exception):
    """qe/qt/removal/qmax 타깃 혼합 시도 차단."""


@dataclass
class Stage2CoreDataset:
    """Core 97 qe 데이터와 메타."""

    df: pd.DataFrame
    source_path: str

    @property
    def n_rows(self) -> int:
        return int(len(self.df))

    @property
    def n_doi(self) -> int:
        return int(self.df[fc.GROUP_DOI].nunique())


def load_core_qe(base: Path | None = None) -> Stage2CoreDataset:
    """Core 97 qe 데이터를 읽는다 (다른 타깃 파일과 합치지 않음).

    읽은 즉시 타깃이 단일 reported uptake 인지 확인하고, 아니면 TargetMixError.
    값을 변경하거나 결측을 채우지 않는다.
    """
    d = resolve_v15_dir(base)
    p = d / CORE_FILE
    df = pd.read_csv(p)
    if "target_name" in df.columns:
        names = set(df["target_name"].dropna().unique())
        if names - {EXPECTED_TARGET_NAME}:
            raise TargetMixError(
                f"Core 97 에 예상치 못한 타깃 유형이 있습니다: {names}. "
                "qe/qt/removal/qmax 를 혼합하지 않습니다."
            )
    return Stage2CoreDataset(df=df, source_path=str(p))


def load_chen16(base: Path | None = None) -> pd.DataFrame:
    """Chen 2022 16개 제조조건 추천 벤치마크를 읽는다 (제거율 %)."""
    d = resolve_v15_dir(base)
    return pd.read_csv(d / CHEN16_FILE)


def _numeric_missing(df: pd.DataFrame, cols: list[str]) -> dict[str, dict[str, Any]]:
    n = len(df)
    out: dict[str, dict[str, Any]] = {}
    for c in cols:
        if c not in df.columns:
            out[c] = {"present": False, "n_missing": n, "missing_pct": 100.0}
            continue
        s = df[c]
        nmiss = int(s.isna().sum())
        entry = {"present": True, "n_missing": nmiss,
                 "missing_pct": round(100.0 * nmiss / n, 2) if n else 0.0}
        num = pd.to_numeric(s, errors="coerce")
        if num.notna().any():
            entry.update({"min": float(num.min()), "max": float(num.max()),
                          "mean": float(num.mean())})
        out[c] = entry
    return out


def validate_core(ds: Stage2CoreDataset) -> dict[str, Any]:
    """Core 97 데이터 정합성·단위·결측·물질수지·제조공정 현황을 검사한다 (학습 아님)."""
    df = ds.df
    n = len(df)
    report: dict[str, Any] = {"source": ds.source_path, "n_rows": n}

    # 1) 행 수·DOI 수
    report["row_count_ok"] = (n == EXPECTED_ROWS)
    report["n_doi"] = int(df[fc.GROUP_DOI].nunique()) if fc.GROUP_DOI in df.columns else None
    report["doi_count_ok"] = (report["n_doi"] == EXPECTED_DOI)

    # 2) observation_id 고유성
    if "observation_id" in df.columns:
        report["observation_id_unique"] = bool(df["observation_id"].is_unique)
        report["n_observation_id"] = int(df["observation_id"].nunique())
    else:
        report["observation_id_unique"] = None

    # 3) DOI·material_uid·experiment_series_id 관계
    rel: dict[str, Any] = {}
    for c in ["doi", "material_uid", "experiment_series_id", "sample_group"]:
        if c in df.columns:
            rel[c] = {"nunique": int(df[c].nunique(dropna=True)),
                      "n_missing": int(df[c].isna().sum())}
    # material 이 여러 DOI 에 걸치는지(공유 시료) 확인
    if {"material_uid", "doi"} <= set(df.columns):
        mu_doi = df.groupby("material_uid")["doi"].nunique()
        rel["material_uid_in_multiple_doi"] = int((mu_doi > 1).sum())
    report["id_relationships"] = rel

    # 4) 입력 단위·결측 현황 (흡착/제조/실측/raw조성)
    report["missing"] = {
        "adsorption": _numeric_missing(df, fc.ADSORPTION_INPUTS),
        "manufacturing": _numeric_missing(df, fc.MANUFACTURING_INPUTS),
        "measured_property": _numeric_missing(df, fc.MEASURED_PROPERTY_INPUTS),
        "raw_composition_optional": _numeric_missing(df, fc.RAW_COMPOSITION_OPTIONAL),
    }

    # 5) 물질수지 상한 검증: target_value <= C0/dose (물리 제약). cap 열과 대조.
    mb: dict[str, Any] = {}
    if {fc.C0_COL, fc.DOSE_COL, fc.TARGET_COLUMN} <= set(df.columns):
        c0 = pd.to_numeric(df[fc.C0_COL], errors="coerce")
        dose = pd.to_numeric(df[fc.DOSE_COL], errors="coerce")
        tgt = pd.to_numeric(df[fc.TARGET_COLUMN], errors="coerce")
        cap = c0 / dose.replace(0, np.nan)
        valid = c0.notna() & dose.notna() & tgt.notna() & (dose != 0)
        over = valid & (tgt > cap * (1 + 1e-9))
        mb["n_evaluable"] = int(valid.sum())
        mb["n_target_exceeds_C0_over_dose"] = int(over.sum())
        mb["exceeding_observation_ids"] = (
            df.loc[over, "observation_id"].tolist() if "observation_id" in df.columns else []
        )
        # 데이터에 기록된 cap 열과 계산값 대조
        if fc.MASS_BALANCE_CAP_COL in df.columns:
            recorded = pd.to_numeric(df[fc.MASS_BALANCE_CAP_COL], errors="coerce")
            diff = (recorded - cap).abs()
            mb["recorded_cap_matches_calc"] = int((diff <= 1e-6 + 1e-6 * cap.abs()).sum())
    report["mass_balance"] = mb

    # 6) 제조공정 분류·활성화 유효성 (값 생성 없이 분포만)
    proc: dict[str, Any] = {}
    if "manufacturing_route" in df.columns:
        proc["manufacturing_route_counts"] = {
            str(k): int(v) for k, v in df["manufacturing_route"].value_counts(dropna=False).items()
        }
    if "activation_or_modification" in df.columns:
        am = df["activation_or_modification"]
        proc["activation_reported"] = int(am.notna().sum())
        proc["activation_missing"] = int(am.isna().sum())
        proc["activation_note"] = "미보고(결측)를 none 으로 자동 보충하지 않는다."
    # 활성화제 원문 라벨은 역할 정규화 전 — 모델 확정 입력 아님(참고 집계만)
    for c in ["reported_activation_agent", "reported_impregnation_agent"]:
        if c in df.columns:
            proc[f"{c}_reported"] = int(df[c].notna().sum())
    report["manufacturing"] = proc

    # 7) 타깃 분포 (단일 유형 확인)
    report["target"] = {
        "column": fc.TARGET_COLUMN,
        "unit": fc.TARGET_UNIT,
        "n_unique_values": int(df[fc.TARGET_COLUMN].nunique()) if fc.TARGET_COLUMN in df.columns else None,
        "target_name_values": sorted(df["target_name"].dropna().unique().tolist())
        if "target_name" in df.columns else None,
    }

    # 8) raw 조성 확보율 요약 (지시: 조성 기반 일반화 주장 불가 근거)
    comp_available = {}
    for c in fc.RAW_COMPOSITION_OPTIONAL:
        if c in df.columns:
            comp_available[c] = int(pd.to_numeric(df[c], errors="coerce").notna().sum())
    report["raw_composition_available_counts"] = comp_available
    report["raw_composition_note"] = (
        "원료 조성 확보율이 극히 낮아(대부분 0~2행) 조성 기반 일반화 모델로 주장할 수 없다. "
        "원료명에서 조성을 생성하지 않는다."
    )

    return report
