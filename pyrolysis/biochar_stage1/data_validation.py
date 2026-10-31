"""
data_validation.py — Stage1 BET 데이터 검증·적격성·현황 집계.

지시서 §1·§5: 컬럼·단위·결측·학습 적격성을 검사한다. 값을 생성하거나
합계를 임의로 100 으로 정규화하지 않는다. 이 모듈은 읽기 전용 집계다.

두 가지 검증을 구분한다:
  - 학습 데이터 현황 (dataset_audit): 학습·평가에 쓰는 자료의 결측·그룹·범위.
  - 사용자 입력 적용 범위 (inference.domain) 는 별도 모듈에서 수행한다.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pyrolysis.biochar_stage1.features import (
    BET_GROUP_COL,
    BET_INPUT_ORDER,
    BET_INPUT_VARIABLES,
    BET_TARGET_COL,
)

__all__ = [
    "column_presence",
    "missing_report",
    "numeric_ranges",
    "provenance_group_summary",
    "training_eligibility",
    "composition_sanity",
    "audit_dataset",
]


def column_presence(df: pd.DataFrame) -> dict[str, bool]:
    """필수 입력·타깃·그룹 열의 존재 여부."""
    cols = BET_INPUT_ORDER + [BET_TARGET_COL, BET_GROUP_COL]
    return {c: (c in df.columns) for c in cols}


def missing_report(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """입력 변수별 결측 수·비율을 집계한다 (값을 채우지 않는다)."""
    n = len(df)
    out: dict[str, dict[str, Any]] = {}
    for c in BET_INPUT_ORDER:
        if c not in df.columns:
            out[c] = {"present_column": False, "n_missing": n, "missing_pct": 100.0}
            continue
        num = pd.to_numeric(df[c], errors="coerce")
        n_missing = int(num.isna().sum())
        out[c] = {
            "present_column": True,
            "n_missing": n_missing,
            "missing_pct": round(100.0 * n_missing / n, 3) if n else 0.0,
            "unit": BET_INPUT_VARIABLES[c].unit,
            "basis": BET_INPUT_VARIABLES[c].basis,
        }
    # 타깃 결측
    if BET_TARGET_COL in df.columns:
        t = pd.to_numeric(df[BET_TARGET_COL], errors="coerce")
        out[BET_TARGET_COL] = {
            "present_column": True,
            "n_missing": int(t.isna().sum()),
            "missing_pct": round(100.0 * int(t.isna().sum()) / n, 3) if n else 0.0,
            "unit": "m2/g",
        }
    return out


def numeric_ranges(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """입력 변수별 관측 최소·최대·평균 (학습 범위 근거)."""
    out: dict[str, dict[str, float]] = {}
    for c in BET_INPUT_ORDER:
        if c not in df.columns:
            continue
        num = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(num) == 0:
            continue
        out[c] = {
            "min": float(num.min()),
            "max": float(num.max()),
            "mean": float(num.mean()),
            "n": int(len(num)),
        }
    return out


def provenance_group_summary(df: pd.DataFrame) -> dict[str, Any]:
    """출처 그룹 수·그룹별 행 수 (GroupKFold 분할 단위)."""
    if BET_GROUP_COL not in df.columns:
        return {"present_column": False}
    vc = df[BET_GROUP_COL].value_counts(dropna=False)
    return {
        "present_column": True,
        "n_groups": int(df[BET_GROUP_COL].dropna().nunique()),
        "group_sizes": {str(k): int(v) for k, v in vc.items()},
        "n_missing_group": int(df[BET_GROUP_COL].isna().sum()),
    }


def training_eligibility(df: pd.DataFrame) -> dict[str, Any]:
    """학습 적격성: 10개 입력이 모두 있고 타깃이 수치인 행 수.

    적격 = 10개 입력 전부 수치 + value 수치. 그렇지 않은 행은 '부적격'으로
    집계만 하고 삭제·대체하지 않는다.
    """
    present_inputs = [c for c in BET_INPUT_ORDER if c in df.columns]
    mask = pd.Series(True, index=df.index)
    for c in present_inputs:
        mask &= pd.to_numeric(df[c], errors="coerce").notna()
    if BET_TARGET_COL in df.columns:
        mask &= pd.to_numeric(df[BET_TARGET_COL], errors="coerce").notna()
    n_eligible = int(mask.sum())
    return {
        "n_rows": int(len(df)),
        "n_all_inputs_present": n_eligible,
        "n_excluded": int((~mask).sum()),
        "note": (
            "적격 = 10개 입력 전부 수치 + BET value 수치. 결측 행을 삭제하지 않고 "
            "집계상 제외만 한다. 추가 후보의 결측은 학습 폴드 내 중앙값 대체로 처리한다."
        ),
    }


def composition_sanity(df: pd.DataFrame) -> dict[str, Any]:
    """조성 합계를 점검한다 (단, 임의로 100 으로 재조정하지 않는다).

    - 원소분석 C+H+N+O 합계 분포를 보고한다 (daf 기준이면 100 근처가 많음).
    - 공업분석 VM+Ash+FC 합계 분포를 보고한다.
    합계만으로 원자료를 자동 수정하지 않는다. 음수·범위 이탈만 표시한다.
    """
    out: dict[str, Any] = {}
    el_cols = ["C_pct", "H_pct", "N_pct", "O_pct"]
    px_cols = ["VM_pct", "Ash_pct", "FC_pct"]

    def _sum_stats(cols: list[str]) -> dict[str, Any] | None:
        if not all(c in df.columns for c in cols):
            return None
        sub = df[cols].apply(pd.to_numeric, errors="coerce")
        complete = sub.dropna()
        if len(complete) == 0:
            return {"n_complete_rows": 0}
        s = complete.sum(axis=1)
        return {
            "n_complete_rows": int(len(complete)),
            "sum_min": float(s.min()),
            "sum_max": float(s.max()),
            "sum_mean": float(s.mean()),
            "n_sum_within_95_105": int(((s >= 95) & (s <= 105)).sum()),
        }

    out["elemental_CHNO_sum"] = _sum_stats(el_cols)
    out["proximate_VM_Ash_FC_sum"] = _sum_stats(px_cols)

    # 음수 조성 검사
    neg = {}
    for c in el_cols + px_cols:
        if c in df.columns:
            num = pd.to_numeric(df[c], errors="coerce")
            n_neg = int((num < 0).sum())
            if n_neg:
                neg[c] = n_neg
    out["negative_values"] = neg
    out["note"] = (
        "C/H/N/O 합을 100 으로 재조정하지 않는다. 합계는 점검용이며 원자료를 수정하지 않는다. "
        "원소분석은 daf, 공업분석은 dry basis 로 기준이 다르다."
    )
    return out


def audit_dataset(base_df: pd.DataFrame, extra_df: pd.DataFrame) -> dict[str, Any]:
    """base(182)·extra(99) 전체 데이터 현황 점검 (학습 아님)."""
    return {
        "base": {
            "n_rows": int(len(base_df)),
            "column_presence": column_presence(base_df),
            "missing": missing_report(base_df),
            "numeric_ranges": numeric_ranges(base_df),
            "provenance_groups": provenance_group_summary(base_df),
            "training_eligibility": training_eligibility(base_df),
            "composition_sanity": composition_sanity(base_df),
        },
        "extra": {
            "n_rows": int(len(extra_df)),
            "missing": missing_report(extra_df),
            "numeric_ranges": numeric_ranges(extra_df),
            "provenance_groups": provenance_group_summary(extra_df),
            "training_eligibility": training_eligibility(extra_df),
        },
        "policy": (
            "base 182행은 기준 평가 데이터, extra 99행은 추가 후보다. 추가 후보는 각 외부 "
            "평가 폴드의 출처·관측값과 겹치면 제외한다 (evaluation 모듈의 누설 방지 로직). "
            "원본 파일·플래그를 변경하지 않는다."
        ),
    }
