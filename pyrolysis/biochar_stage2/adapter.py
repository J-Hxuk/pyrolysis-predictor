"""
adapter.py — Stage2 데이터 검증·학습 적격성 판정·Stage1 결합 어댑터.

지시서 §3:
  - 미검증 qe 는 자동 학습 대상 승인 금지.
  - 실측 qe / Langmuir qmax / 모델 예측값 / 미검증 reported capacity 를 구분.
  - Stage2 는 Stage1 과 독립적으로 학습·검증·저장할 수 있도록 설계.
  - Stage1 예측 BET/TPV 를 Stage2 입력으로 쓸 때 측정값/예측값 구분과 오차 전파 고려.

이번 Phase 1 은 '학습 차단'과 '구조 설계'만 수행한다. Stage2 ML 학습은 하지 않는다.
approve_training_rows 는 적격 행만 통과시키고, 데이터가 부족하면 학습을 차단한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from pyrolysis.biochar_stage2.schema import (
    APPROVED_MEASUREMENT_STATUS,
    STAGE2_FIELD_IDS,
    STAGE2_FIELDS,
    TARGET_FIELD,
    VERIFIED_EQUILIBRIUM_VALUES,
)

__all__ = [
    "Stage2ValidationResult",
    "validate_stage2_frame",
    "approve_training_rows",
    "Stage1LinkInput",
    "Stage1ValueSource",
    "build_stage2_input_row",
    "TrainingBlockedError",
]


class TrainingBlockedError(Exception):
    """미검증 데이터 또는 데이터 부족으로 Stage2 학습을 차단한다."""


@dataclass
class Stage2ValidationResult:
    """Stage2 프레임 검증 결과."""

    n_rows: int
    present_fields: list[str]
    missing_fields: list[str]
    unknown_fields: list[str]
    n_measured_qe: int           # measured 로 표기 + 평형 검증 + qe 수치
    n_unverified_capacity: int   # reported_unverified 등
    n_langmuir_qmax: int
    n_model_predicted: int
    n_training_eligible: int     # 실제 학습 승인 가능한 행
    issues: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "present_fields": self.present_fields,
            "missing_fields": self.missing_fields,
            "unknown_fields": self.unknown_fields,
            "n_measured_qe": self.n_measured_qe,
            "n_unverified_capacity": self.n_unverified_capacity,
            "n_langmuir_qmax": self.n_langmuir_qmax,
            "n_model_predicted": self.n_model_predicted,
            "n_training_eligible": self.n_training_eligible,
            "issues": self.issues,
        }


def _is_verified(value: Any) -> bool:
    if isinstance(value, bool):
        return value is True
    return value in VERIFIED_EQUILIBRIUM_VALUES


def _training_eligible_mask(df: pd.DataFrame) -> pd.Series:
    """학습 승인 가능한 행 마스크.

    승인 조건 (모두 충족):
      - measurement_status == 'measured'
      - equilibrium_verified 가 검증값
      - measured_qe_mg_g 가 유한 수치
      - 필수 조건 필드(required_for_training)가 모두 존재
    """
    mask = pd.Series(True, index=df.index)

    if "measurement_status" in df.columns:
        mask &= df["measurement_status"].astype(str).isin(APPROVED_MEASUREMENT_STATUS)
    else:
        return pd.Series(False, index=df.index)

    if "equilibrium_verified" in df.columns:
        mask &= df["equilibrium_verified"].map(_is_verified)
    else:
        return pd.Series(False, index=df.index)

    if TARGET_FIELD in df.columns:
        mask &= pd.to_numeric(df[TARGET_FIELD], errors="coerce").notna()
    else:
        return pd.Series(False, index=df.index)

    # 필수 학습 필드 존재
    required = [f.id for f in STAGE2_FIELDS if f.required_for_training]
    for c in required:
        if c not in df.columns:
            return pd.Series(False, index=df.index)
        if c not in ("measurement_status", "equilibrium_verified", TARGET_FIELD):
            mask &= df[c].notna()
    return mask


def validate_stage2_frame(df: pd.DataFrame) -> Stage2ValidationResult:
    """Stage2 프레임의 필드·상태·qe 종류별 수를 집계한다 (학습 아님)."""
    present = [c for c in STAGE2_FIELD_IDS if c in df.columns]
    missing = [c for c in STAGE2_FIELD_IDS if c not in df.columns]
    unknown = [c for c in df.columns if c not in STAGE2_FIELD_IDS]
    issues: list[dict[str, Any]] = []

    status = (
        df["measurement_status"].astype(str)
        if "measurement_status" in df.columns
        else pd.Series([], dtype=str)
    )

    def _count_status(kind: str) -> int:
        return int((status == kind).sum()) if len(status) else 0

    eligible_mask = _training_eligible_mask(df)
    n_eligible = int(eligible_mask.sum())

    # 측정 qe: measured 표기지만 평형 미검증인 행 경고
    if "measurement_status" in df.columns and "equilibrium_verified" in df.columns:
        measured_rows = df[df["measurement_status"].astype(str) == "measured"]
        unverified_measured = measured_rows[
            ~measured_rows["equilibrium_verified"].map(_is_verified)
        ]
        if len(unverified_measured):
            issues.append({
                "code": "MEASURED_BUT_EQUILIBRIUM_UNVERIFIED",
                "n_rows": int(len(unverified_measured)),
                "message_ko": (
                    f"measurement_status='measured' 이지만 평형 미검증 행이 "
                    f"{len(unverified_measured)}개 있습니다. 학습 대상으로 자동 승인하지 않습니다."
                ),
            })

    if not present:
        issues.append({
            "code": "NO_STAGE2_FIELDS",
            "message_ko": "Stage2 스키마 필드가 하나도 없습니다. 데이터셋 연결이 필요합니다.",
        })

    return Stage2ValidationResult(
        n_rows=int(len(df)),
        present_fields=present,
        missing_fields=missing,
        unknown_fields=unknown,
        n_measured_qe=n_eligible,
        n_unverified_capacity=_count_status("reported_unverified"),
        n_langmuir_qmax=_count_status("langmuir_qmax"),
        n_model_predicted=_count_status("model_predicted"),
        n_training_eligible=n_eligible,
        issues=issues,
    )


def approve_training_rows(df: pd.DataFrame, *, min_rows: int = 1) -> pd.DataFrame:
    """학습 승인 가능한 행만 반환한다. 승인 행이 부족하면 학습을 차단한다.

    Raises:
        TrainingBlockedError: 승인 가능한 행이 min_rows 미만이면.
            (지시서 §7-7: 미검증 데이터가 입력되면 학습이 차단되어야 한다.)
    """
    result = validate_stage2_frame(df)
    mask = _training_eligible_mask(df)
    approved = df[mask].copy()
    if len(approved) < min_rows:
        blocked = result.n_rows - len(approved)
        raise TrainingBlockedError(
            f"Stage2 학습 승인 행이 부족합니다 (승인 {len(approved)} / 전체 {result.n_rows}, "
            f"차단 {blocked}). 평형 검증·measured 상태가 아닌 행은 학습에서 제외됩니다. "
            "미검증 데이터로 Stage2 모델을 학습하지 않습니다."
        )
    return approved


# ── Stage1 ↔ Stage2 결합 어댑터 (측정값/예측값 구분, 오차 전파 고려) ───────────


# BET/TPV 값의 허용 출처. v60 예측은 별도 출처로 구분한다(측정값과 혼동 금지).
#   measured             : 실측값
#   stage1_predicted     : 일반 Stage1 예측(출처 모델 미지정)
#   stage1_v60_predicted : Stage1 v60 패키지 모델 예측(support_grade·경고 동반)
Stage1ValueSource = Literal["measured", "stage1_predicted", "stage1_v60_predicted"]


@dataclass
class Stage1LinkInput:
    """Stage2 입력으로 들어가는 BET/TPV 값의 출처·불확실성 표시.

    측정값과 Stage1 예측값을 섞지 않고 명시적으로 구분한다. 예측값을 쓰면 오차 전파를
    고려해야 하므로 source 와 (가능하면) uncertainty 를 함께 전달한다.

    v60 예측을 쓰는 경우 support_grade(HIGH/MODERATE/LOW)와 적용범위 경고를 함께 담아,
    향후 Stage2 가 예측 물성의 신뢰도를 입력 메타로 활용할 수 있게 한다. 이는 **인터페이스
    설계일 뿐** 이며 Phase 1 에서는 Stage1↔Stage2 실제 연결·Stage2 학습을 하지 않는다.
    """

    bet_m2_g: float | None
    tpv_cm3_g: float | None
    bet_source: Stage1ValueSource
    tpv_source: Stage1ValueSource
    bet_uncertainty: float | None = None
    tpv_uncertainty: float | None = None
    # v60 예측일 때만 채워진다(측정값/일반 예측이면 None).
    bet_support_grade: str | None = None
    tpv_support_grade: str | None = None
    bet_applicability_warnings: list[str] = field(default_factory=list)
    tpv_applicability_warnings: list[str] = field(default_factory=list)
    stage1_model_version: str | None = None

    @classmethod
    def from_v60_prediction(cls, v60_result: dict[str, Any]) -> "Stage1LinkInput":
        """v60 predictor.predict 결과(dict)를 Stage2 결합 입력으로 변환한다.

        차단된(blocked) 또는 손상 가능 결과는 값을 싣지 않고 경고만 남긴다. 예측 물성을
        measured 로 승격하지 않는다(측정값과 구분). 실제 Stage2 연결은 하지 않는다.
        """
        blocked = bool(v60_result.get("blocked", False))
        compat_ok = bool(v60_result.get("compatibility_ok", not blocked))
        bet = v60_result.get("BET", {}) if isinstance(v60_result.get("BET"), dict) else {}
        tpv = v60_result.get("TPV", {}) if isinstance(v60_result.get("TPV"), dict) else {}

        # 환경 비호환/차단이면 수치를 싣지 않는다(손상 값 전파 금지).
        usable = (not blocked) and compat_ok
        bet_val = float(bet["prediction"]) if (usable and "prediction" in bet) else None
        tpv_val = float(tpv["prediction"]) if (usable and "prediction" in tpv) else None

        def _interval_halfwidth(d: dict) -> float | None:
            iv = d.get("research_interval_approx")
            if isinstance(iv, (list, tuple)) and len(iv) == 2 and all(
                isinstance(x, (int, float)) for x in iv
            ):
                return float(abs(iv[1] - iv[0]) / 2.0)
            return None

        warn_bet = list(bet.get("applicability_warnings", []) or [])
        warn_tpv = list(tpv.get("applicability_warnings", []) or [])
        if not usable:
            reason = v60_result.get("reason") or [v60_result.get("warning_env", "v60 수치 예측 차단됨")]
            if isinstance(reason, str):
                reason = [reason]
            warn_bet = list(reason) + warn_bet
            warn_tpv = list(reason) + warn_tpv

        return cls(
            bet_m2_g=bet_val, tpv_cm3_g=tpv_val,
            bet_source="stage1_v60_predicted", tpv_source="stage1_v60_predicted",
            bet_uncertainty=_interval_halfwidth(bet) if usable else None,
            tpv_uncertainty=_interval_halfwidth(tpv) if usable else None,
            bet_support_grade=bet.get("support_grade") if usable else None,
            tpv_support_grade=tpv.get("support_grade") if usable else None,
            bet_applicability_warnings=warn_bet,
            tpv_applicability_warnings=warn_tpv,
            stage1_model_version=str(v60_result.get("data_version", "v60")),
        )

    @property
    def is_predicted(self) -> bool:
        """BET 또는 TPV 중 하나라도 예측값(측정값 아님)이면 True."""
        pred = {"stage1_predicted", "stage1_v60_predicted"}
        return self.bet_source in pred or self.tpv_source in pred

    def to_dict(self) -> dict[str, Any]:
        return {
            "bet_m2_g": self.bet_m2_g, "bet_source": self.bet_source,
            "bet_uncertainty": self.bet_uncertainty,
            "bet_support_grade": self.bet_support_grade,
            "bet_applicability_warnings": self.bet_applicability_warnings,
            "tpv_cm3_g": self.tpv_cm3_g, "tpv_source": self.tpv_source,
            "tpv_uncertainty": self.tpv_uncertainty,
            "tpv_support_grade": self.tpv_support_grade,
            "tpv_applicability_warnings": self.tpv_applicability_warnings,
            "stage1_model_version": self.stage1_model_version,
            "note": (
                "Stage1 예측 BET/TPV 를 쓰면 측정값과 구분하고 오차 전파를 고려해야 한다. "
                "Phase 1 에서는 결합 구조(인터페이스)만 제공하며 Stage1↔Stage2 실제 연결·"
                "Stage2 모델 학습은 하지 않는다."
            ),
        }


# 예측 출처로 간주하는 값(측정값이 아님). measured_* 필드에 넣지 않는다.
_PREDICTED_SOURCES = {"stage1_predicted", "stage1_v60_predicted"}


def build_stage2_input_row(
    conditions: dict[str, Any], link: Stage1LinkInput
) -> dict[str, Any]:
    """흡착조건 + BET/TPV(출처 표시)를 하나의 Stage2 입력 행으로 조립한다.

    예측 물성(stage1_predicted/stage1_v60_predicted)은 measured_* 필드에 넣지 않는다
    (측정값과 구분). 예측 출처·신뢰도·경고는 `_stage1_link` 에 보존한다.
    Stage2 ML 이 아직 없으므로 예측을 실행하지 않고, 입력 행 구조만 반환한다.
    """
    row = dict(conditions)
    row["measured_BET_m2_g"] = link.bet_m2_g if link.bet_source == "measured" else None
    row["measured_TPV_cm3_g"] = link.tpv_cm3_g if link.tpv_source == "measured" else None
    row["_stage1_link"] = link.to_dict()
    return row
