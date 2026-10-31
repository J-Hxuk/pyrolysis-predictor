"""
inference.py — 저장된 Stage1 BET 모델 로드·예측·적용 범위 판정.

지시서 §2·§5:
  - 모델 저장 후 재로드해도 동일 입력에 동일 예측 (결정성).
  - 입력 단위·음수·비정상 조성 검사.
  - 학습 데이터 변수별 범위 이탈 판정 (적용 범위 검증 모듈 분리).
  - 외삽·지지 데이터 부족 경고. 검증되지 않은 신뢰도 백분율을 만들지 않는다.
  - 변수별 최소·최대 안이라는 사실만으로 조합이 충분히 검증됐다고 판단하지 않는다.

앱은 이 모듈로 활성 모델만 읽어 예측한다. 재학습하지 않는다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from pyrolysis.biochar_stage1.features import (
    BET_INPUT_ORDER,
    BET_INPUT_VARIABLES,
)

__all__ = [
    "BetModelHandle",
    "load_bet_model",
    "predict_bet",
    "check_input_sanity",
    "check_applicability_domain",
    "DomainVerdict",
]


@dataclass
class BetModelHandle:
    """로드된 BET 모델과 메타데이터."""

    pipeline: Any
    metadata: dict[str, Any]
    version_dir: Path

    @property
    def version(self) -> str:
        return self.metadata.get("model_version", "unknown")

    def training_numeric_ranges(self) -> dict[str, dict[str, float]]:
        """메타데이터의 학습 자료 범위 (없으면 빈 dict)."""
        return self.metadata.get("training_numeric_ranges", {})


def load_bet_model(version_dir: Path) -> BetModelHandle:
    """버전 디렉터리에서 model.joblib + metadata.json 을 읽는다."""
    version_dir = Path(version_dir)
    model_path = version_dir / "model.joblib"
    meta_path = version_dir / "metadata.json"
    if not model_path.is_file():
        raise FileNotFoundError(f"model.joblib 이 없습니다: {model_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"metadata.json 이 없습니다: {meta_path}")
    pipeline = joblib.load(model_path)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return BetModelHandle(pipeline=pipeline, metadata=metadata, version_dir=version_dir)


def _as_frame(inputs: dict[str, float] | pd.DataFrame) -> pd.DataFrame:
    """단일 dict 또는 DataFrame 을 입력 순서 컬럼의 DataFrame 으로 정규화한다."""
    if isinstance(inputs, dict):
        df = pd.DataFrame([inputs])
    else:
        df = inputs.copy()
    # 누락 입력 열은 NaN 으로 채우되(파이프라인 중앙값 대체가 처리), 순서를 고정한다.
    for c in BET_INPUT_ORDER:
        if c not in df.columns:
            df[c] = np.nan
    return df[BET_INPUT_ORDER].apply(pd.to_numeric, errors="coerce")


def predict_bet(
    handle: BetModelHandle, inputs: dict[str, float] | pd.DataFrame
) -> np.ndarray:
    """BET 비표면적(m²/g)을 예측한다. 입력 순서는 features.BET_INPUT_ORDER 로 고정.

    결정성: 같은 입력·같은 모델이면 항상 같은 값을 반환한다 (RF 는 학습 후 고정).
    """
    X = _as_frame(inputs)
    pred = handle.pipeline.predict(X)
    return np.asarray(pred, dtype=float)


# ── 입력 정상성 검사 (단위·음수·비정상 조성) ────────────────────────────────


@dataclass
class DomainVerdict:
    """적용 범위 판정 결과 (예측 차단이 아닌 주의 중심)."""

    errors: list[dict[str, Any]] = field(default_factory=list)      # 물리적 위반 등
    warnings: list[dict[str, Any]] = field(default_factory=list)    # 범위 이탈·외삽
    checked: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "checked": self.checked,
        }


def check_input_sanity(inputs: dict[str, float]) -> DomainVerdict:
    """단위·음수·비정상 조성 등 물리적 정상성을 검사한다.

    - 질량% 입력(C/H/N/O/VM/Ash/FC)은 0~100 범위를 벗어나면 오류.
    - 온도·시간·승온속도는 음수면 오류.
    - C/H/N/O 합과 VM/Ash/FC 합은 '점검'만 한다. 100 으로 재조정하지 않으며
      합이 과도하게 벗어나면 basis 확인 주의를 남긴다 (오류 아님).
    """
    v = DomainVerdict()
    pct_cols = ["C_pct", "H_pct", "N_pct", "O_pct", "VM_pct", "Ash_pct", "FC_pct"]
    nonneg_cols = ["temperature_C", "time_min", "heating_rate_C_min"]

    for c in pct_cols:
        if c not in inputs or inputs[c] is None:
            continue
        val = float(inputs[c])
        if val < 0 or val > 100:
            v.errors.append({
                "input_id": c, "value": val,
                "message_ko": f"{BET_INPUT_VARIABLES[c].label_ko}({c})는 질량%이며 0~100 범위여야 합니다.",
            })
    for c in nonneg_cols:
        if c not in inputs or inputs[c] is None:
            continue
        val = float(inputs[c])
        if val < 0:
            v.errors.append({
                "input_id": c, "value": val,
                "message_ko": f"{BET_INPUT_VARIABLES[c].label_ko}({c})는 음수가 될 수 없습니다.",
            })

    # 조성 합계 점검 (재조정 금지)
    def _sum_check(cols, label, basis):
        if all(c in inputs and inputs[c] is not None for c in cols):
            s = sum(float(inputs[c]) for c in cols)
            v.checked[label] = round(s, 3)
            if s < 90 or s > 110:
                v.warnings.append({
                    "message_ko": (
                        f"{label} 합계가 {s:.1f} 입니다. {basis} 기준과 반올림 차이를 "
                        "확인하세요. 합계만으로 입력을 자동 수정하지 않습니다."
                    ),
                })

    _sum_check(["C_pct", "H_pct", "N_pct", "O_pct"], "CHNO 합", "dry ash-free(daf)")
    _sum_check(["VM_pct", "Ash_pct", "FC_pct"], "VM+Ash+FC 합", "dry basis")
    return v


def check_applicability_domain(
    inputs: dict[str, float], handle: BetModelHandle
) -> DomainVerdict:
    """학습 자료 변수별 범위 이탈을 '주의'로 판정한다 (예측 차단 아님).

    - 학습 범위(metadata.training_data 또는 training_numeric_ranges)를 쓴다.
    - 범위를 벗어나면 TRAINING_RANGE_EXCEEDED 주의.
    - 모든 값이 범위 안이어도 조합의 검증을 보장하지 않는다는 주의를 항상 남긴다.
    - 검증되지 않은 신뢰도 백분율을 만들지 않는다.
    """
    v = DomainVerdict()
    ranges = handle.metadata.get("training_numeric_ranges") or {}
    # 메타데이터에 범위가 없으면 비어 있을 수 있다 -> 범위 판정 불가 주의
    if not ranges:
        v.warnings.append({
            "message_ko": (
                "모델 메타데이터에 학습 변수 범위가 없어 범위 이탈 판정을 할 수 없습니다. "
                "적용 범위는 Leng2022 기반 비활성화 바이오매스 열분해로 제한됩니다."
            ),
        })
        return v

    n_exceeded = 0
    for c in BET_INPUT_ORDER:
        if c not in inputs or inputs[c] is None or c not in ranges:
            continue
        val = float(inputs[c])
        lo = ranges[c]["min"]
        hi = ranges[c]["max"]
        v.checked[c] = {"value": val, "train_min": lo, "train_max": hi}
        if val < lo or val > hi:
            n_exceeded += 1
            v.warnings.append({
                "input_id": c, "value": val,
                "code": "TRAINING_RANGE_EXCEEDED",
                "message_ko": (
                    f"{BET_INPUT_VARIABLES[c].label_ko}({c})={val} 가 학습 범위"
                    f"[{lo}, {hi}] 를 벗어났습니다 (외삽 의심)."
                ),
            })

    # 범위 안이어도 조합 검증을 보장하지 않는다는 주의 (항상)
    if n_exceeded == 0:
        v.warnings.append({
            "code": "WITHIN_RANGE_NOT_GUARANTEED",
            "message_ko": (
                "모든 입력이 학습 변수 범위 안에 있습니다. 다만 변수별 범위 안이라는 "
                "사실만으로 이 입력 조합이 충분히 검증됐다고 보장하지 않습니다."
            ),
        })
    return v
