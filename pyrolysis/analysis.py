"""
analysis.py — 변수 기여도와 로컬 연구용 표·그림 생성 (작업 15).

원칙:
  - 분석 방법은 명시적으로 지정한다. 미지원 방법은 오류로 안내한다 (자동 선택 안 함).
  - 서로 다른 단위의 오차를 전체 정확도로 합산하지 않는다 (대상별로 분리 저장).
  - 결과를 반응 메커니즘·인과관계·실제 3차원 기공 구조로 설명하지 않는다
    (산출물에 해석 주의 문구를 남긴다).
  - 웹 예측 결과나 다른 이용자 자료를 연구 기록에 자동 누적하지 않는다
    (호출자가 명시한 자료만 사용한다).

실제 자료에 적용할 분석 방법은 작업 27에서 확정한다.
Streamlit을 import하지 않는다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # headless 환경에서 그림 저장
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = [
    "AnalysisError",
    "SUPPORTED_IMPORTANCE_METHODS",
    "DISCLAIMER",
    "compute_variable_importance",
    "save_candidate_comparison",
    "save_pred_vs_actual",
    "save_variable_importance",
]

DISCLAIMER = (
    "로컬 연구 기록. 변수 기여도는 모델 관점의 상대적 영향이며 "
    "반응 메커니즘·인과관계·실제 3차원 기공 구조를 뜻하지 않습니다. "
    "서로 다른 단위의 지표는 합산하지 않습니다."
)

SUPPORTED_IMPORTANCE_METHODS = ("permutation",)


class AnalysisError(Exception):
    """분석 설정·실행 실패 (미지원 방법 등)."""


def compute_variable_importance(
    fitted_pipeline: Any,
    X: pd.DataFrame,
    y: Any,
    *,
    method: str,
    scoring: str | None = None,
    n_repeats: int = 10,
    random_seed: int | None = None,
) -> dict:
    """변수 기여도를 명시적으로 지정한 방법으로 계산한다.

    - method 가 지원 목록에 없으면 AnalysisError (자동 대체 안 함).
    - permutation: sklearn.inspection.permutation_importance.

    Returns:
        {
            "method": str,
            "scoring": str | None,
            "importances_mean": {feature: float},
            "importances_std": {feature: float},
            "disclaimer": str,
        }
    """
    if method not in SUPPORTED_IMPORTANCE_METHODS:
        raise AnalysisError(
            "지원하지 않는 변수 기여도 방법입니다: " + repr(method)
            + ". 지원: " + str(SUPPORTED_IMPORTANCE_METHODS)
        )

    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        fitted_pipeline, X, y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_seed,
    )
    cols = list(X.columns)
    return {
        "method": method,
        "scoring": scoring,
        "importances_mean": {c: float(m) for c, m in zip(cols, result.importances_mean)},
        "importances_std": {c: float(s) for c, s in zip(cols, result.importances_std)},
        "disclaimer": DISCLAIMER,
    }


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_candidate_comparison(
    rows: list,
    out_dir: Path,
    target_id: str,
    *,
    metric_name: str,
    metric_unit: str | None,
) -> Path:
    """대상 하나의 후보 비교표를 CSV 로 저장한다.

    단위가 다른 다른 대상의 지표와 섞지 않는다 (대상별 파일로 분리).

    Args:
        rows: [{"candidate": str, "score": float|None, "ran": bool, "reason": str}, ...]
        out_dir: 저장 디렉터리.
        target_id: 대상 ID.
        metric_name: 지표 이름.
        metric_unit: 지표 단위 (없으면 None).

    Returns:
        저장한 CSV 경로.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.insert(0, "target_id", target_id)
    df["metric_name"] = metric_name
    df["metric_unit"] = metric_unit if metric_unit is not None else ""
    path = out_dir / ("candidates_" + target_id + ".csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_pred_vs_actual(
    y_true: Any,
    y_pred: Any,
    out_dir: Path,
    target_id: str,
    *,
    unit: str | None = None,
) -> Path:
    """예측·실측 비교 산점도를 PNG 로 저장한다 (회귀).

    대상 하나의 원래 단위 값만 그린다. 다른 단위 대상과 한 그림에 합치지 않는다.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(yt, yp, s=12)
    lo = float(min(yt.min(), yp.min()))
    hi = float(max(yt.max(), yp.max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    unit_txt = (" (" + unit + ")") if unit else ""
    ax.set_xlabel("actual" + unit_txt)
    ax.set_ylabel("predicted" + unit_txt)
    ax.set_title(target_id)
    fig.tight_layout()
    path = out_dir / ("pred_vs_actual_" + target_id + ".png")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def save_variable_importance(
    importance: dict,
    out_dir: Path,
    target_id: str,
    *,
    feature_origin: dict | None = None,
) -> tuple:
    """변수 기여도를 JSON(해석 주의 포함)과 막대그림 PNG 로 저장한다.

    Args:
        importance: compute_variable_importance 결과.
        out_dir: 저장 디렉터리.
        target_id: 대상 ID.
        feature_origin: 변환된 변수 -> 원래 변수 대응 (있으면 기록).

    Returns:
        (json_path, png_path).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    record = dict(importance)
    record["target_id"] = target_id
    if feature_origin is not None:
        record["feature_origin"] = feature_origin

    json_path = out_dir / ("importance_" + target_id + ".json")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    means = importance["importances_mean"]
    feats = list(means.keys())
    vals = [means[k] for k in feats]
    fig, ax = plt.subplots(figsize=(5, max(2, 0.4 * len(feats) + 1)))
    ax.barh(feats, vals)
    ax.set_xlabel("permutation importance (mean)")
    ax.set_title(target_id + "  [not causal]")
    fig.tight_layout()
    png_path = out_dir / ("importance_" + target_id + ".png")
    fig.savefig(png_path, dpi=100)
    plt.close(fig)
    return json_path, png_path