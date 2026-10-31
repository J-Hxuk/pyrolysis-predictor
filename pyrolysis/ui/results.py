"""
results.py — 다중 대상 결과·검증 정보·순위 표시 (작업 5·21).

설계 10 "결과/비교/검증 정보" 화면을 담당한다.

구조:
  - 표시용 포매팅은 Streamlit 에 의존하지 않는 순수 함수로 구현한다.
    (format_target_result_row, build_results_table, format_evaluation_summary,
     build_ranking_table, format_ranking_caption)
    그래야 AppTest 없이 완료 기준을 단위 검증할 수 있다 (session.py 와 같은 방식).
  - st.* 를 호출하는 얇은 래퍼는 그 순수 함수의 결과를 그릴 뿐이다.

완료 기준 (작업 21):
  - 미지원 값·실패값이 0 으로 나타나지 않는다. 수치 0·예측 미검출·미지원·오류를 구별한다.
  - 평가 오차(MAE/RMSE)를 개별 결과의 신뢰구간·최대 오차로 표현하지 않는다.
  - 순위는 제공된 후보의 예측 기준임을 표시하고, 종합 점수·최적 조건 보장·
    미세 차이의 과학적 유의성을 만들지 않는다.
  - 반올림 값이 같아 보이는 후보는 원래 값을 확인할 수 있게 한다.
"""

from __future__ import annotations

import math
from typing import Any

from pyrolysis.contracts import PredictionBatch, RankingResult, TargetPrediction

__all__ = [
    # 순수 포매팅 (환경 비의존, 테스트 대상)
    "ResultCellKind",
    "format_target_value",
    "format_target_result_row",
    "build_results_table",
    "format_evaluation_summary",
    "build_ranking_table",
    "format_ranking_caption",
    # Streamlit 래퍼 (작업 5 호환 + 작업 21 화면)
    "display_prediction_result",
    "display_prediction_error",
    "display_results_screen",
    "display_ranking_screen",
    "display_evaluation_info",
]


# ── 셀 종류 구분 (0 혼동 방지) ──────────────────────────────────────────────


class ResultCellKind:
    """결과 셀 하나의 종류. 수치 0 과 미검출·미지원·오류를 구별한다."""

    NUMERIC = "numeric"            # 회귀 수치값 (0.0 도 유효한 값)
    DETECTED = "detected"          # 분류: 검출 표지
    NOT_DETECTED = "not_detected"  # 분류: 예측 미검출 (0 아님)
    UNSUPPORTED = "unsupported"    # 미지원 대상 (모델 없음)
    ERROR = "error"                # 계산 실패


# ── 단일 값 포매팅 ──────────────────────────────────────────────────────────


def format_target_value(
    value: Any,
    *,
    task: str,
    precision: int = 4,
    positive_label: str | None = None,
    negative_label: str | None = None,
) -> dict[str, Any]:
    """예측값 하나를 표시용 셀로 변환한다.

    수치 0 은 "0" 으로 표시하되 kind=NUMERIC 로 둔다 (미검출·미지원과 구별).
    분류 표지는 검출/미검출로 구별하고, 미지원·오류는 수치 0 과 섞지 않는다.

    Returns:
        {"display": str, "kind": ResultCellKind, "raw": Any}
    """
    if task == "regression":
        try:
            num = float(value)
        except (TypeError, ValueError):
            return {"display": "계산 실패", "kind": ResultCellKind.ERROR, "raw": value}
        if not math.isfinite(num):
            return {"display": "계산 실패", "kind": ResultCellKind.ERROR, "raw": value}
        # 0.0 도 유효한 수치. 미검출·미지원과 구별된다.
        return {
            "display": f"{num:.{precision}f}",
            "kind": ResultCellKind.NUMERIC,
            "raw": num,
        }

    # 분류: 정의된 표지로 반환된 값 (문자열/표지). 검출/미검출을 구별.
    label = value
    is_positive = (positive_label is not None and str(value) == str(positive_label))
    is_negative = (negative_label is not None and str(value) == str(negative_label))
    if is_positive:
        return {"display": f"검출 ({label})", "kind": ResultCellKind.DETECTED, "raw": value}
    if is_negative:
        # 미검출은 수치 0 이 아니다.
        return {"display": f"미검출 ({label})", "kind": ResultCellKind.NOT_DETECTED, "raw": value}
    # positive/negative 표지를 모르면 표지 자체를 그대로 표시 (0 으로 바꾸지 않음)
    return {"display": str(label), "kind": ResultCellKind.DETECTED, "raw": value}


# ── 대상별 결과 행 ──────────────────────────────────────────────────────────


def format_target_result_row(
    target_id: str,
    *,
    label_ko: str,
    family: str,
    task: str,
    unit: str | None,
    serve_state: str,
    prediction: TargetPrediction | None,
    row_index: int = 0,
    precision: int = 4,
    positive_label: str | None = None,
    negative_label: str | None = None,
    measurement_ref: str | None = None,
) -> dict[str, Any]:
    """결과표의 대상 한 줄을 구성한다.

    - serve_state 가 active 가 아니거나 prediction 이 없으면 미지원으로 표시한다
      (수치 0 이 아니라 "미지원" 과 사유).
    - 분류 초기 화면은 검출 표지만 표시한다.

    Returns: 표 한 줄의 딕셔너리.
    """
    base = {
        "target_id": target_id,
        "대상": label_ko,
        "범주": family,
        "단위": unit or "-",
        "지원_상태": serve_state,
        "측정_기준": measurement_ref or "-",
    }

    if serve_state != "active" or prediction is None:
        base.update({
            "값": "미지원",
            "kind": ResultCellKind.UNSUPPORTED,
            "raw": None,
        })
        return base

    if row_index >= len(prediction.values):
        base.update({"값": "계산 실패", "kind": ResultCellKind.ERROR, "raw": None})
        return base

    cell = format_target_value(
        prediction.values[row_index],
        task=task,
        precision=precision,
        positive_label=positive_label,
        negative_label=negative_label,
    )
    base.update({"값": cell["display"], "kind": cell["kind"], "raw": cell["raw"]})
    return base


def build_results_table(
    batch: PredictionBatch,
    target_meta: list[dict[str, Any]],
    *,
    row_index: int = 0,
) -> list[dict[str, Any]]:
    """한 후보(row_index)의 모든 대상 결과를 표로 만든다.

    target_meta 는 활성·미지원을 모두 포함한다. 각 항목:
        {id, label_ko, family, task, unit, serve_state,
         precision?, positive_label?, negative_label?, measurement_ref?}

    배치가 완료되지 않았으면 결과 행을 만들지 않는다 (부분 결과 노출 금지).
    """
    if not batch.completed or batch.error:
        return []

    rows = []
    for meta in target_meta:
        tid = meta["id"]
        pred = batch.predictions.get(tid)
        rows.append(
            format_target_result_row(
                tid,
                label_ko=meta.get("label_ko", tid),
                family=meta.get("family", "-"),
                task=meta.get("task", "regression"),
                unit=meta.get("unit"),
                serve_state=meta.get("serve_state", "active"),
                prediction=pred,
                row_index=row_index,
                precision=int(meta.get("precision", 4)),
                positive_label=meta.get("positive_label"),
                negative_label=meta.get("negative_label"),
                measurement_ref=meta.get("measurement_ref"),
            )
        )
    return rows


# ── 검증 정보 (독립 평가 지표) ───────────────────────────────────────────────


def format_evaluation_summary(
    target_id: str,
    *,
    task: str,
    metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    """선택 대상의 독립 평가 지표를 표시용으로 정리한다.

    - 회귀: MAE/RMSE(원래 단위)·표본 수·평가 범위.
    - 분류: 정밀도/재현율/F1/혼동행렬·클래스별 표본 수.
    - 평가 미완료·계산 불가 지표는 null 과 사유로 둔다.
    - MAE/RMSE 를 개별 결과의 신뢰구간·최대 오차로 표현하지 않는다 (주의 문구 고정).
    """
    disclaimer = (
        "독립 평가 지표는 전체 평가 자료의 요약입니다. "
        "개별 예측의 신뢰구간이나 최대 오차가 아니며, 높은 정확도를 보장하지 않습니다."
    )

    if metrics is None:
        return {
            "target_id": target_id,
            "task": task,
            "available": False,
            "reason": "독립 평가가 완료되지 않았습니다.",
            "disclaimer": disclaimer,
            "rows": [],
        }

    rows: list[dict[str, Any]] = []
    if task == "regression":
        unit = metrics.get("unit") or "-"
        mae = metrics.get("mae")
        rmse = metrics.get("rmse")
        rows.append({"지표": "MAE", "값": _fmt_metric(mae), "단위": unit})
        rows.append({"지표": "RMSE", "값": _fmt_metric(rmse), "단위": unit})
        rows.append({"지표": "평가 표본 수", "값": str(metrics.get("n_samples", "-")), "단위": "건"})
    else:
        for key, name in (("precision", "정밀도"), ("recall", "재현율"), ("f1", "F1")):
            rows.append({"지표": name, "값": _fmt_metric(metrics.get(key)), "단위": "-"})
        cc = metrics.get("class_counts", {})
        rows.append({"지표": "음성 표본 수", "값": str(cc.get("true_0", "-")), "단위": "건"})
        rows.append({"지표": "양성 표본 수", "값": str(cc.get("true_1", "-")), "단위": "건"})
        rows.append({"지표": "평가 표본 수", "값": str(metrics.get("n_samples", "-")), "단위": "건"})

    return {
        "target_id": target_id,
        "task": task,
        "available": True,
        "reason": metrics.get("threshold_note") or metrics.get("note") or "",
        "disclaimer": disclaimer,
        "confusion_matrix": metrics.get("confusion_matrix"),
        "rows": rows,
    }


def _fmt_metric(v: Any, precision: int = 4) -> str:
    """지표 하나를 표시한다. null·계산 불가는 '계산 불가'. 0 은 수치로 유지."""
    if v is None:
        return "계산 불가"
    try:
        num = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not math.isfinite(num):
        return "계산 불가"
    return f"{num:.{precision}f}"


# ── 순위표 ──────────────────────────────────────────────────────────────────


def build_ranking_table(
    result: RankingResult,
    *,
    precision: int = 4,
) -> list[dict[str, Any]]:
    """RankingResult 를 표시용 순위표로 변환한다.

    - 공동 순위(1,1,3)를 그대로 표시한다.
    - 반올림 표시값과 함께 원래 값(원래 정밀도)을 별도 열로 제공한다.
      그래서 같은 표시값이어도 원래 값 차이를 확인할 수 있다.
    - 학습 범위 이탈은 주의 표시로만 둔다 (제외·감점하지 않음).
    """
    rows = []
    for rc in result.ranked:
        display = f"{rc.value:.{precision}f}"
        rows.append({
            "순위": rc.rank,
            "후보": rc.candidate_id,
            "원본_행": rc.source_row,
            "값": display,
            "원래_값": repr(rc.value),  # 반올림 전 원래 값 확인용
            "범위_이탈": "주의" if rc.range_exceeded else "-",
        })
    return rows


def format_ranking_caption(result: RankingResult) -> str:
    """순위표 아래에 붙일 설명. 과장·보장을 만들지 않는다."""
    dir_ko = {
        "maximize": "최대화",
        "minimize": "최소화",
        "target": "목표값 접근",
    }.get(result.direction, result.direction)
    unit = result.unit or "무차원"
    base = (
        f"'{result.target_id}' 대상의 예측값 기준 {dir_ko} 순위입니다 (단위: {unit}). "
        "제공된 후보들의 예측값만 비교한 결과이며, 종합 점수나 최적 조건을 보장하지 않습니다. "
        "표시값이 같아 보여도 원래 값이 다를 수 있으며, 미세한 차이가 "
        "과학적으로 유의하다는 뜻은 아닙니다."
    )
    if result.direction == "target" and result.target_value is not None:
        base += f" 목표값: {result.target_value} {unit}."
    return base


# ── Streamlit 래퍼 ─────────────────────────────────────────────────────────
#
# 아래 함수들은 st.* 를 호출한다. import 는 함수 안에서 하여
# 이 모듈을 순수 함수만 쓰는 테스트에서 Streamlit 없이 import 할 수 있게 한다.


def display_prediction_result(batch: PredictionBatch, target_label_ko: str) -> None:
    """작업 5 호환: 단일 대상 단일 후보 결과를 표시한다."""
    import streamlit as st

    if not batch.completed or batch.error:
        display_prediction_error(batch)
        return
    target_id = list(batch.predictions.keys())[0]
    pred = batch.predictions[target_id]
    value = pred.values[0]
    unit = pred.unit if pred.unit else ""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        st.success(f"**{target_label_ko}:** {value:.4f} {unit}")
    else:
        st.success(f"**{target_label_ko}:** {value}")
    with st.expander("📊 예측 세부 정보"):
        st.json({
            "요청_ID": batch.request_id,
            "묶음_ID": batch.bundle_id,
            "입력_해시": batch.input_snapshot_hash[:12],
        })


def display_prediction_error(batch: PredictionBatch) -> None:
    import streamlit as st

    if batch.error:
        st.error(f"**예측 실패**\n\n{batch.error.message_ko}")
        st.caption(f"오류 코드: {batch.error.code}")
    else:
        st.warning("예측 결과가 없습니다.")


def display_results_screen(
    batch: PredictionBatch,
    target_meta: list[dict[str, Any]],
    *,
    row_index: int = 0,
) -> None:
    """작업 21: 기체·액체·작용기·기공·촉매 성능별 결과표를 표시한다."""
    import streamlit as st

    if not batch.completed or batch.error:
        display_prediction_error(batch)
        return

    rows = build_results_table(batch, target_meta, row_index=row_index)
    if not rows:
        st.warning("표시할 결과가 없습니다.")
        return

    # 화면 표 (내부 정렬은 탐색용. 공식 순위는 순위 서비스만 계산)
    display_rows = [
        {k: v for k, v in r.items() if k not in ("target_id", "kind", "raw")}
        for r in rows
    ]
    st.dataframe(display_rows, use_container_width=True, hide_index=True)
    st.caption(
        "미지원·실패 대상은 '미지원'·'계산 실패'로 표시되며 수치 0 과 다릅니다. "
        "분류 대상은 검출/미검출 표지만 표시합니다."
    )


def display_evaluation_info(
    target_id: str,
    *,
    task: str,
    metrics: dict[str, Any] | None,
) -> None:
    """작업 21: 선택 대상의 독립 평가 지표(검증 정보)를 표시한다."""
    import streamlit as st

    summary = format_evaluation_summary(target_id, task=task, metrics=metrics)
    st.subheader(f"검증 정보 — {target_id}")
    if not summary["available"]:
        st.info(summary["reason"])
        st.caption(summary["disclaimer"])
        return
    st.dataframe(summary["rows"], use_container_width=True, hide_index=True)
    if summary.get("confusion_matrix") is not None:
        st.caption("혼동행렬 (행=실측, 열=예측)")
        st.table(summary["confusion_matrix"])
    if summary["reason"]:
        st.caption(summary["reason"])
    st.caption(summary["disclaimer"])


def display_ranking_screen(result: RankingResult, *, precision: int = 4) -> None:
    """작업 21: 수치형 단일 지표의 후보 순위표를 표시한다."""
    import streamlit as st

    rows = build_ranking_table(result, precision=precision)
    if not rows:
        st.warning("순위를 매길 후보가 없습니다.")
        return
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.caption(format_ranking_caption(result))
