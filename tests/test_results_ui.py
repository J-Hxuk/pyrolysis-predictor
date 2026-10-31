"""
test_results_ui.py — 작업 21 완료 기준 검증.

AppTest(Streamlit 위젯 렌더링)가 아니라 작업 21 의 표시용 포매팅 규칙을
순수 함수로 검증한다 (환경 비의존). 세션 상태는 작업 20 의 test_input_ui 가 담당한다.

확인 항목 (완료 기준):
1. 수치 0·예측 미검출·미지원·오류를 서로 구별한다 (미지원·실패가 0 으로 보이지 않는다).
2. 초기 분류 결과는 검출/미검출 표지만 표시한다.
3. 평가 오차(MAE/RMSE)를 개별 결과의 신뢰구간·최대 오차로 표현하지 않는다.
4. 계산 불가 지표는 '계산 불가'로 두고 0 으로 바꾸지 않는다.
5. 순위는 제공 후보의 예측 기준임을 표시하고 종합 점수·최적 보장을 만들지 않는다.
6. 반올림 표시값이 같아 보여도 원래 값을 확인할 수 있다.
7. 미완료 배치는 결과표를 만들지 않는다 (부분 결과 노출 금지).
"""

from __future__ import annotations

from pyrolysis.contracts import (
    PredictionBatch,
    RankedCandidate,
    RankingResult,
    TargetPrediction,
)
from pyrolysis.ranking import rank_candidates
from pyrolysis.ui.results import (
    ResultCellKind,
    build_ranking_table,
    build_results_table,
    format_evaluation_summary,
    format_ranking_caption,
    format_target_value,
)


def _batch(predictions: dict, completed=True, error=None):
    return PredictionBatch(
        request_id="r1",
        bundle_id="b1",
        bundle_hash="h",
        input_snapshot_hash="snap",
        predictions=predictions,
        completed=completed,
        error=error,
    )


# ── 1. 수치 0 / 미검출 / 미지원 / 오류 구별 ──────────────────────────────────


class TestCellKinds:
    def test_numeric_zero_is_numeric_not_unsupported(self):
        cell = format_target_value(0.0, task="regression")
        assert cell["kind"] == ResultCellKind.NUMERIC
        assert cell["display"] == "0.0000"

    def test_numeric_nonzero(self):
        cell = format_target_value(12.5, task="regression", precision=2)
        assert cell["kind"] == ResultCellKind.NUMERIC
        assert cell["display"] == "12.50"

    def test_regression_nonfinite_is_error(self):
        cell = format_target_value(float("inf"), task="regression")
        assert cell["kind"] == ResultCellKind.ERROR
        assert cell["display"] != "0"

    def test_regression_non_numeric_is_error(self):
        cell = format_target_value("oops", task="regression")
        assert cell["kind"] == ResultCellKind.ERROR

    def test_classification_detected(self):
        cell = format_target_value(
            "present", task="binary_classification",
            positive_label="present", negative_label="absent",
        )
        assert cell["kind"] == ResultCellKind.DETECTED
        assert "검출" in cell["display"]

    def test_classification_not_detected_is_not_zero(self):
        cell = format_target_value(
            "absent", task="binary_classification",
            positive_label="present", negative_label="absent",
        )
        assert cell["kind"] == ResultCellKind.NOT_DETECTED
        # 미검출은 수치 0 이 아니다
        assert cell["display"] != "0"
        assert "미검출" in cell["display"]


# ── 2 & 7. 결과표: 미지원·미완료 처리 ────────────────────────────────────────


class TestResultsTable:
    def _meta(self):
        return [
            {"id": "y_reg", "label_ko": "기체 수율", "family": "gas",
             "task": "regression", "unit": "wt%", "serve_state": "active"},
            {"id": "y_cls", "label_ko": "작용기 검출", "family": "functional_group",
             "task": "binary_classification", "unit": None, "serve_state": "active",
             "positive_label": "present", "negative_label": "absent"},
            {"id": "y_off", "label_ko": "미지원 성능", "family": "performance",
             "task": "regression", "unit": "x", "serve_state": "disabled"},
        ]

    def test_zero_regression_shows_zero_value_not_unsupported(self):
        batch = _batch({
            "y_reg": TargetPrediction(target_id="y_reg", values=[0.0], unit="wt%"),
            "y_cls": TargetPrediction(target_id="y_cls", values=["absent"], unit=None),
        })
        rows = build_results_table(batch, self._meta())
        reg = next(r for r in rows if r["target_id"] == "y_reg")
        assert reg["kind"] == ResultCellKind.NUMERIC
        assert reg["값"] == "0.0000"

    def test_unsupported_target_shows_unsupported_not_zero(self):
        batch = _batch({
            "y_reg": TargetPrediction(target_id="y_reg", values=[1.0], unit="wt%"),
            "y_cls": TargetPrediction(target_id="y_cls", values=["present"], unit=None),
        })
        rows = build_results_table(batch, self._meta())
        off = next(r for r in rows if r["target_id"] == "y_off")
        assert off["kind"] == ResultCellKind.UNSUPPORTED
        assert off["값"] == "미지원"
        assert off["값"] != "0"

    def test_classification_shows_detection_label_only(self):
        batch = _batch({
            "y_reg": TargetPrediction(target_id="y_reg", values=[1.0], unit="wt%"),
            "y_cls": TargetPrediction(target_id="y_cls", values=["present"], unit=None),
        })
        rows = build_results_table(batch, self._meta())
        cls = next(r for r in rows if r["target_id"] == "y_cls")
        assert cls["kind"] == ResultCellKind.DETECTED
        assert "검출" in cls["값"]

    def test_incomplete_batch_yields_no_rows(self):
        batch = _batch({}, completed=False)
        rows = build_results_table(batch, self._meta())
        assert rows == []


# ── 3 & 4. 검증 정보: 신뢰구간이 아님, 계산 불가 유지 ────────────────────────


class TestEvaluationSummary:
    def test_regression_summary_not_confidence_interval(self):
        metrics = {"mae": 1.23, "rmse": 2.0, "unit": "wt%", "n_samples": 40,
                   "note": "..."}
        summary = format_evaluation_summary("y", task="regression", metrics=metrics)
        assert summary["available"] is True
        # 신뢰구간·최대 오차 표현이 아니라는 고정 주의 문구가 있다
        assert "신뢰구간" in summary["disclaimer"]
        assert "최대 오차" in summary["disclaimer"]
        mae_row = next(r for r in summary["rows"] if r["지표"] == "MAE")
        assert mae_row["값"] == "1.2300"

    def test_missing_metric_is_not_zero(self):
        # 분류에서 정밀도가 계산 불가(None)이면 '계산 불가', 0 아님
        metrics = {"precision": None, "recall": None, "f1": None,
                   "confusion_matrix": [[5, 0]], "class_counts": {"true_0": 5, "true_1": 0},
                   "n_samples": 5, "threshold_note": "클래스 부족"}
        summary = format_evaluation_summary("c", task="binary_classification", metrics=metrics)
        prec = next(r for r in summary["rows"] if r["지표"] == "정밀도")
        assert prec["값"] == "계산 불가"
        assert prec["값"] != "0"

    def test_unavailable_when_no_metrics(self):
        summary = format_evaluation_summary("y", task="regression", metrics=None)
        assert summary["available"] is False
        assert summary["rows"] == []


# ── 5 & 6. 순위표: 보장 없음, 원래 값 확인 ───────────────────────────────────


class TestRankingTable:
    def test_caption_states_candidate_basis_no_guarantee(self):
        result = RankingResult(
            request_id="r1", target_id="y", unit="wt%",
            direction="maximize", target_value=None,
            ranked=[RankedCandidate(candidate_id="c0", source_row=1, value=3.0,
                                    sort_value=3.0, rank=1)],
        )
        cap = format_ranking_caption(result)
        assert "제공된 후보" in cap
        assert "보장" in cap  # 최적 조건을 보장하지 않는다는 문구
        # 종합 점수를 만들지 않는다는 취지의 표현
        assert "종합 점수" in cap

    def test_rounding_tie_keeps_original_value(self):
        # 표시 소수점 2자리에서 두 값이 같아 보이지만 원래 값이 다르다
        batch = _batch({
            "y": TargetPrediction(target_id="y", values=[1.001, 1.002], unit="wt%"),
        })
        result = rank_candidates(batch, "y", "maximize")
        rows = build_ranking_table(result, precision=2)
        # 표시값은 같아 보일 수 있다
        assert rows[0]["값"] == rows[1]["값"] == "1.00"
        # 원래 값은 다르게 확인할 수 있다
        assert rows[0]["원래_값"] != rows[1]["원래_값"]
        # 순위는 미세 차이를 반영 (공동 순위가 아니다)
        assert rows[0]["순위"] != rows[1]["순위"]

    def test_tie_shows_shared_rank(self):
        batch = _batch({
            "y": TargetPrediction(target_id="y", values=[5.0, 5.0, 1.0], unit="wt%"),
        })
        result = rank_candidates(batch, "y", "maximize")
        rows = build_ranking_table(result)
        assert [r["순위"] for r in rows] == [1, 1, 3]

    def test_range_exceeded_shown_not_dropped(self):
        batch = _batch({
            "y": TargetPrediction(target_id="y", values=[2.0, 1.0], unit="wt%"),
        })
        result = rank_candidates(batch, "y", "maximize",
                                 range_flags=[True, False])
        rows = build_ranking_table(result)
        # 범위 이탈 후보가 제외되지 않고 표에 남는다
        assert len(rows) == 2
        exceeded = next(r for r in rows if r["범위_이탈"] == "주의")
        assert exceeded is not None
