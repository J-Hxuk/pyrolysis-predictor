"""
test_evaluation.py - Task 14 완료 기준 검증 (제공 상태 중심).

작업 14에서 추가한 determine_serve_status 와, 분류 지표의 클래스 부족 시
null+사유 처리를 검증한다. 기본 지표 계산의 정상 경로는 최소만 확인한다.

합성 자료로만 테스트한다. 실제 성능 목표·임계값은 사용하지 않는다.
"""

from __future__ import annotations

import numpy as np

from pyrolysis.evaluation import (
    ServeStatus,
    compute_classification_metrics,
    determine_serve_status,
)


class TestServeStatus:
    def test_unconfigured_when_not_configured(self):
        s = determine_serve_status("y", configured=False, data_sufficient=False,
                                   trained=False, evaluated=False)
        assert s.state == "unconfigured"
        assert s.web_enabled is False

    def test_data_insufficient(self):
        s = determine_serve_status("y", configured=True, data_sufficient=False,
                                   trained=False, evaluated=False)
        assert s.state == "data_insufficient"

    def test_trained_but_not_evaluated(self):
        s = determine_serve_status("y", configured=True, data_sufficient=True,
                                   trained=True, evaluated=False)
        assert s.state == "trained_unvalidated"
        assert s.web_enabled is False

    def test_evaluated_when_all_ready(self):
        s = determine_serve_status("y", configured=True, data_sufficient=True,
                                   trained=True, evaluated=True)
        assert s.state == "evaluated"
        # evaluated 는 배포 선정 가능 상태일 뿐 아직 active 가 아니다
        assert s.web_enabled is False

    def test_disabled_overrides(self):
        s = determine_serve_status("y", configured=True, data_sufficient=True,
                                   trained=True, evaluated=True, disabled=True)
        assert s.state == "disabled"

    def test_missing_required_classes_blocks_activation(self):
        # 분류에서 필수 클래스가 부족하면 평가·학습이 돼 보여도 활성화 불가
        s = determine_serve_status("y", configured=True, data_sufficient=True,
                                   trained=True, evaluated=True,
                                   required_classes_present=False)
        assert s.state == "data_insufficient"
        assert s.web_enabled is False

    def test_evaluated_does_not_imply_accuracy(self):
        # 평가 완료 상태에 성능 임계값이 없다 (정확도 보장 아님).
        s = determine_serve_status("y", configured=True, data_sufficient=True,
                                   trained=True, evaluated=True)
        # 상태는 evaluated 이지만 정확도 관련 플래그/점수는 상태에 포함되지 않는다
        assert s.state == "evaluated"
        assert not hasattr(s, "accuracy")
        assert not hasattr(s, "passed_threshold")

    def test_unknown_state_rejected(self):
        import pytest
        with pytest.raises(ValueError):
            ServeStatus("y", "superb")


class TestClassificationMetricsClassShortage:
    def test_single_class_yields_null_metrics_with_reason(self):
        # 전부 같은 클래스면 정밀도/재현율/F1 을 계산할 수 없어 null + 사유
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1])
        m = compute_classification_metrics(y_true, y_pred)
        assert m["precision"] is None
        assert m["recall"] is None
        assert m["f1"] is None
        assert "error" in m or m["threshold_note"]

    def test_both_classes_yield_numeric_metrics(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        m = compute_classification_metrics(y_true, y_pred)
        assert m["precision"] is not None
        assert m["recall"] is not None
        assert m["f1"] is not None