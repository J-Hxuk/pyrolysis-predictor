"""
test_input_ui.py - Task 20 세션 상태·실행 조건 검증.

Streamlit 위젯 렌더링(AppTest)이 아니라, 작업 20의 상태 전이 규칙을
순수 함수로 검증한다 (환경 비의존). UI 위젯 자체는 작업 5 의 inputs.py 가 담당한다.

확인 항목:
1. 입력이 바뀌면 이전 결과를 무효화한다 (REQ-14).
2. 입력이 그대로면 결과를 유지한다.
3. 입력·파일 업로드 자체로는 예측하지 않는다 (실행 버튼 필요).
4. 오류가 있으면 예측 흐름이 중단된다.
5. 준비되지 않으면 예측하지 않는다.
6. 초기화는 결과·입력 지문을 지운다.
"""

from __future__ import annotations

from pyrolysis.ui.session import (
    SessionState,
    can_run_prediction,
    input_fingerprint,
    should_invalidate_result,
)


class TestInvalidation:
    def test_changed_input_invalidates(self):
        fp = input_fingerprint({"x1": 1.0, "x2": 2.0})
        assert should_invalidate_result(fp, {"x1": 9.0, "x2": 2.0}) is True

    def test_same_input_keeps(self):
        inputs = {"x1": 1.0, "x2": 2.0}
        fp = input_fingerprint(inputs)
        assert should_invalidate_result(fp, dict(inputs)) is False

    def test_first_time_no_invalidation(self):
        assert should_invalidate_result(None, {"x1": 1.0}) is False

    def test_fingerprint_order_independent(self):
        a = input_fingerprint({"x1": 1.0, "x2": 2.0})
        b = input_fingerprint({"x2": 2.0, "x1": 1.0})
        assert a == b


class TestCanRun:
    def test_requires_run_clicked(self):
        # 업로드·입력만으로 실행하지 않는다
        assert can_run_prediction(has_errors=False, run_clicked=False, ready=True) is False

    def test_blocked_by_errors(self):
        assert can_run_prediction(has_errors=True, run_clicked=True, ready=True) is False

    def test_blocked_when_not_ready(self):
        assert can_run_prediction(has_errors=False, run_clicked=True, ready=False) is False

    def test_runs_when_all_conditions_met(self):
        assert can_run_prediction(has_errors=False, run_clicked=True, ready=True) is True


class TestSessionState:
    def test_sync_clears_result_on_input_change(self):
        store = {}
        s = SessionState(store)
        s.sync_inputs({"x1": 1.0})
        s.set_result("RESULT_A")
        assert s.get_result() == "RESULT_A"
        # 입력 변경 -> 결과 무효화
        s.sync_inputs({"x1": 2.0})
        assert s.get_result() is None

    def test_sync_keeps_result_on_same_input(self):
        store = {}
        s = SessionState(store)
        s.sync_inputs({"x1": 1.0})
        s.set_result("RESULT_A")
        s.sync_inputs({"x1": 1.0})  # 동일 입력
        assert s.get_result() == "RESULT_A"

    def test_clear_resets(self):
        store = {}
        s = SessionState(store)
        s.sync_inputs({"x1": 1.0})
        s.set_result("RESULT_A")
        s.clear()
        assert s.get_result() is None
        # clear 후 같은 입력을 다시 넣어도 (이전 지문 None) 결과는 비어 있다
        s.sync_inputs({"x1": 1.0})
        assert s.get_result() is None