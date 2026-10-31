"""
test_session_state.py — 작업 22 세션 상태·무효화·초기화 검증.

AppTest(위젯 렌더링)가 아니라 작업 22 의 상태 전이·무효화 규칙을 순수
함수/클래스로 검증한다 (환경 비의존).

확인 항목 (완료 기준):
1. 입력·파일·묶음 변경 및 새 요청 시작 시 이전 결과·순위를 무효화한다.
2. 변화가 없으면 결과를 유지한다.
3. 요청 수명 상태(VALIDATING/PREDICTING/INVALID/DONE/FAILED/RETRY)를 구별한다.
4. 초기화 시 결과·순위·요청 상태와 업로더 키 참조를 정리한다.
5. 사용자 데이터는 세션(store)에만 보관한다 (전역·캐시 미사용).
"""

from __future__ import annotations

from pyrolysis.ui.bundle_cache import bundle_cache_key
from pyrolysis.ui.session import (
    RequestState,
    SessionState,
    combined_fingerprint,
    file_fingerprint,
    next_state,
    should_invalidate_result,
)


# ── 지문·무효화 ──────────────────────────────────────────────────────────────


class TestFingerprints:
    def test_file_fingerprint_none(self):
        assert file_fingerprint(None) is None

    def test_file_fingerprint_content_sensitive(self):
        a = file_fingerprint(b"abc", "f.csv")
        b = file_fingerprint(b"abd", "f.csv")
        assert a != b

    def test_file_fingerprint_name_sensitive(self):
        a = file_fingerprint(b"abc", "f1.csv")
        b = file_fingerprint(b"abc", "f2.csv")
        assert a != b

    def test_combined_changes_with_bundle(self):
        base = combined_fingerprint(inputs={"x": 1}, file_fp=None,
                                    bundle_id="b1", bundle_hash="h1")
        other = combined_fingerprint(inputs={"x": 1}, file_fp=None,
                                     bundle_id="b1", bundle_hash="h2")
        assert base != other

    def test_combined_changes_with_file(self):
        base = combined_fingerprint(inputs={"x": 1}, file_fp=None,
                                    bundle_id="b1", bundle_hash="h1")
        other = combined_fingerprint(inputs={"x": 1}, file_fp="ff",
                                     bundle_id="b1", bundle_hash="h1")
        assert base != other

    def test_should_invalidate_accepts_dict(self):
        from pyrolysis.ui.session import input_fingerprint
        fp = input_fingerprint({"x1": 1.0})
        assert should_invalidate_result(fp, {"x1": 2.0}) is True
        assert should_invalidate_result(fp, {"x1": 1.0}) is False
        assert should_invalidate_result(None, {"x1": 1.0}) is False


# ── 상태 머신 ────────────────────────────────────────────────────────────────


class TestStateMachine:
    def test_not_ready(self):
        assert next_state(RequestState.IDLE, "run", ready=False) == RequestState.NOT_READY

    def test_run_goes_validating(self):
        assert next_state(RequestState.IDLE, "run", ready=True) == RequestState.VALIDATING

    def test_validation_errors_go_invalid(self):
        s = next_state(RequestState.VALIDATING, "validated", has_errors=True)
        assert s == RequestState.INVALID

    def test_validation_ok_goes_predicting(self):
        s = next_state(RequestState.VALIDATING, "validated", has_errors=False)
        assert s == RequestState.PREDICTING

    def test_predicted_done(self):
        assert next_state(RequestState.PREDICTING, "predicted") == RequestState.DONE

    def test_failed(self):
        assert next_state(RequestState.PREDICTING, "failed") == RequestState.FAILED

    def test_timeout_retry(self):
        assert next_state(RequestState.PREDICTING, "timeout") == RequestState.RETRY

    def test_edit_invalidates_to_editing(self):
        assert next_state(RequestState.DONE, "edit", ready=True) == RequestState.EDITING


# ── SessionState 무효화·초기화 ──────────────────────────────────────────────


class TestSessionInvalidation:
    def test_input_change_clears_result_and_ranking(self):
        store = {}
        s = SessionState(store)
        s.sync(inputs={"x": 1}, bundle_id="b1", bundle_hash="h1")
        s.set_result("R")
        s.set_ranking("K")
        invalidated = s.sync(inputs={"x": 2}, bundle_id="b1", bundle_hash="h1")
        assert invalidated is True
        assert s.get_result() is None
        assert s.get_ranking() is None

    def test_bundle_change_clears_result(self):
        store = {}
        s = SessionState(store)
        s.sync(inputs={"x": 1}, bundle_id="b1", bundle_hash="h1")
        s.set_result("R")
        s.sync(inputs={"x": 1}, bundle_id="b1", bundle_hash="h2")  # manifest 해시 변경
        assert s.get_result() is None

    def test_file_change_clears_result(self):
        store = {}
        s = SessionState(store)
        s.sync(inputs={"x": 1}, file_fp="f1", bundle_id="b1", bundle_hash="h1")
        s.set_result("R")
        s.sync(inputs={"x": 1}, file_fp="f2", bundle_id="b1", bundle_hash="h1")
        assert s.get_result() is None

    def test_no_change_keeps_result(self):
        store = {}
        s = SessionState(store)
        s.sync(inputs={"x": 1}, bundle_id="b1", bundle_hash="h1")
        s.set_result("R")
        s.sync(inputs={"x": 1}, bundle_id="b1", bundle_hash="h1")
        assert s.get_result() == "R"

    def test_start_request_invalidates(self):
        store = {}
        s = SessionState(store)
        s.set_result("R")
        s.set_ranking("K")
        s.start_request()
        assert s.get_result() is None
        assert s.get_ranking() is None


class TestReset:
    def test_clear_removes_uploader_keys(self):
        store = {"up_1": "FILEOBJ", "other": "keep"}
        s = SessionState(store)
        s.register_uploader_key("up_1")
        s.set_result("R")
        s.clear()
        # 업로더 키가 세션에서 제거된다
        assert "up_1" not in store
        assert s.get_result() is None
        assert s.get_state() == RequestState.IDLE

    def test_clear_then_same_input_stays_empty(self):
        store = {}
        s = SessionState(store)
        s.sync(inputs={"x": 1}, bundle_id="b1", bundle_hash="h1")
        s.set_result("R")
        s.clear()
        s.sync(inputs={"x": 1}, bundle_id="b1", bundle_hash="h1")
        assert s.get_result() is None


class TestNoGlobalLeak:
    def test_data_stays_in_store(self):
        # 사용자 데이터가 전달한 store(세션)에만 들어간다
        store = {}
        s = SessionState(store)
        s.set_result({"secret_value": 42})
        assert store.get("prediction_result") == {"secret_value": 42}


class TestCacheKey:
    def test_cache_key_includes_id_and_hash(self):
        k = bundle_cache_key("bundleX", "deadbeef")
        assert "bundleX" in k and "deadbeef" in k

    def test_cache_key_changes_with_hash(self):
        assert bundle_cache_key("b", "h1") != bundle_cache_key("b", "h2")
