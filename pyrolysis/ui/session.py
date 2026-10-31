"""
session.py — Streamlit 세션 상태 관리 (작업 20·22).

입력·파일·결과·순위·요청 상태를 st.session_state 에만 보관한다.
사용자 데이터를 st.cache_data·전역 변수·공통 작업 큐·영구 파일에 저장하지 않는다.

이 모듈의 상태 전이 규칙은 Streamlit 에 의존하지 않는 순수 함수/클래스로 구현하고,
Streamlit 래퍼는 그 함수를 호출만 한다. 그래서 AppTest 없이 단위 검증이 가능하다.

작업 22 확장:
  - 입력·파일·묶음 변경과 새 요청 시작 시 이전 결과·순위를 무효화한다.
  - 요청 수명 상태(NOT_READY/IDLE/EDITING/VALIDATING/PREDICTING/INVALID/DONE/FAILED/RETRY)를 구별한다.
  - 초기화 시 업로더 키와 관련 세션 참조를 정리한다.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

__all__ = [
    "input_fingerprint",
    "file_fingerprint",
    "combined_fingerprint",
    "should_invalidate_result",
    "can_run_prediction",
    "RequestState",
    "next_state",
    "SessionState",
]


# ── 지문 계산 ────────────────────────────────────────────────────────────────


def input_fingerprint(inputs: dict) -> str:
    """입력 딕셔너리의 결정적 지문. 입력이 바뀌면 지문도 바뀐다."""
    raw = json.dumps(inputs, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def file_fingerprint(file_bytes: bytes | None, filename: str | None = None) -> str | None:
    """업로드 파일 내용의 결정적 지문. 파일이 없으면 None.

    파일 내용과 이름을 함께 해시하여, 같은 이름의 다른 파일·다른 이름의 같은
    내용도 구별한다. 원본 바이트를 세션 밖에 보관하지 않는다 (지문만 유지).
    """
    if file_bytes is None:
        return None
    h = hashlib.sha256()
    if filename:
        h.update(filename.encode("utf-8", "replace"))
        h.update(b"\x00")
    h.update(file_bytes)
    return h.hexdigest()[:16]


def combined_fingerprint(
    *,
    inputs: dict | None,
    file_fp: str | None,
    bundle_id: str | None,
    bundle_hash: str | None,
) -> str:
    """입력·파일·묶음을 아우르는 지문. 셋 중 하나라도 바뀌면 달라진다.

    묶음(ID+manifest 해시)이 바뀌면 이전 결과·순위를 무효화해야 한다.
    """
    parts = {
        "inputs": input_fingerprint(inputs) if inputs is not None else None,
        "file": file_fp,
        "bundle_id": bundle_id,
        "bundle_hash": bundle_hash,
    }
    raw = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def should_invalidate_result(prev_fingerprint: str | None, current: Any) -> bool:
    """지문이 이전과 달라졌으면 이전 결과를 무효화해야 하는지 판정한다.

    입력·파일·묶음 변경 시 기존 결과를 지운다 (REQ-14). 처음(prev=None)이면
    결과가 없으므로 무효화 불필요.

    current 는 지문 문자열 또는 입력 딕셔너리를 받는다 (작업 20 호환).
    딕셔너리이면 입력 지문으로 변환해 비교한다.
    """
    if prev_fingerprint is None:
        return False
    current_fingerprint = (
        input_fingerprint(current) if isinstance(current, dict) else current
    )
    return current_fingerprint != prev_fingerprint


def can_run_prediction(*, has_errors: bool, run_clicked: bool, ready: bool) -> bool:
    """예측을 실행해도 되는지 판정한다.

    - 입력·파일 업로드 자체로는 실행하지 않는다 (run_clicked 가 True 여야 함).
    - 입력/파일에 오류가 있으면 실행하지 않는다 (오류 파일 흐름 중단).
    - 묶음·대상이 준비되지 않았으면 실행하지 않는다.
    """
    return bool(run_clicked) and (not has_errors) and bool(ready)


# ── 요청 수명 상태 ───────────────────────────────────────────────────────────


class RequestState:
    """요청 처리 상태 (설계 10.1)."""

    NOT_READY = "NOT_READY"      # 묶음·설정 미준비
    IDLE = "IDLE"                # 준비됨, 입력 전/대기
    EDITING = "EDITING"          # 입력·파일·묶음 변경 (이전 결과 무효화됨)
    VALIDATING = "VALIDATING"    # 검증 중
    PREDICTING = "PREDICTING"    # 예측 중
    INVALID = "INVALID"          # 입력 오류 (오류만 표시)
    DONE = "DONE"                # 완료
    FAILED = "FAILED"            # 계산·출력 실패
    RETRY = "RETRY"              # 대기 한도 초과 등으로 재시도 안내


_ALL_STATES = {
    v for k, v in vars(RequestState).items() if not k.startswith("_") and isinstance(v, str)
}


def next_state(
    current: str,
    event: str,
    *,
    ready: bool = True,
    has_errors: bool = False,
) -> str:
    """이벤트에 따른 다음 상태를 결정한다 (순수 함수).

    events: "ready", "not_ready", "edit", "run", "validated", "predicted",
            "failed", "timeout", "reset".
    """
    if event == "not_ready":
        return RequestState.NOT_READY
    if event == "ready":
        return RequestState.IDLE if ready else RequestState.NOT_READY
    if event == "reset":
        return RequestState.IDLE if ready else RequestState.NOT_READY
    if event == "edit":
        # 입력·파일·묶음 변경 -> 편집 상태 (이전 결과 무효화)
        return RequestState.EDITING if ready else RequestState.NOT_READY
    if event == "run":
        if not ready:
            return RequestState.NOT_READY
        return RequestState.VALIDATING
    if event == "validated":
        # 검증 결과에 따라 분기
        return RequestState.INVALID if has_errors else RequestState.PREDICTING
    if event == "predicted":
        return RequestState.DONE
    if event == "failed":
        return RequestState.FAILED
    if event == "timeout":
        return RequestState.RETRY
    # 알 수 없는 이벤트는 현재 상태 유지
    return current if current in _ALL_STATES else RequestState.IDLE


# ── 세션 상태 컨테이너 ───────────────────────────────────────────────────────


class SessionState:
    """st.session_state 를 감싸는 얇은 상태 컨테이너.

    테스트에서는 일반 dict 를 넣어 Streamlit 없이 전이를 검증할 수 있다.
    사용자 입력·파일·결과·순위는 이 세션에만 보관한다.
    """

    _RESULT_KEY = "prediction_result"
    _RANKING_KEY = "ranking_result"
    _FP_KEY = "last_combined_fingerprint"
    _STATE_KEY = "request_state"
    _UPLOADER_KEYS = "uploader_keys"  # 정리 대상 업로더 위젯 키 목록

    def __init__(self, store: Any):
        self._store = store

    # 저장소 접근 ------------------------------------------------------------
    def _get(self, key, default=None):
        if hasattr(self._store, "get"):
            return self._store.get(key, default)
        return getattr(self._store, key, default)

    def _set(self, key, value):
        try:
            self._store[key] = value
        except TypeError:
            setattr(self._store, key, value)

    def _del(self, key):
        try:
            if key in self._store:
                del self._store[key]
            return
        except TypeError:
            pass
        if hasattr(self._store, key):
            delattr(self._store, key)

    # 상태 ------------------------------------------------------------------
    def get_state(self) -> str:
        return self._get(self._STATE_KEY, RequestState.IDLE)

    def set_state(self, state: str) -> None:
        self._set(self._STATE_KEY, state)

    def transition(self, event: str, *, ready: bool = True, has_errors: bool = False) -> str:
        new = next_state(self.get_state(), event, ready=ready, has_errors=has_errors)
        self.set_state(new)
        return new

    # 결과·순위 --------------------------------------------------------------
    def set_result(self, result: Any) -> None:
        self._set(self._RESULT_KEY, result)

    def get_result(self) -> Any:
        return self._get(self._RESULT_KEY)

    def set_ranking(self, ranking: Any) -> None:
        self._set(self._RANKING_KEY, ranking)

    def get_ranking(self) -> Any:
        return self._get(self._RANKING_KEY)

    # 업로더 키 관리 ----------------------------------------------------------
    def register_uploader_key(self, key: str) -> None:
        """초기화 시 정리할 업로더 위젯 키를 등록한다."""
        keys = list(self._get(self._UPLOADER_KEYS, []) or [])
        if key not in keys:
            keys.append(key)
        self._set(self._UPLOADER_KEYS, keys)

    # 무효화 ----------------------------------------------------------------
    def sync(
        self,
        *,
        inputs: dict | None = None,
        file_fp: str | None = None,
        bundle_id: str | None = None,
        bundle_hash: str | None = None,
    ) -> bool:
        """현재 입력·파일·묶음을 반영한다. 이전과 달라졌으면 결과·순위를 무효화한다.

        Returns:
            무효화가 일어났으면 True.
        """
        current = combined_fingerprint(
            inputs=inputs, file_fp=file_fp,
            bundle_id=bundle_id, bundle_hash=bundle_hash,
        )
        prev = self._get(self._FP_KEY)
        invalidated = should_invalidate_result(prev, current)
        if invalidated:
            self._set(self._RESULT_KEY, None)
            self._set(self._RANKING_KEY, None)
        self._set(self._FP_KEY, current)
        return invalidated

    # 하위 호환 (작업 20) -----------------------------------------------------
    def sync_inputs(self, current_inputs: dict) -> None:
        """작업 20 호환: 입력만으로 동기화한다."""
        self.sync(inputs=current_inputs)

    def start_request(self) -> None:
        """새 요청 시작 시 이전 결과·순위를 무효화한다 (REQ-14)."""
        self._set(self._RESULT_KEY, None)
        self._set(self._RANKING_KEY, None)

    def clear(self) -> None:
        """초기화: 입력 지문·결과·순위·요청 상태와 업로더 참조를 지운다 (REQ-14).

        업로더 위젯 키를 세션에서 제거하여 파일 업로더가 비워지도록 한다.
        재접속 이력 복원·다운로드 UI 는 만들지 않는다.
        """
        self._set(self._RESULT_KEY, None)
        self._set(self._RANKING_KEY, None)
        self._set(self._FP_KEY, None)
        self._set(self._STATE_KEY, RequestState.IDLE)
        for key in list(self._get(self._UPLOADER_KEYS, []) or []):
            self._del(key)
        self._set(self._UPLOADER_KEYS, [])
