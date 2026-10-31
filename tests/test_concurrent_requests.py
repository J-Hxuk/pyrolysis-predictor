"""
test_concurrent_requests.py — 작업 22 동시 요청·추론 잠금 검증.

묶음별 추론 잠금으로 동시 모델 접근을 직렬화하고, 대기 한도 초과는 재시도로
안내하며, 예외에서도 잠금이 풀리는지 확인한다 (환경 비의존, 스레드 기반).

확인 항목 (완료 기준):
1. 같은 묶음의 두 요청은 직렬화된다 (동시에 임계 구역에 들어가지 않는다).
2. 대기 한도 안에 잠금을 얻지 못하면 재시도(InferenceBusyError)로 응답한다.
3. 한 요청이 실패해도 잠금이 풀려 다음 요청을 처리할 수 있다.
4. 대기 한도 미설정(None)은 무제한 대기로 취급하지 않는다 (ValueError).
5. 다른 묶음은 서로 다른 잠금을 쓴다 (서로 막지 않는다).
"""

from __future__ import annotations

import threading
import time

import pytest

from pyrolysis.inference import (
    InferenceBusyError,
    run_with_inference_lock,
)


class TestSerialization:
    def test_same_bundle_requests_serialized(self):
        bundle_id = "bundle_serial"
        active = {"count": 0, "max": 0}
        lock = threading.Lock()

        def work():
            with lock:
                active["count"] += 1
                active["max"] = max(active["max"], active["count"])
            time.sleep(0.05)
            with lock:
                active["count"] -= 1
            return "ok"

        results = []

        def run():
            results.append(
                run_with_inference_lock(bundle_id, work, wait_timeout_seconds=5)
            )

        threads = [threading.Thread(target=run) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert results == ["ok"] * 4
        # 임계 구역에 동시에 하나만 들어갔다
        assert active["max"] == 1

    def test_different_bundles_dont_block(self):
        # 서로 다른 묶음은 다른 잠금 -> 동시에 진행 가능
        start = threading.Event()
        both_in = threading.Barrier(2, timeout=2)
        errors = []

        def work():
            start.wait()
            try:
                both_in.wait()  # 둘 다 임계 구역에 있어야 통과
            except threading.BrokenBarrierError:
                errors.append("blocked")
            return "ok"

        def run(bid):
            run_with_inference_lock(bid, work, wait_timeout_seconds=5)

        t1 = threading.Thread(target=run, args=("bundle_A",))
        t2 = threading.Thread(target=run, args=("bundle_B",))
        t1.start()
        t2.start()
        start.set()
        t1.join()
        t2.join()
        assert errors == []  # barrier 통과 -> 둘이 동시에 진행함


class TestTimeout:
    def test_wait_timeout_raises_busy(self):
        bundle_id = "bundle_timeout"
        holding = threading.Event()
        release = threading.Event()

        def holder():
            def work():
                holding.set()
                release.wait(timeout=2)
                return "held"
            run_with_inference_lock(bundle_id, work, wait_timeout_seconds=5)

        t = threading.Thread(target=holder)
        t.start()
        assert holding.wait(timeout=2)

        # 잠금을 쥔 동안 짧은 대기 한도로 요청하면 재시도 안내
        with pytest.raises(InferenceBusyError):
            run_with_inference_lock(bundle_id, lambda: "x", wait_timeout_seconds=0.1)

        release.set()
        t.join()

    def test_none_timeout_rejected(self):
        with pytest.raises(ValueError):
            run_with_inference_lock("b", lambda: "x", wait_timeout_seconds=None)


class TestFailureReleasesLock:
    def test_exception_releases_lock_for_next_request(self):
        bundle_id = "bundle_fail"

        def boom():
            raise RuntimeError("예측 실패")

        # 첫 요청 실패
        with pytest.raises(RuntimeError):
            run_with_inference_lock(bundle_id, boom, wait_timeout_seconds=5)

        # 실패 후에도 잠금이 풀려 다음 요청을 처리할 수 있다
        result = run_with_inference_lock(bundle_id, lambda: "recovered",
                                         wait_timeout_seconds=1)
        assert result == "recovered"
