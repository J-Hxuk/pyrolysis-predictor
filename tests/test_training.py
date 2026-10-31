"""
test_training.py - Task 13 완료 기준 검증 (선정 로직 중심).

기존 test_demo_training.py 의 재현성·분리·지표 검증은 중복하지 않는다.
여기서는 작업 13에서 추가한 내부 비교·선정 규칙만 검증한다.

합성 자료로만 테스트한다. 실제 선정 지표·방향·조정 범위는 테스트 설정으로만 지정한다.
"""

from __future__ import annotations

import pytest

from pyrolysis.training import (
    CandidateScore,
    SelectionConfig,
    SelectionError,
    config_hash,
    select_candidate,
)


class TestSelectionConfig:
    def test_missing_metric_rejected(self):
        with pytest.raises(SelectionError):
            SelectionConfig(metric="", direction="minimize").validate()

    def test_bad_direction_rejected(self):
        with pytest.raises(SelectionError):
            SelectionConfig(metric="mae", direction="smallest").validate()

    def test_valid_config(self):
        SelectionConfig(metric="mae", direction="minimize").validate()  # no raise


class TestSelectCandidate:
    def test_minimize_picks_lowest(self):
        cfg = SelectionConfig(metric="mae", direction="minimize")
        scores = [
            CandidateScore("a", 0.5, True),
            CandidateScore("b", 0.2, True),
            CandidateScore("c", 0.9, True),
        ]
        res = select_candidate("y1", scores, cfg)
        assert res.selected == "b"

    def test_maximize_picks_highest(self):
        cfg = SelectionConfig(metric="f1", direction="maximize")
        scores = [
            CandidateScore("a", 0.5, True),
            CandidateScore("b", 0.8, True),
        ]
        res = select_candidate("y1", scores, cfg)
        assert res.selected == "b"

    def test_not_run_candidate_gets_no_score(self):
        # ran=False 후보는 선정 대상에서 제외된다 (점수를 만들지 않음)
        cfg = SelectionConfig(metric="mae", direction="minimize")
        scores = [
            CandidateScore("a", None, False, reason="두 클래스 부족"),
            CandidateScore("b", 0.3, True),
        ]
        res = select_candidate("y1", scores, cfg)
        assert res.selected == "b"

    def test_no_valid_scores_returns_none_with_reason(self):
        cfg = SelectionConfig(metric="mae", direction="minimize")
        scores = [
            CandidateScore("a", None, False, reason="실행 실패"),
            CandidateScore("b", None, False, reason="실행 실패"),
        ]
        res = select_candidate("y1", scores, cfg)
        assert res.selected is None
        assert res.reason

    def test_tie_broken_by_tie_breaker(self):
        cfg = SelectionConfig(metric="mae", direction="minimize",
                              tie_breaker=["c", "a", "b"])
        scores = [
            CandidateScore("a", 0.3, True),
            CandidateScore("b", 0.3, True),
            CandidateScore("c", 0.3, True),
        ]
        res = select_candidate("y1", scores, cfg)
        assert res.selected == "c"  # tie_breaker 우선순위

    def test_tie_broken_deterministically_without_tie_breaker(self):
        cfg = SelectionConfig(metric="mae", direction="minimize")
        scores = [
            CandidateScore("b", 0.3, True),
            CandidateScore("a", 0.3, True),
        ]
        res = select_candidate("y1", scores, cfg)
        assert res.selected == "a"  # 사전순 결정적

    def test_selection_is_reproducible(self):
        cfg = SelectionConfig(metric="mae", direction="minimize")
        scores = [CandidateScore("a", 0.3, True), CandidateScore("b", 0.1, True)]
        r1 = select_candidate("y1", list(scores), cfg)
        r2 = select_candidate("y1", list(scores), cfg)
        assert r1.selected == r2.selected == "b"


class TestConfigHash:
    def test_same_input_same_hash(self):
        a = config_hash({"metric": "mae", "seed": 1})
        b = config_hash({"seed": 1, "metric": "mae"})  # 키 순서 무관
        assert a == b
        assert len(a) == 16

    def test_different_input_different_hash(self):
        assert config_hash({"metric": "mae"}) != config_hash({"metric": "f1"})