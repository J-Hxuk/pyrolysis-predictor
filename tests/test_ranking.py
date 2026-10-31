"""
test_ranking.py - Task 19 완료 기준 검증.

합성 자료로만 테스트한다.

확인 항목:
1. maximize/minimize/target 세 기준이 올바르게 정렬된다.
2. 동점은 공동 순위 (1,1,3), 표시 순서는 원본 행 순서를 유지한다.
3. 목표값 누락·비유한이면 순위를 만들지 않는다 (RankingError).
4. 분류·비수치 대상은 순위에서 제외된다 (RankingError).
5. 반올림 값으로 정렬하지 않는다 (미세 차이도 순위에 반영).
6. 학습 범위 밖 후보를 제외·감점하지 않는다 (표시만).
7. 기준만 바뀌면 같은 batch 를 재정렬한다 (모델 재실행 없음).
"""

from __future__ import annotations

import pytest

from pyrolysis.contracts import PredictionBatch, TargetPrediction
from pyrolysis.ranking import RankingError, rank_candidates


def _batch(values, target_id="y", unit="wt%", completed=True):
    return PredictionBatch(
        request_id="r1", bundle_id="b1", bundle_hash="h", input_snapshot_hash="s",
        predictions={target_id: TargetPrediction(target_id=target_id, values=values, unit=unit)},
        completed=completed,
    )


class TestDirections:
    def test_maximize(self):
        b = _batch([1.0, 3.0, 2.0])
        r = rank_candidates(b, "y", "maximize")
        assert [c.rank for c in r.ranked] == [1, 2, 3]
        assert r.ranked[0].value == 3.0

    def test_minimize(self):
        b = _batch([1.0, 3.0, 2.0])
        r = rank_candidates(b, "y", "minimize")
        assert r.ranked[0].value == 1.0
        assert r.ranked[-1].value == 3.0

    def test_target(self):
        b = _batch([1.0, 5.0, 9.0])
        r = rank_candidates(b, "y", "target", target_value=5.0)
        assert r.ranked[0].value == 5.0  # 절대 차이 0 이 1위


class TestTies:
    def test_tie_shared_rank_1_1_3(self):
        b = _batch([5.0, 5.0, 1.0])
        r = rank_candidates(b, "y", "maximize")
        ranks = [c.rank for c in r.ranked]
        assert ranks == [1, 1, 3]

    def test_tie_display_order_is_source_order(self):
        # 동점 두 후보의 표시 순서가 원본 행 순서를 유지한다
        b = _batch([5.0, 5.0, 1.0])
        r = rank_candidates(b, "y", "maximize",
                            source_rows=[10, 20, 30], candidate_ids=["a", "b", "c"])
        # 공동 1위 두 후보는 원본 순서(a 먼저, b 다음)
        top_two = [c.candidate_id for c in r.ranked if c.rank == 1]
        assert top_two == ["a", "b"]


class TestInvalidTarget:
    def test_missing_target_value_rejected(self):
        b = _batch([1.0, 2.0])
        with pytest.raises(RankingError):
            rank_candidates(b, "y", "target", target_value=None)

    def test_non_finite_target_value_rejected(self):
        b = _batch([1.0, 2.0])
        with pytest.raises(RankingError):
            rank_candidates(b, "y", "target", target_value=float("inf"))


class TestClassificationExcluded:
    def test_label_values_rejected(self):
        b = _batch(["present", "absent"])  # 분류 표지
        with pytest.raises(RankingError):
            rank_candidates(b, "y", "maximize")


class TestPrecision:
    def test_sorts_by_full_precision_not_rounded(self):
        # 반올림하면 같아 보이는 두 값도 원래 정밀도로 구분해 순위를 매긴다
        b = _batch([1.00001, 1.00002, 1.00000])
        r = rank_candidates(b, "y", "maximize")
        # 가장 큰 1.00002 가 1위, 1.00000 이 꼴찌
        assert r.ranked[0].value == 1.00002
        assert r.ranked[-1].value == 1.00000
        # 동점이 아니므로 공동 순위가 아니다
        assert [c.rank for c in r.ranked] == [1, 2, 3]


class TestRangeExceededKept:
    def test_out_of_range_candidate_not_excluded_or_penalized(self):
        b = _batch([10.0, 2.0])
        r = rank_candidates(b, "y", "maximize", range_flags=[True, False])
        # 범위 밖(첫 후보)이 값이 커서 1위 그대로. 제외·감점 없음
        assert r.ranked[0].value == 10.0
        assert r.ranked[0].rank == 1
        assert r.ranked[0].range_exceeded is True
        assert len(r.ranked) == 2  # 제외되지 않음


class TestRerankWithoutRerun:
    def test_same_batch_rerank_by_changing_direction(self):
        b = _batch([1.0, 3.0, 2.0])
        r_max = rank_candidates(b, "y", "maximize")
        r_min = rank_candidates(b, "y", "minimize")
        # 같은 batch 로 기준만 바꿔 다른 순위를 얻는다 (모델 재실행 없음)
        assert r_max.ranked[0].value == 3.0
        assert r_min.ranked[0].value == 1.0


class TestIncompleteBatch:
    def test_incomplete_batch_rejected(self):
        b = _batch([1.0, 2.0], completed=False)
        with pytest.raises(RankingError):
            rank_candidates(b, "y", "maximize")

    def test_unknown_target_rejected(self):
        b = _batch([1.0, 2.0], target_id="y")
        with pytest.raises(RankingError):
            rank_candidates(b, "other", "maximize")