"""
ranking.py — 수치형 지표 하나의 후보 순위 계산 (작업 19).

기준: maximize(내림차순), minimize(오름차순), target(목표값 절대 차이 오름차순)
동점 처리: 공동 순위 (1, 1, 3). 동점 표시 순서는 원본 행 순서를 유지한다.
원래 정밀도로 정렬한다 (반올림 값으로 정렬하지 않는다).
분류·확률·미지원 대상은 순위 후보에서 제외한다.
기준만 바뀌면 모델 재실행 없이 완료된 결과를 재정렬한다 (PredictionBatch 를 입력으로 받음).
Streamlit을 import하지 않는다.
"""

from __future__ import annotations

import math
from typing import Any

from pyrolysis.contracts import PredictionBatch, RankedCandidate, RankingResult

__all__ = [
    "RankingError",
    "rank_candidates",
]


class RankingError(Exception):
    """순위 계산 불가 (분류 대상·목표값 누락·비유한 목표값 등)."""


def rank_candidates(
    batch: PredictionBatch,
    target_id: str,
    direction: str,
    *,
    target_value: float | None = None,
    source_rows: list | None = None,
    candidate_ids: list | None = None,
    range_flags: list | None = None,
) -> RankingResult:
    """완료된 PredictionBatch 에서 한 수치형 대상의 후보 순위를 계산한다.

    - batch.completed=False 이면 순위를 만들지 않는다 (요청 전체 실패).
    - 대상이 batch 에 없거나 값이 수치가 아니면(분류 표지) 거부한다.
    - target 기준은 유한한 target_value 가 필요하다. 없거나 비유한이면 거부.
    - 유효한 학습 범위 밖 후보를 제외·감점하지 않는다 (range_flags 는 표시만).
    - 원래 정밀도로 정렬하고 같은 정렬값은 공동 순위로 처리한다.

    Args:
        batch: predict_all_targets 결과 (completed=True).
        target_id: 순위 대상 ID (수치형이어야 함).
        direction: "maximize" | "minimize" | "target".
        target_value: target 기준의 목표값 (같은 단위, 유한).
        source_rows: 후보별 원본 행 번호 (없으면 1..n).
        candidate_ids: 후보 식별자 (없으면 자동).
        range_flags: 후보별 학습 범위 이탈 표시 (없으면 모두 False).

    Returns:
        RankingResult.

    Raises:
        RankingError: 미완료·분류 대상·목표값 누락/비유한·대상 없음.
    """
    if direction not in ("maximize", "minimize", "target"):
        raise RankingError("지원하지 않는 순위 기준입니다: " + repr(direction))

    if not batch.completed:
        raise RankingError("완료되지 않은 요청에는 순위를 만들지 않습니다.")

    if target_id not in batch.predictions:
        raise RankingError("대상이 결과에 없습니다: " + repr(target_id))

    tp = batch.predictions[target_id]
    raw_values = tp.values

    # 분류 표지(문자열) 또는 비수치는 순위 대상에서 제외
    numeric_values = []
    for v in raw_values:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise RankingError(
                "수치형 대상만 순위를 매길 수 있습니다. 분류·표지·확률은 제외됩니다: "
                + repr(target_id)
            )
        if not math.isfinite(float(v)):
            raise RankingError("예측값에 비유한 값이 있어 순위를 만들 수 없습니다.")
        numeric_values.append(float(v))

    n = len(numeric_values)
    if source_rows is None:
        source_rows = list(range(1, n + 1))
    if candidate_ids is None:
        candidate_ids = ["c" + str(i) for i in range(n)]
    if range_flags is None:
        range_flags = [False] * n
    if not (len(source_rows) == len(candidate_ids) == len(range_flags) == n):
        raise RankingError("메타데이터 길이가 예측값 수와 다릅니다.")

    # target 기준 유효성
    if direction == "target":
        if target_value is None or not math.isfinite(float(target_value)):
            raise RankingError(
                "목표값 접근에는 유한한 목표값이 필요합니다. 수정할 항목을 표시하세요."
            )

    # 정렬 값 계산 (원래 정밀도 유지)
    def sort_value_of(v: float) -> float:
        if direction == "target":
            return abs(v - float(target_value))
        return v

    entries = []
    for i in range(n):
        entries.append({
            "candidate_id": candidate_ids[i],
            "source_row": source_rows[i],
            "value": numeric_values[i],
            "sort_value": sort_value_of(numeric_values[i]),
            "range_exceeded": bool(range_flags[i]),
            "orig_index": i,  # 동점 시 원본 순서 유지
        })

    reverse = (direction == "maximize")
    # 1차: sort_value, 2차: 원본 인덱스(안정) -> 동점 표시 순서 원본 유지
    entries_sorted = sorted(entries, key=lambda e: (e["sort_value"], e["orig_index"]),
                            reverse=False)
    if reverse:
        # maximize: sort_value 내림차순이되 동점은 원본 순서 유지
        entries_sorted = sorted(entries, key=lambda e: (-e["sort_value"], e["orig_index"]))

    # 공동 순위 (1,1,3) 부여
    ranked = []
    prev_sort = None
    prev_rank = 0
    for pos, e in enumerate(entries_sorted, start=1):
        sv = e["sort_value"]
        if prev_sort is not None and sv == prev_sort:
            rank = prev_rank  # 공동 순위
        else:
            rank = pos
        prev_sort = sv
        prev_rank = rank
        ranked.append(RankedCandidate(
            candidate_id=e["candidate_id"],
            source_row=e["source_row"],
            value=e["value"],
            sort_value=e["sort_value"],
            rank=rank,
            range_exceeded=e["range_exceeded"],
        ))

    return RankingResult(
        request_id=batch.request_id,
        target_id=target_id,
        unit=tp.unit,
        direction=direction,
        target_value=float(target_value) if target_value is not None else None,
        ranked=ranked,
    )