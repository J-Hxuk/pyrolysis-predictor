"""
splitting.py — 독립 그룹을 유지하는 분할 계획 생성·검사 (작업 11).

두 평가 방식을 지원한다:
  - holdout     : 그룹 단위 고정 평가 자료 (개발/평가 그룹을 먼저 고정)
  - nested_cv   : 그룹 단위 중첩 교차검증 (외부 구간마다 내부에서만 선택)

핵심 규칙:
  - 분할 단위는 그룹 ID 다. 같은 그룹은 학습·평가 양쪽에 동시에 들어가지 않는다.
  - 외부 분할 역할은 대상 간 공통 그룹에서 유지한다. 대상별 정답 결측은 마스크로 분리하되
    그룹의 외부 역할(어느 외부 구간에 속하는지)은 바꾸지 않는다.
  - 외부 평가 자료를 내부 선택(모델·매개변수 선정)에 재사용하지 않는다.
  - 분할 비율·fold 수는 설정에서 받는다. 데이터 확인 전 기본값으로 확정하지 않는다.
  - 평가 방식·분할 단위·선정 기준이 없거나 독립 그룹을 구성할 수 없으면 해당 대상의
    평가를 중단하고 사유를 남긴다 (SplitUnavailable).

Streamlit을 import하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "SplitConfig",
    "OuterFold",
    "SplitPlan",
    "TargetSplit",
    "SplitUnavailable",
    "build_split_plan",
    "build_target_split",
]


class SplitError(Exception):
    """분할 설정이 구조적으로 잘못된 경우."""


@dataclass
class SplitConfig:
    """분할 설정. 실제 비율·fold 수는 호출자가 명시한다 (기본값 없음).

    method:
        "holdout"   -> eval_fraction 필요 (0<f<1)
        "nested_cv" -> n_outer_folds 필요 (>=2), n_inner_folds 선택
    group_column:
        분할 단위가 되는 그룹 식별자 열 이름. 없으면 분할 불가.
    selection_metric:
        내부 선정 기준 지표 이름 (예: "mae"). 없으면 분할 계획은 만들 수 있으나
        대상별 선정은 중단된다.
    """

    method: str
    group_column: str | None = None
    eval_fraction: float | None = None
    n_outer_folds: int | None = None
    n_inner_folds: int | None = None
    selection_metric: str | None = None
    random_seed: int | None = None

    def validate(self) -> None:
        if self.method not in ("holdout", "nested_cv"):
            raise SplitError("지원하지 않는 평가 방식입니다: " + repr(self.method))
        if not self.group_column:
            raise SplitError("분할 단위(group_column)가 지정되지 않았습니다.")
        if self.method == "holdout":
            if self.eval_fraction is None:
                raise SplitError("holdout 방식은 eval_fraction 이 필요합니다.")
            if not (0.0 < float(self.eval_fraction) < 1.0):
                raise SplitError("eval_fraction 은 0과 1 사이여야 합니다.")
        if self.method == "nested_cv":
            if self.n_outer_folds is None:
                raise SplitError("nested_cv 방식은 n_outer_folds 가 필요합니다.")
            if int(self.n_outer_folds) < 2:
                raise SplitError("n_outer_folds 는 2 이상이어야 합니다.")


@dataclass
class OuterFold:
    """외부 분할 하나. 그룹 ID 기준으로 학습/평가를 나눈다."""

    fold_id: int
    train_groups: list
    eval_groups: list


@dataclass
class SplitPlan:
    """대상 간 공통으로 유지되는 그룹 단위 외부 분할 계획.

    같은 그룹은 한 외부 fold 의 train 과 eval 양쪽에 동시에 들어가지 않는다.
    holdout 은 단일 fold(fold_id=0), nested_cv 는 n_outer_folds 개의 fold 로 구성한다.
    """

    method: str
    group_column: str
    all_groups: list
    outer_folds: list
    n_inner_folds: int | None = None
    selection_metric: str | None = None

    def eval_groups_of(self, fold_id: int) -> list:
        for f in self.outer_folds:
            if f.fold_id == fold_id:
                return list(f.eval_groups)
        raise KeyError("존재하지 않는 fold_id: " + str(fold_id))


@dataclass
class SplitUnavailable:
    """특정 대상의 분할·평가를 수행할 수 없는 상태와 사유."""

    target_id: str
    reason: str


@dataclass
class TargetSplit:
    """대상 하나에 적용한 분할. 공통 외부 분할 역할을 유지하되,

    정답이 있는(결측이 아닌) 행만 각 fold 의 train/eval 에 포함한다.
    그룹의 외부 역할(fold 배정)은 공통 SplitPlan 을 그대로 따른다.
    """

    target_id: str
    plan: SplitPlan
    # fold_id -> (train 행 위치 리스트, eval 행 위치 리스트)
    fold_row_indices: dict
    n_labeled: int


def _unique_groups_in_order(groups: list) -> list:
    seen = set()
    ordered = []
    for g in groups:
        if g not in seen:
            seen.add(g)
            ordered.append(g)
    return ordered


def build_split_plan(groups: Any, config: SplitConfig) -> SplitPlan:
    """그룹 식별자 열에서 대상 간 공통 외부 분할 계획을 만든다.

    Args:
        groups: 행별 그룹 ID (list/Series/array). 길이 = 전체 행 수.
        config: SplitConfig. 미리 validate() 되어야 한다.

    Returns:
        SplitPlan.

    Raises:
        SplitError: 설정이 구조적으로 잘못됐거나 독립 그룹이 부족한 경우.
    """
    config.validate()
    group_list = list(groups)
    if len(group_list) == 0:
        raise SplitError("행이 없어 분할 계획을 만들 수 없습니다.")
    if any(g is None or (isinstance(g, float) and np.isnan(g)) for g in group_list):
        raise SplitError("그룹 식별자에 결측이 있습니다. 독립 분할을 보장할 수 없습니다.")

    unique_groups = _unique_groups_in_order(group_list)
    n_groups = len(unique_groups)

    rng = np.random.default_rng(config.random_seed)
    shuffled = list(unique_groups)
    rng.shuffle(shuffled)

    outer_folds: list = []

    if config.method == "holdout":
        n_eval = int(round(n_groups * float(config.eval_fraction)))
        # 최소 각 1개 그룹씩은 양쪽에 있어야 독립 분할이 성립한다.
        if n_eval < 1 or n_eval >= n_groups:
            raise SplitError(
                "독립 그룹이 부족하여 holdout 분할을 만들 수 없습니다 "
                "(그룹 수 " + str(n_groups) + ", 평가 그룹 " + str(n_eval) + ")."
            )
        eval_groups = shuffled[:n_eval]
        train_groups = shuffled[n_eval:]
        outer_folds.append(OuterFold(fold_id=0, train_groups=train_groups, eval_groups=eval_groups))

    else:  # nested_cv
        k = int(config.n_outer_folds)
        if n_groups < k:
            raise SplitError(
                "독립 그룹이 부족하여 " + str(k) + "-겹 중첩 교차검증을 만들 수 없습니다 "
                "(그룹 수 " + str(n_groups) + ")."
            )
        # 그룹을 k 개의 외부 구간으로 분배
        buckets = [[] for _ in range(k)]
        for i, g in enumerate(shuffled):
            buckets[i % k].append(g)
        for fid in range(k):
            eval_groups = buckets[fid]
            train_groups = [g for g in shuffled if g not in set(eval_groups)]
            if not eval_groups or not train_groups:
                raise SplitError("중첩 교차검증의 한 구간에 그룹이 비어 분할할 수 없습니다.")
            outer_folds.append(OuterFold(fold_id=fid, train_groups=train_groups, eval_groups=eval_groups))

    return SplitPlan(
        method=config.method,
        group_column=config.group_column,
        all_groups=unique_groups,
        outer_folds=outer_folds,
        n_inner_folds=config.n_inner_folds,
        selection_metric=config.selection_metric,
    )


def build_target_split(
    target_id: str,
    plan: SplitPlan,
    groups: Any,
    label_mask: Any,
    config: SplitConfig,
):
    """공통 SplitPlan 을 특정 대상에 적용한다.

    - 정답이 있는 행(label_mask=True)만 각 fold 의 train/eval 에 포함한다.
    - 그룹의 외부 역할(fold 배정)은 공통 plan 을 그대로 따른다 (대상별 결측이 달라도
      같은 그룹은 같은 외부 구간에 남는다).
    - 선정 기준(selection_metric)이 없으면 TargetSplit 대신 SplitUnavailable 을 반환한다.

    Args:
        target_id: 대상 ID.
        plan: build_split_plan 결과.
        groups: 행별 그룹 ID (전체 행 기준, plan 생성 때와 같은 순서·길이).
        label_mask: 행별 정답 존재 여부 (True=정답 있음).
        config: SplitConfig.

    Returns:
        TargetSplit (정상) 또는 SplitUnavailable (수행 불가, 사유 포함).
    """
    if not config.selection_metric:
        return SplitUnavailable(
            target_id=target_id,
            reason="선정 기준(selection_metric)이 없어 내부 선정을 수행할 수 없습니다.",
        )

    group_list = list(groups)
    mask_list = [bool(m) for m in label_mask]
    if len(group_list) != len(mask_list):
        raise SplitError("groups 와 label_mask 의 길이가 다릅니다.")

    n_labeled = sum(mask_list)
    if n_labeled == 0:
        return SplitUnavailable(
            target_id=target_id,
            reason="정답 행이 없어 분할할 수 없습니다.",
        )

    fold_row_indices: dict = {}
    for f in plan.outer_folds:
        eval_set = set(f.eval_groups)
        train_set = set(f.train_groups)
        train_rows = []
        eval_rows = []
        for pos, (g, m) in enumerate(zip(group_list, mask_list)):
            if not m:
                continue  # 정답 결측 행은 이 대상 학습·평가에서 제외
            if g in eval_set:
                eval_rows.append(pos)
            elif g in train_set:
                train_rows.append(pos)
        # 한 fold 의 eval 에 이 대상의 정답 행이 하나도 없으면 평가 불가 사유를 남긴다.
        if not eval_rows:
            return SplitUnavailable(
                target_id=target_id,
                reason=("fold " + str(f.fold_id) + " 의 평가 그룹에 이 대상의 정답 행이 없습니다."),
            )
        if not train_rows:
            return SplitUnavailable(
                target_id=target_id,
                reason=("fold " + str(f.fold_id) + " 의 학습 그룹에 이 대상의 정답 행이 없습니다."),
            )
        fold_row_indices[f.fold_id] = (train_rows, eval_rows)

    return TargetSplit(
        target_id=target_id,
        plan=plan,
        fold_row_indices=fold_row_indices,
        n_labeled=n_labeled,
    )