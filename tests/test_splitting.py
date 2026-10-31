"""
test_splitting.py - Task 11 완료 기준 검증.

합성 자료로만 테스트한다. 실제 분할 비율·fold 수는 테스트 설정으로만 지정한다.

확인 항목:
1. 그룹 누출 방지: 같은 그룹이 한 fold 의 train 과 eval 양쪽에 동시에 들어가지 않는다.
2. 외부 평가를 내부 선택에 재사용하지 않는다 (각 fold eval 과 train 그룹이 분리).
3. 대상별 정답 결측이 달라도 그룹의 외부 역할(fold 배정)은 유지된다.
4. 설정 누락·그룹 부족이면 SplitError 또는 SplitUnavailable(사유 포함)로 처리한다.
5. 비율·fold 수는 설정에서 받는다 (method 만으로는 검증 실패).
"""

from __future__ import annotations

import pytest

from pyrolysis.splitting import (
    SplitConfig,
    SplitError,
    SplitPlan,
    SplitUnavailable,
    TargetSplit,
    build_split_plan,
    build_target_split,
)


# 그룹 6개, 각 그룹에 2행씩 (반복·파생 시료 의존성 모사): 총 12행
_GROUPS = ["g1", "g1", "g2", "g2", "g3", "g3", "g4", "g4", "g5", "g5", "g6", "g6"]


def _all_true(n):
    return [True] * n


class TestConfigValidation:
    def test_method_only_fails(self):
        # 비율·fold 수 없이 method 만으로는 검증 실패 (기본값 확정 안 함)
        with pytest.raises(SplitError):
            SplitConfig(method="holdout", group_column="grp").validate()
        with pytest.raises(SplitError):
            SplitConfig(method="nested_cv", group_column="grp").validate()

    def test_missing_group_column_fails(self):
        with pytest.raises(SplitError):
            SplitConfig(method="holdout", eval_fraction=0.3).validate()

    def test_unknown_method_fails(self):
        with pytest.raises(SplitError):
            SplitConfig(method="bogus", group_column="grp").validate()

    def test_eval_fraction_bounds(self):
        with pytest.raises(SplitError):
            SplitConfig(method="holdout", group_column="grp", eval_fraction=1.5).validate()

    def test_nested_min_folds(self):
        with pytest.raises(SplitError):
            SplitConfig(method="nested_cv", group_column="grp", n_outer_folds=1).validate()


class TestHoldoutGroupLeakage:
    def _plan(self):
        cfg = SplitConfig(method="holdout", group_column="grp",
                          eval_fraction=0.34, selection_metric="mae", random_seed=1)
        return build_split_plan(_GROUPS, cfg), cfg

    def test_no_group_in_both_sides(self):
        plan, _ = self._plan()
        f = plan.outer_folds[0]
        assert set(f.train_groups).isdisjoint(set(f.eval_groups))

    def test_all_groups_covered(self):
        plan, _ = self._plan()
        f = plan.outer_folds[0]
        assert set(f.train_groups) | set(f.eval_groups) == set(["g1","g2","g3","g4","g5","g6"])

    def test_single_fold(self):
        plan, _ = self._plan()
        assert len(plan.outer_folds) == 1

    def test_target_rows_do_not_leak_groups(self):
        plan, cfg = self._plan()
        ts = build_target_split("y1", plan, _GROUPS, _all_true(len(_GROUPS)), cfg)
        assert isinstance(ts, TargetSplit)
        train_rows, eval_rows = ts.fold_row_indices[0]
        train_groups = {_GROUPS[i] for i in train_rows}
        eval_groups = {_GROUPS[i] for i in eval_rows}
        # 행 수준에서도 그룹이 섞이지 않는다
        assert train_groups.isdisjoint(eval_groups)
        # 행 자체도 겹치지 않는다
        assert set(train_rows).isdisjoint(set(eval_rows))


class TestNestedCvGroupLeakage:
    def _plan(self):
        cfg = SplitConfig(method="nested_cv", group_column="grp",
                          n_outer_folds=3, n_inner_folds=2,
                          selection_metric="mae", random_seed=2)
        return build_split_plan(_GROUPS, cfg), cfg

    def test_each_fold_no_leak(self):
        plan, _ = self._plan()
        assert len(plan.outer_folds) == 3
        for f in plan.outer_folds:
            assert set(f.train_groups).isdisjoint(set(f.eval_groups))

    def test_eval_groups_partition_all(self):
        # 외부 평가 그룹들의 합집합이 전체 그룹 (각 그룹이 정확히 한 번 외부 평가)
        plan, _ = self._plan()
        eval_union = []
        for f in plan.outer_folds:
            eval_union.extend(f.eval_groups)
        assert sorted(eval_union) == sorted(set(["g1","g2","g3","g4","g5","g6"]))
        # 중복 없음 = 외부 평가를 여러 fold 가 공유하지 않음
        assert len(eval_union) == len(set(eval_union))

    def test_eval_not_reused_as_train_in_same_fold(self):
        plan, _ = self._plan()
        for f in plan.outer_folds:
            for g in f.eval_groups:
                assert g not in f.train_groups


class TestPerTargetRoleStable:
    def test_different_missing_same_group_roles(self):
        # 두 대상이 서로 다른 결측 패턴을 가져도 그룹의 fold 배정은 공통 plan 을 따른다.
        cfg = SplitConfig(method="nested_cv", group_column="grp",
                          n_outer_folds=3, selection_metric="mae", random_seed=3)
        plan = build_split_plan(_GROUPS, cfg)

        n = len(_GROUPS)
        mask_a = _all_true(n)
        mask_b = _all_true(n)
        mask_b[0] = False  # y2 는 첫 행(그룹 g1) 결측

        ts_a = build_target_split("y1", plan, _GROUPS, mask_a, cfg)
        ts_b = build_target_split("y2", plan, _GROUPS, mask_b, cfg)
        assert isinstance(ts_a, TargetSplit)
        assert isinstance(ts_b, TargetSplit)

        # 각 fold 에서 그룹의 외부 역할(eval 그룹 집합)이 두 대상 모두 동일해야 한다
        for f in plan.outer_folds:
            fid = f.fold_id
            a_eval_groups = {_GROUPS[i] for i in ts_a.fold_row_indices[fid][1]}
            b_eval_groups = {_GROUPS[i] for i in ts_b.fold_row_indices[fid][1]}
            # g1 이 평가 그룹인 fold 에서는 y2 의 g1 행 1개가 빠질 뿐,
            # 평가 그룹 집합 자체(어느 그룹이 평가인지)는 같아야 한다.
            # (g1 에는 2행이 있으므로 1행 결측돼도 그룹은 남는다)
            assert a_eval_groups == b_eval_groups


class TestUnavailableReasons:
    def test_missing_selection_metric_unavailable(self):
        cfg = SplitConfig(method="holdout", group_column="grp", eval_fraction=0.34)
        plan = build_split_plan(_GROUPS, cfg)
        res = build_target_split("y1", plan, _GROUPS, _all_true(len(_GROUPS)), cfg)
        assert isinstance(res, SplitUnavailable)
        assert "선정 기준" in res.reason

    def test_all_missing_labels_unavailable(self):
        cfg = SplitConfig(method="holdout", group_column="grp",
                          eval_fraction=0.34, selection_metric="mae")
        plan = build_split_plan(_GROUPS, cfg)
        res = build_target_split("y1", plan, _GROUPS, [False]*len(_GROUPS), cfg)
        assert isinstance(res, SplitUnavailable)

    def test_insufficient_groups_holdout(self):
        cfg = SplitConfig(method="holdout", group_column="grp",
                          eval_fraction=0.34, selection_metric="mae")
        with pytest.raises(SplitError):
            build_split_plan(["only_one", "only_one"], cfg)

    def test_insufficient_groups_nested(self):
        cfg = SplitConfig(method="nested_cv", group_column="grp",
                          n_outer_folds=5, selection_metric="mae")
        # 그룹 2개로 5겹 불가
        with pytest.raises(SplitError):
            build_split_plan(["g1","g1","g2","g2"], cfg)

    def test_group_missing_value_rejected(self):
        cfg = SplitConfig(method="holdout", group_column="grp",
                          eval_fraction=0.34, selection_metric="mae")
        with pytest.raises(SplitError):
            build_split_plan(["g1", None, "g2", "g3"], cfg)

    def test_target_with_label_only_in_one_outer_group(self):
        # 특정 대상의 정답이 한 외부 구간에만 있으면 다른 fold 의 eval 에 정답이 없어 Unavailable
        cfg = SplitConfig(method="nested_cv", group_column="grp",
                          n_outer_folds=3, selection_metric="mae", random_seed=3)
        plan = build_split_plan(_GROUPS, cfg)
        n = len(_GROUPS)
        # g1 행(인덱스 0,1)만 정답, 나머지 결측
        mask = [False]*n
        mask[0] = True
        mask[1] = True
        res = build_target_split("y_sparse", plan, _GROUPS, mask, cfg)
        assert isinstance(res, SplitUnavailable)


class TestReproducibility:
    def test_same_seed_same_plan(self):
        cfg = SplitConfig(method="nested_cv", group_column="grp",
                          n_outer_folds=3, selection_metric="mae", random_seed=7)
        p1 = build_split_plan(_GROUPS, cfg)
        p2 = build_split_plan(_GROUPS, cfg)
        e1 = [sorted(f.eval_groups) for f in p1.outer_folds]
        e2 = [sorted(f.eval_groups) for f in p2.outer_folds]
        assert e1 == e2