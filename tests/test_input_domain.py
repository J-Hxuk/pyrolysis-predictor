"""
test_input_domain.py - Task 10 완료 기준 검증.

합성 자료로만 테스트한다. 실제 연구 변수·범위·임계값은 사용하지 않는다.

확인 항목:
1. 물리적 제약 위반은 오류(PHYSICAL_VIOLATION)다.
2. 학습 범위 밖이지만 유효한 후보는 통과하고 TRAINING_RANGE_EXCEEDED 주의가 남는다.
3. 미지원 범주(규격에 없음)는 UNKNOWN_CATEGORY 오류이며 유사 값으로 치환하지 않는다.
4. 학습에서 관측하지 못한 (규격상 허용된) 범주는 주의다.
5. 대상별 학습 표본(TrainingDomain)이 다르면 범위 판정도 달라진다.
6. 단일·파일 입력에 같은 validate_candidates 함수를 적용한다.
7. 모든 값이 범위 안이라는 이유로 조합의 신뢰성을 보장하지 않는다(주의 0개여도 추가 보장 없음).
"""

from __future__ import annotations

import pandas as pd
import pytest

from pyrolysis.contracts import (
    AllowedCategory,
    InputSpec,
    NumericRange,
    PhysicalConstraint,
    TrainingDomain,
)
from pyrolysis.validation import (
    check_physical_constraints,
    check_training_range,
    validate_candidates,
)


def _num_spec(id, pmin=None, pmax=None):
    pcs = []
    if pmin is not None or pmax is not None:
        pcs = [PhysicalConstraint(description="physical", min_value=pmin, max_value=pmax)]
    return InputSpec(
        id=id, label_ko=id, description=id, role="feedstock",
        dtype="float", required=True, physical_constraints=pcs,
    )


def _cat_spec(id, cats):
    return InputSpec(
        id=id, label_ko=id, description=id, role="feedstock",
        dtype="category", required=True,
        allowed_categories=[AllowedCategory(value=c, label_ko=c) for c in cats],
    )


class TestPhysicalConstraints:
    def test_below_min_is_error(self):
        specs = [_num_spec("x", pmin=0.0, pmax=100.0)]
        df = pd.DataFrame([{"x": -5.0}])
        errs = check_physical_constraints(df, specs)
        assert any(e.code == "PHYSICAL_VIOLATION" for e in errs)
        assert all(e.severity == "error" for e in errs)

    def test_above_max_is_error(self):
        specs = [_num_spec("x", pmin=0.0, pmax=100.0)]
        df = pd.DataFrame([{"x": 150.0}])
        errs = check_physical_constraints(df, specs)
        assert any(e.code == "PHYSICAL_VIOLATION" for e in errs)

    def test_within_physical_no_error(self):
        specs = [_num_spec("x", pmin=0.0, pmax=100.0)]
        df = pd.DataFrame([{"x": 50.0}])
        errs = check_physical_constraints(df, specs)
        assert errs == []

    def test_no_physical_constraint_no_error(self):
        specs = [_num_spec("x")]
        df = pd.DataFrame([{"x": 999999.0}])
        errs = check_physical_constraints(df, specs)
        assert errs == []


class TestTrainingRangeIsWarning:
    def test_out_of_range_numeric_is_warning_not_error(self):
        specs = [_num_spec("x", pmin=0.0, pmax=1000.0)]
        # 물리 범위 안(0~1000) 이지만 학습 범위(10~20) 밖
        df = pd.DataFrame([{"x": 500.0}])
        td = TrainingDomain(target_id="y1", numeric_ranges={"x": NumericRange(min_value=10.0, max_value=20.0)})
        warns = check_training_range(df, specs, td)
        assert any(w.code == "TRAINING_RANGE_EXCEEDED" for w in warns)
        assert all(w.severity == "warning" for w in warns)
        assert all(w.target_id == "y1" for w in warns)
        # 물리 제약은 위반 아님 -> 오류 없음
        assert check_physical_constraints(df, specs) == []

    def test_within_training_range_no_warning(self):
        specs = [_num_spec("x")]
        df = pd.DataFrame([{"x": 15.0}])
        td = TrainingDomain(target_id="y1", numeric_ranges={"x": NumericRange(min_value=10.0, max_value=20.0)})
        warns = check_training_range(df, specs, td)
        assert warns == []

    def test_unobserved_category_is_warning(self):
        specs = [_cat_spec("c", ["a", "b", "c"])]
        df = pd.DataFrame([{"c": "c"}])
        # 규격엔 c 허용되지만 학습에서 a,b 만 관측
        td = TrainingDomain(target_id="y1", supported_categories={"c": ["a", "b"]})
        warns = check_training_range(df, specs, td)
        assert any(w.code == "TRAINING_RANGE_EXCEEDED" for w in warns)
        assert all(w.severity == "warning" for w in warns)


class TestUnknownCategoryIsError:
    def test_unknown_category_is_error_not_substituted(self):
        specs = [_cat_spec("c", ["a", "b"])]
        df = pd.DataFrame([{"c": "zzz"}])
        res = validate_candidates(df, specs)
        assert any(e.code == "UNKNOWN_CATEGORY" for e in res["errors"])
        # 치환하지 않음: 원본 값이 그대로 남아 있다
        assert df.iloc[0]["c"] == "zzz"


class TestPerTargetDomainDiffers:
    def test_different_domain_different_verdict(self):
        specs = [_num_spec("x")]
        df = pd.DataFrame([{"x": 25.0}])
        td_narrow = TrainingDomain(target_id="y1", numeric_ranges={"x": NumericRange(min_value=10.0, max_value=20.0)})
        td_wide = TrainingDomain(target_id="y2", numeric_ranges={"x": NumericRange(min_value=0.0, max_value=100.0)})
        warns_narrow = check_training_range(df, specs, td_narrow)
        warns_wide = check_training_range(df, specs, td_wide)
        # 좁은 범위(y1)에서는 이탈 주의, 넓은 범위(y2)에서는 주의 없음
        assert len(warns_narrow) == 1
        assert warns_narrow[0].target_id == "y1"
        assert warns_wide == []

    def test_validate_candidates_collects_per_target_warnings(self):
        specs = [_num_spec("x")]
        df = pd.DataFrame([{"x": 25.0}])
        td1 = TrainingDomain(target_id="y1", numeric_ranges={"x": NumericRange(min_value=10.0, max_value=20.0)})
        td2 = TrainingDomain(target_id="y2", numeric_ranges={"x": NumericRange(min_value=0.0, max_value=5.0)})
        res = validate_candidates(df, specs, training_domains=[td1, td2])
        tgts = {w.target_id for w in res["range_warnings"]}
        assert tgts == {"y1", "y2"}
        assert res["errors"] == []


class TestSingleAndFileSameFunction:
    def test_single_row_and_multi_row_same_function(self):
        specs = [_num_spec("x", pmin=0.0, pmax=100.0)]
        single = pd.DataFrame([{"x": -1.0}])
        multi = pd.DataFrame([{"x": 10.0}, {"x": -1.0}, {"x": 20.0}])
        res_single = validate_candidates(single, specs)
        res_multi = validate_candidates(multi, specs)
        # 둘 다 같은 함수로 PHYSICAL_VIOLATION 을 찾는다
        assert any(e.code == "PHYSICAL_VIOLATION" for e in res_single["errors"])
        # 다중 행에서 두 번째 행(row=2)만 위반
        viol_rows = [e.row for e in res_multi["errors"] if e.code == "PHYSICAL_VIOLATION"]
        assert viol_rows == [2]


class TestAllWithinDoesNotGuarantee:
    def test_all_within_range_produces_no_warning_but_no_extra_guarantee(self):
        # 모든 값이 물리·학습 범위 안이면 오류·주의가 없다.
        # 그러나 이는 조합의 신뢰성 보장이 아니라 '표시할 문제가 없음' 일 뿐이다.
        # 검증 결과에 신뢰성·검증완료 플래그가 포함되지 않음을 확인한다.
        specs = [_num_spec("x", pmin=0.0, pmax=100.0), _num_spec("y", pmin=0.0, pmax=100.0)]
        df = pd.DataFrame([{"x": 50.0, "y": 50.0}])
        td = TrainingDomain(target_id="t", numeric_ranges={
            "x": NumericRange(min_value=0.0, max_value=100.0),
            "y": NumericRange(min_value=0.0, max_value=100.0),
        })
        res = validate_candidates(df, specs, training_domains=[td])
        assert res["errors"] == []
        assert res["range_warnings"] == []
        # 반환 구조에 '검증됨/신뢰함' 같은 보장 키가 없다
        assert set(res.keys()) == {"errors", "range_warnings"}