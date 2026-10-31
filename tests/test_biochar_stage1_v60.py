"""
test_biochar_stage1_v60.py — Stage 1 v60 통합 모듈 검증.

환경 인식(Environment-aware): v60 모델은 sklearn 1.8 / xgboost 3.4 로 저장되었다. 현재
프로젝트 venv(sklearn 1.5.2 / xgboost 2.1.3)에서는 예측값이 손상되므로 호환성 게이트가
차단해야 한다. 따라서 수치 예측 검증은 호환 환경에서만 수행하고, 비호환 환경에서는
"차단되는지"를 검증한다. 어느 경우든 손상 값을 정상으로 간주하지 않는다.

검증 항목:
  - 패키지 추출·로드
  - 입력 계약(제조조건)·대상(BET/TPV) 명세
  - 원료 조성 변수가 모델 추정기에서 drop 되는지(사용자 요청)
  - 호환성 게이트: 비호환이면 차단, 호환이면 예제 예측 재현 일치
  - Stage 2 인터페이스가 v60 예측값을 측정값과 구분해 수용(연결·학습 없음)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyrolysis.biochar_stage1_v60 import compat, features, predictor
from pyrolysis.biochar_stage1_v60.extract import extract_v60_package, resolve_v60_final_dir

_REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def final_dir():
    return extract_v60_package(_REPO)


@pytest.fixture(scope="module")
def handle():
    return predictor.load_v60(_REPO)


# ── 추출·계약 ────────────────────────────────────────────────────────────────


def test_package_extracts_core_files(final_dir):
    for name in [
        "predict_stage1_v60.py",
        "MODEL_STAGE1_V60_BET_XGB_Eng59.joblib",
        "MODEL_STAGE1_V60_TPV_XGB_Eng59.joblib",
        "01_Stage1_Master_v60_256.csv",
        "FINAL_STATUS.json",
    ]:
        assert (final_dir / name).is_file(), f"누락: {name}"


def test_input_contract_fields():
    ids = [f["id"] for f in features.V60_PRIMARY_INPUT_FIELDS]
    for required in ["feedstock", "pyrolysis_temperature_C", "activation_method",
                     "activation_agent", "activation_temperature_C"]:
        assert required in ids
    tids = [t["id"] for t in features.V60_TARGETS]
    assert tids == ["BET", "TPV"]
    units = {t["id"]: t["unit"] for t in features.V60_TARGETS}
    assert units["BET"] == "m2/g" and units["TPV"] == "cm3/g"


# ── 원료 조성 변수 사용 여부 (사용자 요청 검증) ──────────────────────────────


def test_feedstock_composition_dropped_from_estimator(final_dir):
    """원료 조성 변수가 모델 추정기에 전달되지 않는지(전처리 drop) 직접 확인한다."""
    import joblib

    comp_cols = {
        "feedstock_C_pct_original", "feedstock_H_pct_original", "feedstock_O_pct_original",
        "feedstock_N_pct_original", "feedstock_ash_pct_original", "feedstock_VM_pct_original",
        "feedstock_FC_pct_original", "feedstock_moisture_pct_original",
    }
    for target in ["BET", "TPV"]:
        obj = joblib.load(final_dir / f"MODEL_STAGE1_V60_{target}_XGB_Eng59.joblib")
        pipe = obj["router"]["global"]
        ct = None
        for _, step in pipe.named_steps.items():
            if step.__class__.__name__ == "ColumnTransformer":
                ct = step
        assert ct is not None, "ColumnTransformer 를 찾을 수 없습니다."
        # remainder 는 drop 이어야 한다 (나머지 열은 추정기로 가지 않음)
        assert ct.remainder == "drop"
        # 활성 변환기(num/cat 등, remainder·drop 제외)가 사용하는 열만 수집한다.
        used = set()
        for name, trans, cols in ct.transformers_:
            if name == "remainder":
                continue
            if isinstance(trans, str) and trans == "drop":
                continue
            if isinstance(cols, (list, tuple)):
                used |= set(cols)
        leaked = comp_cols & used
        assert not leaked, f"{target}: 조성 변수가 추정기로 전달됨: {leaked}"
        # 조성 변수는 실제로 remainder(drop)에만 있어야 한다 (추정기 미전달 확인)
        remainder_cols = set()
        for name, trans, cols in ct.transformers_:
            if name == "remainder" and isinstance(cols, (list, tuple)):
                remainder_cols |= set(cols)
        assert comp_cols <= remainder_cols, (
            f"{target}: 조성 변수가 drop remainder 에 있지 않습니다."
        )


def test_features_metadata_lists_no_composition():
    """features 메타데이터의 모델 사용 변수 목록에 원료 조성이 없어야 한다."""
    used = set(features.V60_MODEL_USED_NUMERIC) | set(features.V60_MODEL_USED_CATEGORICAL)
    for c in features.V60_COMPOSITION_FIELDS_DROPPED:
        assert c not in used


# ── 호환성 게이트 ────────────────────────────────────────────────────────────


def test_compat_result_shape(handle):
    c = handle.compat
    assert c.status in ("OK", "ENVIRONMENT_INCOMPATIBLE", "PACKAGE_ERROR")
    assert "current" in c.version_info and "required" in c.version_info
    # 재현 세부가 기록되어야 한다
    assert "reproduced" in c.reproduction


def test_blocks_or_reproduces_consistently(handle):
    """비호환이면 차단, 호환이면 예제 재현이 정확해야 한다 (손상 값 통과 금지)."""
    if handle.usable:
        # 호환 환경: 예제 재현이 통과했어야 usable=True
        assert handle.compat.reproduction["reproduced"] is True
        for d in handle.compat.reproduction["details"]:
            assert d["match"] is True
    else:
        # 비호환 환경: 상태가 ENVIRONMENT_INCOMPATIBLE 이고 재현이 실패했어야 한다
        assert handle.compat.status in ("ENVIRONMENT_INCOMPATIBLE", "PACKAGE_ERROR")
        assert handle.compat.reproduction.get("reproduced") is not True


def test_predict_blocked_when_incompatible(handle, final_dir):
    import json

    inp = json.loads((final_dir / "example_input.json").read_text(encoding="utf-8"))
    res = predictor.predict(handle, inp)
    if handle.usable:
        assert res["blocked"] is False
        assert "prediction" in res["BET"] and "prediction" in res["TPV"]
    else:
        # 손상 값을 반환하지 않고 차단해야 한다
        assert res["blocked"] is True
        assert res["status"] in ("ENVIRONMENT_INCOMPATIBLE", "PACKAGE_ERROR")
        assert "BET" not in res or "prediction" not in res.get("BET", {})


def test_example_reproduction_when_compatible(handle, final_dir):
    """호환 환경에서만 수행: 동봉 예제 예측값과 정확히 일치."""
    if not handle.usable:
        pytest.skip("현재 환경은 v60 과 비호환이라 수치 예측 재현을 건너뜀(차단이 정상 동작).")
    import json

    for name in ["example", "out_of_domain_test"]:
        inp = json.loads((final_dir / f"{name}_input.json").read_text(encoding="utf-8"))
        exp = json.loads((final_dir / f"{name}_output.json").read_text(encoding="utf-8"))
        got = predictor.predict(handle, inp)
        for tgt in ["BET", "TPV"]:
            ep = float(exp[tgt]["prediction"])
            gp = float(got[tgt]["prediction"])
            assert abs(ep - gp) <= 1e-4 * max(1.0, abs(ep))


def test_model_info_reports_status(handle):
    info = predictor.model_info(handle)
    assert info["status"] == "research_prototype_complete_not_production_validated"
    assert info["usable_in_current_env"] == handle.usable
    assert info["required_libraries"]["sklearn"].startswith("1.8")
    # 외부 일반화 경고가 명시되어야 한다
    assert "external_generalization_warning" in info["applicability"]


# ── Stage 2 인터페이스 (연결·학습 없음) ──────────────────────────────────────


def test_stage2_link_from_v60_blocked_result_carries_no_values():
    """비호환/차단 결과는 BET/TPV 수치를 싣지 않고 경고만 전달한다."""
    from pyrolysis.biochar_stage2.adapter import Stage1LinkInput

    blocked = {"data_version": "v60", "blocked": True,
               "status": "ENVIRONMENT_INCOMPATIBLE", "reason": ["env mismatch"]}
    link = Stage1LinkInput.from_v60_prediction(blocked)
    assert link.bet_m2_g is None
    assert link.tpv_cm3_g is None
    assert link.bet_source == "stage1_v60_predicted"
    assert link.bet_applicability_warnings  # 경고 포함
    assert link.is_predicted is True


def test_stage2_link_from_v60_ok_result_keeps_predicted_separate():
    """호환 결과는 값을 싣되 측정값이 아니라 예측 출처로 구분한다."""
    from pyrolysis.biochar_stage2.adapter import (
        Stage1LinkInput, build_stage2_input_row,
    )

    ok = {
        "data_version": "v60", "blocked": False, "compatibility_ok": True,
        "BET": {"prediction": 849.26, "support_grade": "HIGH",
                "research_interval_approx": [140.0, 5100.0], "applicability_warnings": []},
        "TPV": {"prediction": 0.528, "support_grade": "HIGH",
                "research_interval_approx": [0.1, 1.1], "applicability_warnings": []},
    }
    link = Stage1LinkInput.from_v60_prediction(ok)
    assert link.bet_m2_g == 849.26 and link.tpv_cm3_g == 0.528
    assert link.bet_source == "stage1_v60_predicted"
    assert link.bet_support_grade == "HIGH"
    assert link.is_predicted is True
    assert link.bet_uncertainty == pytest.approx((5100.0 - 140.0) / 2.0)

    # 결합 행: 예측값을 measured_* 에 넣지 않는다
    row = build_stage2_input_row({"pollutant": "Pb"}, link)
    assert row["measured_BET_m2_g"] is None
    assert row["measured_TPV_cm3_g"] is None
    assert row["_stage1_link"]["bet_source"] == "stage1_v60_predicted"
    assert row["_stage1_link"]["stage1_model_version"] == "v60"


def test_stage2_measured_source_still_supported():
    """기존 measured 출처 동작이 회귀 없이 유지된다."""
    from pyrolysis.biochar_stage2.adapter import Stage1LinkInput, build_stage2_input_row

    link = Stage1LinkInput(bet_m2_g=400.0, tpv_cm3_g=0.2,
                           bet_source="measured", tpv_source="measured")
    row = build_stage2_input_row({"pollutant": "Cd"}, link)
    assert row["measured_BET_m2_g"] == 400.0
    assert row["measured_TPV_cm3_g"] == 0.2
    assert link.is_predicted is False
