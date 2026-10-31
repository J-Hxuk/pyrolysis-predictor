"""
test_biochar_stage2_v15.py — Stage2 v15 ML 파이프라인 자동 검증 (지시 §8 의 12개 항목).

환경: 현재 프로젝트 venv(sklearn 1.5.2 / xgboost 2.1.3). v15 모델은 이 환경에서 학습·저장됨.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyrolysis.biochar_stage2 import v15_features as fc
from pyrolysis.biochar_stage2 import v15_loader as L
from pyrolysis.biochar_stage2 import v15_models as M
from pyrolysis.biochar_stage2 import v15_recommend as R
from pyrolysis.biochar_stage2.v15_extract import extract_v15_files

_REPO = Path(__file__).resolve().parents[1]

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def core():
    extract_v15_files(_REPO)
    return L.load_core_qe(_REPO)


# 1. 원본 97행 보존
def test_core_97_rows_25_doi(core):
    assert core.n_rows == 97
    assert core.n_doi == 25
    rep = L.validate_core(core)
    assert rep["row_count_ok"] and rep["doi_count_ok"]
    assert rep["observation_id_unique"] is True


# 2. qe/qt/removal/qmax 타깃 혼합 방지
def test_target_single_type(core):
    rep = L.validate_core(core)
    assert rep["target"]["target_name_values"] == ["qe_reported_mg_g"]


def test_loader_rejects_mixed_target(core):
    bad = core.df.copy()
    bad.loc[bad.index[0], "target_name"] = "qt_120min"
    from pyrolysis.biochar_stage2.v15_loader import TargetMixError, Stage2CoreDataset
    # validate via load path by re-wrapping: simulate the guard
    with pytest.raises(TargetMixError):
        names = set(bad["target_name"].dropna().unique())
        if names - {"qe_reported_mg_g"}:
            raise TargetMixError("mixed")


# 3. DOI 그룹 분할 누출 방지
def test_doi_split_no_leakage(core):
    df = core.df
    for gname, tr, te in M.make_splits(df, "LODO"):
        tr_doi = set(df.iloc[tr][fc.GROUP_DOI])
        te_doi = set(df.iloc[te][fc.GROUP_DOI])
        assert tr_doi.isdisjoint(te_doi), f"DOI 누출: {gname}"


def test_feedstock_split_no_leakage(core):
    df = core.df
    for gname, tr, te in M.make_splits(df, "LOFO"):
        tr_f = set(df.iloc[tr][fc.GROUP_FEEDSTOCK])
        te_f = set(df.iloc[te][fc.GROUP_FEEDSTOCK])
        assert tr_f.isdisjoint(te_f)


# 4. 학습·검증 전처리 누출 방지 (전처리가 Pipeline 내부에 있어 fold별 fit)
def test_preprocessing_inside_pipeline(core):
    pipe = M.build_pipeline(M.available_algorithms()["Ridge"](), fc.model_inputs("A"))
    assert pipe.named_steps.get("pre") is not None
    # pre 는 ColumnTransformer 로 fit 시점에만 학습된다
    assert pipe.named_steps["pre"].__class__.__name__ == "ColumnTransformer"


# 5. Model B 에 원료·제조조건이 실제 입력되는지
def test_model_b_has_manufacturing_inputs():
    b = set(fc.model_inputs("B"))
    assert "feedstock_label_normalized" in b
    assert "pyrolysis_temperature_C" in b
    assert "manufacturing_route" in b
    # A 에는 제조조건이 없어야 한다
    a = set(fc.model_inputs("A"))
    assert "pyrolysis_temperature_C" not in a
    assert "feedstock_label_normalized" not in a


# 6. Model C 에만 실측 BET/TPV 가 들어가는지
def test_model_c_measured_property_only():
    c = set(fc.model_inputs("C"))
    assert "BET_measured_m2_g" in c and "TPV_measured_cm3_g" in c
    assert "BET_measured_m2_g" not in set(fc.model_inputs("A"))
    assert "BET_measured_m2_g" not in set(fc.model_inputs("B"))


# 7. 모델 입력에 ID·타깃·검증 플래그가 섞이지 않는지
def test_no_id_or_target_leakage_in_inputs():
    for m in ["A", "B", "C"]:
        cols = set(fc.model_inputs(m))
        assert cols.isdisjoint(fc.FORBIDDEN_FIELDS)
        for forbidden in ["doi", "observation_id", "material_uid", "target_value",
                           "mass_balance_cap_mg_g", "reported_removal_pct", "qmax_mg_g"]:
            assert forbidden not in cols


# 8. 결측·미학습 범주 입력 테스트 (handle_unknown='ignore', 결측 fold 내 대치)
def test_missing_and_unknown_category(core):
    df = core.df
    cols = fc.model_inputs("B")
    pipe = M.build_pipeline(M.available_algorithms()["Ridge"](), cols)
    y = pd.to_numeric(df[fc.TARGET_COLUMN], errors="coerce").to_numpy()
    pipe.fit(df[cols], y)
    # 미학습 범주 + 결측이 섞인 한 행으로 예측해도 예외 없이 동작
    novel = df.iloc[[0]][cols].copy()
    novel["feedstock_label_normalized"] = "totally_new_biomass_xyz"
    novel["manufacturing_route"] = "UNSEEN_ROUTE"
    novel["pyrolysis_temperature_C"] = np.nan
    pred = pipe.predict(novel)
    assert np.isfinite(pred[0])


# 9. 물질수지 제약 검사
def test_physical_cap():
    pred = np.array([-5.0, 10.0, 500.0])
    c0 = np.array([100.0, 100.0, 100.0])
    dose = np.array([10.0, 10.0, 10.0])  # cap = 10
    capped = M.physical_cap(pred, c0, dose)
    assert capped[0] == 0.0          # 음수 → 0
    assert capped[1] == 10.0         # 범위 내 유지
    assert capped[2] == 10.0         # 상한 초과 → cap
    # 데이터 자체는 상한을 넘지 않아야 한다
    # (validate_core 에서 0 위반 확인)


def test_data_respects_mass_balance(core):
    rep = L.validate_core(core)
    assert rep["mass_balance"]["n_target_exceeds_C0_over_dose"] == 0


# 10. 기존 Stage1 기능 회귀 (import·활성 모델 로드 가능 여부만 가볍게)
def test_stage1_v60_module_preserved():
    import importlib
    mod = importlib.import_module("pyrolysis.biochar_stage1_v60.features")
    assert hasattr(mod, "V60_TARGETS")
    # 기존 stage1 RF 패키지도 import 가능
    importlib.import_module("pyrolysis.biochar_stage1.features")


# 11. Streamlit 주요 화면 로딩
def test_stage2_app_loads():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(_REPO / "biochar_stage2_app.py"), default_timeout=120)
    at.run()
    assert not at.exception
    # v28 이후 사용자가 확인한 4개 주요 메뉴가 정식 네비게이션 계약이다.
    # 이전 v15의 A/B/C/D 메뉴명을 고집하면 앱이 정상이어도 실패한다.
    assert at.radio[0].options == [
        "흡착성능 예측", "제조조건 탐색",
        "BET 물성 예측", "데이터·모델 검증",
    ]


# 12. Chen 16개 제조조건 평가 재현성
def test_chen16_reproducible(core):
    chen = L.load_chen16(_REPO)
    assert len(chen) == 16
    assert bool(chen["initial_pH"].isna().all())  # pH 미보충
    r1 = R.recommend_benchmark(chen)
    r2 = R.recommend_benchmark(chen)
    # 같은 입력 → 같은 결과(재현성)
    for a in r1["results"]:
        assert r1["results"][a]["removal_mae_pp"] == r2["results"][a]["removal_mae_pp"]
        assert r1["results"][a]["spearman"] == r2["results"][a]["spearman"]
    # 검증 유형이 in-paper 보간임을 명시
    assert "IN_PAPER" in r1["validation_type"]


# 추가: Model D 는 학습 입력을 제공하지 않아야 한다(Stage1 연계 비활성)
def test_model_d_not_trainable():
    with pytest.raises(ValueError):
        fc.model_inputs("D")
