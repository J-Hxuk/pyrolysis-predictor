"""
test_biochar_app.py — Biochar 앱 실행·조건 비교 CSV 검증 (AppTest).

완료 기준(지시서 §7) 대응:
  1. Streamlit 앱이 실행된다.
  8. 조건 비교 그래프 및 CSV 저장 기능이 동작한다.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_APP = _REPO / "biochar_app.py"


def _ensure_model():
    """활성 모델이 없으면 학습·활성화한다 (앱 페이지가 모델을 로드할 수 있도록)."""
    from pyrolysis.biochar_stage1.extract import extract_required_files
    from pyrolysis.biochar_stage1.model_registry import Stage1Registry
    from pyrolysis.biochar_stage1.data_loader import load_stage1_dataset
    from pyrolysis.biochar_stage1.training import train_bet_model, save_bet_model

    extract_required_files(_REPO)
    reg = Stage1Registry.default(_REPO)
    if reg.get_active_dir() is None:
        ds = load_stage1_dataset(_REPO)
        model = train_bet_model(ds, base=_REPO, run_cv=False)
        save_bet_model(model, reg.root)
        reg.set_active(model.version)


@pytest.fixture(scope="module", autouse=True)
def _model():
    _ensure_model()


def _apptest():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(str(_APP), default_timeout=120)


def test_app_launches():
    """기준 1: 앱이 예외 없이 실행되고 5개 페이지 메뉴가 있다."""
    at = _apptest()
    at.run()
    assert not at.exception
    assert len(at.radio) == 1
    assert at.radio[0].options == [
        "A. Overview", "B. Pyrolysis Prediction", "C. Condition Comparison",
        "D. Adsorption Analysis", "E. Data & Model Validation",
    ]


def test_prediction_page_predicts():
    """기준 2: 예측 페이지에서 BET 값을 반환한다."""
    at = _apptest()
    at.run()
    at.radio[0].set_value("B. Pyrolysis Prediction").run()
    btns = [b for b in at.button if "예측" in b.label]
    assert btns
    btns[0].set_value(True).run()
    assert not at.error
    assert any("BET" in m.label for m in at.metric)


def test_condition_comparison_table_and_csv():
    """기준 8: 조건 비교 표·곡선·CSV 다운로드가 동작한다."""
    at = _apptest()
    at.run()
    at.radio[0].set_value("C. Condition Comparison").run()
    btns = [b for b in at.button if "조건 비교" in b.label]
    assert btns
    btns[0].set_value(True).run()
    assert not at.error
    assert len(at.dataframe) >= 1
    dls = at.get("download_button")
    assert any("CSV" in getattr(d, "label", "") for d in dls)


def test_adsorption_langmuir_runs():
    at = _apptest()
    at.run()
    at.radio[0].set_value("D. Adsorption Analysis").run()
    btns = [b for b in at.button if "Langmuir" in b.label]
    assert btns
    btns[0].set_value(True).run()
    assert not at.error
    assert any("qmax" in m.label for m in at.metric)
