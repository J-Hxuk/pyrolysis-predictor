"""UI feedstock-catalog regression: deployment never requires the legacy v15 ZIP."""
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from pyrolysis.biochar_ui.v112_unified_page import _build_feedstock_catalog
from pyrolysis.biochar_stage2 import v15_loader, v15_features
from pyrolysis.biochar_stage2.v15_extract import V15ExtractError


ROOT = Path(__file__).resolve().parents[1]


def test_missing_v15_zip_keeps_real_stage1_feedstocks(monkeypatch):
    def missing_zip(base):
        raise V15ExtractError("original v15 ZIP unavailable")

    monkeypatch.setattr(v15_loader, "load_core_qe", missing_zip)
    names, stage1, core = _build_feedstock_catalog(ROOT)
    assert "rice husk" in stage1
    assert not core
    assert names == sorted(stage1)


def test_v15_catalog_is_optional_additive_metadata(monkeypatch):
    def core_data(base):
        return SimpleNamespace(df=pd.DataFrame({
            v15_features.GROUP_FEEDSTOCK: ["verified_core_test_feedstock"]
        }))

    monkeypatch.setattr(v15_loader, "load_core_qe", core_data)
    names, stage1, core = _build_feedstock_catalog(ROOT)
    assert "rice husk" in stage1
    assert "verified core test feedstock" in core
    assert "verified core test feedstock" in names
