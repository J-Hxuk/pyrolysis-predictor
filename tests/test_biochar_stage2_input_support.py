from pyrolysis.biochar_stage2.input_support import (
    match_feedstock, validate_composition, applied_optional_fields,
)

KNOWN = ["rice husk", "rice husk;nanchang,jiangxi", "rice straw", "peanut shell", "bamboo", "bamboo powder"]


def test_exact_match():
    r = match_feedstock("Rice Husk", KNOWN)
    assert (r.status, r.matched) == ("exact", "rice husk")


def test_korean_alias_without_biomass_substitution():
    r = match_feedstock("왕겨", KNOWN)
    assert (r.status, r.matched) == ("alias", "rice husk")
    assert match_feedstock("볏짚", KNOWN).matched == "rice straw"
    assert match_feedstock("bamboo powder", KNOWN).matched == "bamboo powder"


def test_typo_is_suggested_but_never_auto_matched():
    r = match_feedstock("rice hus", KNOWN)
    assert r.matched is None and "rice husk" in r.suggestions


def test_new_feedstock_is_not_replaced():
    r = match_feedstock("banana leaf biochar", KNOWN)
    assert r.matched is None and r.original == "banana leaf biochar"


def test_optional_unknown_is_unreported_and_zero_is_valid():
    vals, errors = validate_composition({"raw_ash_wt_pct": "", "raw_C_wt_pct": "0", "raw_VM_wt_pct": "30"})
    assert not errors and "raw_ash_wt_pct" not in vals and vals["raw_C_wt_pct"] == 0


def test_composition_basis_guard():
    vals, errors = validate_composition({"raw_ash_wt_pct": "50", "raw_VM_wt_pct": "50", "raw_FC_wt_pct": "50"})
    assert vals and errors


def test_ignore_report_identifies_stored_feature_contract():
    use, ignored = applied_optional_fields({"raw_ash_wt_pct": 10, "raw_C_wt_pct": 50}, {"input_cols": ["raw_C_wt_pct", "C0_mg_L"]})
    assert use == ["raw_C_wt_pct"] and ignored == ["raw_ash_wt_pct"]


def test_model_a_prediction_ignores_unknown_feedstock_name():
    from pyrolysis.biochar_stage2.input_support import predict_from_contract
    class FakePipe:
        def predict(self, X):
            assert list(X.columns) == ["C0_mg_L", "dose_g_L"]
            return [float(X["C0_mg_L"].iloc[0]) / 4.0]
    model = {"metadata": {"input_cols": ["C0_mg_L", "dose_g_L"]}, "pipeline": FakePipe()}
    base = {"C0_mg_L": 100, "dose_g_L": 2}
    a = predict_from_contract(model, {**base, "feedstock_label_normalized": "rice husk"})
    b = predict_from_contract(model, {**base, "feedstock_label_normalized": "banana peel"})
    assert a == b == (25.0, 25.0, 50.0)


def test_input_contract_not_backfilled_with_fake_composition():
    from pyrolysis.biochar_stage2.input_support import predict_from_contract
    class FakePipe:
        def predict(self, X):
            assert "raw_C_wt_pct" not in X
            return [123.0]
    model = {"metadata": {"input_cols": ["C0_mg_L", "dose_g_L"]}, "pipeline": FakePipe()}
    assert predict_from_contract(model, {"C0_mg_L": 100, "dose_g_L": 2, "raw_C_wt_pct": 70}) == (123.0, 50.0, 50.0)


def test_actual_core_lookup_has_multiple_feedstocks_and_provenance():
    from pathlib import Path
    from pyrolysis.biochar_stage2 import v15_loader as L
    from pyrolysis.biochar_stage2.input_support import related_records
    base = Path(__file__).resolve().parents[1]
    ds = L.load_core_qe(base)
    assert len(ds.df) == 97 and ds.n_doi == 25
    for name in ["rice husk", "rice straw", "peanut shell", "bamboo"]:
        assert not related_records(ds.df, name).empty
