"""Research-only Stage1 DOI-heldout BET/TPV error and feature ablation.

Use the 2026-11-19 poster archive (or the 3-file input subset from the
ChatGPT experimental package). This file does NOT import or modify the app,
active models, or final v134/v139 predictions.

python research/stage1_phase1_20261120/run_experiment.py --archive /path/to/ZIP --out output_dir

The v60 archival cohort (254 BET, 191 TPV) is NOT the final v134
361 BET / 231 TPV full OOF cohort. Only exploratory assessment is made.
"""
from __future__ import annotations
import argparse
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = "data/imports/Biochar_Stage1_v60/Biochar_Stage1_v60_Final/"
COHORT = {
    "BET": "02_BET_AllData_FixedArchitecture_LODO.csv",
    "TPV": "02_TPV_AllData_FixedArchitecture_LODO.csv",
}
CAT = ["feedstock", "pyrolysis_atmosphere", "activation_method", "activation_agent"]
NUM = ["pyrolysis_temperature_C", "pyrolysis_hold_min", "heating_rate_C_min",
       "activation_temperature_C", "activation_time_min",
       "activation_agent_feedstock_mass_ratio"]
PCAT = ["process_stage", "activation_ratio_basis", "activation_agent_2",
        "pretreatment_agent", "post_treatment_agent"]
PNUM = ["pyrolysis_N2_flow_mL_min", "activation_gas_flow_mL_min",
        "activation_CO2_fraction_pct", "activation_steam_fraction_pct",
        "pretreatment_temperature_C", "pretreatment_time_min",
        "post_treatment_temperature_C", "post_treatment_time_min"]
RAW = ["feedstock_C_pct_original", "feedstock_H_pct_original",
       "feedstock_O_pct_original", "feedstock_N_pct_original",
       "feedstock_ash_pct_original", "feedstock_VM_pct_original",
       "feedstock_FC_pct_original"]
BASIS = {"daf": "daf", "moisture-free feedstock, paper p4": "moisture_free"}


def estimator(cat, num):
    prep = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), num),
        ("category", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="NR",
                                     keep_empty_features=True)),
            ("onehot", OneHotEncoder(handle_unknown="ignore",
                                      sparse_output=False)),
        ]), cat),
    ], remainder="drop", sparse_threshold=0)
    model = HistGradientBoostingRegressor(
        loss="poisson", max_iter=120, learning_rate=.06,
        max_leaf_nodes=15, min_samples_leaf=8,
        l2_regularization=5., random_state=20261120,
        early_stopping=False,
    )
    return Pipeline([("prep", prep), ("reg", model)])


def score(y, yhat):
    return dict(MAE=float(mean_absolute_error(y, yhat)),
                RMSE=float(np.sqrt(mean_squared_error(y, yhat))),
                R2=float(r2_score(y, yhat)))


def main(archive, output):
    output.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as z:
        master = pd.read_csv(io.BytesIO(z.read(
            ROOT + "01_Stage1_Master_v60_256.csv")), low_memory=False)
        refs = {key: pd.read_csv(io.BytesIO(z.read(ROOT + name)))
                for key, name in COHORT.items()}
    assert len(master) == 256 and master.doi.nunique() == 30
    assert not master.material_id.duplicated().any()
    for original in RAW:
        for basis, alias in BASIS.items():
            col = original.replace("_original", "") + "__" + alias
            master[col] = pd.to_numeric(master[original], errors="coerce").where(
                master.feedstock_elemental_basis.eq(basis))
    composition = [c for c in master if c.endswith(("__daf", "__moisture_free"))]
    report = {"status": "EXPLORATORY_POST_LOCK_NOT_V134_FINAL",
              "split": "GroupKFold(n_splits=5) grouped by DOI",
              "composition_basis": BASIS, "targets": {}}

    for target, ref in refs.items():
        expected = 254 if target == "BET" else 191
        assert len(ref) == expected
        master_by_id = master.set_index("material_id", drop=False)
        df = master_by_id.loc[ref.material_id].reset_index(drop=True)
        target_col = (f"{target}_m2_g_training_candidate"
                      if target == "BET" else "TPV_cm3_g_training_candidate")
        y = pd.to_numeric(df[target_col], errors="coerce").to_numpy(float)
        assert np.isfinite(y).all() and (y >= 0).all()
        assert np.allclose(y, ref.actual) and np.array_equal(df.doi, ref.doi)
        groups = df.doi.to_numpy()
        arms = {
            "HGB10": (CAT, NUM),
            "HGB10_plus_process": (CAT + PCAT, NUM + PNUM),
            "HGB10_plus_basis_safe_composition": (CAT, NUM + composition),
        }
        pred = {name: np.full(len(df), np.nan) for name in arms}
        folds = np.full(len(df), -1, dtype=int)
        audit = []
        for fold, (tr, te) in enumerate(GroupKFold(5).split(df, y, groups), 1):
            assert not (set(groups[tr]) & set(groups[te]))
            folds[te] = fold
            audit.append(dict(fold=fold, train_rows=len(tr), test_rows=len(te),
                              train_DOIs=len(set(groups[tr])),
                              test_DOIs=len(set(groups[te])),
                              test_doi_ids=";".join(sorted(set(groups[te])))))
            for name, (cat, num) in arms.items():
                X = df[cat + num].copy()
                for c in cat:
                    X[c] = X[c].astype("string").replace("", pd.NA)
                    X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
                for c in num:
                    X[c] = pd.to_numeric(X[c], errors="coerce")
                fit = estimator(cat, num).fit(X.iloc[tr], y[tr])
                pred[name][te] = np.maximum(0, fit.predict(X.iloc[te]))
        assert all(np.isfinite(value).all() for value in pred.values())
        out = pd.DataFrame({"material_id": df.material_id, "doi": df.doi,
                            "actual": y, "archived_v60_LODO": ref.prediction,
                            "fold": folds, **pred})
        out.to_csv(output / f"{target}_experimental_OOF.csv", index=False)
        pd.DataFrame(audit).to_csv(output / f"{target}_fold_audit.csv", index=False)
        doi = pd.DataFrame([
            {"doi": identifier, "n": len(sub), **{
                "MAE_" + name: float(np.abs(sub.actual-sub[name]).mean())
                for name in pred}}
            for identifier, sub in out.groupby("doi")
        ])
        doi.to_csv(output / f"{target}_per_doi.csv", index=False)
        scores = {}
        for name, values in pred.items():
            scores[name] = {**score(y, values),
                            "DOI_macro_MAE": float(doi["MAE_"+name].mean()),
                            "worst_DOI_MAE": float(doi["MAE_"+name].max())}
        comparisons = {}
        for name in list(arms)[1:]:
            delta = (doi["MAE_HGB10"]-doi["MAE_"+name]).to_numpy()
            rng = np.random.default_rng(1729)
            samples = rng.integers(0, len(delta), size=(4000, len(delta)))
            comparisons[name] = {
                "DOI_macro_improvement": float(delta.mean()),
                "DOI_bootstrap95": np.quantile(delta[samples].mean(axis=1),
                                                [.025, .975]).tolist(),
                "DOIs_improved": int((delta>0).sum()),
                "DOIs_worse": int((delta<0).sum())}
        report["targets"][target] = {
            "n": len(df), "DOI": df.doi.nunique(),
            "archived_v60_LODO": score(y, ref.prediction),
            "composition_rows_with_some_prequalified_data":
                int(df[composition].notna().any(axis=1).sum()),
            "models": scores, "comparisons": comparisons}
        print(target, json.dumps(scores, ensure_ascii=False), flush=True)
    (output/"summary.json").write_text(json.dumps(report, ensure_ascii=False,
                                                   indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    main(args.archive, args.out)
