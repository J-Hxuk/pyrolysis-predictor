"""Stage 1 BET: 조성의 추가 예측정보 검증 (읽기 전용, v60 비변경).

평가 데이터: eval37/base.csv (182 rows / 24 DOI, 비활성화 바이오매스 BET).
분할: LeaveOneGroupOut(provenance_group). 동일 행과 DOI 분할에서 조성 추가 비교.
**이 실험은 v60 배포 모델 재학습/성능이 아니며 TPV도 평가하지 않는다.**
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pyrolysis.biochar_stage1.data_loader import load_bet_base

PROCESS = ("temperature_C", "time_min", "heating_rate_C_min")
ELEMENTAL = ("C_pct", "H_pct", "N_pct", "O_pct")
PROXIMATE = ("VM_pct", "Ash_pct", "FC_pct")
FEATURE_SETS = OrderedDict([
    ("process_only", PROCESS),
    ("process_elemental", PROCESS + ELEMENTAL),
    ("process_proximate", PROCESS + PROXIMATE),
    ("process_all_composition", PROCESS + ELEMENTAL + PROXIMATE),
])
TARGET = "value"
GROUP = "provenance_group"


def make_model(algorithm: str, *, seed: int = 20261030):
    """각 홀드아웃의 학습 데이터에만 전처리 fit."""
    if algorithm == "Ridge":
        estimator = Ridge(alpha=10.0)
        pre = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    elif algorithm == "RandomForest":
        estimator = RandomForestRegressor(
            n_estimators=120, max_depth=7, min_samples_leaf=3,
            max_features=1.0, random_state=seed, n_jobs=1,
        )
        pre = Pipeline([("impute", SimpleImputer(strategy="median"))])
    elif algorithm == "Dummy":
        estimator = DummyRegressor(strategy="mean")
        pre = "passthrough"
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    return Pipeline([("pre", pre), ("model", estimator)])


def validate_data(df: pd.Data) -> dict:
    needed = set(PROCESS + ELEMENTAL + PROXIMATE + (TARGET, GROUP))
    missing = sorted(needed.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df[GROUP].isna().any() or df[GROUP].nunique() < 3:
        raise ValueError("Need at least 3 non-null DOI/provenance groups")
    if df[list(PROCESS + ELEMENTAL + PROXIMATE + (TARGET,))].isna().any().any():
        raise ValueError("eval37 matched ablation requires fully observed input/target rows")
    if (df[TARGET] < 0).any():
        raise ValueError("BET negative training label")
    return {
        "rows": int(len(df)), "doi_groups": int(df[GROUP].nunique()),
        "unique_composition_patterns": int(df[list(ELEMENTAL + PROXIMATE)].drop_duplicates().shape[0]),
        "unique_process_patterns": int(df[list(PROCESS)].drop_duplicates().shape[0]),
        "duplicated_full_feature_rows": int(df.duplicated(subset=list(PROCESS + ELEMENTAL + PROXIMATE)).sum()),
        "target_min": float(df[TARGET].min()),
        "target_max": float(df[TARGET].max()),
    }


def evaluate(df: pd.Data, *, algorithms=("Ridge", "RandomForest")) -> tuple[pd.DataFrame, pd.DataFrame]:
    """LODO OOF. 어느 방법도 test DOI 데이터를 fit에 사용하지 않는다."""
    validate_data(df)
    df = df.reset_index(drop=True)
    y = df[TARGET].to_numpy(dtype=float)
    groups = df[GROUP].astype(str).to_numpy()
    splits = list(LeaveOneGroupOut().split(df, y, groups))
    rows = []
    for algo in algorithms:
        for feature_set, columns in FEATURE_SETS.items():
            preds = np.full(len(df), np.nan)
            for train, test in splits:
                assert set(groups[train]).isdisjoint(set(groups[test]))
                model = make_model(algo)
                model.fit(df.iloc[train][list(columns)], y[train])
                preds[test] = model.predict(df.iloc[test][list(columns)])
            if not np.all(np.isfinite(preds)):
                raise AssertionError("Non-finite OOF predictions")
            for i, value in enumerate(preds):
                rows.append({"row_index": i, "doi": str(groups[i]), "algorithm": algo,
                             "feature_set": feature_set, "actual_BET_m2_g": float(y[i]),
                             "predicted_BET_m2_g": float(value),
                             "abs_error_BET_m2_g": float(abs(value-y[i]))})
    pred_df = pd.DataFrame(rows)
    metrics = []
    for (algo, fset), sub in pred_df.groupby(["algorithm", "feature_set"], sort=True):
        ya = sub["actual_BET_m2_g"].to_numpy()
        yp = sub["predicted_BET_m2_g"].to_numpy()
        metrics.append({"algorithm": algo, "feature_set": fset,
                        "n_rows": len(sub), "n_doi": sub.doi.nunique(),
                        "r2_oof": float(r2_score(ya, yp)),
                        "mae_m2_g": float(mean_absolute_error(ya, yp)),
                        "rmse_m2_g": float(np.sqrt(mean_squared_error(ya, yp))),
                        "doi_macro_mae_m2_g": float(sub.groupby("doi").abs_error_BET_m2_g.mean().mean()),
                        "negative_prediction_count": int((yp < 0).sum())})
    return pred_df, pd.DataFrame(metrics).sort_values(["algorithm", "feature_set"]).reset_index(drop=True)


def doi_cluster_bootstrap(pred_df: pd.DataFrame, *, repeats=2500, seed=20261030) -> pd.DataFrame:
    """DOI cluster resample; composition - process MAE (negative=improvement).

    Cluster bootstrap intervals are descriptive, not independent external validation.
    """
    rng = np.random.default_rng(seed)
    result = []
    for algo in pred_df.algorithm.unique():
        df = pred_df[pred_df.algorithm == algo]
        pivot = df.pivot(index="row_index", columns="feature_set", values="abs_error_BET_m2_g")
        doi = df.drop_duplicates("row_index").set_index("row_index").loc[pivot.index, "doi"]
        uniq = pd.unique(doi).tolist()
        indices = {g: np.flatnonzero(doi.to_numpy() == g) for g in uniq}
        base = pivot["process_only"].to_numpy()
        macro_base = pivot.assign(doi=doi.values).groupby("doi")["process_only"].mean()
        for fset in (x for x in FEATURE_SETS if x != "process_only"):
            compare = pivot[fset].to_numpy()
            deltas = np.empty(repeats)
            macro_deltas = np.empty(repeats)
            diff_g = pivot.assign(doi=doi.values).groupby("doi").apply(
                lambda z: float((z[fset]-z["process_only"]).mean()), include_groups=False)
            for r in range(repeats):
                selected = rng.choice(uniq, size=len(uniq), replace=True)
                idx = np.concatenate([indices[g] for g in selected])
                deltas[r] = float(np.mean(compare[idx]-base[idx]))
                macro_deltas[r] = float(np.mean([diff_g[g] for g in selected]))
            result.append({"algorithm": algo, "feature_set": fset,
                           "delta_mae_m2_g": float(np.mean(compare-base)),
                           "ci95_lo": float(np.quantile(deltas, .025)),
                           "ci95_hi": float(np.quantile(deltas, .975)),
                           "delta_doi_macro_mae_m2_g": float(diff_g.mean()),
                           "macro_ci95_lo": float(np.quantile(macro_deltas, .025)),
                           "macro_ci95_hi": float(np.quantile(macro_deltas, .975)),
                           "n_doi": len(uniq), "bootstrap_repeats": repeats})
    return pd.DataFrame(result)


def v60_composition_coverage(root: Path) -> tuple[pd.DataFrame, dict]:
    p = Path(root)/"data/imports/Biochar_Stage1_v60/Biochar_Stage1_v60_Final/01_Stage1_Master_v60_256.csv"
    d = pd.read_csv(p, low_memory=False)
    fields = ["feedstock_C_pct_original", "feedstock_H_pct_original", "feedstock_O_pct_original",
              "feedstock_N_pct_original", "feedstock_S_pct_original", "feedstock_ash_pct_original",
              "feedstock_VM_pct_original", "feedstock_FC_pct_original",
              "feedstock_moisture_pct_original"]
    out = []
    for c in fields:
        v = pd.to_numeric(d[c], errors="coerce")
        avail = v.notna()
        out.append({"feature": c, "available_rows": int(avail.sum()),
                    "total_rows": len(d), "fraction_available": float(avail.mean()),
                    "n_doi_with_value": int(d.loc[avail, "doi"].nunique()),
                    "distinct_values": int(v.nunique())})
    all_ultimate = pd.concat([pd.to_numeric(d[c], errors="coerce") for c in fields[:4]], axis=1).notna().all(axis=1)
    all_proximate = pd.concat([pd.to_numeric(d[c], errors="coerce") for c in fields[5:8]], axis=1).notna().all(axis=1)
    both = all_ultimate & all_proximate
    meta = {"rows": len(d), "doi_count": int(d.doi.nunique()),
            "all_C_H_O_N_rows": int(all_ultimate.sum()),
            "all_ash_VM_FC_rows": int(all_proximate.sum()),
            "complete_both_rows": int(both.sum()),
            "complete_both_doi_count": int(d.loc[both, "doi"].nunique()),
            "composition_basis_counts": {str(k): int(v) for k,v in d.feedstock_elemental_basis.fillna("NULL").value_counts().items()},
            "warning": "v60 조성 분석기준 및 원료/바이오차 구분 검증 필요; 통합학습 자동 금지"}
    return pd.DataFrame(out), meta



def tpv_data_readiness(root: Path) -> dict:
    """TPV 비교 보류의 근거 수집; 이 자료로 성능을 산출하거나 학습하지 않는다."""
    p = Path(root) / "data/imports/Biochar_Complete_Project_v41/merge30/TPV_default_train_pool.csv"
    d = pd.read_csv(p, low_memory=False)
    accepted = d[d["status"].eq("ACCEPTED")]
    chem = list(ELEMENTAL + PROXIMATE)
    cols = list(("temperature_C", "time_min") + ELEMENTAL + PROXIMATE)
    complete = accepted.dropna(subset=cols + [TARGET]) if TARGET in accepted.columns else accepted.dropna(subset=cols + ["value"])
    from_secondary = complete["source_evidence"].eq("author_SI_secondary_not_all_primary_rechecked")
    return {
        "all_rows": len(d), "accepted_rows": len(accepted),
        "complete_composition_and_temp_time": len(complete),
        "complete_groups": int(complete["provenance_group"].nunique()),
        "complete_from_unverified_secondary_source": int(from_secondary.sum()),
        "complete_with_heating_rate": int(complete["heating_rate_C_min"].notna().sum()),
        "complete_with_named_feedstock": int(complete["feedstock"].notna().sum()),
        "duplicate_rows_when_activation_omitted": int(complete.duplicated(subset=cols).sum()),
        "status": "TPV_COMPOSITION_ABLATION_NOT_CERTIFIED",
        "reason": "Complete cases stem entirely from secondary, not fully primary-verified compilation; heating rate and feedstock names absent; activation conditions may confound identical displayed process+composition features.",
    }


def run(root: Path, outdir: Path, *, repeats=2500) -> dict:
    root, outdir = Path(root), Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    raw = load_bet_base(root)
    info = validate_data(raw)
    preds, metrics = evaluate(raw)
    boots = doi_cluster_bootstrap(preds, repeats=repeats)
    cov, cov_meta = v60_composition_coverage(root)
    tpv_meta = tpv_data_readiness(root)
    preds.to_csv(outdir/"oof_predictions.csv", index=False)
    metrics.to_csv(outdir/"model_comparison.csv", index=False)
    boots.to_csv(outdir/"paired_doi_bootstrap.csv", index=False)
    cov.to_csv(outdir/"v60_composition_coverage.csv", index=False)
    summary = {
        "status": "EXPLORATORY_NOT_DEPLOYED", "dataset": "eval37/base.csv (nonactivated BET only)",
        "data_info": info, "v60_composition": cov_meta, "tpv_readiness": tpv_meta,
        "split": "LeaveOneGroupOut(provenance_group), same held-out DOI per feature set",
        "fitting_note": "All transformations fitted within each train fold; no holdout data fitted",
        "disclaimer": "Not a v60 model comparison; cannot claim TPV or Stage2 improvement",
        "metrics": metrics.to_dict("records"), "paired_deltas": boots.to_dict("records"),
    }
    (outdir/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
