"""
run_v66_logit_fraction.py — 연구용(RESEARCH_ONLY_NOT_ACTIVE).

v65 후속. v65 에서 단순 분율(f=qe/cap) 표적은 상한위반을 0%로 없앴지만,
cap(=C0/dose)이 큰 소수 DOI에서 역변환 증폭으로 최악 MAE 가 악화됐다.

v66 가설: 분율을 logit 공간 z=log(f/(1-f)) 에서 학습하면 경계 포화와
역변환 증폭이 완화되어, 상한위반 0%를 유지하면서 최악 DOI 오차가 줄어든다.

세 방식을 '완전히 동일한' 데이터/분할/피처/알고리즘(XGBoost)에서 비교한다:
  BASELINE : 표적 qe,        예측 후 clip(0, cap)           (기존 활성 방식)
  FRACTION : 표적 qe/cap,     f_hat∈[0,1] → qe=f_hat*cap      (v65)
  LOGIT    : 표적 logit(qe/cap), sigmoid 역변환 → qe=f_hat*cap (신규)

logit 변환 공식은 scripts/evaluate_v26_physics_aware.py 의 수치 안정형을 따른다.
활성 배포 모델·원본 데이터·기존 리포트를 변경하지 않는다. 결과 → reports/v66_logit_fraction/.
"""
from __future__ import annotations

import json
import platform
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import sklearn
import xgboost

from scripts.build_v46_phase8 import features
from scripts.run_v58_failure_diagnostics import build, derive
from scripts.run_v59_doi_material_weighting import PRE, NEW, fit_predict

REPORT = "reports/v66_logit_fraction"
VARIANT = "B"
FEATURE_SET = "cap"
POLICY = "row_equal"
N_SPLITS = 5
EPS = 0.01  # logit 경계 클립


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.where(
        z >= 0,
        1.0 / (1.0 + np.exp(-np.minimum(z, 700))),
        np.exp(np.maximum(z, -700)) / (1.0 + np.exp(np.maximum(z, -700))),
    )


def _fit_fraction(train, test):
    """v65 단순 분율: 표적 f=qe/cap, 예측 clip[0,1] 후 역변환."""
    tr = derive(train, FEATURE_SET)
    te = derive(test, FEATURE_SET)
    model, cols, nums, cats = build(VARIANT, FEATURE_SET)
    cap_tr = (train.C0_mg_L / train.dose_g_L).to_numpy(float)
    f_tr = np.clip(train.target_value.to_numpy(float) / cap_tr, 0.0, 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        model.fit(features(tr, cols, nums, cats), f_tr)
        f_hat = np.clip(np.asarray(model.predict(features(te, cols, nums, cats)), float), 0.0, 1.0)
    cap_te = (test.C0_mg_L / test.dose_g_L).to_numpy(float)
    return f_hat * cap_te


def _fit_logit(train, test):
    """v66 logit 분율: 표적 logit(f), sigmoid 역변환."""
    tr = derive(train, FEATURE_SET)
    te = derive(test, FEATURE_SET)
    model, cols, nums, cats = build(VARIANT, FEATURE_SET)
    cap_tr = (train.C0_mg_L / train.dose_g_L).to_numpy(float)
    f_tr = np.clip(train.target_value.to_numpy(float) / cap_tr, EPS, 1.0 - EPS)
    z_tr = np.log(f_tr / (1.0 - f_tr))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        model.fit(features(tr, cols, nums, cats), z_tr)
        z_hat = np.asarray(model.predict(features(te, cols, nums, cats)), float)
    cap_te = (test.C0_mg_L / test.dose_g_L).to_numpy(float)
    return _sigmoid(z_hat) * cap_te


def metrics(y, p):
    err = p - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    return {"n": int(len(y)), "mae": round(mae, 4), "rmse": round(rmse, 4),
            "r2": round(r2, 4) if r2 is not None else None}


def per_doi(y, p, doi):
    rows = []
    for d in pd.unique(doi):
        m = doi == d
        yt, pt = y[m], p[m]
        n = int(m.sum())
        err = pt - yt
        mae = float(np.mean(np.abs(err)))
        if n >= 2 and np.var(yt) > 0:
            r2 = 1.0 - float(np.sum(err**2)) / float(np.sum((yt - np.mean(yt)) ** 2))
        else:
            r2 = None
        rows.append({"doi": str(d), "n": n, "mae": round(mae, 3),
                     "r2": round(r2, 4) if r2 is not None else None})
    return pd.DataFrame(rows)


def dist_summary(tbl, label):
    maes = tbl["mae"].dropna().to_numpy()
    r2s = tbl["r2"].dropna().to_numpy()
    return {
        "method": label,
        "DOI_MAE_median": round(float(np.median(maes)), 3),
        "DOI_MAE_q75": round(float(np.percentile(maes, 75)), 3),
        "DOI_MAE_worst": round(float(np.max(maes)), 3),
        "DOI_R2_median": round(float(np.median(r2s)), 4) if len(r2s) else None,
        "DOI_R2_worst": round(float(np.min(r2s)), 4) if len(r2s) else None,
        "n_DOI_negative_R2": int(np.sum(r2s < 0)) if len(r2s) else None,
    }


def boot_gain(y, ref, cand, doi, reps=3000):
    """DOI 페어드 부트스트랩: cand 가 ref 대비 MAE 이득(양수=cand 우세)."""
    gain = np.abs(y - ref) - np.abs(y - cand)
    ids = np.unique(doi)
    idx = [np.flatnonzero(doi == g) for g in ids]
    rng = np.random.default_rng(20261031)
    dist = [float(gain[np.concatenate([idx[j] for j in rng.integers(0, len(ids), len(ids))])].mean())
            for _ in range(reps)]
    lo, hi = (float(q) for q in np.quantile(dist, [0.025, 0.975]))
    return {"mean_gain_mg_g": round(float(gain.mean()), 4),
            "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
            "ci_excludes_zero": bool(lo > 0 or hi < 0)}


def main(root: str) -> None:
    root = Path(root).resolve()
    dst = root / REPORT
    dst.mkdir(parents=True, exist_ok=True)

    v152 = pd.read_csv(root / PRE, low_memory=False)
    new = pd.read_csv(root / NEW, low_memory=False)
    d = pd.concat([v152, new], ignore_index=True, sort=False)
    assert len(d) == 188 and d.doi.nunique() == 36 and d.observation_id.is_unique

    y = d.target_value.to_numpy(float)
    cap = (d.C0_mg_L / d.dose_g_L).to_numpy(float)
    doi = d.doi.to_numpy(str)

    base = np.full(len(d), np.nan)
    base_raw = np.full(len(d), np.nan)
    frac = np.full(len(d), np.nan)
    logit = np.full(len(d), np.nan)

    for it, iv in GroupKFold(n_splits=N_SPLITS).split(d, groups=d.doi):
        held = set(d.iloc[iv].doi)
        tr = d[~d.doi.isin(held)].copy()
        te = d.iloc[iv].copy()
        assert set(tr.doi).isdisjoint(held)
        bp, braw = fit_predict(tr, te, VARIANT, FEATURE_SET, POLICY)
        base[iv] = bp
        base_raw[iv] = braw
        frac[iv] = _fit_fraction(tr, te)
        logit[iv] = _fit_logit(tr, te)

    oof = d[["observation_id", "doi", "pollutant", "target_value", "C0_mg_L", "dose_g_L"]].copy()
    oof["cap"] = cap
    oof["baseline_pred"] = base
    oof["baseline_raw"] = base_raw
    oof["fraction_pred"] = frac
    oof["logit_pred"] = logit
    oof.to_csv(dst / "oof_predictions.csv", index=False, encoding="utf-8-sig")

    methods = {"BASELINE_qe_then_clip": base,
               "FRACTION_linear": frac,
               "LOGIT_FRACTION": logit}
    viol = {"BASELINE_qe_then_clip": int(np.sum(base_raw > cap + 1e-8)),
            "FRACTION_linear": int(np.sum(frac > cap + 1e-8)),
            "LOGIT_FRACTION": int(np.sum(logit > cap + 1e-8))}

    agg = []
    doi_tables = {}
    for label, p in methods.items():
        agg.append({"method": label, **metrics(y, p),
                    "upper_cap_violation": viol[label],
                    "violation_rate": round(viol[label] / len(d), 4)})
        doi_tables[label] = per_doi(y, p, doi)
    pd.DataFrame(agg).to_csv(dst / "aggregate_metrics.csv", index=False, encoding="utf-8-sig")

    allper = []
    for label, tbl in doi_tables.items():
        t = tbl.copy()
        t["method"] = label
        allper.append(t)
    pd.concat(allper, ignore_index=True).to_csv(dst / "per_doi_metrics.csv", index=False, encoding="utf-8-sig")

    dist = pd.DataFrame([dist_summary(doi_tables[k], k) for k in methods])
    dist.to_csv(dst / "doi_distribution_summary.csv", index=False, encoding="utf-8-sig")

    boots = {
        "LOGIT_vs_BASELINE": boot_gain(y, base, logit, doi),
        "LOGIT_vs_FRACTION": boot_gain(y, frac, logit, doi),
    }

    manifest = {
        "version": "v66",
        "role": "RESEARCH_ONLY_NOT_ACTIVE",
        "hypothesis": "logit-space fraction target keeps 0% cap violation while reducing worst-DOI inverse-transform blowup seen in v65 linear fraction",
        "fair_comparison": {
            "same_data": "188 rows / 36 DOI (v55_152 + new36)",
            "same_split": f"GroupKFold(DOI, n_splits={N_SPLITS})",
            "same_features": f"Model {VARIANT}, feature_set={FEATURE_SET}",
            "same_algorithm": "XGBoost s2_model (unchanged)",
            "only_changed": "learning target (qe vs qe/cap vs logit(qe/cap))",
            "logit_epsilon": EPS,
        },
        "aggregate": agg,
        "distribution": dist.to_dict(orient="records"),
        "doi_bootstrap": boots,
        "environment": {"python": platform.python_version(),
                        "sklearn": sklearn.__version__, "xgboost": xgboost.__version__},
    }
    (dst / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"aggregate": agg, "distribution": dist.to_dict(orient="records"),
                      "bootstrap": boots}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    main(ap.parse_args().root)
