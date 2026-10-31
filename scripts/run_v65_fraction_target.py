"""
run_v65_fraction_target.py — 연구용(RESEARCH_ONLY_NOT_ACTIVE).

가설: Stage2 qe 를 직접 예측하고 사후 clipping(0≤qe≤C0/dose)하는 대신,
      흡착 분율 f = qe / (C0/dose) 를 표적으로 학습하면
      (1) 물리 상한 위반이 구조적으로 사라지고
      (2) DOI별 성능 분산(음수 R² 다발)이 개선되는가?

공정 비교 설계 — 다음을 기존 v60 과 '완전히 동일'하게 유지한다:
  - 데이터: 188행 / 36 DOI (v55_152 + new36), 동일 소스 파일
  - 분할: DOI GroupKFold(5), v60 과 동일한 시드/구현
  - 피처: Model B 16개 + physical_cap_mg_g (feature_set='cap'), 동일 전처리
  - 알고리즘: 동일 XGBoost 설정 (s2_model)
오직 '학습 표적'만 바꾼다:
  - BASELINE : 표적 = qe,  예측 후 np.clip(raw, 0, cap)   (= 기존 활성 방식 재현)
  - FRACTION : 표적 = qe/cap (0~1로 clip 학습), 예측 f_hat∈[0,1] → qe_hat = f_hat * cap

활성 배포 모델·원본 데이터·기존 리포트를 변경하지 않는다. 결과는 reports/v65_fraction_target/.
"""
from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import sklearn
import xgboost

from scripts.build_v46_phase8 import features, s2_model
from scripts.run_v58_failure_diagnostics import build, derive
from scripts.run_v59_doi_material_weighting import OLD, PRE, NEW, fit_predict

REPORT = "reports/v65_fraction_target"
VARIANT = "B"          # 활성 핵심 모델과 동일 (제조조건 직접 예측)
FEATURE_SET = "cap"    # physical_cap_mg_g 포함 (기존 B_doi_material 과 동일 피처)
POLICY = "row_equal"   # 공정 비교를 위해 행 동일가중(표적 효과만 분리)
N_SPLITS = 5
EPS = 1e-9


def _fit_fraction(train, test):
    """표적을 흡착분율 f=qe/cap 로 바꿔 학습하고 qe 로 역변환해 예측한다.

    기존 fit_predict 와 동일한 파이프라인(build+features)을 쓰되 표적만 교체한다.
    """
    tr = derive(train, FEATURE_SET)
    te = derive(test, FEATURE_SET)
    model, cols, nums, cats = build(VARIANT, FEATURE_SET)

    cap_tr = (train.C0_mg_L / train.dose_g_L).to_numpy(float)
    y_qe_tr = train.target_value.to_numpy(float)
    # 분율 표적 (학습 표적만 [0,1] 로 클립; 소수의 보고 qe>cap 이상치 방어)
    frac_tr = np.clip(y_qe_tr / cap_tr, 0.0, 1.0)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        model.fit(features(tr, cols, nums, cats), frac_tr)
        frac_hat = np.asarray(model.predict(features(te, cols, nums, cats)), float)

    cap_te = (test.C0_mg_L / test.dose_g_L).to_numpy(float)
    frac_hat = np.clip(frac_hat, 0.0, 1.0)       # 구조적 상한 보장
    qe_hat = frac_hat * cap_te                    # 역변환
    raw = qe_hat.copy()                           # 분율 방식은 raw==capped (위반 불가능)
    return qe_hat, raw


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
            ss_res = float(np.sum(err**2))
            ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
        else:
            r2 = None
        rows.append({"doi": str(d), "n": n, "mae": round(mae, 3),
                     "r2": round(r2, 4) if r2 is not None else None})
    return pd.DataFrame(rows)


def dist_summary(doi_table, label):
    maes = doi_table["mae"].dropna().to_numpy()
    r2s = doi_table["r2"].dropna().to_numpy()
    return {
        "method": label,
        "n_doi": int(len(doi_table)),
        "DOI_MAE_median": round(float(np.median(maes)), 3),
        "DOI_MAE_q75": round(float(np.percentile(maes, 75)), 3),
        "DOI_MAE_worst": round(float(np.max(maes)), 3),
        "DOI_R2_median": round(float(np.median(r2s)), 4) if len(r2s) else None,
        "DOI_R2_worst": round(float(np.min(r2s)), 4) if len(r2s) else None,
        "n_DOI_negative_R2": int(np.sum(r2s < 0)) if len(r2s) else None,
    }


def main(root: str) -> None:
    root = Path(root).resolve()
    dst = root / REPORT
    dst.mkdir(parents=True, exist_ok=True)

    v152 = pd.read_csv(root / PRE, low_memory=False)
    new = pd.read_csv(root / NEW, low_memory=False)
    assert len(v152) == 152 and len(new) == 36
    assert not (set(v152.doi) & set(new.doi))
    d = pd.concat([v152, new], ignore_index=True, sort=False)
    assert len(d) == 188 and d.doi.nunique() == 36 and d.observation_id.is_unique

    y = d.target_value.to_numpy(float)
    cap = (d.C0_mg_L / d.dose_g_L).to_numpy(float)
    doi = d.doi.to_numpy(str)

    oof = d[["observation_id", "doi", "pollutant", "target_value",
             "C0_mg_L", "dose_g_L"]].copy()
    oof["cap"] = cap
    base_pred = np.full(len(d), np.nan)
    base_raw = np.full(len(d), np.nan)
    frac_pred = np.full(len(d), np.nan)

    # v60 과 동일한 DOI GroupKFold(5)
    for it, iv in GroupKFold(n_splits=N_SPLITS).split(d, groups=d.doi):
        held = set(d.iloc[iv].doi)
        tr = d[~d.doi.isin(held)].copy()
        te = d.iloc[iv].copy()
        assert set(tr.doi).isdisjoint(held)

        # BASELINE: 기존 방식 재현 (qe 직접 + 사후 clip). 동일 피처/가중.
        bp, braw = fit_predict(tr, te, VARIANT, FEATURE_SET, POLICY)
        base_pred[iv] = bp
        base_raw[iv] = braw

        # FRACTION: 분율 표적
        fp, _ = _fit_fraction(tr, te)
        frac_pred[iv] = fp

    oof["baseline_pred"] = base_pred
    oof["baseline_raw"] = base_raw
    oof["fraction_pred"] = frac_pred
    oof.to_csv(dst / "oof_predictions.csv", index=False, encoding="utf-8-sig")

    # ── 집계 지표 + 상한 위반 ─────────────────────────────────────────
    base_viol = int(np.sum(base_raw > cap + 1e-8))
    frac_viol = int(np.sum(frac_pred > cap + 1e-8))  # 구조적으로 0 이어야 함

    agg = [
        {"method": "BASELINE_qe_then_clip", **metrics(y, base_pred),
         "upper_cap_violation": base_viol,
         "violation_rate": round(base_viol / len(d), 4)},
        {"method": "FRACTION_qe_over_cap", **metrics(y, frac_pred),
         "upper_cap_violation": frac_viol,
         "violation_rate": round(frac_viol / len(d), 4)},
    ]
    agg_df = pd.DataFrame(agg)
    agg_df.to_csv(dst / "aggregate_metrics.csv", index=False, encoding="utf-8-sig")

    # ── DOI별 분포 ────────────────────────────────────────────────────
    base_doi = per_doi(y, base_pred, doi)
    frac_doi = per_doi(y, frac_pred, doi)
    base_doi["method"] = "BASELINE"
    frac_doi["method"] = "FRACTION"
    pd.concat([base_doi, frac_doi], ignore_index=True).to_csv(
        dst / "per_doi_metrics.csv", index=False, encoding="utf-8-sig")

    dist = pd.DataFrame([
        dist_summary(base_doi, "BASELINE_qe_then_clip"),
        dist_summary(frac_doi, "FRACTION_qe_over_cap"),
    ])
    dist.to_csv(dst / "doi_distribution_summary.csv", index=False, encoding="utf-8-sig")

    # ── DOI 단위 페어드 부트스트랩: FRACTION 이 BASELINE 대비 MAE 개선? ──
    ae_base = np.abs(y - base_pred)
    ae_frac = np.abs(y - frac_pred)
    gain = ae_base - ae_frac  # 양수면 FRACTION 이 더 정확
    ids = np.unique(doi)
    idx = [np.flatnonzero(doi == g) for g in ids]
    rng = np.random.default_rng(20261031)
    dist_gain = []
    for _ in range(3000):
        sel = rng.integers(0, len(ids), len(ids))
        pick = np.concatenate([idx[j] for j in sel])
        dist_gain.append(float(gain[pick].mean()))
    lo, hi = (float(q) for q in np.quantile(dist_gain, [0.025, 0.975]))
    boot = {
        "mean_MAE_gain_fraction_minus_baseline_mg_g": round(float(gain.mean()), 4),
        "gain_ci95_low": round(lo, 4),
        "gain_ci95_high": round(hi, 4),
        "positive_means_fraction_better": True,
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
    }

    manifest = {
        "version": "v65",
        "role": "RESEARCH_ONLY_NOT_ACTIVE",
        "hypothesis": "fraction target f=qe/(C0/dose) removes cap violations and stabilizes per-DOI performance",
        "fair_comparison": {
            "same_data": "188 rows / 36 DOI (v55_152 + new36)",
            "same_split": f"GroupKFold(DOI, n_splits={N_SPLITS})",
            "same_features": f"Model {VARIANT}, feature_set={FEATURE_SET}",
            "same_algorithm": "XGBoost s2_model (unchanged hyperparameters)",
            "only_changed": "learning target (qe vs qe/cap)",
            "weighting": POLICY,
        },
        "aggregate": agg,
        "doi_bootstrap": boot,
        "environment": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
    }
    (dst / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"aggregate": agg,
                      "distribution": dist.to_dict(orient="records"),
                      "bootstrap": boot}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    main(ap.parse_args().root)
