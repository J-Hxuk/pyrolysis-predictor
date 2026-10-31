"""
v16_validate.py — Stage2 성능 독립 재검증·Baseline benchmark (v15 모델/데이터 보존).

이 모듈은 새 배포 모델을 추가하지 않는다. v15 의 Model A/B/C 입력 계약과 데이터를 그대로
쓰면서 다음을 수행한다.

  - 물리 기반 Baseline: qe = (C0/dose) × 학습 fold 평균 흡착률.  흡착률 f = qe·dose/C0.
    (타깃/단위 정의가 적합할 때만. 평균 f 는 학습 fold 로만 추정 — 누출 방지.)
  - 동일 행·동일 그룹 fold 에서 Dummy/Ridge/RF/XGBoost 와 공정 비교.
  - A vs B 동일 관측 오차 차이 + DOI cluster bootstrap 95% CI.
  - 제조조건 ablation(동일 fold).
  - 무차원 흡착률 f 타깃 예측(0≤f≤1 보장) vs qe 직접 예측 비교.

Streamlit 을 import 하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pyrolysis.biochar_stage2 import v15_features as fc
from pyrolysis.biochar_stage2 import v15_models as M

__all__ = [
    "RANDOM_STATE",
    "physical_fraction_baseline_oof",
    "oof_predictions",
    "metrics",
    "benchmark_baselines",
    "ablation_manufacturing",
    "fraction_target_experiment",
    "cluster_bootstrap_ci",
    "nested_cv_model_selection",
    "lofo_doi_overlap_audit",
]

RANDOM_STATE = 42


def validation_algorithms() -> dict[str, Callable[[], Any]]:
    """검증용(연산량 절감) 알고리즘 집합.

    v15 배포 모델 설정은 그대로 두되, v16 의 반복 CV(중첩·부트스트랩)가 현실적 시간에
    끝나도록 트리 수를 줄인 보수적 설정을 쓴다. 상대 비교·선택 결론에는 영향이 작다.
    DummyRegressor/Ridge 는 동일. CatBoost 는 환경 있으면 포함.
    """
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    algos: dict[str, Callable[[], Any]] = {
        "Dummy": lambda: DummyRegressor(strategy="mean"),
        "Ridge": lambda: Ridge(alpha=10.0, random_state=RANDOM_STATE),
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=150, max_depth=6, min_samples_leaf=3,
            max_features="sqrt", random_state=RANDOM_STATE, n_jobs=1),
        "ExtraTrees": lambda: ExtraTreesRegressor(
            n_estimators=150, max_depth=6, min_samples_leaf=3,
            max_features="sqrt", random_state=RANDOM_STATE, n_jobs=1),
    }
    try:
        from xgboost import XGBRegressor
        algos["XGBoost"] = lambda: XGBRegressor(
            n_estimators=150, max_depth=2, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0,
            min_child_weight=5, random_state=RANDOM_STATE, n_jobs=1,
            tree_method="hist", verbosity=0)
    except Exception:
        pass
    return algos


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    m = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt, yp = y_true[m], y_pred[m]
    if len(yt) == 0:
        return {"n": 0, "r2": None, "mae": None, "rmse": None}
    r2 = float(r2_score(yt, yp)) if (len(yt) >= 2 and np.var(yt) > 0) else None
    return {"n": int(len(yt)), "r2": r2,
            "mae": float(mean_absolute_error(yt, yp)),
            "rmse": float(np.sqrt(mean_squared_error(yt, yp)))}


def _c0_dose(df: pd.DataFrame):
    c0 = pd.to_numeric(df[fc.C0_COL], errors="coerce").to_numpy(dtype=float)
    dose = pd.to_numeric(df[fc.DOSE_COL], errors="coerce").to_numpy(dtype=float)
    cap = np.divide(c0, dose, out=np.full_like(c0, np.nan), where=(dose != 0))
    return c0, dose, cap


def physical_fraction_baseline_oof(df: pd.DataFrame, scheme: str) -> np.ndarray:
    """물리 Baseline OOF: qe_hat = (C0/dose) × (학습 fold 평균 흡착률 f).

    f = qe·dose/C0. 평균 f 는 **학습 fold 로만** 추정해 누출을 막는다.
    """
    y = pd.to_numeric(df[fc.TARGET_COLUMN], errors="coerce").to_numpy(dtype=float)
    c0, dose, cap = _c0_dose(df)
    f = np.divide(y, cap, out=np.full_like(y, np.nan), where=np.isfinite(cap) & (cap != 0))
    n = len(df)
    oof = np.full(n, np.nan)
    for _, tr, te in M.make_splits(df, scheme):
        f_train = f[tr]
        f_train = f_train[np.isfinite(f_train)]
        if len(f_train) == 0:
            continue
        mean_f = float(np.mean(f_train))
        oof[te] = cap[te] * mean_f
    return oof


def oof_predictions(df: pd.DataFrame, model_name: str, algo_name: str,
                    algo_factory: Callable[[], Any], scheme: str,
                    *, include_raw_composition: bool = False):
    """ML 모델 OOF 예측 배열과 그룹 라벨을 반환한다 (전처리 fold 내 fit)."""
    input_cols = [c for c in fc.model_inputs(model_name, include_raw_composition=include_raw_composition)
                  if c in df.columns]
    y = pd.to_numeric(df[fc.TARGET_COLUMN], errors="coerce").to_numpy(dtype=float)
    n = len(df)
    oof = np.full(n, np.nan)
    grp = np.empty(n, dtype=object)
    for gname, tr, te in M.make_splits(df, scheme):
        pipe = M.build_pipeline(algo_factory(), input_cols)
        pipe.fit(df.iloc[tr][input_cols], y[tr])
        oof[te] = np.asarray(pipe.predict(df.iloc[te][input_cols]), dtype=float)
        grp[te] = gname
    return oof, grp


def benchmark_baselines(df: pd.DataFrame, *, schemes=("LODO", "GroupKFold_DOI")) -> dict[str, Any]:
    """Baseline(Dummy/물리) vs Ridge/RF/XGBoost, Model A/B 를 동일 fold 에서 비교."""
    algos = validation_algorithms()
    want = {k: algos[k] for k in ("Dummy", "Ridge", "RandomForest", "XGBoost") if k in algos}
    y = pd.to_numeric(df[fc.TARGET_COLUMN], errors="coerce").to_numpy(dtype=float)
    out: dict[str, Any] = {"schemes": list(schemes), "results": {}}
    for scheme in schemes:
        out["results"][scheme] = {}
        # 물리 baseline (모델 비종속)
        phys = physical_fraction_baseline_oof(df, scheme)
        out["results"][scheme]["PhysicalMeanFraction"] = {
            "model": "physical", **metrics(y, phys),
            "note": "qe=(C0/dose)×학습fold평균흡착률. 흡착화학 학습 아님(스케일 baseline)."}
        for model_name in ("A", "B"):
            for aname, afac in want.items():
                oof, _ = oof_predictions(df, model_name, aname, afac, scheme)
                out["results"][scheme][f"{model_name}:{aname}"] = {
                    "model": model_name, "algo": aname, **metrics(y, oof)}
    return out


def ablation_manufacturing(df: pd.DataFrame, algo_name="XGBoost", scheme="LODO") -> dict[str, Any]:
    """동일 fold 에서 A(기준)에 제조조건을 하나씩 '추가'했을 때 오차 변화(ablation).

    또한 B 전체, B 에서 조건을 하나씩 '제외'했을 때의 변화를 함께 본다.
    결측률·사용 가능 표본 수를 함께 보고해 결측 효과와 실제 기여를 구분한다.
    """
    algos = validation_algorithms()
    afac = algos.get(algo_name, algos["Ridge"])
    y = pd.to_numeric(df[fc.TARGET_COLUMN], errors="coerce").to_numpy(dtype=float)

    def eval_cols(cols):
        cols = [c for c in cols if c in df.columns]
        n = len(df); oof = np.full(n, np.nan)
        for _, tr, te in M.make_splits(df, scheme):
            pipe = M.build_pipeline(afac(), cols)
            pipe.fit(df.iloc[tr][cols], y[tr])
            oof[te] = np.asarray(pipe.predict(df.iloc[te][cols]), dtype=float)
        return metrics(y, oof)

    base_A = fc.ADSORPTION_INPUTS
    full_B = fc.model_inputs("B")
    manu = fc.MANUFACTURING_INPUTS

    base_A_metrics = eval_cols(base_A)
    out: dict[str, Any] = {"algo": algo_name, "scheme": scheme,
                           "A_only": base_A_metrics, "B_full": eval_cols(full_B),
                           "add_one_to_A": {}, "drop_one_from_B": {},
                           "delta_vs_A_only": {}, "variable_info": {},
                           "baseline_consistency": {
                               "baseline": "A_only",
                               "same_algo": algo_name,
                               "same_scheme": scheme,
                               "same_rows_and_folds": True,
                               "note": ("add_one_to_A·B_full·drop_one_from_B·A_only 모두 동일 "
                                        "eval_cols(동일 algo·동일 scheme·동일 행/fold)로 산출. "
                                        "delta 는 같은 run 의 A_only 기준. 서로 다른 평가절차 점수를 섞지 않음.")}}

    def _delta(m: dict[str, Any]) -> dict[str, Any]:
        # ΔR²/ΔMAE = (A+feature) − (same-run A_only). 양의 ΔR²/음의 ΔMAE = 기여 있음.
        dr2 = (None if (m.get("r2") is None or base_A_metrics.get("r2") is None)
               else round(m["r2"] - base_A_metrics["r2"], 6))
        dmae = (None if (m.get("mae") is None or base_A_metrics.get("mae") is None)
                else round(m["mae"] - base_A_metrics["mae"], 6))
        return {"delta_r2_vs_A": dr2, "delta_mae_vs_A": dmae}

    for c in manu:
        add_m = eval_cols(base_A + [c])
        out["add_one_to_A"][c] = add_m
        out["drop_one_from_B"][c] = eval_cols([x for x in full_B if x != c])
        out["delta_vs_A_only"][c] = _delta(add_m)
        s = df[c] if c in df.columns else pd.Series(dtype=float)
        info = {"missing_pct": round(100 * float(s.isna().mean()), 1) if len(s) else 100.0,
                "n_available": int(s.notna().sum()) if len(s) else 0}
        num = pd.to_numeric(s, errors="coerce")
        if num.notna().any():
            info["n_unique"] = int(num.nunique())
        elif len(s):
            info["n_unique_categories"] = int(s.nunique())
        out["variable_info"][c] = info
    return out


def fraction_target_experiment(df: pd.DataFrame, algo_name="XGBoost",
                               schemes=("LODO", "GroupKFold_DOI")) -> dict[str, Any]:
    """무차원 흡착률 f=qe·dose/C0 타깃(0≤f≤1 보장) vs qe 직접 예측을 동일 fold 비교.

    f 예측 후 qe_hat = f_hat_clipped × (C0/dose). 원시 qe 직접 예측의 물리위반율과 비교.
    """
    algos = validation_algorithms()
    afac = algos.get(algo_name, algos["Ridge"])
    y = pd.to_numeric(df[fc.TARGET_COLUMN], errors="coerce").to_numpy(dtype=float)
    c0, dose, cap = _c0_dose(df)
    f_true = np.divide(y, cap, out=np.full_like(y, np.nan), where=np.isfinite(cap) & (cap != 0))
    cols = [c for c in fc.model_inputs("A") if c in df.columns]
    n = len(df)
    out: dict[str, Any] = {"algo": algo_name, "model": "A", "results": {}}

    for scheme in schemes:
        # (i) qe 직접 예측
        oof_qe = np.full(n, np.nan)
        # (ii) f 예측 -> qe 복원
        oof_qe_from_f = np.full(n, np.nan)
        for _, tr, te in M.make_splits(df, scheme):
            p1 = M.build_pipeline(afac(), cols); p1.fit(df.iloc[tr][cols], y[tr])
            oof_qe[te] = np.asarray(p1.predict(df.iloc[te][cols]), dtype=float)
            ftr = f_true[tr]; ok = np.isfinite(ftr)
            if ok.sum() >= 2:
                p2 = M.build_pipeline(afac(), cols)
                p2.fit(df.iloc[tr][cols].iloc[ok], ftr[ok])
                f_pred = np.asarray(p2.predict(df.iloc[te][cols]), dtype=float)
                f_pred = np.clip(f_pred, 0.0, 1.0)  # 0<=f<=1 보장
                oof_qe_from_f[te] = f_pred * cap[te]
        # 물리 위반율(원시 qe 직접 예측)
        viol_qe = float(np.mean((oof_qe < -1e-9) | (oof_qe > cap * (1 + 1e-9))))
        # f 경로는 설계상 0..cap 이므로 위반 0
        viol_f = float(np.mean((oof_qe_from_f < -1e-9) | (oof_qe_from_f > cap * (1 + 1e-9))))
        out["results"][scheme] = {
            "qe_direct": {**metrics(y, oof_qe), "constraint_violation_rate": round(viol_qe, 4)},
            "f_rate_then_qe": {**metrics(y, oof_qe_from_f), "constraint_violation_rate": round(viol_f, 4)},
            "f_target_metrics": metrics(f_true, np.divide(oof_qe_from_f, cap,
                                        out=np.full(n, np.nan), where=np.isfinite(cap) & (cap != 0))),
        }
    return out


def selection_algorithms() -> dict[str, Callable[[], Any]]:
    """중첩 CV 의 내부 선택 후보(경량). RF/ExtraTrees 는 프로파일상 Ridge/XGBoost 보다
    느리고 성능도 낮아 내부 선택 루프에서 제외한다(선택 결론에 영향 작음, 연산량 대폭 절감).
    Dummy 를 포함해 '아무 신호 없음' 대비 선택이 되는지도 본다."""
    a = validation_algorithms()
    return {k: a[k] for k in ("Dummy", "Ridge", "XGBoost") if k in a}


def nested_cv_model_selection(df: pd.DataFrame, model_name: str,
                              *, outer_scheme="LODO") -> dict[str, Any]:
    """Nested CV: 외부 fold 마다 '내부 DOI 그룹 CV'로 알고리즘을 선택 후 외부 fold 를 평가.

    '단일 점수(최고 알고리즘 cherry-pick)'가 아니라, 모델 선택 과정을 포함한 정직한 OOF 성능을
    준다. A 의 R²=0.695 가 알고리즘 선택을 고려해도 유지되는지 확인하는 핵심 절차.

    내부 선택 기준: 내부 GroupKFold(DOI) OOF MAE 최소 알고리즘.
    """
    algos = selection_algorithms()
    input_cols = [c for c in fc.model_inputs(model_name) if c in df.columns]
    y = pd.to_numeric(df[fc.TARGET_COLUMN], errors="coerce").to_numpy(dtype=float)
    n = len(df)
    oof = np.full(n, np.nan)
    chosen = []

    for gname, tr, te in M.make_splits(df, outer_scheme):
        tr_df = df.iloc[tr].reset_index(drop=True)
        y_tr = y[tr]
        # 내부 선택: 학습 DOI 만으로 GroupKFold OOF MAE 비교
        best_algo, best_mae = None, np.inf
        inner_doi = tr_df[fc.GROUP_DOI].nunique()
        if inner_doi < 2:
            best_algo = "Ridge"  # 그룹이 1개면 선택 불가 → 보수적 기본값
        else:
            for aname, afac in algos.items():
                inner = np.full(len(tr_df), np.nan)
                for _, itr, ite in M.make_splits(tr_df, "GroupKFold_DOI"):
                    pipe = M.build_pipeline(afac(), input_cols)
                    pipe.fit(tr_df.iloc[itr][input_cols], y_tr[itr])
                    inner[ite] = np.asarray(pipe.predict(tr_df.iloc[ite][input_cols]), dtype=float)
                mm = metrics(y_tr, inner)
                if mm["mae"] is not None and mm["mae"] < best_mae:
                    best_mae, best_algo = mm["mae"], aname
        chosen.append(best_algo)
        # 선택한 알고리즘으로 외부 fold 전체 학습·예측
        pipe = M.build_pipeline(algos[best_algo](), input_cols)
        pipe.fit(tr_df[input_cols], y_tr)
        oof[te] = np.asarray(pipe.predict(df.iloc[te][input_cols]), dtype=float)

    from collections import Counter
    return {
        "model": model_name, "outer_scheme": outer_scheme,
        "nested_oof": metrics(y, oof),
        "chosen_algorithms": dict(Counter(chosen)),
        "note": ("Nested CV. 내부 DOI 그룹 CV 로 알고리즘 선택, 외부 fold 평가. "
                 "알고리즘 cherry-pick 을 배제한 정직한 OOF 성능."),
    }, oof


def lofo_doi_overlap_audit(df: pd.DataFrame) -> dict[str, Any]:
    """LOFO 각 fold 에서 train/test 의 DOI·feedstock 교집합을 집계한다.

    feedstock→DOI 가 내포(각 feedstock 이 1 DOI)라도 '하나의 DOI 에 여러 feedstock' 이면
    특정 feedstock 을 holdout 하는 동안 같은 DOI 의 다른 feedstock 이 train 에 남아 DOI
    독립성이 깨진다. 공유가 하나라도 있으면 LOFO 는 LODO 와 동등한 외삽 검증이 아니다.
    """
    doi = df[fc.GROUP_DOI].to_numpy()
    feed = df[fc.GROUP_FEEDSTOCK].to_numpy()
    per_fold = []
    n_overlap = 0
    for gname, tr, te in M.make_splits(df, "LOFO"):
        tr_doi, te_doi = set(doi[tr]), set(doi[te])
        shared = tr_doi & te_doi
        if shared:
            n_overlap += 1
        per_fold.append({
            "holdout_feedstock": str(gname),
            "n_test": int(len(te)),
            "test_doi": sorted(str(x) for x in te_doi),
            "n_shared_doi_with_train": int(len(shared)),
            "shared_doi": sorted(str(x) for x in shared),
            "test_feedstock_in_train": bool(set(feed[te]) & set(feed[tr])),
        })
    n_folds = len(per_fold)
    doi_independent = (n_overlap == 0)
    return {
        "n_folds": n_folds,
        "n_folds_with_doi_overlap": n_overlap,
        "doi_independent": doi_independent,
        "classification": (
            "LOFO == DOI-independent extrapolation (train/test DOI 분리)" if doi_independent
            else "new feedstock within partly seen DOI (DOI 독립 아님; LODO 와 동등 주장 불가)"),
        "note": (
            "DOI 와 원료가 내포돼 새 DOI 효과와 새 원료 효과를 통계적으로 분리하기 어렵다. "
            "DOI 공유가 있으면 LOFO 를 독립·외삽 검증으로 서술하지 않는다."),
        "per_fold": per_fold,
    }


def cluster_bootstrap_ci(df: pd.DataFrame, oof_a: np.ndarray, oof_b: np.ndarray,
                         *, n_boot=2000, seed=RANDOM_STATE) -> dict[str, Any]:
    """DOI 단위 cluster bootstrap 으로 (MAE_B - MAE_A) 의 95% CI 를 추정한다.

    DOI 를 복원추출로 재샘플해 그 안의 모든 관측을 포함, 각 부트스트랩에서 두 모델의 MAE 차이를
    계산한다. 양수면 B 가 더 나쁨(오차 큼), 음수면 B 가 더 나음.
    """
    y = pd.to_numeric(df[fc.TARGET_COLUMN], errors="coerce").to_numpy(dtype=float)
    doi = df[fc.GROUP_DOI].to_numpy()
    uniq = pd.unique(doi)
    err_a = np.abs(oof_a - y)
    err_b = np.abs(oof_b - y)
    valid = ~np.isnan(err_a) & ~np.isnan(err_b)

    rng = np.random.RandomState(seed)
    diffs = []
    idx_by_doi = {g: np.where((doi == g) & valid)[0] for g in uniq}
    for _ in range(n_boot):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_doi[g] for g in sampled if len(idx_by_doi[g])])
        if len(rows) == 0:
            continue
        diffs.append(float(np.mean(err_b[rows]) - np.mean(err_a[rows])))
    diffs = np.array(diffs)
    point = float(np.mean(err_b[valid]) - np.mean(err_a[valid]))
    return {
        "metric": "MAE_B - MAE_A (mg/g)", "point_estimate": round(point, 4),
        "ci95_low": round(float(np.percentile(diffs, 2.5)), 4),
        "ci95_high": round(float(np.percentile(diffs, 97.5)), 4),
        "n_boot": int(len(diffs)),
        "interpretation": (
            "양수 = B 오차가 A 보다 큼(개선 아님). CI 가 0 을 포함하면 차이 불확실."),
    }
