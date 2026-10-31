"""
run_v67_cap_weighted.py — 연구용(RESEARCH_ONLY_NOT_ACTIVE).

v65·v66 후속. 분율 표적(f = qe/cap, cap = C0/dose)은 상한위반 0%와 전형적 DOI 개선을 보였지만,
cap 이 큰 소수 DOI 에서 역변환 증폭으로 최악 DOI 오차가 커졌다.

원인: 분율 공간 MSE 는 모든 샘플의 분율 오차를 같은 무게로 다룬다. 그런데 qe 오차는
    q_hat - q = cap * (f_hat - f)
라서 cap 배로 커진다. 따라서
    sum_i cap_i^2 * (f_hat_i - f_i)^2 = sum_i (q_hat_i - q_i)^2
즉 분율 표적에 sample_weight = cap^2 을 주면 학습 손실이 qe 공간 MSE 와 같아지고,
출력은 [0,1] 분율이므로 상한 위반이 생기지 않는다.

사전 지정한 비교 (가중치 w = cap^alpha, 학습 fold 안에서 평균 1로 정규화):
  alpha = 0   : v65/v66 FRACTION 과 같음 (가중 없음)
  alpha = 0.5 : 중간
  alpha = 1   : 중간
  alpha = 2   : qe 공간 MSE 와 같은 손실
  NESTED      : 외부 fold 마다 학습 DOI 안에서 내부 GroupKFold(4)로 alpha 를 고른다
                (내부 OOF qe MAE 최소). 평가 fold 를 alpha 선택에 쓰지 않으므로 1차 비교 대상이다.
고정 alpha 4개는 같은 OOF 로 고르면 사후 선택이 되므로 2차(참고) 결과다.

데이터(188행/36 DOI)·분할(DOI GroupKFold 5)·피처(Model B + cap)·알고리즘(XGBoost s2_model)은
v60/v65/v66 과 같다. 활성 배포 모델·원본 데이터·기존 리포트는 바꾸지 않는다.

실행: python -m scripts.run_v67_cap_weighted --root .
"""
from __future__ import annotations

import json
import platform
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.model_selection import GroupKFold

from scripts.build_v46_phase8 import features
from scripts.run_v58_failure_diagnostics import build, derive
from scripts.run_v59_doi_material_weighting import NEW, PRE, fit_predict
from scripts.run_v66_logit_fraction import boot_gain, dist_summary, metrics, per_doi

REPORT = "reports/v67_cap_weighted"
VARIANT, FEATURE_SET, POLICY = "B", "cap", "row_equal"
ALPHAS = (0.0, 0.5, 1.0, 2.0)
OUTER_SPLITS, INNER_SPLITS = 5, 4
BASE = "BASELINE_qe_then_clip"
NESTED = "NESTED_alpha"
# v65·v66 에서 분율 표적의 역변환 증폭이 가장 컸던 DOI
TAIL_DOIS = ("10.1371/journal.pone.0218114", "10.1016/j.psep.2017.10.014")


def cap_of(frame: pd.DataFrame) -> np.ndarray:
    return (frame.C0_mg_L / frame.dose_g_L).to_numpy(float)


def label(alpha: float) -> str:
    return f"CAPW_a{alpha:g}"


def fit_capw(train: pd.DataFrame, test: pd.DataFrame, alpha: float) -> np.ndarray:
    """분율 표적 + cap^alpha 가중 학습. 예측 분율을 [0,1]로 자른 뒤 cap 을 곱한다."""
    tr, te = derive(train, FEATURE_SET), derive(test, FEATURE_SET)
    model, cols, nums, cats = build(VARIANT, FEATURE_SET)
    cap_tr = cap_of(train)
    f_tr = np.clip(train.target_value.to_numpy(float) / cap_tr, 0.0, 1.0)
    w = cap_tr**alpha
    w = w * (len(w) / w.sum())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        model.fit(features(tr, cols, nums, cats), f_tr, model__sample_weight=w)
        f_hat = np.asarray(model.predict(features(te, cols, nums, cats)), float)
    return np.clip(f_hat, 0.0, 1.0) * cap_of(test)


def choose_alpha(train: pd.DataFrame) -> tuple[float, dict[float, float]]:
    """학습 DOI 안에서만 내부 GroupKFold 로 alpha 를 고른다 (qe MAE 최소, 동률이면 작은 alpha)."""
    inner = {a: np.full(len(train), np.nan) for a in ALPHAS}
    for it, iv in GroupKFold(n_splits=INNER_SPLITS).split(train, groups=train.doi):
        tr_in, te_in = train.iloc[it], train.iloc[iv]
        assert set(tr_in.doi).isdisjoint(set(te_in.doi))
        for a in ALPHAS:
            inner[a][iv] = fit_capw(tr_in, te_in, a)
    y = train.target_value.to_numpy(float)
    maes = {a: float(np.mean(np.abs(y - inner[a]))) for a in ALPHAS}
    return min(ALPHAS, key=lambda a: (maes[a], a)), maes


def plot_all(dst: Path, agg: pd.DataFrame, dist: pd.DataFrame, tail: pd.DataFrame) -> None:
    methods = list(agg.method)
    short = [m.replace("_qe_then_clip", "").replace("CAPW_", "") for m in methods]

    # 1) DOI MAE median / q75 / worst
    x = np.arange(len(methods))
    w = 0.27
    ds = dist.set_index("method")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w, [ds.loc[m, "DOI_MAE_median"] for m in methods], w, label="median")
    ax.bar(x, [ds.loc[m, "DOI_MAE_q75"] for m in methods], w, label="q75")
    ax.bar(x + w, [ds.loc[m, "DOI_MAE_worst"] for m in methods], w, label="worst")
    ax.set_xticks(x)
    ax.set_xticklabels(short)
    ax.set_ylabel("per-DOI MAE (mg/g)")
    ax.set_title("Per-DOI MAE by cap weight (w = cap^alpha), 188 rows / 36 DOI")
    ax.legend()
    fig.tight_layout()
    fig.savefig(dst / "doi_mae_summary_bars.png", dpi=130)
    plt.close(fig)

    # 2) aggregate MAE vs alpha
    a_vals = list(ALPHAS)
    ag = agg.set_index("method")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(a_vals, [ag.loc[label(a), "mae"] for a in a_vals], "o-", label="fixed alpha (fraction target)")
    ax.axhline(ag.loc[BASE, "mae"], color="#d62728", ls="--", label="BASELINE (qe + clip)")
    ax.axhline(ag.loc[NESTED, "mae"], color="#2ca02c", ls=":", label="NESTED alpha")
    ax.set_xlabel("alpha (weight = cap^alpha)")
    ax.set_ylabel("aggregate MAE (mg/g)")
    ax.set_title("Aggregate MAE vs cap-weight exponent")
    ax.legend()
    fig.tight_layout()
    fig.savefig(dst / "aggregate_mae_vs_alpha.png", dpi=130)
    plt.close(fig)

    # 3) tail DOIs
    fig, ax = plt.subplots(figsize=(10, 4.5))
    xt = np.arange(len(methods))
    tl = tail.set_index("doi")  # "CAPW_a0.5" 같은 열 이름은 itertuples 로 접근할 수 없다
    for i, t in enumerate(tl.index):
        vals = [float(tl.loc[t, m]) for m in methods]
        ax.bar(xt + (i - 0.5) * 0.38, vals, 0.38, label=t)
    ax.set_xticks(xt)
    ax.set_xticklabels(short)
    ax.set_ylabel("DOI MAE (mg/g)")
    ax.set_title("Tail DOIs where fraction target blew up in v65/v66")
    ax.legend()
    fig.tight_layout()
    fig.savefig(dst / "tail_doi_mae.png", dpi=130)
    plt.close(fig)


def main(root: str) -> None:
    root = Path(root).resolve()
    dst = root / REPORT
    dst.mkdir(parents=True, exist_ok=True)

    d = pd.concat([pd.read_csv(root / PRE, low_memory=False), pd.read_csv(root / NEW, low_memory=False)],
                  ignore_index=True, sort=False)
    assert len(d) == 188 and d.doi.nunique() == 36 and d.observation_id.is_unique
    y, cap, doi = d.target_value.to_numpy(float), cap_of(d), d.doi.to_numpy(str)

    pred = {BASE: np.full(len(d), np.nan)}
    for a in ALPHAS:
        pred[label(a)] = np.full(len(d), np.nan)
    pred[NESTED] = np.full(len(d), np.nan)
    base_raw = np.full(len(d), np.nan)
    fold_log = []

    for k, (_, iv) in enumerate(GroupKFold(n_splits=OUTER_SPLITS).split(d, groups=d.doi), 1):
        held = set(d.iloc[iv].doi)
        tr = d[~d.doi.isin(held)].copy()
        te = d.iloc[iv].copy()
        assert set(tr.doi).isdisjoint(held)

        bp, braw = fit_predict(tr, te, VARIANT, FEATURE_SET, POLICY)
        pred[BASE][iv] = bp
        base_raw[iv] = braw

        outer = {}
        for a in ALPHAS:
            outer[a] = fit_capw(tr, te, a)
            pred[label(a)][iv] = outer[a]

        a_star, inner_maes = choose_alpha(tr)
        # 같은 학습 데이터와 같은 alpha 이므로 CAPW_a{a_star} 의 외부 예측과 같은 모델이다 (고정 시드).
        pred[NESTED][iv] = outer[a_star]
        fold_log.append({"fold": k, "n_test": int(len(iv)), "n_test_doi": len(held),
                         "chosen_alpha": a_star,
                         "inner_mae": {f"{a:g}": round(m, 4) for a, m in inner_maes.items()}})

    for name, p in pred.items():
        assert np.isfinite(p).all(), name

    viol = {name: int(np.sum(p > cap + 1e-8)) for name, p in pred.items()}
    viol[BASE] = int(np.sum(base_raw > cap + 1e-8))  # 기준선은 clipping 전 원시 예측 기준

    agg_rows, doi_tables = [], {}
    for name, p in pred.items():
        agg_rows.append({"method": name, **metrics(y, p), "upper_cap_violation": viol[name],
                         "violation_rate": round(viol[name] / len(d), 4)})
        doi_tables[name] = per_doi(y, p, doi)
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(dst / "aggregate_metrics.csv", index=False, encoding="utf-8-sig")

    per = pd.concat([t.assign(method=n) for n, t in doi_tables.items()], ignore_index=True)
    per.to_csv(dst / "per_doi_metrics.csv", index=False, encoding="utf-8-sig")

    dist = pd.DataFrame([dist_summary(doi_tables[n], n) for n in pred])
    dist.to_csv(dst / "doi_distribution_summary.csv", index=False, encoding="utf-8-sig")

    tail_rows = []
    for t in TAIL_DOIS:
        row = {"doi": t, "n": int((doi == t).sum())}
        for name, tbl in doi_tables.items():
            row[name] = float(tbl.loc[tbl.doi == t, "mae"].iloc[0])
        tail_rows.append(row)
    tail = pd.DataFrame(tail_rows)
    tail.to_csv(dst / "tail_doi_mae.csv", index=False, encoding="utf-8-sig")

    boots = {
        "PRIMARY_NESTED_vs_BASELINE": boot_gain(y, pred[BASE], pred[NESTED], doi),
        "NESTED_vs_CAPW_a0": boot_gain(y, pred[label(0.0)], pred[NESTED], doi),
    }
    for a in ALPHAS:
        boots[f"POSTHOC_{label(a)}_vs_BASELINE"] = boot_gain(y, pred[BASE], pred[label(a)], doi)

    oof = d[["observation_id", "doi", "pollutant", "target_value", "C0_mg_L", "dose_g_L"]].copy()
    oof["cap"] = cap
    oof["baseline_raw"] = base_raw
    for name, p in pred.items():
        oof[name] = p
    oof.to_csv(dst / "oof_predictions.csv", index=False, encoding="utf-8-sig")

    plot_all(dst, agg, dist, tail)

    manifest = {
        "version": "v67",
        "role": "RESEARCH_ONLY_NOT_ACTIVE",
        "hypothesis": "cap^alpha sample weights on the fraction target suppress inverse-transform "
                      "blowups while keeping 0% cap violations",
        "identity": "sum cap^2 (f_hat-f)^2 == sum (q_hat-q)^2, so alpha=2 equals qe-space MSE",
        "design": {
            "data": "188 rows / 36 DOI (v55_152 + new36)",
            "outer_split": f"GroupKFold(DOI, {OUTER_SPLITS})",
            "inner_split_for_alpha": f"GroupKFold(DOI, {INNER_SPLITS}) within outer training DOIs",
            "alphas": list(ALPHAS),
            "features": f"Model {VARIANT}, feature_set={FEATURE_SET}",
            "algorithm": "XGBoost s2_model (unchanged hyperparameters)",
            "primary_comparison": "NESTED_alpha vs BASELINE (DOI paired bootstrap of MAE gain)",
            "posthoc_note": "fixed-alpha results share the evaluation OOF; picking the best is post-hoc",
        },
        "folds": fold_log,
        "aggregate": agg_rows,
        "distribution": dist.to_dict(orient="records"),
        "tail_dois": tail_rows,
        "doi_bootstrap": boots,
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                        "xgboost": xgboost.__version__},
    }
    (dst / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    compact = {
        "aggregate": [{k: r[k] for k in ("method", "mae", "rmse", "r2", "violation_rate")} for r in agg_rows],
        "distribution": dist.to_dict(orient="records"),
        "tail": tail_rows,
        "chosen_alpha_per_fold": [f["chosen_alpha"] for f in fold_log],
        "bootstrap": boots,
    }
    print(json.dumps(compact, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    main(ap.parse_args().root)
