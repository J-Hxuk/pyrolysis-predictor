"""
verify_stage1_challenge.py — Stage1 봉인 챌린지 결과 독립 재계산·검증 + 그래프.

목적: 챌린지 폴더의 동결 예측(02_*.csv)으로부터 집계·DOI별·route별 지표를 직접
다시 계산해, 보고된 요약(04)·DOI 분해(03) 수치와 일치하는지 검증한다.
또 route별(none/physical/chemical) 오차 패턴을 그래프로 만든다.

읽기 전용 검증. 챌린지 데이터·활성 모델을 변경하지 않는다.
출력: reports/stage1_sealed_challenge/ (그래프·검증표)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
CH = ROOT / "data/imports/Biochar_Stage1_v60/Biochar_Stage1_v60_Independent_Challenge"
OUT = ROOT / "reports/stage1_sealed_challenge"
OUT.mkdir(parents=True, exist_ok=True)


def metrics(y, p):
    err = p - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    r2 = 1.0 - float(np.sum(err**2)) / float(np.sum((y - np.mean(y)) ** 2)) if len(y) >= 2 and np.var(y) > 0 else None
    return {"n": int(len(y)), "mae": round(mae, 4), "rmse": round(rmse, 4),
            "r2": round(r2, 6) if r2 is not None else None}


def verify_target(tag):
    pred = pd.read_csv(CH / f"02_{tag}_V59_Model_Untouched_Challenge_Predictions.csv")
    y = pred["actual"].to_numpy(float)
    p = pred["prediction"].to_numpy(float)
    overall = metrics(y, p)

    # DOI별 재계산
    per_doi = []
    for d, g in pred.groupby("doi"):
        per_doi.append({"doi": str(d), **metrics(g["actual"].to_numpy(float), g["prediction"].to_numpy(float))})
    per_doi = pd.DataFrame(per_doi)

    # route별 재계산 (예측 CSV의 v59_route_family)
    per_route = []
    if "v59_route_family" in pred.columns:
        for r, g in pred.groupby("v59_route_family"):
            yy = g["actual"].to_numpy(float)
            pp = g["prediction"].to_numpy(float)
            per_route.append({"route": str(r), **metrics(yy, pp),
                              "mean_signed_residual": round(float(np.mean(pp - yy)), 3)})
    per_route = pd.DataFrame(per_route)

    # 보고값(03/04)과 대조
    reported_doi = pd.read_csv(CH / f"03_{tag}_Challenge_By_DOI.csv")
    summ = pd.read_csv(CH / "04_V60_Challenge_Summary.csv")
    rep = summ[summ.target == tag].iloc[0]

    check = {
        "target": tag,
        "recomputed_overall": overall,
        "reported_overall": {"n": int(rep.n), "mae": round(float(rep.MAE), 4),
                             "rmse": round(float(rep.RMSE), 4), "r2": round(float(rep.R2), 6)},
        "overall_match": bool(abs(overall["r2"] - float(rep.R2)) < 1e-3
                              and abs(overall["mae"] - float(rep.MAE)) < 1e-2),
        "doi_match": [],
    }
    for _, rr in reported_doi.iterrows():
        mine = per_doi[per_doi.doi == rr.doi].iloc[0]
        check["doi_match"].append({
            "doi": rr.doi,
            "recomputed_r2": mine["r2"], "reported_r2": round(float(rr.R2), 6),
            "match": bool(abs(mine["r2"] - float(rr.R2)) < 1e-3),
        })
    return pred, overall, per_doi, per_route, check


def main():
    checks = {}
    allroute = {}
    for tag in ("BET", "TPV"):
        pred, overall, per_doi, per_route, check = verify_target(tag)
        checks[tag] = check
        allroute[tag] = per_route
        per_doi.to_csv(OUT / f"{tag}_per_doi_recomputed.csv", index=False, encoding="utf-8-sig")
        per_route.to_csv(OUT / f"{tag}_per_route_recomputed.csv", index=False, encoding="utf-8-sig")

    (OUT / "verification.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 그래프: BET route별 서명 잔차 (과대/과소 예측 방향) ──────────
    predB = pd.read_csv(CH / "02_BET_V59_Model_Untouched_Challenge_Predictions.csv")
    predB["residual"] = predB["prediction"] - predB["actual"]
    order = ["none", "physical", "chemical"]
    colors = {"none": "#1f77b4", "physical": "#ff7f0e", "chemical": "#d62728"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, r in enumerate(order):
        sub = predB[predB.v59_route_family == r]
        if len(sub) == 0:
            continue
        x = np.full(len(sub), i) + np.random.default_rng(0).normal(0, 0.05, len(sub))
        ax.scatter(x, sub["residual"], s=40, alpha=0.7, color=colors[r], label=r)
    ax.axhline(0, color="k", lw=1, ls="--", label="perfect (residual=0)")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylabel("Signed residual = pred - actual (m2/g)")
    ax.set_xlabel("v59 route family")
    ax.set_title("Stage1 BET sealed challenge — residual by activation route\n"
                 "chemical: over-predict (+), physical: under-predict (-)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "bet_residual_by_route.png", dpi=130)
    plt.close(fig)

    # ── 그래프: 예측 vs 실제 (BET), route 색상 ─────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for r in order:
        sub = predB[predB.v59_route_family == r]
        ax.scatter(sub["actual"], sub["prediction"], s=45, alpha=0.75, color=colors[r], label=r)
    lim = max(predB["actual"].max(), predB["prediction"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="y=x")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Actual BET (m2/g)")
    ax.set_ylabel("Predicted BET (m2/g)")
    ax.set_title("Stage1 BET sealed challenge — predicted vs actual (23 rows, 3 sealed DOIs)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "bet_pred_vs_actual.png", dpi=130)
    plt.close(fig)

    print(json.dumps(checks, ensure_ascii=False, indent=2))
    print("\n--- BET per-route ---")
    print(allroute["BET"].to_string(index=False))


if __name__ == "__main__":
    main()
