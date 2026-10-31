"""
analyze_doi_performance.py — Stage 2 qe 예측의 DOI별 성능 분포 분석.

목적: 집계 지표(전체 MAE/R²)가 가리는 "논문(DOI) 단위 성능 편차"를 드러낸다.
입력: reports/v60_separate_effects/all188_doi_oof_predictions.csv (OOF 예측, 재학습 없음)
출력: reports/doi_performance_distribution/ 아래
  - doi_performance_table.csv : DOI별 n, MAE, RMSE, R², 물리상한 위반율
  - model_distribution_summary.csv : 모델별 DOI 성능의 분포 요약(중앙값/사분위/최악)
  - doi_mae_distribution.png : DOI별 MAE 분포 (박스플롯 + 점)
  - doi_r2_distribution.png : DOI별 R² 분포
  - predicted_vs_actual.png : 예측 vs 실제 산점도 (활성 모델 B 기준)
  - RESULTS_KO.md : 한국어 요약

활성 배포 모델·데이터를 변경하지 않는 읽기 전용 분석이다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # 화면 없이 파일로만 저장
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "reports" / "v60_separate_effects" / "all188_doi_oof_predictions.csv"
OUT = ROOT / "reports" / "doi_performance_distribution"
OUT.mkdir(parents=True, exist_ok=True)

# 비교할 모델 예측 컬럼 (188행 시나리오; capped = 물리상한 clipping 적용)
# 라벨 -> 원시 예측 컬럼
MODELS = {
    "A_adsorption_only": "188__A_row",
    "B_direct_equal": "188__B_row",
    "B_direct_weighted": "188__B_doi_material",
}

TRUE_COL = "target_value"
GROUP_COL = "doi"


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """단일 그룹 지표. 샘플<2 또는 분산 0이면 R²=None."""
    n = len(y_true)
    if n == 0:
        return {"n": 0, "mae": None, "rmse": None, "r2": None}
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    if n >= 2 and np.var(y_true) > 0:
        ss_res = float(np.sum(err**2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    else:
        r2 = None
    return {"n": n, "mae": mae, "rmse": rmse, "r2": r2}


def physical_cap(pred: np.ndarray, c0: np.ndarray, dose: np.ndarray) -> np.ndarray:
    """0 <= qe <= C0/dose 로 제약한 예측값."""
    with np.errstate(divide="ignore", invalid="ignore"):
        cap = np.where((dose != 0) & np.isfinite(dose), c0 / dose, np.inf)
    return np.clip(pred, 0.0, cap)


def main() -> None:
    df = pd.read_csv(SRC)
    c0 = df["C0_mg_L"].to_numpy(dtype=float)
    dose = df["dose_g_L"].to_numpy(dtype=float)
    y_true = df[TRUE_COL].to_numpy(dtype=float)
    cap = np.where((dose != 0) & np.isfinite(dose), c0 / dose, np.inf)

    # ── 1. DOI별 성능 테이블 (모델 B 가중, capped 기준) ────────────────
    rows = []
    per_model_doi_metrics: dict[str, list[dict]] = {m: [] for m in MODELS}

    for label, col in MODELS.items():
        raw_pred = df[col].to_numpy(dtype=float)
        capped_pred = physical_cap(raw_pred, c0, dose)
        for doi, idx in df.groupby(GROUP_COL).groups.items():
            iloc = df.index.get_indexer(idx)
            yt = y_true[iloc]
            yp = capped_pred[iloc]
            m = metrics(yt, yp)
            # 물리 상한 위반율 (clipping 전 raw 기준)
            raw_g = raw_pred[iloc]
            cap_g = cap[iloc]
            finite = np.isfinite(cap_g)
            viol = int(np.sum(raw_g[finite] > cap_g[finite]))
            m_row = {
                "model": label,
                "doi": str(doi),
                **m,
                "constraint_violations": viol,
                "violation_rate": round(viol / m["n"], 4) if m["n"] else None,
            }
            rows.append(m_row)
            per_model_doi_metrics[label].append(m)

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "doi_performance_table.csv", index=False, encoding="utf-8-sig")

    # ── 2. 모델별 DOI 성능 분포 요약 ──────────────────────────────────
    summary_rows = []
    for label in MODELS:
        sub = table[table["model"] == label]
        maes = sub["mae"].dropna().to_numpy()
        r2s = sub["r2"].dropna().to_numpy()
        # 전체(행 단위) MAE — 집계 지표
        raw_pred = df[MODELS[label]].to_numpy(dtype=float)
        capped_pred = physical_cap(raw_pred, c0, dose)
        overall = metrics(y_true, capped_pred)
        n_viol = int(np.sum(raw_pred[np.isfinite(cap)] > cap[np.isfinite(cap)]))
        summary_rows.append(
            {
                "model": label,
                "n_doi": len(sub),
                "overall_MAE_capped": round(overall["mae"], 3),
                "overall_R2_capped": round(overall["r2"], 4) if overall["r2"] is not None else None,
                "DOI_MAE_median": round(float(np.median(maes)), 3),
                "DOI_MAE_q25": round(float(np.percentile(maes, 25)), 3),
                "DOI_MAE_q75": round(float(np.percentile(maes, 75)), 3),
                "DOI_MAE_worst": round(float(np.max(maes)), 3),
                "DOI_R2_median": round(float(np.median(r2s)), 4) if len(r2s) else None,
                "DOI_R2_worst": round(float(np.min(r2s)), 4) if len(r2s) else None,
                "n_DOI_negative_R2": int(np.sum(r2s < 0)) if len(r2s) else None,
                "overall_violation_rate": round(n_viol / len(df), 4),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "model_distribution_summary.csv", index=False, encoding="utf-8-sig")

    # ── 3. 그래프: DOI별 MAE 분포 (박스 + 지터 점) ────────────────────
    _boxplot(table, "mae", "DOI-level MAE (mg/g)", OUT / "doi_mae_distribution.png")
    _boxplot(table, "r2", "DOI-level R2", OUT / "doi_r2_distribution.png", ref_line=0.0)

    # ── 4. 예측 vs 실제 (활성 모델 B 가중, capped) ────────────────────
    raw_pred = df["188__B_doi_material"].to_numpy(dtype=float)
    capped_pred = physical_cap(raw_pred, c0, dose)
    _scatter(y_true, capped_pred, OUT / "predicted_vs_actual.png")

    # ── 5. 한국어 요약 ────────────────────────────────────────────────
    _write_markdown(summary, table)

    # 콘솔에도 핵심 결과 (JSON) 남김
    print(json.dumps(summary.to_dict(orient="records"), ensure_ascii=False, indent=2))


def _boxplot(table, metric, ylabel, path, ref_line=None):
    labels = list(MODELS.keys())
    data = [table[(table["model"] == m)][metric].dropna().to_numpy() for m in labels]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, tick_labels=labels, showfliers=False, widths=0.5)
    # 지터 점
    rng = np.random.default_rng(0)
    for i, d in enumerate(data, start=1):
        x = rng.normal(i, 0.05, size=len(d))
        ax.scatter(x, d, alpha=0.5, s=18, color="#1f77b4")
    if ref_line is not None:
        ax.axhline(ref_line, color="red", linestyle="--", linewidth=1, label=f"y={ref_line}")
        ax.legend()
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} — per-DOI distribution (188 rows / 36 DOI, OOF)")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _scatter(y_true, y_pred, path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, s=22, color="#2ca02c")
    lim = max(float(np.max(y_true)), float(np.max(y_pred))) * 1.05
    ax.plot([0, lim], [0, lim], "r--", linewidth=1, label="y=x (perfect)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Actual qe (mg/g)")
    ax.set_ylabel("Predicted qe (mg/g, capped)")
    ax.set_title("Predicted vs Actual — active Model B weighted (capped)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _write_markdown(summary: pd.DataFrame, table: pd.DataFrame):
    lines = []
    lines.append("# Stage 2 qe 예측 — DOI별 성능 분포 분석\n")
    lines.append(
        "입력: `reports/v60_separate_effects/all188_doi_oof_predictions.csv` "
        "(188행 / 36 DOI, DOI-GroupKFold OOF 예측). 재학습 없이 기존 OOF 예측을 분석. "
        "모든 수치는 물리 상한(0≤qe≤C0/dose) clipping 적용(capped).\n"
    )
    lines.append("## 1. 모델별 집계 vs DOI 분포 요약\n")
    lines.append(
        "| 모델 | 집계 MAE | 집계 R² | DOI별 MAE 중앙값 | DOI MAE 최악 | DOI R² 중앙값 | DOI R² 최악 | R²<0 DOI 수 | 상한위반율 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['model']} | {r['overall_MAE_capped']} | {r['overall_R2_capped']} | "
            f"{r['DOI_MAE_median']} | {r['DOI_MAE_worst']} | {r['DOI_R2_median']} | "
            f"{r['DOI_R2_worst']} | {r['n_DOI_negative_R2']} | {r['overall_violation_rate']} |"
        )
    lines.append("")
    lines.append("## 2. 핵심 해석\n")
    lines.append(
        "- **집계 지표가 분포를 가린다**: 집계 MAE는 양호해 보여도 DOI별 MAE 최악값과 "
        "R²<0 DOI 수가 크면 개별 논문 예측은 신뢰하기 어렵다.\n"
        "- **R²<0 DOI**는 해당 논문에서 모델이 평균값보다도 못하다는 뜻이다.\n"
        "- **상한위반율**은 물리적으로 불가능한 예측 비율(clipping 전)로, 모델이 질량보존을 "
        "학습하지 못했음을 보여준다.\n"
    )
    lines.append("## 3. 최악 성능 DOI (활성 모델 B 가중 기준, MAE 상위 10)\n")
    b = table[table["model"] == "B_direct_weighted"].sort_values("mae", ascending=False).head(10)
    lines.append("| DOI | n | MAE | RMSE | R² | 상한위반 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, r in b.iterrows():
        r2 = "N/A" if pd.isna(r["r2"]) else round(r["r2"], 3)
        lines.append(
            f"| {r['doi']} | {int(r['n'])} | {round(r['mae'],2)} | {round(r['rmse'],2)} | {r2} | {int(r['constraint_violations'])} |"
        )
    lines.append("")
    lines.append("## 4. 그래프\n")
    lines.append("- `doi_mae_distribution.png` — DOI별 MAE 분포(박스+점)")
    lines.append("- `doi_r2_distribution.png` — DOI별 R² 분포 (빨간선 R²=0)")
    lines.append("- `predicted_vs_actual.png` — 예측 vs 실제 산점도")
    (OUT / "RESULTS_KO.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
