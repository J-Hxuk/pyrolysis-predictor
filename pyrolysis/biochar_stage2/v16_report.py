"""
v16_report.py — Stage2 v16 재검증 산출물 생성 (CSV·PNG) 및 요약.

reports/stage2_v16/ 아래에 다음을 저장한다:
  - model_comparison.csv       : Baseline/Model A·B × 알고리즘 × 분할 성능
  - per_doi_eval.csv           : DOI별 MAE (대표 모델)
  - ablation.csv               : 제조조건 ablation + 결측·표본 정보
  - fraction_experiment.csv    : qe 직접 vs 흡착률 f 경로 (물리제약·오차)
  - nested_cv.csv              : nested CV 성능과 선택 알고리즘
  - bootstrap_ci.csv           : A vs B MAE 차이 DOI cluster bootstrap 95% CI
  - fig_actual_vs_pred_*.png, fig_residual_*.png, fig_AB_compare.png

matplotlib 는 Agg 백엔드로 저장만 한다(표시 없음).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pyrolysis.biochar_stage2 import v15_features as fc
from pyrolysis.biochar_stage2 import v15_loader as L
from pyrolysis.biochar_stage2 import v15_models as M
from pyrolysis.biochar_stage2 import v16_validate as V
from pyrolysis.biochar_stage2 import v15_recommend as R

__all__ = ["run_v16_report", "DEFAULT_OUT"]

DEFAULT_OUT = Path("reports/stage2_v16")


def _save_fig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def run_v16_report(base: Path | None = None, out_dir: Path | None = None) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = Path(base) if base is not None else Path.cwd()
    out = Path(out_dir) if out_dir is not None else (base / DEFAULT_OUT)
    out.mkdir(parents=True, exist_ok=True)

    ds = L.load_core_qe(base)
    df = ds.df
    y = pd.to_numeric(df[fc.TARGET_COLUMN], errors="coerce").to_numpy(dtype=float)
    summary: dict[str, Any] = {"n_rows": len(df), "n_doi": int(df[fc.GROUP_DOI].nunique())}

    # 1) Baseline benchmark
    bench = V.benchmark_baselines(df)
    rows = []
    for scheme, res in bench["results"].items():
        for key, m in res.items():
            rows.append({"scheme": scheme, "estimator": key, "model": m.get("model"),
                         "algo": m.get("algo", ""), "n": m.get("n"),
                         "r2": m.get("r2"), "mae": m.get("mae"), "rmse": m.get("rmse")})
    comp = pd.DataFrame(rows)
    comp.to_csv(out / "model_comparison.csv", index=False, encoding="utf-8-sig")
    summary["model_comparison_path"] = str(out / "model_comparison.csv")

    # 2) Nested CV (A, B) + representative OOF for plots/bootstrap
    nestedA, oofA = V.nested_cv_model_selection(df, "A")
    nestedB, oofB = V.nested_cv_model_selection(df, "B")
    pd.DataFrame([
        {"model": "A", **nestedA["nested_oof"], "chosen": json.dumps(nestedA["chosen_algorithms"])},
        {"model": "B", **nestedB["nested_oof"], "chosen": json.dumps(nestedB["chosen_algorithms"])},
    ]).to_csv(out / "nested_cv.csv", index=False, encoding="utf-8-sig")
    summary["nested_cv"] = {"A": nestedA["nested_oof"], "B": nestedB["nested_oof"],
                            "A_chosen": nestedA["chosen_algorithms"],
                            "B_chosen": nestedB["chosen_algorithms"]}

    # physical baseline OOF (LODO) for comparison plot
    phys = V.physical_fraction_baseline_oof(df, "LODO")
    summary["physical_baseline_LODO"] = V.metrics(y, phys)

    # 3) per-DOI eval (nested A)
    doi = df[fc.GROUP_DOI].to_numpy()
    per = []
    for g in pd.unique(doi):
        mm = (doi == g)
        per.append({"doi": str(g), "n": int(mm.sum()),
                    "mae_A": V.metrics(y[mm], oofA[mm])["mae"],
                    "mae_B": V.metrics(y[mm], oofB[mm])["mae"],
                    "mae_physical": V.metrics(y[mm], phys[mm])["mae"]})
    pd.DataFrame(per).to_csv(out / "per_doi_eval.csv", index=False, encoding="utf-8-sig")

    # 4) A vs B cluster bootstrap CI
    ci = V.cluster_bootstrap_ci(df, oofA, oofB)
    pd.DataFrame([ci]).to_csv(out / "bootstrap_ci.csv", index=False, encoding="utf-8-sig")
    summary["bootstrap_AB"] = ci

    # 4b) LOFO ≠ LODO: 각 fold train/test DOI 중복 감사
    lofo_audit = V.lofo_doi_overlap_audit(df)
    pd.DataFrame(lofo_audit["per_fold"]).to_csv(
        out / "lofo_doi_overlap.csv", index=False, encoding="utf-8-sig")
    summary["lofo_doi_overlap"] = {
        "n_folds": lofo_audit["n_folds"],
        "n_folds_with_doi_overlap": lofo_audit["n_folds_with_doi_overlap"],
        "doi_independent": lofo_audit["doi_independent"],
        "classification": lofo_audit["classification"],
        "note": lofo_audit["note"],
    }

    # 5) ablation (Ridge: 빠르고, 결측 효과/기여 분리 목적에 충분)
    abl = V.ablation_manufacturing(df, algo_name="Ridge", scheme="LODO")
    arows = [{"config": "A_only", "mae": abl["A_only"]["mae"], "r2": abl["A_only"]["r2"],
              "delta_mae_vs_A": 0.0, "delta_r2_vs_A": 0.0},
             {"config": "B_full", "mae": abl["B_full"]["mae"], "r2": abl["B_full"]["r2"]}]
    for c, m in abl["add_one_to_A"].items():
        vi = abl["variable_info"].get(c, {})
        dl = abl["delta_vs_A_only"].get(c, {})
        arows.append({"config": f"A+{c}", "mae": m["mae"], "r2": m["r2"],
                      "delta_mae_vs_A": dl.get("delta_mae_vs_A"),
                      "delta_r2_vs_A": dl.get("delta_r2_vs_A"),
                      "missing_pct": vi.get("missing_pct"), "n_available": vi.get("n_available")})
    for c, m in abl["drop_one_from_B"].items():
        arows.append({"config": f"B-{c}", "mae": m["mae"], "r2": m["r2"]})
    pd.DataFrame(arows).to_csv(out / "ablation.csv", index=False, encoding="utf-8-sig")
    summary["ablation"] = {
        "A_only": abl["A_only"], "B_full": abl["B_full"],
        "baseline_consistency": abl["baseline_consistency"],
        "delta_vs_A_only": abl["delta_vs_A_only"],
        "interpretation_note": (
            "delta_r2_vs_A/delta_mae_vs_A 는 같은 run 의 A_only 기준 변화량이다. 같은 기준에서 "
            "제조조건 추가가 A 를 넘어서는지 여부만 말한다. 결측률과 기여도의 연관만으로 "
            "'고결측이 성능저하의 주원인'이라고 인과적으로 단정하지 않는다."),
    }

    # 6) fraction target experiment — qe 공간과 f 공간 지표를 '분리' 보고
    #    (같은 held-out sample 에서 target_space 열로 평가공간을 명시; R²(qe)→R²(f) 혼동 방지)
    frac = V.fraction_target_experiment(df, algo_name="Ridge")
    frows = []
    for scheme, r in frac["results"].items():
        frows.append({"scheme": scheme, "path": "qe_direct", "target_space": "qe",
                      "mae": r["qe_direct"]["mae"], "r2": r["qe_direct"]["r2"],
                      "viol_rate": r["qe_direct"]["constraint_violation_rate"]})
        frows.append({"scheme": scheme, "path": "f_rate_then_qe", "target_space": "qe",
                      "mae": r["f_rate_then_qe"]["mae"], "r2": r["f_rate_then_qe"]["r2"],
                      "viol_rate": r["f_rate_then_qe"]["constraint_violation_rate"]})
        ft = r["f_target_metrics"]
        frows.append({"scheme": scheme, "path": "f_rate_then_qe", "target_space": "f",
                      "mae": ft["mae"], "r2": ft["r2"], "viol_rate": None})
    pd.DataFrame(frows).to_csv(out / "fraction_experiment.csv", index=False, encoding="utf-8-sig")
    summary["fraction_experiment"] = frac["results"]
    summary["fraction_experiment_note"] = (
        "타깃 공간(target_space)을 분리해 보고한다. 'R²(qe)' 와 'R²(f)' 는 서로 다른 평가공간이며 "
        "f 를 학습한 예측을 qe 로 환산한 R²(qe) 를 R²(f) 라고 말하지 않는다. "
        "qe_direct·f_rate_then_qe 는 qe 공간, f_target_metrics 는 무차원 f 공간(0≤f≤1) 지표다. "
        "높은 R²(qe) 가 C0/dose 분산에 의존하는지는 물리 baseline(f constant) 대조로 함께 해석한다.")

    # 7) Chen16: simple baseline vs Ridge
    #    (a) pooled LOO diagnostic — Dummy 의 Spearman=-1 은 구조적 산물(해석 주의)
    #    (b) single-fit 반복 12/4 holdout — 단일 모델 동시추천 상황의 주 평가
    chen = L.load_chen16(base)
    rec = R.recommend_benchmark(chen)
    summary["chen16_pooled_loo"] = {
        "diagnostic_type": rec["diagnostic_type"],
        "dummy_anti_correlation_note": rec["dummy_anti_correlation_note"],
        "results": {k: {"mae_pp": v["removal_mae_pp"], "spearman": v["spearman"],
                        "top3": v["top3_overlap_of_3"], "regret": v["regret_pp"]}
                    for k, v in rec["results"].items()},
    }
    holdout = R.recommend_single_fit_holdout(chen)
    summary["chen16_single_fit_holdout"] = {
        "n_train": holdout["n_train"], "n_test": holdout["n_test"],
        "n_repeats": holdout["n_repeats"], "primary_metric": holdout["primary_metric"],
        "generalization_caveat": holdout["generalization_caveat"],
        "tie_handling": holdout["tie_handling"],
        "results": holdout["results"],
    }
    # 하위호환: 기존 'chen16' 키는 pooled LOO 수치로 보존(삭제하지 않음)
    summary["chen16"] = summary["chen16_pooled_loo"]["results"]
    pd.DataFrame([
        {"algo": k, **v} for k, v in holdout["results"].items()
    ]).to_csv(out / "chen16_single_fit_holdout.csv", index=False, encoding="utf-8-sig")

    # ── PNG 그림 ───────────────────────────────────────────────────────────
    # actual vs predicted (nested A, B, physical)
    for name, oof in [("A_nested", oofA), ("B_nested", oofB), ("physical", phys)]:
        m = ~np.isnan(oof) & ~np.isnan(y)
        fig, ax = plt.subplots(figsize=(4.2, 4.2))
        ax.scatter(y[m], oof[m], s=18, alpha=0.6)
        lim = [0, max(float(np.nanmax(y)), float(np.nanmax(oof[m]))) * 1.05]
        ax.plot(lim, lim, "r--", lw=1)
        ax.set_xlabel("actual qe (mg/g)"); ax.set_ylabel("predicted qe (mg/g)")
        mm = V.metrics(y, oof)
        ax.set_title(f"{name}  R2={mm['r2']:.3f}  MAE={mm['mae']:.1f}")
        _save_fig(fig, out / f"fig_actual_vs_pred_{name}.png")

        fig2, ax2 = plt.subplots(figsize=(4.6, 3.4))
        ax2.scatter(oof[m], oof[m] - y[m], s=16, alpha=0.6)
        ax2.axhline(0, color="r", ls="--", lw=1)
        ax2.set_xlabel("predicted qe"); ax2.set_ylabel("residual (pred - actual)")
        ax2.set_title(f"residual — {name}")
        _save_fig(fig2, out / f"fig_residual_{name}.png")

    # A/B/physical comparison bar (LODO + nested)
    fig3, ax3 = plt.subplots(figsize=(6, 3.6))
    labels = ["Physical", "A (nested)", "B (nested)"]
    maes = [summary["physical_baseline_LODO"]["mae"], nestedA["nested_oof"]["mae"], nestedB["nested_oof"]["mae"]]
    r2s = [summary["physical_baseline_LODO"]["r2"], nestedA["nested_oof"]["r2"], nestedB["nested_oof"]["r2"]]
    x = np.arange(len(labels))
    ax3.bar(x, maes, color=["#888", "#3a7", "#37a"])
    for i, (mv, rv) in enumerate(zip(maes, r2s)):
        ax3.text(i, mv, f"MAE {mv:.1f}\nR2 {rv:.2f}" if rv is not None else f"MAE {mv:.1f}",
                 ha="center", va="bottom", fontsize=8)
    ax3.set_xticks(x); ax3.set_xticklabels(labels)
    ax3.set_ylabel("OOF MAE (mg/g)"); ax3.set_title("Baseline vs A vs B (LODO/nested)")
    _save_fig(fig3, out / "fig_AB_compare.png")

    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
