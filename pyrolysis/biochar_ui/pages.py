"""
pages.py — 바이오차 예측 플랫폼 Streamlit 화면.

페이지 (지시서 §4):
  A. Overview            연구 목적·모델 연결 상태·Stage 구현 상태·데이터셋 버전
  B. Pyrolysis Prediction 원료 조성·열분해 조건 입력 -> BET 예측 + 적용 범위 판정
  C. Condition Comparison 동일 원료에서 조건 변화 비교 (표·곡선·CSV)
  D. Adsorption Analysis  Stage2 연결 영역(미확정 표시) + Langmuir fitting
  E. Data & Model Validation 학습 데이터·그룹·결측·GroupKFold·적용 범위·한계

이 파일은 Streamlit 을 import 한다. 예측·피팅 계산은 pyrolysis.* 패키지 함수를 호출한다.
앱 실행 시 모델을 재학습하지 않는다 (저장된 활성 모델만 로드).
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from pyrolysis.biochar_stage1 import data_validation, inference
from pyrolysis.biochar_stage1.data_loader import load_stage1_dataset
from pyrolysis.biochar_stage1.features import (
    APPLICABILITY_SCOPE,
    BET_INPUT_ORDER,
    BET_INPUT_VARIABLES,
    BET_TARGET,
    input_variable_rows,
)
from pyrolysis.biochar_stage1.model_registry import Stage1Registry
from pyrolysis.biochar_stage2 import adapter as s2_adapter
from pyrolysis.biochar_stage2 import langmuir
from pyrolysis.biochar_stage2.schema import field_rows

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ── 모델 로딩 (캐시, 재학습 없음) ────────────────────────────────────────────


@st.cache_resource(show_spinner=False)
def _load_active_model(registry_root: str):
    """활성 BET 모델을 로드한다 (읽기 전용, 캐시). 앱이 재학습하지 않는다."""
    reg = Stage1Registry(root=Path(registry_root))
    vdir = reg.get_active_dir()
    if vdir is None:
        return None
    return inference.load_bet_model(vdir)


def _get_model():
    reg = Stage1Registry.default(_REPO_ROOT)
    return _load_active_model(str(reg.root))


@st.cache_data(show_spinner=False)
def _load_audit():
    ds = load_stage1_dataset(_REPO_ROOT)
    audit = data_validation.audit_dataset(ds.base, ds.extra)
    return audit, ds.version_info()


# ── 입력 위젯 ────────────────────────────────────────────────────────────────


def _elemental_inputs() -> dict[str, float]:
    st.markdown("**원료 원소분석 (dry ash-free, daf)**")
    c1, c2, c3, c4 = st.columns(4)
    vals = {}
    with c1:
        vals["C_pct"] = st.number_input("C (wt%)", value=48.9, min_value=0.0, max_value=100.0, step=0.1, key="C_pct")
    with c2:
        vals["H_pct"] = st.number_input("H (wt%)", value=6.0, min_value=0.0, max_value=100.0, step=0.1, key="H_pct")
    with c3:
        vals["N_pct"] = st.number_input("N (wt%)", value=0.5, min_value=0.0, max_value=100.0, step=0.1, key="N_pct")
    with c4:
        vals["O_pct"] = st.number_input("O (wt%)", value=44.4, min_value=0.0, max_value=100.0, step=0.1, key="O_pct")
    return vals


def _proximate_inputs() -> dict[str, float]:
    st.markdown("**원료 공업분석 (dry basis)**")
    c1, c2, c3 = st.columns(3)
    vals = {}
    with c1:
        vals["VM_pct"] = st.number_input("휘발분 VM (wt%)", value=78.26, min_value=0.0, max_value=100.0, step=0.1, key="VM_pct")
    with c2:
        vals["Ash_pct"] = st.number_input("회분 Ash (wt%)", value=1.13, min_value=0.0, max_value=100.0, step=0.1, key="Ash_pct")
    with c3:
        vals["FC_pct"] = st.number_input("고정탄소 FC (wt%)", value=20.61, min_value=0.0, max_value=100.0, step=0.1, key="FC_pct")
    return vals


def _pyrolysis_inputs() -> dict[str, float]:
    st.markdown("**열분해 조건**")
    c1, c2, c3 = st.columns(3)
    vals = {}
    with c1:
        vals["temperature_C"] = st.number_input("온도 (°C)", value=600.0, min_value=0.0, step=10.0, key="temperature_C")
    with c2:
        vals["time_min"] = st.number_input("유지시간 (min)", value=10.0, min_value=0.0, step=5.0, key="time_min")
    with c3:
        vals["heating_rate_C_min"] = st.number_input("승온속도 (°C/min)", value=30.0, min_value=0.0, step=1.0, key="heating_rate_C_min")
    return vals


def _render_domain_verdict(sanity, domain):
    if sanity.errors:
        for e in sanity.errors:
            st.error(e["message_ko"])
    for w in sanity.warnings:
        st.warning(w["message_ko"])
    for w in domain.warnings:
        code = w.get("code", "")
        msg = w.get("message_ko", "")
        if code == "TRAINING_RANGE_EXCEEDED":
            st.warning("⚠ " + msg)
        elif code == "WITHIN_RANGE_NOT_GUARANTEED":
            st.info(msg)
        else:
            st.warning(msg)


# ── A. Overview ──────────────────────────────────────────────────────────────


def page_overview():
    st.header("A. Overview")
    st.write(
        "원료 조성과 열분해 조건으로 바이오차 물성(BET)을 예측하는 2-Stage 플랫폼입니다. "
        "Stage 1 은 실제 논문 데이터(Leng2022 기반)로 학습한 BET 예측 모델을 제공합니다. "
        "Stage 2(흡착 성능 qe)는 데이터셋 연결을 위한 인터페이스만 준비되어 있습니다."
    )

    model = _get_model()
    _, version_info = _load_audit()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Stage 1 (BET)", "연결됨" if model else "모델 없음")
    with c2:
        st.metric("Stage 2 (qe)", "인터페이스만")
    with c3:
        st.metric("데이터셋 버전", version_info["dataset_version_hash"])

    st.subheader("모델 연결 상태")
    if model is None:
        st.warning(
            "활성 BET 모델이 없습니다. 아래 명령으로 학습·활성화 후 앱을 새로고침하세요.\n\n"
            "`python -m pyrolysis.biochar_stage1.cli train`"
        )
    else:
        meta = model.metadata
        st.success(f"활성 모델: {meta['model_version']}")
        st.table(pd.DataFrame([
            {"항목": "모델 종류", "값": meta["model_type"]},
            {"항목": "학습 관측 수", "값": meta["training_data"]["n_total_with_target"]},
            {"항목": "출처 그룹 수", "값": meta["training_data"]["n_provenance_groups"]},
            {"항목": "학습 시각(UTC)", "값": meta["trained_at_utc"]},
            {"항목": "출력", "값": f"{meta['target']['label_ko']} ({meta['target']['unit']})"},
        ]))

    st.subheader("Stage 구현 상태")
    st.table(pd.DataFrame([
        {"단계": "Stage 1 — BET 예측", "상태": "구현 완료", "비고": "실제 Leng2022 데이터 RF 모델"},
        {"단계": "Stage 1 — TPV 예측", "상태": "미구현", "비고": "학습 후보만 추출 (TPV_default_train_pool)"},
        {"단계": "Stage 2 — qe 예측(ML)", "상태": "인터페이스만", "비고": "데이터셋 제작 중, 학습 미실시"},
        {"단계": "Langmuir fitting", "상태": "구현 완료", "비고": "실측 등온 데이터 피팅 (ML 예측과 별개)"},
    ]))

    st.caption(
        "적용 범위: " + APPLICABILITY_SCOPE["domain"] +
        ". 입력에 없는 인자(활성화·함침·세척·반응기·분위기 등)의 영향은 검증하지 않았습니다."
    )


# ── B. Pyrolysis Prediction ─────────────────────────────────────────────────


def page_prediction():
    st.header("B. Pyrolysis Prediction")
    model = _get_model()
    if model is None:
        st.warning("활성 BET 모델이 없습니다. `python -m pyrolysis.biochar_stage1.cli train` 후 새로고침하세요.")
        return

    vals = {}
    vals.update(_elemental_inputs())
    vals.update(_proximate_inputs())
    vals.update(_pyrolysis_inputs())

    st.caption(
        "원료(feedstock) 조성 기준입니다. 생성 바이오차 조성이 아닙니다. "
        "C/H/N/O 합을 100 으로 자동 정규화하지 않습니다."
    )

    run = st.button("BET 예측 실행", type="primary")
    if not run:
        st.info("입력 후 '예측 실행'을 누르면 BET 비표면적을 예측합니다.")
        return

    sanity = inference.check_input_sanity(vals)
    if not sanity.ok:
        st.error("입력 오류가 있어 예측을 중단합니다.")
        _render_domain_verdict(sanity, inference.DomainVerdict())
        return

    pred = inference.predict_bet(model, vals)
    domain = inference.check_applicability_domain(vals, model)

    st.subheader("예측 결과")
    prec = model.metadata.get("target", {}).get("display_precision", 2)
    st.metric(f"{BET_TARGET.label_ko} ({BET_TARGET.unit})", f"{pred[0]:.2f}")

    meta = model.metadata
    cv = meta.get("cross_validation")
    if cv:
        r = cv["result"]
        st.caption(
            f"모델: {meta['model_version']} · 탐색적 GroupKFold(새 원료·새 논문) "
            f"R²={r['mean_r2']:.3f}, MAE={r['mean_mae']:.1f}, RMSE={r['mean_rmse']:.1f} {BET_TARGET.unit}. "
            "이 오차는 전체 OOF 지표이며 개별 예측의 신뢰구간이 아닙니다."
        )

    st.subheader("적용 범위 판정")
    _render_domain_verdict(sanity, domain)


# ── C. Condition Comparison ─────────────────────────────────────────────────


def page_condition_comparison():
    st.header("C. Condition Comparison")
    model = _get_model()
    if model is None:
        st.warning("활성 BET 모델이 없습니다.")
        return

    st.write("동일 원료에서 열분해 온도를 변화시켜 예측 BET 를 비교합니다. "
             "다른 조건(유지시간·승온속도)은 고정합니다.")

    base_vals = {}
    base_vals.update(_elemental_inputs())
    base_vals.update(_proximate_inputs())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        t_min = st.number_input("온도 시작 (°C)", value=300.0, step=10.0, key="cc_tmin")
    with c2:
        t_max = st.number_input("온도 끝 (°C)", value=800.0, step=10.0, key="cc_tmax")
    with c3:
        t_step = st.number_input("간격 (°C)", value=50.0, min_value=1.0, step=10.0, key="cc_tstep")
    with c4:
        hold = st.number_input("유지시간 (min)", value=10.0, min_value=0.0, step=5.0, key="cc_hold")
    hr = st.number_input("승온속도 (°C/min)", value=30.0, min_value=0.0, step=1.0, key="cc_hr")

    run = st.button("조건 비교 실행", type="primary")
    if not run:
        return

    if t_max < t_min:
        st.error("온도 끝 값이 시작 값보다 작습니다.")
        return

    san = inference.check_input_sanity({**base_vals, "temperature_C": t_min,
                                        "time_min": hold, "heating_rate_C_min": hr})
    if not san.ok:
        st.error("입력 오류가 있어 비교를 중단합니다.")
        _render_domain_verdict(san, inference.DomainVerdict())
        return

    temps = np.arange(t_min, t_max + 1e-9, t_step)
    rows = []
    for t in temps:
        v = {**base_vals, "temperature_C": float(t), "time_min": hold, "heating_rate_C_min": hr}
        pred = inference.predict_bet(model, v)
        dom = inference.check_applicability_domain(v, model)
        exceeded = any(w.get("code") == "TRAINING_RANGE_EXCEEDED" for w in dom.warnings)
        rows.append({
            "temperature_C": float(t),
            "predicted_BET_m2_g": round(float(pred[0]), 2),
            "range_exceeded": exceeded,
        })
    table = pd.DataFrame(rows)

    st.subheader("조건별 예측 BET 비교표")
    st.dataframe(table, use_container_width=True)

    st.subheader("온도에 따른 BET 예측 곡선")
    st.caption("아래 곡선은 모델의 예측 응답이며, 실제 측정 곡선이 아닙니다.")
    chart_df = table.set_index("temperature_C")[["predicted_BET_m2_g"]]
    chart_df = chart_df.rename(columns={"predicted_BET_m2_g": "예측 BET (모델 응답, m²/g)"})
    st.line_chart(chart_df)

    csv = table.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "CSV 결과 다운로드", data=csv,
        file_name="condition_comparison_BET.csv", mime="text/csv",
    )


# ── D. Adsorption Analysis ───────────────────────────────────────────────────


def page_adsorption():
    st.header("D. Adsorption Analysis")
    st.info(
        "Stage 2(흡착 성능 qe) 머신러닝 모델은 현재 **미확정** 상태입니다. "
        "데이터셋이 제작 중이며, 이 단계에서는 임의 데이터로 학습하지 않습니다. "
        "아래는 향후 연결할 입력 스키마와, 실측 등온흡착 데이터에 대한 Langmuir fitting 입니다."
    )

    with st.expander("Stage 2 입력 스키마 (향후 수용 구조)"):
        st.dataframe(pd.DataFrame(field_rows()), use_container_width=True)
        st.caption("모든 필드가 이미 준비됐다는 뜻은 아닙니다. 미검증 qe 는 학습 대상으로 자동 승인하지 않습니다.")

    st.subheader("Langmuir 등온흡착 피팅 (실측 데이터 분석)")
    st.caption(
        "qe = (qmax · K_L · Ce) / (1 + K_L · Ce). "
        "동일 시료·동일 오염물질·동일 조건에서 측정한 (Ce, qe) 쌍을 입력하세요. "
        "이 분석은 입력한 실측 데이터 피팅이며 Stage 2 머신러닝 예측과는 별개입니다."
    )

    default = pd.DataFrame({
        "Ce_mg_L": [1.0, 2.0, 5.0, 10.0, 20.0, 40.0],
        "qe_mg_g": [8.0, 14.0, 28.0, 45.0, 66.0, 86.0],
    })
    edited = st.data_editor(default, num_rows="dynamic", use_container_width=True, key="langmuir_editor")

    if st.button("Langmuir 피팅 실행", type="primary"):
        ce = pd.to_numeric(edited["Ce_mg_L"], errors="coerce").to_numpy()
        qe = pd.to_numeric(edited["qe_mg_g"], errors="coerce").to_numpy()
        result = langmuir.fit_langmuir(ce, qe)
        d = result.to_dict()
        if not d["fitted"]:
            st.error(f"피팅 불가 ({d['reason_code']}): {d['message_ko']}")
            return
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("qmax (mg/g)", f"{d['qmax_mg_g']:.3f}")
        c2.metric("K_L (L/mg)", f"{d['KL_L_mg']:.4f}")
        c3.metric("R²", f"{d['r2']:.4f}")
        c4.metric("RMSE (mg/g)", f"{d['rmse']:.3f}")

        obs = pd.DataFrame({"Ce": d["observed"]["Ce"], "실측 qe": d["observed"]["qe"]}).set_index("Ce")
        fit = pd.DataFrame({"Ce": d["fit_curve"]["Ce"], "피팅 곡선 qe": d["fit_curve"]["qe"]}).set_index("Ce")
        merged = pd.concat([obs, fit], axis=0).sort_index()
        st.line_chart(merged)
        st.caption(d["note"])


# ── E. Data & Model Validation ───────────────────────────────────────────────


def page_validation():
    st.header("E. Data & Model Validation")
    audit, version_info = _load_audit()
    model = _get_model()

    st.subheader("학습 데이터 현황")
    base = audit["base"]
    c1, c2, c3 = st.columns(3)
    c1.metric("기준 평가 데이터(base)", base["n_rows"])
    c2.metric("출처 그룹 수", base["provenance_groups"]["n_groups"])
    c3.metric("추가 후보(extra)", audit["extra"]["n_rows"])

    st.subheader("입력변수 결측 비율")
    miss_rows = []
    for vid in BET_INPUT_ORDER:
        m = base["missing"].get(vid, {})
        miss_rows.append({
            "입력": vid,
            "단위": BET_INPUT_VARIABLES[vid].unit,
            "basis": BET_INPUT_VARIABLES[vid].basis,
            "결측(base) %": m.get("missing_pct", 0.0),
            "결측(extra) %": audit["extra"]["missing"].get(vid, {}).get("missing_pct", 0.0),
        })
    st.dataframe(pd.DataFrame(miss_rows), use_container_width=True)

    st.subheader("학습 대상 포함·제외 현황")
    el = base["training_eligibility"]
    st.write(f"base: 전체 {el['n_rows']} 중 10개 입력+타깃 완비 {el['n_all_inputs_present']}, 제외 {el['n_excluded']}")
    el2 = audit["extra"]["training_eligibility"]
    st.write(f"extra: 전체 {el2['n_rows']} 중 완비 {el2['n_all_inputs_present']} "
             f"(추가 후보는 결측 입력을 학습 폴드 내 중앙값 대체로 처리).")

    st.subheader("GroupKFold 평가 지표 (탐색적 일반화 검증)")
    if model and model.metadata.get("cross_validation"):
        cv = model.metadata["cross_validation"]
        r = cv["result"]
        st.table(pd.DataFrame([
            {"지표": "R² (평균)", "값": round(r["mean_r2"], 4)},
            {"지표": "MAE (m²/g)", "값": round(r["mean_mae"], 3)},
            {"지표": "RMSE (m²/g)", "값": round(r["mean_rmse"], 3)},
            {"지표": "출처 누설 없음", "값": r["leakage_checks_passed"]},
        ]))
        cmp = cv["comparison_with_eval37_reference"]
        if cmp.get("reference_found"):
            st.caption(
                "eval37 참고 수치 대비 차이 (환경·분할 구현 차이로 발생): "
                f"ΔR²={cmp['delta']['r2']:+.4f}, ΔMAE={cmp['delta']['mae']:+.3f}, "
                f"ΔRMSE={cmp['delta']['rmse']:+.3f}"
            )
        st.warning(cv["evaluation_type"])
    else:
        st.info("활성 모델에 교차검증 기록이 없습니다.")

    st.subheader("적용 범위와 모델 한계")
    st.json(APPLICABILITY_SCOPE)
    if model:
        for note in model.metadata.get("unvalidated_notes", []):
            st.caption("• " + note)

    st.caption(f"데이터셋: {version_info['dataset_source']} · 버전 {version_info['dataset_version_hash']}")


# ── 라우팅 ───────────────────────────────────────────────────────────────────

PAGES = {
    "A. Overview": page_overview,
    "B. Pyrolysis Prediction": page_prediction,
    "C. Condition Comparison": page_condition_comparison,
    "D. Adsorption Analysis": page_adsorption,
    "E. Data & Model Validation": page_validation,
}


def render_app():
    st.set_page_config(page_title="Biochar Prediction Platform", layout="wide")
    st.sidebar.title("Biochar Prediction Platform")
    st.sidebar.caption("Stage 1 BET · Stage 2 (인터페이스)")
    choice = st.sidebar.radio("분석 화면", list(PAGES.keys()), label_visibility="collapsed")
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "공학 분석용 연구 도구. 예측 곡선은 모델 응답이며 실측 곡선이 아닙니다. "
        "검증되지 않은 신뢰도 백분율을 제공하지 않습니다."
    )
    PAGES[choice]()
