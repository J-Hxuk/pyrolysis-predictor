"""Stage 2 Streamlit research workbench.

This is a read-only inference UI: no re-training, no automatic substitution of
unknown biomass, and no false claim that optional feedstock chemistry is used
by the frozen v15 model. Stage1-v60 -> Stage2 is deliberately disabled.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from pyrolysis.biochar_stage2 import v15_features as fc
from pyrolysis.biochar_stage2 import v15_loader as L
from pyrolysis.biochar_stage2.input_support import (
    OPTIONAL_COMPOSITION, match_feedstock, related_records,
    validate_composition, applied_optional_fields, predict_from_contract,
)
from pyrolysis.biochar_stage2.v37_sensitivity import VARIABLES, eval_one, sensitivity_frame, make_sweep_values
from pyrolysis.biochar_stage2.v41_scope import inspect_active_scope, scope_to_table
from pyrolysis.biochar_stage2.v43_scenarios import (
    MAX_SCENARIOS, append_snapshot, build_snapshot, comparison_csv,
    comparison_frame, printable_report,
)
from pyrolysis.biochar_stage2.v15_train import Stage2Registry
from pyrolysis.biochar_stage2.candidate_screening import (
    observed_manufacturing_candidates, score_observed_candidates,
)
from pyrolysis.biochar_stage2.v26_screening_policy import assess_scope

_REPO = Path(__file__).resolve().parents[2]


@st.cache_resource(show_spinner=False)
def _load_models(registry_root: str):
    reg = Stage2Registry(root=Path(registry_root))
    versions = reg.list_versions()
    models = {}
    for v in versions:
        obj = reg.load(v)
        models[obj["metadata"]["model_name"]] = obj
    active = reg.get_active()
    return models, (active["metadata"]["model_version"] if active else None)


def _get_models():
    try:
        return _load_models(str(Stage2Registry.default(_REPO).root))
    except Exception as exc:
        st.error("저장 모델을 불러오지 못했습니다. 모델을 학습한 scikit-learn/XGBoost 버전과 "
                 "현재 Python 환경을 확인하세요. 기존 모델은 임의 변환하지 않습니다. "
                 f"({type(exc).__name__}: {exc})")
        return {}, None


@st.cache_data(show_spinner=False)
def _load_core():
    ds = L.load_core_qe(_REPO)
    return ds.df, L.validate_core(ds)


@st.cache_data(show_spinner=False)
def _known_feedstocks():
    df, _ = _load_core()
    return sorted(df[fc.GROUP_FEEDSTOCK].dropna().astype(str).unique().tolist())


@st.cache_data(show_spinner=False)
def _v16_summary():
    path = _REPO / "reports" / "stage2_v16" / "summary.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _feedstock_lookup(text: str):
    """Backward-compatible tuple used by the earlier UI/testing."""
    m = match_feedstock(text, _known_feedstocks())
    return m.matched, ("substring" if m.status == "suggestion" else
                       "out_of_domain" if m.status == "unknown" else m.status), list(m.suggestions)


def _heading(title: str, detail: str):
    st.markdown(f"## {title}")
    st.caption(detail)


def _model_status_caption():
    st.caption("Stage 2 · Core 97 문헌 데이터 · reported uptake (평형 미인증) · 실험 대체 불가")


def _accept_suggestion(widget_key, suggested_name):
    """Called by an explicit user click; never auto-substitutes a biomass."""
    st.session_state[widget_key] = suggested_name


def _feedstock_control(key="s2_bio", default="rice husk"):
    """Single free-form input; suggestions do not overwrite user input."""
    text = st.text_input("바이오매스 원료명", value=default, key=key,
                         help="한국어/영어 입력 가능. 원료명이 물질의 화학조성을 자동으로 알려주지는 않습니다.")
    m = match_feedstock(text, _known_feedstocks())
    if m.status in ("exact", "alias"):
        st.caption(f"등록 원료 확인: {m.matched}" + (" · 동의어 정규화" if m.status == "alias" else ""))
    elif m.status == "empty":
        st.warning("원료명을 입력해 주세요.")
    else:
        st.info("등록되지 않은 원료명입니다. Model A는 원료를 구별하지 않는 흡착조건 기반 예측만 제공합니다. "
                "Model B는 학습되지 않은 원료의 효과를 검증하지 못했습니다.")
    if m.suggestions and m.status not in ("exact", "alias"):
        st.caption("문자열이 유사한 등록 원료 · 실제 다른 원료일 수 있으니 직접 확인 후 선택하세요.")
        for i, suggestion in enumerate(m.suggestions[:4]):
            st.button(f"추천: {suggestion}", key=f"{key}_suggestion_{i}",
                      on_click=_accept_suggestion, args=(key, suggestion))
    return m


def _composition_controls():
    """Data-entry form, not an unvalidated inference path."""
    with st.container(border=True):
        st.markdown("#### 원료 조성 · 공업분석 / 원소분석 (선택 입력)")
        st.caption("원료(열분해 전) 실측값만 입력하세요. 모르는 항목은 공란으로 둡니다. "
                   "현재 저장된 Model A/B는 조성변수를 학습하지 않아 이 값을 바꿔도 qe 예측이 달라지지 않습니다.")
        col_a, col_b = st.columns(2)
        with col_a:
            st.caption("원소분석 기준: dry ash-free (daf) · 원료(열분해 전) 기준 고정")
            st.markdown("**원소분석 (wt%)**")
            fields1 = list(OPTIONAL_COMPOSITION.items())[:5]
            vals = {k: st.text_input(f"{label} (wt%)", key="s2_comp_" + k, placeholder="미보고")
                    for k, label in fields1}
        with col_b:
            st.caption("공업분석 기준: dry basis · 원료(열분해 전) 기준 고정")
            st.markdown("**공업분석 (wt%)**")
            vals.update({k: st.text_input(f"{label} (wt%)", key="s2_comp_" + k, placeholder="미보고")
                        for k, label in list(OPTIONAL_COMPOSITION.items())[5:9]})
        st.markdown("**구성성분 (선택)**")
        extra = st.columns(3)
        for col, (k, label) in zip(extra, list(OPTIONAL_COMPOSITION.items())[9:]):
            with col:
                vals[k] = st.text_input(f"{label} (%)", key="s2_comp_" + k, placeholder="미보고")
        comp, errors = validate_composition(vals)
        if comp:
            st.caption("입력값은 원소분석 daf / 공업분석 dry 기준으로 취급합니다. 해당 기준으로 확인되지 않은 값은 입력하지 마세요.")
        if comp:
            st.caption(f"입력된 특성 {len(comp)}개 · 학습 모델 반영 여부는 결과에서 확인")
        return comp, errors


def _adsorption_widgets(prefix="s2"):
    """Slider values live in session_state. Each drag reruns Streamlit inference."""
    df, _ = _load_core()
    pollutants = sorted(df["pollutant"].dropna().unique().tolist())
    st.markdown("#### 흡착 실험조건 · 슬라이더로 변경")
    pol = st.selectbox("오염물질", pollutants, key=f"{prefix}_pol")
    a, b, c = st.columns(3)
    with a:
        c0 = st.slider("초기농도 C₀ (mg/L)", 1.0, 500.0, 100.0, 1.0, key=f"{prefix}_c0")
        ph = st.slider("초기 pH", 0.0, 14.0, 5.0, 0.1, key=f"{prefix}_ph")
    with b:
        dose = st.slider("흡착제 투입량 (g/L)", 0.1, 10.0, 2.0, 0.1, key=f"{prefix}_dose")
        temp = st.slider("흡착온도 (°C)", 0.0, 70.0, 25.0, 1.0, key=f"{prefix}_T")
    with c:
        ct = st.slider("접촉시간 (min)", 0.0, 2880.0, 1440.0, 10.0, key=f"{prefix}_ct")
    return {"pollutant": pol, "C0_mg_L": c0, "initial_pH": ph, "dose_g_L": dose,
            "adsorption_T_C": temp, "contact_time_min": ct}


_ROUTES = {
    "SINGLE_STAGE_PYROLYSIS": "단일 열분해",
    "TWO_STAGE_THERMAL": "2단 열처리",
    "TWO_STAGE_THERMAL_STEAM": "2단 열처리 + 스팀",
    "CHEMICAL_POST_MODIFICATION": "열분해 후 화학개질",
    "AQUEOUS_POST_MODIFICATION": "수용액 후처리",
    "UREA_CO_PYROLYSIS": "요소 공동열분해",
    "PREIMPREGNATION_PYROLYSIS": "함침 후 열분해",
    "CO_PYROLYSIS_EGGSHELL": "달걀껍질 공동열분해",
}


def _manufacture_widgets(prefix="s2"):
    df, _ = _load_core()
    routes = sorted(df["manufacturing_route"].dropna().unique().tolist())
    atms = sorted(df["pyrolysis_atmosphere"].dropna().unique().tolist())
    st.markdown("#### 제조조건 (Model B에서만 활용)")
    c1, c2, c3 = st.columns(3)
    with c1:
        route = st.selectbox("제조공정", routes, key=f"{prefix}_route",
                             index=routes.index("SINGLE_STAGE_PYROLYSIS") if "SINGLE_STAGE_PYROLYSIS" in routes else 0,
                             format_func=lambda x: f"{_ROUTES.get(x, x.replace('_',' ').title())} ({x})")
        suggested_steps = 2 if ("TWO_STAGE" in route or "POST_" in route) else 1
        steps = st.number_input("공정 단계 수", value=suggested_steps, min_value=1, max_value=5, key=f"{prefix}_steps")
    with c2:
        pt = st.number_input("열분해온도 (°C)", value=500.0, step=10.0, key=f"{prefix}_pt")
        hold = st.number_input("유지시간 (min)", min_value=0.0, value=120.0, step=10.0, key=f"{prefix}_hold")
    with c3:
        hr_unknown = st.checkbox("승온속도 미보고", key=f"{prefix}_hr_unknown")
        hr = st.number_input("승온속도 (°C/min)", value=10.0, min_value=0.0, step=1.0,
                             disabled=hr_unknown, key=f"{prefix}_hr")
        atm = st.selectbox("열분해 분위기", ["(미보고)"] + atms, key=f"{prefix}_atm")
    if ("TWO_STAGE" in route or "POST_" in route) and steps == 1:
        st.caption("공정 경로가 다단계인데 단계 수는 1입니다. 원문 조건을 확인해 주세요.")
    st.caption("후처리·활성화 정보는 원문에 명시된 경우에만 입력합니다. 공란은 미보고로 유지됩니다.")
    c4, c5, c6 = st.columns(3)
    with c4:
        activation = st.text_input("활성화·개질 표기 (선택)", key=f"{prefix}_activation",
                                   placeholder="미보고")
    with c5:
        secondary_temp = st.text_input("2차 처리온도 °C (선택)", key=f"{prefix}_secondary_T",
                                       placeholder="미보고")
    with c6:
        secondary_hold = st.text_input("2차 유지시간 min (선택)", key=f"{prefix}_secondary_hold",
                                       placeholder="미보고")
    input_errors = []
    def _optional_number(value, label):
        if not value.strip():
            return np.nan
        try:
            v = float(value)
        except ValueError:
            input_errors.append(f"{label}: 숫자를 입력하세요.")
            return np.nan
        if not np.isfinite(v) or v < 0:
            input_errors.append(f"{label}: 0 이상의 유한한 수만 입력하세요.")
            return np.nan
        return v
    sec_temp = _optional_number(secondary_temp, "2차 처리온도")
    sec_hold = _optional_number(secondary_hold, "2차 유지시간")
    for e in input_errors:
        st.error(e)
    return {
        "_input_errors": input_errors,
        "manufacturing_route": route, "process_step_count": steps,
        "pyrolysis_temperature_C": pt, "pyrolysis_hold_min": hold,
        "heating_rate_C_min": np.nan if hr_unknown else hr,
        "pyrolysis_atmosphere": np.nan if atm == "(미보고)" else atm,
        "activation_or_modification": activation.strip() if activation.strip() else np.nan,
        "secondary_temperature_C": sec_temp,
        "secondary_hold_min": sec_hold,
    }


def _predict_row(model_obj, row: dict):
    return predict_from_contract(model_obj, row)


def _reference_panel(matched: str | None):
    if matched is None:
        return
    df, _ = _load_core()
    subset = related_records(df, matched)
    with st.expander(f"등록 원료의 문헌 관측치 확인 ({len(subset)}행) · 모델 예측과 별개"):
        if subset.empty:
            st.caption("해당 원료의 관측치가 없습니다.")
            return
        cols = ["doi", "pollutant", "pyrolysis_temperature_C", "C0_mg_L", "initial_pH", "dose_g_L", "target_value"]
        data = subset[cols].copy().rename(columns={
            "doi":"논문 DOI", "pollutant":"오염물질", "pyrolysis_temperature_C":"열분해온도 (°C)",
            "C0_mg_L":"C₀ (mg/L)", "initial_pH":"pH", "dose_g_L":"투입량 (g/L)",
            "target_value":"보고된 qe (mg/g)",
        })
        st.dataframe(data, use_container_width=True, hide_index=True)
        st.caption("위 값은 원문 보고 관측치이며 아래의 ML 예측값을 대신하지 않습니다. 평형 여부 미인증.")


def _prediction_output(models, ads, man, match, comp, use_model_b=True):
    if not ads["C0_mg_L"] > 0 or not ads["dose_g_L"] > 0:
        st.error("초기농도와 투입량은 양수여야 합니다.")
        return
    row = {**ads, **man, **comp, "feedstock_label_normalized":
           match.matched if match.matched else match.normalized}
    st.markdown("#### 예측 결과")
    cols = st.columns(2)
    for c, model_key in zip(cols, ["A", "B"]):
        if model_key == "B" and not use_model_b:
            with c:
                st.caption("Model B · 제조조건 포함")
                st.info("제조조건 입력을 선택하지 않아 실행하지 않았습니다.")
            continue
        if model_key == "B" and match.matched is None:
            with c:
                st.caption("Model B · 원료·제조조건 추가")
                st.info("미학습 원료에서는 Model B 예측값을 제공하지 않습니다. 미지 범주가 실제 원료 특성을 의미하지 않기 때문입니다.")
            continue
        if model_key not in models:
            continue
        raw, capped, cap = _predict_row(models[model_key], row)
        cv = _v16_summary().get("nested_cv", {}).get(model_key, {})
        with c:
            st.caption(f"Model {model_key} · {'흡착조건만' if model_key == 'A' else '원료·제조조건 추가'}")
            st.metric("예측 qe · 원시", f"{raw:,.2f} mg/g")
            st.caption(f"물리 제약 후: {capped:,.2f} mg/g  |  상한 C₀/dose = {cap:,.2f} mg/g")
            if abs(raw - capped) > 1e-9:
                st.warning("원시 예측이 물질수지 제약을 위반합니다. 보정값은 clipping 결과입니다.")
            if cv:
                st.caption(f"독립 DOI Nested CV · R² {cv['r2']:.3f} · MAE {cv['mae']:.1f} mg/g")
            used, ignored = applied_optional_fields(comp, models[model_key]["metadata"])
            if ignored:
                st.caption(f"추가 조성 {len(ignored)}개는 이 저장 모델에서 **사용되지 않음**")
            if used:
                st.caption(f"실제 사용된 추가 조성: {', '.join(used)}")
    if match.matched is None:
        st.warning("미학습 원료: Model A만 탐색적으로 표시합니다. 원료 특성을 반영한 예측이나 제조조건 추천은 불가능합니다.")
    st.caption("A와 B의 숫자 차이는 제조조건의 인과효과가 아닙니다. "
               "현재 Core 97의 qe는 엄격한 평형 흡착량으로 인증되지 않았습니다.")


def page_overview():
    _heading("Biochar Adsorption Workbench", "문헌 데이터 기반 바이오차 흡착성능 예측 · 분석용 프로토타입")
    df, rep = _load_core()
    models, active = _get_models()
    a, b, c = st.columns(3)
    a.metric("실험 관측치", f"{len(df)}")
    b.metric("독립 DOI", f"{df['doi'].nunique()}")
    c.metric("등록 원료 그룹", f"{df[fc.GROUP_FEEDSTOCK].nunique()}")
    _model_status_caption()
    st.markdown("#### 프로그램의 주요 기능")
    st.dataframe(pd.DataFrame([
        {"기능":"흡착량 예측", "모델":"A / B", "의미":"새 입력조건의 qe 모델 추정 · 실측값 조회와 분리"},
        {"기능":"제조조건 후보 탐색", "모델":"B", "의미":"원료·흡착조건 입력 → 문헌 제조조건 후보의 ML 점수 출력(최적조건 보장 아님)"},
        {"기능":"문헌 관측치 조회", "모델":"—", "의미":"일치하는 등록 원료의 논문 데이터 확인"},
        {"기능":"성능·한계 검증", "모델":"DOI CV", "의미":"일반화 오차·물리 제약·추천 검증"},
    ]), hide_index=True, use_container_width=True)
    st.markdown("#### Stage 1–2 연결 상태")
    st.info("Stage 1 v60은 원료명·열분해 및 활성화조건으로 BET/TPV를 예측하지만 원료 조성값을 실제 모델에서 제외합니다. "
            "독립 외부 검증 R²가 음수였으므로 Stage 1→2 자동 연결은 비활성입니다.")
    st.caption("구형 Stage 1 BET 모델은 별도의 `biochar_app.py`에서 실행할 수 있습니다. "
               "이 모델은 원소·공업분석 7항목을 사용하지만 v60과 학습 범위가 다릅니다.")
    with st.expander("모델 구성·검증 상태"):
        st.dataframe(pd.DataFrame([{"모델": k, "정의": v["desc"],
                                   "현재 저장됨": "예" if k in models else "아니오"}
                                  for k,v in fc.MODEL_DEFS.items()]),
                     hide_index=True, use_container_width=True)
        st.caption(f"활성 모델 버전: {active or '없음'} · Model D 비활성 · 평형 qe 미인증")


def _v43_scenarios_panel(row, vals, scopes, *, sensitivity=None, sweep_label=""):
    """Capture and compare user-approved scenarios, never refit or rank them."""
    st.markdown("#### 조건 비교 · 결과 저장")
    st.caption("슬라이더로 조건을 맞춘 다음 저장하세요. 다른 조건을 선택해 반복하면 한 표와 그래프에서 비교할 수 있습니다.")
    history = st.session_state.setdefault("v43_saved_scenarios", [])
    c1, c2 = st.columns([3, 1])
    with c1:
        label = st.text_input("시나리오 이름", value=f"조건 {len(history) + 1}",
                              max_chars=70, key="v43_case_label")
    with c2:
        st.write(" ")
        if st.button("현재 조건 저장", type="primary", key="v43_save_case"):
            try:
                snapshot = build_snapshot(label, row, vals,
                    scope_status={key: scope.status for key, scope in scopes.items()})
                st.session_state["v43_saved_scenarios"] = append_snapshot(history, snapshot)
                history = st.session_state["v43_saved_scenarios"]
            except (ValueError, TypeError, KeyError) as exc:
                st.error(f"현재 조건을 저장하지 못했습니다: {exc}")
    if not history:
        st.info("저장된 조건이 없습니다. 현재 값을 저장하면 조건별 예측값을 비교하고 결과를 내보낼 수 있습니다.")
        return
    actions = st.columns([2, 1, 1])
    with actions[0]:
        st.caption(f"현재 세션에 {len(history)}/{MAX_SCENARIOS}개 저장 · 최대 개수를 넘으면 가장 오래된 조건부터 교체")
    with actions[1]:
        if st.button("마지막 조건 삭제", key="v43_remove_last"):
            st.session_state["v43_saved_scenarios"] = list(history[:-1])
            history = st.session_state["v43_saved_scenarios"]
    with actions[2]:
        if st.button("저장 조건 초기화", key="v43_clear_cases"):
            st.session_state["v43_saved_scenarios"] = []
            history = []
    if not history:
        st.info("저장된 조건이 없습니다.")
        return
    table = comparison_frame(history)
    st.dataframe(table, use_container_width=True, hide_index=True)
    pollutants = set(table["오염물질"].dropna())
    feedstocks = set(table["원료"].dropna())
    if len(pollutants) > 1 or len(feedstocks) > 1:
        st.info("저장된 조건에 서로 다른 원료 또는 오염물질이 포함되어 있습니다. 그래프는 수치 비교용이며 성능 순위가 아닙니다.")
    plot = table.set_index("시나리오")[["A 제약 후 qe (mg/g)", "B 제약 후 qe (mg/g)"]]
    if not plot.isna().all().all():
        st.bar_chart(plot, y_label="물질수지 제약 후 qe (mg/g)", use_container_width=True)
    st.caption("막대의 qe는 물질수지 범위로 보정한 표시값입니다. 원시 qe와 제약 위반 여부는 위 표에 그대로 남겨둡니다.")
    a, b = st.columns(2)
    with a:
        st.download_button("저장 조건·예측 CSV", data=comparison_csv(history),
                           file_name="biochar_v43_scenarios.csv", mime="text/csv",
                           key="v43_export_cases_csv")
    with b:
        st.download_button("발표용 결과 보고서 (HTML)",
                           data=printable_report(history, sensitivity=sensitivity,
                                                 sweep_label=sweep_label).encode("utf-8"),
                           file_name="biochar_v43_report.html", mime="text/html",
                           key="v43_export_cases_html")
    st.caption("보고서 HTML은 인터넷 없이 열 수 있으며 브라우저의 인쇄 기능으로 PDF 저장할 수 있습니다. 시나리오는 현재 브라우저 세션에서만 유지됩니다.")


def _live_prediction_panel(models, ads, man, match, comp, *, use_model_b: bool):
    """Render directly from CURRENT widget values; no st.button-gated results."""
    feedstock = match.matched if match.matched else match.normalized
    row = {**ads, **man, **comp, "feedstock_label_normalized": feedstock}
    enabled = {"A": models["A"]} if "A" in models else {}
    if use_model_b and match.matched is not None and "B" in models:
        enabled["B"] = models["B"]
    if not enabled:
        st.warning("예측 모델이 없습니다.")
        return
    training_core, _ = _load_core()
    # Applicability ALWAYS uses the *active* Core-97 rather than the enlarged
    # v40 research training cohort. Passing feature limits is not certification.
    scopes = {key: inspect_active_scope(training_core, row, key) for key in enabled}
    st.markdown("#### 입력 적용범위 · 활성 모델 기준")
    for key, scope in scopes.items():
        st.caption(f"Model {key} · Core {scope.n_reference_rows}행 / {scope.n_reference_doi} DOI · "
                   f"같은 {'오염물질·원료' if key=='B' else '오염물질'} 관측 "
                   f"{scope.n_cohort_rows}행 / {scope.n_cohort_doi} DOI")
        if scope.status == "INVALID":
            st.error(f"Model {key}: 필수 입력 오류. " + "; ".join(scope.blockers[:3]))
            return
        if scope.status in ("OUTSIDE_RANGE", "LIMITED_EVIDENCE"):
            st.warning(f"Model {key}: 적용범위 확인 필요 — " +
                       ("; ".join(scope.warnings[:3]) if scope.warnings else "동일 실험군 근거 부족"))
        if key == "B" and not scope.can_rank:
            st.caption("현재 조합에서 검증된 최적 제조조건을 추천하지 않습니다. 아래 예측은 탐색적 응답입니다.")
    with st.expander("범위·미보고율 자세히 보기 (이 검사만으로 예측 정확도 보장 불가)"):
        for key, scope in scopes.items():
            st.markdown(f"**Model {key} · 입력 원자료 보고율**")
            st.dataframe(scope_to_table(scope), hide_index=True, use_container_width=True)
            st.caption(scope.note)
    # Do not suppress an exploratory curve just because a sample is OOD;
    # show the provenance warning prominently alongside the raw model output.
    st.markdown("#### 실시간 예측 결과")
    st.caption("슬라이더 변경 시 저장된 모델에서 자동으로 다시 추론합니다. 재학습·기존 실측값 반환이 아닙니다.")
    try:
        vals = {key: eval_one(obj, row) for key, obj in enabled.items()}
    except (ValueError, TypeError, KeyError) as exc:
        st.error(f"현재 조건에서 예측을 계산하지 못했습니다: {exc}")
        return
    if st.button("현재 조건을 변화량 기준으로 고정", key="v37_pin_reference"):
        st.session_state["v37_reference"] = {
            "conditions": {k: ads[k] for k in ads},
            "predictions": {k: v["qe_raw"] for k, v in vals.items()},
        }
    ref = st.session_state.get("v37_reference")
    if ref:
        st.caption("기준 조건 대비 변화량을 아래에 표시합니다. 그래프는 항상 현재 조건 기준으로 계산됩니다.")
    cols = st.columns(len(vals))
    for col, (label, pred) in zip(cols, vals.items()):
        with col:
            ref_pred = ref.get("predictions", {}).get(label) if ref else None
            delta = f"{pred['qe_raw']-ref_pred:+.2f} mg/g" if ref_pred is not None else None
            st.metric(f"Model {label} · 원시 qe", f"{pred['qe_raw']:.2f} mg/g", delta=delta)
            st.caption(f"예측 제거율(원시): {pred['removal_raw_pct']:.1f}% · 물질수지 상한: {pred['cap']:.2f} mg/g")
            if pred['violates']:
                st.warning(f"비물리적 원시 예측. clipping 후 {pred['qe_clipped']:.2f} mg/g — 성능 검증값 아님")
            cv = _v16_summary().get("nested_cv", {}).get(label, {})
            if cv:
                st.caption(f"기존 모델의 DOI Nested CV: MAE {cv.get('mae', float('nan')):.1f} mg/g")
    if use_model_b and match.matched is None:
        st.info("미학습 원료: Model B를 표시하지 않습니다. Model A는 원료별 차이를 구분하지 않습니다.")
    if comp:
        st.caption("현재 A/B에는 입력한 원소·공업분석값이 반영되지 않습니다.")
    st.markdown("#### 변수별 민감도 · 모델 응답 곡선")
    available = list(VARIABLES)
    variable = st.selectbox("변화시킬 변수", available, format_func=lambda v: VARIABLES[v][0],
                            key="v37_sweep_variable")
    label, minval, maxval, step, _ = VARIABLES[variable]
    cur = float(ads[variable])
    # Narrower sweep centred near current input where possible; adjust manually as needed.
    defaults = {
        "initial_pH": (2.0, 10.0), "C0_mg_L": (10.0, 250.0),
        "dose_g_L": (0.5, 5.0), "adsorption_T_C": (15.0, 40.0),
        "contact_time_min": (60.0, 1800.0),
    }
    dlow, dhigh = defaults[variable]
    lo, hi = st.slider("그래프 분석 구간", min_value=minval, max_value=maxval,
                       value=(dlow, dhigh), step=step, key=f"v37_sweep_limits_{variable}")
    sensitivity_table = None
    try:
        # Keep the current marker in range only when the current value is within the chosen interval.
        sweep = make_sweep_values(variable, lo, hi, cur)
        frame = sensitivity_frame(enabled, row, variable=variable, values=sweep,
                                  show_model_b="B" in enabled)
        if frame.empty:
            st.info("이 범위에 표시할 예측이 없습니다.")
            return
        sensitivity_table = frame
        import altair as alt
        long = frame.rename(columns={"qe_raw": "qe (mg/g)"})
        base = alt.Chart(long).encode(x=alt.X("x:Q", title=label, scale=alt.Scale(zero=False)))
        lines = base.mark_line().encode(y=alt.Y("qe (mg/g):Q"), color="model:N")
        marks = base.transform_filter(alt.datum.x == cur).mark_point(filled=True, size=110).encode(
            y="qe (mg/g):Q", color="model:N", tooltip=["model:N", alt.Tooltip("qe (mg/g):Q",format=".2f")])
        cap_line = base.transform_filter(alt.datum.model == "A").mark_line(strokeDash=[4,4],opacity=0.5).encode(
            y=alt.Y("cap:Q", title="qe (mg/g)"))
        st.altair_chart((lines + marks + cap_line).properties(height=290), use_container_width=True)
        removal = frame.rename(columns={"removal_clipped_pct": "제약 후 제거율 (%)"})
        st.altair_chart(alt.Chart(removal).mark_line().encode(
            x=alt.X("x:Q",title=label,scale=alt.Scale(zero=False)),
            y=alt.Y("제약 후 제거율 (%):Q", scale=alt.Scale(domain=[0, 100])), color="model:N"
        ).properties(height=210), use_container_width=True)
        st.caption("아래 제거율은 0~100% 범위로 제한한 값입니다. 물리 제약을 위반한 원시 출력은 상단 qe 곡선과 CSV에 남아 있습니다.")
        violations = int(frame["violates"].sum())
        st.caption(f"현재 입력: {label}={cur:g} · 실선=원시 qe 예측 · 점선=물질수지 상한 · "
                   f"비물리적 예측 {violations}/{len(frame)}개 (clipping으로 숨기지 않음). "
                   "이 곡선은 다른 조건 고정 시 모델 응답이며 인과효과나 검증된 최적조건이 아닙니다.")
        with st.expander("그래프 수치표 / CSV"):
            st.dataframe(frame, hide_index=True, use_container_width=True)
            st.download_button("민감도 데이터 CSV",frame.to_csv(index=False).encode('utf-8-sig'),
                               "biochar_sensitivity.csv", "text/csv", key="v37_sweep_export")
    except (ValueError, TypeError, KeyError) as exc:
        st.error(f"민감도 그래프를 계산하지 못했습니다: {exc}")
    _v43_scenarios_panel(row, vals, scopes, sensitivity=sensitivity_table, sweep_label=label)


def page_predict():
    _heading("흡착성능 예측", "① 원료 선택 → ② 슬라이더 조절 → ③ 그래프 확인 → ④ 조건 저장·보고서 내보내기")
    _model_status_caption()
    handed = st.session_state.get("v44_handoff_summary")
    if handed:
        with st.expander("BET 화면에서 가져온 제조조건", expanded=True):
            st.caption("Stage 1에서 가져온 **마지막 제조 입력값**입니다. 아래에서 수정하면 현재 값과 달라질 수 있습니다. "
                       "이전 화면의 BET 예측값은 Stage 2 모델에 전달하지 않습니다.")
            c1, c2, c3 = st.columns(3)
            c1.metric("제조온도", f"{handed['pyrolysis_temperature_C']:g} °C")
            c2.metric("유지시간", f"{handed['pyrolysis_hold_min']:g} min")
            c3.metric("승온속도", f"{handed['heating_rate_C_min']:g} °C/min")
            st.caption("연결된 원료명: " + handed["feedstock_note_only"] +
                       " · 실제 Model B 학습 원료 여부는 아래에서 확인")
    models, _ = _get_models()
    if not models:
        st.error("저장된 Stage 2 모델을 찾지 못했습니다. 배포 모델 경로를 확인하세요.")
        return
    st.markdown("#### 01 · 바이오매스")
    match = _feedstock_control()
    _reference_panel(match.matched)
    with st.expander("02 · 원료 조성 (선택 · 현재 A/B 예측에는 미사용)", expanded=False):
        comp, comp_errors = _composition_controls()
    st.markdown("#### 03 · 흡착 실험조건")
    ads = _adsorption_widgets()
    use_model_b = st.checkbox("제조조건을 포함하는 Model B와 함께 비교", value=False, key="s2_enable_B")
    man = {"_input_errors": []}
    if use_model_b:
        with st.container(border=True):
            man = _manufacture_widgets()
    for err in comp_errors + man["_input_errors"]:
        st.error(err)
    if match.status == "empty" or comp_errors or man["_input_errors"]:
        st.info("입력 오류를 수정하면 실시간 예측이 표시됩니다.")
        return
    _live_prediction_panel(models, ads, man, match, comp, use_model_b=use_model_b)


def page_screening():
    _heading("제조조건 후보 탐색", "원료와 목표 흡착조건을 먼저 지정하면, 문헌에 보고된 서로 다른 제조조건의 ML 추정치를 비교합니다.")
    _model_status_caption()
    st.warning("현재 Model B는 DOI 독립 검증에서 Model A보다 성능이 낮습니다. 이 화면은 실험 후보 정렬을 위한 탐색적 비교이며, 최적 제조조건을 입증하거나 보장하지 않습니다.")
    st.caption("v28 추가 감사: 이전 순위 검증 11개 비교군에는 열분해온도가 달라지는 실험군이 없었습니다. 현재 순위는 온도 최적화의 검증 근거가 아닙니다.")
    models, _ = _get_models()
    if "B" not in models:
        st.error("저장된 Model B가 없어 후보를 평가할 수 없습니다.")
        return
    st.markdown("#### 01 · 원료 (자연어 입력)")
    matched = _feedstock_control(key="scr_bio", default="rice husk")
    st.markdown("#### 02 · 원료 조성 (선택)")
    _, composition_errors = _composition_controls()
    for e in composition_errors:
        st.error(e)
    st.caption("조성은 현재 Model B에 사용되지 않습니다. 값을 입력해도 추천 순위가 개선되거나 보정되지는 않습니다.")
    st.markdown("#### 03 · 목표 흡착조건")
    ads = _adsorption_widgets(prefix="scr")
    core, _ = _load_core()
    screen = observed_manufacturing_candidates(core, matched.matched, ads["pollutant"])
    if screen.status == "comparable":
        st.success(screen.message)
    else:
        st.info(screen.message)
        if screen.status == "single":
            st.caption("하나의 제조조건만 관측되었습니다. 동일 조건에서 C₀·pH 등 흡착조건을 바꾼 여러 실험을 제조조건 비교로 중복 계산하지 않습니다.")
        if screen.status == "unknown_feedstock":
            st.caption("신규 원료 이름을 다른 원료로 자동 대체하지 않습니다. Model A는 '흡착성능 예측' 탭에서 별도 사용 가능합니다.")
    if screen.count_unique:
        with st.expander("문헌에 기록된 제조조건만 보기", expanded=False):
            st.dataframe(screen.candidates, hide_index=True, use_container_width=True)
    if screen.status != "comparable":
        st.caption("이 입력 조합에 대해 제조조건 순위를 산출하지 않습니다. 원료·오염물질 변경 또는 데이터 보완이 필요합니다.")
        return
    # v27: connect the v26 conservative OOD gate before ANY model scoring.
    # Only explore ranked candidates when target adsorption conditions are
    # non-missing and within the training cohort's empirical range.
    scope = assess_scope(core, matched.matched, ads["pollutant"], ads)
    if not scope.eligible_for_exploratory_ranking:
        st.error("현재 입력조건은 학습자료의 지원범위를 벗어나 정량 제조조건 점수를 계산하지 않습니다.")
        st.caption("차단 사유: " + ", ".join(scope.reasons))
        st.caption(scope.disclaimer)
        return
    # Only one DOI per active feedstock/pollutant pair is present in v15 Core97.
    # Keep read-only exploration accessible but remove ALL recommendation ranks.
    pair_row = {**ads, "feedstock_label_normalized": matched.matched,
                **screen.candidates.iloc[0].to_dict()}
    active_scope = inspect_active_scope(core, pair_row, "B")
    if not active_scope.can_rank:
        st.warning("추천 제한: 같은 원료·오염물질의 독립 DOI "
                   f"{active_scope.n_cohort_doi}편 · 동일 흡착조건의 온도 비교군 "
                   f"{active_scope.n_controlled_temperature_panels}개. 일반화가 확인되지 않아 "
                   "순위나 최고 제조조건을 제시하지 않습니다.")
        st.caption("관측된 제조조건의 *탐색적 모델 응답*은 아래에서 별도로 조회할 수 있습니다. "
                   "점수 차이가 실제 우열을 의미하지 않습니다.")
        if st.button("미검증 제조조건 점수표만 조회 (추천 아님)", key="v41_unvalidated_scores"):
            neutral = score_observed_candidates(screen, models["B"], ads)
            if not neutral.empty:
                # Explicitly undo score sorting and omit rank so the UI cannot
                # silently imply a validated winner or a recommended order.
                neutral = neutral.sort_values("candidate_id").drop(columns=["rank", "eligible_for_ranking"])
                st.dataframe(neutral, hide_index=True, use_container_width=True)
                st.caption("candidate_id는 원자료 입력 순서입니다. 원시 qe는 검증되지 않은 모델 출력이며 "
                           "물리 제약 위반 후보도 제거하지 않고 표시합니다.")
        return
    st.caption("지원범위의 최소·최대값 검사만 통과한 상태입니다. 물리적 타당성과 제조조건 순위의 정확성을 보증하지 않습니다.")
    if st.button("제조조건 후보 점수 계산", type="primary", disabled=bool(composition_errors)):
        result = score_observed_candidates(screen, models["B"], ads)
        if result.empty:
            st.info("비교 가능한 후보가 없습니다.")
            return
        st.markdown("#### 04 · 제조조건 후보 비교 결과 (출력)")
        st.dataframe(result, hide_index=True, use_container_width=True)
        valid = result[result["eligible_for_ranking"]]
        if valid.empty:
            st.error("모든 후보에서 물질수지 제약 위반이 발생하여 순위를 제시하지 않습니다.")
        elif valid["qe_raw_mg_g"].nunique() <= 1:
            st.info("현재 모델이 유효 후보들 사이의 흡착성능 차이를 구별하지 못했습니다(예측값 동률). 최고 조건을 지정하지 않습니다.")
        else:
            top = valid.head(min(3, len(valid)))
            st.caption(f"물리 제약을 위반하지 않은 후보 {len(valid)}/{len(result)}개. 상위 {len(top)}개는 모델 예측에 따른 실험 후보이지 검증된 최적 제조조건이 아닙니다. 동률 후보는 같은 순위입니다.")
            for _, candidate in top.iterrows():
                parts = [f"제조경로: {candidate['manufacturing_route']}"]
                if pd.notna(candidate['pyrolysis_temperature_C']):
                    parts.append(f"열분해온도: {candidate['pyrolysis_temperature_C']:g}°C")
                if pd.notna(candidate['pyrolysis_hold_min']):
                    parts.append(f"유지시간: {candidate['pyrolysis_hold_min']:g} min")
                st.markdown(f"**후보 {int(candidate['rank'])}** · 예상 qe {candidate['qe_raw_mg_g']:.2f} mg/g · " + " / ".join(parts))
            st.bar_chart(valid.set_index("candidate_id")["qe_raw_mg_g"], y_label="예상 qe (mg/g)")
        st.download_button("후보 및 예측 CSV 저장", result.to_csv(index=False).encode("utf-8-sig"),
                           "stage2_manufacture_candidate_screen.csv", "text/csv")
        st.caption("후보는 학습 데이터의 제조입력값에서만 수집합니다. 원료명·오염물질 조합을 그대로 유지하며 보고된 실측 qe를 점수로 사용하지 않습니다."
                   " 불완전한 제조정보는 미보고 상태로 유지하고, 제조조건의 전역 최적화·신규 원료 일반화는 제공하지 않습니다.")


def page_compare():
    _heading("제조조건 비교", "동일 원료와 흡착조건에서 열분해온도 후보별 모델 응답을 비교합니다.")
    _model_status_caption()
    models, _ = _get_models()
    if "B" not in models:
        st.warning("저장된 Model B가 없습니다.")
        return
    match = _feedstock_control(key="cmp_bio", default="rice husk")
    if match.matched is None:
        st.warning("미학습 원료는 Model B 온도 스윕의 유효성이 검증되지 않아 실행하지 않습니다.")
        return
    st.warning("수동 온도 스윕은 관측된 제조조건 범위를 벗어날 수 있으며 최적조건 추천이 아닙니다.")
    ads = _adsorption_widgets(prefix="cmp")
    man = _manufacture_widgets(prefix="cmp")
    a,b,c = st.columns(3)
    with a:
        low = st.number_input("비교 시작온도 (°C)", value=300.0, step=50.0)
    with b:
        high = st.number_input("비교 종료온도 (°C)", value=800.0, step=50.0)
    with c:
        step = st.number_input("온도 간격 (°C)", min_value=1.0, value=50.0, step=10.0)
    if high <= low or (high - low) / step > 100:
        st.error("종료온도를 시작온도보다 높게 하고 후보 수는 최대 101개로 설정해 주세요.")
        return
    if match.status != "empty" and not man["_input_errors"]:
        rows = []
        for t in np.arange(low, high + 1e-6, step):
            row = {**ads, **man, "feedstock_label_normalized":
                   match.matched or match.normalized, "pyrolysis_temperature_C": float(t)}
            raw, capped, cap = _predict_row(models["B"], row)
            rows.append({"열분해온도 (°C)": float(t), "원시 qe (mg/g)": raw,
                         "제약 후 qe (mg/g)": capped,
                         "제약 위반": abs(raw-capped)>1e-9})
        df = pd.DataFrame(rows)
        st.line_chart(df.set_index("열분해온도 (°C)")[["원시 qe (mg/g)", "제약 후 qe (mg/g)"]])
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("비교 CSV 저장", df.to_csv(index=False).encode("utf-8-sig"),
                           "stage2_condition_comparison.csv", "text/csv")
        st.warning("이 곡선은 새로운 실험 결과가 아니라 Model B의 추정 응답입니다. "
                   "제조조건 추가가 Model A보다 일반화 성능을 개선하지 못했고, 전역 최적조건 추천으로 해석하면 안 됩니다.")
        if match.matched is None:
            st.caption("미학습 원료 외삽으로 분류되어 순위의 유효성이 검증되지 않았습니다.")


def page_validation():
    _heading("모델 검증", "저장된 검증 산출물에 기반한 성능과 적용 한계를 확인합니다.")
    summary = _v16_summary()
    nested = summary.get("nested_cv", {})
    if nested:
        st.markdown("#### DOI 그룹 Nested CV · reported uptake qe")
        rows = []
        for m in ("A", "B"):
            if m in nested:
                r = nested[m]
                rows.append({"모델": m, "R²": r["r2"], "MAE (mg/g)": r["mae"],
                             "RMSE (mg/g)": r["rmse"]})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        ci = summary.get("bootstrap_AB", {})
        if ci:
            st.caption(f"MAE(B−A): {ci.get('point_estimate')} mg/g · DOI 부트스트랩 95% CI "
                       f"[{ci.get('ci95_low')}, {ci.get('ci95_high')}] mg/g")
    else:
        st.info("v16 요약 파일이 없습니다. 재학습 없이 배포 메타데이터를 확인하세요.")
    st.markdown("#### Chen16 · 동일 논문 내 조건 순위 진단")
    chen = summary.get("chen16_single_fit_holdout", {})
    if chen:
        rows = [{"모델": key, "Pairwise": val.get("pairwise_ranking_accuracy"),
                 "Top-1 (4후보)": val.get("top1_hit_rate"),
                 "Regret (%p)": val.get("regret_pp_mean")}
                for key,val in chen.get("results", {}).items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption(f"{chen.get('n_train')}개 학습 / {chen.get('n_test')}개 검증 × "
                   f"{chen.get('n_repeats')}회 반복. 동일 논문 내 평가이며 분할 결과는 서로 독립적이지 않습니다. "
                   "Dummy의 예측 동률은 Pairwise에서 불일치 처리됐으므로 무작위 순위 성능과 동일하지 않습니다.")
    v28_path = _REPO / "reports" / "v28_feature_identifiability" / "summary.json"
    if v28_path.is_file():
        v28 = json.loads(v28_path.read_text(encoding="utf-8"))
        st.markdown("#### 제조조건 순위 모델의 판별 가능성 · v28")
        st.caption(f"열분해온도 차이가 있는 독립 비교군 {v28['groups_with_verified_temperature_contrast']}개 / "
                   f"평가 가능한 {v28['evaluable_groups']}개 그룹. "
                   f"동일 제조입력으로 구분되지 않는 {v28['pairs_with_identical_manufacturing_input_signature']}쌍 / "
                   f"{v28['pairwise_comparisons']}쌍. 제조경로·개질 차이를 온도 최적화로 해석하면 안 됩니다.")
    st.markdown("#### 해석 제한")
    st.markdown("- 엄격한 평형 qe가 확인된 관측치는 아직 없습니다.\n"
                "- LOFO의 30개 분할 중 7개는 train/test DOI가 중복됩니다.\n"
                "- Stage 1 v60의 새로운 논문 독립 검증 R²는 BET/TPV 모두 음수였습니다.\n"
                "- 미학습 원료의 예측값을 원료별 검증된 추정치로 간주할 수 없습니다.\n"
                "- 원료 조성값을 입력할 수 있어도 현재 저장된 Stage 2 예측기에는 사용되지 않습니다.")


from pyrolysis.biochar_ui.v48_material_properties_page import render_v48_properties
from pyrolysis.biochar_ui.phase8_evidence_page import render_phase8_evidence


# Four stable user-facing destinations. Old analytical pages remain available under
# secondary choices without adding a new item to the navigation at every research version.
def page_manufacturing_workspace():
    mode = st.radio(
        "제조조건 분석 유형", ["문헌 후보 비교", "연구용 온도 민감도"],
        horizontal=True, key="workbench_manufacturing_mode",
    )
    if mode == "문헌 후보 비교":
        page_screening()
    else:
        st.info("온도 민감도는 수동 모델 진단입니다. 열분해온도 변경에 대한 독립 추천 검증은 아직 없습니다.")
        page_compare()


def page_data_validation_workspace():
    options = ["모델 개요", "예측 검증", "Phase 8 원자료", "입력 적용범위"]
    v24_dir = _REPO / "data" / "imports" / "v24_source_audit"
    if (v24_dir / "stage1_new33_BET_TPV_SOURCE_STAGING.csv").is_file():
        options.append("신규 논문 검증자료")
    mode = st.radio(
        "데이터·검증 보기", options, horizontal=True,
        key="workbench_evidence_mode",
    )
    if mode == "모델 개요":
        page_overview()
    elif mode == "예측 검증":
        page_validation()
    elif mode == "입력 적용범위":
        st.markdown("### 모델별 데이터 적용범위")
        core, _ = _load_core()
        st.caption("활성 A/B 모델: Core 97 원자료만 사용해 범위와 미보고율을 표시합니다. "
                   "범위 통과는 예측 신뢰도를 인증하지 않습니다.")
        for key in ("A", "B"):
            example = core.iloc[0].to_dict()
            scope = inspect_active_scope(core, example, key)
            st.markdown(f"#### 활성 Model {key} · {scope.n_reference_doi} DOI")
            st.dataframe(scope_to_table(scope), hide_index=True, use_container_width=True)
        research = _REPO / "reports" / "v40_new9" / "research_training_envelopes.json"
        if research.is_file():
            st.markdown("### v40 확장 데이터 (연구용 · 활성 모델과 별개)")
            try:
                payload = json.loads(research.read_text(encoding="utf-8"))
                st.dataframe(pd.DataFrame([
                    {"연구용 데이터": key, "행": val.get("n_rows"), "DOI": val.get("n_doi")}
                    for key, val in payload.items()
                ]), hide_index=True, use_container_width=True)
            except (OSError, ValueError, TypeError) as exc:
                st.warning(f"연구용 범위 요약 파일을 읽지 못했습니다: {exc}")
            st.caption("v40 데이터가 더 크더라도 과거 활성 A/B 모델이 이를 학습했다는 뜻은 아닙니다.")
        metrics = _REPO / "reports" / "v40_new9" / "model_checks" / "all_DOI_OOF_metrics.csv"
        if metrics.is_file():
            m = pd.read_csv(metrics)
            s1 = m.loc[m["stage"] == "Stage1", ["target", "model", "n", "n_doi", "MAE", "R2", "raw_negative"]]
            if not s1.empty:
                st.markdown("### Stage 1 모델 비교 · 연구용 DOI 그룹 OOF")
                st.dataframe(s1, hide_index=True, use_container_width=True)
                st.caption("Poisson 모델은 연구 후보이며 현재 Stage 1 배포 모델로 자동 전환하지 않습니다. "
                           "동일 DOI 그룹 교차검증 성능은 미사용 외부 DOI의 블라인드 평가와 다릅니다.")
    elif mode == "Phase 8 원자료":
        render_phase8_evidence()
    else:
        st.header("신규 문헌 데이터 · 연구용 후보")
        st.warning("원문 추출·물질수지 검사 단계의 데이터입니다. 배포 모델 학습 완료 자료가 아닙니다.")
        choices = {
            "제조시료 BET·TPV": "stage1_new33_BET_TPV_SOURCE_STAGING.csv",
            "흡착량 검증 통과 후보": "stage2_new69_PHYSICAL_PASS_STILL_NOT_TRAINED.csv",
            "흡착량 물질수지 격리": "stage2_new3_PHYSICAL_FAIL_QUARANTINE.csv",
            "별도 qmax": "all_qmax_32_FITTED_NOT_QE.csv",
        }
        selection = st.selectbox("검토할 원문 데이터", list(choices), key="v28_literature_table")
        path = v24_dir / choices[selection]
        frame = pd.read_csv(path)
        st.caption(f"원문 검증자료 {len(frame)}행 · 배포 모델 미반영")
        st.dataframe(frame, use_container_width=True, hide_index=True)
        st.download_button(
            "표 CSV 저장", frame.to_csv(index=False).encode("utf-8-sig"),
            f"biochar_v28_{choices[selection]}", "text/csv",
        )


PAGES = {
    "흡착성능 예측": page_predict,
    "제조조건 탐색": page_manufacturing_workspace,
    "BET 물성 예측": render_v48_properties,
    "데이터·모델 검증": page_data_validation_workspace,
}


def render_stage2_app():
    st.set_page_config(page_title="Biochar · Adsorption Workbench", layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown("""<style>
        .block-container {max-width: 1220px; padding-top: 2.1rem;}
        h1,h2,h3 {color: #193650; letter-spacing: -0.025em;}
        div[data-testid="stMetric"] {background: #f5f8fa; border: 1px solid #dbe5ed;
          border-radius: 8px; padding: 13px 18px;}
        div[data-testid="stMetricValue"] {font-variant-numeric:tabular-nums;}
        section[data-testid="stSidebar"] {border-right:1px solid #e2e9ee;}
        .stButton > button[kind="primary"] {background:#1d5279; border:0; border-radius:6px;}
        .stButton > button {border-radius:6px;}
    </style>""", unsafe_allow_html=True)
    st.sidebar.markdown("### BIOCHAR / WORKBENCH")
    st.sidebar.caption("제조조건 · 물성 · 흡착성능 예측")
    st.sidebar.caption("① BET·TPV 물성 → ② 흡착성능 qe → ③ 조건 비교 → ④ 데이터 검증")
    choice = st.sidebar.radio("주요 기능", list(PAGES.keys()), label_visibility="collapsed",
                              key="v48_navigation")
    st.sidebar.divider()
    st.sidebar.caption("활성 Stage 2: Core 97 · 연구 데이터: Stage 1 417 / Stage 2 140")
    with st.sidebar.expander("현재 모델 구조 · 반드시 확인"):
        st.markdown("**BET (v20):** 원료 조성 모델 · 연구용.\n\n"
                    "**BET/TPV (v46):** 문헌 시료 제조특성 모델 · 비활성 연구용.\n\n"
                    "**qe (A/B):** Core 97 활성 모델. BET/TPV 예측값을 qe 입력으로 사용하지 않음.\n\n"
                    "**제조조건 비교:** 탐색적 응답이며 실험 최적화가 아님.")
    PAGES[choice]()
