"""Stage 1 v48: legacy composition BET plus explicit research-only BET/TPV view."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import joblib

from pyrolysis.biochar_ui.stage1_bet_v20_page import render_stage1_bet_v20
from pyrolysis.biochar_stage1.v48_research_bridge import (
    DATA_REL, MODEL_FILES, TARGETS, UNITS, load_research_data, specimen_row,
    source_temperature_range, predict_properties, response_curve,
    measurement_note, export_comparison,
)

ROOT = Path(__file__).resolve().parents[2]


@st.cache_data(show_spinner=False)
def _data():
    return load_research_data(ROOT)


@st.cache_resource(show_spinner=False)
def _models():
    directory = ROOT / "deployment" / "research_v46_NOT_ACTIVE"
    return {key: joblib.load(directory / file) for key, file in MODEL_FILES.items()}


def render_v48_properties():
    st.markdown("## 바이오차 물성 분석 · BET / TPV")
    st.caption("원료·제조조건 → 저장 모델 추론 → 온도 민감도 → 결과 CSV · Stage 2와 구분")
    overview, research = st.tabs(["원료 조성 기반 BET · v20", "확장 제조조건 기반 BET·TPV · v46 연구용"])
    with overview:
        render_stage1_bet_v20()
    with research:
        st.warning("**v46 연구모델 (비활성)** — 전체 후보자료로 재학습한 BET·TPV 모델입니다. "
                   "실험 시료의 제조 입력을 불러오는 연구·시연 기능이며, 별도 외부 검증이나 최적 제조조건 추천이 아닙니다.")
        try:
            data = _data()
        except (OSError, ValueError, KeyError) as error:
            st.error(f"물성 데이터 로딩 실패: {error}")
            return
        st.markdown("#### 01 · 문헌 시료의 실제 입력조건 선택")
        # An unreported temperature must never be invented for a UI slider.
        usable = pd.to_numeric(data["pyrolysis_temperature_C"], errors="coerce")
        data = data.loc[usable.notna() & (usable > 0)].copy()
        st.caption("제조온도 실측 보고가 있는 시료만 선택할 수 있습니다. 미보고 온도는 보완하지 않습니다.")
        feedstocks = sorted(data["feedstock"].dropna().unique().tolist())
        feedstock = st.selectbox("원료", feedstocks, key="v48_feedstock")
        subset = data.loc[data.feedstock == feedstock].copy()
        ids = subset.material_id.astype(str).tolist()
        identifier = st.selectbox("원문 제조시료", ids, key="v48_specimen")
        row = specimen_row(data, identifier)
        note = measurement_note(row)
        st.caption(f"원본 시료: {identifier} · DOI: {note['doi']} · "
                   f"제조경로: {row.get('activation_method', '미보고')}")
        bounds = source_temperature_range(data, row)
        if bounds:
            low, high = bounds
            temp = st.slider("제조온도 비교 (°C) · 해당 원료·경로 관측 범위",
                             min_value=float(low), max_value=float(high),
                             value=float(np.clip(float(row.pyrolysis_temperature_C), low, high)),
                             step=1.0, key="v48_temperature")
        else:
            temp = float(row.pyrolysis_temperature_C) if pd.notna(row.pyrolysis_temperature_C) else None
            st.info("같은 원료·제조경로에서 서로 다른 온도가 확인되지 않아 온도 변경을 허용하지 않습니다.")
        columns = ["pyrolysis_temperature_C", "pyrolysis_hold_min", "heating_rate_C_min",
                   "pyrolysis_atmosphere", "activation_method", "activation_agent",
                   "activation_agent_feedstock_mass_ratio", "activation_temperature_C", "activation_time_min"]
        with st.expander("적용된 실제 제조조건 · 미보고는 결측 처리", expanded=False):
            st.dataframe(pd.DataFrame([row[columns].to_dict()]).T.rename(columns={0:"선택 시료 값"}),
                         use_container_width=True)
        if note["missing_training_inputs"]:
            st.caption("원문에 보고되지 않아 모델 내부 대치가 적용되는 입력: " +
                       ", ".join(note["missing_training_inputs"]))
        if note["source_ratio_not_in_model"]:
            st.warning("이 시료의 K₂CO₃/char 또는 FeCl₃/char 비율은 원문으로 보존되지만 "
                       "현재 모델 특징에 반영되지 않습니다. 투입비별 성능 차이를 판단할 수 없습니다.")
        st.markdown("#### 02 · BET·TPV 연구용 예측")
        if temp is None:
            st.error("원문 제조온도가 없어 예측할 수 없습니다.")
            return
        try:
            models = _models()
            values = predict_properties(models, row, temperature_C=temp)
            curve = response_curve(models, data, row)
        except Exception as exc:
            st.error("v46 연구모델을 불러오거나 예측하지 못했습니다. "
                     "Windows 환경에서 v46 모델을 로컬 재학습한 파일인지 확인하세요. "
                     f"({type(exc).__name__}: {exc})")
            return
        left, right = st.columns(2)
        left.metric("예측 BET · v46 연구용", f"{values['BET']:,.2f} m²/g")
        right.metric("예측 TPV · v46 연구용", f"{values['TPV']:,.4f} cm³/g")
        has_changed = abs(float(temp) - float(row.pyrolysis_temperature_C)) > 1e-8
        measured = [
            {"물성": "BET (m²/g)", "원문 시료 실측": note["measured_BET"], "현재온도 모델 예측": values["BET"]},
            {"물성": "TPV (cm³/g)", "원문 시료 실측": note["measured_TPV"], "현재온도 모델 예측": values["TPV"]},
        ]
        st.dataframe(pd.DataFrame(measured), hide_index=True, use_container_width=True)
        st.caption("원문 실측값은 선택한 원래 시료의 실험값입니다. "
                   + ("지금 변경한 제조온도에서 측정한 값이 아닙니다." if has_changed else "학습에 포함된 시료라 모델 예측과의 근접성이 외부 검증을 뜻하지 않습니다."))
        st.markdown("#### 03 · 열분해온도에 따른 연구모델 응답")
        if curve.empty:
            st.caption("비교할 동일 원료·제조경로의 복수 온도가 없어 곡선을 만들지 않았습니다.")
        else:
            a, b = st.columns(2)
            with a:
                st.line_chart(curve.set_index("열분해온도 (°C)")[["예측 BET (m²/g)"]], use_container_width=True)
            with b:
                st.line_chart(curve.set_index("열분해온도 (°C)")[["예측 TPV (cm³/g)"]], use_container_width=True)
            st.download_button("온도별 BET·TPV 곡선 CSV", curve.to_csv(index=False).encode("utf-8-sig"),
                               file_name="biochar_v48_stage1_temperature.csv", mime="text/csv")
        st.download_button("시료·실측·예측 비교 CSV",
                           export_comparison(row, values, note, pyrolysis_temperature_C=temp),
                           file_name="biochar_v48_stage1_material.csv", mime="text/csv")
        st.info("**Stage 2와의 관계:** 이 연구용 BET·TPV 값은 qe Model A/B에 직접 입력되지 않습니다. "
                "따라서 온도에 따른 BET 변화가 qe 변화의 원인이라는 결론을 제공하지 않습니다.")
