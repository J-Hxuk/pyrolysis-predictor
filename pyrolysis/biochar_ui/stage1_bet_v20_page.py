"""Research-only Streamlit pane: raw feedstock chemistry -> experimental BET.

This pane never modifies Stage 1 v60 or Stage 2 A/B/C/D, and it does not
claim validated new-feedstock generalization or adsorption optimization.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from pyrolysis.biochar_stage1.v44_workflow import (
    bet_temperature_sweep, build_bet_record, handoff_to_stage2, bet_result_csv,
)
from pyrolysis.biochar_stage1.composition_inference_v20 import (
    default_model_path, load_portable_model, assess_inputs, estimate_bet,
)

_ROOT = Path(__file__).resolve().parents[2]


@st.cache_resource(show_spinner=False)
def _get_bet_model(path: str):
    return load_portable_model(Path(path))


def _send_to_stage2(current):
    """Streamlit callback executes BEFORE radio/widget creation on rerun."""
    st.session_state.update(handoff_to_stage2(current))
    st.session_state["v44_handoff_summary"] = current
    st.session_state["v48_navigation"] = "흡착성능 예측"


def render_stage1_bet_v20():
    st.markdown("## BET 물성 예측")
    st.caption("① 원료·공정 선택 → ② 원료 조성 입력 → ③ 실시간 BET/온도 곡선 → ④ 흡착조건으로 제조정보 전달")
    st.info("Stage 1 v20: **비활성화 단일 열분해**의 원료 조성·제조조건 기반 BET 연구모델입니다. "
            "TPV는 이 모델에서 예측하지 않으며, 예측 BET는 Stage 2 qe 입력으로 자동 연결되지 않습니다.")
    model_path = default_model_path(_ROOT)
    if not model_path.is_file():
        st.error("실험용 BET 모델 JSON이 없습니다. 통합 패치의 deployment/experimental_stage1_bet 파일을 확인하세요.")
        return
    try:
        model = _get_bet_model(str(model_path))
    except (OSError, ValueError, KeyError) as err:
        st.error(f"모델 읽기 실패: {err}")
        return

    st.markdown("#### 01 · 원료와 공정 적용범위")
    feedstock = st.text_input("바이오매스 원료명 (자유 입력)", key="v20_feedstock",
                              placeholder="rice husk / 왕겨 / 신규 원료명")
    st.caption("원료명은 기록 및 참고용입니다. 실제 모델 입력은 제조조건과 측정 조성입니다. 미학습 이름이라고 다른 원료로 대체하지 않습니다.")
    col1, col2 = st.columns(2)
    with col1:
        route = st.selectbox("제조경로", ["SINGLE_STAGE_PYROLYSIS", "OTHER"],
                             format_func=lambda s: "비활성화 단일 열분해" if s=="SINGLE_STAGE_PYROLYSIS" else "활성화·다단계·기타")
    with col2:
        activated = st.checkbox("활성화제/후처리/화학적 개질 실시", value=False, key="v20_activated")

    st.markdown("#### 02 · 열분해조건")
    a,b,c = st.columns(3)
    with a: temp = st.number_input("열분해온도 (°C)", min_value=0.0, value=500.0, step=10.0, key="v20_pt")
    with b: hold = st.number_input("유지시간 (min)", min_value=0.0, value=60.0, step=5.0, key="v20_hold")
    with c: heating = st.number_input("승온속도 (°C/min)", min_value=0.0, value=10.0, step=1.0, key="v20_heat")

    st.markdown("#### 03 · 원료 공업분석 (필수 · Dry basis)")
    st.caption("열분해 **전 원료**의 측정값을 입력하세요. VM + Ash + FC ≈ 100 wt%여야 합니다. 임의의 원료 기본값은 채우지 않습니다.")
    proximate_basis = "dry"  # fixed input contract; no as-received values accepted
    st.caption("입력 기준 고정: dry basis (수분 제외). 다른 기준의 값은 직접 입력하지 마세요.")
    prox_vals = {}
    cols = st.columns(3)
    for col, (f, desc) in zip(cols, [
        ("raw_VM_wt_pct","휘발분 VM (wt%)"),
        ("raw_ash_wt_pct","회분 Ash (wt%)"),
        ("raw_FC_wt_pct","고정탄소 FC (wt%)")]):
        with col:
            prox_vals[f] = st.text_input(desc, key="v20_"+f, placeholder="미측정")

    st.markdown("#### 04 · 원소분석 (선택 · Dry ash-free basis)")
    st.caption("4개 값이 모두 있고 분석기준이 daf일 때만 전체 조성 모델로 전환합니다. 일부만 입력하면 임의 보충하지 않습니다.")
    elemental_basis = "dry ash-free (daf)"  # fixed contract, applied only when C/H/N/O all supplied
    st.caption("입력 기준 고정: dry ash-free (daf). 다른 기준의 값은 변환·확인 후 입력하세요.")
    elem_vals = {}
    cols2 = st.columns(4)
    for col, (f, desc) in zip(cols2, [
        ("raw_C_wt_pct","탄소 C (wt%)"), ("raw_H_wt_pct","수소 H (wt%)"),
        ("raw_N_wt_pct","질소 N (wt%)"), ("raw_O_wt_pct","산소 O (wt%)")]):
        with col:
            elem_vals[f] = st.text_input(desc, key="v20_"+f, placeholder="미측정")

    process = {"pyrolysis_temperature_C": temp, "pyrolysis_hold_min": hold,
               "heating_rate_C_min": heating}
    composition = {**prox_vals, **elem_vals}
    check = assess_inputs(model, process, composition,
                          elemental_basis=elemental_basis, proximate_basis=proximate_basis,
                          process_type=route, activated=activated)
    if check["errors"]:
        for message in check["errors"]:
            st.warning(message)
    if check["outside_training_range"]:
        st.warning("학습 데이터의 개별 변수 범위 밖입니다: " + " / ".join(check["outside_training_range"]))
        if check["strongly_outside_feature_names"]:
            st.error("다음 변수는 학습 범위에서 상당히 벗어났습니다: " +
                     ", ".join(check["strongly_outside_feature_names"]) +
                     ". 이 결과는 오차 범위를 정량화할 수 없는 **고불확실성 추정**입니다.")
    st.caption("변수별 범위 내부여도 전체 변수 조합이나 신규 원료에서 정확하다는 보장은 없습니다.")

    exploratory = False
    if check["can_explore_composition_ood"]:
        st.info("제조조건은 학습 범위 내부이나 원료 조성이 범위 밖입니다. "
                "RandomForest는 범위 밖에서 추세를 외삽하지 않으며 가장자리 예측값이 반복될 수 있습니다. "
                "아래 선택 시 검증되지 않은 참고 수치만 표시합니다.")
        exploratory = st.checkbox("한계를 이해했으며, 학습범위 밖 조성에 대한 참고 BET 추정치를 보겠습니다.",
                                  value=False, key="v21_allow_ood")
    elif check["outside_training_range"] and not check["errors"]:
        st.warning("열분해 공정변수가 학습 범위 밖이면 참고 수치도 제공하지 않습니다.")
    can_run = (check["predictable"] or (exploratory and check["can_explore_composition_ood"])) and bool(feedstock.strip())
    # The default live mode makes the Stage 1 experience consistent with Stage 2:
    # changing any valid input updates the model response instead of hiding it.
    live = st.checkbox("입력 변경 시 BET·그래프 자동 갱신", value=True, key="v44_bet_live")
    manual = st.button("BET 예측 실행 (연구용)", type="primary", disabled=not can_run, key="v20_run")
    if can_run and (live or manual):
        is_ood = bool(check["outside_training_range"])
        y = estimate_bet(model, check, allow_composition_ood=exploratory)
        if is_ood:
            st.error("**학습범위 외 연구용 참고 추정** — 이 수치의 정확도·신뢰구간은 검증되지 않았습니다. "
                     "실측이나 신뢰 가능한 정량 예측으로 사용하지 마세요.")
        variant = check["variant"]
        validation = model["validation_results"][variant]
        st.metric("참고 BET 추정치 (범위 밖)" if is_ood else "예상 BET (비표면적)", f"{y:,.2f} m²/g")
        st.caption(f"사용한 모델: {variant} · 실제 원료명 미사용 · LODO 탐색적 R² {validation['r2_oof']:.3f}, MAE {validation['mae_m2_g']:.1f} m²/g")
        if is_ood:
            st.caption("주의: 위 LODO 성능은 이 범위 밖 입력의 오차를 나타내지 않습니다.")
        st.info("이 점수는 동일 데이터의 DOI 교차검증 값이며, 본 입력값의 예측 오차나 신뢰구간이 아닙니다. "
                "현재 Stage 2 qe 모델에는 전달되지 않습니다. TPV·최적 열분해조건 추천도 수행하지 않습니다.")
        record = {"feedstock_note_only": feedstock.strip(), "prediction_type": "EXPERIMENTAL_BET",
                  "model_variant": variant, "predicted_BET_m2_g": y,
                  "prediction_scope": "UNVERIFIED_COMPOSITION_OOD" if is_ood else "IN_MARGINAL_RANGE_EXPLORATORY",
                  "proximate_basis": "dry", "elemental_basis": "daf" if variant=="full" else "not_supplied",
                  **process, **check["values"]}
        st.markdown("#### 05 · 열분해온도별 BET 예측 곡선")
        st.caption("원료 조성·유지시간·승온속도를 고정하고 **열분해온도만** "
                   "모델 학습범위 안에서 변경한 예측 곡선입니다. 실측점이나 최적온도 보장은 아닙니다.")
        try:
            current = build_bet_record(feedstock, process, check, y,
                                       allow_composition_ood=exploratory)
            curve = bet_temperature_sweep(model, check, allow_composition_ood=exploratory)
            st.line_chart(curve.set_index("열분해온도 (°C)")[["예측 BET (m²/g)"]],
                          use_container_width=True)
            st.caption(f"현재 설정: {temp:g}°C → BET {y:,.2f} m²/g · "
                       "그래프는 고정 조성에 대한 모델 응답이며 실험값이 아닙니다.")
            st.download_button("BET 입력·온도별 예측 CSV", bet_result_csv(current, curve),
                               file_name="biochar_v44_bet_temperature_curve.csv",
                               mime="text/csv", key="v44_bet_csv")
            # Save only values verified by the current model's input checker.
            st.session_state["v44_last_bet_result"] = current
            st.button("제조조건 전달 → 흡착성능 화면 열기", key="v44_send_process",
                      on_click=_send_to_stage2, args=(current,))
            st.caption("선택한 원료명·제조온도·유지시간·승온속도만 전달합니다. "
                       "BET 추정치는 qe 입력에 포함되지 않습니다.")
        except (ValueError, TypeError, KeyError) as exc:
            st.warning(f"온도별 BET 곡선을 계산하지 못했습니다: {exc}")

    with st.expander("모델 학습·검증 범위 및 연구상 제한"):
        st.markdown("- 자료: eval37/base.csv, BET 실측 182행·24개 DOI (별도 2차 편찬 평가자료).\n"
                    "- RandomForest 두 모델(공업분석 3개, 공업+원소분석 7개)을 각각 전량 학습한 연구용 추론기입니다.\n"
                    "- 기존 v18 LODO 결과로 후보 모델을 선택한 뒤 182행 전체를 학습했으므로, **잠금 외부 Test의 확증적 성능은 없습니다.**\n"
                    "- 이 자료는 비활성화 단일 열분해 원료만 포함하며 활성화·혼합 원료·개질 조건으로 확장하면 안 됩니다.\n"
                    "- 학습에는 원료명과 TPV가 포함되지 않습니다. 입력 이름이 처음 등장한 원료인지와 무관하게 조성·공정값만 활용합니다.\n"
                    "- 공업분석 dry, 원소분석 daf만 허용합니다. as-received는 입력받지 않으며 서로 다른 분석기준을 임의 변환하지 않습니다.\n"
                    "- 학습 범위를 벗어난 **조성**은 명시적 동의 후에만 고불확실성 참고 추정치로 표시합니다.\n"
                    "- 신규 원료 외삽과 BET→qe 오차전파는 검증되지 않았습니다.")
        limit_rows = [{"변수":k, "학습 최소":v[0], "학습 최대":v[1]}
                      for k,v in model["models"]["full"]["feature_range"].items()]
        st.dataframe(pd.DataFrame(limit_rows), hide_index=True, use_container_width=True)
