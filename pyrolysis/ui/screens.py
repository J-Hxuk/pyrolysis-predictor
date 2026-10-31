from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from pyrolysis.artifacts import BundleError, load_bundle
from pyrolysis.inference import make_single_candidate_table, predict_all
from pyrolysis.ui.inputs import render_input_widgets
from pyrolysis.ui.results import display_prediction_result

__all__ = ["show_single_prediction_screen"]


def show_single_prediction_screen(bundle_dir: Path) -> None:
    """단일 후보 예측 화면을 표시한다 (Task 5).
    
    - 묶음을 로드하여 입력 규격 추출
    - 규격 기반 입력 위젯 표시
    - 예측 실행 버튼
    - 결과 또는 오류 표시
    - demo 모드에서는 "기능 점검용" 표시
    """
    st.title("🔬 열분해 생성물·촉매 성능 예측")
    
    # 묶음 로드
    try:
        bundle = load_bundle(bundle_dir)
    except BundleError as e:
        st.error(f"**배포 묶음 로드 실패**\n\n{e}")
        return
    
    contract = bundle["contract"]
    bundle_type = contract.get("bundle_type", "research")
    
    # demo 모드 표시
    if bundle_type == "demo-synthetic":
        st.info("**기능 점검용 모드**\n\n이 화면은 합성 자료와 demo 모델을 사용합니다.", icon="⚙️")
    
    # 입력 규격 읽기 (Task 2에서 InputSpec 정의 예정, 지금은 contract 직접 사용)
    input_ids = contract["input_ids"]
    input_dtypes = contract["input_dtypes"]
    target_id = contract["target_id"]
    target_unit = contract.get("target_unit", "")
    
    # InputSpec 객체 임시 생성 (Task 2에서 실제 클래스 구현 예정)
    class TempInputSpec:
        def __init__(self, id: str, dtype: str):
            self.id = id
            self.dtype = dtype
            self.label_ko = id.replace("_", " ").title()
            self.unit = ""
            self.min_value = None
            self.max_value = None
            self.default_value = None
            self.categories = None
            
            # dtype별 기본값 설정
            if dtype == "int":
                self.min_value = 0
                self.max_value = 100
                self.default_value = 50
            elif dtype == "float":
                self.min_value = 0.0
                self.max_value = 100.0
                self.default_value = 50.0
            elif dtype == "category":
                self.categories = ["A", "B", "C"]
                self.default_value = "A"
    
    input_specs = [TempInputSpec(id, input_dtypes[id]) for id in input_ids]
    
    # 입력 위젯 표시
    st.subheader("입력 조건")
    
    # 세션 상태 초기화
    if "last_inputs" not in st.session_state:
        st.session_state.last_inputs = None
    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None
    
    # 입력 위젯 렌더링
    current_inputs = render_input_widgets(input_specs, key_prefix="single")
    
    # 입력 변경 감지 - 이전 결과 제거
    if st.session_state.last_inputs != current_inputs:
        st.session_state.prediction_result = None
    
    # 예측 실행 버튼
    if st.button("🔍 예측 실행", type="primary", use_container_width=True):
        with st.spinner("예측 계산 중..."):
            # CandidateTable 생성
            candidates = make_single_candidate_table(
                input_dict=current_inputs,
                input_schema_id=contract["bundle_id"],
            )
            
            # 예측 실행
            batch = predict_all(bundle, candidates)
            
            # 결과 저장
            st.session_state.prediction_result = batch
            st.session_state.last_inputs = current_inputs.copy()
    
    # 결과 표시
    if st.session_state.prediction_result is not None:
        st.markdown("---")
        st.subheader("예측 결과")
        display_prediction_result(
            st.session_state.prediction_result,
            target_label_ko=target_id.replace("_", " ").title(),
        )
