"""Read-only Phase 9 literature-source audit. No model training or activation."""
from pathlib import Path
import pandas as pd
import streamlit as st

_DATA = Path(__file__).resolve().parents[2] / 'data' / 'imports' / 'v24_source_audit'

@st.cache_data(show_spinner=False)
def _read(name):
    return pd.read_csv(_DATA / name, low_memory=False)

def render_v24_evidence():
    st.header('추가 논문 데이터 · 원문 검증 (v24)')
    st.caption('새 4편 + 이전 원문 2편(포멜로·하수슬러지) · 출처·단위·물질수지 보존 · 현재 학습 모델 미변경')
    s1 = _read('stage1_new33_BET_TPV_SOURCE_STAGING.csv')
    s2 = _read('stage2_new72_source_reported_QE_STAGING.csv')
    bad = s2[s2['physical_cap_pass'] == False]
    a,b,c,d=st.columns(4)
    a.metric('신규 Stage 1 물성 시료',f'{len(s1)}')
    b.metric('새 흡착량 기록',f'{len(s2)}')
    c.metric('물질수지 통과',f'{len(s2)-len(bad)}')
    d.metric('수지 위반 격리',f'{len(bad)}')
    st.warning('이 데이터는 연구용 스테이징입니다. 실제 배포 모델이나 예측 결과는 바뀌지 않았습니다. 물질수지 통과는 학습 승인 또는 엄격 평형 인증을 의미하지 않습니다.')
    selected = st.radio('데이터 구분', ['Stage 1 · BET/TPV', 'Stage 2 · reported uptake', '격리 자료', 'qmax · 별도 타깃'], horizontal=True)
    if selected == 'Stage 1 · BET/TPV':
        df=s1
        st.caption('HTC와 화학 활성화는 일반 단일 열분해와 제조공정이 다릅니다. 원료 조성은 비어 있는 값이 많습니다.')
    elif selected == 'Stage 2 · reported uptake':
        df=s2
        st.caption('포멜로: 사용자 확인 단위 오기 mmol/g → μmol/g 반영. 가금분: 3행 물질수지 위반. 모두 평형 인증 전.')
    elif selected == '격리 자료':
        df=_read('stage2_new3_PHYSICAL_FAIL_QUARANTINE.csv')
    else:
        df=_read('all_qmax_32_FITTED_NOT_QE.csv')
        st.caption('Langmuir qmax는 피팅된 용량이며 실측 qe와 합치지 않습니다.')
    doi=st.selectbox('논문 DOI', ['전체'] + sorted(df.doi.unique().tolist()))
    if doi!='전체': df=df[df.doi==doi]
    st.dataframe(df,hide_index=True,use_container_width=True)
    st.download_button('현재 표 CSV 내려받기',df.to_csv(index=False).encode('utf-8-sig'), 'biochar_v24_evidence.csv','text/csv')
    st.info('정량 예측을 개선하려면 신규 출처를 DOI 그룹 외부 검증으로 평가하고, 분석기준과 제조경로가 적합한 행만 따로 승인해야 합니다.')
