"""Read-only viewer for newly integrated evidence. Does NOT auto-retrain models."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import streamlit as st
from pyrolysis.biochar_stage2.phase8_evidence import audit_phase8,phase8_path,read_csv

_ROOT = Path(__file__).resolve().parents[2]

@st.cache_data(show_spinner=False)
def _report():
    return audit_phase8(_ROOT)


def render_phase8_evidence():
    st.header("Phase 8 · 검증된 데이터 확장")
    st.caption("연구실 공동작업자의 출처 확인 자료를 별도의 읽기 전용 코호트로 수록 · 현재 모델은 아직 재학습하지 않음")
    try:
        x=_report()
    except Exception as e:
        st.error(f"Phase8 데이터 경로 또는 무결성 검사 실패: {e}")
        return
    a,b=x['stage1'],x['stage2']
    c1,c2,c3,c4=st.columns(4)
    c1.metric('Stage 1 시료', f"{a['integrated']}", f"+{a['added']}")
    c2.metric('Stage 1 DOI', f"{a['doi_integrated']}", f"+{a['doi_integrated']-a['doi_previous']}")
    c3.metric('Stage 2 관측', f"{b['integrated']}", f"+{b['added']}")
    c4.metric('엄격 평형 신규 승인',f"{b['strict_new']}행")
    st.info('관측 기록과 ML 예측은 다릅니다. v15의 A/B/C 및 v60·v20 저장 모델은 재학습하지 않았습니다. 새 원료에 대한 B 예측/전역 최적화도 아직 검증되지 않았습니다.')
    st.subheader('새 제조시료 8개 · BET/TPV 실측')
    s1=pd.DataFrame(x['stage1_new'])
    st.dataframe(s1.rename(columns={'material_id':'소재 ID','doi':'DOI','feedstock':'원료',
                                     'pyrolysis_temperature_C':'열분해온도 (°C)',
                                     'BET_m2_g':'BET (m²/g)','TPV_cm3_g':'TPV (cm³/g)'}),
                 hide_index=True,use_container_width=True)
    st.subheader('신규 엄격 평형 인증 1개 · 흡착량 관측')
    st.dataframe(pd.DataFrame(x['stage2_new']),hide_index=True,use_container_width=True)
    st.divider()
    st.subheader('원료 조성자료 실제 확보율')
    cohort=st.radio('대상', ['BET','TPV'],horizontal=True)
    coverage=x['composition_coverage'][cohort]
    df=pd.DataFrame([{'변수':name,'실측값 존재':v['n'],'전체 행':coverage['total'],
                      '확보율 (%)':round(v['n']/coverage['total']*100,1),
                      '독립 DOI':v['doi']} for name,v in coverage['fields'].items()])
    st.dataframe(df,hide_index=True,use_container_width=True)
    st.caption('값의 존재와 분석기준(dry/daf 등) 검증·모델 일반화 가능성은 별개입니다.')
    st.subheader('데이터 탐색')
    dataset=st.selectbox('표', ['Stage1_Master_264','Stage2_Reported_98'])
    path=(phase8_path(_ROOT) / ('data/stage1/Stage1_Master_264_UNCHANGED.csv' if dataset.startswith('Stage1') else 'data/stage2/Stage2_Reported_Uptake_98_UNCHANGED.csv'))
    full=pd.DataFrame(read_csv(path))
    q=st.text_input('원료명 또는 DOI 검색 (원자료 행 조회용)', '')
    if q.strip():
        cols=[c for c in ['feedstock','doi','material_key'] if c in full]
        mask=pd.Series(False,index=full.index)
        for col in cols:
            mask |= full[col].fillna('').astype(str).str.contains(q.strip(),case=False,regex=False)
        full=full.loc[mask]
    st.caption(f"{len(full)}행 · 원문값 그대로 (미보고/NR 유지)")
    keep=[c for c in ['material_id','adsorption_id','doi','feedstock','pollutant','pyrolysis_temperature_C',
                      'BET_m2_g_training_candidate','TPV_cm3_g_training_candidate','qe_reported_mg_g','training_policy','source_target_status'] if c in full]
    st.dataframe(full[keep],hide_index=True,use_container_width=True)
    st.download_button('조회 결과 CSV 저장',full.to_csv(index=False).encode('utf-8-sig'),'phase8_search.csv','text/csv')
    st.caption('Phase8에는 Stage 1–2 명시적 동일 시료 연결이 3개 관측(2개 소재)뿐이며, 162/264 recipe만 기존 v60 adapter에서 조건부 실행 가능합니다.')
