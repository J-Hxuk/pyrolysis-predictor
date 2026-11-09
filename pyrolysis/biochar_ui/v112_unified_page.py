"""v113: Restore laboratory-style sliders and evidence-backed feedstock completion.

ONE predictive pipeline and ONE visible page, no extra models or side panels.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
from pyrolysis.biochar_stage2.input_support import match_feedstock,normalize_name
from pyrolysis.biochar_stage2.v112_single_engine import load_chain,predict_one,assess_stage1_input
from pyrolysis.biochar_ui.poster_export_v133 import (
    read_stage2_temperature_bounds, stage1_temperature_bounds, attach_coverage,
    format_bounds, export_csv_with_inputs, render_poster_figure,
    IN_RANGE, OUT_OF_RANGE, NOT_VERIFIED,
)

ROOT=Path(__file__).resolve().parents[2]
FEED_KEY='v113_feedstock'

@st.cache_resource(show_spinner='통합 모델을 준비하고 있습니다...')
def _models(root):
    return load_chain(Path(root))

@st.cache_data(show_spinner=False)
def _feedstock_catalog(root):
    """Names from actual Stage1 417 observations and Stage2 Core97, never synthetic names."""
    path=Path(root)/'data/imports/v46_phase8/Stage1_417_EXPLORATORY_UNAPPROVED.csv'
    s1=pd.read_csv(path,usecols=['feedstock'])
    stage1={normalize_name(v) for v in s1.feedstock.dropna().astype(str) if normalize_name(v)}
    core=set()
    try:
        from pyrolysis.biochar_stage2 import v15_loader as L, v15_features as fc
        src=L.load_core_qe(Path(root)).df
        core={normalize_name(v) for v in src[fc.GROUP_FEEDSTOCK].dropna().astype(str) if normalize_name(v)}
    except (FileNotFoundError,KeyError,ValueError):
        pass  # real Stage1 inventory is still available; no invented Core97 entries
    names=sorted(stage1|core)
    return names,stage1,core



@st.cache_data(show_spinner=False)
def _stage2_temperature_bounds(root: str, pollutant: str):
    return read_stage2_temperature_bounds(Path(root), pollutant)

@st.cache_data(show_spinner=False)
def _poster_image(df: pd.DataFrame, image_format: str) -> bytes:
    return render_poster_figure(df, image_format=image_format, dpi=300)

def _select_feedstock(name):
    st.session_state[FEED_KEY]=name
    st.session_state.pop('v113_last_prediction',None)


def _feedstock_input(root):
    names,s1,core=_feedstock_catalog(str(root))
    query=st.text_input('바이오매스 원료명',value='rice husk',key=FEED_KEY,
             help='등록된 원료명을 검색합니다. 다른 원료로 자동 치환하지 않으며 미등록 원료도 입력할 수 있습니다.')
    m=match_feedstock(query,names,limit=8)
    if m.status in ('exact','alias'):
        label=m.matched
        source=[]
        if normalize_name(label) in s1:source.append('Stage 1 물성 학습')
        if normalize_name(label) in core:source.append('Stage 2 핵심 학습')
        st.caption('등록 원료 일치: **'+label+'** · '+(' / '.join(source) if source else '원료 목록'))
        if m.status=='alias':
            st.caption('동의어로 연결된 원료명입니다. 다른 부산물의 조성으로 자동 변환하지 않습니다.')
    elif m.status=='empty':
        st.warning('원료명을 입력해 주세요.')
    else:
        st.caption('등록되지 않은 원료명도 탐색적 예측은 가능하지만 원료별 정확도는 확인되지 않았습니다.')
        if m.suggestions:
            st.caption('등록 원료 추천 · 원하는 시료를 직접 선택하세요')
            for r in range(0,min(6,len(m.suggestions)),3):
                cs=st.columns(3)
                for i,suggest in enumerate(m.suggestions[r:r+3]):
                    cs[i].button(suggest,key=f'v113_pick_{r+i}',
                                 on_click=_select_feedstock,args=(suggest,),use_container_width=True)
    return query


def _num_optional(value):
    return '' if value is None else value



def _render_temperature_curve(df: pd.DataFrame) -> None:
    """Static Altair line: hover works but wheel zoom/pan is disabled.

    Do not call `.interactive()` or add a `bind='scales'` interval selection.
    Without interactive scale bindings, wheel scroll continues to scroll the page.
    """
    import altair as alt

    base = alt.Chart(df).encode(
        x=alt.X('제조온도 (℃):Q', title='제조온도 (℃)', scale=alt.Scale(zero=False)),
        y=alt.Y('예측 흡착량 (mg/g):Q', title='예측 흡착량 (mg/g)',
                scale=alt.Scale(zero=False)),
        tooltip=[
            alt.Tooltip('제조온도 (℃):Q', title='제조온도 (℃)', format='.0f'),
            alt.Tooltip('예측 흡착량 (mg/g):Q', title='예측 흡착량 (mg/g)', format='.2f'),
            alt.Tooltip('온도 범위 판정:N', title='관측범위'),
            alt.Tooltip('BET 온도범위:N'),
            alt.Tooltip('TPV 온도범위:N'),
            alt.Tooltip('Stage2 오염물질 온도범위:N'),
        ],
    )
    line = base.mark_line(color='#315a72', strokeWidth=2)
    points = base.mark_circle(size=75).encode(
        color=alt.Color('온도 범위 판정:N',
            scale=alt.Scale(domain=[IN_RANGE, OUT_OF_RANGE, NOT_VERIFIED],
                            range=['#157f72', '#bd4d3a', '#777777']),
            title='관측 온도범위'),
    )
    # No .interactive() / bind='scales': wheel scrolling never zooms the chart.
    st.altair_chart((line + points).properties(height=320), use_container_width=True)

def render_v112_unified():
    """Keep existing v112 import/navigation contract while improving the one page."""
    st.markdown('''<style>
      .block-container {padding-top:1.4rem; padding-bottom:2.2rem; max-width:1220px;}
      [data-testid="stMetric"] {padding:0.9rem 1rem; border:1px solid rgba(128,128,128,.23); border-radius:7px;}
      [data-testid="stMetricLabel"] {letter-spacing:.01em;}
      div[data-testid="stForm"] {border:1px solid rgba(128,128,128,.23); border-radius:8px; padding:1rem;}
    </style>''',unsafe_allow_html=True)
    st.title('바이오차 제조·흡착성능 예측')
    st.caption('원료 조성 · 열분해조건 → BET / TPV → 문헌 기반 흡착량 → 제조온도 비교')
    try:
        models=_models(str(ROOT))
    except Exception as exc:
        st.error(f'모델을 읽을 수 없습니다: {type(exc).__name__}: {exc}')
        st.info('v112 단일 모델 설치 및 로컬 학습 상태를 확인해 주세요.')
        return
    st.markdown('### 01 · 원료 선택')
    feedstock=_feedstock_input(ROOT)
    with st.form('v113_prediction_form',clear_on_submit=False):
        st.markdown('### 02 · 열분해 공정')
        a,b,c=st.columns(3)
        with a:temp=st.slider('제조온도 (℃)',min_value=200,max_value=1200,value=500,step=25,
                              help='일부 구간은 학습 범위를 벗어날 수 있습니다.')
        with b:hold=st.slider('유지시간 (min)',min_value=10,max_value=600,value=120,step=10)
        with c:heat=st.slider('승온속도 (℃/min)',min_value=1,max_value=50,value=10,step=1)
        d,e=st.columns([1.0,1.0])
        with d:
            atm=st.selectbox('열분해 분위기',['N2','inert_N2','oxygen_limited','Ar','air','NR'])
        with e:
            activation=st.selectbox('후처리/활성화',[
                '없음','화학적 활성화/함침','물리적 활성화','수증기 활성화'],index=0)
        activation_modes={'없음':'none','화학적 활성화/함침':'chemical_impregnation',
                '물리적 활성화':'physical_activation','수증기 활성화':'physical_steam'}
        activation_method=activation_modes[activation]
        st.caption('활성화 방식은 Stage 1 물성 모델에만 반영됩니다. 활성화제 혼합비 등은 현재 학습 입력에 없습니다.')
        stage1_preview=assess_stage1_input(models,feedstock=feedstock,process={
            'pyrolysis_temperature_C':temp,'pyrolysis_hold_min':hold,
            'heating_rate_C_min':heat,'pyrolysis_atmosphere':atm,
            'activation_method':activation_method})
        if stage1_preview['used_feedstock'] != str(feedstock).strip():
            st.caption('실제 Stage 1 원료 피처: '+stage1_preview['used_feedstock']+
                       ' (동의어/표기 정규화, 원료 조성값 자동대입 아님)')
        if stage1_preview['warnings']:
            st.warning('Stage 1 입력 학습범위 주의: '+'; '.join(stage1_preview['warnings']))

        with st.expander('원료 조성 · 공업분석 및 원소분석 (선택)',expanded=False):
            st.caption('열분해 전 원료의 측정값을 입력합니다. 공업분석 dry basis, 원소분석 daf basis. 미측정 항목은 공란으로 둡니다.')
            composition={}
            for keys,title in [(['raw_VM_wt_pct','raw_ash_wt_pct','raw_FC_wt_pct'],'공업분석 (wt%)'),
                                (['raw_C_wt_pct','raw_H_wt_pct','raw_N_wt_pct','raw_O_wt_pct'],'원소분석 (wt%)')]:
                st.markdown('**'+title+'**')
                cols=st.columns(len(keys))
                for col,k in zip(cols,keys):
                    label=k.removeprefix('raw_').replace('_wt_pct','').upper().replace('ASH','Ash')
                    with col:composition[k]=st.text_input(label,placeholder='미측정',key=f'v113_{k}')
            st.caption('조성 입력의 영향은 Stage 1 BET 경로를 통해 전달됩니다. TPV는 현재 제조조건 기반 모델입니다.')

        st.markdown('### 03 · 흡착 조건')
        labels=models['stage2']['supported_labels']
        c1,c2,c3=st.columns(3)
        with c1:sel=st.selectbox('오염물질',labels+['직접 입력 (미학습)'],index=labels.index('Pb(II)') if 'Pb(II)' in labels else 0)
        with c2:c0=st.number_input('초기농도 C₀ (mg/L)',min_value=0.001,max_value=1e6,value=100.0,step=10.0)
        with c3:dose=st.number_input('흡착제 투입량 (g/L)',min_value=0.001,max_value=1e5,value=1.0,step=0.1,format='%.3f')
        pollutant=st.text_input('미학습 오염물질명',key='v113_custom_pollutant') if sel=='직접 입력 (미학습)' else sel
        with st.expander('추가 흡착조건 · pH / 온도 / 접촉시간',expanded=True):
            x,y,z=st.columns(3)
            with x:
                missing_ph=st.checkbox('초기 pH 미보고',value=False)
                ph=st.slider('초기 pH',min_value=0.0,max_value=14.0,value=5.5,step=0.1,
                             disabled=missing_ph)
            with y:
                missing_ads_temp=st.checkbox('흡착온도 미보고',value=False)
                ads_temp=st.slider('흡착온도 (℃)',min_value=0,max_value=100,value=25,step=1,
                                   disabled=missing_ads_temp)
            with z:
                missing_contact=st.checkbox('접촉시간 미보고',value=False)
                contact=st.slider('접촉시간 (min)',min_value=1,max_value=2880,value=1440,step=1,
                                  disabled=missing_contact)
        st.markdown('### 04 · 제조온도 비교 설정')
        x,y=st.columns([2,1])
        with x: sweep=st.slider('비교 온도 범위 (℃)',min_value=200,max_value=1200,value=(300,700),step=25)
        with y: step=st.selectbox('비교 간격 (℃)',[50,100,25],index=0)
        submitted=st.form_submit_button('흡착성능 예측 및 조건 비교',type='primary',use_container_width=True)

    if submitted:
        try:
            if not str(feedstock).strip():raise ValueError('원료명을 입력해 주세요.')
            process={'pyrolysis_temperature_C':temp,'pyrolysis_hold_min':hold,
                     'heating_rate_C_min':heat,'pyrolysis_atmosphere':atm,
                     'activation_method':activation_method}
            ads={'pollutant':pollutant,'C0_mg_L':c0,'dose_g_L':dose,
                 'initial_pH':_num_optional(None if missing_ph else ph),
                 'adsorption_T_C':_num_optional(None if missing_ads_temp else ads_temp),
                 'contact_time_min':_num_optional(None if missing_contact else contact)}
            active=activation_method!='none'
            result=predict_one(models,ROOT,feedstock=feedstock,process=process,
                               composition=composition,adsorption=ads,activated=active)
            ts=sorted(set([sweep[0],sweep[1]]+list(range(sweep[0],sweep[1]+1,step))))
            if len(ts)>45:raise ValueError('온도 비교 구간은 45개 이하로 설정해 주세요.')
            rows=[]
            for t in ts:
                pr=predict_one(models,ROOT,feedstock=feedstock,
                       process={**process,'pyrolysis_temperature_C':t},
                       composition=composition,adsorption=ads,activated=active)
                rows.append({'제조온도 (℃)':int(t),'예측 흡착량 (mg/g)':pr['reported_capacity_mg_g'],
                             'BET (m²/g)':pr['BET_m2_g'],'TPV (cm³/g)':pr['TPV_cm3_g']})
            df=pd.DataFrame(rows)
            stage2_bounds=_stage2_temperature_bounds(str(ROOT), pollutant)
            df=attach_coverage(df, models, stage2_bounds)
            st.session_state['v113_last_prediction']={
                'result':result,'table':df,'feedstock':feedstock,
                'pollutant':pollutant,'temperature':temp,
                'process':process,'adsorption':ads,'composition':composition,
                'stage2_temperature_bounds':stage2_bounds,
            }
        except Exception as exc:
            st.session_state.pop('v113_last_prediction',None)
            st.error(f'예측을 완료하지 못했습니다: {type(exc).__name__}: {exc}')

    payload=st.session_state.get('v113_last_prediction')
    if payload is None:
        st.caption('조건을 설정한 뒤 위의 예측 버튼을 누르면 결과와 온도별 비교 그래프가 표시됩니다.')
        return
    r=payload['result'];df=payload['table']
    st.markdown('### 예측 결과')
    p,q,t=st.columns(3)
    p.metric('BET (예측)',f"{r['BET_m2_g']:,.2f} m²/g")
    q.metric('TPV (예측)',f"{r['TPV_cm3_g']:,.4f} cm³/g")
    t.metric('Reported Capacity (예측)',f"{r['reported_capacity_mg_g']:,.2f} mg/g")
    st.caption(f"물질수지 상한 {r['mass_balance_upper_mg_g']:,.2f} mg/g · BET 경로 {r['bet_path']} · TPV 경로 {r['tpv_path']}")
    st.caption('Stage 1 실제 원료 피처: '+str(r.get('stage1_feedstock_used',payload['feedstock'])))
    if r.get('stage1_input_warnings'):
        st.warning('예측 당시 입력 학습범위 주의: '+'; '.join(r['stage1_input_warnings']))
    if not r['known_pollutant']:
        st.warning('미학습 오염물질에 대한 외삽 예측입니다.')
    if not r['stage1_property_feature_used']:
        st.info('이 오염물질 계열에서는 Stage 2가 BET·TPV를 사용하지 않습니다. 원료 조성을 바꿔도 최종 흡착량은 변하지 않을 수 있습니다.')
    if r['composition_status']=='FALLBACK':
        st.warning('조성 입력을 반영하지 못하고 제조조건 기반 BET로 계산했습니다: '+'; '.join(map(str,r['warnings'])))
    elif r['composition_status']=='APPLIED':
        st.caption('조성은 Stage 1 BET를 통해 간접 반영되었습니다. 입력 변화에 따른 출력 차이는 인과관계가 아닙니다.')
    st.markdown('### 제조온도 비교')
    _render_temperature_curve(df)
    bounds=stage1_temperature_bounds(models)
    st.caption('관측 제조온도: Stage 1 BET '+format_bounds(bounds['BET'])+
               ' · Stage 1 TPV '+format_bounds(bounds['TPV'])+
               ' · Stage 2 '+str(payload['pollutant'])+' '+
               format_bounds(payload.get('stage2_temperature_bounds')))
    outside=int((df['온도 범위 판정']==OUT_OF_RANGE).sum())
    unknown=int((df['온도 범위 판정']==NOT_VERIFIED).sum())
    if outside or unknown:
        st.warning(f'온도 비교 {len(df)}점 중 관측 온도범위 밖 {outside}점, 범위 미확인 {unknown}점입니다. '
                   '해당 점의 예측은 외삽 또는 근거 부족 가능성에 주의해야 합니다.')
    st.caption('범위 판정은 각 모델/오염물질의 관측 온도 최소·최대만 이용합니다. '
               '범위 안이라는 표시도 원료·흡착조건 조합의 검증이나 정확도 보증은 아닙니다.')
    if not r['stage1_property_feature_used']:
        st.caption('선택 오염물질은 Stage 2에서 BET·TPV를 사용하지 않습니다.')
    top=df.loc[df['예측 흡착량 (mg/g)']==df['예측 흡착량 (mg/g)'].max()]
    st.caption('계산 구간 내 예측 최댓값(최적 조건 검증 아님): '+
               ', '.join(str(t)+'℃' for t in top['제조온도 (℃)'])+' · 동률은 그대로 표시')
    st.dataframe(df,use_container_width=True,hide_index=True)
    st.markdown('### 포스터용 결과 출력')
    st.caption('PNG 300 dpi / SVG 벡터 · BET, TPV, Reported Capacity 3패널. '
               '모델 불확실도나 실측값을 임의로 표시하지 않습니다.')
    export_warnings='; '.join(w for w in r.get('stage1_input_warnings', [])
                              if not w.startswith('제조온도 '))
    model_file=ROOT/'deployment/v112_single_stage2.joblib'
    model_sha=hashlib.sha256(model_file.read_bytes()).hexdigest()
    csv_bytes=export_csv_with_inputs(
        df,feedstock_input=payload['feedstock'],
        feedstock_used=str(r.get('stage1_feedstock_used',payload['feedstock'])),
        process=payload['process'],adsorption=payload['adsorption'],
        composition=payload['composition'],model_sha256=model_sha,
        properties_connected=bool(r['stage1_property_feature_used']),
        stage1_warnings=export_warnings)
    ex1,ex2,ex3=st.columns(3)
    with ex1:
        st.download_button('포스터 PNG (300 dpi)', _poster_image(df,'png'),
                           file_name='biochar_poster_v133_3panel.png',mime='image/png',
                           use_container_width=True)
    with ex2:
        st.download_button('포스터 SVG (벡터)', _poster_image(df,'svg'),
                           file_name='biochar_poster_v133_3panel.svg',mime='image/svg+xml',
                           use_container_width=True)
    with ex3:
        st.download_button('조건·예측값 CSV',csv_bytes,
                           file_name='biochar_poster_v133_inputs_predictions.csv',
                           mime='text/csv',use_container_width=True)
    with st.expander('데이터 범위·연구 한계'):
        st.write('Stage 1: 문헌 기반 BET·TPV 추정. Stage 2: 4,795행·26종 오염물질 통합 예측.')
        st.write('원료명 추천은 Stage 1/핵심 Stage 2에 실제 기록된 명칭만 사용합니다. 원료명을 선택한다고 원소분석값이 자동 채워지지는 않습니다.')
        st.write('문헌 Reported Capacity에는 strict qe가 아닌 관측이 포함될 수 있으며, Stage 1의 예측 오차가 Stage 2에 전파됩니다.')
        st.write('모델 버전: '+str(models['stage2'].get('selected_candidate','unknown')))
