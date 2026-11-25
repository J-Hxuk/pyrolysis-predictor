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
from pyrolysis.biochar_stage2.manufacturing_rank_v140 import load_ranker,rank_sweep
from pyrolysis.biochar_ui.v141_ui_state import input_signature, safe_export_stem, comparison_delta
from pyrolysis.biochar_ui.poster_export_v133 import (
    read_stage2_temperature_bounds, attach_coverage,
    export_csv_with_inputs, render_poster_figure,
    IN_RANGE, OUT_OF_RANGE, NOT_VERIFIED,
)

ROOT=Path(__file__).resolve().parents[2]
FEED_KEY='v113_feedstock'
# Shared UI-only atmosphere options. 'air' has zero Stage 1 training rows.
# Retain the inference layer's air warning for legacy/programmatic inputs.
ATMOSPHERE_OPTIONS=('N2','inert_N2','oxygen_limited','Ar','NR')

@st.cache_resource(show_spinner='통합 모델을 준비하고 있습니다...')
def _models(root):
    return load_chain(Path(root))

@st.cache_resource(show_spinner=False)
def _ranking_model(root):
    return load_ranker(Path(root))


def _build_feedstock_catalog(root):
    """Always use Stage 1 source; legacy v15 Core97 names are optional UI metadata.

    A missing v15 ZIP must not block the deployed v139 Stage 1/2 prediction.
    No Stage 2 feedstock labels are fabricated from material_group codes.
    """
    path=Path(root)/'data/imports/v46_phase8/Stage1_417_EXPLORATORY_UNAPPROVED.csv'
    s1=pd.read_csv(path,usecols=['feedstock'])
    stage1={normalize_name(v) for v in s1.feedstock.dropna().astype(str) if normalize_name(v)}
    core=set()
    try:
        from pyrolysis.biochar_stage2 import v15_loader as L, v15_features as fc
        from pyrolysis.biochar_stage2.v15_extract import V15ExtractError
        src=L.load_core_qe(Path(root)).df
        core={normalize_name(v) for v in src[fc.GROUP_FEEDSTOCK].dropna().astype(str) if normalize_name(v)}
    except (V15ExtractError, FileNotFoundError, KeyError, ValueError):
        # v15 Core97 is a legacy convenience catalog, not a dependency of
        # the currently deployed v139 model. Keep genuine Stage 1 names.
        pass
    names=sorted(stage1|core)
    return names,stage1,core


@st.cache_data(show_spinner=False)
def _feedstock_catalog(root):
    return _build_feedstock_catalog(root)



@st.cache_data(show_spinner=False)
def _stage2_temperature_bounds(root: str, pollutant: str):
    return read_stage2_temperature_bounds(Path(root), pollutant)

@st.cache_data(show_spinner=False)
def _poster_image(df: pd.DataFrame, image_format: str) -> bytes:
    return render_poster_figure(df, image_format=image_format, dpi=300)

def _select_feedstock(name):
    st.session_state[FEED_KEY]=name
    st.session_state.pop('v140_last_prediction',None)
    st.session_state.pop('v141_comparison',None)
    st.session_state.pop('v147_pin_signature',None)
    st.session_state.pop('v147_comparison',None)


def _feedstock_input(root):
    names,s1,core=_feedstock_catalog(str(root))
    # Initialize keyed state once; do not also pass an explicit widget default.
    if FEED_KEY not in st.session_state:
        st.session_state[FEED_KEY]='rice husk'
    query=st.text_input('바이오매스 원료명',key=FEED_KEY,
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



def _render_temperature_curve(df: pd.DataFrame, df_b: pd.DataFrame | None = None,
                              *, pinned: bool = False) -> None:
    """Overlay a frozen A curve (dashed) and optional B curve (solid).

    These are absolute reported-capacity predictions (mg/g), not the separate
    Manufacturing RF's centered relative scores (percentage points).
    Scales are deliberately not scroll-zoomable.
    """
    import altair as alt

    def base_curve(frame, label):
        data=frame.copy()
        data['조건']=label
        return alt.Chart(data).encode(
            x=alt.X('제조온도 (℃):Q', title='제조온도 (℃)',
                    scale=alt.Scale(zero=False)),
            y=alt.Y('예측 흡착량 (mg/g):Q', title='예측 흡착량 (mg/g)',
                    scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip('조건:N'),
                alt.Tooltip('제조온도 (℃):Q', format='.0f'),
                alt.Tooltip('예측 흡착량 (mg/g):Q', format='.3f'),
                alt.Tooltip('BET (m²/g):Q', format='.2f'),
                alt.Tooltip('TPV (cm³/g):Q', format='.4f'),
                alt.Tooltip('온도 범위 판정:N'),
                alt.Tooltip('BET 온도범위:N'),
                alt.Tooltip('TPV 온도범위:N'),
                alt.Tooltip('Stage2 오염물질 온도범위:N'),
            ],
        )

    a=base_curve(df,'A · 고정' if pinned else 'A')
    a_line=a.mark_line(color='#64748b' if pinned else '#315a72',
                       strokeWidth=2.5, strokeDash=[6,4] if pinned else [1,0])
    if not pinned:
        # Show only coverage categories present in this temperature sweep.
        palette={IN_RANGE:'#157f72',OUT_OF_RANGE:'#bd4d3a',NOT_VERIFIED:'#777777'}
        present=set(df['온도 범위 판정'].dropna())
        domain=[label for label in palette if label in present]
        a_points=a.mark_circle(size=75).encode(
            color=alt.Color('온도 범위 판정:N',
                scale=alt.Scale(domain=domain,range=[palette[label] for label in domain]),
                title='관측 온도범위'))
    else:
        a_points=a.mark_circle(size=45,color='#64748b',opacity=0.7)
    chart=a_line+a_points
    if pinned and df_b is not None:
        b=base_curve(df_b,'B · 비교')
        chart=chart+b.mark_line(color='#d58b54',strokeWidth=3)+b.mark_circle(
            size=55,color='#d58b54')
    st.altair_chart(chart.properties(height=330),use_container_width=True)


def _set_b_defaults_from_a(payload):
    """Copy A's manufacturing inputs without altering the pinned A snapshot."""
    process=payload['process']
    st.session_state['v147_b_feedstock']=payload['feedstock']
    atmosphere=process['pyrolysis_atmosphere']
    st.session_state['v147_b_atmosphere']=(
        atmosphere if atmosphere in ATMOSPHERE_OPTIONS else ATMOSPHERE_OPTIONS[0])
    st.session_state['v147_b_hold']=int(process['pyrolysis_hold_min'])
    st.session_state['v147_b_heat']=int(process['heating_rate_C_min'])
    st.session_state['v147_b_same_composition']=False


def _run_b_temperature_sweep(models, payload, feedstock, process, composition):
    """Infer B on exactly the A temperature grid with identical adsorption inputs."""
    kwargs={
        'feedstock':feedstock,'composition':composition,
        'adsorption':payload['adsorption'],
        'activated':process['activation_method']!='none',
        'allow_composition_ood':payload['allow_composition_ood'],
    }
    baseline=predict_one(models,ROOT,
                         process={**process,'pyrolysis_temperature_C':payload['temperature']},
                         **kwargs)
    rows=[]
    for temperature in payload['table']['제조온도 (℃)'].tolist():
        prediction=predict_one(models,ROOT,
                               process={**process,'pyrolysis_temperature_C':int(temperature)},
                               **kwargs)
        rows.append({
            '제조온도 (℃)':int(temperature),
            '예측 흡착량 (mg/g)':prediction['reported_capacity_mg_g'],
            'BET (m²/g)':prediction['BET_m2_g'],
            'TPV (cm³/g)':prediction['TPV_cm3_g'],
        })
    b_table=attach_coverage(pd.DataFrame(rows),models,payload['stage2_temperature_bounds'])
    return baseline,b_table


def _render_pinned_b_inputs(models, payload):
    """Return only the B comparison matching the present B inputs and frozen A."""
    st.markdown('#### B 조건 입력')
    if st.button('A 조건 복사',key='v147_copy_a',use_container_width=True):
        _set_b_defaults_from_a(payload)
        st.session_state.pop('v147_comparison',None)
    b_feedstock=st.text_input('B 원료명',key='v147_b_feedstock')
    atmospheres=ATMOSPHERE_OPTIONS
    b_atmosphere=st.selectbox('B 열분해 분위기',atmospheres,key='v147_b_atmosphere')
    b_hold=st.number_input('B 유지시간 (min)',min_value=10,max_value=600,step=10,
                           key='v147_b_hold')
    b_heat=st.number_input('B 승온속도 (℃/min)',min_value=1,max_value=50,step=1,
                           key='v147_b_heat')
    same_composition=st.checkbox('A의 원료 조성값 사용 (통제 비교)',
                                 key='v147_b_same_composition')
    st.caption('제조온도 범위·흡착조건·활성화 방식은 A와 동일합니다.')
    if same_composition and any(str(v).strip() for v in payload['composition'].values()):
        st.warning('다른 원료에 A의 실측 조성을 재사용하면 그 원료의 실제 조성 차이를 반영하지 않습니다.')
    b_process={**payload['process'],
               'pyrolysis_atmosphere':b_atmosphere,
               'pyrolysis_hold_min':int(b_hold),
               'heating_rate_C_min':int(b_heat)}
    b_composition=payload['composition'] if same_composition else {}
    b_signature=input_signature({
        'a_signature':payload['input_signature'],
        'b_feedstock':b_feedstock,'b_process':b_process,
        'b_composition':b_composition,'adsorption':payload['adsorption'],
        'temperatures':payload['table']['제조온도 (℃)'].tolist(),
        'allow_composition_ood':payload['allow_composition_ood'],
    })
    if st.button('B 곡선 계산',type='primary',key='v147_run_b',
                 use_container_width=True):
        try:
            if not str(b_feedstock).strip():
                raise ValueError('B 원료명을 입력하세요.')
            b_result,b_table=_run_b_temperature_sweep(
                models,payload,b_feedstock,b_process,b_composition)
            st.session_state['v147_comparison']={
                'signature':b_signature,'result':b_result,'table':b_table,
                'feedstock':b_feedstock,'process':b_process,
                'composition':b_composition,
            }
        except Exception as exc:
            st.session_state.pop('v147_comparison',None)
            st.error(f'B 조건 예측 실패: {type(exc).__name__}: {exc}')
    compared=st.session_state.get('v147_comparison')
    if compared and compared['signature']!=b_signature:
        st.info('B 입력값이 변경되었습니다. B 곡선을 다시 계산해 주세요.')
        return None
    return compared


def _render_pinned_ab_details(payload, compared):
    """Show numerical differences without treating model response as causal effect."""
    a=payload['result'];b=compared['result']
    delta=comparison_delta(a,b)
    st.markdown(f"#### 선택 제조온도 {payload['temperature']}℃ · A/B 결과")
    summary=pd.DataFrame([
        {'지표':'BET (m²/g)','A':a['BET_m2_g'],
         'B':b['BET_m2_g'],'B − A':delta['BET_m2_g']},
        {'지표':'TPV (cm³/g)','A':a['TPV_cm3_g'],
         'B':b['TPV_cm3_g'],'B − A':delta['TPV_cm3_g']},
        {'지표':'Reported Capacity (mg/g)','A':a['reported_capacity_mg_g'],
         'B':b['reported_capacity_mg_g'],
         'B − A':delta['reported_capacity_mg_g']},
    ])
    with st.expander('선택 온도 정밀 비교값',expanded=False):
        st.dataframe(summary,hide_index=True,use_container_width=True)
        st.caption('B − A는 동일한 제조온도에서 계산한 모델 예측 차이이며 실측 효과가 아닙니다.')
    combined=payload['table'].rename(columns={
        '예측 흡착량 (mg/g)':'A 예측 흡착량 (mg/g)',
        'BET (m²/g)':'A BET (m²/g)','TPV (cm³/g)':'A TPV (cm³/g)',
    })[['제조온도 (℃)','A 예측 흡착량 (mg/g)','A BET (m²/g)','A TPV (cm³/g)']]
    b_table=compared['table'].rename(columns={
        '예측 흡착량 (mg/g)':'B 예측 흡착량 (mg/g)',
        'BET (m²/g)':'B BET (m²/g)','TPV (cm³/g)':'B TPV (cm³/g)',
    })[['제조온도 (℃)','B 예측 흡착량 (mg/g)','B BET (m²/g)','B TPV (cm³/g)']]
    combined=combined.merge(b_table,on='제조온도 (℃)',validate='one_to_one')
    combined['B − A 흡착량 (mg/g)']=(
        combined['B 예측 흡착량 (mg/g)']-combined['A 예측 흡착량 (mg/g)'])
    st.download_button('A/B 제조온도 곡선 CSV',
        data=combined.to_csv(index=False).encode('utf-8-sig'),
        file_name=safe_export_stem(payload['feedstock'],payload['pollutant'],
                                    payload['temperature'])+'_AB_temperature.csv',
        mime='text/csv',key='v147_export_ab_curve',use_container_width=True)
    if b.get('stage1_input_warnings'):
        with st.expander('B 학습범위 상세 정보',expanded=False):
            st.warning('B 학습범위 주의: '+'; '.join(b['stage1_input_warnings']))
    if b.get('warnings'):
        st.warning('B 예측 경로 주의: '+'; '.join(map(str,b['warnings'])))
    outside=int((compared['table']['온도 범위 판정']==OUT_OF_RANGE).sum())
    unknown=int((compared['table']['온도 범위 판정']==NOT_VERIFIED).sum())
    if outside or unknown:
        with st.expander(f'B 온도 학습범위 상세 · 범위 밖 {outside}점 / 미확인 {unknown}점',expanded=False):
            st.caption('해당 점의 예측은 외삽 또는 근거 부족 가능성에 주의해야 합니다.')
    st.info('A/B 곡선은 문헌 기반 예측모델의 응답 비교입니다. 조건 변경의 인과효과나 신규 시료에서의 정확도를 보증하지 않습니다.')


def _render_ab_comparison_charts(case_a, case_b):
    """Present original A/B model outputs in three unit-specific bar charts.

    Deliberately do not normalize or mix BET, TPV and capacity scales.
    Preserve exact values in the CSV export; graphs are display-only.
    """
    import altair as alt

    specs = [
        ('BET', 'BET_m2_g', 'm²/g', 2),
        ('TPV', 'TPV_cm3_g', 'cm³/g', 4),
        ('Reported Capacity', 'reported_capacity_mg_g', 'mg/g', 3),
    ]
    for col, (title, key, unit, digits) in zip(st.columns(3), specs):
        a, b = float(case_a[key]), float(case_b[key])
        values = pd.DataFrame({
            '조건': ['A', 'B'],
            '예측값': [a, b],
            '표시값': [f'{a:,.{digits}f}', f'{b:,.{digits}f}'],
        })
        max_value = max(a, b, 0.0)
        axis_max = 1.0 if max_value == 0 else max_value * 1.32
        base = alt.Chart(values).encode(
            x=alt.X('예측값:Q', title=unit,
                    scale=alt.Scale(domain=[0, axis_max])),
            y=alt.Y('조건:N', sort=['A', 'B'], title=None),
            tooltip=[alt.Tooltip('조건:N'),
                     alt.Tooltip('예측값:Q', title=title, format='.6f')],
        )
        bars = base.mark_bar(cornerRadiusEnd=4, size=28).encode(
            color=alt.Color('조건:N', legend=None,
                            scale=alt.Scale(domain=['A', 'B'],
                                            range=['#315a72', '#d58b54'])),
        )
        numbers = base.mark_text(align='left', baseline='middle',
                                 dx=5, color='#263845', fontSize=12).encode(
            text=alt.Text('표시값:N'),
        )
        with col:
            st.markdown(f'**{title}**')
            st.altair_chart((bars + numbers).properties(height=145),
                            use_container_width=True)
            st.caption(f'B − A: {b-a:+.{max(digits+3, 6)}f} {unit}')


def render_v112_unified():
    """Keep existing v112 import/navigation contract while improving the one page."""
    st.markdown('''<style>
      .block-container {padding-top:1.4rem; padding-bottom:4.5rem; max-width:1220px;}
      div[data-baseweb="tab-panel"] {padding-top:1.1rem;}
      .v144-tab-bottom-space {height:4.5rem; width:100%;}
      [data-testid="stMetric"] {padding:0.9rem 1rem; border:1px solid rgba(128,128,128,.23); border-radius:7px;}
      [data-testid="stMetricLabel"] {letter-spacing:.01em;}
      div[data-testid="stForm"] {border:1px solid rgba(128,128,128,.23); border-radius:8px; padding:1rem;}
    </style>''',unsafe_allow_html=True)
    st.title('바이오차 제조·흡착성능 예측')
    st.caption('원료 조성 · 열분해조건 → BET / TPV → 문헌 기반 흡착량 · 독립 Manufacturing RF 제조온도 순위')
    try:
        models=_models(str(ROOT))
    except Exception as exc:
        st.error(f'모델을 읽을 수 없습니다: {type(exc).__name__}: {exc}')
        st.info('v139 물성·흡착량 학습 원장과 설치 환경을 확인해 주세요.')
        return
    st.markdown('### 01 · 원료 선택')
    feedstock=_feedstock_input(ROOT)
    if not str(feedstock).strip():
        st.session_state.pop('v140_last_prediction',None)
        st.session_state.pop('v141_comparison',None)
        st.session_state.pop('v147_pin_signature',None)
        st.session_state.pop('v147_comparison',None)
    # Reactive widgets ensure stale results are hidden before a new prediction.
    with st.container():
        st.markdown('### 02 · 열분해 공정')
        a,b,c=st.columns(3)
        with a:temp=st.slider('제조온도 (℃)',min_value=200,max_value=1200,value=300,step=25,
                              help='일부 구간은 학습 범위를 벗어날 수 있습니다.')
        with b:hold=st.slider('유지시간 (min)',min_value=10,max_value=600,value=60,step=10)
        with c:heat=st.slider('승온속도 (℃/min)',min_value=1,max_value=50,value=10,step=1)
        d,e=st.columns([1.0,1.0])
        with d:
            atm=st.selectbox('열분해 분위기',ATMOSPHERE_OPTIONS)
        with e:
            activation=st.selectbox('후처리/활성화',[
                '없음','화학적 활성화','물리적 활성화','수증기 활성화'],index=0)
        activation_modes={'없음':'none','화학적 활성화':'chemical_impregnation',
                '물리적 활성화':'physical_activation','수증기 활성화':'physical_steam'}
        activation_method=activation_modes[activation]
        with st.expander('활성화 세부 조건 (Stage 1 · 선택)',expanded=True):
            st.caption('활성화 방식이 없음이면 아래 조건은 무시됩니다. 미보고 수치는 공란으로 두며 0으로 대체하지 않습니다.')
            ref=models['s1_reference']
            agents=sorted(set().union(*(ref[k]['categorical'].get('activation_agent',()) for k in ('BET','TPV')))-{'UNKNOWN','none'})
            x1,x2=st.columns(2)
            with x1:
                activation_agent=st.selectbox('활성화제 종류',['UNKNOWN']+agents,
                    format_func=lambda v:'미보고 / NR' if v=='UNKNOWN' else v)
                activation_time=st.text_input('활성화 시간 (min)',placeholder='미보고')
            with x2:
                activation_temp=st.text_input('활성화 온도 (℃)',placeholder='미보고')
                activation_ratio=st.text_input('활성화제 / 원료 질량비 (g/g)',placeholder='미보고',
                    help='원료 질량 기준의 비율만 입력. 적용 불가나 미보고는 공란.')
        activation_details={'activation_agent':activation_agent if activation_method!='none' else 'none',
             'activation_temperature_C':activation_temp if activation_method!='none' else '',
             'activation_time_min':activation_time if activation_method!='none' else '',
             'activation_agent_feedstock_mass_ratio':activation_ratio if activation_method!='none' else ''}
        st.caption('기본 Stage 1: Poisson HGB · 10개 제조조건 변수. Stage 2: KNN + 가중 Extra Trees (50:50).')
        # The name field is outside the form and can be blank during editing.
        # Do not call resolve_stage1_feedstock until a non-empty name exists.
        stage1_preview=None
        if str(feedstock).strip():
            stage1_preview=assess_stage1_input(models,feedstock=feedstock,process={
                'pyrolysis_temperature_C':temp,'pyrolysis_hold_min':hold,
                'heating_rate_C_min':heat,'pyrolysis_atmosphere':atm,
                'activation_method':activation_method,**activation_details})
            if stage1_preview['used_feedstock'] != str(feedstock).strip():
                st.caption('실제 Stage 1 원료 피처: '+stage1_preview['used_feedstock']+
                           ' (동의어/표기 정규화, 원료 조성값 자동대입 아님)')
            if stage1_preview['warnings']:
                with st.expander('Stage 1 학습범위 상세 정보',expanded=False):
                    st.warning('Stage 1 입력 학습범위 주의: '+'; '.join(stage1_preview['warnings']))

        # Raw composition is collected only for feedstocks absent from the
        # Stage 1 training inventory; recognized names use the default HGB path.
        composition={}
        allow_comp_ood=False
        unknown_feedstock=(stage1_preview is not None and
                           stage1_preview['feedstock_match_status'] not in ('exact','alias'))
        if unknown_feedstock:
            with st.expander('미등록 원료 · 공업분석 및 원소분석 (선택)',expanded=True):
                st.caption('Stage 1 학습 원료에 없는 시료입니다. 열분해 전 원료의 공업분석(dry basis)·원소분석(daf basis) 실측값만 입력하세요. 모르는 값은 공란으로 둡니다.')
                for keys,title in [(['raw_VM_wt_pct','raw_ash_wt_pct','raw_FC_wt_pct'],'공업분석 (wt%)'),
                                    (['raw_C_wt_pct','raw_H_wt_pct','raw_N_wt_pct','raw_O_wt_pct'],'원소분석 (wt%)')]:
                    st.markdown('**'+title+'**')
                    cols=st.columns(len(keys))
                    for col,k in zip(cols,keys):
                        label=k.removeprefix('raw_').replace('_wt_pct','').upper().replace('ASH','Ash')
                        with col:composition[k]=st.text_input(label,placeholder='미측정',key=f'v113_{k}')
                st.caption('조성 입력 시 별도 연구용 RF BET 경로가 우선 적용될 수 있습니다. 기본 HGB 검증과는 별개이며 TPV는 10변수 HGB입니다.')
                allow_comp_ood=st.checkbox('학습 범위 밖 조성의 탐색용 BET 계산 허용',value=False,
                    key='v141_allow_comp_ood',help='조성값만 범위를 벗어나고 공정조건은 학습범위 안인 경우에 한하여 탐색적 계산. 미허용 시 기본 HGB를 사용합니다.')
        elif stage1_preview is not None:
            st.caption('Stage 1 등록 원료: 공업·원소분석 입력 없이 기존 제조조건 기반 HGB 모델을 사용합니다.')

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
        submitted=st.button('흡착성능 예측 및 조건 비교',type='primary',use_container_width=True,
                            disabled=not bool(str(feedstock).strip()))

    current_process={'pyrolysis_temperature_C':temp,'pyrolysis_hold_min':hold,
         'heating_rate_C_min':heat,'pyrolysis_atmosphere':atm,
         'activation_method':activation_method,**activation_details}
    current_ads={'pollutant':pollutant,'C0_mg_L':c0,'dose_g_L':dose,
         'initial_pH':_num_optional(None if missing_ph else ph),
         'adsorption_T_C':_num_optional(None if missing_ads_temp else ads_temp),
         'contact_time_min':_num_optional(None if missing_contact else contact)}
    current_signature=input_signature({
         'feedstock':feedstock,'process':current_process,'adsorption':current_ads,
         'composition':composition,'sweep':list(sweep),'step':step,
         'allow_composition_ood':allow_comp_ood})

    if submitted:
        try:
            if not str(feedstock).strip():raise ValueError('원료명을 입력해 주세요.')
            process=current_process
            ads=current_ads
            active=activation_method!='none'
            result=predict_one(models,ROOT,feedstock=feedstock,process=process,
                               composition=composition,adsorption=ads,activated=active,
                        allow_composition_ood=allow_comp_ood)
            ts=sorted(set([sweep[0],sweep[1]]+list(range(sweep[0],sweep[1]+1,step))))
            if len(ts)>45:raise ValueError('온도 비교 구간은 45개 이하로 설정해 주세요.')
            rows=[]
            for t in ts:
                pr=predict_one(models,ROOT,feedstock=feedstock,
                       process={**process,'pyrolysis_temperature_C':t},
                       composition=composition,adsorption=ads,activated=active,
                        allow_composition_ood=allow_comp_ood)
                rows.append({'제조온도 (℃)':int(t),'예측 흡착량 (mg/g)':pr['reported_capacity_mg_g'],
                             'BET (m²/g)':pr['BET_m2_g'],'TPV (cm³/g)':pr['TPV_cm3_g']})
            df=pd.DataFrame(rows)
            stage2_bounds=_stage2_temperature_bounds(str(ROOT), pollutant)
            df=attach_coverage(df, models, stage2_bounds)
            st.session_state['v140_last_prediction']={
                'result':result,'table':df,'feedstock':feedstock,
                'pollutant':pollutant,'temperature':temp,
                'process':process,'adsorption':ads,'composition':composition,
                'stage2_temperature_bounds':stage2_bounds,
                'input_signature':current_signature,
                'allow_composition_ood':allow_comp_ood,
            }
            st.session_state.pop('v141_comparison',None)
            st.session_state.pop('v147_pin_signature',None)
            st.session_state.pop('v147_comparison',None)
        except Exception as exc:
            st.session_state.pop('v140_last_prediction',None)
            st.session_state.pop('v141_comparison',None)
            st.session_state.pop('v147_pin_signature',None)
            st.session_state.pop('v147_comparison',None)
            st.error(f'예측을 완료하지 못했습니다: {type(exc).__name__}: {exc}')

    payload=st.session_state.get('v140_last_prediction')
    if payload is None:
        st.caption('조건을 설정한 뒤 위의 예측 버튼을 누르면 결과와 온도별 비교 그래프가 표시됩니다.')
        return
    if payload.get('input_signature') != current_signature:
        if st.session_state.get('v147_pin_signature')==payload['input_signature']:
            st.info('A 조건이 고정되어 있습니다. 상단의 변경값은 새 A 예측을 실행하기 전까지 반영되지 않습니다.')
        else:
            st.warning('입력조건이 변경되었습니다. 이전 결과를 현재 입력의 예측으로 오인하지 않도록 숨겼습니다. 예측 버튼을 다시 눌러 주세요.')
            return
    r=payload['result'];df=payload['table']
    st.caption('학습 문헌 기반 연구용 모델 · 출력값은 실측값이나 독립 검증 결과가 아닙니다.')
    st.markdown('### 예측 결과')
    p,q,t=st.columns(3)
    p.metric('BET (예측)',f"{r['BET_m2_g']:,.2f} m²/g")
    q.metric('TPV (예측)',f"{r['TPV_cm3_g']:,.4f} cm³/g")
    t.metric('Reported Capacity (예측)',f"{r['reported_capacity_mg_g']:,.2f} mg/g")
    if r.get('stage1_input_warnings'):
        with st.expander('예측 당시 학습범위 상세 정보',expanded=False):
            st.warning('예측 당시 입력 학습범위 주의: '+'; '.join(r['stage1_input_warnings']))
    if not r['known_pollutant']:
        st.warning('미학습 오염물질에 대한 외삽 예측입니다.')
    if not r['stage1_property_feature_used']:
        st.info('이 오염물질 계열에서는 Stage 2가 BET·TPV를 사용하지 않습니다. 원료 조성을 바꿔도 최종 흡착량은 변하지 않을 수 있습니다.')
    if r['composition_status']=='FALLBACK':
        st.warning('조성 입력을 반영하지 못하고 제조조건 기반 BET로 계산했습니다: '+'; '.join(map(str,r['warnings'])))
    elif r['composition_status']=='APPLIED':
        st.caption('조성은 Stage 1 BET를 통해 간접 반영되었습니다. 입력 변화에 따른 출력 차이는 인과관계가 아닙니다.')
    tab_sweep,tab_exports=st.tabs(['제조온도·RF 순위 · A/B 비교','내보내기'])
    with tab_sweep:
        heading,pin_col=st.columns([5,1.35])
        with heading:
            st.markdown('### 제조온도 비교')
        pinned=st.session_state.get('v147_pin_signature')==payload['input_signature']
        with pin_col:
            if st.button('고정 해제' if pinned else '📌 A 고정',
                         key='v147_toggle_pin',use_container_width=True):
                if pinned:
                    st.session_state.pop('v147_pin_signature',None)
                    st.session_state.pop('v147_comparison',None)
                    pinned=False
                else:
                    st.session_state['v147_pin_signature']=payload['input_signature']
                    st.session_state.pop('v147_comparison',None)
                    _set_b_defaults_from_a(payload)
                    pinned=True
        compared=None
        if pinned:
            chart_col,b_col=st.columns([2.45,1.05],gap='medium')
            with b_col:
                compared=_render_pinned_b_inputs(models,payload)
            with chart_col:
                _render_temperature_curve(df,
                    compared['table'] if compared else None,pinned=True)
                st.markdown('**A: 점선(고정) · B: 실선(비교)**')
                if compared is None:
                    st.info('오른쪽에서 B 조건을 입력하고 B 곡선 계산을 누르세요.')
            if compared:
                _render_pinned_ab_details(payload,compared)
        else:
            _render_temperature_curve(df)
        outside=int((df['온도 범위 판정']==OUT_OF_RANGE).sum())
        unknown=int((df['온도 범위 판정']==NOT_VERIFIED).sum())
        if outside or unknown:
            with st.expander(f'온도 학습범위 상세 · 범위 밖 {outside}/{len(df)}점 / 미확인 {unknown}점',expanded=False):
                st.caption('해당 점의 예측은 외삽 또는 근거 부족 가능성에 주의해야 합니다.')
        top=df.loc[df['예측 흡착량 (mg/g)']==df['예측 흡착량 (mg/g)'].max()]
        st.download_button(
            '온도 비교 결과 CSV',
            data=df.to_csv(index=False).encode('utf-8-sig'),
            file_name=safe_export_stem(payload['feedstock'],payload['pollutant'],payload['temperature'])+'_temperature_comparison.csv',
            mime='text/csv',key='v146_temperature_csv',use_container_width=True)
        st.markdown('### 제조조건 순위 추천')
        try:
            rank_bundle=_ranking_model(str(ROOT))
            if not r['known_pollutant']:
                st.info('통합 흡착량 모델이 지원하지 않는 오염물질에는 공통 추천을 제공하지 않습니다.')
            else:
                ranking,rank_notes=rank_sweep(rank_bundle,payload['feedstock'],payload['process'],
                                             payload['pollutant'],df['제조온도 (℃)'].tolist())
                if rank_notes:
                    label='RF 추천 미지원 · 상세 정보' if ranking is None else 'RF 적용범위 상세 정보'
                    with st.expander(label,expanded=False):
                        st.caption('순위 모델 적용 범위: '+'; '.join(rank_notes))
                if ranking is not None:
                    rank_table=df[['제조온도 (℃)','예측 흡착량 (mg/g)']].merge(ranking,on='제조온도 (℃)',how='left')
                    rank_table['RF 순위']=rank_table['RF 순위'].astype('Int64')
                    winners=ranking.loc[ranking['RF 순위'].eq(1),'제조온도 (℃)']
                    a,b=st.columns(2)
                    a.metric('예측 흡착량 최대 후보',', '.join(f'{v:g}℃' for v in top['제조온도 (℃)']))
                    b.metric('RF 1위 후보 (탐색)',', '.join(f'{v:g}℃' for v in winners))
                    if len(winners)>1:
                        st.info('RF 1위 동률은 그대로 표시합니다. 임의로 최적 조건을 고르지 않습니다.')
                    if compared:
                        b_ranking,b_notes=rank_sweep(
                            rank_bundle,compared['feedstock'],compared['process'],
                            payload['pollutant'],df['제조온도 (℃)'].tolist())
                        if b_ranking is not None:
                            b_winners=b_ranking.loc[
                                b_ranking['RF 순위'].eq(1),'제조온도 (℃)']
                            st.metric('B RF 1위 후보 (탐색)',
                                      ', '.join(f'{v:g}℃' for v in b_winners))
                        if b_notes:
                            label='B RF 추천 미지원 · 상세 정보' if b_ranking is None else 'B RF 적용범위 상세 정보'
                            with st.expander(label,expanded=False):
                                st.caption('B 순위 모델 적용 범위: '+'; '.join(b_notes))
                    rank_export=rank_table.copy()
                    rank_export['rank_model']=rank_bundle['version']
                    rank_export['rank_training_snapshot_sha256']=rank_bundle['sha256']
                    rank_export['feedstock']=payload['feedstock']
                    rank_export['pollutant']=payload['pollutant']
                    for key,value in {**payload['process'],**payload['adsorption']}.items():
                        if key!='pyrolysis_temperature_C': rank_export[key]=value
                    rank_export['rank_notes']='; '.join(rank_notes)
                    st.download_button('제조조건 순위 CSV',rank_export.to_csv(index=False).encode('utf-8-sig'),
                                       file_name=safe_export_stem(payload['feedstock'],payload['pollutant'],payload['temperature'])+'_ranking.csv',mime='text/csv',
                                       use_container_width=True)
        except Exception as exc:
            st.warning(f'순위 모델을 실행하지 못했습니다: {type(exc).__name__}: {exc}. 기존 흡착량 예측은 유지됩니다.')
    with tab_exports:
        st.markdown('### 포스터용 결과 출력')
        st.caption('PNG 300 dpi / SVG 벡터 · BET, TPV, Reported Capacity 3패널. '
                   '모델 불확실도나 실측값을 임의로 표시하지 않습니다.')
        export_warnings='; '.join(w for w in r.get('stage1_input_warnings', [])
                                  if not w.startswith('제조온도 '))
        model_sha=models['stage2']['training_snapshot_sha256']
        csv_bytes=export_csv_with_inputs(
            df,feedstock_input=payload['feedstock'],
            feedstock_used=str(r.get('stage1_feedstock_used',payload['feedstock'])),
            process=payload['process'],adsorption=payload['adsorption'],
            composition=payload['composition'],model_sha256=model_sha,
            properties_connected=bool(r['stage1_property_feature_used']),
            stage1_warnings=export_warnings,stage1_bet_path=r['bet_path'],stage1_tpv_path=r['tpv_path'])
        export_stem=safe_export_stem(payload['feedstock'],payload['pollutant'],payload['temperature'])
        ex1,ex2,ex3=st.columns(3)
        with ex1:
            st.download_button('포스터 PNG (300 dpi)', _poster_image(df,'png'),
                               file_name=export_stem+'_3panel.png',mime='image/png',
                               use_container_width=True)
        with ex2:
            st.download_button('포스터 SVG (벡터)', _poster_image(df,'svg'),
                               file_name=export_stem+'_3panel.svg',mime='image/svg+xml',
                               use_container_width=True)
        with ex3:
            st.download_button('조건·예측값 CSV',csv_bytes,
                               file_name=export_stem+'_inputs_predictions.csv',
                               mime='text/csv',use_container_width=True)
        st.markdown('<div class="v144-tab-bottom-space" aria-hidden="true"></div>',
                    unsafe_allow_html=True)
