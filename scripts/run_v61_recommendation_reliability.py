"""v61: evidence-aware, tie-aware manufacturing recommendation audit.

Consumes v60 local OOF predictions. This is POSTHOC evaluation only; it does
not retrain estimators and MUST NOT edit source, active, or prior report files.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

SRC = Path('reports/v60_separate_effects')
SRC_CHEN = Path('reports/v59_doi_material_weighting/chen16_weighted_predictions.csv')
DEST = Path('reports/v61_recommendation_reliability')
TIE_TOL = 1e-8
GAP_AUDIT_MG_G = 1.0  # sensitivity analysis, NOT calibrated uncertainty
QE_MODELS = ('A_row', 'B_row', 'B_doi_material')
ALL_MODELS = QE_MODELS + ('mfg_forest',)
SCENARIOS = (152, 188)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_float(value):
    value=float(value)
    return value if math.isfinite(value) else None


def top_set(values, atol=TIE_TOL):
    a=np.asarray(values,dtype=float)
    if a.ndim != 1 or len(a)==0 or not np.isfinite(a).all():
        raise ValueError('Scores must be finite nonempty 1-D values')
    return np.flatnonzero(a >= a.max()-atol)


def pairwise_stats(y, scores, tol=TIE_TOL):
    """Observed ties excluded; predicted ties receive 0.5 credit."""
    y=np.asarray(y,float); s=np.asarray(scores,float)
    if not len(y)==len(s) or not (np.isfinite(y).all() and np.isfinite(s).all()):
        raise ValueError('Invalid pairwise inputs')
    good=tied=total=0
    for i in range(len(y)):
        for j in range(i+1,len(y)):
            dy=y[i]-y[j]; ds=s[i]-s[j]
            if abs(dy)<=tol:continue
            total+=1
            if abs(ds)<=tol:good+=0.5;tied+=1
            elif dy*ds>0:good+=1
    return {'pairwise_concordance':good/total if total else None,'n_comparable_pairs':total,'n_predicted_tied_pairs':tied}


def top_regret(y, scores, atol=TIE_TOL):
    """Conservative regret interval over all predicted joint-top candidates."""
    y=np.asarray(y,float);indices=top_set(scores,atol=atol);best=y.max()
    return {'n_top_tied':len(indices),'top_indices':indices.tolist(),
            'regret_best_case':float(best-y[indices].max()),
            'regret_worst_case':float(best-y[indices].min()),
            'regret_mean_if_random_tiebreak':float(best-y[indices].mean()),
            'unique_top1':bool(len(indices)==1)}


def eval_panel(panel, scenario, model, *, full, strict):
    if panel.empty or panel.panel_id.nunique()!=1 or panel.doi.nunique()!=1:
        raise ValueError('Requires exactly one DOI manufacturing panel')
    panel=panel.copy().sort_values('observation_id')
    y=panel.target_value.to_numpy(float)
    if not np.isfinite(y).all():raise ValueError('Missing observations')
    q_col=f'{scenario}__{model}'
    s=panel[q_col].to_numpy(float)
    is_qe=model in QE_MODELS
    if is_qe:
        cap=panel.C0_mg_L.to_numpy(float)/panel.dose_g_L.to_numpy(float)
        if (not np.isfinite(cap).all()) or (cap<=0).any():raise ValueError('Missing or invalid C0/dose')
        raw=panel[f'{q_col}__raw'].to_numpy(float)
        if not np.isfinite(raw).all():raise ValueError('Missing raw predictions')
        if not np.allclose(s,np.clip(raw,0,cap),atol=2e-5,rtol=1e-8):
            raise ValueError(f'Clipped predictions do not match raw in {q_col}')
        cap_exceed=raw>cap+TIE_TOL
        neg=raw<-TIE_TOL
        top_clipped=top_set(s);top_raw=top_set(raw)
        raw_pairs=pairwise_stats(y,raw)
        final_pairs=pairwise_stats(y,s)
        new_tie_count=0
        for i in range(len(y)):
            for j in range(i+1,len(y)):
                if abs(y[i]-y[j])<=TIE_TOL:continue
                if abs(s[i]-s[j])<=TIE_TOL and abs(raw[i]-raw[j])>TIE_TOL:new_tie_count+=1
        raw_summary=top_regret(y,raw)
        at_cap=int(np.isclose(s,cap,atol=1e-8,rtol=0).sum())
        all_top_at_cap=bool(np.all(np.isclose(s[top_clipped],cap[top_clipped],atol=1e-8,rtol=0)))
        raw_top_is_legal=bool(np.all((raw[top_raw]>=-TIE_TOL)&(raw[top_raw]<=cap[top_raw]+TIE_TOL)))
        margin_units='mg/g'
    else:
        raw=None;cap=None;top_raw=None;raw_summary=None;raw_pairs=None
        final_pairs=pairwise_stats(y,s);new_tie_count=0;cap_exceed=[];neg=[];at_cap=None
        all_top_at_cap=None;raw_top_is_legal=None;margin_units='within-panel removal percentage points (relative)'
    tops=top_regret(y,s)
    gap=float(np.sort(s)[-1]-np.sort(s)[-2]) if len(s)>1 else None
    if not full:status='WITHHOLD_MISSING_ADSORPTION_CONTEXT'
    elif not strict:status='WITHHOLD_TARGET_NOT_STRICT_QE'
    elif tops['n_top_tied']>1:status='WITHHOLD_TIED_TOP'
    elif is_qe and all_top_at_cap:status='WITHHOLD_TOP_AT_MASS_BALANCE_CAP'
    elif is_qe and gap is not None and gap<=GAP_AUDIT_MG_G:
        status='WITHHOLD_SMALL_MARGIN_1MGG_HEURISTIC'
    else:status='RESEARCH_UNIQUE_CANDIDATE_NOT_DEPLOYMENT_CLEARED'
    out={
        'scenario':scenario,'model':model,'doi':str(panel.doi.iloc[0]),
        'panel_id':int(panel.panel_id.iloc[0]),'n_candidates':len(panel),
        'n_full_context':bool(full),'n_strict_qe':bool(strict),
        'score_type':'bounded_qe_mg_g' if is_qe else 'relative_manufacturing_score_NOT_QE',
        'top_score_gap':gap,'top_gap_units':margin_units,
        'n_predicted_top_tied':tops['n_top_tied'],
        'predicted_top_observation_ids':'|'.join(panel.iloc[tops['top_indices']].observation_id.astype(str)),
        'regret_best_case_mg_g':tops['regret_best_case'],
        'regret_worst_case_mg_g':tops['regret_worst_case'],
        'regret_random_tie_break_mg_g':tops['regret_mean_if_random_tiebreak'],
        'pairwise_concordance':final_pairs['pairwise_concordance'],
        'n_comparable_pairs':final_pairs['n_comparable_pairs'],
        'n_predicted_tied_pairs':final_pairs['n_predicted_tied_pairs'],
        'recommendation_gate':status,
        'raw_cap_exceed_rows':int(np.sum(cap_exceed)) if is_qe else None,
        'raw_negative_rows':int(np.sum(neg)) if is_qe else None,
        'n_clipped_at_cap':at_cap,
        'n_top_raw':raw_summary['n_top_tied'] if is_qe else None,
        'raw_regret_worst_case_mg_g_DIAGNOSTIC_NOT_PHYSICAL':raw_summary['regret_worst_case'] if is_qe else None,
        'raw_pairwise_concordance_DIAGNOSTIC_NOT_PHYSICAL':raw_pairs['pairwise_concordance'] if is_qe else None,
        'new_tied_pairs_after_clipping':new_tie_count if is_qe else None,
        'all_clipped_top_at_cap':all_top_at_cap,
        'raw_top_physically_legal':raw_top_is_legal,
    }
    return out


def evaluate_chen(chen):
    required={'material_uid','reported_removal_pct'}
    if not required.issubset(chen.columns) or len(chen)!=16:
        raise ValueError('Chen benchmark shape or fields changed')
    y=chen.reported_removal_pct.to_numpy(float)
    rows=[]
    for col in chen.columns:
        if col not in required and pd.api.types.is_numeric_dtype(chen[col]):
            s=chen[col].to_numpy(float)
            if not np.isfinite(s).all():continue
            tops=top_regret(y,s);pair=pairwise_stats(y,s)
            rows.append({'score_column':col,'n':len(y),
                         'n_top_tied':tops['n_top_tied'],
                         'top_ids':'|'.join(chen.iloc[tops['top_indices']].material_uid.astype(str)),
                         'unique_top1':tops['unique_top1'],
                         'top_regret_best_pp':tops['regret_best_case'],
                         'top_regret_worst_pp':tops['regret_worst_case'],
                         'top_regret_random_tie_break_pp':tops['regret_mean_if_random_tiebreak'],
                         'pairwise_concordance':pair['pairwise_concordance'],
                         'predicted_tied_pairs':pair['n_predicted_tied_pairs'],
                         'recommendation_gate':'WITHHOLD_MISSING_INITIAL_PH_IN_SOURCE',
                         'is_posthoc':True})
    return pd.DataFrame(rows)


def run(root, output=None):
    root=Path(root).resolve();out=Path(output).resolve() if output else root/DEST
    pred_path=root/SRC/'all188_doi_oof_predictions.csv'
    quality_path=root/SRC/'panel_evidence_quality.csv'
    chen_path=root/SRC_CHEN
    for path in (pred_path,quality_path,chen_path):
        if not path.is_file():raise FileNotFoundError(f'Requires existing v59/v60 report: {path}')
    out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(pred_path);qual=pd.read_csv(quality_path);chen=pd.read_csv(chen_path)
    if len(df)!=188 or df.observation_id.duplicated().any():
        raise ValueError('Expected 188 unique v60 OOF observations')
    if len(qual)!=15 or qual.panel_id.duplicated().any():
        raise ValueError('Expected 15 evidence audited manufacturing panels')
    # Guard against mapping a panel from a different dataset version.
    qq=qual.set_index('panel_id')
    panel_ids=sorted(int(v) for v in df.loc[df.panel_id>=0,'panel_id'].unique())
    if panel_ids!=sorted(int(v) for v in qq.index):raise ValueError('Panel IDs differ between predictions and evidence')
    rows=[]
    for sc in SCENARIOS:
        for mod in ALL_MODELS:
            for panel_id,panel in df.loc[df.panel_id>=0].groupby('panel_id'):
                q=qq.loc[panel_id]
                if str(panel.doi.iloc[0])!=str(q.doi):raise ValueError('DOI evidence mismatch')
                rows.append(eval_panel(panel,sc,mod,
                            full=bool(q.full_conditions_reported),
                            strict=bool(q.strict_experimental_qe_confirmed_in_vNext10)))
    # Retain original raw, clipped and cap values for all observations, not only panels.
    clip_rows=[]
    for sc in SCENARIOS:
        for mod in QE_MODELS:
            key=f'{sc}__{mod}'
            cap=df.C0_mg_L.to_numpy(float)/df.dose_g_L.to_numpy(float)
            if not np.isfinite(cap).all() or (cap<=0).any():
                raise ValueError('Nonpositive or missing mass-balance cap in OOF')
            raw=df[f'{key}__raw'].to_numpy(float)
            actual=df[key].to_numpy(float)
            if not np.allclose(actual,np.clip(raw,0,cap),atol=2e-5,rtol=1e-8):
                raise ValueError('Raw/clipped OOF mismatch for '+key)
            clip_rows.append(pd.DataFrame({
                'scenario':sc,'model':mod,'observation_id':df.observation_id,
                'doi':df.doi,'panel_id':df.panel_id,'observed_qe_mg_g':df.target_value,
                'cap_C0_over_dose_mg_g':cap,'prediction_raw_mg_g_DIAGNOSTIC':raw,
                'prediction_clipped_mg_g':actual,
                'raw_exceeds_cap':raw>cap+TIE_TOL,
                'raw_below_zero':raw<-TIE_TOL,
                'removed_by_clipping_mg_g':raw-actual,
            }))
    pd.concat(clip_rows,ignore_index=True).to_csv(out/'observation_clipping_audit.csv',index=False,encoding='utf-8-sig')
    scores=pd.DataFrame(rows)
    scores.to_csv(out/'panel_recommendation_audit.csv',index=False,encoding='utf-8-sig')
    summaries=[]
    for (sc,mod),g in scores.groupby(['scenario','model'],sort=True):
        rank_valid=g.pairwise_concordance.notna()
        summaries.append({'scenario':int(sc),'model':mod,'n_panels':len(g),'n_doi':g.doi.nunique(),
            'n_unique_top':int((g.n_predicted_top_tied==1).sum()),
            'n_tied_top':int((g.n_predicted_top_tied>1).sum()),
            'n_strict_context':int((g.n_strict_qe & g.n_full_context).sum()),
            'n_research_candidates_not_deployment':int(g.recommendation_gate.str.startswith('RESEARCH_').sum()),
            'n_withhold':int(g.recommendation_gate.str.startswith('WITHHOLD_').sum()),
            'mean_best_tie_regret_mg_g':float(g.regret_best_case_mg_g.mean()),
            'mean_worst_tie_regret_mg_g':float(g.regret_worst_case_mg_g.mean()),
            'mean_pairwise_concordance_valid':float(g.loc[rank_valid,'pairwise_concordance'].mean()),
            'n_valid_pairwise_panels':int(rank_valid.sum()),
            'n_new_predicted_ties_after_clipping':int(g.new_tied_pairs_after_clipping.fillna(0).sum()) if mod in QE_MODELS else None,
            'n_raw_cap_exceed_rows_across_panels':int(g.raw_cap_exceed_rows.fillna(0).sum()) if mod in QE_MODELS else None,
        })
    summ=pd.DataFrame(summaries)
    summ.to_csv(out/'panel_audit_summary.csv',index=False,encoding='utf-8-sig')
    ch=evaluate_chen(chen)
    ch.to_csv(out/'chen16_tie_aware_regret.csv',index=False,encoding='utf-8-sig')
    # sensitivity of top choice to increasing the tie tolerance (in same score units).
    tol=[]
    for sc in SCENARIOS:
        for mod in ALL_MODELS:
            col=f'{sc}__{mod}'
            for tolerance in (1e-8,0.1,1.0):
                n_multi=0;n_eligible=0
                for pid,p in df[df.panel_id>=0].groupby('panel_id'):
                    n_top=len(top_set(p[col].to_numpy(float),atol=tolerance))
                    n_multi+=int(n_top>1)
                    n_eligible+=int(n_top==1)
                tol.append({'scenario':sc,'model':mod,'tie_tolerance_in_model_units':tolerance,
                            'n_multi_top':n_multi,'n_unique_top':n_eligible,
                            'note':'SENSITIVITY_ONLY; not a calibrated uncertainty interval'})
    pd.DataFrame(tol).to_csv(out/'top_tie_tolerance_sensitivity.csv',index=False)
    m={'version':'v61','role':'RESEARCH_AUDIT_ONLY_NOT_ACTIVE','posthoc':True,
       'train_new_models':False,'n_observations':len(df),'n_panels':len(qual),'n_dois':int(qual.doi.nunique()),
       'n_complete_panels':int(qual.full_conditions_reported.sum()),
       'n_strict_panels':int(qual.strict_experimental_qe_confirmed_in_vNext10.sum()),
       'source_sha256':{str(path.relative_to(root)):sha256(path) for path in (pred_path,quality_path,chen_path)},
       'top_tie_tolerance':TIE_TOL,'heuristic_gap_not_calibrated_mg_g':GAP_AUDIT_MG_G,
       'warning':'No deployment claim; retrospective reused DOIs; Chen initial pH missing; original v60 local artifacts may differ by library versions.'}
    (out/'manifest_v61.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# v61 — 물질수지 Clipping·순위 동률·추천 보류 감사','',
           '본 보고서는 이 실행 환경의 v60 DOI OOF 예측을 읽어 **재학습 없이** 재계산했다. 연구용 사후 검증이며, 단일 조건을 실용적으로 추천하는 검증은 아니다.','',
           f"- 제조 비교계열 {m['n_panels']}개 / {m['n_dois']} DOI (완전 흡착조건 {m['n_complete_panels']}개, vNext10 엄격 qe {m['n_strict_panels']}개)",
           '- `qe` 상한: C0/dose. Clipping 전 값은 물리적으로 유효하지 않으면 추천 점수로 사용하지 않는다.',
           '- 공동 1위는 실제 최고 관측값을 기준으로 최선·최악 regret 범위를 모두 보고하며 임의의 첫 행 선택은 하지 않는다.',
           '- Spearman undefined일 때 0으로 대체하지 않는다. Pairwise predicted tie는 0.5점.',
           '- 1 mg/g 차이 경고는 **탐색용 휴리스틱**이며 예측구간이나 보장 정확도를 의미하지 않는다.','',
           '## 제조조건 비교계열 요약','',
           '|학습 시나리오|모델|공동 1위 계열|독립 1위 계열|clipping 신규 동률쌍|평균 최악 regret (mg/g)|연구용 후보(배포 불가)|',
           '|---|---|---:|---:|---:|---:|---:|']
    for s in summaries:
        lines.append(f"|{s['scenario']}|{s['model']}|{s['n_tied_top']}|{s['n_unique_top']}|{s['n_new_predicted_ties_after_clipping'] if s['n_new_predicted_ties_after_clipping'] is not None else 'N/A'}|{s['mean_worst_tie_regret_mg_g']:.3f}|{s['n_research_candidates_not_deployment']}|")
    lines.extend(['','## Chen 16조건 벤치마크 — 참고용 사후 분석','',
                  '초기 pH가 확인되지 않아 **모든 후보의 배포 추천은 보류**한다. 기존 클리핑 예측 결과만 비교했으므로 원시 스코어를 복구했다고 주장하지 않는다.','',
                  '|예측 점수|공동 1위 수|최선 regret (%p)|최악 regret (%p)|','|---|---:|---:|---:|'])
    for row in ch[ch.score_column.isin(['v55_152__cap__row_equal','v55_152__cap__doi_sqrt_material_equal','v57_188__cap__doi_sqrt_material_equal'])].itertuples(index=False):
        lines.append(f'|{row.score_column}|{row.n_top_tied}|{row.top_regret_best_pp:.2f}|{row.top_regret_worst_pp:.2f}|')
    lines.extend(['','## 파일 안내','','- `panel_recommendation_audit.csv`: 15개 계열 × 2개 시나리오 × 4개 모델 상세 결과',
                  '- `panel_audit_summary.csv`: 모델별 동률·추천 보류·regret 집계',
                  '- `observation_clipping_audit.csv`: 188행 × 2개 시나리오 × 3개 qe 모델의 raw·clipped·cap 전수 점검',
                  '- `chen16_tie_aware_regret.csv`: Chen 16개 조건 전체 가중치 모델별 공동 최적 처리',
                  '- `top_tie_tolerance_sensitivity.csv`: 수치상 동률 허용오차 민감도',
                  '- `manifest_v61.json`: 실행 입력 데이터 SHA256','',
                  '## 연구 해석 한계','','추천 보류는 미보고 조건이나 유효하지 않은 근거에서 자동 1위를 고르는 것보다 타당하다. 그러나 이 기준은 사전 등록·외부 검증된 decision rule이 아니라 연구용 보수적 게이트이다. 다른 PC에서 재학습한 v60 예측을 넣으면 수치가 달라질 수 있으며 그 값을 우선한다.'])
    (out/'RESULTS_KO.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return m,summ,ch


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='.')
    ap.add_argument('--output',default=None)
    a=ap.parse_args()
    m,s,ch=run(a.root,a.output)
    print(json.dumps({k:m[k] for k in ['version','n_observations','n_panels','n_dois','n_complete_panels','n_strict_panels']},ensure_ascii=False))
    print(s[['scenario','model','n_tied_top','n_raw_cap_exceed_rows_across_panels','n_research_candidates_not_deployment']].to_string(index=False))

if __name__=='__main__':main()
