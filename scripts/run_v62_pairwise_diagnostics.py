"""v62 research-only manufacturing pairwise diagnostic / paper-group OOF.

Always compare to locally regenerated v60 OOF on identical groups. No active deployment edits.
Predicted scores are Borda-style within-panel win probabilities (NOT qe, NOT removal %).
"""
from __future__ import annotations
import argparse, hashlib, json, platform, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import sklearn
from scripts.run_v60_separate_effects import (MFG, MFG_NUM, MFG_CATS, panel_inventory,
                                                panel_centered_training, OLD, PRE, NEW)
from scripts.build_v46_phase8 import features

OUT=Path('reports/v62_pairwise_manufacturing')
SRC=Path('reports/v60_separate_effects')
FEATURE_TOL=1e-10
SCORE_TOL=1e-8

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save_json(path, obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def val(x):
    try:
        f=float(x)
        return f if np.isfinite(f) else None
    except (TypeError,ValueError):return None

def preprocessor():
    numeric=Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('scale',StandardScaler())])
    categ=Pipeline([('impute',SimpleImputer(strategy='constant',fill_value='UNKNOWN')),('ohe',OneHotEncoder(handle_unknown='ignore'))])
    return ColumnTransformer([('num',numeric,MFG_NUM),('cat',categ,MFG_CATS)])

def pair_training_data(t, y, weights):
    pairs=[]; labels=[]; wts=[]; n_pairs=0; n_identical=0
    for p,g in t.groupby('panel_id'):
        ix=g.index.to_numpy(); local=t.index.get_indexer(ix)
        for i in range(len(local)):
            for j in range(i+1,len(local)):
                a,b=local[i],local[j]
                dy=float(y[a]-y[b])
                if abs(dy)<SCORE_TOL:continue
                pairs.append((a,b)); labels.append(int(dy>0));wts.append(.5*(weights[a]+weights[b])/(len(local)-1)); n_pairs+=1
    return pairs,np.asarray(labels,int),np.asarray(wts,float)

def train_pairwise(train):
    """Fits source-blind pairwise logit; DOI and panel used only for training grouping and balancing."""
    t,y,weights=panel_centered_training(train)
    t=t.reset_index(drop=True)
    if len(t)<8 or t.doi.nunique()<2:return None,{'train_rows':len(t),'train_panels':t.panel_id.nunique(),'reason':'insufficient source DOI'}
    X=features(t,MFG,MFG_NUM,MFG_CATS)
    pre=preprocessor(); z=pre.fit_transform(X)
    if not sp.issparse(z):z=sp.csr_matrix(z)
    pairs,labels,w=pair_training_data(t,y,weights)
    if not pairs:return None,{'train_rows':len(t),'train_panels':t.panel_id.nunique(),'reason':'no observed comparable pair'}
    li=np.array([p[0] for p in pairs]);ri=np.array([p[1] for p in pairs]);diff=(z[li]-z[ri]).tocsr()
    norms=np.sqrt(np.asarray(diff.multiply(diff).sum(axis=1)).reshape(-1))
    identical=(norms<=FEATURE_TOL)
    # Keep zero-information pairs in audit, omit from training to avoid fabricating signal.
    diff=diff[~identical]; labels=labels[~identical];w=w[~identical]
    if diff.shape[0]<4:return None,{'train_rows':len(t),'train_panels':t.panel_id.nunique(),'reason':'unidentifiable all pairs'}
    both=sp.vstack([diff,-diff],format='csr')
    yy=np.r_[labels,1-labels];ww=np.r_[w,w]
    mdl=LogisticRegression(C=1.0,solver='liblinear',fit_intercept=False,max_iter=500,random_state=42)
    mdl.fit(both,yy,sample_weight=ww*len(ww)/ww.sum())
    return (pre,mdl),{'train_rows':len(t),'train_panels':int(t.panel_id.nunique()),
                      'train_doi':int(t.doi.nunique()),'n_observed_pairs':len(pairs),'n_identical_pairs':int(identical.sum()),
                      'training_rows_both_directions':len(yy)}

def pairwise_scores(model,test):
    """Return an uncalibrated *relative* win score and indistinguishable-feature indicators per panel."""
    pre,mdl=model
    test=test.reset_index(drop=True)
    z=pre.transform(features(test,MFG,MFG_NUM,MFG_CATS))
    if not sp.issparse(z):z=sp.csr_matrix(z)
    n=len(test);win=np.zeros(n); ties=np.zeros(n,int);pairs=[]
    if n<2:return win,ties,pairs
    for i in range(n):
        for j in range(i+1,n):
            dd=(z[i]-z[j]).tocsr();norm=float(np.sqrt(dd.multiply(dd).sum()))
            p=float(mdl.predict_proba(dd)[0,1])
            if norm<=FEATURE_TOL: ties[i]+=1;ties[j]+=1
            win[i]+=p; win[j]+=1-p
            pairs.append({'left':i,'right':j,'p_left_better':p,'identical_encoded_features':norm<=FEATURE_TOL})
    return win/(n-1),ties,pairs

def pairwise_concord(y,s):
    n=0;correct=0.;tie=0; wrong=0
    for i in range(len(y)):
        for j in range(i+1,len(y)):
            if abs(y[i]-y[j])<=SCORE_TOL:continue
            n+=1;delta=s[i]-s[j]
            if abs(delta)<=SCORE_TOL:correct+=.5;tie+=1
            elif (y[i]-y[j])*delta>0:correct+=1
            else:wrong+=1
    return val(correct/n if n else np.nan),n,tie,wrong

def panel_metric(panel, model, score):
    y=panel.target_value.to_numpy(float);s=panel[score].to_numpy(float)
    if np.ptp(y)>SCORE_TOL and np.ptp(s)>SCORE_TOL and len(y)>2:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');rho=val(spearmanr(y,s).statistic)
    else:rho=None
    top=np.flatnonzero(s>=np.max(s)-SCORE_TOL)
    concord,npair,ties,wrong=pairwise_concord(y,s)
    return {'panel_id':int(panel.panel_id.iloc[0]),'doi':str(panel.doi.iloc[0]),'n':len(panel),'model':model,
            'rho':rho,'pairwise_concordance':concord,'n_pairs':npair,'n_tied_pairs':ties,'n_wrong_pairs':wrong,
            'n_top':len(top),'regret_worst_mg_g':float(y.max()-y[top].min()),
            'top_ids':'|'.join(panel.iloc[top].observation_id.astype(str))}

def panel_comparable_features(panel):
    """Observed *input* information: no y or predictions accessed."""
    ranges={}
    for c in MFG_NUM:
        v=pd.to_numeric(panel[c],errors='coerce')
        ranges[c]={'n_reported':int(v.notna().sum()),'n_distinct':int(v.nunique(dropna=True)),
                   'spread':val(v.max()-v.min()) if v.notna().any() else None}
    return ranges

def bootstrap_doi(d, a='pairwise', b='mfg_forest',n=2500,seed=62031):
    # Paired DOI mean: undefined Spearman kept as NA; 0 is not a valid replacement.
    a0=d[d.model==a][['doi','panel_id','rho']].rename(columns={'rho':'a'})
    b0=d[d.model==b][['doi','panel_id','rho']].rename(columns={'rho':'b'})
    joined=a0.merge(b0,on=['doi','panel_id'],validate='one_to_one')
    valid=joined.dropna(subset=['a','b']).copy()
    if valid.empty:return {'n_panels_total':len(joined),'n_panels_valid':0,'n_doi_valid':0,'difference':None,'ci':None},joined
    rng=np.random.default_rng(seed);dois=valid.doi.unique();group={doi:valid[valid.doi==doi].copy() for doi in dois}
    if len(dois)<2:return {'n_panels_total':len(joined),'n_panels_valid':len(valid),'n_doi_valid':len(dois),
                           'difference':float((valid.a-valid.b).mean()),'ci':None},joined
    arr=[]
    for _ in range(n):
        selected=rng.choice(dois,size=len(dois),replace=True)
        sample=pd.concat([group[x] for x in selected],ignore_index=True)
        arr.append(float((sample.a-sample.b).mean()))
    return {'n_panels_total':len(joined),'n_panels_valid':len(valid),'n_doi_valid':len(dois),
            'difference':float((valid.a-valid.b).mean()),'ci':np.quantile(arr,[.025,.975]).tolist()},joined

def run(root):
    root=Path(root).resolve();dst=root/OUT;dst.mkdir(parents=True,exist_ok=True)
    source=root/SRC/'all188_doi_oof_predictions.csv'; audit=root/SRC/'panel_evidence_quality.csv'
    for p in (source,audit,root/OLD,root/PRE,root/NEW):
        if not p.is_file():raise FileNotFoundError(str(p))
    pred=pd.read_csv(source);quality=pd.read_csv(audit)
    base=pd.concat([pd.read_csv(root/PRE),pd.read_csv(root/NEW)],ignore_index=True,sort=False)
    if len(base)!=188 or len(pred)!=188 or base.observation_id.duplicated().any() or pred.observation_id.duplicated().any():
        raise ValueError('v60 input has changed, expected 188 unique rows')
    if base.observation_id.tolist()!=pred.observation_id.tolist():raise ValueError('v60 OOF and source row order mismatch')
    identified,panels=panel_inventory(base)
    if len(panels)!=15 or not np.array_equal(identified.panel_id.to_numpy(),pred.panel_id.to_numpy()):
        raise ValueError('Manufacturing panel mapping changed; stop for manual reconciliation')
    if quality.panel_id.nunique()!=15:raise ValueError('Evidence inventory mismatch')
    folds=list(GroupKFold(n_splits=5).split(base,groups=base.doi))
    score=np.full(len(base),np.nan);feature_ties=np.full(len(base),np.nan);pair_rows=[];fit_rows=[]
    for k,(tr,te) in enumerate(folds,1):
        train=base.iloc[tr];held=set(base.iloc[te].doi);assert not set(train.doi)&held
        if not (pred.iloc[te].fold.to_numpy()==k).all():
            raise ValueError('v60 OOF fold indices differ from DOI folds; cannot compare estimators')
        model,meta=train_pairwise(train);meta['fold']=k;meta['test_DOI']='|'.join(sorted(held));fit_rows.append(meta)
        if model is None:raise RuntimeError('Insufficient pairwise data in grouped fold: '+str(meta))
        for panel_id,g in pred[(pred.panel_id>=0)&(pred.doi.isin(held))].groupby('panel_id',sort=True):
            g=g.sort_values('observation_id');ix=g.index.to_numpy()
            s,ties,pairs=pairwise_scores(model,base.iloc[ix])
            score[ix]=s;feature_ties[ix]=ties
            for p in pairs:
                pair_rows.append({'fold':k,'doi':str(g.doi.iloc[0]),'panel_id':int(panel_id),
                                  'left_id':str(g.iloc[p['left']].observation_id),'right_id':str(g.iloc[p['right']].observation_id),
                                  'p_left_better':p['p_left_better'],
                                  'identical_encoded_features':p['identical_encoded_features']})
    pp=pred.copy();pp['pairwise_relwin_score_NOT_QE']=score;pp['pairwise_indistinguishable_neighbors']=feature_ties
    if not np.isfinite(pp.loc[pp.panel_id>=0,'pairwise_relwin_score_NOT_QE']).all():raise RuntimeError('Missing pairwise OOF')
    pp.to_csv(dst/'panel_oof_predictions.csv',index=False)
    pr=pd.DataFrame(pair_rows);pr.to_csv(dst/'pairwise_feature_ambiguity.csv',index=False)
    out=[];features_audit=[]
    quality=quality.set_index('panel_id')
    for pid,panel in pp[pp.panel_id>=0].groupby('panel_id'):
        panel=panel.sort_values('observation_id')
        pair_score='pairwise_relwin_score_NOT_QE'
        for model,col in [('pairwise',pair_score),('mfg_forest','188__mfg_forest'),
                          ('B_clipped','188__B_doi_material'),('B_raw_invalid_as_qe','188__B_doi_material__raw')]:
            out.append(panel_metric(panel,model,col))
        ppairs=pr[pr.panel_id==pid]
        q=quality.loc[pid]
        features_audit.append({'doi':str(panel.doi.iloc[0]),'panel_id':int(pid),'n':len(panel),
                               'full_adsorption_context':bool(q.full_conditions_reported),
                               'strict_qe':bool(q.strict_experimental_qe_confirmed_in_vNext10),
                               'n_ambiguous_pairwise_features':int(ppairs.identical_encoded_features.sum()),
                               'total_pairs':len(ppairs)})
    metrics=pd.DataFrame(out);metrics.to_csv(dst/'panel_rank_metrics.csv',index=False)
    fa=pd.DataFrame(features_audit);fa.to_csv(dst/'panel_feature_diagnostics.csv',index=False)
    # Feature spread from original aligned observations, not from the reduced prediction frame.
    spreads=[]
    for pid,g in identified[identified.panel_id>=0].groupby('panel_id'):
        rec={'panel_id':int(pid),'doi':str(g.doi.iloc[0])}
        for c,desc in panel_comparable_features(g).items():rec[f'{c}__distinct']=desc['n_distinct'];rec[f'{c}__reported']=desc['n_reported'];rec[f'{c}__spread']=desc['spread']
        spreads.append(rec)
    spread_table=pd.DataFrame(spreads);spread_table.to_csv(dst/'manufacturing_feature_spreads.csv',index=False)
    variable_panels={c:int((spread_table[f'{c}__distinct']>1).sum()) for c in MFG_NUM}
    categorical_variable_panels={c:int((identified.loc[identified.panel_id>=0].groupby('panel_id')[c].nunique(dropna=True)>1).sum()) for c in MFG_CATS}
    ci={};paired_dois=[]
    for comparator in ('mfg_forest','B_clipped','B_raw_invalid_as_qe'):
        c,rows=bootstrap_doi(metrics,'pairwise',comparator)
        ci[comparator]=c
        rows['comparator']=comparator;paired_dois.append(rows)
    pd.concat(paired_dois,ignore_index=True).to_csv(dst/'paired_panel_rhos.csv',index=False)
    summary=[]
    for m,g in metrics.groupby('model'):
        valid=g.dropna(subset=['rho']);summary.append({'model':m,'n_panel':len(g),'n_defined_rho':len(valid),'n_doi_valid':valid.doi.nunique(),
            'mean_rho_valid_panels':val(valid.rho.mean()),'mean_pairwise_concordance':val(g.pairwise_concordance.mean()),
            'unique_top_count':int((g.n_top==1).sum()),'mean_worst_regret_mg_g':val(g.regret_worst_mg_g.mean()),
            'mean_predicted_tie_pairs':val(g.n_tied_pairs.mean())})
    summary=pd.DataFrame(summary).sort_values('model');summary.to_csv(dst/'model_comparison.csv',index=False)
    pd.DataFrame(fit_rows).to_csv(dst/'fold_training_audit.csv',index=False)
    raw=pred['188__B_doi_material__raw'].to_numpy(float);cap=pred.C0_mg_L.to_numpy(float)/pred.dose_g_L.to_numpy(float)
    clipped=pred['188__B_doi_material'].to_numpy(float)
    if not np.allclose(clipped,np.clip(raw,0,cap),atol=2e-5,rtol=1e-8):raise RuntimeError('Clipping audit failed')
    # Save final fit only in separate research namespace. Full-training model is not OOF and never enters evaluation.
    final_model,full_meta=train_pairwise(base)
    if final_model is None:raise RuntimeError('Could not train a full research pairwise model')
    artifact=root/'deployment/research_v62_NOT_ACTIVE/pairwise_logit_188_RELATIVE_SCORE_NOT_QE.joblib'
    artifact.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump({'preprocessor':final_model[0],'estimator':final_model[1],
                 'features':list(MFG),'output':'within-candidate-panel relative win score, NOT qe',
                 'training_metadata':full_meta},artifact,compress=3)
    stats={'research_artifact':str(artifact.relative_to(root)),'research_artifact_sha256':sha(artifact),
           'full_train_metadata':full_meta,
           'model_role':'RESEARCH_ONLY_DOI_GROUPED_POSTHOC','panel_count':len(panels),'doi_count':int(panels.doi.nunique()),
           'rows_in_manufacturing_panels':int((pred.panel_id>=0).sum()),
           'numeric_feature_variation_panels':variable_panels,
           'categorical_feature_variation_panels':categorical_variable_panels,
           'n_ambiguous_input_pairs':int(pr.identical_encoded_features.sum()),
           'n_pairs_total':len(pr),'raw_upper_cap_violation_all188':int((raw>cap+1e-8).sum()),
           'clipping_at_cap_all188':int(np.isclose(clipped,cap,atol=1e-8,rtol=0).sum()),
           'compare_pairwise':ci,'libraries':{'python':platform.python_version(),'sklearn':sklearn.__version__},
           'inputs':{'v60_oof_sha256':sha(source),'v60_panel_evidence_sha256':sha(audit),
                     'v55_pre152_sha256':sha(root/PRE),'v57_new36_sha256':sha(root/NEW)}}
    save_json(dst/'manifest_v62.json',stats)
    def f3(v): return f'{v:.3f}' if v is not None and pd.notna(v) else '정의 불가'
    lines=['# v62 로컬 DOI별 제조조건 모델 검증 결과', '',
           '이 보고서는 설치된 컴퓨터의 v60 OOF와 누적 학습 데이터를 이용해 새 Pairwise Logit 모델을 DOI GroupKFold로 학습·평가한 **사후 연구용 결과**다.',
           '절대 qe 예측과 상대 우열 점수를 혼합하지 않는다. 신규 모델은 학술·실용 배포 검증을 통과하지 않았다.', '',
           f'- 전체 제조조건 비교계열: {len(panels)}개 / {int(panels.doi.nunique())} DOI',
           f'- 계열 관측: {int((pred.panel_id>=0).sum())}행; 후보 쌍 {len(pr)}개 (서로 독립인 관측 수가 아님)',
           f'- 모델 입력으로 구별할 수 없는 후보 쌍: {int(pr.identical_encoded_features.sum())}개',
           f'- 전체 188행에서 clipping 전 물질수지 상한 초과: {stats["raw_upper_cap_violation_all188"]}행',
           '', '## 모델별 제조조건 순위 검증', '',
           '|모델|Spearman 유효 계열|평균 Spearman ρ (유효 계열)|Pairwise 일치율|고유 1위 계열|평균 최악 Regret (mg/g)|',
           '|---|---:|---:|---:|---:|---:|']
    labels={'pairwise':'신규 Pairwise Logistic (상대 점수)','mfg_forest':'기존 분리 RF (상대 점수)',
            'B_clipped':'기존 Model B clipped qe','B_raw_invalid_as_qe':'기존 Model B raw (물리 위반 가능, 진단만)'}
    for _,r in summary.iterrows():
        lines.append(f'|{labels[r["model"]]}|{int(r["n_defined_rho"])} / {int(r["n_panel"])}|{f3(r["mean_rho_valid_panels"])}|{f3(r["mean_pairwise_concordance"])}|{int(r["unique_top_count"])}|{f3(r["mean_worst_regret_mg_g"])}|')
    lines+=['','## DOI 재표집 비교 (순위 Pairwise − 비교모델)', '']
    for model,c in ci.items():
        lo,hi=c['ci'] if c['ci'] is not None else (None,None)
        lines.append(f'- {labels[model]}: Δρ={f3(c["difference"])}; 95% DOI bootstrap 구간 [{f3(lo)}, {f3(hi)}]; 유효 {c["n_panels_valid"]}계열/{c["n_doi_valid"]}DOI')
    lines+=['','## 데이터 완전성 및 예측 불가능성',
            '',f'- 흡착조건 완전 보고 계열: {int(fa.full_adsorption_context.sum())}개',
            f'- 엄격 평형 qe 계열: {int(fa.strict_qe.sum())}개 (별도 DOI 수가 충분하지 않음)',
            '- 비교계열 내부의 제조조건 수치 변동 수 (계열 15개 기준): '+', '.join(f'{c}={n}' for c,n in variable_panels.items()),
            '- 원료·개질·분위기 등의 범주변수 계열 내부 변동 수: '+', '.join(f'{c}={n}' for c,n in categorical_variable_panels.items()),
            '- 제조조건이 관측상 구별되는 것과 새로운 DOI에서 정확히 우열을 예측하는 것은 별개다.',
            '- 연구용 원시 qe는 물질수지를 초과할 수 있으므로 물리적으로 가능한 흡착량으로 사용하지 않는다.',
            '- 단독 1위 수는 **정답률이 아니며**, 순위 예측의 정확성은 Spearman, pairwise, regret으로 별도 확인한다.',
            '', '## 해석 제한',
            '', '- 전체 15개 중 11개는 흡착조건 일부 미보고. 완전 보고 4개 중 엄격 평형 2개는 같은 DOI에서 나왔다.',
            '- Pairwise 쌍은 동일 제조 비교계열 안에서 상호의존적이므로 95쌍을 독립 표본 수로 보고하지 않는다.',
            '- 본 모델 구조 및 분석은 이전 DOI와 결과를 본 뒤 제안된 사후 실험이므로, 전향·완전 독립 검증이 아니다.',
            '- 다른 로컬 환경에서 재학습된 v60 RF/B OOF를 넣으면 해당 PC에서 생성한 v62 보고서를 기준으로 판단해야 한다.',
            '- 배포 모델을 수정하지 않고 `deployment/research_v62_NOT_ACTIVE/`에 별도 저장했다.', '']
    (dst/'LOCAL_RESULTS_KO.md').write_text('\n'.join(lines),encoding='utf-8')
    print(summary.to_string(index=False));print('DIAGNOSTIC:',json.dumps({k:stats[k] for k in ['panel_count','n_ambiguous_input_pairs','raw_upper_cap_violation_all188']},ensure_ascii=False))
    print('PAIRWISE_VS_FOREST:',ci['mfg_forest'])
    return stats

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');a=p.parse_args();run(a.root)
