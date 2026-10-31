"""Analyze v60 empirical panels and source-grade. No model training or active files modified."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd, numpy as np
from scripts.run_v60_separate_effects import REPORT,PRE,NEW,OLD,panel_inventory,concentration_inventory

SEED=20261031

def quality(df,panels):
    out=[]
    for r in panels.itertuples(index=False):
        s=df[df.panel_id==r.panel_id]
        quals=' | '.join(sorted(set(s.target_qualification.dropna().astype(str)))) if 'target_qualification' in s else ''
        statuses=' | '.join(sorted(set(s.source_target_status.dropna().astype(str)))) if 'source_target_status' in s else ''
        tnames=' | '.join(sorted(set(s.target_name.dropna().astype(str))))
        strict=bool(s.target_qualification.astype(str).str.contains('TABLE3_EXPERIMENTAL_EQUILIBRIUM_QE_EXP',regex=False).all())
        out.append({'panel_id':r.panel_id,'doi':r.doi,'n':r.n,'full_conditions_reported':bool(r.n_missing_ads_context==0),
                    'strict_experimental_qe_confirmed_in_vNext10':strict,'target_qualification_labels':quals,
                    'source_target_status':statuses,'target_name_labels':tnames,
                    'reason': ('COMPLETE_EXACT_CONTEXT' if r.n_missing_ads_context==0 else 'MISSING_ADSORPTION_CONTEXT; assumed same because missing values match')})
    return pd.DataFrame(out)

def paired_bootstrap(x,score_a='mfg_forest_RELATIVE_SCORE_NOT_QE',score_b='B_doi_material',n_iter=5000):
    xx=x[x.model.isin([score_a,score_b])].pivot(index=['doi','panel_id'],columns='model',values='rho').dropna()
    if xx.empty: return {'n':0}
    dd=xx[score_a]-xx[score_b]
    doilist=list(dict.fromkeys(xx.index.get_level_values('doi')))
    diffs=[dd.xs(k,level='doi').to_numpy(float) for k in doilist]
    if len(doilist)<3:
        return dict(n_panels=len(dd),n_doi=len(doilist),mean_delta_rho=float(dd.mean()),
                    ci95_low=None,ci95_high=None,reason='TOO_FEW_INDEPENDENT_DOIS_FOR_CI')
    rng=np.random.default_rng(SEED); vals=[]
    for _ in range(n_iter):
        samples=rng.integers(len(doilist),size=len(doilist))
        yy=np.concatenate([diffs[i] for i in samples]);vals.append(yy.mean())
    return dict(n_panels=len(dd),n_doi=len(doilist),mean_delta_rho=float(dd.mean()),
                ci95_low=float(np.quantile(vals,.025)),ci95_high=float(np.quantile(vals,.975)))

def main(root):
    root=Path(root).resolve();dst=root/REPORT
    d=pd.concat([pd.read_csv(root/PRE),pd.read_csv(root/NEW)],ignore_index=True,sort=False)
    pp,inventory=panel_inventory(d);cc,c_inventory=concentration_inventory(d)
    q=quality(pp,inventory);q.to_csv(dst/'panel_evidence_quality.csv',index=False)
    pred=pd.read_csv(dst/'all188_doi_oof_predictions.csv')
    score=pd.read_csv(dst/'per_panel_rank_results.csv')
    fullstats=[]
    for sc in ('152','188'):
        s=score[(score.scenario==int(sc))&(score.panel_type=='manufacturing')]
        for label,ids in [('all',set(q.panel_id)),('full_ads_conditions',set(q.loc[q.full_conditions_reported,'panel_id'])),
                          ('strict_experimental_qe',set(q.loc[q.strict_experimental_qe_confirmed_in_vNext10,'panel_id']))]:
            z=s[s.panel_id.isin(ids)]
            boot=paired_bootstrap(z)
            fullstats.append({'scenario':sc,'subset':label,**boot})
    x=pd.DataFrame(fullstats);x.to_csv(dst/'paired_doi_bootstrap.csv',index=False)
    qsum={'manufacturing_panels':len(q),'complete_adsorption_panels':int(q.full_conditions_reported.sum()),
          'strict_apple_qe_panels':int(q.strict_experimental_qe_confirmed_in_vNext10.sum()),
          'concentration_series':len(c_inventory), 'concentration_doi':int(c_inventory.doi.nunique()),
          'paired_DOI_bootstrap':'two-sided percentile with DOI resampling, posthoc study'}
    (dst/'source_grade_summary.json').write_text(json.dumps(qsum,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Evidence:',qsum)
    print(x.round(3).to_string(index=False))
    return x

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='.');a=ap.parse_args();main(a.root)
