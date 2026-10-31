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

def _paired_rank_wide(x, score_a, score_b):
    """Keep every observed panel even if Spearman rho is undefined.

    A constant prediction has undefined rho, *not* zero correlation. Such
    a panel is not silently dropped from the reported inventory.
    """
    selected = x[x.model.isin([score_a, score_b])]
    if selected.empty or not {score_a, score_b}.issubset(set(selected.model)):
        raise ValueError('Both ranking model rows must be present for paired comparison')
    keys = ['doi', 'panel_id', 'model']
    if selected.duplicated(keys).any():
        raise ValueError('Duplicate DOI-panel-model ranking records')
    wide = selected.pivot(index=['doi','panel_id'],columns='model',values='rho')
    return wide.reindex(columns=[score_a,score_b])


def paired_bootstrap(x,score_a='mfg_forest_RELATIVE_SCORE_NOT_QE',score_b='B_doi_material',n_iter=5000):
    wide = _paired_rank_wide(x,score_a,score_b)
    valid = wide.dropna(subset=[score_a,score_b])
    invalid = wide.loc[wide[[score_a,score_b]].isna().any(axis=1)]
    out = dict(n_panels_inventory=int(len(wide)),
               n_doi_inventory=int(wide.index.get_level_values('doi').nunique()),
               n_panels=int(len(valid)),
               n_doi=int(valid.index.get_level_values('doi').nunique()),
               n_panels_undefined_rho=int(len(invalid)),
               undefined_panel_keys=' | '.join(f'{doi}:{pid}' for doi,pid in invalid.index))
    if valid.empty:
        return dict(**out, mean_delta_rho=None, ci95_low=None, ci95_high=None,
                    reason='NO_VALID_PAIRED_SPEARMAN')
    delta=valid[score_a]-valid[score_b]
    out['mean_delta_rho']=float(delta.mean())
    doilist=list(dict.fromkeys(valid.index.get_level_values('doi')))
    if len(doilist)<3:
        return dict(**out,ci95_low=None,ci95_high=None,
                    reason='TOO_FEW_INDEPENDENT_DOIS_FOR_CI')
    diffs=[delta.xs(k,level='doi').to_numpy(float) for k in doilist]
    rng=np.random.default_rng(SEED); vals=[]
    for _ in range(n_iter):
        samples=rng.integers(len(doilist),size=len(doilist))
        vals.append(np.concatenate([diffs[i] for i in samples]).mean())
    out.update(ci95_low=float(np.quantile(vals,.025)),ci95_high=float(np.quantile(vals,.975)),reason='')
    return out


def rank_coverage_diagnostics(scores, pred, score_a='mfg_forest_RELATIVE_SCORE_NOT_QE', score_b='B_doi_material'):
    """Audit each model/panel, including constant-prediction causes of undefined rho."""
    rows=[]
    for sc in sorted(scores.scenario.dropna().unique()):
        z=scores[(scores.scenario==sc)&(scores.panel_type=='manufacturing')]
        wide=_paired_rank_wide(z,score_a,score_b)
        for (doi,pid), r in wide.iterrows():
            obs=pred[(pred.doi==doi)&(pred.panel_id==pid)]
            if obs.empty:
                raise ValueError(f'Panel not present in predictions: {doi} {pid}')
            a=obs[f'{int(sc)}__mfg_forest'].to_numpy(float)
            b=obs[f'{int(sc)}__B_doi_material'].to_numpy(float)
            y=obs.target_value.to_numpy(float)
            const_y=bool(np.ptp(y)<1e-10)
            const_a=bool(np.ptp(a)<1e-10)
            const_b=bool(np.ptp(b)<1e-10)
            valid=bool(np.isfinite(r[score_a]) and np.isfinite(r[score_b]))
            reasons=[]
            if const_y:reasons.append('CONSTANT_OBSERVED_VALUES')
            if const_a:reasons.append('CONSTANT_RANKER_PREDICTIONS')
            if const_b:reasons.append('CONSTANT_BASELINE_PREDICTIONS')
            if not valid and not reasons:reasons.append('UNDEFINED_SPEARMAN_OTHER')
            rows.append(dict(scenario=int(sc),doi=doi,panel_id=int(pid),n_observations=len(obs),
                             rho_ranker=r[score_a],rho_baseline=r[score_b],
                             eligible_for_paired_rho=valid,
                             observed_constant=const_y,ranker_constant=const_a,
                             baseline_constant=const_b,
                             exclusion_reason='|'.join(reasons)))
    return pd.DataFrame(rows)

def main(root):
    root=Path(root).resolve();dst=root/REPORT
    d=pd.concat([pd.read_csv(root/PRE),pd.read_csv(root/NEW)],ignore_index=True,sort=False)
    pp,inventory=panel_inventory(d);cc,c_inventory=concentration_inventory(d)
    q=quality(pp,inventory);q.to_csv(dst/'panel_evidence_quality.csv',index=False)
    pred=pd.read_csv(dst/'all188_doi_oof_predictions.csv')
    score=pd.read_csv(dst/'per_panel_rank_results.csv')
    coverage=rank_coverage_diagnostics(score,pred)
    coverage.to_csv(dst/'panel_rank_coverage.csv',index=False)
    fullstats=[]
    for sc in ('152','188'):
        s=score[(score.scenario==int(sc))&(score.panel_type=='manufacturing')]
        for label,ids in [('all',set(q.panel_id)),('full_ads_conditions',set(q.loc[q.full_conditions_reported,'panel_id'])),
                          ('strict_experimental_qe',set(q.loc[q.strict_experimental_qe_confirmed_in_vNext10,'panel_id']))]:
            z=s[s.panel_id.isin(ids)]
            boot=paired_bootstrap(z)
            fullstats.append({'scenario':sc,'subset':label,**boot})
    x=pd.DataFrame(fullstats);x.to_csv(dst/'paired_doi_bootstrap.csv',index=False)
    # Local result note supersedes any historical PDF/Markdown benchmark claim.
    current=x[(x.scenario=='188')&(x.subset=='all')]
    note=['# v60.1 로컬 제조조건 순위 검증 기록',
          '',
          '이 기록은 설치된 PC의 `per_panel_rank_results.csv`로 재계산한 값이다. 이전 제작 환경의 15개 유효 계열 통계와 혼용하지 않는다.',
          '',
          f'- 전체 제조 비교계열: {len(q)}개 / {int(q.doi.nunique())} DOI',
          '- Spearman ρ가 정의되지 않는 비교계열은 0으로 대체하지 않고 CI에서만 제외한다.',
          '- 전체 결과: `panel_rank_coverage.csv`',
          '- DOI bootstrap: `paired_doi_bootstrap.csv`',
          '']
    if not current.empty:
        row=current.iloc[0]
        note.extend([f'- 188행 모델의 양측 Spearman 유효 계열: {int(row.n_panels)} / {int(row.n_panels_inventory)}',
                     f'- 양측 유효 DOI: {int(row.n_doi)} / {int(row.n_doi_inventory)}',
                     f'- 평가 제외 계열: {row.undefined_panel_keys if row.undefined_panel_keys else "없음"}',
                     f'- 평균 ρ 차이(RF−B): {row.mean_delta_rho if pd.notna(row.mean_delta_rho) else "정의 불가"}',
                     f'- DOI bootstrap 95% 구간: [{row.ci95_low}, {row.ci95_high}]',
                     ''])
    nbad=coverage[(coverage.scenario==188)&(~coverage.eligible_for_paired_rho)]
    if len(nbad):
        note.append('## 188행 모델의 제외 원인')
        for r in nbad.itertuples(index=False):
            note.append(f'- DOI `{r.doi}` / panel {r.panel_id}: {r.exclusion_reason}; RF rho={r.rho_ranker}, B rho={r.rho_baseline}')
        note.append('')
    note.append('이 결과는 사후 DOI 분할 평가이다. 데이터 추가·모델 선택에 이미 사용된 DOI를 미래의 독립 검증으로 간주하지 않는다.')
    (dst/'V60_1_LOCAL_RANK_AUDIT_KO.md').write_text('\n'.join(note)+'\n',encoding='utf-8')
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
