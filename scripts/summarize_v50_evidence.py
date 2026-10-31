"""Summarize precomputed v50 DOI OOF experiments, without re-fitting or deployment."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

OUT='reports/v50_model_utility'
CASES=[('Cu(II)','A_clipped','A_bounded_fraction'),
       ('Cd(II)','A_clipped','B_clipped'),
       ('Cu(II)','B_clipped','CdCu_local_B_clipped')]

def run(root:Path):
    out=Path(root)/OUT
    p=pd.read_csv(out/'doi_oof_predictions.csv')
    rows=[];dois=[]
    rng=np.random.default_rng(42)
    for pol,reference,candidate in CASES:
        df=p[p.pollutant==pol].copy()
        df['change_abs_error']=(df.observed_qe-df[candidate]).abs()-(df.observed_qe-df[reference]).abs()
        groups=[(doi,g) for doi,g in df.groupby('doi')]
        diffs=[g.change_abs_error.to_numpy(float) for _,g in groups]
        bs=[]
        for _ in range(3000):
            draw=rng.integers(len(groups),size=len(groups))
            bs.append(float(np.concatenate([diffs[i] for i in draw]).mean()))
        for doi,g in groups:
            dois.append(dict(pollutant=pol,reference=reference,candidate=candidate,doi=doi,
                       n=len(g),mae_delta_candidate_minus_reference=float(g.change_abs_error.mean())))
        rows.append(dict(pollutant=pol,reference=reference,candidate=candidate,n=len(df),doi=len(groups),
                  reference_mae=float((df.observed_qe-df[reference]).abs().mean()),
                  candidate_mae=float((df.observed_qe-df[candidate]).abs().mean()),
                  delta_mae=float(df.change_abs_error.mean()),improved_doi=sum(x.mean()<0 for x in diffs),
                  ci2_5=float(np.quantile(bs,.025)),ci97_5=float(np.quantile(bs,.975)),
                  bootstrap_unit='DOI',bootstrap_replications=3000,
                  uncertainty_note='Exploratory reused-DOI CV; intervals are descriptive, not prospectively validated'))
    pd.DataFrame(dois).to_csv(out/'per_doi_candidate_comparison.csv',index=False)
    summary=pd.DataFrame(rows)
    summary.to_csv(out/'doi_cluster_bootstrap.csv',index=False)
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path.cwd())
    print(run(parser.parse_args().root).to_string(index=False))
