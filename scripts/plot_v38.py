"""Render reproducible v38 research figures; no arbitrary data smoothing."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def render(root=Path('.')):
    root=Path(root)
    rep=root/'reports/v38_new_dois'
    df=pd.read_csv(rep/'paired_old_DOI_evaluation.csv')
    s2=df[df.stage=='Stage2']
    fig,ax=plt.subplots(figsize=(7.2,4.2))
    x=range(len(s2));w=.34
    ax.bar([i-w/2 for i in x],s2.MAE_old,width=w,label='Original 113 training')
    ax.bar([i+w/2 for i in x],s2.MAE_new,width=w,label='Full 125 research training')
    ax.set_xticks(list(x),['Model '+x for x in s2.target])
    ax.set_ylabel('Same original DOI test MAE (mg/g)')
    ax.set_title('Paired DOI CV: effect of adding conditional coffee observations')
    ax.legend(loc='best');fig.tight_layout();fig.savefig(rep/'fig_stage2_paired_MAE.png',dpi=160);plt.close(fig)
    pred=pd.read_csv(rep/'paired_predictions_and_new_DOI.csv')
    focus=pred[(pred.stage=='Stage1')&(pred.target=='BET')&(pred.evaluation_role=='NEW_DOI_DIAGNOSTIC_HELD_FROM_TRAIN')]
    fig,ax=plt.subplots(figsize=(7.2,4.5))
    for doi,g in focus.groupby('doi'):
        ax.scatter(g.actual,g.old_oof,label=doi.rsplit('/',1)[-1],alpha=.8)
    ax.axhline(0,linestyle='--',linewidth=1)
    ax.set_xlabel('Measured BET (m²/g)');ax.set_ylabel('Base Ridge prediction (m²/g)')
    ax.set_title('New DOI challenge: physically negative BET predictions')
    ax.legend();fig.tight_layout();fig.savefig(rep/'fig_new_DOI_BET_failure.png',dpi=160);plt.close(fig)
if __name__=='__main__':render()
