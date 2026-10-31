"""Reproduce three figures from the Stage 1 composition ablation report (optional)."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    folder = Path(__file__).resolve().parents[1] / 'reports/stage1_composition_v18'
    metrics = pd.read_csv(folder/'model_comparison.csv')
    predictions = pd.read_csv(folder/'oof_predictions.csv')
    sub = metrics[metrics.algorithm=='RandomForest'].set_index('feature_set')
    names=['process_only','process_elemental','process_proximate','process_all_composition']
    labels=['Process','+ Elemental','+ Proximate','+ All composition']
    fig,ax=plt.subplots(figsize=(9,4.2))
    ax.barh(labels,sub.loc[names,'mae_m2_g'].to_numpy())
    ax.set_xlabel('Leave-one-DOI-out MAE (m²/g)')
    ax.invert_yaxis()
    ax.grid(axis='x',alpha=.2)
    for i,v in enumerate(sub.loc[names,'mae_m2_g']):ax.text(v+1,i,f'{v:.1f}',va='center')
    fig.tight_layout()
    fig.savefig(folder/'fig_bet_mae_by_features.png',dpi=170)
    plt.close(fig)

    r=predictions[predictions.algorithm=='RandomForest'].pivot_table(index=['row_index','doi'],columns='feature_set',values='abs_error_BET_m2_g').reset_index()
    per_doi=r.assign(improvement=r.process_only-r.process_all_composition).groupby('doi').improvement.mean().sort_values()
    fig,ax=plt.subplots(figsize=(10,6.2))
    ax.barh(np.arange(len(per_doi)),per_doi.values)
    ax.set_yticks(np.arange(len(per_doi)),per_doi.index,fontsize=7.5)
    ax.axvline(0,linewidth=1)
    ax.set_xlabel('MAE improvement within DOI (m²/g; positive = better)')
    fig.tight_layout()
    fig.savefig(folder/'fig_doi_improvement.png',dpi=170)
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(10,4.2),sharex=True,sharey=True)
    for ax,feature in zip(axes,['process_only','process_all_composition']):
        v=predictions[(predictions.algorithm=='RandomForest')&(predictions.feature_set==feature)]
        a=v.actual_BET_m2_g.to_numpy();p=v.predicted_BET_m2_g.to_numpy()
        ax.scatter(a,p,s=10,alpha=.65)
        hi=max(a.max(),p.max());ax.plot([0,hi],[0,hi],linestyle='--')
        ax.set_title('Process only' if feature=='process_only' else 'Process + composition')
        ax.set_xlabel('Measured BET (m²/g)')
    axes[0].set_ylabel('OOF predicted BET (m²/g)')
    fig.tight_layout()
    fig.savefig(folder/'fig_bet_oof_comparison.png',dpi=170)
    plt.close(fig)

if __name__=='__main__':main()
