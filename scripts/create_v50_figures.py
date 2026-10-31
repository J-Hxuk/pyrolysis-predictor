"""Export original, non-stylized comparison graphics from v50 metrics."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
OUT=Path('reports/v50_model_utility')

def main():
    o=OUT
    g=pd.read_csv(o/'global_model_comparison.csv')
    chosen=['A_raw','B_raw','A_clipped','B_clipped','A_bounded_fraction','B_bounded_fraction']
    p=g.set_index('model').loc[chosen]
    fig,ax=plt.subplots(figsize=(10,4.6));ax.bar(range(len(p)),p.mae)
    ax.set_xticks(range(len(p)));ax.set_xticklabels([x.replace('_','\n') for x in chosen],fontsize=8)
    ax.set_ylabel('MAE (mg/g)');ax.set_title('Stage 2: fixed DOI-group 5-fold (140 observations; 32 DOI)')
    ax.set_ylim(0,max(p.mae)*1.15)
    for i,y in enumerate(p.mae):ax.text(i,y+.35,f'{y:.2f}',ha='center',fontsize=9)
    fig.tight_layout();fig.savefig(o/'v50_overall_mae.png',dpi=140);plt.close(fig)
    s=pd.read_csv(o/'per_pollutant_comparison.csv')
    q=s[(s.pollutant=='Cu(II)')&s.model.isin(['A_clipped','B_clipped','A_bounded_fraction','B_bounded_fraction'])].set_index('model')
    selected=['A_clipped','B_clipped','A_bounded_fraction','B_bounded_fraction']
    fig,ax=plt.subplots(figsize=(8,4.2));ax.bar(range(4),q.loc[selected,'mae'])
    ax.set_xticks(range(4));ax.set_xticklabels([x.replace('_','\n') for x in selected],fontsize=8)
    ax.set_ylabel('MAE (mg/g)');ax.set_title('Cu(II): exploratory 31 observations / 9 DOI')
    for i,y in enumerate(q.loc[selected,'mae']):ax.text(i,y+.25,f'{y:.2f}',ha='center',fontsize=9)
    fig.tight_layout();fig.savefig(o/'v50_cu_mae.png',dpi=140);plt.close(fig)

if __name__=='__main__':main()
