"""Render fixed comparison summary and non-overstating charts from v23 CSV/JSON."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/phase8_v23'

def main(folder=OUT):
    folder=Path(folder)
    data=json.loads((folder/'summary.json').read_text(encoding='utf-8'))['models']
    stages={'stage1':[('Stage1_BET','BET (m²/g)'),('Stage1_TPV','TPV (cm³/g)')],
            'stage2':[('Stage2_A_XGBoost','Model A (mg/g)'),('Stage2_B_Ridge','Model B (mg/g)')]}
    for stage,blocks in stages.items():
        for key,unit in blocks:
            vals=[data[key]['paired_original']['mae'],data[key]['paired_expanded']['mae']]
            fig,ax=plt.subplots(figsize=(7,3))
            ax.barh(['Old training','Phase 8 expanded'],vals)
            ax.invert_yaxis()
            ax.set_xlabel('DOI-held-out MAE, old evaluation rows only')
            ax.set_title(f'{unit} | fixed research model')
            ax.set_xlim(0, max(vals)*1.3)
            for idx,v in enumerate(vals):
                ax.text(v+max(vals)*.015,idx,f'{v:.4g}',va='center')
            fig.tight_layout()
            fig.savefig(folder/f'fig_{key}.png',dpi=160)
            plt.close(fig)
    for key in stages['stage1']+stages['stage2']:
        k=key[0]
        if k.startswith('Stage1'):
            target=k.split('_')[1].lower()
            d=pd.read_csv(folder/f'stage1_{target}_paired_old_DOI.csv')
        else:
            target=k.split('_')[1]
            d=pd.read_csv(folder/f'stage2_{target}_paired_old_DOI.csv')
        bydoi=d.groupby('doi',as_index=False).agg(n=('err_original','size'),
              mae_before=('err_original','mean'),mae_after=('err_expanded','mean'))
        bydoi['delta_mae']=bydoi.mae_after-bydoi.mae_before
        bydoi.to_csv(folder/f'{k}_per_DOI_delta.csv',index=False)
    return data

if __name__=='__main__':
    main()
