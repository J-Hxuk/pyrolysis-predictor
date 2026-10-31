"""Plots for completed v39 report; requires precomputed CSVs."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def plot(root):
    p=Path(root)/'reports/v39_quality_and_positivity'
    s1=pd.read_csv(p/'stage1_model_comparison.csv');b=s1[s1.target=='BET']
    fig,ax=plt.subplots(figsize=(8,4.6))
    ax.bar(b.variant,b.MAE)
    ax.set_ylabel('DOI GroupKFold MAE (m²/g)')
    ax.set_title('Stage 1: BET — full-range prediction error')
    ax.set_ylim(0,max(b.MAE)*1.25)
    for idx,v in enumerate(b.MAE):ax.text(idx,v+5,f'{v:.1f}',ha='center',fontsize=9)
    fig.tight_layout();fig.savefig(p/'fig_stage1_BET_models.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.6))
    ax.bar(b.variant,b.MAE_low)
    ax.set_ylabel('MAE on measured BET ≤25 m²/g')
    ax.set_title('Stage 1: low-BET region — 60 observations / 17 DOI')
    ax.set_ylim(0,max(b.MAE_low)*1.25)
    for idx,v in enumerate(b.MAE_low):ax.text(idx,v+3,f'{v:.1f}',ha='center',fontsize=9)
    fig.tight_layout();fig.savefig(p/'fig_stage1_BET_low.png',dpi=180);plt.close(fig)
    s2=pd.read_csv(p/'stage2_quality_sensitivity.csv')
    fig,ax=plt.subplots(figsize=(8.4,4.6))
    x=list(range(3));a=s2[s2.model=='A'];b=s2[s2.model=='B']
    ax.plot(x,a.MAE,marker='o',label='Model A (adsorption inputs)')
    ax.plot(x,b.MAE,marker='o',label='Model B (+ manufacturing inputs)')
    ax.set_xticks(x,['Core 98','+15 temperature series','+12 coffee conditional'])
    ax.set_ylabel('MAE on same Core 98 held-out observations (mg/g)')
    ax.set_title('Stage 2: training-data quality sensitivity')
    ax.legend();fig.tight_layout();fig.savefig(p/'fig_stage2_quality_effect.png',dpi=180);plt.close(fig)

if __name__=='__main__':plot(Path(__file__).resolve().parents[1])
