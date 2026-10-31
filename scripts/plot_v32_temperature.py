"""Read-only charts from v32 CSV and existing v30 source evidence."""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'reports/v32_temperature_identifiability'
EVIDENCE=ROOT/'reports/v30_graph_evidence/rice_figure9_qt_1440_APPROX_NOT_TRAINED.csv'


def render(root=ROOT):
    c=pd.read_csv(root/'reports/v32_temperature_identifiability/rice_temperature_counterfactual_curve_RESEARCH_ONLY.csv')
    obs=pd.read_csv(root/'reports/v30_graph_evidence/rice_figure9_qt_1440_APPROX_NOT_TRAINED.csv')
    obs=obs.loc[obs.modification_route.eq('UNMODIFIED')]
    for pollutant,slug in [('Cd(II)','cd'),('Pb(II)','pb')]:
        fig,ax=plt.subplots(figsize=(8,4.5))
        s=c.loc[c.pollutant.eq(pollutant)]
        for name in ['Ridge','XGBoost']:
            sub=s[s.algorithm.eq(name)].sort_values('pyrolysis_temperature_C')
            ax.plot(sub.pyrolysis_temperature_C,sub.pred_raw_mg_g,marker='o',markersize=4,label=f'Model B {name} (raw)')
        measured=obs.loc[obs.pollutant.eq(pollutant)].sort_values('pyrolysis_temp_C')
        ax.errorbar(measured.pyrolysis_temp_C,measured.qe_reported_approx_mg_g,
                    yerr=measured.digitization_tolerance_mg_g,
                    fmt='s',capsize=4,linewidth=2,label='External graph readings (approx.)')
        ax.axhline(0,color='gray',lw=0.8,ls=':')
        ax.axhline(20,color='gray',lw=1,ls='--',label='Mass-balance cap (20 mg/g)')
        ax.set_title(f'{pollutant}: rice husk / unmodified / held-out DOI')
        ax.set_xlabel('Pyrolysis temperature (°C)')
        ax.set_ylabel('Reported uptake / predicted qe (mg/g)')
        ax.set_xticks([300,350,400,450,500]); ax.legend(fontsize=8,loc='best')
        ax.grid(alpha=.2)
        fig.tight_layout()
        dst=root/f'reports/v32_temperature_identifiability/fig_temperature_curve_{slug}.png'
        fig.savefig(dst,dpi=160);plt.close(fig)

if __name__=='__main__': render()
