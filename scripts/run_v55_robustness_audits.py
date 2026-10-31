"""Supplemental v55 DOI-cluster bootstrap, Chen16 rank, ambiguous-unit qmax ledger."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
from scripts.run_v55_source_cumulative_refit import DOI,OUT,DATA,BASE2,APPLE,metrics,stage2_fit
from scripts.run_v54_bounded_rank import chen_features,rank_metrics,CHEN

def main(root):
 root=Path(root).resolve();report=root/OUT;data=root/DATA
 pred=pd.read_csv(report/'stage2_same140_group_oof_predictions.csv')
 groups=list(pred.groupby('doi',sort=True).indices.values());rng=np.random.default_rng(55)
 samples=[np.concatenate([groups[j] for j in rng.integers(len(groups),size=len(groups))]) for _ in range(3000)]
 y=pred.target_value.to_numpy(float)
 rows=[]
 for model in ['A','B']:
  a=np.abs(y-pred[f'{model}__old140'].to_numpy(float))
  for scen in ['v53_148','v53_164_sensitivity','v55_152_plus_sugar','v55_156_plus_cocoa_conditional',
               'v55_168_all24_plus_sugar_sensitivity','v55_172_all24_plus_cocoa_sensitivity']:
   b=np.abs(y-pred[f'{model}__{scen}'].to_numpy(float));d=a-b
   bs=np.array([d[k].mean() for k in samples])
   rows.append({'model':model,'scenario':scen,'mae_improvement_mg_g':d.mean(),
    'doi_cluster_bootstrap95_low':np.percentile(bs,2.5),'doi_cluster_bootstrap95_high':np.percentile(bs,97.5),
    'all_original_32_DOI_fixed':True,'n_boot':3000,'caution':'Exploratory post-comparison CI; not confirmation after tuning'})
 pd.DataFrame(rows).to_csv(report/'doi_paired_bootstrap_v55.csv',index=False)
 b=pd.read_csv(root/BASE2);apple=pd.read_csv(root/APPLE)
 apple8=apple[apple.target_origin=='V52_ZHAO2020_TABLE3_QE_EXP'].copy()
 new=pd.read_csv(data/'New_Reported_QE_8.csv')
 from scripts.run_v55_source_cumulative_refit import to_stage2
 nn=to_stage2(new,b)
 addons={'old140':nn.iloc[:0], 'v53_148':apple8,
         'v55_152_plus_sugar':pd.concat([apple8,nn[nn.source_key=='sugar']],ignore_index=True),
         'v55_156_plus_cocoa_conditional':pd.concat([apple8,nn],ignore_index=True),
         'v55_168_all24_plus_sugar_sensitivity':pd.concat([apple,nn[nn.source_key=='sugar']],ignore_index=True),
         'v55_172_all24_plus_cocoa_sensitivity':pd.concat([apple,nn],ignore_index=True)}
 chen=pd.read_csv(root/CHEN); inp=chen_features(chen)
 out=chen[['material_uid','reported_removal_pct','temperature_C','hold_min','heating_rate_C_min']].copy()
 rank=[]
 for scen,ad in addons.items():
  tr=pd.concat([b,ad],ignore_index=True)
  for v in ['A','B']:
   pp,raw,_=stage2_fit(tr,inp,v)
   pct=100*pp/(inp.C0_mg_L/inp.dose_g_L).to_numpy(float)
   out[f'{v}__{scen}__estimated_removal_percent']=pct
   r=rank_metrics(out.material_uid.tolist(),out.reported_removal_pct.to_numpy(float),pct)
   rank.append({'model':v,'scenario':scen,'benchmark':'CHEN2022_SOURCE_COMPARISON_16',
    'pH_evidence':'NR_IMPUTED','time_of_measure':'SOURCE_FIXED','status':'IN_PAPER_RANK_NOT_INDEPENDENT_AFTER_SELECTION',**r})
 pd.DataFrame(rank).to_csv(report/'chen16_cumulative_rank_metrics.csv',index=False)
 out.to_csv(report/'chen16_cumulative_rank_predictions.csv',index=False)
 # Original printed Qmax units: table heading mg/kg; retained as printed, not equated to mg/g.
 sludge=[]
 for metal,vals in [('Pb(II)',[12.6,15.3,18.2,16.6,15.4,12.4,12.8]),
                    ('Cr(VI)',[208,69,36.6,141,45.2,118,19.6])]:
  for i,(T,hold,val) in enumerate(zip([300,400,400,500,500,600,600],[120,60,120,60,120,60,120],vals),1):
   sludge.append({'observation_id':f'V55_ZHANG_SDBC{i}_{metal[:2]}_QMAX','doi':DOI['sludge'],
    'material_uid':f'V55_SLUDGE_SDBC_{i}','feedstock_raw':'municipal wastewater sludge',
    'pollutant':metal,'pyrolysis_temperature_C':T,'pyrolysis_hold_min':hold,
    'target_value_as_printed':val,'target_units_as_printed':'mg kg-1',
    'reported_source_page':'p6 Table 2','type':'Langmuir_fitted_Qmax',
    'fit_warning':'UNRESOLVED_UNIT_NOT_CONVERTED_NO_DIRECT_QE','source_url':'https://doi.org/'+DOI['sludge']})
 pd.DataFrame(sludge).to_csv(data/'Sludge_14_QMAX_UNRESOLVED_UNIT_SOURCE_LEDGER.csv',index=False)
 print('bootstrap, Chen rank, sludge original Qmax 14 entries saved')
 return pd.DataFrame(rank)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--root',default='.');args=a.parse_args()
 print(main(args.root).to_string(index=False))
