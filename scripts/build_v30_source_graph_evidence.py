"""v30 source-figure digitization: cautious approximate evidence extraction.

Manual readings of published graphical markers, not model-fitted parameters.
Data are NOT authorized for model training. Requires independent QA.
"""
from __future__ import annotations
import csv, json, math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/v30_graph_evidence'
OUT.mkdir(parents=True,exist_ok=True)
DOI_R='10.3390/molecules30173616'
DOI_U='10.3390/toxics12010008'
DOI_S='10.1039/d5va00202h'

# Figure 9, Molecules p.14. Manually read the ~1440-minute markers by magnifying
# the original vector PDF. Values have intentionally only 1 decimal place.
# Order per tuple: [unmodified black square, HNO3 blue circle, KOH red triangle].
# Color-to-sample conflicts with fitted kinetic Table 4/5 are NOT corrected.
RICE = {
 ('Cd(II)',300):(2.7,16.9,19.9),
 ('Cd(II)',400):(7.2,19.9,15.5),
 ('Cd(II)',500):(4.2,5.4,19.5),
 ('Pb(II)',300):(4.3,20.5,20.3),
 ('Pb(II)',400):(12.6,20.6,20.4),
 ('Pb(II)',500):(11.1,20.2,15.8),
}
# Values copied only for Figure-vs-table discrepancy auditing, NEVER targets.
RICE_FITTED = {
 ('Cd(II)',300):(2.66,16.98,20.02),
 ('Cd(II)',400):(7.22,15.49,19.84),
 ('Cd(II)',500):(4.13,5.17,19.46),
 ('Pb(II)',300):(4.37,20.61,20.31),
 ('Pb(II)',400):(12.57,20.39,20.72),
 ('Pb(II)',500):(11.19,15.37,20.03),
}
PROCESS=('UNMODIFIED','HNO3_POST','KOH_POST')
SAMPLE=('BC','HNO3-BC','KOH-BC')
rice=[]
for (poll,t),vals in RICE.items():
 for i,qe in enumerate(vals):
  cap=50/2.5; fitted=RICE_FITTED[(poll,t)][i]; 
  ambiguous=abs(fitted-qe)>2.0
  physical = qe <= cap + 1e-9 and qe>=0
  s={
   'doi':DOI_R,'source_pdf':'molecules-30-03616.pdf','source_page':14,
   'source_figure':'Figure 9','sample_id':f'{SAMPLE[i]}{t}',
   'raw_feedstock':'rice husk','pollutant':poll,'pyrolysis_temp_C':t,
   'modification_route':PROCESS[i],
   'C0_mg_L':50.0,'dose_g_L':2.5,'initial_pH':5.0,'adsorption_T_C':25.0,'contact_time_min':1440,
   'qe_reported_approx_mg_g':qe,'digitization_tolerance_mg_g':0.5,
   'qe_physical_cap_mg_g':cap,'physical_cap_pass':physical,
   'fitted_table_qe2_DO_NOT_USE':fitted,
   'fig9_vs_table4_5_conflict':ambiguous,
   'source_measure_type':'GRAPH_EXPERIMENTAL_MARKER_QT_AT_1440_MIN',
   'training_approved':False,
   'possible_within_paper_temperature_ranking': i==0 and physical and not ambiguous,
   'exclusion_reason': 'FIGURE_FITTED_DISCORDANCE' if ambiguous else ('PHYSICAL_CAP_EXCEEDED' if not physical else 'APPROX_GRAPH_POINT_SINGLE_DOI_NOT_INDEPENDENTLY_QA'),
  }
  rice.append(s)

# Toxics p.8 Fig.4 actual experimental markers: (equilibrium Ce mg/L, q mg/kg).
# Approximate readings. Each preparation has a different experimental dose;
# q and Ce cannot be transformed to a verified reported initial C0.
UGW_POINTS = {
 350:[(0.4,1700),(1.0,3000),(4.1,3750),(7.2,4500),(11.6,4700)],
 450:[(0.5,2450),(1.2,4100),(4.8,5100),(8.4,5500),(13.0,5600)],
 550:[(0.9,2700),(4.1,3800),(6.6,4500),(10.2,4850),(15.0,4900)],
 650:[(0.15,1000),(0.8,3100),(3.2,4900),(5.3,6150),(9.0,6600),(13.4,7050)],
}
DOSAGE={350:1.5,450:1.0,550:0.75,650:0.75}
ugw=[]
for temp,pts in UGW_POINTS.items():
 for ix,(ce,qkg) in enumerate(pts,1):
  # diagnostic only, not an independently known experimental input
  reconstructed_c0=ce+qkg/1000*DOSAGE[temp]
  ugw.append({'doi':DOI_U,'source_pdf':'toxics-12-00008.pdf',
   'source_page':8,'source_figure':'Figure 4',
   'sample_id':f'UGW{temp}','point_index_approx':ix,
   'pyrolysis_temp_C':temp,'pollutant':'Cd(II)',
   'Ce_graph_approx_mg_L':ce,'qe_graph_approx_mg_kg':qkg,
   'qe_converted_approx_mg_g':round(qkg/1000,3),'dose_g_L':DOSAGE[temp],
   'C0_reconstructed_from_q_and_Ce_DIAGNOSTIC_ONLY_mg_L':round(reconstructed_c0,3),
   'reported_C0_for_this_point':'NR','initial_pH':7.0,'adsorption_T_C':'NR','contact_time_min':2880,
   'q_tolerance_mg_g':0.35,'Ce_tolerance_mg_L':0.4,
   'source_measure_type':'GRAPH_EXPERIMENTAL_CE_QE_PAIR',
   'training_approved':False,'ranking_approved':False,
   'exclusion_reason':'C0_PER_POINT_NR; DIFFERENT_DOSE_BY_PYROLYSIS_TEMP; GRAPH_APPROX; INFERRED_C0_TARGET_LEAKAGE_IF_USED_AS_FEATURE'
  })

# 2026 sludge p.5 Fig.2, reported experimental removal% at C0 >=20 mg/L.
# Top-to-bottom visible marker positions manually read (tolerance ~2 percentage points).
# DO NOT infer valid qe directly as paper section 2.3 says 20 g/20 mL whereas Fig2 says 1 g/L.
SLUDGE_PCT={
 400:[(20,62),(50,53),(100,36),(150,29),(200,22),(250,19),(300,16)],
 500:[(20,69),(50,60),(100,51),(150,41),(200,32),(250,27),(300,22)],
 600:[(20,61),(50,63),(100,50),(150,42),(200,38),(250,33),(300,27)],
 700:[(20,69),(50,70),(100,55),(150,46),(200,40),(250,34),(300,29)],
}
sludge=[]
for temp, pts in SLUDGE_PCT.items():
 for c0,r in pts:
  sludge.append({'doi':DOI_S,'source_pdf':'d5va00202h.pdf',
   'source_page':5,'source_figure':'Figure 2','sample_id':f'SSB{temp}',
   'pyrolysis_temp_C':temp,'pollutant':'Pb(II)',
   'C0_mg_L':c0,'removal_graph_approx_pct':r,'removal_tolerance_percentage_points':2,
   'initial_pH':5.0,'adsorption_T_C':25.0,'contact_time_min':1440,
   'dose_in_figure_caption_g_L':1.0,'dose_implied_by_method_20g_in_20mL_g_L':1000,
   'qe_if_caption_dose_1_gL_DIAGNOSTIC_ONLY_mg_g':round(c0*r/100,2),
   'qe_training_value':'NR','source_measure_type':'GRAPH_EXPERIMENTAL_REMOVAL_FRACTION',
   'training_approved':False,'ranking_approved':False,
   'exclusion_reason':'MATERIAL_DOSAGE_CONFLICT_1000x_BETWEEN_METHOD_AND_CAPTION; GRAPH_APPROX; CONVERSION_NOT_APPROVED'
  })

def save(name,rows):
 with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
save('rice_figure9_qt_1440_APPROX_NOT_TRAINED.csv',rice)
# Join on exact (DOI, reported sample) to the v24 Stage 1 material registry.
# No fuzzy names, no inference: a stable 9-material x 2-pollutant link.
stage1_path=ROOT/'data/imports/v24_source_audit/Stage1_Master_297_RESEARCH_STAGING_NOT_TRAINED.csv'
if stage1_path.exists():
 with stage1_path.open(encoding='utf-8-sig',newline='') as f:
  stage1=list(csv.DictReader(f))
 index={}
 for row in stage1:
  if row['doi']==DOI_R:
   key=(row['doi'],row['original_sample_name'].strip().upper())
   if key in index: raise ValueError('Duplicate Stage1 sample key: '+str(key))
   index[key]=row
 links=[]
 for r in rice:
  a=index.get((r['doi'],r['sample_id'].upper()))
  if a is None: raise ValueError('Missing Stage1 link '+r['sample_id'])
  links.append({**r,'stage1_material_id':a['material_id'],
     'stage1_BET_measured_m2_g':a['BET_m2_g_training_candidate'],
     'stage1_TPV_measured_cm3_g':a['TPV_cm3_g_training_candidate'],
     'link_provenance':'SAME_DOI_EXACT_SAMPLE_NAME_NOT_PREDICTED_STAGE1'})
 save('rice_figure9_stage1_linked_18_QA_NOT_TRAINED.csv',links)

save('urban_waste_fig4_ce_qe_APPROX_NOT_TRAINED.csv',ugw)
save('sewage_sludge_fig2_removal_APPROX_DOSAGE_CONFLICT.csv',sludge)

# One exact source unit: Toxics graph y mg/kg, the qmax table in mg/kg
# are not the same kind of value. The 2025 400Cd/500Pb curves contradict fit tables.
from collections import Counter
conflicts=[r for r in rice if r['fig9_vs_table4_5_conflict']]
physicalfails=[r for r in rice if not r['physical_cap_pass']]
allowed=[r for r in rice if r['possible_within_paper_temperature_ranking']]
summary={
 'version':'v30','date':'2026-10-30',
 'rice_experimental_marker_rows':len(rice),
 'rice_physical_cap_failure_count':len(physicalfails),
 'rice_figure_vs_fitted_table_conflict_count':len(conflicts),
 'rice_conflict_samples':[(r['pollutant'],r['sample_id']) for r in conflicts],
 'rice_ranking_usable_provisional_unmodified':len(allowed),
 'rice_ranking_dois':1,
 'UGW_approx_Ce_qe_records_not_independently_qualified':len(ugw),
 'sludge_approx_removal_records_with_dosage_conflict':len(sludge),
 'all_training_approved_count':sum(int(r['training_approved']) for seq in [rice,ugw,sludge] for r in seq),
 'actual_new_Doi_independent_ranking_validation_ready':False,
 'rationale':'Existing figures do supply experimental points, but 3 DOIs have discordance, different dose, or explicit 1000x dosage contradiction. Digitization is source-backed but approximate and needs manual independent review. Do not mix fitting qmax with measured qt/qe.'
}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')

# Stage 1 only exact matched-series q ranking snapshot, no claims of ML rank validation.
rank=[]
for pol in ['Cd(II)','Pb(II)']:
 d=[r for r in allowed if r['pollutant']==pol]
 if len(d)==3:
  for order,r in enumerate(sorted(d,key=lambda v:-v['qe_reported_approx_mg_g']),1):
   rank.append({'doi':DOI_R,'pollutant':pol,'modification_route':'UNMODIFIED','rank_within_one_paper':order,
     'pyrolysis_temp_C':r['pyrolysis_temp_C'],'qe_graph_approx_mg_g':r['qe_reported_approx_mg_g'],
     'rank_validation_status':'DESCRIPTIVE_ONLY_SINGLE_DOI_NO_ML_EVALUATION',
     'resolution_warning':'GRAPH_READ_0.5_MG_G_TOLERANCE'})
save('rice_unmodified_within_paper_ranking_DESCRIPTIVE_ONLY.csv',rank)

print(json.dumps(summary,indent=2,ensure_ascii=False))
