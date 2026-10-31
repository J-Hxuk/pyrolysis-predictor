"""Curated numeric transcription of newly supplied PDFs, separate incompatible targets.
Run from installed project root; original values are preserved in provenance fields.
"""
import pandas as pd
from pathlib import Path

D=Path('data/imports/v38_new_dois')
D.mkdir(parents=True,exist_ok=True)
COFF='10.1016/j.chemosphere.2022.134051'
WHEAT='10.1016/j.colsurfa.2021.128100'
SUGAR='10.21577/0103-5053.20200038'
SOY='10.1016/j.arabjc.2024.105946'
rows=[]
for t,sa,tpv in [(300,1.127,0.0003822),(500,1.180,0.0005503),(700,1.238,0.0006103)]:
 for aged in [False,True]:
  sca=[1.137,0.897,0.5242][[300,500,700].index(t)] if aged else sa
  vol=[0.0004069,0.0004260,0.000046][[300,500,700].index(t)] if aged else tpv
  rows.append(dict(material_id=f'V38_CGB_{t}_'+('AGED' if aged else 'FRESH'),doi=COFF,paper_year=2022,
    original_sample_name=('Aged-' if aged else '')+f'CGB{t}',feedstock='spent coffee grounds',feedstock_group='coffee grounds',
    pyrolysis_temperature_C=t,pyrolysis_hold_min=120,heating_rate_C_min=10,pyrolysis_atmosphere='inert_N2',
    pyrolysis_N2_flow_mL_min=400,activation_method='oxidative_aging_H2O2' if aged else 'none',
    activation_agent='H2O2' if aged else 'none',activation_temperature_C=80 if aged else None,activation_time_min=360 if aged else None,
    process_stage='PYROLYSIS_PLUS_POST_AGING' if aged else 'SINGLE_STAGE_PYROLYSIS',
    BET_m2_g_original_source=sca,BET_m2_g_training_candidate=sca,
    TPV_cm3_g_original_source=vol,TPV_cm3_g_training_candidate=vol,
    BET_measurement_method='N2 adsorption BET',TPV_measurement_method='pore volume as reported',
    BET_value_evidence='Table S7 numeric',TPV_value_evidence='Table S7 numeric',
    BET_candidate_flag='SOURCE_TABLE_RESEARCH_CANDIDATE',TPV_candidate_flag='SOURCE_TABLE_RESEARCH_CANDIDATE',
    source_pdf_page='Supplement p7 Table S7',source_reference=COFF,
    raw_feedstock_basis_note='Raw composition not reported as verified input; postchar EDS forbidden',
    new_data_release_gate='RESEARCH_STAGING_NOT_APPROVED',source_access='USER_UPLOADED_MAIN_SUPP'))
for atm,vals,amt in [('inert_N2',[1.77,8.47,68.09,22.83],'N'),('oxygen_limited',[2.91,27.98,135.22,25.35],'A')]:
 for t,sa in zip([300,450,600,750],vals):
  rows.append(dict(material_id=f'V38_WHEAT_{amt}{t}',doi=WHEAT,paper_year=2022,original_sample_name=f'{amt}{t}_BC',
    feedstock='wheat straw',feedstock_group='wheat straw',pyrolysis_temperature_C=t,pyrolysis_hold_min=300,
    heating_rate_C_min=20,pyrolysis_atmosphere=atm,pyrolysis_N2_flow_mL_min=500 if atm=='inert_N2' else None,
    activation_method='none',activation_agent='none',process_stage='SINGLE_STAGE_PYROLYSIS',
    BET_m2_g_original_source=sa,BET_m2_g_training_candidate=sa,
    TPV_cm3_g_original_source=None,TPV_cm3_g_training_candidate=None,
    BET_measurement_method='N2 adsorption',BET_value_evidence='Main Table 1 numeric',
    BET_candidate_flag='SOURCE_TABLE_RESEARCH_CANDIDATE',TPV_candidate_flag='NOT_REPORTED',
    source_pdf_page='Main p3 Table 1',source_reference=WHEAT,
    raw_feedstock_basis_note='Biomass ultimate analysis exists but basis unverified; not introduced as model input',
    new_data_release_gate='RESEARCH_STAGING_NOT_APPROVED',source_access='USER_UPLOADED_MAIN_SUPP'))
for t in [300,400,500,600]:
 rows.append(dict(material_id=f'V38_BAGASSE_BC{t}',doi=SUGAR,paper_year=2020,
  original_sample_name=f'BC{t}',feedstock='sugarcane bagasse',feedstock_group='sugarcane bagasse',
  pyrolysis_temperature_C=t,pyrolysis_hold_min=120,heating_rate_C_min=5,
  pyrolysis_atmosphere='oxygen_limited',activation_method='none',activation_agent='none',
  process_stage='SINGLE_STAGE_PYROLYSIS',BET_m2_g_original_source=None,
  BET_m2_g_training_candidate=None,TPV_cm3_g_original_source=None,TPV_cm3_g_training_candidate=None,
  BET_candidate_flag='NOT_REPORTED',TPV_candidate_flag='NOT_REPORTED',
  source_pdf_page='Main methods p2, no tabled BET',source_reference=SUGAR,
  new_data_release_gate='METADATA_ONLY_DO_NOT_TRAIN',source_access='USER_UPLOADED_MAIN'))
assert len(rows)==18
pd.DataFrame(rows).to_csv(D/'new_materials_18_SOURCE_STAGING.csv',index=False)

# The main paper calls these equilibrium capacities; they are NOT the fitted q_e
# from its supplementary PSO kinetic Tables S1-S2. They remain conditional until
# initial pH and per-sample contact time/equilibrium can be verified.
cd={'FRESH':{300:5.24,500:18.92,700:26.90},'AGED':{300:5.73,500:22.56,700:7.70}}
zn={'FRESH':{300:4.53,500:6.13,700:8.99},'AGED':{300:4.03,500:22.67,700:4.78}}
obs=[]
for pollutant,dat in [('Cd(II)',cd),('Zn(II)',zn)]:
 for aged,nm in dat.items():
  for t,y in nm.items():
   mat=f'V38_CGB_{t}_{aged}'
   obs.append(dict(observation_id=f'V38_{mat}_{pollutant.replace("(","").replace(")","")}',doi=COFF,
       pollutant=pollutant,C0_mg_L=50,initial_pH=None,dose_g_L=1.0,adsorption_T_C=25,
       contact_time_min=None,feedstock_label_normalized='spent coffee grounds',
       manufacturing_route='SINGLE_STAGE_PYROLYSIS',process_step_count=2 if aged=='AGED' else 1,
       pyrolysis_temperature_C=t,pyrolysis_hold_min=120,heating_rate_C_min=10,
       pyrolysis_atmosphere='inert_N2',
       activation_or_modification='H2O2_AGED' if aged=='AGED' else 'none',
       secondary_temperature_C=80 if aged=='AGED' else None,
       secondary_hold_min=360 if aged=='AGED' else None,
       reported_activation_agent='H2O2' if aged=='AGED' else None,
       target_name='qe_reported_mg_g',target_value=y,target_unit='mg/g',
       target_origin='MAIN_TEXT_EQUILIBRIUM_UPTAKE_NOT_KINETIC_FIT',
       model_use_tier='V38_RESEARCH_CONDITIONAL_NOT_CERTIFIED',
       target_qualification='EQUILIBRIUM_REPORTED_NOT_INDEPENDENTLY_CERTIFIED',
       source_target_status='RESEARCH_CONDITIONAL',
       physical_cap_pass=True,mass_balance_cap_mg_g=50.0,
       warnings='initial pH NR; per-sample contact time NR; aging feature not fully represented by Model B; no direct C_e table',
       source_material_id=mat,source_pdf='CoffeeGrounds_Biochar_CdZn_main.pdf',source_page=4,
       initial_pH_raw_reported=None))
assert len(obs)==12 and all(0<=r['target_value']<=50 for r in obs)
pd.DataFrame(obs).to_csv(D/'coffee_singlemetal_12_RESEARCH_CONDITIONAL.csv',index=False)

# Multi-component solution: EACH metal initially 500 ug/L, dose 0.2 g / 0.1 L = 2 g/L,
# per-species max mass balance 0.5/2 = 0.25 mg/g. Never blend with single-metal qe.
removal={'Cd(II)':51.50,'Cr(III)':74.35,'Cu(II)':91.18,'Ni(II)':47.05,'Pb(II)':96.17,'Zn(II)':40.50}
mm=[]
for ion,r in removal.items():
 mm.append({'doi':SUGAR,'material_id':'V38_BAGASSE_BC500','pollutant':ion,
            'target_name':'multi_metal_reported_removal_percent','removal_percent':r,
            'C0_per_metal_mg_L':0.5,'dose_g_L':2.0,'converted_uptake_mg_g':round(.25*r/100,8),
            'source':'Main abstract and section Adsorption of trace metals by biochars',
            'experiment_matrix':'SIX_METAL_SIMULTANEOUS','status':'HOLD_SEPARATE_DOMAIN'})
pd.DataFrame(mm).to_csv(D/'bagasse_multimetal_6_SEPARATE_DOMAIN.csv',index=False)

manifest=[
 {'doi':COFF,'title':'Effects of pyrolysis temperature and aging treatment on the adsorption of Cd and Zn by coffee grounds biochar','status':'NEW','main_pdf':'provided','supp_pdf':'provided','stage1_material_rows':6,'bet_rows':6,'tpv_rows':6,'singlemetal_reported_uptake_candidates':12,'notes':'Main text equilibrium amounts; different from kinetic fitted q_e in supplement. Initial pH NR, contact time NR. Research conditional only.'},
 {'doi':WHEAT,'title':'Wheat straw biochar N2 vs air-limitation Cd/Ni sorption','status':'NEW','main_pdf':'provided','supp_pdf':'provided','stage1_material_rows':8,'bet_rows':8,'tpv_rows':0,'singlemetal_reported_uptake_candidates':0,'notes':'Main Table 1 BET; Table S4 fitted Langmuir Qm is not measured qe. Raw source elemental basis not certified.'},
 {'doi':SUGAR,'title':'Sugarcane bagasse biochar simultaneous six-metal adsorption','status':'NEW','main_pdf':'provided','supp_pdf':'not_provided','stage1_material_rows':4,'bet_rows':0,'tpv_rows':0,'singlemetal_reported_uptake_candidates':0,'notes':'4 manufacture cases metadata only; 6 metal removal values at BC500 multi-ion system in ug/L. Exclude from single-ion model.'},
 {'doi':SOY,'title':'Soybean straw Cd adsorption','status':'DUPLICATE_V33','main_pdf':'provided','supp_pdf':'not_provided','stage1_material_rows':0,'bet_rows':0,'tpv_rows':0,'singlemetal_reported_uptake_candidates':0,'notes':'Three B400/B500/B600 material rows already in v33. No duplicate insert.'},
]
pd.DataFrame(manifest).to_csv(D/'source_audit_4_DOIs.csv',index=False)
print('new materials',len(rows),'BET',sum(pd.notna(x.get('BET_m2_g_training_candidate')) for x in rows),'TPV',sum(pd.notna(x.get('TPV_cm3_g_training_candidate')) for x in rows),'conditional qe',len(obs))
