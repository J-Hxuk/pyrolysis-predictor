"""Reproducible v40 SOURCE-STAGING; no source files/models/UI are deleted or overwritten.
All new numerical values below are transcribed from the uploaded original PDFs;
consult reports/v40_new9/EVIDENCE_KO.md before approving production training.
"""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd

DOIS = {
 'jiang':'10.3390/ma17123030', 'kelp':'10.3390/app11199223',
 'rice':'10.1016/j.biteb.2021.100920', 'coffee':'10.1016/j.egyr.2021.07.032',
 'chicken':'10.1039/C9RA08199B','soy':'10.1021/acsomega.1c04717',
 'husk':'10.3390/en13226033', 'wang':'10.3390/ijerph16050845',
 'hu':'10.3390/su14159022'
}
S1_SOURCE='data/research_v40_inputs/Stage1_Master_351_RESEARCH_STAGING.csv'
S2_SOURCE='data/research_v40_inputs/Stage2_reported_uptake_125_RESEARCH_CONDITIONAL.csv'
OUT='reports/v40_new9'

def source_materials():
    """30 new material IDs: Jiang18, Kelp3, Rice3, Coffee2, Chicken4."""
    rows=[]
    def add(doi, sample, feedstock,temp, hold,rate,atm, bet=None,tpv=None,loc='',flow=None,**extra):
        d=dict(material_id='V40_'+sample, doi=doi, original_sample_name=sample,
            feedstock=feedstock, feedstock_group=feedstock, pyrolysis_temperature_C=temp,
            pyrolysis_hold_min=hold,heating_rate_C_min=rate,pyrolysis_atmosphere=atm,
            pyrolysis_N2_flow_mL_min=flow,
            BET_m2_g_original_source=bet, BET_m2_g_training_candidate=bet,
            TPV_cm3_g_original_source=tpv, TPV_cm3_g_training_candidate=tpv,
            BET_candidate_flag='V40_NUMERIC_SOURCE_STAGED' if bet is not None else 'NOT_REPORTED',
            TPV_candidate_flag='V40_NUMERIC_SOURCE_STAGED' if tpv is not None else 'NOT_REPORTED',
            new_data_release_gate='V40_RESEARCH_STAGING_DO_NOT_AUTO_DEPLOY' if (bet is not None or tpv is not None) else 'METADATA_ONLY_DO_NOT_TRAIN',
            source_reference=loc, source_primary_verified=True,
            source_stage='V40_USER_UPLOADED_PRIMARY_PDF',release_status='RESEARCH_NOT_PRODUCTION_APPROVED',
            source_value_type='published_table_numeric',
            paper_reuse_risk='DOI_GROUPED',
            v40_source_locator=loc, v40_review_status='ORIGINAL_PDF_NUMERIC_CHECKED',
        );d.update(extra);rows.append(d)
    composition={
      'CS':{'feedstock':'coconut shell','vm':81.58,'fc':17.97,'ash':0.45,'c':50.92,'h':6.43,'o':41.98,'n':0.18,'s':0.09,
         'bet':[2.51,0.72,0.31,84.28,89.58,81.56]},
      'RH':{'feedstock':'rice husk','vm':68.59,'fc':10.69,'ash':13.38,'c':43.51,'h':5.82,'o':49.83,'n':0.56,'s':0.28,
         'bet':[0.21,4.48,18.53,202.39,125.76,64.34]},
      'CD':{'feedstock':'cow dung','vm':83.02,'fc':6.31,'ash':6.65,'c':47.31,'h':6.45,'o':44.52,'n':1.44,'s':0.29,
         'bet':[None,None,12.18,6.71,8.19,42.45]},
    }
    for kind,d in composition.items():
        for j,temp in enumerate([300,400,500,600,700,800]):
            add(DOIS['jiang'],f'Jiang_BC_{kind}_{temp}',d['feedstock'],temp,0,10,'inert_N2',
                bet=d['bet'][j],loc='Materials 2024 Table 1 p5; Table 2 p7; Methods p3',flow=500,
                feedstock_VM_pct_original=d['vm'],feedstock_FC_pct_original=d['fc'],
                feedstock_ash_pct_original=d['ash'],feedstock_C_pct_original=d['c'],
                feedstock_H_pct_original=d['h'],feedstock_O_pct_original=d['o'],
                feedstock_N_pct_original=d['n'],feedstock_S_pct_original=d['s'],
                feedstock_elemental_basis='dry (Table 1; elemental+ash do not sum to 100)',
                raw_feedstock_basis_note='Table 1 dry basis; 0 min hold is explicit, not missing',
                BET_measurement_method='N2-BET')
    for t,b,vol in zip([300,450,600],[2.0,4.0,4.0],[.009211,.012871,.018898]):
        add(DOIS['kelp'],f'Kelp_BC{t}','Macrocystis pyrifera',t,60,3,'inert_N2',b,vol,
            'Applied Sciences 2021 Methods p3; Table 2 p6',flow=None,
            raw_feedstock_basis_note='N2 4 L/min described for cooling; not claimed to be pyrolysis flow',
            BET_measurement_method='N2-BET',TPV_measurement_method='N2 adsorption at P/P0=0.99',
            v40_pore_type='TOTAL')
    for t,b,vol in zip([300,400,500],[12.78,26.52,64.56],[.0087,.0142,.0303]):
        add(DOIS['rice'],f'RiceStraw_RSB{t}','rice straw',t,60,10,'inert_N2',b,vol,
            'Bioresource Technology Reports 2022 Methods p2; Table 3 p4',flow=1500,
            feedstock_VM_pct_original=76.32,feedstock_FC_pct_original=9.08,
            feedstock_ash_pct_original=13.91,feedstock_C_pct_original=37.18,
            feedstock_H_pct_original=5.81,feedstock_N_pct_original=0.62,
            feedstock_elemental_basis='NOT_EXPLICITLY_HARMONIZED',
            raw_feedstock_basis_note='Table 1 raw O=56.39 by difference but ash=13.91: basis audit needed; O excluded from model input',
            BET_measurement_method='BELSORP; reported area',TPV_measurement_method='BELSORP reported TOTAL',v40_pore_type='TOTAL')
    for variant,bet,tpv in [('NoSteam',None,.482),('SteamActivated',445,.266)]:
        add(DOIS['coffee'],f'Coffee_{variant}','spent coffee grounds',450,60,9,'inert_N2',bet,tpv,
            'Energy Reports 2021 Methods p2; Table 6 p6',flow=500,
            feedstock_VM_pct_original=76.0,feedstock_FC_pct_original=12.5,
            feedstock_ash_pct_original=1.5,feedstock_moisture_pct_original=10.0,
            feedstock_C_pct_original=54.7,feedstock_H_pct_original=6.5,
            feedstock_N_pct_original=2.7,
            feedstock_elemental_basis='ultimate DAF, proximate AIR-DRIED',
            raw_feedstock_basis_note='O and S lumped as 36.1% difference; do not put 36.1 in O alone',
            activation_method='steam' if variant=='SteamActivated' else 'none',
            activation_temperature_C=800 if variant=='SteamActivated' else None,
            activation_time_min=30 if variant=='SteamActivated' else None,
            BET_measurement_method='reported Table 6; ND is not zero',
            TPV_measurement_method='reported total pore volume',v40_pore_type='TOTAL')
    for t,ash in zip([200,400,600,800],[40.57,44.78,60.95,64.63]):
        add(DOIS['chicken'],f'Chicken_CMB{t}','chicken manure',t,120,None,'inert_N2',
            loc='RSC Advances 2019 Methods p2; Table 1 p2; no BET or TPV in supplied main text',
            raw_feedstock_basis_note='ash and elemental Table 1 are CHAR compositions, not feedstock; do not copy raw',
            v40_char_ash_pct=ash)
    assert len(rows)==30
    return pd.DataFrame(rows)

def new_qe():
    # Kelp paper Table 5 qe,exp; kinetic experiment C0=125 mg/L, dose=1 g/L, 1440 min.
    vals=[(300,71.91),(450,119.9),(600,101.9)]
    rows=[]
    for t,v in vals:
        rows.append(dict(observation_id=f'V40_Kelp_BC{t}_Cu_24h',doi=DOIS['kelp'],
            doi_group=DOIS['kelp'], material_uid=f'V40_Kelp_BC{t}',
            sample_group='Kelp_paper_Cu_kinetics_125',experiment_series_id='V40_Kelp_Cu_Temperature',
            feedstock_group_for_novelty_check='macroalgae',feedstock_raw='Macrocystis pyrifera',
            feedstock_label_normalized='Macrocystis pyrifera',manufacturing_route='SINGLE_STAGE_PYROLYSIS',
            process_step_count=1,pyrolysis_temperature_C=t,pyrolysis_hold_min=60,
            heating_rate_C_min=3,pyrolysis_atmosphere='inert_N2',
            activation_or_modification='none',pollutant='Cu(II)',C0_mg_L=125.0,
            initial_pH=np.nan,dose_g_L=1.0,adsorption_T_C=23.0,contact_time_min=1440,
            target_name='qe_reported_mg_g',target_value=v,target_unit='mg/g',
            target_origin='TABLE_5_EXPERIMENTAL_QE',
            model_use_tier='V40_RESEARCH_CONDITIONAL_NOT_CERTIFIED',
            target_qualification='24H_EXP_QE_NOT_INDEPENDENTLY_EQ_CERTIFIED',
            physical_cap_pass=bool(0<=v<=125),mass_balance_cap_mg_g=125.0,
            BET_measured_m2_g={300:2.0,450:4.0,600:4.0}[t],
            TPV_measured_cm3_g={300:.009211,450:.012871,600:.018898}[t],
            source_url='https://doi.org/10.3390/app11199223',
            source_pdf='Kelp_Biochar_Cu_DOI_10.3390-app11199223_main.pdf',
            source_page='Methods p4; Table 2 p6; Table 5 p13',
            warnings='initial pH not reported; alkaline seaweed char; precipitation/speciation not excluded',
            initial_pH_raw_reported='NR',source_target_status='RESEARCH_CONDITIONAL',
            source_material_id=f'V40_Kelp_BC{t}',route_warning='salt-rich algae: potential non-adsorptive removal',
            stage1_v60_eligible=False))
    return pd.DataFrame(rows)

def source_catalog():
    items=[('10.1016/j.egyr.2021.07.032','feedstock+char property','2 char; 1 BET + 2 TPV', 'Table 1 p4; Table 6 p6'),
    ('10.1021/acsomega.1c04717','independent soybean raw feedstock reference','not matched to v33 soybean batch','Table 1 p2'),
    ('10.1016/j.biteb.2021.100920','paired Stage1, adsorption numeric blocked','3 BET+3 TPV; Pb/Zn mass balance unresolved','Methods p2; Table 3 p4; Fig 1 p5'),
    ('10.3390/ma17123030','paired raw composition + BET','18 char, 16 BET; 3 source feedstock batches','Tables 1/2 pp5/7'),
    (DOIS['wang'],'duplicate DOI','already v33','Table 1 p5'),
    ('10.3390/en13226033','independent raw feedstock reference','not matched to existing adsorption batch','Table 1 p3'),
    ('10.3390/app11199223','paired Stage1 + 24h qe','3 BET+3 TPV + 3 experimental qe','Tables 2/5 pp6/13'),
    ('10.1039/C9RA08199B','manufacturing + fitted qmax','4 char; 0 BET/TPV, 4 qmax candidate','Table 1 p2; Table 2 pp4-5'),
    (DOIS['hu'],'duplicate DOI','already v33','Table 1 p5')]
    return pd.DataFrame(items,columns=['doi','category','v40_action','source_locator'])

def separate_targets():
    # Values NOT from experimental qe - do not mix.
    rows=[]
    for t,q in [(300,49.24),(450,58.00),(600,93.55)]:
        rows.append(dict(doi=DOIS['kelp'],sample=f'Kelp_BC{t}',pollutant='Cu(II)',
          pyrolysis_temperature_C=t,value_mg_g=q,value_kind='Langmuir_fitted_qmax',
          source_locator='Appl. Sci. 2021 Table 4 p11',training_use='QMAX_ONLY_NEVER_QE'))
    # Chicken paper has conflicting CMB800 mistaken as CMB200 in abstract; table source must be checked separately.
    for t,q in [(200,180.21),(400,200.80),(600,239.59),(800,242.57)]:
        rows.append(dict(doi=DOIS['chicken'],sample=f'CMB{t}',pollutant='Pb(II)',
          pyrolysis_temperature_C=t,value_mg_g=q,value_kind='Langmuir_fitted_qmax',
          source_locator='RSC Adv 2019 abstract + Table 2 pp4-5; verify label conflict',training_use='QMAX_ONLY_NEVER_QE'))
    # Unresolved rice paper values; user/team should seek clarification of typo 5 mL vs 50 mL.
    for metal,t,v in [('Pb(II)',500,17.93),('Zn(II)',500,25.73)]:
        rows.append(dict(doi=DOIS['rice'],sample=f'RSB{t}',pollutant=metal,
          pyrolysis_temperature_C=t,value_mg_g=v,value_kind='reported_temperature_comparison',
          source_locator='Bioresour Technol Rep 2022 Methods p2, Fig 1 p5',
          training_use='BLOCKED_QE_MASS_BALANCE_CAP_5'))
    return pd.DataFrame(rows)

def references():
    # These are independent source-specific raw biomass records, NOT matches to other papers.
    return pd.DataFrame([
     dict(doi=DOIS['soy'],feedstock='soybean straw',VM_pct=73.61,FC_pct=14.59,ash_pct=5.03,moisture_pct=6.77,
          C_pct=53.68,H_pct=5.38,O_pct=37.17,N_pct=2.59,S_pct=1.17,
          proximate_basis='as listed incl moisture',ultimate_basis='dry',
          source_locator='ACS Omega 2021 Table 1 p2; elemental sum may exceed 100 due reporting',
          join_policy='INDEPENDENT_REFERENCE_NO_CROSS_DOI_JOIN'),
     dict(doi=DOIS['husk'],feedstock='rice husk',VM_pct=71.94,FC_pct=4.16,ash_pct=14.45,moisture_pct=9.45,
          C_pct=41.28,H_pct=5.12,O_pct=53.11,N_pct=.42,S_pct=.07,
          proximate_basis='as received',ultimate_basis='dry ash free',source_locator='Energies 2020 Table 1 p3',
          join_policy='INDEPENDENT_REFERENCE_NO_CROSS_DOI_JOIN'),
     dict(doi=DOIS['coffee'],feedstock='spent coffee grounds',VM_pct=76.0,FC_pct=12.5,ash_pct=1.5,
          moisture_pct=10.0,C_pct=54.7,H_pct=6.5,O_pct=np.nan,N_pct=2.7,S_pct=np.nan,
          proximate_basis='air dried',ultimate_basis='dry ash free',
          source_locator='Energy Reports 2021 Table 1 p4; O+S reported combined 36.1, not O alone',
          join_policy='SAME_DOI_COFFEE_CHAR_ONLY'),
    ])

def run(root):
    root=Path(root); out=root/OUT;out.mkdir(parents=True,exist_ok=True)
    old1=pd.read_csv(root/S1_SOURCE,low_memory=False);old2=pd.read_csv(root/S2_SOURCE,low_memory=False)
    new1=source_materials();new2=new_qe()
    assert not set(old1.doi)&{DOIS[x] for x in ['jiang','kelp','rice','coffee','chicken']},'previous DOI overlap'
    assert not set(old1.material_id)&set(new1.material_id)
    assert not set(old2.observation_id)&set(new2.observation_id)
    # Preserve input schema; extra auditable fields in appended source rows retained as extra cols.
    whole1=pd.concat([old1,new1],ignore_index=True,sort=False)
    whole2=pd.concat([old2,new2],ignore_index=True,sort=False)
    assert whole1.material_id.is_unique and whole2.observation_id.is_unique
    assert len(whole1)==381 and len(whole2)==128
    assert len(whole1[whole1.BET_m2_g_training_candidate.notna()])==336
    assert len(whole1[whole1.TPV_cm3_g_training_candidate.notna()])==213
    assert all(new2.target_value<=new2.C0_mg_L/new2.dose_g_L)
    for fname,frame in [('new_materials_30_SOURCE_STAGING.csv',new1),('new_qe_3_CONDITIONAL.csv',new2),
        ('Stage1_Master_381_RESEARCH_STAGING.csv',whole1),('Stage2_reported_uptake_128_RESEARCH_CONDITIONAL.csv',whole2),
        ('source_catalog_9_papers.csv',source_catalog()),('separate_target_and_blocked_9.csv',separate_targets()),
        ('independent_feedstock_references_3.csv',references())]:
        frame.to_csv(out/fname,index=False)
    summary=dict(papers_in_zip=9,existing_doi_repeats=2,new_doi=7,material_rows_added=30,
      stage1_all=381,bet_observations=336,tpv_observations=213,qe_added_conditional=3,
      stage2_all_conditional=128,stage2_doi=int(whole2.doi.nunique()),
      separate_target_count=len(separate_targets()),
      no_auto_deploy=True,notes='All rows staged; no source transferred from unrelated DOI as same sample.')
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);args=parser.parse_args()
    print(json.dumps(run(args.root),ensure_ascii=False,indent=2))
