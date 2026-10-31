"""v45: deterministic source-preserving staging of four NEW independent DOI papers.
Run from project root: python -m scripts.build_v45_independent_dois --root .
This intentionally does not change active models; digitized numeric values need second reading.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

OUT='data/imports/v45_independent_do_is'
OLD_S1='reports/v40_new9/Stage1_Master_381_RESEARCH_STAGING.csv'
OLD_S2='reports/v40_new9/Stage2_reported_uptake_128_RESEARCH_CONDITIONAL.csv'
NEW_DOI={
 'kitchen':'10.1016/j.jhazmat.2021.125785',
 'cu':'10.1016/j.psep.2017.10.014',
 'municipal':'10.1016/j.biortech.2014.04.048',
 'beans':'10.3390/w17070918',
}

def material_rows():
    rows=[]
    def add(doi,code,feedstock,temp,bet=None,tpv=None,**extra):
        d=dict(material_id=f'V45_{code}',doi=doi,original_sample_name=code,feedstock=feedstock,
               feedstock_group='sludge' if 'sludge' in feedstock else 'agricultural waste',
               pyrolysis_temperature_C=temp,activation_method='none',
               BET_m2_g_original_source=bet,BET_m2_g_training_candidate=bet,
               TPV_cm3_g_original_source=tpv,TPV_cm3_g_training_candidate=tpv,
               source_stage='v45_NEW_DOI',new_data_release_gate='RESEARCH_ONLY_NOT_PRODUCTION_APPROVED',
               BET_candidate_flag='verified_numeric' if bet is not None else 'exclude_missing',
               TPV_candidate_flag='verified_numeric' if tpv is not None else 'exclude_missing')
        d.update(extra);rows.append(d)
    # Chen 2014 Bioresour Technol 164 pp51 Table 3, BET + total volume at P/P0=.99
    for t,b,v in zip([500,600,700,800,900],[25.424,20.268,32.167,48.499,67.603],
                     [.05608,.05268,.06842,.08989,.09855]):
        add(NEW_DOI['municipal'],f'Chen2014_BC{t}','municipal sewage sludge (biophysically dried)',t,b,v,
            pyrolysis_atmosphere='inert_N2',pyrolysis_hold_min=np.nan,heating_rate_C_min=np.nan,
            process_stage='fast_pyrolysis',BET_value_evidence='MAIN_TABLE_3',TPV_value_evidence='MAIN_TABLE_3_PPO_0.99',
            source_pdf_page='main PDF p5 Table 3',source_reference='Chen et al 2014 doi:10.1016/j.biortech.2014.04.048',
            raw_feedstock_basis_note='char elemental Table 1 is NOT raw feedstock composition')
    # Xu 2021 J Hazard Mater 416 p3 Table 1; two-hour process 4 K/min, N2 flow unspecified
    for t,b in zip([300,400,500,600],[1.19,2.09,14.80,10.27]):
        add(NEW_DOI['kitchen'],f'Xu2021_KWB{t}','kitchen waste residue',t,b,None,
            pyrolysis_atmosphere='inert_N2',pyrolysis_hold_min=120,heating_rate_C_min=4,
            process_stage='single_stage_pyrolysis',BET_value_evidence='MAIN_TABLE_1',
            source_pdf_page='main PDF p3 Table 1',source_reference='Xu et al 2021 J Hazard Mater',
            raw_feedstock_basis_note='Table 1 elemental and ash values are CHAR, NOT raw feedstock',
            input_verification_limitations='TPV not numerically reported in reviewed Table 1')
    # Xu et al 2018 PSEP Fig 3: 600,700,900 stated explicitly in prose, 300/500 Figure only, preserved missing
    for t,b in zip([300,500,600,700,900],[None,None,92.3,65.5,64.2]):
        add(NEW_DOI['cu'],f'Xu2018_BC{t}','sewage sludge (oven dried)',t,b,None,
            pyrolysis_atmosphere='inert_N2',heating_rate_C_min=10,pyrolysis_N2_flow_mL_min=650,
            pyrolysis_hold_min=np.nan,process_stage='single_stage_pyrolysis',
            feedstock_VM_pct_original=55.8,feedstock_FC_pct_original=10.5,feedstock_ash_pct_original=33.7,
            raw_feedstock_basis_note='Raw sludge Table 1, VM/FC/ash moisture-free, NOT char; 300/500 BET Figure 3 not exact',
            BET_value_evidence='TEXT_SECTION_3.4' if b is not None else 'FIGURE_ONLY_NOT_DIGITIZED',
            source_pdf_page='main PDF p2 Table 1; p5 Fig 3 + section 3.4',source_reference='Xu et al 2018 PSEP')
    # Water 2025: 9 materials, BET only in Figure 1B, do not turn graphical approximations into exact training values
    for prefix,raw in [('PB','peanut shells'),('PW','pea shells'),('PM','soybean shells')]:
        for t in (400,500,600):
            add(NEW_DOI['beans'],f'Water2025_{prefix}{t}',raw,t,None,None,
                pyrolysis_atmosphere='oxygen_limited',pyrolysis_hold_min=240,
                process_stage='single_stage_pyrolysis',
                BET_value_evidence='FIGURE_1B_NOT_DIGITIZED',
                source_pdf_page='main PDF p3 Methods; p5 Figure 1B; p8 Figures 4-5',
                source_reference='Water 2025 17 918',
                input_verification_limitations='BET figure bars no exact table; experimental Pb/Cd curves require independent reading')
    f=pd.DataFrame(rows);assert len(f)==23
    return f

def uptake_rows():
    rows=[]
    def add(doi,code,feedstock,t,metal,c0,dose,time,adsT,qe,source,uncertainty,process=None):
        assert 0<=qe<=c0/dose+1e-5,(code,qe,c0/dose)
        rows.append(dict(observation_id=f'V45_{code}_{metal.replace("(","").replace(")","")}',
           doi=doi,doi_group=doi,material_uid=f'V45_{code}',sample_group=f'V45_{code}',
           experiment_series_id=f'V45_{doi}_{metal}_fixed_adsorption',
           feedstock_raw=feedstock,feedstock_label_normalized=feedstock,
           feedstock_group_for_novelty_check='sewage sludge',manufacturing_route='SINGLE_STAGE_PYROLYSIS',
           process_step_count=1,pyrolysis_temperature_C=t,
           pyrolysis_atmosphere='inert_N2',activation_or_modification='none',
           pollutant=metal,C0_mg_L=c0,initial_pH=np.nan,dose_g_L=dose,adsorption_T_C=adsT,
           contact_time_min=time,target_name='qe_reported_mg_g',target_value=qe,target_unit='mg/g',
           target_origin=source,model_use_tier='V45_FIGURE_APPROX_RESEARCH_ONLY',
           target_qualification='FIGURE_APPROX_EXPERIMENTAL_UPTAKE_NOT_SECOND_READER_CERTIFIED',
           physical_cap_pass=True,mass_balance_cap_mg_g=c0/dose,
           source_page=process,source_url=f'https://doi.org/{doi}',
           source_target_status='RESEARCH_ONLY_DIGITIZED_APPROX',
           graph_readout_uncertainty_mg_g=uncertainty,
           warnings='Approximate manual graph reading; initial pH NR; do not represent as exact table value'))
    # PSEP 2018 p6 Fig 5a, direct BC600 numeric in Abstract/Sec 3.6. Bar readout approx ±2 mg/g
    for t,q in zip([300,500,600,700,900],[138.5,143.4,146.7,143.9,143.6]):
        add(NEW_DOI['cu'],f'Xu2018_BC{t}','sewage sludge (oven dried)',t,'Cu(II)',1000,0.2/.030,1440,25,q,
            'MAIN_FIG_5A_BAR_APPROX' if t!=600 else 'TEXT_SECTION_3.6_146.7',
            2 if t!=600 else 0.1,'main PDF p4 Methods 2.5; p6 Fig5a and p7 section 3.6')
    # Chen 2014 p6 Fig 2A inset, C0=47.44 mg/L, dose 50mg/25mL = 2g/L, 4h
    # approximate derived from plotted removal % at C0~=47.44; uncertainty ~1.0 mg/g
    for t,q in zip([500,600,700,800,900],[5.6,3.3,6.6,23.0,20.9]):
        add(NEW_DOI['municipal'],f'Chen2014_BC{t}', 'municipal sewage sludge (biophysically dried)',t,
            'Cd(II)',47.44,2.0,240,25,q,'MAIN_FIG_2A_INSET_DIGITIZED_APPROX',1.0,
            'main PDF p3 Methods 2.3; p6 Fig 2A inset; p6 section 3.6')
    return pd.DataFrame(rows)

def hold_out():
    rows=[]
    # KWB paper presents *pseudo-second-order-derived* equilibrium capacity, not direct qe
    for t,q in zip([300,400,500,600],[18.2,23.6,46.5,49.0]):
        rows.append(dict(doi=NEW_DOI['kitchen'],sample=f'KWB{t}',pollutant='Cd(II)',temperature_C=t,
                         value_mg_g=q,value_kind='kinetic_PSO_fitted_qe',
                         decision='SEPARATE_DO_NOT_TRAIN_EXPERIMENTAL_QE',
                         source_locator='J Hazard Mater 2021 p1 abstract; p3 Methods 2.4'))
    # Water curves not transcribed to precise numeric observations in this release
    for feedstock in ('PB','PW','PM'):
        for t in [400,500,600]:
            for pollutant in ['Cd(II)','Pb(II)']:
                rows.append(dict(doi=NEW_DOI['beans'],sample=f'{feedstock}{t}',pollutant=pollutant,
                    temperature_C=t,value_kind='experimental_isotherm_figure_4_or_5',
                    decision='HOLD_DIGITIZATION_NOT_NUMERICALLY_APPROVED',
                    source_locator='Water 2025 main PDF p8 Figures 4-5, C0 variable 0-500 (Pb) / 0-400 (Cd)'))
    return pd.DataFrame(rows)

def sakhiya_sensitivity():
    r=[]
    for t,pb,zn in [(300,5.12,7.06),(400,12.03,16.58),(500,17.93,25.73)]:
        for pol,val in [('Pb(II)',pb),('Zn(II)',zn)]:
            r.append(dict(paper='Sakhiya 2022',temperature_C=t,pollutant=pol,qe_reported_mg_g=val,
                 C0_mg_L=100,adsorbent_mass_g=.1,Volume_reported_mL=5,
                 Volume_assumed_corrected_mL=50,Correction_status='inferred_typo_not_author_verified',
                 QC_flag='MASS_BALANCE_CONFLICT',Main_training='EXCLUDED',
                 Sensitivity_only='OPTIONAL_SEPARATE_NOT_IN_V45_TRAINING',
                 value_provenance='figure_digitized' if t==400 else 'reported',
                 physical_cap_reported_mg_g=5,physical_cap_corrected_mg_g=50,
                 caution='5mL confirmed in primary comparison; 50mL applies to DIFFERENT experiments. Never claim corrected volume is in paper.'))
    return pd.DataFrame(r)

def run(root:Path):
    root=Path(root);out=root/OUT;out.mkdir(parents=True,exist_ok=True)
    old1=pd.read_csv(root/OLD_S1,low_memory=False);old2=pd.read_csv(root/OLD_S2,low_memory=False)
    new1,new2=material_rows(),uptake_rows()
    assert not set(new1.doi)&set(old1.doi), 'DOI collision in Stage1'
    assert not set(new2.doi)&set(old2.doi), 'DOI collision in Stage2'
    assert not set(new1.material_id)&set(old1.material_id)
    assert not set(new2.observation_id)&set(old2.observation_id)
    s1=pd.concat([old1,new1],ignore_index=True,sort=False)
    s2=pd.concat([old2,new2],ignore_index=True,sort=False)
    assert s1.material_id.is_unique and s2.observation_id.is_unique
    assert s2.target_name.eq('qe_reported_mg_g').all()
    assert (s2.target_value<=s2.C0_mg_L/s2.dose_g_L+1e-5).all()
    for filename,df in [ ('new_stage1_23_SOURCE_STAGING.csv',new1),
           ('new_stage2_10_FIGURE_APPROX_RESEARCH_ONLY.csv',new2),
           ('Stage1_Master_404_RESEARCH_STAGING.csv',s1),
           ('Stage2_reported_uptake_138_RESEARCH_CONDITIONAL.csv',s2),
           ('separate_fitted_or_unread_targets.csv',hold_out()),
           ('Sakhiya_5mL_reported_50mL_inferred_NOT_TRAINED.csv',sakhiya_sensitivity())]:
        df.to_csv(out/filename,index=False)
    manifest=[]
    original_src=root/'raw_papers/v45_independent_dois'
    for p in sorted(original_src.glob('*.pdf')):
        manifest.append(dict(file=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size=p.stat().st_size))
    # Distributed patches omit bulky PDFs. Preserve the verified build-time manifest on user reruns.
    manifest_path=out/'source_sha256_manifest.json'
    if manifest or not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    report=dict(new_dois=4,material_added=len(new1),materials_total=len(s1),
      new_BET_exact=int(new1.BET_m2_g_training_candidate.notna().sum()),
      new_TPV_exact=int(new1.TPV_cm3_g_training_candidate.notna().sum()),
      BET_total=int(s1.BET_m2_g_training_candidate.notna().sum()),
      TPV_total=int(s1.TPV_cm3_g_training_candidate.notna().sum()),
      added_approx_experimental_qe=len(new2),stage2_total=len(s2),stage2_doi=s2.doi.nunique(),
      hold_separate=len(hold_out()),inferred_sakhiya=len(sakhiya_sensitivity()),
      signed_off_training_rows=0,release='RESEARCH_ONLY_NOT_ACTIVE')
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd())
    args=p.parse_args();print(json.dumps(run(args.root),ensure_ascii=False,indent=2))
