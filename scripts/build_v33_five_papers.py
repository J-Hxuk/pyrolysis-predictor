"""v33 research-only transcription from five uploaded primary papers.

Source values are manually read from named PDF tables/text; graphic points are NOT digitized.
Do not use the staged datasets for training without independent source audit.
Run: python -m scripts.build_v33_five_papers
"""
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'imports' / 'v33_five_papers'
REPORT = ROOT / 'reports' / 'v33_five_papers'

SOURCES = {
 'WANG19': ('10.3390/ijerph16050845','ijerph-16-00845(1).pdf','Wang et al. 2019'),
 'JIA18': ('10.3390/app8071019','applsci-08-01019.pdf','Jia et al. 2018'),
 'HU22': ('10.3390/su14159022','sustainability-14-09022.pdf','Hu et al. 2022'),
 'LV24': ('10.1016/j.arabjc.2024.105946','K1RRR3NXT3E0c2Z5b2NSSHNpYmxYdz09.pdf','Lv et al. 2024'),
 'ABY24': ('10.1038/s41598-024-54028-6','s41598-024-54028-6.pdf','Abyaneh et al. 2024'),
}

def store_csv(path, rows):
    if not rows: raise ValueError(f'empty table: {path}')
    keys=list(dict.fromkeys(k for row in rows for k in row))
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as fp:
        w=csv.DictWriter(fp,fieldnames=keys); w.writeheader();w.writerows(rows)

def material(source,sample,feedstock,temp,bet=None,pore=None,pore_type=None,**extra):
    doi,pdf,author=SOURCES[source]
    v={
      'material_id':f'V33_{source}_{sample}', 'source_id':source,'doi':doi,'source_pdf':pdf,
      'sample_name':sample, 'feedstock':feedstock, 'pyrolysis_temperature_C':temp,
      'BET_m2_g':bet, 'pore_volume_cm3_g':pore, 'pore_volume_type':pore_type or 'NOT_REPORTED',
      'feedstock_C_pct':'', 'feedstock_H_pct':'','feedstock_O_pct':'','feedstock_N_pct':'',
      'feedstock_ash_pct':'','feedstock_VM_pct':'','feedstock_FC_pct':'',
      'feedstock_composition_basis':'NOT_REPORTED',
      'status':'SOURCE_STAGING_NOT_TRAINED', 'remark':'Raw biomass composition NOT inferred from biochar product composition',
    }
    v.update(extra)
    return v

def stage1_rows():
    r=[]
    # Wang IJERPH 2019 table 1, PDF page 5. PLB600 missing BOTH BET and pore volume.
    for sample,temp,bet,pv in [
      ('MSB300',300,10.3,.034),('MSB400',400,17.1,.047),('MSB500',500,24.5,.065),('MSB600',600,31.1,.062),
      ('PLB300',300,.2,.001),('PLB400',400,.6,.003),('PLB500',500,7.8,.023),('PLB600',600,None,None)]:
        feed='maize straw' if sample.startswith('MSB') else 'Platanus leaves'
        r.append(material('WANG19',sample,feed,temp,bet,pv,'SOURCE_PORE_VOLUME' if pv is not None else 'NOT_REPORTED',pyrolysis_hold_min=120,heating_rate_C_min=None,pyrolysis_atmosphere='oxygen_limited',source_table='Table 1',source_pdf_page=5,source_route='single_stage_pyrolysis',pore_comparability='Article reports pore volume; confirm TPV protocol before TPV training'))
    # Jia Applied Sciences 2018 table 3, PDF page 7; micropore NOT TPV.
    vals={'RH':[(300,78.401,.009),(400,101.295,.008),(500,292.595,.015),(600,377.717,.032),(700,406.244,.049)],
          'CS':[(300,35.258,.010),(400,117.536,.011),(500,300.103,.013),(600,500.807,.014),(700,553.709,.052)]}
    for p, sub in vals.items():
        for t,bet,micro in sub:
            r.append(material('JIA18',f'{p}{t}','rice husk' if p=='RH' else 'cotton straw',t,bet,micro,'MICROPORE_VOLUME_NOT_TPV',pyrolysis_hold_min=180,heating_rate_C_min=15,pyrolysis_atmosphere='inert_N2',pyrolysis_N2_flow_mL_min=2,source_route='single_stage_pyrolysis',source_table='Table 3',source_pdf_page=7,pore_comparability='Micropore volume: MUST NOT be merged into TPV target'))
    # Hu Sustainability 2022 table 1, PDF page 5, maize straw only; WBC400 not material in table 1.
    for temp,bet,pv in [(300,29.598,.03295),(400,12.110,.02517),(500,23.303,.03170)]:
        r.append(material('HU22',f'BC{temp}','maize straw',temp,bet,pv,'SOURCE_PV_P_P0_0P5',pyrolysis_hold_min=600,heating_rate_C_min=8.5,pyrolysis_atmosphere='LIMITED_OXYGEN_UNSPECIFIED',source_route='single_stage_pyrolysis',source_table='Table 1',source_pdf_page=5,pore_comparability='Authors estimate PV at N2 P/P0 ~0.5; do not assume standard TPV equivalence'))
    # Lv 2024 table 1, PDF page 2; BET present, no numeric TPV.
    for temp,bet in [(400,50.17),(500,126.32),(600,190.56)]:
        r.append(material('LV24',f'B{temp}','soybean straw',temp,bet,source_route='single_stage_pyrolysis',pyrolysis_atmosphere='inert_N2',source_table='Table 1',source_pdf_page=2,remark='Product CHON is NOT raw feedstock CHON; hold/ramp NR'))
    # Abyaneh 2024 table 3 PDF page 5; two FEEDSTOCK particle size classes; TPV unreported.
    props=[('Lv400-1',400,'1-2 mm',2.15),('Lv550-1',550,'1-2 mm',52.35),('Lv700-1',700,'1-2 mm',229.44),
           ('Lv400-63',400,'63-75 um',4.07),('Lv550-63',550,'63-75 um',94.28),('Lv700-63',700,'63-75 um',369.63)]
    for sam,temp,sz,bet in props:
        r.append(material('ABY24',sam,'Ligustrum vulgare pruning waste',temp,bet,source_route='single_stage_pyrolysis',pyrolysis_hold_min=60,heating_rate_C_min=10,pyrolysis_atmosphere='oxygen_limited',raw_particle_size=sz,source_table='Table 3',source_pdf_page=5,remark='Table 3 CHON/Ash/VM/FC are POST-pyrolysis biochar, NOT feedstock. SI EDS also post-pyrolysis'))
    return r

def qe(source,sample,pollutant,c0,dose,ph,t,qe_value,origin,table,page,**rest):
    doi,pdf,_=SOURCES[source]
    v={'observation_id':f'V33_{source}_{sample}_{pollutant}_{c0}', 'source_id':source,'doi':doi,'source_pdf':pdf,
       'material_id':f'V33_{source}_{sample}', 'sample_name':sample, 'pollutant':pollutant,
       'C0_mg_L':c0,'dose_g_L':dose,'initial_pH':ph,'contact_time_min':t,
       'qe_mg_g':round(qe_value,8),'qe_origin':origin,
       'qe_physical_cap_mg_g':round(c0/dose,8) if c0 and dose else '',
       'source_table_or_text':table,'source_pdf_page':page,'is_fitted_qmax':'NO',
       'quality_status':'SOURCE_STAGING_NOT_TRAINED'}
    v.update(rest)
    return v

def stage2_measured_rows():
    r=[]
    # Wang pdf p8: measured removal percentage (%) in prose and +/- uncertainties in %p.
    removal={40:[(300,94.4,.5),(400,95.3,.3),(500,97.1,.8),(600,90,2)],
             80:[(300,80,5),(400,97.3,.2),(500,97.9,.3),(600,81,1)],
             170:[(300,72,5),(400,83,.5),(500,92,3),(600,51,2)]}
    for c0,values in removal.items():
        for temp,pct,sd in values:
            r.append(qe('WANG19',f'MSB{temp}','Cd(II)',c0,8.0,'4.5-5.0',1440, (c0/8.0)*(pct/100.0),
                        'DERIVED_FROM_REPORTED_EXPERIMENTAL_REMOVAL_PERCENT',
                        'Section 3.2 reported text; Figure 6(a)',8,
                        removal_pct=pct,reported_uncertainty_pct_points=sd,
                        derived_qe_uncertainty_mg_g=round((c0/8)*(sd/100),8),
                        adsorption_temperature_C='',solution_matrix='single_metal_aqueous',
                        note='Direct text percentages; calculated qe from C0/dose; equilibrium certification not independently audited'))
    # Hu pdf p9 section 3.4 direct reported experimental capacities, controlled single-metal 50 mg/L.
    for temp,value in [(300,6.03),(400,9.11),(500,7.56)]:
        r.append(qe('HU22',f'BC{temp}','Cd(II)',50.0,5.0,5.0,1440,value,
                    'REPORTED_EXPERIMENTAL_UPTAKE','Section 3.4, single-solution Cd',9,
                    adsorption_temperature_C=25,solution_matrix='single_metal_aqueous',
                    note='0.125 g / 25 mL per Methods; Hg/Pb paper also includes mixture but these Cd values are single-solution'))
    return r

def stage2_leachate_rows():
    # This paper is MULTICOMPONENT landfill leachate, differing from single-metal Stage 2 contract.
    r=[]
    for metal,c0,qe_exp in [('Pb(II)',4.94,4.877),('Cd(II)',1.81,1.492),('Mn(II)',7.23,7.255)]:
        v=qe('ABY24','Lv700-63',metal,c0,0.1,'AS_RECEIVED_LEACHATE_pH_7.84_PLUS_MINUS_0.8',None,qe_exp,
             'REPORTED_QE_EXP_FROM_KINETIC_TABLE','Table 4 qe,exp; Table 2 C0; Methods 0.1 g/L',11,
             adsorption_temperature_C='25 ± 2',solution_matrix='MULTICOMPONENT_REAL_LANDFILL_LEACHATE',
             quality_status='SEPARATE_DOMAIN_NOT_CORE',
             note='C0 = leachate mean; contact time for qe_exp not explicitly tagged to single point. Not a single-metal aqueous uptake dataset.')
        r.append(v)
    return r

def fitted_qmax_rows():
    r=[]
    def add(src,sample,metal,value,table,page,note=''):
        doi,pdf,_=SOURCES[src]
        r.append({'source_id':src,'doi':doi,'source_pdf':pdf, 'material_id':f'V33_{src}_{sample}',
                  'sample_name':sample,'pollutant':metal,'qmax_mg_g':value,
                  'target_type':'LANGMUIR_FITTED_QMAX_NOT_MEASURED_QE', 'source_table':table,
                  'source_pdf_page':page,'note':note,'status':'SEPARATE_TARGET_NOT_TRAINED'})
    for sam,qm in zip(['MSB300','MSB400','MSB500','MSB600','PLB300','PLB400','PLB500','PLB600'],[30.30,30.12,35.46,17.21,21.83,14.16,25.45,19.49]):
        add('WANG19',sam,'Cd(II)',qm,'Table 2',9)
    for metal,values in {'Hg(II)':[129.31,173.27,190.24], 'Cd(II)':[9.27,22.83,17.59], 'Pb(II)':[45.45,81.47,56.84]}.items():
        for t,v in zip([300,400,500],values):
            add('HU22',f'BC{t}',metal,v,'Table 3',8,'Cd BC300: TABLE says 9.27 vs prose says 9.97: retain table value and flag conflict' if metal=='Cd(II)' and t==300 else '')
    for t,v in zip([400,500,600],[33.91,39.18,47.30]):
        add('LV24',f'B{t}','Cd(II)',v,'Table 3',3,'Do not use fitted kinetic qe.cal as experimental qe')
    for metal,v in [('Pb(II)',13.855),('Cd(II)',5.598),('Mn(II)',28.141)]:
        add('ABY24','Lv700-63',metal,v,'Table 5',11,'Real landfill leachate multi-metal system')
    return r

def audit(rows1,rows2,leach,qmax):
    flags=[]
    def issue(source,item,issue,severity):flags.append({'source_id':source,'item':item,'issue':issue,'severity':severity})
    issue('WANG19','PLB600','BET and PV explicitly NR in Table 1; do not impute', 'MISSING')
    issue('WANG19','raw feedstock analysis','Table 1 elemental analysis is BIOCHAR output, not raw biomass input','FEATURE_LEAKAGE')
    issue('WANG19','MSB series','qe derived from mean removal%, not separately reported or Langmuir fitted','DERIVED_MEASURED')
    issue('JIA18','all RH and CS samples','Micropore volume != TPV; do not populate TPV label','TARGET_MISMATCH')
    issue('JIA18','Fig. 5 vs Table 4','Graph measured removal and kinetic fitted qm differ; Table 4 not measured qe','TARGET_MISMATCH')
    issue('JIA18','Table 2','Some extracted PDF table text duplicated/misaligned; source image checked','PARSER_CAUTION')
    issue('HU22','BC300 Cd qmax','Table 3 reports 9.27 vs narrative 9.97 mg/g; retain table and flag','SOURCE_CONFLICT')
    issue('HU22','all pore volume','Authors PV at P/P0~0.5; standard TPV compatibility not demonstrated','METHOD_MISMATCH')
    issue('HU22','BC series','24h, pH5 single-metal Cd uptake in sec3.4 distinct from fitted qmax','VALID_TARGET')
    issue('LV24','B400/B500/B600','Only BET numeric. CHON table is biochar after pyrolysis; do not copy to feedstock','FEATURE_LEAKAGE')
    issue('LV24','Table 2','Pseudo-second-order qe.cal marked mg/L despite being uptake; do not substitute for measured qe','UNIT_AND_TARGET')
    issue('ABY24','6 samples','Table 3 VM/FC/CHON are produced BIOCHAR; supplementary EDS not raw feedstock','FEATURE_LEAKAGE')
    issue('ABY24','all measured qe','Multi-component real landfill leachate, not single-metal synthetic solution','DOMAIN_SHIFT')
    issue('ABY24','Table 4','Experimental qe only for Lv700-63, other five samples removal values are a plotted Figure 4','PARTIAL_TARGET')
    issue('ABY24','Table 4','Kinetic qe_exp contact-time mapping and dose must be source-audited independently','CONDITIONS_UNCERTAINTY')
    issues=flags[:]
    for row in rows2+leach:
        cap=float(row['qe_physical_cap_mg_g']); qe=float(row['qe_mg_g']);
        if qe<0 or qe>cap*(1+1e-9):
            issue(row['source_id'],row['observation_id'],f'qe={qe} out of 0..{cap} mg/g', 'PHYSICAL_FAIL')
    return flags

def build():
    s1=stage1_rows(); s2=stage2_measured_rows(); leach=stage2_leachate_rows(); qmax=fitted_qmax_rows()
    assert len(s1)==30 and sum(row['BET_m2_g'] is not None for row in s1)==29
    assert len(s2)==15 and len(leach)==3 and len(qmax)==23
    assert len({r['material_id'] for r in s1})==len(s1)
    assert set(r['material_id'] for r in s2+leach).issubset(set(r['material_id'] for r in s1))
    flags=audit(s1,s2,leach,qmax)
    # Never expose the raw instrument's particle composition as feedstock composition.
    assert all(not r['feedstock_C_pct'] and not r['feedstock_ash_pct'] for r in s1)
    OUT.mkdir(parents=True,exist_ok=True); REPORT.mkdir(parents=True,exist_ok=True)
    store_csv(OUT/'stage1_new30_temperature_series_RESEARCH_STAGING.csv',s1)
    store_csv(OUT/'stage2_new15_single_metal_measured_or_derived_RESEARCH_STAGING.csv',s2)
    store_csv(OUT/'stage2_new3_multimetal_leachate_SEPARATE_DOMAIN.csv',leach)
    store_csv(OUT/'qmax_new23_FITTED_NOT_QE.csv',qmax)
    store_csv(OUT/'source_quality_flags.csv',flags)
    group={}
    for r in s2:
        k=(r['doi'],r['sample_name'].rstrip('0123456789'),r['pollutant'],r['C0_mg_L'],r['dose_g_L'])
        group.setdefault(k,[]).append(r)
    ranks=[]
    # Fixed absorption conditions within each DOI, same source material and C0/dose.
    for (doi,series,metal,c0,dose), rows in sorted(group.items()):
        if len(rows)<3:continue
        best=max(rows,key=lambda x:x['qe_mg_g'])
        ranks.append({'doi':doi,'sample_family':series,'pollutant':metal,'C0_mg_L':c0,'dose_g_L':dose,
                      'number_of_temperatures':len(rows),'best_sample_in_observed_set':best['sample_name'],
                      'best_temperature_C':int(best['sample_name'].lstrip('ABCDEFGHIJKLMNOPQRSTUVWXYZ')),
                      'observed_not_global':'YES', 'independent_Doi_evaluation_only':'YES'})
    store_csv(OUT/'within_paper_measured_temperature_ranking_DESCRIPTIVE.csv',ranks)
    # Audit overlap against previously staged source master, if present in installed project.
    old=ROOT/'data/imports/v24_source_audit/Stage1_Master_297_RESEARCH_STAGING_NOT_TRAINED.csv'
    overlap_status={'baseline_file_found':old.exists(),'existing_master_overlap':None,'extended_master_row_count':None}
    if old.exists():
        with old.open(encoding='utf-8-sig',newline='') as fp: original=list(csv.DictReader(fp))
        doi0=set((r.get('doi') or '').lower() for r in original)
        duplicates=sorted(set(r['doi'].lower() for r in s1)&doi0)
        if duplicates:raise RuntimeError('New DOI present in existing master; manual dedup required: '+str(duplicates))
        # candidate master is for inspection only, preserves all baseline columns; explicit provenance columns appended.
        keys=list(original[0]);keys.extend(['v33_pore_type','v33_original_sample_id'])
        merged=[{**r,'v33_pore_type':'','v33_original_sample_id':''} for r in original]
        for row in s1:
            m={k:'' for k in keys}
            m.update({
                'material_id':row['material_id'],'doi':row['doi'],
                'original_sample_name':row['sample_name'],'feedstock':row['feedstock'],
                'feedstock_group':row['feedstock'], 'pyrolysis_temperature_C':row['pyrolysis_temperature_C'],
                'pyrolysis_hold_min':row.get('pyrolysis_hold_min'),
                'heating_rate_C_min':row.get('heating_rate_C_min'),
                'pyrolysis_atmosphere':row.get('pyrolysis_atmosphere'),
                'BET_m2_g_original_source':row['BET_m2_g'],
                'BET_m2_g_training_candidate':row['BET_m2_g'],
                'BET_candidate_flag':'PENDING_SOURCE_APPROVAL' if row['BET_m2_g'] is not None else 'NR',
                'TPV_cm3_g_original_source':row['pore_volume_cm3_g'] if row['pore_volume_type']=='SOURCE_PORE_VOLUME' else '',
                'TPV_candidate_flag':'ON_HOLD_METHOD_VALIDATION',
                'v33_pore_type':row['pore_volume_type'],'v33_original_sample_id':row['sample_name'],
                'source_pdf_page':row['source_pdf_page'],'source_reference':row['source_table'],
                'release_status':'RESEARCH_STAGING_NOT_TRAINED',
                'source_stage':'V33_RESEARCH_STAGING'})
            merged.append(m)
        store_csv(OUT/'Stage1_Master_327_RESEARCH_STAGING_NOT_TRAINED.csv',merged)
        overlap_status={'baseline_file_found':True,'existing_master_overlap':duplicates,'extended_master_row_count':len(merged)}
    summary={'source_papers':len(SOURCES),'source_doi':sorted(x[0] for x in SOURCES.values()),
             'stage1_material_records':len(s1),'stage1_BET_numeric':sum(r['BET_m2_g'] is not None for r in s1),
             'stage1_reported_pore_volume_numeric':sum(r['pore_volume_cm3_g'] is not None for r in s1),
             'stage1_pore_volume_type_counts':dict(Counter(r['pore_volume_type'] for r in s1)),
             'stage2_single_metal_experimental_derived_rows':len(s2),
             'stage2_leachate_separate_domain_rows':len(leach),'fitted_qmax_rows':len(qmax),
             'stage2_training_approved_rows':0,'new_feedstock_composition_complete_rows':0,
             'within_doi_temperature_ranking_comparisons':len(ranks),
             'physical_fail_on_new_18':sum(float(r['qe_mg_g'])>float(r['qe_physical_cap_mg_g']) for r in s2+leach),
             'source_quality_flags':len(flags), 'baseline_comparison':overlap_status,
             'note':'Source-transcribed, not independent two-reviewer QA, no model updated'}
    (REPORT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary

if __name__=='__main__':print(json.dumps(build(),ensure_ascii=False,indent=2))
