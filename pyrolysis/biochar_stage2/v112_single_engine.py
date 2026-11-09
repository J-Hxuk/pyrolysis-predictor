"""One user-facing pipeline: raw feedstock -> Stage1 BET/TPV -> ONE broad Stage2 predictor.
Research demonstration, not strict equilibrium adsorption or external DOI validation.
No legacy Stage2 A/B/C model is loaded by this module.
"""
from __future__ import annotations
from pathlib import Path
from typing import Mapping
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pyrolysis.biochar_stage2.input_support import match_feedstock, normalize_name
from pyrolysis.biochar_stage1.composition_inference_v20 import (
    default_model_path, load_portable_model, assess_inputs, estimate_bet)

STAGE1_DATA = Path('data/imports/v46_phase8/Stage1_417_EXPLORATORY_UNAPPROVED.csv')
S2_MODEL = Path('deployment/v112_single_stage2.joblib')
S1_NUM = ['pyrolysis_temperature_C','pyrolysis_hold_min','heating_rate_C_min']
S1_CAT = ['feedstock','pyrolysis_atmosphere','activation_method']
S1_FEATURES = S1_NUM + S1_CAT
CATS = ['pollutant','dataset','pollutant_domain']



def _stage1_reference(s1: pd.DataFrame, bet_target: str, tpv_target: str) -> dict:
    """Training-category/range inventory, scoped to each property model's fitted rows."""
    reference = {}
    for label, target in (("BET", bet_target), ("TPV", tpv_target)):
        mask = pd.to_numeric(s1[target], errors='coerce').notna()
        subset = s1.loc[mask]
        reference[label] = {
            'categorical': {name: frozenset(subset[name].astype(str)) for name in S1_CAT},
            'numeric': {name: (float(subset[name].min()), float(subset[name].max()))
                        for name in S1_NUM if subset[name].notna().any()},
        }
    # Use a single, evidence-backed spelling across the two property models.
    # When multiple spellings differ only by case, take the most frequent one.
    union = s1.loc[(pd.to_numeric(s1[bet_target], errors='coerce').notna()) |
                   (pd.to_numeric(s1[tpv_target], errors='coerce').notna()), 'feedstock']
    variants = union.astype(str).value_counts()
    preferred = {}
    for name in variants.index:
        preferred.setdefault(normalize_name(name), name)
    reference['feedstock_known'] = tuple(preferred.values())
    return reference


def resolve_stage1_feedstock(models: Mapping, value: str) -> tuple[str, str]:
    """Map only evidence-backed exact spellings and explicit synonyms, never fuzzy guesses."""
    raw = str(value or '').strip()
    if not raw:
        raise ValueError('원료명이 필요합니다')
    known = models.get('s1_reference', {}).get('feedstock_known', ())
    if not known:
        return raw, 'unverified'
    match = match_feedstock(raw, list(known))
    if match.status in ('exact', 'alias') and match.matched:
        return match.matched, match.status
    return raw, match.status


def assess_stage1_input(models: Mapping, *, feedstock: str, process: Mapping) -> dict:
    """Flag unseen categories and out-of-range numeric values without imputing or blocking."""
    used, status = resolve_stage1_feedstock(models, feedstock)
    reference = models.get('s1_reference', {})
    warnings_list = []
    labels = {'feedstock':'원료명', 'pyrolysis_atmosphere':'열분해 분위기',
              'activation_method':'활성화 방식', 'pyrolysis_temperature_C':'제조온도',
              'pyrolysis_hold_min':'유지시간', 'heating_rate_C_min':'승온속도'}
    vals = {'feedstock':used, **dict(process)}
    for field in S1_CAT:
        val = str(vals.get(field, 'UNKNOWN'))
        missing = [name for name in ('BET','TPV')
                   if name in reference and val not in reference[name]['categorical'].get(field, ())]
        if missing:
            warnings_list.append(f"{labels[field]} '{val}'은(는) Stage 1 {'·'.join(missing)} 학습 범주에 없습니다. "
                                 '해당 범주 효과는 OneHotEncoder에서 무시될 수 있습니다.')
    for field in S1_NUM:
        value = vals.get(field)
        try: num = float(value)
        except (ValueError, TypeError): continue
        if not np.isfinite(num): continue
        for name in ('BET','TPV'):
            bounds = reference.get(name, {}).get('numeric', {}).get(field)
            if bounds and not (bounds[0] <= num <= bounds[1]):
                warnings_list.append(f'{labels[field]} {num:g}은(는) Stage 1 {name} 학습 관측 범위 '
                                     f'[{bounds[0]:g}, {bounds[1]:g}] 밖입니다.')
    if str(process.get('pyrolysis_atmosphere','')) == 'NR':
        warnings_list.append('NR은 분위기 미보고 표기이며 재현 가능한 공정 분위기 조건이 아닙니다.')
    return {'used_feedstock':used, 'feedstock_match_status':status, 'warnings':warnings_list}

def _finite(value, *, optional=False):
    if value is None or (isinstance(value,str) and not value.strip()):
        return np.nan if optional else _raise('Required numeric field is missing')
    try: v=float(value)
    except (TypeError,ValueError): return np.nan if optional else _raise(f'Invalid numeric field: {value}')
    if not np.isfinite(v):
        return np.nan if optional else _raise('Required numeric field must be finite')
    return v


def _raise(msg):
    raise ValueError(msg)


def _s1_forest():
    prep=ColumnTransformer([
        ('num',Pipeline([('imp',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())]),S1_NUM),
        ('cat',Pipeline([('imp',SimpleImputer(strategy='constant',fill_value='UNKNOWN')),
                         ('onehot',OneHotEncoder(handle_unknown='ignore'))]),S1_CAT)])
    forest=RandomForestRegressor(n_estimators=150,max_depth=9,min_samples_leaf=3,max_features=.9,n_jobs=1,random_state=42)
    return Pipeline([('pre',prep),('model',forest)])


def load_chain(root: Path):
    """Build Stage1 surrogates only; load just one Stage2 regressor trained locally."""
    root=Path(root)
    s1_path=root/STAGE1_DATA
    s2_path=root/S2_MODEL
    if not s1_path.is_file():raise FileNotFoundError(f'Stage1 417 source missing: {s1_path}')
    if not s2_path.is_file():raise FileNotFoundError(f'v112 model missing: {s2_path}. Run v112 training/install first.')
    s1=pd.read_csv(s1_path,low_memory=False)
    if len(s1)!=417 or s1.material_id.duplicated().any():
        raise ValueError('Unexpected Stage1 source version: expected 417 unique material IDs')
    for c in S1_NUM:s1[c]=pd.to_numeric(s1[c],errors='coerce')
    for c in S1_CAT:s1[c]=s1[c].fillna('UNKNOWN').astype(str)
    out={'s1_reference': _stage1_reference(s1, 'BET_m2_g_training_candidate',
                                         'TPV_cm3_g_training_candidate')}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',FutureWarning)
        for name,target in [('BET','BET_m2_g_training_candidate'),('TPV','TPV_cm3_g_training_candidate')]:
            y=pd.to_numeric(s1[target],errors='coerce')
            x=s1.loc[y.notna()].copy()
            if len(x)<30:raise ValueError(f'Not enough Stage1 {name} data')
            m=_s1_forest();m.fit(x[S1_FEATURES],np.log1p(y[y.notna()].clip(lower=0).to_numpy(float)))
            out['s1_'+name]=m
    out['composition_bet']=load_portable_model(default_model_path(root))
    out['stage2']=joblib.load(s2_path)
    b=out['stage2']
    if b.get('n_rows')!=4795 or len(b.get('supported_labels',[]))!=26 or b.get('model_target')!='removal_fraction':
        raise ValueError('Unexpected Stage2 artifact schema; refuse to predict')
    out['metadata']={'stage1_BET_n':int(s1.BET_m2_g_training_candidate.notna().sum()),
                     'stage1_TPV_n':int(s1.TPV_cm3_g_training_candidate.notna().sum()),
                     'stage2_n':int(b['n_rows']),'pollutants':len(b['supported_labels'])}
    return out


def predict_properties(models, root, *,feedstock,temperature,hold,heat,atmosphere,activated,composition,activation_method=None):
    if not str(feedstock).strip():raise ValueError('원료명이 필요합니다')
    temp,hold,heat=[_finite(x) for x in (temperature,hold,heat)]
    if min(temp,hold,heat)<=0:raise ValueError('제조 온도·유지시간·승온속도는 양수여야 합니다')
    process={'pyrolysis_temperature_C':temp,'pyrolysis_hold_min':hold,
             'heating_rate_C_min':heat,'pyrolysis_atmosphere':atmosphere,
             'activation_method':(activation_method or ('chemical_activation' if activated else 'none'))}
    input_contract=assess_stage1_input(models,feedstock=feedstock,process=process)
    s1row={'feedstock':input_contract['used_feedstock'],**process}
    m=pd.DataFrame([s1row])
    bet=max(0.0,float(np.expm1(models['s1_BET'].predict(m[S1_FEATURES])[0])))
    tpv=max(0.0,float(np.expm1(models['s1_TPV'].predict(m[S1_FEATURES])[0])))
    comp_status='NOT_PROVIDED';warnings_list=[]
    bet_path='manufacturing_surrogate'
    if any(v is not None and str(v).strip() for v in composition.values()):
        checked=assess_inputs(models['composition_bet'],process,composition,
                    elemental_basis='daf',proximate_basis='dry',
                    process_type='OTHER' if activated else 'SINGLE_STAGE_PYROLYSIS',activated=activated)
        if checked['predictable'] or checked['can_explore_composition_ood']:
            bet=float(estimate_bet(models['composition_bet'],checked,
                      allow_composition_ood=bool(checked['can_explore_composition_ood'])))
            bet_path='composition_BET_v20'
            comp_status='APPLIED'
            if checked['can_explore_composition_ood']:
                warnings_list.append('원료 조성 기반 BET 계산이 학습범위 밖일 수 있습니다.')
        else:
            comp_status='FALLBACK'
            warnings_list+=list(checked.get('errors',[]))+list(checked.get('outside_training_range',[]))
    if not np.isfinite(bet) or not np.isfinite(tpv) or min(bet,tpv)<0:
        raise ValueError('Stage1 property prediction invalid')
    return {'BET_m2_g':bet,'TPV_cm3_g':tpv,'bet_path':bet_path,'tpv_path':'manufacturing_surrogate',
            'composition_status':comp_status,'warnings':warnings_list,
            'stage1_feedstock_used':input_contract['used_feedstock'],
            'stage1_input_warnings':input_contract['warnings']}


def stage2_row(bundle,adsorption,temperature,hold,properties):
    pollutant=str(adsorption.get('pollutant','')).strip()
    if not pollutant:raise ValueError('오염물질명이 필요합니다')
    c0=_finite(adsorption.get('C0_mg_L'));dose=_finite(adsorption.get('dose_g_L'))
    if c0<=0 or dose<=0:raise ValueError('초기농도와 투입량은 양수여야 합니다')
    ph=_finite(adsorption.get('initial_pH'),optional=True)
    if np.isfinite(ph) and (ph<0 or ph>14):raise ValueError('pH는 0~14 범위여야 합니다')
    tads=_finite(adsorption.get('adsorption_T_C'),optional=True)
    time=_finite(adsorption.get('contact_time_min'),optional=True)
    if np.isfinite(time) and time<=0:raise ValueError('접촉시간은 양수여야 합니다')
    core=pollutant in bundle['labels_core'];organic=pollutant in bundle['labels_organic_compilation'];inorg=pollutant in bundle['labels_inorganic_compilation']
    known=core or organic or inorg
    use_props=core or organic
    domain='core_pollutants' if core else 'organic_pollutants' if organic else 'other_inorganic_pollutants'
    family='organic' if pollutant=='Phenol' or organic else 'inorganic'
    bet=properties['BET_m2_g'];tpv=properties['TPV_cm3_g']
    cap=c0/dose
    data={'pollutant':pollutant,'dataset':family,'pollutant_domain':domain,
        'initial_conc_mg_L':c0,'initial_pH':ph,'dose_g_L_recovered':dose,
        'adsorption_temp_C':tads,'contact_time_min':time,
        'pyrolysis_temperature_C':float(temperature),'pyrolysis_hold_min':float(hold),
        'organic_BET_m2_g':bet if use_props else np.nan,
        'organic_TPV_cm3_g':tpv if use_props else np.nan,
        'organic_log_bet':np.log1p(bet) if use_props else np.nan,
        'organic_log_tpv':np.log1p(tpv) if use_props else np.nan,
        'log_initial_conc':np.log1p(c0),'log_dose':np.log1p(dose),
        'log_contact':np.log1p(time) if np.isfinite(time) else np.nan,
        'log_capacity_cap':np.log1p(cap)}
    row=pd.DataFrame([data])[bundle['feature_names']]
    return row,{'known_pollutant':known,'property_effect_enabled':use_props,'mass_balance_upper':cap}


def predict_one(models,root,*,feedstock,process,composition,adsorption,activated=False):
    prop=predict_properties(models,root,feedstock=feedstock,
        temperature=process['pyrolysis_temperature_C'],hold=process['pyrolysis_hold_min'],
        heat=process['heating_rate_C_min'],atmosphere=process['pyrolysis_atmosphere'],
        activated=activated,composition=composition,
        activation_method=process.get('activation_method'))
    bundle=models['stage2']
    row,meta=stage2_row(bundle,adsorption,process['pyrolysis_temperature_C'],process['pyrolysis_hold_min'],prop)
    raw=float(bundle['pipeline'].predict(row)[0])
    target_transform=bundle.get('target_transform','fraction')
    if target_transform=='logit':
        frac=1.0/(1.0+np.exp(-np.clip(raw,-20.0,20.0)))
    elif target_transform=='fraction':
        frac=raw
    else:
        raise ValueError(f'Unexpected Stage2 target transform: {target_transform}')
    if not np.isfinite(frac):raise ValueError('Stage2 returned nonfinite fraction')
    cap=meta['mass_balance_upper'];q=float(np.clip(frac,0,1)*cap)
    return {'reported_capacity_mg_g':q,'raw_removal_fraction':frac,
            'BET_m2_g':prop['BET_m2_g'],'TPV_cm3_g':prop['TPV_cm3_g'],
            'bet_path':prop['bet_path'],'tpv_path':prop['tpv_path'],
            'composition_status':prop['composition_status'],'mass_balance_upper_mg_g':cap,
            'known_pollutant':meta['known_pollutant'],'stage1_property_feature_used':meta['property_effect_enabled'],
            'warnings':prop['warnings'],
            'stage1_feedstock_used':prop.get('stage1_feedstock_used',feedstock),
            'stage1_input_warnings':prop.get('stage1_input_warnings',[])}
