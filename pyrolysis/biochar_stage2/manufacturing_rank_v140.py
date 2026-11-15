"""Fixed v60 Manufacturing RF, deployed separately from absolute capacity.
Training snapshot contains only the original 59 manufacturing-panel records.
"""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_VERSION = 'v140_manufacturing_rf_v60'
SNAPSHOT = Path('data/ranking/v140_manufacturing_panels_59.csv')
MFG = ['pollutant','feedstock_label_normalized','manufacturing_route','process_step_count',
       'pyrolysis_temperature_C','pyrolysis_hold_min','heating_rate_C_min',
       'pyrolysis_atmosphere','activation_or_modification','secondary_temperature_C','secondary_hold_min']
CATS = ['pollutant','feedstock_label_normalized','manufacturing_route','pyrolysis_atmosphere','activation_or_modification']
NUM = [c for c in MFG if c not in CATS]


def features(frame):
    x = frame.reindex(columns=MFG).copy()
    for c in NUM:
        x[c] = pd.to_numeric(x[c],errors='coerce')
    for c in CATS:
        x[c] = x[c].astype(object).where(x[c].notna(),np.nan)
    return x


def fit_ranker(d):
    # Identical target, sample weights, preprocessing and RF configuration to v60.
    cap = d.C0_mg_L.to_numpy(float)/d.dose_g_L.to_numpy(float)
    y = 100*d.target_value.to_numpy(float)/cap
    y = y-pd.Series(y,index=d.index).groupby(d.panel_id).transform('mean').to_numpy()
    n_panels = d.groupby('doi').panel_id.nunique().to_dict()
    sizes = d.groupby(['doi','panel_id']).size().to_dict()
    w = np.array([1/(np.sqrt(n_panels[doi])*sizes[doi,p]) for doi,p in zip(d.doi,d.panel_id)])
    w *= len(w)/w.sum()
    num = Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('scale',StandardScaler())])
    cat = Pipeline([('impute',SimpleImputer(strategy='constant',fill_value='UNKNOWN')),('ohe',OneHotEncoder(handle_unknown='ignore'))])
    prep = ColumnTransformer([('num',num,NUM),('cat',cat,CATS)])
    reg = RandomForestRegressor(n_estimators=160,max_depth=3,min_samples_leaf=2,max_features=.9,random_state=42,n_jobs=1)
    return Pipeline([('pre',prep),('model',reg)]).fit(features(d),y,model__sample_weight=w)


def load_ranker(root):
    path = Path(root)/SNAPSHOT
    d = pd.read_csv(path,low_memory=False)
    if (len(d)!=59 or d.doi.nunique()!=9 or d.panel_id.nunique()!=15 or not d.observation_id.is_unique
            or not d.groupby('doi').fold.nunique().eq(1).all()
            or not d.groupby('panel_id').doi.nunique().eq(1).all()
            or not d.groupby('panel_id').pyrolysis_temperature_C.nunique().ge(3).all()):
        raise ValueError('Unexpected v60 manufacturing panel snapshot')
    vals=d[['target_value','C0_mg_L','dose_g_L']].to_numpy(float)
    if not np.isfinite(vals).all() or (vals[:,0]<0).any() or (vals[:,1:]<=0).any():
        raise ValueError('Invalid ranking source target or mass-balance inputs')
    return {'model':fit_ranker(d),'data':d,'version':MODEL_VERSION,
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def rank_sweep(bundle, feedstock, process, pollutant, temperatures):
    """Only the current UI's single-stage, unmodified route; all else is unsupported.
    Scores are *relative* centered removal percentage points, never mg/g or removal %.
    Unknown categories are shown as exploratory, never silently replaced by another material.
    """
    d=bundle['data']; notes=[]
    if pollutant not in set(d.pollutant):
        return None,['순위 RF에 학습되지 않은 오염물질입니다.']
    if process.get('activation_method')!='none':
        return None,['현재 순위 UI는 단일 열분해·무활성화 조건만 지원합니다. 활성화 후처리 조건은 별도 검증 필요.']
    ts=np.asarray(temperatures,dtype=float)
    if not np.isfinite(ts).all() or len(np.unique(ts))!=len(ts):
        raise ValueError('Invalid ranking temperatures')
    def exact_category(value,col):
        choices={str(v).strip().casefold():v for v in d[col].dropna().unique()}
        key=str(value).strip().casefold()
        return choices.get(key,str(value).strip())
    name=exact_category(feedstock,'feedstock_label_normalized')
    atm=exact_category(process.get('pyrolysis_atmosphere','NR'),'pyrolysis_atmosphere')
    if atm=='NR': atm=np.nan
    context=d[(d.pollutant==pollutant)&(d.feedstock_label_normalized==name)&(d.process_step_count==1)]
    if context.empty: notes.append('해당 원료·오염물질 조합은 순위 학습 자료에 없습니다. 탐색 점수만 표시합니다.')
    if not pd.isna(atm) and atm not in set(d.pyrolysis_atmosphere.dropna()): notes.append('미학습 분위기: 범주 외삽입니다.')
    if pd.isna(atm): notes.append('분위기 미보고: 학습 전처리로 처리합니다.')
    for field,title in [('pyrolysis_hold_min','유지시간'),('heating_rate_C_min','승온속도')]:
        value=float(process[field]);lo=d[field].min();hi=d[field].max()
        if not np.isfinite(value) or value<=0: raise ValueError('Invalid '+field)
        if not lo<=value<=hi: notes.append(f'{title}이 순위 RF 관측 범위({lo:g}–{hi:g}) 밖입니다.')
    lo,hi=d.pyrolysis_temperature_C.min(),d.pyrolysis_temperature_C.max()
    valid=(ts>=lo)&(ts<=hi)
    if not valid.all(): notes.append(f'순위 RF 관측 제조온도({lo:g}–{hi:g}℃) 밖의 후보는 제외합니다.')
    if valid.sum()<3: return None,notes+['순위 비교에는 관측 범위 내 서로 다른 온도 후보 3개 이상이 필요합니다.']
    x=pd.DataFrame({'pollutant':pollutant,'feedstock_label_normalized':name,
        'manufacturing_route':'SINGLE_STAGE_PYROLYSIS','process_step_count':1,
        'pyrolysis_temperature_C':ts[valid],'pyrolysis_hold_min':float(process['pyrolysis_hold_min']),
        'heating_rate_C_min':float(process['heating_rate_C_min']),'pyrolysis_atmosphere':atm,
        'activation_or_modification':'none','secondary_temperature_C':np.nan,'secondary_hold_min':np.nan})
    scores=bundle['model'].predict(features(x))
    if not np.isfinite(scores).all(): raise ValueError('Nonfinite rank score')
    order=np.argsort(-scores,kind='stable');ranks=np.zeros(len(scores),int)
    for pos,i in enumerate(order):
        ranks[i]=ranks[order[pos-1]] if pos and np.isclose(scores[i],scores[order[pos-1]],atol=1e-9,rtol=0) else pos+1
    return pd.DataFrame({'제조온도 (℃)':ts[valid],'RF 상대 점수 (pp)':scores,'RF 순위':ranks}),notes
