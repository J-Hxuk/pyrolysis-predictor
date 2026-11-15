"""Fixed 50:50 fraction ensemble; shared deployment / evaluation factory."""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_VERSION = 'v139_knn10_et_capgroup_50_50'
SNAPSHOT = Path('v114_patch/v112_training_4795.csv')
SNAPSHOT_SHA = 'de2dd067e8a02c481466875fa927c42b93049734f1b255739a7587bc1338388c'
FEATURES = ['initial_conc_mg_L','initial_pH','dose_g_L_recovered','adsorption_temp_C',
 'contact_time_min','pyrolysis_temperature_C','pyrolysis_hold_min','organic_BET_m2_g',
 'organic_TPV_cm3_g','log_initial_conc','log_dose','log_contact','log_capacity_cap',
 'organic_log_bet','organic_log_tpv','pollutant','dataset','pollutant_domain']
CATS = ['pollutant','dataset','pollutant_domain']
NUM = [c for c in FEATURES if c not in CATS]

def make_model(kind):
    imp = SimpleImputer(strategy='median', keep_empty_features=True)
    if kind == 'knn':
        num = Pipeline([('imp',imp),('scale',StandardScaler())])
        reg = KNeighborsRegressor(n_neighbors=10, weights='distance')
    elif kind == 'et':
        num = imp
        reg = ExtraTreesRegressor(n_estimators=100,min_samples_leaf=2,max_features=.9,random_state=107,n_jobs=1)
    else:
        raise ValueError(kind)
    prep = ColumnTransformer([('num',num,NUM),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),CATS)],sparse_threshold=0)
    return Pipeline([('prep',prep),('model',reg)])

class FractionBlend:
    def fit(self, d):
        y = d.removal_fraction.to_numpy(float)
        cap = d.cap.to_numpy(float)
        if not np.isfinite(y).all() or ((y<0)|(y>1)).any() or not np.isfinite(cap).all() or (cap<=0).any():
            raise ValueError('Invalid fraction/cap')
        w = (1/d.groupby('material_group').material_group.transform('size')).to_numpy()
        w *= np.clip(np.sqrt(cap/np.median(cap)),.25,4)
        self.knn = make_model('knn').fit(d[FEATURES],y)
        self.et = make_model('et').fit(d[FEATURES],y,model__sample_weight=w/w.mean())
        return self
    def predict(self, x):
        return .5*np.clip(self.knn.predict(x[FEATURES]),0,1)+.5*np.clip(self.et.predict(x[FEATURES]),0,1)

def load_blend(root):
    path = Path(root)/SNAPSHOT
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SNAPSHOT_SHA:
        raise ValueError('Stage2 training snapshot hash mismatch')
    d = pd.read_csv(path,low_memory=False)
    if len(d)!=4795 or d.material_group.nunique()!=56 or d.pollutant.nunique()!=26:
        raise ValueError('Unexpected Stage2 training snapshot')
    labels = {key:sorted(d.loc[d.eval_domain.eq(scope),'pollutant'].unique()) for key,scope in
        [('labels_core','core97'),('labels_organic_compilation','organic_2023'),('labels_inorganic_compilation','inorg_2024')]}
    return dict(pipeline=FractionBlend().fit(d),feature_names=list(FEATURES),target_transform='fraction',
        model_target='removal_fraction',n_rows=len(d),supported_labels=sorted(d.pollutant.unique()),
        selected_candidate=MODEL_VERSION,training_snapshot_sha256=digest,**labels)
