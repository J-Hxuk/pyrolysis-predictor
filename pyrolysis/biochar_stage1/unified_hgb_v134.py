"""Shared Stage 1 manufacturing model for deployment and DOI OOF validation."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_VERSION = 'v134_hgb_poisson_10_raw'
S1_NUM = ['pyrolysis_temperature_C', 'pyrolysis_hold_min', 'heating_rate_C_min',
          'activation_temperature_C', 'activation_time_min', 'activation_agent_feedstock_mass_ratio']
S1_CAT = ['feedstock', 'pyrolysis_atmosphere', 'activation_method', 'activation_agent']
S1_FEATURES = S1_NUM + S1_CAT
TARGETS = {'BET':'BET_m2_g_training_candidate', 'TPV':'TPV_cm3_g_training_candidate'}

def prepare_inputs(frame):
    x = frame.reindex(columns=S1_FEATURES).copy()
    for c in S1_NUM:
        x[c] = pd.to_numeric(x[c], errors='coerce')
    for c in S1_CAT:
        x[c] = x[c].replace('', pd.NA).fillna('UNKNOWN').astype(str)
    return x

def make_stage1_model():
    prep = ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)),
                          ('scale', StandardScaler())]), S1_NUM),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), S1_CAT),
    ], sparse_threshold=0)
    reg = HistGradientBoostingRegressor(loss='poisson', max_iter=140, max_leaf_nodes=9,
        min_samples_leaf=10, learning_rate=.065, l2_regularization=5, random_state=42)
    return Pipeline([('pre', prep), ('model', reg)])
