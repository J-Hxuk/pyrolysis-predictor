"""Research-only Stage 2 physically bounded uptake regressors.

The physical bound 0 <= uptake <= C0_mg_L / dose_g_L follows a batch
mass balance for an initially dissolved single-metal analyte, not proof that
removal is due to adsorption rather than precipitation/other processes.

This estimator is intentionally NOT connected to deployment/active.json.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from pyrolysis.biochar_stage2.v15_features import model_inputs
from scripts.build_v37_full_refit import S2_CAT, input_frame, pipeline


def physical_cap(frame: pd.DataFrame) -> np.ndarray:
    """Per-row maximum uptake in mg/g; never silently repair invalid inputs."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError('X must be a pandas DataFrame with C0_mg_L and dose_g_L')
    if any(col not in frame.columns for col in ('C0_mg_L', 'dose_g_L')):
        raise ValueError('C0_mg_L and dose_g_L are required')
    c0 = pd.to_numeric(frame['C0_mg_L'], errors='coerce').to_numpy(float)
    dose = pd.to_numeric(frame['dose_g_L'], errors='coerce').to_numpy(float)
    if not np.isfinite(c0).all() or not np.isfinite(dose).all() or (c0<=0).any() or (dose<=0).any():
        raise ValueError('C0 and dose must be finite and strictly positive')
    return c0/dose


class BoundedUptakeRegressor(BaseEstimator, RegressorMixin):
    """Predict fraction removed, then multiply by mass-balance upper bound.

    kind='logit_ridge' estimates logit(fraction) with a fixed epsilon to handle
    measured fractions at exactly 0 and 1; kind='fraction_forest' fits RF to
    observed fractions (RF averages preserve range of observed [0,1] labels).
    This transformation is NOT automatically an accurate physical mechanism.
    """
    def __init__(self, model='A', kind='logit_ridge', epsilon=0.001):
        self.model=model
        self.kind=kind
        self.epsilon=epsilon

    def fit(self, X, y):
        if self.kind not in ('logit_ridge', 'fraction_forest'):
            raise ValueError('Unsupported bound model')
        if not 0 < self.epsilon < .1:
            raise ValueError('epsilon must lie in (0,0.1)')
        inputs=model_inputs(self.model)
        self.inputs_=list(inputs)
        self.cats_=[i for i in inputs if i in S2_CAT]
        self.nums_=[i for i in inputs if i not in self.cats_]
        cap=physical_cap(X)
        yy=np.asarray(y,dtype=float).reshape(-1)
        if len(yy)!=len(cap) or not np.isfinite(yy).all():
            raise ValueError('y must be finite and same length as X')
        frac=yy/cap
        if (frac<-1e-8).any() or (frac>1+1e-8).any():
            raise ValueError('physically incompatible training target; quarantine at source')
        frac=np.clip(frac,0,1)
        if self.kind=='logit_ridge':
            self.estimator_=pipeline(self.nums_,self.cats_,'Ridge')
            transformed=logit(np.clip(frac,self.epsilon,1-self.epsilon))
        else:
            self.estimator_=pipeline(self.nums_,self.cats_,'RF')
            transformed=frac
        self.estimator_.fit(input_frame(X,self.nums_,self.cats_),transformed)
        self.n_train_=len(yy)
        return self

    def predict_fraction(self,X):
        if not hasattr(self,'estimator_'):
            raise ValueError('fit must be called before predict')
        physical_cap(X)
        pred=self.estimator_.predict(input_frame(X,self.nums_,self.cats_))
        if self.kind=='logit_ridge':
            return expit(pred)
        return np.clip(pred,0,1)

    def predict(self,X):
        cap=physical_cap(X)
        return cap*self.predict_fraction(X)


def clip_raw_predictions(raw, frame):
    """An explicit post-processing comparator, NOT learned physics."""
    cap=physical_cap(frame)
    p=np.asarray(raw,float).reshape(-1)
    if len(p)!=len(cap) or not np.isfinite(p).all():
        raise ValueError('raw predictions must match the rows and be finite')
    return np.clip(p,0,cap)
