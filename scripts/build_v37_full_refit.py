"""v37: Rebuild research-only models from all compatible staged observations.

NOT a deployment update. Targets are deliberately separated by scientific
meaning. Comparisons use DOI-held-out GroupKFold; never random row split.
Do not claim a nested CV or independent external validation from this run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge

from pyrolysis.biochar_stage2 import v15_features as FC

STAGE2_PATH = "data/imports/v34_temperature/stage2_core98_plus_v33_15_RESEARCH_NOT_APPROVED.csv"
STAGE1_PATH = "data/imports/v34_temperature/Stage1_Master_333_RESEARCH_STAGING_NOT_TRAINED.csv"
OUT_REPORT = "reports/v37_full_refit"
OUT_MODELS = "deployment/research_v37_NOT_ACTIVE"
# Avoid treating biochar POST-pyrolysis element composition as raw feedstock chemistry.
S1_NUM = ["pyrolysis_temperature_C", "pyrolysis_hold_min", "heating_rate_C_min",
          "pyrolysis_N2_flow_mL_min", "activation_temperature_C", "activation_time_min",
          "activation_agent_feedstock_mass_ratio", "component_1_fraction", "component_2_fraction"]
S1_CAT = ["feedstock", "feedstock_group", "pyrolysis_atmosphere", "activation_method",
          "activation_agent", "process_stage"]
S1_TARGETS = {"BET": "BET_m2_g_training_candidate", "TPV": "TPV_cm3_g_training_candidate"}
S2_CAT = {"pollutant", "feedstock_label_normalized", "manufacturing_route",
          "pyrolysis_atmosphere", "activation_or_modification"}


def audit_s2(frame: pd.DataFrame) -> dict:
    req = ["observation_id", "doi", "target_name", "target_value", "C0_mg_L", "dose_g_L"]
    if any(c not in frame for c in req):
        raise ValueError("Stage 2 원본 입력 필드 누락")
    if not frame.observation_id.is_unique:
        raise ValueError("observation_id가 중복됨: 중복 논문 병합 금지")
    if not frame.target_name.eq("qe_reported_mg_g").all():
        raise ValueError("qe 외 qt/qmax 타깃 혼합 금지")
    if frame.doi.isna().any():
        raise ValueError("DOI 누락")
    y = pd.to_numeric(frame.target_value, errors="coerce")
    c0 = pd.to_numeric(frame.C0_mg_L, errors="coerce")
    dose = pd.to_numeric(frame.dose_g_L, errors="coerce")
    if not (np.isfinite(y).all() and np.isfinite(c0).all() and np.isfinite(dose).all()):
        raise ValueError("Stage2 비유한 qe/C0/dose")
    if ((dose <= 0) | (c0 <= 0) | (y < 0) | (y > c0/dose + 1e-5)).any():
        raise ValueError("Stage2 물질수지 위반 또는 음수")
    return {"rows": len(frame), "n_doi": int(frame.doi.nunique()),
            "unknown_pH": int(pd.to_numeric(frame.initial_pH, errors="coerce").isna().sum()),
            "note": "reported uptake. 엄격 평형 qe는 인증하지 않음; 신규 15행은 연구용 조건부 자료"}


def audit_s1(frame: pd.DataFrame) -> dict:
    if not frame.material_id.is_unique or frame.doi.isna().any():
        raise ValueError("Stage1 중복 material_id 또는 DOI 누락")
    counts = {}
    for name, col in S1_TARGETS.items():
        x = pd.to_numeric(frame[col], errors="coerce")
        if (x.dropna() < 0).any():
            raise ValueError(f"{name} 음수 자료")
        counts[name] = {"rows": int(x.notna().sum()),
                        "n_doi": int(frame.loc[x.notna(), "doi"].nunique())}
    return {"rows": len(frame), "n_doi": int(frame.doi.nunique()), "targets": counts,
            "note": "신규 Stage1 원문검증 후보를 포함한 연구용 시료; 출처 분류/시험 승인 여부 재감사 필요"}


def input_frame(df, nums, cats):
    X = df.reindex(columns=nums+cats).copy()
    for c in nums:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in cats:
        X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
    return X


def pipeline(nums, cats, kind, *, log_target=False):
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                        ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
                            ("encoder", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("num", numeric, nums), ("cat", categorical, cats)])
    if kind == "Dummy":
        reg = DummyRegressor(strategy="mean")
    elif kind == "Ridge":
        reg = Ridge(alpha=10.0)
    elif kind in ("RF", "RF_log"):
        reg = RandomForestRegressor(n_estimators=160,max_depth=6,
                                    min_samples_leaf=3,max_features="sqrt",random_state=42,n_jobs=1)
    elif kind == "XGBoost":
        from xgboost import XGBRegressor
        reg = XGBRegressor(n_estimators=220,max_depth=2,learning_rate=0.05,
                           subsample=0.8,colsample_bytree=0.8,reg_lambda=5,
                           min_child_weight=5,random_state=42,n_jobs=1,tree_method="hist")
    else:
        raise ValueError(kind)
    if log_target:
        reg = TransformedTargetRegressor(regressor=reg, func=np.log1p, inverse_func=np.expm1)
    return Pipeline([("pre", prep), ("model", reg)])


def oof(df: pd.DataFrame, *, inputs_num, inputs_cat, target_col, algo, log_target=False):
    if len(df) < 5 or df.doi.nunique() < 2:
        raise ValueError("그룹 CV를 구성할 자료 부족")
    y = pd.to_numeric(df[target_col], errors="raise").to_numpy(float)
    groups = df["doi"].astype(str).to_numpy()
    x = input_frame(df,inputs_num,inputs_cat)
    splits = GroupKFold(n_splits=min(5,len(set(groups))))
    pred = np.full(len(y),np.nan)
    for train,test in splits.split(x,y,groups):
        pipe = pipeline(inputs_num,inputs_cat,algo,log_target=log_target)
        pipe.fit(x.iloc[train],y[train])
        pred[test] = pipe.predict(x.iloc[test])
    if not np.isfinite(pred).all():
        raise ValueError("비유한 OOF 예측")
    return pred


def report_metric(y, pred, c0=None, dose=None):
    out = {"n": len(y), "r2": float(r2_score(y,pred)),
           "MAE": float(mean_absolute_error(y,pred)),
           "RMSE": float(np.sqrt(mean_squared_error(y,pred)))}
    if c0 is not None and dose is not None:
        cap = np.asarray(c0,float) / np.asarray(dose,float)
        out["raw_violation_count"] = int(((pred<0)|(pred>cap)).sum())
        out["clipped_MAE"] = float(mean_absolute_error(y,np.clip(pred,0,cap)))
    return out


def inventory(root: Path):
    """Track available separate targets; never silently drop or merge records."""
    candidates = [
        ("QE_PRIMARY_98_PLUS_WANG_HU_15",STAGE2_PATH,"qe_reported_mg_g","RESEARCH_CV_ONLY"),
        ("STAGE1_PHYSICAL_MASTER_333",STAGE1_PATH,"BET_and_TPV","RESEARCH_CV_ONLY"),
        ("GRAPH_READOUT_13","data/imports/v36_temperature/v36_new_graph13_research_only.csv","qe_from_graph","HOLD_SECOND_READER"),
        ("QT_240MIN_ZHOU","data/imports/v34_temperature/zhou_BBD_raw34_QT_240min_NOT_QE.csv","qt_240_min","SEPARATE_TARGET"),
        ("QMAX_CEDAR_20","data/imports/v34_temperature/cedar_qmax20_LANGMUIR_SEPARATE.csv","qmax_fitted","SEPARATE_TARGET"),
        ("NEW_QMAX_V33_23","data/imports/v33_five_papers/qmax_new23_FITTED_NOT_QE.csv","qmax_fitted","SEPARATE_TARGET"),
        ("MULTIMETAL_V33","data/imports/v33_five_papers/stage2_new3_multimetal_leachate_SEPARATE_DOMAIN.csv","mixed_matrix_uptake","SEPARATE_DOMAIN"),
        ("PREVIOUS_69_REVIEW","data/imports/v24_source_audit/stage2_new69_PHYSICAL_PASS_STILL_NOT_TRAINED.csv","reported_uptake","HOLD_Precipitation_pH_Route_QA"),
    ]
    rows = []
    for id,rel,target,status in candidates:
        p=root/rel
        rows.append({"dataset":id,"relative_path":rel,"target":target,"status":status,
                     "exists":p.is_file(),"n_rows":int(len(pd.read_csv(p,low_memory=False))) if p.is_file() else None})
    return pd.DataFrame(rows)


def run(root: Path, *, save_models=True):
    root=Path(root)
    stage2=pd.read_csv(root/STAGE2_PATH,low_memory=False)
    stage1=pd.read_csv(root/STAGE1_PATH,low_memory=False)
    audit={"stage2":audit_s2(stage2), "stage1":audit_s1(stage1)}
    out=root/OUT_REPORT; out.mkdir(parents=True,exist_ok=True)
    inventory(root).to_csv(out/"source_status_manifest.csv",index=False)
    scores=[]; predictions=[]
    # Scientifically comparable targets only. Fixed hyperparameters; not nested selection.
    for model in ["A","B"]:
        inputs=FC.model_inputs(model)
        cats=[x for x in inputs if x in S2_CAT]
        nums=[x for x in inputs if x not in cats]
        for algo in ["Dummy","Ridge","RF","XGBoost"]:
            y=stage2.target_value.to_numpy(float)
            pred=oof(stage2,inputs_num=nums,inputs_cat=cats,target_col="target_value",algo=algo)
            metric=report_metric(y,pred,stage2.C0_mg_L,stage2.dose_g_L)
            scores.append({"stage":"Stage2", "target":"reported_qe", "variant":model,"algorithm":algo, **metric})
            for doi,oid,yy,pp in zip(stage2.doi,stage2.observation_id,y,pred):
                predictions.append({"stage":"Stage2","model":model,"algorithm":algo,"doi":doi,
                                    "id":oid,"target":"reported_qe","actual":yy,"pred_raw":pp})
    # Stage 1, inclusive of all numeric candidate rows, DOI CV. Split by DOI, not material.
    for target, col in S1_TARGETS.items():
        work=stage1.copy()
        work[col]=pd.to_numeric(work[col],errors="coerce")
        work=work[work[col].notna()].copy()
        if len(work) < 10 or work.doi.nunique() < 5:
            raise ValueError(f"Stage1 {target} 독립 DOI 검증 불가")
        for algo,log in [("Dummy",False),("Ridge",False),("RF",False),("RF_log",True)]:
            y=work[col].to_numpy(float)
            pred=oof(work,inputs_num=S1_NUM,inputs_cat=S1_CAT,
                     target_col=col,algo=algo,log_target=log)
            scores.append({"stage":"Stage1", "target":target,"variant":"manufacture_inputs",
                           "algorithm":algo,**report_metric(y,pred)})
    metrics=pd.DataFrame(scores)
    metrics.to_csv(out/"group_cv_model_comparison.csv",index=False)
    pd.DataFrame(predictions).to_csv(out/"stage2_group_cv_predictions.csv",index=False)
    if save_models:
        folder=root/OUT_MODELS;folder.mkdir(parents=True,exist_ok=True)
        # Fixed, predeclared research models only. Not selected based on test OOF.
        models_to_save=[("A","XGBoost"),("B","XGBoost"),("B","Ridge")]
        for name,algo in models_to_save:
            inputs=FC.model_inputs(name)
            cats=[x for x in inputs if x in S2_CAT];nums=[x for x in inputs if x not in cats]
            obj=pipeline(nums,cats,algo)
            obj.fit(input_frame(stage2,nums,cats),stage2.target_value.to_numpy(float))
            joblib.dump(obj,folder/f"Stage2_{name}_{algo}_113_RESEARCH_ONLY.joblib")
        for target,col in S1_TARGETS.items():
            work=stage1.copy();work[col]=pd.to_numeric(work[col],errors="coerce")
            work=work[work[col].notna()]
            obj=pipeline(S1_NUM,S1_CAT,"RF",log_target=True)
            obj.fit(input_frame(work,S1_NUM,S1_CAT),work[col].to_numpy(float))
            joblib.dump(obj,folder/f"Stage1_{target}_RF_log_RESEARCH_ONLY.joblib")
    audit["cv"]={"n_folds":"5 (GroupKFold DOI)","algorithm_settings":"fixed a priori, non-nested",
                 "source_selection":"same-target only; noncertified candidates used for research"}
    audit["auto_deploy"] = False
    (out/"summary.json").write_text(json.dumps({"audit":audit,"scores":scores},ensure_ascii=False,indent=2),encoding="utf-8")
    return audit,metrics


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--no-save-models",action="store_true")
    args=parser.parse_args()
    audit,metrics=run(args.root,save_models=not args.no_save_models)
    print(json.dumps(audit,ensure_ascii=False,indent=2))
    print(metrics.to_string(index=False))
