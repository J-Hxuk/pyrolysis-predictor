"""V41 non-deploying paired evaluation of V40 Stage-1 OOF rows.

Uses actual persisted OOF predictions; does not fit models. Bootstrap DOI
clusters, not individual rows. The v40 file has no explicit fold id, so this
cannot independently verify identical fold assignment, only exact same row/DOI.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

KEYS = ["doi", "record_id"]
MODELS = ("Ridge", "Poisson_Process", "Poisson_Composition")
TARGETS = ("BET_m2_g_training_candidate", "TPV_cm3_g_training_candidate")


def paired_oof(frame: pd.DataFrame, target: str, challenger: str,
               baseline: str = "Ridge", *, seed: int = 41,
               bootstrap: int = 1500) -> dict:
    if challenger == baseline:
        raise ValueError("동일 모델 비교 불가")
    x = frame.loc[(frame.stage == 'Stage1') & (frame.target == target) &
                  (frame.model.isin([baseline, challenger]))].copy()
    sets = {}
    for name in (baseline, challenger):
        sub = x.loc[x.model == name, KEYS + ["actual", "pred"]].copy()
        if sub.duplicated(KEYS).any():
            raise ValueError(f"중복 OOF 식별자: {name}")
        sets[name] = sub
    a, b = sets[baseline], sets[challenger]
    if len(a) == 0 or len(a) != len(b):
        raise ValueError("비교 가능한 동일 OOF 관측 미확보")
    paired = a.merge(b, on=KEYS, how='outer', validate='one_to_one', suffixes=("_base", "_new"), indicator=True)
    if not (paired['_merge'] == 'both').all():
        raise ValueError("서로 다른 DOI·시료 OOF 결과")
    if not np.allclose(paired.actual_base, paired.actual_new, atol=1e-9, rtol=1e-9):
        raise ValueError("실측값이 다른 OOF 결과를 비교할 수 없음")
    actual = paired.actual_base.to_numpy(float)
    base = paired.pred_base.to_numpy(float)
    new = paired.pred_new.to_numpy(float)
    if not np.isfinite(np.r_[actual,base,new]).all():
        raise ValueError("비유한 값")
    delta = np.abs(new - actual) - np.abs(base - actual)
    # Paired source-cluster bootstrap: resample DOI, then all observations in DOI.
    rng = np.random.default_rng(seed)
    doi = paired.doi.astype(str).to_numpy()
    unique = np.unique(doi)
    group_ix = [np.flatnonzero(doi == d) for d in unique]
    samples = np.empty(bootstrap)
    for k in range(bootstrap):
        draw = rng.integers(0, len(unique), size=len(unique))
        ix = np.concatenate([group_ix[i] for i in draw])
        samples[k] = float(delta[ix].mean())
    result = {
        "target": target, "baseline": baseline, "challenger": challenger,
        "n": len(paired), "n_doi": len(unique),
        "MAE_baseline": float(np.mean(np.abs(base-actual))),
        "MAE_challenger": float(np.mean(np.abs(new-actual))),
        "delta_MAE_challenger_minus_baseline": float(delta.mean()),
        "bootstrap_DOI_95CI": [float(x) for x in np.quantile(samples,[0.025,0.975])],
        "n_negative_baseline": int((base<0).sum()), "n_negative_challenger": int((new<0).sum()),
        "n_low_target_le_25": int((actual<=25).sum()),
        "low_target_MAE_baseline": float(np.mean(np.abs(base[actual<=25]-actual[actual<=25]))) if (actual<=25).any() else None,
        "low_target_MAE_challenger": float(np.mean(np.abs(new[actual<=25]-actual[actual<=25]))) if (actual<=25).any() else None,
        "limitations": "v40 OOF 파일에 fold ID가 없어 동일 split의 구조적 보증은 원본 평가 코드에서 확인해야 하며, 모델 선택에 이미 사용한 DOI에서의 평가임",
    }
    return result


def run(root: Path, output_dir: Path | None = None) -> list[dict]:
    p = root / 'reports/v40_new9/model_checks/all_DOI_OOF_predictions.csv'
    if not p.is_file():
        raise FileNotFoundError(f"v40 OOF 파일 없음: {p}. v40 패치 먼저 설치하세요.")
    df = pd.read_csv(p)
    results = [paired_oof(df,target,model) for target in TARGETS
               for model in MODELS if model != 'Ridge']
    out = output_dir or (root / 'reports/v41_scope')
    out.mkdir(parents=True,exist_ok=True)
    (out/'stage1_paired_DOI_bootstrap.json').write_text(json.dumps(results,ensure_ascii=False,indent=2), encoding='utf-8')
    pd.DataFrame([{k:v for k,v in r.items() if k not in ('bootstrap_DOI_95CI','limitations')}
                  for r in results]).to_csv(out/'stage1_paired_comparison.csv',index=False,encoding='utf-8-sig')
    return results

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    print(json.dumps(run(args.root,args.out),ensure_ascii=False,indent=2))
