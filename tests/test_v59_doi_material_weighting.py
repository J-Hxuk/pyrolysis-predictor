from pathlib import Path
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from scripts.run_v59_doi_material_weighting import weights_for, material_key, POLICIES,sha,derive
from scripts.build_v46_phase8 import features

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/v59_doi_material_weighting'

def toy():
    return pd.DataFrame({'doi':['a']*4+['b']*2,
        'observation_id':[f'id{i}' for i in range(6)],
        'material_uid':['m1','m1','m1','m2','n1','n1'],
        'target_value':[2,3,4,5,6,7]})

def test_weights_normalized_and_doi_equal():
    f=toy()
    assert np.allclose(weights_for(f,'row_equal'),np.ones(len(f)))
    for policy in POLICIES:
        w=weights_for(f,policy)
        assert np.isfinite(w).all() and np.all(w>0) and np.isclose(w.mean(),1)
    w=weights_for(f,'doi_equal_material_equal')
    assert np.isclose(w[:4].sum(),w[4:].sum())
    assert np.isclose(w[:3].sum(),w[3]) # equal material weight in DOI a
    assert np.isclose(w[0],w[1])

def test_label_changes_do_not_affect_weights():
    f=toy()
    before=weights_for(f,'doi_sqrt_material_equal')
    f['target_value']=[90000]*len(f)
    assert np.array_equal(before,weights_for(f,'doi_sqrt_material_equal'))

def test_legacy_material_mapping_and_evidence():
    df=pd.read_csv(ROOT/'data/imports/v46_phase8/Stage2_140_SOURCE_VERIFIED_AUGMENTED_RESEARCH.csv',low_memory=False)
    mapping, origins=material_key(df)
    assert len(mapping)==140 and mapping.notna().all()
    assert (origins=='legacy_id_wang19').sum()==12
    assert (origins=='legacy_id_hu22').sum()==3
    assert (origins=='legacy_id_v38').sum()==12
    assert not (origins=='unresolved_unique_observation').any()
    o=df[df.doi=='10.3390/ijerph16050845'].copy()
    idx=list(o.index)
    assert len(set(mapping.iloc[idx]))==4 # four temps, not twelve independent materials

def test_data_hashes_and_base_eval_are_fixed():
    manifest=json.loads((REPORT/'run_manifest.json').read_text(encoding='utf-8'))
    for key,rel in [('old','data/imports/v46_phase8/Stage2_140_SOURCE_VERIFIED_AUGMENTED_RESEARCH.csv'),
                    ('v55','data/imports/v55_cumulative_sources/Stage2_152_v55_152_plus_sugar_RESEARCH_NOT_ACTIVE.csv'),
                    ('new','data/imports/v57_handoff_sources/Stage2_New36_Harmonized_RESEARCH.csv')]:
        assert sha(ROOT/rel)==manifest['training_data_sha256'][key]
    oof=pd.read_csv(REPORT/'fixed140_weighted_oof.csv')
    assert len(oof)==140 and oof.observation_id.is_unique and oof.doi.nunique()==32
    assert oof.fold.between(1,5).all()
    m=pd.read_csv(REPORT/'same140_weighted_metrics.csv')
    for row in m.itertuples(index=False):
        key=f'{row.model}__{row.scenario}__{row.feature_set}__{row.weight_policy}'
        assert np.isclose(np.abs(oof.target_value-oof[key]).mean(),row.mae_mg_g,atol=1e-10)
    # Never pin a cross-environment historical MAE: sklearn/XGBoost versions can differ.
    assert m.mae_mg_g.notna().all() and (m.mae_mg_g>=0).all()

def test_sourceout_and_chen_ties_really_recorded():
    m=pd.read_csv(REPORT/'sourceout_weighted_metrics.csv')
    assert m.heldout_doi.nunique()==2
    assert set(m.source_n)=={7,29}
    rank=pd.read_csv(REPORT/'chen16_weighted_rank.csv')
    tied=rank[rank.n_top_tied>1]
    assert len(tied)>0 and tied.unique_top1.eq(False).all()
    assert tied.regret_if_unique.isna().all()

def test_saved_models_reloaded_predict():
    mm=json.loads((REPORT/'run_manifest.json').read_text(encoding='utf-8'))
    df=pd.read_csv(ROOT/'data/imports/v57_handoff_sources/Stage2_New36_Harmonized_RESEARCH.csv').iloc[:2]
    assert len(mm['models'])==6
    for a in mm['models']:
        path=ROOT/a['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==a['sha256']
        model=joblib.load(path)
        data=derive(df,a['feature_set'])
        cols=a['required_feature_columns'];cats=[c for c in cols if c in ['pollutant','feedstock_label_normalized','manufacturing_route','pyrolysis_atmosphere','activation_or_modification']]
        num=[c for c in cols if c not in cats]
        preds=model.predict(features(data,cols,num,cats))
        assert len(preds)==2 and np.isfinite(preds).all()
