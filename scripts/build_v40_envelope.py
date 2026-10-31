"""Generate deploy-agnostic input-applicability envelopes, read-only for active models."""
import argparse,json
from pathlib import Path
import pandas as pd
from scripts.v40_applicability import training_envelope
from scripts.evaluate_v40_research import S1_NUM,S1_CAT,S2_CAT,model_inputs

def run(root):
    root=Path(root);p=root/'reports/v40_new9'
    s1=pd.read_csv(p/'Stage1_Master_381_RESEARCH_STAGING.csv',low_memory=False)
    s2=pd.read_csv(p/'Stage2_reported_uptake_128_RESEARCH_CONDITIONAL.csv',low_memory=False)
    envelopes={}
    for target,col in [('BET','BET_m2_g_training_candidate'),('TPV','TPV_cm3_g_training_candidate')]:
        d=s1[s1[col].notna()]
        envelopes['Stage1_'+target]=training_envelope(d,S1_NUM,S1_CAT)
    for v in ['A','B']:
        inp=model_inputs(v);cats=[x for x in inp if x in S2_CAT];nums=[x for x in inp if x not in cats]
        envelopes['Stage2_'+v]=training_envelope(s2,nums,cats)
    (p/'research_training_envelopes.json').write_text(json.dumps(envelopes,ensure_ascii=False,indent=2),encoding='utf-8')
    return {k:{'n_rows':v['n_rows'],'n_doi':v['n_doi']} for k,v in envelopes.items()}
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,default=Path.cwd())
    z=a.parse_args();print(json.dumps(run(z.root),ensure_ascii=False,indent=2))
