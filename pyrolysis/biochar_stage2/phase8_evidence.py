"""Phase 8: audited, immutable external evidence; never mutates active model training.

A canonical snapshot of the colleague's verified data package lives at
`data/imports/Biochar_Phase8_Integrated_Data_Package`.
This module is deliberately standard-library only and safe to import offline.
"""
from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

PHASE_DIR = "data/imports/Biochar_Phase8_Integrated_Data_Package"
CORE97_PATH = "data/imports/Stage1_Stage2_Model_Ready_v15/Stage1_Stage2_Model_Ready_v15_20261029/03_Stage2_QE_Core_97.csv"
V60_PATH = "data/imports/Biochar_Stage1_v60/Biochar_Stage1_v60_Final/01_Stage1_Master_v60_256.csv"


def read_csv(path: Path) -> list[dict[str,str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def phase8_path(base: Path) -> Path:
    return Path(base) / PHASE_DIR


def _filled(r: dict[str,str], col: str) -> bool:
    v = str(r.get(col, "")).strip()
    return bool(v) and v.upper() not in {"NR", "NA", "N/A", "NONE", "NOT REPORTED", "NAN"}


def _ids(rows: list[dict[str,str]], field: str) -> set[str]:
    names = [r[field] for r in rows]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate {field} detected")
    return set(names)


def audit_phase8(base: Path) -> dict:
    """Compare original current v60 and Core97 with Phase8, preserving row roles.

    This audit is not original-paper validation, model retraining, or external CV.
    """
    base = Path(base)
    imported = phase8_path(base)
    if not imported.is_dir():
        raise FileNotFoundError(f"Phase8 snapshot missing: {imported}")
    s1_old = read_csv(base / V60_PATH)
    s2_old = read_csv(base / CORE97_PATH)
    s1 = read_csv(imported / "data/stage1/Stage1_Master_264_UNCHANGED.csv")
    s2 = read_csv(imported / "data/stage2/Stage2_Reported_Uptake_98_UNCHANGED.csv")
    strict = read_csv(imported / "data/stage2/Stage2_Strict_Newly_Verified_1_UNCHANGED.csv")
    roles = read_csv(imported / "data/stage1/Stage1_Target_Training_Roles_UNCHANGED.csv")
    recipes = read_csv(imported / "data/recipes/Manufacturing_Recipe_Candidates.csv")
    dictionary = read_csv(imported / "data/feedstocks/Biomass_Name_Dictionary.csv")
    aliases = read_csv(imported / "data/feedstocks/Aliases.csv")

    old1_id = _ids(s1_old, "material_id")
    new1_id = _ids(s1, "material_id")
    old2_id = _ids(s2_old, "observation_id")
    new2_id = _ids(s2, "adsorption_id")
    _ids(strict,"adsorption_id")
    _ids(roles,"material_id")
    _ids(recipes,"material_id")
    if old1_id - new1_id or old2_id - new2_id:
        raise ValueError("Phase8 missing previously existing rows; refusing merge")
    old1_lookup={r["material_id"]:r for r in s1_old}
    for r in s1:
        old = old1_lookup.get(r["material_id"])
        if old is not None and old != r:
            raise ValueError(f"Existing Stage1 row unexpectedly modified: {r['material_id']}")
    # Read-only Stage2 is a different (lossy) schema, so match core values by ID.
    old2_lookup = {r["observation_id"]:r for r in s2_old}
    fields = [ ("pollutant","pollutant"), ("C0_mg_L","C0_mg_L"),
               ("initial_pH","initial_pH"), ("dose_g_L","dose_g_L"),
               ("target_value","qe_reported_mg_g") ]
    for r in s2:
        old=old2_lookup.get(r["adsorption_id"])
        if old is not None:
            for l,rr in fields:
                if str(old[l]).strip() != str(r[rr]).strip():
                    try:
                        same=abs(float(old[l])-float(r[rr]))<1e-8
                    except (ValueError,TypeError):
                        same=False
                    if not same:
                        raise ValueError(f"Core97 mismatch for {r['adsorption_id']} {l}")
    added1 = [r for r in s1 if r["material_id"] not in old1_id]
    added2 = [r for r in s2 if r["adsorption_id"] not in old2_id]
    if {r["adsorption_id"] for r in strict} != {r["adsorption_id"] for r in added2}:
        raise ValueError("strict row is not exactly the newly added Stage2 row")
    role_map={r["material_id"]:r for r in roles}
    for r in added1:
        role=role_map[r["material_id"]]
        if role["BET_training_use"] != "True" or role["TPV_training_use"] != "True":
            raise ValueError("New Stage1 observation is not approved for both target cohorts")
    fresh_doi=sorted({r["doi"] for r in added1}-{r["doi"] for r in s1_old})
    summary = {
        "source_name": "Biochar_Phase8_Integrated_Data_Package", "model_retrained":False,
        "model_deployment_changed":False,
        "stage1":{"previous":len(s1_old),"integrated":len(s1),"added":len(added1),
                  "doi_previous":len({r['doi'] for r in s1_old}),"doi_integrated":len({r['doi'] for r in s1}),
                  "new_dois":fresh_doi,"bet_approved":sum(r['BET_training_use']=='True' for r in roles),
                  "tpv_approved":sum(r['TPV_training_use']=='True' for r in roles)},
        "stage2":{"previous":len(s2_old),"integrated":len(s2),"added":len(added2),
                  "doi_previous":len({r['doi'] for r in s2_old}),"doi_integrated":len({r['doi'] for r in s2}),
                  "strict_new":len(strict),"pollutants":dict(sorted(Counter(r['pollutant'] for r in s2).items()))},
        "feedstock":{"dictionary_rows":len(dictionary),"alias_rows":len(aliases)},
        "recipes":{"total":len(recipes),"adapter_allowed":sum(r.get('stage1_adapter_allowed')=='True' for r in recipes)},
        "composition_coverage":{},
        "stage1_new": [{"material_id":r['material_id'],"doi":r['doi'],
                        "feedstock":r['feedstock'],"pyrolysis_temperature_C":r['pyrolysis_temperature_C'],
                        "BET_m2_g":r['BET_m2_g_training_candidate'],
                        "TPV_cm3_g":r['TPV_cm3_g_training_candidate']} for r in added1],
        "stage2_new": [{"adsorption_id":r['adsorption_id'],"doi":r['doi'],
                        "pollutant":r['pollutant'],"qe_mg_g":r['qe_reported_mg_g'],
                        "C0_mg_L":r['C0_mg_L'],"dose_g_L":r['dose_g_L'],
                        "source_target_status":r['source_target_status']} for r in added2],
    }
    names={'C':'feedstock_C_pct_original','H':'feedstock_H_pct_original',
           'O':'feedstock_O_pct_original','N':'feedstock_N_pct_original',
           'Ash':'feedstock_ash_pct_original','VM':'feedstock_VM_pct_original',
           'FC':'feedstock_FC_pct_original'}
    for cohort,items in [('BET',[r for r in s1 if role_map[r['material_id']]['BET_training_use']=='True']),
                         ('TPV',[r for r in s1 if role_map[r['material_id']]['TPV_training_use']=='True'])]:
        counts={name:{"n":sum(_filled(r,col) for r in items),
                      "doi":len({r['doi'] for r in items if _filled(r,col)})} for name,col in names.items()}
        counts['CHON_complete']={"n":sum(all(_filled(r,names[c]) for c in 'CHON') for r in items),
                                  "doi":len({r['doi'] for r in items if all(_filled(r,names[c]) for c in 'CHON')})}
        counts['proximate_VM_Ash_FC_complete']={"n":sum(all(_filled(r,names[c]) for c in ['VM','Ash','FC']) for r in items),
                                  "doi":len({r['doi'] for r in items if all(_filled(r,names[c]) for c in ['VM','Ash','FC'])})}
        summary['composition_coverage'][cohort]={"total":len(items),"fields":counts}
    summary['limitations']=[
      'Phase8 added 8 Stage1 rows and 1 Stage2 row, NOT 264+98 independent new observations.',
      'No training/deployed model is updated by importing this package.',
      'The new strict equilibrium observation is part of Stage2=98, not a separate 99th row.',
      'Most Stage1-Stage2 observations lack an approved material-level bridge.',
      'Stage1 composition values with non-comparable analysis basis must not be silently harmonized.',
      'Research ranking is not a validated global optimum.'
    ]
    return summary


def new_records(base: Path) -> tuple[list[dict],list[dict]]:
    r=audit_phase8(base)
    return r['stage1_new'], r['stage2_new']


def build_stage2_core98_candidate(base: Path) -> list[dict[str,str]]:
    """Create an A/B/C schema-matched research candidate, never train or activate.

    The single new observation has verified measured material descriptors, but
    NO artificial raw-biomass composition. Training requires new DOI CV.
    """
    base=Path(base)
    audit_phase8(base)  # fail closed if lineage/invariants are inconsistent
    original=read_csv(base / CORE97_PATH)
    s1=read_csv(phase8_path(base) / 'data/stage1/Stage1_Master_264_UNCHANGED.csv')
    s2=read_csv(phase8_path(base) / 'data/stage2/Stage2_Reported_Uptake_98_UNCHANGED.csv')
    s1idx={r['material_id']:r for r in s1}
    original_ids={r['observation_id'] for r in original}
    result=[dict(r) for r in original]
    for record in s2:
        if record['adsorption_id'] in original_ids:
            continue
        material=s1idx.get(record['material_key'])
        if material is None:
            raise ValueError('new Stage2 material has no exact Phase8 material ID')
        if material['doi'].lower()!=record['doi'].lower():
            raise ValueError('Stage1 and Stage2 DOI mismatch')
        if material['process_stage']!='single_stage_pyrolysis':
            raise ValueError('cannot map manufacturing_route for new material')
        new={field:'' for field in original[0].keys()}
        new.update({
            'observation_id':record['adsorption_id'],
            'doi_group':record['doi'],
            'feedstock_group_for_novelty_check':material['feedstock'].strip().casefold(),
            'sample_group':record['material_key'],
            'experiment_series_id':record['source_experiment_group'],
            'pollutant':record['pollutant'],
            'C0_mg_L':record['C0_mg_L'],
            'initial_pH':record['initial_pH'],
            'dose_g_L':record['dose_g_L'],
            'adsorption_T_C':record['adsorption_T_C'],
            'contact_time_min':record['contact_time_min'],
            'material_uid':record['material_key'],
            'doi':record['doi'],
            'feedstock_raw':material['feedstock'],
            'feedstock_label_normalized':material['feedstock'].strip().casefold(),
            'manufacturing_route':'SINGLE_STAGE_PYROLYSIS',
            'process_step_count':'1',
            'pyrolysis_temperature_C':material['pyrolysis_temperature_C'],
            'pyrolysis_hold_min':material['pyrolysis_hold_min'],
            'heating_rate_C_min':material['heating_rate_C_min'],
            'pyrolysis_atmosphere':material['pyrolysis_atmosphere'],
            'BET_measured_m2_g':record['BET_measured_m2_g'],
            'TPV_measured_cm3_g':record['TPV_measured_cm3_g'],
            'raw_composition_basis':'NOT_REPORTED_RAW_FEEDSTOCK',
            'stage1_v60_eligible':'NOT_IN_V60',
            'target_name':'qe_reported_mg_g',
            'target_value':record['qe_reported_mg_g'],
            'target_unit':'mg/g',
            'target_origin':'PHASE8_DIRECT_REPORTED_STRICT_EQUILIBRIUM',
            'model_use_tier':'PHASE8_RESEARCH_CANDIDATE_NOT_TRAINED',
            'target_qualification':record['source_target_status'],
            'physical_cap_pass':'YES' if float(record['qe_reported_mg_g'])<=float(record['C0_mg_L'])/float(record['dose_g_L']) else 'NO',
            'mass_balance_cap_mg_g':record['qe_mass_balance_upper_mg_g'],
            'source_url':record['source_url'],
            'warnings':record['warnings'],
        })
        result.append(new)
    return result
