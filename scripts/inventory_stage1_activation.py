"""
inventory_stage1_activation.py — Stage 1 활성화 공정 자료 인벤토리 (읽기 전용).

목적: 저장소 안에 이미 있는 Stage 1 자료 중, v60 master(256행/30 DOI)에 없는 활성화 공정
(physical/chemical) BET 행이 얼마나 있는지, 그 행들의 검증·승인 상태가 무엇인지 확인한다.
새 자료를 만들거나 값을 추정하지 않는다. 원본 파일을 변경하지 않는다.

출력: reports/v68_stage1_activation/inventory.json
실행: python -m scripts.inventory_stage1_activation
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FINAL = ROOT / "data/imports/Biochar_Stage1_v60/Biochar_Stage1_v60_Final"
MASTER = FINAL / "01_Stage1_Master_v60_256.csv"
LODO = {"BET": FINAL / "02_BET_AllData_FixedArchitecture_LODO.csv",
        "TPV": FINAL / "02_TPV_AllData_FixedArchitecture_LODO.csv"}
OTHER = {
    "v45_404": "data/imports/v46_phase8/v45_Stage1_404_BASELINE.csv",
    "v46_417": "data/imports/v46_phase8/Stage1_417_EXPLORATORY_UNAPPROVED.csv",
    "v52_421": "data/imports/v52_apple_series/Stage1_421_RESEARCH_NOT_ACTIVE.csv",
    "v55_431": "data/imports/v55_cumulative_sources/Stage1_431_RESEARCH_NOT_ACTIVE.csv",
    "v55_446": "data/imports/v55_cumulative_sources/Stage1_446_RESEARCH_NOT_ACTIVE.csv",
    "v57_450": "data/imports/v57_handoff_sources/Stage1_450_RESEARCH_NOT_ACTIVE.csv",
}
OUT = ROOT / "reports/v68_stage1_activation"
BET, TPV = "BET_m2_g_training_candidate", "TPV_cm3_g_training_candidate"
STATUS_HINTS = ("status", "role", "tier", "verified", "gate", "flag", "approved", "release", "exclu")
COMPOSITION = ["feedstock_C_pct_original", "feedstock_H_pct_original", "feedstock_O_pct_original",
               "feedstock_N_pct_original", "feedstock_ash_pct_original", "feedstock_VM_pct_original",
               "feedstock_FC_pct_original"]


def route_of(row: pd.Series) -> str:
    """predict_stage1_v60.route_from 과 같은 키워드 규칙."""
    s = (str(row.get("activation_method", "none")) + " " + str(row.get("activation_agent", "none"))).lower()
    if any(x in s for x in ["physical", "steam", "co2", "air", "recycled pyrolysis gas"]):
        return "physical"
    if any(x in s for x in ["chemical", "koh", "zncl", "h3po", "k2co", "k3po", "kmno", "impreg"]):
        return "chemical"
    return "none"


def routes(df: pd.DataFrame) -> pd.Series:
    computed = df.apply(route_of, axis=1)
    if "v59_route_family" in df.columns:
        return df["v59_route_family"].where(df["v59_route_family"].notna(), computed).astype(str)
    return computed.astype(str)


def norm_doi(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(float("nan"), index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def status_summary(df: pd.DataFrame) -> dict:
    out = {}
    for c in df.columns:
        if any(h in c.lower() for h in STATUS_HINTS):
            vc = df[c].dropna().astype(str).value_counts()
            if len(vc):
                out[c] = {k: int(v) for k, v in vc.head(4).items()}
    return out


def master_section(m: pd.DataFrame) -> dict:
    r = routes(m)
    bet, tpv = numeric(m, BET).notna(), numeric(m, TPV).notna()
    comp = m.reindex(columns=COMPOSITION).notna().any(axis=1)
    per_route = {}
    for rt in sorted(r.unique()):
        k = r == rt
        per_route[rt] = {"rows": int(k.sum()), "dois": int(m.loc[k, "doi"].nunique()),
                         "bet_rows": int((k & bet).sum()), "tpv_rows": int((k & tpv).sum()),
                         "rows_with_any_composition": int((k & comp).sum())}
    year = numeric(m, "paper_year")
    doi_year = pd.DataFrame({"doi": norm_doi(m["doi"]), "year": year}).dropna().groupby("doi").year.min()
    return {"rows": int(len(m)), "dois": int(m["doi"].nunique()), "columns": int(m.shape[1]),
            "per_route": per_route,
            "paper_year": {"rows_with_year": int(year.notna().sum()),
                           "dois_with_year": int(len(doi_year)),
                           "dois_by_year": {str(int(y)): int(n) for y, n in doi_year.value_counts().sort_index().items()}},
            "eng_columns_present": sorted(c for c in m.columns if c.startswith("eng_")),
            "status_columns": status_summary(m)}


def lodo_section(m: pd.DataFrame) -> dict:
    res = {}
    for tag, p in LODO.items():
        if not p.exists():
            res[tag] = {"exists": False}
            continue
        lo = pd.read_csv(p, low_memory=False)
        info = {"exists": True, "rows": int(len(lo)), "columns": list(lo.columns)}
        if "material_id" in lo.columns and "material_id" in m.columns:
            target = BET if tag == "BET" else TPV
            cand = m[numeric(m, target).notna()]
            missing = cand[~cand["material_id"].astype(str).isin(set(lo["material_id"].astype(str)))]
            cols = [c for c in ["material_id", "doi", "pyrolysis_temperature_C", "activation_temperature_C",
                                "v53_bet_process_temperature_status", "v53_table2_vs_table3_temperature"]
                    if c in missing.columns]
            info["target_rows_not_in_lodo"] = missing[cols].astype(str).to_dict(orient="records")
        res[tag] = info
    return res


def other_section(m: pd.DataFrame) -> dict:
    m_doi = set(norm_doi(m["doi"]))
    m_key = set(norm_doi(m["doi"]) + "::" + m["material_id"].astype(str)) if "material_id" in m.columns else set()
    res = {}
    for name, rel in OTHER.items():
        p = ROOT / rel
        if not p.exists():
            res[name] = {"exists": False}
            continue
        df = pd.read_csv(p, low_memory=False)
        if "doi" not in df.columns:
            res[name] = {"exists": True, "rows": int(len(df)), "note": "no doi column"}
            continue
        r = routes(df)
        d = norm_doi(df["doi"])
        bet = numeric(df, BET).notna()
        new_doi = ~d.isin(m_doi)
        key = d + "::" + df["material_id"].astype(str) if "material_id" in df.columns else d
        new_row = ~key.isin(m_key) if "material_id" in df.columns else new_doi
        per_route = {}
        for rt in sorted(r.unique()):
            k = r == rt
            per_route[rt] = {"rows": int(k.sum()), "bet_rows": int((k & bet).sum()),
                             "bet_rows_new_doi": int((k & bet & new_doi).sum()),
                             "new_dois": int(d[k & bet & new_doi].nunique()),
                             "bet_rows_master_doi_new_material": int((k & bet & ~new_doi & new_row).sum())}
        cand = df[(r != "none") & bet & new_row].copy()
        cand_routes = r[cand.index]
        by_doi = []
        for dd, g in cand.groupby(norm_doi(cand["doi"])):
            by_doi.append({"doi": dd, "rows": int(len(g)), "in_master_doi": bool(dd in m_doi),
                           "routes": {k: int(v) for k, v in cand_routes[g.index].value_counts().items()},
                           "activation_agents": sorted(set(g.get("activation_agent", pd.Series(dtype=str)).dropna().astype(str))),
                           "status": status_summary(g)})
        res[name] = {"exists": True, "rows": int(len(df)), "dois": int(d.nunique()),
                     "master_columns_present": int(sum(c in df.columns for c in m.columns)),
                     "master_columns_total": int(m.shape[1]),
                     "per_route": per_route, "activated_bet_candidates_not_in_master": by_doi}
    return res


def model_params() -> dict:
    out = {}
    try:
        import joblib
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            obj = joblib.load(FINAL / "MODEL_STAGE1_V60_BET_XGB_Eng59.joblib")
        out["top_keys"] = sorted(obj.keys()) if isinstance(obj, dict) else str(type(obj))
        router = obj["router"]
        out["router_keys"] = sorted(router.keys())
        out["experts"] = sorted(router.get("experts", {}).keys())
        est = router["global"].steps[-1][1]
        keep = ("n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree",
                "min_child_weight", "reg_lambda", "reg_alpha", "gamma", "objective", "random_state")
        out["global_estimator"] = type(est).__name__
        out["global_params"] = {k: v for k, v in est.get_params().items() if k in keep}
        out["pipeline_steps"] = [s[0] for s in router["global"].steps]
    except Exception as exc:  # 버전 불일치 등. 실패를 그대로 기록한다.
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(MASTER, low_memory=False)
    inv = {"master": master_section(m), "published_lodo": lodo_section(m),
           "other_sources": other_section(m), "v60_bet_model": model_params()}
    (OUT / "inventory.json").write_text(json.dumps(inv, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    # 짧은 요약만 출력 (상세는 inventory.json)
    print("MASTER", json.dumps(inv["master"]["per_route"]), "years", json.dumps(inv["master"]["paper_year"]["dois_by_year"]))
    print("ENG cols in CSV:", len(inv["master"]["eng_columns_present"]))
    for tag, info in inv["published_lodo"].items():
        print("LODO", tag, info.get("rows"), info.get("columns"), "not_in_lodo:", info.get("target_rows_not_in_lodo"))
    for name, info in inv["other_sources"].items():
        if not info.get("exists"):
            print(name, "missing")
            continue
        cands = info.get("activated_bet_candidates_not_in_master", [])
        print(name, "rows", info.get("rows"), "dois", info.get("dois"),
              "cols", info.get("master_columns_present"), "/", info.get("master_columns_total"),
              "| activated BET candidates:", sum(c["rows"] for c in cands),
              [(c["doi"], c["rows"], c["routes"], c["in_master_doi"]) for c in cands])
    print("MODEL", json.dumps(inv["v60_bet_model"], default=str))


if __name__ == "__main__":
    main()
