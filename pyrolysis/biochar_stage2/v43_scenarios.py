"""v43: user-facing scenario snapshots & portable reports for the Biochar UI.

Pure presentation helpers: never train models, revise raw predictions or treat
clipping as learned adsorption physics. This module has no Streamlit dependency.
"""
from __future__ import annotations

from html import escape
import math
from typing import Mapping, Sequence

import pandas as pd

MAX_SCENARIOS = 12
SHOW_FIELDS = (
    ("feedstock_label_normalized", "원료"),
    ("pollutant", "오염물질"),
    ("C0_mg_L", "C₀ (mg/L)"),
    ("initial_pH", "초기 pH"),
    ("dose_g_L", "투입량 (g/L)"),
    ("contact_time_min", "접촉시간 (min)"),
    ("adsorption_T_C", "흡착온도 (°C)"),
    ("manufacturing_route", "제조경로"),
    ("pyrolysis_temperature_C", "열분해온도 (°C)"),
    ("pyrolysis_hold_min", "유지시간 (min)"),
)


def _plain(value):
    """Keep snapshots portable and free of numpy/pandas non-finite scalars."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            return None
        return float(value)
    try:
        # catches pandas.NA and numpy.nan without changing text values
        if bool(pd.isna(value)):
            return None
    except (ValueError, TypeError):
        pass
    return str(value)


def build_snapshot(name: str, inputs: Mapping, predictions: Mapping, *,
                   scope_status: Mapping | None = None) -> dict:
    """Capture a CURRENT inference run, preserving BOTH raw and bounded values."""
    label = str(name).strip()
    if not label or len(label) > 70:
        raise ValueError("시나리오 이름은 1~70자로 입력하세요")
    if not predictions:
        raise ValueError("저장할 예측 결과가 없습니다")
    features = {key: _plain(inputs.get(key)) for key, _ in SHOW_FIELDS}
    output = {}
    for model, pred in predictions.items():
        if model not in ("A", "B"):
            raise ValueError("지원하지 않는 모델")
        vals = [float(pred[k]) for k in ("qe_raw", "qe_clipped", "cap")]
        raw, clipped, cap = vals
        if not all(math.isfinite(x) for x in vals) or cap <= 0:
            raise ValueError("비유한 모델 결과 또는 잘못된 물질수지 상한")
        output[model] = {
            "qe_raw": raw,
            "qe_clipped": clipped,
            "cap": cap,
            "removal_bounded_pct": 100.0 * clipped / cap,
            "violates": bool(pred.get("violates", abs(raw - clipped) > 1e-9)),
            "scope": str((scope_status or {}).get(model, "미평가")),
        }
    return {"name": label, "inputs": features, "predictions": output}


def append_snapshot(history: Sequence[dict], snapshot: dict,
                    *, limit: int = MAX_SCENARIOS) -> list[dict]:
    if limit < 1:
        raise ValueError("양의 저장 한도 필요")
    # Chart row labels must be unique even if the text input retained its old value.
    names = {str(item.get("name")) for item in history}
    item = dict(snapshot)
    original = str(item["name"])
    if original in names:
        idx = 2
        while f"{original} ({idx})" in names:
            idx += 1
        item["name"] = f"{original} ({idx})"
    return [*list(history), item][-limit:]


def comparison_frame(history: Sequence[dict]) -> pd.DataFrame:
    """Show saved cases. Different pollutants are labeled, NEVER ranked."""
    cols = ["시나리오", "원료", "오염물질", "C₀ (mg/L)", "초기 pH", "투입량 (g/L)",
            "접촉시간 (min)", "흡착온도 (°C)", "제조온도 (°C)",
            "A 원시 qe (mg/g)", "A 제약 후 qe (mg/g)", "A 제거율 (%)",
            "B 원시 qe (mg/g)", "B 제약 후 qe (mg/g)", "B 제거율 (%)", "A 적용범위", "B 적용범위", "제약 경고"]
    rows = []
    for case in history:
        x, y = case["inputs"], case["predictions"]
        row = {"시나리오": case["name"], "원료": x.get("feedstock_label_normalized"),
               "오염물질": x.get("pollutant"), "C₀ (mg/L)": x.get("C0_mg_L"),
               "초기 pH": x.get("initial_pH"), "투입량 (g/L)": x.get("dose_g_L"),
               "접촉시간 (min)": x.get("contact_time_min"),
               "흡착온도 (°C)": x.get("adsorption_T_C"),
               "제조온도 (°C)": x.get("pyrolysis_temperature_C")}
        for model in ("A", "B"):
            data = y.get(model, {})
            row[f"{model} 원시 qe (mg/g)"] = data.get("qe_raw")
            row[f"{model} 제약 후 qe (mg/g)"] = data.get("qe_clipped")
            row[f"{model} 제거율 (%)"] = data.get("removal_bounded_pct")
            row[f"{model} 적용범위"] = data.get("scope")
        row["제약 경고"] = ", ".join(k for k, v in y.items() if v["violates"]) or "없음"
        rows.append(row)
    return pd.DataFrame(rows, columns=cols)


def _html_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p>저장된 항목 없음</p>"
    head = "".join(f"<th>{escape(str(x))}</th>" for x in df.columns)
    body = []
    for values in df.itertuples(index=False, name=None):
        tds = []
        for value in values:
            if pd.isna(value):
                text = "—"
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                text = f"{value:,.3f}" if not float(value).is_integer() else f"{value:g}"
            else:
                text = str(value)
            tds.append(f"<td>{escape(text)}</td>")
        body.append("<tr>" + "".join(tds) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _bar_svg(history: Sequence[dict]) -> str:
    """Portable, no-network graphic. Bars use bounded qe and label the model."""
    if not history:
        return ""
    width, left, right = 940, 245, 60
    height = 55 + 52 * len(history)
    max_value = max((max(float(v.get("qe_clipped", 0)) for v in case["predictions"].values())
                     for case in history), default=1.0)
    scale = (width - left - right) / max(max_value, 1e-9)
    chunks = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
              f'role="img" aria-label="저장 조건별 제약 후 흡착량 비교" style="max-width:100%">',
              '<text x="240" y="18" font-size="13" fill="#183044">물질수지 제약 후 qe (mg/g) · 수치 비교 (추천 순위 아님)</text>']
    colors = {"A": "#267b96", "B": "#ce9850"}
    for i, case in enumerate(history):
        y = 46 + i * 52
        name = escape(str(case["name"])[:32])
        chunks.append(f'<text x="8" y="{y+11}" font-size="12" fill="#183044">{name}</text>')
        for key, off in (("A", 0), ("B", 19)):
            if key not in case["predictions"]:
                continue
            val = max(0., float(case["predictions"][key]["qe_clipped"]))
            w = val * scale
            chunks.append(f'<rect x="{left}" y="{y+off}" width="{w:.2f}" height="14" rx="2" fill="{colors[key]}"/>')
            chunks.append(f'<text x="{left+w+5:.2f}" y="{y+off+11}" font-size="10" fill="#183044">{key}: {val:.2f}</text>')
    chunks.append('</svg>')
    return ''.join(chunks)


def _sweep_svg(frame: pd.DataFrame) -> str:
    """Embedded SVG curves, derived ONLY from saved sweep data."""
    if frame.empty or not {"x", "qe_clipped", "model"}.issubset(frame.columns):
        return ""
    series = frame.copy()
    series["x"] = pd.to_numeric(series["x"], errors="coerce")
    series["qe_clipped"] = pd.to_numeric(series["qe_clipped"], errors="coerce")
    series = series.dropna(subset=["x", "qe_clipped"])
    if series.empty:
        return ""
    xmin, xmax = float(series.x.min()), float(series.x.max())
    ymin, ymax = min(0.0, float(series.qe_clipped.min())), float(series.qe_clipped.max())
    if xmax <= xmin: xmax = xmin + 1
    if ymax <= ymin: ymax = ymin + 1
    width, height, left, top, inner_w, inner_h = 940, 300, 64, 32, 790, 210
    chunks = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
              f'role="img" aria-label="슬라이더 변수별 제약 후 qe 변화 곡선" style="max-width:100%">',
              f'<path d="M{left},{top} V{top+inner_h} H{left+inner_w}" stroke="#667e8a" fill="none"/>',
              f'<text x="{left}" y="{top+inner_h+25}" font-size="12">x: {xmin:g}–{xmax:g}</text>',
              f'<text x="{left+340}" y="{top+inner_h+25}" font-size="12">제약 후 qe (mg/g): {ymin:.2f}–{ymax:.2f}</text>']
    for model, color in (("A", "#267b96"), ("B", "#ce9850")):
        part = series[series["model"] == model].sort_values("x")
        if part.empty:
            continue
        points = []
        for x, q in zip(part.x, part.qe_clipped):
            px = left + (float(x) - xmin) / (xmax - xmin) * inner_w
            py = top + inner_h - (float(q) - ymin) / (ymax - ymin) * inner_h
            points.append(f"{px:.2f},{py:.2f}")
        chunks.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        chunks.append(f'<text x="{left+inner_w+12}" y="{top+30*(1 if model=="A" else 2)}" fill="{color}" font-size="14">Model {model}</text>')
    chunks.append('</svg>')
    return ''.join(chunks)


def printable_report(history: Sequence[dict], *,
                     sensitivity: pd.DataFrame | None = None,
                     sweep_label: str = "") -> str:
    """Self-contained HTML usable offline, including printing to PDF in browser."""
    data = comparison_frame(history)
    if len(history) == 0:
        raise ValueError("보고서로 저장할 시나리오가 없습니다")
    pollutants = {case["inputs"].get("pollutant") for case in history}
    note = ("서로 다른 오염물질을 포함합니다. 흡착량 크기로 재료 우열을 판정하지 마세요."
            if len(pollutants) > 1 else
            "여러 입력변수가 함께 바뀌었다면 결과 차이를 특정 변수의 영향으로 해석하지 마세요.")
    sensitivity_html = ""
    if sensitivity is not None and not sensitivity.empty:
        chosen = [c for c in ("x", "model", "qe_raw", "qe_clipped", "cap", "violates")
                  if c in sensitivity.columns]
        sensitivity_html = ("<h2>변수별 모델 응답</h2>" +
                            f"<p>변화 변수: {escape(sweep_label)} · 아래 선은 물질수지 제약 후 qe</p>" +
                            _sweep_svg(sensitivity) + _html_table(sensitivity[chosen]))
    style = """<style>
    body{font-family:Arial,'Malgun Gothic',sans-serif;max-width:1100px;margin:32px auto;color:#183044;line-height:1.55}
    h1{border-bottom:3px solid #275a78;padding-bottom:14px}h2{margin-top:34px}
    .note{background:#f3f6f8;padding:12px 16px;border-left:3px solid #738ea0}
    table{border-collapse:collapse;width:100%;font-size:12px;margin:12px 0 30px}
    td,th{border:1px solid #d6e1e7;padding:7px 8px;text-align:left}
    th{background:#eaf0f4}tr:nth-child(even){background:#f8fafb}
    @page{size:A4 landscape;margin:10mm}
    @media print{body{margin:0}table{font-size:9px}tr{break-inside:avoid}}
    </style>"""
    return ("<!doctype html><html lang='ko'><head><meta charset='utf-8'>" +
            "<title>Biochar · 실험조건 비교 보고서</title>" + style + "</head><body>" +
            "<h1>Biochar · 실험조건 비교 보고서</h1>" +
            "<p>저장된 시나리오: " + str(len(history)) + "개 · 모델 응답 비교</p>" +
            "<h2>저장된 실험조건 비교 그래프</h2>" + _bar_svg(history) +
            "<h2>입력조건 및 예측 결과</h2>" + _html_table(data) +
            "<p class='note'>" + escape(note) + "</p>" + sensitivity_html +
            "<h2>해석 범위</h2><p>예측은 저장된 머신러닝 모델의 추정값이며 실험 결과가 아닙니다. " +
            "'제약 후 qe'는 물질수지 0 ≤ qe ≤ C₀/Dose를 적용한 값으로, 새로운 모델의 학습 성능을 의미하지 않습니다. " +
            "Model A는 원료·제조조건을 사용하지 않고, Model B는 이 정보를 사용하지만 제조조건 최적화는 검증되지 않았습니다. " +
            "사용자는 독립 실험으로 확인해야 합니다.</p>" +
            "</body></html>")


def comparison_csv(history: Sequence[dict]) -> bytes:
    return comparison_frame(history).to_csv(index=False).encode("utf-8-sig")
