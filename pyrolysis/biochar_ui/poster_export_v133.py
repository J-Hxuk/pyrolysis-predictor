"""Poster export utilities for the installed v113/v128/v132 single UI.

Read-only research visualization: no model training, artifact writes or state changes.
'Within observed ranges' is not a confidence interval or independent validation.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

STAGE2_SNAPSHOT = Path('v114_patch/v112_training_4795.csv')
# Keep these stable for both Altair and Matplotlib legends.
IN_RANGE = '관측 온도범위 내'
OUT_OF_RANGE = '관측 온도범위 밖'
NOT_VERIFIED = '관측 온도범위 미확인'


def _valid_bounds(series) -> tuple[float, float] | None:
    values = pd.to_numeric(pd.Series(series), errors='coerce').dropna()
    if values.empty:
        return None
    return float(values.min()), float(values.max())


def read_stage2_temperature_bounds(root: Path, pollutant: str) -> tuple[float, float] | None:
    """Bounds of recorded temperatures for THIS pollutant, not a global range.

    No fallback to other pollutants, and no extrapolated safe-range claim.
    """
    path = Path(root) / STAGE2_SNAPSHOT
    if not path.is_file():
        return None
    data = pd.read_csv(path, usecols=['pollutant', 'pyrolysis_temperature_C'], low_memory=False)
    return _valid_bounds(data.loc[data['pollutant'] == pollutant, 'pyrolysis_temperature_C'])


def stage1_temperature_bounds(models: Mapping) -> dict[str, tuple[float, float] | None]:
    ref = models.get('s1_reference', {})
    return {
        target: ref.get(target, {}).get('numeric', {}).get('pyrolysis_temperature_C')
        for target in ('BET', 'TPV')
    }


def _inside(value: float, bounds: tuple[float, float] | None) -> bool | None:
    if bounds is None:
        return None
    lo, hi = map(float, bounds)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    return bool(lo <= float(value) <= hi)


def status_for_temperature(temp: float, bet_bounds, tpv_bounds, stage2_bounds) -> dict:
    """Univariate train-data coverage, NOT joint support or prediction accuracy."""
    checks = {
        'BET': _inside(temp, bet_bounds),
        'TPV': _inside(temp, tpv_bounds),
        'Stage2': _inside(temp, stage2_bounds),
    }
    if False in checks.values():
        status = OUT_OF_RANGE
    elif None in checks.values():
        status = NOT_VERIFIED
    else:
        status = IN_RANGE
    return {
        '온도 범위 판정': status,
        'BET 온도범위': _describe(checks['BET']),
        'TPV 온도범위': _describe(checks['TPV']),
        'Stage2 오염물질 온도범위': _describe(checks['Stage2']),
    }


def _describe(value: bool | None) -> str:
    return '미확인' if value is None else ('범위 내' if value else '범위 밖')


def format_bounds(bounds: tuple[float, float] | None) -> str:
    return '미확인' if bounds is None else f'{float(bounds[0]):g}–{float(bounds[1]):g} °C'


def attach_coverage(df: pd.DataFrame, models: Mapping, stage2_bounds) -> pd.DataFrame:
    """Add descriptive range flags. Does not modify predictions or model inputs."""
    bounds = stage1_temperature_bounds(models)
    out = df.copy()
    diagnostics = [status_for_temperature(t, bounds['BET'], bounds['TPV'], stage2_bounds)
                   for t in out['제조온도 (℃)']]
    return pd.concat([out.reset_index(drop=True), pd.DataFrame(diagnostics)], axis=1)


def export_csv_with_inputs(df: pd.DataFrame, *, feedstock_input: str, feedstock_used: str,
                           process: Mapping, adsorption: Mapping, composition: Mapping,
                           model_sha256: str, properties_connected: bool,
                           stage1_warnings: str = '') -> bytes:
    """One row per prediction temperature, with fixed inputs and provenance columns."""
    out = df.copy()
    fixed = {
        '입력 원료명': feedstock_input,
        'Stage1 실제 원료명': feedstock_used,
        '유지시간 (min)': process.get('pyrolysis_hold_min'),
        '승온속도 (℃/min)': process.get('heating_rate_C_min'),
        '열분해 분위기': process.get('pyrolysis_atmosphere'),
        '활성화 방식': process.get('activation_method'),
        '오염물질': adsorption.get('pollutant'),
        'C0 (mg/L)': adsorption.get('C0_mg_L'),
        'Dose (g/L)': adsorption.get('dose_g_L'),
        '초기 pH': adsorption.get('initial_pH'),
        '흡착온도 (℃)': adsorption.get('adsorption_T_C'),
        '접촉시간 (min)': adsorption.get('contact_time_min'),
        'Stage2 BET/TPV 전달': '예' if properties_connected else '아니오',
        '학습 범위 판정 방법': '각 모델의 관측 제조온도 최소·최대(단변량)만 비교',
        '타깃 의미': 'Predicted reported adsorption capacity (strict qe 인증 아님)',
        '운영 모델 SHA256': model_sha256,
        '프로그램 출력 버전': 'v133 / v113 UI + v114 Stage2',
        '비온도 입력 경고': stage1_warnings,
    }
    for key, value in composition.items():
        fixed['원료 조성 '+str(key)] = value
    for key, value in fixed.items():
        out[key] = value
    return out.to_csv(index=False).encode('utf-8-sig')


def render_poster_figure(df: pd.DataFrame, *, image_format: str = 'png', dpi: int = 300) -> bytes:
    """3-panel paper-ready figure; vector SVG or print-resolution PNG.

    No speculative observed values, synthetic confidence intervals or optimality claims.
    """
    if image_format not in ('png', 'svg'):
        raise ValueError('Only PNG and SVG exports are supported.')
    if df.empty:
        raise ValueError('Cannot export an empty temperature sweep.')

    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.lines import Line2D
    from matplotlib import rc_context

    temp = pd.to_numeric(df['제조온도 (℃)'], errors='raise').to_numpy(float)
    status = df['온도 범위 판정'].fillna(NOT_VERIFIED)
    palette = {IN_RANGE: '#157f72', OUT_OF_RANGE: '#bd4d3a', NOT_VERIFIED: '#777777'}
    series = [
        ('BET (m²/g)', 'BET (m²/g)'),
        ('TPV (cm³/g)', 'TPV (cm³/g)'),
        ('Predicted reported capacity (mg/g)', '예측 흡착량 (mg/g)'),
    ]
    with rc_context({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
                     'axes.edgecolor': '#8d969c', 'axes.labelcolor': '#263845',
                     'xtick.color': '#263845', 'ytick.color': '#263845'}):
        fig = Figure(figsize=(9.4, 9.0), facecolor='white')
        FigureCanvasAgg(fig)
        axes = fig.subplots(3, 1, sharex=True)
        for ax, (ylabel, key) in zip(axes, series):
            yy = pd.to_numeric(df[key], errors='raise').to_numpy(float)
            ax.plot(temp, yy, color='#315a72', linewidth=1.8, zorder=2)
            for category, color in palette.items():
                mask = status.eq(category).to_numpy(bool)
                if np.any(mask):
                    ax.scatter(temp[mask], yy[mask], color=color, s=32, zorder=3,
                               edgecolors='white', linewidths=0.5)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.grid(True, color='#e5ebef', linewidth=0.7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(labelsize=9)
            if len(temp) == 1:
                ax.margins(x=0.3)
        axes[-1].set_xlabel('Pyrolysis temperature (°C)', fontsize=11)
        legend = [
            Line2D([0], [0], marker='o', color='none', markerfacecolor=c,
                   markeredgecolor='white', markersize=7, label=label)
            for label, c in [('Within observed ranges', palette[IN_RANGE]),
                             ('Outside at least one range', palette[OUT_OF_RANGE]),
                             ('Coverage not verified', palette[NOT_VERIFIED])]
        ]
        fig.legend(handles=legend, loc='upper center', ncol=3,
                   bbox_to_anchor=(0.51, 0.992), frameon=False, fontsize=9)
        fig.text(0.50, 0.012,
                 'Temperature min–max checks only; in-range does not imply validated accuracy.',
                 ha='center', va='bottom', fontsize=8, color='#666666')
        fig.subplots_adjust(left=0.22, right=0.97, top=0.935, bottom=0.085, hspace=0.19)
        buf = BytesIO()
        fig.savefig(buf, format=image_format, dpi=int(dpi), facecolor='white',
                    metadata={'Creator': 'Biochar v133 research export'} if image_format=='png' else None)
        return buf.getvalue()
