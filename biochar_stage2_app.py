"""
biochar_stage2_app.py — Stage 2 흡착성능 예측 앱 진입점.

실행:
  streamlit run biochar_stage2_app.py

저장된 Stage 2 모델만 로드하며 실행 시 재학습하지 않는다.
모델이 없으면 먼저 학습한다:
  python -m pyrolysis.biochar_stage2.v15_cli train

계산은 pyrolysis.biochar_stage2.* 에 있다. 이 파일은 화면만 띄운다.
"""

from __future__ import annotations

from pyrolysis.biochar_ui.stage2_pages import render_stage2_app

render_stage2_app()
