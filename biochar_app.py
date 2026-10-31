"""
biochar_app.py — Biochar Prediction Platform 앱 진입점 (Stage1 BET + Stage2 인터페이스).

실행:
  streamlit run biochar_app.py

이 앱은 저장된 활성 BET 모델만 로드하며, 실행 시 재학습하지 않는다.
모델이 없으면 먼저 학습한다:
  python -m pyrolysis.biochar_stage1.cli train

계산 로직은 pyrolysis.biochar_stage1 / biochar_stage2 에 있다. 이 파일은 화면만 띄운다.
"""

from __future__ import annotations

from pyrolysis.biochar_ui.pages import render_app

render_app()
