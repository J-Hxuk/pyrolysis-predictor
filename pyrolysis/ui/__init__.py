"""
pyrolysis.ui — 한국어 입력·결과·오류 화면과 세션 상태 관리.

이 서브패키지만 Streamlit을 import할 수 있다.
계산 모듈(pyrolysis.*)은 Streamlit을 import하지 않는다.

주의: 이 __init__ 은 import 시점에 Streamlit 을 끌어오지 않는다.
show_single_prediction_screen 을 실제로 접근할 때만 screens(=streamlit) 를 지연 로드한다.
그래야 pyrolysis.ui.session 처럼 Streamlit 비의존 모듈을 import 해도
계산 계층 테스트의 Streamlit 비의존성 검사가 깨지지 않는다.
"""

from __future__ import annotations

__all__ = ["show_single_prediction_screen"]


def __getattr__(name):
    # PEP 562: 속성 접근 시점에만 Streamlit 의존 화면을 로드한다.
    if name == "show_single_prediction_screen":
        from pyrolysis.ui.screens import show_single_prediction_screen
        return show_single_prediction_screen
    raise AttributeError("module 'pyrolysis.ui' has no attribute " + repr(name))