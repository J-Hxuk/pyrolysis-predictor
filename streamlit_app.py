"""
streamlit_app.py — 앱 진입점.

역할:
  - 배포 묶음 읽기 및 준비 상태 확인
  - pyrolysis.ui 화면 함수 호출

이 파일 자체는 Streamlit 을 import한다.
예측 계산 로직은 pyrolysis.* 패키지에 있으며 여기에 복사하지 않는다.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from pyrolysis.artifacts import DEMO_BUNDLE_MARKER
from pyrolysis.ui import show_single_prediction_screen

# 페이지 기본 설정
st.set_page_config(
    page_title="열분해 예측 시스템",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 배포 묶음 확인 ────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).parent
_BUNDLE_DIR = _REPO_ROOT / "deployment" / "bundles"
_ACTIVE_POINTER = _REPO_ROOT / "deployment" / "active.json"

def get_active_bundle_dir() -> Path | None:
    """활성 배포 묶음 디렉터리를 반환한다.

    Task 5에서는 demo 묶음만 지원한다.
    deployment/bundles/ 아래에 demo로 시작하는 디렉터리를 찾는다.
    """
    if not _BUNDLE_DIR.exists():
        return None

    # demo 묶음 찾기
    for item in _BUNDLE_DIR.iterdir():
        if item.is_dir() and item.name.startswith("demo"):
            # contract.json 확인
            contract_file = item / "contract.json"
            if contract_file.exists():
                import json
                try:
                    with contract_file.open(encoding="utf-8") as f:
                        contract = json.load(f)
                    if contract.get("bundle_type") == DEMO_BUNDLE_MARKER:
                        return item
                except Exception:
                    continue

    return None

# ── 메인 화면 ────────────────────────────────────────────────────────────────

bundle_dir = get_active_bundle_dir()

if bundle_dir is None:
    # 배포 묶음이 없는 경우
    st.title("🔬 열분해 생성물·촉매 성능 예측")
    st.caption("연구용 예측 시스템 | 실제 모델 준비 중")

    st.info(
        "**시스템 준비 중**\n\n"
        "현재 배포 묶음이 없습니다. "
        "학습·평가를 완료한 뒤 `python -m pyrolysis.cli export` 명령으로 "
        "배포 묶음을 생성하고 `activate` 명령으로 활성화하세요.\n\n"
        "자세한 절차는 `docs/setup.md` 를 참고하세요.",
        icon="⚙️",
    )

    st.markdown("---")
    st.markdown(
        "**구현 진행 상황**\n"
        "- ✅ 작업 1: 프로젝트 구조와 개발 환경 구성\n"
        "- ✅ 작업 2: 빈 연구 설정과 최소 계약 구현\n"
        "- ✅ 작업 3: 가상 자료로 최소 학습·독립 평가 경로 연결\n"
        "- ✅ 작업 4: 가상 모델 저장·다시 읽기·공통 예측 연결\n"
        "- 🔄 작업 5: 로컬 단일 예측 화면으로 첫 흐름 완성 (진행 중)\n"
        "- ⬜ 작업 6–24: 전체 기능 구현\n"
    )
else:
    # 배포 묶음이 있는 경우 - 단일 예측 화면 표시
    show_single_prediction_screen(bundle_dir)
