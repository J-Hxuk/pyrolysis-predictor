"""
biochar_stage1 — 실제 논문 데이터 기반 Stage1 BET 예측 플랫폼 (Phase 1).

이 패키지는 지시서(Phase 1 개발 지시서)에 따라 다음 책임을 분리해 구현한다.

  - extract        : v41 ZIP 에서 필요한 파일만 선택 추출 (원본 미변경)
  - data_loader    : 데이터 불러오기 (학습과 분리)
  - data_validation: 컬럼·단위·결측·적격성 검사
  - features       : 입력변수 계약 (10개 입력 + BET 출력)
  - evaluation     : eval37 GroupKFold 재현 평가
  - training       : 최종 RF 모델 학습·버전 저장
  - inference      : 저장 모델 로드·예측·적용 범위 판정
  - model_registry : 모델·버전 관리

Stage2 연결과 Langmuir 피팅은 pyrolysis.biochar_stage2 에 있다.
이 패키지는 계산 전용이며 Streamlit 을 import 하지 않는다.
"""

from __future__ import annotations

__all__ = [
    "features",
    "data_loader",
    "data_validation",
    "evaluation",
    "training",
    "inference",
    "model_registry",
]
