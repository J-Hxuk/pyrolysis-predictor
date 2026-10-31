"""
biochar_stage2 — Stage2 흡착 성능(qe) 예측의 확장 인터페이스 (Phase 1).

지시서 §3:
  - Stage2 데이터셋은 별도 제작 중이다. 이번 단계에서 Stage2 ML 모델을 임의 데이터로
    학습하지 않는다.
  - 향후 데이터 연결을 위해 입력 스키마와 검증 모듈만 설계한다.
  - 미검증(평형 미입증) qe 는 자동 학습 대상으로 승인하지 않는다.
  - 실측 qe / Langmuir qmax / 모델 예측값 / 미검증 reported capacity 를 구분 관리한다.
  - Stage2 인터페이스는 Stage1 과 독립적으로 학습·검증·저장할 수 있도록 설계한다.
  - Stage1 의 예측 BET/TPV 를 Stage2 입력으로 쓸 때는 측정값/예측값 구분과 오차 전파를
    고려하도록 연결 구조를 분리한다.

Langmuir fitting(선택 기능)은 입력한 실측 등온흡착 데이터에 대한 피팅이며,
Stage2 머신러닝 예측과는 별개다.

이 패키지는 계산 전용이며 Streamlit 을 import 하지 않는다.
"""

from __future__ import annotations

__all__ = ["schema", "langmuir", "adapter"]
