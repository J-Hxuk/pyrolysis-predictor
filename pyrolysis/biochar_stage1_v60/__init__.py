"""
biochar_stage1_v60 — Stage 1 v60 Final Package 통합 (독립·교체 가능 모듈).

이 패키지는 외부에서 전달된 `Biochar_Stage1_v60_Final_Package.zip`(BET·TPV XGBoost route
모델, 예측기, master CSV, 메타데이터)을 **기존 프로그램에 연결**한다.

설계 원칙 (사용자 지시):
  - 기존 모델·파일을 보존한다(이 패키지는 v41 RF 모델·biochar_stage1 을 건드리지 않는다).
  - Stage 1 v60 을 **독립 모듈**로 구성해 향후 패키지 교체만으로 업데이트할 수 있게 한다.
  - v60 패키지의 예측 로직(`predict_stage1_v60.py`)을 **재구현하지 않고 그대로 위임**한다.
  - Stage 2 연결·학습은 **수행하지 않는다**(스키마 인터페이스만 별도 모듈에서 설계).

환경 호환성 (중요):
  - v60 모델은 scikit-learn 1.8.0 / xgboost 3.4.1 로 저장되었다. 현재 프로젝트 venv 는
    sklearn 1.5.2 / xgboost 2.1.3 이며, 이 불일치에서는 전처리 역직렬화가 어긋나 **예측값이
    손상**된다(검증됨: BET 849 → 5). support_grade·경고는 순수 pandas 로직이라 재현되지만
    수치 예측은 신뢰할 수 없다.
  - 따라서 로드 시 호환성 게이트(`compat.py`)가 예제 예측 재현으로 자가 검증하고, 재현에
    실패하면 수치 예측을 **거부**한다. 손상된 값을 정상으로 반환하지 않는다.

Streamlit 을 import 하지 않는다(계산 전용).
"""

from __future__ import annotations

__all__ = ["extract", "compat", "predictor", "features"]
