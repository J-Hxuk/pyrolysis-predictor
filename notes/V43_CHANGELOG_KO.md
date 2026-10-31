# v43 변경 이력

- UI 목적: 학술대회 시연 경로 완성. 과도한 학술 검증 대신 입력→예측→변수별 그래프→시나리오 비교→보고서 저장 제공.
- `pyrolysis/biochar_ui/stage2_pages.py`: 조성 선택입력 접기; 시나리오 저장·삭제·비교표·막대그래프·내보내기; 제거율 그래프 물질수지 제약 후 값으로 표시. 기준값 고정 버튼 문구 명확화.
- `pyrolysis/biochar_stage2/v43_scenarios.py`: 시나리오 스냅샷·중복 이름 처리·비교표·CSV·SVG 내장 HTML 보고서.
- `tests/test_v43_scenarios.py`: 저장·유한성·모델 누락·CSV·보고서 내용·HTML escaping·시나리오 한도 테스트.
- **변경 없음**: 활성 모델, 학습·검증 데이터, 모델 알고리즘, 신뢰성 게이트, 사용자 연구 파일, Stage 1 시스템.
- 검증 범위: 새 기능과 관련된 순수 코드 32개 통과. Streamlit 실제 브라우저와 Windows 전체 회귀 테스트는 미검증.
