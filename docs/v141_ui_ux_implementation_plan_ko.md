# v141 통합 시뮬레이터 UI/UX·안전성 개선 작업계획

> 범위: **표시, 입력 상태, 적용범위 안내, 비교, 내보내기 기능만 변경.** ML 모델·학습셋·하이퍼파라미터·분할 방식·성능지표는 수정하지 않는다.

## 변경 금지 영역 (freeze)
- `pyrolysis/biochar_stage1/unified_hgb_v134.py`: Poisson HGB 10변수, 학습 파이프라인 및 모델 설정
- `pyrolysis/biochar_stage2/stage2_blend_v139.py`: KNN(k=10) + weighted ET(50:50), 4,795행 및 모델 파일/해시
- `pyrolysis/biochar_stage2/manufacturing_rank_v140.py`: Manufacturing RF 학습·순위·지원 정책(59행/15계열/9 DOI), 특별히 필요한 경우 표시 계층에서만 변경
- 논문 데이터·평가지표·결과 그래프의 원자료·데이터 스냅샷은 변경 금지
- 학습 데이터의 신규 행 추가·묵시적 원료 치환·가상 신뢰구간 생성 금지

## P0 — 입력 결과 일치성 / 계산 경로 확인
1. **Stale result guard** (`pyrolysis/biochar_ui/v112_unified_page.py`):
   - 모든 결과/랭킹/내보내기는 **마지막 계산 당시의 입력 스냅샷**에 연결.
   - 원료 변경이나 조성·제조·흡착 입력 변경 후, 계산 전 이전 결과를 현재 결과처럼 표시하지 않는다.
   - 재계산 전에는 「이전 입력조건 결과」 표시 또는 결과 숨김. Streamlit `st.form` 내부 위젯의 상태 갱신 시점도 고려.
   - `st.session_state`의 과거 결과와 현재 입력 스냅샷 비교. 데이터 자체는 임의로 변경하지 않는다.
2. **조성 RF 적용 범위 명시** (`v112_unified_page.py`, `v112_single_engine.py`):
   - 공업분석 dry 기준 VM/Ash/FC 완비, 원소분석 daf 기준 C/H/N/O 완비일 때만 사용.
   - `inside_training_range`와 `can_explore_composition_ood`의 차이를 사용자에게 명시.
   - 조성만 학습 범위 밖인 경우 **명시적 opt-in**으로 연구용 추정; opt-in 없으면 기본 HGB 사용과 fallback 사유 표시.
   - 공정 범위 밖·활성화된 경로·잘못된 basis·불완전 원소분석은 조성 RF를 사용하지 않는다.
   - `BET path`, `TPV path`와 `Stage2 use_props`를 결과 근처에 표시.

## P1 — 실제 사용성을 높이는 기능
3. **변수 적용 상태 매트릭스**:
   - Stage1 HGB: 원료명/제조조건 사용, 조성 직접 미사용.
   - Composition RF: 지원 범위 내 BET에만 영향, TPV는 HGB.
   - Stage2: 원료명·분위기 직접 입력 안 함; 일부 오염물질에는 BET/TPV 미전달.
   - Manufacturing RF: 원료명/제조변수 사용, pH·투입량·조성 등은 순위 입력 아님.
4. **A/B 시나리오 비교**: 동일 예측 엔진을 각각 한 번 호출해 BET, TPV, reported capacity와 차이를 병렬 표시. 값의 미세 차이를 숨기지 않고 모델/변수/지원 범위 상태를 함께 표시. 비교를 실제 독립 검증으로 주장하지 않는다.
5. **학습범위/지원범위 미리보기**: 오염물질별 관측 행 수, 학습 범주 여부, 제조온도 범위를 '관측 데이터 설명'으로만 표시; 범위 내라고 정확도 인증 불가.

## P2 — 시각 디자인 / 자료 출력
6. 입력-결과 2열 레이아웃(가독성), 결과 3카드(BET/TPV/Capacity), 결과 아래 온도 Sweep/A·B/근거·내보내기 탭.
7. 기존 PNG(300dpi), SVG, CSV 기능 유지. 파일명에 원료·오염물질/조건 식별자 추가; 보고서에 실제 입력 스냅샷과 모델 경로·경고·버전 동봉.
8. 통합 사용자 진입점 `biochar_stage2_app.py`를 README에서 분명히 안내; 구형 `biochar_app.py`·`streamlit_app.py`와 혼동 방지.

## 반드시 추가·실행할 테스트
- 입력 변경 시 이전 결과 무효화/명시적 이전 결과 표기 (원료 변경, 분위기, 조성, pH/농도/투입량, 온도 등).
- OOD 조성 opt-in 없음 → HGB; 명시 동의 → 탐색적 조성 RF; 공정 범위 밖은 동의해도 거부.
- 동일 입력에 대한 **수정 전후 BET·TPV·흡착량·Manufacturing RF 순위의 수치 일치성** (표시 자릿수 제외).
- Phenol: 원료/분위기를 바꾼 뒤 결과가 실제로 갱신됨을 확인; 입력만 바꾼 경우 stale 표시.
- Zn(II)·미학습 오염물질·활성화 조건에서 지원하지 않는 RF 추천을 보이거나 강제로 실행하지 않음.
- CSV/PNG/SVG 결과 수치·제조온도/모델 버전/상태 일치성.
- `pytest tests/test_stage1_hgb_v134.py tests/test_stage2_blend_v139.py tests/test_manufacturing_rank_v140.py tests/test_stage1_bet_v21_ood.py -q`로 모델 동결 회귀확인.
- UI/상태 신규 테스트 별도 작성; Streamlit AppTest 또는 순수 상태 함수 테스트.

## 작업 규칙 / 배포
- **`main` 직접 수정 금지.** 현재 브랜치 `feature/v141-ui-ux-safety`에서만 구현.
- 작은 단위 커밋 → 테스트/실행확인 → Draft PR → 리뷰 후 merge.
- 기존 모형 성능 표기·메트릭 변경 없이, 사용자에게 불확실성과 한계의 올바른 표현을 보장.
- 외부 검증과 구별하여 '문헌 기반 개발평가', '탐색적 예측', '관측 범위 내'로 표시.
