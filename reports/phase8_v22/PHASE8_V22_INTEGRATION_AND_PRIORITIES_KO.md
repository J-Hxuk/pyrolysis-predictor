# Phase 8 → 기존 Biochar Workbench v22 데이터 통합 감사 및 다음 수집 우선순위

## 1. 수행한 실제 작업

- 공동연구자가 제공한 `Biochar_Phase8_Integrated_Data_Package.zip`을 **별도 읽기 전용 스냅샷**으로 프로젝트 `data/imports/Biochar_Phase8_Integrated_Data_Package/` 아래 반영. 여러 중복 사본은 이 스냅샷 내부에서 그대로 유지하되 학습행 수에는 합산하지 않음.
- 현행 v60 256개 material_id와 Phase 8의 264개 material_id를 대조: 기존 256개 행의 전체 필드가 동일하며 신규 **8개** 확인. DOI 30 → 32.
- 현행 Core97의 observation_id와 Phase 8의 98개 adsorption_id를 대조: 기존 97개 주요 흡착 입력/타깃 유지, 신규 **1개** 확인. DOI 25 → 26.
- 기존 Stage 2 v15 입력 계약 53개 열에 맞춘 **98행 연구용 후보표** 생성 (`stage2_98_RESEARCH_CANDIDATE_NOT_TRAINED.csv`); 새 원료 조성 미보고값을 임의 대치하지 않았으며 Stage1 material_id 및 DOI를 일치시키고 새 행의 공정 분류를 원본의 명시적인 `single_stage_pyrolysis`에서만 매핑함.
- Streamlit 사이드바의 `G. Phase 8 Data Evidence`에 **새 관측 8+1, 전체 기록 검색, 주요 조성 확보율**을 표시. 기존 v15 A/B/C, Stage 1 v60, 실험용 v20 BET는 재학습/활성모델 교체하지 않음.
- 출처 원문 재확인 검증을 이번 작업에서 다시 수행한 것은 아님. 공동연구자의 검증 상태/승인 범위를 그대로 계승함.

## 2. 확정된 집계

| 구분 | 기존 | Phase 8 통합 | 순증 | 주의 |
|---|---:|---:|---:|---|
| Stage 1 시료 | 256 | 264 | +8 | +2 DOI |
| Stage 1 BET 훈련역할 True | 254 | 262 | +8 | 모델 업데이트 아님 |
| Stage 1 TPV 훈련역할 True | 191 | 199 | +8 | 모델 업데이트 아님 |
| Stage 2 흡착 관측 | 97 | 98 | +1 | +1 DOI |
| 신규 엄격 평형 확인 | 기존 Core 97과 별도 | 1 | +1 | 98행 내부의 부분집합, 99행 아님 |
| 원료 사전/별칭 | UI Core97 원료명 사용 | 84 원료/228 별칭 | 사전 확장 | 인식 건수≠학습 시료 증가 |
| Stage 1 recipe | 기존 v60 | 264 | — | 기존 adapter 조건부 가능 162/264 |
| Stage1–2 명시적 소재 연결 | — | 3 흡착행/2 소재 | — | 95/98행은 명시적 동일 시료 연결 없음 |

### Stage 1 신규 8시료

- DOI `10.1039/c8ra03958e`: 느타리·표고 폐배지 2원료 × 열분해온도 300/500/700°C = 6시료. BET 및 Total pore volume 실측 추가. 새로운 원료의 온도별 경향 연구에 활용 가능하나 *단일 논문의 6개 관측은 독립 DOI 6개가 아님*.
- DOI `10.3390/molecules30204121`: 아몬드 껍질 2시료, 600°C에서 비활성/화학처리 경로 비교. BET 및 Total pore volume 실측 추가.
- 8개 신규 시료에는 원료의 C/H/O/Ash/VM/FC 수치가 미보고로 남아 있어, 별도 v20 조성 학습 데이터의 저 VM·고 FC 빈 구간을 해소하지 못함.

### Stage 2 신규 1관측

- DOI `10.3390/molecules30204121`, 아몬드 껍질 비활성 바이오차, Pb(II), C0 500 mg/L, initial pH 4, dose 2 g/L, 흡착온도 22°C, 시간 60 min, 보고된 흡착량 103 mg/g. 물질수지 상한 250 mg/g 이하.
- `DIRECT_OBSERVED_QE_SEC2.4`로 새롭게 인증된 직접 관측 평형 값. **Langmuir qmax 103 mg/g가 아니다.**
- 이미 Stage 1에 존재하던 다른 아몬드 시료를 동일한 흡착실험의 연결 대상으로 임의 확장하지 않음.

## 3. 데이터 보완 우선순위 (데이터 양이 아니라 연구질문 달성 기준)

### P0 · 제조조건 → 물성 → 흡착량으로 연결되는 **동일 소재 실험계열** (가장 급함)

현재 98개 흡착 관측 중 Stage 1 소재와 **명시적으로 연결된 것은 3관측/2소재뿐**. Stage 1–2 연계 모델의 일반화 및 최적 제조조건 추천을 주장하기 위해서는 **동일 원료 배치/DOI에서 서로 다른 제조조건 최소 3개 이상의 실측 BET+Total TPV+동일 흡착조건 qe**와 세부 제조 recipe가 짝지어진 자료가 급함. `source_material_id`, `experiment_series_id`, activation route, 초기 pH, C0, dose, adsorption temperature, 평형 시간 근거까지 확보. 단순히 Stage 2 관측 수만 늘려서는 이 병목이 해결되지 않음.

### P1 · 원료 공업분석/원소분석의 **출처 다양성**과 분석 basis

| Stage 1 학습 역할 코호트 | 원료 CHON 완비 | VM/Ash/FC 완비 | VM만 존재 |
|---|---:|---:|---:|
| BET 262 | 36행 / 3 DOI | 17행 / 1 DOI | 17행 / 1 DOI |
| TPV 199 | 19행 / 2 DOI | **0행 / 0 DOI** | **0행 / 0 DOI** |

이 수치만으로는 조성의 새 DOI 일반화 오차보정을 주장하기 곤란함. 동일 원료의 **열분해 전** C/H/O/N/S, Ash/VM/FC/Moisture(실측), dry/daf 기준, 측정방법을 다른 DOI에서 추가 확보. `VM 20% / FC 60%` 등 기존 v20 BET 학습범위 밖에 있는 *실제 원료*를 우선 수집하되, 열분해 후 char 조성을 원료 조성으로 혼입하지 말 것. 8개 신규 행은 조성이 미보고이므로 이 빈 구간을 메우지 않음.

### P2 · 실험조건 전체가 알려진 **엄격 평형 qe**

현재 98개 reported uptake는 서로 다른 평형 근거의 혼합 코호트임. 이번 신규 엄격 평형 확인 1행을 기존 97행 모두의 평형 인증과 동일시하지 말 것. 실측 Ce, 시간별 plateau 또는 논문의 평형 확인, 측정시간/반복/오차, pH/C0/dose/온도를 모두 보존하면서 타깃을 구분. `qmax`·kinetic `Qt`·제거율 환산값은 자동 병합 금지.

### P3 · 독립 DOI 및 오염물질 범위 확장

현재 Stage 2: Cd(II) 41행/8 DOI, Cu(II) 22행/7 DOI, Pb(II) 14행/7 DOI, Ni(II) 11행/4 DOI, Phenol 10행/6 DOI. 우선 Ni(II)처럼 독립 DOI가 적고 조건 다양성이 부족한 타깃과, 제조경로 비교가 가능한 Pb(II) 계열을 보강. 같은 DOI에서 단지 pH만 바꾼 다수 행을 새 원료/논문 일반화의 증거로 세지 말 것.

### P4 · 제조 공정 메타데이터/재현 가능한 레시피

264 recipe 가운데 162개만 기존 v60 adapter로 조건부 실행 가능. 나머지 102개에 대해 실제 원문에 있는 승온속도, 유지시간, 분위기, 기체유량, 활성화제 투입비/순서/온도/시간을 확인. 미보고 조건을 평균 또는 다른 논문 값으로 자동 대체하지 않음. 특히 서로 다른 활성화 경로의 같은 원료 다중 시료가 유용함.

## 4. 이번 작업에서 **하지 않은 일** 및 후속 검증

- 기존 모델 파일을 새 데이터로 재학습하지 않았으며 배포 성능이 개선됐다고 주장하지 않음.
- v20 BET 모델은 별도 182행/24 DOI 비활성화 바이오매스 구성임. Phase 8 +8 시료를 자동 합치면 대상·공정·조성 basis가 일치한다는 보장이 없음.
- Stage 2 기존 Model A/B의 Nested CV 및 Chen16 추천 성능을 새 98행 성능으로 재표기하지 않음.
- 별도 연구용 코호트 98행을 실제 새 모델 학습에 사용하려면 **이전 97행과 동일 방법으로 DOI 그룹 분할, 타깃 상태 계층화, 물리 제약/원시·보정 오류 비교 및 재평가**가 필요함. 새 1 DOI가 기존 25 DOI에 대해 얼마나 영향을 미치는지 Paired OOF로 평가한 뒤에만 승격 판단.
- Stage 1 264행으로 새 모델을 학습할 때도 256 vs 264 같은 기존 DOI 기반 평가와 외부 DOI 평가를 분리해야 함.
- Streamlit 실사용 검증은 Windows 사용자 기기에서 필요함.

## 5. 파일/실행

`data/imports/Biochar_Phase8_Integrated_Data_Package/`에 원본 패키지 콘텐츠를 그대로 보존. 핵심 출처 관련 파일명 `*_UNCHANGED`는 유지.

`python -m scripts.build_phase8_v22_reports`로 보고서의 CSV/JSON 재생성.

`python -m pytest tests/test_phase8_v22.py -q`로 6개 무결성·계약 검증.

Streamlit: 기존 `biochar_stage2_app.py`의 `G. Phase 8 Data Evidence` 탭에서 확인.

**시험 결과:** 신규 6개 + 이전 v16/v19/v20/v21 대상 34개 = 40개 통과. 전체 회귀 및 Windows UI 화면은 미실행.
