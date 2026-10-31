# Biochar v24 — 4개 신규 논문·보충자료 원문 검증 및 연구용 데이터 반영

**데이터 등록(스테이징)만 수행했습니다. 기존 Stage 1 v60·실험용 BET v20·Stage 2 A/B/C 저장 모델과 학습 결과는 변경하지 않습니다.**

## 반영 범위

- 신규 원문 4편: `10.3390/ma14216566`, `10.3390/toxics12010008`, `10.3390/molecules27154868`, `10.1016/j.heliyon.2023.e14930`.
- 앞선 원문 2편의 아직 학습 미반영 후보: `10.3390/w13060868` (포멜로), `10.1039/d5va00202h` (하수슬러지).
- 기존 Phase 8 물성 264행 **변경 없음**; 새로운 물성 테이블 33행과 원본을 이어놓은 읽기 전용 Stage 1 297행 생성.
- 기존 Stage 2 reported uptake 98행 **변경 없음**; 가금분 직접 보고 흡착량 54행, 포멜로 실험 `Qe_exp` 18행을 별도 등록. 총 72행 중 물질수지 기준 69행 통과, 3행 격리. 기존과 결합한 167행 테이블은 **학습 승인판이 아닌 스테이징**.
- 새 논문 Langmuir qmax 10행과 이전 논문 22행을 결합한 `all_qmax_32_FITTED_NOT_QE.csv`는 **qe와 절대 합치지 않음**.

## 논문별 데이터 계약

| DOI | 원료 / 공정 | BET+TPV 직접 실측 | reported uptake | 별도 qmax | 주의 사항 |
|---|---|---:|---:|---:|---|
| 10.3390/ma14216566 | poultry manure; 425,575,725 °C | 3 | 54 (51 수지 통과) | 0 | 100 mg/L Cd(II) 3행이 C0/dose 위반, 최종 pH >9.6의 침전 위험 |
| 10.3390/toxics12010008 | urban green waste; 350–650 °C | 4 | 그래프만; 미추출 | 4 | 표 3의 TPV 변동은 본문 단조증가 서술과 불일치, raw 조성은 미보고 |
| 10.3390/molecules27154868 | cotton stalks+LDPE; 400–500 °C | 10 (12조건 중 2조건 BET/TPV 미보고) | 그래프만; 미추출 | 4 | 표 2의 TPV 단위 10^-2 cm³/g 변환, Methods에 Corn/Cotton 명칭 충돌 |
| 10.1016/j.heliyon.2023.e14930 | HTC(180 °C)+KHCO3 activation | 6 (원물 1종 제외) | 그래프만; 미추출 | 2 | 180 °C는 건식 열분해온도가 아님, 일반 열분해 모델에 직접 병합 금지 |
| 10.3390/w13060868 (이전) | pomelo peel; 일반/자성 개질 | 6 | `Qe_exp` 18 | 기존 별도 | 원문의 `mmol/g` 오기를 사용자 확인에 근거해 `μmol/g`로 해석; 문서화된 출판사 정오표 미제공, initial pH 미확인 |
| 10.1039/d5va00202h (이전) | sewage sludge; 400–700 °C | 4 | 미추출 | 기존 별도 | Methods 20g/20mL와 Fig 2 투입량 1g/L 모순 해결 필요 |

> 포멜로 `μmol/g → mg/g`: 원소량(μmol/g) × 원자량(mg/mmol) / 1000. Cu 63.546, Pb 207.2, Zn 65.38. 18행 중 Pb/MBC600은 20.001016mg/g으로 물질수지 상한 20.0mg/g보다 **0.0051%** 높아, 반올림 허용차(0.01%) 기준으로 통과 표시하되 `ROUNDING_TOLERANCE_CAP_EXCESS` 기록.

## 각 파일의 의미

- `stage1_new33_BET_TPV_SOURCE_STAGING.csv`: 원문 출처 위치·시료별 물성·공정 정보 추출. **최우선 파일**.
- `stage2_new72_source_reported_QE_STAGING.csv`: 흡착량 원문값, 수정 단위, 물질수지 상한, 격리 여부.
- `stage2_new69_PHYSICAL_PASS_STILL_NOT_TRAINED.csv`: 수지만 통과한 보류 자료. 평형 또는 흡착 메커니즘 미인증.
- `stage2_new3_PHYSICAL_FAIL_QUARANTINE.csv`: 100mg/L Cd에 대한 가금분 3행 격리.
- `all_qmax_32_FITTED_NOT_QE.csv`: 이전 22+신규 10 qmax, 실측 qe와 별개.
- `raw_feedstock_composition_PARTIAL_BASIS_UNVERIFIED.csv`: 가금분, 면화 줄기의 **원료 원물** 일부 화학조성. DAF/dry basis 감사 전 모델에 강제하지 않음.
- `Stage1_Master_297_RESEARCH_STAGING_NOT_TRAINED.csv`: 기존 264+신규 33. 신규 행의 `*_training_candidate` 비워둠.
- `Stage2_Core_167_RESEARCH_STAGING_NOT_TRAINED.csv`: 기존 98+물질수지 조건부 통과 69. 새 행 `model_use_tier`는 모두 학습 승인 아님.

## 추가 검증에서 중요한 과학적 문제

1. 가금분 바이오차는 높은 회분·알칼리성으로 용액에서 **침전과 흡착을 구분하기 어려움**. qe라고 기록되어도 물질수지 제거량을 순수 흡착에 의한 제거량으로 단정하면 안 됨.
2. 포멜로 Qe_exp는 실험 kinetic table이지만 Cu의 180 min에서 평형 도달이 명확하지 않음. 초기 pH도 Methods로 확정하지 못했으므로 Stage 2 A/B 필수 입력 충족 여부를 다시 검사해야 함.
3. Cottons stalks + LDPE 공열분해 및 HTC+활성화는 기존의 단일 열분해 모델과 다른 제조경로. Stage 1 신규 DOI 외부 검증에서 경로 분리 필수.
4. 4편 신규 원료에서 조성 완비 자료는 여전히 부족. 특히 **열분해 전** 원료의 VM/FC 자료는 확보되지 않음.
5. 현재 파일은 원문 표의 숫자와 단위 및 사용자 제공 정정 사실을 기록한 것. Stage 1 또는 Stage 2의 성능 향상을 증명하지 않음.

## 설치

현재 v23 프로젝트의 백업을 만든 뒤 패치 ZIP의 **내용물**을 `프로그램/` 폴더에 병합한다. 기존 `.venv`, `data/imports/Phase8...`, `deployment/`는 변경하지 않는다. `stage2_pages.py`는 H 페이지 등록을 위해 전체 파일이 포함되어 있으므로 이전 버전과 사용자 커스터마이징이 있으면 백업한다.

```powershell
Set-Location "C:\Users\jh030\Desktop\대학\학술대회\프로그램"
Copy-Item ".\pyrolysis\biochar_ui\stage2_pages.py" "..\stage2_pages_before_v24.py"
Expand-Archive -LiteralPath ".\Biochar_v24_FourPapers_SourceStaging_Patch.zip" -DestinationPath "." -Force
.\.venv\Scripts\python.exe -m pytest tests\test_biochar_v24_source_ingest.py -q
.\.venv\Scripts\python.exe -m streamlit run biochar_stage2_app.py
```

사이드바 **H. New Literature Data (v24)**에서 직접 관측 기록을 검토할 수 있다. 기존 예측 숫자는 달라지지 않는다. 새 데이터로 재학습하려면 DOI 분할 고정, 원료/공정 정의 확인, 타깃 엄격화 및 홀드아웃 검증이 필요하다.
