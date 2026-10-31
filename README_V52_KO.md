# Biochar v52 — 새 논문 실측 관측 24행 반영 + 전체 연구용 재학습

## 핵심 결과

- 출처: Zhao et al., *Energies* **13** (2020) 3498, DOI `10.3390/en13133498`.
  원문: https://www.mdpi.com/1996-1073/13/13/3498
- Stage 2 기준 **140행/32 DOI → 164행/33 DOI**. 신규 24행 모두 **같은 DOI 1편**.
  - **8행**: Table 3의 실험 평형 흡착량 `qe,exp`. Cu4, Zn4. 투입량 5 g/L.
  - **16행**: Section 3.1 텍스트에 직접 기재된 1440-min 흡착량. Cu8, Zn8. 투입량 1 또는 20 g/L.
  - 흡착조건: Cu C0=64 mg/L / Zn C0=65 mg/L, 초기 pH=5, 흡착온도 25°C,
    접촉시간 1440 min. 제조온도 300/400/500/600°C.
  - 24행 모두 `0 ≤ qe ≤ C0/dose` 만족. qe(exp), 1440-min 관측값과 qmax/속도식 fitted qe를 구분.
  - **동일 DOI의 동일 제조시료를 투입량만 바꾸어 측정**한 관측은 독립 출처가 아님.
- Stage 1 연구용 제조시료 **417→421**, BET 실측 **361→365**, TPV 실측 **231→231**.
  - BET: Table 1의 2.39/7.00/37.25/108.59 m²/g.
  - 원료의 탄소·수소·산소 함량은 **열분해 후 char** 측정치이므로 열분해 전 원료 조성으로 입력하지 않음.
  - 제조 유지시간/승온속도/분위기가 이 원문에 구체적으로 제시되지 않은 경우 NA로 보존.

## 모델 평가

동일한 기존 140행, 동일한 DOI GroupKFold(5개 fold)에서 비교. 새 DOI는 각 fold의 **학습 부분에만** 추가되고, 동일 평가 140행으로 성능을 계산한다. A/B는 XGBoost, Stage1은 Poisson HistGradientBoostingRegressor.

| 모델 | 기존 학습 | +신규 24행 | 단위 |
|---|---:|---:|---|
| Stage2 A · MAE (clipped) | 17.2468 | **16.7601** | mg/g |
| Stage2 B · MAE (clipped) | **17.6904** | 17.7610 | mg/g |
| Stage1 BET · MAE | **247.9710** | 252.9732 | m²/g |

범위 제한(clipping)은 물질수지 상한 안에서의 예측을 만드는 후처리이며 제조온도 영향의 물리 학습을 입증하지 않음.

추가 DOI를 **전혀 학습하지 않은 기존 모델**로 먼저 예측한 결과, 실제 최고 제조온도는 모든 금속·투입량 계열에서 500°C였지만,
Model A는 같은 흡착조건에서 제조온도가 입력되지 않아 온도별 예측이 동률이고,
Model B는 일부 계열에서 600°C 등 다른 온도를 최고로 예측했다.
따라서 최적 제조온도 예측 능력은 **아직 미확보**.

이 비교는 기존 검증 DOI를 반복 사용하는 후속 실험이며, 독립 시험으로 모델 개선을 확증하는 결과가 아니다. 신규 24행은 독립 표본 24편이 아니라 **1개 논문의 상관된 24관측**이다.

## 설치

이 ZIP은 기존 **v47에 v48~v51 패치를 적용한 프로젝트**에 덮어쓴다.

```powershell
Set-Location "C:\Users\jh030\Desktop\대학\학술대회\Biochar_v47_Integrated_Project"
Expand-Archive -LiteralPath ".\Biochar_v52_AppleTree_24Rows_FullRefit_Research_Patch.zip" -DestinationPath "." -Force
$env:BIOCHAR_PYTHON = "C:\Users\jh030\Desktop\대학\학술대회\프로그램\.venv\Scripts\python.exe"
.\CHECK_V52.cmd
```

모델 `.joblib`은 리눅스의 Python 3.13 / NumPy 2.3.5 / scikit-learn 1.8.0 / XGBoost 3.1.3에서 저장된 **비활성 연구용 모델**이므로 로딩 호환성이 보장되지 않는다. Windows에서 다음 명령으로 **해당 PC의 패키지 환경으로 재학습/재저장 가능**:

```powershell
& $env:BIOCHAR_PYTHON -m scripts.build_v52_apple_series --root .
```

이 명령은 기존 v46 입력 CSV와 v52 원문 검증값을 사용하여 5-fold 비교 후 Stage1 BET·TPV, Stage2 A·B 4모델을 `deployment/research_v52_NOT_ACTIVE/`에 저장한다. 기존 `deployment`의 활성 모델 파일을 덮어쓰지 않는다.

## 중요한 파일

- `data/imports/v52_apple_series/AppleTree_24_QE_2_SOURCE_TYPES_VERIFIED.csv`: 신규 24관측, 출처 품질 열 포함.
- `data/imports/v52_apple_series/AppleTree_Table1_BET_4_VERIFIED.csv`: 신규 제조시료 4행.
- `data/imports/v52_apple_series/Stage2_164_RESEARCH_NOT_ACTIVE.csv`: 전체 Stage2 연구용 통합자료.
- `data/imports/v52_apple_series/Stage1_421_RESEARCH_NOT_ACTIVE.csv`: 전체 Stage1 연구용 통합자료.
- `reports/v52_apple_series/`: DOI 분할 결과, 신규 DOI 사전 외부 예측, 원본 해시, 결과 보고서.
- `scripts/build_v52_apple_series.py`: 재생성·평가·재학습.
- `tests/test_v52_apple_series.py`: 관측 유형·DOI·물질수지·행 수 등 핵심 테스트.

v51에서 보류했던 대나무 DOI `10.3390/toxics10120787`는 투입량과 접촉시간의 실험별 근거가 아직 확정되지 않아 승인 데이터에 넣지 않았다.
