# v50 — 시연 기능이 아니라 모델 실제 성능을 검토하는 연구용 패치

## 변경 내용

1. v46 연구용 Stage 2 전체 140행/32 DOI의 동일 GroupKFold 평가를 재현하는 코드
2. A/B, 물질수지 사후 제한, 제거율 직접 학습, Cd/Cu 전문 모델, 오염물질 기준선 비교
3. 같은 관측 103행에 대해 실측 BET·TPV를 추가한 C의 효과 평가
4. 11개 제조온도 비교 패널의 순위 정확도 확인 (실험군 중복 DOI 구분)
5. 오차의 DOI 단위 부트스트랩 및 결과 보고서/그래프
6. Cu 제한용 연구 후보를 **사용자 PC에서만** 재학습·저장하는 선택적 명령

### 적용

이 파일은 **기존 v47 폴더에 v48·v49를 덮어쓴 상태**를 기준으로 한 v50 패치다.

압축파일 `Biochar_v50_Model_Utility_Research_Patch.zip`을 같은 폴더에 복사하고 PowerShell에서 실행한다.

```powershell
Set-Location "C:\Users\jh030\Desktop\대학\학술대회\Biochar_v47_Integrated_Project"
Expand-Archive -LiteralPath ".\Biochar_v50_Model_Utility_Research_Patch.zip" -DestinationPath "." -Force
$env:BIOCHAR_PYTHON = "C:\Users\jh030\Desktop\대학\학술대회\프로그램\.venv\Scripts\python.exe"
.\CHECK_V50.cmd
```

`CHECK_V50.cmd`는 별도 재학습을 강제하지 않으며, v50 결과 CSV와 코드 계약에 대한 테스트를 수행한다.

### 결과 검토

`reports/v50_model_utility/RESULTS_KO.md`를 읽는다. 기본 실행 앱은 그대로다.

추가 학습을 직접 재현하려면:

```powershell
& $env:BIOCHAR_PYTHON -m scripts.evaluate_v50_model_utility --root .
& $env:BIOCHAR_PYTHON -m scripts.summarize_v50_evidence --root .
& $env:BIOCHAR_PYTHON -m scripts.train_v50_bounded_candidate --root .
```

옵션 모델은 `deployment/research_v50_NOT_ACTIVE/` 안에 **연구용으로만** 저장한다. 기존 활성 모델은 덮어쓰지 않는다.

### 주의

- 이 버전은 전반적인 예측 성능 향상을 주장하는 제품 출시 버전이 아니다. 검증되지 않은 제조온도 추천 기능을 추가하지 않는다.
- A는 제조온도를 입력받지 않으므로 제조온도별 랭킹에 사용할 수 없다.
- 실측 BET·TPV를 사용한 모델 C는 원료의 열분해조건만으로 물성을 자동 추정하는 D와 다르다.
- 그래프 판독값과 논문별 조건이 혼합된 140행은 연구용이며, 기존 앱 활성 Core 97과 구분한다.
- 평가 환경의 scikit-learn·XGBoost 버전 차이로 재학습 수치는 소폭 달라질 수 있다.
