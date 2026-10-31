# v61 — 제조조건 추천 신뢰성 진단 (v47 누적 연구 패치)

## 수행한 작업

- 기존 v60 DOI별 OOF 예측의 전체 188행에 대해 raw qe, 물질수지 상한 `C0/dose`, clipping 후 qe를 원문 데이터와 결합해서 감사
- **제조조건 비교계열 15개(9 DOI)**에서 공동 1위, 최선/최악 top-1 regret, pairwise 동률, clipping으로 새로 발생한 동률을 비교
- v60 A / Model B(row-equal) / Model B(DOI+시료 가중) / 제조조건 RF(relative score)를 동일 패널에서 비교
- Chen 2022 16개 조건은 **초기 pH 미보고**이므로 연구용 사후 동률 민감도 분석만 수행; 자동 추천 불허
- 완전한 흡착조건과 엄격한 실험 평형 qe 근거가 모두 있는 경우만 *연구용 후보*로 분류

## 개별 설치 파일

- `scripts/run_v61_recommendation_reliability.py`
- `tests/test_v61_recommendation_reliability.py`
- `reports/v61_recommendation_reliability/` : 기존 제작 환경의 참고용 분석 산출물. **PC 재실행 결과로 덮어써야 함**

## 설치 전 요구

v59의 `reports/v59_doi_material_weighting/chen16_weighted_predictions.csv`와
v60.1의 다음 CSV가 필요하다.

- `reports/v60_separate_effects/all188_doi_oof_predictions.csv`
- `reports/v60_separate_effects/panel_evidence_quality.csv`

v60.1 사용자 PC에서 14개 Spearman 유효패널이 나온 사실은 성능 평가에서 별도로 유지한다. v61은 15개 **관측 패널** 전체를 손실 없이 감사한다. Spearman이 정의되지 않아도 패널·regret을 삭제하지 않는다.

## 실행 (PowerShell; v47 프로젝트 최상위 폴더)

```powershell
$ErrorActionPreference = 'Stop'
$python = Join-Path ([Environment]::GetFolderPath('Desktop')) '대학\학술대회\프로그램\.venv\Scripts\python.exe'
$zip = '.\Biochar_v61_Recommendation_Reliability_v47_Patch.zip'
if (!(Test-Path $zip)) { throw 'v61 ZIP을 v47 프로젝트 폴더에 옮기세요.' }
if (!(Test-Path $python)) { throw '기존 프로그램\.venv Python 경로를 확인하세요.' }
if (!(Test-Path '.\reports\v60_separate_effects\all188_doi_oof_predictions.csv')) { throw 'v60 OOF를 찾지 못했습니다.' }
if (!(Test-Path '.\reports\v59_doi_material_weighting\chen16_weighted_predictions.csv')) { throw 'v59 Chen16 결과를 찾지 못했습니다.' }
Expand-Archive -LiteralPath $zip -DestinationPath '.' -Force
& $python -m scripts.run_v61_recommendation_reliability --root .
if ($LASTEXITCODE -ne 0) { throw 'v61 감사 실패' }
& $python -m pytest tests/test_v61_recommendation_reliability.py -q --tb=short
if ($LASTEXITCODE -ne 0) { throw 'v61 테스트 실패' }
Get-Content '.\reports\v61_recommendation_reliability\RESULTS_KO.md' -Encoding UTF8
```

## 판정 로직 및 한계

1. 입력: 학습에 사용되지 않은 DOI의 각 평가 fold 예측값만 사용. 하지만 연구 자체가 **사후 탐색**이고 DOI가 이전 선택에 노출되었으므로 외부 전향 독립검증이 아님.
2. 보수적 추천 보류: 흡착조건 불완전, qe 평형 근거 부족, 공동 1위, 물질수지 상한 포화, 단독 1위라도 미세한 예측 간격을 각각 기록.
3. 동률에 의한 regret은 실제 관측 최적값과 공동 1위 후보의 각 관측값 차이를 모두 고려: 최선과 최악의 범위. 통계적 신뢰구간이 아님.
4. `1 mg/g` 간격 기준은 계산적 민감도 분석을 위한 보수적 휴리스틱일 뿐 검증된 오차 경계가 아님.
5. 제조조건 RF 출력은 상대 순위 점수(`NOT_QE`); `qe`나 % 제거율로 해석 금지.
6. 원본 학습 CSV, 기존 v52–v60 모델, 활성 배포 모델 및 Streamlit 앱 **변경 없음**. 이번 패치는 신규 모델 재학습을 포함하지 않는 추천 검증 패치임.
