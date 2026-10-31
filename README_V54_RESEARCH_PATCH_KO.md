# v54 Stage2 물리 제약·추천 검증 패치 — v47+v53에 압축 해제

**이 ZIP은 연구용으로 학습 완료한 모델과 직접 재현 코드를 포함합니다.** 활성 프로그램을 자동 교체하거나 데이터를 변조하지 않습니다.

### PowerShell (v47 프로젝트 루트에서)

```powershell
$python = Join-Path ([Environment]::GetFolderPath('Desktop')) '대학\학술대회\프로그램\.venv\Scripts\python.exe'
if (!(Test-Path $python)) { throw '기존 프로그램 .venv Python 경로를 확인하세요.' }
& $python --version
& $python -m pytest tests/test_v52_apple_series.py tests/test_v53_evidence_tier_refit.py tests/test_v54_bounded_rank.py -q --tb=short
```

**현재 PC에는 scikit-learn 1.5.2가 설치되어 있고 배포된 연구용 Joblib은 scikit-learn 1.8.0에서 학습한 파일**입니다. 테스트가 통과하더라도 버전 불일치 경고가 나면 그 자체로 호환성이 보장되지는 않습니다.

Windows 기존 `.venv` 환경과 같은 버전으로 새 모델을 다시 저장하려면 (기존 연구용 v54 파일만 갱신):

```powershell
& $python -m scripts.run_v54_bounded_rank --root .
```

위 스크립트는 연구용 `deployment/research_v54_NOT_ACTIVE/`와 `reports/v54_bounded_ranking/`만 갱신하며, 활성 배포 및 구버전 데이터와 모델은 변경하지 않습니다. v53 원문 검증 및 학습 데이터가 존재해야 실행되며, 없으면 오류로 중단됩니다.

### 해석에서 금지할 주장

- 'v54가 v53보다 높은 예측 정확도를 갖는다' → **거짓**. 현재 세 학습집합 모두 기존 direct+clipping 방식이 더 낮은 MAE.
- '14~16개 제조조건에서 1위 적중=독립 추천 검증' → **거짓**. 논문 내부 실험이며 pH 미보고, 기존 모델 출처 노출 감사 불완전.
- '8개 Zhao 원문 실험 qe가 완전 새 관측' → **거짓**. v52 24행에 이미 포함.
- '물질수지 상한을 지키면 물리 모델이 실험적으로 타당' → **거짓**. 출력 범위만 만족한 것.

전체 지표와 구분은 `reports/v54_bounded_ranking/RESULTS_KO.md` 참조.
