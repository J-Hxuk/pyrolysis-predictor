# v39 설치 및 검증 안내

## 사전 조건

- **v38까지 적용된** 프로젝트 디렉터리 (v37 `scripts/build_v37_full_refit.py`, v15 입력계약 모듈 및 v38 `reports/v38_new_dois` 필요)
- `.venv`와 현재 활성 모델은 그대로 보존. 기존 모델을 덮어쓰는 파일은 패치에 없음.
- 설치 전 작업 프로젝트 전체 백업 권장.

## 순서: ZIP을 `프로그램` 디렉터리에 저장한 후 PowerShell

```powershell
Set-Location "C:\Users\jh030\Desktop\대학\학술대회\프로그램"

# v39 파일은 새 경로에만 기록됨
Expand-Archive -LiteralPath ".\Biochar_v39_PositiveBET_QualitySensitivity_Research_Patch.zip" -DestinationPath "." -Force

# 이번 연구용 논문 데이터의 품질 영향과 양수 제약 모델을 재계산
.\.venv\Scripts\python.exe -m scripts.build_v39_quality_positivity --root .

# 검증 실행
.\.venv\Scripts\python.exe -m pytest tests\test_v39_quality_positivity.py -q
```

## 결과

- `reports/v39_quality_and_positivity/RESULTS_KO.md` — 결과와 해석
- 같은 폴더의 CSV·PNG·summary.json — 상세 근거
- `deployment/research_v39_NOT_ACTIVE/Stage1_BET_Poisson_HGB_RESEARCH_ONLY.joblib` — **연구용, 활성화 금지**

**중요:** 배포 모델, `active.json`, Streamlit UI는 변경하지 않음. 모델 파일은 Python/scikit-learn 버전 차이로 작동하지 않을 수 있으므로 환경이 다르면 재학습 스크립트를 실행. 전체 회귀 suite와 Windows 실제 실행은 아직 미확인.
