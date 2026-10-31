# v36 연구 패치 적용

**목적:** 신규 그래프 후보 판독 13행 + v35의 고정된 15행을 4 DOI 온도별 비교 연구용 데이터셋으로 연결하고, 독립 DOI 상대순위 평가 수행. 기존 모델/데이터/UI 미수정.

1. 기존 프로그램 폴더 전체 또는 수정 파일 백업.
2. 이 ZIP의 최상위 폴더 구조(`scripts/`, `tests/`, `data/`, `reports/`)를 기존 프로젝트 루트에 병합.
3. PowerShell:

```powershell
Set-Location 'C:\Users\jh030\Desktop\대학\학술대회\프로그램'
Expand-Archive -LiteralPath '.\Biochar_v36_GraphReadout_DOIRanking_Research_Patch.zip' -DestinationPath '.' -Force
.\.venv\Scripts\python.exe scripts\build_v36_temperature.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_v36_temperature.py' -v
```

- 출력: `reports/v36_temperature/RESULTS_KO.md`, `summary.json`, `v36_LODO4DOI_ranking.csv`, `v36_temperature_panel28_RESEARCH_ONLY.csv`
- Lv 논문의 3점은 별도 보류 CSV에 남고 학습되지 않는다.
- 일반 모델 A/B의 예측값, `deployment`, `active.json`, `.venv`는 변경하지 않는다.
- 그래프는 단일 판독으로 정확도가 미인증되었고, 수치가 있다고 임의로 qe 학습 승인을 부여하지 않는다.
