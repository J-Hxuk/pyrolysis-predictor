# Biochar v45 — 신규 DOI 4편 통합 및 모델 갱신 비교

이 패치는 v44까지 설치된 프로젝트에 **연구용 데이터·학습결과·모델**만 추가하며 기존 Streamlit UI 및 활성 모델을 교체하지 않습니다.

1. 다운로드한 `Biochar_v45_Independent_DOIs_FullRefit_Research_Patch.zip`을 프로그램 루트에 저장합니다.
2. PowerShell:

```powershell
Set-Location "C:\Users\jh030\Desktop\대학\학술대회\프로그램"
Expand-Archive -LiteralPath '.\Biochar_v45_Independent_DOIs_FullRefit_Research_Patch.zip' -DestinationPath . -Force

# 신규 자료 재구축
.\.venv\Scripts\python.exe -m scripts.build_v45_independent_dois --root .

# 신규 전용 테스트
.\.venv\Scripts\python.exe -m pytest tests\test_v45_independent_dois.py -q

# 선택: 전체 연구용 모델 재평가 및 4개 모델 재학습
.\.venv\Scripts\python.exe -m scripts.evaluate_v45_independent_dois --root .
```

- 결과: `reports/v45_independent_dois/RESULTS_KO.md` 및 `model_comparison.csv`.
- 추가 자료: `data/imports/v45_independent_do_is/`.
- 활성 모델은 변경되지 않음. `deployment/research_v45_NOT_ACTIVE/`는 연구용으로만 사용.
- 새 논문 PDF 원본은 기존에 받은 `Biochar_Independent_DOI_Temperature_Series (2).zip`에 있으며, 이 패치 ZIP에는 다시 포함되지 않음.
- Windows에서 학습 모델을 재생성할 때는 해당 PC의 가상환경으로 재학습하는 것을 권장함.
- Sakhiya 50 mL 가정은 논문 보고값이 아닌 **별도 시나리오**로만 보존, 메인 학습 제외.

**주의:** Stage2의 신규 그래프 판독값 10개는 연구용 근사치이고 원문 Table의 직접 수치가 아님. 재판독 전 활성 모델에 포함하지 마세요.
