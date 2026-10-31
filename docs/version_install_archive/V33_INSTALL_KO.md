# v33 적용 및 연구용 데이터 확인

이 패치는 **v32까지 적용한 프로그램**에서 연구용 데이터 CSV·분석 재현 스크립트·검증 보고서만 추가한다. 기존 저장 모델·Streamlit UI·canonical dataset을 수정하지 않는다.

1. `Biochar_v33_FivePapers_Temperature_Staging_Patch.zip`을 `프로그램` 폴더 안으로 다운로드한다.
2. PowerShell:

```powershell
Set-Location "C:\Users\jh030\Desktop\대학\학술대회\프로그램"
Test-Path ".\Biochar_v33_FivePapers_Temperature_Staging_Patch.zip"
# 위 값이 True일 때만 다음 명령 수행
Expand-Archive -LiteralPath ".\Biochar_v33_FivePapers_Temperature_Staging_Patch.zip" -DestinationPath "." -Force
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_v33_five_papers.py" -v
```

CSV는 `data\imports\v33_five_papers\`, 보고서는 `reports\v33_five_papers\RESULTS_KO.md`.

중요: `Stage1_Master_327_RESEARCH_STAGING_NOT_TRAINED.csv`는 이전 v24 297행에 새 제조시료 30개를 붙인 **별도 연구용 후보 복사본**이다. 기존 297행 표, canonical Stage 1, Stage 2 Core 167, 배포 모델은 변경되지 않는다. 이 파일로 자동 재학습하지 말 것.

선택적 재생성: `.\.venv\Scripts\python.exe -m scripts.build_v33_five_papers`
기존 v24 통합 파일이 설치된 경로에 존재하면 신규 DOI 중복을 먼저 검사한다.

원문 PDF·보충자료는 패치에 포함되지 않는다. 전달받은 PDF 파일은 별도로 보존한다.
