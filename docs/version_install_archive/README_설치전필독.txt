BIOCHAR v24 데이터 원문 검증 패치 (기존 v23 설치 후 적용)

1) 이 ZIP은 기존 프로젝트를 삭제하지 않고 추가되는 데이터/검증화면용 패치입니다.
2) 신규 Stage1 33시료 (추가 논문 4편의 23 + 종전 2편의 10), 신규 Stage2 원문값 72행 중 3개 수지위반 격리.
3) 기존 데이터 264행/98행 및 기존 배포 모델은 변경하지 않습니다.
4) 포멜로 Table S1 mmol/g은 사용자가 확인한 오기를 반영해 umol/g으로 환산했습니다.
5) 원료/제조경로 및 평형 여부 미인증 자료는 학습 모드에 넣지 않았습니다.

설치: ZIP을 기존 프로그램 폴더에 복사하고, 해당 폴더에서 아래 명령 실행
Copy-Item ".\pyrolysis\biochar_ui\stage2_pages.py" "..\stage2_pages_before_v24.py"
Expand-Archive -LiteralPath ".\Biochar_v24_FourPapers_SourceStaging_Patch.zip" -DestinationPath "." -Force
.\.venv\Scripts\python.exe -m pytest tests\test_biochar_v24_source_ingest.py -q
.\.venv\Scripts\python.exe -m streamlit run biochar_stage2_app.py

사이드바 H. New Literature Data (v24) 메뉴에서 기록 확인 가능.
세부 연구 검증: docs/BIOCHAR_V24_SOURCE_STAGING_KO.md
