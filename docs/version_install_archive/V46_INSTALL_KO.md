# v46 설치 안내 — Phase8 v2 원문검증 반영 및 추가 학습

v45까지 적용된 기존 프로그램 폴더에 `Biochar_v46_Phase8_Verified_Refit_Research_Patch.zip`을 저장합니다.

```powershell
Set-Location "C:\Users\jh030\Desktop\대학\학술대회\프로그램"
Expand-Archive -LiteralPath ".\Biochar_v46_Phase8_Verified_Refit_Research_Patch.zip" -DestinationPath "." -Force
.\.venv\Scripts\python.exe -m pytest tests\test_v46_phase8.py -q
```

이 패치에는 **재학습 완료된 연구용 모델 4개와 최신 데이터**가 이미 포함돼 있습니다. 재학습은 선택 사항입니다.

```powershell
.\.venv\Scripts\python.exe -m scripts.build_v46_phase8 --root .
```

학습 결과는 `reports/v46_phase8/`에, 연구용 모델은 `deployment/research_v46_NOT_ACTIVE/`에 저장됩니다.

- 기존 앱의 모델·가중치, `.venv`, UI, 과거 raw PDF를 덮어쓰지 않습니다.
- Stage1 신규 13행은 source 측정값이지만 모델 특징 입력이 약품비율을 충분히 표현하지 못하므로 연구용 사용만 허용합니다.
- Phase8 검증된 기존 43개는 재삽입하지 않습니다.
- 연구용 활성 모델 자동 교체 없음. Streamlit 시연은 기존 명령을 사용합니다.
- 기록이 많은 이전 패치 ZIP은 테스트에 필요한 파일/추출본과 구분하여 외부 아카이브로 이동할 수 있습니다. 원본 ZIP을 코드가 직접 참조하는지 먼저 점검해야 합니다.
