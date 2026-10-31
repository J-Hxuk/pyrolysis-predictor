# Biochar v42 설치 (v41 설치 이후, 연구용 증분 패치)

**기존 활성 모델·데이터·Streamlit 화면은 수정하지 않습니다.** v42는 물질수지 기반 연구모델의 DOI 검증 실험만 추가합니다.

1. Streamlit 프로그램이 실행 중이면 종료. `Biochar_v42_Physics_BoundedUptake_Research_Patch.zip`를 `프로그램` 폴더에 저장.
2. PowerShell에서:

```powershell
Set-Location "C:\Users\jh030\Desktop\대학\학술대회\프로그램"
Expand-Archive -LiteralPath ".\Biochar_v42_Physics_BoundedUptake_Research_Patch.zip" -DestinationPath "." -Force

# 신규 + 관련 테스트
.\.venv\Scripts\python.exe -m pytest tests\test_v42_physics.py tests\test_v41_scope.py tests\test_v41_paired_oof.py tests\test_v40_new9.py tests\test_v37_sensitivity.py -q

# 개인 환경에서 재현/모델 재학습 (연구용 결과에만 덮어씀)
.\.venv\Scripts\python.exe -m scripts.evaluate_v42_physics --root .
```

3. `reports\v42_physics\doi_group5_metrics.csv`, `paired_doi_bootstrap.json`, `qa_tier_oof_diagnostics.csv`, `RESULTS_KO.md` 확인.
4. 기존 앱 실행은 평소처럼: ` .\.venv\Scripts\python.exe -m streamlit run biochar_stage2_app.py`.

**주의:** 패치에는 v40의 `reports/v40_new9/Stage2_reported_uptake_128_RESEARCH_CONDITIONAL.csv`가 반드시 필요하다. 소스가 없으면 `scripts.build_v40_new9`를 먼저 사용해 원래 근거 데이터에서 복구한다. 새 데이터나 임의의 숫자로 대신하지 않는다. 모델 바이너리는 실행 환경의 `scikit-learn` 버전에 따라 재학습이 필요할 수 있으므로 연구에 활용할 때는 위 재현 명령으로 재학습한다.

물질수지 `q <= C0/Dose`는 **계산 일관성 검증**이다. 실제 흡착/침전 기여 분리는 확인하지 않으며, 사후 clipping의 개선이 머신러닝 성능 개선을 의미하지 않는다. 이번 버전은 `deployment/active.json`을 변경하거나 기존 배포 모델을 교체하지 않는다.

프로젝트 폴더 정리는 `python -m scripts.audit_project_hygiene_v40 --root .`로 점검만 수행. 데이터/모델/추출본/원문 SHA256 증거는 자동 삭제 금지.
