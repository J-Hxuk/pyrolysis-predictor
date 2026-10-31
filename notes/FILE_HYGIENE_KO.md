# v41 프로젝트 정리 원칙

- `학술대회_Archive_20261030`: 설치 완료한 과거 증분 패치 ZIP, 구형 설치 설명서, 디버깅 로그 등의 **보관용**. 기본적으로 자동 이동/삭제하지 않음.
- 원문 PDF와 보충자료는 DOI·파일 해시 목록이 있어도 원본을 최소 1곳에 안전하게 보관. 설치 패치에 중복 PDF를 포함하지 않음.
- 프로젝트 내 `data/imports/*`는 구형 Stage1/Stage2 로더가 직접 참조할 수 있으므로 삭제/이동 전에 `pytest tests -q` 통과와 재현성을 확인.
- `deployment/stage2_models/active.json` 및 활성 `model.joblib`, 데이터 출처/감사표, 실측·피팅 타깃 분리 자료는 유지.
- `__pycache__`, `.pytest_cache` 등은 프로그램 종료 후 정리 가능. `.venv`는 프로젝트 의존성 환경이므로 자동 정리 금지.
- 다음 정리 시점: v43~v45 즈음, 버전별 독립 ZIP을 하나의 **검증된 기준본**으로 통합한 뒤 과거 패치 ZIP만 아카이브로 이동.

진단 명령 (v40에서 제공):

```powershell
.\.venv\Scripts\python.exe -m scripts.audit_project_hygiene_v40 --root . --out reports\v41_hygiene\local_audit.json
```
