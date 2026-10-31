# v45 파일 정리 지침

- 이 ZIP은 신규 소스 PDF 자체를 중복 포함하지 않습니다. 원문 PDF 7개는 사용자가 받은 `Biochar_Independent_DOI_Temperature_Series (2).zip`으로 보관하고, `source_sha256_manifest.json`에 검증용 SHA-256이 남습니다.
- 현재 활성 모델 경로 및 `deployment/...ACTIVE...`는 변경하지 않습니다. 연구용 모델은 `deployment/research_v45_NOT_ACTIVE`에만 추가됩니다.
- v17~v44 설치 ZIP은 새 프로젝트에 필요한 파일이 모두 있는지 확인한 다음 `학술대회_Archive_YYYYMMDD`로 **이동(삭제 금지)**해도 됩니다.
- `data/imports/`, `reports/v40_new9`, `deployment/research_*` 등을 임의로 이동하면 재현 스크립트가 실패할 수 있습니다. 정리할 때는 의존 경로를 먼저 통합합니다.
- 캐시(`__pycache__`, `.pytest_cache`)는 프로그램 종료 후 삭제 가능하지만, 테스트 후 재생성됩니다.
- v46~v47에서는 누적 패치의 의존성 목록을 만든 다음 설치 1회로 재현되는 통합 기준본을 제작하는 게 효율적입니다.
