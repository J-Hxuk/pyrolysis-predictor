# 개발 환경 설치와 실행 안내

## 사전 요건

- Windows 10/11 (64-bit)
- Python 3.12 (공식 배포판: https://www.python.org/downloads/)
- Git

## 1. 저장소 복제

```powershell
git clone <저장소-URL>
cd <프로젝트-폴더>
```

## 2. Python 3.12 가상 환경 생성

```powershell
py -3.12 -m venv .venv
```

> `py -3.12` 가 없으면 `python3.12` 또는 설치된 Python의 전체 경로를 사용한다.

## 3. 의존성 설치

### 웹 실행만 필요한 경우

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

### 연구·개발 환경 (학습·노트북·테스트 포함)

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

## 4. Jupyter 커널 등록

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --user --name pyrolysis-research --display-name "Python (pyrolysis-research)"
```

## 5. 로컬 Streamlit 앱 실행

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

브라우저에서 http://localhost:8501 로 접속한다.

## 6. 테스트 실행

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## 7. 실제 데이터 준비 (데이터 확보 후)

1. `configs/local/` 폴더를 만든다.
2. `configs/*.template.yaml` 을 복사하고 실제 값을 채운다.

   ```powershell
   copy configs\problem.template.yaml configs\local\problem.yaml
   copy configs\dataset.template.yaml configs\local\dataset.yaml
   copy configs\training.template.yaml configs\local\training.yaml
   copy configs\release.template.yaml configs\local\release.yaml
   ```

3. `data/` 폴더에 학습 자료를 넣는다 (Git 제외 대상).
4. `audit` 명령으로 자료 준비 상태를 확인한다.

   ```powershell
   .\.venv\Scripts\python.exe -m pyrolysis.cli audit --problem configs/local/problem.yaml --dataset configs/local/dataset.yaml
   ```

## 8. CLI·노트북 실행

명령 인자·오류·실행 순서와 노트북 대응은 `docs/cli.md` 를 참고한다.

합성 설정으로 전체 경로를 점검하는 예:

```powershell
# 학습 → 보고서 → 묶음 저장·재읽기 (합성 설정)
.\.venv\Scripts\python.exe -m pyrolysis.cli train  --demo --output runs
.\.venv\Scripts\python.exe -m pyrolysis.cli report --demo --output runs
.\.venv\Scripts\python.exe -m pyrolysis.cli export --demo --bundle-dir deployment/bundles --bundle-id demo_cli_export

# 양식 생성·묶음 검사·운영 설정·활성화
.\.venv\Scripts\python.exe -m pyrolysis.cli make-template --problem configs/demo.yaml --output templates/local
.\.venv\Scripts\python.exe -m pyrolysis.cli check-bundle --bundle deployment/bundles/<bundle_id>
.\.venv\Scripts\python.exe -m pyrolysis.cli prepare-runtime
.\.venv\Scripts\python.exe -m pyrolysis.cli activate --bundle deployment/bundles/<bundle_id>
```

실제 학습·독립 평가·완전 묶음 내보내기는 실제 설정과 평가 완료 대상이 필요하므로
작업 28/29 에서 수행한다. 그 전에는 위 `--demo` 경로로 기능만 점검한다.

## 9. Git 커밋 전 확인 사항

- `configs/local/` 파일이 스테이징에 포함되지 않았는지 확인
- `data/`, `runs/` 폴더가 스테이징에 포함되지 않았는지 확인
- 노트북 셀 출력을 제거했는지 확인 (Kernel → Restart & Clear Output)
- `.streamlit/secrets.toml` 이 없거나 스테이징에서 제외되었는지 확인

```powershell
git status
git diff --cached --name-only
```

## 확인하지 못한 환경

| 환경 | 상태 | 확인 예정 |
|------|------|-----------|
| Windows 11, Python 3.12 | ✅ 개발 기준 환경 | — |
| Linux (Ubuntu 22.04), Python 3.12 | ⬜ 미확인 | 작업 24·29에서 확인 |
| Streamlit Community Cloud (Linux) | ⬜ 미확인 | 작업 32에서 확인 |

Linux 환경 확인은 Windows 생성 합성 묶음을 Linux에서 읽고 예측하는 방식으로 수행한다.
실제 학습 행을 서버용 검증 자료로 복사하지 않는다.
