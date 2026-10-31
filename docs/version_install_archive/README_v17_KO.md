# 폐기물 열분해 생성물·고체 특성·촉매 성능 예측 프로그램

연구팀 내부용 웹 예측 시스템입니다.

## 현재 상태

- 작업 1 완료: 프로젝트 구조와 개발 환경 구성
- 실제 데이터 확보 및 학습 전 단계

## 빠른 시작

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

자세한 설치·실행 방법은 [docs/setup.md](docs/setup.md) 를 참고하세요.

## 주요 구조

```
pyrolysis/          공통 계산 패키지 (Streamlit 미포함)
├── contracts.py    자료형·구조 검증
├── config.py       설정 읽기
├── io.py           파일 읽기·양식 생성
├── validation.py   입력 검증
├── preprocessing.py 전처리 Pipeline
├── models.py       모델 후보
├── splitting.py    분할 계획
├── training.py     학습·선정
├── evaluation.py   독립 평가
├── analysis.py     변수 분석
├── artifacts.py    배포 묶음
├── inference.py    예측 서비스
├── ranking.py      후보 순위
├── cli.py          명령행 진입점
└── ui/             한국어 화면 (Streamlit 사용)
streamlit_app.py    웹 앱 진입점
configs/            설정 원형 (실제 설정은 configs/local/ — Git 제외)
deployment/         배포 묶음 (active.json + bundles/)
notebooks/          연구용 노트북
tests/              pytest 테스트
docs/               설치·배포 안내
```

## 주의 사항

- `data/`, `runs/`, `configs/local/` 는 Git에 포함하지 않습니다.
- 노트북 셀 출력은 커밋 전에 제거합니다.
- 이 시스템의 예측 결과는 실제 최적 조건을 보장하지 않습니다.
