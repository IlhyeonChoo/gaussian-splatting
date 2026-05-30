# Plan: 학습 실시간 모니터링 + 폼/UX 개선 (webui)

## Context

`webui/`는 FastAPI + HTMX 기반 단일 페이지로 프레임 추출 → prepare → COLMAP → train → render → metrics 워크플로우를
서버에서 실행한다. 현재 작업 상세 페이지는 **로그 텍스트(`<pre>`)를 2초마다 폴링**하는 것이 전부라, 사용자가
"학습이 지금 몇 번째 iteration인지, loss/PSNR가 어떻게 변하는지, 언제 끝나는지"를 한눈에 볼 수 없다.

목표:
1. **학습 실시간 모니터링** — `train.py` 로그를 파싱해 진행률 바 + 핵심 지표 타일 + PSNR/Loss 스파크라인으로 시각화.
2. **기존 폼/UX 개선** — 워크플로우 생성 폼의 단계 인지성·자동입력·가독성 향상.

제약: `train.py`는 수정하지 않는다(로그 파싱 방식). 외부 JS 차트 라이브러리 도입 없이 인라인 SVG + CSS만 사용해
현재의 "htmx CDN 하나뿐" 무의존성 구조를 유지한다.

## 파싱 가능한 학습 출력 (train.py 기준)

- tqdm 진행바(매 10 iter): `Training progress:  45%|███▌  | 13500/30000 [02:14<02:43,  100.50it/s, Loss=0.0312000, Depth Loss=0.0000000]`
  - tqdm은 `\r`로 갱신하지만 `jobs.py`가 text 모드(universal newline)로 읽으므로 각 갱신이 로그 파일에 줄 단위로 누적됨 → 폴링으로 추적 가능.
- 평가(test/train iters): `\n[ITER 7000] Evaluating test: L1 0.034 PSNR 28.51`
- 저장: `\n[ITER 7000] Saving Gaussians`, 체크포인트: `\n[ITER 7000] Saving Checkpoint`
- 출력 폴더: `Output folder: output/<name>`

## 변경/신규 파일

### 1. 신규 `webui/progress.py` — 로그 파서 (순수 함수, 무의존성)
- `@dataclass(frozen=True) Evaluation`: `iteration:int`, `split:str("test"|"train")`, `l1:float`, `psnr:float`
- `@dataclass(frozen=True) TrainingProgress`:
  `iteration`, `total_iterations`, `percent`, `rate`(it/s), `elapsed`, `eta`,
  `loss`, `depth_loss`, `evaluations:list[Evaluation]`, `saved_iterations:list[int]`,
  `output_folder:str|None`, `has_data:bool`
- `parse_training_progress(text:str) -> TrainingProgress`:
  - 컴파일된 정규식으로 라인 스캔. tqdm 라인은 **마지막 매치**를 현재 상태로 채택, 평가/저장은 전체 누적.
  - 정규식(대략):
    - tqdm: `Training progress:\s+(\d+)%.*?(\d+)/(\d+)\s+\[([\d:]+)<([\d:?]+),\s+([\d.]+)\s*it/s(?:,\s*Loss=([\d.eE+-]+),\s*Depth Loss=([\d.eE+-]+))?\]`
    - eval: `\[ITER (\d+)\] Evaluating (test|train): L1 ([\d.eE+-]+) PSNR ([\d.eE+-]+)`
    - save: `\[ITER (\d+)\] Saving Gaussians`
    - output: `Output folder:\s*(.+)`
  - 어떤 패턴도 매치 안 되면 `has_data=False` 반환(예: COLMAP 단계 진행 중).

### 2. `webui/jobs.py` — 진행 데이터 접근자
- `get_progress(job_id) -> TrainingProgress`: 로그 파일을 읽어(상한 캡 5MB, 초과 시 tail) `parse_training_progress` 호출.
  기존 `get_log_tail` 패턴 재사용. 파일 없으면 빈 `TrainingProgress`.

### 3. `webui/app.py` — 엔드포인트 + 렌더링 + 폼/UX
- **신규 엔드포인트** `GET /partials/jobs/{job_id}/progress` → `_job_progress(job)` HTML 프래그먼트 반환(404 처리는 기존 패턴 동일).
- **작업 상세 페이지**(`job_detail`): 로그 `<pre>` 위에 진행 패널 섹션 추가
  `<section hx-get="/partials/jobs/{id}/progress" hx-trigger="load, every 2s" hx-swap="innerHTML">`.
  종료 상태에서도 최종값 표시(무해).
- **신규 렌더 헬퍼**:
  - `_job_progress(job)`: `has_data=False`면 "학습 진행 정보 없음" 안내. 있으면:
    - CSS 진행률 바(percent) + `iteration/total`,
    - 지표 타일: 현재 iter, Loss, Depth Loss, it/s, ETA, 최신 test PSNR,
    - `_sparkline()`로 PSNR(test) 추이 + Loss 추이.
  - `_sparkline(points, ...) -> str`: 값 목록을 0~100 정규화해 `<svg><polyline>`로 그리는 인라인 SVG(외부 의존성 없음).
- **CSS 추가**(기존 `CSS` 문자열): `.progressbar/.progressbar > span`(채움 바), `.metrics`(타일 그리드), `.metric .value/.unit`, `.spark`(svg 컨테이너). 기존 색 변수(`--accent/--ok/--muted`) 재사용.
- **폼/UX 개선**(저위험·additive):
  - `JS`에 `syncStages()` 추가: 단계 체크박스 상태에 따라 대응 `<details>` 섹션을 dim(`.section-off` 클래스, opacity↓) 처리.
    매핑 Frame Extraction↔`stage_frames`, COLMAP↔`stage_colmap`, Training↔`stage_train`, Render↔`stage_render`.
    각 `<details>`에 `data-stage` 속성 부여, 체크박스에 `onchange="syncStages()"`, `load` 시 1회 실행.
    **제출 동작은 변경하지 않음**(실제 단계 게이팅은 이미 `commands.py`가 체크박스로 처리).
  - `pickCandidate()` 확장: 비어있을 때 `scene_name`뿐 아니라 `model_output_name`도 stem으로 자동 채움.
  - 폼 상단에 "선택된 자료" readout(선택 시 JS로 갱신) 추가로 현재 선택 가시화.
- **라벨**: `LABEL_KO`에 모니터링/폼 신규 문자열(예: "Progress","Iteration","Rate","ETA","Latest PSNR","Loss curve","PSNR curve","No training progress yet.","Selected") 추가.

### 4. 신규 `tests/test_webui_progress.py`
- 기존 `tests/test_webui_*` 스타일(표준 `unittest`/`pytest`, 외부 의존 없음) 따름.
- 케이스: 대표 tqdm 라인 1개 → iteration/total/percent/rate/loss 파싱, 평가 라인 여러 개 → `evaluations` 누적/정렬,
  `Saving Gaussians` → `saved_iterations`, 매치 없는 텍스트 → `has_data=False`, ETA `?` 형태 견고성.

## 구현 방식
- 사용자 글로벌 규칙에 따라 각 코딩 하위 단계는 우선 **Codex(`codex-rescue`)에 위임**, 실패 시 직접 처리하고 결과를 한국어 한 줄로 보고.
- 단계 순서: (1) `progress.py`+테스트 → (2) `jobs.get_progress` → (3) `app.py` 엔드포인트/렌더/CSS/JS → (4) 라벨/문구 마무리.

## 검증
1. 단위 테스트: `uv run pytest tests/test_webui_progress.py -q` (그리고 회귀로 `uv run pytest tests/ -q`).
2. 파서 스모크: `uv run python -c "from webui.progress import parse_training_progress; print(parse_training_progress(open('<기존 로그>').read()))"`.
3. 서버 기동: `uv run python -m webui.app` 후 작업 상세 페이지에서 진행 패널이 2초 폴링으로 갱신되는지, 진행바/타일/스파크라인이 그려지는지 확인.
4. 폼 UX: 단계 체크 해제 시 해당 섹션 dim, 후보 선택 시 scene/model 이름 자동입력 및 "선택된 자료" readout 확인.

## 범위 밖(이번 미적용)
- `train.py` 수정, 별도 메트릭 DB/스키마, WebSocket(현재 htmx 폴링 유지), Gaussian 포인트 수(현재 stdout 미출력) 표시.
