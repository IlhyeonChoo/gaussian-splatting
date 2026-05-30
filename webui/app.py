"""FastAPI + HTMX web interface for 3DGS workflows."""

from __future__ import annotations

import html
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, RedirectResponse

from .commands import (
    BOOL_TRAIN_FIELDS,
    FLOAT_COLMAP_FIELDS,
    FLOAT_TRAIN_FIELDS,
    INT_COLMAP_FIELDS,
    INT_TRAIN_FIELDS,
    PATH_COLMAP_FIELDS,
    TEXT_COLMAP_FIELDS,
    TEXT_TRAIN_FIELDS,
    ZERO_ONE_FIELDS,
    build_job_spec,
)
from .config import WebUIConfig, load_config
from .data_browser import (
    DataCandidate,
    discover_data_candidates,
    discover_model_candidates,
    is_within_directory,
)
from .jobs import JobManager, JobRecord, TERMINAL_STATUSES
from .progress import TrainingProgress
from .security import build_allowed_networks, detect_tailscale_ipv4, is_client_allowed, resolve_bind_host


CSS = """
:root {
  color-scheme: light;
  --bg: #f7f8fa;
  --panel: #ffffff;
  --text: #1f2937;
  --muted: #667085;
  --line: #d8dee8;
  --accent: #0f766e;
  --danger: #b42318;
  --ok: #087443;
  --warn: #b54708;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-size: 14px;
}
header {
  padding: 18px 24px;
  border-bottom: 1px solid var(--line);
  background: var(--panel);
}
h1, h2, h3 { margin: 0; line-height: 1.2; }
h1 { font-size: 20px; }
h2 { font-size: 16px; margin-bottom: 12px; }
h3 { font-size: 14px; margin: 16px 0 8px; }
main { padding: 20px 24px 40px; }
.layout {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(320px, 448px);
  gap: 20px;
  align-items: start;
}
.panel {
  min-width: 0;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 16px;
}
.grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
}
.grid.two { grid-template-columns: repeat(2, minmax(0, 1fr)); }
label { display: grid; gap: 5px; min-width: 0; }
label span, .label { color: var(--muted); font-size: 12px; font-weight: 600; }
input, select, textarea {
  width: 100%;
  min-width: 0;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 8px 10px;
  font: inherit;
  background: #fff;
}
textarea { min-height: 72px; resize: vertical; }
.checks {
  display: flex;
  flex-wrap: wrap;
  gap: 10px 16px;
  margin: 8px 0 14px;
}
.checks label {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: var(--text);
  font-size: 13px;
}
.checks input { width: auto; }
details {
  border-top: 1px solid var(--line);
  margin-top: 16px;
  padding-top: 12px;
}
summary { cursor: pointer; color: var(--accent); font-weight: 700; }
button, .button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 1px solid var(--accent);
  border-radius: 6px;
  padding: 8px 12px;
  background: var(--accent);
  color: #fff;
  font-weight: 700;
  text-decoration: none;
  cursor: pointer;
}
button.secondary, .button.secondary {
  background: #fff;
  color: var(--accent);
}
button.danger {
  border-color: var(--danger);
  background: var(--danger);
}
table { width: 100%; border-collapse: collapse; }
th, td {
  padding: 8px 6px;
  border-bottom: 1px solid var(--line);
  text-align: left;
  vertical-align: top;
}
th { color: var(--muted); font-size: 12px; }
code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px;
}
pre {
  overflow: auto;
  max-height: 520px;
  margin: 0;
  padding: 12px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #101828;
  color: #eef2f6;
  white-space: pre-wrap;
}
.status {
  display: inline-flex;
  border-radius: 999px;
  padding: 2px 8px;
  font-size: 12px;
  font-weight: 700;
  background: #eef2f6;
}
.status.succeeded { color: var(--ok); background: #ecfdf3; }
.status.failed, .status.canceled { color: var(--danger); background: #fef3f2; }
.status.running, .status.canceling { color: var(--warn); background: #fff7ed; }
.muted { color: var(--muted); }
.job-list {
  display: grid;
  gap: 10px;
}
.job-card {
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 10px 12px;
  background: #fbfcfe;
  min-width: 0;
}
.job-card-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 10px;
}
.job-card-head a,
.job-card-name,
.job-card dd {
  min-width: 0;
  overflow-wrap: anywhere;
}
.job-card-head .status {
  flex: 0 0 auto;
}
.job-card-name {
  margin-top: 8px;
  font-weight: 700;
}
.job-card-meta {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px;
  margin: 10px 0 0;
}
.job-card-meta div {
  min-width: 0;
}
.job-card dt {
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
}
.job-card dd {
  margin: 2px 0 0;
}
.error {
  border: 1px solid #fecdca;
  background: #fffbfa;
  color: var(--danger);
  border-radius: 8px;
  padding: 12px;
}
.actions { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; margin-top: 14px; }
.path { word-break: break-all; }
.selected-readout {
  margin: 6px 0 14px;
  padding: 8px 10px;
  border: 1px dashed var(--line);
  border-radius: 6px;
  background: #fbfcfe;
  font-size: 13px;
}
.selected-readout code { color: var(--accent); word-break: break-all; }
details.section-off {
  opacity: 0.5;
}
details.section-off > summary::after {
  content: " · 단계 꺼짐";
  color: var(--muted);
  font-weight: 500;
  font-size: 12px;
}
.progress-head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 10px;
  margin-bottom: 6px;
}
.progress-head .count { color: var(--muted); font-variant-numeric: tabular-nums; }
.progressbar {
  height: 12px;
  border-radius: 999px;
  background: #eef2f6;
  overflow: hidden;
  border: 1px solid var(--line);
}
.progressbar > span {
  display: block;
  height: 100%;
  background: var(--accent);
  transition: width 0.4s ease;
}
.metrics {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 10px;
  margin: 14px 0;
}
.metric {
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 10px 12px;
  background: #fbfcfe;
}
.metric .label { display: block; margin-bottom: 4px; }
.metric .value {
  font-size: 18px;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
}
.metric .unit { font-size: 12px; color: var(--muted); font-weight: 500; margin-left: 3px; }
.charts {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 14px;
  margin-top: 8px;
}
.chart .label { display: block; margin-bottom: 4px; }
.spark {
  display: block;
}
.chart .spark {
  width: 100%;
  height: 90px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fbfcfe;
}
.candidate-browser {
  margin-bottom: 14px;
}
.candidate-tree {
  max-height: 320px;
  overflow: auto;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 10px;
  background: #fbfcfe;
}
.candidate-tree details {
  border-top: 0;
  margin-top: 0;
  padding-top: 0;
}
.tree-list {
  list-style: none;
  margin: 0;
  padding-left: 16px;
}
.tree-list.root {
  padding-left: 0;
}
.tree-list li {
  margin: 3px 0;
}
.tree-folder > summary {
  color: var(--text);
  font-weight: 700;
}
.candidate-button {
  width: 100%;
  justify-content: flex-start;
  border-color: transparent;
  background: transparent;
  color: var(--text);
  font-weight: 500;
  padding: 6px 8px;
  text-align: left;
}
.candidate-button:hover,
.candidate-button.selected {
  border-color: var(--accent);
  background: #ecfdf3;
}
.candidate-kind {
  margin-left: 8px;
  color: var(--muted);
  font-size: 12px;
}
@media (max-width: 1100px) {
  .layout { grid-template-columns: 1fr; }
  .grid, .grid.two { grid-template-columns: 1fr; }
}
"""

JS = """
function pickCandidate(button) {
  if (!button || !button.dataset.path) return;
  document.querySelector('[name="source_path"]').value = button.dataset.path;
  document.querySelector('[name="source_kind"]').value = button.dataset.kind;
  const stem = button.dataset.stem || '';
  const sceneInput = document.querySelector('[name="scene_name"]');
  if (sceneInput && !sceneInput.value) sceneInput.value = stem;
  const modelInput = document.querySelector('[name="model_output_name"]');
  if (modelInput && !modelInput.value) modelInput.value = stem;
  document.querySelectorAll('.candidate-button.selected').forEach((item) => item.classList.remove('selected'));
  button.classList.add('selected');
  updateSelectedReadout();
}

function updateSelectedReadout() {
  const readout = document.getElementById('selected-readout');
  if (!readout) return;
  const path = (document.querySelector('[name="source_path"]') || {}).value || '';
  const kind = (document.querySelector('[name="source_kind"]') || {}).value || '';
  readout.textContent = '';
  if (path) {
    const code = document.createElement('code');
    code.textContent = path;
    readout.append(
      document.createTextNode('선택된 자료(Selected): '),
      code,
      document.createTextNode(' · ' + kind)
    );
  } else {
    readout.textContent = '선택된 자료 없음 (no source selected)';
  }
}

function syncStages() {
  document.querySelectorAll('details[data-stage]').forEach((section) => {
    const stage = section.getAttribute('data-stage');
    const checkbox = document.querySelector('input[name="' + stage + '"]');
    const on = checkbox ? checkbox.checked : true;
    section.classList.toggle('section-off', !on);
  });
}

document.addEventListener('DOMContentLoaded', function () {
  syncStages();
  updateSelectedReadout();
  const sourcePath = document.querySelector('[name="source_path"]');
  if (sourcePath) sourcePath.addEventListener('input', updateSelectedReadout);
});

function logScrollTarget(event) {
  const detail = event.detail || {};
  const target = detail.target || detail.elt;
  if (target && target.matches && target.matches('[data-log-scroll]')) return target;
  return null;
}

document.addEventListener('htmx:beforeSwap', function (event) {
  const target = logScrollTarget(event);
  if (!target) return;
  const distanceFromBottom = target.scrollHeight - target.scrollTop - target.clientHeight;
  target.dataset.wasAtBottom = String(distanceFromBottom < 24);
  target.dataset.previousScrollTop = String(target.scrollTop);
});

document.addEventListener('htmx:afterSwap', function (event) {
  const target = logScrollTarget(event);
  if (!target) return;
  window.requestAnimationFrame(function () {
    if (target.dataset.wasAtBottom === 'true') {
      target.scrollTop = target.scrollHeight;
    } else {
      target.scrollTop = Number(target.dataset.previousScrollTop || 0);
    }
  });
});
"""

LABEL_KO = {
    "3DGS Web UI": "3DGS 웹 UI",
    "New Workflow": "새 워크플로우",
    "Jobs": "작업",
    "Candidate data": "후보 자료",
    "Source kind": "자료 유형",
    "Source path": "자료 경로",
    "Job name": "작업 이름",
    "Scene name": "씬 이름",
    "Model output name": "모델 출력 이름",
    "Existing model for render/metrics only": "렌더/평가 전용 기존 모델",
    "Stages": "단계",
    "Extract frames": "프레임 추출",
    "Prepare": "준비",
    "COLMAP": "COLMAP 전처리",
    "Train": "학습",
    "Render": "렌더",
    "Metrics": "평가",
    "Frame Extraction": "프레임 추출",
    "Mode": "모드",
    "Prepared frame set": "준비할 프레임 세트",
    "Target FPS": "목표 FPS",
    "Every nth": "N프레임마다",
    "Width": "가로",
    "Height": "세로",
    "Scale": "배율",
    "Custom format": "사용자 포맷",
    "Original format": "원본 포맷",
    "JPEG/WebP quality": "JPEG/WebP 품질",
    "Max images": "최대 이미지 수",
    "Preset": "프리셋",
    "Device": "장치",
    "Matcher": "매처",
    "Mapper": "매퍼",
    "Camera mode": "카메라 모드",
    "Camera model": "카메라 모델",
    "Feature type": "특징점 유형",
    "Matching type": "매칭 유형",
    "Camera params": "카메라 파라미터",
    "Advanced COLMAP options": "고급 COLMAP 옵션",
    "Training": "학습",
    "Iterations": "반복 횟수",
    "Resolution": "해상도",
    "Images folder": "이미지 폴더",
    "Depths folder": "깊이 폴더",
    "Data device": "자료 장치",
    "Optimizer": "옵티마이저",
    "Max train cameras": "최대 학습 카메라 수",
    "Camera quality ratio": "카메라 품질 비율",
    "Camera selection seed": "카메라 선택 시드",
    "Test iterations": "테스트 반복 지점",
    "Save iterations": "저장 반복 지점",
    "Checkpoint iterations": "체크포인트 반복 지점",
    "Advanced training options": "고급 학습 옵션",
    "Render flags": "렌더 플래그",
    "Iteration": "반복 지점",
    "Create job": "작업 생성",
    "Dashboard": "대시보드",
    "Outputs": "출력",
    "Output": "출력",
    "Invalid Job": "잘못된 작업",
    "Source": "자료",
    "Scene": "씬",
    "Model": "모델",
    "Current step": "현재 단계",
    "Commands": "명령",
    "Log": "로그",
    "Cancel": "취소",
    "ID": "식별자",
    "Name": "이름",
    "Status": "상태",
    "Step": "단계",
    "Updated": "갱신 시각",
    "Type": "유형",
    "No jobs yet.": "아직 작업 없음",
    "No candidates found.": "후보 없음",
    "Select server data from the tree.": "트리에서 서버 자료 선택",
    "None": "없음",
    "skip matching": "매칭 건너뛰기",
    "create images_2/4/8": "images_2/4/8 생성",
    "white background": "흰 배경",
    "eval split": "평가 분할",
    "train/test exposure": "학습/테스트 노출",
    "antialiasing": "안티앨리어싱",
    "random background": "랜덤 배경",
    "quiet": "조용히",
    "skip train": "학습 렌더 건너뛰기",
    "skip test": "테스트 렌더 건너뛰기",
    "Progress": "진행 상황",
    "Rate": "속도",
    "Elapsed": "경과 시간",
    "ETA": "남은 시간",
    "Latest test PSNR": "최신 테스트 PSNR",
    "Latest train PSNR": "최신 학습 PSNR",
    "Loss": "손실",
    "Depth Loss": "깊이 손실",
    "Saved iterations": "저장된 반복 지점",
    "PSNR curve": "PSNR 추이",
    "Loss curve": "손실 추이",
    "No training progress yet.": "아직 학습 진행 정보 없음",
}

KIND_LABEL_KO = {
    "image_folder": "이미지 폴더",
    "video_file": "동영상 파일",
    "colmap_scene": "COLMAP 씬",
    "model": "모델",
}

FIELD_LABEL_KO = {
    "aliked_lightglue_min_score": "ALIKED LightGlue 최소 점수",
    "aliked_matching_max_ratio": "ALIKED 매칭 최대 비율",
    "aliked_matching_min_cossim": "ALIKED 매칭 최소 코사인 유사도",
    "aliked_max_num_features": "ALIKED 최대 특징점 수",
    "aliked_min_score": "ALIKED 최소 점수",
    "ba_refine_extra_params": "BA 추가 파라미터 보정",
    "ba_refine_focal_length": "BA 초점거리 보정",
    "ba_refine_principal_point": "BA 주점 보정",
    "camera_mask_path": "카메라 마스크 경로",
    "colmap_project_path": "COLMAP 프로젝트 경로",
    "debug": "디버그",
    "debug_from": "디버그 시작 반복",
    "densification_interval": "밀도화 간격",
    "densify_from_iter": "밀도화 시작 반복",
    "densify_grad_threshold": "밀도화 그래디언트 임계값",
    "densify_until_iter": "밀도화 종료 반복",
    "depth_l1_weight_final": "최종 깊이 L1 가중치",
    "depth_l1_weight_init": "초기 깊이 L1 가중치",
    "detect_anomaly": "이상 감지",
    "exhaustive_block_size": "Exhaustive 블록 크기",
    "exposure_lr_delay_mult": "노출 LR 지연 배율",
    "exposure_lr_delay_steps": "노출 LR 지연 단계",
    "exposure_lr_final": "최종 노출 LR",
    "exposure_lr_init": "초기 노출 LR",
    "extra_feature_args": "추가 특징점 인자",
    "extra_mapper_args": "추가 매퍼 인자",
    "extra_matching_args": "추가 매칭 인자",
    "extra_undistort_args": "추가 왜곡 보정 인자",
    "feature_gpu_index": "특징점 GPU 인덱스",
    "feature_lr": "특징 LR",
    "feature_max_image_size": "특징점 최대 이미지 크기",
    "guided_matching": "가이드 매칭",
    "image_list_path": "이미지 목록 경로",
    "lambda_dssim": "DSSIM 가중치",
    "mapper_ba_global_function_tolerance": "매퍼 전역 BA 함수 허용오차",
    "mapper_ba_gpu_index": "매퍼 BA GPU 인덱스",
    "mapper_ba_use_gpu": "매퍼 BA GPU 사용",
    "mapper_filter_max_reproj_error": "매퍼 최대 재투영 오차 필터",
    "mapper_max_runtime_seconds": "매퍼 최대 실행 시간",
    "mapper_min_num_matches": "매퍼 최소 매칭 수",
    "mapper_multiple_models": "매퍼 다중 모델",
    "mapper_tri_ignore_two_view_tracks": "2-view 트랙 삼각화 무시",
    "mapper_tri_min_angle": "삼각화 최소 각도",
    "mask_path": "마스크 경로",
    "matching_gpu_index": "매칭 GPU 인덱스",
    "matching_max_num_matches": "최대 매칭 수",
    "num_threads": "스레드 수",
    "opacity_lr": "불투명도 LR",
    "opacity_reset_interval": "불투명도 초기화 간격",
    "percent_dense": "밀도화 비율",
    "pose_outlier_mad_scale": "포즈 이상치 MAD 배율",
    "pose_outlier_min_cameras": "포즈 이상치 최소 카메라 수",
    "position_lr_delay_mult": "위치 LR 지연 배율",
    "position_lr_final": "최종 위치 LR",
    "position_lr_init": "초기 위치 LR",
    "position_lr_max_steps": "위치 LR 최대 단계",
    "rotation_lr": "회전 LR",
    "scaling_lr": "스케일 LR",
    "sequential_loop_detection": "Sequential 루프 감지",
    "sequential_overlap": "Sequential 겹침 수",
    "sh_degree": "SH 차수",
    "sift_edge_threshold": "SIFT edge 임계값",
    "sift_lightglue_min_score": "SIFT LightGlue 최소 점수",
    "sift_matching_cross_check": "SIFT 매칭 교차 확인",
    "sift_matching_max_distance": "SIFT 매칭 최대 거리",
    "sift_matching_max_ratio": "SIFT 매칭 최대 비율",
    "sift_max_num_features": "SIFT 최대 특징점 수",
    "sift_peak_threshold": "SIFT peak 임계값",
    "spatial_ignore_z": "Spatial Z 무시",
    "spatial_max_distance": "Spatial 최대 거리",
    "spatial_max_num_neighbors": "Spatial 최대 이웃 수",
    "start_checkpoint": "시작 체크포인트",
    "two_view_max_error": "Two-view 최대 오차",
    "two_view_min_num_inliers": "Two-view 최소 inlier 수",
    "undistort_copy_policy": "왜곡 보정 복사 정책",
    "undistort_jpeg_quality": "왜곡 보정 JPEG 품질",
    "undistort_max_image_size": "왜곡 보정 최대 이미지 크기",
    "vocab_tree_num_images": "Vocab tree 이미지 수",
    "vocab_tree_num_nearest_neighbors": "Vocab tree 최근접 이웃 수",
    "vocab_tree_path": "Vocab tree 경로",
}


def create_app(config: WebUIConfig | None = None, manager: JobManager | None = None) -> FastAPI:
    config = config or load_config()
    tailscale_ip = detect_tailscale_ipv4()
    build_allowed_networks(config, tailscale_ip)
    manager = manager or JobManager(config)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        manager.start()
        yield
        manager.stop()

    app = FastAPI(title="3DGS Web UI", lifespan=lifespan)
    app.state.config = config
    app.state.jobs = manager
    app.state.tailscale_ip = tailscale_ip

    @app.middleware("http")
    async def restrict_network(request: Request, call_next):
        client_host = request.client.host if request.client else ""
        if not is_client_allowed(client_host, config, tailscale_ip):
            return PlainTextResponse("Forbidden", status_code=403)
        return await call_next(request)

    @app.get("/", response_class=HTMLResponse)
    async def dashboard() -> HTMLResponse:
        data_candidates = discover_data_candidates(config.data_roots, config.repo_root)
        model_candidates = discover_model_candidates(config.output_root, config.repo_root)
        jobs = manager.list_jobs()
        return HTMLResponse(_page("3DGS Web UI", _dashboard(config, data_candidates, model_candidates, jobs)))

    @app.get("/healthz", response_class=PlainTextResponse)
    async def healthz() -> PlainTextResponse:
        return PlainTextResponse("ok")

    @app.post("/jobs")
    async def create_job(request: Request):
        try:
            spec = build_job_spec(await request.form(), config)
            job_id = manager.enqueue(spec)
        except ValueError as exc:
            return HTMLResponse(_page("Invalid Job", _error_page(str(exc))), status_code=400)
        return RedirectResponse(f"/jobs/{job_id}", status_code=303)

    @app.get("/partials/jobs", response_class=HTMLResponse)
    async def jobs_partial() -> HTMLResponse:
        return HTMLResponse(_jobs_table(manager.list_jobs()))

    @app.post("/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str):
        manager.cancel_job(job_id)
        return RedirectResponse(f"/jobs/{job_id}", status_code=303)

    @app.get("/jobs/{job_id}/log", response_class=HTMLResponse)
    async def job_log(job_id: str) -> HTMLResponse:
        if not manager.get_job(job_id):
            raise HTTPException(status_code=404)
        return HTMLResponse(html.escape(manager.get_log_tail(job_id)))

    @app.get("/partials/jobs/{job_id}/progress", response_class=HTMLResponse)
    async def job_progress_partial(job_id: str) -> HTMLResponse:
        if not manager.get_job(job_id):
            raise HTTPException(status_code=404)
        return HTMLResponse(_job_progress(manager.get_progress(job_id)))

    @app.get("/partials/jobs/{job_id}/overview", response_class=HTMLResponse)
    async def job_overview_partial(job_id: str) -> HTMLResponse:
        job = manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404)
        return HTMLResponse(_job_overview(config, job))

    @app.get("/partials/jobs/{job_id}", response_class=HTMLResponse)
    async def job_detail_partial(job_id: str) -> HTMLResponse:
        job = manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404)
        return HTMLResponse(_job_detail(config, job, manager.get_progress(job_id)))

    @app.get("/jobs/{job_id}", response_class=HTMLResponse)
    async def job_detail(job_id: str) -> HTMLResponse:
        job = manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404)
        body = (
            f'<div class="actions"><a class="button secondary" href="/">{_label("Dashboard")}</a></div>'
            f'<section class="panel" id="job-detail">'
            f"{_job_detail(config, job, manager.get_progress(job_id))}</section>"
        )
        return HTMLResponse(_page(f"Job {job.id}", body))

    @app.get("/outputs", response_class=HTMLResponse)
    async def outputs_root() -> HTMLResponse:
        return HTMLResponse(_page("Outputs", _output_listing(config, config.output_root)))

    @app.get("/outputs/{path:path}")
    async def output_file(path: str):
        target = _resolve_output_path(config, path)
        if target.is_dir():
            return HTMLResponse(_page(f"Output {path}", _output_listing(config, target)))
        return FileResponse(target)

    return app


def _page(title: str, body: str) -> str:
    display_title = _label(title)
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{display_title}</title>
  <script src="https://unpkg.com/htmx.org@1.9.12"></script>
  <style>{CSS}</style>
  <script>{JS}</script>
</head>
<body>
  <header><h1>{display_title}</h1></header>
  <main>{body}</main>
</body>
</html>"""


def _dashboard(
    config: WebUIConfig,
    data_candidates: list[DataCandidate],
    model_candidates: list[DataCandidate],
    jobs: list[JobRecord],
) -> str:
    access = (
        f"Bind mode(바인드 모드): {_e(config.bind_mode)}, "
        f"Data roots(자료 루트): {_e(', '.join(str(root) for root in config.data_roots))}, "
        f"Output(출력): {_e(str(config.output_root))}"
    )
    return f"""
<p class="muted">{access}</p>
<div class="layout">
  <section class="panel">
    <h2>{_label("New Workflow")}</h2>
    {_job_form(data_candidates, model_candidates)}
  </section>
  <section class="panel">
    <h2>{_label("Jobs")}</h2>
    <div hx-get="/partials/jobs" hx-trigger="load, every 3s" hx-swap="innerHTML">
      {_jobs_table(jobs)}
    </div>
  </section>
</div>
"""


def _job_form(data_candidates: list[DataCandidate], model_candidates: list[DataCandidate]) -> str:
    return f"""
<form method="post" action="/jobs">
  <div class="candidate-browser">
    <div class="label">{_label("Candidate data")}</div>
    {_candidate_tree(data_candidates)}
  </div>
  <div id="selected-readout" class="selected-readout">선택된 자료 없음 (no source selected)</div>
  <div class="grid">
    <label><span>{_label("Source kind")}</span>{_source_kind_select()}</label>
    <label><span>{_label("Source path")}</span><input name="source_path" required></label>
  </div>
  <div class="grid">
    <label><span>{_label("Job name")}</span><input name="job_name" placeholder="optional / 선택 사항"></label>
    <label><span>{_label("Scene name")}</span><input name="scene_name" placeholder="source name / 자료 이름"></label>
    <label><span>{_label("Model output name")}</span><input name="model_output_name" placeholder="scene name / 씬 이름"></label>
  </div>
  <div class="grid two">
    <label><span>{_label("Existing model for render/metrics only")}</span>{_model_select(model_candidates)}</label>
  </div>

  <h3>{_label("Stages")}</h3>
  <div class="checks">
    {_stage_checkbox("stage_frames", "Extract frames", True)}
    {_stage_checkbox("stage_prepare", "Prepare", True)}
    {_stage_checkbox("stage_colmap", "COLMAP", True)}
    {_stage_checkbox("stage_train", "Train", True)}
    {_stage_checkbox("stage_render", "Render", False)}
    {_stage_checkbox("stage_metrics", "Metrics", False)}
  </div>

  <details data-stage="stage_frames">
    <summary>{_label("Frame Extraction")}</summary>
    <div class="grid">
      <label><span>{_label("Mode")}</span>{_select("frame_mode", [("both", "both"), ("original", "original"), ("custom", "custom")], "both")}</label>
      <label><span>{_label("Prepared frame set")}</span>{_select("prepared_frame_set", [("custom", "custom"), ("original", "original")], "custom")}</label>
      <label><span>{_label("Target FPS")}</span><input name="target_fps" value="2" inputmode="decimal"></label>
      <label><span>{_label("Every nth")}</span><input name="every_nth" inputmode="numeric"></label>
      <label><span>{_label("Width")}</span><input name="width" inputmode="numeric"></label>
      <label><span>{_label("Height")}</span><input name="height" inputmode="numeric"></label>
      <label><span>{_label("Scale")}</span><input name="scale" value="1.0" inputmode="decimal"></label>
      <label><span>{_label("Custom format")}</span>{_select("custom_format", [("jpg", "jpg"), ("png", "png"), ("webp", "webp")], "jpg")}</label>
      <label><span>{_label("Original format")}</span>{_select("original_format", [("png", "png"), ("jpg", "jpg"), ("webp", "webp")], "png")}</label>
      <label><span>{_label("JPEG/WebP quality")}</span><input name="jpeg_quality" value="90" inputmode="numeric"></label>
      <label><span>{_label("Max images")}</span><input name="max_images" value="0" inputmode="numeric"></label>
    </div>
  </details>

  <details open data-stage="stage_colmap">
    <summary>{_label("COLMAP")}</summary>
    {_colmap_form()}
  </details>

  <details open data-stage="stage_train">
    <summary>{_label("Training")}</summary>
    {_training_form()}
  </details>

  <details data-stage="stage_render">
    <summary>{_label("Render")}</summary>
    <div class="grid">
      <label><span>{_label("Iteration")}</span><input name="render_iteration" value="-1" inputmode="numeric"></label>
      <label><span>{_label("Render flags")}</span><span class="checks">
        {_checkbox("render_skip_train", "skip train", False)}
        {_checkbox("render_skip_test", "skip test", False)}
        {_checkbox("render_quiet", "quiet", False)}
        {_checkbox("render_antialiasing", "antialiasing", False)}
      </span></label>
    </div>
  </details>

  <div class="actions">
    <button type="submit">{_label("Create job")}</button>
  </div>
</form>
"""


def _candidate_tree(candidates: list[DataCandidate]) -> str:
    if not candidates:
        return f'<div class="candidate-tree"><p class="muted">{_label("No candidates found.")}</p></div>'

    tree: dict[str, object] = {"children": {}, "candidate": None}
    for candidate in candidates:
        parts = [part for part in candidate.label.replace("\\", "/").split("/") if part]
        if not parts:
            parts = [candidate.path.name]
        node = tree
        for part in parts:
            children = node.setdefault("children", {})
            assert isinstance(children, dict)
            node = children.setdefault(part, {"children": {}, "candidate": None})
            assert isinstance(node, dict)
        node["candidate"] = candidate

    children = tree["children"]
    assert isinstance(children, dict)
    return (
        f'<div class="candidate-tree" aria-label="{_label("Candidate data")}">'
        f'<p class="muted">{_label("Select server data from the tree.")}</p>'
        f'<ul class="tree-list root">{_render_candidate_nodes(children)}</ul>'
        "</div>"
    )


def _render_candidate_nodes(nodes: dict[str, object]) -> str:
    rendered = []
    for name, raw_node in sorted(nodes.items(), key=lambda item: item[0]):
        node = raw_node
        assert isinstance(node, dict)
        children = node.get("children", {})
        candidate = node.get("candidate")
        assert isinstance(children, dict)
        if candidate is not None:
            assert isinstance(candidate, DataCandidate)
            rendered.append(f"<li>{_candidate_button(name, candidate)}</li>")
            continue
        rendered.append(
            '<li><details class="tree-folder">'
            f"<summary>{_e(name)}</summary>"
            f'<ul class="tree-list">{_render_candidate_nodes(children)}</ul>'
            "</details></li>"
        )
    return "".join(rendered)


def _candidate_button(name: str, candidate: DataCandidate) -> str:
    suffix = f" · {candidate.count}" if candidate.count else ""
    return (
        '<button type="button" class="candidate-button" onclick="pickCandidate(this)" '
        f'data-path="{_e(str(candidate.path))}" '
        f'data-kind="{_e(candidate.kind)}" '
        f'data-stem="{_e(candidate.path.stem)}">'
        f"{_e(name)}"
        f'<span class="candidate-kind">{_kind_label(candidate.kind)}{_e(suffix)}</span>'
        "</button>"
    )


def _model_select(candidates: list[DataCandidate]) -> str:
    options = [f'<option value="">{_label("None")}</option>']
    options.extend(
        f'<option value="{_e(str(candidate.path))}">{_e(candidate.label)}</option>'
        for candidate in candidates
    )
    return f'<select name="existing_model_path">{"".join(options)}</select>'


def _source_kind_select() -> str:
    return _select(
        "source_kind",
        [
            ("image_folder", "image_folder (이미지 폴더)"),
            ("video_file", "video_file (동영상 파일)"),
            ("colmap_scene", "colmap_scene (COLMAP 씬)"),
        ],
        "image_folder",
    )


def _colmap_form() -> str:
    basic = f"""
    <div class="grid">
      <label><span>{_label("Preset")}</span>{_select("colmap_preset", [("default", "default"), ("video", "video"), ("low-memory", "low-memory"), ("hard-scene", "hard-scene")], "default")}</label>
      <label><span>{_label("Device")}</span>{_select("colmap_device", [("auto", "auto"), ("gpu", "gpu"), ("cpu", "cpu")], "auto")}</label>
      <label><span>{_label("Matcher")}</span>{_select("colmap_matcher", [("", "preset/default"), ("exhaustive", "exhaustive"), ("sequential", "sequential"), ("spatial", "spatial"), ("vocab_tree", "vocab_tree")], "")}</label>
      <label><span>{_label("Mapper")}</span>{_select("mapper_type", [("", "default"), ("incremental", "incremental"), ("global", "global")], "")}</label>
      <label><span>{_label("Camera mode")}</span>{_select("camera_mode", [("single", "single"), ("shared_off", "shared off"), ("per_folder", "per folder"), ("per_image", "per image")], "single")}</label>
      <label><span>{_label("Camera model")}</span><input name="camera" value="OPENCV"></label>
      <label><span>{_label("Feature type")}</span>{_select("feature_type", [("", "default"), ("SIFT", "SIFT"), ("ALIKED", "ALIKED")], "")}</label>
      <label><span>{_label("Matching type")}</span>{_select("matching_type", [("", "default"), ("SIFT_BRUTEFORCE", "SIFT_BRUTEFORCE"), ("SIFT_LIGHTGLUE", "SIFT_LIGHTGLUE"), ("ALIKED_BRUTEFORCE", "ALIKED_BRUTEFORCE"), ("ALIKED_LIGHTGLUE", "ALIKED_LIGHTGLUE")], "")}</label>
      <label><span>{_label("Camera params")}</span><input name="camera_params"></label>
    </div>
    <div class="checks">
      {_checkbox("skip_matching", "skip matching", False)}
      {_checkbox("resize", "create images_2/4/8", False)}
    </div>
    """
    advanced_fields = _field_grid(sorted(INT_COLMAP_FIELDS), "number")
    advanced_fields += _field_grid(sorted(FLOAT_COLMAP_FIELDS), "text")
    advanced_fields += _zero_one_grid(sorted(ZERO_ONE_FIELDS))
    advanced_fields += _field_grid(sorted(TEXT_COLMAP_FIELDS), "text")
    advanced_fields += _field_grid(sorted(PATH_COLMAP_FIELDS), "text")
    return basic + f"<details><summary>{_label('Advanced COLMAP options')}</summary>{advanced_fields}</details>"


def _training_form() -> str:
    defaults = {
        "sh_degree": "3",
        "images": "images",
        "resolution": "-1",
        "data_device": "cuda",
        "max_train_cameras": "0",
        "camera_quality_ratio": "0.7",
        "camera_selection_seed": "42",
        "pose_outlier_mad_scale": "8.0",
        "pose_outlier_min_cameras": "12",
        "iterations": "30000",
        "position_lr_init": "0.00016",
        "position_lr_final": "0.0000016",
        "position_lr_delay_mult": "0.01",
        "position_lr_max_steps": "30000",
        "feature_lr": "0.0025",
        "opacity_lr": "0.025",
        "scaling_lr": "0.005",
        "rotation_lr": "0.001",
        "exposure_lr_init": "0.01",
        "exposure_lr_final": "0.001",
        "exposure_lr_delay_steps": "0",
        "exposure_lr_delay_mult": "0.0",
        "percent_dense": "0.01",
        "lambda_dssim": "0.2",
        "densification_interval": "100",
        "opacity_reset_interval": "3000",
        "densify_from_iter": "500",
        "densify_until_iter": "15000",
        "densify_grad_threshold": "0.0002",
        "depth_l1_weight_init": "1.0",
        "depth_l1_weight_final": "0.01",
        "debug_from": "-1",
    }
    basic = f"""
    <div class="grid">
      <label><span>{_label("Iterations")}</span><input name="iterations" value="{defaults["iterations"]}" inputmode="numeric"></label>
      <label><span>{_label("Resolution")}</span><input name="resolution" value="{defaults["resolution"]}" inputmode="numeric"></label>
      <label><span>{_label("Images folder")}</span><input name="images" value="{defaults["images"]}"></label>
      <label><span>{_label("Depths folder")}</span><input name="depths"></label>
      <label><span>{_label("Data device")}</span>{_select("data_device", [("cuda", "cuda"), ("cpu", "cpu")], "cuda")}</label>
      <label><span>{_label("Optimizer")}</span>{_select("optimizer_type", [("default", "default"), ("sparse_adam", "sparse_adam")], "default")}</label>
      <label><span>{_label("Max train cameras")}</span><input name="max_train_cameras" value="{defaults["max_train_cameras"]}" inputmode="numeric"></label>
      <label><span>{_label("Camera quality ratio")}</span><input name="camera_quality_ratio" value="{defaults["camera_quality_ratio"]}" inputmode="decimal"></label>
      <label><span>{_label("Camera selection seed")}</span><input name="camera_selection_seed" value="{defaults["camera_selection_seed"]}" inputmode="numeric"></label>
      <label><span>{_label("Test iterations")}</span><input name="test_iterations" value="7000 30000"></label>
      <label><span>{_label("Save iterations")}</span><input name="save_iterations" value="7000 30000"></label>
      <label><span>{_label("Checkpoint iterations")}</span><input name="checkpoint_iterations"></label>
    </div>
    <div class="checks">
      {_checkbox("white_background", "white background", False)}
      {_checkbox("eval", "eval split", False)}
      {_checkbox("train_test_exp", "train/test exposure", False)}
      {_checkbox("antialiasing", "antialiasing", False)}
      {_checkbox("random_background", "random background", False)}
      {_checkbox("quiet", "quiet", False)}
    </div>
    """
    advanced_names = sorted((INT_TRAIN_FIELDS | FLOAT_TRAIN_FIELDS | TEXT_TRAIN_FIELDS) - {"iterations", "resolution", "images", "depths", "data_device", "optimizer_type", "max_train_cameras", "camera_quality_ratio", "camera_selection_seed"})
    advanced = _field_grid(advanced_names, "text", defaults)
    bools = "".join(
        _checkbox(name, name, False)
        for name in sorted(BOOL_TRAIN_FIELDS - {"white_background", "eval", "train_test_exp", "antialiasing", "random_background", "quiet"})
    )
    advanced += f'<div class="checks">{bools}</div>'
    advanced += f'<div class="grid"><label><span>{_field_label("start_checkpoint")}</span><input name="start_checkpoint"></label></div>'
    return basic + f"<details><summary>{_label('Advanced training options')}</summary>{advanced}</details>"


def _jobs_table(jobs: list[JobRecord]) -> str:
    if not jobs:
        return f'<p class="muted">{_label("No jobs yet.")}</p>'
    cards = []
    for job in jobs:
        cards.append(
            '<article class="job-card">'
            '<div class="job-card-head">'
            f'<a href="/jobs/{_e(job.id)}"><code>{_e(job.id)}</code></a>'
            f'<span class="status {_e(job.status)}">{_e(job.status)}</span>'
            "</div>"
            f'<div class="job-card-name">{_e(job.name)}</div>'
            '<dl class="job-card-meta">'
            f"<div><dt>{_label('Step')}</dt><dd>{_e(job.current_step or '-')}</dd></div>"
            f"<div><dt>{_label('Updated')}</dt><dd>{_e(job.updated_at)}</dd></div>"
            "</dl>"
            "</article>"
        )
    return f'<div class="job-list">{"".join(cards)}</div>'


def _job_detail(config: WebUIConfig, job: JobRecord, progress: TrainingProgress | None = None) -> str:
    spec = job.spec
    steps = "".join(
        f"<li><strong>{_e(step.name)}</strong><br><code>{_e(' '.join(step.argv))}</code></li>"
        for step in spec.steps
    )
    return f"""
<div hx-get="/partials/jobs/{_e(job.id)}/overview" hx-trigger="load, every 3s" hx-swap="innerHTML">
  {_job_overview(config, job)}
</div>
<details>
  <summary>{_label("Commands")}</summary>
  <ol>{steps}</ol>
</details>
<h3>{_label("Progress")}</h3>
<section hx-get="/partials/jobs/{_e(job.id)}/progress" hx-trigger="load, every 2s" hx-swap="innerHTML">{_job_progress(progress)}</section>
<h3>{_label("Log")}</h3>
<pre id="job-log" data-log-scroll hx-get="/jobs/{_e(job.id)}/log" hx-trigger="load, every 2s" hx-swap="innerHTML"></pre>
"""


def _job_overview(config: WebUIConfig, job: JobRecord) -> str:
    cancel = ""
    if job.status not in TERMINAL_STATUSES:
        cancel = f"""
        <form method="post" action="/jobs/{_e(job.id)}/cancel">
          <button class="danger" type="submit">{_label("Cancel")}</button>
        </form>
        """
    output_link = _output_link(config, job.model_path)
    return f"""
<h2>{_e(job.name)}</h2>
<p><span class="status {_e(job.status)}">{_e(job.status)}</span></p>
<div class="grid two">
  <p><span class="label">{_label("Source")}</span><br><span class="path">{_e(job.source_path)}</span></p>
  <p><span class="label">{_label("Scene")}</span><br><span class="path">{_e(job.scene_path)}</span></p>
  <p><span class="label">{_label("Model")}</span><br><span class="path">{output_link}</span></p>
  <p><span class="label">{_label("Current step")}</span><br>{_e(job.current_step or "-")}</p>
</div>
{f'<p class="error">{_e(job.error)}</p>' if job.error else ''}
<div class="actions">{cancel}</div>
"""


def _job_progress(progress: TrainingProgress | None) -> str:
    if progress is None or not progress.has_data:
        return f'<p class="muted">{_label("No training progress yet.")}</p>'

    pct = progress.percent if progress.percent is not None else 0.0
    pct = max(0.0, min(100.0, pct))
    if progress.iteration is not None and progress.total_iterations:
        count = f"{progress.iteration}/{progress.total_iterations}"
    elif progress.iteration is not None:
        count = str(progress.iteration)
    else:
        count = ""
    bar = (
        f'<div class="progress-head"><strong>{pct:.1f}%</strong>'
        f'<span class="count">{_e(count)}</span></div>'
        f'<div class="progressbar"><span style="width: {pct:.1f}%"></span></div>'
    )

    test_evals = [e for e in progress.evaluations if e.split == "test"]
    train_evals = [e for e in progress.evaluations if e.split == "train"]
    latest_test = test_evals[-1] if test_evals else None
    latest_train = train_evals[-1] if train_evals else None

    metrics = [
        _metric("Iteration", count or "-"),
        _metric("Loss", _fmt_num(progress.loss, 6)),
        _metric("Depth Loss", _fmt_num(progress.depth_loss, 6)),
        _metric("Rate", _fmt_num(progress.rate, 2), "it/s"),
        _metric("Elapsed", progress.elapsed or "-"),
        _metric("ETA", progress.eta or "-"),
    ]
    if latest_test is not None:
        metrics.append(_metric("Latest test PSNR", f"{latest_test.psnr:.2f}"))
    if latest_train is not None:
        metrics.append(_metric("Latest train PSNR", f"{latest_train.psnr:.2f}"))
    if progress.saved_iterations:
        metrics.append(_metric("Saved iterations", str(len(progress.saved_iterations))))
    metrics_html = f'<div class="metrics">{"".join(metrics)}</div>'

    charts = []
    loss_values = [sample.loss for sample in progress.loss_samples]
    if test_evals:
        charts.append(_chart("PSNR curve", [e.psnr for e in test_evals], "#0f766e"))
    if loss_values:
        charts.append(_chart("Loss curve", loss_values, "#b42318"))
    charts_html = f'<div class="charts">{"".join(charts)}</div>' if charts else ""

    return bar + metrics_html + charts_html


def _metric(label: str, value: str, unit: str = "") -> str:
    unit_html = f'<span class="unit">{_e(unit)}</span>' if unit else ""
    return (
        f'<div class="metric"><span class="label">{_label(label)}</span>'
        f'<span class="value">{_e(value)}{unit_html}</span></div>'
    )


def _chart(label: str, values: list[float], color: str) -> str:
    return (
        f'<div class="chart"><span class="label">{_label(label)}</span>'
        f"{_sparkline(values, color)}</div>"
    )


def _sparkline(values: list[float], color: str) -> str:
    if not values:
        return ""
    width, height, pad = 100.0, 40.0, 3.0
    low, high = min(values), max(values)
    span = (high - low) or 1.0
    if len(values) == 1:
        mid = height / 2
        points = f"{pad:.2f},{mid:.2f} {width - pad:.2f},{mid:.2f}"
    else:
        coords = []
        for index, value in enumerate(values):
            x = pad + (width - 2 * pad) * index / (len(values) - 1)
            y = pad + (height - 2 * pad) * (1 - (value - low) / span)
            coords.append(f"{x:.2f},{y:.2f}")
        points = " ".join(coords)
    return (
        f'<svg class="spark" viewBox="0 0 {width:.0f} {height:.0f}" preserveAspectRatio="none">'
        f'<polyline fill="none" stroke="{_e(color)}" stroke-width="1.5" '
        f'vector-effect="non-scaling-stroke" points="{_e(points)}"/>'
        "</svg>"
    )


def _fmt_num(value: float | None, digits: int) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _output_link(config: WebUIConfig, raw_path: str) -> str:
    if not raw_path:
        return "-"
    path = Path(raw_path)
    try:
        resolved = path.resolve()
        root = config.output_root.resolve()
        if is_within_directory(root, resolved):
            rel = resolved.relative_to(root)
            href = "/outputs/" + quote(str(rel), safe="/")
            return f'<a href="{_e(href)}">{_e(str(path))}</a>'
    except (OSError, ValueError):
        pass
    return _e(raw_path)


def _output_listing(config: WebUIConfig, directory: Path) -> str:
    rows = []
    for entry in sorted(directory.iterdir(), key=lambda item: (not item.is_dir(), item.name)):
        rel = entry.resolve().relative_to(config.output_root.resolve())
        href = "/outputs/" + quote(str(rel), safe="/")
        label = entry.name + ("/" if entry.is_dir() else "")
        rows.append(f'<tr><td><a href="{_e(href)}">{_e(label)}</a></td><td>{_e("dir" if entry.is_dir() else "file")}</td></tr>')
    return (
        f'<div class="actions"><a class="button secondary" href="/">{_label("Dashboard")}</a></div>'
        f'<p class="path">{_e(str(directory))}</p>'
        f'<section class="panel"><table><thead><tr><th>{_label("Name")}</th><th>{_label("Type")}</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></section>"
    )


def _resolve_output_path(config: WebUIConfig, raw_path: str) -> Path:
    target = (config.output_root / raw_path).resolve(strict=True)
    if not is_within_directory(config.output_root.resolve(), target):
        raise HTTPException(status_code=404)
    return target


def _field_grid(names: list[str], input_type: str, defaults: dict[str, str] | None = None) -> str:
    defaults = defaults or {}
    fields = []
    for name in names:
        value = defaults.get(name, "")
        fields.append(
            f'<label><span>{_field_label(name)}</span><input type="{_e(input_type)}" name="{_e(name)}" value="{_e(value)}"></label>'
        )
    return f'<div class="grid">{"".join(fields)}</div>'


def _zero_one_grid(names: list[str]) -> str:
    fields = []
    for name in names:
        fields.append(
            f'<label><span>{_field_label(name)}</span>{_select(name, [("", "default"), ("0", "0"), ("1", "1")], "")}</label>'
        )
    return f'<div class="grid">{"".join(fields)}</div>'


def _checkbox(name: str, label: str, checked: bool) -> str:
    state = " checked" if checked else ""
    return f'<label><input type="checkbox" name="{_e(name)}"{state}> {_label(label)}</label>'


def _stage_checkbox(name: str, label: str, checked: bool) -> str:
    state = " checked" if checked else ""
    return (
        f'<label><input type="checkbox" name="{_e(name)}"{state} '
        f'onchange="syncStages()"> {_label(label)}</label>'
    )


def _select(name: str, options: list[tuple[str, str]], selected: str) -> str:
    rendered = []
    for value, label in options:
        state = " selected" if value == selected else ""
        rendered.append(f'<option value="{_e(value)}"{state}>{_e(label)}</option>')
    return f'<select name="{_e(name)}">{"".join(rendered)}</select>'


def _error_page(message: str) -> str:
    return (
        f'<div class="actions"><a class="button secondary" href="/">{_label("Dashboard")}</a></div>'
        f'<section class="panel"><p class="error">{_e(message)}</p></section>'
    )


def _label(text: str) -> str:
    korean = LABEL_KO.get(text)
    if korean is None:
        return _e(text)
    return f"{_e(text)} ({_e(korean)})"


def _field_label(name: str) -> str:
    korean = FIELD_LABEL_KO.get(name, "고급 옵션")
    return f"{_e(name)} ({_e(korean)})"


def _kind_label(kind: str) -> str:
    korean = KIND_LABEL_KO.get(kind)
    if korean is None:
        return _e(kind)
    return f"{_e(kind)} ({_e(korean)})"


def _e(value: object) -> str:
    return html.escape(str(value), quote=True)


app = create_app()


def main() -> None:
    import uvicorn

    config: WebUIConfig = app.state.config
    host = resolve_bind_host(config)
    print(f"Starting 3DGS Web UI on http://{host}:{config.port}")
    uvicorn.run(app, host=host, port=config.port)


if __name__ == "__main__":
    main()
