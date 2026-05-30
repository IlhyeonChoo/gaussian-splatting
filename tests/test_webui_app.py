from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from webui.app import _job_detail, _jobs_table
from webui.commands import JobSpec
from webui.config import WebUIConfig
from webui.jobs import JobRecord


def make_config(root: Path) -> WebUIConfig:
    output = root / "output"
    data = root / "data"
    state = output / ".webui"
    log_dir = state / "logs"
    output.mkdir(parents=True)
    data.mkdir(parents=True)
    log_dir.mkdir(parents=True)
    return WebUIConfig(
        repo_root=root,
        data_roots=(data,),
        output_root=output,
        state_dir=state,
        log_dir=log_dir,
        database_path=state / "jobs.sqlite3",
        python_bin="python",
        bind_mode="auto",
        host_override=None,
        port=7860,
        unsafe_allow_all=False,
        allowed_cidrs=(),
    )


def make_job(root: Path, *, job_id: str = "abc123", status: str = "running") -> JobRecord:
    spec = JobSpec(
        name="image_folder:very_long_running_job_name_that_should_wrap",
        source_kind="image_folder",
        source_path=str(root / "data" / "input"),
        scene_path=str(root / "data" / "scene"),
        model_path=str(root / "output" / "model"),
        steps=[],
    )
    return JobRecord(
        id=job_id,
        name=spec.name,
        status=status,
        created_at="2026-05-30T00:00:00+00:00",
        updated_at="2026-05-30T00:01:00+00:00",
        source_path=spec.source_path,
        scene_path=spec.scene_path,
        model_path=spec.model_path,
        current_step="colmap",
        log_path=str(root / "output" / ".webui" / "logs" / f"{job_id}.log"),
        spec_json=json.dumps(spec.to_dict()),
        pid=123,
        return_code=None,
        error="",
    )


class AppRenderTest(unittest.TestCase):
    def test_jobs_dashboard_uses_cards_for_narrow_sidebar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            html = _jobs_table([make_job(Path(tmp))])

        self.assertIn('class="job-list"', html)
        self.assertIn('class="job-card"', html)
        self.assertIn("very_long_running_job_name", html)
        self.assertNotIn("<table", html)

    def test_job_detail_does_not_poll_entire_panel_or_reset_log_node(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            html = _job_detail(make_config(root), make_job(root))

        self.assertIn("/partials/jobs/abc123/overview", html)
        self.assertIn("/partials/jobs/abc123/progress", html)
        self.assertIn('data-log-scroll', html)
        self.assertIn('id="job-log"', html)
        self.assertNotIn('id="job-detail" hx-get', html)


if __name__ == "__main__":
    unittest.main()
