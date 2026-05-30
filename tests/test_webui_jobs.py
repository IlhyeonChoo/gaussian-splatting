from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path

from webui.commands import CommandStep, JobSpec
from webui.config import WebUIConfig
from webui.jobs import JobManager, TERMINAL_STATUSES


def make_config(root: Path) -> WebUIConfig:
    output = root / "output"
    data = root / "data"
    state = output / ".webui"
    log_dir = state / "logs"
    log_dir.mkdir(parents=True)
    data.mkdir(parents=True)
    return WebUIConfig(
        repo_root=root,
        data_roots=(data,),
        output_root=output,
        state_dir=state,
        log_dir=log_dir,
        database_path=state / "jobs.sqlite3",
        python_bin=sys.executable,
        bind_mode="auto",
        host_override=None,
        port=7860,
        unsafe_allow_all=False,
        allowed_cidrs=(),
    )


class JobsTest(unittest.TestCase):
    def test_queue_executes_successful_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = make_config(root)
            manager = JobManager(config)
            manager.start()
            try:
                spec = JobSpec(
                    name="smoke",
                    source_kind="image_folder",
                    source_path=str(config.primary_data_root),
                    scene_path=str(config.primary_data_root / "scene"),
                    model_path=str(config.output_root / "model"),
                    steps=[
                        CommandStep(
                            name="print",
                            argv=[sys.executable, "-c", "print('job ok')"],
                            cwd=str(root),
                        )
                    ],
                )
                job_id = manager.enqueue(spec)

                deadline = time.time() + 5
                job = manager.get_job(job_id)
                while job and job.status not in TERMINAL_STATUSES and time.time() < deadline:
                    time.sleep(0.05)
                    job = manager.get_job(job_id)

                self.assertIsNotNone(job)
                assert job is not None
                self.assertEqual(job.status, "succeeded")
                self.assertIn("job ok", manager.get_log_tail(job_id))
            finally:
                manager.stop()

    def test_get_progress_parses_job_log_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = make_config(root)
            manager = JobManager(config)
            spec = JobSpec(
                name="progress",
                source_kind="image_folder",
                source_path=str(config.primary_data_root),
                scene_path=str(config.primary_data_root / "scene"),
                model_path=str(config.output_root / "model"),
                steps=[],
            )
            job_id = manager.enqueue(spec)
            job = manager.get_job(job_id)
            self.assertIsNotNone(job)
            assert job is not None
            Path(job.log_path).write_text(
                "Training progress:  20%|██  | 6000/30000 "
                "[01:00<04:00, 100.00it/s, Loss=0.05, Depth Loss=0.0]\n"
                "[ITER 6000] Evaluating test: L1 0.02 PSNR 31.2\n",
                encoding="utf-8",
            )

            progress = manager.get_progress(job_id)

            self.assertTrue(progress.has_data)
            self.assertEqual(progress.iteration, 6000)
            self.assertEqual(progress.total_iterations, 30000)
            self.assertAlmostEqual(progress.loss, 0.05)
            self.assertEqual(len(progress.loss_samples), 1)
            self.assertEqual(progress.evaluations[-1].psnr, 31.2)


if __name__ == "__main__":
    unittest.main()

