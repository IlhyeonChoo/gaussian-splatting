"""SQLite-backed single-worker job queue for web UI workflows."""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .commands import JobSpec
from .config import WebUIConfig
from .progress import TrainingProgress, parse_training_progress


TERMINAL_STATUSES = {"succeeded", "failed", "canceled"}


@dataclass(frozen=True)
class JobRecord:
    id: str
    name: str
    status: str
    created_at: str
    updated_at: str
    source_path: str
    scene_path: str
    model_path: str
    current_step: str
    log_path: str
    spec_json: str
    pid: int | None
    return_code: int | None
    error: str

    @property
    def spec(self) -> JobSpec:
        return JobSpec.from_dict(json.loads(self.spec_json))


class JobManager:
    """Manage queued workflow jobs and execute one job at a time."""

    def __init__(self, config: WebUIConfig):
        self.config = config
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._initialize_database()
        self._recover_interrupted_jobs()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.config.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    scene_path TEXT NOT NULL,
                    model_path TEXT NOT NULL,
                    current_step TEXT NOT NULL DEFAULT '',
                    log_path TEXT NOT NULL,
                    spec_json TEXT NOT NULL,
                    pid INTEGER,
                    return_code INTEGER,
                    error TEXT NOT NULL DEFAULT ''
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs(status, created_at)")

    def _recover_interrupted_jobs(self) -> None:
        now = _now()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE jobs
                SET status = 'failed',
                    updated_at = ?,
                    pid = NULL,
                    error = 'Web UI restarted before this job finished.'
                WHERE status IN ('running', 'canceling')
                """,
                (now,),
            )

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._worker_loop, name="webui-job-worker", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def enqueue(self, spec: JobSpec) -> str:
        job_id = uuid.uuid4().hex[:12]
        now = _now()
        log_path = self.config.log_dir / f"{job_id}.log"
        payload = json.dumps(spec.to_dict(), ensure_ascii=False)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO jobs (
                    id, name, status, created_at, updated_at, source_path, scene_path,
                    model_path, log_path, spec_json
                )
                VALUES (?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    spec.name,
                    now,
                    now,
                    spec.source_path,
                    spec.scene_path,
                    spec.model_path,
                    str(log_path),
                    payload,
                ),
            )
        _write_log(log_path, f"[{now}] queued job {job_id}: {spec.name}\n")
        return job_id

    def list_jobs(self, limit: int = 50) -> list[JobRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [_job_from_row(row) for row in rows]

    def get_job(self, job_id: str) -> JobRecord | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return _job_from_row(row) if row else None

    def get_log_tail(self, job_id: str, max_bytes: int = 120_000) -> str:
        job = self.get_job(job_id)
        if not job:
            return ""
        path = Path(job.log_path)
        if not path.exists():
            return ""
        with path.open("rb") as file:
            file.seek(0, os.SEEK_END)
            size = file.tell()
            file.seek(max(0, size - max_bytes))
            data = file.read()
        return data.decode("utf-8", errors="replace")

    def get_progress(self, job_id: str, max_bytes: int = 5_000_000) -> TrainingProgress:
        """Parse the job log into structured training progress.

        Reads up to ``max_bytes`` from the end of the log file. Evaluation and
        ``Saving Gaussians`` markers are sparse, so the tail comfortably covers
        a full run while bounding memory for very long logs.
        """
        job = self.get_job(job_id)
        if not job:
            return TrainingProgress()
        path = Path(job.log_path)
        if not path.exists():
            return TrainingProgress()
        with path.open("rb") as file:
            file.seek(0, os.SEEK_END)
            size = file.tell()
            file.seek(max(0, size - max_bytes))
            data = file.read()
        return parse_training_progress(data.decode("utf-8", errors="replace"))

    def cancel_job(self, job_id: str) -> bool:
        job = self.get_job(job_id)
        if not job or job.status in TERMINAL_STATUSES:
            return False

        now = _now()
        if job.status == "queued":
            with self._connect() as connection:
                connection.execute(
                    "UPDATE jobs SET status = 'canceled', updated_at = ?, error = 'Canceled before start.' WHERE id = ?",
                    (now, job_id),
                )
            _write_log(Path(job.log_path), f"[{now}] canceled before start\n")
            return True

        with self._connect() as connection:
            connection.execute(
                "UPDATE jobs SET status = 'canceling', updated_at = ? WHERE id = ?",
                (now, job_id),
            )
        if job.pid:
            _terminate_process_group(job.pid)
        _write_log(Path(job.log_path), f"[{now}] cancellation requested\n")
        return True

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            job = self._claim_next_job()
            if not job:
                time.sleep(1)
                continue
            self._execute_job(job)

    def _claim_next_job(self) -> JobRecord | None:
        now = _now()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE status = 'queued' ORDER BY created_at ASC LIMIT 1"
            ).fetchone()
            if not row:
                return None
            connection.execute(
                "UPDATE jobs SET status = 'running', updated_at = ? WHERE id = ?",
                (now, row["id"]),
            )
        claimed = self.get_job(row["id"])
        return claimed

    def _execute_job(self, job: JobRecord) -> None:
        log_path = Path(job.log_path)
        spec = job.spec
        _write_log(log_path, f"[{_now()}] started job {job.id}\n")
        for index, step in enumerate(spec.steps, start=1):
            if self._is_cancel_requested(job.id):
                self._finish_job(job.id, "canceled", None, "Canceled before next step.")
                _write_log(log_path, f"[{_now()}] canceled before step {step.name}\n")
                return

            self._set_step(job.id, step.name)
            _write_log(log_path, f"\n[{_now()}] step {index}/{len(spec.steps)}: {step.name}\n")
            _write_log(log_path, "$ " + " ".join(step.argv) + "\n")
            return_code = self._run_step(job.id, step.argv, step.cwd, log_path)

            if self._is_cancel_requested(job.id):
                self._finish_job(job.id, "canceled", return_code, "Canceled by user.")
                _write_log(log_path, f"[{_now()}] canceled\n")
                return
            if return_code != 0:
                self._finish_job(job.id, "failed", return_code, f"Step '{step.name}' failed.")
                _write_log(log_path, f"[{_now()}] failed with code {return_code}\n")
                return

        self._finish_job(job.id, "succeeded", 0, "")
        _write_log(log_path, f"[{_now()}] succeeded\n")

    def _run_step(self, job_id: str, argv: list[str], cwd: str, log_path: Path) -> int:
        try:
            process = subprocess.Popen(
                argv,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                start_new_session=True,
            )
        except OSError as exc:
            _write_log(log_path, f"[{_now()}] failed to start process: {exc}\n")
            return 127

        self._set_pid(job_id, process.pid)
        assert process.stdout is not None
        with process.stdout, log_path.open("a", encoding="utf-8", errors="replace") as log_file:
            for line in process.stdout:
                log_file.write(line)
                log_file.flush()
        return_code = process.wait()
        self._set_pid(job_id, None)
        return return_code

    def _is_cancel_requested(self, job_id: str) -> bool:
        job = self.get_job(job_id)
        return bool(job and job.status in {"canceling", "canceled"})

    def _set_pid(self, job_id: str, pid: int | None) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE jobs SET pid = ?, updated_at = ? WHERE id = ?",
                (pid, _now(), job_id),
            )

    def _set_step(self, job_id: str, step_name: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE jobs SET current_step = ?, updated_at = ? WHERE id = ?",
                (step_name, _now(), job_id),
            )

    def _finish_job(self, job_id: str, status: str, return_code: int | None, error: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE jobs
                SET status = ?, updated_at = ?, current_step = '', pid = NULL,
                    return_code = ?, error = ?
                WHERE id = ?
                """,
                (status, _now(), return_code, error, job_id),
            )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _job_from_row(row: sqlite3.Row) -> JobRecord:
    return JobRecord(
        id=row["id"],
        name=row["name"],
        status=row["status"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        source_path=row["source_path"],
        scene_path=row["scene_path"],
        model_path=row["model_path"],
        current_step=row["current_step"],
        log_path=row["log_path"],
        spec_json=row["spec_json"],
        pid=row["pid"],
        return_code=row["return_code"],
        error=row["error"],
    )


def _write_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", errors="replace") as file:
        file.write(text)


def _terminate_process_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            return

