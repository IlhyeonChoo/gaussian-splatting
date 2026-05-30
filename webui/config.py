"""Configuration loading for the 3DGS web UI."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


VALID_BIND_MODES = {"auto", "tailscale", "localhost"}


def _parse_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _split_env_list(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _split_paths(value: str | None) -> tuple[str, ...]:
    if not value:
        return ("data",)
    return tuple(item.strip() for item in value.split(os.pathsep) if item.strip())


def _resolve_repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _resolve_executable_path(repo_root: Path, value: str) -> str:
    """Return an absolute executable path without dereferencing venv symlinks."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return str(path.absolute())


def _resolve_python_bin(repo_root: Path, value: str | None) -> str:
    if value:
        if os.sep in value or value.startswith("."):
            return _resolve_executable_path(repo_root, value)
        return value

    venv_python = repo_root / "venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python.absolute())
    return sys.executable


@dataclass(frozen=True)
class WebUIConfig:
    """Runtime configuration for the web UI."""

    repo_root: Path
    data_roots: tuple[Path, ...]
    output_root: Path
    state_dir: Path
    log_dir: Path
    database_path: Path
    python_bin: str
    bind_mode: str
    host_override: str | None
    port: int
    unsafe_allow_all: bool
    allowed_cidrs: tuple[str, ...]

    @property
    def primary_data_root(self) -> Path:
        return self.data_roots[0]

    def ensure_runtime_dirs(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.primary_data_root.mkdir(parents=True, exist_ok=True)


def find_repo_root() -> Path:
    """Return the repository root, assuming this module lives under webui/."""
    return Path(__file__).resolve().parents[1]


def load_config(env: dict[str, str] | None = None) -> WebUIConfig:
    """Load configuration from environment variables."""
    env = env or os.environ
    repo_root = find_repo_root()

    data_roots = tuple(
        _resolve_repo_path(repo_root, value)
        for value in _split_paths(env.get("WEBUI_DATA_ROOTS"))
    )
    output_root = _resolve_repo_path(repo_root, env.get("WEBUI_OUTPUT_ROOT", "output"))
    state_dir = output_root / ".webui"
    log_dir = state_dir / "logs"
    database_path = state_dir / "jobs.sqlite3"

    bind_mode = env.get("WEBUI_BIND_MODE", "auto").strip().lower()
    if bind_mode not in VALID_BIND_MODES:
        valid = ", ".join(sorted(VALID_BIND_MODES))
        raise ValueError(f"WEBUI_BIND_MODE must be one of: {valid}")

    port_raw = env.get("WEBUI_PORT", "7860").strip()
    try:
        port = int(port_raw)
    except ValueError as exc:
        raise ValueError("WEBUI_PORT must be an integer.") from exc
    if not (1 <= port <= 65535):
        raise ValueError("WEBUI_PORT must be between 1 and 65535.")

    host_override = env.get("WEBUI_HOST")
    if host_override is not None:
        host_override = host_override.strip() or None

    config = WebUIConfig(
        repo_root=repo_root,
        data_roots=data_roots,
        output_root=output_root,
        state_dir=state_dir,
        log_dir=log_dir,
        database_path=database_path,
        python_bin=_resolve_python_bin(repo_root, env.get("WEBUI_PYTHON_BIN")),
        bind_mode=bind_mode,
        host_override=host_override,
        port=port,
        unsafe_allow_all=_parse_bool(env.get("WEBUI_UNSAFE_ALLOW_ALL")),
        allowed_cidrs=_split_env_list(env.get("WEBUI_ALLOWED_CIDRS")),
    )
    config.ensure_runtime_dirs()
    return config
