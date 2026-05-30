from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from webui.config import WebUIConfig
from webui.security import is_client_allowed, validate_bind_host


def make_config(root: Path, *, unsafe: bool = False, allowed_cidrs: tuple[str, ...] = ()) -> WebUIConfig:
    output = root / "output"
    return WebUIConfig(
        repo_root=root,
        data_roots=(root / "data",),
        output_root=output,
        state_dir=output / ".webui",
        log_dir=output / ".webui" / "logs",
        database_path=output / ".webui" / "jobs.sqlite3",
        python_bin="python",
        bind_mode="auto",
        host_override=None,
        port=7860,
        unsafe_allow_all=unsafe,
        allowed_cidrs=allowed_cidrs,
    )


class SecurityTest(unittest.TestCase):
    def test_bind_host_rejects_all_interfaces_without_unsafe(self) -> None:
        with self.assertRaises(ValueError):
            validate_bind_host("0.0.0.0", unsafe_allow_all=False)
        with self.assertRaises(ValueError):
            validate_bind_host("192.168.1.10", unsafe_allow_all=False)

    def test_bind_host_allows_loopback_and_tailscale(self) -> None:
        validate_bind_host("127.0.0.1", unsafe_allow_all=False)
        validate_bind_host("localhost", unsafe_allow_all=False)
        validate_bind_host("100.80.1.2", unsafe_allow_all=False)

    def test_client_filter_defaults_to_loopback_and_tailscale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(Path(tmp))
            self.assertTrue(is_client_allowed("127.0.0.1", config))
            self.assertTrue(is_client_allowed("100.64.0.5", config))
            self.assertFalse(is_client_allowed("192.168.1.10", config))

    def test_unsafe_allows_all_clients(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(Path(tmp), unsafe=True)
            self.assertTrue(is_client_allowed("192.168.1.10", config))


if __name__ == "__main__":
    unittest.main()

