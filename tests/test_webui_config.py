from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from webui.config import _resolve_python_bin


class ConfigTest(unittest.TestCase):
    def test_default_python_bin_preserves_venv_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bindir = root / "venv" / "bin"
            bindir.mkdir(parents=True)
            python = bindir / "python"
            python.symlink_to(sys.executable)

            resolved = _resolve_python_bin(root, None)

            self.assertEqual(resolved, str(python.absolute()))
            self.assertNotEqual(resolved, str(Path(sys.executable).resolve()))

    def test_explicit_relative_python_bin_preserves_venv_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bindir = root / "venv" / "bin"
            bindir.mkdir(parents=True)
            python = bindir / "python"
            python.symlink_to(sys.executable)

            resolved = _resolve_python_bin(root, "venv/bin/python")

            self.assertEqual(resolved, str(python.absolute()))
            self.assertNotEqual(resolved, str(Path(sys.executable).resolve()))


if __name__ == "__main__":
    unittest.main()
