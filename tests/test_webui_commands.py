from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from webui.commands import build_job_spec
from webui.config import WebUIConfig


def make_config(root: Path) -> WebUIConfig:
    output = root / "output"
    data = root / "data"
    output.mkdir(parents=True)
    data.mkdir(parents=True)
    return WebUIConfig(
        repo_root=root,
        data_roots=(data,),
        output_root=output,
        state_dir=output / ".webui",
        log_dir=output / ".webui" / "logs",
        database_path=output / ".webui" / "jobs.sqlite3",
        python_bin="python",
        bind_mode="auto",
        host_override=None,
        port=7860,
        unsafe_allow_all=False,
        allowed_cidrs=(),
    )


class CommandsTest(unittest.TestCase):
    def test_builds_image_colmap_train_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = make_config(root)
            images = config.primary_data_root / "raw"
            images.mkdir()
            (images / "a.jpg").write_bytes(b"fake")

            spec = build_job_spec(
                {
                    "source_kind": "image_folder",
                    "source_path": str(images),
                    "scene_name": "my scene",
                    "model_output_name": "my model",
                    "stage_prepare": "on",
                    "stage_colmap": "on",
                    "stage_train": "on",
                    "colmap_preset": "low-memory",
                    "colmap_device": "cpu",
                    "camera_mode": "single",
                    "iterations": "10",
                    "optimizer_type": "default",
                    "data_device": "cuda",
                    "extra_feature_args": "--SiftExtraction.domain_size_pooling 1",
                },
                config,
            )

            self.assertEqual([step.name for step in spec.steps], ["prepare", "colmap", "train"])
            colmap_argv = spec.steps[1].argv
            self.assertIn("--colmap_device", colmap_argv)
            self.assertIn("cpu", colmap_argv)
            self.assertIn("--feature_max_image_size", colmap_argv)
            self.assertIn("--extra_feature_args", colmap_argv)
            raw_index = colmap_argv.index("--extra_feature_args") + 1
            self.assertEqual(colmap_argv[raw_index], "--SiftExtraction.domain_size_pooling 1")
            self.assertIn("--disable_viewer", spec.steps[2].argv)
            self.assertTrue(spec.scene_path.endswith("my_scene"))
            self.assertTrue(spec.model_path.endswith("my_model"))

    def test_render_without_training_uses_existing_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = make_config(root)
            scene = config.primary_data_root / "scene"
            (scene / "images").mkdir(parents=True)
            sparse = scene / "sparse" / "0"
            sparse.mkdir(parents=True)
            (sparse / "images.bin").write_bytes(b"")
            (sparse / "cameras.bin").write_bytes(b"")
            (sparse / "points3D.bin").write_bytes(b"")
            model = config.output_root / "model"
            model.mkdir()
            (model / "cfg_args").write_text("Namespace()", encoding="utf-8")

            spec = build_job_spec(
                {
                    "source_kind": "colmap_scene",
                    "source_path": str(scene),
                    "stage_render": "on",
                    "existing_model_path": str(model),
                },
                config,
            )

            self.assertEqual([step.name for step in spec.steps], ["render"])
            self.assertEqual(spec.model_path, str(model.resolve()))


if __name__ == "__main__":
    unittest.main()

