from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from webui.data_browser import discover_data_candidates, is_colmap_scene_dir, require_allowed_path


class DataBrowserTest(unittest.TestCase):
    def test_require_allowed_path_blocks_symlink_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            data.mkdir()
            inside = data / "images"
            inside.mkdir()
            outside = root / "outside"
            outside.mkdir()
            escaped = data / "escaped"
            escaped.symlink_to(outside, target_is_directory=True)

            self.assertEqual(require_allowed_path(inside, (data,)), inside.resolve())
            with self.assertRaises(ValueError):
                require_allowed_path(escaped, (data,))

    def test_colmap_scene_detection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp) / "scene"
            (scene / "images").mkdir(parents=True)
            sparse = scene / "sparse" / "0"
            sparse.mkdir(parents=True)
            (sparse / "images.bin").write_bytes(b"")
            (sparse / "cameras.bin").write_bytes(b"")
            (sparse / "points3D.ply").write_text("ply\n", encoding="utf-8")

            self.assertTrue(is_colmap_scene_dir(scene))

    def test_discover_candidates_finds_images_video_and_scene(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            data = repo / "data"
            image_dir = data / "raw"
            image_dir.mkdir(parents=True)
            (image_dir / "a.jpg").write_bytes(b"fake")
            (data / "clip.mp4").write_bytes(b"fake")
            scene = data / "scene"
            (scene / "images").mkdir(parents=True)
            sparse = scene / "sparse" / "0"
            sparse.mkdir(parents=True)
            (sparse / "images.bin").write_bytes(b"")
            (sparse / "cameras.bin").write_bytes(b"")
            (sparse / "points3D.bin").write_bytes(b"")

            candidates = discover_data_candidates((data,), repo)
            kinds = {candidate.kind for candidate in candidates}
            self.assertIn("image_folder", kinds)
            self.assertIn("video_file", kinds)
            self.assertIn("colmap_scene", kinds)


if __name__ == "__main__":
    unittest.main()

