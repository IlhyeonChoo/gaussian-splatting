#!/usr/bin/env python3
"""Undistort existing COLMAP scene directories for 3DGS training."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import struct
import subprocess


CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}

REQUIRED_MODEL_FILES = ("cameras.bin", "images.bin", "points3D.bin")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-roots", nargs="+", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--colmap", default="colmap")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--scenes", nargs="*", default=None)
    return parser.parse_args()


def validate_model_path(model_path: Path) -> None:
    for filename in REQUIRED_MODEL_FILES:
        path = model_path / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing COLMAP model file: {path}")


def discover_scenes(input_roots: list[Path], selected_scenes: list[str] | None) -> list[Path]:
    scenes: list[Path] = []
    selected = set(selected_scenes or [])
    for root in input_roots:
        for scene_dir in sorted(root.iterdir()):
            if not scene_dir.is_dir():
                continue
            if selected and scene_dir.name not in selected:
                continue
            if (scene_dir / "images").is_dir() and (scene_dir / "sparse" / "0" / "images.bin").is_file():
                scenes.append(scene_dir)
    if selected:
        found = {scene.name for scene in scenes}
        missing = selected - found
        if missing:
            raise FileNotFoundError(f"Requested scenes not found: {', '.join(sorted(missing))}")
    return scenes


def normalize_sparse_zero(scene_dir: Path) -> Path:
    sparse_root = scene_dir / "sparse"
    sparse_zero = sparse_root / "0"
    if (sparse_zero / "images.bin").is_file():
        validate_model_path(sparse_zero)
        return sparse_zero

    loose_files = [path for path in sparse_root.iterdir() if path.is_file()]
    if not loose_files:
        raise FileNotFoundError(f"No sparse model files found under {sparse_root}")

    sparse_zero.mkdir(parents=True, exist_ok=True)
    for source in loose_files:
        shutil.move(str(source), sparse_zero / source.name)

    validate_model_path(sparse_zero)
    return sparse_zero


def read_camera_models(cameras_bin: Path) -> list[str]:
    models: list[str] = []
    with cameras_bin.open("rb") as file:
        num_cameras = struct.unpack("<Q", file.read(8))[0]
        for _ in range(num_cameras):
            _, model_id, _, _ = struct.unpack("<iiQQ", file.read(24))
            model_name, num_params = CAMERA_MODELS[model_id]
            file.seek(num_params * 8, 1)
            models.append(model_name)
    return models


def count_images(images_dir: Path) -> int:
    return sum(
        1
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def run_image_undistorter(colmap: str, scene_dir: Path, output_dir: Path, log_path: Path) -> None:
    command = [
        colmap,
        "image_undistorter",
        "--image_path",
        str(scene_dir / "images"),
        "--input_path",
        str(scene_dir / "sparse" / "0"),
        "--output_path",
        str(output_dir),
        "--output_type",
        "COLMAP",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        subprocess.run(command, check=True, stdout=log_file, stderr=subprocess.STDOUT)


def write_summary(output_root: Path, rows: list[dict[str, object]]) -> None:
    headers = [
        "scene",
        "source",
        "undistorted_images",
        "camera_models",
        "output",
    ]
    lines = ["\t".join(headers)]
    for row in rows:
        lines.append("\t".join(str(row[header]) for header in headers))
    (output_root / "undistort_summary.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_roots = [root.resolve() for root in args.input_roots]
    output_root = args.output_root.resolve()
    scenes = discover_scenes(input_roots, args.scenes)
    if not scenes:
        raise FileNotFoundError("No COLMAP scenes found")

    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for scene_dir in scenes:
        output_dir = output_root / scene_dir.name
        if output_dir.exists():
            if not args.force:
                raise FileExistsError(f"Output scene already exists: {output_dir}")
            shutil.rmtree(output_dir)

        log_path = output_root / "colmap_logs" / f"{scene_dir.name}.log"
        run_image_undistorter(args.colmap, scene_dir, output_dir, log_path)
        sparse_zero = normalize_sparse_zero(output_dir)
        camera_models = sorted(set(read_camera_models(sparse_zero / "cameras.bin")))
        image_count = count_images(output_dir / "images")
        rows.append(
            {
                "scene": scene_dir.name,
                "source": scene_dir,
                "undistorted_images": image_count,
                "camera_models": ",".join(camera_models),
                "output": output_dir,
            }
        )
        print(
            f"{scene_dir.name}: undistorted_images={image_count} "
            f"camera_models={','.join(camera_models)} output={output_dir}",
            flush=True,
        )

    write_summary(output_root, rows)
    print(f"summary: {output_root / 'undistort_summary.tsv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
