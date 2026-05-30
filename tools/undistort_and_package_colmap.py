#!/usr/bin/env python3
"""Undistort COLMAP split scenes and package them for 3DGS/Colab."""

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


def count_registered_images(images_bin: Path) -> int:
    """Return the number of registered images in COLMAP images.bin."""
    with images_bin.open("rb") as file:
        num_images = struct.unpack("<Q", file.read(8))[0]
        for _ in range(num_images):
            file.seek(64, 1)

            while True:
                current_byte = file.read(1)
                if current_byte == b"\x00":
                    break
                if current_byte == b"":
                    raise ValueError(f"Unexpected EOF while reading image name from {images_bin}")

            num_points_2d = struct.unpack("<Q", file.read(8))[0]
            file.seek(num_points_2d * 24, 1)

    return num_images


def read_camera_models(cameras_bin: Path) -> list[str]:
    """Return camera model names from COLMAP cameras.bin."""
    models: list[str] = []
    with cameras_bin.open("rb") as file:
        num_cameras = struct.unpack("<Q", file.read(8))[0]
        for _ in range(num_cameras):
            _, model_id, _, _ = struct.unpack("<iiQQ", file.read(24))
            model_name, num_params = CAMERA_MODELS[model_id]
            file.seek(num_params * 8, 1)
            models.append(model_name)
    return models


def validate_model_path(model_path: Path) -> None:
    for filename in REQUIRED_MODEL_FILES:
        path = model_path / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing COLMAP model file: {path}")


def find_best_sparse_model(scene_split_dir: Path) -> tuple[Path, int, list[tuple[str, int]]]:
    sparse_root = scene_split_dir / "sparse"
    candidates: list[tuple[Path, int]] = []

    loose_images = sparse_root / "images.bin"
    if loose_images.is_file():
        candidates.append((sparse_root, count_registered_images(loose_images)))

    for entry in sorted(sparse_root.iterdir(), key=lambda path: path.name):
        if not entry.is_dir():
            continue
        images_bin = entry / "images.bin"
        if images_bin.is_file():
            candidates.append((entry, count_registered_images(images_bin)))

    if not candidates:
        raise FileNotFoundError(f"No sparse COLMAP models found under {sparse_root}")

    for model_path, _ in candidates:
        validate_model_path(model_path)

    candidates.sort(key=lambda item: (-item[1], item[0].name))
    best_path, best_count = candidates[0]
    details = [(path.name, count) for path, count in candidates]
    return best_path, best_count, details


def normalize_sparse_zero(scene_dir: Path) -> Path:
    sparse_root = scene_dir / "sparse"
    sparse_zero = sparse_root / "0"
    if (sparse_zero / "images.bin").is_file():
        validate_model_path(sparse_zero)
        return sparse_zero

    loose_files = [path for path in sparse_root.iterdir() if path.is_file()]
    if not loose_files:
        raise FileNotFoundError(f"No loose sparse files found under {sparse_root}")

    sparse_zero.mkdir(parents=True, exist_ok=True)
    for src in loose_files:
        shutil.move(str(src), sparse_zero / src.name)

    validate_model_path(sparse_zero)
    return sparse_zero


def count_images(images_dir: Path) -> int:
    suffixes = {".jpg", ".jpeg", ".png", ".webp"}
    return sum(1 for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in suffixes)


def run_image_undistorter(colmap: str, input_dir: Path, model_path: Path, output_dir: Path) -> None:
    command = [
        colmap,
        "image_undistorter",
        "--image_path",
        str(input_dir),
        "--input_path",
        str(model_path),
        "--output_path",
        str(output_dir),
        "--output_type",
        "COLMAP",
    ]
    subprocess.run(command, check=True)


def create_archive(scene_dir: Path, archive_path: Path, compression_level: int, force: bool) -> None:
    if archive_path.exists():
        if not force:
            raise FileExistsError(f"Archive already exists: {archive_path}")
        archive_path.unlink()

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "tar",
        "-C",
        str(scene_dir),
        f"--use-compress-program=gzip -{compression_level}",
        "-cf",
        str(archive_path),
        "images",
        "sparse",
    ]
    subprocess.run(command, check=True)
    subprocess.run(["gzip", "-t", str(archive_path)], check=True)


def process_scene(args: argparse.Namespace, label: str) -> dict[str, object]:
    split_scene = args.split_root / label
    input_dir = split_scene / "input"
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Missing input directory: {input_dir}")

    best_model, registered_count, model_details = find_best_sparse_model(split_scene)
    output_scene = args.scene_root / label
    archive_path = args.archive_root / f"{args.prefix}_{label}.tar.gz"

    if output_scene.exists():
        if not args.force:
            raise FileExistsError(f"Output scene already exists: {output_scene}")
        shutil.rmtree(output_scene)
    output_scene.parent.mkdir(parents=True, exist_ok=True)

    run_image_undistorter(args.colmap, input_dir, best_model, output_scene)
    sparse_zero = normalize_sparse_zero(output_scene)
    models = sorted(set(read_camera_models(sparse_zero / "cameras.bin")))
    image_count = count_images(output_scene / "images")
    create_archive(output_scene, archive_path, args.compression_level, args.force)

    return {
        "label": label,
        "best_model": str(best_model),
        "registered_cameras": registered_count,
        "model_details": ",".join(f"{name}:{count}" for name, count in model_details),
        "undistorted_images": image_count,
        "camera_models": ",".join(models),
        "archive": str(archive_path),
        "archive_size_gib": archive_path.stat().st_size / (1024**3),
    }


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    headers = [
        "label",
        "best_model",
        "registered_cameras",
        "model_details",
        "undistorted_images",
        "camera_models",
        "archive",
        "archive_size_gib",
    ]
    lines = ["\t".join(headers)]
    for row in rows:
        lines.append("\t".join(str(row[header]) for header in headers))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--prefix", default="W2_4_3")
    parser.add_argument("--compression-level", type=int, default=1)
    parser.add_argument("--colmap", default="colmap")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("labels", nargs="+")
    args = parser.parse_args()

    args.split_root = args.split_root.resolve()
    args.scene_root = args.scene_root.resolve()
    args.archive_root = args.archive_root.resolve()
    args.scene_root.mkdir(parents=True, exist_ok=True)
    args.archive_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for label in args.labels:
        row = process_scene(args, label)
        rows.append(row)
        print(
            f"{label}: best={row['best_model']} cameras={row['registered_cameras']} "
            f"undistorted_images={row['undistorted_images']} "
            f"camera_models={row['camera_models']} "
            f"archive={row['archive']} size={row['archive_size_gib']:.2f}G",
            flush=True,
        )

    summary_path = args.archive_root / f"{args.prefix}_undistort_package_summary.tsv"
    write_summary(summary_path, rows)
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
