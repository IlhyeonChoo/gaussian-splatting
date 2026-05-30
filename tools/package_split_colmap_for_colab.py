#!/usr/bin/env python3
"""Package split COLMAP scenes as Colab-ready 3DGS tar archives."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess


SPARSE_FILES = ("rigs.bin", "cameras.bin", "frames.bin", "images.bin", "points3D.bin")


def copy_file(src: Path, dst: Path) -> bool:
    """Copy src to dst, skipping existing files with the same size."""
    if dst.exists():
        if dst.is_file() and dst.stat().st_size == src.stat().st_size:
            return False
        raise FileExistsError(f"Refusing to overwrite different existing file: {dst}")
    shutil.copy2(src, dst)
    return True


def package_cell(split_root: Path, scene_root: Path, cell: str) -> tuple[int, int]:
    src_scene = split_root / cell
    input_dir = src_scene / "input"
    sparse_src = src_scene / "sparse" / "0"
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Missing input directory: {input_dir}")
    if not sparse_src.is_dir():
        raise FileNotFoundError(f"Missing sparse/0 directory: {sparse_src}")

    scene_dir = scene_root / cell
    images_dir = scene_dir / "images"
    sparse_dst = scene_dir / "sparse" / "0"
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dst.mkdir(parents=True, exist_ok=True)

    image_count = 0
    copied_images = 0
    for src in sorted(input_dir.glob("*.png")):
        image_count += 1
        if copy_file(src, images_dir / src.name):
            copied_images += 1

    for filename in SPARSE_FILES:
        src = sparse_src / filename
        if not src.is_file():
            raise FileNotFoundError(f"Missing sparse file: {src}")
        copy_file(src, sparse_dst / filename)

    project_ini = sparse_src / "project.ini"
    if project_ini.is_file():
        copy_file(project_ini, sparse_dst / "project.ini")

    return image_count, copied_images


def create_archive(scene_root: Path, archive_root: Path, cell: str, prefix: str, compression_level: int) -> Path:
    archive_root.mkdir(parents=True, exist_ok=True)
    archive_path = archive_root / f"{prefix}_{cell}.tar.gz"
    if archive_path.exists():
        raise FileExistsError(f"Archive already exists: {archive_path}")

    command = [
        "tar",
        "-C",
        str(scene_root / cell),
        f"--use-compress-program=gzip -{compression_level}",
        "-cf",
        str(archive_path),
        "images",
        "sparse",
    ]
    subprocess.run(command, check=True)
    return archive_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--prefix", default="W2_4_3")
    parser.add_argument("--compression-level", type=int, default=1)
    parser.add_argument("cells", nargs="+")
    args = parser.parse_args()

    split_root = args.split_root.resolve()
    scene_root = args.scene_root.resolve()
    archive_root = args.archive_root.resolve()
    scene_root.mkdir(parents=True, exist_ok=True)
    archive_root.mkdir(parents=True, exist_ok=True)

    for cell in args.cells:
        image_count, copied_images = package_cell(split_root, scene_root, cell)
        archive_path = create_archive(
            scene_root,
            archive_root,
            cell,
            args.prefix,
            args.compression_level,
        )
        size_gib = archive_path.stat().st_size / (1024**3)
        print(
            f"{cell}: images={image_count}, copied_images={copied_images}, "
            f"archive={archive_path}, size={size_gib:.2f}G"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
