#!/usr/bin/env python3
"""Package selected COLMAP bundle models as Colab-ready 3DGS scenes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.read_write_model import read_images_binary  # noqa: E402


@dataclass(frozen=True)
class BundleSpec:
    name: str
    model_dir: Path
    source_cells: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--merge-root", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument(
        "--compression-level",
        default="1",
        help="gzip compression level for tar archives. Use 1 for faster packaging.",
    )
    return parser.parse_args()


def bundle_specs(merge_root: Path) -> list[BundleSpec]:
    return [
        BundleSpec("pair_401_passage", merge_root / "pair_401_passage", ("passage", "401")),
        BundleSpec("pair_416_passage", merge_root / "pair_416_passage", ("passage", "416")),
        BundleSpec("pair_415_passage", merge_root / "pair_415_passage", ("passage", "415")),
        BundleSpec(
            "triple_401_passage_416",
            merge_root / "triple_401_passage_416",
            ("passage", "401", "416"),
        ),
        BundleSpec(
            "triple_416_passage_415",
            merge_root / "triple_416_passage_415",
            ("passage", "416", "415"),
        ),
        BundleSpec(
            "triple_404_passage_405",
            merge_root / "triple_404_passage_405",
            ("passage", "404", "405"),
        ),
    ]


def ensure_link_or_copy(src: Path, dst: Path) -> bool:
    """Create dst as a hardlink to src, falling back to a real copy.

    Returns True when a new file was created.
    """
    if dst.exists():
        if dst.stat().st_size == src.stat().st_size:
            return False
        raise FileExistsError(f"Refusing to overwrite existing file with different size: {dst}")

    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return True


def source_image_dirs(split_root: Path, cells: tuple[str, ...]) -> list[tuple[str, Path]]:
    return [(cell, split_root / cell / "images") for cell in cells]


def package_scene(split_root: Path, scene_root: Path, spec: BundleSpec) -> tuple[int, int, int]:
    scene_dir = scene_root / spec.name
    sparse_dir = scene_dir / "sparse" / "0"
    images_dir = scene_dir / "images"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    for filename in ("rigs.bin", "cameras.bin", "frames.bin", "images.bin", "points3D.bin"):
        src = spec.model_dir / filename
        if src.exists():
            ensure_link_or_copy(src.resolve(), sparse_dir / filename)

    images = read_images_binary(str(spec.model_dir / "images.bin"))
    image_dirs = source_image_dirs(split_root, spec.source_cells)
    records = []
    missing = []
    created = 0
    kept = 0

    for image in sorted(images.values(), key=lambda item: item.name):
        source = None
        source_cell = None
        for cell, image_dir in image_dirs:
            candidate = image_dir / image.name
            if candidate.exists():
                source = candidate.resolve()
                source_cell = cell
                break

        if source is None:
            missing.append(image.name)
            continue

        if ensure_link_or_copy(source, images_dir / image.name):
            created += 1
        else:
            kept += 1
        records.append((image.name, source_cell, str(source)))

    (scene_dir / "image_sources.tsv").write_text(
        "image_name\tsource_cell\tsource_path\n"
        + "".join(f"{name}\t{cell}\t{path}\n" for name, cell, path in records)
    )

    if missing:
        (scene_dir / "missing_images.txt").write_text("\n".join(missing) + "\n")
        raise FileNotFoundError(f"{spec.name}: missing {len(missing)} images")

    return len(images), created, kept


def create_archive(scene_root: Path, archive_root: Path, name: str, compression_level: str) -> Path:
    archive_root.mkdir(parents=True, exist_ok=True)
    archive_path = archive_root / f"{name}.tar.gz"
    if archive_path.exists():
        raise FileExistsError(f"Archive already exists: {archive_path}")

    command = [
        "tar",
        "-C",
        str(scene_root / name),
        f"--use-compress-program=gzip -{compression_level}",
        "-cf",
        str(archive_path),
        "images",
        "sparse",
    ]
    subprocess.run(command, check=True)
    return archive_path


def main() -> int:
    args = parse_args()
    split_root = args.split_root.resolve()
    merge_root = args.merge_root.resolve()
    scene_root = args.scene_root.resolve()
    archive_root = args.archive_root.resolve()
    scene_root.mkdir(parents=True, exist_ok=True)
    archive_root.mkdir(parents=True, exist_ok=True)

    print(f"Scene root: {scene_root}")
    print(f"Archive root: {archive_root}")

    for spec in bundle_specs(merge_root):
        num_images, created, kept = package_scene(split_root, scene_root, spec)
        archive_path = create_archive(scene_root, archive_root, spec.name, args.compression_level)
        size_gb = archive_path.stat().st_size / (1024**3)
        print(
            f"{spec.name}: images={num_images}, linked_or_copied={created}, "
            f"kept={kept}, archive={archive_path.name}, size={size_gb:.2f}G"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
