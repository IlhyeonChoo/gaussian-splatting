"""Prepare image folders for COLMAP training scenes."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .data_browser import collect_image_files


def evenly_sample(items: list[Path], max_items: int) -> list[Path]:
    """Return at most max_items evenly sampled items."""
    if max_items <= 0 or len(items) <= max_items:
        return items
    return [items[index * len(items) // max_items] for index in range(max_items)]


def prepare_images(source: Path, scene_path: Path, max_images: int, input_dir_name: str = "input") -> Path:
    """Flatten source images into scene_path/input_dir_name."""
    images = evenly_sample(collect_image_files(source), max_images)
    if not images:
        raise ValueError(f"No supported images found under: {source}")

    input_dir = scene_path / input_dir_name
    if input_dir.exists():
        shutil.rmtree(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)

    digits = max(6, len(str(len(images))))
    for index, image_path in enumerate(images):
        destination = input_dir / f"image_{index:0{digits}d}{image_path.suffix.lower()}"
        shutil.copy2(image_path, destination)

    return input_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a 3DGS scene input folder.")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--scene_path", required=True, type=Path)
    parser.add_argument("--max_images", default=0, type=int)
    parser.add_argument("--input_dir_name", default="input")
    args = parser.parse_args()

    if args.max_images < 0:
        parser.error("--max_images must be >= 0")
    if "/" in args.input_dir_name or args.input_dir_name in {"", ".", ".."}:
        parser.error("--input_dir_name must be a simple directory name")
    return args


def main() -> None:
    args = parse_args()
    input_dir = prepare_images(args.source, args.scene_path, args.max_images, args.input_dir_name)
    print(f"[DONE] Prepared {input_dir}")


if __name__ == "__main__":
    main()

