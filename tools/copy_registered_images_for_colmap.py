#!/usr/bin/env python3
"""Copy registered COLMAP images into scene directories."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import struct


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--input-dir-name", default="input")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("scenes", nargs="*")
    return parser.parse_args()


def read_registered_image_names(images_bin: Path) -> list[str]:
    names: list[str] = []
    with images_bin.open("rb") as file:
        num_images = struct.unpack("<Q", file.read(8))[0]
        for _ in range(num_images):
            file.seek(64, 1)
            name_bytes = bytearray()
            while True:
                current_byte = file.read(1)
                if current_byte == b"\x00":
                    break
                if current_byte == b"":
                    raise ValueError(f"Unexpected EOF while reading {images_bin}")
                name_bytes.extend(current_byte)
            names.append(name_bytes.decode("utf-8"))
            num_points_2d = struct.unpack("<Q", file.read(8))[0]
            file.seek(num_points_2d * 24, 1)
    return names


def discover_scenes(scene_root: Path) -> list[str]:
    scenes: list[str] = []
    for scene_dir in sorted(scene_root.iterdir()):
        if (scene_dir / "sparse" / "0" / "images.bin").is_file():
            scenes.append(scene_dir.name)
    return scenes


def build_source_lookup(source_root: Path, input_dir_name: str) -> dict[str, Path]:
    lookup: dict[str, Path] = {}
    for input_dir in sorted(source_root.glob(f"*/{input_dir_name}")):
        if not input_dir.is_dir():
            continue
        for image_path in sorted(input_dir.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            relative_name = image_path.relative_to(input_dir).as_posix()
            lookup.setdefault(relative_name, image_path)
            lookup.setdefault(image_path.name, image_path)
    return lookup


def copy_scene_images(
    scene_dir: Path,
    source_lookup: dict[str, Path],
    force: bool,
) -> tuple[int, int, int, list[str]]:
    names = read_registered_image_names(scene_dir / "sparse" / "0" / "images.bin")
    images_dir = scene_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    existing = 0
    missing: list[str] = []
    for name in names:
        source = source_lookup.get(name) or source_lookup.get(Path(name).name)
        if source is None:
            missing.append(name)
            continue

        destination = images_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not force:
            existing += 1
            continue
        shutil.copy2(source, destination)
        copied += 1

    return len(names), copied, existing, missing


def main() -> int:
    args = parse_args()
    scene_root = args.scene_root.resolve()
    source_root = args.source_root.resolve()
    scenes = args.scenes or discover_scenes(scene_root)
    source_lookup = build_source_lookup(source_root, args.input_dir_name)

    if not scenes:
        raise FileNotFoundError(f"No scenes found under {scene_root}")
    if not source_lookup:
        raise FileNotFoundError(f"No source images found under {source_root}")

    failed = False
    for scene in scenes:
        scene_dir = scene_root / scene
        expected, copied, existing, missing = copy_scene_images(
            scene_dir,
            source_lookup,
            args.force,
        )
        print(
            f"{scene}: expected={expected} copied={copied} "
            f"existing={existing} missing={len(missing)}",
            flush=True,
        )
        if missing:
            failed = True
            missing_path = scene_dir / "missing_images.txt"
            missing_path.write_text("\n".join(missing) + "\n", encoding="utf-8")
            print(f"{scene}: missing list written to {missing_path}", flush=True)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
