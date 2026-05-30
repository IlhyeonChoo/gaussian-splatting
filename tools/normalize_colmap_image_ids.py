#!/usr/bin/env python3
"""Normalize COLMAP image IDs across split sparse models.

COLMAP model_merger expects shared registered images to have matching image IDs
in both input reconstructions. This tool preserves the reference model's image
IDs and rewrites the other models to use the same global image-name mapping.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.read_write_model import (  # noqa: E402
    Image,
    Point3D,
    read_model,
    write_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite split COLMAP sparse models with consistent image IDs."
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        required=True,
        help="Directory containing <cell>/sparse/0 COLMAP models.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory for normalized <cell>/sparse/0 COLMAP models.",
    )
    parser.add_argument(
        "--reference-cell",
        default="passage",
        help="Cell whose existing image IDs are kept as the global reference.",
    )
    parser.add_argument(
        "--cells",
        nargs="*",
        default=None,
        help="Optional cell list. Defaults to every cell with sparse/0 under models-root.",
    )
    parser.add_argument(
        "--copy-input-links",
        action="store_true",
        help="Also copy/symlink each cell's input directory into output-root.",
    )
    return parser.parse_args()


def model_path(root: Path, cell: str) -> Path:
    return root / cell / "sparse" / "0"


def has_binary_model(path: Path) -> bool:
    return all((path / name).is_file() for name in ("cameras.bin", "images.bin", "points3D.bin"))


def discover_cells(root: Path) -> list[str]:
    cells = []
    for entry in sorted(root.iterdir()):
        if entry.is_dir() and has_binary_model(model_path(root, entry.name)):
            cells.append(entry.name)
    return cells


def sorted_image_items(images: dict[int, Image]) -> list[tuple[int, Image]]:
    return sorted(images.items(), key=lambda item: (item[1].name, item[0]))


def build_global_image_ids(
    models_root: Path,
    cells: list[str],
    reference_cell: str,
) -> dict[str, int]:
    reference_model = model_path(models_root, reference_cell)
    _, reference_images, _ = read_model(str(reference_model), ".bin")

    name_to_id: dict[str, int] = {}
    used_ids: set[int] = set()
    for image_id, image in sorted_image_items(reference_images):
        if image.name in name_to_id:
            raise ValueError(f"Duplicate image name in reference model: {image.name}")
        name_to_id[image.name] = image_id
        used_ids.add(image_id)

    next_id = max(used_ids, default=0) + 1
    pending_names: set[str] = set()
    for cell in cells:
        _, images, _ = read_model(str(model_path(models_root, cell)), ".bin")
        for _, image in sorted_image_items(images):
            if image.name not in name_to_id:
                pending_names.add(image.name)

    for image_name in sorted(pending_names):
        while next_id in used_ids:
            next_id += 1
        name_to_id[image_name] = next_id
        used_ids.add(next_id)
        next_id += 1

    return name_to_id


def remap_images(images: dict[int, Image], name_to_id: dict[str, int]) -> tuple[dict[int, Image], dict[int, int]]:
    old_to_new: dict[int, int] = {}
    remapped: dict[int, Image] = {}

    for old_id, image in sorted_image_items(images):
        new_id = name_to_id[image.name]
        if new_id in remapped:
            raise ValueError(f"Image ID collision after remap: {new_id} ({image.name})")
        old_to_new[old_id] = new_id
        remapped[new_id] = Image(
            id=new_id,
            qvec=image.qvec,
            tvec=image.tvec,
            camera_id=image.camera_id,
            name=image.name,
            xys=image.xys,
            point3D_ids=image.point3D_ids,
        )

    return remapped, old_to_new


def remap_points3d(points3d: dict[int, Point3D], old_to_new: dict[int, int]) -> dict[int, Point3D]:
    remapped: dict[int, Point3D] = {}

    for point_id, point in points3d.items():
        image_ids = []
        for image_id in point.image_ids:
            old_id = int(image_id)
            if old_id not in old_to_new:
                raise ValueError(f"Point3D {point_id} references unknown image_id {old_id}")
            image_ids.append(old_to_new[old_id])

        remapped[point_id] = Point3D(
            id=point.id,
            xyz=point.xyz,
            rgb=point.rgb,
            error=point.error,
            image_ids=np.array(image_ids, dtype=point.image_ids.dtype),
            point2D_idxs=point.point2D_idxs,
        )

    return remapped


def copy_input_dir(models_root: Path, output_root: Path, cell: str) -> None:
    src = models_root / cell / "input"
    dst = output_root / cell / "input"
    if not src.is_dir():
        return
    if dst.exists():
        return
    os.symlink(os.path.relpath(src.resolve(), start=dst.parent.resolve()), dst)


def write_mapping(output_root: Path, reference_cell: str, name_to_id: dict[str, int]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    mapping_path = output_root / "image_id_mapping.json"
    payload = {
        "reference_cell": reference_cell,
        "num_images": len(name_to_id),
        "image_name_to_id": dict(sorted(name_to_id.items())),
    }
    mapping_path.write_text(json.dumps(payload, indent=2) + "\n")


def normalize_cell(
    models_root: Path,
    output_root: Path,
    cell: str,
    name_to_id: dict[str, int],
    copy_input_links: bool,
) -> tuple[int, int]:
    source_model = model_path(models_root, cell)
    output_model = model_path(output_root, cell)
    output_model.mkdir(parents=True, exist_ok=True)

    cameras, images, points3d = read_model(str(source_model), ".bin")
    remapped_images, old_to_new = remap_images(images, name_to_id)
    remapped_points3d = remap_points3d(points3d, old_to_new)

    write_model(cameras, remapped_images, remapped_points3d, str(output_model), ".bin")
    if copy_input_links:
        copy_input_dir(models_root, output_root, cell)

    changed = sum(1 for old_id, new_id in old_to_new.items() if old_id != new_id)
    return len(remapped_images), changed


def main() -> int:
    args = parse_args()
    models_root = args.models_root.resolve()
    output_root = args.output_root.resolve()

    cells = args.cells if args.cells is not None else discover_cells(models_root)
    if args.reference_cell not in cells:
        cells = [args.reference_cell, *cells]
    cells = list(dict.fromkeys(cells))

    missing = [cell for cell in cells if not has_binary_model(model_path(models_root, cell))]
    if missing:
        raise FileNotFoundError(f"Missing sparse/0 binary model for cells: {', '.join(missing)}")

    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output root is not empty: {output_root}")

    name_to_id = build_global_image_ids(models_root, cells, args.reference_cell)
    write_mapping(output_root, args.reference_cell, name_to_id)

    print(f"Reference cell: {args.reference_cell}")
    print(f"Global registered images: {len(name_to_id)}")
    for cell in cells:
        image_count, changed_count = normalize_cell(
            models_root,
            output_root,
            cell,
            name_to_id,
            args.copy_input_links,
        )
        print(f"{cell}: images={image_count}, changed_image_ids={changed_count}")

    print(f"Output: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
