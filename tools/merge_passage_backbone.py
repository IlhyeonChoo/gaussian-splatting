#!/usr/bin/env python3
"""Build a connected passage backbone from selected COLMAP passage models."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.merge_rooms_with_best_passage import (  # noqa: E402
    run_model_aligner,
    union_reference_and_aligned_model,
    validate_model_path,
)
from utils.read_write_model import read_model  # noqa: E402


def parse_component(component: str) -> tuple[str, str]:
    if "/" not in component:
        raise ValueError(f"Component must be formatted as <label>/<model_id>: {component}")
    label, model_id = component.split("/", 1)
    return label, model_id


def model_path(split_root: Path, component: str) -> Path:
    label, model_id = parse_component(component)
    path = split_root / label / "sparse" / model_id
    validate_model_path(path)
    return path


def count_images(model: Path) -> int:
    _, images, _ = read_model(str(model), ".bin")
    return len(images)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--scene-name", default="passage_all")
    parser.add_argument("--base", default="passage2/2")
    parser.add_argument("--components", nargs="+", default=["passage1/2", "passage3/0"])
    parser.add_argument("--colmap", default="colmap")
    parser.add_argument("--min-common-images", type=int, default=3)
    parser.add_argument("--alignment-max-error", type=float, default=10.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    split_root = args.split_root.resolve()
    output_root = args.output_root.resolve()
    work_root = args.work_root.resolve()
    scene_dir = output_root / args.scene_name
    sparse_zero = scene_dir / "sparse" / "0"

    if output_root.exists() and args.force:
        shutil.rmtree(output_root)
    if work_root.exists() and args.force:
        shutil.rmtree(work_root)
    if output_root.exists() and any(output_root.iterdir()) and not args.force:
        raise FileExistsError(f"Output root is not empty: {output_root}")
    if work_root.exists() and any(work_root.iterdir()) and not args.force:
        raise FileExistsError(f"Work root is not empty: {work_root}")

    work_root.mkdir(parents=True, exist_ok=True)
    sparse_zero.mkdir(parents=True, exist_ok=True)

    current_model = model_path(split_root, args.base)
    print(f"base: {args.base} images={count_images(current_model)}", flush=True)

    for index, component in enumerate(args.components, start=1):
        component_model = model_path(split_root, component)
        aligned_model = work_root / f"{index:02d}_{component.replace('/', '_')}_aligned"
        next_model = work_root / f"{index:02d}_{component.replace('/', '_')}_union"

        result = run_model_aligner(
            args.colmap,
            component_model,
            current_model,
            aligned_model,
            args.min_common_images,
            args.alignment_max_error,
        )
        if result.returncode != 0:
            message = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"Failed to align {component}:\n{message}")

        merged_images, merged_points = union_reference_and_aligned_model(
            current_model,
            aligned_model,
            next_model,
        )
        print(
            f"added: {component} source_images={count_images(component_model)} "
            f"merged_images={merged_images} merged_points3D={merged_points}",
            flush=True,
        )
        current_model = next_model

    for filename in ("cameras.bin", "images.bin", "points3D.bin"):
        shutil.copy2(current_model / filename, sparse_zero / filename)

    validate_model_path(sparse_zero)
    print(f"output: {sparse_zero}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
