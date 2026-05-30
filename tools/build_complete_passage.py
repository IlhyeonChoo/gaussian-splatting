#!/usr/bin/env python3
"""Extend the passage backbone with indirectly connected passage components."""

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


def count_images(model_path: Path) -> int:
    _, images, _ = read_model(str(model_path), ".bin")
    return len(images)


def copy_model(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for filename in ("cameras.bin", "images.bin", "points3D.bin"):
        shutil.copy2(source / filename, destination / filename)


def align_model(
    colmap: str,
    input_model: Path,
    reference_model: Path,
    output_model: Path,
    min_common_images: int,
    alignment_max_error: float,
) -> None:
    result = run_model_aligner(
        colmap,
        input_model,
        reference_model,
        output_model,
        min_common_images,
        alignment_max_error,
    )
    if result.returncode != 0:
        message = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"Failed to align {input_model}:\n{message}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--bridge-model", type=Path, required=True)
    parser.add_argument("--left-passage-model", type=Path, required=True)
    parser.add_argument("--extra-passage-models", nargs="*", type=Path, default=[])
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--colmap", default="colmap")
    parser.add_argument("--min-common-images", type=int, default=3)
    parser.add_argument("--alignment-max-error", type=float, default=10.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    base_model = args.base_model.resolve()
    bridge_model = args.bridge_model.resolve()
    left_passage_model = args.left_passage_model.resolve()
    output_model = args.output_model.resolve()
    work_root = args.work_root.resolve()

    for model_path in [base_model, bridge_model, left_passage_model, *args.extra_passage_models]:
        validate_model_path(model_path.resolve())

    if work_root.exists() and args.force:
        shutil.rmtree(work_root)
    if output_model.exists() and args.force:
        shutil.rmtree(output_model)
    work_root.mkdir(parents=True, exist_ok=True)

    aligned_bridge = work_root / "01_bridge_aligned_to_base"
    aligned_left = work_root / "02_left_passage_aligned_to_base"
    first_union = work_root / "03_base_left_passage_union"

    align_model(
        args.colmap,
        bridge_model,
        base_model,
        aligned_bridge,
        args.min_common_images,
        args.alignment_max_error,
    )
    print(
        f"aligned bridge: images={count_images(bridge_model)} "
        f"reference_images={count_images(base_model)}",
        flush=True,
    )

    align_model(
        args.colmap,
        left_passage_model,
        aligned_bridge,
        aligned_left,
        args.min_common_images,
        args.alignment_max_error,
    )
    merged_images, merged_points = union_reference_and_aligned_model(
        base_model,
        aligned_left,
        first_union,
    )
    print(
        f"added left passage: source_images={count_images(left_passage_model)} "
        f"merged_images={merged_images} merged_points3D={merged_points}",
        flush=True,
    )

    current_model = first_union
    for index, extra_model in enumerate(args.extra_passage_models, start=4):
        extra_model = extra_model.resolve()
        aligned_extra = work_root / f"{index:02d}_{extra_model.parent.parent.parent.name}_{extra_model.name}_aligned"
        next_union = work_root / f"{index:02d}_{extra_model.parent.parent.parent.name}_{extra_model.name}_union"
        align_model(
            args.colmap,
            extra_model,
            current_model,
            aligned_extra,
            args.min_common_images,
            args.alignment_max_error,
        )
        merged_images, merged_points = union_reference_and_aligned_model(
            current_model,
            aligned_extra,
            next_union,
        )
        print(
            f"added extra passage: {extra_model} source_images={count_images(extra_model)} "
            f"merged_images={merged_images} merged_points3D={merged_points}",
            flush=True,
        )
        current_model = next_union

    copy_model(current_model, output_model)
    validate_model_path(output_model)
    print(f"output: {output_model}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
