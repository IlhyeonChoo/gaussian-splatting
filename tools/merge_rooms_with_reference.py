#!/usr/bin/env python3
"""Merge room COLMAP models with a fixed reference model and build a final union."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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


@dataclass(frozen=True)
class RoomModel:
    room: str
    path: Path
    model_id: str
    image_count: int
    point_count: int
    image_names: frozenset[str]


@dataclass(frozen=True)
class SelectedRoom:
    room: str
    model: RoomModel
    overlap: int
    pair_scene: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-model", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--final-scene", default="all_rooms_passage")
    parser.add_argument("--rooms", nargs="+", required=True)
    parser.add_argument("--colmap", default="colmap")
    parser.add_argument("--min-common-images", type=int, default=3)
    parser.add_argument("--alignment-max-error", type=float, default=10.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def model_image_names(model_path: Path) -> frozenset[str]:
    _, images, _ = read_model(str(model_path), ".bin")
    return frozenset(image.name for image in images.values())


def read_model_counts(model_path: Path) -> tuple[int, int, frozenset[str]]:
    _, images, points = read_model(str(model_path), ".bin")
    return len(images), len(points), frozenset(image.name for image in images.values())


def load_room_models(split_root: Path, room: str) -> list[RoomModel]:
    sparse_root = split_root / room / "sparse"
    if not sparse_root.is_dir():
        raise FileNotFoundError(f"Missing sparse root: {sparse_root}")

    models: list[RoomModel] = []
    for model_path in sorted(sparse_root.iterdir(), key=lambda path: path.name):
        if not model_path.is_dir() or not (model_path / "images.bin").is_file():
            continue
        validate_model_path(model_path)
        image_count, point_count, names = read_model_counts(model_path)
        models.append(
            RoomModel(
                room=room,
                path=model_path,
                model_id=model_path.name,
                image_count=image_count,
                point_count=point_count,
                image_names=names,
            )
        )
    if not models:
        raise FileNotFoundError(f"No sparse models found for room: {room}")
    return models


def select_room_model(
    room: str,
    room_models: list[RoomModel],
    reference_names: frozenset[str],
    min_common_images: int,
) -> tuple[RoomModel | None, int]:
    candidates: list[tuple[int, int, int, RoomModel]] = []
    for model in room_models:
        overlap = len(model.image_names & reference_names)
        adds_unique_images = overlap < model.image_count
        if overlap >= min_common_images and adds_unique_images:
            expected_union = len(reference_names) + model.image_count - overlap
            candidates.append((expected_union, model.image_count, overlap, model))

    if not candidates:
        return None, 0

    expected_union, _, overlap, model = sorted(
        candidates,
        key=lambda item: (-item[0], -item[1], -item[2], item[3].model_id),
    )[0]
    _ = expected_union
    return model, overlap


def copy_model(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for filename in ("cameras.bin", "images.bin", "points3D.bin"):
        shutil.copy2(source / filename, destination / filename)


def align_and_union(
    colmap: str,
    reference_model: Path,
    room_model: RoomModel,
    output_model: Path,
    work_model: Path,
    min_common_images: int,
    alignment_max_error: float,
) -> tuple[int, int]:
    aligned_model = work_model / f"{room_model.room}_{room_model.model_id}_aligned"
    result = run_model_aligner(
        colmap,
        room_model.path,
        reference_model,
        aligned_model,
        min_common_images,
        alignment_max_error,
    )
    if result.returncode != 0:
        message = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"Failed to align {room_model.room}/{room_model.model_id}:\n{message}")

    return union_reference_and_aligned_model(reference_model, aligned_model, output_model)


def write_summary(output_root: Path, rows: list[dict[str, object]]) -> None:
    headers = [
        "room",
        "status",
        "selected_model",
        "overlap",
        "room_images",
        "merged_images",
        "merged_points3D",
        "output_model",
    ]
    lines = ["\t".join(headers)]
    for row in rows:
        lines.append("\t".join(str(row.get(header, "")) for header in headers))
    (output_root / "merge_with_reference_summary.tsv").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    reference_model = args.reference_model.resolve()
    split_root = args.split_root.resolve()
    output_root = args.output_root.resolve()
    work_root = args.work_root.resolve()
    validate_model_path(reference_model)

    if output_root.exists() and args.force:
        shutil.rmtree(output_root)
    if work_root.exists() and args.force:
        shutil.rmtree(work_root)
    output_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    reference_names = model_image_names(reference_model)
    rows: list[dict[str, object]] = []
    selected_rooms: list[SelectedRoom] = []

    for room in args.rooms:
        room_models = load_room_models(split_root, room)
        model, overlap = select_room_model(
            room,
            room_models,
            reference_names,
            args.min_common_images,
        )
        if model is None:
            rows.append({"room": room, "status": "skipped:no_connected_component"})
            print(f"{room}: skipped, no connected component", flush=True)
            continue

        pair_scene = f"{room}_passage_complete"
        output_model = output_root / pair_scene / "sparse" / "0"
        work_model = work_root / pair_scene
        merged_images, merged_points = align_and_union(
            args.colmap,
            reference_model,
            model,
            output_model,
            work_model,
            args.min_common_images,
            args.alignment_max_error,
        )
        rows.append(
            {
                "room": room,
                "status": "merged",
                "selected_model": model.path,
                "overlap": overlap,
                "room_images": model.image_count,
                "merged_images": merged_images,
                "merged_points3D": merged_points,
                "output_model": output_model,
            }
        )
        selected_rooms.append(SelectedRoom(room, model, overlap, pair_scene))
        print(
            f"{room}: merged model={model.model_id} overlap={overlap} "
            f"merged_images={merged_images}",
            flush=True,
        )

    current_model = work_root / "final_00_reference"
    copy_model(reference_model, current_model)

    for index, selected in enumerate(selected_rooms, start=1):
        next_model = work_root / f"final_{index:02d}_{selected.room}"
        work_model = work_root / f"final_{index:02d}_{selected.room}_work"
        merged_images, merged_points = align_and_union(
            args.colmap,
            current_model,
            selected.model,
            next_model,
            work_model,
            args.min_common_images,
            args.alignment_max_error,
        )
        print(
            f"final add {selected.room}: model={selected.model.model_id} "
            f"merged_images={merged_images} merged_points3D={merged_points}",
            flush=True,
        )
        current_model = next_model

    final_output = output_root / args.final_scene / "sparse" / "0"
    copy_model(current_model, final_output)
    validate_model_path(final_output)
    final_images, final_points, _ = read_model_counts(final_output)
    rows.append(
        {
            "room": "__final__",
            "status": "merged",
            "merged_images": final_images,
            "merged_points3D": final_points,
            "output_model": final_output,
        }
    )
    print(f"final output: {final_output} images={final_images} points3D={final_points}", flush=True)

    write_summary(output_root, rows)
    print(f"summary: {output_root / 'merge_with_reference_summary.tsv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
