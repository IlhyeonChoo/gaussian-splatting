#!/usr/bin/env python3
"""Merge each room COLMAP reconstruction with its best overlapping passage."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.read_write_model import (  # noqa: E402
    Camera,
    Image,
    Point3D,
    read_model,
    write_model,
)


REQUIRED_MODEL_FILES = ("cameras.bin", "images.bin", "points3D.bin")


@dataclass(frozen=True)
class SparseModel:
    label: str
    path: Path
    model_id: str
    image_count: int
    image_names: frozenset[str]


@dataclass(frozen=True)
class MergeCandidate:
    room: SparseModel
    passage: SparseModel
    overlap: int

    @property
    def output_name(self) -> str:
        return f"{self.room.label}_{self.passage.label}"

    @property
    def expected_union_count(self) -> int:
        return self.room.image_count + self.passage.image_count - self.overlap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--colmap", default="colmap")
    parser.add_argument("--min-overlap", type=int, default=3)
    parser.add_argument("--max-reproj-error", type=float, default=64.0)
    parser.add_argument("--alignment-max-error", type=float, default=10.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--passages", nargs="*", default=None)
    parser.add_argument("--rooms", nargs="*", default=None)
    return parser.parse_args()


def validate_model_path(path: Path) -> None:
    for filename in REQUIRED_MODEL_FILES:
        model_file = path / filename
        if not model_file.is_file():
            raise FileNotFoundError(f"Missing COLMAP model file: {model_file}")


def discover_labels(split_root: Path) -> tuple[list[str], list[str]]:
    labels = sorted(path.name for path in split_root.iterdir() if path.is_dir())
    passages = [label for label in labels if label.startswith("passage")]
    rooms = [label for label in labels if label not in passages]
    return passages, rooms


def load_sparse_models(split_root: Path, label: str) -> list[SparseModel]:
    sparse_root = split_root / label / "sparse"
    if not sparse_root.is_dir():
        raise FileNotFoundError(f"Missing sparse directory: {sparse_root}")

    models: list[SparseModel] = []
    for path in sorted(sparse_root.iterdir(), key=lambda item: item.name):
        if not path.is_dir() or not (path / "images.bin").is_file():
            continue
        validate_model_path(path)
        _, images, _ = read_model(str(path), ".bin")
        models.append(
            SparseModel(
                label=label,
                path=path,
                model_id=path.name,
                image_count=len(images),
                image_names=frozenset(image.name for image in images.values()),
            )
        )

    if not models:
        raise FileNotFoundError(f"No COLMAP sparse models found for {label}: {sparse_root}")
    return models


def build_candidates(
    room_models: list[SparseModel],
    passage_models: dict[str, list[SparseModel]],
    min_overlap: int,
) -> list[MergeCandidate]:
    all_candidates: list[MergeCandidate] = []
    for room_model in room_models:
        for passage_label, models in passage_models.items():
            for passage_model in models:
                overlap = len(room_model.image_names & passage_model.image_names)
                adds_unique_images_from_both_models = (
                    overlap < room_model.image_count and overlap < passage_model.image_count
                )
                if overlap >= min_overlap and adds_unique_images_from_both_models:
                    all_candidates.append(MergeCandidate(room_model, passage_model, overlap))

    if not all_candidates:
        return []

    best_overlap_by_passage = {
        passage_label: max(
            (candidate.overlap for candidate in all_candidates if candidate.passage.label == passage_label),
            default=0,
        )
        for passage_label in passage_models
    }
    best_overlap = max(best_overlap_by_passage.values())
    preferred_passages = {
        passage_label
        for passage_label, overlap in best_overlap_by_passage.items()
        if overlap == best_overlap
    }

    preferred = [
        candidate for candidate in all_candidates if candidate.passage.label in preferred_passages
    ]
    fallback = [
        candidate for candidate in all_candidates if candidate.passage.label not in preferred_passages
    ]

    def sort_key(candidate: MergeCandidate) -> tuple[int, int, int, int, str, str, str]:
        return (
            -candidate.expected_union_count,
            -candidate.room.image_count,
            -candidate.passage.image_count,
            -candidate.overlap,
            candidate.passage.label,
            candidate.room.model_id,
            candidate.passage.model_id,
        )

    preferred.sort(key=sort_key)
    fallback.sort(
        key=lambda candidate: (
            -candidate.overlap,
            -candidate.expected_union_count,
            -candidate.room.image_count,
            -candidate.passage.image_count,
            candidate.passage.label,
            candidate.room.model_id,
            candidate.passage.model_id,
        )
    )
    return preferred + fallback


def build_pair_image_ids(reference_images: dict[int, Image], other_images: dict[int, Image]) -> dict[str, int]:
    name_to_id: dict[str, int] = {}
    used_ids: set[int] = set()

    for image_id, image in sorted(reference_images.items(), key=lambda item: (item[1].name, item[0])):
        if image.name in name_to_id:
            raise ValueError(f"Duplicate image name in reference model: {image.name}")
        name_to_id[image.name] = image_id
        used_ids.add(image_id)

    next_id = max(used_ids, default=0) + 1
    for _, image in sorted(other_images.items(), key=lambda item: (item[1].name, item[0])):
        if image.name in name_to_id:
            continue
        while next_id in used_ids:
            next_id += 1
        name_to_id[image.name] = next_id
        used_ids.add(next_id)
        next_id += 1

    return name_to_id


def remap_cameras(
    cameras: dict[int, Camera],
    reserved_ids: set[int],
) -> tuple[dict[int, Camera], dict[int, int]]:
    old_to_new: dict[int, int] = {}
    remapped: dict[int, Camera] = {}
    next_id = max(reserved_ids, default=0) + 1

    for camera_id, camera in sorted(cameras.items()):
        if camera_id in reserved_ids:
            while next_id in reserved_ids or next_id in remapped:
                next_id += 1
            new_id = next_id
            next_id += 1
        else:
            new_id = camera_id
        old_to_new[camera_id] = new_id
        remapped[new_id] = Camera(
            id=new_id,
            model=camera.model,
            width=camera.width,
            height=camera.height,
            params=camera.params,
        )
    return remapped, old_to_new


def remap_images(
    images: dict[int, Image],
    name_to_id: dict[str, int],
    camera_old_to_new: dict[int, int] | None = None,
) -> tuple[dict[int, Image], dict[int, int]]:
    old_to_new: dict[int, int] = {}
    remapped: dict[int, Image] = {}

    for old_id, image in sorted(images.items(), key=lambda item: (item[1].name, item[0])):
        new_id = name_to_id[image.name]
        if new_id in remapped:
            raise ValueError(f"Image ID collision after remap: {new_id} ({image.name})")
        old_to_new[old_id] = new_id
        camera_id = image.camera_id
        if camera_old_to_new is not None:
            camera_id = camera_old_to_new[camera_id]
        remapped[new_id] = Image(
            id=new_id,
            qvec=image.qvec,
            tvec=image.tvec,
            camera_id=camera_id,
            name=image.name,
            xys=image.xys,
            point3D_ids=image.point3D_ids,
        )

    return remapped, old_to_new


def remap_points3d(points3d: dict[int, Point3D], image_old_to_new: dict[int, int]) -> dict[int, Point3D]:
    remapped: dict[int, Point3D] = {}

    for point_id, point in points3d.items():
        image_ids = []
        for image_id in point.image_ids:
            old_id = int(image_id)
            if old_id not in image_old_to_new:
                raise ValueError(f"Point3D {point_id} references unknown image_id {old_id}")
            image_ids.append(image_old_to_new[old_id])

        remapped[point_id] = Point3D(
            id=point.id,
            xyz=point.xyz,
            rgb=point.rgb,
            error=point.error,
            image_ids=np.array(image_ids, dtype=point.image_ids.dtype),
            point2D_idxs=point.point2D_idxs,
        )

    return remapped


def next_available_id(used_ids: set[int], start: int | None = None) -> int:
    next_id = max(used_ids, default=0) + 1 if start is None else start
    while next_id in used_ids:
        next_id += 1
    return next_id


def union_reference_and_aligned_model(
    reference_model: Path,
    aligned_model: Path,
    output_model: Path,
) -> tuple[int, int]:
    ref_cameras, ref_images, ref_points3d = read_model(str(reference_model), ".bin")
    other_cameras, other_images, other_points3d = read_model(str(aligned_model), ".bin")

    output_cameras = dict(ref_cameras)
    output_images = dict(ref_images)
    output_points3d = dict(ref_points3d)

    ref_image_names = {image.name for image in ref_images.values()}
    used_camera_ids = set(output_cameras)
    used_image_ids = set(output_images)
    used_point_ids = set(output_points3d)

    other_cameras_remapped, camera_old_to_new = remap_cameras(other_cameras, used_camera_ids)
    output_cameras.update(other_cameras_remapped)

    old_image_to_new: dict[int, int] = {}
    added_old_image_ids: set[int] = set()
    next_image_id = next_available_id(used_image_ids)
    for old_image_id, image in sorted(other_images.items(), key=lambda item: (item[1].name, item[0])):
        if image.name in ref_image_names:
            matching_ref_id = next(
                ref_id for ref_id, ref_image in ref_images.items() if ref_image.name == image.name
            )
            old_image_to_new[old_image_id] = matching_ref_id
            continue

        new_image_id = next_available_id(used_image_ids, next_image_id)
        next_image_id = new_image_id + 1
        used_image_ids.add(new_image_id)
        old_image_to_new[old_image_id] = new_image_id
        added_old_image_ids.add(old_image_id)
        output_images[new_image_id] = Image(
            id=new_image_id,
            qvec=image.qvec,
            tvec=image.tvec,
            camera_id=camera_old_to_new[image.camera_id],
            name=image.name,
            xys=image.xys,
            point3D_ids=np.full_like(image.point3D_ids, -1),
        )

    old_point_to_new: dict[int, int] = {}
    filtered_tracks: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    next_point_id = next_available_id(used_point_ids)
    for old_point_id, point in sorted(other_points3d.items()):
        image_ids: list[int] = []
        point2d_idxs: list[int] = []
        for old_image_id_raw, point2d_idx_raw in zip(point.image_ids, point.point2D_idxs):
            old_image_id = int(old_image_id_raw)
            if old_image_id not in added_old_image_ids:
                continue
            image_ids.append(old_image_to_new[old_image_id])
            point2d_idxs.append(int(point2d_idx_raw))

        if len(image_ids) < 2:
            continue

        new_point_id = next_available_id(used_point_ids, next_point_id)
        next_point_id = new_point_id + 1
        used_point_ids.add(new_point_id)
        old_point_to_new[old_point_id] = new_point_id
        filtered_tracks[old_point_id] = (
            np.array(image_ids, dtype=point.image_ids.dtype),
            np.array(point2d_idxs, dtype=point.point2D_idxs.dtype),
        )

    for old_point_id, new_point_id in old_point_to_new.items():
        point = other_points3d[old_point_id]
        image_ids, point2d_idxs = filtered_tracks[old_point_id]
        output_points3d[new_point_id] = Point3D(
            id=new_point_id,
            xyz=point.xyz,
            rgb=point.rgb,
            error=point.error,
            image_ids=image_ids,
            point2D_idxs=point2d_idxs,
        )

    for old_image_id in added_old_image_ids:
        old_image = other_images[old_image_id]
        new_image_id = old_image_to_new[old_image_id]
        point3d_ids = np.full_like(old_image.point3D_ids, -1)
        for point2d_idx, old_point_id in enumerate(old_image.point3D_ids):
            old_point_id_int = int(old_point_id)
            if old_point_id_int in old_point_to_new:
                point3d_ids[point2d_idx] = old_point_to_new[old_point_id_int]

        current_image = output_images[new_image_id]
        output_images[new_image_id] = Image(
            id=current_image.id,
            qvec=current_image.qvec,
            tvec=current_image.tvec,
            camera_id=current_image.camera_id,
            name=current_image.name,
            xys=current_image.xys,
            point3D_ids=point3d_ids,
        )

    output_model.mkdir(parents=True, exist_ok=True)
    write_model(output_cameras, output_images, output_points3d, str(output_model), ".bin")
    return len(output_images), len(output_points3d)


def write_normalized_pair(candidate: MergeCandidate, pair_work_root: Path) -> tuple[Path, Path]:
    reference_path = candidate.passage.path
    other_path = candidate.room.path

    ref_cameras, ref_images, ref_points3d = read_model(str(reference_path), ".bin")
    other_cameras, other_images, other_points3d = read_model(str(other_path), ".bin")
    name_to_id = build_pair_image_ids(ref_images, other_images)

    ref_output = pair_work_root / candidate.passage.label / "sparse" / "0"
    other_output = pair_work_root / candidate.room.label / "sparse" / "0"
    ref_output.mkdir(parents=True, exist_ok=True)
    other_output.mkdir(parents=True, exist_ok=True)

    ref_images_remapped, ref_image_old_to_new = remap_images(ref_images, name_to_id)
    ref_points_remapped = remap_points3d(ref_points3d, ref_image_old_to_new)
    write_model(ref_cameras, ref_images_remapped, ref_points_remapped, str(ref_output), ".bin")

    other_cameras_remapped, camera_old_to_new = remap_cameras(other_cameras, set(ref_cameras))
    other_images_remapped, other_image_old_to_new = remap_images(
        other_images,
        name_to_id,
        camera_old_to_new,
    )
    other_points_remapped = remap_points3d(other_points3d, other_image_old_to_new)
    write_model(other_cameras_remapped, other_images_remapped, other_points_remapped, str(other_output), ".bin")

    return ref_output, other_output


def run_model_merger(
    colmap: str,
    reference_model: Path,
    other_model: Path,
    output_model: Path,
    max_reproj_error: float,
) -> subprocess.CompletedProcess[str]:
    output_model.mkdir(parents=True, exist_ok=True)
    command = [
        colmap,
        "model_merger",
        "--input_path1",
        str(reference_model),
        "--input_path2",
        str(other_model),
        "--output_path",
        str(output_model),
        "--max_reproj_error",
        str(max_reproj_error),
    ]
    return subprocess.run(command, check=False, text=True, capture_output=True)


def run_model_aligner(
    colmap: str,
    input_model: Path,
    reference_model: Path,
    output_model: Path,
    min_common_images: int,
    alignment_max_error: float,
) -> subprocess.CompletedProcess[str]:
    if output_model.exists():
        shutil.rmtree(output_model)
    output_model.mkdir(parents=True, exist_ok=True)
    command = [
        colmap,
        "model_aligner",
        "--input_path",
        str(input_model),
        "--ref_model_path",
        str(reference_model),
        "--output_path",
        str(output_model),
        "--ref_is_gps",
        "0",
        "--alignment_type",
        "custom",
        "--min_common_images",
        str(min_common_images),
        "--alignment_max_error",
        str(alignment_max_error),
    ]
    return subprocess.run(command, check=False, text=True, capture_output=True)


def prepare_empty_dir(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"Directory already exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_summary(output_root: Path, rows: list[dict[str, object]]) -> None:
    headers = [
        "room",
        "passage",
        "status",
        "overlap",
        "room_model",
        "room_registered",
        "passage_model",
        "passage_registered",
        "expected_union",
        "merged_registered",
        "merged_points3D",
        "output_model",
    ]
    lines = ["\t".join(headers)]
    for row in rows:
        lines.append("\t".join(str(row.get(header, "")) for header in headers))
    (output_root / "merge_summary.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    split_root = args.split_root.resolve()
    work_root = args.work_root.resolve()
    output_root = args.output_root.resolve()

    discovered_passages, discovered_rooms = discover_labels(split_root)
    passages = args.passages or discovered_passages
    rooms = args.rooms or discovered_rooms

    prepare_empty_dir(work_root, args.force)
    prepare_empty_dir(output_root, args.force)

    passage_models = {label: load_sparse_models(split_root, label) for label in passages}
    room_models = {label: load_sparse_models(split_root, label) for label in rooms}

    rows: list[dict[str, object]] = []
    for room_label in rooms:
        candidates = build_candidates(room_models[room_label], passage_models, args.min_overlap)
        if not candidates:
            rows.append(
                {
                    "room": room_label,
                    "status": f"skipped:no_overlap_ge_{args.min_overlap}",
                }
            )
            print(f"{room_label}: skipped, no overlap >= {args.min_overlap}", flush=True)
            continue

        merged = False
        last_error = ""
        for index, candidate in enumerate(candidates, start=1):
            pair_work_root = work_root / candidate.output_name / f"try_{index}"
            reference_model, other_model = write_normalized_pair(candidate, pair_work_root)
            output_model = output_root / candidate.output_name / "sparse" / "0"
            result = run_model_merger(
                args.colmap,
                reference_model,
                other_model,
                output_model,
                args.max_reproj_error,
            )
            status = "merged"
            if result.returncode == 0:
                validate_model_path(output_model)
                _, merged_images, merged_points3d = read_model(str(output_model), ".bin")
                if len(merged_images) != candidate.expected_union_count:
                    last_error = (
                        f"incomplete_merge:{len(merged_images)}_of_"
                        f"{candidate.expected_union_count}_expected"
                    )
                    shutil.rmtree(output_model.parent.parent, ignore_errors=True)
                else:
                    rows.append(
                        {
                            "room": room_label,
                            "passage": candidate.passage.label,
                            "status": status,
                            "overlap": candidate.overlap,
                            "room_model": candidate.room.path,
                            "room_registered": candidate.room.image_count,
                            "passage_model": candidate.passage.path,
                            "passage_registered": candidate.passage.image_count,
                            "expected_union": candidate.expected_union_count,
                            "merged_registered": len(merged_images),
                            "merged_points3D": len(merged_points3d),
                            "output_model": output_model,
                        }
                    )
                    print(
                        f"{room_label}: {status} with {candidate.passage.label} "
                        f"(overlap={candidate.overlap}, room_model={candidate.room.model_id}, "
                        f"passage_model={candidate.passage.model_id}, "
                        f"merged_images={len(merged_images)})",
                        flush=True,
                    )
                    merged = True
                    break

            if result.returncode != 0:
                last_error = (result.stderr or result.stdout).strip().splitlines()[-1] if (result.stderr or result.stdout) else ""
                shutil.rmtree(output_model.parent.parent, ignore_errors=True)

            aligned_room = pair_work_root / "aligned_room"
            align_result = run_model_aligner(
                args.colmap,
                candidate.room.path,
                candidate.passage.path,
                aligned_room,
                args.min_overlap,
                args.alignment_max_error,
            )
            if align_result.returncode != 0:
                last_error = (
                    (align_result.stderr or align_result.stdout).strip().splitlines()[-1]
                    if (align_result.stderr or align_result.stdout)
                    else last_error
                )
                continue

            merged_registered, merged_points3d_count = union_reference_and_aligned_model(
                candidate.passage.path,
                aligned_room,
                output_model,
            )
            if merged_registered != candidate.expected_union_count:
                last_error = (
                    f"aligned_union_incomplete:{merged_registered}_of_"
                    f"{candidate.expected_union_count}_expected"
                )
                shutil.rmtree(output_model.parent.parent, ignore_errors=True)
                continue

            rows.append(
                {
                    "room": room_label,
                    "passage": candidate.passage.label,
                    "status": "aligned_union",
                    "overlap": candidate.overlap,
                    "room_model": candidate.room.path,
                    "room_registered": candidate.room.image_count,
                    "passage_model": candidate.passage.path,
                    "passage_registered": candidate.passage.image_count,
                    "expected_union": candidate.expected_union_count,
                    "merged_registered": merged_registered,
                    "merged_points3D": merged_points3d_count,
                    "output_model": output_model,
                }
            )
            print(
                f"{room_label}: aligned_union with {candidate.passage.label} "
                f"(overlap={candidate.overlap}, room_model={candidate.room.model_id}, "
                f"passage_model={candidate.passage.model_id}, "
                f"merged_images={merged_registered})",
                flush=True,
            )
            merged = True
            break

        if not merged:
            best = candidates[0]
            rows.append(
                {
                    "room": room_label,
                    "passage": best.passage.label,
                    "status": f"failed:{last_error}",
                    "overlap": best.overlap,
                    "room_model": best.room.path,
                    "room_registered": best.room.image_count,
                    "passage_model": best.passage.path,
                    "passage_registered": best.passage.image_count,
                }
            )
            print(f"{room_label}: failed, last_error={last_error}", flush=True)

    write_summary(output_root, rows)
    print(f"summary: {output_root / 'merge_summary.tsv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
