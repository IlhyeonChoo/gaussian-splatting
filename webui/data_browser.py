"""Data discovery and path validation helpers for the web UI."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


@dataclass(frozen=True)
class DataCandidate:
    kind: str
    path: Path
    label: str
    count: int = 0


def is_within_directory(directory: Path, target: Path) -> bool:
    directory = directory.resolve()
    target = target.resolve()
    return target == directory or directory in target.parents


def require_allowed_path(raw_path: str | Path, allowed_roots: tuple[Path, ...]) -> Path:
    """Resolve raw_path and ensure it stays inside one of allowed_roots."""
    user_path = Path(raw_path).expanduser()
    if user_path.is_absolute():
        candidates = [user_path]
    else:
        candidates = [root / user_path for root in allowed_roots]
        candidates.append(Path.cwd() / user_path)

    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError:
            continue
        if any(is_within_directory(root, resolved) for root in allowed_roots):
            return resolved

    roots = ", ".join(str(root) for root in allowed_roots)
    raise ValueError(f"Path is outside allowed data roots or does not exist: {raw_path}. Allowed roots: {roots}")


def collect_image_files(source_dir: str | Path) -> list[Path]:
    source = Path(source_dir).resolve(strict=True)
    if not source.is_dir():
        raise NotADirectoryError(f"Image source directory not found: {source}")
    return sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def count_direct_files(path: Path, extensions: set[str]) -> int:
    try:
        return sum(
            1
            for entry in os.scandir(path)
            if entry.is_file(follow_symlinks=False) and Path(entry.name).suffix.lower() in extensions
        )
    except OSError:
        return 0


def is_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTS


def is_colmap_scene_dir(path: str | Path) -> bool:
    scene = Path(path)
    sparse_dir = scene / "sparse" / "0"
    has_sparse = (
        (sparse_dir / "images.bin").is_file()
        or (sparse_dir / "images.txt").is_file()
    ) and (
        (sparse_dir / "cameras.bin").is_file()
        or (sparse_dir / "cameras.txt").is_file()
    )
    has_points = (
        (sparse_dir / "points3D.ply").is_file()
        or (sparse_dir / "points3D.bin").is_file()
        or (sparse_dir / "points3D.txt").is_file()
    )
    return has_sparse and has_points and (scene / "images").is_dir()


def classify_data_path(path: Path) -> str | None:
    if is_video_file(path):
        return "video_file"
    if path.is_dir() and is_colmap_scene_dir(path):
        return "colmap_scene"
    if path.is_dir() and count_direct_files(path, IMAGE_EXTS) > 0:
        return "image_folder"
    return None


def _candidate_label(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def discover_data_candidates(
    data_roots: tuple[Path, ...],
    repo_root: Path,
    *,
    max_depth: int = 5,
) -> list[DataCandidate]:
    """Discover trainable scene, image folder, and video file candidates."""
    candidates: list[DataCandidate] = []
    seen: set[Path] = set()

    for root in data_roots:
        if not root.exists():
            continue
        root = root.resolve()
        stack: list[tuple[Path, int]] = [(root, 0)]
        while stack:
            path, depth = stack.pop()
            if path in seen:
                continue
            seen.add(path)

            kind = classify_data_path(path)
            if kind:
                count = 0
                if kind == "image_folder":
                    count = count_direct_files(path, IMAGE_EXTS)
                candidates.append(
                    DataCandidate(
                        kind=kind,
                        path=path,
                        label=_candidate_label(path, repo_root),
                        count=count,
                    )
                )
                if kind == "colmap_scene":
                    continue

            if path.is_file() or depth >= max_depth:
                continue

            try:
                entries = sorted(os.scandir(path), key=lambda entry: entry.name)
            except OSError:
                continue

            for entry in reversed(entries):
                entry_path = Path(entry.path)
                if entry.is_file(follow_symlinks=False):
                    if is_video_file(entry_path):
                        candidates.append(
                            DataCandidate(
                                kind="video_file",
                                path=entry_path.resolve(),
                                label=_candidate_label(entry_path.resolve(), repo_root),
                            )
                        )
                elif entry.is_dir(follow_symlinks=False):
                    stack.append((entry_path.resolve(), depth + 1))

    return sorted(candidates, key=lambda item: (item.kind, item.label))


def discover_model_candidates(output_root: Path, repo_root: Path, *, max_depth: int = 2) -> list[DataCandidate]:
    """Discover existing trained model directories under output_root."""
    if not output_root.exists():
        return []

    candidates: list[DataCandidate] = []
    stack: list[tuple[Path, int]] = [(output_root.resolve(), 0)]
    while stack:
        path, depth = stack.pop()
        if path != output_root and ((path / "cfg_args").is_file() or (path / "point_cloud").is_dir()):
            candidates.append(
                DataCandidate(
                    kind="model",
                    path=path,
                    label=_candidate_label(path, repo_root),
                )
            )
            continue
        if depth >= max_depth:
            continue
        try:
            entries = sorted(os.scandir(path), key=lambda entry: entry.name)
        except OSError:
            continue
        for entry in reversed(entries):
            if entry.is_dir(follow_symlinks=False):
                stack.append((Path(entry.path).resolve(), depth + 1))

    return sorted(candidates, key=lambda item: item.label)
