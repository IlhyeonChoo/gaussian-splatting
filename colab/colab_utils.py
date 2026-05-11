"""Utility helpers for running this repository in Google Colab."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


def repo_root(path: str | Path | None = None) -> Path:
    """Return a resolved repository root path."""
    return Path(path or Path.cwd()).resolve()


def scene_dir(scene_name: str, root: str | Path | None = None) -> Path:
    """Return the data directory for a scene name."""
    return repo_root(root) / "data" / scene_name


def collect_image_files(source_dir: str | Path) -> list[Path]:
    """Collect supported image files recursively in deterministic order."""
    source = Path(source_dir).expanduser().resolve()
    if not source.is_dir():
        raise NotADirectoryError(f"Image source directory not found: {source}")

    return sorted(
        path for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def evenly_sample(items: list[Path], max_items: int) -> list[Path]:
    """Return at most max_items sampled evenly from items."""
    if max_items <= 0 or len(items) <= max_items:
        return items

    return [items[index * len(items) // max_items] for index in range(max_items)]


def reset_input_dir(target_scene_dir: str | Path) -> Path:
    """Create a clean COLMAP input directory for a scene."""
    input_dir = Path(target_scene_dir) / "input"
    if input_dir.exists():
        shutil.rmtree(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    return input_dir


def prepare_images_from_directory(
    source_dir: str | Path,
    scene_name: str,
    *,
    max_images: int = 0,
    root: str | Path | None = None,
) -> Path:
    """Flatten image files from source_dir into data/<scene_name>/input."""
    images = evenly_sample(collect_image_files(source_dir), int(max_images))
    if not images:
        raise ValueError(f"No supported images found under: {source_dir}")

    target_scene = scene_dir(scene_name, root)
    input_dir = reset_input_dir(target_scene)

    digits = max(6, len(str(len(images))))
    for index, image_path in enumerate(images):
        destination = input_dir / f"image_{index:0{digits}d}{image_path.suffix.lower()}"
        shutil.copy2(image_path, destination)

    print(f"[DONE] Prepared {len(images)} image(s): {input_dir}")
    return target_scene


def _safe_extract_zip(zip_path: Path, output_dir: Path) -> None:
    """Extract a zip file while preventing path traversal."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_root = output_dir.resolve()

    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            target = (output_root / member.filename).resolve()
            if output_root not in target.parents and target != output_root:
                raise ValueError(f"Unsafe path in zip archive: {member.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)


def extract_zip_to_scene(
    zip_path: str | Path,
    scene_name: str,
    *,
    max_images: int = 0,
    root: str | Path | None = None,
    scratch_dir: str | Path = "/content/gaussian_splatting_colab",
) -> Path:
    """Extract an image zip and prepare data/<scene_name>/input."""
    zip_file = Path(zip_path).expanduser().resolve()
    if not zip_file.is_file():
        raise FileNotFoundError(f"Zip file not found: {zip_file}")

    extract_dir = Path(scratch_dir) / "unzipped" / scene_name
    if extract_dir.exists():
        shutil.rmtree(extract_dir)

    _safe_extract_zip(zip_file, extract_dir)
    return prepare_images_from_directory(extract_dir, scene_name, max_images=max_images, root=root)


def extract_video_to_scene(
    video_path: str | Path,
    scene_name: str,
    *,
    target_fps: float = 2.0,
    scale: float = 0.5,
    width: int | None = None,
    height: int | None = None,
    custom_format: str = "jpg",
    jpeg_quality: int = 90,
    max_images: int = 0,
    root: str | Path | None = None,
    scratch_dir: str | Path = "/content/gaussian_splatting_colab",
) -> Path:
    """Extract frames from a video and prepare data/<scene_name>/input."""
    root_path = repo_root(root)
    video_file = Path(video_path).expanduser().resolve()
    if not video_file.is_file():
        raise FileNotFoundError(f"Video file not found: {video_file}")

    frames_dir = Path(scratch_dir) / "frames" / scene_name
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(root_path / "extract_video_frames.py"),
        "--video_path",
        str(video_file),
        "--output_dir",
        str(frames_dir),
        "--mode",
        "custom",
        "--target_fps",
        str(target_fps),
        "--scale",
        str(scale),
        "--custom_format",
        custom_format,
        "--jpeg_quality",
        str(jpeg_quality),
    ]
    if width is not None:
        command.extend(["--width", str(width)])
    if height is not None:
        command.extend(["--height", str(height)])

    subprocess.run(command, cwd=root_path, check=True)
    return prepare_images_from_directory(frames_dir / "custom", scene_name, max_images=max_images, root=root_path)


def prepare_uploaded_path(
    source_path: str | Path,
    scene_name: str,
    *,
    max_images: int = 0,
    root: str | Path | None = None,
    **video_kwargs,
) -> Path:
    """Prepare a scene from an uploaded directory, image zip, video, or image file."""
    source = Path(source_path).expanduser().resolve()
    suffix = source.suffix.lower()

    if source.is_dir():
        return prepare_images_from_directory(source, scene_name, max_images=max_images, root=root)
    if suffix == ".zip":
        return extract_zip_to_scene(source, scene_name, max_images=max_images, root=root)
    if suffix in VIDEO_EXTS:
        return extract_video_to_scene(source, scene_name, max_images=max_images, root=root, **video_kwargs)
    if source.is_file() and suffix in IMAGE_EXTS:
        upload_dir = source.parent
        return prepare_images_from_directory(upload_dir, scene_name, max_images=max_images, root=root)

    raise ValueError(f"Unsupported input path: {source}")


def latest_model_path(output_root: str | Path = "output") -> Path:
    """Return the newest model directory under output_root."""
    root = Path(output_root)
    candidates = [path for path in root.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No model directories found under: {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def copy_model_to_drive(model_path: str | Path, drive_dir: str | Path, *, overwrite: bool = True) -> Path:
    """Copy a model directory to Google Drive or another persistent location."""
    source = Path(model_path).resolve()
    if not source.is_dir():
        raise NotADirectoryError(f"Model directory not found: {source}")

    target_root = Path(drive_dir).expanduser().resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    destination = target_root / source.name

    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"Destination already exists: {destination}")
        shutil.rmtree(destination)

    shutil.copytree(source, destination)
    print(f"[DONE] Copied model to: {destination}")
    return destination
