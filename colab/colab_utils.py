"""Utility helpers for running this repository in Google Colab."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
VALID_WORKFLOW_STEPS = ("prepare", "colmap", "train", "render", "metrics", "copy")
WORKFLOW_STEP_ALIASES = {
    "data": "prepare",
    "extract": "prepare",
    "extract_frames": "prepare",
    "frames": "prepare",
    "video": "prepare",
    "sfm": "colmap",
    "convert": "colmap",
    "training": "train",
    "3dgs": "train",
    "rendering": "render",
    "metric": "metrics",
    "drive": "copy",
}
COLMAP_PRESET_ARGS = {
    "default": (),
    "video": ("--colmap_matcher", "sequential", "--sequential_overlap", "10"),
    "low-memory": (
        "--feature_max_image_size",
        "1600",
        "--sift_max_num_features",
        "4096",
        "--matching_max_num_matches",
        "10000",
    ),
    "hard-scene": (
        "--sift_max_num_features",
        "16384",
        "--sift_peak_threshold",
        "0.003",
        "--guided_matching",
        "1",
    ),
}


def repo_root(path: str | Path | None = None) -> Path:
    """Return a resolved repository root path."""
    return Path(path or Path.cwd()).resolve()


def scene_dir(scene_name: str, root: str | Path | None = None) -> Path:
    """Return the data directory for a scene name."""
    return repo_root(root) / "data" / scene_name


def normalize_workflow_steps(steps: str | list[str] | tuple[str, ...]) -> list[str]:
    """Normalize a comma/space separated step list for the Colab workflow."""
    if isinstance(steps, str):
        raw_steps = [step.strip() for step in steps.replace(",", " ").split()]
    else:
        raw_steps = [str(step).strip() for step in steps]

    normalized = []
    for step in raw_steps:
        if not step:
            continue
        key = step.lower().replace("-", "_")
        resolved = WORKFLOW_STEP_ALIASES.get(key, key)
        resolved = resolved.replace("_", "-")
        if resolved not in VALID_WORKFLOW_STEPS:
            valid = ", ".join(VALID_WORKFLOW_STEPS)
            raise ValueError(f"Unknown workflow step '{step}'. Valid steps: {valid}")
        if resolved not in normalized:
            normalized.append(resolved)

    return normalized


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


def _is_within_directory(directory: Path, target: Path) -> bool:
    """Return whether target is inside directory after resolving symlinks."""
    directory = directory.resolve()
    target = target.resolve()
    return target == directory or directory in target.parents


def _safe_extract_zip(zip_path: Path, output_dir: Path) -> None:
    """Extract a zip file while preventing path traversal."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_root = output_dir.resolve()

    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            target = (output_root / member.filename).resolve()
            if not _is_within_directory(output_root, target):
                raise ValueError(f"Unsafe path in zip archive: {member.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)


def _safe_extract_tar(tar_path: Path, output_dir: Path) -> None:
    """Extract a tar archive while preventing path traversal and links."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_root = output_dir.resolve()

    with tarfile.open(tar_path) as archive:
        for member in archive.getmembers():
            target = (output_root / member.name).resolve()
            if not _is_within_directory(output_root, target):
                raise ValueError(f"Unsafe path in tar archive: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"Links are not allowed in tar archive: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                continue

            source = archive.extractfile(member)
            if source is None:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)


def is_colmap_scene_dir(path: str | Path) -> bool:
    """Return whether path looks like a trainable COLMAP scene directory."""
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


def find_colmap_scene_dir(search_root: str | Path, scene_name: str | None = None) -> Path:
    """Find a COLMAP scene directory inside an extracted archive."""
    root = Path(search_root)
    if is_colmap_scene_dir(root):
        return root

    candidates = []
    for sparse_dir in root.rglob("sparse"):
        candidate = sparse_dir.parent
        if is_colmap_scene_dir(candidate):
            candidates.append(candidate)

    if not candidates:
        raise FileNotFoundError(
            f"No COLMAP scene with images/ and sparse/0/ found under: {root}"
        )
    if scene_name:
        for candidate in candidates:
            if candidate.name == scene_name:
                return candidate

    return sorted(candidates, key=lambda path: (len(path.parts), str(path)))[0]


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


def restore_colmap_archive_to_scene(
    archive_path: str | Path,
    scene_name: str,
    *,
    root: str | Path | None = None,
    scratch_dir: str | Path = "/content/gaussian_splatting_colab",
) -> Path:
    """Restore a tar archive containing an already-converted COLMAP scene."""
    archive_file = Path(archive_path).expanduser().resolve()
    if not archive_file.is_file():
        raise FileNotFoundError(f"Archive file not found: {archive_file}")
    if not tarfile.is_tarfile(archive_file):
        raise ValueError(f"Not a supported tar archive: {archive_file}")

    root_path = repo_root(root)
    extract_dir = Path(scratch_dir) / "archives" / scene_name
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    _safe_extract_tar(archive_file, extract_dir)
    source_scene = find_colmap_scene_dir(extract_dir, scene_name=scene_name)

    target_scene = scene_dir(scene_name, root_path)
    if target_scene.exists():
        shutil.rmtree(target_scene)
    target_scene.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_scene, target_scene)

    verify_colmap_scene(scene_name, root_path)
    print(f"[DONE] Restored COLMAP scene: {target_scene}")
    return target_scene


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
    if source.is_file() and tarfile.is_tarfile(source):
        return restore_colmap_archive_to_scene(source, scene_name, root=root)
    if suffix in VIDEO_EXTS:
        return extract_video_to_scene(source, scene_name, max_images=max_images, root=root, **video_kwargs)
    if source.is_file() and suffix in IMAGE_EXTS:
        upload_dir = source.parent
        return prepare_images_from_directory(upload_dir, scene_name, max_images=max_images, root=root)

    raise ValueError(
        f"Unsupported input path: {source}. Expected a directory, zip, tar archive, video, or image."
    )


def scene_input_images(scene_name: str, root: str | Path | None = None) -> list[Path]:
    """Return supported image files directly under data/<scene_name>/input."""
    input_dir = scene_dir(scene_name, root) / "input"
    if not input_dir.is_dir():
        return []
    return sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def verify_scene_input(scene_name: str, root: str | Path | None = None) -> Path:
    """Validate that a scene has input images prepared for COLMAP."""
    target_scene = scene_dir(scene_name, root)
    images = scene_input_images(scene_name, root)
    if not images:
        raise FileNotFoundError(
            f"No input images found under {target_scene / 'input'}. "
            "Run the prepare step first or place images there."
        )

    print(f"[INFO] Scene path: {target_scene}")
    print(f"[INFO] Input images: {len(images)}")
    print(f"[INFO] First files: {[path.name for path in images[:5]]}")
    return target_scene


def verify_colmap_scene(scene_name: str, root: str | Path | None = None) -> Path:
    """Validate that a scene has COLMAP outputs required by train.py."""
    target_scene = scene_dir(scene_name, root)
    if not is_colmap_scene_dir(target_scene):
        raise FileNotFoundError(
            f"COLMAP scene is incomplete under {target_scene}. "
            "Expected images/ and sparse/0/images.*, cameras.*, and points3D.*."
        )

    image_count = len(collect_image_files(target_scene / "images"))
    if image_count == 0:
        raise FileNotFoundError(f"No training images found under {target_scene / 'images'}.")

    print(f"[INFO] COLMAP scene path: {target_scene}")
    print(f"[INFO] Training images: {image_count}")
    return target_scene


def _run(command: list[str], root: str | Path | None = None) -> None:
    """Run a repository command from the resolved root and print it first."""
    root_path = repo_root(root)
    print(shlex.join(command))
    subprocess.run(command, cwd=root_path, check=True)


def _extra_args(args: list[str] | tuple[str, ...] | None) -> list[str]:
    """Return a mutable list for optional command arguments."""
    if args is None:
        return []
    return [str(arg) for arg in args]


def run_colmap_conversion(
    scene_name: str,
    *,
    root: str | Path | None = None,
    colmap_device: str = "auto",
    colmap_preset: str = "default",
    extra_args: list[str] | tuple[str, ...] | None = None,
) -> Path:
    """Run convert.py for a prepared scene."""
    root_path = repo_root(root)
    target_scene = verify_scene_input(scene_name, root_path)
    if colmap_preset not in COLMAP_PRESET_ARGS:
        valid = ", ".join(COLMAP_PRESET_ARGS)
        raise ValueError(f"Unknown COLMAP preset '{colmap_preset}'. Valid presets: {valid}")

    command = [
        sys.executable,
        "convert.py",
        "-s",
        str(target_scene),
        "--colmap_device",
        colmap_device,
        *COLMAP_PRESET_ARGS[colmap_preset],
        *_extra_args(extra_args),
    ]
    _run(command, root_path)
    return target_scene


def run_training(
    scene_name: str,
    model_path: str | Path,
    *,
    root: str | Path | None = None,
    iterations: int = 30000,
    max_train_cameras: int = 0,
    camera_quality_ratio: float = 0.7,
    camera_selection_seed: int = 42,
    resolution: int = 1,
    data_device: str = "cuda",
    eval_split: bool = False,
    use_sparse_adam: bool = False,
    extra_args: list[str] | tuple[str, ...] | None = None,
) -> Path:
    """Run train.py for a prepared scene."""
    root_path = repo_root(root)
    target_scene = verify_colmap_scene(scene_name, root_path)
    output_path = Path(model_path)

    command = [
        sys.executable,
        "train.py",
        "-s",
        str(target_scene),
        "-m",
        str(output_path),
        "--iterations",
        str(iterations),
        "--max_train_cameras",
        str(max_train_cameras),
        "--camera_quality_ratio",
        str(camera_quality_ratio),
        "--camera_selection_seed",
        str(camera_selection_seed),
        "--resolution",
        str(resolution),
        "--data_device",
        data_device,
        "--disable_viewer",
    ]
    if eval_split:
        command.append("--eval")
    if use_sparse_adam:
        command.extend(["--optimizer_type", "sparse_adam"])
    command.extend(_extra_args(extra_args))

    _run(command, root_path)
    return output_path


def run_rendering(
    model_path: str | Path,
    *,
    root: str | Path | None = None,
    extra_args: list[str] | tuple[str, ...] | None = None,
) -> None:
    """Run render.py for a trained model."""
    command = [sys.executable, "render.py", "-m", str(model_path), *_extra_args(extra_args)]
    _run(command, root)


def run_metrics(
    model_path: str | Path,
    *,
    root: str | Path | None = None,
    extra_args: list[str] | tuple[str, ...] | None = None,
) -> None:
    """Run metrics.py for rendered outputs."""
    command = [sys.executable, "metrics.py", "-m", str(model_path), *_extra_args(extra_args)]
    _run(command, root)


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
