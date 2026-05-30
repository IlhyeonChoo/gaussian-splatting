"""Build safe subprocess command specifications for web UI jobs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .config import WebUIConfig
from .data_browser import (
    classify_data_path,
    collect_image_files,
    is_colmap_scene_dir,
    is_video_file,
    require_allowed_path,
)


COLMAP_PRESET_ARGS: dict[str, tuple[str, ...]] = {
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

COLMAP_MATCHERS = {"exhaustive", "sequential", "spatial", "vocab_tree"}
MAPPER_TYPES = {"incremental", "global"}
FEATURE_TYPES = {"SIFT", "ALIKED"}
MATCHING_TYPES = {
    "",
    "SIFT_BRUTEFORCE",
    "SIFT_LIGHTGLUE",
    "ALIKED_BRUTEFORCE",
    "ALIKED_LIGHTGLUE",
}
COLMAP_DEVICES = {"auto", "gpu", "cpu"}
CAMERA_MODES = {"single", "per_folder", "per_image", "shared_off"}
ZERO_ONE_FIELDS = {
    "guided_matching",
    "sift_matching_cross_check",
    "sequential_loop_detection",
    "spatial_ignore_z",
    "mapper_tri_ignore_two_view_tracks",
    "mapper_multiple_models",
    "ba_refine_focal_length",
    "ba_refine_extra_params",
    "ba_refine_principal_point",
    "mapper_ba_use_gpu",
}
INT_COLMAP_FIELDS = {
    "feature_max_image_size",
    "sift_max_num_features",
    "aliked_max_num_features",
    "matching_max_num_matches",
    "exhaustive_block_size",
    "sequential_overlap",
    "spatial_max_num_neighbors",
    "two_view_min_num_inliers",
    "vocab_tree_num_images",
    "vocab_tree_num_nearest_neighbors",
    "mapper_min_num_matches",
    "num_threads",
    "undistort_max_image_size",
    "undistort_jpeg_quality",
}
FLOAT_COLMAP_FIELDS = {
    "sift_peak_threshold",
    "sift_edge_threshold",
    "aliked_min_score",
    "sift_matching_max_ratio",
    "sift_matching_max_distance",
    "sift_lightglue_min_score",
    "aliked_matching_min_cossim",
    "aliked_matching_max_ratio",
    "aliked_lightglue_min_score",
    "two_view_max_error",
    "spatial_max_distance",
    "mapper_filter_max_reproj_error",
    "mapper_tri_min_angle",
    "mapper_max_runtime_seconds",
    "mapper_ba_global_function_tolerance",
}
TEXT_COLMAP_FIELDS = {
    "camera",
    "camera_params",
    "feature_gpu_index",
    "matching_gpu_index",
    "mapper_ba_gpu_index",
    "undistort_copy_policy",
    "extra_feature_args",
    "extra_matching_args",
    "extra_mapper_args",
    "extra_undistort_args",
}
PATH_COLMAP_FIELDS = {
    "mask_path",
    "camera_mask_path",
    "image_list_path",
    "colmap_project_path",
    "vocab_tree_path",
}

INT_TRAIN_FIELDS = {
    "sh_degree",
    "resolution",
    "max_train_cameras",
    "camera_selection_seed",
    "pose_outlier_min_cameras",
    "iterations",
    "position_lr_max_steps",
    "exposure_lr_delay_steps",
    "densification_interval",
    "opacity_reset_interval",
    "densify_from_iter",
    "densify_until_iter",
    "debug_from",
}
FLOAT_TRAIN_FIELDS = {
    "camera_quality_ratio",
    "pose_outlier_mad_scale",
    "position_lr_init",
    "position_lr_final",
    "position_lr_delay_mult",
    "feature_lr",
    "opacity_lr",
    "scaling_lr",
    "rotation_lr",
    "exposure_lr_init",
    "exposure_lr_final",
    "exposure_lr_delay_mult",
    "percent_dense",
    "lambda_dssim",
    "densify_grad_threshold",
    "depth_l1_weight_init",
    "depth_l1_weight_final",
}
TEXT_TRAIN_FIELDS = {
    "images",
    "depths",
    "data_device",
    "optimizer_type",
}
BOOL_TRAIN_FIELDS = {
    "white_background",
    "train_test_exp",
    "eval",
    "convert_SHs_python",
    "compute_cov3D_python",
    "debug",
    "antialiasing",
    "random_background",
    "detect_anomaly",
    "quiet",
}


@dataclass(frozen=True)
class CommandStep:
    """A subprocess step executed by the job worker."""

    name: str
    argv: list[str]
    cwd: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "argv": self.argv, "cwd": self.cwd}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CommandStep":
        return cls(
            name=str(payload["name"]),
            argv=[str(item) for item in payload["argv"]],
            cwd=str(payload["cwd"]),
        )


@dataclass(frozen=True)
class JobSpec:
    """A complete workflow job requested from the web UI."""

    name: str
    source_kind: str
    source_path: str
    scene_path: str
    model_path: str
    steps: list[CommandStep]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_kind": self.source_kind,
            "source_path": self.source_path,
            "scene_path": self.scene_path,
            "model_path": self.model_path,
            "steps": [step.to_dict() for step in self.steps],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "JobSpec":
        return cls(
            name=str(payload["name"]),
            source_kind=str(payload["source_kind"]),
            source_path=str(payload["source_path"]),
            scene_path=str(payload["scene_path"]),
            model_path=str(payload["model_path"]),
            steps=[CommandStep.from_dict(item) for item in payload["steps"]],
        )


class FormReader:
    """Small adapter over Starlette FormData and plain dict objects."""

    def __init__(self, form: Mapping[str, Any]):
        self.form = form

    def get(self, name: str, default: str = "") -> str:
        value = self.form.get(name, default)
        if isinstance(value, list):
            value = value[-1] if value else default
        if value is None:
            return default
        return str(value).strip()

    def checked(self, name: str) -> bool:
        return self.get(name).lower() in {"1", "true", "yes", "on"}


def _safe_name(value: str, default: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return safe or default


def _next_available_path(base: Path) -> Path:
    if not base.exists():
        return base
    index = 2
    while True:
        candidate = base.with_name(f"{base.name}_v{index}")
        if not candidate.exists():
            return candidate
        index += 1


def _choice(value: str, field_name: str, choices: set[str], *, allow_blank: bool = False) -> str:
    if allow_blank and value == "":
        return value
    if value not in choices:
        valid = ", ".join(sorted(choices))
        raise ValueError(f"{field_name} must be one of: {valid}")
    return value


def _optional_int(reader: FormReader, name: str, *, minimum: int | None = None) -> int | None:
    raw = reader.get(name)
    if raw == "":
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return value


def _optional_float(reader: FormReader, name: str, *, minimum: float | None = None) -> float | None:
    raw = reader.get(name)
    if raw == "":
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number.") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return value


def _append(argv: list[str], flag: str, value: str | int | float | None) -> None:
    if value is None or value == "":
        return
    argv.extend([flag, str(value)])


def _append_bool(argv: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        argv.append(flag)


def _parse_int_list(raw: str, field_name: str) -> list[str]:
    if raw.strip() == "":
        return []
    values: list[str] = []
    for token in raw.replace(",", " ").split():
        try:
            values.append(str(int(token)))
        except ValueError as exc:
            raise ValueError(f"{field_name} must contain only integers.") from exc
    return values


def _append_int_list(argv: list[str], flag: str, raw: str, field_name: str) -> None:
    values = _parse_int_list(raw, field_name)
    if values:
        argv.append(flag)
        argv.extend(values)


def _append_allowed_path(
    argv: list[str],
    flag: str,
    raw: str,
    roots: tuple[Path, ...],
) -> None:
    if raw == "":
        return
    argv.extend([flag, str(require_allowed_path(raw, roots))])


def _build_colmap_options(reader: FormReader, config: WebUIConfig) -> list[str]:
    argv: list[str] = []

    preset = _choice(reader.get("colmap_preset", "default"), "colmap_preset", set(COLMAP_PRESET_ARGS))
    argv.extend(COLMAP_PRESET_ARGS[preset])

    device = _choice(reader.get("colmap_device", "auto"), "colmap_device", COLMAP_DEVICES)
    argv.extend(["--colmap_device", device])

    matcher = reader.get("colmap_matcher")
    if matcher:
        argv.extend(["--colmap_matcher", _choice(matcher, "colmap_matcher", COLMAP_MATCHERS)])

    mapper = reader.get("mapper_type")
    if mapper:
        argv.extend(["--mapper_type", _choice(mapper, "mapper_type", MAPPER_TYPES)])

    feature_type = reader.get("feature_type")
    if feature_type:
        argv.extend(["--feature_type", _choice(feature_type, "feature_type", FEATURE_TYPES)])

    matching_type = reader.get("matching_type")
    if matching_type:
        argv.extend(["--matching_type", _choice(matching_type, "matching_type", MATCHING_TYPES, allow_blank=True)])

    camera_mode = _choice(reader.get("camera_mode", "single"), "camera_mode", CAMERA_MODES)
    if camera_mode == "single":
        argv.extend(["--single_camera", "1"])
    elif camera_mode == "per_folder":
        argv.extend(["--single_camera_per_folder", "1"])
    elif camera_mode == "per_image":
        argv.extend(["--single_camera_per_image", "1"])
    else:
        argv.extend(["--single_camera", "0"])

    _append_bool(argv, "--skip_matching", reader.checked("skip_matching"))
    _append_bool(argv, "--resize", reader.checked("resize"))

    for name in sorted(INT_COLMAP_FIELDS):
        _append(argv, f"--{name}", _optional_int(reader, name, minimum=0))
    for name in sorted(FLOAT_COLMAP_FIELDS):
        _append(argv, f"--{name}", _optional_float(reader, name))
    for name in sorted(ZERO_ONE_FIELDS):
        raw = reader.get(name)
        if raw:
            _append(argv, f"--{name}", _choice(raw, name, {"0", "1"}))
    for name in sorted(TEXT_COLMAP_FIELDS):
        raw = reader.get(name)
        if raw:
            if name == "matching_type":
                raw = _choice(raw, name, MATCHING_TYPES, allow_blank=True)
            _append(argv, f"--{name}", raw)
    for name in sorted(PATH_COLMAP_FIELDS):
        _append_allowed_path(argv, f"--{name}", reader.get(name), config.data_roots)

    return argv


def _build_train_options(reader: FormReader, config: WebUIConfig) -> list[str]:
    argv: list[str] = ["--disable_viewer"]

    for name in sorted(INT_TRAIN_FIELDS):
        _append(argv, f"--{name}", _optional_int(reader, name))
    for name in sorted(FLOAT_TRAIN_FIELDS):
        _append(argv, f"--{name}", _optional_float(reader, name))
    for name in sorted(TEXT_TRAIN_FIELDS):
        raw = reader.get(name)
        if not raw:
            continue
        if name == "data_device":
            raw = _choice(raw, name, {"cuda", "cpu"})
        if name == "optimizer_type":
            raw = _choice(raw, name, {"default", "sparse_adam"})
        _append(argv, f"--{name}", raw)
    for name in sorted(BOOL_TRAIN_FIELDS):
        _append_bool(argv, f"--{name}", reader.checked(name))

    _append_int_list(argv, "--test_iterations", reader.get("test_iterations"), "test_iterations")
    _append_int_list(argv, "--save_iterations", reader.get("save_iterations"), "save_iterations")
    _append_int_list(argv, "--checkpoint_iterations", reader.get("checkpoint_iterations"), "checkpoint_iterations")

    start_checkpoint = reader.get("start_checkpoint")
    if start_checkpoint:
        roots = config.data_roots + (config.output_root,)
        _append_allowed_path(argv, "--start_checkpoint", start_checkpoint, roots)

    return argv


def _build_render_options(reader: FormReader) -> list[str]:
    argv: list[str] = []
    _append(argv, "--iteration", _optional_int(reader, "render_iteration"))
    _append_bool(argv, "--skip_train", reader.checked("render_skip_train"))
    _append_bool(argv, "--skip_test", reader.checked("render_skip_test"))
    _append_bool(argv, "--quiet", reader.checked("render_quiet"))
    _append_bool(argv, "--convert_SHs_python", reader.checked("render_convert_SHs_python"))
    _append_bool(argv, "--compute_cov3D_python", reader.checked("render_compute_cov3D_python"))
    _append_bool(argv, "--debug", reader.checked("render_debug"))
    _append_bool(argv, "--antialiasing", reader.checked("render_antialiasing"))
    return argv


def _validate_source(source_path: Path, source_kind: str) -> None:
    detected = classify_data_path(source_path)
    if source_kind == "colmap_scene" and not is_colmap_scene_dir(source_path):
        raise ValueError("Selected source is not a COLMAP scene with images/ and sparse/0/.")
    if source_kind == "image_folder" and len(collect_image_files(source_path)) == 0:
        raise ValueError("Selected image folder does not contain supported images.")
    if source_kind == "video_file" and not is_video_file(source_path):
        raise ValueError("Selected source is not a supported video file.")
    if detected and detected != source_kind:
        raise ValueError(f"Source kind is {source_kind}, but the selected path looks like {detected}.")


def build_job_spec(form: Mapping[str, Any], config: WebUIConfig) -> JobSpec:
    """Build a validated job specification from submitted form data."""
    reader = FormReader(form)
    source_kind = _choice(
        reader.get("source_kind", "colmap_scene"),
        "source_kind",
        {"colmap_scene", "image_folder", "video_file"},
    )
    source_path = require_allowed_path(reader.get("source_path"), config.data_roots)
    _validate_source(source_path, source_kind)

    repo_root = config.repo_root
    steps: list[CommandStep] = []

    scene_name = _safe_name(reader.get("scene_name") or source_path.stem, "scene")
    model_output_name = _safe_name(reader.get("model_output_name") or scene_name, "model")

    stage_prepare = reader.checked("stage_prepare")
    stage_frames = reader.checked("stage_frames")
    stage_colmap = reader.checked("stage_colmap")
    stage_train = reader.checked("stage_train")
    stage_render = reader.checked("stage_render")
    stage_metrics = reader.checked("stage_metrics")

    if source_kind == "colmap_scene":
        scene_path = source_path
        stage_frames = False
        stage_prepare = False
        if stage_colmap and not (scene_path / "input").is_dir():
            raise ValueError("COLMAP rerun requires an input/ directory in the selected scene.")
    else:
        if not stage_prepare and (stage_colmap or stage_train):
            raise ValueError("Image/video inputs require the Prepare stage before COLMAP or training.")
        scene_path = _next_available_path(config.primary_data_root / scene_name)

    if source_kind == "video_file":
        if not stage_frames and stage_prepare:
            raise ValueError("Video inputs require the Extract Frames stage before Prepare.")
        if stage_frames:
            frame_mode = _choice(reader.get("frame_mode", "both"), "frame_mode", {"original", "custom", "both"})
            frame_output_dir = scene_path / "frames"
            argv = [
                config.python_bin,
                "extract_video_frames.py",
                "--video_path",
                str(source_path),
                "--output_dir",
                str(frame_output_dir),
                "--mode",
                frame_mode,
            ]
            target_fps = _optional_float(reader, "target_fps", minimum=0.000001)
            every_nth = _optional_int(reader, "every_nth", minimum=1)
            if target_fps is not None:
                argv.extend(["--target_fps", str(target_fps)])
            elif every_nth is not None:
                argv.extend(["--every_nth", str(every_nth)])
            for name in ("width", "height"):
                _append(argv, f"--{name}", _optional_int(reader, name, minimum=1))
            _append(argv, "--scale", _optional_float(reader, "scale", minimum=0.000001))
            jpeg_quality = _optional_int(reader, "jpeg_quality", minimum=1)
            if jpeg_quality is not None and jpeg_quality > 100:
                raise ValueError("jpeg_quality must be between 1 and 100.")
            _append(argv, "--jpeg_quality", jpeg_quality)
            custom_format = reader.get("custom_format")
            if custom_format:
                _append(argv, "--custom_format", _choice(custom_format, "custom_format", {"jpg", "jpeg", "png", "webp"}))
            original_format = reader.get("original_format")
            if original_format:
                _append(argv, "--original_format", _choice(original_format, "original_format", {"jpg", "jpeg", "png", "webp"}))
            steps.append(CommandStep("extract_frames", argv, str(repo_root)))

    if stage_prepare:
        if source_kind == "video_file":
            prepared_set = _choice(reader.get("prepared_frame_set", "custom"), "prepared_frame_set", {"original", "custom"})
            frame_mode = reader.get("frame_mode", "both")
            if stage_frames and frame_mode != "both" and prepared_set != frame_mode:
                raise ValueError("prepared_frame_set must be generated by frame_mode.")
            prepare_source = scene_path / "frames" / prepared_set
        else:
            prepare_source = source_path
        max_images = _optional_int(reader, "max_images", minimum=0)
        argv = [
            config.python_bin,
            "-m",
            "webui.prepare_data",
            "--source",
            str(prepare_source),
            "--scene_path",
            str(scene_path),
            "--max_images",
            str(max_images or 0),
        ]
        steps.append(CommandStep("prepare", argv, str(repo_root)))

    if stage_colmap:
        argv = [
            config.python_bin,
            "convert.py",
            "-s",
            str(scene_path),
        ]
        argv.extend(_build_colmap_options(reader, config))
        steps.append(CommandStep("colmap", argv, str(repo_root)))

    model_path = Path("")
    if stage_train:
        model_path = _next_available_path(config.output_root / model_output_name)
        argv = [
            config.python_bin,
            "train.py",
            "-s",
            str(scene_path),
            "-m",
            str(model_path),
        ]
        argv.extend(_build_train_options(reader, config))
        steps.append(CommandStep("train", argv, str(repo_root)))
    elif stage_render or stage_metrics:
        existing_model = reader.get("existing_model_path")
        if not existing_model:
            raise ValueError("Render or metrics without training requires an existing model path.")
        model_path = require_allowed_path(existing_model, (config.output_root,))

    if stage_render:
        argv = [config.python_bin, "render.py", "-m", str(model_path)]
        argv.extend(_build_render_options(reader))
        steps.append(CommandStep("render", argv, str(repo_root)))

    if stage_metrics:
        steps.append(
            CommandStep(
                "metrics",
                [config.python_bin, "metrics.py", "-m", str(model_path)],
                str(repo_root),
            )
        )

    if not steps:
        raise ValueError("Select at least one workflow stage.")

    job_name = reader.get("job_name") or f"{source_kind}:{scene_name}"
    return JobSpec(
        name=job_name,
        source_kind=source_kind,
        source_path=str(source_path),
        scene_path=str(scene_path),
        model_path=str(model_path) if str(model_path) else "",
        steps=steps,
    )
