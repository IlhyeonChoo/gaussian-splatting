#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import logging
import os
import shlex
import shutil
import struct
import subprocess
import sys
from argparse import ArgumentParser


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


MATCHER_COMMANDS = {
    "exhaustive": "exhaustive_matcher",
    "sequential": "sequential_matcher",
    "spatial": "spatial_matcher",
    "vocab_tree": "vocab_tree_matcher",
}
MATCHER_OPTION_PREFIXES = {
    "exhaustive": "ExhaustiveMatching",
    "sequential": "SequentialMatching",
    "spatial": "SpatialMatching",
    "vocab_tree": "VocabTreeMatching",
}
MATCHING_TYPES = (
    "SIFT_BRUTEFORCE",
    "SIFT_LIGHTGLUE",
    "ALIKED_BRUTEFORCE",
    "ALIKED_LIGHTGLUE",
)
MAPPER_COMMANDS = {
    "incremental": "mapper",
    "global": "global_mapper",
}


parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action="store_true")
parser.add_argument(
    "--colmap_device",
    default="auto",
    choices=("auto", "gpu", "cpu"),
    help="COLMAP device for SIFT extraction/matching: auto, gpu, or cpu.",
)
parser.add_argument("--skip_matching", action="store_true")
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--camera_params", default="", type=str)
parser.add_argument("--single_camera", default=1, choices=(0, 1), type=int)
parser.add_argument("--single_camera_per_folder", default=0, choices=(0, 1), type=int)
parser.add_argument("--single_camera_per_image", default=0, choices=(0, 1), type=int)
parser.add_argument("--mask_path", default="", type=str)
parser.add_argument("--camera_mask_path", default="", type=str)
parser.add_argument("--image_list_path", default="", type=str)
parser.add_argument("--colmap_project_path", default="", type=str)
parser.add_argument(
    "--colmap_matcher",
    default="exhaustive",
    choices=tuple(MATCHER_COMMANDS.keys()),
    help="COLMAP matcher command to run after feature extraction.",
)
parser.add_argument(
    "--mapper_type",
    default="incremental",
    choices=tuple(MAPPER_COMMANDS.keys()),
    help="Sparse mapper implementation to run after matching.",
)
parser.add_argument("--feature_type", default="SIFT", choices=("SIFT", "ALIKED"))
parser.add_argument("--matching_type", default="")
parser.add_argument("--feature_max_image_size", default=None, type=int)
parser.add_argument("--sift_max_num_features", default=None, type=int)
parser.add_argument("--sift_peak_threshold", default=None, type=float)
parser.add_argument("--sift_edge_threshold", default=None, type=float)
parser.add_argument("--aliked_max_num_features", default=None, type=int)
parser.add_argument("--aliked_min_score", default=None, type=float)
parser.add_argument("--matching_max_num_matches", default=None, type=int)
parser.add_argument("--guided_matching", default=None, choices=(0, 1), type=int)
parser.add_argument("--sift_matching_max_ratio", default=None, type=float)
parser.add_argument("--sift_matching_max_distance", default=None, type=float)
parser.add_argument("--sift_matching_cross_check", default=None, choices=(0, 1), type=int)
parser.add_argument("--sift_lightglue_min_score", default=None, type=float)
parser.add_argument("--aliked_matching_min_cossim", default=None, type=float)
parser.add_argument("--aliked_matching_max_ratio", default=None, type=float)
parser.add_argument("--aliked_lightglue_min_score", default=None, type=float)
parser.add_argument("--two_view_min_num_inliers", default=None, type=int)
parser.add_argument("--two_view_max_error", default=None, type=float)
parser.add_argument("--exhaustive_block_size", default=None, type=int)
parser.add_argument("--sequential_overlap", default=None, type=int)
parser.add_argument("--sequential_loop_detection", default=None, choices=(0, 1), type=int)
parser.add_argument("--spatial_ignore_z", default=None, choices=(0, 1), type=int)
parser.add_argument("--spatial_max_num_neighbors", default=None, type=int)
parser.add_argument("--spatial_max_distance", default=None, type=float)
parser.add_argument("--vocab_tree_path", default="", type=str)
parser.add_argument("--vocab_tree_num_images", default=None, type=int)
parser.add_argument("--vocab_tree_num_nearest_neighbors", default=None, type=int)
parser.add_argument("--mapper_min_num_matches", default=None, type=int)
parser.add_argument("--mapper_filter_max_reproj_error", default=None, type=float)
parser.add_argument("--mapper_tri_min_angle", default=None, type=float)
parser.add_argument("--mapper_tri_ignore_two_view_tracks", default=None, choices=(0, 1), type=int)
parser.add_argument("--mapper_multiple_models", default=None, choices=(0, 1), type=int)
parser.add_argument("--mapper_max_runtime_seconds", default=None, type=float)
parser.add_argument("--mapper_ba_global_function_tolerance", default=0.000001, type=float)
parser.add_argument("--ba_refine_focal_length", default=None, choices=(0, 1), type=int)
parser.add_argument("--ba_refine_extra_params", default=None, choices=(0, 1), type=int)
parser.add_argument("--ba_refine_principal_point", default=None, choices=(0, 1), type=int)
parser.add_argument("--mapper_ba_use_gpu", default=None, choices=(0, 1), type=int)
parser.add_argument("--feature_gpu_index", default="", type=str)
parser.add_argument("--matching_gpu_index", default="", type=str)
parser.add_argument("--mapper_ba_gpu_index", default="", type=str)
parser.add_argument("--num_threads", default=None, type=int)
parser.add_argument("--undistort_max_image_size", default=None, type=int)
parser.add_argument("--undistort_copy_policy", default="")
parser.add_argument("--undistort_jpeg_quality", default=None, type=int)
parser.add_argument("--extra_feature_args", default="", type=str)
parser.add_argument("--extra_matching_args", default="", type=str)
parser.add_argument("--extra_mapper_args", default="", type=str)
parser.add_argument("--extra_undistort_args", default="", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
args = parser.parse_args()


single_camera_explicit = any(
    argument == "--single_camera" or argument.startswith("--single_camera=")
    for argument in sys.argv
)
if (args.single_camera_per_folder or args.single_camera_per_image) and not single_camera_explicit:
    args.single_camera = 0

camera_mode_count = sum(
    value == 1
    for value in (
        args.single_camera,
        args.single_camera_per_folder,
        args.single_camera_per_image,
    )
)
if camera_mode_count > 1:
    parser.error("Use only one of --single_camera, --single_camera_per_folder, or --single_camera_per_image.")

colmap_device_explicit = any(
    argument == "--colmap_device" or argument.startswith("--colmap_device=")
    for argument in sys.argv
)
if args.no_gpu and colmap_device_explicit:
    parser.error("--no_gpu cannot be used together with --colmap_device.")

if args.no_gpu:
    logging.warning("--no_gpu is deprecated; use --colmap_device cpu instead.")
    args.colmap_device = "cpu"

if args.matching_type == "" and args.feature_type == "ALIKED":
    args.matching_type = "ALIKED_BRUTEFORCE"

if args.matching_type != "" and args.matching_type not in MATCHING_TYPES:
    parser.error(f"--matching_type must be one of: {', '.join(MATCHING_TYPES)}.")
if args.matching_type.startswith("SIFT") and args.feature_type != "SIFT":
    parser.error("SIFT matching types require --feature_type SIFT.")
if args.matching_type.startswith("ALIKED") and args.feature_type != "ALIKED":
    parser.error("ALIKED matching types require --feature_type ALIKED.")
if args.guided_matching == 1 and "LIGHTGLUE" in args.matching_type:
    parser.error("--guided_matching 1 cannot be used with LightGlue matching types.")
if args.colmap_matcher == "vocab_tree" and len(args.vocab_tree_path) == 0:
    parser.error("--colmap_matcher vocab_tree requires --vocab_tree_path.")
if args.sequential_loop_detection == 1 and len(args.vocab_tree_path) == 0:
    parser.error("--sequential_loop_detection 1 requires --vocab_tree_path.")

if args.undistort_copy_policy:
    copy_policies = ("COPY", "SOFT_LINK", "HARD_LINK", "copy", "soft-link", "hard-link")
    if args.undistort_copy_policy not in copy_policies:
        parser.error("--undistort_copy_policy must be COPY, SOFT_LINK, HARD_LINK, copy, soft-link, or hard-link.")
    args.undistort_copy_policy = args.undistort_copy_policy.upper().replace("-", "_")


colmap_command = args.colmap_executable if len(args.colmap_executable) > 0 else "colmap"
magick_command = args.magick_executable if len(args.magick_executable) > 0 else "magick"


def get_help_output(command):
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return completed.returncode, completed.stdout


def run_command(command, description):
    logging.info("Running %s", description)
    logging.info("Command: %s", " ".join(command))
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        logging.error("%s failed with code %s.", description, completed.returncode)
    return completed.returncode


def has_colmap_option(help_text, option_name):
    for line in help_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("--"):
            continue
        option_token = stripped.split()[0].split("=")[0]
        if option_token == f"--{option_name}":
            return True
    return False


def detect_colmap_option(help_text, candidates):
    for candidate in candidates:
        if has_colmap_option(help_text, candidate):
            return candidate
    return None


def inspect_colmap_environment():
    _, main_help = get_help_output([colmap_command, "-h"])
    _, feature_help = get_help_output([colmap_command, "feature_extractor", "-h"])
    _, matching_help = get_help_output([colmap_command, MATCHER_COMMANDS[args.colmap_matcher], "-h"])
    _, mapper_help = get_help_output([colmap_command, MAPPER_COMMANDS[args.mapper_type], "-h"])
    _, undistorter_help = get_help_output([colmap_command, "image_undistorter", "-h"])

    return {
        "cuda_available": "without CUDA" not in main_help,
        "feature_gpu_option": detect_colmap_option(
            feature_help,
            ("FeatureExtraction.use_gpu", "SiftExtraction.use_gpu"),
        ),
        "matching_gpu_option": detect_colmap_option(
            matching_help,
            ("FeatureMatching.use_gpu", "SiftMatching.use_gpu"),
        ),
        "feature_max_image_size_option": detect_colmap_option(
            feature_help,
            ("FeatureExtraction.max_image_size", "SiftExtraction.max_image_size"),
        ),
        "main_help": main_help,
        "feature_help": feature_help,
        "matching_help": matching_help,
        "mapper_help": mapper_help,
        "undistorter_help": undistorter_help,
    }


def append_colmap_option(command, help_text, option_name, value):
    if value is None or value == "":
        return
    if option_name is None:
        return
    if not has_colmap_option(help_text, option_name):
        logging.warning("Installed COLMAP command does not support --%s; ignoring it.", option_name)
        return
    command.extend([f"--{option_name}", str(value)])


def append_first_supported_option(command, help_text, candidates, value):
    if value is None or value == "":
        return
    option_name = detect_colmap_option(help_text, candidates)
    if option_name is None:
        logging.warning(
            "Installed COLMAP command does not support any of these options: %s; ignoring value %s.",
            ", ".join(f"--{candidate}" for candidate in candidates),
            value,
        )
        return
    command.extend([f"--{option_name}", str(value)])


def append_project_path(command):
    if len(args.colmap_project_path) > 0:
        command.extend(["--project_path", args.colmap_project_path])


def append_extra_args(command, raw_args, option_name):
    if len(raw_args) == 0:
        return
    try:
        command.extend(shlex.split(raw_args))
    except ValueError as exc:
        parser.error(f"Invalid {option_name}: {exc}")


def build_feature_extraction_command(use_gpu, capabilities):
    feature_help = capabilities["feature_help"]
    command = [
        colmap_command,
        "feature_extractor",
        "--database_path",
        os.path.join(args.source_path, "distorted", "database.db"),
        "--image_path",
        os.path.join(args.source_path, "input"),
        "--ImageReader.single_camera",
        str(args.single_camera),
        "--ImageReader.camera_model",
        args.camera,
    ]
    append_project_path(command)
    append_colmap_option(command, feature_help, "image_list_path", args.image_list_path)
    append_colmap_option(command, feature_help, "ImageReader.single_camera_per_folder", args.single_camera_per_folder)
    append_colmap_option(command, feature_help, "ImageReader.single_camera_per_image", args.single_camera_per_image)
    append_colmap_option(command, feature_help, "ImageReader.camera_params", args.camera_params)
    append_colmap_option(command, feature_help, "ImageReader.mask_path", args.mask_path)
    append_colmap_option(command, feature_help, "ImageReader.camera_mask_path", args.camera_mask_path)
    append_colmap_option(command, feature_help, "FeatureExtraction.type", args.feature_type)
    append_colmap_option(command, feature_help, "FeatureExtraction.num_threads", args.num_threads)
    append_colmap_option(command, feature_help, capabilities["feature_gpu_option"], use_gpu)
    append_colmap_option(command, feature_help, "FeatureExtraction.gpu_index", args.feature_gpu_index)
    append_colmap_option(command, feature_help, capabilities["feature_max_image_size_option"], args.feature_max_image_size)
    append_colmap_option(command, feature_help, "SiftExtraction.max_num_features", args.sift_max_num_features)
    append_colmap_option(command, feature_help, "SiftExtraction.peak_threshold", args.sift_peak_threshold)
    append_colmap_option(command, feature_help, "SiftExtraction.edge_threshold", args.sift_edge_threshold)
    append_colmap_option(command, feature_help, "AlikedExtraction.max_num_features", args.aliked_max_num_features)
    append_colmap_option(command, feature_help, "AlikedExtraction.min_score", args.aliked_min_score)
    append_extra_args(command, args.extra_feature_args, "--extra_feature_args")
    return command


def build_feature_matching_command(use_gpu, capabilities):
    matching_help = capabilities["matching_help"]
    matcher_prefix = MATCHER_OPTION_PREFIXES[args.colmap_matcher]
    command = [
        colmap_command,
        MATCHER_COMMANDS[args.colmap_matcher],
        "--database_path",
        os.path.join(args.source_path, "distorted", "database.db"),
    ]
    append_project_path(command)
    append_colmap_option(command, matching_help, "FeatureMatching.type", args.matching_type)
    append_colmap_option(command, matching_help, "FeatureMatching.num_threads", args.num_threads)
    append_colmap_option(command, matching_help, capabilities["matching_gpu_option"], use_gpu)
    append_colmap_option(command, matching_help, "FeatureMatching.gpu_index", args.matching_gpu_index)
    append_colmap_option(command, matching_help, "FeatureMatching.guided_matching", args.guided_matching)
    append_colmap_option(command, matching_help, "FeatureMatching.max_num_matches", args.matching_max_num_matches)
    append_colmap_option(command, matching_help, "SiftMatching.max_ratio", args.sift_matching_max_ratio)
    append_colmap_option(command, matching_help, "SiftMatching.max_distance", args.sift_matching_max_distance)
    append_colmap_option(command, matching_help, "SiftMatching.cross_check", args.sift_matching_cross_check)
    append_colmap_option(command, matching_help, "SiftMatching.lightglue_min_score", args.sift_lightglue_min_score)
    append_colmap_option(command, matching_help, "AlikedMatching.brute_force_min_cossim", args.aliked_matching_min_cossim)
    append_colmap_option(command, matching_help, "AlikedMatching.brute_force_max_ratio", args.aliked_matching_max_ratio)
    append_colmap_option(command, matching_help, "AlikedMatching.lightglue_min_score", args.aliked_lightglue_min_score)
    append_colmap_option(command, matching_help, "TwoViewGeometry.min_num_inliers", args.two_view_min_num_inliers)
    append_colmap_option(command, matching_help, "TwoViewGeometry.max_error", args.two_view_max_error)

    append_colmap_option(command, matching_help, "ExhaustiveMatching.block_size", args.exhaustive_block_size)
    append_colmap_option(command, matching_help, "SequentialMatching.overlap", args.sequential_overlap)
    append_colmap_option(command, matching_help, "SequentialMatching.loop_detection", args.sequential_loop_detection)
    append_colmap_option(command, matching_help, "SequentialMatching.vocab_tree_path", args.vocab_tree_path)
    append_colmap_option(command, matching_help, "SpatialMatching.ignore_z", args.spatial_ignore_z)
    append_colmap_option(command, matching_help, "SpatialMatching.max_num_neighbors", args.spatial_max_num_neighbors)
    append_colmap_option(command, matching_help, "SpatialMatching.max_distance", args.spatial_max_distance)
    append_colmap_option(command, matching_help, "VocabTreeMatching.vocab_tree_path", args.vocab_tree_path)
    append_colmap_option(command, matching_help, "VocabTreeMatching.num_images", args.vocab_tree_num_images)
    append_colmap_option(
        command,
        matching_help,
        "VocabTreeMatching.num_nearest_neighbors",
        args.vocab_tree_num_nearest_neighbors,
    )
    if args.num_threads is not None and args.colmap_matcher in ("sequential", "vocab_tree"):
        append_colmap_option(command, matching_help, f"{matcher_prefix}.num_threads", args.num_threads)
    append_extra_args(command, args.extra_matching_args, "--extra_matching_args")
    return command


def build_mapper_command(capabilities):
    mapper_help = capabilities["mapper_help"]
    command = [
        colmap_command,
        MAPPER_COMMANDS[args.mapper_type],
        "--database_path",
        os.path.join(args.source_path, "distorted", "database.db"),
        "--image_path",
        os.path.join(args.source_path, "input"),
        "--output_path",
        os.path.join(args.source_path, "distorted", "sparse"),
    ]
    append_project_path(command)
    if args.mapper_type == "incremental":
        append_first_supported_option(command, mapper_help, ("Mapper.image_list_path", "image_list_path"), args.image_list_path)
        append_colmap_option(command, mapper_help, "Mapper.min_num_matches", args.mapper_min_num_matches)
        append_colmap_option(command, mapper_help, "Mapper.filter_max_reproj_error", args.mapper_filter_max_reproj_error)
        append_colmap_option(command, mapper_help, "Mapper.tri_min_angle", args.mapper_tri_min_angle)
        append_colmap_option(command, mapper_help, "Mapper.tri_ignore_two_view_tracks", args.mapper_tri_ignore_two_view_tracks)
        append_colmap_option(command, mapper_help, "Mapper.multiple_models", args.mapper_multiple_models)
        append_colmap_option(command, mapper_help, "Mapper.max_runtime_seconds", args.mapper_max_runtime_seconds)
        append_colmap_option(command, mapper_help, "Mapper.num_threads", args.num_threads)
        append_colmap_option(command, mapper_help, "Mapper.ba_refine_focal_length", args.ba_refine_focal_length)
        append_colmap_option(command, mapper_help, "Mapper.ba_refine_extra_params", args.ba_refine_extra_params)
        append_colmap_option(command, mapper_help, "Mapper.ba_refine_principal_point", args.ba_refine_principal_point)
        append_colmap_option(command, mapper_help, "Mapper.ba_use_gpu", args.mapper_ba_use_gpu)
        append_colmap_option(command, mapper_help, "Mapper.ba_gpu_index", args.mapper_ba_gpu_index)
        append_colmap_option(
            command,
            mapper_help,
            "Mapper.ba_global_function_tolerance",
            args.mapper_ba_global_function_tolerance,
        )
    else:
        append_first_supported_option(command, mapper_help, ("GlobalMapper.image_list_path", "image_list_path"), args.image_list_path)
        append_colmap_option(command, mapper_help, "GlobalMapper.min_num_matches", args.mapper_min_num_matches)
        append_colmap_option(command, mapper_help, "GlobalMapper.tri_min_angle", args.mapper_tri_min_angle)
        append_colmap_option(command, mapper_help, "GlobalMapper.num_threads", args.num_threads)
        append_colmap_option(command, mapper_help, "GlobalMapper.ba_refine_focal_length", args.ba_refine_focal_length)
        append_colmap_option(command, mapper_help, "GlobalMapper.ba_refine_extra_params", args.ba_refine_extra_params)
        append_colmap_option(command, mapper_help, "GlobalMapper.ba_refine_principal_point", args.ba_refine_principal_point)
        append_colmap_option(command, mapper_help, "GlobalMapper.ba_ceres_use_gpu", args.mapper_ba_use_gpu)
        append_colmap_option(command, mapper_help, "GlobalMapper.ba_ceres_gpu_index", args.mapper_ba_gpu_index)
        if args.mapper_filter_max_reproj_error is not None:
            logging.warning("--mapper_filter_max_reproj_error is only supported by the incremental mapper.")
        if args.mapper_tri_ignore_two_view_tracks is not None:
            logging.warning("--mapper_tri_ignore_two_view_tracks is only supported by the incremental mapper.")
        if args.mapper_multiple_models is not None:
            logging.warning("--mapper_multiple_models is only supported by the incremental mapper.")
        if args.mapper_max_runtime_seconds is not None:
            logging.warning("--mapper_max_runtime_seconds is only supported by the incremental mapper.")
    append_extra_args(command, args.extra_mapper_args, "--extra_mapper_args")
    return command


def build_image_undistorter_command(input_path, capabilities):
    undistorter_help = capabilities["undistorter_help"]
    command = [
        colmap_command,
        "image_undistorter",
        "--image_path",
        os.path.join(args.source_path, "input"),
        "--input_path",
        input_path,
        "--output_path",
        args.source_path,
        "--output_type",
        "COLMAP",
    ]
    append_project_path(command)
    append_colmap_option(command, undistorter_help, "image_list_path", args.image_list_path)
    append_colmap_option(command, undistorter_help, "max_image_size", args.undistort_max_image_size)
    append_colmap_option(command, undistorter_help, "copy_policy", args.undistort_copy_policy)
    append_colmap_option(command, undistorter_help, "jpeg_quality", args.undistort_jpeg_quality)
    append_colmap_option(command, undistorter_help, "num_threads", args.num_threads)
    append_extra_args(command, args.extra_undistort_args, "--extra_undistort_args")
    return command


def cleanup_matching_artifacts():
    database_path = os.path.join(args.source_path, "distorted", "database.db")
    sparse_path = os.path.join(args.source_path, "distorted", "sparse")

    if os.path.isfile(database_path):
        os.remove(database_path)
    if os.path.isdir(sparse_path):
        shutil.rmtree(sparse_path)

    os.makedirs(sparse_path, exist_ok=True)


def count_registered_images(images_bin_path):
    with open(images_bin_path, "rb") as file:
        num_images = struct.unpack("<Q", file.read(8))[0]
        for _ in range(num_images):
            file.seek(64, os.SEEK_CUR)

            while True:
                current_byte = file.read(1)
                if current_byte == b"\x00":
                    break
                if current_byte == b"":
                    raise ValueError(f"Unexpected EOF while reading image name from {images_bin_path}")

            num_points_2d = struct.unpack("<Q", file.read(8))[0]
            file.seek(num_points_2d * 24, os.SEEK_CUR)

    return num_images


def select_largest_sparse_model():
    sparse_root = os.path.join(args.source_path, "distorted", "sparse")
    candidates = []

    root_images_bin_path = os.path.join(sparse_root, "images.bin")
    if os.path.isfile(root_images_bin_path):
        registered_images = count_registered_images(root_images_bin_path)
        candidates.append((sparse_root, registered_images))

    for entry in os.scandir(sparse_root):
        if not entry.is_dir():
            continue

        images_bin_path = os.path.join(entry.path, "images.bin")
        if not os.path.isfile(images_bin_path):
            continue

        registered_images = count_registered_images(images_bin_path)
        candidates.append((entry.path, registered_images))

    if not candidates:
        raise ValueError(f"No sparse COLMAP models found under {sparse_root}")

    candidates.sort(key=lambda item: (-item[1], item[0]))
    for model_path, registered_images in candidates:
        logging.info(
            "Found sparse model %s with %d registered images.",
            os.path.basename(model_path),
            registered_images,
        )

    selected_model_path, selected_model_size = candidates[0]
    logging.info(
        "Selected sparse model %s with %d registered images.",
        os.path.basename(selected_model_path),
        selected_model_size,
    )
    return selected_model_path


def run_matching_pipeline(use_gpu, attempt_name, capabilities):
    gpu_state = "enabled" if use_gpu else "disabled"
    logging.info("Starting COLMAP matching attempt '%s' with GPU %s.", attempt_name, gpu_state)

    exit_code = run_command(
        build_feature_extraction_command(use_gpu, capabilities),
        f"feature extraction ({attempt_name})",
    )
    if exit_code != 0:
        return exit_code

    return run_command(
        build_feature_matching_command(use_gpu, capabilities),
        f"feature matching ({attempt_name})",
    )


def move_sparse_outputs():
    sparse_root = os.path.join(args.source_path, "sparse")
    files = os.listdir(sparse_root)
    sparse_zero_path = os.path.join(sparse_root, "0")
    loose_model_files = [file for file in files if file != "0"]

    if not loose_model_files:
        logging.info("Sparse outputs are already organized under sparse/0.")
        return

    if os.path.isdir(sparse_zero_path):
        shutil.rmtree(sparse_zero_path)
    os.makedirs(sparse_zero_path, exist_ok=True)

    for file in files:
        if file == "0":
            continue
        source_file = os.path.join(sparse_root, file)
        destination_file = os.path.join(sparse_zero_path, file)
        shutil.move(source_file, destination_file)


def resize_images():
    print("Copying and resizing...")

    os.makedirs(os.path.join(args.source_path, "images_2"), exist_ok=True)
    os.makedirs(os.path.join(args.source_path, "images_4"), exist_ok=True)
    os.makedirs(os.path.join(args.source_path, "images_8"), exist_ok=True)

    files = os.listdir(os.path.join(args.source_path, "images"))
    for file in files:
        source_file = os.path.join(args.source_path, "images", file)

        destination_file = os.path.join(args.source_path, "images_2", file)
        shutil.copy2(source_file, destination_file)
        exit_code = run_command(
            [magick_command, "mogrify", "-resize", "50%", destination_file],
            f"50% resize ({file})",
        )
        if exit_code != 0:
            raise SystemExit(exit_code)

        destination_file = os.path.join(args.source_path, "images_4", file)
        shutil.copy2(source_file, destination_file)
        exit_code = run_command(
            [magick_command, "mogrify", "-resize", "25%", destination_file],
            f"25% resize ({file})",
        )
        if exit_code != 0:
            raise SystemExit(exit_code)

        destination_file = os.path.join(args.source_path, "images_8", file)
        shutil.copy2(source_file, destination_file)
        exit_code = run_command(
            [magick_command, "mogrify", "-resize", "12.5%", destination_file],
            f"12.5% resize ({file})",
        )
        if exit_code != 0:
            raise SystemExit(exit_code)


try:
    capabilities = inspect_colmap_environment()
    logging.info("Detected COLMAP CUDA support: %s", "enabled" if capabilities["cuda_available"] else "disabled")
    logging.info("Detected feature GPU option: %s", capabilities["feature_gpu_option"])
    logging.info("Detected matching GPU option: %s", capabilities["matching_gpu_option"])
    logging.info("COLMAP matcher command: %s", MATCHER_COMMANDS[args.colmap_matcher])
    logging.info("COLMAP mapper command: %s", MAPPER_COMMANDS[args.mapper_type])

    if not args.skip_matching:
        if capabilities["feature_gpu_option"] is None or capabilities["matching_gpu_option"] is None:
            logging.error("Could not detect COLMAP GPU toggle options from the installed binary.")
            raise SystemExit(1)

        if args.colmap_device == "gpu" and not capabilities["cuda_available"]:
            logging.error("GPU-only mode requested, but the installed COLMAP binary reports 'without CUDA'.")
            raise SystemExit(1)

        cleanup_matching_artifacts()
        logging.info("COLMAP matching mode: %s", args.colmap_device)

        if args.colmap_device == "auto":
            if capabilities["cuda_available"]:
                exit_code = run_matching_pipeline(
                    use_gpu=1,
                    attempt_name="auto/gpu",
                    capabilities=capabilities,
                )
                if exit_code != 0:
                    logging.warning("GPU matching failed. Retrying with CPU.")
                    cleanup_matching_artifacts()
                    exit_code = run_matching_pipeline(
                        use_gpu=0,
                        attempt_name="auto/cpu-fallback",
                        capabilities=capabilities,
                    )
            else:
                logging.info("Installed COLMAP binary is built without CUDA. Using CPU directly in auto mode.")
                exit_code = run_matching_pipeline(
                    use_gpu=0,
                    attempt_name="auto/cpu-no-cuda",
                    capabilities=capabilities,
                )
        else:
            use_gpu = 1 if args.colmap_device == "gpu" else 0
            exit_code = run_matching_pipeline(
                use_gpu=use_gpu,
                attempt_name=args.colmap_device,
                capabilities=capabilities,
            )

        if exit_code != 0:
            if args.colmap_device == "gpu":
                logging.error("GPU-only mode failed. Re-run with --colmap_device auto or --colmap_device cpu.")
            raise SystemExit(exit_code)

        exit_code = run_command(build_mapper_command(capabilities), MAPPER_COMMANDS[args.mapper_type])
        if exit_code != 0:
            raise SystemExit(exit_code)
    else:
        logging.info("Skipping COLMAP matching and mapper; using existing sparse reconstruction.")

    selected_sparse_model_path = select_largest_sparse_model()
    exit_code = run_command(
        build_image_undistorter_command(selected_sparse_model_path, capabilities),
        "image undistortion",
    )
    if exit_code != 0:
        raise SystemExit(exit_code)

    move_sparse_outputs()

    if args.resize:
        resize_images()

    print("Done.")
except FileNotFoundError as exc:
    logging.error("Required file or executable not found: %s", exc.filename)
    raise SystemExit(1) from exc
