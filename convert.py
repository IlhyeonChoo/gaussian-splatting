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
import shutil
import struct
import subprocess
import sys
from argparse import ArgumentParser


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


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
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
args = parser.parse_args()


if args.no_gpu and "--colmap_device" in sys.argv:
    parser.error("--no_gpu cannot be used together with --colmap_device.")

if args.no_gpu:
    logging.warning("--no_gpu is deprecated; use --colmap_device cpu instead.")
    args.colmap_device = "cpu"


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


def detect_gpu_option(help_text, candidates):
    for candidate in candidates:
        if f"--{candidate} " in help_text or help_text.rstrip().endswith(f"--{candidate}"):
            return candidate
    return None


def inspect_colmap_environment():
    _, main_help = get_help_output([colmap_command, "-h"])
    _, feature_help = get_help_output([colmap_command, "feature_extractor", "-h"])
    _, matching_help = get_help_output([colmap_command, "exhaustive_matcher", "-h"])

    return {
        "cuda_available": "without CUDA" not in main_help,
        "feature_gpu_option": detect_gpu_option(
            feature_help,
            ("FeatureExtraction.use_gpu", "SiftExtraction.use_gpu"),
        ),
        "matching_gpu_option": detect_gpu_option(
            matching_help,
            ("FeatureMatching.use_gpu", "SiftMatching.use_gpu"),
        ),
        "main_help": main_help,
    }


def build_feature_extraction_command(use_gpu, feature_gpu_option):
    command = [
        colmap_command,
        "feature_extractor",
        "--database_path",
        os.path.join(args.source_path, "distorted", "database.db"),
        "--image_path",
        os.path.join(args.source_path, "input"),
        "--ImageReader.single_camera",
        "1",
        "--ImageReader.camera_model",
        args.camera,
    ]
    if feature_gpu_option is not None:
        command.extend([f"--{feature_gpu_option}", str(use_gpu)])
    return command


def build_feature_matching_command(use_gpu, matching_gpu_option):
    command = [
        colmap_command,
        "exhaustive_matcher",
        "--database_path",
        os.path.join(args.source_path, "distorted", "database.db"),
    ]
    if matching_gpu_option is not None:
        command.extend([f"--{matching_gpu_option}", str(use_gpu)])
    return command


def build_mapper_command():
    return [
        colmap_command,
        "mapper",
        "--database_path",
        os.path.join(args.source_path, "distorted", "database.db"),
        "--image_path",
        os.path.join(args.source_path, "input"),
        "--output_path",
        os.path.join(args.source_path, "distorted", "sparse"),
        "--Mapper.ba_global_function_tolerance=0.000001",
    ]


def build_image_undistorter_command(input_path):
    return [
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
        build_feature_extraction_command(use_gpu, capabilities["feature_gpu_option"]),
        f"feature extraction ({attempt_name})",
    )
    if exit_code != 0:
        return exit_code

    return run_command(
        build_feature_matching_command(use_gpu, capabilities["matching_gpu_option"]),
        f"feature matching ({attempt_name})",
    )


def move_sparse_outputs():
    sparse_root = os.path.join(args.source_path, "sparse")
    files = os.listdir(sparse_root)
    sparse_zero_path = os.path.join(sparse_root, "0")

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

    if capabilities["feature_gpu_option"] is None or capabilities["matching_gpu_option"] is None:
        logging.error("Could not detect COLMAP GPU toggle options from the installed binary.")
        raise SystemExit(1)

    if args.colmap_device == "gpu" and not capabilities["cuda_available"]:
        logging.error("GPU-only mode requested, but the installed COLMAP binary reports 'without CUDA'.")
        raise SystemExit(1)

    if not args.skip_matching:
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

        exit_code = run_command(build_mapper_command(), "mapper")
        if exit_code != 0:
            raise SystemExit(exit_code)
    else:
        logging.info("Skipping COLMAP matching and mapper; using existing sparse reconstruction.")

    selected_sparse_model_path = select_largest_sparse_model()
    exit_code = run_command(
        build_image_undistorter_command(selected_sparse_model_path),
        "image undistortion",
    )
    if exit_code != 0:
        raise SystemExit(exit_code)

    move_sparse_outputs()

    if args.resize:
        resize_images()

    print("Done.")
except FileNotFoundError as exc:
    logging.error("Executable not found: %s", exc.filename)
    raise SystemExit(1) from exc
