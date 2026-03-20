#!/usr/bin/env python3
"""Extract frames from a video in original and/or custom quality."""

import argparse
import math
import os
from pathlib import Path

import cv2


VALID_IMAGE_EXTS = {"png", "jpg", "jpeg", "webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract video frames with original and custom quality outputs"
    )
    parser.add_argument("--video_path", required=True, help="Input video path")
    parser.add_argument("--output_dir", required=True, help="Output base directory")
    parser.add_argument(
        "--mode",
        choices=["original", "custom", "both"],
        default="both",
        help="Which outputs to generate",
    )

    parser.add_argument(
        "--every_nth",
        type=int,
        default=1,
        help="Save one frame every N frames (default: 1)",
    )
    parser.add_argument(
        "--target_fps",
        type=float,
        default=None,
        help="Target output FPS (overrides --every_nth when set)",
    )

    parser.add_argument(
        "--original_format",
        default="png",
        choices=sorted(VALID_IMAGE_EXTS),
        help="Image format for original output (default: png)",
    )
    parser.add_argument(
        "--custom_format",
        default="jpg",
        choices=sorted(VALID_IMAGE_EXTS),
        help="Image format for custom output (default: jpg)",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale factor for custom output resolution (default: 1.0)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Custom output width (keeps aspect ratio if height omitted)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Custom output height (keeps aspect ratio if width omitted)",
    )

    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=95,
        help="JPEG/WebP quality for custom output (1-100, default: 95)",
    )

    args = parser.parse_args()

    if args.every_nth < 1:
        parser.error("--every_nth must be >= 1")
    if args.target_fps is not None and args.target_fps <= 0:
        parser.error("--target_fps must be > 0")
    if args.scale <= 0:
        parser.error("--scale must be > 0")
    if args.width is not None and args.width <= 0:
        parser.error("--width must be > 0")
    if args.height is not None and args.height <= 0:
        parser.error("--height must be > 0")
    if not (1 <= args.jpeg_quality <= 100):
        parser.error("--jpeg_quality must be between 1 and 100")

    return args


def compute_custom_size(src_w: int, src_h: int, scale: float, width: int | None, height: int | None) -> tuple[int, int]:
    if width is not None and height is not None:
        return width, height

    if width is not None:
        return width, max(1, int(round(src_h * (width / src_w))))

    if height is not None:
        return max(1, int(round(src_w * (height / src_h)))), height

    return max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale)))


def imwrite_with_quality(path: Path, img, ext: str, quality: int) -> bool:
    ext = ext.lower()

    if ext in {"jpg", "jpeg"}:
        return cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if ext == "webp":
        return cv2.imwrite(str(path), img, [cv2.IMWRITE_WEBP_QUALITY, quality])
    return cv2.imwrite(str(path), img)


def main() -> None:
    args = parse_args()

    video_path = Path(args.video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir = Path(args.output_dir)
    original_dir = output_dir / "original"
    custom_dir = output_dir / "custom"

    if args.mode in {"original", "both"}:
        original_dir.mkdir(parents=True, exist_ok=True)
    if args.mode in {"custom", "both"}:
        custom_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0:
        src_fps = 30.0

    frame_interval = args.every_nth
    if args.target_fps is not None:
        frame_interval = max(1, int(round(src_fps / args.target_fps)))

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    custom_w, custom_h = compute_custom_size(src_w, src_h, args.scale, args.width, args.height)

    print(f"[INFO] video={video_path}")
    print(f"[INFO] source resolution={src_w}x{src_h}, source fps={src_fps:.3f}")
    print(f"[INFO] save every {frame_interval} frame(s)")
    if args.mode in {"custom", "both"}:
        print(f"[INFO] custom resolution={custom_w}x{custom_h}, format={args.custom_format}")

    frame_idx = 0
    saved_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        stem = f"frame_{saved_idx:06d}"

        if args.mode in {"original", "both"}:
            original_path = original_dir / f"{stem}.{args.original_format}"
            if not imwrite_with_quality(original_path, frame, args.original_format, args.jpeg_quality):
                raise RuntimeError(f"Failed to write: {original_path}")

        if args.mode in {"custom", "both"}:
            if custom_w == frame.shape[1] and custom_h == frame.shape[0]:
                custom_img = frame
            else:
                interpolation = cv2.INTER_AREA if custom_w < frame.shape[1] else cv2.INTER_CUBIC
                custom_img = cv2.resize(frame, (custom_w, custom_h), interpolation=interpolation)

            custom_path = custom_dir / f"{stem}.{args.custom_format}"
            if not imwrite_with_quality(custom_path, custom_img, args.custom_format, args.jpeg_quality):
                raise RuntimeError(f"Failed to write: {custom_path}")

        saved_idx += 1
        frame_idx += 1

    cap.release()

    duration_sec = frame_idx / src_fps if src_fps > 0 else 0.0
    out_fps = src_fps / frame_interval
    print(f"[DONE] processed frames={frame_idx}, saved frames={saved_idx}")
    print(f"[DONE] approx duration={duration_sec:.2f}s, output fps={out_fps:.3f}")
    print(f"[DONE] output dir={output_dir}")


if __name__ == "__main__":
    main()
