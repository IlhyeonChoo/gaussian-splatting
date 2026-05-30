#!/usr/bin/env python3
"""Split frame images into COLMAP input folders from a range text file."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


RANGE_RE = re.compile(r"(\d+)\s*~\s*(\d+)")


def parse_subtxt(path: Path) -> dict[str, list[tuple[int, int]]]:
    groups: dict[str, list[tuple[int, int]]] = {}
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or ":" not in line:
            continue

        label, ranges_text = line.split(":", 1)
        label = label.strip()
        ranges = []
        for start_text, end_text in RANGE_RE.findall(ranges_text):
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"{path}:{line_no}: invalid range {start_text} ~ {end_text}")
            ranges.append((start, end))

        if ranges:
            groups[label] = ranges

    return groups


def frame_name(index: int, suffix: str) -> str:
    return f"frame_{index:06d}{suffix}"


def iter_indices(ranges: list[tuple[int, int]]) -> list[int]:
    indices: list[int] = []
    seen: set[int] = set()
    for start, end in ranges:
        for index in range(start, end + 1):
            if index not in seen:
                indices.append(index)
                seen.add(index)
    return indices


def copy_group(
    label: str,
    ranges: list[tuple[int, int]],
    image_dir: Path,
    output_root: Path,
    suffix: str,
) -> tuple[int, list[str]]:
    output_dir = output_root / label / "input"
    output_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing: list[str] = []
    for index in iter_indices(ranges):
        name = frame_name(index, suffix)
        src = image_dir / name
        dst = output_dir / name
        if not src.is_file():
            missing.append(name)
            continue
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            copied += 1
            continue
        shutil.copy2(src, dst)
        copied += 1

    return copied, missing


def write_summary(
    output_root: Path,
    source_dir: Path,
    subtxt: Path,
    results: list[tuple[str, list[tuple[int, int]], int, list[str]]],
) -> None:
    lines = [
        f"source_dir\t{source_dir}",
        f"subtxt\t{subtxt}",
        "mode\tcopy",
        "",
        "label\tranges\tcopied_or_present\tmissing_count\tmissing",
    ]
    for label, ranges, copied, missing in results:
        ranges_text = ",".join(f"{start:06d}-{end:06d}" for start, end in ranges)
        missing_text = ",".join(missing)
        lines.append(f"{label}\t{ranges_text}\t{copied}\t{len(missing)}\t{missing_text}")

    (output_root / "split_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--subtxt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--suffix", default=".png")
    parser.add_argument(
        "--exclude-prefix",
        action="append",
        default=[],
        help="Skip labels starting with this prefix. Can be passed multiple times.",
    )
    args = parser.parse_args()

    image_dir = args.image_dir.resolve()
    subtxt = args.subtxt.resolve()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    groups = parse_subtxt(subtxt)
    results: list[tuple[str, list[tuple[int, int]], int, list[str]]] = []
    for label, ranges in groups.items():
        if any(label.startswith(prefix) for prefix in args.exclude_prefix):
            continue
        copied, missing = copy_group(label, ranges, image_dir, output_root, args.suffix)
        results.append((label, ranges, copied, missing))
        print(f"{label}: {copied} files, missing {len(missing)}")

    write_summary(output_root, image_dir, subtxt, results)
    print(f"summary: {output_root / 'split_summary.txt'}")


if __name__ == "__main__":
    main()
