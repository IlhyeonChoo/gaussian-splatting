#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

echo "[INFO] Repository: ${REPO_ROOT}"

if ! command -v python >/dev/null 2>&1; then
    echo "[ERROR] python is not available." >&2
    exit 1
fi

python - <<'PY'
import sys

try:
    import torch
except Exception as exc:
    raise SystemExit(f"[ERROR] PyTorch is not available in this runtime: {exc}")

print(f"[INFO] Python: {sys.version.split()[0]}")
print(f"[INFO] PyTorch: {torch.__version__}")
print(f"[INFO] PyTorch CUDA: {torch.version.cuda}")
print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
PY

echo "[INFO] Installing system packages."
if command -v apt-get >/dev/null 2>&1; then
    apt-get update -y
    apt-get install -y --no-install-recommends \
        build-essential \
        colmap \
        git \
        imagemagick \
        ninja-build
else
    echo "[WARN] apt-get is not available. Install COLMAP, ImageMagick, git, build-essential, and ninja manually."
fi

echo "[INFO] Installing Python runtime dependencies."
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r colab/requirements-colab.txt

echo "[INFO] Initializing CUDA extension submodules."
git submodule update --init --recursive \
    submodules/diff-gaussian-rasterization \
    submodules/simple-knn \
    submodules/fused-ssim

if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
    DETECTED_ARCH_LIST="$(python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit(0)

arches = []
for idx in range(torch.cuda.device_count()):
    major, minor = torch.cuda.get_device_capability(idx)
    arch = f"{major}.{minor}"
    if arch not in arches:
        arches.append(arch)
print(";".join(arches))
PY
)"
    if [ -n "${DETECTED_ARCH_LIST}" ]; then
        export TORCH_CUDA_ARCH_LIST="${DETECTED_ARCH_LIST}"
        echo "[INFO] TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
    fi
fi

export MAX_JOBS="${MAX_JOBS:-2}"

echo "[INFO] Building and installing local CUDA extensions."
python -m pip install --no-build-isolation submodules/diff-gaussian-rasterization
python -m pip install --no-build-isolation submodules/simple-knn
python -m pip install --no-build-isolation submodules/fused-ssim

echo "[INFO] Validating installation."
python - <<'PY'
import importlib
import shutil

import cv2
import torch
from plyfile import PlyData

modules = [
    "diff_gaussian_rasterization",
    "simple_knn._C",
    "fused_ssim",
]
for module_name in modules:
    importlib.import_module(module_name)

print(f"[INFO] cv2: {cv2.__version__}")
print(f"[INFO] torch cuda available: {torch.cuda.is_available()}")
print(f"[INFO] colmap: {shutil.which('colmap')}")
print("[DONE] Colab setup is ready.")
PY
