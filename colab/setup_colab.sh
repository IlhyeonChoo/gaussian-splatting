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

export GIT_TERMINAL_PROMPT=0

clone_or_update_extension() {
    local target="$1"
    local url="$2"
    local ref="${3:-}"

    echo "[INFO] Preparing ${target}"
    mkdir -p "$(dirname "${target}")"

    if [ -e "${target}/.git" ]; then
        git -C "${target}" remote set-url origin "${url}" || true
        if [ -n "${ref}" ]; then
            git -C "${target}" fetch --depth 1 origin "${ref}"
            git -C "${target}" checkout --force FETCH_HEAD
        else
            git -C "${target}" fetch --depth 1 origin HEAD
            git -C "${target}" checkout --force FETCH_HEAD
        fi
    elif [ -f "${target}/setup.py" ] || [ -f "${target}/pyproject.toml" ]; then
        echo "[INFO] Using existing populated extension directory: ${target}"
    else
        rm -rf "${target}"
        if [ -n "${ref}" ]; then
            git clone --depth 1 --branch "${ref}" "${url}" "${target}"
        else
            git clone --depth 1 "${url}" "${target}"
        fi
    fi

    if [ ! -f "${target}/setup.py" ] && [ ! -f "${target}/pyproject.toml" ]; then
        echo "[ERROR] ${target} does not contain setup.py or pyproject.toml after checkout." >&2
        exit 1
    fi
}

echo "[INFO] Preparing CUDA extension source directories."
clone_or_update_extension \
    submodules/diff-gaussian-rasterization \
    https://github.com/graphdeco-inria/diff-gaussian-rasterization.git \
    dr_aa
clone_or_update_extension \
    submodules/simple-knn \
    https://gitlab.inria.fr/bkerbl/simple-knn.git
clone_or_update_extension \
    submodules/fused-ssim \
    https://github.com/rahul-goel/fused-ssim.git \
    main

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
