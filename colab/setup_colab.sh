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
        bzip2 \
        build-essential \
        ca-certificates \
        curl \
        git \
        imagemagick \
        ninja-build
else
    echo "[WARN] apt-get is not available. Install ImageMagick, git, build-essential, and ninja manually."
fi

export COLAB_COLMAP_PREFIX="${COLAB_COLMAP_PREFIX:-/content/colmap-conda}"
export COLAB_COLMAP_VERSION="${COLAB_COLMAP_VERSION:-4.0.*}"
export COLAB_COLMAP_CUDA_VERSION="${COLAB_COLMAP_CUDA_VERSION:-12.9}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/content/micromamba}"

install_micromamba() {
    if command -v micromamba >/dev/null 2>&1; then
        command -v micromamba
        return
    fi

    local install_dir="/content/micromamba-bin"
    local executable="${install_dir}/micromamba"
    if [ ! -x "${executable}" ]; then
        echo "[INFO] Installing micromamba." >&2
        mkdir -p "${install_dir}"
        curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
            | tar -xj -C "${install_dir}" --strip-components=1 bin/micromamba
        chmod +x "${executable}"
    fi
    echo "${executable}"
}

MICROMAMBA="$(install_micromamba)"

create_colmap_env() {
    local override="${1:-}"
    local -a command=(
        "${MICROMAMBA}" create
        --yes
        --prefix "${COLAB_COLMAP_PREFIX}"
        --channel conda-forge
        --strict-channel-priority
        "conda-forge::colmap=${COLAB_COLMAP_VERSION}"
        "cuda-version=${COLAB_COLMAP_CUDA_VERSION}"
    )

    if [ -n "${override}" ]; then
        echo "[INFO] Installing COLMAP with CONDA_OVERRIDE_CUDA=${override}."
        CONDA_OVERRIDE_CUDA="${override}" "${command[@]}"
    else
        echo "[INFO] Installing COLMAP with detected CUDA driver metadata."
        "${command[@]}"
    fi
}

if [ ! -x "${COLAB_COLMAP_PREFIX}/bin/colmap" ]; then
    echo "[INFO] Installing conda-forge COLMAP ${COLAB_COLMAP_VERSION} with CUDA ${COLAB_COLMAP_CUDA_VERSION} support."
    if [ -n "${CONDA_OVERRIDE_CUDA:-}" ]; then
        create_colmap_env "${CONDA_OVERRIDE_CUDA}"
    elif ! create_colmap_env ""; then
        echo "[WARN] COLMAP solve failed without CUDA override; retrying with CONDA_OVERRIDE_CUDA=13.0."
        create_colmap_env "13.0"
    fi
else
    echo "[INFO] Reusing existing COLMAP environment: ${COLAB_COLMAP_PREFIX}"
fi

if [ ! -x "${COLAB_COLMAP_PREFIX}/bin/colmap" ]; then
    echo "[ERROR] COLMAP executable was not installed at ${COLAB_COLMAP_PREFIX}/bin/colmap." >&2
    exit 1
fi

echo "[INFO] Installing COLMAP wrapper at /usr/local/bin/colmap."
cat >/usr/local/bin/colmap <<'SH'
#!/usr/bin/env bash
set -euo pipefail

COLAB_COLMAP_PREFIX="${COLAB_COLMAP_PREFIX:-/content/colmap-conda}"
export LD_LIBRARY_PATH="${COLAB_COLMAP_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
exec "${COLAB_COLMAP_PREFIX}/bin/colmap" "$@"
SH
chmod +x /usr/local/bin/colmap

echo "[INFO] Validating COLMAP CUDA build."
COLMAP_HELP="$(colmap -h 2>&1 || true)"
printf '%s\n' "${COLMAP_HELP}" | head -n 40
if printf '%s\n' "${COLMAP_HELP}" | grep -qi "without CUDA"; then
    echo "[ERROR] Installed COLMAP reports that it was built without CUDA." >&2
    exit 1
fi
if ! colmap -h >/dev/null 2>&1; then
    echo "[ERROR] colmap -h failed after installation." >&2
    exit 1
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

    if [ -f "${target}/.gitmodules" ]; then
        git -C "${target}" submodule update --init --recursive --depth 1
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
python -m pip install --no-build-isolation --no-cache-dir -v submodules/diff-gaussian-rasterization
python -m pip install --no-build-isolation --no-cache-dir -v submodules/simple-knn
python -m pip install --no-build-isolation --no-cache-dir -v submodules/fused-ssim

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
