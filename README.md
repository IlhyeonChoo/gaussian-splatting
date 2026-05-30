# 3D Gaussian Splatting Local Workflow

This repository is a personal fork of the original
[graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
implementation for "3D Gaussian Splatting for Real-Time Radiance Field Rendering".

The core training and rendering baseline is still the vanilla 3DGS paper code.
The main additions in this fork are local workflow changes for a personal server
environment, helper scripts for data preparation, and a browser-based Web UI for
configuring and monitoring training without staying inside a server terminal.

This is not an official upstream release. For the original paper, official
documentation, datasets, pre-trained models, and viewer binaries, refer to the
upstream project:

- Project page: <https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/>
- Upstream repository: <https://github.com/graphdeco-inria/gaussian-splatting>
- Paper: <https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_high.pdf>

![Teaser image](assets/teaser.png)

## What Is Different In This Fork

- The 3DGS optimizer, renderer, metrics flow, and SIBR viewer support remain
  based on the vanilla 3DGS codebase.
- Local convenience scripts were added for video frame extraction, COLMAP
  preparation, training, rendering, and scene packaging.
- `convert.py` has additional COLMAP controls, including device selection,
  matcher presets, camera mode options, feature and matching options, and safer
  CPU fallback behavior when GPU COLMAP is unavailable.
- `train.py` and scene loading include local controls for limiting training
  cameras with a quality plus random sampling strategy and filtering obvious
  COLMAP pose outliers.
- `webui/` adds a FastAPI plus HTMX Web UI for launching server-side workflows
  from a local browser or Tailscale-connected device.
- `output/.webui/` stores Web UI job state, logs, and progress metadata.

The Web UI is the biggest workflow difference from upstream: it lets you choose
input data, configure COLMAP and training options, enqueue jobs, watch logs and
training progress, and browse generated outputs from the browser.

## Repository Layout

- `train.py`: main 3DGS optimizer.
- `render.py`, `render_video.py`, `render_interpolated.py`: render trained
  models and camera paths.
- `metrics.py`: compute PSNR, SSIM, and LPIPS for rendered outputs.
- `convert.py`: prepare COLMAP scenes for 3DGS training.
- `extract_video_frames.py`: extract image frames from videos.
- `quickstart.sh`: interactive terminal helper for common local workflows.
- `webui/`: local/Tailscale browser UI and job queue.
- `tools/`: one-off and reusable dataset/COLMAP packaging utilities.
- `scene/`, `gaussian_renderer/`, `utils/`, `arguments/`: core 3DGS modules.
- `submodules/`: CUDA/PyTorch extensions used by the renderer and optimizer.
- `data/`: local datasets and prepared scenes.
- `output/`: trained models, renders, metrics, and Web UI runtime state.

## Environment Notes

The original upstream `environment.yml` is kept for reference, but it targets an
older Python/CUDA/PyTorch stack from the official release. In this fork, the
recommended local workflow is to use a Python virtual environment and keep the
training environment separate from the lightweight Web UI runtime.

Typical local assumptions:

- NVIDIA GPU with a CUDA-capable PyTorch build.
- COLMAP available on `PATH` for custom image or video data.
- Python 3.11 or newer for the Web UI tooling.
- Training dependencies installed in `venv/` or another interpreter selected by
  `WEBUI_PYTHON_BIN`.
- CUDA extension submodules installed into the training environment:
  `diff-gaussian-rasterization`, `simple-knn`, and `fused-ssim`.

Example Web UI dependency setup:

```bash
cd /home/ilhyeonchu/ReCompose3D/3DGS/gaussian-splatting
uv sync
```

If your training environment is not `venv/bin/python`, point the Web UI to the
right interpreter before launching it:

```bash
export WEBUI_PYTHON_BIN=/path/to/training/python
```

## Web UI

Run the browser UI from the repository root:

```bash
uv run python -m webui.app
```

By default, the app:

- uses port `7860`;
- binds to a Tailscale IPv4 address when available;
- falls back to `127.0.0.1` when Tailscale is unavailable;
- reads datasets from `data/`;
- writes model outputs to `output/`;
- stores job state and logs under `output/.webui/`;
- runs training and COLMAP subprocesses with `WEBUI_PYTHON_BIN`.

Useful configuration variables:

```bash
WEBUI_BIND_MODE=auto              # auto, tailscale, or localhost
WEBUI_PORT=7860
WEBUI_DATA_ROOTS=data             # os.pathsep-separated list
WEBUI_OUTPUT_ROOT=output
WEBUI_PYTHON_BIN=venv/bin/python
WEBUI_ALLOWED_CIDRS=192.168.0.0/16 # optional extra client networks
```

Security boundary: the Web UI has no user accounts. It is intended for single
user access from localhost or from devices already joined to the same Tailscale
network. Binding to `0.0.0.0` or `::` is refused unless
`WEBUI_UNSAFE_ALLOW_ALL=1` is set.

The Web UI can run these stages as a queued job:

1. Extract frames from a video.
2. Prepare a scene input folder.
3. Run COLMAP conversion.
4. Train a 3DGS model.
5. Render train/test outputs.
6. Compute metrics.

Only one queued job runs at a time. Commands are built as argument lists and are
not executed through a shell.

## CLI Workflow

Use this when you want direct terminal control instead of the Web UI.

### 1. Prepare Images

For image folders, place images under `data/<scene>/input`:

```bash
mkdir -p data/my_scene/input
# copy jpg, jpeg, png, or webp images into data/my_scene/input
```

For videos, extract frames first:

```bash
python extract_video_frames.py \
  --video_path data/videos/input.mp4 \
  --output_dir data/my_scene/frames \
  --mode both \
  --target_fps 2 \
  --scale 0.5 \
  --custom_format jpg \
  --jpeg_quality 90
```

Then prepare the selected frame set as the scene input:

```bash
python -m webui.prepare_data \
  --source data/my_scene/frames/custom \
  --scene_path data/my_scene \
  --max_images 0
```

### 2. Run COLMAP Conversion

```bash
python convert.py -s data/my_scene --colmap_device auto
```

Common local presets:

```bash
# Sequential matcher for video-like ordered frames.
python convert.py -s data/my_scene \
  --colmap_matcher sequential \
  --sequential_overlap 10 \
  --colmap_device auto

# Lower memory COLMAP settings.
python convert.py -s data/my_scene \
  --feature_max_image_size 1600 \
  --sift_max_num_features 4096 \
  --matching_max_num_matches 10000

# Harder scenes with more features and guided matching.
python convert.py -s data/my_scene \
  --sift_max_num_features 16384 \
  --sift_peak_threshold 0.003 \
  --guided_matching 1
```

`--colmap_device auto` tries GPU COLMAP when available and falls back to CPU
when appropriate. Use `--colmap_device gpu` to fail fast if GPU COLMAP cannot
run, or `--colmap_device cpu` to force CPU mode.

### 3. Train

```bash
python train.py -s data/my_scene -m output/my_scene --iterations 7000
```

Useful local options:

```bash
# Limit the number of training cameras. 70 percent are selected by COLMAP
# correspondence count and the rest are sampled randomly with a fixed seed.
python train.py -s data/my_scene -m output/my_scene \
  --iterations 7000 \
  --max_train_cameras 120 \
  --camera_quality_ratio 0.7 \
  --camera_selection_seed 42

# Reduce image resolution to save VRAM.
python train.py -s data/my_scene -m output/my_scene --resolution 2

# Use CPU image storage to reduce VRAM pressure at some speed cost.
python train.py -s data/my_scene -m output/my_scene --data_device cpu

# Use sparse Adam when the installed rasterizer supports it.
python train.py -s data/my_scene -m output/my_scene --optimizer_type sparse_adam
```

### 4. Render And Evaluate

```bash
python render.py -m output/my_scene
python metrics.py -m output/my_scene
```

Rendered images and metrics are written inside the model directory under
`output/`.

## Interactive Terminal Helper

`quickstart.sh` provides a menu-driven local workflow for:

- preparing image folders with optional image count limits;
- extracting frames from videos;
- running COLMAP with common presets;
- training with camera count controls;
- rendering trained models.

Run it from the repository root:

```bash
bash quickstart.sh
```

## Development Checks

The repository does not include a broad upstream-style test suite, but the Web
UI has focused tests:

```bash
uv run pytest tests
```

For training or renderer changes, validate with the smallest useful smoke test:

```bash
python train.py -s data/<scene> -m output/smoke --iterations 1000
python render.py -m output/smoke
python metrics.py -m output/smoke
```

## Citation

Please cite the original 3DGS paper when using this codebase for research:

```bibtex
@Article{kerbl3Dgaussians,
  author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  journal      = {ACM Transactions on Graphics},
  number       = {4},
  volume       = {42},
  month        = {July},
  year         = {2023},
  url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```

## License

The original 3DGS code is provided for non-commercial, research and evaluation
use under the terms in `LICENSE.md`. Local additions in this fork follow the
same repository license unless a file states otherwise.
