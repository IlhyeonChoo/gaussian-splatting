# Google Colab Support

This directory contains a Colab-first workflow for preparing images from uploads or videos, running COLMAP, training 3DGS, rendering, and computing metrics.

## Files

- `Gaussian_Splatting_Colab.ipynb`: runnable Colab notebook.
- `setup_colab.sh`: installs system packages, installs the conda-forge CUDA COLMAP build, initializes CUDA extension submodules, and builds them against the active Colab PyTorch runtime.
- `requirements-colab.txt`: Python packages that are safe to install without replacing Colab PyTorch.
- `colab_utils.py`: small helpers for preparing uploaded/Drive images, zips, or videos.

## Basic Use

1. Open `Gaussian_Splatting_Colab.ipynb` in Google Colab.
2. Set the runtime to GPU.
3. Run the setup cells.
4. Set `WORKFLOW_STEPS` to the stages you want to run.
5. Provide an input folder, zip, video, or uploaded files when the `prepare` step is selected.
6. Run the selected workflow cell.

The notebook defaults to preserving data and outputs under Google Drive when Drive is mounted.

## Selectable Workflow Stages

Set `WORKFLOW_STEPS` to a comma-separated subset of:

- `prepare`: prepare data from a Drive/local image folder, zip, tar archive, video, single image, or uploaded files. Video inputs are extracted through `extract_video_frames.py`; tar archives are treated as already-converted COLMAP scenes and restored to `data/<scene>`.
- `colmap`: run `convert.py` on the prepared scene.
- `train`: run `train.py` with Colab-safe defaults, including `--disable_viewer`.
- `render`: run `render.py` for `MODEL_PATH`.
- `metrics`: run `metrics.py` for `MODEL_PATH`.
- `copy`: copy `MODEL_PATH` to `DRIVE_PROJECT_DIR`.

Common examples:

```python
WORKFLOW_STEPS = "prepare,colmap,train"
WORKFLOW_STEPS = "prepare,train"  # for a tar.gz that already contains COLMAP results
WORKFLOW_STEPS = "prepare,train,copy"  # train from tar.gz and copy the model back to Drive
WORKFLOW_STEPS = "colmap"
WORKFLOW_STEPS = "train"
WORKFLOW_STEPS = "render,metrics"
WORKFLOW_STEPS = "prepare,colmap,train,render,copy"
```

Aliases such as `video`, `extract_frames`, `sfm`, `training`, and `3dgs` are accepted.

## Training From a Drive COLMAP Archive

If the notebook is stored at `MyDrive/3DGS/Gaussian_Splatting_Colab.ipynb` and an already-converted COLMAP scene is stored at `MyDrive/3DGS/data/W2_4.tar.gz`, use:

```python
SCENE_NAME = "W2_4"
INPUT_PATH = "/content/drive/MyDrive/3DGS/data/W2_4.tar.gz"
WORKFLOW_STEPS = "prepare,train"
```

Use `WORKFLOW_STEPS = "prepare,train,copy"` if you also want to copy the trained model to `DRIVE_PROJECT_DIR` after training completes.

The archive can contain the scene directory itself or a nested directory such as `scene/`, as long as the extracted COLMAP scene contains `images/` and `sparse/0/`.

## Notes

- The setup script does not install or pin PyTorch. It uses the PyTorch/CUDA version already present in the active Colab runtime.
- Colab uses conda-forge COLMAP instead of the Ubuntu `apt` COLMAP package so the workflow can use a recent CUDA-enabled COLMAP build.
- By default, setup installs `conda-forge::colmap=4.0.*` into `/content/colmap-conda` with `cuda-version=12.9`, then exposes it through `/usr/local/bin/colmap`.
- If the conda solver cannot detect the Colab NVIDIA driver, setup retries with `CONDA_OVERRIDE_CUDA=13.0`. You can override the install with `COLAB_COLMAP_PREFIX`, `COLAB_COLMAP_VERSION`, or `COLAB_COLMAP_CUDA_VERSION` before running setup.
- `nvidia-smi` may show `CUDA Version: 13.0`; that is the maximum CUDA API level supported by the driver. It does not prevent running a COLMAP package built for the CUDA 12.9 runtime on a compatible newer NVIDIA driver.
- The SIBR real-time viewer is not supported in Colab.
- Training commands pass `--disable_viewer` because Colab cannot use the local viewer socket workflow.
- For limited Colab VRAM, use `--data_device cpu`, `--resolution 2` or `4`, and a modest `--max_train_cameras` value.
