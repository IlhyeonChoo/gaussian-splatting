# Google Colab Support

This directory contains a Colab-first workflow for running the Python training, rendering, and metrics parts of this Gaussian Splatting fork.

## Files

- `Gaussian_Splatting_Colab.ipynb`: runnable Colab notebook.
- `setup_colab.sh`: installs system packages, initializes CUDA extension submodules, and builds them against the active Colab PyTorch runtime.
- `requirements-colab.txt`: Python packages that are safe to install without replacing Colab PyTorch.
- `colab_utils.py`: small helpers for preparing uploaded/Drive images, zips, or videos.

## Basic Use

1. Open `Gaussian_Splatting_Colab.ipynb` in Google Colab.
2. Set the runtime to GPU.
3. Run the setup cells.
4. Provide an input folder, zip, video, or uploaded files.
5. Run COLMAP conversion, training, rendering, and optional metrics.

The notebook defaults to preserving data and outputs under Google Drive when Drive is mounted.

## Notes

- The setup script does not install or pin PyTorch. It uses the PyTorch/CUDA version already present in the active Colab runtime.
- The SIBR real-time viewer is not supported in Colab.
- Training commands pass `--disable_viewer` because Colab cannot use the local viewer socket workflow.
- For limited Colab VRAM, use `--data_device cpu`, `--resolution 2` or `4`, and a modest `--max_train_cameras` value.
