# Repository Guidelines

## Project Structure & Module Organization
Core training and rendering entry points live at the repository root: `train.py`, `render.py`, `metrics.py`, `convert.py`, and `full_eval.py`.
Model/data internals are split into focused modules:
- `scene/`: camera loading, dataset readers, Gaussian scene representation.
- `gaussian_renderer/`: differentiable rendering pipeline.
- `utils/`: math, I/O, image, and system helpers.
- `arguments/`: CLI argument groups shared by scripts.
- `lpipsPyTorch/`: LPIPS metric implementation.
- `submodules/`: CUDA/PyTorch extensions (`diff-gaussian-rasterization`, `simple-knn`, `fused-ssim`).
Reference assets/docs are in `assets/` and `SIBR_viewers/`. Runtime artifacts should stay in `output/` and user datasets in `data/`.

## Build, Test, and Development Commands
- `conda env create --file environment.yml && conda activate gaussian_splatting`: create the baseline environment.
- `python train.py -s data/<scene>`: train a Gaussian model from COLMAP/Synthetic input.
- `python render.py -m output/<model_dir>`: render train/test views from a trained model.
- `python metrics.py -m output/<model_dir>`: compute PSNR/SSIM/LPIPS on rendered outputs.
- `python convert.py -s data/<scene>`: run COLMAP preprocessing for custom image folders.
- `bash quickstart.sh`: interactive local workflow helper (data prep/train/render).

## Coding Style & Naming Conventions
Follow Python conventions used in this repo:
- 4-space indentation, snake_case for functions/variables/files, PascalCase for classes.
- Keep modules small and purpose-specific; place reusable logic under `utils/` or `scene/`.
- Prefer explicit CLI flags and descriptive argument names consistent with existing scripts.
No enforced formatter/linter config is committed; use PEP 8-compatible formatting and keep imports/order consistent with nearby code.

## Testing Guidelines
There is no dedicated unit-test suite in this repository. Validate changes with script-level smoke tests:
1. Run a short training pass (for example `--iterations 1000`).
2. Render with the produced checkpoint.
3. Run `metrics.py` and confirm outputs are generated without regressions.
For CUDA/submodule changes, rebuild and rerun at least one end-to-end train-render-metrics cycle.

## Commit & Pull Request Guidelines
Recent history favors short, imperative commit messages (`fix ...`, `add ...`, `update ...`).
- Keep commits scoped to one change.
- Use clear subjects under ~72 chars (example: `fix depth path handling in synthetic loader`).
PRs should include:
- What changed and why.
- Reproduction/validation commands you ran.
- Linked issue(s), and screenshots/videos when viewer or rendering behavior changes.
Exclude large generated artifacts (`output/`, datasets, virtualenv files) from commits.
