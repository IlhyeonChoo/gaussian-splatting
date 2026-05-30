# 3DGS Web UI

This package adds a browser-based local/Tailscale UI for preparing server-side data, running COLMAP, training 3DGS models, rendering outputs, and computing metrics.

## Install

Activate the repository environment first, then install the web dependencies:

```bash
uv pip install -r requirements-web.txt
```

## Run

```bash
uv run python -m webui.app
```

`uv run` uses the lightweight web UI dependencies from `pyproject.toml`. Training and COLMAP subprocesses still use `WEBUI_PYTHON_BIN`, which defaults to `venv/bin/python` when that interpreter exists.

Defaults:

- `WEBUI_BIND_MODE=auto`
- `WEBUI_DATA_ROOTS=data`
- `WEBUI_OUTPUT_ROOT=output`
- `WEBUI_PYTHON_BIN=venv/bin/python`
- `WEBUI_PORT=7860`

In `auto` mode the server binds to the first `tailscale ip -4` address. If Tailscale is unavailable, it binds to `127.0.0.1`.

## Network Boundary

The web UI does not implement user accounts. It is intended for single-user access from the server itself or from devices already joined to the same Tailscale network.

Requests are accepted only from:

- loopback addresses
- the server Tailscale address
- Tailscale CGNAT range `100.64.0.0/10`
- optional CIDRs in `WEBUI_ALLOWED_CIDRS`

Binding to `0.0.0.0` or `::` is refused unless `WEBUI_UNSAFE_ALLOW_ALL=1` is set.

## Workflow

The UI creates one queued job per submitted workflow. Only one job runs at a time. Each job stores its state and log under `output/.webui/`.

Workflow stages are subprocess calls to the existing repository scripts:

- `extract_video_frames.py`
- `python -m webui.prepare_data`
- `convert.py`
- `train.py`
- `render.py`
- `metrics.py`

Commands are built as argv lists and are never executed through a shell.
