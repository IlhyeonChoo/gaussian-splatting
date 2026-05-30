"""Parse 3DGS ``train.py`` log output into structured progress data.

This module is pure stdlib and side-effect free. ``train.py`` writes a tqdm
progress bar plus a few ``[ITER ...]`` status lines; the job worker captures
that output line-by-line in text/universal-newline mode, so each tqdm
carriage-return refresh lands as its own line in the log file. We scan that
text and surface the latest training state for the web UI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# tqdm line, e.g.:
#   Training progress:  45%|████▌     | 13500/30000 [02:14<02:43,  100.50it/s, Loss=0.0312000, Depth Loss=0.0000000]
# The trailing ", Loss=..., Depth Loss=...]" postfix is absent in early lines,
# and the ETA / rate can be "?" before tqdm has an estimate.
_TQDM_RE = re.compile(
    r"Training progress:\s+(\d+)%.*?(\d+)/(\d+)\s+"
    r"\[([\d:]+)<([\d:?]+),\s+([\d.?]+)\s*it/s"
    r"(?:,\s*Loss=([\d.eE+-]+),\s*Depth Loss=([\d.eE+-]+))?\]"
)
_EVAL_RE = re.compile(
    r"\[ITER (\d+)\] Evaluating (test|train): L1 ([\d.eE+-]+) PSNR ([\d.eE+-]+)"
)
_SAVE_RE = re.compile(r"\[ITER (\d+)\] Saving Gaussians")
_OUTPUT_RE = re.compile(r"Output folder:\s*(.+)")


@dataclass(frozen=True)
class Evaluation:
    """A single ``[ITER n] Evaluating <split>`` report."""

    iteration: int
    split: str
    l1: float
    psnr: float


@dataclass(frozen=True)
class LossSample:
    """A single tqdm loss sample."""

    iteration: int
    loss: float
    depth_loss: float | None


@dataclass(frozen=True)
class TrainingProgress:
    """Latest parsed state of a training run.

    All scalar fields are ``None`` until the corresponding signal appears in
    the log. ``has_data`` is ``False`` when nothing parseable was found (for
    example while an earlier COLMAP step is still running).
    """

    iteration: int | None = None
    total_iterations: int | None = None
    percent: float | None = None
    rate: float | None = None
    elapsed: str | None = None
    eta: str | None = None
    loss: float | None = None
    depth_loss: float | None = None
    loss_samples: list[LossSample] = field(default_factory=list)
    evaluations: list[Evaluation] = field(default_factory=list)
    saved_iterations: list[int] = field(default_factory=list)
    output_folder: str | None = None
    has_data: bool = False


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_training_progress(text: str) -> TrainingProgress:
    """Parse raw training log ``text`` into a :class:`TrainingProgress`.

    The last tqdm line wins for the current iteration/loss/rate. Evaluation
    and ``Saving Gaussians`` lines accumulate; the last ``Output folder`` wins.
    Malformed lines are skipped rather than raising.
    """

    iteration: int | None = None
    total_iterations: int | None = None
    percent: float | None = None
    rate: float | None = None
    elapsed: str | None = None
    eta: str | None = None
    loss: float | None = None
    depth_loss: float | None = None
    loss_samples: list[LossSample] = []
    evaluations: list[Evaluation] = []
    saved: list[int] = []
    output_folder: str | None = None
    found = False

    for line in text.splitlines():
        tqdm_match = _TQDM_RE.search(line)
        if tqdm_match:
            iteration = _to_int(tqdm_match.group(2))
            total_iterations = _to_int(tqdm_match.group(3))
            percent = _to_float(tqdm_match.group(1))
            elapsed = tqdm_match.group(4)
            eta = tqdm_match.group(5)
            rate = _to_float(tqdm_match.group(6))  # None when "?"
            loss = _to_float(tqdm_match.group(7))
            depth_loss = _to_float(tqdm_match.group(8))
            if iteration is not None and loss is not None:
                loss_samples.append(LossSample(iteration, loss, depth_loss))
            found = True
            continue

        eval_match = _EVAL_RE.search(line)
        if eval_match:
            eval_iter = _to_int(eval_match.group(1))
            l1 = _to_float(eval_match.group(3))
            psnr = _to_float(eval_match.group(4))
            if eval_iter is not None and l1 is not None and psnr is not None:
                evaluations.append(
                    Evaluation(
                        iteration=eval_iter,
                        split=eval_match.group(2),
                        l1=l1,
                        psnr=psnr,
                    )
                )
                found = True
            continue

        save_match = _SAVE_RE.search(line)
        if save_match:
            save_iter = _to_int(save_match.group(1))
            if save_iter is not None and save_iter not in saved:
                saved.append(save_iter)
                found = True
            continue

        output_match = _OUTPUT_RE.search(line)
        if output_match:
            output_folder = output_match.group(1).strip()
            found = True

    return TrainingProgress(
        iteration=iteration,
        total_iterations=total_iterations,
        percent=percent,
        rate=rate,
        elapsed=elapsed,
        eta=eta,
        loss=loss,
        depth_loss=depth_loss,
        loss_samples=loss_samples,
        evaluations=evaluations,
        saved_iterations=saved,
        output_folder=output_folder,
        has_data=found,
    )
