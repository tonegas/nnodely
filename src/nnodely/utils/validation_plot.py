"""Figures for a validation run.

One figure per minimizer with four panels - response, error, parity, error
distribution - and a fifth, the error at each step, for a signal with a
sequence; plus one loss-curve figure when the losses of a training are passed.

Figures are built with the normal Matplotlib pyplot interface, so when they
are shown they come with the interactive toolbar: zoom, pan, rescale, edit
curve and axis properties, and save. The response and error panels share an
x axis, so zooming into a stretch of the run zooms both.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from nnodely.core.results import ValidationResult
from nnodely.core.validation import Signal

#: Channels drawn per panel, worst first. Metrics always cover every channel;
#: this only keeps a figure of a 64-channel signal readable.
MAX_LINES = 6

# _TARGET_STYLE = dict(color="#2E3440", linewidth=1.4, linestyle="--", alpha=0.85)
_PREDICT = ("#6F84FE", "#5B4BFB", "#80BFFF", "#7A3FE8", "#3FA7E8", "#9B6BFF")


def _points(signal: Signal) -> tuple[np.ndarray, np.ndarray]:
    """The source and the target at the points that have a target, laid out
    ``(points, channels)``."""
    channels = int(np.prod(signal.shape.dim))
    keep = signal.valid.reshape(-1)
    return (
        signal.source.reshape(-1, channels)[keep],
        signal.target.reshape(-1, channels)[keep],
    )


def _worst_channels(signal: Signal) -> list[int]:
    order = np.argsort(np.ravel(signal.rmse(dim=True)))[::-1]
    return sorted(int(index) for index in order[:MAX_LINES])


def _label(source: str, channels: int, channel: int) -> str:
    return source if channels == 1 else f"{source}[{channel}]"


def figure_signal(signal: Signal, minimizer: dict):
    """Response, error, parity and error distribution for one minimizer, and
    the error at each step when its signal has a sequence."""
    import matplotlib.pyplot as plt

    source, target = _points(signal)
    source_name, total = minimizer["source"], source.shape[1]
    channels = _worst_channels(signal)
    hidden = total - len(channels)
    # The error at each step of the sequence, every channel apart.
    step_rmse = (
        np.reshape(signal.rmse(seq=True, dim=True), (-1, total))
        if signal.shape.seq_rank
        else None
    )

    rows = 3 if step_rmse is None else 4
    figure = plt.figure(figsize=(12.0, 8.0 if rows == 3 else 10.5))
    grid = figure.add_gridspec(
        rows, 2, height_ratios=(2.0, 1.4, 1.4, 1.4)[:rows], hspace=0.35
    )
    ax_time = figure.add_subplot(grid[0, :])
    ax_error = figure.add_subplot(grid[1, :], sharex=ax_time)
    ax_parity = figure.add_subplot(grid[2, 0])
    ax_hist = figure.add_subplot(grid[2, 1])

    points = np.arange(len(source))
    for position, channel in enumerate(channels):
        colour = _PREDICT[position % len(_PREDICT)]
        ax_time.plot(
            points,
            target[:, channel],
            label=f"target {_label(source_name, total, channel)}",
            color="#2E3440",
            linewidth=1.4,
            linestyle="--",
            alpha=0.85,
        )
        ax_time.plot(
            points,
            source[:, channel],
            color=colour,
            linewidth=1.2,
            label=f"predicted {_label(source_name, total, channel)}",
        )
        ax_error.plot(
            points,
            source[:, channel] - target[:, channel],
            color=colour,
            linewidth=1.0,
        )

    title = f"{signal.name}   {source_name} vs {minimizer['target']}"
    if hidden:
        title += f"   ({len(channels)} of {total} channels, worst first)"
    ax_time.set_title(title)
    ax_time.set_ylabel("value")
    ax_time.grid(alpha=0.25)
    # The error panel below shares this axis and carries the labels for both.
    ax_time.tick_params(labelbottom=False)
    if len(channels) <= 3:
        ax_time.legend(loc="best", fontsize=8)

    ax_error.axhline(0.0, color="#2E3440", linewidth=0.8)
    ax_error.set_ylabel("prediction - target")
    # A point is one sample at one time step; padded steps are left out.
    ax_error.set_xlabel("point")
    ax_error.grid(alpha=0.25)

    flat_target = target[:, channels].ravel()
    flat_source = source[:, channels].ravel()
    ax_parity.scatter(flat_target, flat_source, s=6, alpha=0.35, color=_PREDICT[0])
    if flat_target.size:
        limits = (float(np.min(flat_target)), float(np.max(flat_target)))
        ax_parity.plot(limits, limits, color="#2E3440", linewidth=1.0, linestyle="--")
    ax_parity.set_xlabel("target")
    ax_parity.set_ylabel("prediction")
    ax_parity.set_title("parity")
    ax_parity.grid(alpha=0.25)

    residual = (source[:, channels] - target[:, channels]).ravel()
    residual = residual[np.isfinite(residual)]
    if residual.size:
        # A perfect prediction - or a constant offset - has a residual with no
        # spread to bin, and 40 bins across a zero-width range is an error.
        # The padded range keeps the edges strictly increasing either way.
        low, high = float(np.min(residual)), float(np.max(residual))
        pad = max((high - low) * 0.05, abs(high) * 1e-9, 1e-12)
        ax_hist.hist(
            residual,
            bins=40 if high - low > pad else 1,
            range=(low - pad, high + pad),
            color=_PREDICT[0],
            alpha=0.8,
        )
        ax_hist.axvline(
            float(np.mean(residual)),
            color="#D9534F",
            linewidth=1.2,
            label=f"bias {np.mean(residual):+.3e}",
        )
        ax_hist.legend(fontsize=8)
    ax_hist.set_xlabel("error")
    ax_hist.set_ylabel("count")
    ax_hist.set_title("error distribution")
    ax_hist.grid(alpha=0.25)

    if step_rmse is not None:
        ax_steps = figure.add_subplot(grid[3, :])
        steps = np.arange(1, len(step_rmse) + 1)
        for position, channel in enumerate(channels):
            ax_steps.plot(
                steps,
                step_rmse[:, channel],
                color=_PREDICT[position % len(_PREDICT)],
                linewidth=1.2,
                marker="o",
                markersize=3,
                label=_label(source_name, total, channel),
            )
        ax_steps.set_xlabel("step")
        ax_steps.set_ylabel("RMSE")
        ax_steps.set_title("error at each step of the sequence")
        ax_steps.grid(alpha=0.25)
        if len(channels) > 1:
            ax_steps.legend(fontsize=8)

    figure.text(
        0.5,
        0.005,
        f"RMSE {signal.rmse():.4e}    MAE {signal.mae():.4e}    "
        f"max {signal.max_error():.4e}    FIT {signal.fit_pct():.2f}%    "
        f"R2 {signal.r2():.4f}",
        ha="center",
        fontsize=9,
        color="#4C566A",
    )
    _name_window(figure, f"nnodely validation - {signal.name}")
    return figure


def figure_history(history: dict[str, Any]):
    """Training and validation loss curves, if the history carries any."""
    import matplotlib.pyplot as plt

    curves = {
        key: np.asarray(values, dtype=float)
        for key, values in history.items()
        if np.ndim(values) == 1 and len(values)
    }
    if not curves:
        return None

    figure, axes = plt.subplots(figsize=(10.0, 5.0))
    for position, (key, values) in enumerate(sorted(curves.items())):
        axes.plot(
            np.arange(1, len(values) + 1),
            values,
            label=key,
            linewidth=1.6 if key in ("loss", "val_loss") else 1.0,
            linestyle="--" if key.startswith("val_") else "-",
            color=_PREDICT[position % len(_PREDICT)],
        )
    positive = [v for values in curves.values() for v in values if v > 0]
    if positive and max(positive) / min(positive) > 50:
        axes.set_yscale("log")
    axes.set_xlabel("epoch")
    axes.set_ylabel("loss")
    axes.set_title("training history")
    axes.grid(alpha=0.25)
    axes.legend(fontsize=8)
    _name_window(figure, "nnodely validation - history")
    return figure


def _name_window(figure, title: str) -> None:
    manager = getattr(figure.canvas, "manager", None)
    if manager is not None and hasattr(manager, "set_window_title"):
        manager.set_window_title(title)


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)


def render(
    result: ValidationResult,
    *,
    out_dir: str | os.PathLike | None = None,
    show: bool = False,
    history: dict[str, Any] | None = None,
) -> list[str]:
    """Draw every figure, save it when asked, and return the paths written."""
    import matplotlib.pyplot as plt

    # A file per minimizer is named by its index and its name: the index
    # keeps apart names alike once slugged - "a b" and "a_b", "Fit" and "fit"
    # on a file system blind to case - and keeps them from the history's.
    figures: list[tuple[str, Any]] = [
        (f"{index}_{_slug(name)}", figure_signal(signal, result.minimizers[name]))
        for index, (name, signal) in enumerate(result.signals.items())
    ]
    if history:
        history_figure = figure_history(history)
        if history_figure is not None:
            figures.append(("history", history_figure))

    saved: list[str] = []
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        for name, figure in figures:
            path = os.path.join(str(out_dir), f"{_slug(result.model)}_{name}.png")
            figure.savefig(path, dpi=150, bbox_inches="tight")
            saved.append(path)

    if show:
        plt.show()
    else:
        # Nothing is going to close these, and Matplotlib warns once twenty
        # figures are open.
        for _, figure in figures:
            plt.close(figure)
    return saved
