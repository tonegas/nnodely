"""A validation run: the built model evaluated on a dataset, and every
minimizer scored on what it predicted.

Validation sees what training sees. The forward pass is the one training runs
on its validation data, each target is aligned with its source as training
aligns it, the padding mask of simulations of different lengths is the same,
and the loss is computed by the code that trains the minimizer.

One minimizer produces one :class:`SignalScore`: that loss, plus the
indicators a mechanical system-identification report is read for. They are
computed for each channel - each feature of the compared signal - on its own
scale, over the real points alone: a padded rollout step is in no indicator.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import keras
import numpy as np

from nnodely.core.minimizer import (
    MinimizerModel,
    _resolve_minimizer_terms,
    _rollout_mask,
)

#: Samples evaluated per forward pass during validation. Validation is defined
#: one sample at a time - nothing about the result depends on this number - so
#: it is a memory bound rather than a parameter worth exposing.
_INFERENCE_CHUNK = 256

#: Indicators in the units of the signal: a signal of several channels pools
#: every channel into them.
_ABSOLUTE = ("rmse", "mae", "max_error", "bias", "std_error")
#: Indicators relative to the target's own spread: a signal of several
#: channels reports the mean of its channels' values.
_RELATIVE = ("nrmse_pct", "fit_pct", "r2", "vaf_pct", "correlation")


def loss_name(loss_fn: Any) -> str:
    for attribute in ("name", "__name__"):
        name = getattr(loss_fn, attribute, None)
        if isinstance(name, str) and name:
            return name
    return type(loss_fn).__name__ if loss_fn is not None else "mse"


def _constant(values: np.ndarray) -> bool:
    """True for values that do not move, up to float32 rounding."""
    return bool(np.ptp(values) <= np.finfo(np.float32).eps * np.max(np.abs(values)))


def _indicators(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Every indicator of one channel, from its real points.

    With ``e = pred - true``: RMSE, MAE and max error are the root mean
    square, mean and peak of ``|e|``; bias and error std are the mean and
    standard deviation of ``e``, so ``RMSE^2 = bias^2 + std^2``. Against the
    target's spread: NRMSE is RMSE as a percentage of its range, FIT is
    ``100 (1 - ||e|| / ||true - mean(true)||)``, R2 is
    ``1 - sum(e^2) / sum((true - mean(true))^2)``, VAF is
    ``100 (1 - var(e) / var(true))`` - R2 that forgives a constant offset -
    and correlation is Pearson's - 0 for a prediction that never moves, which
    explains none of a target that does.
    """
    nan = float("nan")
    error = pred - true
    if error.size == 0 or not np.isfinite(error).all():
        return dict.fromkeys(_ABSOLUTE + _RELATIVE, nan)

    rmse = float(np.sqrt(np.mean(error**2)))
    values = {
        "rmse": rmse,
        "mae": float(np.mean(np.abs(error))),
        "max_error": float(np.max(np.abs(error))),
        "bias": float(np.mean(error)),
        "std_error": float(np.std(error)),
    }
    # A target that never moves has no scale to normalize against, so the
    # relative indicators are undefined rather than perfect or zero - saying
    # 100% fit on a constant signal would be a lie.
    if _constant(true):
        return {**values, **dict.fromkeys(_RELATIVE, nan)}

    residual = float(np.sum(error**2))
    spread = float(np.sum((true - np.mean(true)) ** 2))
    values["nrmse_pct"] = 100.0 * rmse / float(np.ptp(true))
    values["fit_pct"] = 100.0 * (1.0 - np.sqrt(residual / spread))
    values["r2"] = 1.0 - residual / spread
    values["vaf_pct"] = 100.0 * (1.0 - float(np.var(error)) / float(np.var(true)))
    values["correlation"] = (
        0.0 if _constant(pred) else float(np.corrcoef(true, pred)[0, 1])
    )
    return values


def _step_rmse(error: np.ndarray, real: np.ndarray, dim_rank: int):
    """RMSE of every channel at every step of the outermost sequence axis,
    over the real points of that step; None for a signal without a sequence.

    ``error`` is laid out ``(samples, *seq, time, *dim)``, ``real`` marks its real
    points ``(samples, *time)``.
    """
    if error.ndim - 1 - dim_rank < 2:
        return None
    samples, steps = error.shape[:2]
    error = error.reshape(samples, steps, -1, int(np.prod(error.shape[-dim_rank:])))
    real = real.reshape(samples, steps, -1, 1)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        squared = np.where(real, error**2, 0.0).sum(axis=(0, 2))
        # A step no simulation reaches has no point, and no RMSE.
        return np.sqrt(squared / real.sum(axis=(0, 2)))


@dataclass
class SignalScore:
    """What one minimizer scored, and the two signals it scored it from.

    ``metrics`` holds the loss and every indicator of the whole signal,
    ``channel_metrics`` each indicator of each channel. ``y_true`` and
    ``y_pred`` are laid out ``(points, channels)``: a point is one sample at
    one time step, a channel one feature, and the padded points are left out.
    ``step_rmse``, for a signal with a sequence - a rollout - is the RMSE
    of each channel at each of its steps, ``(steps, channels)``.
    """

    name: str
    source: str
    target: str
    loss: str
    metrics: dict[str, float]
    channel_metrics: dict[str, np.ndarray]
    y_true: np.ndarray
    y_pred: np.ndarray
    step_rmse: np.ndarray | None = None
    padded: int = 0

    @property
    def channels(self) -> int:
        return self.y_true.shape[1]

    @property
    def points(self) -> int:
        return self.y_true.shape[0]


def score_signal(
    *,
    name: str,
    source: str,
    target: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    valid: np.ndarray | None = None,
    dim_rank: int = 1,
    loss: str = "mse",
    loss_value: float = float("nan"),
) -> SignalScore:
    """Score one minimizer's prediction against its reference.

    Both signals are laid out ``(samples, *time, *dim)``, their last
    ``dim_rank`` axes being the channels. ``valid``, broadcastable to them,
    marks the real points: a padded one is left out of every indicator.
    ``loss_value`` is the minimizer's loss, computed as training computes it.

    Each channel is scored on its own. For the whole signal, the absolute
    indicators pool every channel and the relative ones are the mean of the
    channels' values, so a channel in metres and one in millimetres are each
    scored against their own spread.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Minimizer {name!r} compares {source!r} of shape {y_pred.shape} with "
            f"{target!r} of shape {y_true.shape}. They must match."
        )

    observed = y_true.shape[: y_true.ndim - dim_rank]
    channels = int(np.prod(y_true.shape[y_true.ndim - dim_rank :], dtype=int))
    if valid is None:
        real = np.ones(observed, dtype=bool)
    else:
        # A point is real when every channel of it is.
        real = np.broadcast_to(valid, y_true.shape).reshape(*observed, channels)
        real = np.asarray(real.all(axis=-1))

    keep = real.reshape(-1)
    true = y_true.reshape(-1, channels)[keep]
    pred = y_pred.reshape(-1, channels)[keep]
    # A diverged prediction is counted below, not warned about on the way.
    with np.errstate(invalid="ignore", over="ignore"):
        error = y_pred - y_true
        each = [_indicators(true[:, c], pred[:, c]) for c in range(channels)]
        pooled = _indicators(true.ravel(), pred.ravel())
    non_finite = int(
        np.size(true) - np.isfinite(error.reshape(-1, channels)[keep]).sum()
    )

    channel_metrics = {
        key: np.array([values[key] for values in each]) for key in _ABSOLUTE + _RELATIVE
    }
    metrics = {"loss": float(loss_value)}
    metrics.update({key: pooled[key] for key in _ABSOLUTE})
    for key in _RELATIVE:
        # A non-finite value spoils the signal it is in: averaging the other
        # channels would hide it.
        defined = channel_metrics[key][np.isfinite(channel_metrics[key])]
        metrics[key] = (
            float(np.mean(defined)) if defined.size and not non_finite else float("nan")
        )
    metrics["non_finite"] = non_finite

    return SignalScore(
        name=name,
        source=source,
        target=target,
        loss=loss,
        metrics=metrics,
        channel_metrics=channel_metrics,
        y_true=true,
        y_pred=pred,
        step_rmse=_step_rmse(error, real, dim_rank),
        padded=int(keep.size - keep.sum()),
    )


#: Label, metric key and format for every reported row, in reading order.
_ROWS: tuple[tuple[str, str, str], ...] = (
    ("RMSE", "rmse", "{:.4e}"),
    ("MAE", "mae", "{:.4e}"),
    ("Max error", "max_error", "{:.4e}"),
    ("Bias (mean error)", "bias", "{:+.4e}"),
    ("Error std", "std_error", "{:.4e}"),
    ("NRMSE (% of range)", "nrmse_pct", "{:.3f} %"),
    ("FIT", "fit_pct", "{:.3f} %"),
    ("R2", "r2", "{:.6f}"),
    ("VAF", "vaf_pct", "{:.3f} %"),
    ("Correlation", "correlation", "{:.6f}"),
)

#: Width of one value column of the summary, and how many channels get a
#: column of their own next to the whole signal's within 80 characters.
_COLUMN = 12
_CHANNEL_COLUMNS = 3


def _cell(value: float, spec: str) -> str:
    return f"{'n/a' if value != value else spec.format(value):>{_COLUMN}}"


@dataclass
class ValidationResult:
    """Everything one call to :meth:`Modely.validate` produced."""

    model: str
    samples: int
    signals: dict[str, SignalScore]
    figures: list[str]
    history: dict[str, Any] | None = None

    def __getitem__(self, name: str) -> SignalScore:
        return self.signals[name]

    def __iter__(self):
        return iter(self.signals)

    def __len__(self) -> int:
        return len(self.signals)

    def metrics(self) -> dict[str, dict[str, float]]:
        """Every score as plain numbers, for asserting on or logging."""
        return {name: dict(s.metrics) for name, s in self.signals.items()}

    def summary(self) -> str:
        width = 80
        label = 30
        lines = [" nnodely Validation ".center(width, "=")]
        lines.append(f"{'Model:':<{label}}{self.model}")
        lines.append(f"{'Samples:':<{label}}{self.samples}")
        lines.append(f"{'Minimizers:':<{label}}{len(self.signals)}")

        for score in self.signals.values():
            lines.append("-" * width)
            lines.append(f"{score.name}   ({score.source} vs {score.target})")
            lines.append(
                f"{'  Loss (' + score.loss + ')':<{label}}"
                f"{_cell(score.metrics['loss'], '{:.4e}')}"
            )
            points = f"{'  Points scored':<{label}}{score.points:>{_COLUMN}}"
            if score.padded:
                points += f"  ({score.padded} padded left out)"
            lines.append(points)

            columns = 1 < score.channels <= _CHANNEL_COLUMNS
            if columns:
                lines.append(
                    f"{'':<{label}}{'all':>{_COLUMN}}"
                    + "".join(f"{f'[{c}]':>{_COLUMN}}" for c in range(score.channels))
                )
            elif score.channels > 1:
                lines.append(
                    f"{'  Channels':<{label}}{score.channels:>{_COLUMN}}"
                    "  (each in channel_metrics)"
                )
            for text, key, spec in _ROWS:
                values = [score.metrics[key]]
                if columns:
                    values += list(score.channel_metrics[key])
                lines.append(
                    f"{'  ' + text:<{label}}"
                    + "".join(_cell(value, spec) for value in values)
                )
            if score.step_rmse is not None:
                # Every channel pooled, at the first and the last step.
                rollout = np.sqrt(np.mean(score.step_rmse**2, axis=1))
                lines.append(
                    f"{'  RMSE first / last step':<{label}}"
                    f"{_cell(rollout[0], '{:.4e}')}{_cell(rollout[-1], '{:.4e}')}"
                )
            if score.metrics["non_finite"]:
                lines.append(
                    f"{'  Non-finite values':<{label}}"
                    f"{score.metrics['non_finite']:>{_COLUMN}}"
                )

        if self.figures:
            lines.append("-" * width)
            for path in self.figures:
                lines.append(f"{'  Figure:':<{label}}{path}")
        lines.append("=" * width)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


# ----------------------------------------------------------------------
# The validation run
# ----------------------------------------------------------------------


def _evaluate(model, terms, x_data: dict, mask, n_samples: int) -> dict:
    """What each minimizer compares over the dataset, as training sees it.

    Returns, by minimizer, its loss over the whole dataset, its source, its
    target aligned with it, and the mask of their real points - None when no
    point was padded.

    ``training=False`` is the backend-independent switch: it is what tells
    every Keras layer to take its inference branch, so no per-backend module
    mode has to be toggled here. The loss of a chunk is a mean over its
    samples, so weighing it by their number gives the loss of the dataset.
    """
    input_names = [node.name for node in model.train_inputs]
    missing = [name for name in input_names if name not in x_data]
    if missing:
        raise ValueError(f"Validation data is missing model inputs: {missing}.")

    losses = {term.name: 0.0 for term in terms}
    parts: dict[str, tuple[list, list, list]] = {
        term.name: ([], [], []) for term in terms
    }
    for start in range(0, n_samples, _INFERENCE_CHUNK):
        stop = min(start + _INFERENCE_CHUNK, n_samples)
        batch = {name: x_data[name][start:stop, ...] for name in input_names}
        y = None if mask is None else mask[start:stop]
        outputs = model.model(batch, training=False)
        for term in terms:
            source = outputs[term.source]
            target = MinimizerModel._align_target(term, outputs[term.target], source)
            value = keras.ops.convert_to_numpy(
                MinimizerModel._term_loss(term, source, target, y)
            )
            losses[term.name] += float(value) * (stop - start)  # type: ignore[arg-type]

            sources, targets, valids = parts[term.name]
            sources.append(keras.ops.convert_to_numpy(source))
            targets.append(keras.ops.convert_to_numpy(target))
            valid = _rollout_mask(term, target, y)
            if valid is not None:
                valids.append(keras.ops.convert_to_numpy(valid))

    return {
        name: (
            losses[name] / n_samples,
            np.concatenate(sources),
            np.concatenate(targets),
            np.concatenate(valids) if valids else None,
        )
        for name, (sources, targets, valids) in parts.items()
    }


def _dataset_name(node, x_data: dict) -> str | None:
    """Name of the dataset column a node ultimately reads, if any."""
    if node.name in x_data:
        return node.name
    preds = getattr(node, "preds", [])
    if len(preds) == 1:
        return _dataset_name(preds[0], x_data)
    return None


def validate(
    model,
    val_data,
    out_dir: str | os.PathLike | None = None,
    show: bool = False,
    history: dict[str, Any] | None = None,
    verbose: bool = True,
) -> ValidationResult:
    """The implementation of :meth:`Modely.validate`."""
    if model.model is None:
        raise ValueError("Model is not built. Call build() before validate().")
    if not model.minimizers:
        raise ValueError("No minimizers defined. Cannot infer validation targets.")

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    n_samples = len(val_data)
    if n_samples == 0:
        raise ValueError("Validation dataset is empty.")

    terms = _resolve_minimizer_terms(model)
    x_data, mask = model._training_arrays(val_data)
    compared = _evaluate(model, terms, x_data, mask, n_samples)

    signals = {}
    for minimizer, term in zip(model.minimizers, terms):
        loss_value, y_pred, y_true, valid = compared[term.name]
        target = minimizer["target"]
        signals[term.name] = score_signal(
            name=term.name,
            source=minimizer["source"].name,
            target=_dataset_name(target, x_data) or target.name,
            y_true=y_true,
            y_pred=y_pred,
            valid=valid,
            dim_rank=minimizer["source"].shape.dim_rank,
            loss=loss_name(minimizer["loss"]),
            loss_value=loss_value,
        )

    result = ValidationResult(
        model=model.name,
        samples=n_samples,
        signals=signals,
        figures=[],
        history=history,
    )
    if out_dir is not None or show:
        # Drawing needs matplotlib, and draws a ValidationResult: imported here,
        # where it is used.
        from nnodely.utils import validation_plot

        result.figures = validation_plot.render(result, out_dir=out_dir, show=show)
    if verbose:
        print(result.summary())
    return result
