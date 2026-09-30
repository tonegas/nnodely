"""A validation run: the built model evaluated on a dataset, and every
minimizer scored on what it predicted.

One minimizer produces one :class:`SignalScore`: the loss
it was trained on, plus the handful of indicators a mechanical
system-identification report is actually read for.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import keras
import numpy as np

from nnodely.core.minimizer import _reads_input, _seq_weights_for

#: Samples evaluated per forward pass during validation. Validation is defined
#: one sample at a time - nothing about the result depends on this number - so
#: it is a memory bound rather than a parameter worth exposing.
_INFERENCE_CHUNK = 256


def as_channels(values: np.ndarray) -> np.ndarray:
    """View a signal as ``(samples, channels)``.

    Everything past the sample axis - dimensions, sample windows, rollout
    steps - is one channel, because for scoring purposes each is just another
    number that had to come out right.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 0:
        raise ValueError("A signal must have at least a sample axis.")
    return values.reshape(len(values), -1)


def loss_name(loss_fn: Any) -> str:
    for attribute in ("name", "__name__"):
        name = getattr(loss_fn, attribute, None)
        if isinstance(name, str) and name:
            return name
    return type(loss_fn).__name__ if loss_fn is not None else "mse"


def evaluate_loss(
    loss_fn: Any,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    seq_weights: np.ndarray | None = None,
) -> float:
    """The minimizer's own loss, so validation is scored the way it was trained.

    With ``seq_weights`` the signals keep their layout, whose last axis is the
    one the weights are laid along.
    """
    if loss_fn is None:
        return float(np.mean((y_true - y_pred) ** 2))
    try:
        import keras

        y_true = keras.ops.convert_to_tensor(np.asarray(y_true, dtype=np.float32))  # type: ignore
        y_pred = keras.ops.convert_to_tensor(np.asarray(y_pred, dtype=np.float32))  # type: ignore
        if seq_weights is not None:
            from nnodely.utils.utils import _weighted_sequence_loss

            value = _weighted_sequence_loss(loss_fn, y_true, y_pred, seq_weights)
        else:
            value = loss_fn(y_true, y_pred)
        return float(np.mean(keras.ops.convert_to_numpy(value)))  # type: ignore
    except Exception:
        # A loss that cannot be replayed outside training should not cost the
        # caller the rest of the report.
        return float("nan")


@dataclass
class SignalScore:
    """What one minimizer scored, and the two signals it scored it from."""

    name: str
    source: str
    target: str
    loss: str
    metrics: dict[str, float]
    y_true: np.ndarray
    y_pred: np.ndarray

    @property
    def channels(self) -> int:
        return self.y_true.shape[1]

    def channel_rmse(self) -> np.ndarray:
        return np.sqrt(np.mean((self.y_true - self.y_pred) ** 2, axis=0))


def score_signal(
    *,
    name: str,
    source: str,
    target: str,
    loss_fn: Any,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    seq_weights: np.ndarray | None = None,
) -> SignalScore:
    """Score one minimizer's prediction against its reference."""
    true_channels = as_channels(y_true)
    pred_channels = as_channels(y_pred)
    if true_channels.shape != pred_channels.shape:
        raise ValueError(
            f"Minimizer {name!r} compares {source!r} of shape {y_pred.shape} with "
            f"{target!r} of shape {y_true.shape}. They must match."
        )

    error = pred_channels - true_channels
    finite = np.isfinite(error)
    spread = float(np.std(true_channels))

    # A target that never moves has no scale to normalize against, so the
    # relative indicators are undefined rather than perfect or zero - saying
    # 100% fit on a constant signal would be a lie.
    if spread <= np.finfo(np.float32).eps or not finite.all():
        fit = r2 = correlation = nrmse = float("nan")
    else:
        residual = float(np.linalg.norm(error))
        reference = float(np.linalg.norm(true_channels - np.mean(true_channels)))
        fit = 100.0 * (1.0 - residual / reference)
        r2 = 1.0 - residual**2 / reference**2
        correlation = float(
            np.corrcoef(true_channels.ravel(), pred_channels.ravel())[0, 1]
        )
        span = float(np.ptp(true_channels))
        nrmse = (
            100.0 * float(np.sqrt(np.mean(error**2))) / span if span else float("nan")
        )

    metrics = {
        "loss": (
            evaluate_loss(loss_fn, true_channels, pred_channels)
            if seq_weights is None
            else evaluate_loss(loss_fn, y_true, y_pred, seq_weights)
        ),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "max_error": float(np.max(np.abs(error))) if error.size else float("nan"),
        "bias": float(np.mean(error)),
        "std_error": float(np.std(error)),
        "nrmse_pct": nrmse,
        "fit_pct": fit,
        "r2": r2,
        "correlation": correlation,
        "non_finite": int(error.size - int(finite.sum())),
    }
    return SignalScore(
        name=name,
        source=source,
        target=target,
        loss=loss_name(loss_fn),
        metrics=metrics,
        y_true=true_channels,
        y_pred=pred_channels,
    )


#: Label, metric key and format for every reported row, in reading order.
_ROWS: tuple[tuple[str, str, str], ...] = (
    ("RMSE", "rmse", "{:.6e}"),
    ("MAE", "mae", "{:.6e}"),
    ("Max error", "max_error", "{:.6e}"),
    ("Bias (mean error)", "bias", "{:+.6e}"),
    ("Error std", "std_error", "{:.6e}"),
    ("NRMSE (% of range)", "nrmse_pct", "{:.3f} %"),
    ("FIT (% variance)", "fit_pct", "{:.3f} %"),
    ("R2", "r2", "{:.6f}"),
    ("Correlation", "correlation", "{:.6f}"),
)


def _cell(value: float, spec: str) -> str:
    return "n/a" if value != value else spec.format(value)


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
                f"{_cell(score.metrics['loss'], '{:.6e}')}"
            )
            for text, key, spec in _ROWS:
                lines.append(f"{'  ' + text:<{label}}{_cell(score.metrics[key], spec)}")
            if score.metrics["non_finite"]:
                lines.append(
                    f"{'  Non-finite values':<{label}}{score.metrics['non_finite']}"
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


def _predict(model, x_data: dict, n_samples: int) -> dict[str, np.ndarray]:
    """Run the graph over the whole dataset with the training path disabled.

    ``training=False`` is the backend-independent switch: it is what tells
    every Keras layer to take its inference branch, so no per-backend module
    mode has to be toggled here.
    """
    if model.model is None:
        raise ValueError("Model is not built. Call build() before validate().")
    input_names = [node.name for node in model.train_inputs]
    missing = [name for name in input_names if name not in x_data]
    if missing:
        raise ValueError(f"Validation data is missing model inputs: {missing}.")

    chunks: dict[str, list[np.ndarray]] = {}
    for start in range(0, n_samples, _INFERENCE_CHUNK):
        stop = min(start + _INFERENCE_CHUNK, n_samples)
        batch = {name: x_data[name][start:stop, ...] for name in input_names}
        for name, value in model.model(batch, training=False).items():
            value = keras.ops.convert_to_numpy(value)
            chunks.setdefault(name, []).append(value)  # type: ignore

    return {name: np.concatenate(values, axis=0) for name, values in chunks.items()}


def _minimizer_key(model, node) -> str | None:
    """Name under which the forward pass returns a minimizer's side."""
    return model._minimizer_outputs.get(node, node.name)


def _broadcast_constant(values, source_values: np.ndarray) -> np.ndarray:
    """A target built from constants, laid out like its source's predictions.

    The numpy twin of what training does: append the axes the target
    omits, then broadcast - so a dim-2 Constant lines up with the dim
    axis, not the time axis.
    """
    values = np.asarray(values, dtype=np.float32)
    while values.ndim < source_values.ndim:
        values = values[..., np.newaxis]
    return np.broadcast_to(values, source_values.shape)


def _dataset_name(node, x_data: dict) -> str | None:
    """Name of the dataset column a node ultimately reads, if any."""
    if node.name in x_data:
        return node.name
    preds = getattr(node, "preds", [])
    if len(preds) == 1:
        return _dataset_name(preds[0], x_data)
    return None


def _validation_target(
    model, minimizer: dict, x_data: dict, predictions: dict, n_samples: int
) -> tuple[np.ndarray, str]:
    """Resolve the reference signal a minimizer is scored against.

    The evaluated target stream wins over the dataset column it reads: a
    target declared as ``y.sw(2)`` on an input that carries a wider window
    elsewhere in the graph is two steps long, not as wide as the column.
    The column still names the signal, which is what a reader recognizes.
    """
    source_values = predictions[_minimizer_key(model, minimizer["source"])]
    target = minimizer["target"]
    label = _dataset_name(target, x_data) or target.name

    key = _minimizer_key(model, target)
    if key in predictions:
        values = np.asarray(predictions[key][:n_samples])
        if not _reads_input(target):
            return _broadcast_constant(values, source_values), label
        return values, label

    if label in x_data:
        return np.asarray(x_data[label][:n_samples]), label

    value = getattr(target, "value_numpy", None)
    if value is None:
        raise ValueError(
            f"Validation target {target.name!r} of minimizer "
            f"{minimizer['name']!r} is neither in the dataset nor a constant."
        )
    # The value itself, without the batch axis of the streams.
    values = np.asarray(value)[np.newaxis, ...]
    return _broadcast_constant(values, source_values), target.name


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

    x_data = {name: np.asarray(values) for name, values in val_data.as_dict().items()}
    predictions = _predict(model, x_data, n_samples)

    signals = {}
    for minimizer in model.minimizers:
        name = minimizer["name"]
        source = minimizer["source"]
        source_key = _minimizer_key(model, source)
        if source_key not in predictions:
            raise ValueError(
                f"Minimizer {name!r} sources {source.name!r}, which the built "
                "model does not expose as an output."
            )
        y_true, target_name = _validation_target(
            model, minimizer, x_data, predictions, n_samples
        )
        signals[name] = score_signal(
            name=name,
            source=source.name,
            target=target_name,
            loss_fn=minimizer["loss"],
            seq_weights=(
                None
                if minimizer.get("seq_weights") is None
                else _seq_weights_for(minimizer["seq_weights"], y_true.shape[-1], name)
            ),
            y_true=y_true,
            y_pred=np.asarray(predictions[source_key][:n_samples]),
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
