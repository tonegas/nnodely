"""Scoring for a validation run.

One minimizer produces one :class:`SignalScore`: the loss
it was trained on, plus the handful of indicators a mechanical
system-identification report is actually read for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


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


def evaluate_loss(loss_fn: Any, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """The minimizer's own loss, so validation is scored the way it was trained."""
    if loss_fn is None:
        return float(np.mean((y_true - y_pred) ** 2))
    try:
        import keras

        value = loss_fn(
            keras.ops.convert_to_tensor(y_true.astype(np.float32)),
            keras.ops.convert_to_tensor(y_pred.astype(np.float32)),
        )
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
        "loss": evaluate_loss(loss_fn, true_channels, pred_channels),
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
