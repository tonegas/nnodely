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

from nnodely.core.dag import Table
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

    ``error`` is laid out ``(samples, *seq, tw, *dim)``, ``real`` marks its real
    points ``(samples, *tw)``.
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
    r"""The score of one objective in a validation: its loss, the indicators
    of a system-identification report, and the two signals compared.

    :meth:`~nnodely.Modely.validate` makes one per objective, and returns them
    in a ``ValidationResult``. You do not create one yourself. Every
    indicator is computed for each channel, a feature of the compared signal,
    over the real points: a point is one sample at one step of the window or
    of the rollout, and the steps padded onto shorter simulations are left
    out.

    Attributes
    ----------
    name : str
        The name of the objective.
    source : str
        The name of the prediction stream.
    target : str
        The name of the target stream.
    loss : str
        The name of the loss function, such as ``"mean_squared_error"``.
    metrics : dict of str to float
        The loss and the indicators of the whole signal, under the keys:

        - ``"loss"``: the loss as training computes it, the gain not
          applied. It equals the ``"val_<objective>_loss"`` that
          :meth:`~nnodely.Modely.train` logs for the same data;
        - ``"rmse"``, ``"mae"``, ``"max_error"``: the root mean square, the
          mean and the peak of the absolute error, in the units of the
          signal;
        - ``"bias"`` and ``"std_error"``: the mean and the standard deviation
          of the error;
        - ``"nrmse_pct"``: the RMSE in percent of the range of the target;
        - ``"fit_pct"``: the FIT of system identification, in percent;
        - ``"r2"``: the coefficient of determination;
        - ``"vaf_pct"``: the variance accounted for, in percent;
        - ``"correlation"``: Pearson's correlation of prediction and target;
        - ``"non_finite"``: the number of NaN or infinite errors, an int.

        The indicators in the units of the signal pool every channel. The
        relative ones, ``nrmse_pct`` to ``correlation``, are the mean of the
        channels' values.
    channel_metrics : dict of str to numpy.ndarray
        Every indicator of ``metrics`` but ``loss`` and ``non_finite``, as an
        array of one value per channel.
    y_true : numpy.ndarray
        The target, laid out ``(points, channels)``, the padded points left
        out.
    y_pred : numpy.ndarray
        The prediction, laid out as ``y_true``.
    step_rmse : numpy.ndarray or None
        For a signal with a sequence, such as a rollout, the RMSE of each
        channel at each step of the outermost sequence axis,
        ``(steps, channels)``. A step no simulation reaches is NaN. ``None``
        for a signal without a sequence.
    padded : int
        The padded points left out.

    See Also
    --------
    nnodely.Modely.validate : Score every objective of a model.
    nnodely.Modely.minimize : Register an objective.

    Notes
    -----
    With the error :math:`e = \hat y - y` over the :math:`N` real points of a
    channel, :math:`\bar e` its mean and :math:`\bar y` the mean of the
    target:

    .. math::

        \mathrm{RMSE} = \sqrt{\tfrac{1}{N} \textstyle\sum e^2}, \quad
        \mathrm{MAE} = \tfrac{1}{N} \textstyle\sum |e|, \quad
        \mathrm{bias} = \bar e, \quad
        \mathrm{std} = \sqrt{\tfrac{1}{N} \textstyle\sum (e - \bar e)^2}

    .. math::

        \mathrm{NRMSE} = 100 \, \frac{\mathrm{RMSE}}{\max y - \min y}, \quad
        \mathrm{FIT} = 100 \left(1 - \frac{\lVert e \rVert}
        {\lVert y - \bar y \rVert}\right)

    .. math::

        R^2 = 1 - \frac{\sum e^2}{\sum (y - \bar y)^2}, \quad
        \mathrm{VAF} = 100 \left(1 - \frac{\operatorname{var} e}
        {\operatorname{var} y}\right)

    So :math:`\mathrm{RMSE}^2 = \mathrm{bias}^2 + \mathrm{std}^2`, and VAF is
    :math:`R^2` that forgives a constant offset. A FIT of 100 % is a perfect
    prediction, 0 % no better than the mean of the target.

    A channel whose target never moves has no spread to measure against: its
    relative indicators are NaN, and left out of the mean of the channels.
    The correlation is 0 for a prediction that never moves. A NaN or
    infinite error makes every indicator of its channel and of the whole
    signal NaN, and is counted in ``non_finite``.

    Basic Examples
    --------------
    A prediction 1 above the target at every point:

    >>> from nnodely import DataLoader, Input, ModelyLoop, Modely, Output, Parameter
    >>> x = Input("x", tw=1)
    >>> y = Output("y", x + Parameter("b", value=[1.0]))
    >>> model = Modely("offset", inputs=[x], outputs=[y])
    >>> model = model.minimize("fit", y, Input("y_meas", tw=1)).build()
    >>> data = DataLoader(model, source={"x": [0.0, 2.0, 4.0, 6.0],
    ...                                  "y_meas": [0.0, 2.0, 4.0, 6.0]})
    >>> score = model.validate(data, verbose=False)["fit"]
    >>> score.name, score.source, score.target, score.loss
    ('fit', 'y', 'y_meas', 'mean_squared_error')
    >>> {key: score.metrics[key] for key in ("rmse", "bias", "std_error", "r2", "vaf_pct")}
    {'rmse': 1.0, 'bias': 1.0, 'std_error': 0.0, 'r2': 0.8, 'vaf_pct': 100.0}
    >>> score.points, score.padded, score.y_pred.ravel().tolist()
    (4, 0, [1.0, 3.0, 5.0, 7.0])

    Scope
    -----
    A signal of 2 channels, each scored on its own:

    >>> v = Input("v", dim=2, tw=1)
    >>> w = Output("w", v + Parameter("c", value=[1.0, 0.5]))
    >>> pair = Modely("pair", inputs=[v], outputs=[w])
    >>> pair = pair.minimize("fit", w, Input("w_meas", dim=2, tw=1)).build()
    >>> values = [[0.0, 0.0], [2.0, 4.0], [4.0, 8.0]]
    >>> score = pair.validate(DataLoader(pair, source={"v": values, "w_meas": values}),
    ...                       verbose=False)["fit"]
    >>> score.channels, score.channel_metrics["bias"], score.metrics["bias"]
    (2, array([1. , 0.5]), 0.75)

    A rollout has the RMSE of every step:

    >>> rate, total = Input("rate", tw=1), Input("total", tw=1)
    >>> total_out = Output("total", total.feedback(total + rate))
    >>> counter = ModelyLoop("counter", inputs=[rate], outputs=[total_out])
    >>> counter = counter.minimize("track", "total", Input("measured", tw=1)).build()
    >>> steps = DataLoader(counter, source={"rate": [1.0, 1.0, 1.0],
    ...                                     "measured": [1.0, 2.0, 4.0]}, seq_length=3)
    >>> counter.validate(steps, verbose=False)["track"].step_rmse
    array([[0.],
           [0.],
           [1.]])

    Possible Issues
    ---------------
    A target that never moves leaves the relative indicators undefined:

    >>> flat = Modely("flat", inputs=[x], outputs=[y]).minimize("level", y, 5.0).build()
    >>> metrics = flat.validate(DataLoader(flat, source={"x": [0.0, 2.0]}), verbose=False)[0].metrics
    >>> metrics["rmse"], metrics["r2"]
    (3.1622776601683795, nan)

    .. versionadded:: 2.0
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
        """The number of channels, the features of the compared signal."""
        return self.y_true.shape[1]

    @property
    def points(self) -> int:
        """The number of real points scored, the padded ones left out."""
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

    Both signals are laid out ``(samples, *tw, *dim)``, their last
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


class ValidationResult(Table):
    """The scores of a validation, one per objective, as a list printed as
    a report.

    :meth:`~nnodely.Modely.validate` returns it. You do not create one
    yourself. It is a list of ``SignalScore``, one per objective, in the
    order of :attr:`~nnodely.Modely.minimizers`. Each holds the loss of its
    objective and the indicators of a system-identification report, in its
    ``metrics``. An objective is called a minimizer in the method names of a
    model and in the report.

    Pick a score by the name of its objective, ``result["fit"]``, or by its
    index, ``result[0]``. The result is a snapshot: renaming or removing an
    objective later leaves it as it is. ``print(result)`` prints
    :meth:`summary`.

    Attributes
    ----------
    model : str
        The name of the validated model.
    samples : int
        The samples of the data.
    figures : list of str
        The paths of the PNG files written, empty without ``out_dir``.
    history : dict or None
        The training history drawn, as given to ``validate()``.

    See Also
    --------
    nnodely.Modely.validate : Score every objective of a model.
    nnodely.Modely.train : Train a model, and log the validation losses.

    Basic Examples
    --------------
    >>> from nnodely import DataLoader, Input, Modely, Output, Parameter
    >>> x = Input("x", tw=1)
    >>> y = Output("y", x + Parameter("b", value=[1.0]))
    >>> y_meas = Input("y_meas", tw=1)
    >>> model = Modely("offset", inputs=[x], outputs=[y])
    >>> model = model.minimize("fit", y, y_meas).minimize("level", y, 4.0).build()
    >>> data = DataLoader(model, source={"x": [0.0, 2.0, 4.0, 6.0],
    ...                                  "y_meas": [0.0, 2.0, 4.0, 6.0]})
    >>> result = model.validate(data, verbose=False)
    >>> result.model, result.samples, len(result), result.figures
    ('offset', 4, 2, [])
    >>> result["fit"].metrics["rmse"], result[1].metrics["rmse"]
    (1.0, 2.23606797749979)

    Scope
    -----
    The metrics of every objective:

    >>> metrics = {score.name: score.metrics for score in result}
    >>> sorted(metrics), sorted(metrics["fit"])
    (['fit', 'level'], ['bias', 'correlation', 'fit_pct', 'loss', 'mae', 'max_error',
     'non_finite', 'nrmse_pct', 'r2', 'rmse', 'std_error', 'vaf_pct'])

    Possible Issues
    ---------------
    A name that is no objective of the result raises ``KeyError``:

    >>> result["track"]
    Traceback (most recent call last):
        ...
    nnodely.core.dag.NameNotFound: ValidationResult: no minimizer named 'track'; the names are ['fit', 'level'].

    .. versionadded:: 2.0
    """

    _kind, _attr, _title = "minimizer", "result", "ValidationResult"

    def __init__(
        self,
        scores: list[SignalScore],
        *,
        model: str,
        samples: int,
        figures: list[str] | None = None,
        history: dict[str, Any] | None = None,
    ):
        super().__init__(scores)
        self.model = model
        self.samples = samples
        self.figures = figures or []
        self.history = history

    def summary(self) -> str:
        """Return the report of the scores as text, 80 characters wide.

        For each objective, the report gives its loss, the points scored and
        every indicator. A signal of 2 or 3 channels has a column per
        channel. A rollout adds the RMSE of its first and its last step.

        Returns
        -------
        str
            The report.

        See Also
        --------
        nnodely.Modely.validate : Prints it, unless ``verbose=False``.

        Basic Examples
        --------------
        >>> from nnodely import DataLoader, Input, Modely, Output, Parameter
        >>> x = Input("x", tw=1)
        >>> y = Output("y", x + Parameter("b", value=[1.0]))
        >>> model = Modely("offset", inputs=[x], outputs=[y])
        >>> model = model.minimize("fit", y, Input("y_meas", tw=1)).build()
        >>> data = DataLoader(model, source={"x": [0.0, 2.0, 4.0, 6.0],
        ...                                  "y_meas": [0.0, 2.0, 4.0, 6.0]})
        >>> print(model.validate(data, verbose=False).summary())
        ============================== nnodely Validation ==============================
        Model:                        offset
        Samples:                      4
        Minimizers:                   1
        --------------------------------------------------------------------------------
        fit   (y vs y_meas)
          Loss (mean_squared_error)     1.0000e+00
          Points scored                          4
          RMSE                          1.0000e+00
          MAE                           1.0000e+00
          Max error                     1.0000e+00
          Bias (mean error)            +1.0000e+00
          Error std                     0.0000e+00
          NRMSE (% of range)              16.667 %
          FIT                             55.279 %
          R2                              0.800000
          VAF                            100.000 %
          Correlation                     1.000000
        ================================================================================

        .. versionadded:: 2.0
        """
        width = 80
        label = 30
        lines = [" nnodely Validation ".center(width, "=")]
        lines.append(f"{'Model:':<{label}}{self.model}")
        lines.append(f"{'Samples:':<{label}}{self.samples}")
        lines.append(f"{'Minimizers:':<{label}}{len(self)}")

        for score in self:
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


def _evaluate(model, terms, x_data: dict, mask, n_samples: int) -> list:
    """What each minimizer compares over the dataset, as training sees it.

    Returns, for each term in the order of ``terms``, its loss over the whole dataset, its source, its
    target aligned with it, and the mask of their real points - None when no
    point was padded.

    ``training=False`` is the backend-independent switch: it is what tells
    every Keras layer to take its inference branch, so no per-backend module
    mode has to be toggled here. The loss of a chunk is a mean over its
    samples, so weighing it by their number gives the loss of the dataset.
    """
    # Every input is there: Modely._training_arrays checked it.
    input_names = [node.name for node in model.train_inputs]

    losses = [0.0] * len(terms)
    parts: list[tuple[list, list, list]] = [([], [], []) for _ in terms]
    for start in range(0, n_samples, _INFERENCE_CHUNK):
        stop = min(start + _INFERENCE_CHUNK, n_samples)
        batch = {name: x_data[name][start:stop, ...] for name in input_names}
        y = None if mask is None else mask[start:stop]
        outputs = model.model(batch, training=False)
        for index, term in enumerate(terms):
            source = outputs[term.source]
            target = MinimizerModel._align_target(term, outputs[term.target], source)
            value = keras.ops.convert_to_numpy(
                MinimizerModel._term_loss(term, source, target, y)
            )
            losses[index] += float(value) * (stop - start)  # type: ignore[arg-type]

            sources, targets, valids = parts[index]
            sources.append(keras.ops.convert_to_numpy(source))
            targets.append(keras.ops.convert_to_numpy(target))
            valid = _rollout_mask(term, target, y)
            if valid is not None:
                valids.append(keras.ops.convert_to_numpy(valid))

    return [
        (
            loss / n_samples,
            np.concatenate(sources),
            np.concatenate(targets),
            np.concatenate(valids) if valids else None,
        )
        for loss, (sources, targets, valids) in zip(losses, parts)
    ]


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
    if not model._minimizers:
        raise ValueError("No minimizers defined. Cannot infer validation targets.")

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    n_samples = len(val_data)
    if n_samples == 0:
        raise ValueError("Validation dataset is empty.")

    terms = _resolve_minimizer_terms(model)
    x_data, mask = model._training_arrays(val_data)
    compared = _evaluate(model, terms, x_data, mask, n_samples)

    result = ValidationResult(
        [
            score_signal(
                name=term.name,
                source=minimizer["source"].name,
                target=minimizer["target"].name,
                y_true=y_true,
                y_pred=y_pred,
                valid=valid,
                dim_rank=minimizer["source"].shape.dim_rank,
                loss=loss_name(minimizer["loss"]),
                loss_value=loss_value,
            )
            for minimizer, term, (loss_value, y_pred, y_true, valid) in zip(
                model._minimizers, terms, compared
            )
        ],
        model=model.name,
        samples=n_samples,
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
