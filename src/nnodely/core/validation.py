"""A validation run: the built model evaluated on a dataset, and every
minimizer scored on what it compared.

Validation sees what training sees. The forward pass is the one training runs
on its validation data, each target is aligned with its source as training
aligns it, the padding mask of simulations of different lengths is the same,
and the loss is computed by the code that trains the minimizer.

One minimizer gives one :class:`Signal`: the two signals it compared, whole,
and the metrics a system-identification report is read for, as functions of
the axes to keep.
"""

from __future__ import annotations

import os

import keras
import numpy as np

from nnodely.core.minimizer import (
    MinimizerModel,
    _resolve_minimizer_terms,
    _rollout_mask,
)
from nnodely.core.results import Fields, ValidationResult, minimizer_fields
from nnodely.core.shape import Shape
from nnodely.core.stream import object_repr
from nnodely.utils.printers import _table

#: Samples evaluated per forward pass during validation. Validation is defined
#: one sample at a time - nothing about the result depends on this number - so
#: it is a memory bound rather than a parameter worth exposing.
_INFERENCE_CHUNK = 256

#: Every metric: what it measures, as the table of a Signal prints it, and
#: the format of its value.
_METRICS = {
    "rmse": ("root mean square error", "{:.4e}"),
    "mae": ("mean absolute error", "{:.4e}"),
    "max_error": ("largest absolute error", "{:.4e}"),
    "bias": ("mean error, source - target", "{:.4e}"),
    "std_error": ("standard deviation of the error", "{:.4e}"),
    "nrmse_pct": ("RMSE, % of the target's range", "{:.3f} %"),
    "fit_pct": ("FIT, 100 (1 - |e| / |target - mean|)", "{:.3f} %"),
    "r2": ("R2, 1 - SSE / SST", "{:.6f}"),
    "vaf_pct": ("variance accounted for, %", "{:.3f} %"),
    "correlation": ("Pearson correlation", "{:.6f}"),
}
#: The metrics relative to the spread of the target: measured in each
#: channel, on its own scale, then averaged over the channels.
_RELATIVE = ("nrmse_pct", "fit_pct", "r2", "vaf_pct", "correlation")


def _range(values: np.ndarray, real: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The range of ``values`` along the last axis, at the ``real`` points,
    and where they do not move, up to float32 rounding."""
    highest = np.where(real, values, -np.inf).max(axis=-1)
    lowest = np.where(real, values, np.inf).min(axis=-1)
    span = highest - lowest
    still = span <= np.finfo(np.float32).eps * np.maximum(abs(highest), abs(lowest))
    return span, still


def _indicators(true: np.ndarray, pred: np.ndarray) -> dict[str, np.ndarray]:
    """Every metric along the last axis of ``true`` and ``pred``, over the
    points where the target is not NaN: an array of the other axes each, and
    under ``"constant"`` where the target does not move.

    With ``e = pred - true``: RMSE, MAE and max error are the root mean
    square, mean and peak of ``|e|``; bias and error std are the mean and
    standard deviation of ``e``, so ``RMSE^2 = bias^2 + std^2``. Against the
    spread of the target: NRMSE is RMSE as a percentage of its range, FIT is
    ``100 (1 - ||e|| / ||true - mean(true)||)``, R2 is
    ``1 - sum(e^2) / sum((true - mean(true))^2)``, VAF is
    ``100 (1 - var(e) / var(true))`` - R2 that forgives a constant offset -
    and correlation is Pearson's, 0 for a prediction that never moves. A
    target that never moves has no spread: its relative metrics are NaN.
    A non-finite prediction makes every metric it is in NaN.
    """
    real = ~np.isnan(true)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        n = real.sum(axis=-1)
        error = np.where(real, pred - true, 0.0)
        bias = error.sum(axis=-1) / n
        mean_true = np.where(real, true, 0.0).sum(axis=-1) / n
        mean_pred = np.where(real, pred, 0.0).sum(axis=-1) / n
        spread_error = np.where(real, error - bias[..., None], 0.0)
        spread_true = np.where(real, true - mean_true[..., None], 0.0)
        spread_pred = np.where(real, pred - mean_pred[..., None], 0.0)
        sse = (error**2).sum(axis=-1)
        sst = (spread_true**2).sum(axis=-1)
        ssp = (spread_pred**2).sum(axis=-1)
        span, constant = _range(true, real)
        _, still = _range(pred, real)
        rmse = np.sqrt(sse / n)
        values = {
            "rmse": rmse,
            "mae": np.abs(error).sum(axis=-1) / n,
            "max_error": np.where(n > 0, np.abs(error).max(axis=-1), np.nan),
            "bias": bias,
            "std_error": np.sqrt((spread_error**2).sum(axis=-1) / n),
            "nrmse_pct": 100.0 * rmse / span,
            "fit_pct": 100.0 * (1.0 - np.sqrt(sse / sst)),
            "r2": 1.0 - sse / sst,
            "vaf_pct": 100.0 * (1.0 - (spread_error**2).sum(axis=-1) / sst),
            "correlation": np.where(
                still,
                0.0,
                (spread_true * spread_pred).sum(axis=-1) / np.sqrt(sst * ssp),
            ),
        }
    for key in _RELATIVE:
        values[key] = np.where(constant, np.nan, values[key])
    values["constant"] = constant
    return values


def _along(array: np.ndarray, axes: list[int]) -> np.ndarray:
    """``array`` with ``axes`` first, in their order, and every other axis
    flattened into the last."""
    moved = np.moveaxis(array, axes, range(len(axes)))
    return moved.reshape(*moved.shape[: len(axes)], -1)


class Signal:
    r"""What one minimizer compared over a dataset: its source and its
    target, whole, and their metrics.

    :meth:`~nnodely.Modely.validate` makes one per minimizer, in
    ``result.signals``. You do not create one yourself.

    Attributes
    ----------
    name : str
        The name of the minimizer.
    source, target : numpy.ndarray
        The two signals compared, laid out as the stream,
        ``(samples, *seq, tw, *dim)``. A point without a target is NaN in
        both.
    valid : numpy.ndarray of bool
        The points that have a target, ``(samples, *seq, tw)``.
    shape : Shape
        The shape of one sample.
    samples, scored, missing : int
        The samples, the points scored, and those without a target - the
        steps padded onto shorter rollouts, NaN in the data. A point is one
        sample at one step of the sequence and of the window.

    The metrics are functions: ``rmse``, ``mae``, ``max_error``, ``bias``,
    ``std_error``, ``nrmse_pct``, ``fit_pct``, ``r2``, ``vaf_pct`` and
    ``correlation``, and ``non_finite``. Without arguments they give a
    number, over the whole signal. ``seq=``, ``tw=`` and ``dim=`` keep those
    axes, and give an array, a value per position along them: ``True`` keeps
    every axis of the group, a tuple of indices keeps those positions along
    its first axis. The samples are always pooled.

    The metrics in the units of the signal pool every point kept together.
    Those relative to the spread of the target, ``nrmse_pct`` to
    ``correlation``, are measured in each channel, on its own scale, and
    averaged over the channels not kept.

    ``print`` shows the table of every metric over the whole signal.

    Notes
    -----
    With the error :math:`e = \hat y - y` and :math:`\bar y` the mean of the
    target:

    .. math::

        \mathrm{RMSE} = \sqrt{\tfrac{1}{N} \textstyle\sum e^2}, \quad
        \mathrm{FIT} = 100 \left(1 - \frac{\lVert e \rVert}
        {\lVert y - \bar y \rVert}\right), \quad
        R^2 = 1 - \frac{\sum e^2}{\sum (y - \bar y)^2}

    so ``rmse() ** 2`` is the mean squared error, the loss of a minimizer
    with ``loss="mse"``.

    Basic Examples
    --------------
    >>> from nnodely import DataLoader, Input, Modely, Output, Parameter
    >>> v = Input("v", dim=2, tw=1)
    >>> w = Output("w", v + Parameter("c", value=[1.0, 0.5]))
    >>> pair = Modely("pair", inputs=[v], outputs=[w])
    >>> pair = pair.minimize("fit", w, Input("w_meas", dim=2, tw=1)).build()
    >>> values = [[0.0, 0.0], [2.0, 4.0], [4.0, 8.0]]
    >>> data = DataLoader(pair, source={"v": values, "w_meas": values})
    >>> signal = pair.validate(data, verbose=False).signals["fit"]
    >>> signal
    Signal(name='fit', samples=3, shape=Shape(S=(), T=1, D=(2,)), scored=3, missing=0)
    >>> signal.bias(), signal.bias(dim=True)
    (0.75, array([1. , 0.5]))

    .. versionadded:: 2.0
    """

    def __init__(self, name: str, source, target, valid, shape: Shape):
        source = np.asarray(source, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        dims = shape.dim_axes
        # A point has a target when every channel of it has one.
        real = np.isfinite(target).all(axis=dims)
        if valid is not None:
            real &= np.broadcast_to(np.asarray(valid, bool), target.shape).all(
                axis=dims
            )
        hidden = ~real.reshape(*real.shape, *(1,) * len(dims))
        self.name = name
        self.shape = shape
        self.valid = real
        self.source = np.where(hidden, np.nan, source)
        self.target = np.where(hidden, np.nan, target)

    @property
    def samples(self) -> int:
        return self.source.shape[0]

    @property
    def scored(self) -> int:
        return int(self.valid.sum())

    @property
    def missing(self) -> int:
        return int(self.valid.size - self.valid.sum())

    def __repr__(self) -> str:
        return object_repr(
            self,
            name=self.name,
            samples=self.samples,
            shape=self.shape,
            scored=self.scored,
            missing=self.missing,
        )

    def __str__(self) -> str:
        rows = []
        for key, (meaning, spec) in _METRICS.items():
            value = getattr(self, key)()
            rows.append((key, meaning, "n/a" if value != value else spec.format(value)))
        rows.append(
            ("non_finite", "values of source not finite", str(self.non_finite()))
        )
        return _table(("metric", "meaning", self.name), [("", rows)])

    def _metric(self, key: str, seq, tw, dim) -> float | np.ndarray:
        """The metric ``key`` over the axes not kept: a number, or an array
        of the axes ``seq``, ``tw`` and ``dim`` keep."""
        tw_axes = () if self.shape.tw_axis is None else (self.shape.tw_axis,)
        groups = ((seq, self.shape.seq_axes), (tw, tw_axes), (dim, self.shape.dim_axes))
        true, pred, keep = self.target, self.source, []
        for choice, axes in groups:
            if choice is None or choice is False or not axes:
                continue
            keep += axes
            if choice is not True:
                positions = list(np.atleast_1d(choice))
                true = np.take(true, positions, axis=axes[0])
                pred = np.take(pred, positions, axis=axes[0])
        relative = key in _RELATIVE
        # A relative metric is measured in each channel first.
        by = sorted({*keep, *self.shape.dim_axes}) if relative else keep
        indicators = _indicators(_along(true, by), _along(pred, by))
        values = indicators[key]
        pooled = tuple(i for i, axis in enumerate(by) if axis not in keep)
        if relative and pooled:
            # The mean of the channels; one whose target does not move has no
            # value and is left out, one that diverged spoils the mean.
            constant = indicators["constant"]
            with np.errstate(invalid="ignore", divide="ignore"):
                values = np.where(constant, 0.0, values).sum(axis=pooled) / (
                    ~constant
                ).sum(axis=pooled)
        return float(values) if np.ndim(values) == 0 else values

    def rmse(self, seq=None, tw=None, dim=None):
        """The root mean square error."""
        return self._metric("rmse", seq, tw, dim)

    def mae(self, seq=None, tw=None, dim=None):
        """The mean absolute error."""
        return self._metric("mae", seq, tw, dim)

    def max_error(self, seq=None, tw=None, dim=None):
        """The largest absolute error."""
        return self._metric("max_error", seq, tw, dim)

    def bias(self, seq=None, tw=None, dim=None):
        """The mean error, source - target."""
        return self._metric("bias", seq, tw, dim)

    def std_error(self, seq=None, tw=None, dim=None):
        """The standard deviation of the error."""
        return self._metric("std_error", seq, tw, dim)

    def nrmse_pct(self, seq=None, tw=None, dim=None):
        """The RMSE, in percent of the range of the target."""
        return self._metric("nrmse_pct", seq, tw, dim)

    def fit_pct(self, seq=None, tw=None, dim=None):
        """The FIT of system identification, in percent."""
        return self._metric("fit_pct", seq, tw, dim)

    def r2(self, seq=None, tw=None, dim=None):
        """The coefficient of determination."""
        return self._metric("r2", seq, tw, dim)

    def vaf_pct(self, seq=None, tw=None, dim=None):
        """The variance accounted for, in percent."""
        return self._metric("vaf_pct", seq, tw, dim)

    def correlation(self, seq=None, tw=None, dim=None):
        """Pearson's correlation of source and target."""
        return self._metric("correlation", seq, tw, dim)

    def non_finite(self) -> int:
        """The values of the source that are NaN or infinite where there is
        a target: a prediction that diverged."""
        scored = self.valid.reshape(*self.valid.shape, *(1,) * self.shape.dim_rank)
        return int((scored & ~np.isfinite(self.source)).sum())


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
    history=None,
    verbose: bool = False,
) -> ValidationResult:
    """The implementation of :meth:`Modely.validate`."""
    if model.model is None:
        raise ValueError("Model is not built. Call build() before validate().")
    if not model._minimizers:
        raise ValueError("No minimizers defined. Cannot infer validation targets.")

    n_samples = len(val_data)
    if n_samples == 0:
        raise ValueError("Validation dataset is empty.")

    terms = _resolve_minimizer_terms(model)
    x_data, mask = model._training_arrays(val_data)
    compared = _evaluate(model, terms, x_data, mask, n_samples)

    # The total first, as Keras logs it: every loss weighted by its gain.
    losses = Fields(loss=0.0)
    signals = Fields()
    for minimizer, term, (loss, source, target, valid) in zip(
        model._minimizers, terms, compared
    ):
        stream = minimizer["source"].shape
        shape = Shape.from_tensor_shape(
            source.shape[1:], seq_rank=stream.seq_rank, tw=stream.tw is not None
        )
        losses["loss"] += term.gain * loss
        losses[f"loss_{term.name}"] = loss
        signals[term.name] = Signal(term.name, source, target, valid, shape)

    result = ValidationResult(
        model=model.name,
        samples=n_samples,
        minimizers=minimizer_fields(model._minimizers),
        losses=losses,
        signals=signals,
    )
    if out_dir is not None or show:
        result.plot(out_dir=out_dir, show=show, history=history)
    if verbose:
        print(result)
    return result
