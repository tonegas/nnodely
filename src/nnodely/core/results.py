"""What :meth:`Modely.train` and :meth:`Modely.validate` return.

The two results have one form. They are read by their attributes - the name
of the model, the samples, the minimizers they were computed with, the
losses - and their dictionaries are keyed alike: ``loss`` is the total,
``loss_<minimizer>`` the loss of each minimizer, as Keras logs them and as
``early_stopping`` names them. ``print`` shows a table, a row per minimizer
and the total; ``repr`` the fields, then the keys of every dictionary.
"""

from __future__ import annotations

import os
from typing import Any

from nnodely.layers.parameter import Constant
from nnodely.utils.printers import _table


class Fields(dict):
    """A dict whose ``keys()``, ``values()`` and ``items()`` are lists, as
    those of a table of a model are."""

    def keys(self) -> list:  # pyright: ignore[reportIncompatibleMethodOverride]
        return list(super().keys())

    def values(self) -> list:  # pyright: ignore[reportIncompatibleMethodOverride]
        return list(super().values())

    def items(self) -> list:  # pyright: ignore[reportIncompatibleMethodOverride]
        return list(super().items())


def loss_name(loss_fn: Any) -> str:
    """The name of a loss function, such as ``"mean_squared_error"``."""
    for attribute in ("name", "__name__"):
        name = getattr(loss_fn, attribute, None)
        if isinstance(name, str) and name:
            return name
    return type(loss_fn).__name__ if loss_fn is not None else "mse"


def target_label(target) -> str | float:
    """How the target of a minimizer prints: a Constant of one value - a
    number given as the target - as that number, any other by its name."""
    if (
        isinstance(target, Constant)
        and target.value is not None
        and target.value.size == 1
    ):
        return float(target.value.ravel()[0])
    return target.name


def minimizer_fields(minimizers: list[dict]) -> Fields:
    """A snapshot of the minimizers of a model, by name: the source, the
    target, the loss and the gain of each, as they print. Changing the model
    later leaves it as it is."""
    return Fields(
        (
            m["name"],
            Fields(
                source=m["source"].name,
                target=target_label(m["target"]),
                loss=loss_name(m["loss"]),
                gain=m["gain"],
            ),
        )
        for m in minimizers
    )


def _value(number: float) -> str:
    return "n/a" if number != number else f"{number:.4e}"


class _Result:
    """What a TrainResult and a ValidationResult share: they are read by
    their attributes, and their ``repr`` lists the keys of each dictionary
    named in ``_dicts``."""

    _dicts: tuple[str, ...] = ("minimizers", "losses")
    model: str
    samples: int
    minimizers: Fields
    losses: Fields

    def _head(self) -> str:
        """The fields the ``repr`` opens with."""
        raise NotImplementedError

    def __repr__(self) -> str:
        width = max(len(name) for name in self._dicts) + 1
        lines = [f"{type(self).__name__}({self._head()},"]
        for name in self._dicts:
            lines.append(f"  {name + ':':<{width}} {list(getattr(self, name))!r},")
        lines.append(")")
        return "\n".join(lines)

    def __getitem__(self, key):
        where = next(
            (name for name in self._dicts if key in getattr(self, name)), "losses"
        )
        raise TypeError(
            f"{type(self).__name__} is read by its attributes, not as a dict or a "
            f"list: its {where} are .{where}[{key!r}]."
        )


class TrainResult(_Result):
    """What :meth:`~nnodely.Modely.train` returns: the losses of every epoch.

    You do not create one yourself. The model does not keep it: save it in a
    variable, or it is lost.

    Attributes
    ----------
    model : str
        The name of the trained model.
    samples : int
        The samples of the training data.
    epochs : int
        The epochs run: fewer than those asked for when early stopping ends
        training.
    minimizers : dict
        The minimizers trained, by name: the ``source``, ``target``, ``loss``
        and ``gain`` of each, as they were when training ran.
    losses : dict of str to numpy.ndarray
        A value per epoch, under the keys Keras logs: ``loss``, the total,
        every minimizer weighted by its gain; ``loss_<minimizer>``, the loss
        of each, not weighted; and, with validation data, the same under
        ``val_``. ``dict(result.losses)`` is the Keras history.

    ``print`` shows a table, a row per minimizer with the losses of the last
    epoch, and the total. Every dictionary has ``keys()``, ``values()`` and
    ``items()`` that are lists.

    Basic Examples
    --------------
    >>> import numpy as np
    >>> from nnodely import DataLoader, Input, Modely, Output, Parameter
    >>> x = Input("x", tw=1)
    >>> y = Output("y", x + Parameter("b", value=[1.0]))
    >>> model = Modely("offset", inputs=[x], outputs=[y])
    >>> model = model.minimize("fit", y, Input("y_meas", tw=1)).build()
    >>> data = DataLoader(model, source={"x": np.arange(8.0), "y_meas": np.arange(8.0)})
    >>> trained = model.train(data, val_data=data, epochs=2, batch_size=4, printer=None)
    >>> trained.epochs, trained.samples, trained.losses.keys()
    (2, 8, ['loss', 'loss_fit', 'val_loss', 'val_loss_fit'])
    >>> trained.minimizers["fit"]
    {'source': 'y', 'target': 'y_meas', 'loss': 'mean_squared_error', 'gain': 1.0}

    .. versionadded:: 2.0
    """

    def __init__(self, *, model: str, samples: int, minimizers: Fields, losses: Fields):
        self.model = model
        self.samples = samples
        self.minimizers = minimizers
        self.losses = losses

    @property
    def epochs(self) -> int:
        return len(self.losses["loss"])

    def _head(self) -> str:
        head = (
            f"model={self.model!r}, epochs={self.epochs}, samples={self.samples}, "
            f"loss={self.losses['loss'][-1]:.4g}"
        )
        if "val_loss" in self.losses:
            head += f", val_loss={self.losses['val_loss'][-1]:.4g}"
        return head

    def __str__(self) -> str:
        validated = "val_loss" in self.losses
        header = ("minimizer", "loss", "gain", "loss value")
        if validated:
            header += ("val loss value",)
        rows = []
        for name, m in self.minimizers.items():
            row = (
                name,
                m["loss"],
                str(m["gain"]),
                _value(self.losses[f"loss_{name}"][-1]),
            )
            if validated:
                row += (_value(self.losses[f"val_loss_{name}"][-1]),)
            rows.append(row)
        total = ("total", "", "", _value(self.losses["loss"][-1]))
        if validated:
            total += (_value(self.losses["val_loss"][-1]),)
        title = f"{self.model}: {self.epochs} epochs, {self.samples} samples"
        return _table(header, [(title, rows)], total, indexed=True)


class ValidationResult(_Result):
    """What :meth:`~nnodely.Modely.validate` returns: the loss of every
    minimizer on the data, and the signals it compared.

    You do not create one yourself.

    Attributes
    ----------
    model : str
        The name of the validated model.
    samples : int
        The samples validated.
    minimizers : dict
        The minimizers validated, by name: the ``source``, ``target``,
        ``loss`` and ``gain`` of each, as they were when validation ran.
    losses : dict of str to float
        A number per key, the keys of :class:`TrainResult` without ``val_``:
        ``loss``, the total, and ``loss_<minimizer>``. On the same data they
        are the ``val_`` losses of the last epoch of training.
    signals : dict of str to Signal
        A :class:`~nnodely.core.validation.Signal` per minimizer: the two
        signals it compared, whole, and their metrics.
    figures : list of str
        The PNG files the last :meth:`plot` wrote.

    ``print`` shows a table, a row per minimizer with its loss, and the
    total. Every dictionary has ``keys()``, ``values()`` and ``items()``
    that are lists.

    Basic Examples
    --------------
    >>> from nnodely import DataLoader, Input, Modely, Output, Parameter
    >>> x = Input("x", tw=1)
    >>> y = Output("y", x + Parameter("b", value=[1.0]))
    >>> model = Modely("offset", inputs=[x], outputs=[y])
    >>> model = model.minimize("fit", y, Input("y_meas", tw=1)).build()
    >>> data = DataLoader(model, source={"x": [0.0, 2.0, 4.0, 6.0],
    ...                                  "y_meas": [0.0, 2.0, 4.0, 6.0]})
    >>> result = model.validate(data, verbose=False)
    >>> result
    ValidationResult(model='offset', samples=4, loss=1,
      minimizers: ['fit'],
      losses:     ['loss', 'loss_fit'],
      signals:    ['fit'],
      figures:    [],
    )
    >>> result.losses["loss_fit"], result.signals["fit"].rmse()
    (1.0, 1.0)

    .. versionadded:: 2.0
    """

    _dicts = ("minimizers", "losses", "signals", "figures")

    def __init__(
        self,
        *,
        model: str,
        samples: int,
        minimizers: Fields,
        losses: Fields,
        signals: Fields,
    ):
        self.model = model
        self.samples = samples
        self.minimizers = minimizers
        self.losses = losses
        self.signals: Fields = signals
        self.figures: list[str] = []

    def _head(self) -> str:
        return f"model={self.model!r}, samples={self.samples}, loss={self.losses['loss']:.4g}"

    def __str__(self) -> str:
        header = ("minimizer", "source", "target", "loss", "gain", "loss value")
        rows = [
            (
                name,
                m["source"],
                str(m["target"]),
                m["loss"],
                str(m["gain"]),
                _value(self.losses[f"loss_{name}"]),
            )
            for name, m in self.minimizers.items()
        ]
        total = ("total", "", "", "", "", _value(self.losses["loss"]))
        title = f"{self.model}: {self.samples} samples"
        return _table(header, [(title, rows)], total, indexed=True)

    def plot(
        self,
        out_dir: str | os.PathLike | None = None,
        show: bool = False,
        history: TrainResult | dict | None = None,
    ) -> list[str]:
        """Draw a figure per minimizer - response, error, parity, error
        distribution and, along a sequence, the error at each step - and,
        given ``history``, the loss curves of a training.

        Parameters
        ----------
        out_dir : str, os.PathLike or None, default None
            The folder the figures are written to, as PNG files named
            ``<model>_<index>_<minimizer>.png`` and ``<model>_history.png``.
            ``None`` writes none. It is created when missing.
        show : bool, default False
            ``True`` opens the figures in an interactive Matplotlib window.
        history : TrainResult, dict or None, default None
            The losses of a training, drawn as curves.

        Returns
        -------
        list of str
            The files written, also kept in :attr:`figures`.
        """
        # Drawing needs Matplotlib: imported here, where it is used.
        from nnodely.utils import validation_plot

        losses = history.losses if isinstance(history, TrainResult) else history
        self.figures = validation_plot.render(
            self, out_dir=out_dir, show=show, history=losses
        )
        return self.figures
