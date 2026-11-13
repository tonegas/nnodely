"""Output: names a stream as an output of the model."""

from nnodely.core.dag import Table
from nnodely.core.stream import Stream
from nnodely.utils.printers import _table
from nnodely.layers.input import _seq_cell


class Output(Stream):
    """A stream of the model given a public name: the key of its result.

    Syntax
    ------
    ``Output(name, stream)``
        Names ``stream`` as the output ``name`` of a model.

    A model returns a dict of its outputs, keyed by their names. The names
    also refer to outputs elsewhere: in :meth:`Modely.minimize`, in the
    ``feedback`` of a :class:`Loop`, and in :attr:`Modely.outputs`. An
    Output is a stream of the shape of ``stream``, so other streams can read
    it.

    Parameters
    ----------
    name : str
        The name of the output, the key of its result. Two outputs of one
        model cannot share it.
    stream : Stream
        The stream the output names.

    Returns
    -------
    Output
        A stream of the shape of ``stream``: ``Shape(S, T, D)`` gives
        ``Shape(S, T, D)``.

    Raises
    ------
    ValueError
        When the model is built, if two of its outputs share a name ("has
        two different outputs named").

    See Also
    --------
    Input : A signal read from data.
    Modely : The model, declared with its inputs and outputs.
    Modely.minimize : Adds an objective on an output.

    Basic Examples
    --------------
    >>> import keras
    >>> import numpy as np
    >>> from nnodely import Input, Modely, Output
    >>> x = Input("x", tw=2)
    >>> doubled = Output("doubled", 2.0 * x)
    >>> doubled.shape
    Shape(S=(), T=2, D=(1,))
    >>> model = Modely("double", inputs=[x], outputs=[doubled]).build()
    >>> result = model({"x": np.array([[[1.0], [2.0]]])})
    >>> list(result)
    ['doubled']
    >>> keras.ops.convert_to_numpy(result["doubled"])
    array([[[2.],
            [4.]]], dtype=float32)

    Scope
    -----
    An output keeps the shape of its stream, window and sequence included:

    >>> Output("v", Input("v_in", dim=3, seq=4, tw=2) * 1.0).shape
    Shape(S=(4,), T=2, D=(3,))

    Options
    -------
    ``name`` picks the output among those of the model:

    >>> model.outputs["doubled"] is doubled
    True

    Properties & Relations
    ----------------------
    The Output is the stream the model lists, picked by name or by index:

    >>> model.outputs[0] is model.outputs["doubled"]
    True

    Possible Issues
    ---------------
    Two outputs of one model need two names:

    >>> Modely("twice", inputs=[x], outputs=[Output("y", x * 2.0), Output("y", x * 3.0)]).build()
    Traceback (most recent call last):
        ...
    ValueError: Model 'twice' has two different outputs named 'y'...

    .. versionadded:: 2.0
    """

    def __init__(self, name: str, stream: Stream) -> None:
        super().__init__(
            name=name,
            dim=stream.shape.dim,
            tw=stream.shape.tw,
            seq=stream.shape.seq,
            preds=[stream],
        )

    @classmethod
    def from_config(cls, config: dict, preds=None) -> "Output":
        if preds is None or len(preds) != 1:
            raise ValueError(
                f"Output '{config['name']}' requires exactly one predecessor."
            )
        return super().from_config({**config, "stream": preds[0]})


class Outputs(Table):
    """The outputs of a model, as :attr:`Modely.outputs` lists them.

    It is a list of the Output objects, in the order the model was declared
    with. Pick one by its name, ``outputs["y"]``, or by its index,
    ``outputs[0]``: the ``idx`` column of the table. Printed, it is a table
    of their shapes, the stream each names, the Inputs that stream is fed
    back into and the objectives, called minimizers in the method names,
    that read it."""

    _HEADER = (
        "output",
        "seq",
        "tw",
        "dim",
        "stream",
        "feeds back",
        "minimized by",
    )

    _kind, _attr = "output", "outputs"

    def __init__(self, rows: list[dict]):
        super().__init__(row["output"] for row in rows)
        self._rows = rows

    def _where(self, index, item) -> str:
        return f"(of {self._rows[index]['stream']!r})"

    def __str__(self):
        cells = [
            (
                row["output"].name,
                _seq_cell(row["output"].seq),
                "-" if row["output"].tw is None else str(row["output"].tw),
                str(tuple(row["output"].dim)),
                row["stream"],
                ", ".join(row["feeds back"]) or "-",
                ", ".join(row["minimized by"]) or "-",
            )
            for row in self._rows
        ]
        return _table(self._HEADER, [("", cells)], indexed=True)

    __repr__ = __str__
