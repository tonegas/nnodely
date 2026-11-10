"""Output: names a stream as an output of the model."""

from nnodely.core.stream import Stream
from nnodely.utils.printers import _table
from nnodely.layers.input import _seq_cell


class Output(Stream):
    """Expose ``stream`` as an output of the model, under ``name``.

    Output names are the keys of the model's results and the names used to
    refer to outputs elsewhere, for example in feedback mappings.
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


class Outputs(list):
    """The outputs of a model, as :attr:`Modely.outputs` lists them: a list
    of the Output objects, in the order the model was declared with, printed
    as a table of their shapes, the stream each names, the Inputs that
    stream is fed back into and the minimizers that read it."""

    _HEADER = (
        "index",
        "output",
        "seq",
        "tw",
        "dim",
        "stream",
        "feeds back",
        "minimized by",
    )

    def __init__(self, rows: list[dict]):
        super().__init__(row["output"] for row in rows)
        self._rows = rows

    def __getitem__(self, key):  # type: ignore[override]
        """An output by its name, or by its index - the index column of the
        table - as in a list."""
        from nnodely.core.dag import by_name_or_index, pick  # dag imports this module

        if not by_name_or_index(key, "Outputs"):
            return super().__getitem__(key)
        candidates = [
            (row["output"], index, f"(of {row['stream']!r})")
            for index, row in enumerate(self._rows)
        ]
        picked = pick(
            candidates,
            key,
            kind="output",
            owner="Outputs",
            hint="Pick one by its index, outputs[i], or give it a name of its "
            "own, outputs[i].name = 'new_name'.",
        )
        if not picked:
            raise KeyError(
                f"Outputs: no output named {key!r}; the names are "
                f"{[node.name for node in self]}."
            )
        return picked[0]

    def __str__(self):
        cells = [
            (
                str(index),
                row["output"].name,
                _seq_cell(row["output"].seq),
                "-" if row["output"].tw is None else str(row["output"].tw),
                str(tuple(row["output"].dim)),
                row["stream"],
                ", ".join(row["feeds back"]) or "-",
                ", ".join(row["minimized by"]) or "-",
            )
            for index, row in enumerate(self._rows)
        ]
        return _table(self._HEADER, [("", cells)])

    __repr__ = __str__
