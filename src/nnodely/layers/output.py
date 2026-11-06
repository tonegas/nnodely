"""Output: names a stream as an output of the model."""

from nnodely.core.stream import Stream
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
            time=stream.shape.time,
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

    _HEADER = ("name", "seq", "time", "dim", "stream", "feeds back", "minimized by")

    def __init__(self, rows: list[dict], title: str = ""):
        super().__init__(row["output"] for row in rows)
        self._rows = rows
        self._title = title

    def __str__(self):
        cells = [
            (
                row["output"].name,
                _seq_cell(row["output"].seq),
                "-" if row["output"].time is None else str(row["output"].time),
                str(tuple(row["output"].dim)),
                row["stream"],
                ", ".join(row["feeds back"]) or "-",
                ", ".join(row["minimized by"]) or "-",
            )
            for row in self._rows
        ]
        widths = [
            max(len(line[column]) for line in (self._HEADER, *cells))
            for column in range(len(self._HEADER))
        ]

        def line(row):
            return "  ".join(cell.ljust(width) for cell, width in zip(row, widths))

        fed_back = sum(bool(row["feeds back"]) for row in self._rows)
        minimized = sum(bool(row["minimized by"]) for row in self._rows)
        return "\n".join(
            [
                *([self._title] if self._title else []),
                line(self._HEADER).rstrip(),
                line(tuple("-" * width for width in widths)).rstrip(),
                *(line(row).rstrip() for row in cells),
                f"{len(self)} outputs, {fed_back} fed back, {minimized} minimized",
                "stream: the stream the output names; in a ModelyLoop, the "
                "stream of the step, collected over the steps",
                "feeds back: the Inputs a loop feeds that stream back into at "
                "the next step",
            ]
        )

    __repr__ = __str__
