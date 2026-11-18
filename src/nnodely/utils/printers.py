from __future__ import annotations

import itertools
import os
import sys
import time
from typing import Any, Sequence

import keras

from nnodely.core.stream import _object_repr

#: Width of one loss cell, and of the epoch column. A cell holds a number
#: formatted as ``.4g``, which is at most 9 characters wide for the exponents
#: a loss realistically reaches.
_CELL = 9
_EPOCH_CELL = 10


class _LossTable(keras.callbacks.Callback):
    r"""The progress of :meth:`Modely.train` with ``verbose=True``: the
    losses as a table, one row per sampled epoch.

    Each row holds the epoch, then a train and a val column for each
    objective (``minimizers``, which log ``loss_<name>``) and for the total;
    ``-`` without validation data. The time the training took closes the
    table. A row is printed every :math:`\lceil \text{epochs} /
    \text{max\_rows} \rceil` epochs, and the last epoch always has its row,
    also when early stopping ends the training between two rows. It is drawn
    in the nnodely colours in a terminal or a notebook, plain otherwise
    (``NO_COLOR`` forces it plain, ``FORCE_COLOR`` coloured).
    """

    def __init__(
        self,
        epochs: int,
        minimizers: Sequence[str] = (),
        max_rows: int = 20,
    ):
        super().__init__()
        if max_rows < 1:
            raise ValueError(f"max_rows must be a positive integer, got {max_rows}.")
        self._epochs = int(epochs)
        self._minimizers = list(minimizers)
        self._max_rows = int(max_rows)
        self._stride = max(1, -(-self._epochs // self._max_rows))
        self._start_time = 0.0
        self._rows_printed = 0
        # The last epoch seen, and whether its row was printed: early stopping
        # can end training between two printed rows.
        self._last_epoch: tuple[int, dict] | None = None
        self._last_printed = False

        self._groups = [*self._minimizers, "total"]
        # The columns: the epoch, then a train and a val one per group.
        self._widths = [_EPOCH_CELL] + [_CELL] * (2 * len(self._groups))
        self._width = sum(width + 3 for width in self._widths) + 1
        self._colour = False

    def __repr__(self) -> str:
        return _object_repr(
            self,
            epochs=self._epochs,
            minimizers=self._minimizers,
            max_rows=self._max_rows,
        )

    # ------------------------------------------------------------------
    # Callback hooks
    # ------------------------------------------------------------------

    def on_train_begin(self, logs=None):
        """Called by Keras when training starts.

        :meta private:
        """

        self._start_time = time.time()
        self._colour = _shows_colour(sys.stdout)
        for line, style in self._header():
            print(self._paint(line, style), flush=True)

    def on_epoch_end(self, epoch, logs=None):
        """Called by Keras at the end of every epoch.

        :meta private:
        """

        self._last_epoch = (epoch, dict(logs or {}))
        last = epoch + 1 >= self._epochs
        self._last_printed = last or not (epoch + 1) % self._stride
        if self._last_printed:
            self._print_row(epoch, logs or {})

    def on_train_end(self, logs=None):
        """Called by Keras when training ends.

        :meta private:
        """

        if self._last_epoch is not None and not self._last_printed:
            self._print_row(*self._last_epoch)
        elapsed = time.time() - self._start_time
        if self._rows_printed:
            print()
        # The time spans the box, under the last row.
        inner = self._width - 4
        for line in (
            _box_rule(self._widths, "├", "─", "┴", "┤"),
            f"│ {f'training time: {elapsed:.3f} s'.ljust(inner)} │",
            "└" + "─" * (inner + 2) + "┘",
        ):
            print(self._paint(line, ""))

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _print_row(self, epoch: int, logs: dict) -> None:
        # A row is left unterminated so that whatever ends the training, such
        # as an early-stopping notice, reads as a continuation of its own row.
        prefix = "\n" if self._rows_printed else ""
        print(prefix + self._paint(self._row(epoch, logs), ""), end="", flush=True)
        self._rows_printed += 1

    def _paint(self, line: str, style: str) -> str:
        return _paint_box(line, style, self._width - 1) if self._colour else line

    def _header(self) -> list[tuple[str, str]]:
        """The lines above the rows, each with the style of its text."""
        # The name of a group spans its train and val columns.
        group = 2 * _CELL + 3
        spans = [_EPOCH_CELL] + [group] * len(self._groups)
        names = ["epoch", *(name[:group].center(group) for name in self._groups)]
        sides = ["", *(["train".rjust(_CELL), "val".rjust(_CELL)] * len(self._groups))]
        # A rule under the names, beside the epoch, which spans both rows.
        pair = "━" * (_CELL + 2) + "┳" + "━" * (_CELL + 2)
        split = (
            "┃"
            + " " * (_EPOCH_CELL + 2)
            + "┣"
            + "╋".join([pair] * len(self._groups))
            + "┫"
        )
        bold = "\033[1m"
        return [
            (_box_rule(spans, "┏", "━", "┳", "┓"), ""),
            (_box_row(names, spans, "┃"), bold),
            (split, ""),
            (_box_row(sides, self._widths, "┃"), bold),
            (_box_rule(self._widths, "┡", "━", "╇", "┩"), ""),
        ]

    def _row(self, epoch: int, logs: dict[str, Any]) -> str:
        keys = [f"loss_{name}" for name in self._minimizers] + ["loss"]
        cells = [f"{epoch + 1}/{self._epochs}"]
        for key in keys:
            # A loss training does not log is left blank, never filled in.
            cells.append(_format(logs.get(key)))
            cells.append(_format(logs.get(f"val_{key}")))
        return _box_row(cells, self._widths, "│")


def _format(value: Any) -> str:
    """Render one loss into a fixed-width cell.

    Precision is dropped rather than the column widened, so a stray large or
    negative value cannot knock the whole table out of alignment.
    """
    text = ""
    if value is None:
        return "-".center(_CELL)
    for precision in (4, 3, 2, 1):
        text = f"{float(value):.{precision}g}"
        if len(text) <= _CELL:
            return text.rjust(_CELL)
    return text[:_CELL]


# =====================================================================
# Colours, shared by the tables and the splash
# =====================================================================

#: The nnodely wordmark gradient, sampled from imgs/logo_info.png: sky blue at
#: the top of the glyphs, falling through cobalt into the violet at their feet.
_BRAND = (
    (0xA5, 0xD8, 0xFF),
    (0x80, 0xBF, 0xFF),
    (0x6F, 0x84, 0xFE),
    (0x5B, 0x4B, 0xFB),
    (0x7A, 0x3F, 0xE8),
)

#: Steel the machinery is cut from, for everything that is not brand coloured.
_STEEL = (0x66, 0x70, 0x84)
_BONE = (0xCF, 0xD6, 0xE2)


def _lerp(a: tuple, b: tuple, t: float) -> tuple:
    return tuple(int(round(u + (v - u) * t)) for u, v in zip(a, b))


def _brand(t: float, ramp: Sequence[tuple] = _BRAND) -> tuple:
    """Sample the brand ramp at ``t`` in [0, 1]."""
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    span = t * (len(ramp) - 1)
    i = min(int(span), len(ramp) - 2)
    return _lerp(ramp[i], ramp[i + 1], span - i)


def _shade(rgb: tuple, factor: float) -> tuple:
    """Scale a colour's brightness, clipping into the 8-bit range."""
    return tuple(max(0, min(255, int(round(c * factor)))) for c in rgb)


def _fg(rgb: tuple) -> str:
    return f"\033[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m"


def _supports_colour(stream: Any) -> bool:
    """True when it is safe to write 24-bit colour and cursor moves. A
    notebook is the exception, whose kernel sets ``FORCE_COLOR``: see
    :func:`_is_notebook`."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if os.environ.get("TERM") == "dumb":
        return False
    try:
        return bool(stream.isatty())
    except Exception:
        return False


# =====================================================================
# Notebooks: colour, but no cursor moves
# =====================================================================


def _is_notebook(stream: Any) -> bool:
    """Whether ``stream`` is the output of a notebook kernel, not merely
    ipykernel imported. A notebook renders colour, though it is no terminal,
    but not the cursor moves a terminal is repainted with: each repaint would
    be appended to the cell."""
    return type(stream).__module__.startswith("ipykernel")


def _shows_colour(stream: Any) -> bool:
    """Whether colour written to ``stream`` shows: in a terminal, as
    :func:`_supports_colour` tells, and in a notebook."""
    if os.environ.get("NO_COLOR"):
        return False
    return _is_notebook(stream) or _supports_colour(stream)


# =====================================================================
# Tables: what a model prints of its inputs, outputs, Parameters, ...
# =====================================================================

#: The characters the box of a table is drawn with.
_BOX = set("┏┳┓┃┣╋┫┡╇┻┩│├┬┴┼┤└┘━─")


def _table(
    header: tuple[str, ...],
    sections: list[tuple[str, list[tuple[str, ...]]]],
    total: tuple[str, ...] | None = None,
    indexed: bool = False,
) -> str:
    """A table as nnodely prints one: ``header`` in heavy lines, then the
    rows of each section, a (title, rows) pair, under a row of its title
    when it has one, under a rule when it has none and is not the first, and,
    given, the ``total`` row last, under a rule. The
    columns line up across the sections. ``indexed`` opens each row with its
    position among the rows of every section, under ``idx``: the index it is
    picked by from the list the table prints. In colour (see
    :func:`_shows_colour`) the box takes the brand gradient, the titles the
    brand colour."""
    if indexed:
        header = ("idx", *header)
        position = itertools.count()
        sections = [
            (name, [(str(next(position)), *cells) for cells in section])
            for name, section in sections
        ]
        total = None if total is None else ("", *total)
    rows = [row for _, section in sections for row in section]
    widths = [
        max(len(row[column]) for row in (header, *rows, *([total] if total else [])))
        for column in range(len(header))
    ]
    # The title of a section spans the box, which widens to hold it.
    longest = max((len(name) for name, _ in sections), default=0)
    widths[-1] += max(0, longest - sum(widths) - 3 * (len(widths) - 1))
    inner = sum(widths) + 3 * (len(widths) - 1)

    def rule(left: str, line: str, cross: str, right: str) -> str:
        return _box_rule(widths, left, line, cross, right)

    def row(cells: tuple[str, ...], bar: str) -> str:
        return _box_row(cells, widths, bar)

    bold, brand = "\033[1m", "\033[1m" + _fg(_BRAND[2])
    titled = bool(sections) and bool(sections[0][0])
    lines = [
        (rule("┏", "━", "┳", "┓"), ""),
        (row(header, "┃"), bold),
        (rule("┡", "━", "┻" if titled else "╇", "┩"), ""),
    ]
    for index, (name, section) in enumerate(sections):
        if name:
            if index:
                lines.append((rule("├", "─", "┴", "┤"), ""))
            lines += [
                (f"│ {name.ljust(inner)} │", brand),
                (rule("├", "─", "┬", "┤"), ""),
            ]
        elif index:
            lines.append((rule("├", "─", "┼", "┤"), ""))
        lines += [(row(cells, "│"), "") for cells in section]
    if total is not None:
        if rows:
            lines.append((rule("├", "─", "┼", "┤"), ""))
        lines.append((row(total, "│"), bold))
    lines.append((rule("└", "─", "┴", "┘"), ""))
    if not _shows_colour(sys.stdout):
        return "\n".join(line for line, _ in lines)
    return "\n".join(_paint_box(line, style, inner + 3) for line, style in lines)


def _box_rule(widths: Sequence[int], left: str, line: str, cross: str, right: str):
    """A rule of a table box across columns ``widths`` wide."""
    return left + cross.join(line * (width + 2) for width in widths) + right


def _box_row(cells: Sequence[str], widths: Sequence[int], bar: str) -> str:
    """A row of a table box: each cell padded to its column, between bars."""
    padded = (f" {cell.ljust(width)} " for cell, width in zip(cells, widths))
    return bar + bar.join(padded) + bar


def _paint_box(line: str, style: str, last: int) -> str:
    """A line of a table box in colour: the box in the brand gradient by the
    column it stands at, up to ``last``, so a bar keeps its colour from top to
    bottom; the text in ``style``. A code only where the colour changes."""
    out, current = [], ""
    for x, char in enumerate(line):
        if char in _BOX:
            step = round(16 * min(x, last) / last) / 16
            want = _fg(_shade(_brand(0.15 + 0.85 * step), 0.9))
        else:
            want = current if char == " " else style
        if want != current:
            out.append("\033[0m" + want)
            current = want
        out.append(char)
    return "".join(out) + ("\033[0m" if current else "")
