"""Training progress printers.

A printer is a plain ``keras.callbacks.Callback`` that owns everything about
how an epoch is rendered, so :meth:`Modely.train` only has to pick one.
"""

from __future__ import annotations

import atexit
import math
import os
import shutil
import sys
import time
from collections import deque
from typing import Any, Sequence

import keras

#: Width of one loss cell, and of the epoch column. A cell holds a number
#: formatted as ``.4g``, which is at most 9 characters wide for the exponents
#: a loss realistically reaches.
_CELL = 9
_EPOCH_CELL = 10


def _resolve_printer(
    printer: str | keras.callbacks.Callback | None,
    epochs: int,
    minimizers: Sequence[dict[str, Any]] = (),
    name: str = "model",
) -> keras.callbacks.Callback:
    """Turn the ``printer`` argument of :meth:`train` into a callback."""
    if isinstance(printer, keras.callbacks.Callback):
        return printer
    if printer is None:
        return keras.callbacks.Callback()
    if printer == "tiny":
        return TinyPrinter(epochs=epochs)
    if printer == "legacy":
        return LegacyPrinter(
            epochs=epochs,
            minimizers=[minimizer["name"] for minimizer in minimizers],
        )
    if printer == "nnodely":
        return NNodelyPrinter(
            epochs=epochs,
            minimizers=[minimizer["name"] for minimizer in minimizers],
            model_name=name,
        )
    raise ValueError(
        f"Unknown printer {printer!r}: expected 'tiny', 'legacy', "
        "'nnodely', None, or a keras.callbacks.Callback."
    )


class TinyPrinter(keras.callbacks.Callback):
    """Repaint a single-screen summary of the current epoch.

    The terminal is cleared on every epoch, so only the latest numbers are on
    screen and nothing scrolls; in a notebook the summary is one output of the
    cell, replaced in place a few times a second. Use :class:`LegacyPrinter`
    when the history of the run matters more than the current value.
    """

    def __init__(self, epochs: int | None = None):
        super().__init__()
        self._epochs = epochs
        self._start_time = 0.0
        self._view: _NotebookView | None = None
        # The summary of the last epoch, and when the notebook last showed one.
        self._text = ""
        self._shown = 0.0

    def on_train_begin(self, logs=None):
        self._start_time = time.time()
        self._view = _NotebookView() if _is_notebook(sys.stdout) else None
        self._text, self._shown = "", 0.0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        elapsed = time.time() - self._start_time

        sep = "─" * 70

        if self.params is None:
            self.params = {"epochs": self._epochs}
        text = "\n".join(
            [
                sep,
                f"Epoch {epoch + 1}/{self.params['epochs']}   Elapsed: {elapsed:.1f}s",
                sep,
                *(f"{k:<30} {v:>12.3e}" for k, v in sorted(logs.items())),
                sep,
            ]
        )

        if self._view is not None:
            self._text = text
            if time.time() - self._shown >= 1.0 / _NOTEBOOK_FPS:
                self._view.show(text)
                self._shown = time.time()
            return
        # Repainted in place on a terminal; a file gets one block per epoch.
        if _supports_colour(sys.stdout):
            sys.stdout.write("\033[H\033[J")
            sys.stdout.flush()
        print(text)

    def on_train_end(self, logs=None):
        # The last epoch, which the notebook may not have shown yet.
        if self._view is not None and self._text:
            self._view.show(self._text)


class LegacyPrinter(keras.callbacks.Callback):
    """Print the scrolling loss table of the training, drawn as the tables of
    a model are.

    One row per sampled epoch, one column pair (train, val) per minimizer plus
    a final pair for the total loss, and the time the training took once it
    ends::

        ┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ epoch      ┃       curv_error      ┃         total         ┃
        ┃            ┣━━━━━━━━━━━┳━━━━━━━━━━━╋━━━━━━━━━━━┳━━━━━━━━━━━┫
        ┃            ┃     train ┃       val ┃     train ┃       val ┃
        ┡━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━┩
        │ 60/6000    │ 2.099e-06 │ 1.226e-06 │ 2.099e-06 │ 1.226e-06 │
        ├────────────┴───────────┴───────────┴───────────┴───────────┤
        │ training time: 12.345 s                                    │
        └────────────────────────────────────────────────────────────┘

    In the nnodely colours where the tables of a model are - a terminal or a
    notebook - plain otherwise.

    Rows are sampled every ``epochs / max_rows`` epochs, so a 6000-epoch run
    prints the same number of lines as a 60-epoch one instead of flooding the
    terminal. The final epoch is always printed, whatever the stride, so the
    table ends on the numbers the model actually stopped at.

    Parameters
    ----------
    epochs:
        Total number of epochs, used for the stride and the ``n/total`` label.
    minimizers:
        The minimizers' names, in column order: each logs its loss under
        ``"<name>_loss"``.
    max_rows:
        Upper bound on the number of printed rows.
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

    # ------------------------------------------------------------------
    # Callback hooks
    # ------------------------------------------------------------------

    def on_train_begin(self, logs=None):
        self._start_time = time.time()
        self._colour = _shows_colour(sys.stdout)
        for line, style in self._header():
            print(self._paint(line, style), flush=True)

    def on_epoch_end(self, epoch, logs=None):
        self._last_epoch = (epoch, dict(logs or {}))
        last = epoch + 1 >= self._epochs
        self._last_printed = last or not (epoch + 1) % self._stride
        if self._last_printed:
            self._print_row(epoch, logs or {})

    def on_train_end(self, logs=None):
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
        # A row is left unterminated so that whatever ends the training - an
        # early-stopping notice - reads as a continuation of its own row.
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
        split = "┃" + " " * (_EPOCH_CELL + 2) + "┣" + "╋".join([pair] * len(self._groups)) + "┫"
        bold = "\033[1m"
        return [
            (_box_rule(spans, "┏", "━", "┳", "┓"), ""),
            (_box_row(names, spans, "┃"), bold),
            (split, ""),
            (_box_row(sides, self._widths, "┃"), bold),
            (_box_rule(self._widths, "┡", "━", "╇", "┩"), ""),
        ]

    def _row(self, epoch: int, logs: dict[str, Any]) -> str:
        keys = [f"{name}_loss" for name in self._minimizers] + ["loss"]
        cells = [f"{epoch + 1}/{self._epochs}"]
        for key in keys:
            cells.append(_format(logs.get(key, logs.get("loss"))))
            cells.append(_format(logs.get(f"val_{key}", logs.get("val_loss"))))
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
# NNodelyPrinter - the machine room
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
_STEEL_DIM = (0x2E, 0x34, 0x40)
_BONE = (0xCF, 0xD6, 0xE2)
_EMBER = (0xFF, 0x9E, 0x4A)

_SPARKS = "▁▂▃▄▅▆▇█"
_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

#: A 10-row bitmap for the six glyphs of the wordmark. Rows 0-1 carry the
#: ascenders, 2-6 the x-height, 7-8 the descender of the 'y'.
_GLYPHS = {
    "n": (
        ".....",
        ".....",
        "#.##.",
        "##..#",
        "#...#",
        "#...#",
        "#...#",
        ".....",
        ".....",
        ".....",
    ),
    "o": (
        ".....",
        ".....",
        ".###.",
        "#...#",
        "#...#",
        "#...#",
        ".###.",
        ".....",
        ".....",
        ".....",
    ),
    "d": (
        "....#",
        "....#",
        ".####",
        "#...#",
        "#...#",
        "#...#",
        ".####",
        ".....",
        ".....",
        ".....",
    ),
    "e": (
        ".....",
        ".....",
        ".###.",
        "#...#",
        "#####",
        "#....",
        ".###.",
        ".....",
        ".....",
        ".....",
    ),
    "l": (
        "..##.",
        "...#.",
        "...#.",
        "...#.",
        "...#.",
        "...#.",
        "...##",
        ".....",
        ".....",
        ".....",
    ),
    "y": (
        ".....",
        ".....",
        "#...#",
        "#...#",
        "#...#",
        ".####",
        "....#",
        "#...#",
        ".###.",
        ".....",
    ),
}
_WORDMARK = "nnodely"


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


def _bg(rgb: tuple) -> str:
    return f"\033[48;2;{rgb[0]};{rgb[1]};{rgb[2]}m"


class _Canvas:
    """A half-block pixel canvas.

    Every character cell carries two stacked pixels - the upper one painted as
    the foreground of ``▀``, the lower one as its background - which makes the
    pixels square on a normal terminal grid.
    """

    def __init__(self, cols: int, rows: int):
        self.cols = cols
        self.rows = rows
        self.height = rows * 2
        self.px: list[list[tuple | None]] = [[None] * cols for _ in range(self.height)]

    def set(self, x: int, y: int, rgb: tuple | None) -> None:
        if rgb is not None and 0 <= x < self.cols and 0 <= y < self.height:
            self.px[y][x] = rgb

    def lines(self) -> list[str]:
        out = []
        for row in range(self.rows):
            top_row, bottom_row = self.px[row * 2], self.px[row * 2 + 1]
            chunks, fg, bg = [], None, None
            for x in range(self.cols):
                top, bottom = top_row[x], bottom_row[x]
                if top is None and bottom is None:
                    if bg is not None:
                        chunks.append("\033[49m")
                        bg = None
                    chunks.append(" ")
                    continue
                if top is not None and bottom is not None:
                    char, want_fg, want_bg = "▀", top, bottom
                elif top is not None:
                    char, want_fg, want_bg = "▀", top, None
                else:
                    char, want_fg, want_bg = "▄", bottom, None
                if want_fg != fg:
                    chunks.append(_fg(want_fg))  # type: ignore
                    fg = want_fg
                if want_bg != bg:
                    chunks.append(_bg(want_bg) if want_bg else "\033[49m")
                    bg = want_bg
                chunks.append(char)
            chunks.append("\033[0m")
            out.append("".join(chunks))
        return out


def _loss_graph(losses: Sequence[float], cols: int, rows: int) -> list[str]:
    """The loss along the epochs as an area chart, ``rows`` lines of ``cols``
    columns: its highest and lowest value on the axis, then a column per
    epoch - the last epoch of each span of them, when there are more epochs
    than columns - its height in eighths of a line, on a log scale when every
    loss is positive. A loss that is not finite leaves its column empty."""
    width = cols - _CELL - 2  # the values, a space and the axis
    count = len(losses)
    if count > width:
        losses = [losses[(column + 1) * count // width - 1] for column in range(width)]
    finite = [loss for loss in losses if math.isfinite(loss)]
    log = bool(finite) and min(finite) > 0

    def scale(loss: float) -> float:
        return math.log10(loss) if log else loss

    low, high = (min(finite), max(finite)) if finite else (None, None)
    span = scale(high) - scale(low) if finite else 0.0
    # Never below an eighth: a flat loss is still a line on the axis.
    eighths = [
        (1 + round((scale(loss) - scale(low)) / span * (rows * 8 - 1)) if span else 1)
        if math.isfinite(loss)
        else 0
        for loss in losses
    ]

    out = []
    for row in range(rows):
        floor = (rows - 1 - row) * 8
        value = high if row == 0 else low if row == rows - 1 else None
        line = _Line().add(_format(value) if value is not None else " " * _CELL, _STEEL)
        line.add(" ").add("┤" if value is not None else "│", _STEEL_DIM)
        for column, height in enumerate(eighths):
            fill = height - floor
            if fill <= 0:
                line.add(" ")
            else:
                block = _SPARKS[min(fill, 8) - 1]
                line.add(block, _brand(0.15 + 0.85 * column / max(1, width - 1)))
        out.append(str(line.pad(cols)))
    return out


def _ink(letter: str) -> tuple[int, int]:
    """The first and last column a glyph is drawn in."""
    columns = [x for row in _GLYPHS[letter] for x, cell in enumerate(row) if cell == "#"]
    return min(columns), max(columns)


def _wordmark_width() -> int:
    """Columns of the wordmark: each letter as wide as it is drawn, the next
    one column after it."""
    widths = [last - first + 1 for first, last in map(_ink, _WORDMARK)]
    return sum(widths) + len(widths) - 1


def _draw_wordmark(canvas: _Canvas, x0: int, y0: int, sweep: float) -> None:
    """Paint 'nnodely' in the logo gradient with a highlight sweeping across.

    Each letter is placed by its own width, so they are evenly spaced
    whatever the columns a glyph leaves empty."""
    width = _wordmark_width()
    left = x0
    for letter in _WORDMARK:
        first, last = _ink(letter)
        rows = _GLYPHS[letter]
        for dy, row in enumerate(rows):
            for dx, cell in enumerate(row):
                if cell != "#":
                    continue
                x = left + dx - first
                y = y0 + dy
                # The gradient runs down the glyphs exactly as it does in the
                # logo; the sweep is a soft specular band travelling right.
                colour = _brand((dy - 1) / 7.0)
                distance = abs((x - x0) / max(1, width) - sweep)
                highlight = math.exp(-((distance * 6.0) ** 2))
                canvas.set(x, y, _shade(colour, 1.0 + 0.75 * highlight))
        left += last - first + 2


class _Line:
    """A single terminal line that tracks its visible width past the colours."""

    def __init__(self):
        self.parts: list[str] = []
        self.width = 0

    def add(
        self, text: str, colour: tuple | None = None, bold: bool = False
    ) -> "_Line":
        if text:
            prefix = (_fg(colour) if colour else "") + ("\033[1m" if bold else "")
            self.parts.append(f"{prefix}{text}\033[0m" if prefix else text)
            self.width += len(text)
        return self

    def raw(self, text: str, visible: int) -> "_Line":
        """Append already-coloured text whose visible width is known."""
        self.parts.append(text)
        self.width += visible
        return self

    def pad(self, column: int) -> "_Line":
        return self.add(" " * max(0, column - self.width))

    def __str__(self) -> str:
        return "".join(self.parts)


def _clock(seconds: float) -> str:
    if seconds != seconds or seconds in (float("inf"), float("-inf")) or seconds < 0:
        return "--:--:--"
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


class _Channel:
    """The rolling history of one loss, and how it is drawn."""

    __slots__ = ("name", "key", "train", "val")

    def __init__(self, name: str, key: str, span: int):
        self.name = name
        self.key = key
        self.train: deque[float] = deque(maxlen=span)
        self.val: deque[float] = deque(maxlen=span)

    def push(self, train: float | None, val: float | None) -> None:
        if train is not None:
            self.train.append(float(train))
        if val is not None:
            self.val.append(float(val))

    def trend(self) -> float:
        """Relative change of the loss over the visible history."""
        if len(self.train) < 3:
            return 0.0
        first, last = self.train[0], self.train[-1]
        if first == 0 or not (math.isfinite(first) and math.isfinite(last)):
            return 0.0
        return (last - first) / abs(first)

    def sparkline(self, width: int) -> str:
        values = list(self.train)[-width:]
        if not values:
            return " " * width
        # A loss falls over orders of magnitude, so the spark is drawn on a log
        # scale - otherwise everything after the first few epochs is one flat
        # line pinned to the bottom of the cell.
        positive = [v for v in values if math.isfinite(v) and v > 0]
        floor = min(positive) if positive else 1.0
        scaled = [
            math.log10(max(v, floor * 1e-3)) if math.isfinite(v) else None
            for v in values
        ]
        known = [value for value in scaled if value is not None]
        low, high = (min(known), max(known)) if known else (0.0, 0.0)
        span = high - low
        out = []
        for index, value in enumerate(scaled):
            # A loss that is not finite diverged: drawn at the top.
            if value is None:
                level = 1.0
            else:
                level = 0.0 if span <= 0 else (value - low) / span
            block = _SPARKS[min(len(_SPARKS) - 1, int(level * len(_SPARKS)))]
            colour = _brand(0.15 + 0.85 * (1.0 - level))
            fade = 0.45 + 0.55 * (index + 1) / len(scaled)
            out.append(_fg(_shade(colour, fade)) + block)
        return "".join(out) + "\033[0m" + " " * (width - len(values))


class NNodelyPrinter(keras.callbacks.Callback):
    """A machine room for your training run.

    Renders the run as a live industrial console painted in the nnodely logo
    gradient: the wordmark with a highlight sweeping across it, a chart of the
    total loss along the epochs - on a log scale, its highest and lowest value
    on the axis - beside the telemetry of the run, a gradient progress bar,
    and a log-scale spark per minimizer.

    Everything is repainted in place at a capped frame rate, so a fast run
    animates smoothly without the terminal becoming the bottleneck. In a
    notebook the console is one output of the cell, replaced in place a few
    times a second. On a non-terminal (a log file, a CI job) or with
    ``NO_COLOR`` set it falls back to one plain line per sampled epoch, so
    nothing downstream sees escapes.
    """

    #: Columns of the loss chart, and how many character rows it is tall.
    _BAY = 34
    _BAY_ROWS = 10

    def __init__(
        self,
        epochs: int,
        minimizers: Sequence[str] = (),
        model_name: str = "model",
        width: int | None = None,
        fps: float = 24.0,
        stream: Any = None,
    ):
        super().__init__()
        self._epochs = max(1, int(epochs))
        self._model_name = model_name
        self._fps = max(1.0, float(fps))
        self._stream = stream if stream is not None else sys.stdout
        self._colour = _shows_colour(self._stream)
        self._notebook = _is_notebook(self._stream)
        if self._notebook:
            self._fps = min(self._fps, _NOTEBOOK_FPS)
        self._view: _NotebookView | None = None

        terminal = shutil.get_terminal_size(fallback=(80, 24)).columns
        self._width = max(72, min(110, int(width or terminal - 2)))
        self._spark = max(10, min(28, self._width - 62))

        self._channels = [
            _Channel(name, f"{name}_loss", self._spark) for name in minimizers
        ] + [_Channel("Total", "loss", self._spark)]

        self._start_time = 0.0
        self._last_paint = 0.0
        self._epoch = 0
        self._load = 0.0
        # The total loss of every epoch, as the chart draws it.
        self._history: list[float] = []
        self._last_total: float | None = None
        self._restored = False
        # Without a terminal there is nothing to repaint, so the console
        # degrades to one line per sampled epoch rather than 24 lines a second
        # of log file.
        self._plain_stride = max(1, self._epochs // 40)

    # ------------------------------------------------------------------
    # Callback hooks
    # ------------------------------------------------------------------

    def on_train_begin(self, logs=None):
        self._start_time = time.time()
        self._last_paint = 0.0
        if not self._colour:
            return
        if self._notebook:
            # A run of its own: a new output, not the one of an earlier run.
            self._view = None
            return
        # A crash or a Ctrl-C must never leave the user staring at a terminal
        # with no cursor, so the restore is armed before anything is hidden -
        # and disarmed once training ends, since the hook keeps this printer
        # alive until the interpreter exits.
        self._restored = False
        atexit.register(self._restore)
        self._stream.write("\033[?25l\033[2J")
        self._stream.flush()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self._epoch = epoch + 1

        total = logs.get("loss")
        for channel in self._channels:
            channel.push(
                logs.get(channel.key, logs.get("loss")),
                logs.get(f"val_{channel.key}", logs.get("val_loss")),
            )

        # The machine works harder the faster the loss is falling: one epoch's
        # relative improvement, smoothed, is the drive load it shows.
        if total is not None and self._last_total:
            gain = (self._last_total - float(total)) / abs(self._last_total)
            self._load = max(0.0, min(1.0, 0.82 * self._load + 18.0 * max(0.0, gain)))
        if total is not None:
            self._last_total = float(total)
            self._history.append(float(total))

        now = time.time()
        final = self._epoch >= self._epochs
        if self._colour:
            if not final and now - self._last_paint < 1.0 / self._fps:
                return
        elif not final and self._epoch % self._plain_stride:
            return
        self._last_paint = now
        self._paint(now - self._start_time)

    def on_train_end(self, logs=None):
        elapsed = time.time() - self._start_time
        if self._colour:
            self._paint(elapsed, done=True)
            if not self._notebook:
                atexit.unregister(self._restore)
        self._restore()
        self._stream.write("\n")
        for line in self._plaque(elapsed):
            self._stream.write(line + "\n")
        self._stream.flush()

    def _restore(self) -> None:
        if self._colour and not self._notebook and not self._restored:
            self._restored = True
            self._stream.write("\033[?25h\033[0m")
            self._stream.flush()

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def _paint(self, elapsed: float, done: bool = False) -> None:
        if not self._colour:
            self._paint_plain(elapsed)
            return
        lines = self._banner(elapsed)
        lines += self._machine(elapsed, done)
        lines.append("")
        lines.append(self._progress(elapsed))
        lines.append("")
        lines += self._gauges()
        if self._notebook:
            if self._view is None:
                self._view = _NotebookView()
            self._view.show("\n".join(lines))
            return
        # Home the cursor and overwrite: clearing first would make the whole
        # console flicker on every frame.
        body = "".join(f"{line}\033[K\n" for line in lines)
        self._stream.write(f"\033[H{body}\033[J")
        self._stream.flush()

    def _paint_plain(self, elapsed: float) -> None:
        cells = []
        for channel in self._channels:
            value = channel.train[-1] if channel.train else float("nan")
            cells.append(f"{channel.name}={value:.4g}")
        self._stream.write(
            f"epoch {self._epoch}/{self._epochs}  "
            + "  ".join(cells)
            + f"  [{_clock(elapsed)}]\n"
        )
        self._stream.flush()

    def _banner(self, elapsed: float) -> list[str]:
        # The wordmark two columns in, and one after it.
        mark_width = _wordmark_width() + 3
        canvas = _Canvas(mark_width, 5)
        _draw_wordmark(canvas, 2, 0, (elapsed * 0.32) % 1.7 - 0.35)

        out = []
        for index, row in enumerate(canvas.lines()):
            line = _Line().raw(row, mark_width)
            if index == 3:
                line.pad(mark_width + 4)
                line.add(" ".join("neuralize your model"), _shade(_BRAND[1], 0.85))
            out.append(str(line))
        return ["", *out, self._rule()]

    def _rule(self) -> str:
        """A horizontal rule carrying the logo gradient left to right."""
        out = []
        for x in range(self._width):
            out.append(_fg(_shade(_brand(x / max(1, self._width - 1)), 0.75)) + "━")
        return "".join(out) + "\033[0m"

    def _machine(self, elapsed: float, done: bool) -> list[str]:
        bay = _loss_graph(self._history, self._BAY, self._BAY_ROWS)

        done_epochs = max(1, self._epoch)
        rate = done_epochs / max(1e-6, elapsed)
        eta = (self._epochs - self._epoch) / rate if rate > 0 else float("nan")
        spinner = _SPINNER[int(elapsed * 12) % len(_SPINNER)]
        status = (
            ("✔ COMPLETE", (0x7C, 0xE0, 0xA8))
            if done
            else (f"{spinner} RUNNING", _BRAND[1])
        )

        telemetry = [
            ("MODEL", self._model_name, _BONE),
            ("EPOCH", f"{self._epoch} / {self._epochs}", _BONE),
            ("ELAPSED", _clock(elapsed), _BONE),
            ("REMAINING", "00:00:00" if done else _clock(eta), _BONE),
            ("THROUGHPUT", f"{rate:.2f} ep/s", _BONE),
            ("DRIVE LOAD", None, None),
            ("STATUS", status[0], status[1]),
        ]

        out = []
        for index in range(self._BAY_ROWS):
            line = _Line().raw(bay[index], self._BAY)
            line.add("  ").add("┃", _STEEL_DIM).add("  ")
            slot = index - (self._BAY_ROWS - len(telemetry)) // 2
            if 0 <= slot < len(telemetry):
                label, value, colour = telemetry[slot]
                line.add(f"{label:<11}", _STEEL)
                if value is None:
                    line.raw(self._load_bar(), 12)
                else:
                    line.add(value, colour, bold=label in ("EPOCH", "STATUS"))
            out.append(str(line))
        return ["", *out]

    def _load_bar(self, cells: int = 12) -> str:
        filled = int(round(self._load * cells))
        out = []
        for index in range(cells):
            if index < filled:
                colour = _lerp(
                    _brand(index / max(1, cells - 1)), _EMBER, self._load * 0.5
                )
                out.append(_fg(colour) + "▰")
            else:
                out.append(_fg(_STEEL_DIM) + "▱")
        return "".join(out) + "\033[0m"

    def _progress(self, elapsed: float) -> str:
        label = f" {100.0 * self._epoch / self._epochs:5.1f}% "
        track = self._width - len(label) - 2
        done = int(track * self._epoch / self._epochs)
        line = _Line().add("▐", _STEEL_DIM)
        for index in range(track):
            if index < done:
                colour = _brand(index / max(1, track - 1))
                # A travelling sheen on the filled bar, plus a hot leading edge
                # where the run is actually working.
                sheen = math.exp(
                    -(((index - (elapsed * 14) % (track + 40)) / 5.0) ** 2)
                )
                if index >= done - 2:
                    colour = _lerp(colour, _EMBER, 0.55)
                line.add("█", _shade(colour, 1.0 + 0.6 * sheen))
            else:
                line.add("░", _STEEL_DIM)
        line.add("▌", _STEEL_DIM).add(label, _BONE, bold=True)
        return str(line)

    def _gauges(self) -> list[str]:
        out = []
        for channel in self._channels:
            trend = channel.trend()
            arrow, colour = (
                ("▼", (0x7C, 0xE0, 0xA8))
                if trend < -1e-9
                else ("▲", (0xFF, 0x8A, 0x8A))
                if trend > 1e-9
                else ("=", _STEEL)
            )
            train = channel.train[-1] if channel.train else None
            val = channel.val[-1] if channel.val else None

            line = _Line().add("  ")
            line.add(
                f"{channel.name[:14]:<14}", _BRAND[1], bold=channel.name == "Total"
            )
            line.raw(channel.sparkline(self._spark), self._spark)
            line.add("  train ", _STEEL).add(f"{_format(train).strip():>10}", _BONE)
            line.add("  val ", _STEEL).add(f"{_format(val).strip():>10}", _BONE)
            line.add("  ").add(f"{arrow} {abs(trend) * 100:5.1f}%", colour)
            out.append(str(line))
        return out

    def _plaque(self, elapsed: float) -> list[str]:
        """The permanent record left in the scrollback once the machine stops."""
        title = " nnodely Training Complete "
        lines = [title.center(80, "=") if not self._colour else self._stamp(title)]
        lines.append(f"{'Model:':<30}{self._model_name}")
        lines.append(f"{'Epochs completed:':<30}{self._epoch} / {self._epochs}")
        lines.append(f"{'Total time of Training:':<30}{elapsed}")
        for channel in self._channels:
            if channel.train:
                value = f"{channel.train[-1]:.6e}"
                if channel.val:
                    value += f"   (val {channel.val[-1]:.6e})"
                lines.append(f"{'Final ' + channel.name + ':':<30}{value}")
        lines.append("=" * 80 if not self._colour else self._stamp(""))
        return lines

    def _stamp(self, title: str) -> str:
        text = title.center(80, "=")
        return (
            "".join(
                _fg(_shade(_brand(i / 79.0), 0.9)) + ch for i, ch in enumerate(text)
            )
            + "\033[0m"
        )


def _supports_colour(stream: Any) -> bool:
    """True when it is safe to write 24-bit colour and cursor moves - but in
    a notebook, whose kernel sets ``FORCE_COLOR``: see :func:`_is_notebook`."""
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
# Notebooks - colour, but no cursor moves
# =====================================================================

#: How many times a second a printer redraws its output in a notebook: each
#: redraw is a message of the kernel to the notebook.
_NOTEBOOK_FPS = 5.0


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


class _NotebookView:
    """One output of a notebook cell, replaced in place as a terminal screen
    is repainted. The notebook renders the colours of its text."""

    def __init__(self):
        self._handle = None

    def show(self, text: str) -> None:
        from IPython.display import display

        bundle = {"text/plain": text}
        if self._handle is None:
            self._handle = display(bundle, raw=True, display_id=True)
        else:
            self._handle.update(bundle, raw=True)


# =====================================================================
# Tables - what a model prints of its inputs, outputs, Parameters, ...
# =====================================================================

#: The characters the box of a table is drawn with.
_BOX = set("┏┳┓┃┣╋┫┡╇┻┩│├┬┴┼┤└┘━─")


def _table(
    header: tuple[str, ...],
    sections: list[tuple[str, list[tuple[str, ...]]]],
    total: tuple[str, ...] | None = None,
) -> str:
    """A table as nnodely prints one: ``header`` in heavy lines, then the
    rows of each section - a (title, rows) pair - under a row of its title
    when it has one, and, given, the ``total`` row last, under a rule. The
    columns line up across the sections. In colour (see
    :func:`_shows_colour`) the box takes the brand gradient, the titles the
    brand colour."""
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
