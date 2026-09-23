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
            minimizers=[
                (minimizer["name"], minimizer["source"].name)
                for minimizer in minimizers
            ],
        )
    if printer == "nnodely":
        return NNodelyPrinter(
            epochs=epochs,
            minimizers=[
                (minimizer["name"], minimizer["source"].name)
                for minimizer in minimizers
            ],
            model_name=name,
        )
    raise ValueError(
        f"Unknown printer {printer!r}: expected 'tiny', 'legacy', "
        "'nnodely', None, or a keras.callbacks.Callback."
    )


class TinyPrinter(keras.callbacks.Callback):
    """Repaint a single-screen summary of the current epoch.

    The terminal is cleared on every epoch, so only the latest numbers are on
    screen and nothing scrolls. Use :class:`LegacyPrinter` when the history of
    the run matters more than the current value.
    """

    def __init__(self, epochs: int | None = None):
        super().__init__()
        self.epochs = epochs
        self.start_time = 0.0

    def on_train_begin(self, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # Clear terminal
        sys.stdout.write("\033[H\033[J")
        sys.stdout.flush()

        elapsed = time.time() - self.start_time

        sep = "─" * 70

        if self.params is None:
            self.params = {"epochs": self.epochs}
        print(sep)
        print(f"Epoch {epoch + 1}/{self.params['epochs']}   Elapsed: {elapsed:.1f}s")
        print(sep)

        for k, v in sorted(logs.items()):
            print(f"{k:<30} {v:>12.3e}")

        print(sep)


class LegacyPrinter(keras.callbacks.Callback):
    """Print the scrolling loss table of the original nnodely trainer.

    One row per sampled epoch, one column pair (train, val) per minimizer plus
    a final pair for the total loss::

        ================= nnodely Training =================
        |  Epoch   |     curv_error    |       Total       |
        |          |        Loss       |        Loss       |
        |          |  train  |   val   |  train  |   val   |
        |--------------------------------------------------|
        | 60/6000  |2.099e-06|1.226e-06|2.099e-06|1.226e-06|

    Rows are sampled every ``epochs / max_rows`` epochs, so a 6000-epoch run
    prints the same number of lines as a 60-epoch one instead of flooding the
    terminal. The final epoch is always printed, whatever the stride, so the
    table ends on the numbers the model actually stopped at.

    Parameters
    ----------
    epochs:
        Total number of epochs, used for the stride and the ``n/total`` label.
    minimizers:
        ``(display name, keras output name)`` pairs, in column order. The
        display name is the minimizer's name; the output name is what Keras
        prefixes its per-output loss with. When a model has a single output,
        Keras reports only the total loss and that column mirrors it.
    max_rows:
        Upper bound on the number of printed rows.
    """

    def __init__(
        self,
        epochs: int,
        minimizers: Sequence[tuple[str, str]] = (),
        max_rows: int = 100,
    ):
        super().__init__()
        if max_rows < 1:
            raise ValueError(f"max_rows must be a positive integer, got {max_rows}.")
        self.epochs = int(epochs)
        self.minimizers = list(minimizers)
        self.max_rows = int(max_rows)
        self.stride = max(1, -(-self.epochs // self.max_rows))
        self.start_time = 0.0
        self.rows_printed = 0

        self.groups = [name for name, _ in self.minimizers] + ["Total"]
        self.width = 2 + _EPOCH_CELL + len(self.groups) * (2 * _CELL + 2)

    # ------------------------------------------------------------------
    # Callback hooks
    # ------------------------------------------------------------------

    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        for line in self._header():
            print(line, flush=True)

    def on_epoch_end(self, epoch, logs=None):
        last = epoch + 1 >= self.epochs
        if not last and (epoch + 1) % self.stride:
            return
        # A row is left unterminated so that whatever ends the training - an
        # early-stopping notice - reads as a continuation of its own row.
        prefix = "\n" if self.rows_printed else ""
        print(prefix + self._row(epoch, logs or {}), end="", flush=True)
        self.rows_printed += 1

    def on_train_end(self, logs=None):
        elapsed = time.time() - self.start_time
        print()
        print(" nnodely Training Time ".center(80, "="))
        print(f"{'Total time of Training:':<30}{elapsed}")
        print("=" * 80)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _header(self) -> list[str]:
        group_width = 2 * _CELL + 1

        def group_row(cells: list[str]) -> str:
            return "|" + " " * _EPOCH_CELL + "|" + "|".join(cells) + "|"

        names = [name[:group_width].center(group_width) for name in self.groups]
        return [
            " nnodely Training ".center(self.width, "="),
            "|" + "Epoch".center(_EPOCH_CELL) + "|" + "|".join(names) + "|",
            group_row(["Loss".center(group_width)] * len(self.groups)),
            group_row(
                ["train".center(_CELL) + "|" + "val".center(_CELL)] * len(self.groups)
            ),
            "|" + "-" * (self.width - 2) + "|",
        ]

    def _row(self, epoch: int, logs: dict[str, Any]) -> str:
        keys = [f"{output}_loss" for _, output in self.minimizers] + ["loss"]
        cells = []
        for key in keys:
            # Keras only emits per-output losses when the model has more than
            # one output; with a single one the total is the only loss there is.
            train = logs.get(key, logs.get("loss"))
            val = logs.get(f"val_{key}", logs.get("val_loss"))
            cells.append(_format(train) + "|" + _format(val))
        label = f"{epoch + 1}/{self.epochs}".center(_EPOCH_CELL)
        return "|" + label + "|" + "|".join(cells) + "|"


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
    pixels square on a normal terminal grid and is what lets the gears come out
    round instead of squashed.
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


def _draw_gear(
    canvas: _Canvas,
    cx: float,
    cy: float,
    radius: float,
    teeth: int,
    phase: float,
    spokes: int = 4,
    glow: float = 0.0,
) -> None:
    """Paint one spur gear, anti-aliased by supersampling.

    A pixel's brightness is its coverage of the gear body, which is what keeps
    the rim looking like a circle at this resolution instead of a staircase.
    ``glow`` heats the rim towards ember as the machine works harder.
    """
    tooth = radius * 0.24
    rim = radius * 0.60
    hub = radius * 0.30
    bore = radius * 0.13
    reach = radius + tooth + 1.0
    offsets = (-0.25, 0.25)

    def inside(px: float, py: float) -> bool:
        dx, dy = px - cx, py - cy
        r = math.hypot(dx, dy)
        if r > reach or r < bore:
            return False
        angle = math.atan2(dy, dx)
        # Trapezoidal teeth riding on the root circle: they only ever add to
        # the radius, so the rim behind them stays unbroken as they turn.
        wave = math.cos(teeth * (angle - phase))
        outer = radius + tooth * max(0.0, min(1.0, (wave - 0.1) * 3.2))
        if r > outer:
            return False
        if r >= rim or r <= hub:
            return True
        return math.cos(spokes * (angle - phase)) > 0.80

    y0 = max(0, int(cy - reach) - 1)
    y1 = min(canvas.height - 1, int(cy + reach) + 1)
    x0 = max(0, int(cx - reach) - 1)
    x1 = min(canvas.cols - 1, int(cx + reach) + 1)

    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            hits = sum(inside(x + ox, y + oy) for ox in offsets for oy in offsets)
            if not hits:
                continue
            coverage = hits / 4.0
            depth = (y - (cy - reach)) / (2.0 * reach)
            colour = _brand(depth)
            if glow > 0.0:
                r = math.hypot(x - cx, y - cy)
                if r > rim:
                    colour = _lerp(colour, _EMBER, glow * 0.55)
            canvas.set(x, y, _shade(colour, 0.34 + 0.66 * coverage))


def _draw_wordmark(canvas: _Canvas, x0: int, y0: int, sweep: float) -> None:
    """Paint 'nnodely' in the logo gradient with a highlight sweeping across."""
    width = len(_WORDMARK) * 6
    for index, letter in enumerate(_WORDMARK):
        rows = _GLYPHS[letter]
        for dy, row in enumerate(rows):
            for dx, cell in enumerate(row):
                if cell != "#":
                    continue
                x = x0 + index * 6 + dx
                y = y0 + dy
                # The gradient runs down the glyphs exactly as it does in the
                # logo; the sweep is a soft specular band travelling right.
                colour = _brand((dy - 1) / 7.0)
                distance = abs((x - x0) / max(1, width) - sweep)
                highlight = math.exp(-((distance * 6.0) ** 2))
                canvas.set(x, y, _shade(colour, 1.0 + 0.75 * highlight))


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
        return 0.0 if first == 0 else (last - first) / abs(first)

    def sparkline(self, width: int) -> str:
        values = list(self.train)[-width:]
        if not values:
            return " " * width
        # A loss falls over orders of magnitude, so the spark is drawn on a log
        # scale - otherwise everything after the first few epochs is one flat
        # line pinned to the bottom of the cell.
        floor = min(v for v in values if v > 0) if any(v > 0 for v in values) else 1.0
        scaled = [math.log10(max(v, floor * 1e-3)) for v in values]
        low, high = min(scaled), max(scaled)
        span = high - low
        out = []
        for index, value in enumerate(scaled):
            level = 0.0 if span <= 0 else (value - low) / span
            block = _SPARKS[min(len(_SPARKS) - 1, int(level * len(_SPARKS)))]
            colour = _brand(0.15 + 0.85 * (1.0 - level))
            fade = 0.45 + 0.55 * (index + 1) / len(scaled)
            out.append(_fg(_shade(colour, fade)) + block)
        return "".join(out) + "\033[0m" + " " * (width - len(values))


class NNodelyPrinter(keras.callbacks.Callback):
    """A machine room for your training run.

    Renders the run as a live industrial console painted in the nnodely logo
    gradient: the wordmark with a highlight sweeping across it, a meshing gear
    train whose speed and rim heat follow how fast the loss is actually
    falling, a gradient progress bar, and a log-scale spark per minimizer.

    The gears are drawn with half-block pixels - two square pixels stacked in
    every character cell - and anti-aliased by supersampling, which is what
    makes them come out round on a terminal grid.

    Everything is repainted in place at a capped frame rate, so a fast run
    animates smoothly without the terminal becoming the bottleneck. On a
    non-terminal (a log file, a CI job) or with ``NO_COLOR`` set it falls back
    to one plain line per sampled epoch, so nothing downstream sees escapes.
    """

    #: Columns of the gear bay, and how many character rows it is tall.
    _BAY = 34
    _BAY_ROWS = 10

    def __init__(
        self,
        epochs: int,
        minimizers: Sequence[tuple[str, str]] = (),
        model_name: str = "model",
        width: int | None = None,
        fps: float = 24.0,
        stream: Any = None,
    ):
        super().__init__()
        self.epochs = max(1, int(epochs))
        self.model_name = model_name
        self.fps = max(1.0, float(fps))
        self.stream = stream if stream is not None else sys.stdout
        self.colour = _supports_colour(self.stream)

        terminal = shutil.get_terminal_size(fallback=(80, 24)).columns
        self.width = max(72, min(110, int(width or terminal - 2)))
        self.spark = max(10, min(28, self.width - 62))

        self.channels = [
            _Channel(name, f"{output}_loss", self.spark) for name, output in minimizers
        ] + [_Channel("Total", "loss", self.spark)]

        self.start_time = 0.0
        self.last_paint = 0.0
        self.epoch = 0
        self.load = 0.0
        self.phase = 0.0
        self.last_total: float | None = None
        self._restored = False
        # Without a terminal there is nothing to repaint, so the console
        # degrades to one line per sampled epoch rather than 24 lines a second
        # of log file.
        self._plain_stride = max(1, self.epochs // 40)

    # ------------------------------------------------------------------
    # Callback hooks
    # ------------------------------------------------------------------

    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        self.last_paint = 0.0
        if not self.colour:
            return
        # A crash or a Ctrl-C must never leave the user staring at a terminal
        # with no cursor, so the restore is armed before anything is hidden.
        atexit.register(self._restore)
        self.stream.write("\033[?25l\033[2J")
        self.stream.flush()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch = epoch + 1

        total = logs.get("loss")
        for channel in self.channels:
            channel.push(
                logs.get(channel.key, logs.get("loss")),
                logs.get(f"val_{channel.key}", logs.get("val_loss")),
            )

        # The machine works harder the faster the loss is falling: one epoch's
        # relative improvement, smoothed, drives the gear speed and rim heat.
        if total is not None and self.last_total:
            gain = (self.last_total - float(total)) / abs(self.last_total)
            self.load = max(0.0, min(1.0, 0.82 * self.load + 18.0 * max(0.0, gain)))
        if total is not None:
            self.last_total = float(total)

        now = time.time()
        final = self.epoch >= self.epochs
        if self.colour:
            if not final and now - self.last_paint < 1.0 / self.fps:
                return
        elif not final and self.epoch % self._plain_stride:
            return
        self.phase += (now - self.last_paint if self.last_paint else 0.04) * (
            1.1 + 4.5 * self.load
        )
        self.last_paint = now
        self._paint(now - self.start_time)

    def on_train_end(self, logs=None):
        elapsed = time.time() - self.start_time
        if self.colour:
            self._paint(elapsed, done=True)
        self._restore()
        self.stream.write("\n")
        for line in self._plaque(elapsed):
            self.stream.write(line + "\n")
        self.stream.flush()

    def _restore(self) -> None:
        if self.colour and not self._restored:
            self._restored = True
            self.stream.write("\033[?25h\033[0m")
            self.stream.flush()

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def _paint(self, elapsed: float, done: bool = False) -> None:
        if not self.colour:
            self._paint_plain(elapsed)
            return
        lines = self._banner(elapsed)
        lines += self._machine(elapsed, done)
        lines.append("")
        lines.append(self._progress(elapsed))
        lines.append("")
        lines += self._gauges()
        # Home the cursor and overwrite: clearing first would make the whole
        # console flicker on every frame.
        body = "".join(f"{line}\033[K\n" for line in lines)
        self.stream.write(f"\033[H{body}\033[J")
        self.stream.flush()

    def _paint_plain(self, elapsed: float) -> None:
        cells = []
        for channel in self.channels:
            value = channel.train[-1] if channel.train else float("nan")
            cells.append(f"{channel.name}={value:.4g}")
        self.stream.write(
            f"epoch {self.epoch}/{self.epochs}  "
            + "  ".join(cells)
            + f"  [{_clock(elapsed)}]\n"
        )
        self.stream.flush()

    def _banner(self, elapsed: float) -> list[str]:
        mark_width = len(_WORDMARK) * 6 + 2
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
        for x in range(self.width):
            out.append(_fg(_shade(_brand(x / max(1, self.width - 1)), 0.75)) + "━")
        return "".join(out) + "\033[0m"

    def _machine(self, elapsed: float, done: bool) -> list[str]:
        canvas = _Canvas(self._BAY, self._BAY_ROWS)
        big_r, big_teeth = 7.2, 9
        small_r, small_teeth = 4.6, 6
        centre = float(self._BAY_ROWS)
        _draw_gear(canvas, 10.0, centre, big_r, big_teeth, self.phase, 3, self.load)
        _draw_gear(
            canvas,
            10.0 + big_r + small_r + 0.8,
            centre,
            small_r,
            small_teeth,
            -self.phase * big_teeth / small_teeth
            + math.pi * (small_teeth - 1) / small_teeth,
            3,
            self.load,
        )
        bay = canvas.lines()

        done_epochs = max(1, self.epoch)
        rate = done_epochs / max(1e-6, elapsed)
        eta = (self.epochs - self.epoch) / rate if rate > 0 else float("nan")
        spinner = _SPINNER[int(elapsed * 12) % len(_SPINNER)]
        status = (
            ("✔ COMPLETE", (0x7C, 0xE0, 0xA8))
            if done
            else (f"{spinner} RUNNING", _BRAND[1])
        )

        telemetry = [
            ("MODEL", self.model_name, _BONE),
            ("EPOCH", f"{self.epoch} / {self.epochs}", _BONE),
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
        filled = int(round(self.load * cells))
        out = []
        for index in range(cells):
            if index < filled:
                colour = _lerp(
                    _brand(index / max(1, cells - 1)), _EMBER, self.load * 0.5
                )
                out.append(_fg(colour) + "▰")
            else:
                out.append(_fg(_STEEL_DIM) + "▱")
        return "".join(out) + "\033[0m"

    def _progress(self, elapsed: float) -> str:
        label = f" {100.0 * self.epoch / self.epochs:5.1f}% "
        track = self.width - len(label) - 2
        done = int(track * self.epoch / self.epochs)
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
        for channel in self.channels:
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
            line.raw(channel.sparkline(self.spark), self.spark)
            line.add("  train ", _STEEL).add(f"{_format(train).strip():>10}", _BONE)
            line.add("  val ", _STEEL).add(f"{_format(val).strip():>10}", _BONE)
            line.add("  ").add(f"{arrow} {abs(trend) * 100:5.1f}%", colour)
            out.append(str(line))
        return out

    def _plaque(self, elapsed: float) -> list[str]:
        """The permanent record left in the scrollback once the machine stops."""
        title = " nnodely Training Complete "
        lines = [title.center(80, "=") if not self.colour else self._stamp(title)]
        lines.append(f"{'Model:':<30}{self.model_name}")
        lines.append(f"{'Epochs completed:':<30}{self.epoch} / {self.epochs}")
        lines.append(f"{'Total time of Training:':<30}{elapsed}")
        for channel in self.channels:
            if channel.train:
                value = f"{channel.train[-1]:.6e}"
                if channel.val:
                    value += f"   (val {channel.val[-1]:.6e})"
                lines.append(f"{'Final ' + channel.name + ':':<30}{value}")
        lines.append("=" * 80 if not self.colour else self._stamp(""))
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
    """True when it is safe to write 24-bit colour and cursor moves."""
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
