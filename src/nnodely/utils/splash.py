"""The splash nnodely prints when it is imported.

One band, 80 columns by 6 lines: a neural network of 3 inputs, 5 hidden
neurons and 2 outputs feeds the ``nn`` of the wordmark, and the wordmark
drives, with a shaft, the base of a mass held by a spring and a damper. The
``nn`` in the gradient of nnodely, the network in sky blue and violet from
where it joins the ``nn``, the other letters white, the machine from yellow
to red. Under the wordmark, ending at the tail of the ``y``, the versions of
nnodely and of Keras and the backend Keras runs on.

It shows in a terminal or a notebook, and never when the output is piped or
captured. ``NNODELY_SPLASH=0`` turns it off.
"""

from __future__ import annotations

import colorsys
import os
import sys

import keras

from nnodely.utils.printers import (
    _BONE,
    _STEEL,
    _brand,
    _fg,
    _is_notebook,
    _shows_colour,
)

#: The width of the splash: a standard line.
_WIDTH = 80

#: The network - its neurons (●), every connection between two layers and its
#: joint into the ``nn`` in Braille dots - the shaft, and the mass, spring
#: and damper: the wordmark is drawn between them. Over its top the network
#: flows on, in dots of ordinary width, with neurons here and there, broken
#: by the stems of d and l, down to the spring; a branch drops to a neuron
#: in the hole of d, another to a neuron in the middle of y. Every line of the network
#: holds as many Braille characters, some blank: a terminal that draws them
#: wider than a column shifts every line alike, and keeps them aligned.
_MACHINE = """
  ⠀⠀⠀⣀●⣤⣤⡤⠔⠒⠒⠒:˙˙:●·:˙˙˙:●·:˙˙˙:··:●˙:···:    ··●˙˙●·    ˙:●·:.
⠀●⣴⣮⢽⣯●⡬⢮⠭⣿●⡖⠪·::····:::····:·●····::···●·    ···::··  : :···●:·. ┃       ┏━━━┓
 ●⣾⣿⢽⣿●⢜⣛⢝⡏⠀⠈⠢                      ::                 ˙·.     ˙:·┃╲╱╲╱╲╱╲┫   ┃
⠀●⢾⡿⢽⡿●⠮⡽⠭⣿●⠒⠉                       ˙····●              ˙˙˙●   ━━┫       ┃ m ┃
  ⠀⠈⠉⠛●⠞⠊⠉⠀⠀⠀⠀                               ˙                    ┃━━┳━┓━━┫   ┃
   ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀                                                    ┃  ┗━┛  ┗━━━┛
""".strip("\n").split("\n")

#: Where the wordmark is drawn over the machine: its first line and column.
_LOGO_LINE, _LOGO_COLUMN = 0, 14

#: The wordmark, a pixel per ``#``, drawn two pixels to a line: the
#: ascenders of d and l on its first two lines, the tail of y on its last.
#: Its ``nn`` fills the first columns.
_LOGO = """
.............................##.........##
.............................##.........##
.............................##.........##
.............................##.........##
.#####...#####...#####...######..#####..##.##...##
##...##.##...##.##...##.##...##.##...##.##.##...##
##...##.##...##.##...##.##...##.#######.##.##...##
##...##.##...##.##...##.##...##.##......##.##...##
##...##.##...##.##...##.##...##.##......##.##...##
##...##.##...##..#####...######..######.##..######
................................................##
............................................#####
""".strip("\n").split("\n")
#: The columns and the lines of the ``nn``, in the gradient of nnodely.
_NN_COLUMNS, _NN_LINES = 15, range(2, 5)
#: Where the network is joined to the ``nn``: its gradient starts there.
_JOIN = (13, 2)
#: The columns the flows over the wordmark run between: from the network to
#: the wall the spring is fixed to.
_FLOW_START, _WALL = 14, 66


#: The half block a pair of pixels, one above the other, is drawn with.
_HALVES = {
    (False, False): " ",
    (True, False): "▀",
    (False, True): "▄",
    (True, True): "█",
}


def _wordmark_lines() -> list[str]:
    """The wordmark in half blocks, two rows of pixels to a line."""
    width = max(len(row) for row in _LOGO)
    rows = [row.ljust(width, ".") for row in _LOGO]
    return [
        "".join(_HALVES[(a == "#", b == "#")] for a, b in zip(top, bottom))
        for top, bottom in zip(rows[::2], rows[1::2])
    ]


def _hsv(hue: float, saturation: float = 0.6) -> tuple:
    red, green, blue = colorsys.hsv_to_rgb(hue % 1.0, saturation, 1.0)
    return (round(red * 255), round(green * 255), round(blue * 255))


def _colour(x: int, y: int) -> tuple:
    """The colour of the machine at column ``x`` and line ``y``. The network
    starts from the sky blue of the ``nn`` where it is joined to it, and
    turns violet diagonally away from there. The flows over the wordmark run
    from blue through violet, magenta and orange to the yellow of the
    spring, and the machine from that yellow to red at the edge."""
    if x < _FLOW_START:
        away = 0.6 * (_JOIN[0] - x) / _JOIN[0] + 0.4 * abs(y - _JOIN[1]) / 2
        return _brand(away)
    if x < _WALL:
        t = (x - _FLOW_START) / (_WALL - _FLOW_START)
        return _hsv(0.62 + 0.54 * t)
    t = (x - _WALL) / (_WIDTH - _WALL)
    return _hsv(0.16 * (1 - t), 0.85)


def _logo_colour(x: int, y: int) -> tuple:
    """The colour of the wordmark at column ``x`` and line ``y``: the ``nn``
    in the gradient of nnodely, sky blue on top and violet at its feet, the
    other letters white."""
    if x < _LOGO_COLUMN + _NN_COLUMNS:
        return _brand((y - _NN_LINES.start) / (len(_NN_LINES) - 1))
    return _BONE


def splash(colour: bool = False) -> str:
    """The splash as text, in colour when ``colour``."""
    from nnodely import __version__

    # One cell per character, with its colour: the machine, the wordmark
    # over it, and the versions on the last line, up to the tail of the y.
    wordmark = _wordmark_lines()
    cells: list[list[tuple[str, tuple | None]]] = [
        [(" ", None)] * _WIDTH for _ in _MACHINE
    ]
    for y, row in enumerate(_MACHINE):
        for x, char in enumerate(row):
            if char != " ":
                cells[y][x] = (char, _colour(x, y))
    for index, row in enumerate(wordmark):
        y = _LOGO_LINE + index
        for x, char in enumerate(row, start=_LOGO_COLUMN):
            if char != " ":
                cells[y][x] = (char, _logo_colour(x, y))
    about = (
        f"v{__version__} · Keras {keras.__version__} · "
        f"backend: {keras.backend.backend()}"
    )
    tail = _LOGO_COLUMN + len(wordmark[-1]) - len(wordmark[-1].lstrip())
    for x, char in enumerate(about, start=tail - 1 - len(about)):
        cells[-1][x] = (char, _STEEL)

    lines = []
    for row in cells:
        if colour:
            text = "".join(c if rgb is None else _fg(rgb) + c for c, rgb in row)
            lines.append(text.rstrip() + "\033[0m")
        else:
            lines.append("".join(c for c, _ in row).rstrip())
    return "\n".join(lines)


def show_splash() -> None:
    """Print the splash, in a terminal or a notebook, unless
    ``NNODELY_SPLASH=0``."""
    if os.environ.get("NNODELY_SPLASH", "1") == "0":
        return
    stream = sys.stdout
    try:
        interactive = _is_notebook(stream) or stream.isatty()
    except Exception:
        interactive = False
    if interactive:
        print(splash(colour=_shows_colour(stream)))
