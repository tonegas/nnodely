import io
import re

import keras
import numpy as np
import pytest

from nnodely import Input, Output, Fir, Modely, DataLoader
from nnodely.utils.printers import (
    LegacyPrinter,
    NNodelyPrinter,
    TinyPrinter,
    _resolve_printer,
)


def _single_minimizer_model():
    x = Input("printer_x", dim=1, tw=5)
    t = Input("printer_t", dim=1, tw=1)
    out = Output("printer_pred", Fir(features=1)([x]))
    model = Modely("printer_model", inputs=[x, t], outputs=[out])
    model.minimize("curv_error", source=out, target=t)
    model.build()

    n = 40
    source = {
        "printer_x": np.arange(n, dtype=np.float32),
        "printer_t": np.arange(n, dtype=np.float32),
    }
    return model, DataLoader(model, source=source)


def _rows(out):
    """The epoch rows of a loss table: the cells of each, stripped."""
    return [
        [cell.strip() for cell in line.split("│")[1:-1]]
        for line in out.splitlines()
        if line.startswith("│") and "/" in line
    ]


def test_legacy_printer_draws_the_loss_table(capsys):
    printer = LegacyPrinter(epochs=6000, minimizers=["curv_error"], max_rows=100)

    ## 6000 epochs over at most 100 rows is one row every 60 epochs
    assert printer._stride == 60

    printer.on_train_begin()
    printer.on_epoch_end(
        59,
        {
            "curv_error_loss": 2.099e-06,
            "val_curv_error_loss": 1.226e-06,
            "loss": 2.099e-06,
            "val_loss": 1.226e-06,
        },
    )
    lines = capsys.readouterr().out.splitlines()

    ## Drawn as the tables of a model are, the name of each minimizer over
    ## its train and val columns
    assert lines == [
        "┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┓",
        "┃ epoch      ┃       curv_error      ┃         total         ┃",
        "┃            ┣━━━━━━━━━━━┳━━━━━━━━━━━╋━━━━━━━━━━━┳━━━━━━━━━━━┫",
        "┃            ┃     train ┃       val ┃     train ┃       val ┃",
        "┡━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━┩",
        "│ 60/6000    │ 2.099e-06 │ 1.226e-06 │ 2.099e-06 │ 1.226e-06 │",
    ]
    assert all(len(line) == printer._width for line in lines)


def test_legacy_printer_keeps_the_row_count_bounded(capsys):
    printer = LegacyPrinter(epochs=1000, minimizers=["a"], max_rows=10)
    printer.on_train_begin()
    for epoch in range(1000):
        printer.on_epoch_end(epoch, {"loss": 1.0, "a_loss": 1.0})
    printer.on_train_end()

    rows = _rows(capsys.readouterr().out)
    assert len(rows) == 10
    ## The last epoch is always reported, whatever the stride
    assert rows[-1][0] == "1000/1000"


def test_legacy_printer_always_prints_the_final_epoch(capsys):
    ## 7 epochs over at most 3 rows gives a stride of 3: epochs 3, 6 and then 7
    printer = LegacyPrinter(epochs=7, minimizers=["a"], max_rows=3)
    printer.on_train_begin()
    for epoch in range(7):
        printer.on_epoch_end(epoch, {"loss": 1.0})
    printer.on_train_end()

    assert [row[0] for row in _rows(capsys.readouterr().out)] == ["3/7", "6/7", "7/7"]


def test_legacy_printer_falls_back_to_the_total_loss(capsys):
    ## A minimizer missing from the logs mirrors the total loss
    printer = LegacyPrinter(epochs=1, minimizers=["only"])
    printer.on_train_begin()
    printer.on_epoch_end(0, {"loss": 0.25})

    (row,) = _rows(capsys.readouterr().out)
    assert row[1:] == ["0.25", "-", "0.25", "-"]


def test_legacy_printer_columns_stay_aligned_for_wide_values(capsys):
    printer = LegacyPrinter(epochs=1, minimizers=["a"])
    printer.on_train_begin()
    printer.on_epoch_end(0, {"loss": -1.23456789e-123, "val_loss": 9.87654321e-321})
    printer.on_train_end()
    lines = capsys.readouterr().out.splitlines()

    ## Precision is dropped instead of the table widening
    assert all(len(line) == printer._width for line in lines)


def test_legacy_printer_reports_the_training_time(capsys):
    printer = LegacyPrinter(epochs=1, minimizers=["a"])
    printer.on_train_begin()
    printer.on_epoch_end(0, {"loss": 1.0})
    printer.on_train_end()
    lines = capsys.readouterr().out.splitlines()

    ## The last row of the table, across it
    assert lines[-3].startswith("├") and lines[-1].startswith("└")
    (seconds,) = re.fullmatch(r"│ training time: (\S+) s +│", lines[-2]).groups()  # type: ignore
    assert float(seconds) >= 0.0


def test_legacy_printer_is_coloured_where_the_tables_are(capsys, monkeypatch):
    def run():
        printer = LegacyPrinter(epochs=2, minimizers=["a"])
        printer.on_train_begin()
        for epoch in range(2):
            printer.on_epoch_end(epoch, {"loss": 1.0, "a_loss": 1.0})
        printer.on_train_end()
        return capsys.readouterr().out

    plain = run()
    assert "\x1b[" not in plain  # the tests run with NO_COLOR
    monkeypatch.delenv("NO_COLOR")
    monkeypatch.setenv("FORCE_COLOR", "1")
    coloured = run()
    assert "\x1b[" in coloured
    ## Only the time it took differs
    strip = re.compile(r"\x1b\[[0-9;]*m|training time: \S+")
    assert strip.sub("", coloured) == strip.sub("", plain)


def test_legacy_printer_rejects_a_non_positive_max_rows():
    with pytest.raises(ValueError):
        LegacyPrinter(epochs=10, minimizers=["a"], max_rows=0)


def test_train_resolves_the_printer_argument():
    model, _ = _single_minimizer_model()

    assert isinstance(
        _resolve_printer("tiny", 10, model._minimizers, model.name), TinyPrinter
    )

    legacy = _resolve_printer("legacy", 10, model._minimizers, model.name)
    assert isinstance(legacy, LegacyPrinter)
    ## Columns are named after the minimizers, which log their losses by name
    assert legacy._minimizers == ["curv_error"]
    assert legacy._epochs == 10

    ## None is silent, and a callback is taken as given
    assert (
        type(_resolve_printer(None, 10, model._minimizers, model.name))
        is keras.callbacks.Callback
    )
    given = LegacyPrinter(epochs=3, minimizers=[])
    assert _resolve_printer(given, 10, model._minimizers, model.name) is given

    with pytest.raises(ValueError):
        _resolve_printer("loud", 10, model._minimizers, model.name)


def test_tiny_printer_writes_no_terminal_codes_into_a_file(capsys):
    # Once cleared the screen unconditionally: a redirected run was full of
    # escape sequences.
    printer = TinyPrinter(epochs=2)
    printer.on_train_begin()
    printer.on_epoch_end(0, {"loss": 1.0})
    printer.on_epoch_end(1, {"loss": 0.5})

    out = capsys.readouterr().out
    assert "Epoch 2/2" in out
    assert "\033" not in out


def test_nnodely_printer_arms_the_cursor_restore_only_while_training(monkeypatch):
    # Once registered at every run and never removed, each hook keeping its
    # printer, and the model it trained, alive until the interpreter exits.
    registered, unregistered = [], []
    monkeypatch.setattr("atexit.register", registered.append)
    monkeypatch.setattr("atexit.unregister", unregistered.append)
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.delenv("NO_COLOR", raising=False)
    stream = io.StringIO()
    printer = NNodelyPrinter(epochs=1, minimizers=["a"], stream=stream)

    for _ in range(2):
        printer.on_train_begin()
        printer.on_epoch_end(0, {"loss": 1.0, "a_loss": 1.0})
        printer.on_train_end()

    assert len(registered) == 2
    assert unregistered == registered
    # The cursor is shown again at the end of each run, not only the first.
    assert stream.getvalue().count("\033[?25h") == 2


@pytest.mark.parametrize(
    "losses", [[1.0, float("inf")], [float("inf")], [float("nan")]]
)
def test_a_loss_that_is_not_finite_is_drawn_not_raised(losses):
    # Its log once raised "cannot convert float NaN to integer", which aborted
    # train() on a terminal.
    from nnodely.utils.printers import _Channel

    channel = _Channel("fit", "fit_loss", span=10)
    for loss in losses:
        channel.push(loss, None)
    assert isinstance(channel.sparkline(10), str)
    assert channel.trend() == 0.0


def test_the_nnodely_printer_charts_the_total_loss():
    from nnodely.utils.printers import _loss_graph

    def chart(losses, rows):
        return [
            re.sub(r"\x1b\[[0-9;]*m", "", line)
            for line in _loss_graph(losses, 34, rows)
        ]

    # A loss falling over three decades, a column per epoch.
    lines = chart([10.0 ** -(epoch / 4) for epoch in range(13)], 10)
    assert all(len(line) == 34 for line in lines)
    # Its highest and lowest value on the axis.
    assert lines[0].startswith("        1 ┤") and lines[-1].startswith("    0.001 ┤")
    # On a log scale: the first column fills the chart, the last only its floor.
    columns = [line[11:24] for line in lines]
    assert [column[0] for column in columns] == ["█"] * 10
    assert [column[-1] for column in columns] == [" "] * 9 + ["▁"]
    # A loss that is not finite leaves its column empty.
    assert [line[11:14] for line in chart([1.0, float("nan"), 2.0], 2)] == [
        "  █",
        "▁ █",
    ]


def test_tables_are_coloured_where_colour_shows(monkeypatch):
    x = Input("tab_colour_x", tw=1)
    model = Modely("tab_colour", inputs=[x], outputs=[Output("tab_colour_y", x * 2)])
    plain = str(model.inputs)
    assert "\x1b[" not in plain  # the tests run with NO_COLOR

    # In a terminal, or forced: the same table, in colour.
    monkeypatch.delenv("NO_COLOR")
    monkeypatch.setenv("FORCE_COLOR", "1")
    coloured = str(model.inputs)
    assert "\x1b[" in coloured
    assert re.sub(r"\x1b\[[0-9;]*m", "", coloured) == plain
    # NO_COLOR wins, as it does for the printers.
    monkeypatch.setenv("NO_COLOR", "1")
    assert str(model.inputs) == plain


def test_tables_are_coloured_in_a_notebook_not_wherever_ipykernel_is_imported(
    monkeypatch,
):
    import sys
    import types

    x = Input("tab_nb_x", tw=1)
    model = Modely("tab_nb", inputs=[x], outputs=[Output("tab_nb_y", x * 2)])
    monkeypatch.delenv("NO_COLOR")
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    # Imported, but the output is a pipe.
    monkeypatch.setitem(sys.modules, "ipykernel", types.ModuleType("ipykernel"))
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    assert "\x1b[" not in str(model.inputs)
    # The output stream of a notebook kernel.
    OutStream = type("OutStream", (io.StringIO,), {"__module__": "ipykernel.iostream"})
    monkeypatch.setattr(sys, "stdout", OutStream())
    assert "\x1b[" in str(model.inputs)


def test_a_model_prints_as_a_table_of_its_inputs_outputs_and_minimizers():
    x, y = Input("repr_x", tw=1), Input("repr_y", tw=1)
    model = Modely("repr_model", inputs=[x, y], outputs=[Output("repr_out", x + y)])
    model.minimize("fit", "repr_out", Input("repr_ref", tw=1))
    model.minimize("reg", "repr_out", 1.0)
    # Its name heads the table; the names of each kind are inline: every
    # Input it reads, as inputs lists them, the target it is not declared
    # with included.
    assert repr(model) == "\n".join(
        [
            "┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┓",
            "┃ model      ┃ repr_model               ┃",
            "┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━┩",
            "│ inputs     │ repr_x, repr_y, repr_ref │",
            "│ outputs    │ repr_out                 │",
            "│ minimizers │ fit, reg                 │",
            "└────────────┴──────────────────────────┘",
        ]
    )
    assert str(model) == repr(model)
    # A model without minimizers.
    bare = Modely("repr_bare", inputs=[x], outputs=[Output("repr_bare_out", x * 2)])
    assert repr(bare).splitlines()[5] == "│ minimizers │ -             │"


def _notebook(monkeypatch):
    """Run as a notebook does: the output stream of a kernel, which sets
    FORCE_COLOR, and a display recording the text of each output it shows."""
    import sys
    import types

    shown = []

    class Handle:
        def update(self, bundle, raw):
            shown.append(bundle["text/plain"])

    def display(bundle, raw, display_id):
        shown.append(bundle["text/plain"])
        return Handle()

    module = types.ModuleType("IPython.display")
    module.display = display  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "IPython.display", module)
    OutStream = type("OutStream", (io.StringIO,), {"__module__": "ipykernel.iostream"})
    monkeypatch.setattr(sys, "stdout", OutStream())
    monkeypatch.delenv("NO_COLOR")
    monkeypatch.setenv("FORCE_COLOR", "1")
    return sys.stdout, shown


def test_the_nnodely_printer_redraws_one_output_in_a_notebook(monkeypatch):
    # A notebook ignores the cursor moves a terminal is repainted with: every
    # frame was appended to the cell, thousands in a run of a few minutes.
    stream, shown = _notebook(monkeypatch)
    monkeypatch.setattr("atexit.register", lambda _: pytest.fail("no cursor to show"))
    printer = NNodelyPrinter(epochs=3, minimizers=["a"])
    printer.on_train_begin()
    for epoch in range(3):
        printer.on_epoch_end(epoch, {"loss": 1.0 / (epoch + 1), "a_loss": 1.0})
    printer.on_train_end()

    # Every frame in one output, in colour, the last one the finished run.
    assert len(shown) >= 2 and all("\x1b[38;2" in frame for frame in shown)
    assert "COMPLETE" in shown[-1]
    # The cell's text holds no cursor moves and no frame, only the summary.
    out = stream.getvalue()  # type: ignore
    assert not any(code in out for code in ("\x1b[H", "\x1b[2J", "\x1b[?25"))
    assert "nnodely Training Complete" in re.sub(r"\x1b\[[0-9;]*m", "", out)


def test_the_tiny_printer_redraws_one_output_in_a_notebook(monkeypatch):
    stream, shown = _notebook(monkeypatch)
    printer = TinyPrinter(epochs=50)
    printer.on_train_begin()
    for epoch in range(50):
        printer.on_epoch_end(epoch, {"loss": 1.0})
    printer.on_train_end()

    # A few times a second, and the last epoch once training ends.
    assert 2 <= len(shown) < 50
    assert "Epoch 50/50" in shown[-1]
    assert stream.getvalue() == ""  # type: ignore
