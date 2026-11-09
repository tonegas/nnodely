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
    x = Input("printer_x", dim=1, sw=5)
    t = Input("printer_t", dim=1, sw=1)
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


def test_legacy_printer_reproduces_the_original_table(capsys):
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

    assert lines == [
        "================= nnodely Training =================",
        "|  Epoch   |     curv_error    |       Total       |",
        "|          |        Loss       |        Loss       |",
        "|          |  train  |   val   |  train  |   val   |",
        "|--------------------------------------------------|",
        "| 60/6000  |2.099e-06|1.226e-06|2.099e-06|1.226e-06|",
    ]
    assert all(len(line) == 52 for line in lines)


def test_legacy_printer_keeps_the_row_count_bounded(capsys):
    printer = LegacyPrinter(epochs=1000, minimizers=["a"], max_rows=10)
    printer.on_train_begin()
    for epoch in range(1000):
        printer.on_epoch_end(epoch, {"loss": 1.0, "a_loss": 1.0})
    printer.on_train_end()

    rows = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("|") and "/" in line
    ]
    assert len(rows) == 10
    ## The last epoch is always reported, whatever the stride
    assert rows[-1].startswith("|1000/1000")


def test_legacy_printer_always_prints_the_final_epoch(capsys):
    ## 7 epochs over at most 3 rows gives a stride of 3: epochs 3, 6 and then 7
    printer = LegacyPrinter(epochs=7, minimizers=["a"], max_rows=3)
    printer.on_train_begin()
    for epoch in range(7):
        printer.on_epoch_end(epoch, {"loss": 1.0})
    printer.on_train_end()

    rows = [
        line.split("|")[1].strip()
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("|") and "/" in line
    ]
    assert rows == ["3/7", "6/7", "7/7"]


def test_legacy_printer_falls_back_to_the_total_loss(capsys):
    ## A minimizer missing from the logs mirrors the total loss
    printer = LegacyPrinter(epochs=1, minimizers=["only"])
    printer.on_train_begin()
    printer.on_epoch_end(0, {"loss": 0.25})
    row = capsys.readouterr().out.splitlines()[-1]

    assert row.split("|")[2:6] == ["     0.25", "    -    ", "     0.25", "    -    "]


def test_legacy_printer_columns_stay_aligned_for_wide_values(capsys):
    printer = LegacyPrinter(epochs=1, minimizers=["a"])
    printer.on_train_begin()
    printer.on_epoch_end(0, {"loss": -1.23456789e-123, "val_loss": 9.87654321e-321})
    lines = capsys.readouterr().out.splitlines()

    ## Precision is dropped instead of the table widening
    assert all(len(line) == printer._width for line in lines)


def test_legacy_printer_reports_the_training_time(capsys):
    printer = LegacyPrinter(epochs=1, minimizers=["a"])
    printer.on_train_begin()
    printer.on_epoch_end(0, {"loss": 1.0})
    printer.on_train_end()
    lines = capsys.readouterr().out.splitlines()

    assert " nnodely Training Time " in lines[-3]
    assert lines[-2].startswith("Total time of Training:")
    assert float(lines[-2][30:]) >= 0.0
    assert lines[-1] == "=" * 80


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


def test_tables_are_coloured_where_colour_shows(monkeypatch):
    x = Input("tab_colour_x", sw=1)
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

    x = Input("tab_nb_x", sw=1)
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
    x, y = Input("repr_x", sw=1), Input("repr_y", sw=1)
    model = Modely("repr_model", inputs=[x, y], outputs=[Output("repr_out", x + y)])
    model.minimize("fit", "repr_out", Input("repr_ref", sw=1))
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
