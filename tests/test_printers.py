import io
import re

import numpy as np
import pytest

from nnodely import Input, Output, Fir, Modely, DataLoader
from nnodely.utils.printers import _LossTable


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


def test_the_loss_table_draws_a_row_per_sampled_epoch(capsys):
    printer = _LossTable(epochs=6000, minimizers=["curv_error"], max_rows=100)

    ## 6000 epochs over at most 100 rows is one row every 60 epochs
    assert printer._stride == 60

    printer.on_train_begin()
    printer.on_epoch_end(
        59,
        {
            "loss_curv_error": 2.099e-06,
            "val_loss_curv_error": 1.226e-06,
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


def test_the_loss_table_keeps_the_row_count_bounded(capsys):
    printer = _LossTable(epochs=1000, minimizers=["a"], max_rows=10)
    printer.on_train_begin()
    for epoch in range(1000):
        printer.on_epoch_end(epoch, {"loss": 1.0, "loss_a": 1.0})
    printer.on_train_end()

    rows = _rows(capsys.readouterr().out)
    assert len(rows) == 10
    ## The last epoch is always reported, whatever the stride
    assert rows[-1][0] == "1000/1000"


def test_the_loss_table_always_prints_the_final_epoch(capsys):
    ## 7 epochs over at most 3 rows gives a stride of 3: epochs 3, 6 and then 7
    printer = _LossTable(epochs=7, minimizers=["a"], max_rows=3)
    printer.on_train_begin()
    for epoch in range(7):
        printer.on_epoch_end(epoch, {"loss": 1.0})
    printer.on_train_end()

    assert [row[0] for row in _rows(capsys.readouterr().out)] == ["3/7", "6/7", "7/7"]


def test_the_loss_table_leaves_a_loss_it_is_not_given_blank(capsys):
    ## A minimizer missing from the logs is blank: once it showed the total
    printer = _LossTable(epochs=1, minimizers=["only"])
    printer.on_train_begin()
    printer.on_epoch_end(0, {"loss": 0.25})

    (row,) = _rows(capsys.readouterr().out)
    assert row[1:] == ["-", "-", "0.25", "-"]


def test_the_loss_table_columns_stay_aligned_for_wide_values(capsys):
    printer = _LossTable(epochs=1, minimizers=["a"])
    printer.on_train_begin()
    printer.on_epoch_end(0, {"loss": -1.23456789e-123, "val_loss": 9.87654321e-321})
    printer.on_train_end()
    lines = capsys.readouterr().out.splitlines()

    ## Precision is dropped instead of the table widening
    assert all(len(line) == printer._width for line in lines)


def test_the_loss_table_reports_the_training_time(capsys):
    printer = _LossTable(epochs=1, minimizers=["a"])
    printer.on_train_begin()
    printer.on_epoch_end(0, {"loss": 1.0})
    printer.on_train_end()
    lines = capsys.readouterr().out.splitlines()

    ## The last row of the table, across it
    assert lines[-3].startswith("├") and lines[-1].startswith("└")
    (seconds,) = re.fullmatch(r"│ training time: (\S+) s +│", lines[-2]).groups()  # type: ignore
    assert float(seconds) >= 0.0


def test_the_loss_table_is_coloured_where_the_tables_are(capsys, monkeypatch):
    def run():
        printer = _LossTable(epochs=2, minimizers=["a"])
        printer.on_train_begin()
        for epoch in range(2):
            printer.on_epoch_end(epoch, {"loss": 1.0, "loss_a": 1.0})
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


def test_the_loss_table_rejects_a_non_positive_max_rows():
    with pytest.raises(ValueError):
        _LossTable(epochs=10, minimizers=["a"], max_rows=0)


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
    assert str(model) == "\n".join(
        [
            "┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┓",
            "┃ model      ┃ repr_model               ┃",
            "┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━┩",
            "│ inputs     │ repr_x, repr_y, repr_ref │",
            "│ outputs    │ repr_out                 │",
            "│ parameters │ 0                        │",
            "│ minimizers │ fit, reg                 │",
            "└────────────┴──────────────────────────┘",
        ]
    )
    # repr is one line, the fields of its row in a table of models.
    assert repr(model) == (
        "Modely(name='repr_model', inputs=['repr_x', 'repr_y', 'repr_ref'], "
        "outputs=['repr_out'], parameters=0, built=False)"
    )
    # A model without minimizers.
    bare = Modely("repr_bare", inputs=[x], outputs=[Output("repr_bare_out", x * 2)])
    assert str(bare).splitlines()[6] == "│ minimizers │ -             │"


def test_train_prints_the_loss_table_unless_verbose_is_false(capsys):
    model, data = _single_minimizer_model()

    model.train(data, epochs=2)
    assert "training time" in capsys.readouterr().out
    model.train(data, epochs=2, verbose=False)
    assert capsys.readouterr().out == ""
    with pytest.raises(TypeError, match="verbose is True or False"):
        model.train(data, verbose="yes")  # type: ignore[arg-type]
