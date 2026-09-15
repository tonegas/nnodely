import keras
import numpy as np
import pytest

from nnodely import Input, Output, Fir, Modely, DataLoader
from nnodely.utils.printers import TinyPrinter, LegacyPrinter, _resolve_printer


def _single_minimizer_model():
    x = Input("printer_x", dim=1)
    t = Input("printer_t", dim=1)
    out = Output("printer_pred", Fir(out_features=1)([x.sw(5)]))
    model = Modely("printer_model", inputs=[x, t], outputs=[out])
    model.minimize("curv_error", source=out, target=t.last())
    model.build()

    n = 40
    source = {
        "printer_x": np.arange(n, dtype=np.float32),
        "printer_t": np.arange(n, dtype=np.float32),
    }
    return model, DataLoader(model, source=source)


def test_legacy_printer_reproduces_the_original_table(capsys):
    printer = LegacyPrinter(epochs=6000, minimizers=[("curv_error", "curv")])

    ## 6000 epochs over at most 100 rows is one row every 60 epochs
    assert printer.stride == 60

    printer.on_train_begin()
    printer.on_epoch_end(
        59,
        {
            "curv_loss": 2.099e-06,
            "val_curv_loss": 1.226e-06,
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
    printer = LegacyPrinter(epochs=1000, minimizers=[("a", "a")], max_rows=10)
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
    printer = LegacyPrinter(epochs=7, minimizers=[("a", "a")], max_rows=3)
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
    ## Keras reports no per-output loss for a single-output model, so the
    ## minimizer column mirrors the total one
    printer = LegacyPrinter(epochs=1, minimizers=[("only", "only_out")])
    printer.on_train_begin()
    printer.on_epoch_end(0, {"loss": 0.25})
    row = capsys.readouterr().out.splitlines()[-1]

    assert row.split("|")[2:6] == ["     0.25", "    -    ", "     0.25", "    -    "]


def test_legacy_printer_columns_stay_aligned_for_wide_values(capsys):
    printer = LegacyPrinter(epochs=1, minimizers=[("a", "a")])
    printer.on_train_begin()
    printer.on_epoch_end(0, {"loss": -1.23456789e-123, "val_loss": 9.87654321e-321})
    lines = capsys.readouterr().out.splitlines()

    ## Precision is dropped instead of the table widening
    assert all(len(line) == printer.width for line in lines)


def test_legacy_printer_reports_the_training_time(capsys):
    printer = LegacyPrinter(epochs=1, minimizers=[("a", "a")])
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
        LegacyPrinter(epochs=10, minimizers=[("a", "a")], max_rows=0)


def test_train_resolves_the_printer_argument():
    model, _ = _single_minimizer_model()

    assert isinstance(
        _resolve_printer("tiny", 10, model.minimizers, model.name), TinyPrinter
    )

    legacy = _resolve_printer("legacy", 10, model.minimizers, model.name)
    assert isinstance(legacy, LegacyPrinter)
    ## Columns are named after the minimizers and keyed by their keras output
    assert legacy.minimizers == [("curv_error", "printer_pred")]
    assert legacy.epochs == 10

    ## None is silent, and a callback is taken as given
    assert (
        type(_resolve_printer(None, 10, model.minimizers, model.name))
        is keras.callbacks.Callback
    )
    given = LegacyPrinter(epochs=3, minimizers=[])
    assert _resolve_printer(given, 10, model.minimizers, model.name) is given

    with pytest.raises(ValueError):
        _resolve_printer("loud", 10, model.minimizers, model.name)


@pytest.mark.slow
def test_train_with_the_legacy_printer_prints_a_table(capsys):
    model, data = _single_minimizer_model()
    model.train(
        train_data=data, epochs=4, batch_size=8, val_data=data, printer="legacy"
    )
    out = capsys.readouterr().out

    ## Keras must stay silent so that only the printer writes to the terminal
    assert "nnodely Training" in out
    assert "\nEpoch 1/4" not in out
    rows = [line for line in out.splitlines() if line.startswith("|") and "/4" in line]
    assert len(rows) == 4
