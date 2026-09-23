import numpy as np
import pytest

from nnodely import DataLoader, Input, Modely, Output
from nnodely.core import validation
from nnodely.utils import validation_plot


def _identity_model(name, *, dim=None, window=1, loss="mse"):
    """A model whose prediction is its input, so the error is fully controlled."""
    x = Input(f"{name}_x", dim=dim)
    target = Input(f"{name}_target", dim=dim)
    prediction = Output(f"{name}_prediction", x.sw(window))

    model = Modely(name, inputs=[x], outputs=[prediction])
    reference = target if window == 1 else target.sw(window)
    model.minimize("tracking", source=prediction, target=reference, loss=loss)
    model.build()
    return model


def _data(model, name, x_values, target_values):
    return DataLoader(
        model,
        source={
            f"{name}_x": np.asarray(x_values),
            f"{name}_target": np.asarray(target_values),
        },
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_a_perfect_prediction_scores_as_perfect():
    values = np.linspace(0.0, 1.0, 32)[:, None]
    score = validation.score_signal(
        name="tracking",
        source="pred",
        target="ref",
        loss_fn=None,
        y_true=values,
        y_pred=values,
    )

    assert score.metrics["rmse"] == pytest.approx(0.0)
    assert score.metrics["fit_pct"] == pytest.approx(100.0)
    assert score.metrics["r2"] == pytest.approx(1.0)
    assert score.metrics["max_error"] == pytest.approx(0.0)


def test_metrics_measure_error_not_magnitude():
    true = np.array([[0.0], [1.0], [2.0], [3.0]])
    pred = true + 0.5
    score = validation.score_signal(
        name="t", source="p", target="r", loss_fn=None, y_true=true, y_pred=pred
    )

    assert score.metrics["rmse"] == pytest.approx(0.5)
    assert score.metrics["mae"] == pytest.approx(0.5)
    assert score.metrics["max_error"] == pytest.approx(0.5)
    ## A constant offset is a bias, which is what separates it from noise
    assert score.metrics["bias"] == pytest.approx(0.5)
    assert score.metrics["std_error"] == pytest.approx(0.0)
    ## RMSE as a percentage of the target's peak-to-peak range
    assert score.metrics["nrmse_pct"] == pytest.approx(100.0 * 0.5 / 3.0)


def test_relative_indicators_are_undefined_on_a_constant_target():
    ## Saying "100% fit" against a signal that never moves would be a lie
    score = validation.score_signal(
        name="t",
        source="p",
        target="r",
        loss_fn=None,
        y_true=np.ones((8, 1)),
        y_pred=np.ones((8, 1)) + 0.1,
    )

    assert score.metrics["rmse"] == pytest.approx(0.1)
    for key in ("fit_pct", "r2", "correlation", "nrmse_pct"):
        assert np.isnan(score.metrics[key])


def test_every_element_of_the_tensor_is_scored():
    ## (samples, dim, time) - an error hidden past the first scalar still counts
    true = np.zeros((4, 2, 3))
    pred = np.zeros((4, 2, 3))
    pred[2, 1, 2] = 1.0
    score = validation.score_signal(
        name="t", source="p", target="r", loss_fn=None, y_true=true, y_pred=pred
    )

    assert score.channels == 6
    assert score.metrics["max_error"] == pytest.approx(1.0)
    assert score.metrics["rmse"] == pytest.approx(np.sqrt(1.0 / 24.0))


def test_shapes_that_do_not_match_are_refused():
    ## Truncating to the shorter of the two hides exactly the mistakes that
    ## matter - a target window offset by one step
    with pytest.raises(ValueError, match="They must match"):
        validation.score_signal(
            name="t",
            source="p",
            target="r",
            loss_fn=None,
            y_true=np.zeros((4, 1)),
            y_pred=np.zeros((4, 3)),
        )


def test_non_finite_predictions_are_counted_and_not_hidden():
    pred = np.zeros((6, 1))
    pred[2] = np.nan
    score = validation.score_signal(
        name="t",
        source="p",
        target="r",
        loss_fn=None,
        y_true=np.linspace(0, 1, 6)[:, None],
        y_pred=pred,
    )

    assert score.metrics["non_finite"] == 1
    assert np.isnan(score.metrics["fit_pct"])


# ---------------------------------------------------------------------------
# Modely.validate
# ---------------------------------------------------------------------------


def test_validate_scores_every_minimizer():
    name = "score_model"
    model = _identity_model(name)
    values = np.linspace(0.0, 1.0, 20)
    result = model.validate(_data(model, name, values, values))

    assert result.model == name
    assert result.samples == 20
    assert set(result.metrics()) == {"tracking"}
    assert result["tracking"].metrics["rmse"] == pytest.approx(0.0, abs=1e-6)
    ## The result maps minimizer name to score
    assert list(result) == ["tracking"]
    assert len(result) == 1


def test_validate_reports_the_configured_loss():
    name = "loss_model"
    model = _identity_model(name, loss="mae")
    values = np.linspace(0.0, 1.0, 12)
    result = model.validate(_data(model, name, values, values + 0.25))

    assert result["tracking"].loss == "mean_absolute_error"
    assert result["tracking"].metrics["loss"] == pytest.approx(0.25, abs=1e-5)


def test_validate_names_the_source_and_the_target_signal():
    name = "label_model"
    model = _identity_model(name)
    values = np.linspace(0.0, 1.0, 10)
    score = model.validate(_data(model, name, values, values))["tracking"]

    assert score.source == f"{name}_prediction"
    assert score.target == f"{name}_target"


def test_validate_draws_nothing_unless_asked():
    name = "quiet_model"
    model = _identity_model(name)
    values = np.linspace(0.0, 1.0, 10)

    result = model.validate(_data(model, name, values, values))

    assert result.figures == []


def test_validate_writes_one_png_per_minimizer(tmp_path):
    name = "figure_model"
    model = _identity_model(name)
    values = np.linspace(0.0, 1.0, 16)

    result = model.validate(_data(model, name, values, values), out_dir=tmp_path)

    written = sorted(p.name for p in tmp_path.glob("*.png"))
    assert written == [f"{name}_tracking.png"]
    assert result.figures == [str(tmp_path / f"{name}_tracking.png")]


def test_validate_adds_a_history_figure_when_given_one(tmp_path):
    name = "history_model"
    model = _identity_model(name)
    values = np.linspace(0.0, 1.0, 16)
    data = _data(model, name, values, values)
    history = model.train(train_data=data, epochs=2, batch_size=4, printer=None)

    model.validate(data, out_dir=tmp_path, history=history)

    assert (tmp_path / f"{name}_history.png").exists()


def test_validate_shows_the_figures_on_request(monkeypatch):
    name = "show_model"
    model = _identity_model(name)
    values = np.linspace(0.0, 1.0, 10)
    shown = []
    monkeypatch.setattr("matplotlib.pyplot.show", lambda *a, **k: shown.append(True))

    result = model.validate(_data(model, name, values, values), show=True)

    assert shown == [True]
    ## Shown but not saved: there was nowhere to save to
    assert result.figures == []


def test_validate_rejects_an_empty_dataset():
    name = "empty_model"
    model = _identity_model(name)
    values = np.linspace(0.0, 1.0, 10)
    data = _data(model, name, values, values)
    data._num_steps = 0

    with pytest.raises(ValueError, match="empty"):
        model.validate(data)


def test_validate_requires_a_built_model_and_a_minimizer():
    x = Input("unbuilt_x")
    model = Modely("unbuilt", inputs=[x], outputs=[Output("unbuilt_out", x.sw(1))])

    with pytest.raises(ValueError, match="not built"):
        model.validate(None)

    model.build()
    with pytest.raises(ValueError, match="No minimizers"):
        model.validate(None)


def test_validate_reports_its_summary_and_returns_the_numbers(capsys):
    name = "summary_model"
    model = _identity_model(name)
    values = np.linspace(0.0, 1.0, 12)

    result = model.validate(_data(model, name, values, values + 0.1))
    ## Reported on the way out, and still available as data afterwards
    assert "nnodely Validation" in capsys.readouterr().out

    text = result.summary()
    assert text == repr(result)
    assert "nnodely Validation" in text
    for label in ("RMSE", "MAE", "Max error", "Bias", "FIT", "R2", "Correlation"):
        assert label in text
    ## Every line fits the 80-column banner the rest of nnodely prints in
    assert all(len(line) <= 80 for line in text.splitlines())


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def test_a_wide_signal_caps_the_drawn_lines_but_never_the_metrics():
    channels = validation_plot.MAX_LINES + 4
    true = np.zeros((20, channels))
    pred = np.tile(np.arange(channels, dtype=float), (20, 1))
    score = validation.score_signal(
        name="wide", source="p", target="r", loss_fn=None, y_true=true, y_pred=pred
    )

    figure = validation_plot.figure_signal(score)
    try:
        ## Metrics see every channel; the figure draws the worst few
        assert score.channels == channels
        assert score.metrics["max_error"] == pytest.approx(channels - 1)
        assert f"of {channels} channels" in figure.axes[0].get_title()
    finally:
        validation_plot.close_all([figure])


def test_the_response_and_error_panels_share_an_x_axis():
    values = np.linspace(0.0, 1.0, 20)[:, None]
    score = validation.score_signal(
        name="t",
        source="p",
        target="r",
        loss_fn=None,
        y_true=values,
        y_pred=values + 0.05,
    )

    figure = validation_plot.figure_signal(score)
    try:
        ## Zooming a stretch of the run has to zoom both panels
        response, error = figure.axes[0], figure.axes[1]
        assert error in response.get_shared_x_axes().get_siblings(response)
    finally:
        validation_plot.close_all([figure])


def test_the_history_figure_is_skipped_without_curves():
    assert validation_plot.figure_history({}) is None
    assert validation_plot.figure_history({"loss": []}) is None


def test_the_history_figure_draws_every_curve():
    figure = validation_plot.figure_history(
        {"loss": [1.0, 0.5, 0.25], "val_loss": [1.1, 0.6, 0.4]}
    )
    try:
        assert figure is not None
        assert {line.get_label() for line in figure.axes[0].lines} == {
            "loss",
            "val_loss",
        }
    finally:
        validation_plot.close_all([figure])
