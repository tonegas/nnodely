import matplotlib.pyplot as plt
import numpy as np
import pytest

from nnodely import DataLoader, Input, Linear, Loop, Modely, Output
from nnodely.core import validation
from nnodely.utils import validation_plot
from nnodely import TimeSelect


def _identity_model(name, *, dim=None, window=1, loss="mse"):
    """A model whose prediction is its input, so the error is fully controlled."""
    x = Input(f"{name}_x", dim=dim, sw=window)
    target = Input(f"{name}_target", dim=dim, sw=window)
    prediction = Output(f"{name}_prediction", x)

    model = Modely(name, inputs=[x], outputs=[prediction])
    reference = TimeSelect(-1)(target) if window == 1 else target
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
        name="t", source="p", target="r", y_true=true, y_pred=pred
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
        y_true=np.ones((8, 1)),
        y_pred=np.ones((8, 1)) + 0.1,
    )

    assert score.metrics["rmse"] == pytest.approx(0.1)
    for key in ("fit_pct", "r2", "correlation", "nrmse_pct"):
        assert np.isnan(score.metrics[key])


def test_every_element_of_the_tensor_is_scored():
    ## (samples, time, dim) - an error hidden past the first scalar still counts
    true = np.zeros((4, 3, 2))
    pred = np.zeros((4, 3, 2))
    pred[2, 2, 1] = 1.0
    score = validation.score_signal(
        name="t", source="p", target="r", y_true=true, y_pred=pred
    )

    ## A channel is a feature; every sample at every step is a point of it
    assert score.channels == 2
    assert score.points == 12
    assert score.metrics["max_error"] == pytest.approx(1.0)
    assert score.metrics["rmse"] == pytest.approx(np.sqrt(1.0 / 24.0))
    np.testing.assert_allclose(score.channel_metrics["rmse"], [0.0, np.sqrt(1 / 12)])


def test_shapes_that_do_not_match_are_refused():
    ## Truncating to the shorter of the two hides exactly the mistakes that
    ## matter - a target window offset by one step
    with pytest.raises(ValueError, match="They must match"):
        validation.score_signal(
            name="t",
            source="p",
            target="r",
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
        y_true=np.linspace(0, 1, 6)[:, None],
        y_pred=pred,
    )

    assert score.metrics["non_finite"] == 1
    assert np.isnan(score.metrics["fit_pct"])


def test_the_indicators_match_a_worked_example():
    ## e = [0, 0, 0, 1]: sum(e^2) = 1, and the target's spread
    ## sum((y - 1.5)^2) = 5, var(y) = 1.25, var(e) = 0.25 - 0.25^2 = 0.1875
    true = np.array([0.0, 1.0, 2.0, 3.0])[:, None]
    pred = np.array([0.0, 1.0, 2.0, 4.0])[:, None]
    metrics = validation.score_signal(
        name="t", source="p", target="r", y_true=true, y_pred=pred
    ).metrics

    assert metrics["rmse"] == pytest.approx(0.5)
    assert metrics["mae"] == pytest.approx(0.25)
    assert metrics["max_error"] == pytest.approx(1.0)
    assert metrics["bias"] == pytest.approx(0.25)
    assert metrics["std_error"] == pytest.approx(np.sqrt(0.1875))
    assert metrics["nrmse_pct"] == pytest.approx(100.0 * 0.5 / 3.0)
    assert metrics["fit_pct"] == pytest.approx(100.0 * (1.0 - 1.0 / np.sqrt(5.0)))
    assert metrics["r2"] == pytest.approx(1.0 - 1.0 / 5.0)
    assert metrics["vaf_pct"] == pytest.approx(100.0 * (1.0 - 0.1875 / 1.25))
    ## cov(y, y_hat) / (std(y) std(y_hat)), in sums: 6.5 / sqrt(5 * 8.75)
    assert metrics["correlation"] == pytest.approx(6.5 / np.sqrt(5.0 * 8.75))


def test_the_indicators_keep_the_identities_that_relate_them():
    rng = np.random.default_rng(0)
    true = rng.normal(2.0, 3.0, size=(500, 1))
    pred = 0.9 * true + rng.normal(0.4, 0.5, size=(500, 1))
    metrics = validation.score_signal(
        name="t", source="p", target="r", y_true=true, y_pred=pred
    ).metrics

    ## The error splits into a systematic part and a random one
    assert metrics["rmse"] ** 2 == pytest.approx(
        metrics["bias"] ** 2 + metrics["std_error"] ** 2
    )
    ## FIT and R2 weigh the same residual, as a norm and as its square
    assert (1.0 - metrics["fit_pct"] / 100.0) ** 2 == pytest.approx(1.0 - metrics["r2"])
    ## VAF is R2 that forgives a constant offset
    assert metrics["vaf_pct"] / 100.0 - metrics["r2"] == pytest.approx(
        metrics["bias"] ** 2 / np.var(true)
    )


def test_each_channel_is_scored_against_its_own_spread():
    ## A position in metres and a velocity a thousand times smaller: pooled,
    ## the position's spread made a prediction of the velocity by its mean
    ## alone look perfect
    t = np.linspace(0.0, 10.0, 200)
    true = np.stack([1000.0 * np.sin(t), 0.01 * np.cos(t)], axis=-1)
    pred = true.copy()
    pred[:, 1] = np.mean(true[:, 1])
    score = validation.score_signal(
        name="t", source="p", target="r", y_true=true, y_pred=pred
    )

    assert score.channels == 2
    np.testing.assert_allclose(score.channel_metrics["r2"], [1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(
        score.channel_metrics["fit_pct"], [100.0, 0.0], atol=1e-9
    )
    ## A prediction that never moves follows none of the target
    np.testing.assert_allclose(score.channel_metrics["correlation"], [1.0, 0.0])
    ## The whole signal: the relative indicators average the channels...
    assert score.metrics["r2"] == pytest.approx(0.5)
    assert score.metrics["fit_pct"] == pytest.approx(50.0)
    ## ... and the absolute ones pool every point of every channel
    assert score.metrics["rmse"] == pytest.approx(np.sqrt(np.mean((pred - true) ** 2)))
    assert score.metrics["max_error"] == pytest.approx(np.max(np.abs(pred - true)))


def test_a_constant_channel_is_left_out_of_the_relative_indicators():
    true = np.stack([np.full(20, 2.0), np.linspace(0.0, 1.0, 20)], axis=-1)
    pred = true + [0.1, 0.0]
    score = validation.score_signal(
        name="t", source="p", target="r", y_true=true, y_pred=pred
    )

    assert np.isnan(score.channel_metrics["r2"][0])
    assert score.channel_metrics["r2"][1] == pytest.approx(1.0)
    ## The channel that moves still has a fit to report
    assert score.metrics["r2"] == pytest.approx(1.0)
    np.testing.assert_allclose(score.channel_metrics["rmse"], [0.1, 0.0], atol=1e-12)


def _rollout_mask(lengths, steps):
    """The DataLoader's mask: one row per simulation, True on its real steps."""
    return np.arange(steps) < np.asarray(lengths)[:, np.newaxis]


def test_padded_points_are_left_out_of_every_indicator():
    ## Three simulations rolled out over 5 steps, (samples, steps, window, dim)
    real = _rollout_mask([5, 2, 4], 5)
    rng = np.random.default_rng(1)
    true = rng.normal(size=(3, 5, 1, 1))
    pred = true + rng.normal(scale=0.1, size=true.shape)
    ## Whatever a padded step holds - a rollout that diverged included - is
    ## not the model's error
    pred[~real] = np.array([1e6, -np.inf, np.nan, 7.0])[:, None, None]

    score = validation.score_signal(
        name="t",
        source="p",
        target="r",
        y_true=true,
        y_pred=pred,
        valid=real[:, :, None, None],
    )
    alone = validation.score_signal(
        name="t", source="p", target="r", y_true=true[real], y_pred=pred[real]
    )

    assert (score.points, score.padded) == (11, 4)
    assert score.metrics["non_finite"] == 0
    for key in ("rmse", "mae", "max_error", "bias", "std_error", "fit_pct", "r2"):
        assert score.metrics[key] == pytest.approx(alone.metrics[key]), key
    np.testing.assert_array_equal(score.y_pred, alone.y_pred)


def test_the_rollout_error_is_reported_step_by_step():
    ## Every real point is 0.1 off per step of the rollout, on both channels
    real = _rollout_mask([5, 2, 4], 5)
    true = np.zeros((3, 5, 1, 2))
    pred = np.broadcast_to(0.1 * np.arange(1.0, 6.0)[:, None, None], true.shape).copy()
    pred[~real] = 1e3

    score = validation.score_signal(
        name="t",
        source="p",
        target="r",
        y_true=true,
        y_pred=pred,
        valid=real[:, :, None, None],
    )

    assert score.step_rmse is not None
    np.testing.assert_allclose(
        score.step_rmse, np.repeat(0.1 * np.arange(1.0, 6.0)[:, None], 2, axis=1)
    )
    ## A signal with only its window has no rollout to follow
    window_only = validation.score_signal(
        name="t", source="p", target="r", y_true=true[:, 0], y_pred=pred[:, 0]
    )
    assert window_only.step_rmse is None


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
    x = Input("unbuilt_x", sw=1)
    model = Modely("unbuilt", inputs=[x], outputs=[Output("unbuilt_out", x)])

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


def test_validate_scores_each_channel_of_a_signal():
    name = "channels_model"
    model = _identity_model(name, dim=2)
    t = np.linspace(0.0, 10.0, 50)
    target = np.stack([1000.0 * np.sin(t), 0.01 * np.cos(t)], axis=-1)
    x = target.copy()
    x[:, 1] = np.mean(target[:, 1])

    result = model.validate(_data(model, name, x, target), verbose=False)
    score = result["tracking"]

    assert (score.channels, score.points) == (2, 50)
    np.testing.assert_allclose(score.channel_metrics["r2"], [1.0, 0.0], atol=1e-5)
    assert score.metrics["r2"] == pytest.approx(0.5, abs=1e-5)
    ## A column per channel, next to the whole signal's
    text = result.summary()
    assert "[0]" in text and "[1]" in text
    assert all(len(line) <= 80 for line in text.splitlines())


def _decay_model(name, seq_weights=None):
    """x[k + 1] = w x[k], rolled out over each whole simulation.

    The weight is 1 and the data decays by 0.8 a step, so the error grows
    along the rollout.
    """
    body_x = Input(f"{name}_body_x", sw=1)
    body_out = Output(
        f"{name}_body_out", Linear(features=1, kernel="ones", bias=False)(body_x)
    )
    body = Modely(f"{name}_body", inputs=[body_x], outputs=[body_out]).build()
    seed = Input(f"{name}_x", seq=-1, sw=1)
    loop = Loop(
        f=body,
        callback={body_x: body_out},
        length=6,
        init={body_x: seed},
    )()
    out = Output(f"{name}_out", loop)
    model = Modely(name, inputs=[seed], outputs=[out])
    model.minimize(
        "rollout", out, Input(f"{name}_target", seq=-1, sw=1), seq_weights=seq_weights
    )
    model.build()
    return model


def _simulations(name):
    """Three simulations of 6, 3 and 5 steps."""
    return [
        {f"{name}_x": signal, f"{name}_target": 0.8 * signal}
        for signal in (
            start * 0.8 ** np.arange(length)
            for start, length in ((1.0, 6), (2.0, 3), (-1.5, 5))
        )
    ]


def test_validate_reports_the_loss_train_reports_on_the_same_data():
    ## Padded simulations and step weights: masked and weighted the same way
    name = "same_loss"
    model = _decay_model(name, seq_weights=lambda steps: np.arange(1.0, steps + 1.0))
    data = DataLoader(model, source=_simulations(name), seq_length="full")
    assert data.mask is not None

    history = model.train(
        data,
        val_data=data,
        epochs=1,
        batch_size=len(data),
        optimizer="sgd",
        lr=0.0,
        shuffle=False,
        printer=None,
    )
    score = model.validate(data, verbose=False)["rollout"]

    assert score.metrics["loss"] == pytest.approx(
        history["val_rollout_loss"][0], rel=1e-5
    )


def test_a_short_last_batch_weighs_only_its_samples():
    ## 10 samples in batches of 4 leave a last batch of 2. The loss of each
    ## minimizer was logged as the plain mean of the batch losses, so it
    ## disagreed with Keras's own loss and with the loss of the dataset
    name = "batches_model"
    model = _identity_model(name)
    values = np.linspace(0.0, 1.0, 10)
    data = _data(model, name, values, values**2)

    history = model.train(
        train_data=data,
        val_data=data,
        epochs=1,
        batch_size=4,
        shuffle=False,
        printer=None,
    )
    loss = model.validate(data, verbose=False)["tracking"].metrics["loss"]

    expected = np.mean((values - values**2) ** 2)
    assert history["tracking_loss"][0] == pytest.approx(expected, rel=1e-5)
    assert history["val_tracking_loss"][0] == pytest.approx(expected, rel=1e-5)
    assert history["val_loss"][0] == pytest.approx(expected, rel=1e-5)
    assert loss == pytest.approx(expected, rel=1e-5)


def test_padded_simulations_are_scored_as_if_each_ran_alone():
    name = "fair_model"
    model = _decay_model(name)
    simulations = _simulations(name)

    result = model.validate(
        DataLoader(model, source=simulations, seq_length="full"), verbose=False
    )
    padded = result["rollout"]
    alone = [
        model.validate(
            DataLoader(model, source=[simulation], seq_length="full"), verbose=False
        )["rollout"]
        for simulation in simulations
    ]

    ## Every real step is scored once, and no padded one: 3 + 1 steps pad the
    ## two shorter simulations to the longest
    assert (padded.points, padded.padded) == (6 + 3 + 5, 4)
    assert all(score.padded == 0 for score in alone)
    reference = validation.score_signal(
        name="t",
        source="p",
        target="r",
        y_true=np.concatenate([score.y_true for score in alone]),
        y_pred=np.concatenate([score.y_pred for score in alone]),
    )
    for key, value in reference.metrics.items():
        if key != "loss":
            assert padded.metrics[key] == pytest.approx(value, rel=1e-5), key
    ## The error grows along the rollout, step by step
    assert padded.step_rmse is not None and padded.step_rmse.shape == (6, 1)
    assert padded.step_rmse[-1, 0] > padded.step_rmse[0, 0]
    text = result.summary()
    assert "(4 padded left out)" in text
    assert "RMSE first / last step" in text


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def test_a_wide_signal_caps_the_drawn_lines_but_never_the_metrics():
    channels = validation_plot.MAX_LINES + 4
    true = np.zeros((20, channels))
    pred = np.tile(np.arange(channels, dtype=float), (20, 1))
    score = validation.score_signal(
        name="wide", source="p", target="r", y_true=true, y_pred=pred
    )

    figure = validation_plot.figure_signal(score)
    try:
        ## Metrics see every channel; the figure draws the worst few
        assert score.channels == channels
        assert score.metrics["max_error"] == pytest.approx(channels - 1)
        assert f"of {channels} channels" in figure.axes[0].get_title()
    finally:
        plt.close(figure)


def test_the_response_and_error_panels_share_an_x_axis():
    values = np.linspace(0.0, 1.0, 20)[:, None]
    score = validation.score_signal(
        name="t",
        source="p",
        target="r",
        y_true=values,
        y_pred=values + 0.05,
    )

    figure = validation_plot.figure_signal(score)
    try:
        ## Zooming a stretch of the run has to zoom both panels
        response, error = figure.axes[0], figure.axes[1]
        assert error in response.get_shared_x_axes().get_siblings(response)
    finally:
        plt.close(figure)


def test_a_rollout_figure_draws_the_error_step_by_step():
    real = _rollout_mask([5, 2, 4], 5)
    true = np.zeros((3, 5, 1, 1))
    pred = np.broadcast_to(0.1 * np.arange(1.0, 6.0)[:, None, None], true.shape)
    score = validation.score_signal(
        name="t",
        source="p",
        target="r",
        y_true=true,
        y_pred=pred,
        valid=real[:, :, None, None],
    )

    figure = validation_plot.figure_signal(score)
    try:
        steps = figure.axes[-1]
        assert steps.get_title() == "error along the rollout"
        assert score.step_rmse is not None
        np.testing.assert_allclose(
            np.asarray(steps.lines[0].get_ydata()), score.step_rmse[:, 0]
        )
    finally:
        plt.close(figure)


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
        plt.close(figure)


def test_validate_prints_its_summary_unless_told_not_to(capsys):
    # Once printed on every call, with no way to keep a log or a loop quiet.
    name = "verbose_model"
    model = _identity_model(name)
    values = np.linspace(0.0, 1.0, 10)
    data = _data(model, name, values, values + 0.25)

    shown = model.validate(data)
    assert "nnodely Validation" in capsys.readouterr().out

    quiet = model.validate(data, verbose=False)
    assert capsys.readouterr().out == ""
    assert quiet["tracking"].metrics["mae"] == pytest.approx(
        shown["tracking"].metrics["mae"]
    )
