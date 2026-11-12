"""Name tags: how the objects of a model are picked when names clash.

Nothing keeps two objects from sharing a name - layers named alike by hand, a
block and the model calling it, two copies of one loaded model. A call taking
a name looks it up among the objects the graph reaches, in that order: a name
one object carries picks it; one several carry raises listing them with
their index - the index column of the table - which picks one, ``parameters[2]``;
renaming one, ``obj.name = "new"``, gives each a name of its own. An item is
picked by its name or by its index, never both. Inside nnodely the objects themselves are
used, so an object renamed is still found: a renamed Input is read by its new
name, a renamed output returned under it.
"""

import json
import re

import keras
import numpy as np
import pytest

import nnodely
from conftest import to_numpy
from nnodely import (
    DataLoader,
    Fir,
    Gradient,
    Loop,
    Input,
    Integrate,
    Modely,
    ModelyLoop,
    Output,
    Parameter,
)
from nnodely.core.layer import Layer

_WINDOW = np.array([[[1.0], [2.0], [3.0]]], np.float32)


class Sin(Layer):
    """A layer of the user's own, named like nnodely's ``Sin``."""

    def __init__(self, name=None):
        super().__init__(name=name)

    def build_layer(self):
        return keras.layers.Lambda(lambda t: 100.0 * t)

    def output_shape(self, *inputs):
        return inputs[0].shape


def _block(name, k=2.0):
    """``k * fir(x)``: a Parameter ``k`` and a Fir ``fir``, named alike in
    every block made here."""
    x = Input("x", tw=3)
    fir = Fir(features=1, kernel="ones", bias=False, name="fir")([x])
    return Modely(
        name, inputs=[x], outputs=[Output("y", fir * Parameter("k", value=[k]))]
    )


def _two_ks():
    """A model whose block and own graph each have a Parameter ``k``: the
    model's own 10, then the block's 2."""
    block = _block("tag_inner").build()
    x = Input("x", tw=3)
    k = Parameter("k", value=[10.0])
    return Modely(
        "tag_outer", inputs=[x], outputs=[Output("total", block([x * k]))]
    ).build()


def _html_nodes(model, folder):
    model.export_html(folder)
    page = (folder / f"{model.name}.html").read_text()
    nodes = re.search(r"const nodes = new vis\.DataSet\((\[.*?\])\);\n", page, re.S)
    assert nodes is not None
    return json.loads(nodes.group(1))


# ---------------------------------------------------------------------------
# Picking by name, and by name and index
# ---------------------------------------------------------------------------


def test_a_name_one_object_carries_picks_it():
    block = _block("tag_alone").build()
    x = Input("tag_alone_x", tw=3)
    model = Modely(
        "tag_alone_model", inputs=[x], outputs=[Output("tag_out", block([x]))]
    )

    assert model.parameters["k"] is block.parameters["k"]
    assert model.models["tag_alone"] is block
    assert model.inputs["tag_alone_x"] is x
    assert model.outputs["tag_out"] is model._outputs[0]
    # Or by its index, the index column of the table.
    assert model.parameters[1] is model.parameters["k"]
    assert "│ 1   │ k " in str(model.parameters)


def test_a_name_several_parameters_carry_is_refused_listing_them():
    model = _two_ks()

    with pytest.raises(ValueError) as error:
        model.parameters["k"]
    message = str(error.value)
    assert "'k' is ambiguous, 2 Parameters carry it, at indices [0, 2]" in message
    # Each with its index and where it is, in the order the graph reaches
    # them: the model's own first, then those of the models it calls.
    assert "[0] k in 'tag_outer'; [2] k in 'tag_inner'" in message
    assert "Pick one by its index, parameters[i]" in message
    assert "parameters[i].name = 'new_name'" in message
    np.testing.assert_allclose(to_numpy(model.parameters[0].param), [10.0])
    np.testing.assert_allclose(to_numpy(model.parameters[2].param), [2.0])


def test_two_parameters_of_one_name_in_one_model_are_told_apart_by_index():
    x = Input("tag_one_x", tw=1)
    first, second = Parameter("gain", value=[2.0]), Parameter("gain", value=[3.0])
    model = Modely(
        "tag_one", inputs=[x], outputs=[Output("tag_one_y", x * first + x * second)]
    ).build()

    assert model.parameters[0] is first
    assert model.parameters[1] is second
    result = model({"tag_one_x": np.ones((1, 1, 1), np.float32)})["tag_one_y"]
    np.testing.assert_allclose(to_numpy(result), [[[5.0]]])


def test_an_unknown_name_or_index_is_refused():
    model = _two_ks()

    with pytest.raises(KeyError, match="no Parameter named 'nope'"):
        model.parameters["nope"]
    with pytest.raises(IndexError):
        model.parameters[3]
    # By name or by index, never both.
    for listed in (model.parameters, model.models, model.inputs, model.outputs):
        with pytest.raises(TypeError, match="by its name, a string, or by its index"):
            listed["k", 0]
    assert model.parameters[-1] is model.parameters[2]


def test_the_indices_are_the_same_once_saved_and_loaded(tmp_path):
    model = _two_ks()
    model.save(tmp_path / "two_ks")
    loaded = Modely.load(tmp_path / "two_ks")

    for index, value in ((0, 10.0), (2, 2.0)):
        assert loaded.parameters[index].name == "k"
        np.testing.assert_allclose(to_numpy(loaded.parameters[index].param), [value])
    np.testing.assert_allclose(
        to_numpy(loaded({"x": _WINDOW})["total"]),
        to_numpy(model({"x": _WINDOW})["total"]),
    )


def test_two_copies_of_one_loaded_model_are_picked_by_index(tmp_path):
    _block("tag_copy").build().save(tmp_path / "copy")
    left, right = Modely.load(tmp_path / "copy"), Modely.load(tmp_path / "copy")
    a = Input("tag_copy_a", tw=3)
    both = Modely(
        "tag_copies", inputs=[a], outputs=[Output("tag_copy_o", left([a]) + right([a]))]
    ).build()

    with pytest.raises(
        ValueError,
        match=r"'tag_copy' is ambiguous.*at indices \[0, 1\].*\[0\] tag_copy in 'tag_copies'",
    ):
        both.models["tag_copy"]
    assert both.models[0] is left
    assert both.models[1] is right
    # Each copy is its own: the Parameter of one is not the other's.
    assert both.parameters[1] is left.parameters["k"]
    assert both.parameters[3] is right.parameters["k"]


# ---------------------------------------------------------------------------
# freeze and unfreeze: by name; one of several by itself
# ---------------------------------------------------------------------------


def test_freeze_refuses_a_name_several_parameters_carry():
    model = _two_ks()

    # As every call taking a name: nothing is frozen, and the listing says
    # where each is, and how to freeze one.
    with pytest.raises(
        ValueError,
        match=r"'k' is ambiguous.*\(parameters\[0\]\) in 'tag_outer'.*"
        r"model\.parameters\[i\]\.freeze\(\)",
    ):
        model.freeze("k")
    with pytest.raises(ValueError, match="'k' is ambiguous"):
        model.unfreeze(["fir_kernel", "k"])
    assert all(p.trainable for p in model.parameters)

    model.parameters[2].freeze()
    assert model.parameters[0].trainable
    assert not model.parameters[2].trainable
    # Renamed, each is picked by its name.
    model.parameters[0].name = "k_outer"
    model.freeze("k")
    assert not model.parameters["k"].trainable
    model.freeze("k_outer")
    assert not model.parameters["k_outer"].trainable


def test_freeze_takes_names_only():
    model = _two_ks()
    model.parameters[0].name = "k_outer"

    model.freeze(["k_outer", "fir_kernel"])
    assert [p.name for p in model.parameters if not p.trainable] == [
        "k_outer",
        "fir_kernel",
    ]
    with pytest.raises(TypeError, match="got float"):
        model.freeze([1.5])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="got tuple"):
        model.unfreeze([("k", 0)])  # type: ignore[list-item]


def test_freeze_a_model_name_two_copies_carry(tmp_path):
    _block("tag_frozen").build().save(tmp_path / "frozen")
    left, right = Modely.load(tmp_path / "frozen"), Modely.load(tmp_path / "frozen")
    a = Input("tag_frozen_a", tw=3)
    both = Modely(
        "tag_frozen_both",
        inputs=[a],
        outputs=[Output("tag_frozen_o", left([a]) + right([a]))],
    ).build()

    with pytest.raises(ValueError, match="'tag_frozen' is ambiguous"):
        both.freeze("tag_frozen")
    assert all(p.trainable for p in both.parameters)
    both.models[1].freeze()
    assert [p.trainable for p in left.parameters] == [True, True]
    assert [p.trainable for p in right.parameters] == [False, False]


# ---------------------------------------------------------------------------
# minimize: a stream by name, or itself
# ---------------------------------------------------------------------------


def _twice():
    """One block called on two Inputs: two streams named 'y'."""
    block = _block("tag_twice_block").build()
    a, b = Input("tag_a", tw=3), Input("tag_b", tw=3)
    ya, yb = block([a]), block([b])
    model = Modely(
        "tag_twice", inputs=[a, b], outputs=[Output("oa", ya), Output("ob", yb)]
    )
    return model, ya, yb


def test_minimize_refuses_a_name_two_streams_carry():
    model, ya, yb = _twice()

    with pytest.raises(
        ValueError,
        match=r"'y' is ambiguous.*y \(an IntermediateOutput.*Pass the stream itself",
    ):
        model.minimize("fit", "y", 1.0)
    assert model._minimizers == []
    model.minimize("fit", yb, 1.0)
    assert model._minimizers[0]["source"] is yb
    with pytest.raises(TypeError, match="a Stream or the name of one"):
        model.minimize("fit_a", ("y", 0), 1.0)  # type: ignore[arg-type]


def test_two_calls_of_one_block_are_minimized_apart():
    # Once both streams were returned by the training model under one key,
    # their name, and the build refused them.
    model, ya, yb = _twice()
    ta, tb = Input("tag_ta", tw=1), Input("tag_tb", tw=1)
    model.minimize("fit_a", ya, ta).minimize("fit_b", yb, tb).build()

    data = {
        "tag_a": _WINDOW,
        "tag_b": 2 * _WINDOW,
        "tag_ta": np.zeros((1, 1, 1), np.float32),
        "tag_tb": np.zeros((1, 1, 1), np.float32),
    }
    result = model.validate(DataLoader(model, data, windowed=True), verbose=False)
    # y = 2 * (sum of the window): 12 for a, 24 for b, against 0.
    assert result["fit_a"].metrics["loss"] == pytest.approx(144.0)
    assert result["fit_b"].metrics["loss"] == pytest.approx(576.0)


def test_a_minimized_stream_named_like_an_output_is_its_own():
    x = Input("tag_same_x", tw=1)
    doubled = x * 2.0
    doubled.name = "tag_same_y"  # named like the output, another stream
    model = Modely("tag_same", inputs=[x], outputs=[Output("tag_same_y", x * 3.0)])
    model.minimize("fit", doubled, Input("tag_same_t", tw=1)).build()

    data = {
        "tag_same_x": np.ones((1, 1, 1), np.float32),
        "tag_same_t": np.zeros((1, 1, 1), np.float32),
    }
    # The doubled stream is compared, not the output: (2 - 0)^2.
    result = model.validate(DataLoader(model, data, windowed=True), verbose=False)
    assert result["fit"].metrics["loss"] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Inputs and outputs: the names of the data, so still one per model
# ---------------------------------------------------------------------------


def test_two_inputs_of_one_name_are_refused():
    first, second = Input("tag_twin", tw=1), Input("tag_twin", tw=1)
    model = Modely(
        "tag_twins", inputs=[first, second], outputs=[Output("o", first + second)]
    )

    # The data is keyed by input name: an index cannot tell two arrays apart.
    with pytest.raises(ValueError, match="two different Inputs named 'tag_twin'"):
        model.build()


def test_a_loaded_model_gives_its_own_inputs_by_name(tmp_path):
    z, v = Input("tag_z", tw=1), Input("tag_v", tw=3)
    model = Modely(
        "tag_reads_z",
        inputs=[v, z],
        outputs=[Output("tag_rz", Fir(features=1, kernel="ones", bias=False)([v]) + z)],
    ).build()
    model.save(tmp_path / "reads_z")
    loaded = Modely.load(tmp_path / "reads_z")

    z_loaded = loaded.inputs["tag_z"]
    assert any(node is z_loaded for node in loaded._order)
    # So a model calling it can read that same Input, and give it to the call.
    x = Input("tag_x", tw=3)
    outer = Modely(
        "tag_reads_too",
        inputs=[x, z_loaded],
        outputs=[Output("tag_o", loaded([x, z_loaded]) + z_loaded)],
    ).build()
    result = outer({"tag_x": _WINDOW, "tag_z": np.full((1, 1, 1), 5.0, np.float32)})
    np.testing.assert_allclose(to_numpy(result["tag_o"]), [[[6.0 + 5.0 + 5.0]]])
    assert loaded.outputs["tag_rz"] is loaded._outputs[0]
    with pytest.raises(KeyError, match="no Input named 'nope'"):
        loaded.inputs["nope"]


def test_a_kept_state_yields_its_name_to_an_input_of_the_user():
    # Integrate keeps its state in an Input named after it and its rate,
    # tag_tank_tag_rate: the user's own Input of that name keeps it.
    rate = Input("tag_rate", tw=1)
    plant = Modely(
        "tag_plant",
        inputs=[rate],
        outputs=[Output("tag_level", Integrate(dt=1.0, name="tag_tank")(rate))],
    ).build()
    q, mine = Input("tag_q", tw=1), Input("tag_tank_tag_rate", tw=1)
    sim = ModelyLoop(
        "tag_sim",
        inputs=[q, mine],
        outputs=[Output("tag_lv", plant([q])), Output("tag_mine", mine * 1.0)],
    ).build()

    result = sim(
        {
            "tag_q": np.ones((1, 4, 1, 1), np.float32),
            "tag_tank_tag_rate": np.full((1, 4, 1, 1), 100.0, np.float32),
        }
    )
    np.testing.assert_allclose(to_numpy(result["tag_lv"]).ravel(), [1, 2, 3, 4])
    np.testing.assert_allclose(to_numpy(result["tag_mine"]).ravel(), [100.0] * 4)


# ---------------------------------------------------------------------------
# Classes: saved with their module
# ---------------------------------------------------------------------------


def _sin_model():
    x = Input("tag_sin_x", tw=3)
    return Modely(
        "tag_sin", inputs=[x], outputs=[Output("tag_sin_y", nnodely.Sin(x))]
    ).build()


def test_a_class_of_the_user_named_like_nnodely_s_does_not_replace_it(tmp_path):
    # Once classes were saved by name alone, and loading took the last class
    # of that name: the Sin of this module, 100 x, in place of sin x.
    model = _sin_model()
    model.save(tmp_path / "sin")
    loaded = Modely.load(tmp_path / "sin")

    np.testing.assert_allclose(
        to_numpy(loaded({"tag_sin_x": _WINDOW})["tag_sin_y"]),
        np.sin(_WINDOW),
        rtol=1e-6,
    )


def test_a_class_saved_by_its_name_alone_loads_when_one_class_has_it(tmp_path):
    model = _sin_model()
    model.save(tmp_path / "bare")
    config = tmp_path / "bare" / "model.json"
    saved = json.loads(config.read_text())
    for node in saved["nodes"]:
        node["class_name"] = node["class_name"].rsplit(".", 1)[-1]
    config.write_text(json.dumps(saved))

    # Output and Input: one class each of that name.
    names = [node["class_name"] for node in saved["nodes"]]
    assert "Sin" in names and "Input" in names
    # Sin: nnodely's and this module's - which one is not known.
    with pytest.raises(ValueError, match=r"2 classes are named 'Sin'"):
        Modely.load(tmp_path / "bare")


# ---------------------------------------------------------------------------
# HTML: every node drawn apart, whatever its name
# ---------------------------------------------------------------------------


def test_html_draws_one_block_called_twice_as_two_calls(tmp_path):
    model, _, _ = _twice()
    model.build()

    # Two nodes each, under their own names: told apart by id, not renamed.
    nodes = _html_nodes(model, tmp_path)
    labels = [node["label"] for node in nodes]
    assert labels.count("tag_twice_block") == 2
    assert labels.count("y") == 2
    assert len({node["id"] for node in nodes}) == len(nodes)
    assert (tmp_path / "tag_twice__tag_twice_block.html").exists()
    assert (tmp_path / "tag_twice__tag_twice_block_2.html").exists()


def test_html_draws_two_layers_of_one_name_apart(tmp_path):
    x = Input("tag_html_x", tw=3)
    first, second = Fir(features=1, name="tag_f"), Fir(features=1, name="tag_f")
    model = Modely(
        "tag_html", inputs=[x], outputs=[Output("tag_html_y", first([x]) + second([x]))]
    ).build()

    labels = [node["label"] for node in _html_nodes(model, tmp_path)]
    assert labels.count("tag_f") == 2
    assert labels.count("tag_f_kernel") == 2


# ---------------------------------------------------------------------------
# Renaming: obj.name = "new", before or after the build
# ---------------------------------------------------------------------------


def _gain_model():
    x = Input("rn_x", tw=3)
    k = Parameter("rn_k", value=[2.0])
    y = Output("rn_y", Fir(features=1, kernel="ones", bias=False)([x]) * k)
    return Modely("rn_model", inputs=[x], outputs=[y]), x, k, y


def test_an_input_renamed_before_the_build_is_read_by_its_new_name():
    model, x, _, _ = _gain_model()
    x.name = "rn_speed"
    model.build()

    assert [node.name for node in model.inference_inputs] == ["rn_speed"]
    assert model.inference_model is not None
    assert [tensor.name for tensor in model.inference_model.inputs] == ["rn_speed"]
    np.testing.assert_allclose(
        to_numpy(model({"rn_speed": _WINDOW})["rn_y"]), [[[12.0]]]
    )
    np.testing.assert_allclose(to_numpy(model({x: _WINDOW})["rn_y"]), [[[12.0]]])


def test_an_input_renamed_after_the_build_is_read_by_its_new_name(tmp_path):
    model, x, _, y = _gain_model()
    model.minimize("rn_fit", y, Input("rn_t", tw=1))
    model.build()
    x.name = "rn_speed"

    # Called, the model reads it by its name now, without a new build.
    np.testing.assert_allclose(
        to_numpy(model({"rn_speed": _WINDOW})["rn_y"]), [[[12.0]]]
    )
    with pytest.raises(ValueError, match=r"reads the inputs \['rn_speed'\]"):
        model({"rn_x": _WINDOW})
    # Training data is keyed by the names of the build: built again first.
    source = {"rn_speed": np.arange(8.0), "rn_t": np.arange(8.0)}
    with pytest.raises(ValueError, match="'rn_x' is now 'rn_speed': build it again"):
        DataLoader(model, source)
    model.build()
    assert set(DataLoader(model, source).as_dict()) == {"rn_speed", "rn_t"}
    # Saved under its name now - without the minimizer and its target.
    model.save(tmp_path / "renamed")
    assert [node.name for node in Modely.load(tmp_path / "renamed").inputs] == [
        "rn_speed"
    ]


def test_an_output_renamed_after_the_build_returns_under_its_new_name():
    model, _, _, y = _gain_model()
    model.build()
    y.name = "rn_force"

    assert list(model({"rn_x": _WINDOW})) == ["rn_force"]
    assert model.outputs["rn_force"] is y
    # Renamed like another output, the results could not hold both.
    x = Input("rn_x2", tw=1)
    first, second = Output("rn_a", x * 1.0), Output("rn_b", x * 2.0)
    pair = Modely("rn_pair", inputs=[x], outputs=[first, second]).build()
    second.name = "rn_a"
    with pytest.raises(ValueError, match=r"several outputs named \['rn_a'\]"):
        pair({"rn_x2": np.ones((1, 1, 1), np.float32)})


def test_renaming_into_a_clash_and_out_of_it(tmp_path):
    model = _two_ks()
    # Two 'k': picked by index, then one renamed, each by its name.
    with pytest.raises(ValueError, match=r"parameters\[i\]\.name = 'new_name'"):
        model.parameters["k"]
    inner = model.parameters[2]
    inner.name = "k_inner"
    np.testing.assert_allclose(to_numpy(model.parameters["k"].param), [10.0])
    assert model.parameters["k_inner"] is inner
    model.freeze("k_inner")
    assert not inner.trainable
    # Saved and loaded under its new name.
    model.save(tmp_path / "renamed_k")
    loaded = Modely.load(tmp_path / "renamed_k")
    np.testing.assert_allclose(to_numpy(loaded.parameters["k_inner"].param), [2.0])
    # Renamed like the other, it is ambiguous again.
    inner.name = "k"
    with pytest.raises(ValueError, match="'k' is ambiguous"):
        model.freeze("k")


def test_a_model_renamed_is_picked_and_saved_by_its_new_name(tmp_path):
    block = _block("rn_block").build()
    a = Input("rn_a", tw=3)
    outer = Modely("rn_outer", inputs=[a], outputs=[Output("rn_o", block([a]))]).build()
    block.name = "rn_plant"

    assert outer.models["rn_plant"] is block
    outer.freeze("rn_plant")
    assert not any(p.trainable for p in outer.parameters)
    outer.save(tmp_path / "rn_outer")
    assert (tmp_path / "rn_outer" / "rn_plant").is_dir()
    assert [m.name for m in Modely.load(tmp_path / "rn_outer").models] == ["rn_plant"]


@pytest.mark.slow
def test_a_minimizer_is_renamed_through_minimizers():
    model, _, _, y = _gain_model()
    t = Input("rn_t", tw=1)
    model.minimize("rn_fit", y, t).minimize("rn_other", y * 2.0, t).build()

    model.minimizers["rn_fit"]["name"] = "rn_track"
    assert [row["name"] for row in model.minimizers] == ["rn_track", "rn_other"]
    assert model.minimizers["rn_track"]["source"] is y
    with pytest.raises(KeyError, match="no minimizer named 'rn_fit'"):
        model.minimizers["rn_fit"]
    # Each name its own; a name like another's validation key is no clash,
    # its loss is logged as loss_val_rn_other, that one's as val_loss_rn_other.
    with pytest.raises(ValueError, match="has a minimizer named 'rn_other' already"):
        model.minimizers["rn_track"]["name"] = "rn_other"
    model.minimizers["rn_track"]["name"] = "val_rn_other"
    model.minimizers["val_rn_other"]["name"] = "rn_track"
    # The loss is logged under the new name from the next train.
    data = DataLoader(model, {"rn_x": np.arange(8.0), "rn_t": np.arange(8.0)})
    history = model.train(data, epochs=1, printer=None)
    assert "loss_rn_track" in getattr(history, "history", history)


def test_a_loop_body_renamed_after_the_loop_still_rolls_out(tmp_path):
    x, u = Input("rn_state", tw=1), Input("rn_push", tw=1)
    after = Output("rn_after", x + u)
    body = Modely("rn_body", inputs=[x, u], outputs=[after]).build()
    x0, pushes = Input("rn_x0", tw=1), Input("rn_pushes", seq=3, tw=1)
    loop = Loop(f=body, feedback={x: after}, init={x: x0})({u: pushes})
    model = Modely(
        "rn_loop", inputs=[x0, pushes], outputs=[Output("rn_traj", loop)]
    ).build()
    data = {
        "rn_x0": np.ones((1, 1, 1), np.float32),
        "rn_pushes": np.ones((1, 3, 1, 1), np.float32),
    }

    x.name, after.name = "rn_level", "rn_level_next"
    model.build()
    np.testing.assert_allclose(
        to_numpy(model(data)["rn_traj"]).ravel(), [2.0, 3.0, 4.0]
    )
    model.save(tmp_path / "rn_loop")
    loaded = Modely.load(tmp_path / "rn_loop")
    np.testing.assert_allclose(
        to_numpy(loaded(data)["rn_traj"]).ravel(), [2.0, 3.0, 4.0]
    )


def test_a_gradient_follows_the_input_it_differentiates_renamed(tmp_path):
    x = Input("rn_g_x", tw=1)
    square = x * x
    model = Modely(
        "rn_grad", inputs=[x], outputs=[Output("rn_dy", Gradient(wrt=x)(square))]
    )
    x.name = "rn_g_position"
    model.build()
    three = np.full((1, 1, 1), 3.0, np.float32)

    np.testing.assert_allclose(
        to_numpy(model({"rn_g_position": three})["rn_dy"]), [[[6.0]]]
    )
    model.save(tmp_path / "rn_grad")
    loaded = Modely.load(tmp_path / "rn_grad")
    np.testing.assert_allclose(
        to_numpy(loaded({"rn_g_position": three})["rn_dy"]), [[[6.0]]]
    )


def test_a_modelyloop_input_is_renamed_on_the_modelyloop():
    x, u = Input("rn_ml_x", tw=1), Input("rn_ml_u", tw=1)
    x.feedback(x + u)
    sim = ModelyLoop("rn_sim", inputs=[x, u], outputs=[Output("rn_ml_out", x)]).build()
    steps = np.ones((1, 4, 1, 1), np.float32)

    # Its inputs are its own, the step's rolled out: renamed there, the data
    # is keyed by the new name; the state stays optional.
    sim.inputs["rn_ml_x"].name = "rn_ml_position"
    sim.build()
    assert {node.name for node in sim._optional_inputs} == {"rn_ml_position"}
    start = np.full((1, 4, 1, 1), 10.0, np.float32)
    result = sim({"rn_ml_position": start, "rn_ml_u": steps})["rn_ml_out"]
    np.testing.assert_allclose(to_numpy(result).ravel(), [10.0, 11.0, 12.0, 13.0])


def test_an_input_renamed_like_another_is_listed_apart():
    # Once the table kept a row per name: the renamed Input's row was lost.
    a, b = Input("rl_a", dim=3, tw=5), Input("rl_b", dim=1, tw=5)
    out = Output("rl_out", a * b)
    model = Modely("rl_model", inputs=[a, b], outputs=[out])
    model.minimize("rl_loss", out, Input("rl_target", dim=3, tw=5))
    model.inputs["rl_a"].name = "rl_b"

    rows = model.inputs
    assert [(node.name, node.dim) for node in rows] == [
        ("rl_b", (3,)),
        ("rl_b", (1,)),
        ("rl_target", (3,)),
    ]
    assert rows[0] is a and rows[1] is b
    with pytest.raises(ValueError, match="'rl_b' is ambiguous"):
        rows["rl_b"]
    assert rows[0] is a
    assert "rl_b, rl_b, rl_target" in str(model)
    # The data is keyed by name: refused when built, until renamed apart.
    with pytest.raises(ValueError, match="two different Inputs named 'rl_b'"):
        model.build()
    a.name = "rl_a"
    model.build()


def test_the_outputs_table_tells_the_stream_fed_back_by_its_object():
    # Once matched by name: an output whose stream was named like the one an
    # Input is fed back with was said to feed it back too.
    x, u = Input("ft_x", tw=1), Input("ft_u", tw=1)
    after = x + u
    other = u * 2.0
    after.name = other.name = "ft_s"
    x.feedback(after)
    sim = ModelyLoop(
        "ft_sim",
        inputs=[x, u],
        outputs=[Output("ft_next", after), Output("ft_other", other)],
    )
    feeds = [row["feeds back"] for row in sim.outputs._rows]
    assert feeds == [["ft_x"], []]


# ---------------------------------------------------------------------------
# Every table: picked by name or by index, renamed through what it gives
# ---------------------------------------------------------------------------


def _full_model():
    """A model with two Inputs and a target, a block, two outputs and two
    minimizers, every name its own."""
    block = _block("all_block").build()
    x, w = Input("all_x", tw=3), Input("all_w", tw=1)
    y = Output("all_y", block([x]) * Parameter("all_gain", value=[3.0]))
    z = Output("all_z", w * 2.0)
    model = Modely("all_model", inputs=[x, w], outputs=[y, z])
    t = Input("all_t", tw=1)
    model.minimize("all_fit", y, t).minimize("all_reg", z, 0.0)
    return model


def _index_column(table) -> list[str]:
    """The index and the name of every row of a printed table."""
    rows = []
    for line in str(table).splitlines():
        cells = [cell.strip() for cell in line.split("│")[1:-1]]
        if len(cells) > 1 and cells[0].isdigit():
            rows.append((int(cells[0]), cells[1]))
    return rows


@pytest.mark.parametrize(
    "table", ["parameters", "models", "inputs", "outputs", "minimizers"]
)
def test_every_table_is_picked_by_name_or_index_and_renamed(table):
    model = _full_model()
    items = getattr(model, table)

    # The index column is the position, and the name picks the same item.
    printed = _index_column(items)
    assert printed == [(i, item.name) for i, item in enumerate(items)]
    for index, name in printed:
        same = items[name]
        assert same is items[index] or (
            table == "minimizers" and same["source"] is items[index]["source"]
        )
    assert items[-1].name == printed[-1][1]
    assert [item.name for item in items[:1]] == [printed[0][1]]
    with pytest.raises(TypeError, match="not both"):
        items[printed[0][1], 0]

    # Renamed through the table, it is found by its new name only.
    old = printed[-1][1]
    items[len(printed) - 1].name = "all_renamed"
    again = getattr(model, table)
    assert again["all_renamed"].name == "all_renamed"
    assert _index_column(again)[-1] == (len(printed) - 1, "all_renamed")
    with pytest.raises(KeyError, match=repr(old)):
        again[old]
    # And the model still builds and runs on it - the last Input is the
    # target, which running does not read.
    model.build()
    data = {"all_x": _WINDOW, "all_w": np.ones((1, 1, 1), np.float32)}
    assert len(model(data)) == 2


def test_a_minimizer_is_renamed_as_any_object():
    model = _full_model()

    # By attribute, as every other object, or by key.
    model.minimizers[0].name = "all_track"
    assert [row.name for row in model.minimizers] == ["all_track", "all_reg"]
    model.minimizers["all_reg"]["gain"] = 0.5
    assert model.minimizers[1].gain == 0.5
    # What it compares, or anything else, is not lost silently.
    with pytest.raises(TypeError, match="not its 'source'"):
        model.minimizers[0].source = model.outputs[1]
    with pytest.raises(TypeError, match="not its 'colour'"):
        model.minimizers[0].colour = "red"  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        model.minimizers[0].colour  # type: ignore[attr-defined]
    # Removed by name or by index.
    model.minimizers[1].remove()
    assert [row.name for row in model.minimizers] == ["all_track"]
    model.minimizers["all_track"].remove()
    assert list(model.minimizers) == []


def test_a_validation_result_is_read_by_name_or_index():
    model = _full_model().build()
    data = DataLoader(
        model,
        {
            "all_x": _WINDOW,
            "all_w": np.ones((1, 1, 1), np.float32),
            "all_t": np.zeros((1, 1, 1), np.float32),
        },
        windowed=True,
    )
    result = model.validate(data, verbose=False)

    # In the order of the minimizers table.
    assert result[0] is result["all_fit"] and result[1] is result["all_reg"]
    with pytest.raises(TypeError, match="not both"):
        result["all_fit", 0]  # type: ignore[index]


# ---------------------------------------------------------------------------
# Inputs that clash once built, and saved: two objects, whatever their names
# ---------------------------------------------------------------------------

_ONE = np.ones((1, 1, 1), np.float32)


def test_inputs_renamed_into_a_clash_after_the_build_are_refused_when_run():
    a, b = Input("cb_a", tw=1), Input("cb_b", tw=1)
    model = Modely(
        "cb_model", inputs=[a, b], outputs=[Output("cb_y", a + 10.0 * b)]
    ).build()
    b.name = "cb_a"

    # Once both read the one array given, 1 + 10 * 1, silently.
    with pytest.raises(ValueError, match=r"'cb_a' is ambiguous, 2 Inputs carry it"):
        model({"cb_a": _ONE})
    with pytest.raises(ValueError, match=r"several Inputs named \['cb_a'\]"):
        DataLoader(model, {"cb_a": _ONE}, windowed=True)
    # Renamed apart again, it runs by the new name, no build needed.
    b.name = "cb_b2"
    result = model({"cb_a": _ONE, "cb_b2": 2.0 * _ONE})["cb_y"]
    np.testing.assert_allclose(to_numpy(result), [[[21.0]]])


def test_a_model_whose_inputs_clash_is_refused_at_save(tmp_path):
    # Loaded, a model is built, and its Keras inputs need names of their own:
    # it is refused when saved, not when loaded.
    xa, xb = Input("cs_x", tw=1), Input("cs_x", tw=1)
    model = Modely(
        "cs_twins", inputs=[xa, xb], outputs=[Output("cs_y", xa + 10.0 * xb)]
    )
    for weights in (True, False):
        with pytest.raises(ValueError, match=r"several Inputs named \['cs_x'\]"):
            model.save(tmp_path / "twins", weights=weights)
    assert not (tmp_path / "twins").exists()
    # Renamed, it saves, and loads as the two Inputs it has.
    xb.name = "cs_x2"
    model.save(tmp_path / "twins")
    loaded = Modely.load(tmp_path / "twins")
    result = loaded({"cs_x": _ONE, "cs_x2": 2.0 * _ONE})["cs_y"]
    np.testing.assert_allclose(to_numpy(result), [[[21.0]]])


def test_a_model_renamed_into_a_clash_is_refused_at_save(tmp_path):
    a, b = Input("cw_a", tw=3), Input("cw_b", tw=1)
    fir = Fir(features=1, name="cw_fir")([a])  # a kernel drawn at random
    model = Modely(
        "cw_model", inputs=[a, b], outputs=[Output("cw_y", fir + 10.0 * b)]
    ).build()
    before = to_numpy(model({"cw_a": _WINDOW, "cw_b": _ONE})["cw_y"])
    b.name = "cw_a"
    with pytest.raises(ValueError, match=r"several Inputs named \['cw_a'\]"):
        model.save(tmp_path / "clash")
    # Renamed back, it saves with its weights.
    b.name = "cw_b"
    model.save(tmp_path / "clash")
    loaded = Modely.load(tmp_path / "clash")
    after = to_numpy(loaded({"cw_a": _WINDOW, "cw_b": _ONE})["cw_y"])
    np.testing.assert_allclose(after, before, rtol=1e-6)


def test_a_model_renamed_since_its_build_is_built_again_to_be_exported(tmp_path):
    # The Keras model keeps the names of its build: exported, two outputs of
    # one shape could swap their values, as ONNX names were paired by shape.
    x = Input("ex_x", tw=1)
    a, b = Output("ex_a", x * 2.0), Output("ex_b", x * 3.0)
    model = Modely("ex_model", inputs=[x], outputs=[a, b]).build()
    a.name, x.name = "ex_z", "ex_u"
    for export in (model.export_keras, model.export_onnx):
        with pytest.raises(ValueError, match="'ex_a' is now 'ex_z'.*build it again"):
            export(tmp_path)
    assert not any(tmp_path.iterdir())
    model.build().export_keras(tmp_path)
    exported = keras.saving.load_model(tmp_path / "ex_model.keras")
    assert isinstance(exported, keras.Model)
    assert set(exported.output) == {"ex_z", "ex_b"}
    assert [tensor.name for tensor in exported.inputs] == ["ex_u"]


def test_a_block_whose_inputs_share_a_name_is_called_by_position(tmp_path):
    xa, xb = Input("cp_x", tw=1), Input("cp_x", tw=1)
    block = Modely(
        "cp_block", inputs=[xa, xb], outputs=[Output("cp_y", xa + 10.0 * xb)]
    )
    a, b = Input("cp_a", tw=1), Input("cp_b", tw=1)
    outer = Modely(
        "cp_outer", inputs=[a, b], outputs=[Output("cp_o", block([a, b]))]
    ).build()
    data = {"cp_a": _ONE, "cp_b": 2.0 * _ONE}

    # A list pairs the streams with the inputs in order: once by name, both
    # read b, 2 + 10 * 2.
    np.testing.assert_allclose(to_numpy(outer(data)["cp_o"]), [[[21.0]]])
    outer.save(tmp_path / "outer")
    loaded = Modely.load(tmp_path / "outer")
    np.testing.assert_allclose(to_numpy(loaded(data)["cp_o"]), [[[21.0]]])
    # By name they are ambiguous.
    with pytest.raises(ValueError, match=r"'cp_x' is ambiguous, 2 Inputs carry it"):
        block({"cp_x": a})


def test_a_block_is_saved_with_the_model_calling_it_and_on_its_own(tmp_path):
    x = Input("bs_x", tw=2)
    block = Modely(
        "bs_block",
        inputs=[x],
        outputs=[Output("bs_y", Fir(features=1, bias=False, name="bs_fir")(x))],
    )
    a = Input("bs_a", tw=2)
    outer = Modely("bs_outer", inputs=[a], outputs=[Output("bs_o", block([a]))])
    outer.build()
    assert outer.inference_model is not None
    for weight in outer.inference_model.weights:  # as trained
        weight.assign(np.full(weight.shape, 3.0, np.float32))
    data = np.ones((1, 2, 1), np.float32)

    # The block is not built to be saved: the weights it computes with are in
    # the file of the model calling it, its folder holds its graph.
    outer.save(tmp_path / "outer")
    assert not block.built
    assert not (tmp_path / "outer" / "bs_block" / "model.weights.h5").exists()
    loaded = Modely.load(tmp_path / "outer")
    np.testing.assert_allclose(to_numpy(loaded({"bs_a": data})["bs_o"]), [[[6.0]]])
    # Saved itself, it is built, keeping the weights it was trained with.
    block.save(tmp_path / "block")
    loaded = Modely.load(tmp_path / "block")
    np.testing.assert_allclose(to_numpy(loaded({"bs_x": data})["bs_y"]), [[[6.0]]])


def test_weights_all_saves_every_block_with_its_own_weights(tmp_path):
    x = Input("ba_x", tw=2)
    block = Modely(
        "ba_block",
        inputs=[x],
        outputs=[Output("ba_y", Fir(features=1, bias=False, name="ba_fir")(x))],
    )
    a = Input("ba_a", tw=2)
    outer = Modely("ba_outer", inputs=[a], outputs=[Output("ba_o", block([a]))])
    outer.build()
    assert outer.inference_model is not None
    for weight in outer.inference_model.weights:  # as trained
        weight.assign(np.full(weight.shape, 3.0, np.float32))
    data = np.ones((1, 2, 1), np.float32)

    # Each folder is then a trained model.
    outer.save(tmp_path / "outer", weights="all")
    assert block.built
    alone = Modely.load(tmp_path / "outer" / "ba_block")
    np.testing.assert_allclose(to_numpy(alone({"ba_x": data})["ba_y"]), [[[6.0]]])
    loaded = Modely.load(tmp_path / "outer")
    np.testing.assert_allclose(to_numpy(loaded({"ba_a": data})["ba_o"]), [[[6.0]]])
    with pytest.raises(ValueError, match="True, False or 'all', got 'yes'"):
        outer.save(tmp_path / "other", weights="yes")  # type: ignore[arg-type]


def test_weights_all_refuses_a_block_whose_inputs_share_a_name(tmp_path):
    # Called by position the block runs, and saves with weights=True; its
    # own weights need a Keras model of its own, which needs names apart.
    xa, xb = Input("bt_x", tw=1), Input("bt_x", tw=1)
    block = Modely(
        "bt_block", inputs=[xa, xb], outputs=[Output("bt_y", xa + 10.0 * xb)]
    )
    a, b = Input("bt_a", tw=1), Input("bt_b", tw=1)
    outer = Modely(
        "bt_outer", inputs=[a, b], outputs=[Output("bt_o", block([a, b]))]
    ).build()
    with pytest.raises(ValueError, match=r"'bt_block' reads several Inputs named"):
        outer.save(tmp_path / "outer", weights="all")
    assert not (tmp_path / "outer").exists()
    outer.save(tmp_path / "outer")


def test_a_call_resolves_its_keys_to_the_inputs_at_once():
    a, b = Input("ck_a", tw=1), Input("ck_b", tw=1)
    model = Modely(
        "ck_model", inputs=[a, b], outputs=[Output("ck_y", a + 10.0 * b)]
    ).build()

    # Renamed alike, keyed by the Inputs themselves they are still two: the
    # Keras model keeps the keys of its build. Once both were the name 'ck_a'.
    b.name = "ck_a"
    result = model({a: _ONE, b: 2.0 * _ONE})["ck_y"]
    np.testing.assert_allclose(to_numpy(result), [[[21.0]]])
    # By the name both carry: refused as every name is, listing them.
    with pytest.raises(ValueError, match=r"'ck_a' is ambiguous, 2 Inputs carry it"):
        model({"ck_a": _ONE})
    b.name = "ck_b"
    # A name no Input carries is ignored, as another column of the data...
    with pytest.raises(ValueError, match=r"reads the inputs \['ck_b'\]"):
        model({"ck_a": _ONE, "ck_bb": _ONE})
    # ... but an Input of another model, named alike, is not taken for its
    # own: once it was read as b, 1 + 10 * 5.
    with pytest.raises(ValueError, match="does not read the Input 'ck_b'"):
        model({a: _ONE, Input("ck_b", tw=1): 5.0 * _ONE})

    # A block whose inputs share a name is called keyed by them too.
    xa, xb = Input("ck_x2", tw=1), Input("ck_x2", tw=1)
    block = Modely(
        "ck_block", inputs=[xa, xb], outputs=[Output("ck_by", xa + 10.0 * xb)]
    )
    p, q = Input("ck_p", tw=1), Input("ck_q", tw=1)
    outer = Modely(
        "ck_outer", inputs=[p, q], outputs=[Output("ck_o", block({xa: p, xb: q}))]
    ).build()
    result = outer({"ck_p": _ONE, "ck_q": 2.0 * _ONE})["ck_o"]
    np.testing.assert_allclose(to_numpy(result), [[[21.0]]])


def test_a_modelyloop_state_keeps_the_row_of_its_input_renamed(tmp_path):
    x, u = Input("mr_x", tw=1), Input("mr_u", tw=1)
    x.feedback(x + u)
    sim = ModelyLoop("mr_sim", inputs=[x, u], outputs=[Output("mr_out", x)]).build()
    sim.inputs["mr_x"].name = "mr_position"
    sim.build()
    sim.save(tmp_path / "sim")

    # The state starts from the data of the Input renamed, by object: one row,
    # in the ModelyLoop and in its loaded copy, a Modely.
    for model in (sim, Modely.load(tmp_path / "sim")):
        rows = model.inputs._rows
        assert [row["input"].name for row in rows] == ["mr_position", "mr_u"]
        assert rows[0]["fed back with"] is not None
        assert rows[0]["starts from"] == "its data, first step, else 0"


def test_every_lookup_resolves_a_key_alike():
    from nnodely.core.dag import NameNotFound

    # A key that picks nothing is a KeyError, as a table's, and a ValueError,
    # as a call's: either is caught.
    model = _two_ks()
    for lookup in (
        lambda: model.parameters["nope"],
        lambda: model.freeze("nope"),
        lambda: model.minimize("fit", "nope", 1.0),
    ):
        with pytest.raises(NameNotFound) as error:
            lookup()
        assert isinstance(error.value, KeyError) and isinstance(error.value, ValueError)

    # A Loop takes its body inputs by name or by themselves: a stream of
    # another graph named like one is no longer taken for it by its name,
    # and a key of no name is refused rather than ignored.
    s, u = Input("lk_s", tw=1), Input("lk_u", tw=1)
    after = Output("lk_after", s + u)
    body = Modely("lk_body", inputs=[s, u], outputs=[after]).build()
    s0, pushes = Input("lk_s0", tw=1), Input("lk_pushes", seq=3, tw=1)
    with pytest.raises(ValueError, match="'lk_u' is not an input the body"):
        Loop(f=body, feedback={s: after}, init={s: s0})({Input("lk_u", tw=1): pushes})
    with pytest.raises(TypeError, match="by its name, a string, or by itself"):
        Loop(f=body, feedback={s: after}, init={s: s0})({5: pushes})
    with pytest.raises(TypeError, match="by its name, a string, or by itself"):
        Loop(f=body, feedback={5: after})
    loop = Loop(f=body, feedback={"lk_s": "lk_after"}, init={"lk_s": s0})
    model = Modely(
        "lk_model", inputs=[s0, pushes], outputs=[Output("lk_traj", loop({u: pushes}))]
    ).build()
    data = {"lk_s0": _ONE, "lk_pushes": np.ones((1, 3, 1, 1), np.float32)}
    np.testing.assert_allclose(to_numpy(model(data)["lk_traj"]).ravel(), [2, 3, 4])


def test_a_dataloader_resolves_its_keys_as_a_call_does():
    import pandas as pd

    x, w = Input("dk_x", tw=1), Input("dk_w", tw=1)
    y = Output("dk_y", x * 2.0 + w)
    model = Modely("dk_model", inputs=[x, w], outputs=[y])
    model.minimize("dk_fit", y, Input("dk_t", tw=1)).build()
    values = np.arange(4.0)

    # Keyed by the Inputs themselves, as a call is: once refused as missing.
    data = DataLoader(model, {x: values, w: values, "dk_t": values})
    assert data.inputs == ["dk_x", "dk_w", "dk_t"]
    np.testing.assert_array_equal(data.get_input(x).ravel(), values)
    # format by Input: once ignored, x read its own column.
    frame = pd.DataFrame({"dk_x": values, "dk_w": 100 + values, "dk_t": values})
    mapped = DataLoader(model, frame, format={x: "dk_w"})
    np.testing.assert_array_equal(mapped.get_input("dk_x").ravel(), 100 + values)
    # A name no input carries is ignored, an Input of another model refused.
    DataLoader(
        model, {**{"dk_x": values, "dk_w": values, "dk_t": values}, "other": values}
    )
    with pytest.raises(ValueError, match="does not read the Input 'dk_x'"):
        DataLoader(model, {Input("dk_x", tw=1): values, "dk_w": values, "dk_t": values})

    # Made before the model was built again under other names, it is refused
    # with the names on both sides - once Keras's own error, in train.
    w.name = "dk_w2"
    model.build()
    with pytest.raises(
        ValueError, match=r"reads the inputs \['dk_w2'\], which the data"
    ):
        model.train(data, epochs=1, printer=None)


def test_a_loop_tells_a_state_inside_a_block_by_object():
    # A fed-back Input inside a block the body calls is a state of the loop:
    # given as a call key, it is that state - once "not an input the body
    # model reads", since only the copies of the body's own were matched.
    s, u = Input("is_s", tw=1), Input("is_u", tw=1)
    s.feedback(s + u)
    block = Modely("is_block", inputs=[u], outputs=[Output("is_y", s * 1.0)]).build()
    v = Input("is_v", tw=1)
    body = Modely("is_body", inputs=[v], outputs=[Output("is_o", block([v]))]).build()
    loop = Loop(f=body)
    with pytest.raises(ValueError, match="'is_s' is fed back: give its initial value"):
        loop({s: Input("is_pushes", seq=3, tw=1)})


def test_two_undeclared_inputs_of_one_name_are_both_listed():
    # The check is by object; the message once listed a set of names, so two
    # Inputs of one name read as one.
    first, second = Input("und_x", tw=1), Input("und_x", tw=1)
    with pytest.raises(ValueError, match=r"reads the Inputs \['und_x', 'und_x'\]"):
        Modely("und_model", inputs=[], outputs=[Output("und_y", first + second)])
