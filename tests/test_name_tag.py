"""Name tags: how the objects of a model are picked when names clash.

Nothing keeps two objects from sharing a name - layers named alike by hand, a
block and the model calling it, two copies of one loaded model. A call taking
a name looks it up among the objects the graph reaches, in that order: a name
one object carries picks it; one several carry raises listing them - or, for
freeze and unfreeze, picks them all with a warning - and ``(name, index)``
picks one of them. Inside nnodely the objects themselves are used.
"""

import json
import re
import warnings

import keras
import numpy as np
import pytest

import nnodely
from conftest import to_numpy
from nnodely import (
    DataLoader,
    Fir,
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


def _html_ids(model, folder):
    model.export_html(folder)
    page = (folder / f"{model.name}.html").read_text()
    nodes = re.search(r"const nodes = new vis\.DataSet\((\[.*?\])\);\n", page, re.S)
    assert nodes is not None
    return [node["id"] for node in json.loads(nodes.group(1))]


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
    # Or by its index among those of that name, the only one: 0.
    assert model.parameters["k", 0] is model.parameters["k"]


def test_a_name_several_parameters_carry_is_refused_listing_them():
    model = _two_ks()

    with pytest.raises(ValueError) as error:
        model.parameters["k"]
    message = str(error.value)
    assert "'k' is ambiguous, 2 Parameters carry it - indices [0, 1]" in message
    # Each with where it is, in the order the graph reaches them: the
    # model's own first, then those of the models it calls.
    assert "[0] k in 'tag_outer'; [1] k in 'tag_inner'" in message
    assert "Pick one with ('k', index)" in message
    np.testing.assert_allclose(to_numpy(model.parameters["k", 0].param), [10.0])
    np.testing.assert_allclose(to_numpy(model.parameters["k", 1].param), [2.0])


def test_two_parameters_of_one_name_in_one_model_are_told_apart_by_index():
    x = Input("tag_one_x", tw=1)
    first, second = Parameter("gain", value=[2.0]), Parameter("gain", value=[3.0])
    model = Modely(
        "tag_one", inputs=[x], outputs=[Output("tag_one_y", x * first + x * second)]
    ).build()

    assert model.parameters["gain", 0] is first
    assert model.parameters["gain", 1] is second
    result = model({"tag_one_x": np.ones((1, 1, 1), np.float32)})["tag_one_y"]
    np.testing.assert_allclose(to_numpy(result), [[[5.0]]])


def test_an_unknown_name_or_index_is_refused():
    model = _two_ks()

    with pytest.raises(KeyError, match="no Parameter named 'nope'"):
        model.parameters["nope"]
    with pytest.raises(KeyError, match="no Parameter named 'nope'"):
        model.parameters["nope", 0]
    with pytest.raises(IndexError, match=r"\('k', 2\) picks .* there are 2"):
        model.parameters["k", 2]
    with pytest.raises(IndexError, match=r"\('k', -1\)"):
        model.parameters["k", -1]
    # Positions pick as in a list.
    assert model.parameters[0] is model.parameters["k", 0]


def test_the_indices_are_the_same_once_saved_and_loaded(tmp_path):
    model = _two_ks()
    model.save(tmp_path / "two_ks")
    loaded = Modely.load(tmp_path / "two_ks")

    for index, value in ((0, 10.0), (1, 2.0)):
        np.testing.assert_allclose(
            to_numpy(loaded.parameters["k", index].param), [value]
        )
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
        ValueError, match=r"'tag_copy' is ambiguous.*\[0\] tag_copy in 'tag_copies'"
    ):
        both.models["tag_copy"]
    assert both.models["tag_copy", 0] is left
    assert both.models["tag_copy", 1] is right
    # Each copy is its own: the Parameter of one is not the other's.
    assert both.parameters["k", 0] is left.parameters["k"]
    assert both.parameters["k", 1] is right.parameters["k"]


# ---------------------------------------------------------------------------
# freeze and unfreeze: every match, with a warning, or one by index
# ---------------------------------------------------------------------------


def test_freeze_picks_every_parameter_of_a_name_and_warns():
    model = _two_ks()

    with pytest.warns(UserWarning, match=r"'k' is ambiguous.*All of them are picked"):
        model.freeze("k")
    assert [p.trainable for p in model.parameters if p.name == "k"] == [False, False]

    model.unfreeze()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # one picked: no warning
        model.freeze(("k", 1))
    assert model.parameters["k", 0].trainable
    assert not model.parameters["k", 1].trainable


def test_a_freeze_that_raises_warns_of_nothing():
    model = _two_ks()

    # "k" alone is ambiguous, "nope" unknown: nothing is frozen, and the
    # warning that every k would be is not given.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="no Parameter and no model named 'nope'"):
            model.freeze(["k", "nope"])
    assert not [w for w in caught if issubclass(w.category, UserWarning)]
    assert all(p.trainable for p in model.parameters)
    # When it freezes, the warning points at the line calling freeze.
    with pytest.warns(UserWarning) as record:
        model.freeze("k")
    assert record[0].filename == __file__


def test_freeze_takes_a_list_of_names_and_indices():
    model = _two_ks()

    model.freeze([("k", 0), "fir_kernel"])
    assert sorted(
        (p.name, i) for i, p in enumerate(model.parameters) if not p.trainable
    ) == [("fir_kernel", 1), ("k", 0)]
    with pytest.raises(TypeError, match="got float"):
        model.freeze([1.5])  # type: ignore[list-item]
    with pytest.raises(IndexError, match=r"\('k', 5\)"):
        model.unfreeze(("k", 5))


def test_freeze_a_model_name_two_copies_carry(tmp_path):
    _block("tag_frozen").build().save(tmp_path / "frozen")
    left, right = Modely.load(tmp_path / "frozen"), Modely.load(tmp_path / "frozen")
    a = Input("tag_frozen_a", tw=3)
    both = Modely(
        "tag_frozen_both",
        inputs=[a],
        outputs=[Output("tag_frozen_o", left([a]) + right([a]))],
    ).build()

    with pytest.warns(UserWarning, match="'tag_frozen' is ambiguous"):
        both.freeze("tag_frozen")
    assert not any(p.trainable for p in both.parameters)
    both.unfreeze()
    both.freeze(("tag_frozen", 1))
    assert [p.trainable for p in left.parameters] == [True, True]
    assert [p.trainable for p in right.parameters] == [False, False]


# ---------------------------------------------------------------------------
# minimize: a stream by name, by name and index, or itself
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


def test_minimize_picks_a_stream_by_name_and_index():
    model, ya, yb = _twice()

    with pytest.raises(
        ValueError, match=r"'y' is ambiguous.*\[0\] y \(an IntermediateOutput"
    ):
        model.minimize("fit", "y", 1.0)
    assert model._minimizers == []
    model.minimize("fit", ("y", 1), 1.0)
    assert model._minimizers[0]["source"] is yb
    model.minimize("fit_a", ("y", 0), 1.0)
    assert model._minimizers[1]["source"] is ya


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

    ids = _html_ids(model, tmp_path)
    assert ids.count("tag_twice_block_call") == 1
    assert "tag_twice_block_call~2" in ids
    assert {"y", "y~2"} <= set(ids)
    assert (tmp_path / "tag_twice__tag_twice_block_call.html").exists()
    assert (tmp_path / "tag_twice__tag_twice_block_call-2.html").exists()


def test_html_draws_two_layers_of_one_name_apart(tmp_path):
    x = Input("tag_html_x", tw=3)
    first, second = Fir(features=1, name="tag_f"), Fir(features=1, name="tag_f")
    model = Modely(
        "tag_html", inputs=[x], outputs=[Output("tag_html_y", first([x]) + second([x]))]
    ).build()

    ids = _html_ids(model, tmp_path)
    assert {"tag_f", "tag_f~2", "tag_f_kernel", "tag_f_kernel~2"} <= set(ids)
