import os
import warnings

from collections.abc import Mapping
from typing import Any, Callable, Literal, cast

from nnodely.core import export, validation
from nnodely.core.dag import (
    free_name,
    toposort,
    flatten,
    _flatten_graph,
    _undeclared_inputs,
    _parameters,
    _nodes_and_models,
    _nodes_by_model,
    _read_nodes,
    _source_input_tensors,
    NameNotFound,
    Table,
    resolve,
)
from nnodely.core.minimizer import (
    MinimizerModel,
    _check_minimizers_train_weights,
    _check_sides,
    _minimizer_stream,
    _resolve_minimizer_terms,
    _validate_gain,
)
from nnodely.core.results import (
    Fields,
    TrainResult,
    ValidationResult,
    loss_name,
    minimizer_fields,
    target_label,
)
from nnodely.utils.utils import (
    _resolve_early_stopping,
    _resolve_loss,
    _resolve_optimizer,
)
from nnodely.utils.printers import _resolve_printer, _table
from nnodely.core.registry import ModelSerializer
from nnodely.core.stream import Stream, Node, object_repr
from nnodely.layers.parameter import Constant, Parameter, Parameters, _Value
from nnodely.core.dataloader import DataLoader

import numpy as np

from nnodely.layers.output import Output, Outputs
from nnodely.layers.input import Input, Inputs
from nnodely.core.layer import Layer, _claim_keras_name, _refuse_uncalled

import keras


class Modely:
    r"""A model made of the streams that lead from its Inputs to its Outputs,
    trained, run and saved as one network.

    Syntax
    ------
    ``Modely(name, inputs, outputs)``
        Declares the model of the streams that lead from ``inputs`` to
        ``outputs``.
    ``model.minimize(name, source, target)``, then ``model.build()``
        Registers an objective, then creates the Keras model and its weights.
    ``model({"x": array})``
        Runs the built model on arrays, keyed by input name or by Input.
    ``model([array_1, array_2])``
        The same, one array per input, in the order of ``inputs``.
    ``model(array)``
        The same, for a model with one input.
    ``model({"x": stream})``, ``model([stream_1, stream_2])``, ``model(stream)``
        Uses the model as a block of a larger graph: returns its outputs as
        streams.
    ``model.loop()``, ``model.loop(seq=n)``
        Returns the model of the rollout of a model declared one step at a
        time, over the steps of the data or ``n`` steps: see :meth:`loop`.

    A model is declared once, on streams. Objectives, called minimizers in the
    method names (:meth:`minimize`, :attr:`minimizers`,
    :meth:`minimizers[name].remove() <nnodely.core.modely.Minimizer.remove>`), are registered next. :meth:`build` then
    creates the Keras model. A :class:`DataLoader` made for the built model
    feeds :meth:`train` and :meth:`validate`.

    Called on arrays, a built model runs: every array has its batch axis
    first. Called on streams, the model is a block: its outputs become streams
    of the graph that calls it, and its weights are shared with it.

    Parameters
    ----------
    name : str
        The name of the model. It heads the printed tables, names the files
        :meth:`export_keras`, :meth:`export_onnx` and :meth:`export_html`
        write, and names the folder of a block saved inside a larger model.
    inputs : list of Input
        Every Input the model reads. The list form of the call follows its
        order. It lists the Inputs the outputs read, the fed-back ones too,
        those given to a block, and the ``init`` of an :class:`Integrate`, a
        :class:`Derivative` or a :class:`Buffer`. Two kinds are left out:

        - a hidden state, ``x.feedback(stream, hidden=True)``: the model's own
          memory, which cannot be declared, see :meth:`Input.feedback`;
        - an Input only an objective reads, such as a measured target.

        The list is copied: changing it later leaves the model as it was.
    outputs : list of Output
        The streams the model exposes, each named by its :class:`Output`.

    Returns
    -------
    dict of str to tensor, Stream or list of Stream
        Called on arrays: ``{output name: tensor}``, a tensor of the active
        backend for every declared output. Its shape is ``(batch, *S, T,
        *D)`` for an output of ``Shape(S, T, D)``: 4 samples of an output of
        ``Shape(S=(), T=1, D=(1,))`` give ``(4, 1, 1)``. Convert it with
        ``keras.ops.convert_to_numpy``. Called on streams: the output stream
        of a model with one output, else the list of them, in the order of
        ``outputs``.

    Raises
    ------
    ValueError
        If the outputs read an Input that is not among ``inputs`` ("which are
        not among its inputs: declare them there"), or if ``inputs`` holds a
        hidden state ("declares the hidden states"). If the model is called on
        arrays before :meth:`build` ("Model is not built"), or without an
        input it reads ("which are missing from the data"). If a list holds
        one value too many or too few ("takes one value for each"). If a
        block is given a stream of another shape ("takes a stream of that
        shape").
    TypeError
        If the model is called on something else than a dict, a list, an
        array or a stream ("is called with a dict keyed by Input or input
        name"), or on streams and arrays together ("streams and arrays
        together").

    See Also
    --------
    Modely.loop : The rollout of a model declared one step at a time.
    DataLoader : The data of a model, laid out for training.
    Input : A signal read from data.
    Output : A stream the model exposes, with its name.
    Loop : A rollout of a model inside a larger graph.

    Notes
    -----
    The lifecycle is fixed: declare the streams, then the model, then the
    objectives, then :meth:`build`. Make the :class:`DataLoader` after the
    build. Train, validate, call and export the built model.

    The array of an Input is laid out ``(batch, *S, T, *D)``, as the Input
    is declared: its ``shape`` gives ``S``, ``T`` and ``D``. An Input without
    a window has no ``T`` axis. Arrays need not be ``float32``.

    A dict may hold more keys than the model reads. A key that names no Input
    of the model, such as a target or another column, is ignored. So the
    arrays of a :class:`DataLoader`, ``loader.as_dict()``, pass as they are.
    Keyed by the Input objects, two Inputs of one name are told apart. A name
    several Inputs of the model carry raises: key them by the Inputs.

    ``print(model)``, or ``model`` as the last line of a notebook cell, shows
    a table of the names of its inputs, outputs and objectives.

    Basic Examples
    --------------
    A model that doubles its input, run on one sample of a window of 2:

    >>> import keras
    >>> import numpy as np
    >>> from nnodely import DataLoader, Fir, Input, Modely, Output, Parameter
    >>> x = Input("x", tw=2)
    >>> double = Modely("double", inputs=[x], outputs=[Output("y", 2.0 * x)])
    >>> double = double.build()
    >>> result = double({"x": np.array([[[1.0], [2.0]]])})  # (batch, T, D)
    >>> keras.ops.convert_to_numpy(result["y"])
    array([[[2.],
            [4.]]], dtype=float32)

    The model prints as a table of its names:

    >>> print(double)
    ┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
    ┃ model      ┃ double (built) ┃
    ┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
    │ inputs     │ x              │
    │ outputs    │ y              │
    │ parameters │ 0              │
    │ minimizers │ -              │
    └────────────┴────────────────┘

    Scope
    -----
    The same array, keyed by the Input, in a list, or alone:

    >>> sample = np.array([[[1.0], [2.0]]])
    >>> [keras.ops.convert_to_numpy(double(form)["y"]).ravel().tolist()
    ...  for form in ({x: sample}, [sample], sample)]
    [[2.0, 4.0], [2.0, 4.0], [2.0, 4.0]]

    A batch of samples gives one result each:

    >>> keras.ops.convert_to_numpy(double({"x": np.ones((3, 2, 1))})["y"]).shape
    (3, 2, 1)

    The arrays of a :class:`DataLoader` pass as they are, batch axis included:

    >>> loader = DataLoader(double, source={"x": [1.0, 2.0, 3.0]})
    >>> keras.ops.convert_to_numpy(double(loader.as_dict())["y"]).shape
    (2, 2, 1)

    Called on a stream, the model is a block, and returns a stream:

    >>> a = Input("a", tw=2)
    >>> double(a).shape
    Shape(S=(), T=2, D=(1,))

    Options
    -------
    ``inputs`` sets the order of the list form, here ``u`` then ``v``:

    >>> u, v = Input("u", tw=1), Input("v", tw=1)
    >>> diff = Modely("diff", inputs=[u, v], outputs=[Output("d", u - v)]).build()
    >>> keras.ops.convert_to_numpy(diff([np.array([[[5.0]]]), np.array([[[2.0]]])])["d"])
    array([[[3.]]], dtype=float32)

    ``outputs`` names the keys of the result:

    >>> sorted(diff({"u": np.ones((1, 1, 1)), "v": np.ones((1, 1, 1))}))
    ['d']

    Applications
    ------------
    The next position of a mass, from its last positions and the force, as a
    block reused by a larger model:

    >>> position = Input("position", tw=3)
    >>> force = Input("force", tw=1)
    >>> step = Modely("step", inputs=[position, force],
    ...               outputs=[Output("next", Fir()(position) + Fir()(force))])
    >>> p, f = Input("p", tw=3), Input("f", tw=1)
    >>> plant = Modely("plant", inputs=[p, f], outputs=[Output("p_next", step([p, f]))])
    >>> [model.name for model in plant.models]
    ['step']

    Properties & Relations
    ----------------------
    A block computes inside a larger model what it computes alone:

    >>> k = Parameter("k", value=[[3.0]])
    >>> gain = Modely("gain", inputs=[u], outputs=[Output("g", Fir(kernel=k, bias=False)(u))])
    >>> w = Input("w", tw=1)
    >>> outer = Modely("outer", inputs=[w], outputs=[Output("o", gain(w))]).build()
    >>> one = np.array([[[2.0]]])
    >>> keras.ops.convert_to_numpy(outer({"w": one})["o"])
    array([[[6.]]], dtype=float32)
    >>> keras.ops.convert_to_numpy(gain.build()({"u": one})["g"])
    array([[[6.]]], dtype=float32)

    Possible Issues
    ---------------
    The arrays need their batch axis. ``loader[0]`` is one sample without
    it, so the window of 2 samples is read as a batch of 2. A model whose
    layers accept that shape runs silently and gives a wrong result. Add the
    batch axis:

    >>> sample = loader[0]
    >>> sample["x"].shape
    (2, 1)
    >>> keras.ops.convert_to_numpy(double(sample)["y"]).shape  # wrong: 2 samples
    (2, 1, 1)
    >>> batched = {key: value[None] for key, value in loader[0].items()}
    >>> keras.ops.convert_to_numpy(double(batched)["y"])
    array([[[2.],
            [4.]]], dtype=float32)

    A model runs on arrays only once built:

    >>> Modely("late", inputs=[x], outputs=[Output("z", 3.0 * x)])(sample)
    Traceback (most recent call last):
        ...
    ValueError: Model is not built. Call build() before calling it.

    Every Input the outputs read is declared among ``inputs``:

    >>> Modely("partial", inputs=[u], outputs=[Output("s", u + v)])
    Traceback (most recent call last):
        ...
    ValueError: Model 'partial' reads the Inputs ['v'], which are not among its inputs: declare them there...

    .. versionadded:: 2.0
    """

    name: str
    _inputs: list[Input]
    _outputs: list[Output]
    _order: list[Node]

    def __init__(self, name: str, inputs: list[Input], outputs: list[Output]) -> None:
        self._declare(name, inputs, outputs)
        hidden = [node.name for node in self._inputs if node._hidden]
        if hidden:
            raise ValueError(
                f"Model {name!r} declares the hidden states {hidden} among its "
                "inputs. A hidden state (fed back with hidden=True, or kept by "
                "an Integrate, a Derivative or a Buffer) is the model's own "
                "memory, which starts from its init, and none of its inputs: "
                "leave it out of inputs, or feed it back without hidden=True "
                "to read it from data."
            )

    @classmethod
    def _of(cls, name: str, inputs: list[Input], outputs: list[Output]) -> "Modely":
        """A model of a graph nnodely makes, which may declare hidden states:
        the body of a Loop, which reads its states as inputs at every step,
        or a model loaded back, checked when it was declared."""
        model = cls.__new__(cls)
        model._declare(name, inputs, outputs)
        return model

    def _declare(self, name: str, inputs: list[Input], outputs: list[Output]) -> None:
        self.name = name
        # Lists of its own: one the caller goes on changing, reused for
        # another model, leaves this one as it was declared.
        self._inputs = list(inputs)
        self._outputs = list(outputs)
        self._order = toposort(self._outputs)
        # Read as an output, or bound to a Loop, it passed no layer call.
        _refuse_uncalled(self._order)
        undeclared = _undeclared_inputs(self._inputs, self._outputs)
        if undeclared:
            # One entry per Input: two of one name are listed twice.
            fed_back = sorted(
                node.name for node in undeclared if node.update is not None
            )
            raise ValueError(
                f"Model {name!r} reads the Inputs "
                f"{sorted(node.name for node in undeclared)}, which are not "
                "among its inputs: declare them there. A model declares every "
                "Input it reads, the init of an Integrate, a Derivative or a "
                "Buffer too; only a hidden state, or an Input only a minimizer "
                "reads, such as a target, may be left out."
                + (
                    f" The fed-back ones, {fed_back}, are states: each is "
                    "declared, an input the data or the caller gives, or "
                    "hidden, the model's own memory, x.feedback(stream, "
                    "init=..., hidden=True)."
                    if fed_back
                    else ""
                )
            )
        self._keras_model = None  # Keras model build from this DAG
        self._train_inputs = []  # List of Input nodes that are required for training (derived from minimizers)
        self._train_outputs = []  # List of Output nodes that are required for training (derived from minimizers)
        self._keras_inference_model = (
            None  # Keras model of the declared outputs only, without minimizers
        )
        self._inference_inputs = []  # List of Input nodes the declared outputs read (no minimizer-only input)
        # Inputs the data may leave out: they read zeros.
        self._optional_inputs: set[Input] = set()
        # Set by build(): the key every Input and output has in the Keras
        # models, its name then, by the id of the object; the Input of the
        # graph each copy the build made stands for, by the id of the copy.
        self._keys: dict[int, str] = {}
        self._originals: dict[int, Input] = {}
        self._minimizers = []  # List of dicts with keys: 'source', 'target', 'loss', 'name'
        self._minimizer_outputs: dict[Node, str] = {}  # minimizer side -> graph output
        # Set by loop(): the Input of the rollout it made of each Input of
        # its step, by the Input it stands for (_Step.key). The graph draws the
        # two as one node.
        self._rolled_inputs: dict[int, Input] = {}

    def __str__(self) -> str:
        """The model as a table: its name, then the names of every Input it
        reads, as :attr:`inputs` lists them, of its outputs, the number of its
        Parameters and the names of its minimizers."""
        rows = [
            ("inputs", ", ".join(self.inputs.keys()) or "-"),
            ("outputs", ", ".join(node.name for node in self._outputs) or "-"),
            ("parameters", str(len(_parameters(self)))),
            ("minimizers", ", ".join(m["name"] for m in self._minimizers) or "-"),
        ]
        title = f"{self.name} (built)" if self.built else self.name
        return _table(("model", title), [("", rows)])

    def __repr__(self) -> str:
        """One line, the fields of its row in :attr:`models`."""
        return object_repr(
            self,
            name=self.name,
            inputs=self.inputs.keys(),
            outputs=[node.name for node in self._outputs],
            parameters=len(_parameters(self)),
            built=self.built,
        )

    def __call__(self, inputs) -> Any:
        """Run the model on arrays, or use it as a block on streams.

        Either way ``inputs`` is a dict keyed by Input or input name, a list
        with one value per input, in the order of ``inputs``, or the value
        alone of a model with one input. Every key is resolved to its Input
        at once, as ``model.inputs[name]`` resolves it: a name several Inputs
        carry raises, and so does an Input the model does not read. Keyed by
        the Inputs themselves, two of one name are told apart. A key that
        names no input the model reads, such as a target or another column,
        is ignored.
        """
        if isinstance(inputs, Node) or hasattr(inputs, "shape"):
            # A stream or an array alone: the value of the one input.
            if len(self._inputs) != 1:
                names = [node.name for node in self._inputs]
                raise ValueError(
                    f"Model {self.name!r} has {len(self._inputs)} inputs {names}: "
                    "called with a value alone it has one; give a list, one "
                    "value per input in that order, or a dict."
                )
            inputs = [inputs]
        if isinstance(inputs, dict):
            values = list(inputs.values())
            given = self._given_by_input(inputs)
        elif isinstance(inputs, (list, tuple)):
            values = list(inputs)
            # One value per input, no more and no less: paired off with zip, an
            # input left without one would be read from the data of the graph
            # above, and an extra value dropped.
            if len(values) != len(self._inputs):
                names = [node.name for node in self._inputs]
                raise ValueError(
                    f"Model {self.name!r} has {len(self._inputs)} inputs {names}: "
                    "called with a list it takes one value for each, in that "
                    f"order, got {len(values)}."
                )
            if all(isinstance(value, Node) for value in values):
                # Paired by position, not by name: two inputs may share one.
                return self._call_stream(values)
            given = dict(zip(self._inputs, values))
        else:
            raise TypeError(
                f"Model {self.name!r} is called with a dict keyed by Input or "
                "input name, with a list of streams or arrays, one per input "
                "in their order, or with the stream or array of its one input; "
                f"got {type(inputs).__name__}."
            )
        streams = [isinstance(value, Node) for value in values]
        if all(streams):
            # A declared input left out would be read from the data of the
            # graph above, as an input of its own.
            missing = [node.name for node in self._inputs if node not in given]
            if missing:
                names = [node.name for node in self._inputs]
                raise ValueError(
                    f"Model {self.name!r} has {len(names)} inputs {names}: used "
                    f"as a block it takes a stream for each, and {missing} have "
                    "none."
                )
            return self._call_stream([given[node] for node in self._inputs])
        if any(streams):
            raise TypeError(
                f"Model {self.name!r} is called with streams and arrays together: "
                "streams make it a block of a larger graph, arrays run it. Give "
                "it one or the other."
            )
        return self._call_tensor(given)

    def _input_of(self, key, known: list | None = None) -> Input:
        """The Input a key the user gives stands for. The key is the Input
        itself, or its name, resolved as ``model.inputs[name]`` resolves it: a name
        several Inputs carry raises, listing them, and a name none carries, or
        an Input the model does not read, raises :class:`NameNotFound`.
        ``known`` is ``self.inputs._candidates()``, when the caller holds it."""
        return resolve(
            key,
            self.inputs._candidates() if known is None else known,
            kind="Input",
            owner=f"Model {self.name!r}",
            hint="Key it by the Input itself, model.inputs[i], or give it a "
            "name of its own, model.inputs[i].name = 'new_name'.",
            objects=Input,
            missing=lambda key: (
                f"Model {self.name!r} has no Input named {key!r}; its Inputs are "
                f"{[node.name for node in self.inputs]}."
                if isinstance(key, str)
                else f"Model {self.name!r} does not read the Input {key.name!r} "
                "it is keyed by: key it by one of its own, model.inputs[i]."
            ),
        )

    def _given_by_input(self, inputs: Mapping) -> dict[Input, Any]:
        """``{Input: value}`` from a dict keyed by Input or input name, every
        key resolved by :meth:`_input_of`: a name no Input carries is ignored,
        as another column of the data."""
        known = self.inputs._candidates()
        given: dict[Input, Any] = {}
        for key, value in inputs.items():
            if not isinstance(key, (str, Input)):
                continue  # no name and no Input: ignored, as a name none carries
            try:
                node = self._input_of(key, known)
            except NameNotFound:
                if isinstance(key, str):
                    continue  # a name no Input of the model carries: ignored
                raise
            if node in given:
                raise ValueError(
                    f"Model {self.name!r} is given two values for its input "
                    f"{node.name!r}: key it by the Input or by its name, once."
                )
            given[node] = value
        return given

    def _call_stream(self, arguments: list) -> Any:
        """The model as a block of a larger graph: every declared input reads
        the stream given for it, ``arguments`` in their order, and the outputs
        are returned as streams."""
        for node, stream in zip(self._inputs, arguments):
            self._check_block_input(node, stream)
        mc = ModelCall(self.name, self)
        mc._preds = arguments
        mc._inputs_map = {old: new for old, new in zip(self._inputs, arguments)}
        outputs = [IntermediateOutput(output, mc) for output in self._outputs]
        mc._outputs_map = {new: old for old, new in zip(self._outputs, outputs)}
        return outputs[0] if len(outputs) == 1 else outputs

    def _call_tensor(self, inputs: dict[Input, Any]) -> Any:
        """The model run on arrays, ``{Input: array}``: its declared
        outputs, without minimizers, ``{output name: tensor}``.

        Every Input is handed to Keras under the key it was built with, and
        every output returned under its name now: an Input or an output
        renamed since the build is still found, and two Inputs renamed alike
        are still two."""
        # The arrays come batched, as a DataLoader lays them out; an input only
        # a minimizer reads, such as a target, is neither needed nor passed on.
        if self._keras_inference_model is None:
            raise ValueError("Model is not built. Call build() before calling it.")
        read = [(self._original(node), node.name) for node in self._inference_inputs]
        given = [(node, key) for node, key in read if node in inputs]
        left_out = [(node, key) for node, key in read if node not in inputs]
        missing = [
            node.name
            for node, _ in left_out
            if node not in self._optional_inputs or not given
        ]
        if missing:
            raise ValueError(
                f"Model {self.name!r} reads the inputs {missing}, which are "
                "missing from the data it was called with."
            )
        # An optional input left out reads zeros, one step of a dynamic
        # sequence: a rollout of loop() starts from the first step of it alone.
        batch = np.shape(inputs[given[0][0]])[0]
        arrays = {key: inputs[node] for node, key in given}
        for node, key in left_out:
            arrays[key] = np.zeros(
                (batch, *(1 if size is None else size for size in node.shape.tuple)),
                np.float32,
            )
        results = self._keras_inference_model(arrays)
        names = [output.name for output in self._outputs]
        twins = sorted({name for name in names if names.count(name) > 1})
        if twins:
            raise ValueError(
                f"Model {self.name!r} has several outputs named {twins}, renamed "
                "since it was built: the results are keyed by output name, so "
                "give each output a name of its own."
            )
        return {
            output.name: results[self._keys[id(output)]] for output in self._outputs
        }

    def _original(self, node: Input) -> Input:
        """The Input of the graph ``node``, a copy :meth:`build` made, stands
        for: ``node`` itself if it is none."""
        return self._originals.get(id(node), node)

    def _check_inputs_apart(self, nodes) -> None:
        """Refuse to read data for the Inputs ``nodes``, those a build read,
        while several carry one name: the data is keyed by input name, so
        they would read one array. Renamed alike after the build, or loaded
        so, they are still two Inputs, which a rename tells apart again."""
        names = [self._original(node).name for node in nodes]
        twins = sorted({name for name in names if names.count(name) > 1})
        if twins:
            raise ValueError(
                f"Model {self.name!r} reads several Inputs named {twins}: the "
                "data is keyed by input name, so give one of them a name of "
                "its own, model.inputs[i].name = 'new_name', then build "
                "again to train with it."
            )

    def _check_names_built(self) -> None:
        """Refuse to train, validate or export while an Input or an output
        has another name than the one it was built with: the data and the
        Keras models are keyed by those names."""
        self._check_inputs_apart(self._train_inputs)
        renamed = [
            f"{key!r} is now {node.name!r}"
            for node, key in [
                *((self._original(node), node.name) for node in self._train_inputs),
                *((node, self._keys[id(node)]) for node in self._outputs),
            ]
            if node.name != key
        ]
        if renamed:
            raise ValueError(
                f"Model {self.name!r} was built before some of its names "
                f"changed - {', '.join(renamed)}: build it again to use them."
            )

    def _check_block_input(self, node: Input, stream) -> None:
        """Refuse a stream that this model, used as a block, cannot read as
        its input ``node``: a block keeps the shapes it was declared with, so
        the stream must have the shape of ``node``. A dynamic sequence axis
        matches any length.
        """
        if node not in self._order:
            return  # an input this model does not read
        if (
            stream.tw != node.tw
            or not _same_sizes(stream.seq, node.seq)
            or tuple(stream.dim) != tuple(node.dim)
        ):
            raise ValueError(
                f"Model {self.name!r} reads its input {node.name!r} as "
                f"{node.shape}, so used as a block it takes a stream of that "
                "shape for it, with the same dim, sequence and window; "
                f"{stream.name!r} is {stream.shape}. A block keeps "
                "the shapes it was declared with: give it a stream of that "
                "shape, or declare the model on the shape it is used with."
            )

    def loop(self, seq: int | None = None) -> "Modely":
        r"""A new :class:`Modely` that rolls this model out over a sequence
        of steps: every Input it feeds back is fed back at the next step.

        Syntax
        ------
        ``model.loop()``
            Rolls the model out over as many steps as the data has.
        ``model.loop(seq=n)``
            Rolls it out over ``n`` steps.

        This model is one step. A state is an Input fed back with
        :meth:`Input.feedback`: ``x_next = x.feedback(stream)``. Inside the
        step, ``x`` holds the value at the end of the previous step, and
        ``stream`` the value at the end of this one. Every fed-back Input the
        step reads, also inside the models it calls (one per call), is fed
        back together, in one rollout, so states may read each other. A state
        is declared among the inputs of this model, or hidden,
        ``x.feedback(stream, hidden=True)``: the model's own memory.

        - A declared state starts from the first step of its own data, and
          from zero where the data leaves it out.
        - A hidden state starts from its ``init``, zero without one; so does
          the state of an :class:`Integrate`, a :class:`Derivative` or a
          :class:`Buffer`, from the ``init`` of the layer.

        A model the step calls as a block keeps the feedback of the fed-back
        Inputs it declares: the stream the call gives one is its initial
        value, read at the first step. Every Input has a window: ``tw=1``
        reads one sample per step.

        The model returned is a plain :class:`Modely` of the rollout, of the
        name of this one: its Inputs and outputs have the names of those of
        the step, and the steps as a new outermost sequence axis. Every
        output is a trajectory: its element ``k`` is computed at step ``k``.
        Everything else works as in any Modely: objectives, :meth:`build`,
        :meth:`train`, calls and export.

        The one-step model it rolls out is ``rollout.models[0]``, and shares
        its weights with this model. It is this model itself when this model
        declares every state, and has an output for every stream a state is
        fed back with; it is then built, if it is not. Otherwise it is a model
        of the same name made of its graph, with its states among its inputs
        and an output added for each stream they are fed back with, as
        :class:`Loop` rolls out. The objectives of this model are not those of
        the rollout.

        Parameters
        ----------
        seq : int or None, default None
            The number of steps, at least 1. ``None`` takes as many steps as
            the data has.

        Returns
        -------
        Modely
            The model of the rollout. With ``seq=n``, an Input of the step
            declared with ``seq=s`` is read from data of ``seq=(n, *s)``. On
            arrays, an output of the step of ``Shape(S, T, D)`` gives
            ``(batch, steps, *S, T, *D)``: with ``seq=3``, a step output of
            ``Shape(S=(), T=1, D=(1,))`` gives ``(batch, 3, 1, 1)``.

        Raises
        ------
        ValueError
            If the model reads no fed-back Input ("reads no fed-back Input").
            If ``seq`` is not a positive int or None ("seq is a positive
            int"). If an input has no window ("declare it with tw=1"). If the
            stream a hidden state is fed back with reads an Input the model
            does not declare ("without declaring them"). If a declared
            fed-back Input also gets an initial value from the call of a
            block ("is among the inputs").

        See Also
        --------
        Loop : A rollout of a model inside a larger graph.
        Input.feedback : Feed a stream back into an Input at the next step.
        IntegrateStep : One step of an integral.
        DataLoader : The data of a model, cut into sequences with ``seq_length``.

        Notes
        -----
        With the state :math:`x_k` at the start of step :math:`k`, the input
        :math:`u_k` and the step :math:`f`:

        .. math:: x_{k+1} = f(x_k, u_k), \qquad y_k = x_{k+1}

        where the output :math:`y` collects the stream ``x`` is fed back with,
        and :math:`x_0` is the start of the state.

        :meth:`minimize` compares the outputs of the rollout,
        ``rollout.outputs["x"]`` or their names, which are trajectories, with
        targets declared with the horizon, ``Input("x_meas", seq=-1, tw=1)``.
        A stream of the step is of another graph, and refused.

        Basic Examples
        --------------
        A running total, the hidden state ``total``, fed back with
        ``total + rate``:

        >>> import keras
        >>> import numpy as np
        >>> from nnodely import Input, IntegrateStep, Modely, Output
        >>> rate = Input("rate", tw=1)
        >>> total = Input("total", tw=1)
        >>> total_next = total.feedback(total + rate, hidden=True)
        >>> step = Modely("counter", inputs=[rate], outputs=[Output("total", total_next)])
        >>> counter = step.loop().build()
        >>> result = counter({"rate": np.ones((1, 3, 1, 1))})  # (batch, steps, T, D)
        >>> keras.ops.convert_to_numpy(result["total"]).ravel()
        array([1., 2., 3.], dtype=float32)

        Scope
        -----
        Without ``seq``, the number of steps follows the data:

        >>> keras.ops.convert_to_numpy(counter({"rate": np.ones((1, 5, 1, 1))})["total"]).ravel()
        array([1., 2., 3., 4., 5.], dtype=float32)

        The rollout is a :class:`Modely` of the same name. Its one-step model,
        its first model, reads the hidden state as an input:

        >>> type(counter) is Modely, counter.name
        (True, 'counter')
        >>> sorted(node.name for node in counter.models[0].inputs)
        ['rate', 'total']

        Options
        -------
        ``seq`` fixes the number of steps:

        >>> fixed = step.loop(seq=3)
        >>> fixed.outputs[0].shape
        Shape(S=(3,), T=1, D=(1,))

        A declared state starts from the first step of its data, and from
        zero when the data leaves it out:

        >>> level = Input("level", tw=1)
        >>> level_next = level.feedback(level + rate)
        >>> listed = Modely("listed", inputs=[rate, level], outputs=[Output("level", level_next)])
        >>> rollout = listed.loop().build()
        >>> data = {"rate": np.ones((1, 3, 1, 1)), "level": np.full((1, 3, 1, 1), 10.0)}
        >>> keras.ops.convert_to_numpy(rollout(data)["level"]).ravel()
        array([11., 12., 13.], dtype=float32)
        >>> keras.ops.convert_to_numpy(rollout({"rate": np.ones((1, 3, 1, 1))})["level"]).ravel()
        array([1., 2., 3.], dtype=float32)

        It declares its state, and has the stream it is fed back with as an
        output, so it is the one-step model itself:

        >>> rollout.models[0] is listed
        True

        A hidden state starts from its ``init``, whatever the data holds:

        >>> memory = Input("memory", tw=1)
        >>> memory_next = memory.feedback(memory + rate, init=5.0, hidden=True)
        >>> kept = Modely("kept", inputs=[rate], outputs=[Output("memory", memory_next)]).loop()
        >>> data = {"rate": np.ones((1, 3, 1, 1)), "memory": np.full((1, 3, 1, 1), 10.0)}
        >>> keras.ops.convert_to_numpy(kept.build()(data)["memory"]).ravel()
        array([6., 7., 8.], dtype=float32)

        Applications
        ------------
        A mass of 2 kg pushed by a force, its velocity and position integrated
        with the Euler method, ``dt = 1``:

        >>> force = Input("force", tw=1)
        >>> v, x = Input("v", tw=1), Input("x", tw=1)
        >>> v_next = v.feedback(IntegrateStep(dt=1.0)(v, force / 2.0), hidden=True)
        >>> x_next = x.feedback(IntegrateStep(dt=1.0)(x, v_next), hidden=True)
        >>> mass = Modely("mass", inputs=[force], outputs=[Output("x", x_next)]).loop().build()
        >>> keras.ops.convert_to_numpy(mass({"force": np.ones((1, 3, 1, 1))})["x"]).ravel()
        array([0.5, 1.5, 3. ], dtype=float32)

        Properties & Relations
        ----------------------
        The running total is the cumulative sum of the rates:

        >>> rates = np.array([1.0, -2.0, 4.0]).reshape(1, 3, 1, 1)
        >>> totals = keras.ops.convert_to_numpy(counter({"rate": rates})["total"]).ravel()
        >>> bool(np.all(totals == np.cumsum(rates)))
        True

        Possible Issues
        ---------------
        A model without a fed-back Input has nothing to roll out. Feed one
        back, or use the model as it is:

        >>> Modely("plain", inputs=[rate], outputs=[Output("o", 2.0 * rate)]).loop()
        Traceback (most recent call last):
            ...
        ValueError: Model 'plain' reads no fed-back Input, so it has nothing to roll out...

        A state is declared among the inputs, or hidden. One that is neither
        is refused when the model is declared:

        >>> s = Input("s", tw=1)
        >>> s_next = s.feedback(s + rate)
        >>> Modely("open", inputs=[rate], outputs=[Output("s", s_next)])
        Traceback (most recent call last):
            ...
        ValueError: Model 'open' reads the Inputs ['s'], which are not among its inputs: declare them there...

        A stream of the step has no value in the rollout. The rollout is
        minimized on its outputs, the trajectories:

        >>> counter.minimize("bad", total_next, Input("measured", seq=-1, tw=1))
        Traceback (most recent call last):
            ...
        ValueError: Minimizer 'bad': its source ... is a stream of the graph of 'counter'...
        >>> counter = counter.minimize("fit", "total", Input("measured", seq=-1, tw=1))

        .. versionadded:: 2.0
        """
        from nnodely.layers.loop import Loop, _Step

        if seq is not None and (
            isinstance(seq, bool) or not isinstance(seq, int) or seq < 1
        ):
            raise ValueError(
                f"Model {self.name!r}: the seq of loop() is a positive int, or "
                f"None for the length of the data, got {seq!r}."
            )

        # This model is the step; the one Loop feeds back every Input it
        # feeds back.
        step = _Step(self, {}, who=f"Model {self.name!r}", among="its inputs")
        if not step.pairs:
            raise ValueError(
                f"Model {self.name!r} reads no fed-back Input, so it has nothing "
                "to roll out: feed one back with x.feedback(stream), or use the "
                "model as it is."
            )

        # The Input of the rollout of each Input of the step, by the Input it
        # stands for (_Step.key): the same name, the horizon its new outermost
        # sequence axis.
        rolled_inputs: dict[int, Input] = {}

        def rolled_input(node: Input) -> Input:
            key = step.key(node)
            rolled = rolled_inputs.get(key)
            if rolled is None:
                if node.tw is None:
                    raise ValueError(
                        f"Model {self.name!r} reads its Input {node.name!r} one "
                        "step at a time when rolled out, so its data carries the "
                        "horizon as a sequence axis, which needs a window: "
                        "declare it with tw=1."
                    )
                horizon = -1 if seq is None else seq
                rolled = Input(
                    node.name,
                    dim=node.dim,
                    seq=(horizon, *(-1 if size is None else size for size in node.seq)),
                    tw=(node.past, node.future),
                )
                rolled_inputs[key] = rolled
            return rolled

        # A fed-back Input among the inputs starts from its data, which may
        # leave it out: then from zero.
        listed = {id(node) for node in self._inputs}
        from_data = [state for state in step.states if step.key(state) in listed]
        for state in from_data:
            if state in step.starts:
                raise ValueError(
                    f"Model {self.name!r}: {state.name!r} has an initial value "
                    "(the stream a call gives it) and is among the inputs, "
                    "which would start it from its data: leave it out of "
                    "inputs, or give it no initial value."
                )
        declaration = Loop._of(
            step,
            init={state: rolled_input(state) for state in from_data},
            name=f"{self.name}_loop",
            seq=seq,
        )
        read = [*declaration._static_inputs, *declaration._seed_inputs]
        if not read and not from_data:
            raise ValueError(
                f"Model {self.name!r} reads no Input, in its steps or as the "
                "initial value of a fed-back Input, so no data sets the batch "
                "and number of steps of its rollout."
            )
        loop = declaration({node: rolled_input(node) for node in read})

        # One trajectory per output of the body, the declared outputs first.
        model = Modely(
            self.name,
            inputs=[rolled_input(node) for node in self._inputs],
            outputs=[
                Output(output.name, trajectory)
                for output, trajectory in zip(self._outputs, loop)
            ],
        )
        model._optional_inputs = {rolled_input(state) for state in from_data}
        model._rolled_inputs = rolled_inputs
        return model

    @property
    def built(self):
        """Whether :meth:`build` has created the Keras model, ``True`` or
        ``False``.

        A model is called on arrays, trained, validated and exported only once
        built. A model returned by :meth:`load` is built.

        See Also
        --------
        Modely.build : Create the Keras model and its weights.

        Basic Examples
        --------------
        >>> from nnodely import Input, Modely, Output
        >>> x = Input("x", tw=1)
        >>> model = Modely("double", inputs=[x], outputs=[Output("y", 2.0 * x)])
        >>> model.built
        False
        >>> model.build().built
        True

        .. versionadded:: 2.0
        """
        return self._keras_model is not None

    def build(self, keep_weights: bool = True):
        """Create the Keras model of the graph and its weights, and return the
        model itself.

        Call it after the objectives and the feedbacks are declared: they add
        to the graph that is built. Build again after a new objective. A
        :class:`DataLoader` is made for the built model.

        Parameters
        ----------
        keep_weights : bool, default True
            ``True`` keeps the weights the graph already holds: those of an
            earlier build, and those of a block built on its own. ``False``
            gives every weight its initial value again, in place: the
            declared value of a :class:`Parameter`, or a new draw of its
            initializer. Frozen weights, those of the blocks it calls and of
            its Loop bodies are reset too, and every model that shares them
            sees the new values.

        Returns
        -------
        Modely
            The model itself, so that ``Modely(...).build()`` chains.

        Raises
        ------
        ValueError
            If the graph reads no Input ("reads no Input"). If two different
            Inputs of the graph have one name ("reads two different Inputs
            named"): the data is keyed by input name. If a hidden state the
            user named, ``x.feedback(stream, hidden=True)``, has the name of
            another Input of the graph ("and one is a hidden state"): the
            state of a layer takes a free name instead.

        See Also
        --------
        Modely.built : Whether the model is built.
        Modely.minimize : Register an objective, before the build.
        Modely.train : Train the built model.
        DataLoader : The data of the built model.

        Notes
        -----
        A build that raises leaves the model as it was. Building creates the
        Keras variables of the weights: after it, a Parameter's ``param`` is
        the variable the model computes with.

        Basic Examples
        --------------
        >>> import keras
        >>> from nnodely import Constant, Input, Modely, Output, Parameter
        >>> x = Input("x", tw=1)
        >>> p = Parameter("p", value=[3.0])
        >>> model = Modely("scale", inputs=[x], outputs=[Output("y", p * x)]).build()
        >>> model.built
        True

        Options
        -------
        A new build keeps the weights; ``keep_weights=False`` starts them
        again from their declared value:

        >>> _ = p.param.assign([7.0])
        >>> model = model.build()
        >>> keras.ops.convert_to_numpy(p.param)
        array([7.], dtype=float32)
        >>> model = model.build(keep_weights=False)
        >>> keras.ops.convert_to_numpy(p.param)
        array([3.], dtype=float32)

        Possible Issues
        ---------------
        A graph of Parameters and Constants alone has no data to run on. Read
        an Input:

        >>> c = Constant("c", value=[1.0])
        >>> Modely("no_input", inputs=[], outputs=[Output("o", 2.0 * c)]).build()
        Traceback (most recent call last):
            ...
        ValueError: Model 'no_input' reads no Input...

        .. versionadded:: 2.0
        """
        from nnodely.core.layer import Identity

        _rename_hidden_states(self)
        _refresh_keras_inputs(self)
        # A minimizer reads both of its sides from the forward pass, so every
        # source and target has to be an output of the graph. An Input is not
        # the value of any layer, so it is exposed through an Identity.
        # Everything is computed aside and checked, and taken by the model
        # only once its Keras models are made.
        # The training model returns them by a key of their own: the name of
        # a declared output for one, else its name when no other output has
        # it, since two calls of one block return streams of one name. An Input
        # is keyed by its own name, not by the Identity's.
        extra_outputs, extra_keys = [], []
        minimizer_outputs: dict[Node, str] = {}
        keys = {node.name for node in self._outputs}
        for minimizer in self._minimizers:
            for node in (minimizer["source"], minimizer["target"]):
                if node in minimizer_outputs:
                    continue
                exposed = Identity()([node]) if isinstance(node, Input) else node
                if exposed in self._outputs:
                    minimizer_outputs[node] = exposed.name
                    continue
                key = free_name(node.name, keys)
                keys.add(key)
                extra_outputs.append(exposed)
                extra_keys.append(key)
                minimizer_outputs[node] = key

        train_outputs = self._outputs + extra_outputs
        output_keys = [node.name for node in self._outputs] + extra_keys
        flat_outputs, flatten_memo = _flatten_graph(
            self._inputs, train_outputs, one_step=True
        )
        flat_order = toposort(flat_outputs)
        # A minimizer side, too, is computed by a layer called on streams.
        _refuse_uncalled(flat_order)
        train_inputs = [node for node in flat_order if isinstance(node, Input)]
        if not train_inputs:
            # Keras would only say that its outputs are not KerasTensors.
            raise ValueError(
                f"Model {self.name!r} reads no Input. A model is evaluated on "
                "data, and its Inputs are what the data feeds, the batch too: "
                "a graph of Parameters, Constants and numbers alone, such as a "
                "Loop seeded and driven by numbers only, has nothing to run on. "
                "Read at least one Input."
            )
        # The Input of the graph each copy stands for: the copy shares its
        # Keras input, which a hidden state read as its init does not.
        originals = {
            id(copied): node
            for node, copied in flatten_memo.items()
            if isinstance(node, Input)
            and isinstance(copied, Input)
            and copied._input is node._input
        }
        # The data is keyed by input name: two Inputs of one name would read
        # the same array, whatever their shapes.
        names = [node.name for node in train_inputs]
        twins = [name for name in names if names.count(name) > 1]
        if twins:
            raise ValueError(
                f"Model {self.name!r} reads two different Inputs named "
                f"{twins[0]!r}. The data is fed by input name, so every "
                "Input of a model needs a name of its own."
            )
        keras_inputs, keras_outputs = self._resolve_graph(
            flat_order, train_inputs, list(zip(output_keys, flat_outputs))
        )
        model = MinimizerModel(
            name=self.name + "_train",
            inputs=keras_inputs,
            outputs=keras_outputs,
        )

        # Inference sees the model as it was declared: its outputs, and only
        # the inputs those read. What the minimizers add (their sides, and the
        # inputs only they read, such as targets) is there for the loss and
        # stays in the training model. The inputs are read off the Keras graph
        # rather than the DAG, because a Parameter or Constant is wired to an
        # arbitrary input for its batch.
        inference_outputs = {
            node.name: keras_outputs[node.name] for node in self._outputs
        }
        read = {
            id(source)
            for tensor in inference_outputs.values()
            for source in _source_input_tensors(tensor)
        }
        inference_inputs = [node for node in train_inputs if id(node._input) in read]
        inference_model = keras.Model(
            name=self.name,
            inputs={node.name: keras_inputs[node.name] for node in inference_inputs},
            outputs=inference_outputs,
        )

        # Graph flattening builds shallow copies. Keep the public symbolic nodes
        # connected to the concrete Keras layers created from those copies, and
        # to the input shapes those were built for. The handle and what it
        # fits travel together, so composing this model later can tell whether
        # its layers can be reused where they land.
        for source_node, flat_node in flatten_memo.items():
            if isinstance(source_node, Layer) and isinstance(flat_node, Layer):
                source_node._layer = flat_node._layer
                source_node._layer_signature = flat_node._layer_signature
        self._minimizer_outputs = minimizer_outputs
        self._train_outputs = train_outputs
        self._originals = originals
        self._keys = {
            **{id(originals.get(id(node), node)): node.name for node in train_inputs},
            **{id(node): node.name for node in self._outputs},
        }
        self._train_inputs = train_inputs
        self._keras_model = model
        self._inference_inputs = inference_inputs
        self._keras_inference_model = inference_model

        if not keep_weights:
            # In place, so every model computing with them (a block built on
            # its own, a body) starts from them too. Each Keras layer once,
            # however many nodes apply it.
            layers: dict[int, Layer] = {}
            for node in _nodes_and_models(self)[0]:
                if isinstance(node, Layer) and node._layer is not None:
                    layers.setdefault(id(node._layer), node)
            for node in layers.values():
                node._reinitialize()
        return self

    def _resolve_graph(self, order, inputs, keyed_outputs):
        # Applying one layer several times yields one node per application, so
        # tensors are keyed by node identity: keying them by name made each
        # application overwrite the previous one. The concrete Keras layer is
        # keyed by the layer object the node applies, its source: applications
        # of one layer share its weights, the way repeated calls do in Keras,
        # and layers created apart never do, even under one name, as a model
        # and its reloaded copy have.
        tensor_map: dict[Node, Any] = {
            node: node._input for node in order if isinstance(node, Input)
        }
        layers_by_source: dict[int, Any] = {}
        keras_names: dict[str, Any] = {
            node._input.name: Input for node in order if isinstance(node, Input)
        }
        # A Parameter that layers only read as one of their weights (never as
        # their input, nor as an output) is read whole: it keeps a batch axis
        # of one rather than a copy for every sample, which the backward pass
        # would sum again.
        readers: dict[Node, list[tuple[Node, int]]] = {}
        for node in order:
            for index, pred in enumerate(node._preds):
                readers.setdefault(pred, []).append((node, index))
        exposed = {id(node) for _, node in keyed_outputs}

        def read_whole(node) -> bool:
            uses = readers.get(node, [])
            return (
                bool(uses)
                and id(node) not in exposed
                and all(
                    index > 0 and isinstance(reader, Layer) and reader._weight_labels
                    for reader, index in uses
                )
            )

        for node in [n for n in order if not isinstance(n, Input)]:
            if isinstance(node, Layer):
                shared = layers_by_source.get(id(node._source))
                if shared is not None:
                    node._layer, signature = shared
                    # A layer without weights may be built for its input's
                    # shape, an axis included: applied to another, it is built
                    # again. One with weights stays shared, as its weights are.
                    if not node._layer.weights:
                        node._layer_signature = signature
                if len(node._preds) == 0:  ## Parameters and Constants
                    anchor = next(iter(tensor_map.values()), None)
                    if isinstance(node, _Value):
                        tensor_map[node] = node.call([anchor], whole=read_whole(node))
                    else:
                        tensor_map[node] = node.call([anchor])
                else:
                    tensor_map[node] = node.call(
                        [tensor_map[pred] for pred in node._preds]
                    )
                layers_by_source[id(node._source)] = (
                    node._layer,
                    node._layer_signature,
                )
                _claim_keras_name(node._layer, keras_names)
            else:  ## Output or other non-Layer node
                tensor = tensor_map[node._preds[0]]
                if isinstance(node, Output) and node._loop_index is not None:
                    # One output of a Loop of several, among those it returns.
                    tensor = tensor[node._loop_index]
                if isinstance(node._preds[0], Input):
                    # A Keras model exposes what its layers compute, never one
                    # of its own inputs: an Input read out is passed through.
                    identity = keras.layers.Identity(name=f"{node.name}_identity")
                    _claim_keras_name(identity, keras_names)
                    tensor = identity(tensor)
                tensor_map[node] = tensor

        keras_inputs = {node.name: tensor_map[node] for node in inputs}

        # The declared outputs are addressed by name, so two different nodes
        # claiming one name would silently drop one of them. Calling a model
        # several times produces exactly that: its outputs all carry the
        # body's output names. The keys of the others are their own.
        keras_outputs = {}
        claimed: dict[str, Node] = {}
        for key, node in keyed_outputs:
            owner = claimed.get(key)
            if owner is not None and owner is not node:
                raise ValueError(
                    f"Model {self.name!r} has two different outputs named "
                    f"{key!r}. Calling one model several times returns outputs "
                    "that share its output names, so wrap them in Output nodes with "
                    "distinct names before exposing them."
                )
            claimed[key] = node
            keras_outputs[key] = tensor_map[node]
        return keras_inputs, keras_outputs

    # -------------------------------------------------------------------------
    # Train API
    # -------------------------------------------------------------------------

    def minimize(
        self,
        name: str,
        source: Output | Stream | str,
        target: Output | Stream | str | float | None = None,
        loss: str | dict[str, Any] | keras.losses.Loss | Callable = "mse",
        gain: float = 1.0,
    ):
        r"""Register an objective: a loss between two streams that
        :meth:`train` minimizes. Return the model itself.

        An objective is called a minimizer in the method names:
        ``minimize``, :attr:`minimizers` and
        :meth:`minimizers[name].remove() <nnodely.core.modely.Minimizer.remove>`. Register
        the objectives before :meth:`build`, which adds their streams to the
        training model. The name of an objective is its key: registering a
        name again replaces that objective, in its place, with a warning.

        Both sides are computed from the graph of the model, from values, and
        from Inputs only the objectives read, such as targets. They have one
        shape, but for a value, which broadcasts: nothing is reshaped. A
        stream of the graph of another model (a block called, the body of a
        :class:`Loop`, the step :meth:`loop` rolls out) is refused: the model
        has no value for it. A trajectory is minimized as the output of the
        model that collects it, ``msd.outputs["x"]`` or its name, against a
        target declared with its horizon, ``Input("x_meas", seq=-1, tw=1)``.

        Parameters
        ----------
        name : str
            The name of the objective. Training logs its loss as
            ``"loss_<name>"``, and ``"val_loss_<name>"`` with validation data.
        source : Output, Stream or str
            The prediction: an :class:`Output`, any stream of the model, or the
            name of one.
        target : Output, Stream, str, float or None, default None
            What the prediction is compared with:

            - an Input, or a stream computed from Inputs, read from the data,
              of the shape of ``source``;
            - the name of a stream of the model;
            - a number, compared with every value of ``source``;
            - ``None``, the same as ``0.0``: the objective drives ``source``
              to zero, as a physical residual.

            An Input only the target reads need not be among the inputs of
            the model.
        loss : str, dict, keras.losses.Loss or callable, default "mse"
            The loss function: a Keras loss name (``"mse"``, ``"mae"``,
            ``"huber"``, ``"log_cosh"``, ...), a serialized Keras loss, a
            ``keras.losses.Loss`` instance, or a function
            ``loss(y_true, y_pred)``.
        gain : float, default 1.0
            The weight of the objective in the total loss, a finite number not
            below zero. ``0.5`` halves its importance, ``2.0`` doubles it. Its
            own logged and validated loss is not weighted.

        Returns
        -------
        Modely
            The model itself.

        Raises
        ------
        ValueError
            If ``name`` is not a non-empty string ("A minimizer name is a
            non-empty string"). If ``loss`` is unknown ("Unknown or invalid Keras loss"). If ``gain``
            is negative or not finite ("must be finite and not negative").
            If a side reads the graph of another model ("is a stream of the
            graph of"), a fed-back Input the model does not read ("is fed
            back, and model"), or another Input of the name of one the model
            reads ("is not the Input"). If the target, read from the data,
            has another shape than the source ("must have the same shape").
        KeyError
            If a name given for ``source`` or ``target`` is the name of no
            stream of the model ("is not the name of a stream"). The error is
            also a ``ValueError``.
        TypeError
            If ``gain`` is not a number, or ``source`` is neither a stream nor
            a name.

        Warns
        -----
        UserWarning
            If an objective of that name exists ("it is replaced by the new
            one").

        See Also
        --------
        Modely.minimizers : The objectives, as a table.
        Modely.train : Minimize the objectives.
        Modely.validate : Score every objective on a dataset.

        Notes
        -----
        Training minimizes the total loss of the :math:`n` objectives, each
        loss :math:`L_i` weighted by its gain :math:`g_i`:

        .. math:: L = \sum_{i=1}^{n} g_i \, L_i

        :math:`L_i` is the mean of the loss function over the samples. The
        rollout steps padded onto shorter simulations add zero. Only the
        objectives of this model are trained: those of the models it calls
        are each trained by their own model.

        Basic Examples
        --------------
        The output ``y`` is compared with a measured Input, ``y_meas``:

        >>> from nnodely import Input, Modely, Output, Parameter
        >>> x = Input("x", tw=1)
        >>> y = Output("y", x + Parameter("b", value=[1.0]))
        >>> y_meas = Input("y_meas", tw=1)
        >>> model = Modely("offset", inputs=[x], outputs=[y])
        >>> model = model.minimize("fit", y, y_meas)
        >>> print(model.minimizers)
        ┏━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━┓
        ┃ idx ┃ minimizer ┃ source ┃ target ┃ loss               ┃ gain ┃
        ┡━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━┩
        │ 0   │ fit       │ y      │ y_meas │ mean_squared_error │ 1.0  │
        └─────┴───────────┴────────┴────────┴────────────────────┴──────┘

        Scope
        -----
        A stream given by its name, compared with a number, and a stream
        driven to zero, ``target=None``:

        >>> model = model.minimize("level", "y", 2.0)
        >>> model = model.minimize("residual", y)
        >>> print(model.minimizers)
        ┏━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━┓
        ┃ idx ┃ minimizer ┃ source ┃ target ┃ loss               ┃ gain ┃
        ┡━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━┩
        │ 0   │ fit       │ y      │ y_meas │ mean_squared_error │ 1.0  │
        │ 1   │ level     │ y      │ 2.0    │ mean_squared_error │ 1.0  │
        │ 2   │ residual  │ y      │ 0.0    │ mean_squared_error │ 1.0  │
        └─────┴───────────┴────────┴────────┴────────────────────┴──────┘

        Options
        -------
        ``loss`` and ``gain``:

        >>> model = model.minimize("robust", y, y_meas, loss="mae", gain=0.5)
        >>> model.minimizers["robust"].gain
        0.5

        Possible Issues
        ---------------
        A name registered again replaces its objective, with a warning:

        >>> import warnings
        >>> with warnings.catch_warnings(record=True) as caught:
        ...     warnings.simplefilter("always")
        ...     model = model.minimize("level", y, 3.0)
        >>> str(caught[0].message)
        "Minimizer 'level' already exists in model 'offset': it is replaced by the new one."

        A target of another shape than the source is refused, not broadcast:

        >>> model.minimize("window", y, Input("y_window", tw=3))
        Traceback (most recent call last):
            ...
        ValueError: Minimizer 'window' compares 'y' of shape Shape(S=(), T=1, D=(1,)) with 'y_window' of shape Shape(S=(), T=3, D=(1,))...

        .. versionadded:: 2.0
        """
        self._check_minimizer_name(
            name, [m for m in self._minimizers if m["name"] != name]
        )
        source = _minimizer_stream(self, source, "source", name)
        if target is None:  ## Transform it into a Constant with value zero
            target = 0.0
        # Any real number, numpy scalars included; a bool is not a target value.
        if isinstance(target, (int, float, np.integer, np.floating)) and not isinstance(
            target, (bool, np.bool_)
        ):
            target = Constant(name=None, value=[float(target)])
        target = _minimizer_stream(self, target, "target", name)
        _check_sides(self, source, target, name)
        resolved_loss = _resolve_loss(loss)
        gain = _validate_gain(gain, name)
        minimizer = {
            "name": name,
            "source": source,
            "target": target,
            "loss": resolved_loss,
            "gain": gain,
        }
        for index, existing in enumerate(self._minimizers):
            if existing["name"] == name:
                warnings.warn(
                    f"Minimizer {name!r} already exists in model {self.name!r}: "
                    "it is replaced by the new one.",
                    UserWarning,
                    stacklevel=2,
                )
                self._minimizers[index] = minimizer
                return self
        self._minimizers.append(minimizer)
        return self

    def _check_minimizer_name(self, name, others: list[dict]) -> None:
        """Refuse ``name`` for a minimizer beside the minimizers ``others``."""
        if not isinstance(name, str) or not name:
            raise ValueError(f"A minimizer name is a non-empty string, got {name!r}.")
        # Its loss is logged under a key of its name, loss_<name>, which no
        # other name and no total of Keras can have: only the name is checked.
        if any(other["name"] == name for other in others):
            raise ValueError(
                f"Model {self.name!r} has a minimizer named {name!r} already: "
                "each minimizer has a name of its own, the key of its loss."
            )

    def _set_minimizer(self, minimizer: dict, key: str, value):
        """Set the name, loss or gain, ``key``, of ``minimizer``, one of
        the model's, checked as :meth:`minimize` checks it."""
        if not any(m is minimizer for m in self._minimizers):
            raise ValueError(
                f"Minimizer {minimizer['name']!r} was removed from model "
                f"{self.name!r}: declare it again with minimize()."
            )
        if key == "name":
            self._check_minimizer_name(
                value, [m for m in self._minimizers if m is not minimizer]
            )
        elif key == "gain":
            value = _validate_gain(value, minimizer["name"])
        else:
            value = _resolve_loss(value)
        minimizer[key] = value

    # -------------------------------------------------------------------------
    # Parameters and freezing
    # -------------------------------------------------------------------------

    @property
    def parameters(self) -> Parameters:
        """The Parameters the model computes with, as a list printed as a table.

        It holds every :class:`Parameter` of the graph, in the order the
        graph reaches them:

        - those the streams read;
        - the weights of its layers, such as the kernel of a :class:`Fir`;
        - those of the models it calls as blocks and of its Loop bodies.

        The list holds the Parameter objects themselves. Pick one by its name,
        ``parameters["k"]``, or by its index, the ``idx`` column of the table,
        ``parameters[0]``. A name several Parameters carry raises
        ``ValueError``, listing their indices. A name none carries raises
        ``KeyError``.

        The table gives the name, ``seq``, ``tw`` and ``dim`` of each, its
        number of values, and whether training changes it. When the models it
        calls have Parameters, a ``model`` column tells whose each is, and the
        rows of each model stand together, between rules. A Parameter is
        listed with the first model whose graph reaches it. The
        last row totals the values of all the Parameters and of the trainable
        ones.

        See Also
        --------
        Parameter : A trainable value of the graph.
        Modely.freeze : Keep Parameters as they are through training.
        Modely.models : The models the model calls.

        Basic Examples
        --------------
        >>> from nnodely import Fir, Input, Modely, Output, Parameter
        >>> x = Input("x", tw=2)
        >>> y = Fir(name="fir")(x) * Parameter("gain", value=[2.0])
        >>> model = Modely("filter", inputs=[x], outputs=[Output("y", y)])
        >>> print(model.parameters)
        ┏━━━━━┳━━━━━━━━━━━━┳━━━━━┳━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┓
        ┃ idx ┃ parameter  ┃ seq ┃ tw ┃ dim    ┃ values ┃ trainable ┃
        ┡━━━━━╇━━━━━━━━━━━━╇━━━━━╇━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━┩
        │ 0   │ fir_kernel │ ()  │ -  │ (2, 1) │ 2      │ yes       │
        │ 1   │ fir_bias   │ ()  │ -  │ (1,)   │ 1      │ yes       │
        │ 2   │ gain       │ ()  │ -  │ (1,)   │ 1      │ yes       │
        ├─────┼────────────┼─────┼────┼────────┼────────┼───────────┤
        │     │ total      │     │    │        │ 4      │ 4         │
        └─────┴────────────┴─────┴────┴────────┴────────┴───────────┘

        Scope
        -----
        By name or by index:

        >>> model.parameters["gain"] is model.parameters[2]
        True

        The Parameters of a block are listed under its name:

        >>> u = Input("u", tw=2)
        >>> outer = Modely("outer", inputs=[u], outputs=[Output("z", model(u) - 1.0)])
        >>> [p.name for p in outer.parameters]
        ['fir_kernel', 'fir_bias', 'gain']

        Possible Issues
        ---------------
        A name nothing carries raises ``KeyError``:

        >>> model.parameters["bias"]
        Traceback (most recent call last):
            ...
        nnodely.core.dag.NameNotFound: ...

        .. versionadded:: 2.0
        """
        return Parameters(
            [
                (model.name, [node for node in nodes if isinstance(node, Parameter)])
                for model, nodes in _nodes_by_model(self)
            ],
            title=f"Parameters of {self.name}",
        )

    @property
    def models(self) -> "Models":
        """The models this model computes with, as a list printed as a table.

        It holds the models it calls as blocks, the bodies of its Loops, and
        theirs, at every depth. They come in the order the graph reaches
        them, each once however many times it is called. A
        :class:`LocalModel`, an :class:`EquationLearner` and the rollout of
        :meth:`loop` call models of their own, listed too.

        The list holds the Modely objects themselves, not copies. They share
        their weights with this model: training, freezing or building one
        again changes this model too. Pick one by its name,
        ``models["block"]``, or by its index, the ``idx`` column of the
        table. A name several models carry raises ``ValueError``: pick one by
        its index.

        The table gives the name of each, its inputs, its outputs, its number
        of Parameters, and whether it is built. :meth:`save` writes each in a
        folder of its own, so a loaded model lists the same models.

        See Also
        --------
        Modely.parameters : The Parameters, those of these models included.
        Modely.freeze : Freeze the Parameters of a model by its name.
        Loop : A rollout of a model inside a larger graph.

        Basic Examples
        --------------
        >>> from nnodely import Input, Modely, Output, Parameter
        >>> x = Input("x", tw=1)
        >>> y = Output("y", x + Parameter("b", value=[1.0]))
        >>> offset = Modely("offset", inputs=[x], outputs=[y])
        >>> u = Input("u", tw=1)
        >>> outer = Modely("outer", inputs=[u], outputs=[Output("z", 2.0 * offset(u))])
        >>> print(outer.models)
        ┏━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━┓
        ┃ idx ┃ model  ┃ inputs ┃ outputs ┃ parameters ┃ built ┃
        ┡━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━┩
        │ 0   │ offset │ x      │ y       │ 1          │ no    │
        └─────┴────────┴────────┴─────────┴────────────┴───────┘

        Scope
        -----
        By name or by index, the block itself:

        >>> outer.models["offset"] is offset, outer.models[0] is offset
        (True, True)

        A model that calls none has an empty list:

        >>> len(offset.models)
        0

        .. versionadded:: 2.0
        """
        holders: dict[int, Modely] = {}
        models = [model for model, _ in _nodes_by_model(self, holders)][1:]
        return Models(
            models,
            title=f"Models of {self.name}",
            where=[f"in {holders[id(model)].name!r}" for model in models],
        )

    @property
    def inputs(self) -> Inputs:
        """The Inputs the model reads, as a list printed as a table.

        It holds, inside the models it calls too:

        - the Inputs it declares;
        - the Inputs only an objective reads, such as targets;
        - its hidden states (``x.feedback(stream, hidden=True)``, the states
          of an Integrate, a Derivative or a Buffer), which it reads the init
          of;
        - the states fed back inside the step of a Loop it rolls out, as the
          rollout of :meth:`loop` does.

        The list holds the Input objects themselves. Pick one by its name,
        ``inputs["x"]``, or by its index, the ``idx`` column of the table.
        Renaming one, ``inputs[0].name = "speed"``, renames it in the model;
        build again to train with the new name.

        The table gives the name, ``seq``, ``tw`` and ``dim`` of each Input,
        and how it is fed:

        - flags: ``target``, read to train and validate only; ``optional``,
          read as zeros where the data leaves it out (only the rollout of
          :meth:`loop` has one); ``hidden``, a state that is the model's own
          memory and starts from its init, never from data.
        - update: ``<- v``, what it reads at every step; ``x₀ = v``, what it
          starts from. ``#['name']`` is the data of an Input, its first step
          in a rollout. A state shows both: ``<- w``, the stream it is fed
          back with, and its initial value (``0`` without one). The one-step
          model shows them too, though only a rollout feeds a state back: it
          reads a declared one from data at every step, and a hidden one its
          init.

        See Also
        --------
        Input : A signal read from data.
        Modely.outputs : The outputs, as a table.
        DataLoader : The data of the Inputs.

        Basic Examples
        --------------
        >>> from nnodely import Input, Modely, Output
        >>> x = Input("x", tw=1)
        >>> y = Output("y", 2.0 * x)
        >>> model = Modely("double", inputs=[x], outputs=[y])
        >>> model = model.minimize("fit", y, Input("y_meas", tw=1))
        >>> [(node.name, node.tw) for node in model.inputs]
        [('x', 1), ('y_meas', 1)]
        >>> model.inputs["x"] is x
        True

        A running total, a hidden state, against a target:

        >>> x, total = Input("x", tw=1), Input("total", tw=1)
        >>> running = x + total
        >>> running.name = "running"
        >>> _ = total.feedback(running, hidden=True)
        >>> model = Modely("sum", inputs=[x], outputs=[Output("y", running)])
        >>> model = model.minimize("fit", "y", Input("y_meas", tw=1))
        >>> print(model.inputs)
        ┏━━━━━┳━━━━━━━━┳━━━━━┳━━━━┳━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ idx ┃ input  ┃ seq ┃ tw ┃ dim  ┃ flags  ┃ update                 ┃
        ┡━━━━━╇━━━━━━━━╇━━━━━╇━━━━╇━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ 0   │ x      │ ()  │ 1  │ (1,) │ -      │ <- #['x']              │
        │ 1   │ y_meas │ ()  │ 1  │ (1,) │ target │ <- #['y_meas']         │
        │ 2   │ total  │ ()  │ 1  │ (1,) │ hidden │ <- running, total₀ = 0 │
        └─────┴────────┴─────┴────┴──────┴────────┴────────────────────────┘

        .. versionadded:: 2.0
        """
        return Inputs(self._input_rows())

    def _input_rows(self) -> list[dict]:
        """A row of :attr:`inputs` per Input the graph reads, then one per
        state of its Loops that does not start from the data of one."""
        from nnodely.layers.input import _HiddenInit
        from nnodely.layers.loop import Loop

        # A row per Input object: two Inputs of one name (renamed alike)
        # are two rows, and the model refuses them when built.
        declared = {id(node) for node in self._inputs}
        sides = [
            side
            for minimizer in self._minimizers
            for side in (minimizer["source"], minimizer["target"])
        ]
        # The rows give the Inputs of the graph, not the copies reading it
        # makes: an Input a called model reads is the one to pass on.
        forward_originals: dict[int, Input] = {}
        forward = {
            id(forward_originals.get(id(node), node))
            for node in _read_nodes(self._outputs, self._inputs, forward_originals)
            if isinstance(node, Input)
        }
        originals: dict[int, Input] = {}
        copies: dict[int, Node] = {}
        hidden: list[Input] = []
        read = _read_nodes(
            [*self._outputs, *sides], self._inputs, originals, copies, hidden
        )
        # The Inputs read at every step: not those only the initial value of
        # a state reads, of a Loop or a hidden one.
        every_step = {
            id(originals.get(id(root), root)) for root in [*self._outputs, *sides]
        }
        for node in read:
            preds = node._preds
            if isinstance(node, Loop):
                preds = preds[len(node._feedback_inputs) :]
            elif isinstance(node, _HiddenInit):
                preds = []
            every_step.update(id(originals.get(id(pred), pred)) for pred in preds)
        rows = []
        by_input: dict[int, dict] = {}
        for node in read:
            if not isinstance(node, Input):
                continue
            original = originals.get(id(node), node)
            if id(original) in by_input:
                continue
            data = (
                "target"
                if id(original) not in forward
                else "optional"
                if original in self._optional_inputs
                else "required"
            )
            data_cell = f"#[{original.name!r}]"
            by_input[id(original)] = {
                "input": original,
                "seq": tuple(node.seq),
                "declared": id(original) in declared,
                "flags": None if data == "required" else data,
                "update": f"<- {data_cell}"
                if node.update is None and id(original) in every_step
                else f"{original.name}₀ = {data_cell}"
                if node.update is None
                else f"<- {node.update.name}, {original.name}₀ = {data_cell}",
                # The stream of the graph it is fed back with.
                "update stream": None
                if node.update is None
                else copies.get(id(node.update), node.update),
                "data": data,
            }
            rows.append(by_input[id(original)])

        # The states: the hidden ones the one-step model reads the init of, at
        # every step, then those inside the steps of its Loops, with the
        # stream of the graph each is fed back with.
        states: list[tuple[Input, Stream, Any, Any]] = [
            (state, cast(Stream, state.update), state.init, state.update)
            for state in hidden
        ]
        for loop in (node for node in read if isinstance(node, Loop)):
            states += [
                (state, update, start, loop._step.copied_from(update))
                for state, update, start, _ in loop._state_starts()
            ]
        seen: set[int] = set()
        for state, update, start, graph_update in states:
            if id(state) in seen:
                continue
            seen.add(id(state))
            # Read here as a copy, too.
            if isinstance(start, Input):
                start = originals.get(id(start), start)
            # One that starts from an optional Input (the Input of a rollout
            # of loop() it starts from, by object, renamed or not) is that
            # Input, read from data.
            row = (
                by_input.get(id(start))
                if isinstance(start, Input) and start in self._optional_inputs
                else None
            )
            if row is not None:
                starts = f"#[{start.name!r}] or 0"
            else:
                row = {
                    "input": state,
                    "seq": tuple(state.seq),
                    "declared": False,
                    "flags": "hidden" if state._hidden else None,
                    "data": None,
                }
                rows.append(row)
                starts = (
                    "0"
                    if start is None
                    else f"#[{start.name!r}]"
                    if isinstance(start, Input)
                    else start.name
                    if isinstance(start, Stream)
                    else str(start)
                )
            row["update"] = f"<- {update.name}, {row['input'].name}₀ = {starts}"
            row["update stream"] = graph_update
        return rows

    @property
    def outputs(self) -> Outputs:
        """The outputs the model was declared with, as a list printed as a
        table.

        The list holds the Output objects, in the order of ``outputs``. Pick
        one by its name, ``outputs["y"]``, or by its index, the ``idx``
        column of the table.

        The table gives the name, ``seq``, ``tw`` and ``dim`` of each output,
        and:

        - update: ``-> s``, the Inputs a loop feeds the stream of the output
          back into at the next step, as :attr:`inputs` shows them with
          ``<- w``. In the rollout of :meth:`loop`, that stream is the one of
          the step the trajectory collects;
        - minimized by: the objectives that read it, as ``name (source)`` or
          ``name (target)``.

        See Also
        --------
        Output : A stream the model exposes.
        Modely.inputs : The Inputs, as a table.
        Modely.minimizers : The objectives, as a table.

        Basic Examples
        --------------
        >>> from nnodely import Input, Modely, Output
        >>> x = Input("x", tw=1)
        >>> doubled = 2.0 * x
        >>> doubled.name = "doubled"
        >>> y = Output("y", doubled)
        >>> model = Modely("double", inputs=[x], outputs=[y]).minimize("fit", y, 4.0)
        >>> print(model.outputs)
        ┏━━━━━┳━━━━━━━━┳━━━━━┳━━━━┳━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━┓
        ┃ idx ┃ output ┃ seq ┃ tw ┃ dim  ┃ update ┃ minimized by ┃
        ┡━━━━━╇━━━━━━━━╇━━━━━╇━━━━╇━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━┩
        │ 0   │ y      │ ()  │ 1  │ (1,) │ -      │ fit (source) │
        └─────┴────────┴─────┴────┴──────┴────────┴──────────────┘
        >>> model.outputs["y"] is y
        True

        .. versionadded:: 2.0
        """
        return Outputs(self._output_rows())

    def _output_rows(self) -> list[dict]:
        """A row of :attr:`outputs` per declared output."""
        # By the stream of the graph each Input is fed back with: the rows
        # of :attr:`inputs` read copies of it, and give it.
        fed_back: dict[int, list[str]] = {}
        for row in self._input_rows():
            if row["update stream"] is not None:
                fed_back.setdefault(id(row["update stream"]), []).append(
                    row["input"].name
                )
        minimized: dict[int, list[str]] = {}
        for minimizer in self._minimizers:
            for side in ("source", "target"):
                minimized.setdefault(id(minimizer[side]), []).append(
                    f"{minimizer['name']} ({side})"
                )
        rows = []
        for output in self._outputs:
            stream = self._output_stream(output)
            rows.append(
                {
                    "output": output,
                    "stream": stream.name,
                    "update": fed_back.get(id(stream), fed_back.get(id(output), [])),
                    "minimized by": [
                        *minimized.get(id(output), []),
                        *minimized.get(id(output._preds[0]), []),
                    ],
                }
            )
        return rows

    def _output_stream(self, output: Output) -> Node:
        """The stream ``output`` names: for the trajectory of a Loop, the
        stream of its body it collects, as the graph the Loop was declared
        over holds it (the step of a rollout of loop(), which the body copies)."""
        from nnodely.layers.loop import Loop

        stream = output._preds[0]
        # One output of a Loop of several is an Output of the loop.
        index = stream._loop_index if isinstance(stream, Output) else None
        loop = stream._preds[0] if index is not None else stream
        if not isinstance(loop, Loop):
            return stream
        if index is not None:
            collected = loop._model._outputs[index]
        else:
            collected = loop._returns[0]  # the one output it returns
        return loop._step.copied_from(collected._preds[0])

    @property
    def minimizers(self) -> "Minimizers":
        """The objectives :meth:`train` minimizes, as a list printed as a table.

        An objective is called a minimizer in the method names. The list holds
        one entry per objective, in the order they were registered. Pick one
        by its name, ``minimizers["fit"]``, or by its index, the ``idx``
        column of the table. A name none carries raises ``KeyError``.

        Each entry reads as a dict and as an object, ``m["gain"]`` or
        ``m.gain``. Its keys are:

        - ``name``: the name of the objective;
        - ``source`` and ``target``: the streams it compares. A number given
          as the target is a :class:`Constant`;
        - ``loss``: the loss function;
        - ``gain``: its weight in the total loss;
        - ``model``: the model it belongs to.

        Setting ``name``, ``loss`` or ``gain`` of an entry changes the
        objective in the model, checked as :meth:`minimize` checks it. The
        next :meth:`train` uses it, without a new build. Setting anything
        else raises ``TypeError``: register the objective again with
        :meth:`minimize` to change what it compares. :meth:`minimizers[name].remove() <nnodely.core.modely.Minimizer.remove>`
        removes the objective from the model.

        The table gives the name, source, target, loss and gain of each. The
        objectives of the models it calls are not listed and not trained:
        each is minimized by the :meth:`train` of its own model.

        See Also
        --------
        Modely.minimize : Register an objective.
        Modely.train : Minimize the objectives.

        Basic Examples
        --------------
        >>> from nnodely import Input, Modely, Output
        >>> x = Input("x", tw=1)
        >>> y = Output("y", 2.0 * x)
        >>> model = Modely("double", inputs=[x], outputs=[y])
        >>> model = model.minimize("fit", y, Input("y_meas", tw=1))
        >>> print(model.minimizers)
        ┏━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━┓
        ┃ idx ┃ minimizer ┃ source ┃ target ┃ loss               ┃ gain ┃
        ┡━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━┩
        │ 0   │ fit       │ y      │ y_meas │ mean_squared_error │ 1.0  │
        └─────┴───────────┴────────┴────────┴────────────────────┴──────┘

        Scope
        -----
        Every key of an entry, read as a key or as an attribute:

        >>> fit = model.minimizers["fit"]
        >>> fit["source"] is y, fit.target.name, fit.gain, fit.model is model
        (True, 'y_meas', 1.0, True)

        Options
        -------
        Setting the gain, the loss or the name changes the objective:

        >>> model.minimizers[0]["gain"] = 0.5
        >>> model.minimizers[0].loss = "mae"
        >>> fit.name = "track"
        >>> print(model.minimizers)
        ┏━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━┓
        ┃ idx ┃ minimizer ┃ source ┃ target ┃ loss                ┃ gain ┃
        ┡━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━┩
        │ 0   │ track     │ y      │ y_meas │ mean_absolute_error │ 0.5  │
        └─────┴───────────┴────────┴────────┴─────────────────────┴──────┘

        Possible Issues
        ---------------
        What an objective compares is set by :meth:`minimize` only:

        >>> fit.source = y
        Traceback (most recent call last):
            ...
        TypeError: Minimizer 'track': its name, loss and gain can be set, not its 'source'; ...

        .. versionadded:: 2.0
        """
        return Minimizers(
            [Minimizer(minimizer, self) for minimizer in self._minimizers]
        )

    def freeze(self, names: str | list[str] | None = None):
        """Keep Parameters of the model as they are through training, and
        return the model itself.

        Without ``names``, every Parameter :attr:`parameters` lists is frozen:
        those of its streams, the weights of its layers, and those of the
        models it calls and of its Loop bodies. :meth:`unfreeze` undoes it.
        It works before and after :meth:`build`, and the next :meth:`train`
        follows it.

        Parameters
        ----------
        names : str, list of str or None, default None
            What to freeze. ``None`` freezes every Parameter. A name, or a
            list of names, picks:

            - a Parameter, by its name as :attr:`parameters` lists it;
            - a model the graph reaches, by its name, as :attr:`models` lists
              it: every Parameter of that model. It is a block the model
              calls, the body of a Loop, or the model of a
              :class:`LocalModel`.

        Returns
        -------
        Modely
            The model itself.

        Raises
        ------
        KeyError
            If a name is the name of no Parameter and no model ("reaches no
            Parameter and no model named"). The error is also a
            ``ValueError``.
        ValueError
            If a name is carried by several Parameters or models ("is
            ambiguous"). The message lists them with their index: freeze one
            by itself, ``model.parameters[i].freeze()``.
        TypeError
            If a name is not a string.

        See Also
        --------
        Modely.unfreeze : Let training change Parameters again.
        Modely.parameters : The Parameters, with whether each is trainable.
        Parameter : A trainable value, frozen alone with its ``freeze()``.

        Notes
        -----
        Every name is checked before any Parameter changes: a wrong name
        changes nothing. A Parameter is one object wherever it is read, so a
        Parameter shared with another model is frozen there too. Two kinds of
        weights are not Parameters, and are not frozen:

        - the Keras weights of a :class:`KerasFunction`: set ``trainable`` on
          its Keras layer;
        - the moving statistics of a :class:`BatchNorm`, which training keeps
          updating.

        Basic Examples
        --------------
        >>> from nnodely import Input, Modely, Output, Parameter
        >>> x = Input("x", tw=1)
        >>> y = Parameter("a", value=[1.0]) * x + Parameter("b", value=[0.0])
        >>> model = Modely("line", inputs=[x], outputs=[Output("y", y)])
        >>> trainable = lambda: [(p.name, p.trainable) for p in model.parameters]
        >>> model = model.freeze()
        >>> trainable()
        [('a', False), ('b', False)]
        >>> model = model.unfreeze()

        Options
        -------
        A name, or a list of names:

        >>> model = model.freeze("b")
        >>> trainable()
        [('a', True), ('b', False)]
        >>> model = model.unfreeze("b")

        The name of a block freezes all its Parameters:

        >>> u = Input("u", tw=1)
        >>> outer = Modely("outer", inputs=[u], outputs=[Output("z", 2.0 * model(u))])
        >>> outer = outer.freeze("line")
        >>> trainable()
        [('a', False), ('b', False)]
        >>> outer = outer.unfreeze()

        Possible Issues
        ---------------
        A name several Parameters carry is refused. Freeze one by its index:

        >>> p, q = Parameter("w", value=[1.0]), Parameter("w", value=[2.0])
        >>> twins = Modely("twins", inputs=[x], outputs=[Output("o", p * x + q)])
        >>> twins.freeze("w")
        Traceback (most recent call last):
            ...
        ValueError: Model 'twins': the name 'w' is ambiguous, 2 Parameters and models carry it: w (parameters[0]) in 'twins'; w (parameters[1]) in 'twins'...
        >>> _ = twins.parameters[1].freeze()
        >>> [p.trainable for p in twins.parameters]
        [True, False]
        >>> _ = twins.parameters[1].unfreeze()

        .. versionadded:: 2.0
        """
        for parameter in self._parameters_named(names):
            parameter.freeze()
        return self

    def unfreeze(self, names: str | list[str] | None = None):
        """Let training change Parameters of the model again, and return the
        model itself.

        It undoes :meth:`freeze`, and picks the Parameters as it does.

        Parameters
        ----------
        names : str, list of str or None, default None
            What to unfreeze. ``None`` unfreezes every Parameter. A name, or a
            list of names, picks a Parameter by its name, or a model the graph
            reaches by its name, for every Parameter of that model.

        Returns
        -------
        Modely
            The model itself.

        Raises
        ------
        KeyError
            If a name is the name of no Parameter and no model. The error is
            also a ``ValueError``.
        ValueError
            If a name is carried by several Parameters or models ("is
            ambiguous"): unfreeze one by itself,
            ``model.parameters[i].unfreeze()``.
        TypeError
            If a name is not a string.

        See Also
        --------
        Modely.freeze : Keep Parameters as they are through training.
        Modely.parameters : The Parameters, with whether each is trainable.

        Basic Examples
        --------------
        >>> from nnodely import Input, Modely, Output, Parameter
        >>> x = Input("x", tw=1)
        >>> y = Parameter("a", value=[1.0]) * x + Parameter("b", value=[0.0])
        >>> model = Modely("line", inputs=[x], outputs=[Output("y", y)]).freeze()
        >>> model = model.unfreeze("a")
        >>> [(p.name, p.trainable) for p in model.parameters]
        [('a', True), ('b', False)]
        >>> model = model.unfreeze()
        >>> [p.trainable for p in model.parameters]
        [True, True]

        Possible Issues
        ---------------
        A name nothing carries raises, and nothing changes:

        >>> model.unfreeze(["a", "c"])
        Traceback (most recent call last):
            ...
        nnodely.core.dag.NameNotFound: Model 'line' reaches no Parameter and no model named 'c'...

        .. versionadded:: 2.0
        """
        for parameter in self._parameters_named(names):
            parameter.unfreeze()
        return self

    def _parameters_named(self, names) -> list[Parameter]:
        """The Parameters ``names`` picks, every one for None. A name picks
        among the Parameters and the models the graph reaches; a model stands
        for all its Parameters. A name several carry raises, listing them.
        Each name is checked before any of them is used, so a wrong one
        changes nothing."""
        holders: dict[int, Modely] = {}
        groups = _nodes_by_model(self, holders)
        parameters = [
            node for _, nodes in groups for node in nodes if isinstance(node, Parameter)
        ]
        if names is None:
            return parameters
        keys = [names] if isinstance(names, str) else list(names)
        given = [key for key in keys if not isinstance(key, str)]
        if given:
            raise TypeError(
                f"Model {self.name!r} picks Parameters and models by name, a "
                "string or a list of strings. One of several of a name is "
                "frozen by itself, model.parameters[i].freeze(); got "
                f"{type(given[0]).__name__}."
            )
        models = [model for model, _ in groups[1:]]
        # Each with the list it is in, model.parameters or model.models, and
        # its index there.
        candidates = [
            *(
                (parameter, None, f"(parameters[{index}]) in {model.name!r}")
                for index, (model, parameter) in enumerate(
                    (model, node)
                    for model, nodes in groups
                    for node in nodes
                    if isinstance(node, Parameter)
                )
            ),
            *(
                (model, None, f"(models[{index}]) in {holders[id(model)].name!r}")
                for index, model in enumerate(models)
            ),
        ]
        picked: dict[int, Parameter] = {}
        for key in keys:
            item = resolve(
                key,
                candidates,
                kind="Parameter or model",
                kinds="Parameters and models",
                owner=f"Model {self.name!r}",
                hint="Freeze or unfreeze one by itself, model.parameters[i]"
                ".freeze() or model.models[i].freeze(), or give it a name of "
                "its own, obj.name = 'new_name'.",
                missing=lambda key: (
                    f"Model {self.name!r} reaches no Parameter "
                    f"and no model named {key!r}. Its Parameters are "
                    f"{[parameter.name for parameter in parameters]}, its models "
                    f"{list(dict.fromkeys(model.name for model in models))}."
                ),
            )
            for parameter in (
                [item] if isinstance(item, Parameter) else _parameters(item)
            ):
                picked.setdefault(id(parameter), parameter)
        return list(picked.values())

    # -------------------------------------------------------------------------
    # Validation API
    # -------------------------------------------------------------------------

    def validate(
        self,
        val_data: DataLoader,
        out_dir: str | os.PathLike | None = None,
        show: bool = False,
        history: TrainResult | dict[str, Any] | None = None,
        verbose: bool = False,
    ) -> ValidationResult:
        """Validate every minimizer of the built model on a dataset.

        Each minimizer is evaluated as training evaluates it: the same
        forward pass, the same target, aligned with the prediction the same
        way, and the same padding mask. Its loss is the one :meth:`train`
        logs as ``"val_loss_<minimizer>"`` for the same data.

        Parameters
        ----------
        val_data : DataLoader
            The data, made for this build of the model. It holds every Input
            training reads, the targets included.
        out_dir : str, os.PathLike or None, default None
            The folder :meth:`~nnodely.core.results.ValidationResult.plot` writes the figures to,
            as PNG files. ``None`` writes none.
        show : bool, default False
            ``True`` opens the figures in an interactive Matplotlib window.
        history : TrainResult, dict or None, default None
            The losses of a training, drawn as curves in a figure of their
            own.
        verbose : bool, default False
            ``True`` prints the table of the result.

        Returns
        -------
        ~nnodely.core.results.ValidationResult
            Read by its attributes: ``losses``, ``loss`` the total and
            ``loss_<minimizer>`` the loss of each, the keys of :meth:`train`;
            ``signals``, a :class:`~nnodely.core.validation.Signal` per
            minimizer, the signals it compared and their metrics;
            ``minimizers``, a snapshot of those validated; ``figures``, the
            files written.

        Raises
        ------
        ValueError
            If the model is not built ("Call build() before validate()"),
            has no objective ("No minimizers defined"), or the data is empty
            ("Validation dataset is empty"). If the data lacks an Input the
            model reads ("which the data does not hold"). If an Input or an
            output was renamed after the build ("build it again").

        See Also
        --------
        Modely.train : Train the model, with ``val_data`` too.
        Modely.minimize : Register an objective.
        DataLoader : The data of a model.

        Notes
        -----
        The forward pass is the inference one: a :class:`BatchNorm` uses its
        moving statistics. ``loss_<minimizer>`` is not weighted by the gain,
        ``loss`` weights each by its own, as training does.

        Basic Examples
        --------------
        A model that adds 1 to the target it should copy errs by 1 at every
        point:

        >>> from nnodely import DataLoader, Input, Modely, Output, Parameter
        >>> x = Input("x", tw=1)
        >>> y = Output("y", x + Parameter("b", value=[1.0]))
        >>> model = Modely("offset", inputs=[x], outputs=[y])
        >>> model = model.minimize("fit", y, Input("y_meas", tw=1)).build()
        >>> data = DataLoader(model, source={"x": [0.0, 2.0, 4.0, 6.0],
        ...                                  "y_meas": [0.0, 2.0, 4.0, 6.0]})
        >>> result = model.validate(data)
        >>> fit = result.signals["fit"]
        >>> result.losses["loss_fit"], fit.rmse(), fit.bias(), fit.r2()
        (1.0, 1.0, 1.0, 0.8)

        Options
        -------
        ``print`` shows a row per minimizer, and the total:

        >>> print(result)
        ┏━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━┓
        ┃ idx ┃ minimizer ┃ source ┃ target ┃ loss               ┃ gain ┃ loss value ┃
        ┡━━━━━┻━━━━━━━━━━━┻━━━━━━━━┻━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━┻━━━━━━┻━━━━━━━━━━━━┩
        │ offset: 4 samples                                                          │
        ├─────┬───────────┬────────┬────────┬────────────────────┬──────┬────────────┤
        │ 0   │ fit       │ y      │ y_meas │ mean_squared_error │ 1.0  │ 1.0000e+00 │
        ├─────┼───────────┼────────┼────────┼────────────────────┼──────┼────────────┤
        │     │ total     │        │        │                    │      │ 1.0000e+00 │
        └─────┴───────────┴────────┴────────┴────────────────────┴──────┴────────────┘

        ``out_dir`` writes the figures, and ``history`` adds the loss curves:

        >>> trained = model.train(data, epochs=1, batch_size=4, printer=None)
        >>> result = model.validate(data, out_dir="figures", history=trained)
        >>> from pathlib import Path
        >>> [Path(path).as_posix() for path in result.figures]
        ['figures/offset_0_fit.png', 'figures/offset_history.png']

        Possible Issues
        ---------------
        A model is validated on its objectives: declare one first.

        >>> plain = Modely("plain", inputs=[x], outputs=[Output("z", 2.0 * x)]).build()
        >>> plain.validate(data)
        Traceback (most recent call last):
            ...
        ValueError: No minimizers defined. Cannot infer validation targets.

        .. versionadded:: 2.0
        """
        if self._keras_model is not None:
            self._check_names_built()
        return validation.validate(
            self,
            val_data,
            out_dir=out_dir,
            show=show,
            history=history,
            verbose=verbose,
        )

    def _training_arrays(self, data: DataLoader) -> tuple[dict, np.ndarray | None]:
        """The model's inputs and, when simulations were padded, their mask.

        Both sides of every minimizer come from the forward pass, so the only
        label left is the mask that tells real rollout steps from padded ones.
        """
        x_data = {name: np.asarray(values) for name, values in data.as_dict().items()}
        # The data is keyed by the names of the build it was made for: a
        # DataLoader made before the model was built again holds others.
        missing = [node.name for node in self._train_inputs if node.name not in x_data]
        if missing:
            raise ValueError(
                f"Model {self.name!r} reads the inputs {missing}, which the data "
                f"does not hold; it holds {list(x_data)}. A DataLoader is keyed "
                "by the input names of the build it was made for: make it again "
                "once the model is built again."
            )
        mask = getattr(data, "mask", None)
        return x_data, None if mask is None else np.asarray(mask, dtype=np.float32)

    def train(
        self,
        train_data: DataLoader,
        val_data: DataLoader | None = None,
        epochs: int = 10,
        batch_size: int = 1,
        optimizer: str | dict[str, Any] | keras.optimizers.Optimizer | None = None,
        early_stopping: str | dict[str, Any] | keras.callbacks.Callback | None = None,
        lr: float = 1e-3,
        shuffle: bool = True,
        optimizer_kwargs: dict[str, Any] | None = None,
        early_stopping_kwargs: dict[str, Any] | None = None,
        printer: str | keras.callbacks.Callback | None = "legacy",
        jit_compile: str | bool = "auto",
    ):
        r"""Train the built model on its objectives, and return the losses of
        every epoch.

        Every epoch goes once through the samples of ``train_data``, in
        batches of ``batch_size``, and updates the trainable Parameters. Each
        call continues from the current weights. The defaults train 10 epochs
        with a batch of 1 sample: one update per sample, slow on any real
        dataset. Pass ``epochs`` and ``batch_size``.

        Parameters
        ----------
        train_data : DataLoader
            The training data, made for this build of the model.
        val_data : DataLoader or None, default None
            The validation data, scored at the end of every epoch. Its losses
            come under ``val_`` keys. Pass it whenever a held-out set exists:
            the two curves together tell learning from memorizing.
        epochs : int, default 10
            The number of passes through ``train_data``, at least 1.
        batch_size : int, default 1
            The samples of one update, at least 1.
        optimizer : str, dict, keras.optimizers.Optimizer or None, default None
            The optimizer:

            - a name, in any case: ``"adadelta"``, ``"adafactor"``,
              ``"adagrad"``, ``"adam"``, ``"adamax"``, ``"adamw"``,
              ``"ftrl"``, ``"lion"``, ``"nadam"``, ``"rmsprop"`` or
              ``"sgd"``. It is made with ``lr`` and ``optimizer_kwargs``;
            - ``None``, the same as ``"adam"``;
            - a serialized Keras optimizer, or an optimizer instance. It keeps
              its own learning rate.
        early_stopping : str, dict, keras.callbacks.Callback or None, default None
            Ends training before ``epochs`` once it stops improving:

            - the name of a loss training logs, such as ``"val_loss_fit"``,
              made into a
              ``keras.callbacks.EarlyStopping`` with ``early_stopping_kwargs``
              as its other arguments. It looks for a minimum;
            - a dict of the arguments of ``keras.callbacks.EarlyStopping``;
            - a callback: a configured ``keras.callbacks.EarlyStopping``, or
              one of your own that sets ``self.model.stop_training = True``;
            - ``None``, training runs every epoch.
        lr : float, default 1e-3
            The learning rate of a named optimizer, at least 0.
        shuffle : bool, default True
            ``True`` shuffles the samples at every epoch.
        optimizer_kwargs : dict or None, default None
            More arguments of a named optimizer, such as
            ``{"weight_decay": 1e-4, "global_clipnorm": 1.0}``. It cannot
            hold ``"learning_rate"``: that is ``lr``.
        early_stopping_kwargs : dict or None, default None
            More arguments of the ``EarlyStopping`` that a name given as
            ``early_stopping`` makes, such as
            ``{"patience": 10, "restore_best_weights": True}``.
        printer : str, keras.callbacks.Callback or None, default "legacy"
            How progress is shown:

            - ``"legacy"``: a table of the losses of every objective, a row
              per epoch, drawn as the tables of a model are;
            - ``"tiny"``: one compact block, redrawn at every epoch;
            - ``"nnodely"``: an animated console with a chart of the total
              loss;
            - ``None``: nothing;
            - a Keras callback, used as given, such as a
              :class:`LegacyPrinter`.
        jit_compile : str or bool, default "auto"
            Whether Keras compiles the training step: ``"auto"``, ``True`` or
            ``False``, as ``keras.Model.compile`` takes it.

        Returns
        -------
        ~nnodely.core.results.TrainResult
            Read by its attributes. ``losses`` holds an array, a value per
            epoch, under the keys Keras logs:

            - ``"loss"``: the total, every objective weighted by its gain;
            - ``"loss_<objective>"``: the loss of each objective, not
              weighted;
            - ``"val_loss"`` and ``"val_loss_<objective>"``: the same on
              ``val_data``, when it is given.

            Each value is the mean over the samples of the epoch. Early
            stopping gives fewer values than ``epochs``: ``epochs`` is the
            number run. The model does not keep the result.

        Raises
        ------
        ValueError
            If the model has no objective ("No minimizers defined. Call
            minimize() before train()"), or is not built ("Call build()
            before training"). If ``epochs`` or ``batch_size`` is not a
            positive integer, or ``lr`` is negative. If the data is empty, or
            lacks an Input the model reads ("which the data does not hold").
            If the optimizer name is unknown ("The supported optimizers
            are"), or ``optimizer_kwargs`` holds ``"learning_rate"``. If early
            stopping monitors a quantity training does not log ("which
            training does not log"). If ``printer`` is unknown.
        TypeError
            If ``train_data`` is not a :class:`DataLoader`, ``shuffle`` is not
            a bool, or ``early_stopping`` has another type.

        See Also
        --------
        Modely.minimize : Register an objective.
        Modely.validate : Score the trained model.
        Modely.freeze : Keep Parameters as they are through training.
        DataLoader : The data of a model.
        set_seed : Make the initial weights reproducible.

        Notes
        -----
        Training minimizes the total loss of the objectives, each loss
        :math:`L_i` weighted by its gain :math:`g_i`:

        .. math:: L = \sum_i g_i \, L_i

        Frozen Parameters keep their values. The objectives of the models it
        calls are not trained: each is minimized by the ``train()`` of its own
        model. The rollout steps padded onto shorter simulations add zero to
        every loss.

        Basic Examples
        --------------
        >>> import numpy as np
        >>> from nnodely import DataLoader, Input, Modely, Output, Parameter
        >>> x = Input("x", tw=1)
        >>> y = Output("y", x + Parameter("b", value=[1.0]))
        >>> y_meas = Input("y_meas", tw=1)
        >>> model = Modely("offset", inputs=[x], outputs=[y])
        >>> model = model.minimize("fit", y, y_meas).build()
        >>> samples = np.arange(8.0)
        >>> data = DataLoader(model, source={"x": samples, "y_meas": samples})
        >>> trained = model.train(data, epochs=2, batch_size=4, printer=None)
        >>> trained.losses.keys(), trained.epochs
        (['loss', 'loss_fit'], 2)
        >>> trained
        TrainResult(model='offset', epochs=2, samples=8, loss=...,
          minimizers: ['fit'],
          losses:     ['loss', 'loss_fit'],
        )

        Options
        -------
        ``val_data`` adds the validation losses:

        >>> trained = model.train(data, val_data=data, epochs=1, batch_size=4, printer=None)
        >>> trained.losses.keys()
        ['loss', 'loss_fit', 'val_loss', 'val_loss_fit']

        ``optimizer`` by name, with ``lr`` and ``optimizer_kwargs``. An
        instance, such as ``keras.optimizers.SGD(learning_rate=1e-2)``, is
        passed the same way:

        >>> trained = model.train(data, epochs=1, batch_size=4, optimizer="adamw", lr=1e-2,
        ...                       optimizer_kwargs={"weight_decay": 1e-4}, printer=None)

        ``early_stopping`` by the name of a logged loss:

        >>> trained = model.train(data, epochs=50, batch_size=4, printer=None,
        ...                       early_stopping="loss_fit",
        ...                       early_stopping_kwargs={"patience": 1, "min_delta": 1e9})
        >>> trained.epochs
        2

        ``shuffle=False`` keeps the order of the samples, and ``printer=None``
        prints nothing.

        Properties & Relations
        ----------------------
        With ``lr=0`` the weights do not change, and the total loss is the sum
        of the losses weighted by their gains, here :math:`1 \cdot 1 + 0.5
        \cdot 1`:

        >>> model = model.minimize("again", y, y_meas, loss="mae", gain=0.5)
        >>> model = model.build(keep_weights=False)
        >>> trained = model.train(data, epochs=1, batch_size=4, optimizer="sgd", lr=0.0,
        ...                       printer=None)
        >>> print(trained)
        ┏━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━┓
        ┃ idx ┃ minimizer ┃ loss                ┃ gain ┃ loss value ┃
        ┡━━━━━┻━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━┻━━━━━━┻━━━━━━━━━━━━┩
        │ offset: 1 epochs, 8 samples                               │
        ├─────┬───────────┬─────────────────────┬──────┬────────────┤
        │ 0   │ fit       │ mean_squared_error  │ 1.0  │ 1.0000e+00 │
        │ 1   │ again     │ mean_absolute_error │ 0.5  │ 1.0000e+00 │
        ├─────┼───────────┼─────────────────────┼──────┼────────────┤
        │     │ total     │                     │      │ 1.5000e+00 │
        └─────┴───────────┴─────────────────────┴──────┴────────────┘

        Possible Issues
        ---------------
        A validation loss is logged only with ``val_data``:

        >>> model.train(data, epochs=5, batch_size=4, early_stopping="val_loss", printer=None)
        Traceback (most recent call last):
            ...
        ValueError: Early stopping monitors 'val_loss', which training does not log: ...Pass val_data to monitor a validation loss.

        A model trains only on its objectives:

        >>> plain = Modely("plain", inputs=[x], outputs=[Output("z", 2.0 * x)]).build()
        >>> plain.train(data)
        Traceback (most recent call last):
            ...
        ValueError: No minimizers defined. Call minimize() before train().

        .. versionadded:: 2.0
        """
        if not self._minimizers:
            raise ValueError("No minimizers defined. Call minimize() before train().")
        for argument, value in (("epochs", epochs), ("batch_size", batch_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{argument} is a positive integer, got {value!r}.")
        if not isinstance(lr, (int, float)) or isinstance(lr, bool) or lr < 0:
            raise ValueError(f"lr is a non-negative number, got {lr!r}.")
        if not isinstance(shuffle, bool):
            raise TypeError(f"shuffle is True or False, got {shuffle!r}.")
        if not hasattr(train_data, "as_dict"):
            raise TypeError(
                f"train_data is a DataLoader, got {type(train_data).__name__}: "
                "DataLoader(model, source=...) lays the data out for training."
            )

        n_samples = len(train_data)
        if n_samples == 0:
            raise ValueError("train_data is empty.")

        # Ensure model is built
        if not self._keras_model:
            raise ValueError("Model is not built. Call build() before training.")
        self._check_names_built()

        resolved_optimizer = _resolve_optimizer(optimizer, lr, optimizer_kwargs)
        logged = [
            "loss",
            *(f"loss_{minimizer['name']}" for minimizer in self._minimizers),
        ]
        if val_data is not None:
            logged += [f"val_{name}" for name in logged]
        stopper = _resolve_early_stopping(early_stopping, early_stopping_kwargs, logged)
        self._keras_model.set_minimizers(_resolve_minimizer_terms(self))
        _check_minimizers_train_weights(self, self._keras_model)

        x_data, mask = self._training_arrays(train_data)

        validation_data = None
        val_mask = None
        if val_data is not None:
            if len(val_data) == 0:
                raise ValueError("val_data is empty.")
            val_x, val_mask = self._training_arrays(val_data)
            validation_data = val_x if val_mask is None else (val_x, val_mask)

        # Keras types jit_compile as a str, but takes True and False as well.
        self._keras_model.compile(
            optimizer=resolved_optimizer,
            jit_compile=jit_compile,  # pyright: ignore[reportArgumentType]
        )
        history = self._keras_model.fit(
            x=x_data,
            y=mask,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            verbose=0,  # type: ignore
            callbacks=[
                _resolve_printer(printer, epochs, self._minimizers, self.name),
                *([stopper] if stopper is not None else []),
            ],
            validation_data=validation_data,
        )

        return TrainResult(
            model=self.name,
            samples=n_samples,
            minimizers=minimizer_fields(self._minimizers),
            losses=Fields((key, np.asarray(history.history[key])) for key in logged),
        )

    # -------------------------------------------------------------------------
    # Flatten API
    # -------------------------------------------------------------------------

    def flatten(self) -> "Modely":
        """Return an equivalent model with every block inlined.

        The new model, named ``<name>_flat``, computes the same outputs from
        the same inputs. Every model this one calls as a block is replaced by
        its streams, at every depth. Its Loop bodies are not inlined: a
        :class:`Loop` and the rollout of :meth:`loop` keep their body.

        Returns
        -------
        Modely
            A new, unbuilt model named ``<name>_flat``.

        See Also
        --------
        Modely.models : The models a model calls.
        Modely.export_html : Draws the graph, flattened on a page of its own.

        Notes
        -----
        - The objectives are not carried over: the new model has none.
        - A declared Input the outputs do not read is dropped from its inputs.
        - The new model reads copies of the Inputs and of the Parameters, of
          the same names.
        - Flattened after :meth:`build`, it shares the weights of this model:
          training one changes the other.
        - Flattened before :meth:`build`, its layers draw weights of their
          own when it is built.

        Basic Examples
        --------------
        >>> import keras
        >>> import numpy as np
        >>> from nnodely import Input, Modely, Output, Parameter
        >>> x = Input("x", tw=1)
        >>> b = Parameter("b", value=[1.0])
        >>> offset = Modely("offset", inputs=[x], outputs=[Output("y", x + b)])
        >>> u = Input("u", tw=1)
        >>> outer = Modely("outer", inputs=[u], outputs=[Output("z", 2.0 * offset(u))])
        >>> flat = outer.flatten()
        >>> flat.name, len(outer.models), len(flat.models)
        ('outer_flat', 1, 0)

        Properties & Relations
        ----------------------
        Flattened after the build, it computes with the same weights:

        >>> outer = outer.build()
        >>> flat = outer.flatten().build()
        >>> sample = {"u": np.array([[[3.0]]])}
        >>> _ = outer.parameters["b"].param.assign([5.0])
        >>> keras.ops.convert_to_numpy(outer(sample)["z"]), keras.ops.convert_to_numpy(flat(sample)["z"])
        (array([[[16.]]], dtype=float32), array([[[16.]]], dtype=float32))

        Possible Issues
        ---------------
        The objectives stay with the model. Register them on the flattened
        model again:

        >>> outer = outer.minimize("fit", "z", 0.0)
        >>> len(outer.flatten().minimizers)
        0

        .. versionadded:: 2.0
        """
        return flatten(model=self)

    # -------------------------------------------------------------------------
    # Visualization API
    # -------------------------------------------------------------------------

    def export_html(
        self,
        out_dir: str | os.PathLike,
        filename: str | None = None,
        *,
        open_subgraph_in_new_tab: bool = False,
        physics: bool = True,
    ) -> None:
        """Write the graph of the model as interactive HTML pages, to open in a
        browser.

        Every stream is a card, colored by its role: Inputs, fed-back Inputs,
        outputs, Parameters, Constants, blocks and objectives. Clicking a card
        shows its shape and what it reads and feeds. A block, or a Loop body,
        has a page of its own, opened with a double click. The model need not
        be built.

        Parameters
        ----------
        out_dir : str or os.PathLike
            The folder the pages are written to, created when missing. A path
            ending in ``.html`` names the main page instead, in its folder.
        filename : str or None, default None
            The name of the main page. ``None`` names it after the model. The
            suffix ``.html`` is added when missing.
        open_subgraph_in_new_tab : bool, default False
            ``True`` opens the page of a block in a new browser tab.
        physics : bool, default True
            The pages open on a left-to-right layout, and switch to a force
            layout with a button. ``True`` lets the force layout place the
            cards by a simulation; ``False`` turns the simulation off.

        See Also
        --------
        Modely.flatten : The model with every block inlined.
        Modely.models : The models a model calls.

        Notes
        -----
        Next to the main page, ``<name>.html``, the folder holds:

        - ``<name>__flattened.html``, every block spliced in;
        - a page per call of a block, such as ``<name>__<block>.html``, and
          ``<name>__<block>_2.html`` for a second call;
        - the folder ``imgs``, of the images the pages show.

        Basic Examples
        --------------
        >>> import os
        >>> from nnodely import Input, Modely, Output
        >>> x = Input("x", tw=1)
        >>> offset = Modely("offset", inputs=[x], outputs=[Output("y", x + 1.0)])
        >>> u = Input("u", tw=1)
        >>> outer = Modely("outer", inputs=[u], outputs=[Output("z", 2.0 * offset(u))])
        >>> outer.export_html("graph")
        >>> sorted(os.listdir("graph"))
        ['imgs', 'outer.html', 'outer__flattened.html', 'outer__offset.html']

        Options
        -------
        ``filename`` names the main page:

        >>> outer.export_html("graph", "main")
        >>> os.path.exists("graph/main.html")
        True

        .. versionadded:: 2.0
        """
        from nnodely.utils.html_export import export_html

        export_html(
            model=self,
            out_dir=out_dir,
            filename=filename,
            open_subgraph_in_new_tab=open_subgraph_in_new_tab,
            physics=physics,
        )

    def summary(self):
        """Print the Keras summary of the built model: its layers, their output
        shapes and their numbers of weights.

        The summary is the one of the Keras model training runs, named
        ``<name>_train``. A model not built prints nothing.

        See Also
        --------
        Modely.parameters : The Parameters, as a table.
        Modely.build : Create the Keras model.

        Basic Examples
        --------------
        >>> from nnodely import Input, Modely, Output
        >>> x = Input("x", tw=2)
        >>> model = Modely("double", inputs=[x], outputs=[Output("y", 2.0 * x)]).build()
        >>> model.summary()
        Model: "double_train"
        ...

        .. versionadded:: 2.0
        """
        if self._keras_model is not None:
            self._keras_model.summary()

    # -------------------------------------------------------------------------
    # Save and load
    # -------------------------------------------------------------------------

    def save(self, path: str | os.PathLike, weights: bool | Literal["all"] = True):
        """Save the model, its graph and its weights, to a folder.

        :meth:`load` makes the model again from the folder. The objectives are
        not saved: register them again on the loaded model, then build it.

        Parameters
        ----------
        path : str or os.PathLike
            The folder, created when missing. A name such as
            ``"model.nnodely"`` is a folder too.
        weights : bool or "all", default True
            ``True`` saves the weights. ``False`` saves the graph alone, to
            be trained later or elsewhere. ``"all"`` also saves every block
            with weights of its own, so that each folder is a trained model.

        Raises
        ------
        ValueError
            If ``weights`` is not ``True``, ``False`` or ``"all"``. If Inputs
            of the model, or of a block that saves weights of its own, share
            a name: a model is built once loaded.

        See Also
        --------
        Modely.load : Load a saved model.
        Modely.export_keras : Save the model as a Keras file.
        Modely.export_onnx : Save the model as an ONNX file.

        Notes
        -----
        The folder holds:

        - ``model.json``, the graph;
        - ``model.weights.h5``, the weights, unless ``weights=False``. They
          include the weights of the blocks it calls;
        - a folder for every model it calls as a block or rolls out in a
          Loop, named after that model, laid out the same way. Each loads on
          its own. Its folder holds weights of its own only if the block was
          built on its own, as a Loop body always is, or with
          ``weights="all"``; otherwise it loads with initial weights.

        A model not built is built to save its weights. Without weights it is
        saved as it is.

        Basic Examples
        --------------
        >>> import os
        >>> from nnodely import Input, Modely, Output, Parameter
        >>> x = Input("x", tw=1)
        >>> y = Output("y", x + Parameter("b", value=[1.0]))
        >>> offset = Modely("offset", inputs=[x], outputs=[y])
        >>> u = Input("u", tw=1)
        >>> outer = Modely("outer", inputs=[u], outputs=[Output("z", 2.0 * offset(u))]).build()
        >>> outer.save("outer_model")
        >>> sorted(os.listdir("outer_model")), sorted(os.listdir("outer_model/offset"))
        (['model.json', 'model.weights.h5', 'offset'], ['model.json'])

        Options
        -------
        ``weights=False`` saves the graph alone:

        >>> outer.save("outer_graph", weights=False)
        >>> sorted(os.listdir("outer_graph"))
        ['model.json', 'offset']

        ``weights="all"`` saves the weights of every block too, so the block
        is a trained model of its own:

        >>> outer.save("outer_all", weights="all")
        >>> sorted(os.listdir("outer_all/offset"))
        ['model.json', 'model.weights.h5']

        .. versionadded:: 2.0
        """
        if weights not in (True, False, "all"):
            raise ValueError(f"weights is True, False or 'all', got {weights!r}.")
        # Loaded, a model with weights of its own is built: its Inputs need
        # names of their own, and its weights a Keras model to write them
        # from. It is built to write them, not to run: what that would warn
        # about is said once it is built to run.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for model in _nodes_and_models(self)[1]:
                # The model, and every block saving weights of its own.
                if model is self or weights == "all" or (weights and model.built):
                    model._check_inputs_apart(model._inputs)
                    if weights and not model.built:
                        model.build()
        ModelSerializer.serialize(self, path, weights=weights)

    @classmethod
    def load(cls, path: str | os.PathLike, weights: bool = True):
        """Load a model saved with :meth:`save`, and return it built.

        The models it calls are loaded from their folders, each once however
        many times it is called. So is every Parameter, Constant or Input they
        share. The loaded model has no objectives: register them again, then
        build it, before training.

        Parameters
        ----------
        path : str or os.PathLike
            The folder :meth:`save` wrote. The folder of a block inside it
            loads that block alone.
        weights : bool, default True
            ``True`` loads the saved weights. ``False``, or a folder saved
            without them, initializes the weights as :meth:`build` does: a
            Parameter takes its declared value, or a new draw.

        Returns
        -------
        Modely
            The model, built. A saved rollout of :meth:`loop` loads as a
            :class:`Modely` that rolls out the same step: it computes the
            same outputs.

        Raises
        ------
        FileNotFoundError
            If the folder holds no ``model.json`` ("Could not find nnodely
            model configuration").

        See Also
        --------
        Modely.save : Save a model.
        Modely.import_keras : Load a Keras file.

        Notes
        -----
        The loaded layers are layers of their own. A layer added to the
        loaded model, the model it was saved from, or another model loaded
        from the same folder never shares their weights. A model loaded and
        used as a block keeps its weights when the larger model is built.

        A custom Keras layer, such as one given to a :class:`KerasFunction`,
        loads only when its class is defined and registered with
        ``@keras.saving.register_keras_serializable()``.

        Basic Examples
        --------------
        >>> import keras
        >>> import numpy as np
        >>> from nnodely import Input, Modely, Output, Parameter
        >>> x = Input("x", tw=1)
        >>> b = Parameter("b", value=[1.0])
        >>> model = Modely("offset", inputs=[x], outputs=[Output("y", x + b)]).build()
        >>> _ = b.param.assign([5.0])
        >>> model.save("offset_model")
        >>> loaded = Modely.load("offset_model")
        >>> loaded.built
        True
        >>> keras.ops.convert_to_numpy(loaded({"x": np.array([[[1.0]]])})["y"])
        array([[[6.]]], dtype=float32)

        Options
        -------
        ``weights=False`` starts from the declared values:

        >>> fresh = Modely.load("offset_model", weights=False)
        >>> keras.ops.convert_to_numpy(fresh({"x": np.array([[[1.0]]])})["y"])
        array([[[2.]]], dtype=float32)

        Possible Issues
        ---------------
        The objectives are not saved:

        >>> model = model.minimize("fit", "y", 0.0)
        >>> model.save("offset_model")
        >>> len(Modely.load("offset_model").minimizers)
        0

        .. versionadded:: 2.0
        """
        return ModelSerializer.load(path, weights=weights)

    def export_keras(
        self, path: str | os.PathLike, filename: str | None = None
    ) -> None:
        """Write the built model as a ``.keras`` file, a stand-alone Keras
        model for inference.

        The file holds the model as declared: its outputs, and only the
        inputs they read. The objectives are left out. :meth:`import_keras`
        reads it back as a ``keras.Model``, on any Keras backend.

        Parameters
        ----------
        path : str or os.PathLike
            The folder the file is written to, created when missing.
        filename : str or None, default None
            The name of the file. ``None`` names it after the model. The
            suffix ``.keras`` is added when missing.

        Raises
        ------
        ValueError
            If the model is not built ("Call build() before export_keras()").

        See Also
        --------
        Modely.import_keras : Read the file back.
        Modely.save : Save the model to keep working on it with nnodely.
        Modely.export_onnx : Write the model as an ONNX file.

        Notes
        -----
        A weight read from several places, such as a body rolled out by two
        Loops, or a Parameter read inside a Loop body and outside it, is
        written as a copy for each place. The copies hold the same value, so
        the file gives the same outputs. Trained further as a Keras model,
        they would drift apart: to keep training, use :meth:`save` and
        :meth:`load`.

        Basic Examples
        --------------
        >>> import os
        >>> from nnodely import Input, Modely, Output
        >>> x = Input("x", tw=2)
        >>> model = Modely("double", inputs=[x], outputs=[Output("y", 2.0 * x)]).build()
        >>> model.export_keras("exports")
        >>> sorted(os.listdir("exports"))
        ['double.keras']

        Options
        -------
        ``filename`` names the file:

        >>> model.export_keras("exports", "copy")
        >>> sorted(os.listdir("exports"))
        ['copy.keras', 'double.keras']

        .. versionadded:: 2.0
        """
        export.export_keras(self, path, filename)

    @staticmethod
    def import_keras(filename: str, safe_mode: bool = True):
        """Read a ``.keras`` file written by :meth:`export_keras`, and return
        it as a ``keras.Model``.

        The Keras model takes a dict ``{input name: array}``, every array with
        its batch axis, and returns ``{output name: tensor}``, as the model
        did.

        Parameters
        ----------
        filename : str
            The path of the ``.keras`` file.
        safe_mode : bool, default True
            Keras' safe mode, which refuses to load code stored in the file,
            such as a Python lambda. Pass ``False`` only for a file you
            trust.

        Returns
        -------
        keras.Model
            The model, ready for inference on the active backend.

        See Also
        --------
        Modely.export_keras : Write the file.
        Modely.load : Load a model saved with :meth:`save`.

        Basic Examples
        --------------
        >>> import keras
        >>> import numpy as np
        >>> from nnodely import Input, Modely, Output
        >>> x = Input("x", tw=2)
        >>> model = Modely("double", inputs=[x], outputs=[Output("y", 2.0 * x)]).build()
        >>> model.export_keras("exports")
        >>> imported = Modely.import_keras("exports/double.keras")
        >>> result = imported({"x": np.array([[[1.0], [2.0]]], dtype="float32")})
        >>> keras.ops.convert_to_numpy(result["y"])
        array([[[2.],
                [4.]]], dtype=float32)

        .. versionadded:: 2.0
        """
        return export.import_keras(filename, safe_mode=safe_mode)

    def export_onnx(
        self,
        path: str | os.PathLike,
        filename: str | None = None,
        *,
        input_signature: list | None = None,
        batch_size: int | None = None,
        opset_version: int | None = None,
        verbose: bool = False,
    ) -> None:
        """Write the built model as an ``.onnx`` file, to run it with any ONNX
        runtime.

        The file holds the model as declared: its outputs, and only the
        inputs they read, under their names. The objectives are left out.
        The export needs the ``onnx`` extra, ``pip install "nnodely[onnx]"``,
        and the TensorFlow or PyTorch backend.

        Parameters
        ----------
        path : str or os.PathLike
            The folder the file is written to, created when missing.
        filename : str or None, default None
            The name of the file. ``None`` names it after the model. The
            suffix ``.onnx`` is added when missing.
        input_signature : list or None, default None
            The shapes the exported graph takes, as Keras'
            ``Model.export`` takes them, to fix axes left dynamic. ``None``
            keeps the shapes of the model, the batch axis dynamic.
        batch_size : int or None, default None
            Fixes the batch axis to that many samples. ``None`` leaves it
            dynamic. ``input_signature`` overrides it.
        opset_version : int or None, default None
            The ONNX opset. ``None`` takes the exporter's default.
        verbose : bool, default False
            ``True`` prints the messages of the exporter.

        Raises
        ------
        ValueError
            If the model is not built ("Call build() before export_onnx()").
        NotImplementedError
            On the JAX backend ("not available on the jax backend"). On the
            PyTorch backend, for a model with a :class:`Gradient` with respect
            to an Input.

        See Also
        --------
        Modely.validate_onnx : Run the exported file.
        Modely.export_keras : Write the model as a Keras file.
        Modely.save : Save the model to keep working on it with nnodely.

        Notes
        -----
        A :class:`Gradient` with respect to an Input records the backward
        pass in the exported graph. Its shape arithmetic converts only with a
        fixed batch axis: such a model is exported with a batch of 1, unless
        ``batch_size`` or ``input_signature`` says otherwise.

        Basic Examples
        --------------
        A built model named ``double``, on the TensorFlow or the PyTorch
        backend::

            model.export_onnx("exports")                 # exports/double.onnx
            model.export_onnx("exports", "fixed", batch_size=4)

        .. versionadded:: 2.0
        """
        export.export_onnx(
            self,
            path,
            filename,
            input_signature=input_signature,
            batch_size=batch_size,
            opset_version=opset_version,
            verbose=verbose,
        )

    @staticmethod
    def validate_onnx(
        filename: str | os.PathLike,
        inputs: dict,
        *,
        return_dict: bool = False,
        providers: list[str] | None = None,
    ):
        """Run an ``.onnx`` file written by :meth:`export_onnx` with ONNX
        Runtime, and return its outputs.

        Compare them with the outputs of the model to check the export.

        Parameters
        ----------
        filename : str or os.PathLike
            The path of the file. A path without the ``.onnx`` suffix gets
            it, in place of the suffix it has.
        inputs : dict
            ``{input name: array}``, every array with its batch axis. The
            arrays are cast to ``float32``. Keys that name no input of the
            file are ignored.
        return_dict : bool, default False
            ``False`` returns the outputs as a list of arrays, in the order
            of the file. ``True`` returns ``{output name: array}``.
        providers : list of str or None, default None
            The ONNX Runtime execution providers, such as
            ``["CPUExecutionProvider"]``. ``None`` takes the default ones.

        Returns
        -------
        list of numpy.ndarray or dict of str to numpy.ndarray
            The outputs, as ``return_dict`` selects.

        Raises
        ------
        ImportError
            If ``onnxruntime`` is not installed.
        FileNotFoundError
            If the file does not exist ("ONNX model not found").
        ValueError
            If an input of the file is missing from ``inputs`` ("Missing ONNX
            inputs").

        See Also
        --------
        Modely.export_onnx : Write the file.

        Basic Examples
        --------------
        With a model ``double`` that doubles its input ``x``, exported with
        the TensorFlow or the PyTorch backend::

            sample = {"x": np.array([[[1.0], [2.0]]])}
            Modely.validate_onnx("exports/double.onnx", sample)
            # [array([[[2.], [4.]]], dtype=float32)]
            Modely.validate_onnx("exports/double.onnx", sample, return_dict=True)
            # {'y': array([[[2.], [4.]]], dtype=float32)}

        .. versionadded:: 2.0
        """
        return export.validate_onnx(
            filename, inputs, return_dict=return_dict, providers=providers
        )


class ModelCall(Node):
    _inputs_map: dict[Node, Node]
    _outputs_map: dict[Node, Node]

    def __init__(self, name: str, model: Modely):
        super().__init__(name=name)
        self._model = model
        self._inputs_map = {}
        self._outputs_map = {}


class IntermediateOutput(Output):
    _pred: ModelCall

    def __init__(self, out: Stream, model_call: ModelCall) -> None:
        super().__init__(name=out.name, stream=out)
        self._pred = model_call
        self._preds = [model_call]


class Models(Table):
    """The models a model computes with, as :attr:`Modely.models` lists them:
    a list of the Modely objects, printed as a table of their inputs, their
    outputs, the number of Parameters each computes with and whether it is
    built."""

    _kind, _attr = "model", "models"

    def __init__(self, models, title: str = "", where: list[str] | None = None):
        super().__init__(models)
        self._title = title
        # Where each one is, for the listing of a name several carry.
        self._places = where or ["" for _ in models]

    def _where(self, index, item) -> str:
        return self._places[index]

    def __str__(self):
        header = ("model", "inputs", "outputs", "parameters", "built")
        rows = [
            (
                model.name,
                ", ".join(node.name for node in model._inputs) or "-",
                ", ".join(node.name for node in model._outputs) or "-",
                str(len(_parameters(model))),
                "yes" if model.built else "no",
            )
            for model in self
        ]
        return _table(header, [("", rows)], indexed=True)


class Minimizer(Mapping):
    """One objective of a model, as :attr:`~nnodely.Modely.minimizers` lists
    it.

    You do not create one yourself: :meth:`~nnodely.Modely.minimize`
    registers the objective, and ``model.minimizers["fit"]`` or
    ``model.minimizers[0]`` picks its row.

    Each is read as an attribute or as a key, ``m.gain`` or ``m["gain"]``.
    Setting ``name``, ``loss`` or ``gain`` changes the objective in its
    model, checked as :meth:`~nnodely.Modely.minimize` checks it, from the
    next :meth:`~nnodely.Modely.train`. Setting anything else raises
    ``TypeError``: register the objective again to change what it compares.
    :meth:`remove` removes it from its model.

    Attributes
    ----------
    name : str
        The name of the objective, the key of its loss, ``loss_<name>``.
    source : Stream
        The prediction it compares.
    target : Stream
        The reference it compares the prediction with. A number given as the
        target is a :class:`~nnodely.Constant`.
    loss : keras.losses.Loss or callable
        The loss function.
    gain : float
        The weight of its loss in the total loss.
    model : Modely
        The model it belongs to.

    Basic Examples
    --------------
    >>> from nnodely import Input, Modely, Output
    >>> x = Input("x", tw=1)
    >>> model = Modely("double", inputs=[x], outputs=[Output("y", 2.0 * x)])
    >>> model = model.minimize("fit", "y", Input("y_meas", tw=1))
    >>> fit = model.minimizers["fit"]
    >>> fit.name, fit.source.name, fit.target.name, fit.gain
    ('fit', 'y', 'y_meas', 1.0)
    >>> fit.gain = 2.0
    >>> model.minimizers[0]["gain"]
    2.0

    Possible Issues
    ---------------
    What it compares is set by :meth:`~nnodely.Modely.minimize`:

    >>> fit.source = x
    Traceback (most recent call last):
        ...
    TypeError: Minimizer 'fit': its name, loss and gain can be set, not its 'source'; ...

    .. versionadded:: 2.0
    """

    _SETTABLE = ("name", "loss", "gain")
    # No attribute of its own to set: one set by mistake would be lost.
    __slots__ = ("_minimizer", "_model")

    def __init__(self, minimizer: dict, model: "Modely"):
        # The model's own entry, not a copy: what is set is set there.
        object.__setattr__(self, "_minimizer", minimizer)
        object.__setattr__(self, "_model", model)

    def __getattr__(self, key):
        if key in ("name", "source", "target", "loss", "gain"):
            return self._minimizer[key]
        if key == "model":
            return self._model
        raise AttributeError(f"Minimizer has no attribute {key!r}.")

    def __setattr__(self, key, value):
        self[key] = value

    def __getitem__(self, key):
        return self._model if key == "model" else self._minimizer[key]

    def __iter__(self):
        return iter([*self._minimizer, "model"])

    def __len__(self):
        return len(self._minimizer) + 1

    def __repr__(self):
        return object_repr(
            self,
            name=self.name,
            source=self.source,
            target=target_label(self.target),
            loss=loss_name(self.loss),
            gain=self.gain,
            model=self._model,
        )

    def remove(self) -> None:
        """Remove this objective from its model.

        Pick it first, by its name or by its index in
        :attr:`~nnodely.Modely.minimizers`. The next
        :meth:`~nnodely.Modely.train` and :meth:`~nnodely.Modely.validate`
        leave it out. The model needs no new build. The row stays readable,
        but no longer belongs to the model: setting it raises.

        Raises
        ------
        KeyError
            When it is picked, if no objective has that name. Nothing is
            removed.
        IndexError
            When it is picked, if the index is out of range.

        See Also
        --------
        nnodely.Modely.minimize : Register an objective.
        nnodely.Modely.minimizers : The objectives, as a table.

        Basic Examples
        --------------
        >>> from nnodely import Input, Modely, Output
        >>> x = Input("x", tw=1)
        >>> model = Modely("double", inputs=[x], outputs=[Output("y", 2.0 * x)])
        >>> model = model.minimize("a", "y", 1.0).minimize("b", "y", 2.0)
        >>> model.minimizers["a"].remove()
        >>> [m.name for m in model.minimizers]
        ['b']

        Options
        -------
        By index:

        >>> model.minimizers[0].remove()
        >>> len(model.minimizers)
        0

        Possible Issues
        ---------------
        A name no objective has raises, as in every table:

        >>> model.minimizers["c"].remove()
        Traceback (most recent call last):
            ...
        nnodely.core.dag.NameNotFound: ...no minimizer named 'c'...

        A row removed is no longer the model's:

        >>> fit = model.minimize("fit", "y", 1.0).minimizers["fit"]
        >>> fit.remove()
        >>> fit.gain = 2.0
        Traceback (most recent call last):
            ...
        ValueError: Minimizer 'fit' was removed from model 'double': declare it again with minimize().

        .. versionadded:: 2.0
        """
        model = self._model
        model._minimizers = [m for m in model._minimizers if m is not self._minimizer]

    def __setitem__(self, key, value):
        if key not in self._SETTABLE:
            raise TypeError(
                f"Minimizer {self._minimizer['name']!r}: its name, loss and gain "
                f"can be set, not its {key!r}; declare it again with minimize() "
                "to change what it compares."
            )
        self._model._set_minimizer(self._minimizer, key, value)


class Minimizers(Table):
    """The minimizers of a model, as :attr:`Modely.minimizers` lists them: a
    list of :class:`Minimizer`, printed as a table of the streams each
    compares, its loss and its gain."""

    _HEADER = ("minimizer", "source", "target", "loss", "gain")

    _kind, _attr = "minimizer", "minimizers"

    def __init__(self, minimizers: list[Minimizer]):
        super().__init__(minimizers)

    def __str__(self):
        rows = [
            (
                m.name,
                m.source.name,
                str(target_label(m.target)),
                loss_name(m.loss),
                str(m.gain),
            )
            for m in self
        ]
        return _table(self._HEADER, [("", rows)], indexed=True)


def _refresh_keras_inputs(model: Modely) -> None:
    """Give every Input the model reads a Keras input of its name now: an
    Input renamed since it was declared, as by ``x.name = "speed"``, keeps
    the Keras input of its first name, which the data would be keyed by."""
    for node in _nodes_and_models(model)[0]:
        if isinstance(node, Input) and node._input.name != node.name:
            node._input = keras.Input(shape=node.shape.tuple, name=node.name)


def _rename_hidden_states(model: Modely) -> None:
    """Give a free name to every state a layer (Integrate, Derivative,
    Buffer) keeps in an Input of its own that has the name of another Input
    the model reads: named after the layer and its signal, ``tank_rate``, it can
    be named like an Input of the user's, ``tank_rate`` too. The user never
    names or reads it, so it yields. The data is fed by input name, so the
    model could not be built otherwise.

    A hidden state the user named, ``x.feedback(stream, hidden=True)``, keeps
    its name, as any Input of the user's: one named like another Input the
    model reads is refused."""
    inputs = [node for node in _nodes_and_models(model)[0] if isinstance(node, Input)]
    taken: dict[str, Input] = {}
    for node in inputs:
        if node._hidden and not node._hidden_by_user:
            continue
        other = taken.setdefault(node.name, node)
        if other is not node and (node._hidden_by_user or other._hidden_by_user):
            raise ValueError(
                f"Model {model.name!r} reads two Inputs named {node.name!r}, "
                "and one is a hidden state, fed back with hidden=True. A hidden "
                "state keeps the name it was given, and the Inputs of a model "
                "are told apart by name: give the state a name of its own."
            )
    for node in inputs:
        if (
            node._hidden
            and not node._hidden_by_user
            and taken.setdefault(node.name, node) is not node
        ):
            node.name = free_name(node.name, taken)
            node._input = keras.Input(shape=node.shape.tuple, name=node.name)
            taken[node.name] = node


def _same_sizes(sizes: tuple, other: tuple) -> bool:
    """Whether two tuples of axis sizes match: as many axes, of the same
    lengths, a dynamic one (None) matching any."""
    return len(sizes) == len(other) and all(
        None in (size, other_size) or size == other_size
        for size, other_size in zip(sizes, other)
    )
