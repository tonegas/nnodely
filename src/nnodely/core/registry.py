import json
import warnings
from pathlib import Path
from nnodely.core.stream import Node, class_path, registered_class


def _saved_order(model) -> list:
    """The nodes of ``model``'s own graph to save, every one after its
    predecessors: those its outputs are computed from, its declared inputs -
    read or not, they are what it is called with - and those its fed-back
    Inputs are fed back with, and start from, which are no predecessors of
    theirs. A model it calls is a node of it, saved in a folder of its own."""
    from nnodely.core.dag import toposort
    from nnodely.layers.input import Input

    roots = [*model._outputs, *model._inputs]
    while True:
        order = toposort(roots)
        reached = set(order)
        extra = [
            stream
            for node in order
            if isinstance(node, Input) and node.update is not None
            for stream in (node.update, node.init)
            if isinstance(stream, Node) and stream not in reached
        ]
        if not extra:
            return order
        roots += extra


class _Saving:
    """What one save shares between the models of the saved tree: an id for
    every layer object, every Input and every model, the same in every file
    that writes it, so that what the models share is shared again once
    loaded."""

    def __init__(self) -> None:
        self.layers: dict[int, int] = {}
        self.inputs: dict[int, int] = {}
        self.models: dict[int, int] = {}

    @staticmethod
    def id_of(table: dict[int, int], obj) -> int:
        return table.setdefault(id(obj), len(table))


class _Loading:
    """What one load shares between the models of the saved tree.

    Every layer was saved under an id, the same for every node applying it
    in any model of the tree. Those nodes are given back one source, so they
    share their weights again, and the Keras layer the first model built for
    it - a called model, or a body, is built before the model holding it - is
    handed on to the models built after it. A model and an Input written in
    several files of the tree are loaded once, and are one object again.
    """

    def __init__(self, weights: bool):
        self.weights = weights
        self.sources: dict[int, object] = {}
        self.layers: dict[int, object] = {}
        self.ids: dict[int, int] = {}  # id(node) -> the id its layer was saved under
        # The Parameters and Constants loaded, by the id they were saved under.
        self.values: dict[int, object] = {}
        self.inputs: dict[int, object] = {}
        self.models: dict[int, object] = {}

    def model(self, node_data: dict, path: Path):
        """The model a node calls, or rolls out, loaded from its folder the
        first time the tree names it."""
        model_id = node_data["model_id"]
        if model_id not in self.models:
            self.models[model_id] = ModelSerializer.load(
                path / node_data["model"], loading=self
            )
        return self.models[model_id]

    def restore(self, node, layer_id: int | None) -> None:
        if layer_id is None:
            return
        self.ids[id(node)] = layer_id
        node._source = self.sources.setdefault(layer_id, node._source)
        if layer_id in self.layers:
            node._layer = self.layers[layer_id]

    def keep_built(self, model) -> None:
        for node in model._order:
            layer_id = self.ids.get(id(node))
            layer = getattr(node, "_layer", None)
            if layer_id is not None and getattr(layer, "weights", None):
                self.layers.setdefault(layer_id, layer)


class ModelSerializer:
    FORMAT = "nnodely"
    VERSION = 14

    @staticmethod
    def serialize(model, path, *, weights=True, saving=None):
        """Write ``model`` to the folder ``path``: its own graph in
        ``model.json``, its weights in ``model.weights.h5``, and every model
        it calls as a block, or rolls out in a Loop, in a folder of its own,
        written the same way - so the models nest as their folders do, and
        each folder is a model that loads on its own."""
        from nnodely.core.layer import Layer
        from nnodely.core.modely import (
            IntermediateOutput,
            ModelCall,
            Modely,
            _build_apart,
        )
        from nnodely.layers.input import Input
        from nnodely.layers.parameter import Constant

        saving = _Saving() if saving is None else saving
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        order = _saved_order(model)
        node_ids = {node: f"node_{i}" for i, node in enumerate(order)}

        if weights and model.inference_model is not None:
            saved = {id(variable) for variable in model.inference_model.weights}
            # A called model, or a body, saves its weights in its own folder,
            # and a Constant is its value, saved in its config.
            unsaved = sorted(
                {
                    node.name
                    for node in order
                    if isinstance(node, Layer)
                    and not isinstance(node, Constant)
                    and node._model is None
                    and any(
                        id(variable) not in saved
                        for variable in getattr(node._layer, "weights", None) or []
                    )
                }
            )
            if unsaved:
                warnings.warn(
                    f"Model {model.name!r} is saved without the weights of "
                    f"{unsaved}: only the streams its fed-back Inputs are fed "
                    "back with read them, and the model does not compute those, "
                    "so they load back as declared. Save the Loop or ModelyLoop "
                    "that computes them, or declare those streams among the "
                    "outputs.",
                    UserWarning,
                    stacklevel=3,
                )

        nodes = []
        # The folder of every model this one holds, by model: one called
        # several times is written once.
        folders: dict[int, str] = {}

        for node in order:
            node_data = {
                "id": node_ids[node],
                "class_name": class_path(type(node)),
                "config": node.get_config(),
                "preds": [node_ids[pred] for pred in node._preds],
            }
            if isinstance(node, Input):
                # An Input a called model reads without declaring it is the
                # one this model reads too: written in both files, it is
                # loaded as one.
                node_data["input"] = saving.id_of(saving.inputs, node)
                # A fed-back Input keeps the stream it is fed back with and
                # its init, a number or a node: a loop that calls the loaded
                # model feeds it back as it would the saved one.
                if node.update is not None:
                    node_data["feedback"] = {
                        "update": node_ids[node.update],
                        "init": (
                            node_ids[node.init]
                            if isinstance(node.init, Node)
                            else node.init
                        ),
                        "hidden": node._hidden,
                    }
            if isinstance(node, Layer):
                node_data["layer"] = saving.id_of(saving.layers, node._source)
            if isinstance(node, IntermediateOutput):
                call = node._pred
                inner = call._outputs_map[node]
                node_data["output"] = next(
                    index
                    for index, output in enumerate(call._model._outputs)
                    if output is inner
                )
            # A model this one calls as a block, or a Loop rolls out, is saved
            # as a model of its own, in a folder named after it - the model
            # called, or the Loop - so nested models nest their folders the
            # same way. Two can share a name, so a folder already taken gets
            # a suffix.
            held = node._model
            if isinstance(held, Modely):
                folder = folders.get(id(held))
                if folder is None:
                    base = held.name if isinstance(node, ModelCall) else node.name
                    folder, index = base, 1
                    while folder in folders.values():
                        folder, index = f"{base}_{index}", index + 1
                    folders[id(held)] = folder
                    if weights and not held.built:
                        # Built for the weights it computes with here - the
                        # ones this model shares with it - not to run on its
                        # own: what that would warn about does not apply.
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", UserWarning)
                            _build_apart(held)
                    ModelSerializer.serialize(
                        held, path / folder, weights=weights, saving=saving
                    )
                node_data["model"] = folder
                node_data["model_id"] = saving.id_of(saving.models, held)
            nodes.append(node_data)

        data = {
            "format": ModelSerializer.FORMAT,
            "version": ModelSerializer.VERSION,
            "model": {
                "name": model.name,
                # By position among its inputs.
                "optional_inputs": [
                    index
                    for index, node in enumerate(model._inputs)
                    if node in model._optional_inputs
                ],
            },
            "nodes": nodes,
            "inputs": [node_ids[x] for x in model._inputs],
            "outputs": [node_ids[x] for x in model._outputs],
        }

        with open(path / "model.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        # The saved model is the declared one: minimizers, and whatever only
        # they read, belong to a training session, not to the model. Saved
        # without weights, the model is its architecture only: a weights file
        # left by an earlier save would otherwise be loaded with it.
        weights_path = path / "model.weights.h5"
        if weights and model.inference_model is not None:
            model.inference_model.save_weights(weights_path)
        else:
            weights_path.unlink(missing_ok=True)

    @staticmethod
    def deserialize(data, path, *, loading):
        from nnodely.core.modely import IntermediateOutput, ModelCall, Modely
        from nnodely.layers.parameter import _Value

        if data["model"].get("roll") is not None:
            raise ValueError(
                f"The model saved in {path} has a rollback, which this nnodely "
                "does not have: declare it again, rolled out with a Loop."
            )

        node_map = {}
        # The streams each call returned, by the id of the call.
        calls: dict[str, list] = {}

        for node_data in data["nodes"]:
            cls = registered_class(node_data["class_name"])
            if cls is None:
                raise ValueError(
                    f"The saved model holds a {node_data['class_name']!r} node, "
                    "which this nnodely does not know: it was saved by a newer "
                    "version, or by a layer class defined in code not imported."
                )

            layer_id = node_data.get("layer")
            input_id = node_data.get("input")
            # A Parameter or Constant read by several models of the tree is
            # one object, written in the file of each: loaded once, it is the
            # same object in all of them. So is an Input a called model reads
            # without declaring it.
            if issubclass(cls, _Value) and layer_id in loading.values:
                node_map[node_data["id"]] = loading.values[layer_id]
                continue
            if input_id is not None and input_id in loading.inputs:
                node_map[node_data["id"]] = loading.inputs[input_id]
                continue

            preds = [node_map[pred_id] for pred_id in node_data["preds"]]
            if cls is ModelCall:
                # The called model is loaded, built and given its weights from
                # its folder, then called on the streams it was called on.
                returned = loading.model(node_data, path)(preds)
                calls[node_data["id"]] = (
                    returned if isinstance(returned, list) else [returned]
                )
                node_map[node_data["id"]] = calls[node_data["id"]][0]._pred
                continue
            if cls is IntermediateOutput:
                node_map[node_data["id"]] = calls[node_data["preds"][0]][
                    node_data["output"]
                ]
                continue
            config = node_data["config"]
            if "model" in node_data:
                # The body is loaded, built and given its weights first, then
                # handed to the block as the `f` it was built with.
                config = {**config, "f": loading.model(node_data, path)}
            node = cls.from_config(
                config,
                preds=preds,
            )
            loading.restore(node, layer_id)
            if issubclass(cls, _Value) and layer_id is not None:
                loading.values[layer_id] = node
            if input_id is not None:
                loading.inputs[input_id] = node
            node_map[node_data["id"]] = node

        # Fed back once the streams they are fed back with, which read them,
        # exist.
        for node_data in data["nodes"]:
            feedback = node_data.get("feedback")
            state = node_map[node_data["id"]]
            # An Input loaded in another file of the tree already is.
            if feedback is not None and state.update is None:
                init = feedback["init"]
                state._feed_back(
                    node_map[feedback["update"]],
                    init=node_map[init] if isinstance(init, str) else init,
                )
                state._hidden = feedback["hidden"]

        inputs = [node_map[node_id] for node_id in data["inputs"]]
        outputs = [node_map[node_id] for node_id in data["outputs"]]
        model = Modely(
            name=data["model"]["name"],
            inputs=inputs,
            outputs=outputs,
        )
        model._optional_inputs = {
            inputs[index] for index in data["model"].get("optional_inputs", ())
        }
        return model

    @staticmethod
    def load(path, *, weights=True, loading=None):
        from nnodely.core.modely import _build_apart

        path = Path(path)

        config_path = path / "model.json"
        weights_path = path / "model.weights.h5"

        if not config_path.exists():
            raise FileNotFoundError(
                f"Could not find nnodely model configuration: {config_path}"
            )

        loading = _Loading(weights) if loading is None else loading
        with open(config_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as error:
                raise ValueError(f"{config_path} is not a valid model file: {error}.")
        if not isinstance(data, dict) or data.get("format") != ModelSerializer.FORMAT:
            raise ValueError(
                f"{config_path} is not a model saved by nnodely 2: models saved "
                "with nnodely 1.x cannot be loaded."
            )
        if data.get("version") != ModelSerializer.VERSION:
            raise ValueError(
                f"{config_path} was saved in format version {data.get('version')}, "
                f"and this nnodely reads version {ModelSerializer.VERSION} only: "
                "declare the model again with this nnodely and save it."
            )

        model = ModelSerializer.deserialize(data, path, loading=loading)
        # Saved with Inputs of one name, it loads with them: two Inputs, built
        # apart, which a rename lets run.
        _build_apart(model)

        # A model loaded without its weights - saved without them, or loaded
        # with weights=False - keeps the ones build() just gave it, as a model
        # declared from scratch would.
        if loading.weights and weights_path.exists():
            assert model.inference_model is not None
            model.inference_model.load_weights(weights_path)

        loading.keep_built(model)
        return model
