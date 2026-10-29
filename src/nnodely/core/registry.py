import json
import warnings
from pathlib import Path
from nnodely.core.stream import NODE_REGISTRY, Node


def _bound_closed(model) -> list[str]:
    """The closed Inputs ``model``, or a model it calls, gives a stream for
    when it calls a model that declares them: flattened, the call reads that
    stream in their place, and their feedback is gone."""
    from nnodely.core.modely import ModelCall
    from nnodely.layers.input import Input

    bound: set[str] = set()
    seen: set[int] = set()
    stack = list(model.outputs)
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, ModelCall):
            bound.update(
                inner.name
                for inner in node.inputs_map
                if isinstance(inner, Input) and inner.update is not None
            )
            stack.extend(node.model.outputs)
        if isinstance(node, Input) and node.update is not None:
            stack.extend(s for s in (node.update, node.init) if isinstance(s, Node))
        stack.extend(node.preds)
    return sorted(bound)


def _saved_order(flat) -> list:
    """The nodes of ``flat`` to save, every one after its predecessors: those
    its outputs are computed from, and those its closed Inputs are closed
    with, and start from, which are no predecessors of theirs."""
    from nnodely.core.dag import toposort_outputs
    from nnodely.layers.input import Input

    roots = list(flat.outputs)
    while True:
        order = toposort_outputs(roots)
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


class _Loading:
    """What one load shares between the models of the saved tree.

    Every layer was saved under an id, the same for every node applying it
    in any model of the tree. Those nodes are given back one source, so they
    share their weights again, and the Keras layer the first model built for
    it - a body is built before the model holding it - is handed on to the
    models built after it.
    """

    def __init__(self, weights: bool):
        self.weights = weights
        self.sources: dict[int, object] = {}
        self.layers: dict[int, object] = {}
        self.ids: dict[int, int] = {}  # id(node) -> the id its layer was saved under
        # The Parameters and Constants loaded, by the id they were saved under.
        self.values: dict[int, object] = {}

    def restore(self, node, layer_id: int | None) -> None:
        if layer_id is None:
            return
        self.ids[id(node)] = layer_id
        node._source = self.sources.setdefault(layer_id, node._source)
        if layer_id in self.layers:
            node._layer = self.layers[layer_id]

    def keep_built(self, model) -> None:
        for node in model.order:
            layer_id = self.ids.get(id(node))
            layer = getattr(node, "_layer", None)
            if layer_id is not None and getattr(layer, "weights", None):
                self.layers.setdefault(layer_id, layer)


class ModelSerializer:
    FORMAT = "nnodely"
    VERSION = 7

    @staticmethod
    def serialize(model, path, *, weights=True, layer_ids=None):
        from nnodely.core.layer import Layer
        from nnodely.core.modely import Modely
        from nnodely.layers.input import Input
        from nnodely.layers.parameter import Constant

        # One id per layer object, shared by every model the save writes: a
        # Parameter used by two bodies, or one body rolled out by two Loops,
        # is one layer when loaded back.
        layer_ids = {} if layer_ids is None else layer_ids
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        flat = model.flatten()
        order = _saved_order(flat)
        node_ids = {node: f"node_{i}" for i, node in enumerate(order)}

        # What a ModelyLoop calling the loaded model would feed back other
        # than it feeds back the saved one.
        bound = _bound_closed(model)
        if bound:
            warnings.warn(
                f"Model {model.name!r} calls models that declare the closed "
                f"Inputs {bound}, giving a stream for each: saved, it reads that "
                "stream in their place, as a Modely does, and keeps nothing of "
                "their feedback, so a ModelyLoop that calls the loaded model "
                "does not feed them back. Save the ModelyLoop that closes them "
                "instead.",
                UserWarning,
                stacklevel=3,
            )
        if weights and model.inference_model is not None:
            saved = {id(variable) for variable in model.inference_model.weights}
            # A wrapped body saves its weights with its own model, and a
            # Constant is its value, saved in its config.
            unsaved = sorted(
                {
                    node.name
                    for node in order
                    if isinstance(node, Layer)
                    and not isinstance(node, Constant)
                    and not isinstance(getattr(node, "f", None), Modely)
                    and any(
                        id(variable) not in saved
                        for variable in getattr(node._layer, "weights", None) or []
                    )
                }
            )
            if unsaved:
                warnings.warn(
                    f"Model {model.name!r} is saved without the weights of "
                    f"{unsaved}: only the streams its closed Inputs are closed "
                    "with read them, and the model does not compute those, so "
                    "they load back as declared. Save the ModelyLoop that "
                    "computes them, or declare those streams among the outputs.",
                    UserWarning,
                    stacklevel=3,
                )

        nodes = []
        folders: set[str] = set()

        for node in order:
            node_data = {
                "id": node_ids[node],
                "class_name": node.__class__.__name__,
                "config": node.get_config(),
                "preds": [node_ids[pred] for pred in node.preds],
            }
            # A closed Input keeps the stream it is closed with and its init,
            # a number or a node: a ModelyLoop that calls the loaded model
            # feeds it back as it would the saved one.
            if isinstance(node, Input) and node.update is not None:
                node_data["closed"] = {
                    "update": node_ids[node.update],
                    "init": (
                        node_ids[node.init]
                        if isinstance(node.init, Node)
                        else node.init
                    ),
                    "hidden": node._hidden,
                }
            if isinstance(node, Layer):
                node_data["layer"] = layer_ids.setdefault(
                    id(node._source), len(layer_ids)
                )
            # A block that wraps a model (Loop, Roll, OdeNet) saves that model
            # as a model of its own, in a folder named after the block, so
            # nested blocks nest their folders the same way. Two blocks can
            # share a name, so a folder already taken gets a suffix.
            body = getattr(node, "f", None)
            if isinstance(body, Modely):
                folder, index = node.name, 1
                while folder in folders:
                    folder, index = f"{node.name}_{index}", index + 1
                folders.add(folder)
                node_data["model"] = folder
                ModelSerializer.serialize(
                    body, path / folder, weights=weights, layer_ids=layer_ids
                )
            nodes.append(node_data)

        data = {
            "format": ModelSerializer.FORMAT,
            "version": ModelSerializer.VERSION,
            "model": {
                "name": model.name,
                "roll": (
                    {
                        "callbacks": {
                            input_node.name: stream.name
                            for input_node, stream in model._roll_callbacks.items()
                        },
                        "steps": model._roll_steps,
                        "name": model._roll_name,
                    }
                    if model._roll_callbacks
                    else None
                ),
                "optional_inputs": sorted(model.optional_inputs),
            },
            "nodes": nodes,
            "inputs": [node_ids[x] for x in flat.inputs],
            "outputs": [node_ids[x] for x in flat.outputs],
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
        from nnodely.core.modely import Modely
        from nnodely.layers.parameter import _Value

        node_map = {}

        for node_data in data["nodes"]:
            cls = NODE_REGISTRY.get(node_data["class_name"])
            if cls is None:
                raise ValueError(
                    f"The saved model holds a {node_data['class_name']!r} node, "
                    "which this nnodely does not know: it was saved by a newer "
                    "version, or by a layer class defined in code not imported."
                )

            layer_id = node_data.get("layer")
            # A Parameter or Constant read by several models of the tree is
            # one object, written in the file of each: loaded once, it is the
            # same object in all of them.
            if issubclass(cls, _Value) and layer_id in loading.values:
                node_map[node_data["id"]] = loading.values[layer_id]
                continue

            preds = [node_map[pred_id] for pred_id in node_data["preds"]]
            config = node_data["config"]
            if "model" in node_data:
                # The wrapped model is loaded, built and given its weights
                # first, then handed to the block as the `f` it was built with.
                config = {
                    **config,
                    "f": ModelSerializer.load(
                        path / node_data["model"], loading=loading
                    ),
                }
            node = cls.from_config(
                config,
                preds=preds,
            )
            loading.restore(node, layer_id)
            if issubclass(cls, _Value) and layer_id is not None:
                loading.values[layer_id] = node
            node_map[node_data["id"]] = node

        # Closed once the streams they are closed with, which read them, exist.
        for node_data in data["nodes"]:
            closed = node_data.get("closed")
            if closed is not None:
                init = closed["init"]
                node_map[node_data["id"]].close(
                    node_map[closed["update"]],
                    init=node_map[init] if isinstance(init, str) else init,
                )
                node_map[node_data["id"]]._hidden = closed.get("hidden", False)

        inputs = [node_map[node_id] for node_id in data["inputs"]]
        outputs = [node_map[node_id] for node_id in data["outputs"]]
        model = Modely(
            name=data["model"]["name"],
            inputs=inputs,
            outputs=outputs,
        )
        model.optional_inputs = set(data["model"].get("optional_inputs", ()))
        roll = data["model"].get("roll")
        if roll is not None:
            model.rollback(
                roll["callbacks"],
                steps=roll["steps"],
                name=roll.get("name"),
            )
        return model

    @staticmethod
    def load(path, *, weights=True, loading=None):
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
        model.build()

        # A model loaded without its weights - saved without them, or loaded
        # with weights=False - keeps the ones build() just gave it, as a model
        # declared from scratch would.
        if loading.weights and weights_path.exists():
            assert model.inference_model is not None
            model.inference_model.load_weights(weights_path)

        loading.keep_built(model)
        return model
