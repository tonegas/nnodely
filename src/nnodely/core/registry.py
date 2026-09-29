# NODE_REGISTRY = {}


# def register_node(cls):
#     NODE_REGISTRY[cls.__name__] = cls
#     return cls
import json
import uuid
from pathlib import Path
from nnodely.core.dag import GENERATED_NAMES, is_generated, next_name
from nnodely.core.stream import NODE_NAMES, NODE_REGISTRY, reserve_name

#: Written into every save, to tell the files of this process from the others.
_SESSION = uuid.uuid4().hex


def _saved_names(path: Path) -> list[str]:
    """The node names of a saved model and of every model saved inside it."""
    with open(path / "model.json", "r") as f:
        nodes = json.load(f)["nodes"]
    names = [node["config"]["name"] for node in nodes]
    for node in nodes:
        if "model" in node:
            names += _saved_names(path / node["model"])
    return names


class _LoadedNames:
    """The names a load gives the nodes it reads back.

    Saved names are kept, and a user's always. A generated name, though, is
    unique only within the process that made it: a Fir created here before
    the load and one saved from another process can both be "Fir1". One name
    builds one layer, so composing the two would silently train a single one.
    Such a name is replaced by a fresh one, the same for every node saved
    under it - the applications of one shared layer stay shared.

    A file saved by this process keeps every name: its generated names are
    unique here by construction, so they can only be taken by the layers it
    was saved from, and a model reloaded next to its original stays the same.
    """

    def __init__(self, path: Path):
        with open(path / "model.json", "r") as f:
            foreign = json.load(f).get("session") != _SESSION
        self.taken = set(NODE_NAMES) if foreign else set()
        self.renamed: dict[str, str] = {}
        # The whole saved tree is reserved first, so a fresh name never lands
        # on one that a node read later still has to take.
        for name in _saved_names(path):
            reserve_name(name)

    def name(self, saved: str, class_name: str, generated: bool) -> str:
        if saved not in self.renamed:
            fresh = generated and saved in self.taken
            self.renamed[saved] = next_name(class_name) if fresh else saved
            if generated:
                GENERATED_NAMES.add(self.renamed[saved])
        return self.renamed[saved]

    def saved(self, name: str) -> str:
        """The name a node was saved under, which keys the layers it shares."""
        return next((saved for saved, new in self.renamed.items() if new == name), name)


class ModelSerializer:
    FORMAT = "nnodely"
    VERSION = 1

    @staticmethod
    def serialize(model, path, *, weights=True, layers=None, folder=""):
        from nnodely.core.layer import Layer
        from nnodely.core.modely import Modely

        # Every weighted layer saved so far by this save, across all the
        # models it writes, mapped by identity to the key it is reloaded under:
        # the folder of its model and its name.
        layers = {} if layers is None else layers
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        flat = model.flatten()

        node_ids = {node: f"node_{i}" for i, node in enumerate(flat.order)}

        nodes = []

        for node in flat.order:
            node_data = {
                "id": node_ids[node],
                "class_name": node.__class__.__name__,
                "config": node.get_config(),
                "preds": [node_ids[pred] for pred in node.preds],
            }
            if isinstance(node, Layer) and is_generated(node.name):
                node_data["generated"] = True
            # A block that wraps a model (Loop, Roll, OdeNet) saves that model
            # as a model of its own, in a folder named after the block, so
            # nested blocks nest their folders the same way.
            body = getattr(node, "f", None)
            if isinstance(body, Modely):
                node_data["model"] = node.name
                ModelSerializer.serialize(
                    body,
                    path / node.name,
                    weights=weights,
                    layers=layers,
                    folder=f"{folder}{node.name}/",
                )
            nodes.append(node_data)

        # One layer can belong to several models of the tree - a Parameter
        # used by two bodies, or one body rolled out by two Loops. Keras saves
        # its weights once, so it has to be shared again when loading. A
        # wrapped model is built before the model that holds it, both here and
        # when loading, so the first model to claim a layer is its owner.
        for node, node_data in zip(flat.order, nodes):
            layer = getattr(node, "_layer", None)
            if not getattr(layer, "weights", None):
                continue
            key = f"{folder}{node.name}"
            owner = layers.setdefault(id(layer), key)
            if owner != key:
                node_data["shared"] = owner

        data = {
            "format": ModelSerializer.FORMAT,
            "version": ModelSerializer.VERSION,
            "session": _SESSION,
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
            },
            "nodes": nodes,
            "inputs": [node_ids[x] for x in flat.inputs],
            "outputs": [node_ids[x] for x in flat.outputs],
        }

        with open(path / "model.json", "w") as f:
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
    def deserialize(data, path, *, layers=None, folder="", names):
        from nnodely.core.modely import Modely

        layers = {} if layers is None else layers

        node_map = {}

        for node_data in data["nodes"]:
            cls = NODE_REGISTRY[node_data["class_name"]]

            preds = [node_map[pred_id] for pred_id in node_data["preds"]]
            config = node_data["config"]
            name = names.name(
                config["name"],
                node_data["class_name"],
                node_data.get("generated", False),
            )
            if name != config["name"]:
                config = {**config, "name": name}
            if "model" in node_data:
                # The wrapped model is loaded, built and given its weights
                # first, then handed to the block as the `f` it was built with.
                config = {
                    **config,
                    "f": ModelSerializer.load(
                        path / node_data["model"],
                        layers=layers,
                        folder=f"{folder}{node_data['model']}/",
                        names=names,
                    ),
                }
            node = cls.from_config(
                config,
                preds=preds,
            )
            if node_data.get("shared") in layers:
                # Reuse the layer its owner, loaded earlier, has already built.
                node._layer = layers[node_data["shared"]]  # type: ignore
            # node.preds = preds
            node_map[node_data["id"]] = node

        inputs = [node_map[node_id] for node_id in data["inputs"]]
        outputs = [node_map[node_id] for node_id in data["outputs"]]
        model = Modely(
            name=data["model"]["name"],
            inputs=inputs,
            outputs=outputs,
        )
        roll = data["model"].get("roll")
        if roll is not None:
            model.rollback(
                {
                    names.renamed.get(input_name, input_name): names.renamed.get(
                        stream_name, stream_name
                    )
                    for input_name, stream_name in roll["callbacks"].items()
                },
                steps=roll["steps"],
                name=roll.get("name"),
            )
        return model

    @staticmethod
    def load(path, *, layers=None, folder="", names=None):
        # The weighted layers built so far by this load, by the key they were
        # saved under, for the models loaded after them to share.
        layers = {} if layers is None else layers
        path = Path(path)

        config_path = path / "model.json"
        weights_path = path / "model.weights.h5"

        if not config_path.exists():
            raise FileNotFoundError(
                f"Could not find nnodely model configuration: {config_path}"
            )

        names = _LoadedNames(path) if names is None else names
        with open(config_path, "r") as f:
            data = json.load(f)

        model = ModelSerializer.deserialize(
            data, path, layers=layers, folder=folder, names=names
        )
        model.build()

        # A model saved without its weights keeps the ones build() just gave
        # it, as a model declared from scratch would.
        if weights_path.exists():
            assert model.inference_model is not None
            model.inference_model.load_weights(weights_path)

        for node in model.order:
            layer = getattr(node, "_layer", None)
            if getattr(layer, "weights", None):
                layers.setdefault(f"{folder}{names.saved(node.name)}", layer)
        return model
