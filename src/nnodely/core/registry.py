# NODE_REGISTRY = {}


# def register_node(cls):
#     NODE_REGISTRY[cls.__name__] = cls
#     return cls
import json
from pathlib import Path
from nnodely.core.stream import NODE_REGISTRY


class ModelSerializer:
    FORMAT = "nnodely"
    VERSION = 1

    @staticmethod
    def serialize(model, path, *, layers=None, folder=""):
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
            # A block that wraps a model (Loop, Roll, OdeNet) saves that model
            # as a model of its own, in a folder named after the block, so
            # nested blocks nest their folders the same way.
            body = getattr(node, "f", None)
            if isinstance(body, Modely):
                node_data["model"] = node.name
                ModelSerializer.serialize(
                    body,
                    path / node.name,
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

        if model.model is not None:
            model.model.save_weights(path / "model.weights.h5")

    @staticmethod
    def deserialize(data, path, *, layers=None, folder=""):
        from nnodely.core.modely import Modely

        layers = {} if layers is None else layers

        node_map = {}

        for node_data in data["nodes"]:
            cls = NODE_REGISTRY[node_data["class_name"]]

            preds = [node_map[pred_id] for pred_id in node_data["preds"]]
            config = node_data["config"]
            if "model" in node_data:
                # The wrapped model is loaded, built and given its weights
                # first, then handed to the block as the `f` it was built with.
                config = {
                    **config,
                    "f": ModelSerializer.load(
                        path / node_data["model"],
                        layers=layers,
                        folder=f"{folder}{node_data['model']}/",
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
                roll["callbacks"],
                steps=roll["steps"],
                name=roll.get("name"),
            )
        return model

    @staticmethod
    def load(path, *, layers=None, folder=""):
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

        with open(config_path, "r") as f:
            data = json.load(f)

        model = ModelSerializer.deserialize(data, path, layers=layers, folder=folder)
        model.build()

        if weights_path.exists():
            if model.model is not None:
                model.model.load_weights(weights_path)
            else:
                print(f"the model {model.name} has no keras model to load weights.")
        else:
            print(f"the weights path: {weights_path} does not exist.")

        for node in model.order:
            layer = getattr(node, "_layer", None)
            if getattr(layer, "weights", None):
                layers.setdefault(f"{folder}{node.name}", layer)
        return model
