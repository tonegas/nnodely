import json
from pathlib import Path
from nnodely.core.stream import NODE_REGISTRY


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
    VERSION = 2

    @staticmethod
    def serialize(model, path, *, weights=True, layer_ids=None):
        from nnodely.core.layer import Layer
        from nnodely.core.modely import Modely

        # One id per layer object, shared by every model the save writes: a
        # Parameter used by two bodies, or one body rolled out by two Loops,
        # is one layer when loaded back.
        layer_ids = {} if layer_ids is None else layer_ids
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        flat = model.flatten()

        node_ids = {node: f"node_{i}" for i, node in enumerate(flat.order)}

        nodes = []
        folders: set[str] = set()

        for node in flat.order:
            node_data = {
                "id": node_ids[node],
                "class_name": node.__class__.__name__,
                "config": node.get_config(),
                "preds": [node_ids[pred] for pred in node.preds],
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
                        path / node_data["model"], loading=loading
                    ),
                }
            node = cls.from_config(
                config,
                preds=preds,
            )
            loading.restore(node, node_data.get("layer"))
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
            data = json.load(f)

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
