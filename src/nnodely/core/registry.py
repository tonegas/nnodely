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
    def serialize(model, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        flat = model.flatten()

        node_ids = {node: f"node_{i}" for i, node in enumerate(flat.order)}

        nodes = []

        for node in flat.order:
            nodes.append(
                {
                    "id": node_ids[node],
                    "class_name": node.__class__.__name__,
                    "config": node.get_config(),
                    "preds": [node_ids[pred] for pred in node.preds],
                }
            )

        data = {
            "format": ModelSerializer.FORMAT,
            "version": ModelSerializer.VERSION,
            "model": {
                "name": model.name,
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
    def deserialize(data):
        from nnodely.core.modely import Modely

        node_map = {}

        for node_data in data["nodes"]:
            cls = NODE_REGISTRY[node_data["class_name"]]

            preds = [node_map[pred_id] for pred_id in node_data["preds"]]
            node = cls.from_config(
                node_data["config"],
                preds=preds,
            )
            # node.preds = preds
            node_map[node_data["id"]] = node

        inputs = [node_map[node_id] for node_id in data["inputs"]]
        outputs = [node_map[node_id] for node_id in data["outputs"]]
        return Modely(
            name=data["model"]["name"],
            inputs=inputs,
            outputs=outputs,
        )

    @staticmethod
    def load(path):
        path = Path(path)

        config_path = path / "model.json"
        weights_path = path / "model.weights.h5"

        if not config_path.exists():
            raise FileNotFoundError(
                f"Could not find nnodely model configuration: {config_path}"
            )

        with open(config_path, "r") as f:
            data = json.load(f)

        model = ModelSerializer.deserialize(data)
        model.build()

        if weights_path.exists():
            if model.model is not None:
                model.model.load_weights(weights_path)
            else:
                print(f"the model {model.name} has no keras model to load weights.")
        else:
            print(f"the weights path: {weights_path} does not exist.")

        return model
