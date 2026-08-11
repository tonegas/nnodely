NODE_REGISTRY = {}


def register_node(cls):
    NODE_REGISTRY[cls.__name__] = cls
    return cls


class ModelSerializer:
    FORMAT = "nnodely"
    VERSION = 1

    @staticmethod
    def serialize(model):
        flat = model.flatten()
        nodes = []

        for node in flat.order:
            nodes.append(
                {
                    "id": node.name,
                    "class_name": node.__class__.__name__,
                    "config": node.get_config(),
                    "preds": [pred.name for pred in node.preds],
                }
            )

        return {
            "format": ModelSerializer.FORMAT,
            "version": ModelSerializer.VERSION,
            "model": {
                "name": model.name,
            },
            "nodes": nodes,
            "inputs": [x.name for x in model.inputs],
            "outputs": [x.name for x in model.outputs],
        }

    @staticmethod
    def deserialize(data):
        from nnodely.core.modely import Modely

        node_map = {}

        for node_data in data["nodes"]:
            cls = NODE_REGISTRY[node_data["class_name"]]

            preds = [node_map[pred_id] for pred_id in node_data["preds"]]
            # time = 1
            # if node_data["class_name"] == "Input":
            #     # remove time from node_data["config"] if it exists
            #     time = node_data["config"]["time"]
            #     del node_data["config"]["time"]
            node = cls.from_config(
                node_data["config"],
                preds=preds,
            )
            # if node_data["class_name"] == "Input":
            #     node.shape.time = time
            node.preds = preds
            node_map[node_data["id"]] = node

        inputs = [node_map[node_id] for node_id in data["inputs"]]
        outputs = [node_map[node_id] for node_id in data["outputs"]]
        return Modely(
            name=data["model"]["name"],
            inputs=inputs,
            outputs=outputs,
        )
