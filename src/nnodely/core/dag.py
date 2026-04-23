"""
DAG
"""

import copy
from typing import Sequence, cast

from nnodely.core.modely import ModelCall, Modely
from nnodely.core.stream import Stream
from nnodely.layers.output import Output

_node_counter: int = 0


def next_name(prefix: str) -> str:
    """Generate a unique name for nodes

    :param prefix: Prefix to add to the unique name.
    :return: The unique name.
    """

    global _node_counter
    _node_counter += 1
    return f"{prefix}{_node_counter}"


def flatten(model: Modely) -> Modely:
    """Flatten a model

    :param model: The model to flatten.
    :return: The flattened model.
    """

    flattened = model
    for node in model._order:
        if type(node) is ModelCall:
            flattened = _aggregate_models(model=flattened, submodel=node)
    if any(type(n) is ModelCall for n in flattened._order):
        return flatten(flattened)
    return flattened


def _clone_graph(order: Sequence[Stream]) -> tuple[list[Stream], dict[Stream, Stream]]:
    clone_map = {}
    cloned_order = []
    for node in order:
        new_node = copy.copy(node)
        new_node.predecessors = [
            clone_map[p] for p in node.predecessors if p in clone_map
        ]
        clone_map[node] = new_node
        cloned_order.append(new_node)
    return cloned_order, clone_map


def _aggregate_models(model: Modely, submodel: ModelCall) -> Modely:
    model_order, model_map = _clone_graph(model._order)
    submodel_cloned = model_map[submodel]
    submodel_order, _ = _clone_graph(submodel.model._order)

    cloned_sub_outputs = {n.name: n for n in submodel_order if n.node_type == "Output"}
    # Replace submodel inputs with external mapped streams
    input_map_cloned = {}
    for in_name, ext_stream in submodel.input_map.items():
        if ext_stream in model_map:
            input_map_cloned[in_name] = model_map[ext_stream]

    # remove child input/output nodes from inserted subgraph
    submodel_order = [
        n for n in submodel_order if n.node_type not in ("Input", "Output")
    ]

    # reconnect predecessors inside child graph
    for node in submodel_order:
        new_preds = []
        for pred in node.predecessors:
            if pred.node_type == "Input" and pred.name in input_map_cloned:
                new_preds.append(input_map_cloned[pred.name])
            else:
                new_preds.append(pred)
        node.predecessors = new_preds

    # Replace parent references to the ModelCall with the selected child output predecessor
    selected_output = cloned_sub_outputs[submodel.output_name]
    replacement_preds = selected_output.predecessors[:]  # usually one predecessor

    model_order.remove(submodel_cloned)  # remove the ModelCall node

    for node in model_order:
        new_preds = []
        for pred in node.predecessors:
            if pred is submodel_cloned:
                new_preds.extend(replacement_preds)
            else:
                new_preds.append(pred)
        node.predecessors = new_preds

    # Rebuild outputs from cloned parent outputs
    cloned_outputs = [
        cast(Output, model_map[out]) for out in model.outputs
    ]  # HACK: requires further attention

    flat_model = Modely(
        name=model.name + "_" + submodel.model.name,
        outputs=cloned_outputs,
    )
    # keep minimizers
    flat_model._minimizers = copy.copy(model._minimizers)

    return flat_model


def toposort(output_nodes: list[Output] | Output) -> list[Stream]:
    """Topologically sort a graph given its output nodes

    DFS post-order from outputs to inputs

    :param output_nodes: The output nodes of the graph.
    :return: The topological sort of the graph.
    """

    if not isinstance(output_nodes, (list, tuple)):
        output_nodes = [output_nodes]
    result = []
    visited = set()

    def dfs(node: Stream):
        visited.add(id(node))
        for pred in node.predecessors:
            if id(pred) not in visited:
                dfs(pred)
        result.append(node)

    for out in output_nodes:
        dfs(out)
    return result
