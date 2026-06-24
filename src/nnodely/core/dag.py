"""
DAG - Lazy/DAG approach. No global graph.
Each node has predecessors; Model traverses from output backwards.

Dimensioni: seq, time, dim. Shape = seq + (time,) + dim.
Default: seq=(), time=1, dim=(1,)
"""

from copy import copy
from nnodely.core.stream import Node
from nnodely.layers.output import Output
from typing import Any

_node_counter = 0


def next_name(prefix: str) -> str:
    """Nome univoco per nodi generati."""
    global _node_counter
    _node_counter += 1
    return f"{prefix}{_node_counter}"


# ------------------------------------------------------------------
# flattening logic
# ------------------------------------------------------------------


def get_preds(node: Node) -> list[Node]:
    from nnodely.core.modely import IntermediateOutput

    if type(node) is IntermediateOutput:
        return [node]

    return node.preds


def flatten_node(node: Node, memo: dict[Node, Node]) -> Any:
    from nnodely.core.modely import IntermediateOutput

    if node in memo:
        return memo[node]

    if type(node) is IntermediateOutput:
        inputs_map = node.pred.inputs_map
        outputs_map = node.pred.outputs_map
        for old, new in inputs_map.items():
            if new not in memo:
                new_node = copy(new)
                new_node.preds = [flatten_node(pred, memo) for pred in get_preds(new)]
                memo[new] = new_node
            memo[old] = memo[new]

        new_preds = [flatten_node(pred, memo) for pred in get_preds(outputs_map[node])]
    else:
        new_preds = [flatten_node(pred, memo) for pred in node.preds]

    # new_node = copy(node)
    # new_node.preds = new_preds
    # memo[node] = new_node
    # return new_node
    node.preds = new_preds
    memo[node] = node
    return node


def flatten(model):
    return _flatten_graph(model.name, model.inputs, model.outputs)


def _flatten_graph(name, inputs: list[Any], outputs: list[Output]) -> Any:
    from nnodely.core.modely import Modely

    memo: dict[Node, Any] = {}
    flat_outputs: list[Output] = [flatten_node(output, memo) for output in outputs]
    flat_inputs: list[Any] = [memo[input] for input in inputs if input in memo]

    return Modely(f"{name}_flat", flat_inputs, flat_outputs)


# ------------------------------------------------------------------
# DAG topological ordering
# ------------------------------------------------------------------
def toposort(model) -> list[Node]:
    return toposort_outputs(model.outputs)


def toposort_outputs(outputs: list[Output]) -> list[Node]:
    order = []
    visited = set()

    def dfs(node: Node) -> None:
        visited.add(node)

        for pred in node.preds:
            if pred not in visited:
                dfs(pred)
        order.append(node)

    for output in outputs:
        dfs(output)

    return order
