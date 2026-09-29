"""
DAG - Lazy/DAG approach. No global graph.
Each node has predecessors; Model traverses from output backwards.

Dimensioni: dim, time, seq. Shape = dim, time, seq.
Default: dim=1
"""

from copy import copy

import keras

from nnodely.core.stream import NODE_NAMES, Node, Stream, reserve_name
from nnodely.layers.output import Output
from typing import Any

_node_counter = 0

#: The names next_name() made. Unlike a given name, a load may replace one.
GENERATED_NAMES: set[str] = set()


def next_name(prefix: str) -> str:
    """A name for a node left unnamed, used by no node of this process.

    The counter alone is not enough: a given name, or one read back from a
    saved model, can already have the form it would produce.
    """
    global _node_counter
    while True:
        _node_counter += 1
        name = f"{prefix}{_node_counter}"
        if name not in NODE_NAMES:
            break
    # Reserved here as well, because a LocalModel or an EquationLearner takes
    # its name from here without being a node, and derives its layers' names.
    reserve_name(name)
    GENERATED_NAMES.add(name)
    return name


def is_generated(name: str) -> bool:
    """True for a name next_name() made, or one derived from it - a LocalModel
    cell is ``"<LocalModel name>_out0"``."""
    parts = name.split("_")
    return any(
        "_".join(parts[:index]) in GENERATED_NAMES for index in range(1, len(parts) + 1)
    )


# ------------------------------------------------------------------
# flattening logic
# ------------------------------------------------------------------


class Scope:
    """What one graph has already inlined, plus a scope per call it makes.

    A model called several times has to be inlined once per call, so the nodes
    inside a body belong to the call rather than to the graph: sharing a single
    memo made the second call return the first call's subgraph. The scopes of
    nested calls hang off the enclosing scope, because a call written inside a
    body exists once in its definition but is evaluated once per enclosing call.
    """

    def __init__(self, memo: dict[Node, Any] | None = None) -> None:
        self.memo: dict[Node, Any] = {} if memo is None else memo
        self.calls: dict[Node, "Scope"] = {}

    def merged_memo(self) -> dict[Node, Any]:
        """Every node inlined anywhere, with the enclosing scope taking priority.

        Callers use this to reconnect the public symbolic nodes to the concrete
        layers built from their copies, and a node inside a body is public too:
        a block that wraps a model still exposes the layers it holds. A body
        inlined once per call has one copy per call here, but they share a name
        and therefore a single concrete layer, so either copy answers for it.
        """
        merged: dict[Node, Any] = {}
        for call_scope in self.calls.values():
            merged.update(call_scope.merged_memo())
        merged.update(self.memo)
        return merged


def flatten_node(node: Node, scope: Scope) -> Any:
    from nnodely.core.modely import IntermediateOutput
    from nnodely.core.layer import Layer

    if node in scope.memo:
        return scope.memo[node]

    if type(node) is IntermediateOutput:
        model_call = node.pred

        # One scope per call, not per output: a body with several outputs is
        # still a single call, and its outputs must share the inlined body.
        call_scope = scope.calls.get(model_call)
        if call_scope is None:
            # The arguments belong to the calling graph, so they are inlined in
            # the enclosing scope and stay shared between calls; only the
            # bindings they produce are local to this one.
            bindings = {
                internal_input: flatten_node(external_input, scope)
                for internal_input, external_input in model_call.inputs_map.items()
            }
            call_scope = Scope({**scope.memo, **bindings})
            scope.calls[model_call] = call_scope

        internal_output = model_call.outputs_map[node]
        flat_output = flatten_node(internal_output, call_scope)

        scope.memo[node] = flat_output
        return flat_output

    new_preds = [flatten_node(pred, scope) for pred in node.preds]
    new_node = copy(node)
    new_node.preds = new_preds
    if isinstance(new_node, Layer) and not bool(
        getattr(new_node._layer, "weights", None)
    ):
        new_node._layer = None
    scope.memo[node] = new_node
    return new_node


def flatten(model):
    return _flatten_graph(model.name, model.inputs, model.outputs)


def _flatten_graph(
    name,
    inputs: list[Any],
    outputs: list[Stream],
    *,
    return_memo: bool = False,
) -> Any:
    from nnodely.core.modely import Modely

    scope = Scope()
    flat_outputs: list[Output] = [flatten_node(output, scope) for output in outputs]
    memo = scope.merged_memo()
    flat_inputs: list[Any] = [memo[input] for input in inputs if input in memo]

    flat_model = Modely(f"{name}_flat", flat_inputs, flat_outputs)
    if return_memo:
        return flat_model, memo
    return flat_model


# ------------------------------------------------------------------
# Keras graph introspection
# ------------------------------------------------------------------
def _source_input_tensors(tensor):
    """The model-input tensors ``tensor`` is computed from.

    The Keras graph is walked rather than the nnodely DAG because the two do
    not always agree: a Constant or Parameter is wired to an arbitrary anchor
    input to give it a batch context, so the Keras sub-graph of a relation can
    need a tensor its symbolic predecessors never mention. Sources are ordered
    by name so a reloaded model resolves them exactly like the saved one did.
    """
    sources: dict[str, object] = {}
    seen: set[int] = set()
    stack = [tensor]
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        history = getattr(current, "_keras_history", None)
        if history is None:
            continue
        operation = history[0]
        if isinstance(operation, keras.layers.InputLayer):
            sources[operation.name] = current
            continue
        node = operation._inbound_nodes[history[1]]
        stack.extend(node.input_tensors)
    return [sources[name] for name in sorted(sources)]


def _source_names(tensors):
    return [tensor._keras_history[0].name for tensor in tensors]


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


def _streams_from(roots) -> list[Stream]:
    """Every Stream ``roots`` are computed from, the roots themselves included.

    Iterative, so a graph of any depth is walked; one stream reached along
    several paths is listed once.
    """
    streams: dict[int, Stream] = {}
    seen: set[int] = set()
    stack = list(roots)
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, Stream):
            streams[id(node)] = node
        stack.extend(node.preds)
    return list(streams.values())
