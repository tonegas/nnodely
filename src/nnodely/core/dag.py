"""
DAG - Lazy/DAG approach. No global graph.
Each node has predecessors; Model traverses from output backwards.

Axes: a stream is laid out (batch, *seq, tw, *dim), or (batch, *seq, *dim)
without a window, see core/shape.py. Default: dim=1, no window, no sequence.
"""

from copy import copy

import keras

from nnodely.core.names import by_name_or_index, pick  # noqa: F401 - re-exported
from nnodely.core.stream import NODE_NAMES, Node, Stream, reserve_name
from nnodely.layers.output import Output
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nnodely.core.modely import Modely
    from nnodely.layers.parameter import Parameter

_node_counter = 0


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
    return name


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

    def __init__(
        self,
        memo: dict[Node, Any] | None = None,
        values: dict[Node, Any] | None = None,
        pending: list | None = None,
        loop: bool = False,
        declared: set | None = None,
    ) -> None:
        self.memo: dict[Node, Any] = {} if memo is None else memo
        self.calls: dict[Node, "Scope"] = {}
        # The Parameters and Constants inlined, one copy each, shared with
        # every call: they are one object in every model that reads them. So
        # is an Input read from data - one array, whatever reads it - unless a
        # call binds it; a fed-back Input is one state per call in a loop.
        self.values: dict[Node, Any] = {} if values is None else values
        # The fed-back Inputs inlined, with the scope of each, when the
        # streams they are fed back with are inlined too (see
        # flatten_fed_back); None leaves them as they are.
        self.pending = pending
        # Whether the graph is the step of a loop: a call then keeps the
        # feedback of the fed-back Inputs its model declares, where a Modely
        # reads the argument given for one as its data.
        self.loop = loop
        # The streams a call gives the fed-back Inputs its model declares, when
        # their feedback is inlined: their initial values.
        self.inits: dict[Node, Any] = {}
        # The Inputs a Modely being built declares, or None outside one: a
        # hidden fed-back Input it does not declare is nothing it feeds back,
        # and reads its init - zero without one - as the first step of a
        # loop would.
        self.declared = declared

    def merged_memo(self) -> dict[Node, Any]:
        """Every node inlined anywhere, with the enclosing scope taking priority.

        Callers use this to reconnect the public symbolic nodes to the concrete
        layers built from their copies, and a node inside a body is public too:
        a block that wraps a model still exposes the layers it holds. A body
        inlined once per call has one copy per call here, but they apply one
        layer, so they hold one concrete layer and either copy answers for it.
        """
        merged: dict[Node, Any] = {}
        for call_scope in self.calls.values():
            merged.update(call_scope.merged_memo())
        merged.update(self.memo)
        return merged


def flatten_node(node: Node, scope: Scope) -> Any:
    from nnodely.core.modely import IntermediateOutput
    from nnodely.core.layer import Layer
    from nnodely.layers.parameter import _Value
    from nnodely.layers.input import Input

    if node in scope.memo:
        return scope.memo[node]
    if (
        scope.declared is not None
        and isinstance(node, Input)
        and node._hidden
        and node not in scope.declared
    ):
        init = Stream._coerce_operand(0.0 if node.init is None else node.init)
        scope.memo[node] = flatten_node(init, scope)
        return scope.memo[node]
    shared = isinstance(node, _Value) or (
        isinstance(node, Input) and (node.update is None or not scope.loop)
    )
    if shared and node in scope.values:
        scope.memo[node] = scope.values[node]
        return scope.values[node]

    if type(node) is IntermediateOutput:
        model_call = node._pred

        # One scope per call, not per output: a body with several outputs is
        # still a single call, and its outputs must share the inlined body.
        call_scope = scope.calls.get(model_call)
        if call_scope is None:
            # The arguments belong to the calling graph, so they are inlined in
            # the enclosing scope and stay shared between calls; only the
            # bindings they produce are local to this one.
            arguments = {
                internal_input: flatten_node(external_input, scope)
                for internal_input, external_input in model_call._inputs_map.items()
            }
            # A fed-back Input the model declares reads its argument as data:
            # at every step in a Modely; in a loop it keeps its feedback, and
            # the argument is its initial value.
            inits: dict[Node, Any] = {
                internal_input: argument
                for internal_input, argument in arguments.items()
                if scope.loop
                and isinstance(internal_input, Input)
                and internal_input.update is not None
            }
            bindings = {
                internal_input: argument
                for internal_input, argument in arguments.items()
                if internal_input not in inits
            }
            # A call starts from its bindings alone: what the caller has
            # inlined so far depends on the order the graph is walked in, and
            # sharing it made the result depend on that order too.
            call_scope = Scope(
                bindings, scope.values, scope.pending, scope.loop, scope.declared
            )
            call_scope.inits = inits
            scope.calls[model_call] = call_scope

        internal_output = model_call._outputs_map[node]
        flat_output = flatten_node(internal_output, call_scope)

        scope.memo[node] = flat_output
        return flat_output

    new_preds = [flatten_node(pred, scope) for pred in node._preds]
    new_node = copy(node)
    new_node._preds = new_preds
    if isinstance(new_node, Layer) and not bool(
        getattr(new_node._layer, "weights", None)
    ):
        new_node._layer = None
    scope.memo[node] = new_node
    if shared:
        scope.values[node] = new_node
    if (
        scope.pending is not None
        and isinstance(node, Input)
        and node.update is not None
    ):
        scope.pending.append((new_node, node, scope))
    return new_node


def flatten_fed_back(scope: Scope) -> None:
    """Inline the stream every fed-back Input inlined in ``scope`` is fed
    back with, and its init, each in the call the Input was inlined in, and so
    on for the fed-back Inputs those read in turn.

    They are no predecessors of the Input, and wait for the graph to be
    inlined: an update can read the output of a call whose arguments read the
    Input, and inlined on reaching the Input it would make that call twice.
    """
    while scope.pending:
        inlined, source, source_scope = scope.pending.pop(0)
        inlined.update = flatten_node(source.update, source_scope)
        if source in source_scope.inits:
            # A state starts from one value.
            if source.init is not None:
                raise ValueError(
                    f"{source.name!r} has two initial values: the init of the "
                    "block that keeps it, and the stream a call gives it as an "
                    "input of its model. Give one of them."
                )
            inlined.init = source_scope.inits[source]
        elif isinstance(source.init, Node):
            inlined.init = flatten_node(source.init, source_scope)


def flatten(model):
    return _flatten_graph(model.name, model._inputs, model._outputs, fed_back=True)


def _flatten_graph(
    name,
    inputs: list[Any],
    outputs: list[Output],
    *,
    return_memo: bool = False,
    fed_back: bool = False,
    one_step: bool = False,
) -> Any:
    """``fed_back`` inlines the stream every fed-back Input is fed back with,
    and its init, too: without it they are left as they are, in the graph of
    the model that declared them. ``one_step`` reads the init of every hidden
    fed-back Input not among ``inputs`` in its place, as a Modely is built."""
    from nnodely.core.modely import Modely

    scope = Scope(
        pending=[] if fed_back else None, declared=set(inputs) if one_step else None
    )
    flat_outputs: list[Output] = [flatten_node(output, scope) for output in outputs]
    flatten_fed_back(scope)
    memo = scope.merged_memo()
    flat_inputs: list[Any] = [memo[input] for input in inputs if input in memo]

    flat_model = Modely(f"{name}_flat", flat_inputs, flat_outputs)
    if return_memo:
        return flat_model, memo
    return flat_model


# ------------------------------------------------------------------
# Keras graph introspection
# ------------------------------------------------------------------
def _source_input_tensors(tensor) -> list[Any]:
    """The model-input tensors ``tensor`` is computed from.

    The Keras graph is walked rather than the nnodely DAG because the two do
    not always agree: a Constant or Parameter is wired to an arbitrary anchor
    input to give it a batch context, so the Keras sub-graph of a relation can
    need a tensor its symbolic predecessors never mention. Sources are told
    apart by object, and ordered by name so a reloaded model resolves them
    exactly like the saved one did.
    """
    sources: dict[int, Any] = {}
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
            sources[id(current)] = current
            continue
        node = operation._inbound_nodes[history[1]]
        stack.extend(node.input_tensors)
    return sorted(sources.values(), key=lambda tensor: tensor._keras_history[0].name)


# ------------------------------------------------------------------
# DAG topological ordering
# ------------------------------------------------------------------
def toposort(model) -> list[Node]:
    return toposort_outputs(model._outputs)


def toposort_outputs(outputs: list[Output]) -> list[Node]:
    order = []
    visited = set()

    def dfs(node: Node) -> None:
        visited.add(node)

        for pred in node._preds:
            if pred not in visited:
                dfs(pred)
        order.append(node)

    for output in outputs:
        # An output can also feed another one, e.g. a minimizer source that
        # an Output exposes: visiting it again would apply its layer twice.
        if output not in visited:
            dfs(output)

    return order


def _copies(scope: Scope) -> dict[int, Node]:
    """The node of the graph every node ``scope`` - and the scope of every
    call it makes - holds stands for, by the id of the node held: the copy
    inlining made, or what a call returns for the output it is read as. A
    node of the enclosing scope wins over one of a call."""
    found: dict[int, Node] = {}
    for call in scope.calls.values():
        found.update(_copies(call))
    found.update(
        (id(held), node) for node, held in scope.memo.items() if held is not node
    )
    return found


def _read_nodes(roots, declared=None, originals=None, streams=None) -> list[Node]:
    """Every node ``roots`` read, inside the models they call too, in the
    order the graph reaches them; with ``declared``, the inputs of a Modely,
    as it reads them (see Scope.declared). They are copies, made inlining
    the models called: ``originals``, if given, is filled with the node of
    the graph each Input copied stands for, by the id of the copy, and
    ``streams`` with the node every node read stands for (:func:`_copies`)."""
    from nnodely.layers.input import Input

    scope = Scope(declared=None if declared is None else set(declared))
    nodes = toposort_outputs([flatten_node(root, scope) for root in roots])
    if streams is not None:
        streams.update(_copies(scope))
    if originals is not None:
        # A hidden state left undeclared is inlined as its init, which can be
        # an Input of another name: only a copy of its own name is the node.
        for node, copied in scope.merged_memo().items():
            if (
                isinstance(node, Input)
                and isinstance(copied, Input)
                and copied._input is node._input
            ):
                originals.setdefault(id(copied), node)
    return nodes


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
        stack.extend(node._preds)
    return list(streams.values())


def _parameters(model: "Modely") -> list["Parameter"]:
    """Every Parameter ``model`` computes with, each once: reached from its
    outputs and minimizers, through the models it calls and the bodies of its
    Loops, in the order the graph reaches them - inputs
    before the nodes they feed, the model's own before its bodies'."""
    return _parameters_and_models(model)[0]


def _parameters_and_models(
    model: "Modely",
) -> tuple[list["Parameter"], list["Modely"]]:
    """:func:`_parameters`, and every model the walk reaches, each once:
    ``model`` first, then the models it calls and the bodies of its Loops,
    theirs too."""
    from nnodely.layers.parameter import Parameter

    nodes, models = _nodes_and_models(model)
    return [node for node in nodes if isinstance(node, Parameter)], models


def _nodes_and_models(model: "Modely") -> tuple[list[Node], list["Modely"]]:
    """Every node ``model`` computes with, each once, in the order
    :func:`_parameters` reaches them, and every model the walk reaches."""
    groups = _nodes_by_model(model)
    return [node for _, nodes in groups for node in nodes], [m for m, _ in groups]


def _nodes_by_model(
    model: "Modely", holders: dict[int, "Modely"] | None = None
) -> list[tuple["Modely", list[Node]]]:
    """Every model the walk of :func:`_nodes_and_models` reaches, in that
    order, each with the nodes it reaches first: those of its own graph -
    not of the models it calls - but those an earlier model reached already.
    ``holders``, if given, is filled with the model whose graph first reaches
    each one, by id, ``model`` aside."""
    from nnodely.core.modely import Modely

    found: set[int] = set()
    groups: list[tuple[Modely, list[Node]]] = []
    models: list[Modely] = [model]
    reached: set[int] = set()
    while models:
        current = models.pop(0)
        if id(current) in reached:
            continue
        reached.add(id(current))
        nodes: list[Node] = []
        groups.append((current, nodes))
        roots: list[Any] = [
            *current._outputs,
            *(
                side
                for minimizer in current._minimizers
                for side in (minimizer["source"], minimizer["target"])
            ),
        ]
        # Depth first, each node's predecessors in order.
        stack = roots[::-1]
        visited = set()
        while stack:
            node = stack.pop()
            if not isinstance(node, Node) or id(node) in visited:
                continue
            visited.add(id(node))
            if id(node) not in found:
                found.add(id(node))
                nodes.append(node)
            # A called model, or the body a block rolls out.
            for inner in (getattr(node, "_model", None), getattr(node, "_f", None)):
                if isinstance(inner, Modely):
                    models.append(inner)
                    if holders is not None:
                        holders.setdefault(id(inner), current)
            stack.extend(reversed(node._preds))
    return groups
