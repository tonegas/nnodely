from copy import copy

import keras

from nnodely.core.stream import _NODE_NAMES, Node, Stream, _reserve_name
from typing import TYPE_CHECKING, Any, Callable, Container, Generator, Sequence

if TYPE_CHECKING:
    from nnodely.core.modely import Modely
    from nnodely.layers.output import Output
    from nnodely.layers.parameter import Parameter

_node_counter = 0


def _next_name(prefix: str) -> str:
    """A name for a node left unnamed, used by no node of this process.

    The counter alone is not enough: a given name, or one read back from a
    saved model, can already have the form it would produce.
    """
    global _node_counter
    while True:
        _node_counter += 1
        name = f"{prefix}{_node_counter}"
        if name not in _NODE_NAMES:
            break
    # Reserved here as well, because a LocalModel or an EquationLearner takes
    # its name from here without being a node, and derives its layers' names.
    _reserve_name(name)
    return name


def _free_name(name: str, taken: Container[str]) -> str:
    """``name``, or the first of ``name_2``, ``name_3``, ... not in
    ``taken``, a set of names or a dict keyed by them. It is how a name
    nnodely generates, or shows, keeps apart from those already used. The caller
    records the name it takes."""
    free, index = name, 2
    while free in taken:
        free, index = f"{name}_{index}", index + 1
    return free


# ------------------------------------------------------------------
# picking the objects of a model by name, by index or by themselves
# ------------------------------------------------------------------
#
# Names are how the user reaches a Parameter, a model or a stream, and nothing
# keeps two objects from sharing one: layers named alike by hand, a block and
# the model calling it, two copies of one loaded model. So every call taking a
# name looks it up among the objects the graph reaches, in that order, and
# refuses a name several carry; the lists the user reads them from are also
# indexed by position, as the index column of their tables shows. A key is
# resolved to its object where the user gives it, by resolve(), and inside
# nnodely the objects themselves are used, never their names.

#: ``(object, index, where)``: an object a key may pick, its index in the
#: list it is picked from (or None) and where it is, for the listing of a
#: name several carry.
_Candidate = tuple[Any, "int | None", str]


class NameNotFound(KeyError, ValueError):
    """A key that picks nothing: a ``KeyError``, as a missing key of a table
    is, and a ``ValueError``, as a wrong argument of a call is."""

    def __str__(self) -> str:
        # KeyError quotes its message; this one reads as a sentence.
        return str(self.args[0]) if self.args else ""


def _by_name_or_index(key, owner: str) -> bool:
    """Whether ``key`` picks by name: a string; an int or a slice picks by
    position, as in a list. Anything else, such as a name and an index
    together, raises ``TypeError``."""
    if isinstance(key, str):
        return True
    if isinstance(key, (int, slice)) and not isinstance(key, bool):
        return False
    raise TypeError(
        f"{owner}: an item is picked by its name, a string, or by its index, "
        f"an int (the index column of the table), not both; got {key!r}."
    )


def _pick(
    candidates: Sequence[_Candidate],
    name: str,
    *,
    kind: str,
    owner: str,
    hint: str,
    kinds: str | None = None,
) -> list:
    """The object of ``candidates`` that carries the name ``name``, as a
    list: ``[]`` for none. When several carry it, raises ``ValueError``
    listing them, and ``hint``: how to pick one, or to rename one. ``kinds``
    is the plural of ``kind``, its name with an s by default."""
    found = [
        (item, index, where) for item, index, where in candidates if item.name == name
    ]
    if len(found) <= 1:
        return [item for item, _, _ in found]
    indices = [index for _, index, _ in found]
    at = "" if None in indices else f", at indices {indices}"
    listing = "; ".join(
        f"{'' if index is None else f'[{index}] '}{name} {where}".rstrip()
        for _, index, where in found
    )
    raise ValueError(
        f"{owner}: the name {name!r} is ambiguous, {len(found)} "
        f"{kinds or kind + 's'} carry it{at}: {listing}. {hint}"
    )


def _resolve(
    key,
    candidates: Sequence[_Candidate],
    *,
    kind: str,
    owner: str,
    hint: str,
    kinds: str | None = None,
    indices: bool = False,
    objects: type | tuple[type, ...] | None = None,
    same: Callable[[Any, Any], bool] | None = None,
    missing: Callable[[Any], str] | None = None,
) -> Any:
    """The object of ``candidates`` the user's ``key`` picks. It is the one
    place a key becomes an object:

    - a name, a string: the one candidate carrying it; a name several carry
      raises ``ValueError`` listing them (:func:`_pick`);
    - with ``indices``, an int or a slice: by position, as in a list;
    - with ``objects``, an instance of those types: the candidate it is or,
      with ``same(candidate, key)``, the one it stands for, as an Input
      stands for the copies a build makes of it.

    Any other key raises ``TypeError``. A key that picks nothing raises
    :class:`NameNotFound`, with ``missing(key)`` as its message when given."""
    items = [item for item, _, _ in candidates]
    plural = kinds or kind + "s"
    if objects is not None and isinstance(key, objects):
        match = same or (lambda item, given: item is given)
        found = next((item for item in items if match(item, key)), None)
        if found is None:
            raise NameNotFound(
                missing(key)
                if missing
                else f"{owner}: {getattr(key, 'name', key)!r} is not one of its "
                f"{plural}."
            )
        return found
    if indices and not _by_name_or_index(key, owner):
        return items[key]
    if not isinstance(key, str):
        article = "an" if kind[:1].lower() in "aeiou" else "a"
        raise TypeError(
            f"{owner}: {article} {kind} is picked by its name, a string"
            f"{', or by itself' if objects is not None else ''}; got {key!r}."
        )
    picked = _pick(candidates, key, kind=kind, kinds=kinds, owner=owner, hint=hint)
    if not picked:
        raise NameNotFound(
            missing(key)
            if missing
            else f"{owner}: no {kind} named {key!r}; the names are "
            f"{[item.name for item in items]}."
        )
    return picked[0]


class Table(tuple):
    """A tuple of the objects of a model a table prints (its Parameters,
    Inputs, outputs, models or minimizers), picked by name or by index, the
    index column of the table, as :func:`_resolve` picks them. A subclass
    sets ``_kind`` and ``_attr``, the name of its list on a model, and gives
    where each item is with :meth:`_where`.

    It reads as a tuple, which iterated gives the objects, and as a dict of
    them by name: ``keys()``, ``values()`` and ``items()`` are lists, and
    ``in`` takes a name or an object. ``print`` shows the table, with the
    ``idx`` column; ``repr`` the same items, one per line, as
    ``[i] 'name': Object(...)``. A tuple, so read only: the model makes a
    new one at every read, and changing one would change nothing."""

    _kind = "item"
    _attr = "items"
    _title = ""

    def _owner(self) -> str:
        return self._title or self._attr.capitalize()

    def _where(self, index: int, item) -> str:
        """Where ``item``, at ``index``, is: for the listing of a name several
        carry."""
        return ""

    def _candidates(self) -> list[_Candidate]:
        return [
            (item, index, self._where(index, item)) for index, item in enumerate(self)
        ]

    def __getitem__(self, key):  # type: ignore[override]
        """An item by its name, or by its index (the index column of the
        table) as in a list. A name several items carry raises
        ``ValueError`` listing them, with their index."""
        return _resolve(
            key,
            self._candidates(),
            kind=self._kind,
            owner=self._owner(),
            hint=f"Pick one by its index, {self._attr}[i], or give it a name of "
            f"its own, {self._attr}[i].name = 'new_name'.",
            indices=True,
        )

    def __contains__(self, key) -> bool:
        """Whether an item has the name ``key``, or is the object ``key``."""
        if isinstance(key, str):
            return key in self.keys()
        return any(item is key for item in self)

    def keys(self) -> list[str]:
        return [item.name for item in self]

    def values(self) -> list:
        return list(self)

    def items(self) -> list[tuple[str, Any]]:
        return [(item.name, item) for item in self]

    def __repr__(self) -> str:
        cls = type(self).__name__
        if not self:
            return f"{cls}()"
        keys = [f"[{index}] {item.name!r}:" for index, item in enumerate(self)]
        width = max(len(key) for key in keys)
        lines = [f"  {key:<{width}} {item!r}," for key, item in zip(keys, self)]
        return "\n".join([f"{cls}(", *lines, ")"])


# ------------------------------------------------------------------
# flattening logic
# ------------------------------------------------------------------


class _Scope:
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
        self.calls: dict[Node, "_Scope"] = {}
        # The Parameters and Constants inlined, one copy each, shared with
        # every call: they are one object in every model that reads them. So
        # is an Input read from data, one array whatever reads it, unless a
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
        # and reads its init (zero without one) as the first step of a
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


def _flatten_node(node: Node, scope: _Scope) -> Any:
    """The copy of ``node`` inlined in ``scope``, every model it calls
    inlined too, each call in a scope of its own (see :class:`_Scope`).

    Without recursion: the steps of each node (:func:`_flatten_steps`) ask
    for the copies of the nodes it reads, and a stack runs the steps of the
    nodes being copied, in the order a recursion would. So a graph of any
    depth is inlined."""
    stack = [_flatten_steps(node, scope)]
    copied = None
    while stack:
        try:
            wanted = stack[-1].send(copied)
        except StopIteration as done:
            stack.pop()
            copied = done.value
        else:
            stack.append(_flatten_steps(*wanted))
            copied = None
    return copied


def _flatten_steps(
    node: Node, scope: _Scope
) -> Generator[tuple[Node, _Scope], Any, Any]:
    """The steps of :func:`_flatten_node` for one node: ``yield (pred,
    scope)`` is the copy of ``pred`` inlined in ``scope``, as a recursive
    call would give it; the copy of ``node`` is returned."""
    from nnodely.core.modely import IntermediateOutput
    from nnodely.core.layer import Layer
    from nnodely.layers.parameter import _Value
    from nnodely.layers.input import Input, _HiddenInit

    if node in scope.memo:
        return scope.memo[node]
    if (
        scope.declared is not None
        and isinstance(node, Input)
        and node._hidden
        and node not in scope.declared
    ):
        init = Stream._coerce_operand(0.0 if node._init is None else node._init)
        scope.memo[node] = yield (_HiddenInit(state=node.shape)([init]), scope)
        return scope.memo[node]
    shared = isinstance(node, _Value) or (
        isinstance(node, Input) and (node._update is None or not scope.loop)
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
            arguments: dict[Node, Any] = {}
            for internal_input, external_input in model_call._inputs_map.items():
                arguments[internal_input] = yield (external_input, scope)
            # A fed-back Input the model declares reads its argument as data:
            # at every step in a Modely; in a loop it keeps its feedback, and
            # the argument is its initial value.
            inits: dict[Node, Any] = {
                internal_input: argument
                for internal_input, argument in arguments.items()
                if scope.loop
                and isinstance(internal_input, Input)
                and internal_input._update is not None
            }
            bindings = {
                internal_input: argument
                for internal_input, argument in arguments.items()
                if internal_input not in inits
            }
            # A call starts from its bindings alone: what the caller has
            # inlined so far depends on the order the graph is walked in, and
            # sharing it made the result depend on that order too.
            call_scope = _Scope(
                bindings, scope.values, scope.pending, scope.loop, scope.declared
            )
            call_scope.inits = inits
            scope.calls[model_call] = call_scope

        internal_output = model_call._outputs_map[node]
        flat_output = yield (internal_output, call_scope)

        scope.memo[node] = flat_output
        return flat_output

    new_preds = []
    for pred in node._preds:
        new_preds.append((yield (pred, scope)))
    new_node = copy(node)
    new_node._preds = new_preds
    if isinstance(new_node, Layer):
        new_node._copied(scope.memo)
        if not getattr(new_node._layer, "weights", None):
            new_node._layer = None
    scope.memo[node] = new_node
    if shared:
        scope.values[node] = new_node
    if (
        scope.pending is not None
        and isinstance(node, Input)
        and node._update is not None
    ):
        scope.pending.append((new_node, node, scope))
    return new_node


def _flatten_fed_back(scope: _Scope) -> None:
    """Inline the stream every fed-back Input inlined in ``scope`` is fed
    back with, and its init, each in the call the Input was inlined in, and so
    on for the fed-back Inputs those read in turn.

    They are no predecessors of the Input, and wait for the graph to be
    inlined: an update can read the output of a call whose arguments read the
    Input, and inlined on reaching the Input it would make that call twice.
    """
    while scope.pending:
        inlined, source, source_scope = scope.pending.pop(0)
        inlined._update = _flatten_node(source._update, source_scope)
        if source in source_scope.inits:
            # A state starts from one value.
            if source._init is not None:
                raise ValueError(
                    f"{source.name!r} has two initial values: the init of the "
                    "block that keeps it, and the stream a call gives it as an "
                    "input of its model. Give one of them."
                )
            inlined._init = source_scope.inits[source]
        elif isinstance(source._init, Node):
            inlined._init = _flatten_node(source._init, source_scope)


def flatten(model):
    from nnodely.core.modely import Modely

    flat_outputs, memo = _flatten_graph(model._inputs, model._outputs, fed_back=True)
    flat_inputs = [memo[node] for node in model._inputs if node in memo]
    return Modely(f"{model.name}_flat", flat_inputs, flat_outputs)


def _flatten_graph(
    inputs: list[Any],
    outputs: list["Output"],
    *,
    fed_back: bool = False,
    one_step: bool = False,
) -> tuple[list[Any], dict[Node, Any]]:
    """The copies of ``outputs`` with every model they call inlined, and the
    copy of every node inlined, by node. ``fed_back`` inlines the stream
    every fed-back Input is fed back with, and its init, too: without it they
    are left as they are, in the graph of the model that declared them.
    ``one_step`` reads the init of every hidden fed-back Input not among
    ``inputs`` in its place, as a Modely is built."""
    scope = _Scope(
        pending=[] if fed_back else None, declared=set(inputs) if one_step else None
    )
    flat_outputs: list["Output"] = [_flatten_node(output, scope) for output in outputs]
    _flatten_fed_back(scope)
    return flat_outputs, scope.merged_memo()


def _undeclared_inputs(inputs, outputs, read=None) -> list:
    """Every Input the model of ``outputs`` reads that is not among
    ``inputs`` and is no hidden state, the model's own memory, which it
    starts from its init:

    - every one its one-step model reads, through its streams and the
      arguments of the models it calls, and as the init of a hidden state;
    - every fed-back one a rollout reads, through the streams the fed-back
      Inputs are fed back with too: a state is declared or hidden. The
      others read there are the loop's to check.

    ``read``, if given, is filled with the id of every one of ``inputs``
    the model reads, in one step or in a rollout."""
    from nnodely.layers.input import Input

    declared = {id(node) for node in inputs}
    found: dict[int, Any] = {}
    for rollout in (False, True):
        seen: set[int] = set()
        stack = list(outputs)
        while stack:
            node = stack.pop()
            if not isinstance(node, Node) or id(node) in seen:
                continue
            seen.add(id(node))
            if isinstance(node, Input):
                fed_back = node._update is not None
                if rollout and fed_back:
                    stack.append(node._update)
                if id(node) in declared:
                    if read is not None:
                        read.add(id(node))
                elif node._hidden and fed_back:
                    stack.append(node._init)
                elif fed_back or not rollout:
                    found[id(node)] = node
            stack.extend(node._preds)
    return list(found.values())


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
def _toposort(roots: Sequence[Node]) -> list[Node]:
    """Every node ``roots`` are computed from, the roots included, each
    after its predecessors: a model's graph is ``_toposort(model._outputs)``.

    A depth-first walk, each node's predecessors in their order, without
    recursion: a stack holds every node of the path being walked with the
    predecessors it has left to visit, so a graph of any depth is sorted."""
    order: list[Node] = []
    visited: set[Node] = set()
    for root in roots:
        # A root can also feed another one, e.g. a minimizer source that an
        # Output exposes: visiting it again would apply its layer twice.
        if root in visited:
            continue
        visited.add(root)
        stack = [(root, iter(root._preds))]
        while stack:
            node, preds = stack[-1]
            for pred in preds:
                if pred not in visited:
                    visited.add(pred)
                    stack.append((pred, iter(pred._preds)))
                    break  # its predecessors first, as a recursion would
            else:
                stack.pop()
                order.append(node)  # after every one of its predecessors
    return order


def _copies(scope: _Scope) -> dict[int, Node]:
    """The node of the graph every node held by ``scope``, and by the scope
    of every call it makes, stands for, by the id of the node held: the copy
    inlining made, or what a call returns for the output it is read as. A
    node of the enclosing scope wins over one of a call."""
    found: dict[int, Node] = {}
    for call in scope.calls.values():
        found.update(_copies(call))
    found.update(
        (id(held), node) for node, held in scope.memo.items() if held is not node
    )
    return found


def _read_nodes(
    roots, declared=None, originals=None, streams=None, hidden=None
) -> list[Node]:
    """Every node ``roots`` read, inside the models they call too, in the
    order the graph reaches them; with ``declared``, the inputs of a Modely,
    as it reads them (see Scope.declared). They are copies, made inlining
    the models called: ``originals``, if given, is filled with the node of
    the graph each Input copied stands for, by the id of the copy, and
    ``streams`` with the node every node read stands for (:func:`_copies`).
    ``hidden``, given with ``declared``, is filled with every hidden state
    the Modely reads its init in place of."""
    from nnodely.layers.input import Input

    scope = _Scope(declared=None if declared is None else set(declared))
    nodes = _toposort([_flatten_node(root, scope) for root in roots])
    if streams is not None:
        streams.update(_copies(scope))
    if hidden is not None and scope.declared is not None:
        hidden.extend(
            node
            for node in scope.merged_memo()
            if isinstance(node, Input) and node._hidden and node not in scope.declared
        )
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
    Loops, in the order the graph reaches them: inputs before the nodes
    they feed, the model's own before its bodies'."""
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
    order, each with the nodes it reaches first: those of its own graph, not
    of the models it calls, but those an earlier model reached already.
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
            if isinstance(node._model, Modely):
                models.append(node._model)
                if holders is not None:
                    holders.setdefault(id(node._model), current)
            stack.extend(reversed(node._preds))
    return groups
