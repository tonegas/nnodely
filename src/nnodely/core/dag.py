"""
DAG - Lazy/DAG approach. No global graph.
Each node has predecessors; Model traverses from output backwards.

Dimensioni: seq, time, dim. Shape = seq + (time,) + dim.
Default: seq=(), time=1, dim=(1,)
"""

import copy

SEQ_TIME_DIM_DEFAULT = ((), 1, (1,))
_node_counter = 0


def next_name(prefix: str) -> str:
    """Nome univoco per nodi generati."""
    global _node_counter
    _node_counter += 1
    return f"{prefix}{_node_counter}"


# ------------------------------------------------------------------
# flattening logic
# ------------------------------------------------------------------
def flatten(model):
    flattened = model
    for node in model._order:
        if node.node_type == "Model":
            flattened = _aggregate_models(model=flattened, submodel=node)
    if any(n.node_type == "Model" for n in flattened._order):
        return flatten(flattened)
    return flattened


def _clone_graph(order):
    """
    Clone all nodes in `order` and reconnect predecessors to the cloned nodes.
    """
    clone_map = {}

    # first pass: clone nodes only
    for node in order:
        new_node = copy.copy(node)
        new_node.predecessors = []
        clone_map[node] = new_node

    # second pass: reconnect predecessors
    for node in order:
        new_node = clone_map[node]
        new_node.predecessors = [
            clone_map[p] for p in node.predecessors if p in clone_map
        ]

    cloned_order = [clone_map[node] for node in order]
    return cloned_order, clone_map


def _aggregate_models(model, submodel):
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

    model_order = [n for n in model_order if n is not submodel_cloned]

    for node in model_order:
        new_preds = []
        for pred in node.predecessors:
            if pred is submodel_cloned:
                new_preds.extend(replacement_preds)
            else:
                new_preds.append(pred)
        node.predecessors = new_preds

    # Rebuild outputs from cloned parent outputs
    cloned_outputs = [model_map[out] for out in model.outputs]

    from nnodely.core.modely import Modely

    flat_model = Modely(
        name=model.name + "_" + submodel.model.name,
        outputs=cloned_outputs,
    )
    # keep minimizers
    flat_model._minimizers = copy.copy(model._minimizers)

    return flat_model


# ------------------------------------------------------------------
# DAG logic
# ------------------------------------------------------------------
def to_tuple(x, default=(1,)):
    """Converte int/tuple/None in tuple. 0 -> default."""
    if x is None:
        return default
    if isinstance(x, int):
        return (x,) if x != 0 else default
    return tuple(x) if x else default


def get_seq_time_dim(node):
    """Estrae (seq, time, dim) da Stream, Input o Layer. Default: seq=(), time=1, dim=(1,)."""
    seq = tuple(getattr(node, "seq", ()) or ())
    time = getattr(node, "time", None) or 1
    dim = to_tuple(getattr(node, "dim", (1,)), (1,))
    return seq, time, dim


def seq_time_dim_to_shape(seq, time, dim):
    """Converte (seq, time, dim) nella shape Keras (senza batch)."""
    seq = tuple(seq or ())
    dim = to_tuple(dim, (1,))
    return seq + (time,) + dim


def same_shape(a, b):
    return get_seq_time_dim(a) == get_seq_time_dim(b)


def collect_and_order(output_nodes):
    """
    Raccoglie tutti i nodi dagli output e restituisce ordine topologico.
    DFS post-order da output verso input.
    """
    if not isinstance(output_nodes, (list, tuple)):
        output_nodes = [output_nodes]
    result = []
    visited = set()

    def dfs(node):
        if id(node) in visited:
            return
        visited.add(id(node))
        for pred in getattr(node, "predecessors", []):
            dfs(pred)
        result.append(node)

    for out in output_nodes:
        dfs(out)
    return result
