"""
DAG - Lazy/DAG approach. No global graph.
Each node has predecessors; Model traverses from output backwards.

Dimensioni: seq, time, dim. Shape = seq + (time,) + dim.
Default: seq=(), time=1, dim=(1,)
"""

import copy

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


# def _clone_graph(order):
#     """
#     Clone all nodes in `order` and reconnect predecessors to the cloned nodes.
#     """
#     cloned_order = []
#     for node in order:
#         new_node = copy.copy(node)
#         new_node.predecessors = copy.copy(node.predecessors)
#         cloned_order.append(new_node)
#     return cloned_order

# def _aggregate_models(model, submodel):
#     model_order = _clone_graph(model._order)
#     submodel_order = _clone_graph(submodel.model._order)

#     cloned_sub_outputs = [n for n in submodel_order if n.node_type == "Output"]
#     cloned_model_outputs = [n for n in model_order if n.node_type == "Output"]

#     ## Remove Inputs and Outputs
#     submodel_order = [n for n in submodel_order if n.node_type not in ["Input", "Output"]]

#     ## substitute submodel inputs predecessors with external mapped streams
#     for node in submodel_order:
#         for pred in node.predecessors:
#             if pred.node_type == "Input" and pred.name in submodel.input_map:
#                 ext_stream = submodel.input_map[pred.name]
#                 node.predecessors.remove(pred)
#                 node.predecessors.append(ext_stream)

#     ## Remove ModelCall node from parent graph and reconnect to selected child output predecessors
#     model_order = [n for n in model_order if n is not submodel]

#     for node in model_order:
#         for pred in node.predecessors:
#             if pred is submodel:
#                 node.predecessors.remove(pred)
#                 node.predecessors.extend(p for p in cloned_sub_outputs)

#     from nnodely.core.modely import Modely
#     flat_model = Modely(
#         name=model.name + "_" + submodel.model.name,
#         outputs=cloned_sub_outputs + cloned_model_outputs,
#     )
#     # keep minimizers
#     flat_model._minimizers = copy.copy(model._minimizers)

#     return flat_model


def _clone_graph(order):
    """
    Clone all nodes in `order` and reconnect predecessors to the cloned nodes.
    """
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
# DAG topological ordering
# ------------------------------------------------------------------
def toposort(output_nodes):
    """
    Restituisce l'ordine topologico dei nodi a partire dagli output.
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
