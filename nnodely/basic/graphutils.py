import networkx as nx

def to_graph(model_json : dict) -> nx.DiGraph:
    """
    Convert the nnodely JSON dictionary into a directed NetworkX graph.
    """
    G = nx.DiGraph()

    # ---- Add Inputs ----
    for name, attrs in model_json.get("Inputs", {}).items():
        G.add_node(name, type="Input", **attrs)

    # ---- Add Constants ----
    for name, attrs in model_json.get("Constants", {}).items():
        G.add_node(name, type="Constant", **attrs)

    # ---- Add Parameters ----
    for name, attrs in model_json.get("Parameters", {}).items():
        G.add_node(name, type="Parameter", **attrs)

    # ---- Add Functions ----
    for name, attrs in model_json.get("Functions", {}).items():
        G.add_node(name, type="Function", **attrs)

    # ---- Add Relations (Blocks) ----
    relations = model_json.get("Relations", {})

    for node_name, rel in relations.items():
        block_type = rel[0]
        inputs = rel[1]

        # attach entire relation info for later serialization
        G.add_node(node_name, type=block_type, relation=rel)

        # add edges from inputs to node
        for inp in inputs:
            if isinstance(inp, str):  # simple case (string reference)
                G.add_edge(inp, node_name)

    # ---- Add Output mapping ----
    for out_name, src in model_json.get("Outputs", {}).items():
        G.add_node(out_name, type="Output")
        G.add_edge(src, out_name)

    return G

def to_json(G : nx.DiGraph) -> dict:
    """
    Serialize a NetworkX nnodely graph back into the original JSON structure.
    """
    out = {
        "Inputs": {},
        "Constants": {},
        "Parameters": {},
        "Functions": {},
        "Relations": {},
        "Outputs": {}
    }

    # Categorize nodes by type
    for n, attrs in G.nodes(data=True):
        ntype = attrs.get("type")

        if ntype == "Input":
            out["Inputs"][n] = {k: v for k, v in attrs.items() if k != "type"}
        elif ntype == "Constant":
            out["Constants"][n] = {k: v for k, v in attrs.items() if k != "type"}
        elif ntype == "Parameter":
            out["Parameters"][n] = {k: v for k, v in attrs.items() if k != "type"}
        elif ntype == "Function":
            out["Functions"][n] = {k: v for k, v in attrs.items() if k != "type"}
        elif ntype == "Output":
            # Output always has exactly 1 input
            src = next(G.predecessors(n))
            out["Outputs"][n] = src
        else:
            # Relations (Add, Fir, TimePart, etc.)
            rel = attrs.get("relation")
            if rel:
                out["Relations"][n] = rel

    return out