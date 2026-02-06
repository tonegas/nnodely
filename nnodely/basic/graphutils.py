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

def plot_graphviz_structure(graph:nx.DiGraph, filename='nnodely_graph', view=True): # pragma: no cover
    import shutil
    from graphviz import view
    from graphviz import Digraph

    # Check if Graphviz is installed
    if shutil.which('dot') is None:
        print(
            "Graphviz does not appear to be installed on your system. "
            "Please install it from https://graphviz.org/download/"
        )
        return
    
    dot = Digraph(comment='Structured Neural Network')

    # Set graph attributes for top-down layout and style
    dot.attr(rankdir='LR', size='21')
    dot.attr('node', shape='box', style='filled', color='lightgray', fontname='Helvetica')
    color_map = {
        'Input': 'lightgreen',
        'Output': 'red',
        'Parameter': 'orange',
        'Function': 'blue',
        'Constant': 'lightgray',
        'Minimize': 'purple',
        'Relation': 'lightblue'
    }
    ## Add nodes
    for n, attr in graph.nodes(data=True):
        shape_map = 'ellipse' if attr.get('type') in ['Function', 'Parameter', 'Minimize', 'Input', 'Output'] else 'box'
        dot.node(n, label=f"{n}\nType: {attr.get('type')}", fillcolor=color_map.get(attr.get('type'), 'lightgray'), shape=shape_map)
    ## Add edges
    for u, v, attr in graph.edges(data=True):
        if attr.get('type', None) == 'connect':
            dot.edge(u, v, label='connect', color='blue', fontcolor='black')
        elif attr.get('type', None) == 'loop':
            dot.edge(u, v, label='loop', color='red', fontcolor='black')
        else:
            dot.edge(u, v, color='black')

    # Render the graph
    dot.render(filename=filename, view=view, format='svg')  # opens in default viewer and saves as SVG