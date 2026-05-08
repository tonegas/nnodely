import copy
from pprint import pformat

from nnodely.support.utils import check
from nnodely.support.logger import logging, nnLogger

log = nnLogger(__name__, logging.WARNING)

# Sections whose inner *entries* are mutated in place by callers after merge() and
# therefore must be detached (shallow-copied) in the result:
#   Inputs      -> connect / closedLoop / local / ns / ntot
#   Functions   -> Fuzzify / LocalModel / dim_out / params_and_consts
#   Parameters  -> values / init_values / init_fun / dim
#   Constants   -> values
# All other sections (Relations / Outputs / Models / Minimizers) are append-only at
# the section level, so their inner values are shared between operands and result.
_MUTABLE_ENTRY_SECTIONS = frozenset(("Inputs", "Functions", "Parameters", "Constants"))


def get_window(obj):
    """Return the window key (``'tw'``/``'sw'``) of ``obj.dim``, or ``None``."""
    return "tw" if "tw" in obj.dim else ("sw" if "sw" in obj.dim else None)


def _shallow_copy_entries(d):
    """Return a fresh dict where dict-typed values are shallow-copied."""
    if not d:
        return {}
    return {k: dict(v) if isinstance(v, dict) else v for k, v in d.items()}


def _window_union(a, b):
    """Return ``[min(a[0], b[0]), max(a[1], b[1])]`` without mutating *a* or *b*."""
    return [a[0] if a[0] <= b[0] else b[0], a[1] if a[1] >= b[1] else b[1]]


def _overlay_with_windows(dst, src):
    """Overlay *src* onto a fresh copy of *dst*; union ``tw``/``sw`` lists.

    Used for the ``Info`` section and for individual ``Inputs`` descriptors,
    which share the same shape (flat dict of scalars/2-element windows).
    """
    out = dict(dst) if isinstance(dst, dict) else {}
    if not isinstance(src, dict) or not src:
        return out
    for k, v in src.items():
        if (
            (k == "tw" or k == "sw")
            and isinstance(v, list)
            and isinstance(out.get(k), list)
        ):
            out[k] = _window_union(out[k], v)
        else:
            out[k] = v
    return out


def _merge_section(sec, dst, src):
    """Merge a single named section of the model JSON.

    Three regimes:
    * ``Info`` — flat dict: scalar overlay + ``tw``/``sw`` window union.
    * ``_MUTABLE_ENTRY_SECTIONS`` — union of named entries; each surviving
      entry is shallow-copied so callers can mutate the result.
      ``Inputs`` additionally union'es per-entry windows.
    * Everything else (``Relations`` / ``Outputs`` / ``Models`` / ``Minimizers``):
      union of named entries; inner values are *shared* with the operands
      (callers only add new keys, never mutate inner values in place).
    """
    if sec == "Info":
        return _overlay_with_windows(dst, src)

    if sec in _MUTABLE_ENTRY_SECTIONS:
        out = _shallow_copy_entries(dst)
        if not src:
            return out
        if sec == "Inputs":
            for k, sv in src.items():
                out[k] = _overlay_with_windows(out.get(k), sv)
        else:
            for k, sv in src.items():
                out[k] = dict(sv) if isinstance(sv, dict) else sv
        return out

    # Shared-value sections: dict-union if both are dicts, else source overrides.
    if isinstance(dst, dict) and isinstance(src, dict):
        out = dict(dst)
        out.update(src)
        return out
    if isinstance(src, dict):
        return dict(src)
    if isinstance(dst, dict):
        return dict(dst)
    return src if src is not None else dst


def _iter_overlap(a, b):
    """Yield ``(key, a[key], b[key])`` for keys present in both dicts, scanning
    the smaller side."""
    if not a or not b:
        return
    small, big = (a, b) if len(a) <= len(b) else (b, a)
    for key, value in small.items():
        other = big.get(key)
        if other is not None:
            yield key, value, other


def _validate_compat(source, destination):
    """Verify that overlapping Functions/Parameters agree on their declared shape
    (``n_input``, ``dim``, ``tw``/``sw``)."""
    for key, a, b in _iter_overlap(
        source.get("Functions") or {}, destination.get("Functions") or {}
    ):
        if a and b and "n_input" in a and "n_input" in b:
            check(
                a["n_input"] == b["n_input"],
                TypeError,
                f"The ParamFun {key} is present multiple times, with different number of inputs. "
                f"The ParamFun {key} is called with {a['n_input']} parameters and with {b['n_input']} parameters.",
            )

    for key, a, b in _iter_overlap(
        source.get("Parameters") or {}, destination.get("Parameters") or {}
    ):
        if "dim" in a and "dim" in b:
            check(
                a["dim"] == b["dim"],
                TypeError,
                f"The Parameter {key} is present multiple times, with different dimensions. "
                f"The Parameter {key} is called with {a['dim']} dimension and with {b['dim']} dimension.",
            )
        wa = "tw" if "tw" in a else ("sw" if "sw" in a else None)
        if wa is None:
            continue
        wb = "tw" if "tw" in b else ("sw" if "sw" in b else None)
        check(
            wa == wb and a[wa] == b[wb],
            TypeError,
            f"The Parameter {key} is present multiple times, with different window. "
            f"The Parameter {key} is called with {wa}={a[wa]} dimension and with {wb}={b[wb] if wb else None} dimension.",
        )


def merge(source, destination):
    """
    Combine two model JSONs into a fresh, independent dict.

    Complexity: ``O(|sections| + |mutable entries| + |new keys in source|)``
    per call. No recursion, no full deep-copy of *destination*. Inner values
    of named sections are shared with the operands wherever safe; entries in
    ``Inputs``/``Functions``/``Parameters``/``Constants`` are shallow-copied
    because callers mutate them in place. ``tw``/``sw`` ranges are union'd.
    """
    if source is destination:
        return {sec: _merge_section(sec, v, None) for sec, v in destination.items()}

    _validate_compat(source, destination)

    sections = set(destination) | set(source)
    result = {
        sec: _merge_section(sec, destination.get(sec), source.get(sec))
        for sec in sections
    }

    if log.isEnabledFor(logging.DEBUG):
        log.debug("Merge Source\n" + pformat(source))
        log.debug("Merge Destination\n" + pformat(destination))
        log.debug("Merge Result\n" + pformat(result))
    return result


def get_models_json(json):
    model_json = {}
    model_json["Parameters"] = list(json["Parameters"].keys())
    model_json["Constants"] = list(json["Constants"].keys())
    model_json["Inputs"] = list(json["Inputs"].keys())
    model_json["Outputs"] = list(json["Outputs"].keys())
    model_json["Functions"] = list(json["Functions"].keys())
    model_json["Relations"] = list(json["Relations"].keys())
    return model_json


def check_model(json):
    all_inputs = json["Inputs"].keys()
    all_outputs = json["Outputs"].keys()

    from nnodely.basic.relation import MAIN_JSON

    subjson = MAIN_JSON
    for name in all_outputs:
        subjson = merge(subjson, subjson_from_output(json, name))
    needed_inputs = subjson["Inputs"].keys()
    extenal_inputs = set(all_inputs) - set(needed_inputs)

    check(
        all_inputs == needed_inputs,
        RuntimeError,
        f"Connect or close loop operation on the inputs {list(extenal_inputs)}, that are not used in the model.",
    )
    return json


def binary_cheks(self, obj1, obj2, name):
    from nnodely.basic.relation import Stream, toStream

    obj1, obj2 = toStream(obj1), toStream(obj2)
    check(
        type(obj1) is Stream,
        TypeError,
        f"The type of {obj1} is {type(obj1)} and is not supported for add operation.",
    )
    check(
        type(obj2) is Stream,
        TypeError,
        f"The type of {obj2} is {type(obj2)} and is not supported for add operation.",
    )
    window_obj1 = get_window(obj1)
    window_obj2 = get_window(obj2)
    if window_obj1 is not None and window_obj2 is not None:
        check(
            window_obj1 == window_obj2,
            TypeError,
            f"For {name} the time window type must match or None but they were {window_obj1} and {window_obj2}.",
        )
        check(
            obj1.dim[window_obj1] == obj2.dim[window_obj2],
            ValueError,
            f"For {name} the time window must match or None but they were {window_obj1}={obj1.dim[window_obj1]} and {window_obj2}={obj2.dim[window_obj2]}.",
        )
    check(
        obj1.dim["dim"] == obj2.dim["dim"]
        or obj1.dim == {"dim": 1}
        or obj2.dim == {"dim": 1},
        ValueError,
        f"For {name} the dimension of {obj1.name} = {obj1.dim} must be the same of {obj2.name} = {obj2.dim}.",
    )
    dim = obj1.dim | obj2.dim
    dim["dim"] = max(obj1.dim["dim"], obj2.dim["dim"])
    return obj1, obj2, dim


def subjson_from_relation(json, relation):
    # Read-only DAG walk with iterative DFS + visited memo (relations form a DAG with
    # heavy reuse, e.g. RK4 expansions: recursive untracked walks blow up exponentially).
    inputs = set()
    relations = set()
    constants = set()
    parameters = set()
    functions = set()

    j_inputs = json["Inputs"]
    j_constants = json["Constants"]
    j_parameters = json["Parameters"]
    j_functions = json["Functions"]
    j_relations = json["Relations"]

    visited = set()
    stack = [relation]
    while stack:
        rel = stack.pop()
        if rel in visited:
            continue
        visited.add(rel)
        if rel in j_inputs:
            inputs.add(rel)
            entry = j_inputs[rel]
            if "connect" in entry and entry.get("local") == 1:
                stack.append(entry["connect"])
            if "closed_loop" in entry and entry.get("local") == 1:
                stack.append(entry["closed_loop"])
        elif rel in j_constants:
            constants.add(rel)
        elif rel in j_parameters:
            parameters.add(rel)
        elif rel in j_functions:
            functions.add(rel)
            f_entry = j_functions[rel]
            pcs = (
                f_entry.get("params_and_consts") if isinstance(f_entry, dict) else None
            )
            if pcs:
                stack.extend(pcs)
        elif rel in j_relations:
            relations.add(rel)
            r_entry = j_relations[rel]
            stack.extend(r_entry[1])
            kind = r_entry[0]
            if kind in ("Fir", "Linear", "Fuzzify", "ParamFun") and len(r_entry) > 2:
                stack.extend(r_entry[2:])

    from nnodely.basic.relation import MAIN_JSON

    sub_json = copy.deepcopy(MAIN_JSON)
    sub_json["Relations"] = {
        key: value for key, value in json["Relations"].items() if key in relations
    }
    sub_json["Inputs"] = {
        key: value for key, value in json["Inputs"].items() if key in inputs
    }
    sub_json["Constants"] = {
        key: value for key, value in json["Constants"].items() if key in constants
    }
    sub_json["Parameters"] = {
        key: value for key, value in json["Parameters"].items() if key in parameters
    }
    sub_json["Functions"] = {
        key: value for key, value in json["Functions"].items() if key in functions
    }
    sub_json["Outputs"] = {}
    sub_json["Info"] = {}
    return sub_json


def subjson_from_output(json, outputs: str | list):
    from nnodely.basic.relation import MAIN_JSON

    sub_json = copy.deepcopy(MAIN_JSON)
    if type(outputs) is str:
        outputs = [outputs]
    for output in outputs:
        sub_json = merge(sub_json, subjson_from_relation(json, json["Outputs"][output]))
        sub_json["Outputs"][output] = json["Outputs"][output]
    return sub_json


def subjson_from_model(json, models: str | list):
    from nnodely.basic.relation import MAIN_JSON

    sub_json = copy.deepcopy(MAIN_JSON)
    models_names = (
        set([json["Models"]])
        if type(json["Models"]) is str
        else set(json["Models"].keys())
    )
    if type(models) is str or len(models) == 1:
        if len(models) == 1:
            models = models[0]
        check(models in models_names, AttributeError, f"Model [{models}] not found!")
        if type(json["Models"]) is str:
            outputs = set(json["Outputs"].keys())
        else:
            outputs = set(json["Models"][models]["Outputs"])
        sub_json["Models"] = models
    else:
        outputs = set()
        sub_json["Models"] = {}
        for model in models:
            check(model in models_names, AttributeError, f"Model [{model}] not found!")
            outputs |= set(json["Models"][model]["Outputs"])
            sub_json["Models"][model] = {
                key: value for key, value in json["Models"][model].items()
            }

    # Remove the extern connections not keys in the graph
    final_json = merge(sub_json, subjson_from_output(json, outputs))
    for key, value in final_json["Inputs"].items():
        if "connect" in value and (
            value["local"] == 0
            and value["connect"] not in final_json["Relations"].keys()
        ):
            del final_json["Inputs"][key]["connect"]
            del final_json["Inputs"][key]["local"]
            log.warning(
                f'The input {key} is "connect" outside the model connection removed for subjson'
            )
        if "closedLoop" in value and (
            value["local"] == 0
            and value["closedLoop"] not in final_json["Relations"].keys()
        ):
            del final_json["Inputs"][key]["closedLoop"]
            del final_json["Inputs"][key]["local"]
            log.warning(
                f'The input {key} is "closedLoop" outside the model connection removed for subjson'
            )
    return final_json


def subjson_from_minimize(json, minimizers: str | list):
    from nnodely.basic.relation import MAIN_JSON

    sub_json = copy.deepcopy(MAIN_JSON)

    if "Minimizers" in json:
        rel_A = [json["Minimizers"][key]["A"] for key in minimizers]
        rel_B = [json["Minimizers"][key]["B"] for key in minimizers]
        relations_name = set(rel_A) | set(rel_B)
        for rel_name in relations_name:
            minimizers_json = subjson_from_relation(json, rel_name)
            sub_json = merge(sub_json, minimizers_json)
        sub_json["Minimizers"] = {key: json["Minimizers"][key] for key in minimizers}

    return sub_json


def stream_to_str(obj, type="Stream"):
    from nnodely.visualizer.emptyvisualizer import color, GREEN
    from pprint import pformat

    stream = f" {type} "
    stream_name = f" {obj.name} {obj.dim} "

    title = color((stream).center(80, "="), GREEN, True)
    json = color(pformat(obj.json), GREEN)
    stream = color((stream_name).center(80, "-"), GREEN, True)
    return title + "\n" + json + "\n" + stream


def plot_structure(json, filename="nnodely_graph", library="matplotlib", view=True):
    # json = self.modely.json if json is None else json
    # if json is None:
    #     raise ValueError("No JSON model definition provided. Please provide a valid JSON model definition.")
    if library not in ["matplotlib", "graphviz"]:
        raise ValueError("Invalid library specified. Use 'matplotlib' or 'graphviz'.")
    if library == "matplotlib":
        plot_matplotlib_structure(json, filename, view=view)
    elif library == "graphviz":
        plot_graphviz_structure(json, filename, view=view)


def plot_matplotlib_structure(json, filename="nnodely_graph", view=True):
    import matplotlib.pyplot as plt
    from matplotlib import patches
    from matplotlib.lines import Line2D

    layer_positions = {}
    x, y = 0, 0  # Initial position
    dy, dx = 1.5, 2.5  # Spacing

    ## Layer Inputs:
    for input_name, input_type in json["Inputs"].items():
        layer_positions[input_name] = (x, y)
        y -= dy
    for constant_name in json["Constants"].keys():
        layer_positions[constant_name] = (x, y)
        y -= dy
    y_limit = abs(y)

    # Layers Relations:
    available_inputs = list(json["Inputs"].keys() | json["Constants"].keys())
    available_outputs = list(set(json["Outputs"].values()))
    while available_outputs:
        x += dx
        y = 0
        inputs_to_add, outputs_to_remove = [], []
        for relation_name, (relation_type, dependencies, *_) in json[
            "Relations"
        ].items():
            if all(dep in available_inputs for dep in dependencies) and (
                relation_name not in available_inputs
            ):
                inputs_to_add.append(relation_name)
                if relation_name in available_outputs:
                    outputs_to_remove.append(relation_name)
                layer_positions[relation_name] = (x, y)
                y -= dy
        y_limit = max(y_limit, abs(y))
        available_inputs.extend(inputs_to_add)
        available_outputs = [
            out for out in available_outputs if out not in outputs_to_remove
        ]

    ## Layer Outputs:
    x += dx
    y = 0
    for idx, output_name in enumerate(json["Outputs"].keys()):
        layer_positions[output_name] = (x, y)
        y -= dy  # Move down for the next input
    x_limit = abs(x)
    y_limit = max(y_limit, abs(y))

    # Create the plot
    fig, ax = plt.subplots(figsize=(x_limit, y_limit))
    # fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    # Plot rectangles for each layer
    colors, labels = (
        ["lightgreen", "lightblue", "orange", "lightgray"],
        ["Inputs", "Relations", "Outputs", "Constants"],
    )
    legend_info = [
        patches.Patch(facecolor=color, edgecolor="black", label=label)
        for color, label in zip(colors, labels)
    ]
    for layer in (
        json["Inputs"].keys()
        | json["Outputs"].keys()
        | json["Relations"].keys()
        | json["Constants"].keys()
    ):
        x1, y1 = layer_positions[layer]
        if layer in json["Inputs"].keys():
            color = "lightgreen"
            tag = f"{layer}\ndim: {json['Inputs'][layer]['dim']}\nWindow: {json['Inputs'][layer]['ntot']}"
        elif layer in json["Outputs"].keys():
            color = "orange"
            tag = layer
        elif layer in json["Constants"].keys():
            color = "lightgray"
            tag = f"{layer}\ndim: {json['Constants'][layer]['dim']}"
        else:
            color = "lightblue"
            tag = f"{json['Relations'][layer][0]}\n({layer})"
        rect = patches.Rectangle((x1, y1), 2, 1, edgecolor="black", facecolor=color)
        ax.add_patch(rect)
        ax.text(
            x1 + 1,
            y1 + 0.5,
            f"{tag}",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
        )

    # Draw arrows for dependencies
    for layer, (_, dependencies, *_) in json["Relations"].items():
        x1, y1 = layer_positions[layer]  # Get position of the current layer
        for dep in dependencies:
            if dep in layer_positions:
                x2, y2 = layer_positions[dep]  # Get position of the dependent layer
                ax.annotate(
                    "",
                    xy=(x1, y1),
                    xytext=(x2 + 2, y2 + 0.5),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1),
                )
    for out_name, rel_name in json["Outputs"].items():
        x1, y1 = layer_positions[out_name]
        x2, y2 = layer_positions[rel_name]
        ax.annotate(
            "",
            xy=(x1, y1 + 0.5),
            xytext=(x2 + 2, y2 + 0.5),
            arrowprops=dict(arrowstyle="->", color="black", lw=1),
        )
    for key, state in json["Inputs"].items():
        if "closedLoop" in state.keys():
            x1, y1 = layer_positions[key]
            x2, y2 = layer_positions[state["closedLoop"]]
            # ax.annotate("", xy=(x2+1, y2), xytext=(x2+1, y_limit), arrowprops=dict(arrowstyle="-", color='red', lw=1, linestyle='dashed'))
            ax.add_patch(
                patches.FancyArrowPatch(
                    (x2 + 1, y2),
                    (x2 + 1, -y_limit),
                    arrowstyle="-",
                    mutation_scale=15,
                    color="red",
                    linestyle="dashed",
                )
            )
            ax.add_patch(
                patches.FancyArrowPatch(
                    (x2 + 1, -y_limit),
                    (x1 - 1, -y_limit),
                    arrowstyle="-",
                    mutation_scale=15,
                    color="red",
                    linestyle="dashed",
                )
            )
            ax.add_patch(
                patches.FancyArrowPatch(
                    (x1 - 1, -y_limit),
                    (x1 - 1, y1 + 0.5),
                    arrowstyle="-",
                    mutation_scale=15,
                    color="red",
                    linestyle="dashed",
                )
            )
            ax.add_patch(
                patches.FancyArrowPatch(
                    (x1 - 1, y1 + 0.5),
                    (x1, y1 + 0.5),
                    arrowstyle="->",
                    mutation_scale=15,
                    color="red",
                    linestyle="dashed",
                )
            )
        elif "connect" in state.keys():
            x1, y1 = layer_positions[key]
            x2, y2 = layer_positions[state["connect"]]
            ax.add_patch(
                patches.FancyArrowPatch(
                    (x1, y1),
                    (x2, y2),
                    arrowstyle="->",
                    mutation_scale=15,
                    color="green",
                    linestyle="dashed",
                )
            )

    legend_info.extend(
        [
            Line2D([0], [0], color="black", lw=2, label="Dependency"),
            Line2D(
                [0], [0], color="red", lw=2, linestyle="dashed", label="Closed Loop"
            ),
            Line2D([0], [0], color="green", lw=2, linestyle="dashed", label="Connect"),
        ]
    )

    # Adjust the plot limits
    ax.set_xlim(-dx, x_limit + dx)
    ax.set_ylim(-y_limit, dy)
    ax.set_aspect("equal")
    ax.legend(handles=legend_info, loc="lower right")
    ax.axis("off")  # Hide axes

    plt.title(
        f"Neural Network Diagram - Sampling [{json['Info']['SampleTime']}]",
        fontsize=12,
        fontweight="bold",
    )
    ## Save the figure
    plt.savefig(filename, format="png", bbox_inches="tight")
    if view:
        plt.show()


def plot_graphviz_structure(
    json, filename="nnodely_graph", view=True
):  # pragma: no cover
    import shutil
    from graphviz import view
    from graphviz import Digraph

    # Check if Graphviz is installed
    if shutil.which("dot") is None:
        # raise RuntimeError(
        #     "Graphviz does not appear to be installed on your system. "
        #     "Please install it from https://graphviz.org/download/"
        # )
        log.warning(
            "Graphviz does not appear to be installed on your system. "
            "Please install it from https://graphviz.org/download/"
        )
        return

    dot = Digraph(comment="Structured Neural Network")

    # Set graph attributes for top-down layout and style
    dot.attr(rankdir="LR", size="21")
    dot.attr(
        "node", shape="box", style="filled", color="lightgray", fontname="Helvetica"
    )

    # Add metadata/info box
    if "Info" in json:
        info = json["Info"]
        info_text = "\n".join([f"{k}: {v}" for k, v in info.items()])
        dot.node(
            "INFO_BOX",
            label=f"Model Info\n{info_text}",
            shape="note",
            fillcolor="white",
            fontsize="10",
        )

    # Add input nodes
    for inp, data in json["Inputs"].items():
        dim = data["dim"]
        window = data["sw"] if "sw" in data else data["tw"]
        window_tag = "sw" if "sw" in data else "tw"
        label = f"{inp}\nDim: {dim}\nWindow({window_tag}): {window}"
        dot.node(inp, label=label, fillcolor="lightgreen")
        if "connect" in data.keys():
            dot.edge(
                data["connect"], inp, label="connect", color="blue", fontcolor="blue"
            )
        if "closedLoop" in data.keys():
            dot.edge(
                data["closedLoop"],
                inp,
                label="closedLoop",
                color="red",
                fontcolor="red",
            )

    # Add constant nodes
    if "Constants" in json:
        for const, data in json["Constants"].items():
            dim = data["dim"]
            label = f"{const}\nDim: {dim}"
            dot.node(const, label=label, fillcolor="lightgray")

    # Add relation nodes
    for name, rel in json["Relations"].items():
        op_type = rel[0]
        parents = rel[1]
        param1 = rel[2] if len(rel) > 2 else None
        param2 = rel[3] if len(rel) > 3 else None
        label = f"{name}\nType: {op_type}"
        dot.node(name, label=label, fillcolor="lightblue")
        for i in [param1, param2]:
            if isinstance(i, str):
                if i in json["Parameters"]:
                    param_dim = json["Parameters"][i]["dim"]
                    dot.node(
                        i,
                        label=f"{i}\nDim: {param_dim}",
                        shape="ellipse",
                        fillcolor="orange",
                    )
                    dot.edge(
                        i, name, label="Parameter", color="orange", fontcolor="orange"
                    )
                elif i in json["Functions"]:
                    dot.node(
                        i, label=f"{param1}", shape="ellipse", fillcolor="darkorange"
                    )
                    dot.edge(
                        i,
                        name,
                        label="function",
                        color="darkorange",
                        fontcolor="darkorange",
                    )
        for parent in parents:
            dot.edge(parent, name)

    # Add output nodes
    for out, rel in json["Outputs"].items():
        dot.node(out, fillcolor="lightcoral")
        dot.edge(rel, out)

    # Add Minimize nodes if present
    if "Minimizers" in json:
        for name, rel in json["Minimizers"].items():
            rel_a, rel_b = rel["A"], rel["B"]
            loss = rel["loss"]
            dot.node(
                name, label=f"{name}\nLoss:{loss}", shape="ellipse", fillcolor="purple"
            )
            dot.edge(rel_a, name, label="Minimize", color="purple", fontcolor="purple")
            dot.edge(rel_b, name, label="Minimize", color="purple", fontcolor="purple")

    # Add a legend as a subgraph
    # with dot.subgraph(name='cluster_legend') as legend:
    #     legend.attr(label='Legend', style='dashed')
    #     legend.node('LegendInput', 'Inputs', shape='box', fillcolor='lightgreen', style='filled')
    #     legend.node('LegendRel', 'Relation', shape='box', fillcolor='lightblue', style='filled')
    #     legend.node('LegendOutput', 'Outputs', shape='box', fillcolor='lightcoral', style='filled')
    #     # Hide the edges inside the legend box
    #     legend.attr('edge', style='invis')
    #     legend.edge('LegendInput', 'LegendRel')
    #     legend.edge('LegendRel', 'LegendOutput')

    # Render the graph
    dot.render(
        filename=filename, view=view, format="svg"
    )  # opens in default viewer and saves as SVG
