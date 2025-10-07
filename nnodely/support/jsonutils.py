import copy
from pprint import pformat


from nnodely.support.utils import check

from nnodely.support.logger import logging, nnLogger
log = nnLogger(__name__, logging.WARNING)

def get_window(obj):
    return 'tw' if 'tw' in obj.dim else ('sw' if 'sw' in obj.dim else None)

# Codice per comprimere le relazioni
        #print(self.json['Relations'])
        # used_rel = {string for values in self.json['Relations'].values() for string in values[1]}
        # if obj1.name not in used_rel and obj1.name in self.json['Relations'].keys() and self.json['Relations'][obj1.name][0] == add_relation_name:
        #     self.json['Relations'][self.name] = [add_relation_name, self.json['Relations'][obj1.name][1]+[obj2.name]]
        #     del self.json['Relations'][obj1.name]
        # else:
        # Devo aggiungere un operazione che rimuove un operazione di Add,Sub,Mul,Div se può essere unita ad un'altra operazione dello stesso tipo
        #
def merge(source, destination, main = True):
    if main:
        for key, value in destination["Functions"].items():
            if key in source["Functions"].keys() and 'n_input' in value.keys() and 'n_input' in source["Functions"][key].keys():
                check(value == {} or source["Functions"][key] == {} or value['n_input'] == source["Functions"][key]['n_input'],
                      TypeError,
                      f"The ParamFun {key} is present multiple times, with different number of inputs. "
                      f"The ParamFun {key} is called with {value['n_input']} parameters and with {source['Functions'][key]['n_input']} parameters.")
        for key, value in destination["Parameters"].items():
            if key in source["Parameters"].keys():
                if 'dim' in value.keys() and 'dim' in source["Parameters"][key].keys():
                    check(value['dim'] == source["Parameters"][key]['dim'],
                          TypeError,
                          f"The Parameter {key} is present multiple times, with different dimensions. "
                          f"The Parameter {key} is called with {value['dim']} dimension and with {source['Parameters'][key]['dim']} dimension.")
                window_dest = 'tw' if 'tw' in value else ('sw' if 'sw' in value else None)
                window_source = 'tw' if 'tw' in source["Parameters"][key] else ('sw' if 'sw' in source["Parameters"][key] else None)
                if window_dest is not None:
                    check(window_dest == window_source and value[window_dest] == source["Parameters"][key][window_source] ,
                          TypeError,
                          f"The Parameter {key} is present multiple times, with different window. "
                          f"The Parameter {key} is called with {window_dest}={value[window_dest]} dimension and with {window_source}={source['Parameters'][key][window_source]} dimension.")

        log.debug("Merge Source")
        log.debug("\n"+pformat(source))
        log.debug("Merge Destination")
        log.debug("\n"+pformat(destination))
        result = copy.deepcopy(destination)
    else:
        result = destination
    for key, value in source.items():
        if isinstance(value, dict):
            # get node or create one
            node = result.setdefault(key, {})
            merge(value, node, False)
        else:
            if key in result and type(result[key]) is list:
                if key == 'tw' or key == 'sw':
                    if result[key][0] > value[0]:
                        result[key][0] = value[0]
                    if result[key][1] < value[1]:
                        result[key][1] = value[1]
            else:
                result[key] = value
    if main == True:
        log.debug("Merge Result")
        log.debug("\n" + pformat(result))
    return result

def get_models_json(json):
    model_json = {}
    model_json['Parameters'] = list(json['Parameters'].keys())
    model_json['Constants'] = list(json['Constants'].keys())
    model_json['Inputs'] = list(json['Inputs'].keys())
    model_json['Outputs'] = list(json['Outputs'].keys())
    model_json['Functions'] = list(json['Functions'].keys())
    model_json['Relations'] = list(json['Relations'].keys())
    return model_json

def check_model(json):
    all_inputs = json['Inputs'].keys()
    all_outputs = json['Outputs'].keys()

    from nnodely.basic.relation import MAIN_JSON
    subjson = MAIN_JSON
    for name in all_outputs:
        subjson = merge(subjson, subjson_from_output(json, name))
    needed_inputs = subjson['Inputs'].keys()
    extenal_inputs = set(all_inputs) - set(needed_inputs)

    check(all_inputs == needed_inputs, RuntimeError,
          f'Connect or close loop operation on the inputs {list(extenal_inputs)}, that are not used in the model.')
    return json

def binary_cheks(self, obj1, obj2, name):
    from nnodely.basic.relation import Stream, toStream
    obj1,obj2 = toStream(obj1),toStream(obj2)
    check(type(obj1) is Stream,TypeError,
          f"The type of {obj1} is {type(obj1)} and is not supported for add operation.")
    check(type(obj2) is Stream,TypeError,
          f"The type of {obj2} is {type(obj2)} and is not supported for add operation.")
    window_obj1 = get_window(obj1)
    window_obj2 = get_window(obj2)
    if window_obj1 is not None and window_obj2 is not None:
        check(window_obj1==window_obj2, TypeError,
              f"For {name} the time window type must match or None but they were {window_obj1} and {window_obj2}.")
        check(obj1.dim[window_obj1] == obj2.dim[window_obj2], ValueError,
              f"For {name} the time window must match or None but they were {window_obj1}={obj1.dim[window_obj1]} and {window_obj2}={obj2.dim[window_obj2]}.")
    check(obj1.dim['dim'] == obj2.dim['dim'] or obj1.dim == {'dim':1} or obj2.dim == {'dim':1}, ValueError,
          f"For {name} the dimension of {obj1.name} = {obj1.dim} must be the same of {obj2.name} = {obj2.dim}.")
    dim = obj1.dim | obj2.dim
    dim['dim'] = max(obj1.dim['dim'], obj2.dim['dim'])
    return obj1, obj2, dim

def subjson_from_relation(json, relation):
    json = copy.deepcopy(json)
    # Get all the inputs needed to compute a specific relation from the json graph
    inputs = set()
    relations = set()
    constants = set()
    parameters = set()
    functions = set()

    def search(rel):
        if rel in json['Inputs']:  # Found an input
            inputs.add(rel)
            if rel in json['Inputs']:
                if 'connect' in json['Inputs'][rel] and json['Inputs'][rel]['local'] == 1:
                    search(json['Inputs'][rel]['connect'])
                if 'closed_loop' in json['Inputs'][rel] and json['Inputs'][rel]['local'] == 1:
                    search(json['Inputs'][rel]['closed_loop'])
                # if 'init' in json['Inputs'][rel]:
                #     search(json['Inputs'][rel]['init'])
        elif rel in json['Constants']:  # Found a constant or parameter
            constants.add(rel)
        elif rel in json['Parameters']:
            parameters.add(rel)
        elif rel in json['Functions']:
            functions.add(rel)
            if 'params_and_consts' in json['Functions'][rel]:
                for sub_rel in json['Functions'][rel]['params_and_consts']:
                    search(sub_rel)
        elif rel in json['Relations']:  # Another relation
            relations.add(rel)
            for sub_rel in json['Relations'][rel][1]:
                search(sub_rel)
            for sub_rel in json['Relations'][rel][2:]:
                if json['Relations'][rel][0] in ('Fir', 'Linear'):
                    search(sub_rel)
                if json['Relations'][rel][0] in ('Fuzzify'):
                    search(sub_rel)
                if json['Relations'][rel][0] in ('ParamFun'):
                    search(sub_rel)

    search(relation)
    from nnodely.basic.relation import MAIN_JSON
    sub_json = copy.deepcopy(MAIN_JSON)
    sub_json['Relations'] = {key: value for key, value in json['Relations'].items() if key in relations}
    sub_json['Inputs'] = {key: value for key, value in json['Inputs'].items() if key in inputs}
    sub_json['Constants'] = {key: value for key, value in json['Constants'].items() if key in constants}
    sub_json['Parameters'] = {key: value for key, value in json['Parameters'].items() if key in parameters}
    sub_json['Functions'] = {key: value for key, value in json['Functions'].items() if key in functions}
    sub_json['Outputs'] = {}
    sub_json['Info'] = {}
    return sub_json


def subjson_from_output(json, outputs:str|list):
    json = copy.deepcopy(json)
    from nnodely.basic.relation import MAIN_JSON
    sub_json = copy.deepcopy(MAIN_JSON)
    if type(outputs) is str:
        outputs = [outputs]
    for output in outputs:
        sub_json = merge(sub_json, subjson_from_relation(json,json['Outputs'][output]))
        sub_json['Outputs'][output] = json['Outputs'][output]
    return sub_json

def subjson_from_model(json, models:str|list):
    from nnodely.basic.relation import MAIN_JSON
    json = copy.deepcopy(json)
    sub_json = copy.deepcopy(MAIN_JSON)
    models_names = set([json['Models']]) if type(json['Models']) is str else set(json['Models'].keys())
    if type(models) is str or len(models) == 1:
        if len(models) == 1:
            models = models[0]
        check(models in models_names, AttributeError, f"Model [{models}] not found!")
        if type(json['Models']) is str:
            outputs = set(json['Outputs'].keys())
        else:
            outputs = set(json['Models'][models]['Outputs'])
        sub_json['Models'] = models
    else:
        outputs = set()
        sub_json['Models'] = {}
        for model in models:
            check(model in models_names, AttributeError, f"Model [{model}] not found!")
            outputs |= set(json['Models'][model]['Outputs'])
            sub_json['Models'][model] = {key: value for key, value in json['Models'][model].items()}

    # Remove the extern connections not keys in the graph
    final_json = merge(sub_json, subjson_from_output(json, outputs))
    for key, value in final_json['Inputs'].items():
        if 'connect' in value and (value['local'] == 0 and value['connect'] not in final_json['Relations'].keys()):
            del final_json['Inputs'][key]['connect']
            del final_json['Inputs'][key]['local']
            log.warning(f'The input {key} is "connect" outside the model connection removed for subjson')
        if 'closedLoop' in value and (value['local'] == 0 and value['closedLoop'] not in final_json['Relations'].keys()):
            del final_json['Inputs'][key]['closedLoop']
            del final_json['Inputs'][key]['local']
            log.warning(f'The input {key} is "closedLoop" outside the model connection removed for subjson')
    return final_json

def stream_to_str(obj, type = 'Stream'):
    from nnodely.visualizer.emptyvisualizer import color, GREEN
    from pprint import pformat
    stream = f" {type} "
    stream_name = f" {obj.name} {obj.dim} "

    title = color((stream).center(80, '='), GREEN, True)
    json = color(pformat(obj.json), GREEN)
    stream = color((stream_name).center(80, '-'), GREEN, True)
    return title + '\n' + json + '\n' + stream

def plot_structure(json, filename='nnodely_graph', library='matplotlib', view=True):
        #json = self.modely.json if json is None else json
        # if json is None:
        #     raise ValueError("No JSON model definition provided. Please provide a valid JSON model definition.")
        if library not in ['matplotlib', 'graphviz']:
            raise ValueError("Invalid library specified. Use 'matplotlib' or 'graphviz'.")
        if library == 'matplotlib':
            plot_matplotlib_structure(json, filename, view=view)
        elif library == 'graphviz':
            plot_graphviz_structure(json, filename, view=view)

def plot_matplotlib_structure(json, filename='nnodely_graph', view=True):
    import matplotlib.pyplot as plt
    from matplotlib import patches
    from matplotlib.lines import Line2D
    layer_positions = {}
    x, y = 0, 0  # Initial position
    dy, dx = 1.5, 2.5  # Spacing

    ## Layer Inputs: 
    for input_name, input_type in json['Inputs'].items():
        layer_positions[input_name] = (x, y)
        y -= dy
    for constant_name in json['Constants'].keys():
        layer_positions[constant_name] = (x, y)
        y -= dy
    y_limit = abs(y)

    # Layers Relations:
    available_inputs = list(json['Inputs'].keys() | json['Constants'].keys())
    available_outputs = list(set(json['Outputs'].values()))
    while available_outputs:
        x += dx
        y = 0
        inputs_to_add, outputs_to_remove = [], []
        for relation_name, (relation_type, dependencies, *_) in json['Relations'].items():
            if all(dep in available_inputs for dep in dependencies) and (relation_name not in available_inputs):
                inputs_to_add.append(relation_name)
                if relation_name in available_outputs:
                    outputs_to_remove.append(relation_name)
                layer_positions[relation_name] = (x, y)
                y -= dy
        y_limit = max(y_limit, abs(y))
        available_inputs.extend(inputs_to_add)
        available_outputs = [out for out in available_outputs if out not in outputs_to_remove]

    ## Layer Outputs: 
    x += dx
    y = 0
    for idx, output_name in enumerate(json['Outputs'].keys()):
        layer_positions[output_name] = (x, y)
        y -= dy  # Move down for the next input
    x_limit = abs(x)
    y_limit = max(y_limit, abs(y))

    # Create the plot
    fig, ax = plt.subplots(figsize=(x_limit, y_limit))
    #fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    # Plot rectangles for each layer
    colors, labels = ['lightgreen', 'lightblue', 'orange', 'lightgray'], ['Inputs', 'Relations', 'Outputs', 'Constants']
    legend_info = [patches.Patch(facecolor=color, edgecolor='black', label=label) for color, label in zip(colors, labels)]
    for layer in (json['Inputs'].keys() | json['Outputs'].keys() | json['Relations'].keys() | json['Constants'].keys()):
        x1, y1 = layer_positions[layer]
        if layer in json['Inputs'].keys():
            color = 'lightgreen'
            tag = f'{layer}\ndim: {json["Inputs"][layer]["dim"]}\nWindow: {json["Inputs"][layer]["ntot"]}'
        elif layer in json['Outputs'].keys():
            color = 'orange'
            tag = layer
        elif layer in json['Constants'].keys():
            color = 'lightgray'
            tag = f'{layer}\ndim: {json["Constants"][layer]["dim"]}'
        else:
            color = 'lightblue'
            tag = f'{json["Relations"][layer][0]}\n({layer})'
        rect = patches.Rectangle((x1, y1), 2, 1, edgecolor='black', facecolor=color)
        ax.add_patch(rect)
        ax.text(x1 + 1, y1 + 0.5, f"{tag}", ha='center', va='center', fontsize=8, fontweight='bold')

    # Draw arrows for dependencies
    for layer, (_, dependencies, *_) in json['Relations'].items():
        x1, y1 = layer_positions[layer]  # Get position of the current layer
        for dep in dependencies:
            if dep in layer_positions:
                x2, y2 = layer_positions[dep]  # Get position of the dependent layer
                ax.annotate("", xy=(x1, y1), xytext=(x2 + 2, y2 + 0.5), arrowprops=dict(arrowstyle="->", color='black', lw=1))
    for out_name, rel_name in json['Outputs'].items():
        x1, y1 = layer_positions[out_name]
        x2, y2 = layer_positions[rel_name]
        ax.annotate("", xy=(x1, y1 + 0.5), xytext=(x2 + 2, y2 + 0.5),
                    arrowprops=dict(arrowstyle="->", color='black', lw=1))
    for key, state in json['Inputs'].items():
        if 'closedLoop' in state.keys():
            x1, y1 = layer_positions[key]
            x2, y2 = layer_positions[state['closedLoop']]
            #ax.annotate("", xy=(x2+1, y2), xytext=(x2+1, y_limit), arrowprops=dict(arrowstyle="-", color='red', lw=1, linestyle='dashed'))
            ax.add_patch(patches.FancyArrowPatch((x2+1, y2), (x2+1, -y_limit), arrowstyle='-', mutation_scale=15, color='red', linestyle='dashed'))
            ax.add_patch(patches.FancyArrowPatch((x2+1, -y_limit), (x1-1, -y_limit), arrowstyle='-', mutation_scale=15, color='red', linestyle='dashed'))
            ax.add_patch(patches.FancyArrowPatch((x1-1, -y_limit), (x1-1, y1+0.5), arrowstyle='-', mutation_scale=15, color='red', linestyle='dashed'))
            ax.add_patch(patches.FancyArrowPatch((x1-1, y1+0.5), (x1, y1+0.5), arrowstyle='->', mutation_scale=15, color='red', linestyle='dashed'))
        elif 'connect' in state.keys():
            x1, y1 = layer_positions[key]
            x2, y2 = layer_positions[state['connect']]
            ax.add_patch(patches.FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', mutation_scale=15, color='green', linestyle='dashed'))
        
    legend_info.extend([Line2D([0], [0], color='black', lw=2, label='Dependency'),
                        Line2D([0], [0], color='red', lw=2, linestyle='dashed', label='Closed Loop'),
                        Line2D([0], [0], color='green', lw=2, linestyle='dashed', label='Connect')])

    # Adjust the plot limits
    ax.set_xlim(-dx, x_limit+dx)
    ax.set_ylim(-y_limit, dy)
    ax.set_aspect('equal')
    ax.legend(handles=legend_info, loc='lower right')
    ax.axis('off')  # Hide axes

    plt.title(f"Neural Network Diagram - Sampling [{json['Info']['SampleTime']}]", fontsize=12, fontweight='bold')
    ## Save the figure
    plt.savefig(filename, format="png", bbox_inches='tight')
    if view:
        plt.show()

def plot_graphviz_structure(json, filename='nnodely_graph', view=True): # pragma: no cover
    import shutil
    from graphviz import view
    from graphviz import Digraph

    # Check if Graphviz is installed
    if shutil.which('dot') is None:
        # raise RuntimeError(
        #     "Graphviz does not appear to be installed on your system. "
        #     "Please install it from https://graphviz.org/download/"
        # )
        log.warning(
            "Graphviz does not appear to be installed on your system. "
            "Please install it from https://graphviz.org/download/"
        )
        return
    
    dot = Digraph(comment='Structured Neural Network')

    # Set graph attributes for top-down layout and style
    dot.attr(rankdir='LR', size='21')  
    dot.attr('node', shape='box', style='filled', color='lightgray', fontname='Helvetica')

    # Add metadata/info box
    if 'Info' in json:
        info = json['Info']
        info_text = '\n'.join([f"{k}: {v}" for k, v in info.items()])
        dot.node('INFO_BOX', label=f"Model Info\n{info_text}", shape='note', fillcolor='white', fontsize='10')

    # Add input nodes
    for inp, data in json['Inputs'].items():
        dim = data['dim']
        window = data['sw']
        label = f"{inp}\nDim: {dim}\nWindow: {window}"
        dot.node(inp, label=label, fillcolor='lightgreen')
        if 'connect' in data.keys():
            dot.edge(data['connect'], inp, label='connect', color='blue', fontcolor='blue')
        if 'closedLoop' in data.keys():
            dot.edge(data['closedLoop'], inp, label='closedLoop', color='red', fontcolor='red')

    # Add constant nodes
    if 'Constants' in json:
        for const, data in json['Constants'].items():
            dim = data['dim']
            label = f"{const}\nDim: {dim}"
            dot.node(const, label=label, fillcolor='lightgray')

    # Add relation nodes
    for name, rel in json['Relations'].items():
        op_type = rel[0]
        parents = rel[1]
        param1 = rel[2] if len(rel) > 2 else None
        param2 = rel[3] if len(rel) > 3 else None
        label = f"{name}\nType: {op_type}"
        dot.node(name, label=label, fillcolor='lightblue')
        for i in [param1,param2]:
            if isinstance(i, str):
                if i in json['Parameters']:
                    param_dim = json['Parameters'][i]['dim']
                    dot.node(i, label=f"{i}\nDim: {param_dim}", shape='ellipse', fillcolor='orange')
                    dot.edge(i, name, label='Parameter', color='orange', fontcolor='orange')
                elif i in json['Functions']:
                    dot.node(i, label=f"{param1}", shape='ellipse', fillcolor='darkorange')
                    dot.edge(i, name, label='function', color='darkorange', fontcolor='darkorange')
        for parent in parents:
            dot.edge(parent, name)

    # Add output nodes
    for out, rel in json['Outputs'].items():
        dot.node(out, fillcolor='lightcoral')
        dot.edge(rel, out)

    # Add Minimize nodes if present
    if 'Minimizers' in json:
        for name, rel in json['Minimizers'].items():
            rel_a, rel_b = rel['A'], rel['B']
            loss = rel['loss']
            dot.node(name, label=f"{name}\nLoss:{loss}", shape='ellipse', fillcolor='purple')
            dot.edge(rel_a, name, label='Minimize', color='purple', fontcolor='purple')
            dot.edge(rel_b, name, label='Minimize', color='purple', fontcolor='purple')

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
    dot.render(filename=filename, view=view, format='svg')  # opens in default viewer and saves as SVG