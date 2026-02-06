import copy
from pprint import pformat


from nnodely.support.utils import check
import networkx as nx

from nnodely.support.logger import logging, nnLogger
log = nnLogger(__name__, logging.WARNING)

def get_window(obj):
    return 'tw' if 'tw' in obj.dim else ('sw' if 'sw' in obj.dim else None)


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

def subjson_from_minimize(json, minimizers:str|list):
    from nnodely.basic.relation import MAIN_JSON
    json = copy.deepcopy(json)
    sub_json = copy.deepcopy(MAIN_JSON)

    if 'Minimizers' in json:
        rel_A = [json['Minimizers'][key]['A'] for key in minimizers]
        rel_B = [json['Minimizers'][key]['B'] for key in minimizers]
        relations_name = set(rel_A) | set(rel_B)
        for rel_name in relations_name:
            minimizers_json = subjson_from_relation(json, rel_name)
            sub_json = merge(sub_json, minimizers_json)
        sub_json['Minimizers'] = { key : json['Minimizers'][key] for key in minimizers }

    return sub_json

def stream_to_str(obj, type = 'Stream'):
    from nnodely.visualizer.emptyvisualizer import color, GREEN
    from pprint import pformat
    stream = f" {type} "
    stream_name = f" {obj.name} {obj.dim} "

    title = color((stream).center(80, '='), GREEN, True)
    json = color(pformat(obj.json), GREEN)
    stream = color((stream_name).center(80, '-'), GREEN, True)
    return title + '\n' + json + '\n' + stream