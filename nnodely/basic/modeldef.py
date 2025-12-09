import copy

import numpy as np

from nnodely.support.utils import check, check_and_get_list
from nnodely.support.jsonutils import merge, subjson_from_model, subjson_from_minimize, check_model, get_models_json
from nnodely.basic.relation import MAIN_JSON, Stream, check_names
from nnodely.layers.output import Output
from nnodely.layers.input import Input

from nnodely.support.logger import logging, nnLogger
log = nnLogger(__name__, logging.INFO)


class ModelDef:
    def __init__(self, model_def = MAIN_JSON):
        # Models definition
        self.__json_base = copy.deepcopy(model_def)

        # Initialize the model definition
        self.__json = copy.deepcopy(self.__json_base)
        if "SampleTime" in self.__json['Info']:
            self.__sample_time = self.__json['Info']["SampleTime"]
        else:
            self.__sample_time = None

    def __contains__(self, key):
        return key in self.__json

    def __getitem__(self, key):
        if key in self.__json:
            return self.__json[key]
        else:
            return None

    def __setitem__(self, key, value):
        self.__json[key] = value

    def __rebuild_json(self, models_names, minimizers):
        models_json = subjson_from_model(self.__json, list(models_names))
        if 'Minimizers' in self.__json and len(minimizers) > 0:
            minimizers_json = subjson_from_minimize(self.__json, list(minimizers))
            models_json = merge(models_json, minimizers_json)
        return copy.deepcopy(models_json)

    def recurrentInputs(self):
        return {key:value for key, value in self.__json['Inputs'].items() if ('closedLoop' in value.keys() or 'connect' in value.keys())}

    def getJson(self, models:list|str|None = None) -> dict:
        if models is None:
            return copy.deepcopy(self.__json)
        else:
            json = subjson_from_model(self.__json, models)
            check_model(json)
            return copy.deepcopy(json)

    def getSampleTime(self):
        check(self.__sample_time is not None, AttributeError, "Sample time is not defined the model is not neuralized!")
        return self.__sample_time

    def isDefined(self):
        return self.__json is not None

    def addConnection(self, stream_out:str|Output|Stream, input_in:str|Input, type:str, local:bool = False):
        outputs = self.__json['Outputs']

        if isinstance(stream_out, (Output, Stream)):
            stream_name = outputs[stream_out.name] if stream_out.name in outputs.keys() else stream_out.name
        else:
            output_name = check_and_get_list(stream_out, set(outputs.keys()),
                                             lambda name: f"The name {name} is not part of the available Outputs")[0]
            stream_name = outputs[output_name]

        if isinstance(input_in, Input):
            input_name = input_in.name
        else:
            input_name = input_in #TODO Add tests

        input_name = check_and_get_list(input_name, set(self.__json['Inputs'].keys()),
                                       lambda name: f"The name {name} is not part of the available Inputs")[0]
        stream_name = check_and_get_list(stream_name, set(self.__json['Relations'].keys()),
                                        lambda name: f"The name {name} is not part of the available Relations")[0]
        self.__json['Inputs'][input_name][type] = stream_name
        self.__json['Inputs'][input_name]['local'] = int(local)

    def removeConnection(self, name_list:str|list[str]):
        name_list = check_and_get_list(name_list, set(self.__json['Inputs'].keys()), lambda name: f"The name {name} is not part of the available Inputs")
        for input_in in name_list:
            if 'closedLoop' in self.__json['Inputs'][input_in].keys():
                del self.__json['Inputs'][input_in]['closedLoop']
                del self.__json['Inputs'][input_in]['local']
            elif 'connect' in self.__json['Inputs'][input_in].keys():
                del self.__json['Inputs'][input_in]['connect']
                del self.__json['Inputs'][input_in]['local']
            else:
                raise ValueError(f"The input '{input_in}' has no connection or closed loop defined")

    def addModel(self, name:str, stream_list):
        if isinstance(stream_list, Output):
            stream_list = [stream_list]

        json = MAIN_JSON
        for stream in stream_list:
            json = merge(json, stream.json)
        check_model(json)

        if 'Models' not in self.__json:
            self.__json = merge(self.__json, json)
            self.__json['Models'] = name
        else:
            models_names = set((self.__json['Models'],)) if type(self.__json['Models']) is str else set(self.__json['Models'].keys())
            check_names(name, models_names, 'Models')
            if type(self.__json['Models']) is str:
                self.__json['Models'] = {self.__json['Models']: get_models_json(self.__json)}
            self.__json = merge(self.__json, json)
            self.__json['Models'][name] = get_models_json(json)

    def removeModel(self, name_list):
        if 'Models' not in self.__json:
            raise ValueError("No Models are defined")
        models_names = {self.__json['Models']} if type(self.__json['Models']) is str else set(self.__json['Models'].keys())
        name_list = check_and_get_list(name_list, models_names, lambda name: f"The name {name} is not part of the available models")
        models_names -= set(name_list)
        minimizers = set(self.__json['Minimizers'].keys()) if 'Minimizers' in self.__json else None
        self.__json = self.__rebuild_json(models_names, minimizers)

    def addMinimize(self, name, streamA, streamB, loss_function='mse'):
        if 'Minimizers' not in self.__json:
            self.__json['Minimizers'] = {}
        check_names(name, set(self.__json['Minimizers'].keys()), 'Minimizers')

        if isinstance(streamA, str):
            streamA_name = streamA
        else:
            check(isinstance(streamA, (Output, Stream)), TypeError, 'streamA must be an instance of Output or Stream')
            streamA_name = streamA.json['Outputs'][streamA.name] if isinstance(streamA, Output) else streamA.name
            self.__json = merge(self.__json, streamA.json)

        if isinstance(streamB, str):
            streamB_name = streamB
        else:
            check(isinstance(streamB, (Output, Stream)), TypeError, 'streamA must be an instance of Output or Stream')
            streamB_name = streamB.json['Outputs'][streamB.name] if isinstance(streamB, Output) else streamB.name
            self.__json = merge(self.__json, streamB.json)
        #check(streamA.dim == streamB.dim, ValueError, f'Dimension of streamA={streamA.dim} and streamB={streamB.dim} are not equal.')

        self.__json['Minimizers'][name] = {}
        self.__json['Minimizers'][name]['A'] = streamA_name
        self.__json['Minimizers'][name]['B'] = streamB_name
        self.__json['Minimizers'][name]['loss'] = loss_function

    def removeMinimize(self, name_list):
        if 'Minimizers' not in self.__json:
            raise ValueError("No Minimizers are defined")
        name_list = check_and_get_list(name_list, self.__json['Minimizers'].keys(), lambda name: f"The name {name} is not part of the available minimizers")
        models_names = {self.__json['Models']} if type(self.__json['Models']) is str else set(self.__json['Models'].keys())
        remaining_minimizers = set(self.__json['Minimizers'].keys()) - set(name_list) if 'Minimizers' in self.__json else None
        self.__json = self.__rebuild_json(models_names, remaining_minimizers)

    def setBuildWindow(self, sample_time = None):
        check(self.__json is not None, RuntimeError, "No model is defined!")
        if sample_time is not None:
            check(sample_time > 0, RuntimeError, 'Sample time must be strictly positive!')
            self.__sample_time = sample_time
        else:
            if self.__sample_time is None:
                self.__sample_time = 1

        self.__json['Info'] = {"SampleTime": self.__sample_time}
        if 'SampleTime' in self.__json['Constants']:
            self.__json['Constants']['SampleTime'] = {'dim': 1, 'values': self.__sample_time}

        check(self.__json['Inputs'] != {}, RuntimeError, "No model is defined!")
        json_inputs = self.__json['Inputs']

        input_ns_backward, input_ns_forward = {}, {}
        for key, value in json_inputs.items():
            if 'sw' not in value and 'tw' not in value:
                assert False, f"Input '{key}' has no time window or sample window"
            if 'sw' not in value and self.__sample_time is not None:
                ## check if value['tw'] is a multiple of sample_time
                absolute_tw = abs(value['tw'][0]) + abs(value['tw'][1])
                check(round(absolute_tw % self.__sample_time) == 0, ValueError,
                      f"Time window of input '{key}' is not a multiple of sample time. This network cannot be neuralized")
                input_ns_backward[key] = round(-value['tw'][0] / self.__sample_time)
                input_ns_forward[key] = round(value['tw'][1] / self.__sample_time)
            elif self.__sample_time is not None:
                if 'tw' in value:
                    input_ns_backward[key] = max(round(-value['tw'][0] / self.__sample_time), -value['sw'][0])
                    input_ns_forward[key] = max(round(value['tw'][1] / self.__sample_time), value['sw'][1])
                else:
                    input_ns_backward[key] = -value['sw'][0]
                    input_ns_forward[key] =  value['sw'][1]
            else:
                check(value['tw'] == [0,0], RuntimeError, f"Sample time is not defined for input '{key}'")
                input_ns_backward[key] = -value['sw'][0]
                input_ns_forward[key] = value['sw'][1]
            value['ns'] = [input_ns_backward[key], input_ns_forward[key]]
            value['ntot'] = sum(value['ns'])

        self.__json['Info']['ns'] = [max(input_ns_backward.values()), max(input_ns_forward.values())]
        self.__json['Info']['ntot'] = sum(self.__json['Info']['ns'])
        if self.__json['Info']['ns'][0] < 0:
            log.warning(
                f"The input is only in the far past the max_samples_backward is: {self.__json['Info']['ns'][0]}")
        if self.__json['Info']['ns'][1] < 0:
            log.warning(
                f"The input is only in the far future the max_sample_forward is: {self.__json['Info']['ns'][1]}")

        for k, v in (self.__json['Parameters'] | self.__json['Constants']).items():
            if 'values' in v:
                window = 'tw' if 'tw' in v.keys() else ('sw' if 'sw' in v.keys() else None)
                if window == 'tw':
                    check(np.array(v['values']).shape[0] == v['tw'] / self.__sample_time, ValueError,
                      f"{k} has a different number of values for this sample time.")
                if v['values'] == "SampleTime":
                    v['values'] = self.__sample_time

    def updateParameters(self, model = None, *, clear_model = False):
        if clear_model:
            for key in self.__json['Parameters'].keys():
                if 'init_values' in self.__json['Parameters'][key]:
                    self.__json['Parameters'][key]['values'] = self.__json['Parameters'][key]['init_values']
                elif 'values' in self.__json['Parameters'][key]:
                    del self.__json['Parameters'][key]['values']
        elif model is not None:
            for key in self.__json['Parameters'].keys():
                if key in model.all_parameters:
                    self.__json['Parameters'][key]['values'] = model.all_parameters[key].tolist()

