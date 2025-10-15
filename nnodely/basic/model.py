import torch
import copy

import torch.nn as nn
import numpy as np

from itertools import product

from nnodely.support.utils import TORCH_DTYPE
from nnodely.support import initializer



@torch.fx.wrap
def update_state(data_in, rel):
    #virtual = torch.roll(data_in, shifts=-1, dims=1)
    max_dim = min(rel.size(1), data_in.size(1))
    data_out = data_in.clone()
    data_out[:, -max_dim:, :] = rel[:, -max_dim:, :]
    return data_out

class Model(nn.Module):
    def __init__(self, model_def):
        super(Model, self).__init__()
        model_def = copy.deepcopy(model_def)

        self.states = {key: value for key, value in model_def['Inputs'].items() if ('closedLoop' in value.keys() or 'connect' in value.keys())}

        self.inputs = model_def['Inputs']
        self.outputs = model_def['Outputs']
        self.relations = model_def['Relations']
        self.params = model_def['Parameters']
        self.constants = model_def['Constants']
        self.sample_time = model_def['Info']['SampleTime']
        self.functions = model_def['Functions']

        self.minimizers = model_def['Minimizers'] if 'Minimizers' in model_def else {}
        self.minimizers_keys = [self.minimizers[key]['A'] for key in self.minimizers] + [self.minimizers[key]['B'] for key in self.minimizers]

        self.input_ns_backward = {key:value['ns'][0] for key, value in model_def['Inputs'].items()}
        self.input_n_samples = {key:value['ntot'] for key, value in model_def['Inputs'].items()}

        ## Build the network
        self.all_parameters = {}
        self.all_constants = {}
        self.relation_forward = {}
        self.relation_inputs = {}
        self.closed_loop_update = {}
        self.connect_update = {}

        ## Update the connect_update and closed_loop_update
        self.update()

        ## Define the correct slicing
        for _, items in self.relations.items():
            if items[0] == 'SamplePart':
                if items[1][0] in self.inputs.keys():
                    items[3][0] = self.input_ns_backward[items[1][0]] + items[3][0]
                    items[3][1] = self.input_ns_backward[items[1][0]] + items[3][1]
                    if len(items) > 4: ## Offset
                        items[4] = self.input_ns_backward[items[1][0]] + items[4]
            if items[0] == 'TimePart':
                if items[1][0] in self.inputs.keys():
                    items[3][0] = self.input_ns_backward[items[1][0]] + round(items[3][0]/self.sample_time)
                    items[3][1] = self.input_ns_backward[items[1][0]] + round(items[3][1]/self.sample_time)
                    if len(items) > 4: ## Offset
                        items[4] = self.input_ns_backward[items[1][0]] + round(items[4]/self.sample_time)
                else:
                    items[3][0] = round(items[3][0]/self.sample_time)
                    items[3][1] = round(items[3][1]/self.sample_time)
                    if len(items) > 4: ## Offset
                        items[4] = round(items[4]/self.sample_time)

        ## Create all the parameters
        for name, param_data in self.params.items():
            window = 'tw' if 'tw' in param_data.keys() else ('sw' if 'sw' in param_data.keys() else None)
            aux_sample_time = self.sample_time if 'tw' == window else 1
            sample_window = round(param_data[window] / aux_sample_time) if window else None
            if sample_window is None:
                param_size = tuple(param_data['dim']) if type(param_data['dim']) is list else (param_data['dim'],)
            else:
                param_size = (sample_window,)+tuple(param_data['dim']) if type(param_data['dim']) is list else (sample_window, param_data['dim'])
            if 'values' in param_data:
                self.all_parameters[name] = nn.Parameter(torch.tensor(param_data['values'], dtype=TORCH_DTYPE), requires_grad=True)
            # TODO clean code
            elif 'init_fun' in param_data:
                if 'code' in param_data['init_fun'].keys():
                    exec(param_data['init_fun']['code'], globals())
                    function_to_call = globals()[param_data['init_fun']['name']]
                else:
                    function_to_call = getattr(initializer, param_data['init_fun']['name'])
                values = np.zeros(param_size)
                for indexes in product(*(range(v) for v in param_size)):
                    if 'params' in param_data['init_fun']:
                        values[indexes] = function_to_call(indexes, param_size, param_data['init_fun']['params'])
                    else:
                        values[indexes] = function_to_call(indexes, param_size)
                self.all_parameters[name] = nn.Parameter(torch.tensor(values.tolist(), dtype=TORCH_DTYPE), requires_grad=True)
            else:
                self.all_parameters[name] = nn.Parameter(torch.rand(size=param_size, dtype=TORCH_DTYPE), requires_grad=True)

        ## Create all the constants
        for name, param_data in self.constants.items():
            self.all_constants[name] = nn.Parameter(torch.tensor(param_data['values'], dtype=TORCH_DTYPE), requires_grad=False)
        all_params_and_consts = self.all_parameters | self.all_constants

        ## Create all the relations
        for relation, inputs in self.relations.items():
            ## Take the relation name and the inputs needed to solve the relation
            rel_name, input_var = inputs[0], inputs[1]
            ## Create All the Relations
            func = getattr(self,rel_name)
            if func:
                layer_inputs = []
                for item in inputs[2:]:
                    if item in list(self.params.keys()): ## the relation takes parameters
                        layer_inputs.append(self.all_parameters[item])
                    elif item in list(self.constants.keys()): ## the relation takes a constant
                        layer_inputs.append(self.all_constants[item])
                    elif item in list(self.functions.keys()): ## the relation takes a custom function
                        layer_inputs.append(self.functions[item])
                        if 'params_and_consts' in self.functions[item].keys() and len(self.functions[item]['params_and_consts']) >= 0: ## Parametric function that takes parameters
                            layer_inputs.append([all_params_and_consts[par] for par in self.functions[item]['params_and_consts']])
                        if 'map_over_dim' in self.functions[item].keys():
                            layer_inputs.append(self.functions[item]['map_over_dim'])
                    else:
                        layer_inputs.append(item)

                if rel_name == 'SamplePart':
                    if layer_inputs[0] == -1:
                        layer_inputs[0] = self.input_n_samples[input_var[0]]
                elif rel_name == 'TimePart':
                    if layer_inputs[0] == -1:
                        layer_inputs[0] = self.input_n_samples[input_var[0]]
                    else:
                        layer_inputs[0] = round(layer_inputs[0] / self.sample_time)
                ## Initialize the relation
                self.relation_forward[relation] = func(*layer_inputs)
                ## Save the inputs needed for the relative relation
                self.relation_inputs[relation] = input_var

        ## Add the gradient to all the relations and parameters that requires it
        self.relation_forward = nn.ParameterDict(self.relation_forward)
        self.all_constants = nn.ParameterDict(self.all_constants)
        self.all_parameters = nn.ParameterDict(self.all_parameters)
        ## list of network outputs
        self.network_output_predictions = set(self.outputs.values())
        ## list of network minimization outputs
        self.network_output_minimizers = []
        for _,value in self.minimizers.items():
            self.network_output_minimizers.append(self.outputs[value['A']]) if value['A'] in self.outputs.keys() else self.network_output_minimizers.append(value['A'])
            self.network_output_minimizers.append(self.outputs[value['B']]) if value['B'] in self.outputs.keys() else self.network_output_minimizers.append(value['B'])
        self.network_output_minimizers = set(self.network_output_minimizers)
        ## list of all the network Outputs
        self.network_outputs = self.network_output_predictions.union(self.network_output_minimizers)

    def forward(self, kwargs):
        result_dict = {}

        ## Initially i have only the inputs from the dataset, the parameters, and the constants
        available_inputs = [key for key in self.inputs.keys() if key not in self.connect_update.keys()]  ## remove connected inputs
        available_keys = set(available_inputs + list(self.all_parameters.keys()) + list(self.all_constants.keys()))

        ## Forward pass through the relations
        while not self.network_outputs.issubset(available_keys): ## i need to climb the relation tree until i get all the outputs
            for relation in self.relations.keys():
                ## if i have all the variables i can calculate the relation
                if set(self.relation_inputs[relation]).issubset(available_keys) and (relation not in available_keys):
                    ## Collect all the necessary inputs for the relation
                    layer_inputs = []
                    for key in self.relation_inputs[relation]:
                        if key in self.all_constants.keys(): ## relation that takes a constant
                            layer_inputs.append(self.all_constants[key])
                        elif key in available_inputs:  ## relation that takes inputs
                            layer_inputs.append(kwargs[key])
                        elif key in self.all_parameters.keys(): ## relation that takes parameters
                            layer_inputs.append(self.all_parameters[key])
                        else: ## relation than takes another relation or a connect variable
                            layer_inputs.append(result_dict[key])

                    ## Execute the current relation
                    result_dict[relation] = self.relation_forward[relation](*layer_inputs)
                    available_keys.add(relation)

                    ## Check if the relation is inside the connect
                    for connect_input, connect_rel in self.connect_update.items():
                        if relation == connect_rel:
                            result_dict[connect_input] = update_state(kwargs[connect_input], result_dict[relation])
                            available_keys.add(connect_input)

        ## Return a dictionary with all the connected inputs
        connect_update_dict = {key: result_dict[key] for key in self.connect_update.keys()}
        ## Return a dictionary with all the relations that updates the state variables
        closed_loop_update_dict = {key: result_dict[value] for key, value in self.closed_loop_update.items()}
        ## Return a dictionary with all the outputs final values
        output_dict = {key: result_dict[value] for key, value in self.outputs.items()}
        ## Return a dictionary with the minimization relations
        minimize_dict = {}
        for key in self.minimizers_keys:
            minimize_dict[key] = result_dict[self.outputs[key]] if key in self.outputs.keys() else result_dict[key]
        return output_dict, minimize_dict, closed_loop_update_dict, connect_update_dict

    def update(self, *, closed_loop = {}, connect = {}, disconnect = False):
        self.closed_loop_update = {}
        self.connect_update = {}

        if disconnect:
            return

        for key, state in self.states.items():
            if 'connect' in state.keys():
                self.connect_update[key] = state['connect']
            elif 'closedLoop' in state.keys():
                self.closed_loop_update[key] = state['closedLoop']

        # Get relation from outputs
        for connect_in, connect_rel in connect.items():
            set_relation = self.outputs[connect_rel] if connect_rel in self.outputs.keys() else connect_rel
            self.connect_update[connect_in] = set_relation
        for close_in, close_rel in closed_loop.items():
            set_relation = self.outputs[close_rel] if close_rel in self.outputs.keys() else close_rel
            self.closed_loop_update[close_in] = set_relation
