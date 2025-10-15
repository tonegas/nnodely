import copy, torch

import numpy as np

from nnodely.operators.network import Network

from nnodely.basic.modeldef import ModelDef
from nnodely.basic.model import Model
from nnodely.support.utils import check, TORCH_DTYPE, NP_DTYPE, enforce_types
from nnodely.support.mathutils import argmax_dict, argmin_dict
from nnodely.basic.relation import Stream
from nnodely.layers.input import Input
from nnodely.layers.output import Output

from nnodely.support.logger import logging, nnLogger
log = nnLogger(__name__, logging.WARNING)

class Composer(Network):
    @enforce_types
    def __init__(self):
        check(type(self) is not Composer, TypeError, "Composer class cannot be instantiated directly")
        super().__init__()

    def __addInfo(self) -> None:
        total_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        self._model_def['Info']['num_parameters'] = total_params
        from nnodely import __version__
        self._model_def['Info']['nnodely_version'] = __version__

    @enforce_types
    def addModel(self, name:str, stream_list:list|Output) -> None:
        """
        Adds a new model with the given name along with a list of Outputs.

        Parameters
        ----------
        name : str
            The name of the model.
        stream_list : list of Stream
            The list of Outputs stream in the model.

        Example
        -------
        Example usage:
            >>> model = Modely()
            >>> x = Input('x')
            >>> out = Output('out', Fir(x.last()))
            >>> model.addModel('example_model', [out])
        """
        self._model_def.addModel(name, stream_list)

    @enforce_types
    def removeModel(self, name_list:list|str) -> None:
        """
        Removes models with the given list of names.

        Parameters
        ----------
        name_list : list of str
            The list of model names to remove.

        Example
        -------
        Example usage:
            >>> model.removeModel(['sub_model1', 'sub_model2'])
        """
        self._model_def.removeModel(name_list)

    @enforce_types
    def addConnect(self, stream_out:str|Output|Stream, input_in:str|Input, *, local:bool=False) -> None:
        """
        Adds a connection from a relation stream to an input.

        Parameters
        ----------
        stream_out : Stream
            The relation stream to connect from.
        input_in : Input or list of inputs
            The input or list of input to connect to.

        Examples
        --------
        .. image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/tonegas/nnodely/blob/main/examples/states.ipynb
            :alt: Open in Colab

        Example:
            >>> model = Modely()
            >>> x = Input('x')
            >>> y = Input('y')
            >>> relation = Fir(x.last())
            >>> model.addConnect(relation, y)
        """
        self._model_def.addConnection(stream_out, input_in,'connect', local)

    @enforce_types
    def addClosedLoop(self, stream_out:str|Output|Stream, input_in:str|Input, *, local:bool=False) -> None:
        """
        Adds a closed loop connection from a relation stream to an input.

        Parameters
        ----------
        stream_out : Stream
            The relation stream to connect from.
        input_in : Input or list of inputs
            The Input or the list of inputs to connect to.

        Examples
        --------
        .. image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/tonegas/nnodely/blob/main/examples/states.ipynb
            :alt: Open in Colab

        Example:
            >>> model = Modely()
            >>> x = Input('x')
            >>> y = Input('y')
            >>> relation = Fir(x.last())
            >>> model.addClosedLoop(relation, y)
        """
        self._model_def.addConnection(stream_out, input_in,'closedLoop', local)

    @enforce_types
    def removeConnection(self, input_in:str|Input) -> None:
        """
        Remove a closed loop or connect connection from an input.

        Parameters
        ----------
        input_in : Input or name of the input of inputs
            The Input to disconnect.

        Examples
        --------
        .. image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/tonegas/nnodely/blob/main/examples/states.ipynb
            :alt: Open in Colab

        Example:
            >>> model = Modely()
            >>> x = Input('x')
            >>> y = Input('y')
            >>> relation = Fir(x.last())
            >>> model.addConnect(relation, y)
            >>> model.removeConnection(y)
        """
        if isinstance(input_in, Input):
            input_name = input_in.name
        else:
            input_name = input_in
        self._model_def.removeConnection(input_name)

    @enforce_types
    def neuralizeModel(self, sample_time:float|int|None = None, *, clear_model:bool = False, model_def:dict|None = None) -> None:
        """
        Neuralizes the model, preparing it for inference and training. This method creates a neural network model starting from the model definition.
        It will also create all the time windows and correct slicing for all the inputs defined.

        Parameters
        ----------
        sample_time : float or None, optional
            The sample time for the model. Default is 1.0
        clear_model : bool, optional
            Whether to clear the existing model definition. Default is False.
        model_def : dict or None, optional
            A dictionary defining the model. If provided, it overrides the existing model definition. Default is None.

        Raises
        ------
        ValueError
            If sample_time is not None and model_def is provided.
            If clear_model is True and model_def is provided.

        Example
        -------
        Example usage:
            >>> model = Modely(name='example_model')
            >>> model.neuralizeModel(sample_time=0.1, clear_model=True)
        """
        if model_def is not None:
            check(sample_time == None, ValueError, 'The sample_time must be None if a model_def is provided')
            check(clear_model == False, ValueError, 'The clear_model must be False if a model_def is provided')
            self._model_def = ModelDef(model_def)
        else:
            self._model_def.updateParameters(model = None, clear_model = clear_model)

        self._model_def.setBuildWindow(sample_time)
        self._model = Model(self._model_def.getJson())
        self.__addInfo()

        self._input_ns_backward = {key:value['ns'][0] for key, value in self._model_def['Inputs'].items()}
        self._input_ns_forward = {key:value['ns'][1] for key, value in self._model_def['Inputs'].items()}
        self._max_samples_backward = max(self._input_ns_backward.values())
        self._max_samples_forward = max(self._input_ns_forward.values())
        self._input_n_samples = {}
        for key, value in self._model_def['Inputs'].items():
            if self._input_ns_forward[key] >= 0:
                if 'closedLoop' in value:
                    log.warning(f"Closed loop on {key} with sample in the future.")
                if 'connect' in value:
                    log.warning(f"Connect on {key} with sample in the future.")
            self._input_n_samples[key] = self._input_ns_backward[key] + self._input_ns_forward[key]
        self._max_n_samples = max(self._input_ns_backward.values()) + max(self._input_ns_forward.values())

        ## Initialize States
        self.resetStates()

        self._neuralized = True
        self._traced = False
        self._model_def.updateParameters(self._model)
        self.visualizer.showModel(self._model_def.getJson())
        self.visualizer.showModelInputWindow()
        self.visualizer.showBuiltModel()

    @enforce_types
    def __call__(self, inputs:dict={}, *, sampled:bool=False, closed_loop:dict={}, connect:dict={}, prediction_samples:str|int='auto', num_of_samples:int|None=None) -> dict:
        """
        Performs inference on the model.

        Parameters
        ----------
        inputs : dict, optional
            A dictionary of input data. The keys are input names and the values are the corresponding data. Default is an empty dictionary.
        sampled : bool, optional
            A boolean indicating whether the inputs are already sampled. Default is False.
        closed_loop : dict, optional
            A dictionary specifying closed loop connections. The keys are input names and the values are output names. Default is an empty dictionary.
        connect : dict, optional
            A dictionary specifying direct connections. The keys are input names and the values are output names. Default is an empty dictionary.
        prediction_samples : str or int, optional
            The number of prediction samples. Can be 'auto', None or an integer. Default is 'auto'.
        num_of_samples : str or int, optional
            The number of samples. Can be 'auto', None or an integer. Default is 'auto'.

        Returns
        -------
        dict
            A dictionary containing the model's prediction outputs.

        Raises
        ------
        RuntimeError
            If the network is not neuralized.
        ValueError
            If an input variable is not in the model definition or if an output variable is not in the model definition.

        Examples
        --------
        .. image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/tonegas/nnodely/blob/main/examples/inference.ipynb
            :alt: Open in Colab

        Example usage:
            >>> model = Modely()
            >>> x = Input('x')
            >>> out = Output('out', Fir(x.last()))
            >>> model.addModel('example_model', [out])
            >>> model.neuralizeModel()
            >>> predictions = model(inputs={'x': [1, 2, 3]})
        """

        ## Copy dict for avoid python bug
        inputs = copy.deepcopy(inputs)
        all_closed_loop = copy.deepcopy(closed_loop) #| self._model_def._input_closed_loop
        all_connect = copy.deepcopy(connect) #| self._model_def._input_connect

        ## Check neuralize
        check(self.neuralized, RuntimeError, "The network is not neuralized.")

        ## Check closed loop integrity
        prediction_samples = self._setup_recurrent_variables(prediction_samples, all_closed_loop, all_connect)

        ## List of keys
        model_inputs = list(self._model_def['Inputs'].keys())
        json_inputs = self._model_def['Inputs']
        extra_inputs = list(set(list(inputs.keys())) - set(model_inputs))
        non_mandatory_inputs = list(all_closed_loop.keys()) + list(all_connect.keys()) +  list(self._model_def.recurrentInputs().keys())
        mandatory_inputs = list(set(model_inputs) - set(non_mandatory_inputs))

        ## Remove extra inputs
        for key in extra_inputs:
            log.warning(
                f'The provided input {key} is not used inside the network. the inference will continue without using it')
            del inputs[key]

        ## Get the number of data windows for each input
        num_of_windows = {key: len(value) for key, value in inputs.items()} if sampled else {
            key: len(value) - self._input_n_samples[key] + 1 for key, value in inputs.items()}

        if num_of_samples is not None and sampled == True:
            log.warning(f'num_of_samples is ignored if sampled is equal to True')

        ## Get the maximum inference window
        if num_of_samples and not sampled:
            window_dim = num_of_samples
            for key in inputs.keys():
                input_dim = self._model_def['Inputs'][key]['dim']
                new_samples = num_of_samples - (len(inputs[key]) - self._input_n_samples[key] + 1)
                if input_dim > 1:
                    log.warning(f'The variable {key} is filled with {new_samples} samples equal to zeros.')
                    inputs[key] += [[0 for _ in range(input_dim)] for _ in range(new_samples)]
                else:
                    log.warning(f'The variable {key} is filled with {new_samples} samples equal to zeros.')
                    inputs[key] += [0 for _ in range(new_samples)]
        elif inputs:
            windows = []
            for key in inputs.keys():
                if key in mandatory_inputs:
                    n_samples = len(inputs[key]) if sampled else len(inputs[key]) - self._model_def['Inputs'][key]['ntot'] + 1
                    windows.append(n_samples)
            if not windows:
                for key in inputs.keys():
                    if key in non_mandatory_inputs:
                        if key in model_inputs:
                            n_samples = len(inputs[key]) if sampled else len(inputs[key]) - self._model_def['Inputs'][key]['ntot'] + 1
                        windows.append(n_samples)
            window_dim = min(windows) if windows else 0
        else:  ## No inputs
            window_dim = 1 if non_mandatory_inputs else 0
        check(window_dim > 0, StopIteration, f'Missing samples in the input window')

        if len(set(num_of_windows.values())) > 1:
            max_ind_key, max_dim = argmax_dict(num_of_windows)
            min_ind_key, min_dim = argmin_dict(num_of_windows)
            log.warning(
                f'Different number of samples between inputs [MAX {num_of_windows[max_ind_key]} = {max_dim}; MIN {num_of_windows[min_ind_key]} = {min_dim}]')

        ## Autofill the missing inputs
        provided_inputs = list(inputs.keys())
        missing_inputs = list(set(mandatory_inputs) - set(provided_inputs))
        if missing_inputs:
            log.warning(f'Inputs not provided: {missing_inputs}. Autofilling with zeros..')
            for key in missing_inputs:
                inputs[key] = np.zeros(
                    shape=(self._input_n_samples[key] + window_dim - 1, self._model_def['Inputs'][key]['dim']),
                    dtype=NP_DTYPE).tolist()

        ## Transform inputs in 3D Tensors
        for key in inputs.keys():
            input_dim = json_inputs[key]['dim']
            inputs[key] = torch.from_numpy(np.array(inputs[key])).to(TORCH_DTYPE)

            if input_dim > 1:
                correct_dim = 3 if sampled else 2
                check(len(inputs[key].shape) == correct_dim, ValueError,
                      f'The input {key} must have {correct_dim} dimensions')
                check(inputs[key].shape[correct_dim - 1] == input_dim, ValueError,
                      f'The second dimension of the input "{key}" must be equal to {input_dim}')

            if input_dim == 1 and inputs[key].shape[-1] != 1:  ## add the input dimension
                inputs[key] = inputs[key].unsqueeze(-1)
            if inputs[key].ndim <= 1:  ## add the batch dimension
                inputs[key] = inputs[key].unsqueeze(0)
            if inputs[key].ndim <= 2:  ## add the time dimension
                inputs[key] = inputs[key].unsqueeze(0)

        ## initialize the resulting dictionary
        result_dict = {}
        for key in self._model_def['Outputs'].keys():
            result_dict[key] = []

        ## Inference
        with (torch.enable_grad() if self._get_gradient_on_inference() else torch.inference_mode()):
            ## Update with virtual states
            if prediction_samples == 'auto' or prediction_samples >= 0:
                self._model.update(closed_loop=all_closed_loop, connect=all_connect)
            else:
                self._model.update(disconnect=True)
                prediction_samples = 0
            X = {}
            count = 0
            first = True
            for idx in range(window_dim):
                ## Get mandatory data inputs
                for key in mandatory_inputs:
                    X[key] = inputs[key][idx:idx + 1] if sampled else inputs[key][:,idx:idx + self._input_n_samples[key]]
                    if 'type' in json_inputs[key].keys():
                        X[key] = X[key].requires_grad_(True)
                ## reset states
                if count == 0 or prediction_samples == 'auto':
                    init_states = []
                    count = prediction_samples
                    for key in non_mandatory_inputs:  ## Get non mandatory data (from inputs, from states, or with zeros)
                        ## If it is given as input AND
                        ## if prediction_samples is 'auto' and there are enough samples OR
                        ## if prediction_samples is NOT 'auto'
                        if key in inputs.keys() and (
                                (prediction_samples == 'auto' and idx < num_of_windows[key]) or \
                                (prediction_samples != 'auto')
                        ):
                            X[key] = inputs[key][idx:idx + 1] if sampled else inputs[key][:,idx:idx + self._input_n_samples[key]]
                        ## if it is a state AND
                        ## if prediction_samples = 'auto' and there are not enough samples OR
                        ## it is the first iteration with prediction_samples = None
                        elif key in self._states.keys() and (
                                prediction_samples == 'auto' or
                                (first and prediction_samples == None)
                        ):
                            X[key] = self._states[key]
                        else:
                        ## if there are no samples
                            window_size = self._input_n_samples[key]
                            dim = json_inputs[key]['dim']
                            X[key] = torch.zeros(size=(1, window_size, dim), dtype=TORCH_DTYPE, requires_grad=False)

                        if 'type' in json_inputs[key].keys():
                            X[key] = X[key].requires_grad_(True)
                    first = False
                else:
                    # Remove the gradient of the previous forward
                    for key in X.keys():
                        if 'type' in json_inputs[key].keys():
                            X[key] = X[key].detach().requires_grad_(True)
                    count -= 1
                ## Forward pass
                result, _, out_closed_loop, out_connect = self._model(X)

                if init_states:
                    for key in init_states:
                        del self._model.connect_update[key]
                    init_states = []

                ## Append the prediction of the current sample to the result dictionary
                for key in self._model_def['Outputs'].keys():
                    if result[key].shape[-1] == 1:
                        result[key] = result[key].squeeze(-1)
                        if result[key].shape[-1] == 1:
                            result[key] = result[key].squeeze(-1)
                    result_dict[key].append(result[key].detach().squeeze(dim=0).tolist())

                ## Update closed_loop and connect
                if prediction_samples:
                    self._update_state(X, out_closed_loop, out_connect)

        ## Remove virtual states
        self._remove_virtual_states(connect, closed_loop)

        return result_dict


