import copy
from collections import defaultdict
from unittest import result

import numpy as np
import  torch, random

from nnodely.support.utils import TORCH_DTYPE, NP_DTYPE, check, enforce_types, tensor_to_list
from nnodely.basic.modeldef import ModelDef

from nnodely.support.logger import logging, nnLogger
log = nnLogger(__name__, logging.WARNING)

class Network:
    @enforce_types
    def __init__(self):
        check(type(self) is not Network, TypeError, "Loader class cannot be instantiated directly")

        # Models definition
        self._model_def = ModelDef()
        self._model = None
        self._neuralized = False
        self._traced = False

        # Model components
        self._states = {}
        self._input_n_samples = {}
        self._input_ns_backward = {}
        self._input_ns_forward = {}
        self._max_samples_backward = None
        self._max_samples_forward = None
        self._max_n_samples = 0

        # Dataset information
        self._data_loaded = False
        self._file_count = 0
        self._num_of_samples = {}
        self._data = {}
        self._multifile = {}

        # Training information
        self._standard_train_parameters = {
            'models': None,
            'train_dataset': None, 'validation_dataset': None,
            'dataset': None, 'splits': [100, 0, 0],
            'closed_loop': {}, 'connect': {}, 'step': 0, 'prediction_samples': 0,
            'shuffle_data': True,
            'early_stopping': None, 'early_stopping_params': {},
            'select_model': 'last', 'select_model_params': {},
            'minimize_gain': {},
            'num_of_epochs': 100,
            'train_batch_size': 128, 'val_batch_size': 128,
            'optimizer': 'Adam',
            'lr': 0.001, 'lr_param': {},
            'optimizer_params': [], 'add_optimizer_params': [],
            'optimizer_defaults': {}, 'add_optimizer_defaults': {}
        }
        self._training = {}

        # Save internal
        self._log_internal = False
        self._internals = {}

    def _save_internal(self, key, value):
        self._internals[key] = tensor_to_list(value)

    def _set_log_internal(self, log_internal:bool):
        self._log_internal = log_internal

    def _clean_log_internal(self):
        self._internals = {}

    def _remove_virtual_states(self, connect, closed_loop):
        if connect or closed_loop:
            for key in (connect.keys() | closed_loop.keys()):
                if key in self._states.keys():
                    del self._states[key]

    def _update_state(self, X, out_closed_loop, out_connect):
        for key, value in out_connect.items():
            X[key] = value
            self._states[key] = X[key].clone().detach()
        for key, val in out_closed_loop.items():
            shift = val.shape[1] #+ self._input_ns_forward[key]  ## take the output time dimension + forward samples
            X[key] = torch.roll(X[key], shifts=-1, dims=1)  ## Roll the time window
            X[key][:, -shift:, :] = val  ## substitute with the predicted value
            self._states[key] = X[key].clone().detach()

    def _get_gradient_on_inference(self):
        for key, value in self._model_def['Inputs'].items():
            if 'type' in value.keys():
                return True
        return False

    def _get_mandatory_inputs(self, connect, closed_loop):
        model_inputs = list(self._model_def['Inputs'].keys())
        non_mandatory_inputs = list(closed_loop.keys()) + list(connect.keys()) + list(self._model_def.recurrentInputs().keys())
        mandatory_inputs = list(set(model_inputs) - set(non_mandatory_inputs))
        return mandatory_inputs, non_mandatory_inputs
    
    def _get_batch_indexes(self, datasets:str|list|dict|None, n_samples:int=0, prediction_samples:int=0):
        if datasets is None:
            return []
        batch_indexes = list(range(n_samples))
        if prediction_samples > 0 and not isinstance(datasets, dict):
            datasets = [datasets] if type(datasets) is str else datasets
            forbidden_idxs = []
            n_samples_count = 0
            for dataset in datasets:
                if dataset in self._multifile.keys(): ## i have some forbidden indexes
                    for i in self._multifile[dataset]:
                        if i+n_samples_count < batch_indexes[-1]:
                            forbidden_idxs.extend(range((i+n_samples_count) - prediction_samples, (i+n_samples_count), 1))
                n_samples_count += self._num_of_samples[dataset]
            batch_indexes = [idx for idx in batch_indexes if idx not in forbidden_idxs]
            batch_indexes = batch_indexes[:-prediction_samples]
        return batch_indexes
    
    def _get_data(self, dataset:str|list|dict|None):
        if dataset is None:
            return {}
        if isinstance(dataset, dict):
            self.__check_data_integrity(dataset)
            return dataset
        dataset = [dataset] if type(dataset) is str else dataset
        loaded_datasets = list(self._data.keys())
        check(len([data for data in dataset if data in loaded_datasets]) > 0, KeyError, f'the datasets: {dataset} are not loaded!')
        total_data = defaultdict(list)
        for data in dataset:
            if data not in loaded_datasets:
                log.warning(f'{data} is not loaded. Ignoring this dataset...') 
                dataset.remove(data)
                continue
            for k, v in self._data[data].items():
                total_data[k].append(v)
        total_data = {key: np.concatenate(arrays) for key, arrays in total_data.items()}
        total_data = {key: torch.from_numpy(val).to(TORCH_DTYPE) for key, val in total_data.items()}
        return total_data

    def _clip_step(self, step, batch_indexes, batch_size):
        clipped_step = copy.deepcopy(step)
        if clipped_step < 0:  ## clip the step to zero
            log.warning(f"The step is negative ({clipped_step}). The step is set to zero.", stacklevel=5)
            clipped_step = 0
        if clipped_step > (len(batch_indexes) - batch_size):  ## Clip the step to the maximum number of samples
            log.warning(f"The step ({clipped_step}) is greater than the number of available samples ({len(batch_indexes) - batch_size}). The step is set to the maximum number.", stacklevel=5)
            clipped_step = len(batch_indexes) - batch_size
        check((batch_size + clipped_step) > 0, ValueError, f"The sum of batch_size={batch_size} and the step={clipped_step} must be greater than 0.")
        return clipped_step

    def _clip_batch_size(self, n_samples, batch_size=None):
        batch_size = batch_size if batch_size <= n_samples else max(0, n_samples)
        check((n_samples - batch_size + 1) > 0, ValueError, f"The number of available sample are {n_samples - batch_size + 1}")
        check(batch_size > 0, ValueError, f'The batch_size must be greater than 0.')
        return batch_size
    
    def __split_dataset(self, dataset:str|list|dict, splits:list):
        check(len(splits) == 3, ValueError, '3 elements must be inserted for the dataset split in training, validation and test')
        check(sum(splits) == 100, ValueError, 'Training, Validation and Test splits must sum up to 100.')
        check(splits[0] > 0, ValueError, 'The training split cannot be zero.')
        train_size, val_size, test_size = splits[0] / 100, splits[1] / 100, splits[2] / 100
        XY_train, XY_val, XY_test = {}, {}, {}
        if isinstance(dataset, dict):
            self.__check_data_integrity(dataset)
            num_of_samples = next(iter(dataset.values())).size(0)
            XY_train = {key: value[:round(num_of_samples*train_size), :, :] for key, value in dataset.items()}
            XY_val = {key: value[round(num_of_samples*train_size):round(num_of_samples*(train_size + val_size)), :, :] for key, value in dataset.items()}
            XY_test = {key: value[round(num_of_samples*(train_size + val_size)):, :, :] for key, value in dataset.items()}
        else:
            dataset = [dataset] if type(dataset) is str else dataset
            check(len([data for data in dataset if data in self._data.keys()]) > 0, KeyError, f'the datasets: {dataset} are not loaded!')
            for data in dataset:
                if data not in self._data.keys():
                    log.warning(f'{data} is not loaded. The training will continue without this dataset.') 
                    dataset.remove(data)

            num_of_samples = sum([self._num_of_samples[data] for data in dataset])
            n_samples_train, n_samples_val = round(num_of_samples * train_size), round(num_of_samples * val_size)
            n_samples_test = num_of_samples - n_samples_train - n_samples_val
            check(n_samples_train > 0, ValueError, f'The number of train samples {n_samples_train} must be greater than 0.')
            total_data = defaultdict(list)
            for data in dataset:
                for k, v in self._data[data].items():
                    total_data[k].append(v)
            total_data = {key: np.concatenate(arrays, dtype=NP_DTYPE) for key, arrays in total_data.items()}
            for key, samples in total_data.items():
                if val_size == 0.0 and test_size == 0.0:  ## we have only training set
                    XY_train[key] = torch.from_numpy(samples).to(TORCH_DTYPE)
                elif val_size == 0.0 and test_size != 0.0:  ## we have only training and test set
                    XY_train[key] = torch.from_numpy(samples[:n_samples_train]).to(TORCH_DTYPE)
                    XY_test[key] = torch.from_numpy(samples[n_samples_train:]).to(TORCH_DTYPE)
                elif val_size != 0.0 and test_size == 0.0:  ## we have only training and validation set
                    XY_train[key] = torch.from_numpy(samples[:n_samples_train]).to(TORCH_DTYPE)
                    XY_val[key] = torch.from_numpy(samples[n_samples_train:]).to(TORCH_DTYPE)
                else:  ## we have training, validation and test set
                    XY_train[key] = torch.from_numpy(samples[:n_samples_train]).to(TORCH_DTYPE)
                    XY_val[key] = torch.from_numpy(samples[n_samples_train:-n_samples_test]).to(TORCH_DTYPE)
                    XY_test[key] = torch.from_numpy(samples[n_samples_train + n_samples_val:]).to(TORCH_DTYPE)
        return XY_train, XY_val, XY_test

    def _get_tag(self, dataset: str | list | dict | None) -> str:
        """
        Helper function to get the tag for a dataset.
        """
        if isinstance(dataset, str):
            return dataset
        elif isinstance(dataset, list):
            return f"{dataset[0]}_{len(dataset)}" if len(dataset) > 1 else f"{dataset[0]}"
        elif isinstance(dataset, dict):
            return "custom_dataset"
        return dataset

    def _setup_dataset(self, train_dataset:str|list|dict, validation_dataset:str|list|dict, test_dataset:str|list|dict, dataset:str|list|dict, splits:list):
        if train_dataset is None: ## use the splits
            train_dataset = list(self._data.keys()) if dataset is None else dataset
            return self.__split_dataset(train_dataset, splits)
        else: ## use each dataset
            return self._get_data(train_dataset), self._get_data(validation_dataset), self._get_data(test_dataset)

    def __check_data_integrity(self, dataset:dict):
        if bool(dataset):
            check(len(set([t.size(0) for t in dataset.values()])) == 1, ValueError, "All the tensors in the dataset must have the same number of samples.")
            #TODO check why is wrong
            #check(len([t for t in self._model_def['Inputs'].keys() if t in dataset.keys()]) == len(list(self._model_def['Inputs'].keys())), ValueError, "Some inputs are missing.")
            for key, value in dataset.items():
                if key not in self._model_def['Inputs']:
                    log.warning(f"The key '{key}' is not an input of the network. It will be ignored.")
                else:
                    check(isinstance(value, torch.Tensor), TypeError, f"The value of the input '{key}' must be a torch.Tensor.")
                    check(value.size(1) == self._model_def['Inputs'][key]['ntot'], ValueError, f"The time size of the input '{key}' is not correct. Expected {self._model_def['Inputs'][key]['ntot']}, got {value.size(1)}.")
                    check(value.size(2) == self._model_def['Inputs'][key]['dim'], ValueError, f"The dimension of the input '{key}' is not correct. Expected {self._model_def['Inputs'][key]['dim']}, got {value.size(2)}.")

    def _get_not_mandatory_inputs(self, data, X, non_mandatory_inputs, remaning_indexes, batch_size, step, shuffle = False):
        related_indexes = random.sample(remaning_indexes, batch_size) if shuffle else remaning_indexes[:batch_size]
        for num in related_indexes:
            remaning_indexes.remove(num)
        if step > 0:
            if len(remaning_indexes) >= step:
                step_idxs = random.sample(remaning_indexes, step) if shuffle else remaning_indexes[:step]
                for num in step_idxs:
                    remaning_indexes.remove(num)
            else:
                remaning_indexes.clear()
        for key in non_mandatory_inputs:
            if key in data.keys(): ## with data
                X[key] = data[key][related_indexes]
            else:  ## with zeros
                window_size = self._input_n_samples[key]
                dim = self._model_def['Inputs'][key]['dim']
                if 'type' in self._model_def['Inputs'][key]:
                    X[key] = torch.zeros(size=(batch_size, window_size, dim), dtype=TORCH_DTYPE, requires_grad=True)
                else:
                    X[key] = torch.zeros(size=(batch_size, window_size, dim), dtype=TORCH_DTYPE, requires_grad=False)
                self._states[key] = X[key]
        return related_indexes

    def _inference(self, data, n_samples, batch_size, loss_gains, loss_functions,
                    shuffle = False, optimizer = None,
                    total_losses = None, A = None, B = None):
        if shuffle:
            randomize = torch.randperm(n_samples)
            data = {key: val[randomize] for key, val in data.items()}
        ## Initialize the train losses vector
        aux_losses = torch.zeros([len(self._model_def['Minimizers']), n_samples // batch_size])
        for idx in range(0, (n_samples - batch_size + 1), batch_size):
            ## Build the input tensor
            XY = {key: val[idx:idx + batch_size] for key, val in data.items()}
            ## Reset gradient
            if optimizer:
                optimizer.zero_grad()
            ## Model Forward
            _, minimize_out, _, _ = self._model(XY)  ## Forward pass
            ## Loss Calculation
            total_loss = 0
            for ind, (key, value) in enumerate(self._model_def['Minimizers'].items()):
                if A is not None:
                    A[key].append(minimize_out[value['A']].detach().numpy())
                if B is not None:
                    B[key].append(minimize_out[value['B']].detach().numpy())
                loss = loss_functions[key](minimize_out[value['A']], minimize_out[value['B']])
                loss = (loss * loss_gains[key]) if key in loss_gains.keys() else loss
                if total_losses is not None:
                    total_losses[key].append(loss.detach().numpy())
                aux_losses[ind][idx // batch_size] = loss.item()
                total_loss += loss
            ## Gradient step
            if optimizer:
                total_loss.backward()
                optimizer.step()
                self.visualizer.showWeightsInTrain(batch=idx // batch_size)

        ## return the losses
        return aux_losses

    def _recurrent_inference(self, data, batch_indexes, batch_size, loss_gains, prediction_samples,
                             step, non_mandatory_inputs, mandatory_inputs, loss_functions,
                             shuffle = False, optimizer = None,
                             total_losses = None, A = None, B = None):
        indexes = copy.deepcopy(batch_indexes)
        aux_losses = torch.zeros([len(self._model_def['Minimizers']), round((len(indexes) + step) / (batch_size + step))])
        X = {}
        batch_val = 0
        while len(indexes) >= batch_size:
            selected_indexes = self._get_not_mandatory_inputs(data, X, non_mandatory_inputs, indexes, batch_size, step, shuffle)
            horizon_losses = {ind: [] for ind in range(len(self._model_def['Minimizers']))}
            if optimizer:
                optimizer.zero_grad()  ## Reset the gradient

            for horizon_idx in range(prediction_samples + 1):
                ## Get data
                for key in mandatory_inputs:
                    X[key] = data[key][[idx + horizon_idx for idx in selected_indexes]]
                ## Forward pass
                out, minimize_out, out_closed_loop, out_connect = self._model(X)

                if self._log_internal:
                    #assert (check_gradient_operations(self._states) == 0)
                    #assert (check_gradient_operations(data) == 0)
                    internals_dict = {'XY': tensor_to_list(X), 'out': out, 'param': self._model.all_parameters,
                                      'closedLoop': self._model.closed_loop_update, 'connect': self._model.connect_update}

                ## Loss Calculation
                for ind, (key, value) in enumerate(self._model_def['Minimizers'].items()):
                    if A is not None:
                        A[key][horizon_idx].append(minimize_out[value['A']].detach().numpy())
                    if B is not None:
                        B[key][horizon_idx].append(minimize_out[value['B']].detach().numpy())
                    loss = loss_functions[key](minimize_out[value['A']], minimize_out[value['B']])
                    loss = (loss * loss_gains[key]) if key in loss_gains.keys() else loss
                    horizon_losses[ind].append(loss)

                ## Update
                self._update_state(X, out_closed_loop, out_connect)

                if self._log_internal:
                    internals_dict['state'] = self._states
                    self._save_internal('inout_' + str(batch_val) + '_' + str(horizon_idx), internals_dict)

            ## Calculate the total loss
            total_loss = 0
            for ind, key in enumerate(self._model_def['Minimizers'].keys()):
                loss = sum(horizon_losses[ind]) / (prediction_samples + 1)
                aux_losses[ind][batch_val] = loss.item()
                if total_losses is not None:
                    total_losses[key].append(loss.detach().numpy())
                total_loss += loss

            ## Gradient Step
            if optimizer:
                total_loss.backward()  ## Backpropagate the error
                optimizer.step()
                self.visualizer.showWeightsInTrain(batch=batch_val)
            batch_val += 1

        ## return the losses
        return aux_losses

    def _setup_recurrent_variables(self, prediction_samples, closed_loop, connect):
        ## Prediction samples
        check(prediction_samples == 'auto' or prediction_samples >= -1, KeyError, "The sample horizon must be positive, -1, 'auto', for disconnect connection!")
        ## Close loop information
        for input, output in closed_loop.items():
            check(input in self._model_def['Inputs'], ValueError, f'the tag {input} is not an input variable.')
            check(output in self._model_def['Outputs'], ValueError, f'the tag {output} is not an output of the network')
            log.info(f'Recurrent train: closing the loop between the the input ports {input} and the output ports {output} for {prediction_samples} samples')
            if self._input_ns_forward[input] > 0:
                    log.warning(f"Closed loop on variable '{input}' with sample in the future.")
        ## Connect information
        for input, output in connect.items():
            check(input in self._model_def['Inputs'], ValueError, f'the tag {input} is not an input variable.')
            check(output in self._model_def['Outputs'], ValueError, f'the tag {output} is not an output of the network')
            log.info(f'Recurrent train: connecting the input ports {input} with output ports {output} for {prediction_samples} samples')
            if self._input_ns_forward[input] > 0:
                    log.warning(f"Connect on variable '{input}' with sample in the future.")
        ## Disable recurrent training if there are no recurrent variables
        if len(connect|closed_loop|self._model_def.recurrentInputs()) == 0:
            if type(prediction_samples) is not str and prediction_samples >= 0:
                log.warning(f"The value of the prediction_samples={prediction_samples} but the network has no recurrent variables.")
            prediction_samples = -1
        return prediction_samples

    @enforce_types
    def resetStates(self, states:set={}, *, batch:int=1) -> None:
        """
        Resets the state of all the recurrent inputs of the network to zero.
        Parameters
        ----------
        states : set, optional
            A set of recurrent inputs names to reset. If provided, only those inputs will be resetted.
        batch : int, optional
            The batch size for the reset states. Default is 1.
        """
        if states: ## reset only specific states
            for key in states:
                window_size = self._input_n_samples[key]
                dim = self._model_def['Inputs'][key]['dim']
                self._states[key] = torch.zeros(size=(batch, window_size, dim), dtype=TORCH_DTYPE, requires_grad=False)
        else: ## reset all states
            self._states = {}
            for key, state in self._model_def.recurrentInputs().items():
                window_size = self._input_n_samples[key]
                dim = state['dim']
                self._states[key] = torch.zeros(size=(batch, window_size, dim), dtype=TORCH_DTYPE, requires_grad=False)

