import copy, torch, time, inspect

from collections.abc import Callable
from functools import wraps

from nnodely.basic.modeldef import ModelDef
from nnodely.basic.model import Model
from nnodely.basic.optimizer import Optimizer, SGD, Adam
from nnodely.basic.loss import CustomLoss
from nnodely.operators.network import Network
from nnodely.support.utils import check, enforce_types
from nnodely.basic.relation import Stream
from nnodely.layers.output import Output

from nnodely.support.logger import logging, nnLogger
log = nnLogger(__name__, logging.INFO)


class Trainer(Network):
    def __init__(self):
        check(type(self) is not Trainer, TypeError, "Trainer class cannot be instantiated directly")
        super().__init__()

        ## User Parameters
        self.running_parameters = {}

        # Training Losses
        self.__loss_functions = {}

        # Optimizer
        self.__optimizer = None

    @enforce_types
    def addMinimize(self, name:str, streamA:str|Stream|Output, streamB:str|Stream|Output, loss_function:str='mse') -> None:
        """
        Adds a minimize loss function to the model.

        Parameters
        ----------
        name : str
            The name of the cost function.
        streamA : Stream
            The first relation stream for the minimize operation.
        streamB : Stream
            The second relation stream for the minimize operation.
        loss_function : str, optional
            The loss function to use from the ones provided. Default is 'mse'.

        Example
        -------
        Example usage:
            >>> model.addMinimize('minimize_op', streamA, streamB, loss_function='mse')
        """
        self._model_def.addMinimize(name, streamA, streamB, loss_function)
        self.visualizer.showaddMinimize(name)

    @enforce_types
    def removeMinimize(self, name_list:list|str) -> None:
        """
        Removes minimize loss functions using the given list of names.

        Parameters
        ----------
        name_list : list of str
            The list of minimize operation names to remove.

        Example
        -------
        Example usage:
            >>> model.removeMinimize(['minimize_op1', 'minimize_op2'])
        """
        self._model_def.removeMinimize(name_list)

    def __preliminary_checks(self, **kwargs):
        check(self._data_loaded, RuntimeError, 'There is no data loaded! The Training will stop.')
        check('Models' in self._model_def.getJson(), RuntimeError, 'There are no models to train. Load a model using the addModel function.')
        check(list(self._model.parameters()), RuntimeError, 'There are no modules with learnable parameters! The Training will stop.')
        if kwargs.get('train_dataset', None) is None:
            check(kwargs.get('validation_dataset', None) is None, ValueError, 'If train_dataset is None, validation_dataset must also be None.')
        for model in kwargs['models']:
            check(model in kwargs['all_models'], ValueError, f'The model {model} is not in the model definition')

    def __fill_parameters(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            sig = inspect.signature(func)
            bound = sig.bind(self, *args, **kwargs)
            bound.apply_defaults()
            # Get standard parameters
            standard = self._standard_train_parameters
            # Get user_parameters
            users = bound.arguments.get('training_params', None)
            # Fill missing (None) arguments
            for param in sig.parameters.values():
                if param.name == 'self' or param.name == 'lr' or param.name == 'lr_param':
                    continue
                if bound.arguments.get(param.name, None) is None:
                    if param.name in users.keys():
                        bound.arguments[param.name] = users[param.name]
                    else:
                        bound.arguments[param.name] = standard.get(param.name, None)
            return func(**bound.arguments)
        return wrapper

    def __initialize_optimizer(self, models, optimizer, training_params, optimizer_params, optimizer_defaults, add_optimizer_defaults, add_optimizer_params, lr, lr_param):
        ## Get models
        params_to_train = set()
        for model in models:
            if type(self._model_def['Models']) is dict:
                params_to_train |= set(self._model_def['Models'][model]['Parameters'])
            else:
                params_to_train |= set(self._model_def['Parameters'].keys())

        # Get the optimizer
        if type(optimizer) is str:
            if optimizer == 'SGD':
                optimizer = SGD({}, [])
            elif optimizer == 'Adam':
                optimizer = Adam({}, [])
        else:
            optimizer = copy.deepcopy(optimizer)
            check(issubclass(type(optimizer), Optimizer), TypeError, "The optimizer must be an Optimizer or str")

        optimizer.set_params_to_train(self._model.all_parameters, params_to_train)

        optimizer.add_defaults('lr', self._standard_train_parameters['lr'])

        if training_params and 'lr' in training_params:
            optimizer.add_defaults('lr', training_params['lr'])
        if training_params and 'lr_param' in training_params:
            optimizer.add_option_to_params('lr', training_params['lr_param'])

        if optimizer_defaults != {}:
            optimizer.set_defaults(optimizer_defaults)
        if optimizer_params != []:
            optimizer.set_params(optimizer_params)

        for key, value in add_optimizer_defaults.items():
            optimizer.add_defaults(key, value)

        add_optimizer_params = optimizer.unfold(add_optimizer_params)
        for param in add_optimizer_params:
            par = param['params']
            del param['params']
            for key, value in param.items():
                optimizer.add_option_to_params(key, {par: value})

        # Modify the parameter
        optimizer.add_defaults('lr', lr)
        if lr_param:
            optimizer.add_option_to_params('lr', lr_param)

        self.__optimizer = optimizer

    def __initialize_loss(self):
        for name, values in self._model_def['Minimizers'].items():
            self.__loss_functions[name] = CustomLoss(values['loss'])

    def getTrainingInfo(self):
        """
        Returns a dictionary with the training parameters and information.
        Parameters
        ----------
        **kwargs : dict
            Additional parameters to include in the training information.
        Returns
        -------
        dict
            A dictionary containing the training parameters and information.
        """
        to_remove =  ['XY_train','XY_val','XY_test','train_indexes','val_indexes','test_indexes']
        tp = copy.deepcopy({key:value for key, value in self.running_parameters.items() if key not in to_remove})

        ## training
        tp['update_per_epochs'] = len(self.running_parameters['train_indexes']) // (tp['train_batch_size'] + tp['step'])
        if tp['prediction_samples'] >= 0: # TODO
            tp['n_first_samples_train'] = len(self.running_parameters['train_indexes'])
            if tp['n_samples_val'] > 0:
                tp['n_first_samples_val'] = len(self.running_parameters['val_indexes'])
            if tp['n_samples_test'] > 0:
                tp['n_first_samples_test'] = len(self.running_parameters['test_indexes'])


        ## optimizer
        tp['optimizer'] = self.__optimizer.name
        tp['optimizer_defaults'] = self.__optimizer.optimizer_defaults
        tp['optimizer_params'] = self.__optimizer.optimizer_params

        ## early stopping
        early_stopping = tp['early_stopping']
        if early_stopping:
            tp['early_stopping'] = early_stopping.__name__

        ## Loss functions
        tp['minimizers'] = {}
        for name, values in self._model_def['Minimizers'].items():
            tp['minimizers'][name] = {}
            tp['minimizers'][name]['A'] = values['A']
            tp['minimizers'][name]['B'] = values['B']
            tp['minimizers'][name]['loss'] = values['loss']
            if name in tp['minimize_gain']:
                tp['minimizers'][name]['gain'] = tp['minimize_gain'][name]

        return tp

    def __check_needed_keys(self, train_data, connect, closed_loop):
        # Needed keys
        keys = set(self._model_def['Inputs'].keys())
        keys |= ({value['A'] for value in self._model_def['Minimizers'].values()} | {value['B'] for value in  self._model_def['Minimizers'].values()})
        # Available keys
        keys -= set(self._model_def['Outputs'].keys()|self._model_def['Relations'].keys())
        keys -= set(self._model_def.recurrentInputs().keys())
        keys -= (set(connect.keys()|closed_loop.keys()))
        # Check if the keys are in the dataset
        check(set(keys).issubset(set(train_data.keys())), KeyError, f"Not all the mandatory keys {keys} are present in the training dataset {set(train_data.keys())}.")

    @enforce_types
    @__fill_parameters
    def trainModel(self, *,
                   name: str | None = None,
                   models: str | list | None = None,
                   train_dataset: str | list | dict | None = None, validation_dataset: str | list | dict | None = None,
                   dataset: str | list | None = None, splits: list | None = None,
                   closed_loop: dict | None = None, connect: dict | None = None, step: int | None = None, prediction_samples: int | None = None,
                   shuffle_data: bool | None = None,
                   early_stopping: Callable | None = None, early_stopping_params: dict | None = None,
                   select_model: Callable | None = None, select_model_params: dict | None = None,
                   minimize_gain: dict | None = None,
                   num_of_epochs: int = None,
                   train_batch_size: int = None, val_batch_size: int = None,
                   optimizer: str | Optimizer | None = None,
                   lr: int | float | None = None, lr_param: dict | None = None,
                   optimizer_params: list | None = None, optimizer_defaults: dict | None = None,
                   add_optimizer_params: list | None = None, add_optimizer_defaults: dict | None = None,
                   training_params: dict | None = {}
                   ) -> None:
        """
        Trains the model using the provided datasets and parameters.

        Notes
        -----
        .. note::
            If no datasets are provided, the model will use all the datasets loaded inside nnodely.

        Parameters
        ----------
        name : str or None, optional
            A name used to identify the training operation.
        models : str or list or None, optional
            A list or name of models to train. Default is all the models loaded.
        train_dataset : str or None, optional
            The name of datasets to use for training.
        validation_dataset : str or None, optional
            The name of datasets to use for validation.
        dataset : str or None, optional
            The name of the datasets to use for training, validation and test.
        splits : list or None, optional
            A list of 3 elements specifying the percentage of splits for training, validation, and testing. The three elements must sum up to 100! default is [100, 0, 0]
            The parameter splits is only used when 'dataset' is not None.
        closed_loop : dict or None, optional
            A dictionary specifying closed loop connections. The keys are input names and the values are output names. Default is None.
        connect : dict or None, optional
            A dictionary specifying connections. The keys are input names and the values are output names. Default is None.
        step : int or None, optional
            The step size for training. A big value will result in less data used for each epochs and a faster train. Default is zero.
        prediction_samples : int or None, optional
            The size of the prediction horizon. Number of samples at each recurrent window Default is zero.
        shuffle_data : bool or None, optional
            Whether to shuffle the data during training. Default is True.
        early_stopping : Callable or None, optional
            A callable for early stopping. Default is None.
        early_stopping_params : dict or None, optional
            A dictionary of parameters for early stopping. Default is None.
        select_model : Callable or None, optional
            A callable for selecting the best model. Default is None.
        select_model_params : dict or None, optional
            A dictionary of parameters for selecting the best model. Default is None.
        minimize_gain : dict or None, optional
            A dictionary specifying the gain for each minimization loss function. Default is None.
        num_of_epochs : int or None, optional
            The number of epochs to train the model. Default is 100.
        train_batch_size : int or None, optional
            The batch size for training. Default is 128.
        val_batch_size : int or None, optional
            The batch size for validation. Default is 128.
        optimizer : Optimizer or None, optional
            The optimizer to use for training. Default is 'Adam'.
        lr : float or None, optional
            The learning rate. Default is 0.001
        lr_param : dict or None, optional
            A dictionary of learning rate parameters. Default is None.
        optimizer_params : list or dict or None, optional
            A dictionary of optimizer parameters. Default is None.
        optimizer_defaults : dict or None, optional
            A dictionary of default optimizer settings. Default is None.
        training_params : dict or None, optional
            A dictionary of training parameters. Default is None.
        add_optimizer_params : list or None, optional
            Additional optimizer parameters. Default is None.
        add_optimizer_defaults : dict or None, optional
            Additional default optimizer settings. Default is None.

        Examples
        --------
        .. image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/tonegas/nnodely/blob/main/examples/training.ipynb
            :alt: Open in Colab

        Example - basic feed-forward training:
            >>> x = Input('x')
            >>> F = Input('F')

            >>> xk1 = Output('x[k+1]', Fir()(x.tw(0.2))+Fir()(F.last()))

            >>> mass_spring_damper = Modely(seed=0)
            >>> mass_spring_damper.addModel('xk1',xk1)
            >>> mass_spring_damper.neuralizeModel(sample_time = 0.05)

            >>> data_struct = ['time','x','dx','F']
            >>> data_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)),'dataset','data')
            >>> mass_spring_damper.loadData(name='mass_spring_dataset', source=data_folder, format=data_struct, delimiter=';')

            >>> params = {'num_of_epochs': 100,'train_batch_size': 128,'lr':0.001}
            >>> mass_spring_damper.trainModel(splits=[70,20,10], training_params = params)

        Example - recurrent training:
            >>> x = Input('x')
            >>> F = Input('F')

            >>> xk1 = Output('x[k+1]', Fir()(x.tw(0.2))+Fir()(F.last()))

            >>> mass_spring_damper = Modely(seed=0)
            >>> mass_spring_damper.addModel('xk1',xk1)
            >>> mass_spring_damper.addClosedLoop(xk1, x)
            >>> mass_spring_damper.neuralizeModel(sample_time = 0.05)

            >>> data_struct = ['time','x','dx','F']
            >>> data_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)),'dataset','data')
            >>> mass_spring_damper.loadData(name='mass_spring_dataset', source=data_folder, format=data_struct, delimiter=';')

            >>> params = {'num_of_epochs': 100,'train_batch_size': 128,'lr':0.001}
            >>> mass_spring_damper.trainModel(splits=[70,20,10], prediction_samples=10, training_params = params)
        """
        ## Get model for train
        all_models = list(self._model_def['Models'].keys()) if type(self._model_def['Models']) is dict else [self._model_def['Models']]
        if models is None:
            models = all_models
        if isinstance(models, str):
            models = [models]

        ## Preliminary Checks
        self.__preliminary_checks(models = models, all_models = all_models, train_dataset = train_dataset, validation_dataset = validation_dataset)

        ## Recurret variables
        prediction_samples = self._setup_recurrent_variables(prediction_samples, closed_loop, connect)

        ## Get the dataset
        XY_train, XY_val, XY_test = self._setup_dataset(train_dataset, validation_dataset, None, dataset, splits)
        self.__check_needed_keys(train_data=XY_train, connect=connect, closed_loop=closed_loop)

        n_samples_train = next(iter(XY_train.values())).size(0)
        n_samples_val = next(iter(XY_val.values())).size(0) if XY_val else 0
        n_samples_test = next(iter(XY_test.values())).size(0) if XY_test else 0

        if train_dataset is not None:
            train_tag = self._get_tag(train_dataset)
            val_tag = self._get_tag(validation_dataset)
        else: ## splits is used
            if dataset is None:
                dataset = list(self._data.keys())
            tag = self._get_tag(dataset)
            train_tag = f"{tag}_train"
            val_tag = f"{tag}_val" if n_samples_val > 0 else None
            test_tag = f"{tag}_test" if n_samples_test > 0 else None

        train_indexes, val_indexes = [], []
        if train_dataset is not None:
            train_indexes, val_indexes = self._get_batch_indexes(train_dataset, n_samples_train, prediction_samples), self._get_batch_indexes(validation_dataset, n_samples_val, prediction_samples)
        else:
            dataset = list(self._data.keys()) if dataset is None else dataset
            train_indexes = self._get_batch_indexes(dataset, n_samples_train, prediction_samples)
            check(len(train_indexes) > 0, ValueError,
                  'The number of valid train samples is less than the number of prediction samples.')
            if n_samples_val > 0:
                val_indexes = self._get_batch_indexes(dataset, n_samples_train + n_samples_val, prediction_samples)
                val_indexes = [i - n_samples_train for i in val_indexes if i >= n_samples_train]
                if len(val_indexes) < 0:
                    log.warning('The number of valid validation samples is less than the number of prediction samples.')
            if n_samples_test > 0:
                test_indexes = self._get_batch_indexes(dataset, n_samples_train + n_samples_val + n_samples_test, prediction_samples)
                test_indexes = [i - (n_samples_train+n_samples_val)for i in test_indexes if i >= (n_samples_train+n_samples_val)]
                if len(test_indexes) < 0:
                    log.warning('The number of valid test samples is less than the number of prediction samples.')

        ## clip batch size and step
        train_batch_size = self._clip_batch_size(len(train_indexes), train_batch_size)
        train_step = self._clip_step(step, train_indexes, train_batch_size)
        if n_samples_val > 0:
            val_batch_size = self._clip_batch_size(len(val_indexes), val_batch_size)
            val_step = self._clip_step(step, val_indexes, val_batch_size)

        ## Save the training parameters
        self.running_parameters = {key:value for key,value in locals().items() if key not in ['self', 'kwargs', 'training_params', 'lr', 'lr_param']}

        ## Define the optimizer
        self.__initialize_optimizer(models, optimizer, training_params, optimizer_params, optimizer_defaults, add_optimizer_defaults, add_optimizer_params, lr, lr_param)
        torch_optimizer = self.__optimizer.get_torch_optimizer()

        ## Define the loss functions
        self.__initialize_loss()

        ## Define mandatory inputs
        mandatory_inputs, non_mandatory_inputs = self._get_mandatory_inputs(connect, closed_loop)

        ## Check close loop and connect
        self._clean_log_internal()

        ## Create the train, validation and test loss dictionaries
        train_losses, val_losses = {}, {}
        for key in self._model_def['Minimizers'].keys():
            train_losses[key] = []
            if n_samples_val > 0:
                val_losses[key] = []

        ## Set the gradient to true if necessary
        model_inputs = self._model_def['Inputs']
        for key in model_inputs.keys():
            if 'type' in model_inputs[key]:
                if key in XY_train:
                    XY_train[key].requires_grad_(True)
                if key in XY_val:
                    XY_val[key].requires_grad_(True)
        selected_model_def = ModelDef(self._model_def.getJson())

        ## Show the training parameters
        self.visualizer.showTrainParams()
        self.visualizer.showStartTraining()

        ## Update with virtual states
        if prediction_samples >= 0:
            self._model.update(closed_loop=closed_loop, connect=connect)
        else:
            self._model.update(disconnect=True)

        self.resetStates()  ## Reset the states

        ## start the train timer
        start = time.time()
        for epoch in range(num_of_epochs):
            ## TRAIN
            self._model.train()
            if prediction_samples >= 0:
                losses = self._recurrent_inference(XY_train, train_indexes, train_batch_size, minimize_gain, prediction_samples, train_step, non_mandatory_inputs, mandatory_inputs, self.__loss_functions, shuffle=shuffle_data, optimizer=torch_optimizer)
            else:
                losses = self._inference(XY_train, n_samples_train, train_batch_size, minimize_gain, self.__loss_functions, shuffle=shuffle_data, optimizer=torch_optimizer)
            ## save the losses
            for ind, key in enumerate(self._model_def['Minimizers'].keys()):
                train_losses[key].append(torch.mean(losses[ind]).tolist())

            if n_samples_val > 0:
                ## VALIDATION
                self._model.eval()
                setted_log_internal = self._log_internal
                self._set_log_internal(False)  # TODO To remove when the function is moved outside the train
                if prediction_samples >= 0:
                    losses = self._recurrent_inference(XY_val, val_indexes, val_batch_size, minimize_gain, prediction_samples, val_step,
                                                       non_mandatory_inputs, mandatory_inputs, self.__loss_functions)
                else:
                    losses = self._inference(XY_val, n_samples_val, val_batch_size, minimize_gain, self.__loss_functions)
                self._set_log_internal(setted_log_internal)

                ## save the losses
                for ind, key in enumerate(self._model_def['Minimizers'].keys()):
                    val_losses[key].append(torch.mean(losses[ind]).tolist())

            ## Early-stopping
            if callable(early_stopping):
                if early_stopping(train_losses, val_losses, early_stopping_params):
                    log.info(f'Stopping the training at epoch {epoch} due to early stopping.')
                    break

            if callable(select_model):
                if select_model(train_losses, val_losses, select_model_params):
                    best_model_epoch = epoch
                    selected_model_def.updateParameters(self._model)

            ## Visualize the training...
            self.visualizer.showTraining(epoch, train_losses, val_losses)
            self.visualizer.showWeightsInTrain(epoch=epoch)

        ## Visualize the training time
        end = time.time()
        self.visualizer.showTrainingTime(end - start)

        for key in self._model_def['Minimizers'].keys():
            self._training[key] = {'train': train_losses[key]}
            if n_samples_val > 0:
                self._training[key]['val'] = val_losses[key]
        self.visualizer.showEndTraining(num_of_epochs - 1, train_losses, val_losses)

        ## Select the model
        if callable(select_model):
            log.info(f'Selected the model at the epoch {best_model_epoch + 1}.')
            self._model = Model(selected_model_def)
        else:
            log.info('The selected model is the LAST model of the training.')

        ## Remove virtual states
        self._remove_virtual_states(connect, closed_loop)

        ## Get trained model from torch and set the model_def
        self._model_def.updateParameters(self._model)

#from 685
#from 840