# Extern packages
import random, torch, copy
import numpy as np

# Main operators
from nnodely.operators.composer import Composer
from nnodely.operators.trainer import Trainer
from nnodely.operators.loader import Loader
from nnodely.operators.validator import Validator
from nnodely.operators.exporter import Exporter

# nnodely packages
from nnodely.visualizer import EmptyVisualizer, TextVisualizer
from nnodely.exporter import EmptyExporter
from nnodely.basic.relation import NeuObj
from nnodely.support.utils import ReadOnlyDict, ParamDict, enforce_types, check

from nnodely.support.logger import logging, nnLogger
log = nnLogger(__name__, logging.INFO)


@enforce_types
def clearNames(names:str|list|None = None):
    NeuObj.clearNames(names)

class Modely(Composer, Trainer, Loader, Validator, Exporter):
    """
    Create the main object, the nnodely object, that will be used to create the network, train and export it.

    Parameters
    ----------
    visualizer : str, Visualizer, optional
        The visualizer to be used. Default is the 'Standard' visualizer.
    exporter : str, Exporter, optional
        The exporter to be used. Default is the 'Standard' exporter.
    seed : int, optional
        Set the seed for all the random modules inside the nnodely framework. Default is None.
    workspace : str
        The path of the workspace where all the exported files will be saved.
    log_internal : bool
        Whether or not save the logs. Default is False.
    save_history : bool
        Whether or not save the history. Default is False.

    Example
    -------
        >>> model = Modely()
    """
    @enforce_types
    def __init__(self, *,
                 visualizer:str|EmptyVisualizer|None = 'Standard',
                 exporter:str|EmptyExporter|None = 'Standard',
                 seed:int|None = None,
                 workspace:str|None = None,
                 log_internal:bool = False,
                 save_history:bool = False):

        ## Set the random seed for reproducibility
        if seed is not None:
            self.resetSeed(seed)

        # Visualizer
        if visualizer == 'Standard':
            self.visualizer = TextVisualizer(1)
        elif visualizer != None:
            self.visualizer = visualizer
        else:
            self.visualizer = EmptyVisualizer()
        self.visualizer.setModely(self)

        Composer.__init__(self)
        Loader.__init__(self)
        Trainer.__init__(self)
        Validator.__init__(self)
        Exporter.__init__(self, exporter, workspace, save_history=save_history)

        self._set_log_internal(log_internal)
        self._clean_log_internal()

    @property
    def internals(self):
        return ReadOnlyDict(self._internals)

    @property
    def neuralized(self):
        return self._neuralized

    @neuralized.setter
    def neuralized(self, value):
        raise AttributeError("Cannot modify read-only property 'neuralized' use neuralizeModel() instead.")

    @property
    def traced(self):
        return self._traced

    @traced.setter
    def traced(self, value):
        raise AttributeError("Cannot modify read-only property 'traced'.")

    @property
    def parameters(self):
        if self._neuralized:
            return ParamDict(self._model_def['Parameters'], self._model.all_parameters)
        else:
            return ParamDict(self._model_def['Parameters'])

    @property
    def constants(self):
        return ReadOnlyDict({key:value.detach().numpy().tolist() for key,value in self._model.all_constants})

    @property
    def states(self):
        return {key:value.detach().numpy().tolist() for key,value in self._states.items()}

    @property
    def json(self):
        return copy.deepcopy(self._model_def._ModelDef__json)

    @enforce_types
    def resetSeed(self, seed:int) -> None:
        """
        Resets the random seed for reproducibility.

        This method sets the seed for various random number generators used in the project to ensure reproducibility of results.

        :param seed: The seed value to be used for the random number generators.
        :type seed: int

        Example:
            >>> model = nnodely()
            >>> model.resetSeed(42)
        """
        torch.manual_seed(seed)  ## set the pytorch seed
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)  ## set the random module seed
        np.random.seed(seed)  ## set the numpy seed

    def trainAndAnalyze(self, *, test_dataset: str | list | dict | None = None, test_batch_size: int = 128, **kwargs):
        """
        Trains the model using the provided datasets and parameters. After training, it analyzes the results on the training, validation, and test datasets.

        Parameters
        ----------
        test_dataset : str or None, optional
            The name of the datasets used for test. Default is None.
        test_batch_size : int, optional
            The batch size for testing. Default is 1.
        models : list or None, optional
            A list of models to train. Default is None.
        train_dataset : str or None, optional
            The name of datasets to use for training. Default is None.
        validation_dataset : str or None, optional
            The name of datasets to use for validation. Default is None.
        dataset : str or None, optional
            The name of the datasets to use for training, validation and test.
        splits : list or None, optional
            A list of 3 elements specifying the percentage of splits for training, validation, and testing. The three elements must sum up to 100!
            The parameter splits is only used when dataset is not None
        closed_loop : dict or None, optional
            A dictionary specifying closed loop connections. The keys are input names and the values are output names. Default is None.
        connect : dict or None, optional
            A dictionary specifying connections. The keys are input names and the values are output names. Default is None.
        step : int or None, optional
            The step size for training. A big value will result in less data used for each epochs and a faster train. Default is None.
        prediction_samples : int or None, optional
            The size of the prediction horizon. Number of samples at each recurrent window Default is None.
        shuffle_data : bool or None, optional
            Whether to shuffle the data during training. Default is None.
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
            The number of epochs to train the model. Default is None.
        train_batch_size : int or None, optional
            The batch size for training. Default is None.
        val_batch_size : int or None, optional
            The batch size for validation. Default is None.
        optimizer : Optimizer or None, optional
            The optimizer to use for training. Default is None.
        lr : float or None, optional
            The learning rate. Default is None.
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

        Raises
        ------
        RuntimeError
            If no data is loaded or if there are no modules with learnable parameters.
        KeyError
            If the sample horizon is not positive.
        ValueError
            If an input or output variable is not in the model definition.

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
        ## Train the model
        self.trainModel(**kwargs)
        params = self.running_parameters

        minimize_gain = params['minimize_gain']
        closed_loop, connect, prediction_samples = params['closed_loop'], params['connect'], params['prediction_samples']

        if kwargs.get('train_dataset', None) is None:
            check(test_dataset is None, ValueError, 'If train_dataset is None, test_dataset must also be None.')
        else:
            params['test_tag'] = self._get_tag(test_dataset)
            params['XY_test'] = self._get_data(test_dataset)
            params['n_samples_test'] = next(iter(params['XY_test'].values())).size(0) if params['XY_test'] else 0
            params['test_indexes'] = self._get_batch_indexes(test_dataset, params['n_samples_test'], prediction_samples)

        ## Training set Results
        self.resultAnalysis(params['XY_train'], name = params['train_tag'], minimize_gain = minimize_gain,
                            closed_loop = closed_loop, connect = connect, prediction_samples = prediction_samples, step = params['train_step'], batch_size = params['train_batch_size'])
        
        ## Validation set Results
        if params['n_samples_val'] > 0:
            self.resultAnalysis(params['XY_val'], name = params['val_tag'], minimize_gain = minimize_gain,
                            closed_loop = closed_loop, connect = connect, prediction_samples = prediction_samples, step = params['val_step'], batch_size = params['val_batch_size'])
        else:
            log.warning("Validation dataset is empty. Skipping validation results analysis.")

        ## Test set Results
        if params['n_samples_test'] > 0:
            params['test_batch_size'] = self._clip_batch_size(len(params['test_indexes']), test_batch_size)
            params['test_step'] = self._clip_step(params['step'], params['test_indexes'], params['test_batch_size'])
            self.resultAnalysis(params['XY_test'], name = params['test_tag'], minimize_gain = minimize_gain,
                            closed_loop = closed_loop, connect = connect, prediction_samples = prediction_samples, step = params['test_step'], batch_size = test_batch_size)
        else:
            log.warning("Test dataset is empty. Skipping test results analysis.")

nnodely = Modely