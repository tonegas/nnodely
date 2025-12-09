import torch, warnings
import numpy as np

from nnodely.support.utils import ReadOnlyDict, get_batch_size

from nnodely.basic.loss import CustomLoss
from nnodely.operators.network import Network
from nnodely.support.utils import  check, TORCH_DTYPE, enforce_types

class Validator(Network):
    @enforce_types
    def __init__(self):
        check(type(self) is not Validator, TypeError, "Validator class cannot be instantiated directly")
        super().__init__()

        # Validation Parameters
        self.__performance = {}
        self.__prediction = {}

    @property
    def performance(self):
        return ReadOnlyDict(self.__performance)

    @property
    def prediction(self):
        return ReadOnlyDict(self.__prediction)

    @enforce_types
    def _analyze(self,
                  dataset: dict,
                  dataset_tag: str,
                  indexes: list = None,
                  minimize_gain: dict = {},
                  closed_loop: dict = {},
                  connect: dict = {},
                  prediction_samples: int | str = 0,
                  step: int = 0,
                  batch_size: int | None = None
                ) -> None:
        with torch.enable_grad() if self._get_gradient_on_inference() else torch.inference_mode():
            self._model.eval()
            self.__performance[dataset_tag] = {}
            self.__prediction[dataset_tag] = {}
            A = {}
            B = {}
            idxs = None
            total_losses = {}

            # Create the losses
            losses = {}
            for name, values in self._model_def['Minimizers'].items():
                losses[name] = CustomLoss(values['loss'])

            #data = self._get_data(dataset)
            n_samples = len(dataset[list(dataset.keys())[0]])

            batch_size = get_batch_size(n_samples, batch_size, prediction_samples)
            prediction_samples = self._setup_recurrent_variables(prediction_samples, closed_loop, connect)

            if prediction_samples >= 0:
                mandatory_inputs, non_mandatory_inputs = self._get_mandatory_inputs(connect,closed_loop)

                idxs = []
                for horizon_idx in range(prediction_samples + 1):
                    idxs.append([])
                for key, value in self._model_def['Minimizers'].items():
                    total_losses[key], A[key], B[key] = [], [], []
                    for horizon_idx in range(prediction_samples + 1):
                        A[key].append([])
                        B[key].append([])

                ## Update with virtual states
                self._model.update(closed_loop = closed_loop, connect = connect)
                self._recurrent_inference(dataset, indexes, batch_size, minimize_gain, prediction_samples,
                                          step, non_mandatory_inputs, mandatory_inputs, losses,
                                          total_losses = total_losses, A = A, B = B, idxs = idxs)

                for horizon_idx in range(prediction_samples + 1):
                    idxs[horizon_idx] = np.concatenate(idxs[horizon_idx])
                for key, value in self._model_def['Minimizers'].items():
                    for horizon_idx in range(prediction_samples + 1):
                        if A is not None:
                            A[key][horizon_idx] = np.concatenate(A[key][horizon_idx])
                        if B is not None:
                            B[key][horizon_idx] = np.concatenate(B[key][horizon_idx])
                    if total_losses is not None:
                        total_losses[key] = np.mean(total_losses[key])
            else:
                for key, value in self._model_def['Minimizers'].items():
                    total_losses[key], A[key], B[key] = [], [], []

                self._model.update(disconnect=True)
                self._inference(dataset, n_samples, batch_size, minimize_gain, losses,
                                total_losses = total_losses, A = A, B = B)

                for key, value in self._model_def['Minimizers'].items():
                    A[key] = np.concatenate(A[key])
                    B[key] = np.concatenate(B[key])
                    total_losses[key] = np.mean(total_losses[key])

            for ind, (key, value) in enumerate(self._model_def['Minimizers'].items()):
                A_np = np.array(A[key])
                B_np = np.array(B[key])
                self.__performance[dataset_tag][key] = {}
                self.__performance[dataset_tag][key][value['loss']] = np.mean(total_losses[key]).item()
                self.__performance[dataset_tag][key]['fvu'] = {}
                # Compute FVU
                residual = A_np - B_np
                error_var = np.var(residual)
                error_mean = np.mean(residual)
                #error_var_manual = np.sum((residual-error_mean) ** 2) / (len(self.__prediction['B'][ind]) - 0)
                #print(f"{key} var np:{new_error_var} and var manual:{error_var_manual}")
                with warnings.catch_warnings(record=True) as w:
                    self.__performance[dataset_tag][key]['fvu']['A'] = (error_var / np.var(A_np)).item()
                    self.__performance[dataset_tag][key]['fvu']['B'] = (error_var / np.var(B_np)).item()
                    if w and np.var(A_np) == 0.0 and  np.var(B_np) == 0.0:
                        self.__performance[dataset_tag][key]['fvu']['A'] = np.nan
                        self.__performance[dataset_tag][key]['fvu']['B'] = np.nan
                self.__performance[dataset_tag][key]['fvu']['total'] = np.mean([self.__performance[dataset_tag][key]['fvu']['A'],self.__performance[dataset_tag][key]['fvu']['B']]).item()
                # Compute AIC
                #normal_dist = norm(0, error_var ** 0.5)
                #probability_of_residual = normal_dist.pdf(residual)
                #log_likelihood_first = sum(np.log(probability_of_residual))
                p1 = -len(residual)/2.0*np.log(2*np.pi)
                with warnings.catch_warnings(record=True) as w:
                    p2 = -len(residual)/2.0*np.log(error_var)
                    p3 = -1 / (2.0 * error_var) * np.sum(residual ** 2)
                    if w and p2 == np.float32(np.inf) and p3 == np.float32(-np.inf):
                        p2 = p3 = 0.0
                log_likelihood = p1+p2+p3
                #print(f"{key} log likelihood second mode:{log_likelihood} = {p1}+{p2}+{p3} first mode: {log_likelihood_first}")
                total_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
                #print(f"{key} total_params:{total_params}")
                aic = - 2 * log_likelihood + 2 * total_params
                #print(f"{key} aic:{aic}")
                self.__performance[dataset_tag][key]['aic'] = {'value':aic,'total_params':total_params,'log_likelihood':log_likelihood}
                # Prediction and target
                self.__prediction[dataset_tag][key] = {}
                self.__prediction[dataset_tag][key]['A'] = A_np.tolist()
                self.__prediction[dataset_tag][key]['B'] = B_np.tolist()

            if idxs is not None:
                self.__prediction[dataset_tag]['idxs'] = np.array(idxs).tolist()
            self.__performance[dataset_tag]['total'] = {}
            self.__performance[dataset_tag]['total']['mean_error'] = np.mean([value for key,value in total_losses.items()])
            self.__performance[dataset_tag]['total']['fvu'] = np.mean([self.__performance[dataset_tag][key]['fvu']['total'] for key in self._model_def['Minimizers'].keys()])
            self.__performance[dataset_tag]['total']['aic'] = np.mean([self.__performance[dataset_tag][key]['aic']['value']for key in self._model_def['Minimizers'].keys()])

        self.visualizer.showResult(dataset_tag)

    # @enforce_types
    # def analyzeModel(self,
    #                    dataset: str | list | dict | None = None, *,
    #                    splits: list | None = None,
    #                    name: str | None = None,
    #                    minimize_gain: dict = {},
    #                    closed_loop: dict = {},
    #                    connect: dict = {},
    #                    prediction_samples: int | str = 0,
    #                    step: int = 0,
    #                    batch_size: int | None = None
    #                    ) -> None:
    #     """
    #     The function is used to analyze the performance of the model on the provided dataset.

    #     Parameters
    #     ----------
    #     dataset : str | list | dict
    #         Dataset to analyze the performance of the model on.
    #     splits : list or None, optional
    #         A list of 3 elements specifying the percentage of splits for training, validation, and testing.
    #         The three elements must sum up to 100! default is [100, 0, 0]
    #     name : str or None
    #         Label to be used in the plots
    #     minimize_gain : dict
    #         A dictionary specifying the gain for each minimization loss function.
    #     closed_loop : dict or None, optional
    #         A dictionary specifying closed loop connections. The keys are input names and the values are output names. Default is None.
    #     connect : dict or None, optional
    #         A dictionary specifying connections. The keys are input names and the values are output names. Default is None.
    #     step : int or None, optional
    #         The step size to analyze the model on the provided dataset. A big value will result in less data used for each epochs and a faster train. Default is None.
    #     prediction_samples : int or None, optional
    #         The size of the prediction horizon. Number of samples at each recurrent window Default is None.
    #     batch_size :
    #         The batch size use for analyse the performance of the model on the provided dataset.


    #     """
    #     # Get the dataset if is None take all datasets
    #     if dataset is None:
    #         dataset = list(self._data.keys())

    #     data = self._get_data(dataset) 
    #     n_samples = len(data[list(data.keys())[0]])
    #     data_tag = self._get_tag(dataset) if name is None else name
    #     indexes = list(range(n_samples))
    #     dataset_name = data_tag.replace('_train','').replace('_val','').replace('_test','')
    #     if prediction_samples > 0 and dataset_name in self._multifile.keys():
    #         forbidden_idxs = []
    #         for i in self._multifile[dataset_name]:
    #             if i < indexes[-1]:
    #                 forbidden_idxs.extend(range((i) - prediction_samples, (i), 1))
    #         indexes = [idx for idx in indexes if idx not in forbidden_idxs]
    #         indexes = indexes[:-prediction_samples]

    #     # If splits is None it uses all the dataset
    #     if splits is None:
    #         data = self._get_data(dataset)
    #         self._analyze(data, data_tag, indexes, minimize_gain, closed_loop, connect, prediction_samples, step, batch_size)
    #     else:
    #         data_train, data_val, data_test = self._setup_dataset(None, None, None, dataset, splits)
    #         n_samples_val = next(iter(data_val.values())).size(0) if data_val else 0
    #         n_samples_test = next(iter(data_test.values())).size(0) if data_test else 0

    #         self._analyze(data_train, f"{data_tag}_train", indexes, minimize_gain, closed_loop, connect, prediction_samples, step, batch_size)
    #         if n_samples_val > 0:
    #             self._analyze(data_val, f"{data_tag}_val", indexes, minimize_gain, closed_loop, connect, prediction_samples, step, batch_size)
    #         if n_samples_test > 0:
    #             self._analyze(data_test, f"{data_tag}_test", indexes, minimize_gain, closed_loop, connect, prediction_samples, step, batch_size)


    @enforce_types
    def analyzeModel(self,
                dataset: str | list | dict | None = None, *,
                tag: str | None = None,
                splits: list | None = None,
                minimize_gain: dict = {},
                closed_loop: dict = {},
                connect: dict = {},
                prediction_samples: int | str = 0,
                step: int = 0,
                batch_size: int | None = None
                ) -> None:
        """
        The function is used to analyze the performance of the model on the provided dataset.

        Parameters
        ----------
        dataset : str | list | dict
            Dataset to analyze the performance of the model on.
        tag : str or None
            Label to be used in the plots
        minimize_gain : dict
            A dictionary specifying the gain for each minimization loss function.
        closed_loop : dict or None, optional
            A dictionary specifying closed loop connections. The keys are input names and the values are output names. Default is None.
        connect : dict or None, optional
            A dictionary specifying connections. The keys are input names and the values are output names. Default is None.
        step : int or None, optional
            The step size to analyze the model on the provided dataset. A big value will result in less data used for each epochs and a faster train. Default is None.
        prediction_samples : int or None, optional
            The size of the prediction horizon. Number of samples at each recurrent window Default is None.
        batch_size :
            The batch size use for analyse the performance of the model on the provided dataset.


        """
        # Get the dataset if is None take all datasets
        if tag is None:
            tag = dataset if isinstance(dataset, str) else 'default'
        if dataset is None:
            dataset = list(self._data.keys())

        if splits: ## splits is used
            ## Get the dataset
            XY_train, XY_val, XY_test = self._setup_dataset(None, None, None, dataset, splits)
            n_samples_train = next(iter(XY_train.values())).size(0)
            n_samples_val = next(iter(XY_val.values())).size(0) if XY_val else 0
            n_samples_test = next(iter(XY_test.values())).size(0) if XY_test else 0
            tag = self._get_tag(dataset)
            train_tag = f"{tag}_train"
            val_tag = f"{tag}_val" if n_samples_val > 0 else None
            test_tag = f"{tag}_test" if n_samples_test > 0 else None

            train_indexes, val_indexes = [], []
            train_indexes = self._get_batch_indexes(dataset, n_samples_train, prediction_samples)
            check(len(train_indexes) > 0, ValueError,
                  'The number of valid train samples is less than the number of prediction samples.')
            if n_samples_val > 0:
                val_indexes = self._get_batch_indexes(dataset, n_samples_train + n_samples_val, prediction_samples)
                val_indexes = [i - n_samples_train for i in val_indexes if i >= n_samples_train]
            if n_samples_test > 0:
                test_indexes = self._get_batch_indexes(dataset, n_samples_train + n_samples_val + n_samples_test, prediction_samples)
                test_indexes = [i - (n_samples_train+n_samples_val)for i in test_indexes if i >= (n_samples_train+n_samples_val)]
            ## Training set Results
            self._analyze(XY_train, dataset_tag=train_tag, indexes=train_indexes, minimize_gain=minimize_gain,
                            closed_loop=closed_loop, connect=connect, prediction_samples=prediction_samples,
                            step=step, batch_size=batch_size)
            ## Validation set Results
            if n_samples_val > 0:
                self._analyze(XY_val, dataset_tag=val_tag, indexes=val_indexes, minimize_gain=minimize_gain,
                                closed_loop=closed_loop, connect=connect, prediction_samples=prediction_samples,
                                step=step, batch_size=batch_size)
            ## Test set Results
            if n_samples_test > 0:
                self._analyze(XY_test, dataset_tag=test_tag, indexes=test_indexes, minimize_gain=minimize_gain,
                                closed_loop=closed_loop, connect=connect, prediction_samples=prediction_samples,
                                step=step, batch_size=batch_size)
        else:
            data = self._get_data(dataset) 
            n_samples = next(iter(data.values())).size(0)
            indexes = self._get_batch_indexes(dataset, n_samples, prediction_samples)
            self._analyze(data, tag, indexes, minimize_gain, closed_loop, connect, prediction_samples, step, batch_size)