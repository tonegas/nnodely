import os, random

import pandas as pd
import numpy as np
import pandas.api.types as ptypes
from collections.abc import Sequence, Callable

from nnodely.basic.relation import check_names
from nnodely.operators.network import Network
from nnodely.support.utils import check, enforce_types

from nnodely.support.logger import logging, nnLogger
log = nnLogger(__name__, logging.WARNING)

class Loader(Network):
    @enforce_types
    def __init__(self):
        check(type(self) is not Loader, TypeError, "Loader class cannot be instantiated directly")
        super().__init__()

        # Dataaset Parameters
        self.__n_datasets = 0
        self.__datasets_loaded = set()

    @enforce_types
    def getSamples(self, dataset:str, *, index:int|None = None, window:int=1) -> dict:
        """
        Retrieves a window of samples from a given dataset.

        Parameters
        ----------
        dataset : str
            The name of the dataset to retrieve samples from.
        index : int, optional
            The starting index of the samples. If None, a random index is chosen. Default is None.
        window : int, optional
            The number of consecutive samples to retrieve. Default is 1.

        Returns
        -------
        dict
            A dictionary containing the retrieved samples. The keys are input names, and the values are lists of samples.

        Raises
        ------
        ValueError
            If the dataset is not loaded.

        Examples
        --------
        .. image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/tonegas/nnodely/blob/main/examples/dataset.ipynb
            :alt: Open in Colab

        Example usage:
            >>> model = Modely()
            >>> model.loadData('dataset_name')
            >>> samples = model.getSamples('dataset_name', index=10, window=5)
        """
        if index is None:
            index = random.randint(0, self._num_of_samples[dataset] - window)
        check(self._data_loaded, ValueError, 'The Dataset must first be loaded using <loadData> function!')
        if self._data_loaded:
            result_dict = {}
            for key in self._model_def['Inputs'].keys():
                result_dict[key] = []
            for idx in range(window):
                for key ,samples in self._data[dataset].items():
                    if key in self._model_def['Inputs'].keys():
                        result_dict[key].append(samples[index+idx])
            return result_dict

    @enforce_types
    def filterData(self, filter_function:Callable, dataset_name:str|None = None) -> None:
        """
        Filters the data in the dataset using the provided filter function.

        Parameters
        ----------
        filter_function : Callable
            A function that takes a sample as input and returns True if the sample should be kept, and False if it should be removed.
        dataset_name : str or None, optional
            The name of the dataset to filter. If None, all datasets are filtered. Default is None.

        Examples
        --------
        .. image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/tonegas/nnodely/blob/main/examples/dataset.ipynb
            :alt: Open in Colab

        Example usage:
            >>> model = Modely()
            >>> model.loadData('dataset_name', 'path/to/data')
            >>> def filter_fn(sample):
            >>>     return sample['input1'] > 0
            >>> model.filterData(filter_fn, 'dataset_name')
        """
        idx_to_remove = []
        if dataset_name is None:
            for name in self._data.keys():
                dataset = self._data[name]
                n_samples = len(dataset[list(dataset.keys())[0]])

                data_for_filter = []
                for i in range(n_samples):
                    new_sample = {key: val[i] for key, val in dataset.items()}
                    data_for_filter.append(new_sample)

                for idx, sample in enumerate(data_for_filter):
                    if not filter_function(sample):
                        idx_to_remove.append(idx)

                for key in self._data[name].keys():
                    self._data[name][key] = np.delete(self._data[name][key], idx_to_remove, axis=0)
                    self._num_of_samples[name] = self._data[name][key].shape[0]
                self.visualizer.showDataset(name=name)

        else:
            dataset = self._data[dataset_name]
            n_samples = len(dataset[list(dataset.keys())[0]])

            data_for_filter = []
            for i in range(n_samples):
                new_sample = {key: val[i] for key, val in dataset.items()}
                data_for_filter.append(new_sample)

            for idx, sample in enumerate(data_for_filter):
                if not filter_function(sample):
                    idx_to_remove.append(idx)

            for key in self._data[dataset_name].keys():
                self._data[dataset_name][key] = np.delete(self._data[dataset_name][key], idx_to_remove, axis=0)
                self._num_of_samples[dataset_name] = self._data[dataset_name][key].shape[0]
            self.visualizer.showDataset(name=dataset_name)

    @enforce_types
    def resamplingData(self, df:pd.DataFrame, *, scale:float = 1e9) -> None:
        """
        Resamples the DataFrame to a specified sample time.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to resample.
        scale : float, optional
            The scale factor to convert the sample time to nanoseconds. Default is 1e9

        Returns
        -------
        pd.DataFrame
            The resampled DataFrame.

        Raises
        ------
        TypeError
            If the DataFrame does not contain a time column or if the time column is not in datetime format.

        Examples
        --------
        .. image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/tonegas/nnodely/blob/main/examples/dataset.ipynb
            :alt: Open in Colab

        Example usage:
            >>> model = Modely()
            >>> df = pd.DataFrame({'time': np.array(range(60), dtype=np.float32),'x': np.array(10*[10] + 20*[20] + 30*[30], dtype=np.float32)})
            >>> resampled_df = model.resamplingData(df, scale=1e9)
        """
        sample_time_ns = int(self._model_def.getSampleTime() * scale)
        method = 'linear'
        if type(df.index) is pd.DatetimeIndex:
            df = df.resample(f"{sample_time_ns}ns").interpolate(method=method)
        elif 'time' in df.columns:
            if not ptypes.is_datetime64_any_dtype(df['time']):
                df['time'] = pd.to_datetime(df['time'], unit='s')
            df = df.set_index('time', drop=True)
            df = df.resample(f"{sample_time_ns}ns").interpolate(method=method)
        else:
            raise TypeError("No time column found in the DataFrame. Please provide a time column for resampling.")
        return df
    
    @enforce_types
    def __get_format_idxs(self, format: list | None = None) -> dict:
        model_inputs = self._model_def['Inputs']
        format_idx = {}
        idx = 0
        for item in format:
            if isinstance(item, tuple):
                for key in item:
                    if key not in model_inputs.keys():
                        idx += 1
                        break
                    n_cols = model_inputs[key]['dim']
                    format_idx[key] = (idx, idx + n_cols)
                idx += n_cols
            else:
                if item not in model_inputs.keys():
                    idx += 1
                    continue
                n_cols = model_inputs[item]['dim']
                format_idx[item] = (idx, idx + n_cols)
                idx += n_cols
        return format_idx
    
    @enforce_types
    def __get_files(self, folder:str) -> list:
        try:
            _, _, files = next(os.walk(folder))
            files.sort()
        except StopIteration as e:
            check(False, StopIteration, f'ERROR: The path "{folder}" does not exist!')
            return []
        return files
    
    @enforce_types
    def __stack_arrays(self, data: dict) -> tuple:
        ## Convert lists to numpy arrays
        num_of_samples = {}
        for key in data:
            data[key] = np.stack(data[key])
            if self._model_def['Inputs'][key]['dim'] > 1:
                data[key] = np.array(data[key].tolist(), dtype=np.float64)
            if data[key].ndim == 2:  ## Add the sample dimension
                data[key] = np.expand_dims(data[key], axis=-1)
            if data[key].ndim > 3:
                data[key] = np.squeeze(data[key], axis=1)
            num_of_samples[key] = data[key].shape[0]
        return num_of_samples

    @enforce_types
    def loadData(self, name:str,
                 source: str | dict | pd.DataFrame, *,
                 format: list | None = None,
                 skiplines: int = 0,
                 delimiter: str = ',',
                 header: int | str | Sequence | None = None,
                 resampling: bool = False
                 ) -> None:
        """
        Loads data into the model. The data can be loaded from a directory path containing the csv files or from a crafted dataset.

        Parameters
        ----------
        name : str
            The name of the dataset.
        source : str or list or pd.DataFrame
            The source of the data. Can be a directory path containing the csv files or a custom dataset provided as a dictionary or a pandas DataFrame.
        format : list or None, optional
            The format of the data. When loading multiple csv files the format parameter will define how to read each column of the file. Default is None.
        skiplines : int, optional
            The number of lines to skip at the beginning of the file. Default is 0.
        delimiter : str, optional
            The delimiter used in the data files. Default is ','.
        header : list or None, optional
            The header of the data files. Default is None.

        Raises
        ------
        ValueError
            If the network is not neuralized.
            If the delimiter is not valid.

        Examples
        --------
        .. image:: https://colab.research.google.com/assets/colab-badge.svg
            :target: https://colab.research.google.com/github/tonegas/nnodely/blob/main/examples/dataset.ipynb
            :alt: Open in Colab

        Example - load data from files:
            >>> x = Input('x')
            >>> y = Input('y')
            >>> out = Output('out',Fir(x.tw(0.05)))
            >>> test = Modely(visualizer=None)
            >>> test.addModel('example_model', out)
            >>> test.neuralizeModel(0.01)
            >>> data_struct = ['x', '', 'y']
            >>> test.loadData(name='example_dataset', source='path/to/data', format=data_struct)

        Example - load data from a crafted dataset:
            >>> x = Input('x')
            >>> y = Input('y')
            >>> out = Output('out',Fir(x.tw(0.05)))
            >>> test = Modely(visualizer=None)
            >>> test.addModel('example_model', out)
            >>> test.neuralizeModel(0.01)
            >>> data_x = np.array(range(10))
            >>> dataset = {'x': data_x, 'y': (2*data_x)}
            >>> test.loadData(name='example_dataset',source=dataset)
        """
        check(self.neuralized, ValueError, "The network is not neuralized.")
        check(delimiter in ['\t', '\n', ';', ',', ' '], ValueError, 'delimiter not valid!')

        json_inputs = self._model_def['Inputs']
        ## Initialize the dictionary containing the data
        check_names(name, self._data.keys(), f"Dataset")
        self._data[name] = {}

        if type(source) is str:  ## we have a directory path containing the files
            ## collect column indexes
            format_idx = self.__get_format_idxs(format)
            ## Initialize each input key
            for key in format_idx.keys():
                self._data[name][key] = []
            ## obtain the file names
            files = self.__get_files(source)
            self._file_count = len(files)
            if self._file_count > 1:  ## Multifile
                self._multifile[name] = []

            ## Cycle through all the files
            for file in files:
                try:
                    ## read the csv
                    df = pd.read_csv(os.path.join(source, file), skiprows=skiplines, delimiter=delimiter, header=header)
                    if not all(df.iloc[0].apply(lambda x: isinstance(x, (int, float)))):
                        log.warning(f"The file {file} does not contain a numerical column.")
                    ## Resampling if the time column is provided (must be a Datetime object)
                    if resampling:
                        self.resamplingData(df)
                except:
                    log.warning(f'Cannot read file {os.path.join(source, file)}')
                    continue
                if self._file_count > 1:
                    self._multifile[name].append((self._multifile[name][-1] + (len(df) - self._max_n_samples + 1)) if self._multifile[name] else len(df) - self._max_n_samples + 1)
                ## Cycle through all the windows
                for key, idxs in format_idx.items():
                    back, forw = self._input_ns_backward[key], self._input_ns_forward[key]
                    ## Save as numpy array the data
                    data = df.iloc[:, idxs[0]:idxs[1]].to_numpy()
                    self._data[name][key] += [data[i - back:i + forw] for i in range(self._max_samples_backward, len(df) - self._max_samples_forward + 1)]
        else:  ## we have a crafted dataset
            self._file_count = 1
            if isinstance(source, dict):
                # Merge a list of inputs into a single dictionary
                for key in json_inputs.keys():
                    if key not in source.keys():
                        continue
                    self._data[name][key] = []  ## Initialize the dataset
                    back, forw = self._input_ns_backward[key], self._input_ns_forward[key]
                    for idx in range(len(source[key]) - self._max_n_samples + 1):
                        self._data[name][key].append(source[key][idx + (self._max_samples_backward - back):idx + (self._max_samples_backward + forw)])
            else:
                if resampling:
                    source = self.resamplingData(source)
                for key in json_inputs.keys():
                    if key not in source.columns:
                        continue
                    self._data[name][key] = []  ## Initialize the dataset
                    back, forw = self._input_ns_backward[key], self._input_ns_forward[key]
                    for idx in range(len(source) - self._max_n_samples + 1):
                        window = source[key].iloc[idx + (self._max_samples_backward - back):idx + (self._max_samples_backward + forw)]
                        self._data[name][key].append(window.to_numpy())

        ## Convert lists to numpy arrays
        num_of_samples = self.__stack_arrays(self._data[name])
        # Check dim of the samples
        check(len(set(num_of_samples.values())) == 1, ValueError, f"The number of the sample of the dataset {name} are not the same for all input in the dataset: {num_of_samples}")
        self._num_of_samples[name] = num_of_samples[list(num_of_samples.keys())[0]]
        ## Set the Loaded flag to True
        self._data_loaded = True
        ## Update the number of datasets loaded
        self.__n_datasets = len(self._data.keys())
        self.__datasets_loaded.add(name)
        ## Show the dataset
        self.visualizer.showDataset(name=name)