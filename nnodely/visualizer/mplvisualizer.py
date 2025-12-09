import subprocess, json, os, importlib
import numpy as np

from nnodely.visualizer.textvisualizer import TextVisualizer
from nnodely.layers.fuzzify import return_fuzzify
from nnodely.layers.parametricfunction import return_standard_inputs, return_function
from nnodely.support.utils import check
from nnodely.basic.modeldef import ModelDef

from nnodely.support.logger import logging, nnLogger
log = nnLogger(__name__, logging.INFO)

def get_library_path(library_name):
    spec = importlib.util.find_spec(library_name)
    if spec is None:
        raise ImportError(f"Library {library_name} not found")
    return os.path.dirname(spec.origin)

class MPLVisualizer(TextVisualizer):
    def __init__(self, verbose = 1):
        super().__init__(verbose)
        # Path to the data visualizer script
        import signal
        import sys
        get_library_path('nnodely')
        self.__training_visualizer_script = os.path.join(get_library_path('nnodely'),'visualizer','dynamicmpl','trainingplot.py')
        self.__time_series_visualizer_script = os.path.join(get_library_path('nnodely'),'visualizer','dynamicmpl','resultsplot.py')
        self.__fuzzy_visualizer_script = os.path.join(get_library_path('nnodely'),'visualizer','dynamicmpl','fuzzyplot.py')
        self.__function_visualizer_script = os.path.join(get_library_path('nnodely'),'visualizer','dynamicmpl','functionplot.py')
        self.__process_training = {}
        self.__process_results = {}
        self.__process_function = {}
        def signal_handler(sig, frame):
            for key in self.__process_training.keys():
                self.__process_training[key].terminate()
                self.__process_training[key].wait()
            for name_data in self.__process_results.keys():
                for key in self.__process_results[name_data].keys():
                    self.__process_results[name_data][key].terminate()
                    self.__process_results[name_data][key].wait()
            self.__process_results = {}
            for key in self.__process_function.keys():
                self.__process_function[key].terminate()
                self.__process_functios[key].wait()
            sys.exit()

        signal.signal(signal.SIGINT, signal_handler)

    def showStartTraining(self):
        pass

    def showTraining(self, epoch, train_losses, val_losses):
        if epoch == 0:
            for key in self.__process_training.keys():
                if self.__process_training[key] is not None and self.__process_training[key].poll() is None:
                    self.__process_training[key].terminate()
                    self.__process_training[key].wait()
                self.__process_training[key] = None

            self.__process_training = {}
            for key in self.modely._model_def['Minimizers'].keys():
                self.__process_training[key] = subprocess.Popen(['python', self.__training_visualizer_script], stdin=subprocess.PIPE, text=True)

        num_of_epochs = self.modely.running_parameters['num_of_epochs']
        train_tag = self.modely.running_parameters['train_tag']
        val_tag = self.modely.running_parameters['val_tag']
        if epoch+1 <= num_of_epochs:
            for key in self.modely._model_def['Minimizers'].keys():
                if val_losses:
                    val_loss = val_losses[key][epoch]
                    title = f"Training on {train_tag} and {val_tag}"
                else:
                    val_loss = []
                    title = f"Training on {train_tag}"
                data = {"title":title, "key": key, "last": num_of_epochs - (epoch + 1), "epoch": epoch,
                        "train_losses": train_losses[key][epoch], "val_losses": val_loss}
                try:
                    # Send data to the visualizer process
                    self.__process_training[key].stdin.write(f"{json.dumps(data)}\n")
                    self.__process_training[key].stdin.flush()
                except BrokenPipeError:
                    self.closeTraining()
                    log.warning("The visualizer process has been closed.")

        if epoch+1 == num_of_epochs:
            for key in self.modely._model_def['Minimizers'].keys():
                self.__process_training[key].stdin.close()

    def showResult(self, name_data):
        super().showResult(name_data)
        check(name_data in self.modely.performance, ValueError, f"Results not available for {name_data}.")
        if name_data in self.__process_results:
            for key in self.modely._model_def['Minimizers'].keys():
                if key in self.__process_results[name_data] and self.__process_results[name_data][key].poll() is None:
                    self.__process_results[name_data][key].terminate()
                    self.__process_results[name_data][key].wait()
                self.__process_results[name_data][key] = None
        self.__process_results[name_data] = {}

        for key in self.modely._model_def['Minimizers'].keys():
            # Start the data visualizer process
            self.__process_results[name_data][key] = subprocess.Popen(['python', self.__time_series_visualizer_script], stdin=subprocess.PIPE,
                                                    text=True)
            np_data_A = np.array(self.modely.prediction[name_data][key]['A'])
            if len(np_data_A.shape) > 3 and np_data_A.shape[1] > 30:
                np_data_B = np.array(self.modely.prediction[name_data][key]['B'])
                indices = np.linspace(0, np_data_A.shape[1] - 1, 30, dtype=int)
                data_A = np_data_A[:, indices, :, :].tolist()
                data_B = np_data_B[:, indices, :, :].tolist()
                data_idxs = np.array(self.modely.prediction[name_data]['idxs'])[:,indices].tolist()
            else:
                data_A = self.modely.prediction[name_data][key]['A']
                data_B = self.modely.prediction[name_data][key]['B']
                data_idxs = self.modely.prediction[name_data]['idxs'] if len(np_data_A.shape) > 3 else None

            data = {"name_data": name_data,
                    "key": key,
                    "performance": self.modely.performance[name_data][key],
                    "prediction_A": data_A,
                    "prediction_B": data_B,
                    "data_idxs": data_idxs,
                    "sample_time": self.modely._model_def['Info']["SampleTime"]}
            try:
                # Send data to the visualizer process
                self.__process_results[name_data][key].stdin.write(f"{json.dumps(data)}\n")
                self.__process_results[name_data][key].stdin.flush()
                self.__process_results[name_data][key].stdin.close()
            except BrokenPipeError:
                self.closeResult(self, name_data)
                log.warning(f"The visualizer {name_data} process has been closed.")

    def showWeights(self, weights = None):
        pass

    def showFunctions(self, functions = None, xlim = None, num_points = 1000):
        check(self.modely.neuralized, ValueError, "The model has not been neuralized.")
        for key, value in self.modely._model_def['Functions'].items():
            if key in functions:
                if key in self.__process_function and self.__process_function[key].poll() is None:
                    self.__process_function[key].terminate()
                    self.__process_function[key].wait()

                if 'functions' in self.modely._model_def['Functions'][key]:
                    x, activ_fun = return_fuzzify(value, xlim, num_points)
                    data = {"name": key,
                            "x": x,
                            "y": activ_fun,
                            "chan_centers": value['centers']}
                    # Start the data visualizer process
                    self.__process_function[key] = subprocess.Popen(['python', self.__fuzzy_visualizer_script],
                                                                  stdin=subprocess.PIPE,
                                                                  text=True)
                elif 'code':
                    model_def = ModelDef(self.modely._model_def)
                    model_def.updateParameters(self.modely._model)
                    function_inputs = return_standard_inputs(value, model_def, xlim, num_points)
                    function_output, function_input_list = return_function(value, function_inputs)

                    data = {"name": key}
                    if value['n_input'] == 2:
                        data['x0'] = function_inputs[0].reshape(num_points, num_points).tolist()
                        data['x1'] = function_inputs[1].reshape(num_points, num_points).tolist()
                        data['output'] = function_output.reshape(num_points, num_points).tolist()
                    else:
                        data['x0'] = function_inputs[0].reshape(num_points).tolist()
                        data['output'] = function_output.reshape(num_points).tolist()
                    data['params'] = []
                    for i, key in enumerate(value['params_and_consts']):
                        data['params'] += [function_inputs[i+value['n_input']].tolist()]
                    data['input_names'] = function_input_list

                    # Start the data visualizer process
                    self.__process_function[key] = subprocess.Popen(['python', self.__function_visualizer_script],
                                                                  stdin=subprocess.PIPE,
                                                                  text=True)
                try:
                    # Send data to the visualizer process
                    self.__process_function[key].stdin.write(f"{json.dumps(data)}\n")
                    self.__process_function[key].stdin.flush()
                    self.__process_function[key].stdin.close()
                except BrokenPipeError:
                    self.closeFunctions()
                    log.warning(f"The visualizer {functions} process has been closed.")

    def closeFunctions(self, functions = None):
        if functions is None:
            for key in self.__process_function.keys():
                self.__process_function[key].terminate()
                self.__process_function[key].wait()
            self.__process_function = {}
        else:
            for key in functions:
                self.__process_function[key].terminate()
                self.__process_function[key].wait()
                self.__process_function.pop(key)

    def closeTraining(self, minimizer = None):
        if minimizer is None:
            for key in self.modely._model_def['Minimizers'].keys():
                if key in self.__process_training and self.__process_training[key].poll() is None:
                    self.__process_training[key].terminate()
                    self.__process_training[key].wait()
                self.__process_training[key] = None
        else:
            self.__process_training[minimizer].terminate()
            self.__process_training[minimizer].wait()
            self.__process_training.pop(minimizer)

    def closeResult(self, name_data = None, minimizer = None):
        if name_data is None:
            check(minimizer is None, ValueError, "If name_data is None, minimizer must be None.")
            for name_data in self.__process_results.keys():
                for key in self.__process_results[name_data].keys():
                    self.__process_results[name_data][key].terminate()
                    self.__process_results[name_data][key].wait()
            self.__process_results = {}
        else:
            if minimizer is None:
                for key in self.__process_results[name_data].keys():
                    self.__process_results[name_data][key].terminate()
                    self.__process_results[name_data][key].wait()
                self.__process_results[name_data] = {}
            else:
                self.__process_results[name_data][minimizer].terminate()
                self.__process_results[name_data][minimizer].wait()
                self.__process_results[name_data].pop(minimizer)





