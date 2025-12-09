import matplotlib.pyplot as plt
import numpy as np

from nnodely.visualizer.textvisualizer import TextVisualizer
from nnodely.layers.fuzzify import return_fuzzify
from nnodely.layers.parametricfunction import return_standard_inputs, return_function
from nnodely.support.utils import check
from mplplots import plots

class MPLNotebookVisualizer(TextVisualizer):
    def __init__(self, verbose = 1, *, test = False):
        super().__init__(verbose)
        self.test = test
        if self.test:
            plt.ion()

    def showEndTraining(self, epoch, train_losses, val_losses):
        train_tag = self.modely.running_parameters['train_tag']
        val_tag = self.modely.running_parameters['val_tag']
        for key in self.modely.json['Minimizers'].keys():
            fig = plt.figure()
            ax = fig.add_subplot(111)
            if val_losses:
                plots.plot_training(ax, f"Training on {train_tag} and {val_tag}", key, train_losses[key], val_losses[key])
            else:
                plots.plot_training(ax, f"Training on {train_tag}", key, train_losses[key])
        plt.show()

    def showResult(self, name_data):
        super().showResult(name_data)
        for key in self.modely.json['Minimizers'].keys():
            fig = plt.figure()
            ax = fig.add_subplot(111)
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

            plots.plot_results(ax, name_data, key, data_A,
                               data_B, data_idxs, self.modely._model_def['Info']["SampleTime"])
        plt.show()

    def showWeights(self, weights = None):
        pass

    def showFunctions(self, functions = None, xlim = None, num_points = 1000):
        check(self.modely.neuralized, ValueError, "The model has not been neuralized.")
        for fun, value in self.modely._model_def['Functions'].items():
            if fun in functions:
                if 'functions' in self.modely._model_def['Functions'][fun]:
                    x, activ_fun = return_fuzzify(value, xlim, num_points)
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    plots.plot_fuzzy(ax, fun, x, activ_fun, value['centers'])
                elif 'code':
                    function_inputs = return_standard_inputs(value, self.modely._model_def, xlim, num_points)
                    function_output, function_input_list = return_function(value, function_inputs)
                    if value['n_input'] == 2:
                        x0 = function_inputs[0].reshape(num_points, num_points).tolist()
                        x1 = function_inputs[1].reshape(num_points, num_points).tolist()
                        output = function_output.reshape(num_points, num_points).tolist()
                        params = []
                        for i, key in enumerate(value['params_and_consts']):
                            params += [function_inputs[i + value['n_input']].tolist()]
                        plots.plot_3d_function(plt, fun, x0, x1, params, output, function_input_list)
                    else:
                        x = function_inputs[0].reshape(num_points).tolist()
                        output = function_output.reshape(num_points).tolist()
                        params = []
                        for i, key in enumerate(value['params_and_consts']):
                            params += [function_inputs[i + value['n_input']].tolist()]
                        plots.plot_2d_function(plt, fun, x, params, output, function_input_list)
        plt.show()

    def closePlots(self):
        plt.close()



