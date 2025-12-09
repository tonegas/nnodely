import sys, io, os, unittest, torch
import numpy as np

from nnodely import *
from nnodely.basic.relation import NeuObj
from nnodely.support.logger import logging, nnLogger
from nnodely.support.jsonutils import plot_structure

log = nnLogger(__name__, logging.ERROR)
log.setAllLevel(logging.ERROR)

sys.path.append(os.getcwd())

# 3 Tests
# Test of visualizers

class ModelyTestVisualizer(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        NeuObj.clearNames()
        super(ModelyTestVisualizer, self).__init__(*args, **kwargs)

        self.x = x = Input('x')
        self.y = y = Input('y')
        self.z = z = Input('z')
        self.a = a = Input('a', dimensions=2)
        self.b = b = Input('b', dimensions=2)

        ## create the relations
        def myFun(K1, p1, p2):
            return K1 * p1 * p2

        P_time = Parameter('P_time', dimensions=2, sw=5, values=[[0,0],[-0.1,0.1],[-0.2,0.2],[-0.3,0.3],[-0.4,0.4]])
        K_x = Parameter('k_x', dimensions=1, tw=1, init='init_constant', init_params={'value': 1})
        K_y = Parameter('k_y', dimensions=1, tw=1)
        w = Parameter('w', dimensions=1, tw=1, init='init_constant', init_params={'value': 1})
        t = Parameter('t', dimensions=1, tw=1)
        c_v = Constant('c_v', tw=1, values=[[1], [2]])
        c = 5
        w_5 = Parameter('w_5', dimensions=1, tw=5)
        t_5 = Parameter('t_5', dimensions=1, tw=5)
        c_5 = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
        c_5_2 = Constant('c_5_2', tw=5, values=c_5)
        parfun_x = ParamFun(myFun, parameters_and_constants=[K_x,c_v])
        parfun_y = ParamFun(myFun, parameters_and_constants=[K_y])
        parfun_zz = ParamFun(myFun)
        parfun_2d = ParamFun(myFun, parameters_and_constants=[K_x, K_x])
        parfun_3d = ParamFun(myFun, parameters_and_constants=[K_x])
        fir_w = Fir(W=w_5)(x.tw(5))
        fir_t = Fir(W=t_5)(y.tw(5))
        time_part = TimePart(x.tw(5), i=1, j=3)
        sample_select = SampleSelect(x.sw(5), i=1)

        def fuzzyfun(x):
            return torch.sin(x)

        def fuzzyfunth(x):
            return torch.tanh(x)

        fuzzy = Fuzzify(output_dimension=4, range=[0, 4], functions=fuzzyfun)(x.tw(1))
        fuzzyTriang = Fuzzify(centers=[1, 2, 3, 7])(x.tw(1))
        fuzzyRect = Fuzzify(centers=[1, 2, 3, 7], functions='Rectangular')(x.tw(1))
        fuzzyList = Fuzzify(centers=[1, 3, 2, 7], functions=[fuzzyfun,fuzzyfunth])(x.tw(1))
        self.stream = fuzzyList

        self.out = Output('out', Fir(parfun_x(x.tw(1)) + parfun_y(y.tw(1), c_v)))
        self.out2 = Output('out2', Add(w, x.tw(1)) + Add(t, y.tw(1)) + Add(w, c))
        self.out3 = Output('out3', Add(fir_w, fir_t))
        self.out4 = Output('out4', Linear(output_dimension=1)(fuzzy+fuzzyTriang+fuzzyRect+fuzzyList))
        self.out5 = Output('out5', Fir(time_part) + Fir(sample_select))
        self.out6 = Output('out6', LocalModel(output_function=Fir())(x.tw(1), fuzzy))
        self.out7 = Output('out7', parfun_zz(z.last()))
        self.out8 = Output('out8', Fir(parfun_x(x.tw(1)) + parfun_y(y.tw(1), c_v)) + Fir(parfun_zz(x.tw(5), t_5, c_5_2)))
        self.out9 = Output('out9', Fir(parfun_2d(x.tw(1)) + parfun_3d(x.tw(1),x.tw(1))))
        self.out10 = Output('out10', a.sw(5)+P_time)
        self.out11 = Output('out11', TimeConcatenate(TimeConcatenate(
                            TimeConcatenate(Integrate(a.last()),Integrate(a.last())),
                            TimeConcatenate(Integrate(a.last()),Integrate(a.last()))
        ),Integrate(a.last()))+P_time)

    def setUp(self):
        # Reindirizza stdout e stderr
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

    def tearDown(self):
        # Ripristina stdout e stderr
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

    def test_rper_of_objects(self):
        print(repr(self.x))
        print(repr(self.stream))
        print(repr(self.out9))

    def test_export_textvisualizer(self):
        t = TextVisualizer(5)
        test = Modely(visualizer=t, seed=42, workspace='./results')
        test.addModel('modelA', self.out)
        test.addModel('modelB', [self.out2, self.out3, self.out4])
        test.addModel('modelC', [self.out4, self.out5, self.out6])
        test.addModel('modelD', self.out7)
        test.addMinimize('error1', self.x.last(), self.out)
        test.addMinimize('error2', self.y.last(), self.out3, loss_function='rmse')
        test.addMinimize('error3', self.z.last(), self.out6, loss_function='rmse')

        test.neuralizeModel(0.5)

        data_x = np.arange(0.0, 5, 0.1)
        data_y = np.arange(0.0, 5, 0.1)
        a, b = -1.0, 2.0
        dataset = {'x': data_x, 'y': data_y, 'z': a * data_x + b * data_y}
        params = {'num_of_epochs': 10, 'lr': 0.01}
        test.loadData(name='dataset', source=dataset)  # Create the datastest.trainModel(optimizer='SGD', training_params=params)  # Train the traced model
        t.showMinimize('error1')
        t.showMinimize('error2')
        t.showMinimize('error3')
        test.trainModel(optimizer='SGD', training_params=params)
        t.showWeights()
        test.trainModel(optimizer='SGD', training_params=params, closed_loop={'x':'out'}, prediction_samples=1)
        test.saveModel()
        test.loadModel()

        test.neuralizeModel(0.5)
        test.exportPythonModel()
        test.importPythonModel()
        test.exportReport()

        test = Modely(visualizer='Standard')
        test.addModel('modelA', self.out)
        test.neuralizeModel(0.5)

    def test_export_mplvisualizer(self):
        m = MPLVisualizer(5)
        test = Modely(visualizer=m, seed=42)
        test.addModel('modelA', self.out)
        test.addModel('modelB', [self.out2, self.out3, self.out4])
        test.addModel('modelC', [self.out4, self.out5, self.out6])
        test.addModel('modelD', self.out7)
        test.addModel('modelE', self.out9)
        test.addMinimize('error1', self.x.last(), self.out)
        test.addMinimize('error2', self.y.last(), self.out3, loss_function='rmse')
        test.addMinimize('error3', self.z.last(), self.out6, loss_function='rmse')

        test.neuralizeModel(0.5)

        data_x = np.arange(0.0, 1000, 0.1)
        data_y = np.arange(0.0, 1000, 0.1)
        a, b = -1.0, 2.0
        dataset = {'x': data_x, 'y': data_y, 'z': a * data_x + b * data_y}
        params = {'num_of_epochs': 10, 'lr': 0.01}
        test.loadData(name='dataset', source=dataset)  # Create the dataset
        test.trainAndAnalyze(optimizer='SGD', training_params=params)  # Train the traced model
        test.trainAndAnalyze(optimizer='SGD', training_params=params)
        m.closeResult()
        m.closeTraining()
        list_of_functions = list(test.json['Functions'].keys())
        try:
            for f in list_of_functions:
                m.showFunctions(f)
        except ValueError:
            pass
        with self.assertRaises(ValueError):
            m.showFunctions(list_of_functions[1])
        m.closeFunctions()
        test.trainAndAnalyze(optimizer='SGD', splits=[70, 20, 10], training_params=params, closed_loop={'x': 'out2'},
                             prediction_samples=5)
        m.closeResult()
        m.closeTraining()

    def test_export_mplvisualizer2(self):
        clearNames(['x', 'F'])
        x = Input('x')
        F = Input('F')
        def myFun(K1, K2, p1, p2):
            import torch
            return p1 * K1 + p2 * torch.sin(K2)

        parfun = ParamFun(myFun)
        out = Output('fun', parfun(x.last(), F.last()))
        m = MPLVisualizer()
        example = Modely(visualizer=m)
        example.addModel('out', out)
        example.neuralizeModel()
        m.showFunctions(list(example.json['Functions'].keys()), xlim=[[-5, 5], [-1, 1]])
        m.closeFunctions()

    @unittest.skipIf(
        sys.platform.startswith("win"),
        reason="MPLNotebookVisualizer ask for backend GUI not available in Windows CI"
    )
    def test_export_mplnotebookvisualizer(self):
        m = MPLNotebookVisualizer(5, test=True)
        test = Modely(visualizer=m, seed=42)
        test.addModel('modelB', [self.out2, self.out3, self.out4])
        test.addModel('modelC', [self.out4, self.out5, self.out6])
        test.addModel('modelD', [self.out9])
        test.addMinimize('error2', self.y.last(), self.out3, loss_function='rmse')
        test.addMinimize('error3', self.z.last(), self.out6, loss_function='rmse')

        test.neuralizeModel(1)

        data_x = np.arange(0.0, 1000, 0.1)
        data_y = np.arange(0.0, 1000, 0.1)
        a, b = -1.0, 2.0
        dataset = {'x': data_x, 'y': data_y, 'z': a * data_x + b * data_y}
        params = {'num_of_epochs': 1, 'lr': 0.01}
        test.loadData(name='dataset', source=dataset)  # Create the dataset
        test.trainAndAnalyze(optimizer='SGD', splits=[70,20,10], training_params=params)  # Train the traced mode
        m.closePlots()
        list_of_functions = list(test.json['Functions'].keys())
        try:
            for f in list_of_functions:
                m.showFunctions(f)
        except ValueError:
            pass
        m.closePlots()
        test.trainAndAnalyze(optimizer='SGD', splits=[70, 20, 10], training_params=params, closed_loop={'x':'out2'}, prediction_samples=5)
        m.closePlots()


    def test_structure_plot(self):
        clearNames()
        X = Input('X')
        Y = Input('Y')
        Z = Input('Z')
        t_state = Input('t_state')
        k_state = Input('k_state')

        func1 = Fir(X.last()) + Fir(Y.last())
        func1.closedLoop(t_state)
        func2 = Fir(Z.last()) + t_state.last()
        func2.connect(k_state)
        func3 = Fir(k_state.last()) * Constant('g', sw=1, values=[[9.8]])

        out = Output('out', func1 + func2 + func3)

        example = Modely(visualizer=None)
        example.addModel('model', out)
        example.neuralizeModel()
        with self.assertRaises(ValueError):
            plot_structure(example.json, filename='results/structure_plot', library='invalid_library')
        plot_structure(example.json, filename='results/structure_plot', library='matplotlib', view=False)
        #plot_structure(example.json, filename='results/structure_plot', library='graphviz', view=False)

    def test_window_vector_plot(self):
        m = MPLNotebookVisualizer(5, test=True)
        test = Modely(visualizer=m, seed=42)
        test.addModel('modelA', self.out10)
        test.addMinimize('error1', self.b.sw(5), self.out10, loss_function='rmse')
        test.neuralizeModel()
        data_x = np.sin(np.arange(0.0, 5, 0.01))
        data_y = np.cos(np.arange(0.0, 5, 0.01))
        data_a = np.transpose(np.array([data_x,data_y]))
        dataset = {'a':data_a, 'b': data_a}
        test.loadData(name='dataset', source=dataset)
        test.analyzeModel()

    def test_window_vector_plot_recurrent(self):
        m = MPLNotebookVisualizer(5, test=True)
        test = Modely(visualizer=m, seed=42)
        test.addModel('modelA', self.out10)
        test.addMinimize('error1', self.b.sw(5), self.out11, loss_function='rmse')
        test.neuralizeModel(0.1)
        data_x = np.sin(np.arange(0.0, 5, 0.01))
        data_y = np.cos(np.arange(0.0, 5, 0.01))
        data_a = np.transpose(np.array([data_x,data_y]))
        dataset = {'a':data_a, 'b': data_a}
        test.loadData(name='dataset', source=dataset)
        test.analyzeModel(prediction_samples=20)