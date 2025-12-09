import unittest, os, sys
import numpy as np

from nnodely import *
from nnodely.basic.relation import NeuObj
from nnodely.support.logger import logging, nnLogger

log = nnLogger(__name__, logging.CRITICAL)
log.setAllLevel(logging.CRITICAL)

sys.path.append(os.getcwd())

# 13 Tests
# Test the train parameter and the optimizer options

data_folder = os.path.join(os.path.dirname(__file__), 'data/')

def funIn(x, w):
    return x * w

def funOut(x, w):
    return x / w

def linear_fun(x,a,b):
    return x*a+b

class ModelyTrainingTestParameter(unittest.TestCase):
    def test_network_mass_spring_damper(self):
        NeuObj.clearNames()
        x = Input('x')  # Position
        F = Input('F')  # Force

        # List the output of the model
        x_z = Output('x_z', Fir(x.tw(0.3)) + Fir(F.last()))

        # Add the neural model to the nnodely structure and neuralization of the model
        test = Modely(visualizer=None)
        test.addModel('x_z',x_z)
        test.addMinimize('next-pos', x.z(-1), x_z, 'mse')

        # Create the neural network
        test.neuralizeModel(sample_time=0.05)  # The sampling time depends to the dataset

        # Data load
        data_struct = ['x','F','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']
        test.loadData(name='dataset', source=data_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        test.trainModel(splits=[80,10,10])
        tp = test.getTrainingInfo()

        self.assertEqual((15-6), test._num_of_samples['dataset'])
        self.assertEqual(round((15-6)*80/100),tp['n_samples_train'])
        self.assertEqual(round((15-6)*10/100),tp['n_samples_val'])
        self.assertEqual(round((15-6)*10/100),tp['n_samples_test'])
        self.assertEqual(round((15-6)*80/100),tp['train_batch_size'])
        self.assertEqual(1, tp['update_per_epochs'])
        #self.assertEqual(0, tp['unused_samples'])
        self.assertEqual(1,tp['val_batch_size'])
        self.assertEqual(100,tp['num_of_epochs'])
        self.assertEqual(0.001,tp['optimizer_defaults']['lr'])

    def test_build_dataset_batch_connect(self):
        NeuObj.clearNames()
        data_x = np.random.rand(500) * 20 - 10
        data_a = 2
        data_b = -3
        dataset = {'in1': data_x, 'out': linear_fun(data_x, data_a, data_b)}

        input1 = Input('in1')
        out = Input('out')
        rel1 = Fir(input1.tw(0.05))
        y = Output('y', rel1)

        test = Modely(visualizer=None, seed=42)
        test.addModel('y',y)
        test.addMinimize('pos', out.next(), y)
        test.neuralizeModel(0.01)

        test.loadData(name='dataset',source=dataset)

        training_params = {}
        training_params['train_batch_size'] = 4
        training_params['val_batch_size'] = 4
        training_params['lr'] = 0.1
        training_params['num_of_epochs'] = 5
        test.trainModel(splits=[70,20,10], closed_loop={'in1':'y'}, prediction_samples=5, training_params = training_params)
        tp = test.getTrainingInfo()

        self.assertEqual(346,tp['n_samples_train']) ## ((500 - 5) * 0.7)  = 346
        self.assertEqual(99,tp['n_samples_val']) ## ((500 - 5) * 0.2)  = 99
        self.assertEqual(50,tp['n_samples_test']) ## ((500 - 5) * 0.1)  = 50
        self.assertEqual(495, test._num_of_samples['dataset']) ## 500 - 5 = 495
        self.assertEqual(4,tp['train_batch_size'])
        self.assertEqual(4,tp['val_batch_size'])
        self.assertEqual(5,tp['num_of_epochs'])
        self.assertEqual(5, tp['prediction_samples'])
        self.assertEqual(0, tp['step'])
        self.assertEqual({'in1':'y'}, tp['closed_loop'])
        self.assertEqual(0.1,tp['optimizer_defaults']['lr'])
        self.assertEqual(((494*0.7)-tp['prediction_samples'])//4, tp['update_per_epochs'])
        #self.assertEqual(1, tp['unused_samples'])

    def test_recurrent_train_closed_loop(self):
        NeuObj.clearNames()
        data_x = np.random.rand(500) * 20 - 10
        data_a = 2
        data_b = -3
        dataset = {'in1': data_x, 'out': linear_fun(data_x, data_a, data_b)}

        x = Input('in1')
        p = Parameter('p', dimensions=1, sw=1, values=[[1.0]])
        fir = Fir(W=p)(x.last())
        out = Output('out', fir)

        test = Modely(visualizer=None, seed=42)
        test.addModel('out',out)
        test.addMinimize('pos', x.next(), out)
        test.neuralizeModel(0.01)

        test.loadData(name='dataset',source=dataset)

        training_params = {}
        training_params['train_batch_size'] = 4
        training_params['val_batch_size'] = 4
        training_params['lr'] = 0.1
        training_params['num_of_epochs'] = 50

        test.trainModel(splits=[100,0,0], closed_loop={'in1':'out'}, prediction_samples=3, step=1, training_params = training_params)
        tp = test.getTrainingInfo()

        self.assertEqual((len(data_x)-1)*100/100,tp['n_samples_train']) ## ((500 - 1) * 1)  = 499
        self.assertEqual(0,tp['n_samples_val']) ## ((500 - 5) * 0)  = 0
        self.assertEqual(0,tp['n_samples_test']) ## ((500 - 5) * 0)  = 0
        self.assertEqual((len(data_x)-1) * 100 / 100, test._num_of_samples['dataset'])
        self.assertEqual(4,tp['train_batch_size'])
        self.assertEqual(4,tp['val_batch_size'])
        self.assertEqual(50,tp['num_of_epochs'])
        self.assertEqual(3, tp['prediction_samples'])
        self.assertEqual(1, tp['step'])
        self.assertEqual({'in1':'out'}, tp['closed_loop'])
        self.assertEqual(0.1,tp['optimizer_defaults']['lr'])

        self.assertEqual(99, tp['update_per_epochs']) ## 499 // (4+1) = 99
        #self.assertEqual(100, tp['unused_samples']) ## 99 * step + 1

    def test_recurrent_train_single_close_loop(self):
        NeuObj.clearNames()
        data_x = np.array(list(range(1, 101, 1)), dtype=np.float32)
        dataset = {'x': data_x, 'y': 2 * data_x}

        x = Input('x')
        y = Input('y')
        out = Output('out', Fir(x.last()))

        test = Modely(visualizer=None, seed=42)
        test.addModel('out', out)
        test.addMinimize('pos', y.last(), out)
        test.neuralizeModel(0.01)

        test.loadData(name='dataset', source=dataset)

        training_params = {}
        training_params['train_batch_size'] = 4
        training_params['val_batch_size'] = 4
        training_params['lr'] = 0.01
        training_params['num_of_epochs'] = 50
        test.trainModel(splits=[80, 20, 0], closed_loop={'x': 'out'}, prediction_samples=3, step=2, training_params=training_params)
        tp = test.getTrainingInfo()
        self.assertEqual(round((len(data_x) - 0) * 80 / 100), tp['n_samples_train'])
        self.assertEqual((len(data_x) - 0) * 20 / 100, tp['n_samples_val'])
        self.assertEqual(0, tp['n_samples_test'])
        self.assertEqual((len(data_x) - 0) * 100 / 100, test._num_of_samples['dataset'])
        self.assertEqual(4, tp['train_batch_size'])
        self.assertEqual(4, tp['val_batch_size'])
        self.assertEqual(50, tp['num_of_epochs'])
        self.assertEqual(3, tp['prediction_samples'])
        self.assertEqual(2, tp['step'])
        self.assertEqual({'x': 'out'}, tp['closed_loop'])
        self.assertEqual(0.01, tp['optimizer_defaults']['lr'])
        self.assertEqual(((100*0.8)-3)//(4+2), tp['update_per_epochs'])
        #self.assertEqual(100*0.8 - (tp['update_per_epochs']*4) - 3, tp['unused_samples'])

    def test_recurrent_train_multiple_close_loop(self):
        NeuObj.clearNames()
        data_x = np.array(list(range(1, 101, 1)), dtype=np.float32)
        dataset = {'x': data_x, 'y': 2 * data_x}

        x = Input('x')
        y = Input('y')
        out_x = Output('out_x', Fir(x.last()))
        out_y = Output('out_y', Fir(y.last()))

        test = Modely(visualizer=None, seed=42)
        test.addModel('out_x', out_x)
        test.addModel('out_y', out_y)
        test.addMinimize('pos_x', x.next(), out_x)
        test.addMinimize('pos_y', y.next(), out_y)
        test.neuralizeModel(0.01)

        test.loadData(name='dataset', source=dataset)

        training_params = {}
        training_params['train_batch_size'] = 4
        training_params['val_batch_size'] = 4
        training_params['lr'] = 0.01
        training_params['num_of_epochs'] = 32

        test.trainModel(splits=[80, 20, 0], closed_loop={'x': 'out_x', 'y': 'out_y'}, prediction_samples=3,
                        training_params=training_params)
        tp = test.getTrainingInfo()

        self.assertEqual(round((len(data_x) - 1) * 80 / 100), tp['n_samples_train'])
        self.assertEqual(round((len(data_x) - 1) * 20 / 100), tp['n_samples_val'])
        self.assertEqual(0, tp['n_samples_test'])
        self.assertEqual((len(data_x) - 1) * 100 / 100, test._num_of_samples['dataset'])
        self.assertEqual(4, tp['train_batch_size'])
        self.assertEqual(4, tp['val_batch_size'])
        self.assertEqual(32, tp['num_of_epochs'])
        self.assertEqual(3, tp['prediction_samples'])
        self.assertEqual(0, tp['step'])
        self.assertEqual({'x': 'out_x', 'y': 'out_y'}, tp['closed_loop'])
        self.assertEqual(0.01, tp['optimizer_defaults']['lr'])
        self.assertEqual(19, tp['update_per_epochs'])
        #self.assertEqual(0, tp['unused_samples'])

    def test_build_dataset_batch(self):
        NeuObj.clearNames()
        input1 = Input('in1')
        output = Input('out')
        rel1 = Output('out1',Fir(input1.tw(0.05)))

        test = Modely(visualizer=None)
        test.addMinimize('out', output.z(-1), rel1)
        test.neuralizeModel(0.01)

        data_struct = ['x','F','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']
        test.loadData(name='dataset', source=data_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        self.assertEqual((10,5,1), test._data['dataset']['in1'].shape)

        training_params = {}
        training_params['train_batch_size'] = 1
        training_params['val_batch_size'] = 1
        training_params['lr'] = 0.1
        training_params['num_of_epochs'] = 5
        with self.assertRaises(RuntimeError):
            test.trainModel(splits=[70,20,10],training_params = training_params)
        test.addModel('out',rel1)
        test.neuralizeModel(0.01)
        test.trainModel(splits=[70,20,10],training_params = training_params)
        tp = test.getTrainingInfo()

        # 15 lines in the dataset
        # 5 lines for input + 1 for output -> total of sample 10
        # 10 / 1 * 0.7 = 7 for training
        # 10 / 1 * 0.2 = 2 for validation
        # 10 / 1 * 0.1 = 1 for test

        self.assertEqual(7,tp['n_samples_train'])
        self.assertEqual(2,tp['n_samples_val'])
        self.assertEqual(1,tp['n_samples_test'])
        self.assertEqual(10, test._num_of_samples['dataset'])
        self.assertEqual(1,tp['train_batch_size'])
        self.assertEqual(1,tp['val_batch_size'])
        self.assertEqual(5,tp['num_of_epochs'])
        self.assertEqual(0.1,tp['optimizer_defaults']['lr'])

        n_samples = tp['n_samples_train']
        batch_size = tp['train_batch_size']
        list_of_batch_indexes = range(0, n_samples - batch_size + 1, batch_size)
        self.assertEqual(len(list_of_batch_indexes), tp['update_per_epochs'])
        #self.assertEqual(n_samples - list_of_batch_indexes[-1] - batch_size, tp['unused_samples'])

        test.trainModel(splits=[70, 20, 10], training_params=training_params, num_of_epochs=100)
        tp = test.getTrainingInfo()

        self.assertEqual(100, tp['num_of_epochs'])

    def test_build_dataset_batch2(self):
        NeuObj.clearNames()
        input1 = Input('in1')
        output = Input('out')
        rel1 = Output('out1',Fir(input1.tw(0.05)))

        test = Modely(visualizer=None)
        test.addMinimize('out', output.z(-1), rel1)
        test.addModel('model', rel1)
        test.neuralizeModel(0.01)

        data_struct = ['x','F','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']
        test.loadData(name='dataset',source=data_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        self.assertEqual((10,5,1), test._data['dataset']['in1'].shape)

        training_params = {}
        training_params['train_batch_size'] = 25
        training_params['val_batch_size'] = 25
        training_params['lr'] = 0.1
        training_params['num_of_epochs'] = 5
        test.trainModel(splits=[50,0,50],training_params = training_params)
        tp = test.getTrainingInfo()

        # 15 lines in the dataset
        # 5 lines for input + 1 for output -> total of sample 10
        # batch_size > 5 use batch_size = 1
        # 10 / 1 * 0.5 = 5 for training
        # 10 / 1 * 0.0 = 0 for validation
        # 10 / 1 * 0.5 = 5 for test
        self.assertEqual((15 - 5), test._num_of_samples['dataset'])
        self.assertEqual(round((15 - 5) * 50 / 100), tp['n_samples_train'])
        self.assertEqual(round((15 - 5) * 0 / 100), tp['n_samples_val'])
        self.assertEqual(round((15 - 5) * 50 / 100), tp['n_samples_test'])
        self.assertEqual(round((15 - 5) * 50 / 100), tp['train_batch_size'])
        self.assertEqual(25, tp['val_batch_size'])
        self.assertEqual(5, tp['num_of_epochs'])
        self.assertEqual(0.1, tp['optimizer_defaults']['lr'])
        self.assertEqual(1, tp['update_per_epochs'])
        #self.assertEqual(0, tp['unused_samples'])

    def test_build_dataset_batch3(self):
        NeuObj.clearNames()
        input1 = Input('in1')
        output = Input('out')
        rel1 = Output('out1',Fir(input1.tw(0.05)))

        test = Modely(workspace='results', visualizer=None)
        test.addMinimize('out', output.next(), rel1)
        test.addModel('model', rel1)
        test.neuralizeModel(0.01)

        data_struct = ['x', 'F', 'x2', 'y2', '', 'A1x', 'A1y', 'B1x', 'B1y', '', 'A2x', 'A2y', 'B2x', 'out', '', 'x3',
                       'in1', 'in2', 'time']
        test.loadData(name='dataset', source=data_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        self.assertEqual((10, 5, 1), test._data['dataset']['in1'].shape)

        training_params = {}
        training_params['train_batch_size'] = 2
        training_params['val_batch_size'] = 2
        training_params['lr'] = 0.1
        training_params['num_of_epochs'] = 5
        test.trainModel(splits=[40, 30, 30], training_params=training_params)
        tp = test.getTrainingInfo()
        # 15 lines in the dataset
        # 5 lines for input + 1 for output -> total of sample 10
        # batch_size > 5 -> NO
        # num_of_training_sample must be multiple of batch_size
        # num_of_test_sample must be multiple of batch_size and at least 50%
        # 10 * 0.4 = 2 for training
        # 10 * 0.3 = 1 for validation
        # 10 * 0.3 = 1 for test
        self.assertEqual((15 - 5), test._num_of_samples['dataset'])
        self.assertEqual(round((15 - 5) * 40 / 100), tp['n_samples_train'])
        self.assertEqual(round((15 - 5) * 30 / 100), tp['n_samples_val'])
        self.assertEqual(round((15 - 5) * 30 / 100), tp['n_samples_test'])
        self.assertEqual(2, tp['train_batch_size'])
        self.assertEqual(2, tp['val_batch_size'])
        self.assertEqual(5, tp['num_of_epochs'])
        self.assertEqual(0.1, tp['optimizer_defaults']['lr'])
        self.assertEqual(2, tp['update_per_epochs'])
        #self.assertEqual(0, tp['unused_samples'])

    def test_build_dataset_batch4(self):
        NeuObj.clearNames()
        input1 = Input('in1')
        output = Input('out')
        rel1 = Output('out1',Fir(input1.tw(0.05)))

        test = Modely(visualizer=None)
        test.addMinimize('out', output.z(-1), rel1)
        test.addModel('model', rel1)
        test.neuralizeModel(0.01)

        data_struct = ['x', 'F', 'x2', 'y2', '', 'A1x', 'A1y', 'B1x', 'B1y', '', 'A2x', 'A2y', 'B2x', 'out', '', 'x3',
                       'in1', 'in2', 'time']
        test.loadData(name='dataset', source=data_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        self.assertEqual((10, 5, 1), test._data['dataset']['in1'].shape)

        training_params = {}
        training_params['train_batch_size'] = 2
        training_params['val_batch_size'] = 2
        training_params['lr'] = 0.1
        training_params['num_of_epochs'] = 5
        test.trainModel(splits=[80, 10, 10], training_params=training_params)
        tp = test.getTrainingInfo()

        # 15 lines in the dataset
        # 5 lines for input + 1 for output -> total of sample 10
        # batch_size > 1 -> YES
        # num_of_training_sample must be multiple of batch_size
        # num_of_test_sample must be multiple of batch_size and at least 10%
        # 10 * 0.8 = 8 for training
        # 10 * 0.1 = 1 for validation
        # 10 * 0.1 = 1 for test
        self.assertEqual((15 - 5), test._num_of_samples['dataset'])
        self.assertEqual(round((15 - 5) * 80 / 100), tp['n_samples_train'])
        self.assertEqual(round((15 - 5) * 10 / 100), tp['n_samples_val'])
        self.assertEqual(round((15 - 5) * 10 / 100), tp['n_samples_test'])
        self.assertEqual(2, tp['train_batch_size'])
        self.assertEqual(1, tp['val_batch_size'])
        self.assertEqual(5, tp['num_of_epochs'])
        self.assertEqual(0.1, tp['optimizer_defaults']['lr'])
        self.assertEqual(4, tp['update_per_epochs'])
        #self.assertEqual(0, tp['unused_samples'])

    def test_build_dataset_from_code(self):
        NeuObj.clearNames()
        input1 = Input('in1')
        output = Input('out')
        rel1 = Output('out1',Fir(input1.tw(0.05)))

        test = Modely(visualizer=None)
        test.addMinimize('out', output.next(), rel1)
        test.addModel('model', rel1)
        test.neuralizeModel(0.01)

        x_size = 20
        data_x = np.random.rand(x_size) * 20 - 10
        data_a = 2
        data_b = -3
        dataset = {'in1': data_x, 'out': data_x * data_a + data_b}

        test.loadData(name='dataset', source=dataset, skiplines=0)
        self.assertEqual((15, 5, 1), test._data['dataset']['in1'].shape)  ## 20 data - 5 tw = 15 sample | 0.05/0.01 = 5 in1

        training_params = {}
        training_params['train_batch_size'] = 2
        training_params['val_batch_size'] = 2
        training_params['lr'] = 0.1
        training_params['num_of_epochs'] = 5
        test.trainModel(splits=[80, 20, 0], training_params=training_params)
        tp = test.getTrainingInfo()

        # 20 lines in the dataset
        # 5 lines for input + 1 for output -> total of sample (20 - 5 - 1) = 16
        # batch_size > 1 -> YES
        # num_of_training_sample must be multiple of batch_size
        # num_of_test_sample must be multiple of batch_size and at least 10%
        # 15 * 0.8 = 12 for training
        # 15 * 0.2 = 3 for validation
        # 15 * 0.0 = 0 for test
        self.assertEqual((20 - 5), test._num_of_samples['dataset'])
        self.assertEqual(round((20 - 5) * 80 / 100), tp['n_samples_train'])
        self.assertEqual(round((20 - 5) * 20 / 100), tp['n_samples_val'])
        self.assertEqual(round((20 - 5) * 0 / 100), tp['n_samples_test'])
        self.assertEqual(2, tp['train_batch_size'])
        self.assertEqual(2, tp['val_batch_size'])
        self.assertEqual(5, tp['num_of_epochs'])
        self.assertEqual(0.1, tp['optimizer_defaults']['lr'])
        self.assertEqual(6, tp['update_per_epochs'])
        #self.assertEqual(0, tp['unused_samples'])

    def test_network_multi_dataset(self):
        NeuObj.clearNames()
        train_folder = os.path.join(os.path.dirname(__file__), 'data/')
        val_folder = os.path.join(os.path.dirname(__file__), 'val_data/')
        test_folder = os.path.join(os.path.dirname(__file__), 'test_data/')

        x = Input('x')  # Position
        F = Input('F')  # Force

        # List the output of the model
        x_z = Output('x_z', Fir(x.tw(0.3)) + Fir(F.last()))

        # Add the neural model to the nnodely structure and neuralization of the model
        test = Modely(visualizer=None)
        test.addModel('x_z', x_z)
        test.addMinimize('next-pos', x.z(-1), x_z, 'mse')

        # Create the neural network
        test.neuralizeModel(sample_time=0.05)  # The sampling time depends to the dataset

        # Data load
        data_struct = ['x', 'F', 'x2', 'y2', '', 'A1x', 'A1y', 'B1x', 'B1y', '', 'A2x', 'A2y', 'B2x', 'out', '', 'x3',
                       'in1', 'in2', 'time']
        test.loadData(name='train_dataset', source=train_folder, format=data_struct, skiplines=4, delimiter='\t',
                      header=None)
        test.loadData(name='validation_dataset', source=val_folder, format=data_struct, skiplines=4, delimiter='\t',
                      header=None)
        test.loadData(name='test_dataset', source=test_folder, format=data_struct, skiplines=4, delimiter='\t',
                      header=None)

        training_params = {}
        training_params['train_batch_size'] = 3
        training_params['val_batch_size'] = 2
        training_params['lr'] = 0.1
        training_params['num_of_epochs'] = 5
        #test.trainModel(train_dataset='train_dataset', validation_dataset='validation_dataset', test_dataset='test_dataset', training_params=training_params)
        test.trainModel(train_dataset='train_dataset', validation_dataset='validation_dataset', training_params=training_params)
        tp = test.getTrainingInfo()

        self.assertEqual(9, test._num_of_samples['train_dataset'])
        self.assertEqual(5, test._num_of_samples['validation_dataset'])
        #self.assertEqual(7, test._num_of_samples['test_dataset'])
        self.assertEqual(9, tp['n_samples_train'])
        self.assertEqual(5, tp['n_samples_val'])
        self.assertEqual(3, tp['train_batch_size'])
        self.assertEqual(2, tp['val_batch_size'])
        self.assertEqual(5, tp['num_of_epochs'])
        self.assertEqual(0.1, tp['optimizer_defaults']['lr'])
        self.assertEqual(3, tp['update_per_epochs'])
        #self.assertEqual(0, tp['unused_samples'])

    def test_train_vector_input(self):
        NeuObj.clearNames()
        x = Input('x', dimensions=4)
        y = Input('y', dimensions=3)
        k = Input('k', dimensions=2)
        w = Input('w')

        out = Output('out', Fir(Linear(Linear(3)(x.tw(0.02)) + y.tw(0.02))))
        out2 = Output('out2', Fir(Linear(k.last() + Fir(2)(w.tw(0.05, offset=-0.02)))))

        test = Modely(visualizer=None)
        test.addMinimize('out', out, out2)
        test.addModel('model', out)
        test.neuralizeModel(0.01)

        data_folder = os.path.join(os.path.dirname(__file__), 'vector_data/')
        data_struct = ['x', 'y', '', '', '', '', 'k', '', '', '', 'w']
        test.loadData(name='dataset', source=data_folder, format=data_struct, skiplines=1, delimiter='\t', header=None)

        training_params = {}
        training_params['train_batch_size'] = 1
        training_params['val_batch_size'] = 1
        training_params['lr'] = 0.01
        training_params['num_of_epochs'] = 7
        test.trainModel(splits=[80, 10, 10], training_params=training_params)
        tp = test.getTrainingInfo()

        self.assertEqual(22, test._num_of_samples['dataset'])
        self.assertEqual(18, tp['n_samples_train'])
        self.assertEqual(2, tp['n_samples_val'])
        self.assertEqual(2, tp['n_samples_test'])
        self.assertEqual(1, tp['train_batch_size'])
        self.assertEqual(1, tp['val_batch_size'])
        self.assertEqual(7, tp['num_of_epochs'])
        self.assertEqual(0.01, tp['optimizer_defaults']['lr'])
        self.assertEqual(18, tp['update_per_epochs'])
        #self.assertEqual(0, tp['unused_samples'])

        training_params = {}
        training_params['train_batch_size'] = 6
        training_params['val_batch_size'] = 2
        test.trainModel(splits=[80, 10, 10], training_params=training_params)
        tp = test.getTrainingInfo()

        self.assertEqual(22, test._num_of_samples['dataset'])
        self.assertEqual(18, tp['n_samples_train'])
        self.assertEqual(2, tp['n_samples_val'])
        self.assertEqual(2, tp['n_samples_test'])
        self.assertEqual(6, tp['train_batch_size'])
        self.assertEqual(2, tp['val_batch_size'])
        self.assertEqual(100, tp['num_of_epochs'])
        self.assertEqual(0.001, tp['optimizer_defaults']['lr'])
        self.assertEqual(3, tp['update_per_epochs'])
        #self.assertEqual(0, tp['unused_samples'])

    def test_optimizer_configuration(self):
        NeuObj.clearNames()
        ## Model1
        input1 = Input('in1')
        a = Parameter('a', dimensions=1, tw=0.05, values=[[1], [1], [1], [1], [1]])
        shared_w = Parameter('w', values=[[5]])
        output1 = Output('out1',
                         Fir(W=a)(input1.tw(0.05)) + ParamFun(funIn, parameters_and_constants={'w': shared_w})(
                             input1.last()))

        test = Modely(visualizer=None, seed=42)
        test.addModel('model1', output1)
        test.addMinimize('error1', input1.last(), output1)

        ## Model2
        input2 = Input('in2')
        b = Parameter('b', dimensions=1, tw=0.05, values=[[1], [1], [1], [1], [1]])
        output2 = Output('out2',
                         Fir(W=b)(input2.tw(0.05)) + ParamFun(funOut, parameters_and_constants={'w': shared_w})(
                             input2.last()))

        test.addModel('model2', output2)
        test.addMinimize('error2', input2.last(), output2)
        test.neuralizeModel(0.01)

        # Dataset for train
        data_in1 = np.linspace(0, 5, 60)
        data_in2 = np.linspace(10, 15, 60)
        data_out1 = 2
        data_out2 = -3
        dataset = {'in1': data_in1, 'in2': data_in2, 'out1': data_in1 * data_out1, 'out2': data_in2 * data_out2}
        test.loadData(name='dataset1', source=dataset)

        data_in1 = np.linspace(0, 5, 100)
        data_in2 = np.linspace(10, 15, 100)
        data_out1 = 2
        data_out2 = -3
        dataset = {'in1': data_in1, 'in2': data_in2, 'out1': data_in1 * data_out1, 'out2': data_in2 * data_out2}
        test.loadData(name='dataset2', source=dataset)

        # Optimizer
        # Basic usage
        # Standard optimizer with standard configuration
        # We train all the models with split [100,0,0], lr =0.01 and epochs = 100
        test.trainModel()
        tp = test.getTrainingInfo()
        self.assertEqual(['model1', 'model2'], tp['models'])
        self.assertEqual(152, tp['n_samples_train'])
        self.assertEqual(0, tp['n_samples_val'])
        self.assertEqual(0, tp['n_samples_test'])
        self.assertEqual(100, tp['num_of_epochs'])
        self.assertEqual(0.001, tp['optimizer_defaults']['lr'])
        self.assertEqual(1, tp['update_per_epochs'])
        #self.assertEqual(24, tp['unused_samples'])

        # We train only model1 with split [100,0,0]
        # TODO Learning rate automoatically optimized based on the mean and variance of the output
        # TODO num_of_epochs automatically defined
        # now is 0.001 for learning rate and 100 for the epochs and optimizer Adam
        test.trainModel(models='model1', splits=[100, 0, 0])
        tp = test.getTrainingInfo()
        self.assertEqual(['model1'], tp['models'])
        self.assertEqual(100, tp['num_of_epochs'])
        self.assertEqual(152, tp['n_samples_train'])
        self.assertEqual(0, tp['n_samples_val'])
        self.assertEqual(0, tp['n_samples_test'])
        self.assertEqual(1, tp['update_per_epochs'])
        #self.assertEqual(24, tp['unused_samples'])

        # Set number of epoch and learning rate via parameters it works only for standard parameters
        test.trainModel(models='model1', splits=[100, 0, 0], lr=0.5, num_of_epochs=5)
        tp = test.getTrainingInfo()
        self.assertEqual(['model1'], tp['models'])
        self.assertEqual(5, tp['num_of_epochs'])
        self.assertEqual(152, tp['n_samples_train'])
        self.assertEqual(0, tp['n_samples_val'])
        self.assertEqual(0, tp['n_samples_test'])
        self.assertEqual(0.5, tp['optimizer_defaults']['lr'])
        self.assertEqual(1, tp['update_per_epochs'])
        #self.assertEqual(24, tp['unused_samples'])

        # Set number of epoch and learning rate via parameters it works only for standard parameters and use two different dataset one for train and one for validation
        test.trainModel(models='model1', train_dataset='dataset1', validation_dataset='dataset2', lr=0.6, num_of_epochs=10)
        tp = test.getTrainingInfo()
        self.assertEqual(['model1'], tp['models'])
        self.assertEqual(10, tp['num_of_epochs'])
        self.assertEqual(56, tp['n_samples_train'])
        self.assertEqual(96, tp['n_samples_val'])
        self.assertEqual(0, tp['n_samples_test'])
        self.assertEqual(0.6,tp['optimizer_defaults']['lr'])
        self.assertEqual(1, tp['update_per_epochs'])
        #self.assertEqual(0, tp['unused_samples'])

        # Use dictionary for set number of epoch, learning rate, etc.. This configuration works only standard parameters (all the parameters that are input of the trainModel).
        training_params = {
            'models': ['model2'],
            'splits': [55, 40, 5],
            'num_of_epochs': 20,
            'lr': 0.7
        }
        test.trainModel(training_params=training_params)
        tp = test.getTrainingInfo()
        self.assertEqual(['model2'], tp['models'])
        self.assertEqual(20, tp['num_of_epochs'])
        self.assertEqual(round(152 * 55 / 100), tp['n_samples_train'])
        self.assertEqual(round(152 * 40 / 100), tp['n_samples_val'])
        self.assertEqual(152 - tp['n_samples_train'] - tp['n_samples_val'], tp['n_samples_test'])
        self.assertEqual(0.7, tp['optimizer_defaults']['lr'])
        self.assertEqual(1, tp['update_per_epochs'])
        #self.assertEqual(0, tp['unused_samples'])

        # If I add a function parameter it has the priority
        # In this case apply train parameter but on a different model
        test.trainModel(models='model1', training_params=training_params)
        tp = test.getTrainingInfo()
        self.assertEqual(['model1'], tp['models'])
        self.assertEqual(20, tp['num_of_epochs'])
        self.assertEqual(round(152 * 55 / 100), tp['n_samples_train'])
        self.assertEqual(round(152 * 40 / 100), tp['n_samples_val'])
        self.assertEqual(152 - tp['n_samples_train'] - tp['n_samples_val'], tp['n_samples_test'])
        self.assertEqual(0.7, tp['optimizer_defaults']['lr'])
        self.assertEqual(1, tp['update_per_epochs'])
        #self.assertEqual(0, tp['unused_samples'])

        ##################################
        # Modify additional parameters in the optimizer that are not present in the standard parameter
        # In this case I modify the learning rate and the betas of the Adam optimizer
        # For the optimizer parameter the priority is the following
        # max priority to the function parameter ('lr' : 0.2)
        # then the standard_optimizer_parameters ('lr' : 0.1)
        # finally the standard_train_parameters  ('lr' : 0.5)
        optimizer_defaults = {
            'lr': 0.1,
            'betas': (0.5, 0.99)
        }
        test.trainModel(training_params=training_params, optimizer_defaults=optimizer_defaults, lr=0.2)
        tp = test.getTrainingInfo()
        self.assertEqual(['model2'], tp['models'])
        self.assertEqual(20, tp['num_of_epochs'])
        self.assertEqual(round(152 * 55 / 100), tp['n_samples_train'])
        self.assertEqual(round(152 * 40 / 100), tp['n_samples_val'])
        self.assertEqual(152 - tp['n_samples_train'] - tp['n_samples_val'], tp['n_samples_test'])
        self.assertEqual(0.2, tp['optimizer_defaults']['lr'])
        self.assertEqual((0.5, 0.99), tp['optimizer_defaults']['betas'])
        self.assertEqual(1, tp['update_per_epochs'])
        #self.assertEqual(0, tp['unused_samples'])

        test.trainModel(training_params=training_params, optimizer_defaults=optimizer_defaults)
        tp = test.getTrainingInfo()
        self.assertEqual(['model2'], tp['models'])
        self.assertEqual(20, tp['num_of_epochs'])
        self.assertEqual(round(152 * 55 / 100), tp['n_samples_train'])
        self.assertEqual(round(152 * 40 / 100), tp['n_samples_val'])
        self.assertEqual(152 - tp['n_samples_train'] - tp['n_samples_val'], tp['n_samples_test'])
        self.assertEqual(0.1, tp['optimizer_defaults']['lr'])
        self.assertEqual((0.5, 0.99), tp['optimizer_defaults']['betas'])
        self.assertEqual(1, tp['update_per_epochs'])
        #self.assertEqual(0, tp['unused_samples'])

        test.trainModel(training_params=training_params)
        tp = test.getTrainingInfo()
        self.assertEqual(['model2'], tp['models'])
        self.assertEqual(20, tp['num_of_epochs'])
        self.assertEqual(round(152 * 55 / 100), tp['n_samples_train'])
        self.assertEqual(round(152 * 40 / 100), tp['n_samples_val'])
        self.assertEqual(152 - tp['n_samples_train'] - tp['n_samples_val'], tp['n_samples_test'])
        self.assertEqual(0.7, tp['optimizer_defaults']['lr'])
        self.assertEqual(1, tp['update_per_epochs'])
        #self.assertEqual(0, tp['unused_samples'])
        ##################################

        # Modify the non standard args of the optimizer using the optimizer_defaults
        # In this case use the SGD with 0.2 of momentum
        optimizer_defaults = {
            'momentum': 0.002
        }
        test.trainModel(optimizer='SGD', training_params=training_params, optimizer_defaults=optimizer_defaults, lr=0.2)
        tp = test.getTrainingInfo()
        self.assertEqual(['model2'], tp['models'])
        self.assertEqual('SGD', tp['optimizer'])
        self.assertEqual(20, tp['num_of_epochs'])
        self.assertEqual(round(152 * 55 / 100), tp['n_samples_train'])
        self.assertEqual(round(152 * 40 / 100), tp['n_samples_val'])
        self.assertEqual(152 - tp['n_samples_train'] - tp['n_samples_val'], tp['n_samples_test'])
        self.assertEqual(0.2, tp['optimizer_defaults']['lr'])
        self.assertEqual(0.002, tp['optimizer_defaults']['momentum'])
        self.assertEqual(1, tp['update_per_epochs'])
        #self.assertEqual(0, tp['unused_samples'])

        # Modify standard optimizer parameter for each training parameter
        training_params = {
            'models': ['model1'],
            'splits': [100, 0, 0],
            'num_of_epochs': 30,
            'lr': 0.5,
            'lr_param': {'a': 0.1}
        }
        test.trainModel(training_params=training_params)
        tp = test.getTrainingInfo()
        self.assertEqual(['model1'], tp['models'])
        self.assertEqual(30, tp['num_of_epochs'])
        self.assertEqual(round(152 * 100 / 100), tp['n_samples_train'])
        self.assertEqual(round(152 * 0 / 100), tp['n_samples_val'])
        self.assertEqual(152 - tp['n_samples_train'] - tp['n_samples_val'], tp['n_samples_test'])
        self.assertEqual(0.5, tp['optimizer_defaults']['lr'])
        self.assertEqual([{'lr': 0.1, 'params': 'a'},
                          {'lr': 0.0, 'params': 'b'},
                          {'params': 'w'}], tp['optimizer_params'])
        self.assertEqual(1, tp['update_per_epochs'])
        #self.assertEqual(24, tp['unused_samples'])

        ##################################
        # Modify standard optimizer parameter for each training parameter using optimizer_params
        # The priority is the following
        # max priority to the function parameter ( 'lr_param'={'a': 0.4})
        # then the optimizer_params ( {'params':'a','lr':0.6} )
        # then the optimizer_params inside the train_parameters ( {'params':['a'],'lr':0.7} )
        # finally the train_parameters  ( 'lr_param'={'a': 0.1})
        training_params = {
            'models': ['model1'],
            'splits': [100, 0, 0],
            'num_of_epochs': 40,
            'lr': 0.5,
            'lr_param': {'a': 0.1},
            'optimizer_params': [{'params': ['a'], 'lr': 0.7}],
            'optimizer_defaults': {'lr': 0.12}
        }
        optimizer_params = [
            {'params': ['a'], 'lr': 0.6}
        ]
        optimizer_defaults = {
            'lr': 0.2
        }
        test.trainModel(training_params=training_params, optimizer_params=optimizer_params,
                        optimizer_defaults=optimizer_defaults, lr_param={'a': 0.4})
        tp = test.getTrainingInfo()
        self.assertEqual(['model1'], tp['models'])
        self.assertEqual(40, tp['num_of_epochs'])
        self.assertEqual(round(152 * 100 / 100), tp['n_samples_train'])
        self.assertEqual(round(152 * 0 / 100), tp['n_samples_val'])
        self.assertEqual(round(152 * 0 / 100), tp['n_samples_test'])
        self.assertEqual(0.2, tp['optimizer_defaults']['lr'])
        self.assertEqual([{'lr': 0.4, 'params': 'a'}], tp['optimizer_params'])
        self.assertEqual(1, tp['update_per_epochs'])
        #self.assertEqual(24, tp['unused_samples'])

        test.trainModel(training_params=training_params, optimizer_params=optimizer_params,
                        optimizer_defaults=optimizer_defaults)
        tp = test.getTrainingInfo()
        self.assertEqual(0.2, tp['optimizer_defaults']['lr'])
        self.assertEqual([{'lr': 0.6, 'params': 'a'}], tp['optimizer_params'])

        test.trainModel(training_params=training_params, optimizer_params=optimizer_params)
        tp = test.getTrainingInfo()
        self.assertEqual(0.12, tp['optimizer_defaults']['lr'])
        self.assertEqual([{'lr': 0.6, 'params': 'a'}], tp['optimizer_params'])

        test.trainModel(training_params=training_params)
        tp = test.getTrainingInfo()
        self.assertEqual(0.12, tp['optimizer_defaults']['lr'])
        self.assertEqual([{'params': 'a', 'lr': 0.7}], tp['optimizer_params'])

        del training_params['optimizer_defaults']
        test.trainModel(training_params=training_params)
        tp = test.getTrainingInfo()
        self.assertEqual(0.5, tp['optimizer_defaults']['lr'])
        self.assertEqual([{'params': 'a', 'lr': 0.7}], tp['optimizer_params'])

        del training_params['optimizer_params']
        test.trainModel(training_params=training_params)
        tp = test.getTrainingInfo()
        self.assertEqual(0.5, tp['optimizer_defaults']['lr'])
        self.assertEqual([{'lr': 0.1, 'params': 'a'},
                          {'lr': 0.0, 'params': 'b'},
                          {'params': 'w'}], tp['optimizer_params'])

        test.trainModel()
        tp = test.getTrainingInfo()
        self.assertEqual(0.001, tp['optimizer_defaults']['lr'])
        self.assertEqual([{'params': 'a'},
                          {'params': 'b'},
                          {'params': 'w'}], tp['optimizer_params'])
        self.assertEqual(1, tp['update_per_epochs'])
        #self.assertEqual(24, tp['unused_samples'])

        ##################################

        ##################################
        # Maximum level of configuration I define a custom optimizer with defaults
        # For the optimizer default the priority is the following
        # max priority to the function parameter ('lr'= 0.4)
        # then the optimizer_defaults ('lr':0.1)
        # then the optimizer_defaults inside the train_parameters ('lr'= 0.12)
        # finally the train_parameters  ('lr'= 0.5)
        class RMSprop(Optimizer):
            def __init__(self, optimizer_defaults={}, optimizer_params=[]):
                super(RMSprop, self).__init__('RMSprop', optimizer_defaults, optimizer_params)

            def get_torch_optimizer(self):
                import torch
                return torch.optim.RMSprop(self.replace_key_with_params(), **self.optimizer_defaults)

        training_params = {
            'models': ['model1'],
            'splits': [100, 0, 0],
            'num_of_epochs': 40,
            'lr': 0.5,
            'lr_param': {'a': 0.1},
            'optimizer_params': [{'params': ['a'], 'lr': 0.7}],
            'optimizer_defaults': {'lr': 0.12}
        }
        optimizer_defaults = {
            'alpha': 0.8
        }
        optimizer = RMSprop(optimizer_defaults)
        test.trainModel(optimizer=optimizer, training_params=training_params, optimizer_defaults={'lr': 0.3}, lr=0.4)
        tp = test.getTrainingInfo()
        self.assertEqual(['model1'], tp['models'])
        self.assertEqual('RMSprop', tp['optimizer'])
        self.assertEqual(40, tp['num_of_epochs'])
        self.assertEqual(round(152 * 100 / 100), tp['n_samples_train'])
        self.assertEqual(round(152 * 0 / 100), tp['n_samples_val'])
        self.assertEqual(round(152 * 0 / 100), tp['n_samples_test'])
        self.assertEqual({'lr': 0.4}, tp['optimizer_defaults'])
        self.assertEqual([{'lr': 0.7, 'params': 'a'}], tp['optimizer_params'])

        test.trainModel(optimizer=optimizer, training_params=training_params, optimizer_defaults={'lr': 0.1})
        tp = test.getTrainingInfo()
        self.assertEqual({'lr': 0.1}, tp['optimizer_defaults'])
        self.assertEqual([{'lr': 0.7, 'params': 'a'}], tp['optimizer_params'])

        test.trainModel(optimizer=optimizer, training_params=training_params)
        tp = test.getTrainingInfo()
        self.assertEqual({'lr': 0.12}, tp['optimizer_defaults'])
        self.assertEqual([{'lr': 0.7, 'params': 'a'}], tp['optimizer_params'])

        del training_params['optimizer_defaults']
        test.trainModel(optimizer=optimizer, training_params=training_params)
        tp = test.getTrainingInfo()
        self.assertEqual({'alpha': 0.8, 'lr': 0.5}, tp['optimizer_defaults'])
        self.assertEqual([{'lr': 0.7, 'params': 'a'}], tp['optimizer_params'])

        del training_params['optimizer_params']
        test.trainModel(optimizer=optimizer, training_params=training_params)
        tp = test.getTrainingInfo()
        self.assertEqual({'alpha': 0.8, 'lr': 0.5}, tp['optimizer_defaults'])
        self.assertEqual([{'lr': 0.1, 'params': 'a'}, {'lr': 0.0, 'params': 'b'}, {'params': 'w'}], tp['optimizer_params'])

        test.trainModel(optimizer=optimizer)
        tp = test.getTrainingInfo()
        self.assertEqual({'alpha': 0.8, 'lr': 0.001}, tp['optimizer_defaults'])
        self.assertEqual([{'params': 'a'}, {'params': 'b'}, {'params': 'w'}], tp['optimizer_params'])

        self.assertEqual(1, tp['update_per_epochs'])
        #self.assertEqual(24, tp['unused_samples'])
        ##################################

        ##################################
        # Maximum level of configuration I define a custom optimizer with custom value for each params
        # The priority is the following
        # max priority to the function parameter ( 'lr_param'={'a': 0.2})
        # then the optimizer_params ( [{'params':['a'],'lr':1.0}] )
        # then the optimizer_params inside the train_parameters (  [{'params':['a'],'lr':0.7}] )
        # then the train_parameters  ( 'lr_param'={'a': 0.1} )
        # finnaly the optimizer_paramsat the time of the optimizer initialization [{'params':['a'],'lr':0.6}]
        training_params = {
            'models': ['model1'],
            'splits': [100, 0, 0],
            'num_of_epochs': 40,
            'lr': 0.5,
            'lr_param': {'a': 0.1},
            'optimizer_params': [{'params': ['a'], 'lr': 0.7}],
            'optimizer_defaults': {'lr': 0.12}
        }
        optimizer_defaults = {
            'alpha': 0.8
        }
        optimizer_params = [
            {'params': ['a'], 'lr': 0.6}, {'params': 'w', 'lr': 0.12, 'alpha': 0.02}
        ]
        optimizer = RMSprop(optimizer_defaults, optimizer_params)
        test.trainModel(optimizer=optimizer, training_params=training_params, optimizer_defaults={'lr': 0.3},
                        optimizer_params=[{'params': ['a'], 'lr': 1.0}, {'params': ['b'], 'lr': 1.2}],
                        lr_param={'a': 0.2})
        tp = test.getTrainingInfo()
        self.assertEqual(['model1'], tp['models'])
        self.assertEqual('RMSprop', tp['optimizer'])
        self.assertEqual(40, tp['num_of_epochs'])
        self.assertEqual(round(152 * 100 / 100), tp['n_samples_train'])
        self.assertEqual(round(152 * 0 / 100), tp['n_samples_val'])
        self.assertEqual(round(152 * 0 / 100), tp['n_samples_test'])
        self.assertEqual({'lr': 0.3}, tp['optimizer_defaults'])
        self.assertEqual([{'params': 'a', 'lr': 0.2}, {'params': 'b', 'lr': 1.2}], tp['optimizer_params'])

        test.trainModel(optimizer=optimizer, training_params=training_params, optimizer_defaults={'lr': 0.3},
                        optimizer_params=[{'params': ['a'], 'lr': 0.1}, {'params': ['b'], 'lr': 0.2}])
        tp = test.getTrainingInfo()
        self.assertEqual({'lr': 0.3}, tp['optimizer_defaults'])
        self.assertEqual([{'params': 'a', 'lr': 0.1}, {'params': 'b', 'lr': 0.2}], tp['optimizer_params'])

        test.trainModel(optimizer=optimizer, training_params=training_params, optimizer_defaults={'lr': 0.3})
        tp = test.getTrainingInfo()
        self.assertEqual({'lr': 0.3}, tp['optimizer_defaults'])
        self.assertEqual([{'params': 'a', 'lr': 0.7}], tp['optimizer_params'])

        test.trainModel(optimizer=optimizer, training_params=training_params)
        tp = test.getTrainingInfo()
        self.assertEqual({'lr': 0.12}, tp['optimizer_defaults'])
        self.assertEqual([{'params': 'a', 'lr': 0.7}], tp['optimizer_params'])

        del training_params['optimizer_defaults']
        test.trainModel(optimizer=optimizer, training_params=training_params)
        tp = test.getTrainingInfo()
        self.assertEqual({'alpha': 0.8, 'lr': 0.5}, tp['optimizer_defaults'])
        self.assertEqual([{'params': 'a', 'lr': 0.7}], tp['optimizer_params'])

        del training_params['optimizer_params']
        test.trainModel(optimizer=optimizer, training_params=training_params)
        tp = test.getTrainingInfo()
        self.assertEqual({'alpha': 0.8, 'lr': 0.5}, tp['optimizer_defaults'])
        self.assertEqual([{'lr': 0.1, 'params': 'a'}, {'alpha': 0.02, 'lr': 0.12, 'params': 'w'}], tp['optimizer_params'])

        test.trainModel(optimizer=optimizer)
        tp = test.getTrainingInfo()
        self.assertEqual({'alpha': 0.8, 'lr': 0.001}, tp['optimizer_defaults'])
        self.assertEqual([{'params': 'a', 'lr': 0.6}, {'params': 'w', 'lr': 0.12, 'alpha': 0.02}],
                         tp['optimizer_params'])
        self.assertEqual(1, tp['update_per_epochs'])
        #self.assertEqual(24, tp['unused_samples'])
        ##################################

        ##################################
        # Maximum level of configuration I define a custom optimizer and add some parameter over the defaults
        # The priority is the following
        # max priority to the function parameter ( 'lr_param'={'a': 0.2})
        # then the optimizer_params ( [{'params':['a'],'lr':1.0}] )
        # then the optimizer_params inside the train_parameters (  [{'params':['a'],'lr':0.7}] )
        # then the train_parameters  ( 'lr_param'={'a': 0.1} )
        # The other parameters are the defaults
        training_params = {
            'models': ['model1'],
            'splits': [100, 0, 0],
            'num_of_epochs': 40,
            'lr': 0.5,
            'lr_param': {'a': 0.1},
            'add_optimizer_params': [{'params': ['a'], 'lr': 0.7}],
            'add_optimizer_defaults': {'lr': 0.12}
        }
        optimizer = RMSprop()
        test.trainModel(optimizer=optimizer, training_params=training_params, add_optimizer_defaults={'lr': 0.3},
                        add_optimizer_params=[{'params': ['a'], 'lr': 1.0}, {'params': ['b'], 'lr': 1.2}],
                        lr_param={'a': 0.2})
        tp = test.getTrainingInfo()
        self.assertEqual(['model1'], tp['models'])
        self.assertEqual('RMSprop', tp['optimizer'])
        self.assertEqual(40, tp['num_of_epochs'])
        self.assertEqual(round(152 * 100 / 100), tp['n_samples_train'])
        self.assertEqual(round(152 * 0 / 100), tp['n_samples_val'])
        self.assertEqual(round(152 * 0 / 100), tp['n_samples_test'])
        self.assertEqual({'lr': 0.3}, tp['optimizer_defaults'])
        self.assertEqual([{'params': 'a', 'lr': 0.2}, {'params': 'b', 'lr': 1.2}, {'params': 'w'}], tp['optimizer_params'])

        test.trainModel(optimizer=optimizer, training_params=training_params, add_optimizer_defaults={'lr': 0.3},
                        add_optimizer_params=[{'params': ['a'], 'lr': 0.23}, {'params': ['b'], 'lr': 0.2}])
        tp = test.getTrainingInfo()
        self.assertEqual({'lr': 0.3}, tp['optimizer_defaults'])
        self.assertEqual([{'params': 'a', 'lr': 0.23}, {'params': 'b', 'lr': 0.2}, {'params': 'w'}], tp['optimizer_params'])

        test.trainModel(optimizer=optimizer, training_params=training_params, add_optimizer_defaults={'lr': 0.3},
                        add_optimizer_params=[{'params': ['b'], 'lr': 0.2}])
        tp = test.getTrainingInfo()
        self.assertEqual({'lr': 0.3}, tp['optimizer_defaults'])
        self.assertEqual([{'params': 'a', 'lr': 0.1}, {'params': 'b', 'lr': 0.2}, {'params': 'w'}], tp['optimizer_params'])

        test.trainModel(optimizer=optimizer, training_params=training_params, add_optimizer_defaults={'lr': 0.3})
        tp = test.getTrainingInfo()
        self.assertEqual({'lr': 0.3}, tp['optimizer_defaults'])
        self.assertEqual([{'params': 'a', 'lr': 0.7}, {'lr': 0.0, 'params': 'b'}, {'params': 'w'}], tp['optimizer_params'])

        test.trainModel(optimizer=optimizer, training_params=training_params)
        tp = test.getTrainingInfo()
        self.assertEqual({'lr': 0.12}, tp['optimizer_defaults'])
        self.assertEqual([{'params': 'a', 'lr': 0.7}, {'lr': 0.0, 'params': 'b'}, {'params': 'w'}], tp['optimizer_params'])

        del training_params['add_optimizer_defaults']
        test.trainModel(optimizer=optimizer, training_params=training_params)
        tp = test.getTrainingInfo()
        self.assertEqual({'lr': 0.5}, tp['optimizer_defaults'])
        self.assertEqual([{'params': 'a', 'lr': 0.7}, {'lr': 0.0, 'params': 'b'}, {'params': 'w'}], tp['optimizer_params'])

        del training_params['add_optimizer_params']
        test.trainModel(optimizer=optimizer, training_params=training_params)
        tp = test.getTrainingInfo()
        self.assertEqual({'lr': 0.5}, tp['optimizer_defaults'])
        self.assertEqual([{'lr': 0.1, 'params': 'a'}, {'lr': 0.0, 'params': 'b'}, {'params': 'w'}], tp['optimizer_params'])

        test.trainModel(optimizer=optimizer)
        tp = test.getTrainingInfo()
        self.assertEqual({'lr': 0.001}, tp['optimizer_defaults'])
        self.assertEqual([{'params': 'a'}, {'params': 'b'}, {'params': 'w'}], tp['optimizer_params'])

        self.assertEqual(1, tp['update_per_epochs'])
        #self.assertEqual(24, tp['unused_samples'])

    def test_train_sampled_datasets(self):
        NeuObj.clearNames()
        x = Input('x')  # Position
        F = Input('F')  # Force

        # List the output of the model
        x_z = Output('x_z', Fir(x.tw(0.3)) + Fir(F.last()))

        # Add the neural model to the nnodely structure and neuralization of the model
        test = Modely(visualizer=None)
        test.addModel('x_z',x_z)
        test.addMinimize('next-pos', x.z(-1), x_z, 'mse')

        # Create the neural network
        test.neuralizeModel(sample_time=0.05)  # The sampling time depends to the dataset

        # Data load
        data_struct = ['x','F','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']
        test.loadData(name='dataset', source=data_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        test.trainModel(train_dataset='dataset', num_of_epochs=3)
        tp = test.getTrainingInfo()

        self.assertEqual((15-6), test._num_of_samples['dataset'])
        self.assertEqual(round(15-6),tp['n_samples_train'])
        self.assertEqual(round(0),tp['n_samples_val'])
        self.assertEqual(round(0),tp['n_samples_test'])
        self.assertEqual(round(15-6),tp['train_batch_size'])
        self.assertEqual(1, tp['update_per_epochs'])
        #self.assertEqual(0, tp['unused_samples'])
        self.assertEqual(128,tp['val_batch_size'])
        self.assertEqual(3,tp['num_of_epochs'])
        self.assertEqual(0.001,tp['optimizer_defaults']['lr'])

        ## Passing a sampled dataset
        import torch

        train_data = {'x': torch.tensor([[[0.8030],[0.8030],[0.8040],[0.8040],[0.8050],[0.8060],[0.8070]],
                                         [[0.8030],[0.8040],[0.8040],[0.8050],[0.8060],[0.8070],[0.8080]],
                                         [[0.8080],[0.8090],[0.8100],[0.8120],[0.8130],[0.8140],[0.8150]],
                                         [[0.8090],[0.8100],[0.8120],[0.8130],[0.8140],[0.8150],[0.8160]]]),
                     'F': torch.tensor([[[0.8240]],[[0.8230]],[[0.8220]],[[0.8200]],[[0.8190]],[[0.8180]],[[0.8160]],[[0.8140]],[[0.8130]]])}
        ## Not the same number of samples 
        with self.assertRaises(ValueError):
            test.trainModel(train_dataset=train_data)
        with self.assertRaises(ValueError):
            test.trainAndAnalyze(test_dataset=train_data)

        train_data = {'x': torch.tensor([[[0.8030],[0.8030],[0.8040],[0.8040],[0.8050],[0.8060]],
                                         [[0.8030],[0.8040],[0.8040],[0.8050],[0.8060],[0.8070]],
                                         [[0.8080],[0.8090],[0.8100],[0.8120],[0.8130],[0.8140]],
                                         [[0.8090],[0.8100],[0.8120],[0.8130],[0.8140],[0.8150]]]),
                     'F': torch.tensor([[[0.8240]],[[0.8230]],[[0.8220]],[[0.8200]]])}
        ## Not the correct number of dimensions
        with self.assertRaises(ValueError):
            test.trainModel(train_dataset=train_data)
        with self.assertRaises(ValueError):
            test.trainModel(validation_dataset=train_data)
        with self.assertRaises(ValueError):
            test.trainAndAnalyze(test_dataset=train_data)
        with self.assertRaises(ValueError):
            test.trainAndAnalyze(train_dataset=train_data, test_dataset=train_data)
        
        train_data = {'x': torch.tensor([[[0.8030],[0.8030],[0.8040],[0.8040],[0.8050],[0.8060],[0.8070]],
                                         [[0.8030],[0.8040],[0.8040],[0.8050],[0.8060],[0.8070],[0.8080]],
                                         [[0.8080],[0.8090],[0.8100],[0.8120],[0.8130],[0.8140],[0.8150]],
                                         [[0.8090],[0.8100],[0.8120],[0.8130],[0.8140],[0.8150],[0.8160]]]),
                     'F': torch.tensor([[[0.8240]],[[0.8230]],[[0.8220]],[[0.8200]]]),
                     't': torch.tensor([[[0.0]],[[0.05]],[[0.1]],[[0.15]]])}
        ## The extra sample is ignored
        test.trainModel(train_dataset=train_data, validation_dataset=train_data, num_of_epochs=3)

        train_data = {'F': torch.tensor([[[0.8240]],[[0.8230]],[[0.8220]],[[0.8200]]])}
        ## If there is a missing input the training fails
        with self.assertRaises(KeyError):
            test.trainModel(train_dataset=train_data, validation_dataset=train_data, num_of_epochs=3)

        train_data = {'x': torch.tensor([[[0.8030],[0.8030],[0.8040],[0.8040],[0.8050],[0.8060],[0.8070]],
                                         [[0.8030],[0.8040],[0.8040],[0.8050],[0.8060],[0.8070],[0.8080]],
                                         [[0.8040],[0.8040],[0.8050],[0.8060],[0.8070],[0.8080],[0.8090]],
                                         [[0.8040],[0.8050],[0.8060],[0.8070],[0.8080],[0.8090],[0.8100]],
                                         [[0.8050],[0.8060],[0.8070],[0.8080],[0.8090],[0.8100],[0.8120]],
                                         [[0.8060],[0.8070],[0.8080],[0.8090],[0.8100],[0.8120],[0.8130]],
                                         [[0.8070],[0.8080],[0.8090],[0.8100],[0.8120],[0.8130],[0.8140]],
                                         [[0.8080],[0.8090],[0.8100],[0.8120],[0.8130],[0.8140],[0.8150]],
                                         [[0.8090],[0.8100],[0.8120],[0.8130],[0.8140],[0.8150],[0.8160]]]),
                     'F': torch.tensor([[[0.8240]],[[0.8230]],[[0.8220]],[[0.8200]],[[0.8190]],[[0.8180]],[[0.8160]],[[0.8140]],[[0.8130]]])}
        val_data = {'x': torch.tensor([[[0.8030],[0.8030],[0.8040],[0.8040],[0.8050],[0.8060],[0.8070]],
                                         [[0.8030],[0.8040],[0.8040],[0.8050],[0.8060],[0.8070],[0.8080]],
                                         [[0.8040],[0.8040],[0.8050],[0.8060],[0.8070],[0.8080],[0.8090]],
                                         [[0.8040],[0.8050],[0.8060],[0.8070],[0.8080],[0.8090],[0.8100]],
                                         [[0.8090],[0.8100],[0.8120],[0.8130],[0.8140],[0.8150],[0.8160]]]),
                     'F': torch.tensor([[[0.8240]],[[0.8230]],[[0.8220]],[[0.8200]],[[0.8130]]])}
        
        test.trainModel(train_dataset=train_data, num_of_epochs=3)
        tp = test.getTrainingInfo()
        self.assertEqual(9, test._num_of_samples['dataset'])
        self.assertEqual(9,tp['n_samples_train'])
        self.assertEqual(0,tp['n_samples_val'])
        self.assertEqual(0,tp['n_samples_test'])
        self.assertEqual(9,tp['train_batch_size'])
        self.assertEqual(1, tp['update_per_epochs'])
        #self.assertEqual(0, tp['unused_samples'])
        self.assertEqual(128,tp['val_batch_size'])
        self.assertEqual(3,tp['num_of_epochs'])
        self.assertEqual(0.001,tp['optimizer_defaults']['lr'])

        test.trainModel(train_dataset=train_data, validation_dataset=val_data, num_of_epochs=3)
        tp = test.getTrainingInfo()
        self.assertEqual(9, test._num_of_samples['dataset'])
        self.assertEqual(9,tp['n_samples_train'])
        self.assertEqual(5,tp['n_samples_val'])
        self.assertEqual(0,tp['n_samples_test'])
        self.assertEqual(9,tp['train_batch_size'])
        self.assertEqual(1, tp['update_per_epochs'])
        #self.assertEqual(0, tp['unused_samples'])
        self.assertEqual(5,tp['val_batch_size'])
        self.assertEqual(3,tp['num_of_epochs'])
        self.assertEqual(0.001,tp['optimizer_defaults']['lr'])

        test_data = {'x': torch.tensor([[[0.8030],[0.8030],[0.8040],[0.8040],[0.8050],[0.8060],[0.8070]],
                                         [[0.8030],[0.8040],[0.8040],[0.8050],[0.8060],[0.8070],[0.8080]],
                                         [[0.8040],[0.8040],[0.8050],[0.8060],[0.8070],[0.8080],[0.8090]],
                                         [[0.8040],[0.8050],[0.8060],[0.8070],[0.8080],[0.8090],[0.8100]]]),
                     'F': torch.tensor([[[0.8240]],[[0.8230]],[[0.8220]],[[0.8200]]])}
        
        test.trainAndAnalyze(test_dataset=test_data, test_batch_size=2, train_dataset=train_data, validation_dataset=val_data, num_of_epochs=3)
        tp = test.getTrainingInfo()
        self.assertEqual(9, test._num_of_samples['dataset'])
        self.assertEqual(9,tp['n_samples_train'])
        self.assertEqual(5,tp['n_samples_val'])
        self.assertEqual(4,tp['n_samples_test'])
        self.assertEqual(9,tp['train_batch_size'])
        self.assertEqual(1, tp['update_per_epochs'])
        #self.assertEqual(0, tp['unused_samples'])
        self.assertEqual(5,tp['val_batch_size'])
        self.assertEqual(3,tp['num_of_epochs'])
        self.assertEqual(0.001,tp['optimizer_defaults']['lr'])

        test.trainAndAnalyze(test_dataset=test_data, test_batch_size=2, train_dataset=train_data, validation_dataset=val_data, num_of_epochs=3, prediction_samples=2)
        tp = test.getTrainingInfo()
        self.assertEqual(9, test._num_of_samples['dataset'])
        self.assertEqual(9,tp['n_samples_train'])
        self.assertEqual(5,tp['n_samples_val'])
        self.assertEqual(4,tp['n_samples_test'])
        self.assertEqual(9,tp['train_batch_size'])
        self.assertEqual(1, tp['update_per_epochs'])
        #self.assertEqual(0, tp['unused_samples'])
        self.assertEqual(5,tp['val_batch_size'])
        self.assertEqual(3,tp['num_of_epochs'])
        self.assertEqual(0.001,tp['optimizer_defaults']['lr'])