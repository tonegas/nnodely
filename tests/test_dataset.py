import sys, os, unittest
import numpy as np

from nnodely import *
from nnodely.basic.relation import NeuObj
from nnodely.support.logger import logging, nnLogger

log = nnLogger(__name__, logging.CRITICAL)
log.setAllLevel(logging.CRITICAL)

sys.path.append(os.getcwd())

# 16 Tests
# This file test the data loading in particular:
# The shape and the value of the inputs

train_folder = os.path.join(os.path.dirname(__file__), 'data/')
val_folder = os.path.join(os.path.dirname(__file__), 'val_data/')
test_folder = os.path.join(os.path.dirname(__file__), 'test_data/')

class ModelyCreateDatasetTest(unittest.TestCase):
    
    def test_build_dataset_simple(self):
        NeuObj.clearNames()
        input = Input('in1')
        output = Input('out')
        relation = Fir(input.tw(0.05))

        test = Modely(visualizer=None)
        test.addMinimize('out', output.z(-1), relation)
        test.neuralizeModel(0.01)

        data_struct = ['x1','y1','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','theta','time']
        test.loadData(name='dataset_1', source=train_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        self.assertEqual((10,5,1), test._data['dataset_1']['in1'].shape)
        self.assertEqual([[0.984],[0.983],[0.982],[0.98],[0.977]], test._data['dataset_1']['in1'][0].tolist())

        self.assertEqual((10,1,1), test._data['dataset_1']['out'].shape)
        self.assertEqual([[1.225]], test._data['dataset_1']['out'][0].tolist())
        self.assertEqual([[[1.225]], [[1.224]], [[1.222]], [[1.22]], [[1.217]], [[1.214]], [[1.211]], [[1.207]], [[1.204]], [[1.2]]], test._data['dataset_1']['out'].tolist())

    def test_build_dataset_tuple(self):
        NeuObj.clearNames()
        inputA = Input('inA')
        inputB = Input('inB')
        out = Output('out', Fir(inputA.tw(0.05)+inputB.tw(0.05)))

        test = Modely(visualizer=None)
        test.addModel('out', out)
        test.neuralizeModel(0.01)

        data_struct = ['','y1','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3',('inA','inB'),'theta','time']
        test.loadData(name='dataset_1', source=train_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        self.assertEqual((11,5,1), test._data['dataset_1']['inA'].shape)
        self.assertEqual([[0.984],[0.983],[0.982],[0.98],[0.977]], test._data['dataset_1']['inA'][0].tolist())
        self.assertEqual((11,5,1), test._data['dataset_1']['inB'].shape)
        self.assertEqual([[0.984],[0.983],[0.982],[0.98],[0.977]], test._data['dataset_1']['inB'][0].tolist())

        out = Output('out2', Fir(inputA.tw(0.05)))

        test2 = Modely(visualizer=None)
        test2.addModel('out2', out)
        test2.neuralizeModel(0.01)

        data_struct = ['','y1','x2','y2','','A1x','A1y','B1x','B1y','',('A2x','f'),'A2y','B2x','out','','x3',('inA','inB'),'theta','time']
        test2.loadData(name='dataset_1', source=train_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        self.assertEqual((11,5,1), test._data['dataset_1']['inA'].shape)
        self.assertEqual([[0.984],[0.983],[0.982],[0.98],[0.977]], test._data['dataset_1']['inA'][0].tolist())

    def test_build_dataset_tuple_dim(self):
        NeuObj.clearNames()
        inputA1 = Input('inA1')
        inputA = Input('inA', dimensions=3)
        inputB = Input('inB', dimensions=3)
        out = Output('out', inputA1.sw(1)+Fir(Linear(inputA.tw(0.05)+inputB.tw(0.05))))

        test = Modely(visualizer=None)
        test.addModel('out', out)
        test.neuralizeModel(0.01)

        data_struct = ['','y1','x2','y2','',('A1x','inA1'),'A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3',('inA','inB'),'theta','time']
        test.loadData(name='dataset_1', source=train_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        self.assertEqual((11,5,3), test._data['dataset_1']['inA'].shape)
        self.assertEqual([[0.984,12.493, 0.0],[0.983,12.493, 0.01],[0.982,12.495, 0.02],[0.98,12.498, 0.03],[0.977,12.502, 0.04]], test._data['dataset_1']['inA'][0].tolist())
        self.assertEqual((11,5,3), test._data['dataset_1']['inB'].shape)
        self.assertEqual([[0.984, 12.493, 0.0], [0.983, 12.493, 0.01], [0.982, 12.495, 0.02], [0.98, 12.498, 0.03],
                          [0.977, 12.502, 0.04]], test._data['dataset_1']['inB'][0].tolist())

        with self.assertRaises(ValueError):
            data_struct = ['','y1','x2','y2','',('inA1','inA'),'A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3',('inA','inB'),'theta','time']
            test.loadData(name='dataset_2', source=train_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)

        with self.assertRaises(ValueError):
            data_struct = ['','y1','x2','y2','',('inA1','inA'),'A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','inB','theta','time']
            test.loadData(name='dataset_3', source=train_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)

    def test_build_multi_dataset_simple(self):
        NeuObj.clearNames()
        input = Input('in1')
        output = Input('out')
        relation = Fir(input.tw(0.05))

        test = Modely(visualizer=None)
        test.addMinimize('out', output.z(-1), relation)
        test.neuralizeModel(0.01)

        data_struct = ['x1','y1','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','theta','time']

        test.loadData(name='train_dataset', source=train_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        test.loadData(name='validation_dataset', source=val_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        test.loadData(name='test_dataset', source=test_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)

        self.assertEqual(3, test._Loader__n_datasets)

        self.assertEqual((10,5,1), test._data['train_dataset']['in1'].shape)
        self.assertEqual([[0.984],[0.983],[0.982],[0.98],[0.977]], test._data['train_dataset']['in1'][0].tolist())
        self.assertEqual((6,5,1), test._data['validation_dataset']['in1'].shape)
        self.assertEqual([[0.884],[0.883],[0.882],[0.88],[0.877]], test._data['validation_dataset']['in1'][0].tolist())
        self.assertEqual((8,5,1), test._data['test_dataset']['in1'].shape)
        self.assertEqual([[0.784],[0.783],[0.782],[0.78],[0.777]], test._data['test_dataset']['in1'][0].tolist())

        self.assertEqual((10,1,1), test._data['train_dataset']['out'].shape)
        self.assertEqual([[1.225]], test._data['train_dataset']['out'][0].tolist())
        self.assertEqual([[[1.225]], [[1.224]], [[1.222]], [[1.22]], [[1.217]], [[1.214]], [[1.211]], [[1.207]], [[1.204]], [[1.2]]], test._data['train_dataset']['out'].tolist())
        self.assertEqual((6,1,1), test._data['validation_dataset']['out'].shape)
        self.assertEqual([[2.225]], test._data['validation_dataset']['out'][0].tolist())
        self.assertEqual([[[2.225]], [[2.224]], [[2.222]], [[2.22]], [[2.217]], [[2.214]]], test._data['validation_dataset']['out'].tolist())
        self.assertEqual((8,1,1), test._data['test_dataset']['out'].shape)
        self.assertEqual([[3.225]], test._data['test_dataset']['out'][0].tolist())
        self.assertEqual([[[3.225]], [[3.224]], [[3.222]], [[3.22]], [[3.217]], [[3.214]], [[3.211]], [[3.207]]], test._data['test_dataset']['out'].tolist())
    
    def test_build_dataset_medium1(self):
        NeuObj.clearNames()
        input = Input('in1')
        output = Input('out')
        rel1 = Fir(input.tw(0.05))
        rel2 = Fir(input.tw(0.01))

        test = Modely(visualizer=None)
        test.addMinimize('out', output.z(-1), rel1 + rel2)
        test.neuralizeModel(0.01)

        data_struct = ['x1','y1','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','theta','time']
        test.loadData(name='dataset', source=train_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        self.assertEqual((10,5,1), test._data['dataset']['in1'].shape)
        self.assertEqual([[0.984],[0.983],[0.982],[0.98],[0.977]], test._data['dataset']['in1'][0].tolist())

        self.assertEqual((10,1,1), test._data['dataset']['out'].shape)
        self.assertEqual([[[1.225]], [[1.224]], [[1.222]], [[1.22]], [[1.217]], [[1.214]], [[1.211]], [[1.207]], [[1.204]], [[1.2]]], test._data['dataset']['out'].tolist())
    
    def test_build_multi_dataset_medium1(self):
        NeuObj.clearNames()
        input = Input('in1')
        output = Input('out')
        rel1 = Fir(input.tw(0.05))
        rel2 = Fir(input.tw(0.01))

        test = Modely(visualizer=None)
        test.addMinimize('out', output.z(-1), rel1 + rel2)
        test.neuralizeModel(0.01)

        data_struct = ['x1','y1','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','theta','time']

        test.loadData(name='train_dataset', source=train_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        test.loadData(name='validation_dataset', source=val_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        test.loadData(name='test_dataset', source=test_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)

        self.assertEqual(3, test._Loader__n_datasets)

        self.assertEqual((10,5,1), test._data['train_dataset']['in1'].shape)
        self.assertEqual([[0.984],[0.983],[0.982],[0.98],[0.977]], test._data['train_dataset']['in1'][0].tolist())
        self.assertEqual((6,5,1), test._data['validation_dataset']['in1'].shape)
        self.assertEqual([[0.884],[0.883],[0.882],[0.88],[0.877]], test._data['validation_dataset']['in1'][0].tolist())
        self.assertEqual((8,5,1), test._data['test_dataset']['in1'].shape)
        self.assertEqual([[0.784],[0.783],[0.782],[0.78],[0.777]], test._data['test_dataset']['in1'][0].tolist())

        self.assertEqual((10,1,1), test._data['train_dataset']['out'].shape)
        self.assertEqual([[[1.225]], [[1.224]], [[1.222]], [[1.22]], [[1.217]], [[1.214]], [[1.211]], [[1.207]], [[1.204]], [[1.2]]], test._data['train_dataset']['out'].tolist())
        self.assertEqual((6,1,1), test._data['validation_dataset']['out'].shape)
        self.assertEqual([[[2.225]], [[2.224]], [[2.222]], [[2.22]], [[2.217]], [[2.214]]], test._data['validation_dataset']['out'].tolist())
        self.assertEqual((8,1,1), test._data['test_dataset']['out'].shape)
        self.assertEqual([[[3.225]], [[3.224]], [[3.222]], [[3.22]], [[3.217]], [[3.214]], [[3.211]], [[3.207]]], test._data['test_dataset']['out'].tolist())
    
    def test_build_dataset_medium2(self):
        NeuObj.clearNames()
        input1 = Input('in1')
        input2 = Input('in2')
        output = Input('out')
        rel1 = Fir(input1.tw(0.05))
        rel2 = Fir(input1.tw(0.01))
        rel3 = Fir(input2.tw(0.02))

        test = Modely(visualizer=None)
        test.addMinimize('out', output.z(-1), rel1 + rel2 + rel3)
        test.neuralizeModel(0.01)

        data_struct = ['x1','y1','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']
        test.loadData(name='dataset', source=train_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        self.assertEqual((10,5,1), test._data['dataset']['in1'].shape)
        self.assertEqual([[0.984],[0.983],[0.982],[0.98],[0.977]], test._data['dataset']['in1'][0].tolist())

        self.assertEqual((10,2,1), test._data['dataset']['in2'].shape)
        self.assertEqual([[12.498], [12.502]], test._data['dataset']['in2'][0].tolist())

        self.assertEqual((10,1,1), test._data['dataset']['out'].shape)
        self.assertEqual([[[1.225]], [[1.224]], [[1.222]], [[1.22]], [[1.217]], [[1.214]], [[1.211]], [[1.207]], [[1.204]], [[1.2]]], test._data['dataset']['out'].tolist())
    
    def test_build_dataset_complex1(self):
        NeuObj.clearNames()
        input1 = Input('in1')
        output = Input('out')
        rel1 = Fir(input1.tw(0.05))
        rel2 = Fir(input1.tw([-0.01,0.02]))

        test = Modely(visualizer=None)
        test.addMinimize('out', output.z(-1), rel1 + rel2)
        test.neuralizeModel(0.01)

        data_struct = ['x1','y1','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']
        test.loadData(name='dataset', source=train_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        self.assertEqual((9,7,1), test._data['dataset']['in1'].shape)
        self.assertEqual([[0.984],[0.983],[0.982],[0.98],[0.977],[0.973],[0.969]], test._data['dataset']['in1'][0].tolist())

        self.assertEqual((9,1,1), test._data['dataset']['out'].shape)
        self.assertEqual([[[1.225]], [[1.224]], [[1.222]], [[1.22]], [[1.217]], [[1.214]], [[1.211]], [[1.207]], [[1.204]]], test._data['dataset']['out'].tolist())

    def test_build_multi_dataset_complex1(self):
        NeuObj.clearNames()
        input1 = Input('in1')
        output = Input('out')
        rel1 = Fir(input1.tw(0.05))
        rel2 = Fir(input1.tw([-0.01,0.02]))

        test = Modely(visualizer=None)
        test.addMinimize('out', output.z(-1), rel1 + rel2)
        test.neuralizeModel(0.01)

        data_struct = ['x1','y1','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']

        test.loadData(name='train_dataset', source=train_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        test.loadData(name='validation_dataset', source=val_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        test.loadData(name='test_dataset', source=test_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)

        self.assertEqual(3, test._Loader__n_datasets)

        self.assertEqual((9,7,1), test._data['train_dataset']['in1'].shape)
        self.assertEqual([[0.984],[0.983],[0.982],[0.98],[0.977],[0.973],[0.969]], test._data['train_dataset']['in1'][0].tolist())
        self.assertEqual((5,7,1), test._data['validation_dataset']['in1'].shape)
        self.assertEqual([[0.884],[0.883],[0.882],[0.88],[0.877],[0.873],[0.869]], test._data['validation_dataset']['in1'][0].tolist())
        self.assertEqual((7,7,1), test._data['test_dataset']['in1'].shape)
        self.assertEqual([[0.784],[0.783],[0.782],[0.78],[0.777],[0.773],[0.769]], test._data['test_dataset']['in1'][0].tolist())

        self.assertEqual((9,1,1), test._data['train_dataset']['out'].shape)
        self.assertEqual([[[1.225]], [[1.224]], [[1.222]], [[1.22]], [[1.217]], [[1.214]], [[1.211]], [[1.207]], [[1.204]]], test._data['train_dataset']['out'].tolist())
        self.assertEqual((5,1,1), test._data['validation_dataset']['out'].shape)
        self.assertEqual([[[2.225]], [[2.224]], [[2.222]], [[2.22]], [[2.217]]], test._data['validation_dataset']['out'].tolist())
        self.assertEqual((7,1,1), test._data['test_dataset']['out'].shape)
        self.assertEqual([[[3.225]], [[3.224]], [[3.222]], [[3.22]], [[3.217]], [[3.214]], [[3.211]]], test._data['test_dataset']['out'].tolist())
    
    def test_build_dataset_complex2(self):
        NeuObj.clearNames()
        input1 = Input('in1')
        output = Input('out')
        rel1 = Fir(input1.tw(0.05))
        rel2 = Fir(input1.tw([-0.01,0.01]))

        test = Modely(visualizer=None)
        test.addMinimize('out', output.z(-1), rel1 + rel2)
        test.neuralizeModel(0.01)

        data_struct = ['x1','y1','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']
        test.loadData(name='dataset', source=train_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        self.assertEqual((10,6,1), test._data['dataset']['in1'].shape)
        self.assertEqual([[0.984],[0.983],[0.982],[0.98],[0.977],[0.973]], test._data['dataset']['in1'][0].tolist())

        self.assertEqual((10,1,1), test._data['dataset']['out'].shape)
        self.assertEqual([[[1.225]], [[1.224]], [[1.222]], [[1.22]], [[1.217]], [[1.214]], [[1.211]], [[1.207]], [[1.204]], [[1.2]]], test._data['dataset']['out'].tolist())
    
    def test_build_dataset_complex3(self):
        NeuObj.clearNames()
        input1 = Input('in1')
        input2 = Input('in2')
        output = Input('out')
        rel1 = Fir(input1.tw(0.05))
        rel2 = Fir(input2.tw(0.02))
        rel3 = Fir(input1.tw([-0.01,0.01]))
        rel4 = Fir(input2.last())
        fun = Output('out-net',rel1+rel2+rel3+rel4)

        test = Modely(visualizer=None)
        test.addModel('fun', fun)
        test.addMinimize('out', output.z(-1), rel1 + rel2 + rel3 + rel4)
        test.neuralizeModel(0.01)

        data_struct = ['x1','y1','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']
        test.loadData(name='dataset', source=train_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        self.assertEqual((10,6,1), test._data['dataset']['in1'].shape)
        self.assertEqual([[0.984],[0.983],[0.982],[0.98],[0.977],[0.973]], test._data['dataset']['in1'][0].tolist())
        
        self.assertEqual((10,2,1), test._data['dataset']['in2'].shape)
        self.assertEqual([[12.498], [12.502]], test._data['dataset']['in2'][0].tolist())

        self.assertEqual((10,1,1), test._data['dataset']['out'].shape)
        self.assertEqual([[[1.225]], [[1.224]], [[1.222]], [[1.22]], [[1.217]], [[1.214]], [[1.211]], [[1.207]], [[1.204]], [[1.2]]], test._data['dataset']['out'].tolist())

    def test_build_multi_dataset_complex3(self):
        NeuObj.clearNames()
        input1 = Input('in1')
        input2 = Input('in2')
        output = Input('out')
        rel1 = Fir(input1.tw(0.05))
        rel2 = Fir(input2.tw(0.02))
        rel3 = Fir(input1.tw([-0.01,0.01]))
        rel4 = Fir(input2.last())
        fun = Output('out-net',rel1+rel2+rel3+rel4)

        test = Modely(visualizer=None)
        test.addModel('fun',fun)
        test.addMinimize('out', output.z(-1), rel1 + rel2 + rel3 + rel4)
        test.neuralizeModel(0.01)

        data_struct = ['x1','y1','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']
        test.loadData(name='train_dataset', source=train_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        test.loadData(name='validation_dataset', source=val_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        test.loadData(name='test_dataset', source=test_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)

        self.assertEqual(3, test._Loader__n_datasets)

        self.assertEqual((10,6,1), test._data['train_dataset']['in1'].shape)
        self.assertEqual([[0.984],[0.983],[0.982],[0.98],[0.977],[0.973]], test._data['train_dataset']['in1'][0].tolist())
        self.assertEqual((6,6,1), test._data['validation_dataset']['in1'].shape)
        self.assertEqual([[0.884],[0.883],[0.882],[0.88],[0.877],[0.873]], test._data['validation_dataset']['in1'][0].tolist())
        self.assertEqual((8,6,1), test._data['test_dataset']['in1'].shape)
        self.assertEqual([[0.784],[0.783],[0.782],[0.78],[0.777],[0.773]], test._data['test_dataset']['in1'][0].tolist())
        
        self.assertEqual((10,2,1), test._data['train_dataset']['in2'].shape)
        self.assertEqual([[12.498], [12.502]], test._data['train_dataset']['in2'][0].tolist())
        self.assertEqual((6,2,1), test._data['validation_dataset']['in2'].shape)
        self.assertEqual([[12.498], [12.502]], test._data['validation_dataset']['in2'][0].tolist())
        self.assertEqual((8,2,1), test._data['test_dataset']['in2'].shape)
        self.assertEqual([[12.498], [12.502]], test._data['test_dataset']['in2'][0].tolist())

        self.assertEqual((10,1,1), test._data['train_dataset']['out'].shape)
        self.assertEqual([[[1.225]], [[1.224]], [[1.222]], [[1.22]], [[1.217]], [[1.214]], [[1.211]], [[1.207]], [[1.204]], [[1.2]]], test._data['train_dataset']['out'].tolist())
        self.assertEqual((6,1,1), test._data['validation_dataset']['out'].shape)
        self.assertEqual([[[2.225]], [[2.224]], [[2.222]], [[2.22]], [[2.217]], [[2.214]]], test._data['validation_dataset']['out'].tolist())
        self.assertEqual((8,1,1), test._data['test_dataset']['out'].shape)
        self.assertEqual([[[3.225]], [[3.224]], [[3.222]], [[3.22]], [[3.217]], [[3.214]], [[3.211]], [[3.207]]], test._data['test_dataset']['out'].tolist())
    
    def test_build_dataset_complex5(self):
        NeuObj.clearNames()
        input1 = Input('in1')
        output = Input('out')
        rel1 = Fir(input1.tw(0.05))
        rel2 = Fir(input1.tw([-0.01,0.01]))
        rel3 = Fir(input1.tw([-0.02,0.02]))
        fun = Output('out-net',rel1+rel2+rel3)

        test = Modely(visualizer=None)
        test.addModel('fun', fun)
        test.addMinimize('out', output.z(-1), fun)
        test.neuralizeModel(0.01)

        data_struct = ['x1','y1','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']
        test.loadData(name='dataset', source=train_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        self.assertEqual((9,7,1), test._data['dataset']['in1'].shape)
        self.assertEqual([[0.984],[0.983],[0.982],[0.98],[0.977],[0.973],[0.969]], test._data['dataset']['in1'][0].tolist())

        self.assertEqual((9,1,1), test._data['dataset']['out'].shape)
        self.assertEqual([[[1.225]], [[1.224]], [[1.222]], [[1.22]], [[1.217]], [[1.214]], [[1.211]], [[1.207]], [[1.204]]], test._data['dataset']['out'].tolist())

    def test_filter_data(self):
        NeuObj.clearNames()
        input1 = Input('in1')
        output = Input('out')
        rel1 = Fir(input1.tw(0.05))
        rel2 = Fir(input1.tw([-0.01,0.01]))
        rel3 = Fir(input1.tw([-0.02,0.02]))
        fun = Output('out-net',rel1+rel2+rel3)

        test = Modely(visualizer=None)
        test.addModel('fun', fun)
        test.addMinimize('out', output.z(-1), fun)
        test.neuralizeModel(0.01)

        data_struct = ['x1','y1','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']
        test.loadData(name='dataset', source=train_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)

        def filter_fn(sample):
            return min(sample['in1']) > 0.957

        test.filterData(filter_fn)

        self.assertEqual((2,7,1), test._data['dataset']['in1'].shape)
        self.assertEqual([[0.984],[0.983],[0.982],[0.98],[0.977],[0.973],[0.969]], test._data['dataset']['in1'][0].tolist())

        self.assertEqual((2,1,1), test._data['dataset']['out'].shape)
        self.assertEqual([[[1.225]], [[1.224]]], test._data['dataset']['out'].tolist())

        test.loadData(name='dataset2', source=train_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        test.filterData(filter_fn, dataset_name='dataset2')
        self.assertEqual((2,7,1), test._data['dataset2']['in1'].shape)
        self.assertEqual((2, 1, 1), test._data['dataset2']['out'].shape)

    def test_build_dataset_complex6(self):
        NeuObj.clearNames()
        input1 = Input('in1')
        output = Input('out')
        rel1 = Fir(input1.tw(0.05))
        rel2 = Fir(input1.tw([-0.01,0.02]))
        rel3 = Fir(input1.tw([-0.05,0.01]))
        fun = Output('out-net',rel1+rel2+rel3)

        test = Modely(visualizer=None)
        test.addModel('fun',fun)
        test.addMinimize('out', output.z(-1), fun)
        test.neuralizeModel(0.01)

        data_struct = ['x1','y1','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time']
        test.loadData(name='dataset', source=train_folder, format=data_struct, skiplines=4, delimiter='\t', header=None)
        self.assertEqual((9,7,1), test._data['dataset']['in1'].shape)
        self.assertEqual([[0.984],[0.983],[0.982],[0.98],[0.977],[0.973],[0.969]], test._data['dataset']['in1'][0].tolist())

        self.assertEqual((9,1,1), test._data['dataset']['out'].shape)
        self.assertEqual([[[1.225]], [[1.224]], [[1.222]], [[1.22]], [[1.217]], [[1.214]], [[1.211]], [[1.207]], [[1.204]]], test._data['dataset']['out'].tolist())

    def test_build_dataset_custom(self):
        NeuObj.clearNames()
        input1 = Input('in1')
        output = Input('out')
        rel1 = Fir(input1.tw(0.05))
        rel2 = Fir(input1.tw([-0.01,0.02]))
        rel3 = Fir(input1.tw([-0.05,0.01]))
        fun = Output('out-net',rel1+rel2+rel3)

        test = Modely(visualizer=None)
        test.addModel('fun',fun)
        test.addMinimize('out', output.z(-1), fun)
        test.neuralizeModel(0.01)

        data_x = np.array(range(10))
        data_a = 2
        data_b = -3
        dataset = {'in1': data_x, 'out': (data_a*data_x) + data_b}
        
        test.loadData(name='dataset',source=dataset)
        self.assertEqual((4,7,1), test._data['dataset']['in1'].shape)
        self.assertEqual([[[0],[1],[2],[3],[4],[5],[6]],
                        [[1],[2],[3],[4],[5],[6],[7]],
                        [[2],[3],[4],[5],[6],[7],[8]],
                        [[3],[4],[5],[6],[7],[8],[9]]],
                         test._data['dataset']['in1'].tolist())

        self.assertEqual((4,1,1), test._data['dataset']['out'].shape)
        self.assertEqual([[[7]],[[9]],[[11]],[[13]]], test._data['dataset']['out'].tolist())

    def test_build_multi_dataset_custom(self):
        NeuObj.clearNames()
        input1 = Input('in1')
        output = Input('out')
        rel1 = Fir(input1.tw(0.05))
        rel2 = Fir(input1.tw([-0.01, 0.02]))
        rel3 = Fir(input1.tw([-0.05, 0.01]))
        fun = Output('out-net', rel1 + rel2 + rel3)

        test = Modely(visualizer=None)
        test.addModel('fun',fun)
        test.addMinimize('out', output.z(-1), fun)
        test.neuralizeModel(0.01)

        train_data_x = np.array(range(10))
        val_data_x = np.array(range(10, 20))
        test_data_x = np.array(range(20, 30))
        data_a = 2
        data_b = -3
        train_dataset = {'in1': train_data_x, 'out': (data_a * train_data_x) + data_b}
        val_dataset = {'in1': val_data_x, 'out': (data_a * val_data_x) + data_b}
        test_dataset = {'in1': test_data_x, 'out': (data_a * test_data_x) + data_b}

        test.loadData(name='train_dataset', source=train_dataset)
        test.loadData(name='val_dataset', source=val_dataset)
        test.loadData(name='test_dataset', source=test_dataset)

        self.assertEqual(3, test._Loader__n_datasets)

        self.assertEqual((4, 7, 1), test._data['train_dataset']['in1'].shape)
        self.assertEqual([[[0], [1], [2], [3], [4], [5], [6]],
                          [[1], [2], [3], [4], [5], [6], [7]],
                          [[2], [3], [4], [5], [6], [7], [8]],
                          [[3], [4], [5], [6], [7], [8], [9]]],
                         test._data['train_dataset']['in1'].tolist())
        self.assertEqual((4, 7, 1), test._data['val_dataset']['in1'].shape)
        self.assertEqual([[[10], [11], [12], [13], [14], [15], [16]],
                          [[11], [12], [13], [14], [15], [16], [17]],
                          [[12], [13], [14], [15], [16], [17], [18]],
                          [[13], [14], [15], [16], [17], [18], [19]]],
                         test._data['val_dataset']['in1'].tolist())
        self.assertEqual((4, 7, 1), test._data['test_dataset']['in1'].shape)
        self.assertEqual([[[20], [21], [22], [23], [24], [25], [26]],
                          [[21], [22], [23], [24], [25], [26], [27]],
                          [[22], [23], [24], [25], [26], [27], [28]],
                          [[23], [24], [25], [26], [27], [28], [29]]],
                         test._data['test_dataset']['in1'].tolist())

        self.assertEqual((4, 1, 1), test._data['train_dataset']['out'].shape)
        self.assertEqual([[[7]], [[9]], [[11]], [[13]]], test._data['train_dataset']['out'].tolist())
        self.assertEqual((4, 1, 1), test._data['val_dataset']['out'].shape)
        self.assertEqual([[[27]], [[29]], [[31]], [[33]]], test._data['val_dataset']['out'].tolist())
        self.assertEqual((4, 1, 1), test._data['test_dataset']['out'].shape)
        self.assertEqual([[[47]], [[49]], [[51]], [[53]]], test._data['test_dataset']['out'].tolist())

    def test_vector_input_dataset(self):
        NeuObj.clearNames()
        x = Input('x', dimensions=4)
        y = Input('y', dimensions=3)
        k = Input('k', dimensions=2)
        w = Input('w')


        out = Output('out', Fir(Linear(Linear(3)(x.tw(0.02)) + y.tw(0.02))))
        out2 = Output('out2', Fir(Linear(k.last() + Fir(2)(w.tw(0.05,offset=-0.02)))))

        test = Modely(visualizer=None)
        test.addMinimize('out', out, out2)
        test.neuralizeModel(0.01)

        ## Custom dataset
        data_x = np.transpose(np.array(
                 [np.linspace(1,100,100, dtype=np.float32),
                  np.linspace(2, 101, 100, dtype=np.float32),
                  np.linspace(3, 102, 100, dtype=np.float32),
                  np.linspace(4, 103, 100, dtype=np.float32)]))
        data_y = np.transpose(np.array(
                 [np.linspace(1,100,100, dtype=np.float32) + 10,
                  np.linspace(2, 101, 100, dtype=np.float32) + 10,
                  np.linspace(3, 102, 100, dtype=np.float32) + 10]))
        data_k = np.transpose(np.array(
                 [np.linspace(1,100,100, dtype=np.float32) + 20,
                  np.linspace(2, 101, 100, dtype=np.float32) + 20]))
        data_w = np.linspace(1,100,100, dtype=np.float32) + 30
        dataset = {'x': data_x, 'y': data_y, 'w': data_w, 'k': data_k}

        test.loadData(name='dataset', source=dataset)

        self.assertEqual((96, 2, 4), test._data['dataset']['x'].shape)
        self.assertEqual((96, 2, 3), test._data['dataset']['y'].shape)
        self.assertEqual((96, 1, 2), test._data['dataset']['k'].shape)
        self.assertEqual((96, 5, 1), test._data['dataset']['w'].shape)

        self.assertEqual([[4.0, 5.0, 6.0, 7.0], [5.0, 6.0, 7.0, 8.0]],
                         test._data['dataset']['x'][0].tolist())
        self.assertEqual([[5.0, 6.0, 7.0, 8.0],[6.0, 7.0, 8.0, 9.0]],
                         test._data['dataset']['x'][1].tolist())
        self.assertEqual([[99, 100, 101, 102], [100, 101, 102, 103]],
                         test._data['dataset']['x'][-1].tolist())
        self.assertEqual([[98, 99, 100, 101],[99, 100, 101, 102]],
                         test._data['dataset']['x'][-2].tolist())

        self.assertEqual([[14.0, 15.0, 16.0], [15.0, 16.0, 17.0]],
                         test._data['dataset']['y'][0].tolist())
        self.assertEqual([[15.0, 16.0, 17.0], [16.0, 17.0, 18.0]],
                         test._data['dataset']['y'][1].tolist())
        self.assertEqual([[109, 110, 111], [110, 111, 112]],
                         test._data['dataset']['y'][-1].tolist())
        self.assertEqual([[108, 109, 110], [109, 110, 111]],
                         test._data['dataset']['y'][-2].tolist())

        self.assertEqual([[25.0, 26.0]],
                         test._data['dataset']['k'][0].tolist())
        self.assertEqual([[26.0, 27.0]],
                         test._data['dataset']['k'][1].tolist())
        self.assertEqual([[120, 121]],
                         test._data['dataset']['k'][-1].tolist())
        self.assertEqual([[119, 120]],
                         test._data['dataset']['k'][-2].tolist())

        self.assertEqual([[31], [32], [33], [34], [35]],
                         test._data['dataset']['w'][0].tolist())
        self.assertEqual([[32], [33], [34], [35], [36]],
                         test._data['dataset']['w'][1].tolist())
        self.assertEqual([[126], [127], [128], [129], [130]],
                         test._data['dataset']['w'][-1].tolist())
        self.assertEqual([[125], [126], [127], [128], [129]],
                         test._data['dataset']['w'][-2].tolist())

    def test_vector_input_dataset_files(self):
        NeuObj.clearNames()
        x = Input('x', dimensions=4)
        y = Input('y', dimensions=3)
        k = Input('k', dimensions=2)
        w = Input('w')

        out = Output('out', Fir(Linear(Linear(3)(x.tw(0.02)) + y.tw(0.02))))
        out2 = Output('out2', Fir(Linear(k.last() + Fir(2)(w.tw(0.05,offset=-0.02)))))

        test = Modely(visualizer=None)
        test.addMinimize('out', out, out2)
        test.neuralizeModel(0.01)

        data_folder = os.path.join(os.path.dirname(__file__), 'vector_data/')
        data_struct = ['x', 'y', '','', '', '', 'k', '', '', '', 'w']
        test.loadData(name='dataset', source=data_folder, format=data_struct, skiplines=1, delimiter='\t', header=None)

        self.assertEqual((22, 2, 4), test._data['dataset']['x'].shape)
        self.assertEqual((22, 2, 3), test._data['dataset']['y'].shape)
        self.assertEqual((22, 1, 2), test._data['dataset']['k'].shape)
        self.assertEqual((22, 5, 1), test._data['dataset']['w'].shape)

        self.assertEqual([[0.804,	0.825,	0.320,	0.488], [0.805,	0.825,	0.322,	0.485]],
                         test._data['dataset']['x'][0].tolist())
        self.assertEqual([[0.805,	0.825,	0.322,	0.485],[0.806,	0.824,	0.325,	0.481]],
                         test._data['dataset']['x'][1].tolist())
        self.assertEqual([[0.806,	0.824,	0.325,	0.481], [0.807,	0.823,	0.329,	0.477]],
                         test._data['dataset']['x'][-1].tolist())
        self.assertEqual([[0.805,	0.825,	0.322,	0.485],[0.806,	0.824,	0.325,	0.481]],
                         test._data['dataset']['x'][-2].tolist())

        self.assertEqual([[0.350,	1.375,	0.586], [0.350,	1.375,	0.585]],
                         test._data['dataset']['y'][0].tolist())
        self.assertEqual([[0.350,	1.375,	0.585], [0.350,	1.375,	0.584]],
                         test._data['dataset']['y'][1].tolist())
        self.assertEqual([[0.350,	1.375,	0.584], [0.350,	1.375,	0.582]],
                         test._data['dataset']['y'][-1].tolist())
        self.assertEqual([[0.350,	1.375,	0.585], [0.350,	1.375,	0.584]],
                         test._data['dataset']['y'][-2].tolist())

        self.assertEqual([[0.714,	1.227]],
                         test._data['dataset']['k'][0].tolist())
        self.assertEqual([[0.712,	1.225]],
                         test._data['dataset']['k'][1].tolist())
        self.assertEqual([[0.710,	1.224]],
                         test._data['dataset']['k'][-1].tolist())
        self.assertEqual([[0.712,	1.225]],
                         test._data['dataset']['k'][-2].tolist())

        self.assertEqual([[12.493], [12.493], [12.495], [12.498], [12.502]],
                         test._data['dataset']['w'][0].tolist())
        self.assertEqual([[12.493], [12.495], [12.498], [12.502], [12.508]],
                         test._data['dataset']['w'][1].tolist())
        self.assertEqual([[12.495], [12.498], [12.502], [12.508], [12.515]],
                         test._data['dataset']['w'][-1].tolist())
        self.assertEqual([[12.493], [12.495], [12.498], [12.502], [12.508]],
                         test._data['dataset']['w'][-2].tolist())

        ## Load from file
        ## Try to train the model
        # test.trainModel(splits=[80, 10, 10],
        #                 training_params={'num_of_epochs': 100, 'train_batch_size': 4, 'test_batch_size': 4})

    def test_multifiles(self):
        NeuObj.clearNames()
        x = Input('x')
        relation = Fir()(x.tw(0.05))
        relation.closedLoop(x)
        output = Output('out', relation)

        test = Modely(visualizer=None, log_internal=True)
        test.addModel('model', output)
        test.addMinimize('error', output, x.next())
        test.neuralizeModel(0.01)

        ## The folder contains 3 files with 10, 20 and 30 samples respectively
        data_struct = ['x']
        ## each folder contains 3 files with 10, 20 and 30 samples respectively
        data_folder = os.path.join(os.path.dirname(__file__), 'multifile/')
        data_folder2 = os.path.join(os.path.dirname(__file__), 'multifile2/')
        ## this folder contains only one file with 50 samples
        data_folder3 = os.path.join(os.path.dirname(__file__), 'multifile3/')
        test.loadData(name='dataset1', source=data_folder, format=data_struct, skiplines=1)
        test.loadData(name='dataset2', source=data_folder2, format=data_struct, skiplines=1)
        test.loadData(name='dataset3', source=data_folder3, format=data_struct, skiplines=1) 

        self.assertListEqual(list(test._data['dataset1']['x'].shape), [45, 6, 1])
        self.assertListEqual(test._multifile['dataset1'], [5, 20, 45])
        self.assertListEqual(list(test._data['dataset2']['x'].shape), [45, 6, 1])
        self.assertListEqual(test._multifile['dataset2'], [5, 20, 45])
        self.assertListEqual(list(test._data['dataset3']['x'].shape), [45, 6, 1])
        self.assertEqual(test._num_of_samples['dataset1'], 45) ## 5 + 15 + 25
        self.assertEqual(test._num_of_samples['dataset2'], 45) ## 5 + 15 + 25
        self.assertEqual(test._num_of_samples['dataset3'], 45) ## 50 - 5

        ## train one dataset using splits
        test.trainModel(dataset='dataset1', splits=[80, 10, 10], prediction_samples=3, num_of_epochs=1)
        tp = test.getTrainingInfo()

        self.assertEqual(tp['n_samples_train'], 36) ## 45 * 0.8
        self.assertEqual(tp['n_samples_val'], 4) ## 45 * 0.1
        self.assertEqual(tp['n_samples_test'], 5) ## 45 * 0.1
        self.assertEqual(test.running_parameters['train_indexes'], [0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32])
        self.assertEqual(test.running_parameters['val_indexes'], [0])

        ## train using one dataset for train and one for validation
        test.trainModel(train_dataset='dataset1', validation_dataset='dataset2', prediction_samples=3, num_of_epochs=1)
        tp = test.getTrainingInfo()

        self.assertEqual(tp['n_samples_train'], 45)
        self.assertEqual(tp['n_samples_val'], 45)
        self.assertEqual(tp['n_samples_test'], 0)
        self.assertEqual(test.running_parameters['train_indexes'], [0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41])
        self.assertEqual(test.running_parameters['val_indexes'], [0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41])

        ## train using two dataset for train and one for validation
        test.trainModel(train_dataset=['dataset1', 'dataset2'], validation_dataset='dataset3', prediction_samples=3, num_of_epochs=1)
        tp = test.getTrainingInfo()

        self.assertEqual(tp['n_samples_train'], 90) ## 45 + 45
        self.assertEqual(tp['n_samples_val'], 45)
        self.assertEqual(tp['n_samples_test'], 0)
        self.assertEqual(test.running_parameters['train_indexes'], [0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
                                               45, 46, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86])
        self.assertEqual(test.running_parameters['val_indexes'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41])

        ## train using two dataset for train and two for validation
        test.trainModel(train_dataset=['dataset1', 'dataset2'], validation_dataset=['dataset2','dataset3'], prediction_samples=3, num_of_epochs=1)
        tp = test.getTrainingInfo()

        self.assertEqual(tp['n_samples_train'], 90) ## 45 + 45
        self.assertEqual(tp['n_samples_val'], 90)
        self.assertEqual(tp['n_samples_test'], 0)
        self.assertEqual(test.running_parameters['train_indexes'], [0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
                                               45, 46, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86])
        self.assertEqual(test.running_parameters['val_indexes'], [0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
                                               45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86])

        ## train using two dataset for train and two for validation (dataset4 is ignored)
        test.trainModel(train_dataset=['dataset1', 'dataset2'], validation_dataset=['dataset2','dataset3','dataset4'], num_of_epochs=1, prediction_samples=3)
        tp = test.getTrainingInfo()

        self.assertEqual(tp['n_samples_train'], 90) ## 45 + 45
        self.assertEqual(tp['n_samples_val'], 90)
        self.assertEqual(tp['n_samples_test'], 0)
        self.assertEqual(test.running_parameters['train_indexes'], [0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
                                               45, 46, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86])
        self.assertEqual(test.running_parameters['val_indexes'], [0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
                                               45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86])

        ## Use all datasets by default
        test.trainModel(splits=[80, 10, 10], prediction_samples=3)
        tp = test.getTrainingInfo()

        self.assertEqual(tp['n_samples_train'], 108) ## (45+45+45) * 0.8
        self.assertEqual(tp['n_samples_val'], 14) ## 135 * 0.1
        self.assertEqual(tp['n_samples_test'], 13) ## 135 * 0.1
        self.assertEqual(test.running_parameters['train_indexes'], [0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
                                               45, 46, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86,
                                               90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104])
        self.assertEqual(test.running_parameters['val_indexes'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        ## splits multifile
        test.trainModel(dataset=['dataset1', 'dataset2', 'dataset3'], splits=[80, 10, 10], prediction_samples=3)
        tp = test.getTrainingInfo()

        self.assertEqual(tp['n_samples_train'], 108) ## (45+45+45) * 0.8
        self.assertEqual(tp['n_samples_val'], 14) ## 90 * 0.1
        self.assertEqual(tp['n_samples_test'], 13) ## 90 * 0.1
        self.assertEqual(test.running_parameters['train_indexes'], [0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
                                               45, 46, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86,
                                               90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104])
        self.assertEqual(test.running_parameters['val_indexes'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        ## train one dataset using splits
        test.trainModel(dataset='dataset1', splits=[80, 10, 10], num_of_epochs=1)
        tp = test.getTrainingInfo()

        self.assertEqual(tp['n_samples_train'], 36) ## 45 * 0.8
        self.assertEqual(tp['n_samples_val'], 4) ## 45 * 0.1
        self.assertEqual(tp['n_samples_test'], 5) ## 45 * 0.1
        self.assertEqual(test.running_parameters['train_indexes'], list(range(36)))
        self.assertEqual(test.running_parameters['val_indexes'], list(range(4)))

        ## train using one dataset for train and one for validation
        test.trainModel(train_dataset='dataset1', validation_dataset='dataset2', num_of_epochs=1)
        tp = test.getTrainingInfo()

        self.assertEqual(tp['n_samples_train'], 45)
        self.assertEqual(tp['n_samples_val'], 45)
        self.assertEqual(tp['n_samples_test'], 0)
        self.assertEqual(test.running_parameters['train_indexes'], list(range(45)))
        self.assertEqual(test.running_parameters['val_indexes'], list(range(45)))

        ## train using two dataset for train and one for validation
        test.trainModel(train_dataset=['dataset1', 'dataset2'], validation_dataset='dataset3', num_of_epochs=1)
        tp = test.getTrainingInfo()

        self.assertEqual(tp['n_samples_train'], 90) ## 45 + 45
        self.assertEqual(tp['n_samples_val'], 45)
        self.assertEqual(tp['n_samples_test'], 0)
        self.assertEqual(test.running_parameters['train_indexes'], list(range(90)))
        self.assertEqual(test.running_parameters['val_indexes'], list(range(45)))

        ## train using two dataset for train and two for validation
        test.trainModel(train_dataset=['dataset1', 'dataset2'], validation_dataset=['dataset2','dataset3'], num_of_epochs=1)
        tp = test.getTrainingInfo()

        self.assertEqual(tp['n_samples_train'], 90) ## 45 + 45
        self.assertEqual(tp['n_samples_val'], 90)
        self.assertEqual(tp['n_samples_test'], 0)
        self.assertEqual(test.running_parameters['train_indexes'], list(range(90)))
        self.assertEqual(test.running_parameters['val_indexes'], list(range(90)))

        ## train using two dataset for train and two for validation (dataset4 is ignored)
        test.trainModel(train_dataset=['dataset1', 'dataset2'], validation_dataset=['dataset2','dataset3','dataset4'], num_of_epochs=1)
        tp = test.getTrainingInfo()

        self.assertEqual(tp['n_samples_train'], 90) ## 45 + 45
        self.assertEqual(tp['n_samples_val'], 90)
        self.assertEqual(tp['n_samples_test'], 0)
        self.assertEqual(test.running_parameters['train_indexes'], list(range(90)))
        self.assertEqual(test.running_parameters['val_indexes'], list(range(90)))

        ## splits multifile
        test.trainModel(dataset=['dataset1', 'dataset2', 'dataset3'], splits=[80, 10, 10])
        tp = test.getTrainingInfo()

        self.assertEqual(tp['n_samples_train'], 108) ## (45+45+45) * 0.8
        self.assertEqual(tp['n_samples_val'], 14) ## 90 * 0.1
        self.assertEqual(tp['n_samples_test'], 13) ## 90 * 0.1
        self.assertEqual(test.running_parameters['train_indexes'], list(range(108)))
        self.assertEqual(test.running_parameters['val_indexes'], list(range(14)))

    def test_multifiles_2(self):
        NeuObj.clearNames()
        x = Input('x')
        y = Input('y')
        relation = Fir()(x.tw(0.05))+Fir(y.sw([-2,2]))
        relation.closedLoop(x)
        output = Output('out', relation)

        test = Modely(visualizer=None, log_internal=True)
        test.addModel('model', output)
        test.addMinimize('error', output, x.next())
        test.neuralizeModel(0.01)

        ## The folder contains 3 files with 10, 20 and 30 samples respectively
        data_struct = ['x', 'y']
        data_folder = os.path.join(os.path.dirname(__file__), 'multifile/')
        test.loadData(name='dataset', source=data_folder, format=data_struct, skiplines=1)
        self.assertListEqual(list(test._data['dataset']['x'].shape), [42, 6, 1])
        self.assertListEqual(list(test._data['dataset']['y'].shape), [42, 4, 1])
        self.assertListEqual(test._multifile['dataset'], [4, 18, 42])

    def test_dataframe_multidimensional(self):
        import pandas as pd
        NeuObj.clearNames()
        x = Input('x', dimensions=4)
        y = Input('y', dimensions=3)
        k = Input('k', dimensions=2)
        w = Input('w')

        out = Output('out', Fir(Linear(Linear(3)(x.tw(0.02)) + y.tw(0.02))))
        out2 = Output('out2', Fir(Linear(k.last() + Fir(2)(w.tw(0.05,offset=-0.02)))))

        test = Modely(visualizer=None)
        test.addMinimize('out', out, out2)
        test.neuralizeModel(0.01)

        # Create a DataFrame with random values for each input
        df = pd.DataFrame({
            'x': [np.array([1.0,2.0,3.0,4.0]) for _ in range(10)],
            'y': [np.array([5.0,6.0,7.0]) for _ in range(10)],
            'k': [np.array([8.0,9.0]) for _ in range(10)],
            'w': np.array([10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0])})

        test.loadData(name='dataset', source=df)
        self.assertEqual((6, 2, 4), test._data['dataset']['x'].shape)
        self.assertEqual((6, 2, 3), test._data['dataset']['y'].shape)
        self.assertEqual((6, 1, 2), test._data['dataset']['k'].shape)
        self.assertEqual((6, 5, 1), test._data['dataset']['w'].shape)

    def test_dataframe_single_dimension(self):
        import pandas as pd
        NeuObj.clearNames()
        x = Input('x')
        y = Input('y')
        k = Input('k')
        w = Input('w')

        out = Output('out', Fir(x.tw(0.02) + y.tw(0.02)))
        out2 = Output('out2', Fir(k.last()) + Fir(w.tw(0.05,offset=-0.02)))

        test = Modely(visualizer=None)
        test.addMinimize('out', out, out2)
        test.neuralizeModel(0.01)

        # Create a DataFrame with random values for each input
        df = pd.DataFrame({
            'x': np.linspace(1,100,100, dtype=np.float32),
            'y': np.linspace(1,100,100, dtype=np.float32),
            'k': np.linspace(1,100,100, dtype=np.float32),
            'w': np.linspace(1,100,100, dtype=np.float32)})

        test.loadData(name='dataset', source=df)
        self.assertEqual((96, 2, 1), test._data['dataset']['x'].shape)
        self.assertEqual((96, 2, 1), test._data['dataset']['y'].shape)
        self.assertEqual((96, 1, 1), test._data['dataset']['k'].shape)
        self.assertEqual((96, 5, 1), test._data['dataset']['w'].shape)

    def test_dataframe_resampling(self):
        import pandas as pd
        NeuObj.clearNames()
        x = Input('x')
        y = Input('y')
        k = Input('k')
        w = Input('w')

        out = Output('out', Fir(x.tw(1.0) + y.tw(1.0)))
        out2 = Output('out2', Fir(k.last()) + Fir(w.tw(2.5,offset=-1.0)))

        test = Modely(visualizer=None)
        test.addMinimize('out', out, out2)
        test.neuralizeModel(0.5)

        # Create a DataFrame with random values for each input
        df = pd.DataFrame({
            'time': np.array([1.0,1.5,2.0,4.0,4.5,5.0,7.0,7.5,8.0,8.5], dtype=np.float32),
            'x': np.linspace(1,10,10, dtype=np.float32),
            'y': np.linspace(1,10,10, dtype=np.float32),
            'k': np.linspace(1,10,10, dtype=np.float32),
            'w': np.linspace(1,10,10, dtype=np.float32)})

        test.loadData(name='dataset1', source=df, resampling=True)
        self.assertEqual((12, 2, 1), test._data['dataset1']['x'].shape)
        self.assertEqual((12, 2, 1), test._data['dataset1']['y'].shape)
        self.assertEqual((12, 1, 1), test._data['dataset1']['k'].shape)
        self.assertEqual((12, 5, 1), test._data['dataset1']['w'].shape)

        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.set_index('time', drop=True)
        test.loadData(name='dataset2', source=df, resampling=True)
        self.assertEqual((12, 2, 1), test._data['dataset2']['x'].shape)
        self.assertEqual((12, 2, 1), test._data['dataset2']['y'].shape)
        self.assertEqual((12, 1, 1), test._data['dataset2']['k'].shape)
        self.assertEqual((12, 5, 1), test._data['dataset2']['w'].shape)

        df2 = pd.DataFrame({
            'x': np.linspace(1,10,10, dtype=np.float32),
            'y': np.linspace(1,10,10, dtype=np.float32),
            'k': np.linspace(1,10,10, dtype=np.float32),
            'w': np.linspace(1,10,10, dtype=np.float32)})
        with self.assertRaises(TypeError):
            test.loadData(name='dataset3', source=df2, resampling=True)

    def test_load_data_modalities(self):
        import pandas as pd
        NeuObj.clearNames()
        x = Input('x')
        relation = Fir()(x.tw(0.05))
        relation.closedLoop(x)
        output = Output('out', relation)

        test = Modely(visualizer=None, log_internal=True)
        test.addModel('model', output)
        test.addMinimize('error', output, x.next())
        test.neuralizeModel(0.01)

        ## Case 1: directory with files
        data_struct = ['x']
        data_folder = os.path.join(os.path.dirname(__file__), 'multifile/')
        test.loadData(name='dataset_directory', source=data_folder, format=data_struct, skiplines=1)
        self.assertListEqual(list(test._data['dataset_directory']['x'].shape), [45, 6, 1])
        self.assertListEqual(test._multifile['dataset_directory'], [5, 20, 45])

        ## Case 2: dictionary
        train_data_x = np.array(10*[10] + 20*[20] + 30*[30], dtype=np.float32)
        train_dataset = {'x': train_data_x, 'time': np.array(range(60), dtype=np.float32)}
        test.loadData(name='dataset_dictionary', source=train_dataset, )
        self.assertListEqual(list(test._data['dataset_dictionary']['x'].shape), [55, 6, 1])

        ## Case 3: pandas DataFrame
        df = pd.DataFrame({
            'time': np.array(range(60), dtype=np.float32),
            'x': np.array(10*[10] + 20*[20] + 30*[30], dtype=np.float32)})
        test.loadData(name='dataset_pandas', source=df, resampling=True)
        self.assertListEqual(list(test._data['dataset_pandas']['x'].shape), [5896, 6, 1])
