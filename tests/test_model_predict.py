import sys, os, torch, unittest
import numpy as np

from nnodely import *
from nnodely.basic.relation import NeuObj
from nnodely.support.logger import logging, nnLogger

log = nnLogger(__name__, logging.CRITICAL)
log.setAllLevel(logging.CRITICAL)

sys.path.append(os.getcwd())

# 19 Tests
# This file test the model prediction in particular the output value
# Dimensions
# The first dimension must indicate the time dimension i.e. how many time samples I asked for
# The second dimension indicates the output time dimension for each sample.
# The third is the size of the signal

def myfun(x, P):
    return x*P

def myfun2(a, b ,c):
    import torch
    return torch.sin(a + b) * c

def myfun3(a, b, p1, p2):
    import torch
    at = torch.transpose(a[:, :, 0:2],1,2)
    bt = torch.transpose(b, 1, 2)
    return torch.matmul(p1,at+bt)+p2.t()

class ModelyPredictTest(unittest.TestCase):
    
    def TestAlmostEqual(self, data1, data2, precision=4):
        assert np.asarray(data1, dtype=np.float32).ndim == np.asarray(data2, dtype=np.float32).ndim, f'Inputs must have the same dimension! Received {type(data1)} and {type(data2)}'
        if type(data1) == type(data2) == list:
            self.assertEqual(len(data1),len(data2))
            for pred, label in zip(data1, data2):
                self.TestAlmostEqual(pred, label, precision=precision)
        else:
            self.assertAlmostEqual(data1, data2, places=precision)

    def test_single_in(self):
        NeuObj.clearNames()
        in1 = Input('in1')
        in2 = Input('in2')
        out_fun = Fir(in1.tw(0.1)) + Fir(in2.last())
        out = Output('out', out_fun)
        test = Modely(visualizer=None, seed=1)
        test.addModel('out',out)
        test.neuralizeModel(0.01)
        results = test({'in1': [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]],'in2': [[5]]})
        self.assertEqual(1, len(results['out']))
        self.TestAlmostEqual([33.74938201904297], results['out'])
        results = test({'in1': [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]], 'in2': [[5], [7]]})
        self.assertEqual(2, len(results['out']))
        self.TestAlmostEqual([33.74938201904297, 40.309326171875], results['out'])
        results = test({'in1': [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12]], 'in2': [[5], [7], [9]]})
        self.assertEqual(3, len(results['out']))
        self.TestAlmostEqual([33.74938201904297, 40.309326171875, 46.86927032470703], results['out'])

    def test_activation(self):
        NeuObj.clearNames()
        in1 = Input('in1')
        out_fun = ELU(in1.last()) + Relu(in1.last()) + Tanh(in1.last())
        out = Output('out', out_fun)
        test = Modely(visualizer=None, seed=1)
        test.addModel('out',out)
        test.neuralizeModel()
        results = test({'in1': [-1, -0.5, 0, 0.2, 2, 10]})
        self.assertEqual(6, len(results['out']))
        self.TestAlmostEqual([-1.3937146663665771,-0.8555865287780762,0,0.5973753333091736,4.964027404785156,21.0], results['out'])

    def test_single_in_window(self):
        NeuObj.clearNames()
        # Here there is more sample for each time step but the dimensions of the input is 1
        in1 = Input('in1')

        # Finestre nel tempo
        out1 = Output('x.tw(1)', in1.tw(1))
        out2 = Output('x.tw([-1,0])', in1.tw([-1, 0]))
        out3 = Output('x.tw([-3,0])', in1.tw([-3, 0]))
        out4 = Output('x.tw([1,3])', in1.tw([1, 3]))
        out5 = Output('x.tw([-1,3])', in1.tw([-1, 3]))
        out6 = Output('x.tw([0,1])', in1.tw([0, 1]))
        out7 = Output('x.tw([-3,-2])', in1.tw([-3, -2]))

        ## TODO: adjust the z function
        # Finesatre nei samples 
        out8 = Output('x.z(-1)',  in1.z(-1))
        out9 = Output('x.z(0)',  in1.z(0))
        out10 = Output('x.z(2)',  in1.z(2))
        out11 = Output('x.sw([-1,0])',  in1.sw([-1, 0]))
        out12 = Output('x.sw([1,2])',  in1.sw([1, 2]))
        out13 = Output('x.sw([-3,1])',  in1.sw([-3, 1]))
        out14 = Output('x.sw([-3,-2])',  in1.sw([-3, -2]))
        out15 = Output('x.sw([0,1])',  in1.sw([0, 1]))

        test = Modely(visualizer=None, seed=1)
        #test.addModel('out',[out0,out1,out2,out3,out4,out5,out6,out7,out11,out12,out13,out14,out15])
        test.addModel('out',[out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15])

        test.neuralizeModel(1)
        # Time                  -2,-1,0,1,2,3 # zero represent the last passed instant
        results = test({'in1': [[-2],[-1],[0],[1],[7],[3]]})
        # Time window
        self.assertEqual((1,), np.array(results['x.tw(1)']).shape)
        self.TestAlmostEqual([0], results['x.tw(1)'])
        self.assertEqual((1,), np.array(results['x.tw([-1,0])']).shape)
        self.TestAlmostEqual([0], results['x.tw([-1,0])'])
        self.assertEqual((1, 3), np.array(results['x.tw([-3,0])']).shape)
        self.TestAlmostEqual([[-2, -1, 0]], results['x.tw([-3,0])'])
        self.assertEqual((1, 2), np.array(results['x.tw([1,3])']).shape)
        self.TestAlmostEqual([[7, 3]], results['x.tw([1,3])'])
        self.assertEqual((1, 4), np.array(results['x.tw([-1,3])']).shape)
        self.TestAlmostEqual([[0, 1, 7, 3]], results['x.tw([-1,3])'])
        self.assertEqual((1,), np.array(results['x.tw([0,1])']).shape)
        self.TestAlmostEqual([1], results['x.tw([0,1])'])
        self.assertEqual((1,), np.array(results['x.tw([-3,-2])']).shape)
        self.TestAlmostEqual([-2],results['x.tw([-3,-2])'])
        # Sample window
        self.assertEqual((1,), np.array(results['x.z(-1)']).shape)
        self.TestAlmostEqual([1], results['x.z(-1)'])
        self.assertEqual((1,), np.array(results['x.z(0)']).shape)
        self.TestAlmostEqual([0], results['x.z(0)'])
        self.assertEqual((1,), np.array(results['x.z(2)']).shape)
        self.TestAlmostEqual([-2], results['x.z(2)'])
        self.assertEqual((1,), np.array(results['x.sw([-1,0])']).shape)
        self.TestAlmostEqual([0], results['x.sw([-1,0])'])
        self.assertEqual((1,), np.array(results['x.sw([1,2])']).shape)
        self.TestAlmostEqual([7], results['x.sw([1,2])'])
        self.assertEqual((1,4), np.array(results['x.sw([-3,1])']).shape)
        self.TestAlmostEqual([[-2,-1,0,1]], results['x.sw([-3,1])'])
        self.assertEqual((1,), np.array(results['x.sw([-3,-2])']).shape)
        self.TestAlmostEqual([-2], results['x.sw([-3,-2])'])
        self.assertEqual((1,), np.array(results['x.sw([0,1])']).shape)
        self.TestAlmostEqual([1],results['x.sw([0,1])'])
    
    def test_single_in_window_offset(self):
        NeuObj.clearNames()
        # Here there is more sample for each time step but the dimensions of the input is 1
        in1 = Input('in1')

        # Finestre nel tempo
        out1 = Output('x.tw(1)', in1.tw(1,offset=-1))
        out2 = Output('x.tw([-1,0])', in1.tw([-1, 0],offset=-1))
        out3 = Output('x.tw([1,3])', in1.tw([1, 3],offset=1))
        out4 = Output('x.tw([-3,-2])', in1.tw([-3, -2],offset=-3))

        # Finesatre nei samples
        out5 = Output('x.sw([-1,0])',  in1.sw([-1, 0],offset=-1))
        out6 = Output('x.sw([-3,1])',  in1.sw([-3, 1],offset=-3))
        out7 = Output('x.sw([0,1])',  in1.sw([0, 1],offset=0))
        out8 = Output('x.sw([-3, 3])', in1.sw([-3, 3], offset=2))
        out9 = Output('x.sw([-3, 3])-2', in1.sw([-3, 3], offset=-1))

        test = Modely(visualizer = None, seed = 1)
        test.addModel('out',[out1,out2,out3,out4,out5,out6,out7,out8,out9])

        test.neuralizeModel(1)
        # Time                  -2,-1,0,1,2,3 # zero represent the last passed instant
        results = test({'in1': [[-2],[-1],[0],[1],[7],[3]]})
        # Time window
        self.assertEqual((1,), np.array(results['x.tw(1)']).shape)
        self.TestAlmostEqual([0], results['x.tw(1)'])
        self.assertEqual((1,), np.array(results['x.tw([-1,0])']).shape)
        self.TestAlmostEqual([0], results['x.tw([-1,0])'])
        self.assertEqual((1,2), np.array(results['x.tw([1,3])']).shape)
        self.TestAlmostEqual([[0, -4]], results['x.tw([1,3])'])
        self.assertEqual((1,), np.array(results['x.tw([-3,-2])']).shape)
        self.TestAlmostEqual([0],results['x.tw([-3,-2])'])
        # # Sample window
        self.assertEqual((1,), np.array(results['x.sw([-1,0])']).shape)
        self.TestAlmostEqual([0], results['x.sw([-1,0])'])
        self.assertEqual((1,4), np.array(results['x.sw([-3,1])']).shape)
        self.TestAlmostEqual([[0,1,2,3]], results['x.sw([-3,1])'])
        self.assertEqual((1,), np.array(results['x.sw([0,1])']).shape)
        self.TestAlmostEqual([0],results['x.sw([0,1])'])
        self.assertEqual((1,6), np.array(results['x.sw([-3, 3])']).shape)
        self.TestAlmostEqual([[-5,-4,-3,-2,4,0]],results['x.sw([-3, 3])'])
        self.assertEqual((1,6), np.array(results['x.sw([-3, 3])-2']).shape)
        self.TestAlmostEqual([[-2,-1,0,1,7,3]],results['x.sw([-3, 3])-2'])
    
    def test_multi_in_window_offset(self):
        NeuObj.clearNames()
        # Here there is more sample for each time step but the dimensions of the input is 1
        in1 = Input('in1',dimensions=3)

        # Finestre nel tempo
        out1 = Output('x.tw(1)', in1.tw(1, offset=-1))
        out2 = Output('x.tw([-1,0])', in1.tw([-1, 0], offset=-1))
        out3 = Output('x.tw([1,3])', in1.tw([1, 3], offset=1))
        out4 = Output('x.tw([-3,-2])', in1.tw([-3, -2], offset=-3))

        # Finesatre nei samples
        out5 = Output('x.sw([-1,0])', in1.sw([-1, 0], offset=-1))
        out6 = Output('x.sw([-3,1])', in1.sw([-3, 1], offset=-3))
        out7 = Output('x.sw([0,1])', in1.sw([0, 1], offset=0))
        out8 = Output('x.sw([-3, 3])', in1.sw([-3, 3], offset=2))
        out9 = Output('x.sw([-3, 3])-2', in1.sw([-3, 3], offset=-1))

        test = Modely(visualizer = None, seed = 1)
        test.addModel('out',[out1, out2, out3, out4, out5, out6, out7, out8, out9])

        test.neuralizeModel(1)

        # Single input
        # Time                  -2,         -1,      0,      1,      2,       3 # zero represent the last passed instant
        results = test({'in1': [[-2,3,4],[-1,2,2],[0,0,0],[1,2,3],[2,7,3],[3,3,3]]})
        # Time window
        self.assertEqual((1,1,3), np.array(results['x.tw(1)']).shape)
        self.TestAlmostEqual([[[0,0,0]]], results['x.tw(1)'])
        self.assertEqual((1,1,3), np.array(results['x.tw([-1,0])']).shape)
        self.TestAlmostEqual([[[0,0,0]]], results['x.tw([-1,0])'])
        self.assertEqual((1,2,3), np.array(results['x.tw([1,3])']).shape)
        self.TestAlmostEqual([[[0,0,0], [1,-4,0]]], results['x.tw([1,3])'])
        self.assertEqual((1,1,3), np.array(results['x.tw([-3,-2])']).shape)
        self.TestAlmostEqual([[[0,0,0]]], results['x.tw([-3,-2])'])
        # # Sample window
        self.assertEqual((1,1,3), np.array(results['x.sw([-1,0])']).shape)
        self.TestAlmostEqual([[[0,0,0]]], results['x.sw([-1,0])'])
        self.assertEqual((1,4,3), np.array(results['x.sw([-3,1])']).shape)
        self.TestAlmostEqual([[[0,0,0], [1,-1,-2], [2,-3,-4], [3,-1,-1]]], results['x.sw([-3,1])'])
        self.assertEqual((1,1,3), np.array(results['x.sw([0,1])']).shape)
        self.TestAlmostEqual([[[0,0,0]]], results['x.sw([0,1])'])
        self.assertEqual((1,6,3), np.array(results['x.sw([-3, 3])']).shape)
        self.TestAlmostEqual([[[-5,0,1],[-4,-1,-1], [-3,-3,-3], [-2,-1,0], [-1,4,0], [0,0,0]]], results['x.sw([-3, 3])'])
        self.assertEqual((1,6,3), np.array(results['x.sw([-3, 3])-2']).shape)
        self.TestAlmostEqual([[[-2,3,4],[-1,2,2],[0,0,0],[1,2,3],[2,7,3],[3,3,3]]], results['x.sw([-3, 3])-2'])

        # Multi input
        results = test({'in1': [[-2, 3, 4], [-1, 2, 2], [0, 0, 0], [1, 2, 3], [2, 7, 3], [3, 3, 3], [2, 2, 2]]})
        self.assertEqual((2,6,3), np.array(results['x.sw([-3, 3])']).shape)
        self.TestAlmostEqual([[[-5,0,1], [-4,-1,-1], [-3,-3,-3], [-2,-1,0], [-1,4,0], [0,0,0]],
                                    [[-3,0,0], [-2,-2,-2], [-1,0,1],   [0,5,1],   [1,1,1],  [0,0,0]]], results['x.sw([-3, 3])'])
        self.assertEqual((2,6,3), np.array(results['x.sw([-3, 3])-2']).shape)
        self.TestAlmostEqual([[[-2,3,4],[-1,2,2],[0,0,0],[1,2,3],[2,7,3],[3,3,3]],
                                    [[-2,0,-1],[-1,-2,-3],[0,0,0],[1,5,0],[2,1,0],[1,0,-1]]], results['x.sw([-3, 3])-2'])
    
    def test_single_in_window_offset_aritmetic(self):
        NeuObj.clearNames()
        # Elementwise arithmetic, Activation, Trigonometric
        # the dimensions and time window remain unchanged, for the
        # binary operators must be equal

        in1 = Input('in1')
        in2 = Input('in2', dimensions=2)
        out1 = Output('sum', in1.tw(1, offset=-1) + in1.tw([-1,0]))
        out2 = Output('sub', in1.tw([1, 3], offset=1) - in1.tw([-3, -1], offset=-2))
        out3 = Output('mul', in1.tw([-2, 2]) * in1.tw([-3, 1], offset=-2))

        out4 = Output('sum2', in2.tw(1, offset=-1) + in2.tw([-1,0]))
        out5 = Output('sub2', in2.tw([1, 3], offset=1) - in2.tw([-3, -1], offset=-2))
        out6 = Output('mul2', in2.tw([-2, 2]) * in2.tw([-3, 1], offset=-2))

        test = Modely(visualizer=None)
        test.addModel('out',[out1, out2, out3, out4, out5, out6])

        test.neuralizeModel(1)
        # Single input
        #Time                  -2    -1    0    1     2    3              -2       -1      0        1      2        3
        results = test({'in1': [[1], [2], [8], [4], [-1], [6]], 'in2': [[-2, 3], [-1, 2], [0, 5], [1, 2], [2, 7], [3, 3]]})

        self.assertEqual((1,), np.array(results['sum']).shape)
        self.TestAlmostEqual([8], results['sum'])
        self.assertEqual((1,2), np.array(results['sub']).shape) # [-1,6]+1 - [1,2]-2
        self.TestAlmostEqual([[1,7]], results['sub'])
        self.assertEqual((1,4), np.array(results['mul']).shape) #[2, 8, 4, -1]*[1, 2, 8, 4]-2
        self.TestAlmostEqual([[-2,0,24,-2]], results['mul'])#[2, 8, 4, -1]*[-1, 0, 6, 2]

        self.assertEqual((1,1,2), np.array(results['sum2']).shape)
        self.TestAlmostEqual([[[0,5]]], results['sum2'])
        self.assertEqual((1,2,2), np.array(results['sub2']).shape) #[[2,7],[3,3]]-[2,7] - [[-2,3],[-1,2]]-[-1,2]
        self.TestAlmostEqual([[[1,-1],[1,-4]]], results['sub2']) # [[0,0],[1,-4]] - [[-1,1],[0,0]]
        #[[-1, 2], [0, 5], [1, 2], [2, 7]] * [[-2, 3], [-1, 2], [0, 5], [1, 2]]-[-1, 2]
        # [[-1, 2], [0, 5], [1, 2], [2, 7]] * [[-1, 1], [0, 0], [1, 3], [2, 0]]
        self.assertEqual((1,4,2), np.array(results['mul2']).shape)
        self.TestAlmostEqual([[[1, 2], [0, 0], [1, 6], [4, 0]]], results['mul2'])

        # Multi input
        # Time                  -2 -1  0  1  2  3  4             -2       -1      0        1      2        3         4
        results = test({'in1': [1, 2, 8, 4, -1, 6, 9], 'in2': [[-2, 3], [-1, 2], [0, 5], [1, 2], [2, 7], [3, 3], [0, 0]]})
        self.assertEqual((2,), np.array(results['sum']).shape)
        self.TestAlmostEqual([8,4], results['sum'])
        # [6,9]-6 - [2,8]-8 = [0,3] - [-6,0]
        self.assertEqual((2,2), np.array(results['sub']).shape) # [-1,6]+1 - [1,2]-2
        self.TestAlmostEqual([[1,7],[6,3]], results['sub'])
        #[8, 4, -1, 6] * [2, 8, 4, -1]-8 = [8, 4, -1, 6] * [-6, 0, -4, -9]
        self.assertEqual((2,4), np.array(results['mul']).shape)
        self.TestAlmostEqual([[-2,0,24,-2],[-48,0,4,-54]], results['mul'])

        self.assertEqual((2,1,2), np.array(results['sum2']).shape)
        self.TestAlmostEqual([[[0,5]],[[1, 2]]], results['sum2'])
        self.assertEqual((2,2,2), np.array(results['sub2']).shape)
        #[[3, 3], [0, 0]]-[3,3] - [[-1, 2], [0, 5]]-[0, 5] = [[0, 0], [-3, -3]] - [[-1, -3], [0, 0]]
        self.TestAlmostEqual([[[1,-1],[1,-4]],[[1,3],[-3,-3]]], results['sub2'])
        #[[0, 5], [1, 2], [2, 7], [3, 3]] * [[-1, 2], [0, 5], [1, 2], [2, 7]]-[0, 5]
        #[[0, 5], [1, 2], [2, 7], [3, 3]] * [[-1, -3], [0, 0], [1, -3], [2, 2]]
        self.assertEqual((2,4,2), np.array(results['mul2']).shape)
        self.TestAlmostEqual([[[1, 2], [0, 0], [1, 6], [4, 0]],[[0, -15], [0, 0], [2, -21], [6, 6]]], results['mul2'])

    def test_single_in_window_offset_fir(self):
        NeuObj.clearNames()
        # The input must be scalar and the time dimension is compress to 1,
        # Vector input not allowed, it could be done that a number of fir filters equal to the size of the vector are constructed
        # Should weights be shared or not?
        in1 = Input('in1')
        out1 = Output('Fir3', Fir(3)(in1.last()))
        out2 = Output('Fir5', Fir(5)(in1.tw(1)))#
        out3 = Output('Fir2', Fir(2)(in1.tw([-1,0])))#
        out4 = Output('Fir1', Fir(1)(in1.tw([-3,3])))#
        out5 = Output('Fir7', Fir(7)(in1.tw(3,offset=-1)))#
        out6 = Output('Fir4', Fir(4)(in1.tw([2,3],offset=2)))#
        out7 = Output('Fir6', Fir(6)(in1.sw([-2,-1], offset=-2)))#

        test = Modely(visualizer = None, seed = 1)
        test.addModel('out',[out1,out2,out3,out4,out5,out6,out7])
        test.neuralizeModel(1)
        # Single input
        # Time                 -2    -1    0    1    2    3
        results = test({'in1': [[1], [2], [7], [4], [5], [6]]})
        self.assertEqual((1,1,3), np.array(results['Fir3']).shape)
        self.assertEqual((1,1,5), np.array(results['Fir5']).shape)
        self.assertEqual((1,1,2), np.array(results['Fir2']).shape)
        self.assertEqual((1,), np.array(results['Fir1']).shape)
        self.assertEqual((1,1,7), np.array(results['Fir7']).shape)
        self.assertEqual((1,1,4), np.array(results['Fir4']).shape)
        self.assertEqual((1,1,6), np.array(results['Fir6']).shape)

        # Multi input there are 3 temporal instant
        # Time                 -2 -1  0  1  2  3  4  5
        results = test({'in1': [[1], [2], [7], [4], [5], [6], [7], [8]]})
        self.assertEqual((3,1,3), np.array(results['Fir3']).shape)
        self.assertEqual((3,1,5), np.array(results['Fir5']).shape)
        self.assertEqual((3,1,2), np.array(results['Fir2']).shape)
        self.assertEqual((3,), np.array(results['Fir1']).shape)
        self.assertEqual((3,1,7), np.array(results['Fir7']).shape)
        self.assertEqual((3,1,4), np.array(results['Fir4']).shape)
        self.assertEqual((3,1,6), np.array(results['Fir6']).shape)
    
    def test_fir_and_parameter(self):
        NeuObj.clearNames()
        x = Input('x')
        p1 = Parameter('p1', tw=3, values=[[1],[2],[3],[6],[2],[3]])
        with self.assertRaises(TypeError):
            Fir(W=p1)(x)
        with self.assertRaises(ValueError):
            Fir(W=p1)(x.tw([-3, 1]))
        out1 = Output('out1', Fir(W=p1)(x.tw([-2, 1])))

        p2 = Parameter('p2', sw=1, values=[[-2]])
        with self.assertRaises(TypeError):
            Fir(W=p2)(x.tw([-2, 1]))
        out2 = Output('out2', Fir(W=p2)(x.last()))

        p3 = Parameter('p3', dimensions=2, sw=1, values=[[-2,1]])
        with self.assertRaises(TypeError):
            Fir(W=p3)(x.tw([-2, 1]))
        out3 = Output('out3', Fir(W=p3)(x.last()))

        p4 = Parameter('p4', dimensions=2, tw=2, values=[[-2,1],[2,0],[0,1],[4,0]])
        with self.assertRaises(TypeError):
            Fir(W=p4)(x.sw([-2, 0]))
        out4 = Output('out4', Fir(W=p4)(x.tw([-2, 0])))

        p5 = Parameter('p6', sw=2, dimensions=2, values=[[-2,1],[2,0]])
        with self.assertRaises(TypeError):
            Fir(W = p5)(x)
        with self.assertRaises(TypeError):
            Fir(W = p5)(x.tw([-2,1]))
        with self.assertRaises(ValueError):
            Fir(W = p5)(x.sw([-2,1]))
        out5 = Output('out5', Fir(W=p5)(x.sw([-2, 0])))

        test = Modely(visualizer=None)
        test.addModel('out',[out1, out2, out3, out4, out5])
        test.neuralizeModel(0.5)
        # Time   -3, -2, -1, 0, 1, 2, 3
        input = [-2, -1, 0, 1, 2, 3, 12]
        results = test({'x': input})

        self.assertEqual((2,), np.array(results['out1']).shape)
        self.TestAlmostEqual([15,56], results['out1'])
        self.assertEqual((2,), np.array(results['out2']).shape)
        self.TestAlmostEqual([-2, -4], results['out2'])
        self.assertEqual((2, 1, 2), np.array(results['out3']).shape)
        self.TestAlmostEqual([[[-2,1]], [[-4,2]]], results['out3'])
        self.assertEqual((2, 1, 2), np.array(results['out4']).shape)
        self.TestAlmostEqual([[[6.0, -2.0]], [[10.0, 0.0]]], results['out4'])
        self.assertEqual((2, 1, 2), np.array(results['out5']).shape)
        self.TestAlmostEqual([[[2.0, 0.0]], [[2.0, 1.0]]], results['out5'])
    
    def test_single_in_window_offset_parametric_function(self):
        NeuObj.clearNames()
        # An input dimension is temporal and does not remain unchanged unless redefined on output
        # If there are multiple inputs the function returns an error if the dimensions are not defined
        in1 = Input('in1')
        parfun = ParamFun(myfun)
        out = Output('out', parfun(in1.last()))
        test = Modely(visualizer = None, seed = 1)
        test.addModel('out',out)
        test.neuralizeModel(0.1)

        #results = test({'in1': 1})   
        #self.TestAlmostEqual(results['out'], [0.7576315999031067])
        results = test({'in1': [[1]]})
        self.assertEqual((1,), np.array(results['out']).shape)
        self.TestAlmostEqual(results['out'],[0.7576315999031067])
        results = test({'in1': [2]})
        self.TestAlmostEqual(results['out'],[1.5152631998062134])
        results = test({'in1': [1,2]})
        self.assertEqual((2,), np.array(results['out']).shape)
        self.TestAlmostEqual(results['out'],[0.7576315999031067,1.5152631998062134])

        NeuObj.clearNames('out')
        out = Output('out', ParamFun(myfun)(in1.tw(0.2)))
        test = Modely(visualizer=None, seed = 1)
        test.addModel('out',out)
        test.neuralizeModel(0.1)

        with self.assertRaises(StopIteration):
            results = test({'in1': [1]})
        with self.assertRaises(StopIteration):
            results = test({'in1': [2]})
        results = test({'in1': [1,2]})
        self.assertEqual((1,2), np.array(results['out']).shape)
        self.TestAlmostEqual(results['out'], [[0.7576315999031067, 1.5152631998062134]])
        test({'in1': [[1, 2]]}, num_of_samples=5, sampled=True)
        results = test({'in1': [[1,2]]}, sampled=True)
        self.assertEqual((1,2), np.array(results['out']).shape)
        self.TestAlmostEqual(results['out'], [[0.7576315999031067, 1.5152631998062134]])
        results = test({'in1': [1, 2, 3, 4, 5]})# Qui vengono costruite gli input a due a due con shift di 1
        self.assertEqual((4,2), np.array(results['out']).shape)
        self.TestAlmostEqual(results['out'], [[0.7576315999031067, 1.5152631998062134], [1.5152631998062134, 2.272894859313965], [2.272894859313965, 3.0305263996124268], [3.0305263996124268, 3.7881579399108887]])
        results = test({'in1': [[1, 2], [2, 3], [3, 4], [4, 5]]}, sampled=True)
        self.assertEqual((4,2), np.array(results['out']).shape)
        self.TestAlmostEqual(results['out'], [[0.7576315999031067, 1.5152631998062134], [1.5152631998062134, 2.272894859313965], [2.272894859313965, 3.0305263996124268], [3.0305263996124268, 3.7881579399108887]])

        out = Output('out2', ParamFun(myfun)(in1.last(),in1.last()))
        test = Modely(visualizer=None, seed = 1)
        test.addModel('out',out)
        test.neuralizeModel(0.1)

        #results = test({'in1': 1})
        #self.TestAlmostEqual(results['out'],[1])
        results = test({'in1': [1]})
        self.TestAlmostEqual(results['out2'],[1])
        results = test({'in1': [2]})
        self.TestAlmostEqual(results['out2'],[4])
        results = test({'in1': [1,2]})
        self.TestAlmostEqual(results['out2'],[1,4])

        out = Output('out3', ParamFun(myfun)(in1.tw(0.1), in1.tw(0.1)))
        test = Modely(visualizer=None)
        test.addModel('out',out)
        test.neuralizeModel(0.1)

        #results = test({'in1': 2})
        #self.TestAlmostEqual(results['out'], [4])
        results = test({'in1': [2]})
        self.TestAlmostEqual(results['out3'], [4])
        results = test({'in1': [2,1]})
        self.TestAlmostEqual(results['out3'], [4,1])

        out = Output('out4', ParamFun(myfun)(in1.tw(0.2), in1.tw(0.2)))
        test = Modely(visualizer=None)
        test.addModel('out',out)
        test.neuralizeModel(0.1)

        results = test({'in1': [2,4]})
        self.assertEqual((1,2), np.array(results['out4']).shape)
        self.TestAlmostEqual(results['out4'], [[4,16]])
        results = test({'in1': [[1, 2], [3, 2]]}, sampled=True)
        self.assertEqual((2,2), np.array(results['out4']).shape)
        self.TestAlmostEqual(results['out4'], [[1.0, 4.0], [9.0, 4.0]])

        out = Output('out5', ParamFun(myfun)(in1.tw(0.3), in1.tw(0.3)))
        test = Modely(visualizer=None)
        test.addModel('out',out)
        test.neuralizeModel(0.1)

        with self.assertRaises(StopIteration):
            test({'in1': [2]})

        with self.assertRaises(StopIteration):
            test({'in1': [2, 4]})

        results = test({'in1': [3,2,1]})
        self.assertEqual((1,3), np.array(results['out5']).shape)
        self.TestAlmostEqual(results['out5'], [[9,4,1]])
        results = test({'in1': [[1,2,2],[3,4,5]]}, sampled=True)
        self.assertEqual((2,3), np.array(results['out5']).shape)
        self.TestAlmostEqual(results['out5'], [[1, 4, 4], [9, 16, 25]])
        results = test({'in1': [[3, 2, 1], [2, 1, 0]]}, sampled=True)
        self.assertEqual((2,3), np.array(results['out5']).shape)
        self.TestAlmostEqual(results['out5'], [[9, 4, 1],[4, 1, 0]])
        results = test({'in1': [3,2,1,0]})
        self.assertEqual((2,3), np.array(results['out5']).shape)
        self.TestAlmostEqual(results['out5'], [[9, 4, 1],[4, 1, 0]])
        
        out = Output('out6', ParamFun(myfun)(in1.tw(0.4), in1.tw(0.4)))
        test = Modely(visualizer=None)
        test.addModel('out',out)
        test.neuralizeModel(0.1)
        with self.assertRaises(StopIteration):
            test({'in1': [[1, 2, 2], [3, 4, 5]]})
    
    def test_vectorial_input_parametric_function(self):
        NeuObj.clearNames()
        # Vector input for parametric function
        in1 = Input('in1', dimensions=3)
        in2 = Input('in2', dimensions=2)
        p1 = Parameter('p1', sw=1, dimensions=(3,2), values=[[[1,2],[3,4],[5,6]]])
        p2 = Parameter('p2', sw=1, dimensions=(1,3), values=[[1,2,3]])
        parfun = ParamFun(myfun3, parameters_and_constants=[p1,p2])
        out = Output('out', parfun(in1.last(),in2.last()))
        test = Modely(visualizer = None, seed = 1)
        test.addModel('out',out)
        test.neuralizeModel(0.1)

        results = test({'in1': [[1,2,3]],'in2':[[5,6]]})
        self.assertEqual((1,3), np.array(results['out']).shape)
        self.TestAlmostEqual(results['out'], [[23,52,81]])

        results = test({'in1': [[1,2,3],[5,6,7]],'in2':[[5,6],[7,8]]})
        self.assertEqual((2,3), np.array(results['out']).shape)
        self.TestAlmostEqual(results['out'], [[23,52,81],[41,94,147]])
    
    def test_parametric_function_and_fir(self):
        NeuObj.clearNames()
        in1 = Input('in1')
        in2 = Input('in2')
        out = Output('out', Fir(ParamFun(myfun2)(in1.tw(0.4))))
        test = Modely(visualizer=None, seed=1)
        test.addModel('out',out)
        test.neuralizeModel(0.1)

        with self.assertRaises(StopIteration):
            test({'in1': [1, 2, 2]})
        results = test({'in1': [[1], [2], [2], [4]]})
        self.TestAlmostEqual(results['out'][0], -0.03262542933225632)
        results = test({'in1': [[[1], [2], [2], [4]],[[2], [2], [4], [5]]]}, sampled=True)
        self.TestAlmostEqual(results['out'], [-0.03262542933225632, -0.001211114227771759])
        results = test({'in1': [[1], [2], [2], [4], [5]]})
        self.TestAlmostEqual(results['out'], [-0.03262542933225632, -0.001211114227771759])

        with self.assertRaises(RuntimeError):
            Output('out', Fir(ParamFun(myfun2)(in1.tw(0.4),in2.tw(0.2))))

        NeuObj.clearNames()
        in1 = Input('in1')
        in2 = Input('in2')
        out = Output('out', Fir(ParamFun(myfun2)(in1.tw(0.4),in2.tw(0.4))))
        test = Modely(visualizer=None, seed=1)
        test.addModel('out',out)
        test.neuralizeModel(0.1)
        with self.assertRaises(StopIteration): ## TODO: change to KeyError when checking the inputs
            test({'in1': [[1, 2, 2, 4]]})

        results = test({'in1': [1, 2, 2, 4], 'in2': [1, 2, 2, 4]})
        self.TestAlmostEqual(results['out'], [-0.4379930794239044])
        results = test({'in1': [[1, 2, 2, 4]], 'in2': [[1, 2, 2, 4]]}, sampled=True)
        self.TestAlmostEqual(results['out'], [-0.4379930794239044])
        results = test({'in1': [1, 2, 2, 4, 5], 'in2': [1, 2, 2, 4]})
        self.TestAlmostEqual(results['out'], [-0.4379930794239044])
        results = test({'in1': [[1, 2, 2, 4], [1, 2, 2, 4]], 'in2': [[1, 2, 2, 4]]}, sampled=True)
        self.TestAlmostEqual(results['out'], [-0.4379930794239044])
        results = test({'in1': [[1, 2, 2, 4], [1, 2, 2, 4]], 'in2': [[1, 2, 2, 4], [5, 5, 5, 5]]}, sampled=True)
        self.TestAlmostEqual(results['out'], [-0.4379930794239044,  0.5163354873657227])

        results = test({'in1': [[1, 2, 2, 4]], 'in2': [[1, 2, 2, 4]]}, sampled=True)
        self.TestAlmostEqual(results['out'], [-0.4379930794239044])
        results = test({'in1': [1, 2, 2, 4, 5], 'in2': [1, 2, 2, 4]})
        self.TestAlmostEqual(results['out'], [-0.4379930794239044])
        results = test({'in1': [[1, 2, 2, 4], [1, 2, 2, 4]], 'in2': [[1, 2, 2, 4]]}, sampled=True)
        self.TestAlmostEqual(results['out'], [-0.4379930794239044])
        results = test({'in1': [[1, 2, 2, 4], [1, 2, 2, 4]], 'in2': [[1, 2, 2, 4], [5, 5, 5, 5]]}, sampled=True)
        self.TestAlmostEqual(results['out'], [-0.4379930794239044, 0.5163354873657227])

        NeuObj.clearNames()
        in1 = Input('in1')
        in2 = Input('in2')
        out = Output('out', Fir(3)(ParamFun(myfun2)(in1.tw(0.4), in2.tw(0.4))))
        test = Modely(visualizer=None, seed=1)
        test.addModel('out',out)
        test.neuralizeModel(0.1)

        results = test({'in1': [[1, 2, 2, 4], [1, 2, 2, 4]], 'in2': [[1, 2, 2, 4], [1, 2, 2, 4]]}, sampled=True)
        self.assertEqual((2,1,3), np.array(results['out']).shape)
        self.TestAlmostEqual(results['out'], [[[0.22182656824588776, -0.11421152949333191, 0.5385046601295471]], [[0.22182656824588776, -0.11421152949333191,  0.5385046601295471]]])

        NeuObj.clearNames('out')
        parfun = ParamFun(myfun2)
        with self.assertRaises(TypeError):
            Output('out', parfun(Fir(3)(parfun(in1.tw(0.4), in2.tw(0.4)))))

        parfun = ParamFun(myfun2)
        out = Output('out', parfun(Fir(3)(parfun(in1.tw(0.4)))))
        test = Modely(visualizer=None, seed=1)
        test.addModel('out',out)
        test.neuralizeModel(0.1)

        results = test({'in1': [[1, 2, 2, 4], [2, 1, 1, 3]]}, sampled=True)
        self.assertEqual((2,1,3), np.array(results['out']).shape)
        self.TestAlmostEqual(results['out'], [[[0.2126065045595169, 0.21099068224430084, 0.20540902018547058]], [[0.24776744842529297, 0.2278038114309311, 0.2481299340724945]]])

        results = test({'in1': [1, 2, 2, 4, 3],'in2': [6, 2, 2, 4, 4]})
        self.assertEqual((2,1,3), np.array(results['out']).shape)
        self.TestAlmostEqual(results['out'], [[[0.2126065045595169, 0.21099068224430084, 0.20540902018547058]], [[0.1667831689119339,0.16757671535015106, 0.1605043113231659]]])
        results = test({'in1': [[1, 2, 2, 4],[2, 2, 4, 3]],'in2': [[6, 2, 2, 4],[2, 2, 4, 4]]}, sampled=True)
        self.assertEqual((2,1,3), np.array(results['out']).shape)
        self.TestAlmostEqual(results['out'], [[[0.2126065045595169, 0.21099068224430084, 0.20540902018547058]], [[0.1667831689119339, 0.16757671535015106,0.1605043113231659]]])

    def test_parametric_function_and_fir_with_parameters(self):
        NeuObj.clearNames()
        in1 = Input('in1')
        in2 = Input('in2')
        k1 = Parameter('k1', dimensions=1, tw=0.4, values=[[1.0], [1.0], [1.0], [1.0]])
        k2 = Parameter('k2', dimensions=1, tw=0.4, values=[[1.0], [1.0], [1.0], [1.0]])
        out = Output('out', Fir(ParamFun(myfun2, parameters_and_constants=[k1, k2])(in1.tw(0.4))))
        test = Modely(visualizer=None, seed=42)
        test.addModel('out', out)
        test.neuralizeModel(0.1)

        with self.assertRaises(StopIteration):
            test({'in1': [1, 2, 2]})
        results = test({'in1': [[1], [2], [2], [4]]})
        self.TestAlmostEqual(results['out'][0], 0.06549876928329468)
        results = test({'in1': [[[1], [2], [2], [4]], [[2], [2], [4], [5]]]}, sampled=True)
        self.TestAlmostEqual(results['out'], [0.06549876928329468, -0.38155099749565125])
        results = test({'in1': [[1], [2], [2], [4], [5]]})
        self.TestAlmostEqual(results['out'], [0.06549876928329468, -0.38155099749565125])

        with self.assertRaises(RuntimeError):
            Output('out', Fir(ParamFun(myfun2)(in1.tw(0.4), in2.tw(0.2))))

        NeuObj.clearNames()
        in1 = Input('in1')
        in2 = Input('in2')
        k1 = Parameter('k1', dimensions=1, tw=0.4, values=[[1.0], [1.0], [1.0], [1.0]])
        k_fir = Parameter('k_fir', dimensions=1, tw=0.4, values=[[1.0], [1.0], [1.0], [1.0]])
        out = Output('out', Fir(W=k_fir)(ParamFun(myfun2, parameters_and_constants=[k1])(in1.tw(0.4), in2.tw(0.4))))
        test = Modely(visualizer=None, seed=42)
        test.addModel('out', out)
        test.neuralizeModel(0.1)
        with self.assertRaises(StopIteration):
            test({'in1': [[1, 2, 2, 4]]})

        results = test({'in1': [1, 2, 2, 4], 'in2': [1, 2, 2, 4]})
        self.TestAlmostEqual(results['out'], [0.3850506544113159])
        results = test({'in1': [[1, 2, 2, 4]], 'in2': [[1, 2, 2, 4]]}, sampled=True)
        self.TestAlmostEqual(results['out'], [0.3850506544113159])
        results = test({'in1': [1, 2, 2, 4, 5], 'in2': [1, 2, 2, 4]})
        self.TestAlmostEqual(results['out'], [0.3850506544113159])
        results = test({'in1': [[1, 2, 2, 4], [1, 2, 2, 4]], 'in2': [[1, 2, 2, 4]]}, sampled=True)
        self.TestAlmostEqual(results['out'], [0.3850506544113159])
        results = test({'in1': [[1, 2, 2, 4], [1, 2, 2, 4]], 'in2': [[1, 2, 2, 4], [5, 5, 5, 5]]}, sampled=True)
        self.TestAlmostEqual(results['out'], [0.3850506544113159, 1.446676254272461])

        results = test({'in1': [[1, 2, 2, 4]], 'in2': [[1, 2, 2, 4]]}, sampled=True)
        self.TestAlmostEqual(results['out'], [0.3850506544113159])
        results = test({'in1': [1, 2, 2, 4, 5], 'in2': [1, 2, 2, 4]})
        self.TestAlmostEqual(results['out'], [0.3850506544113159])
        results = test({'in1': [[1, 2, 2, 4], [1, 2, 2, 4]], 'in2': [[1, 2, 2, 4]]}, sampled=True)
        self.TestAlmostEqual(results['out'], [0.3850506544113159])
        results = test({'in1': [[1, 2, 2, 4], [1, 2, 2, 4]], 'in2': [[1, 2, 2, 4], [5, 5, 5, 5]]}, sampled=True)
        self.TestAlmostEqual(results['out'], [0.3850506544113159, 1.446676254272461])

        NeuObj.clearNames()
        in1 = Input('in1')
        in2 = Input('in2')
        k1 = Parameter('k1', dimensions=1, tw=0.4, values=[[1.0], [1.0], [1.0], [1.0]])
        k_fir = Parameter('k_fir', dimensions=3, tw=0.4,
                          values=[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        out = Output('out', Fir(3, W=k_fir)(ParamFun(myfun2, parameters_and_constants=[k1])(in1.tw(0.4), in2.tw(0.4))))
        test = Modely(visualizer=None)
        test.addModel('out', out)
        test.neuralizeModel(0.1)

        results = test({'in1': [[1, 2, 2, 4], [1, 2, 2, 4]], 'in2': [[1, 2, 2, 4], [1, 2, 2, 4]]}, sampled=True)
        self.assertEqual((2, 1, 3), np.array(results['out']).shape)
        self.TestAlmostEqual(results['out'], [[[0.3850506544113159, 0.3850506544113159, 0.3850506544113159]],
                                              [[0.3850506544113159, 0.3850506544113159, 0.3850506544113159]]])

        parfun = ParamFun(myfun2)
        with self.assertRaises(TypeError):
            Output('out', parfun(Fir(3)(parfun(in1.tw(0.4), in2.tw(0.4)))))

        parfun = ParamFun(myfun2)
        out = Output('out2', parfun(Fir(3)(parfun(in1.tw(0.4)))))
        test = Modely(visualizer=None, seed=42)
        test.addModel('out', out)
        test.neuralizeModel(0.1)

        results = test({'in1': [[1, 2, 2, 4], [2, 1, 1, 3]]}, sampled=True)
        self.assertEqual((2, 1, 3), np.array(results['out2']).shape)
        self.TestAlmostEqual(results['out2'], [[[0.790096640586853, 0.7821592688560486, 0.8219361901283264]],
                                              [[0.8505557179450989, 0.7224123477935791, 0.77630215883255]]])

        results = test({'in1': [1, 2, 2, 4, 3], 'in2': [6, 2, 2, 4, 4]})
        self.assertEqual((2, 1, 3), np.array(results['out2']).shape)
        self.TestAlmostEqual(results['out2'], [[[0.790096640586853, 0.7821592688560486, 0.8219361901283264]],
                                              [[-0.09300854057073593, 0.44719645380973816, -0.03044888749718666]]])
        results = test({'in1': [[1, 2, 2, 4], [2, 2, 4, 3]], 'in2': [[6, 2, 2, 4], [2, 2, 4, 4]]}, sampled=True)
        self.assertEqual((2, 1, 3), np.array(results['out2']).shape)
        self.TestAlmostEqual(results['out2'], [[[0.790096640586853, 0.7821592688560486, 0.8219361901283264]],
                                              [[-0.09300854057073593, 0.44719645380973816, -0.03044888749718666]]])

    def test_trigonometri_parameter_and_numeric_constant(self):
        NeuObj.clearNames()
        in1 = Input('in1').last()
        par = Parameter('par', values=5)
        in4 = Input('in4', dimensions=4).last()
        par4 = Parameter('par4', values=[1,2,3,4])
        add = in1 + par + 5.2
        sub = in1 - par - 5.2
        mul = in1 * par * 5.2
        div = in1 / par / 5.2
        pow = in1 ** par ** 2
        sin1 = Sin(par) + Sin(5.2)
        cos1 = Cos(par) + Cos(5.2)
        tan1 = Tan(par) + Tan(5.2)
        relu1 = Relu(par) + Relu(5.2)
        tanh1 = Tanh(par) + Tanh(5.2)
        tot1 = add + sub + mul + div + pow + sin1 + cos1 + tan1 + relu1 + tanh1

        add = 5.2 + in1 + (3 + par)
        sub = - 5.2 - par + (3 - in1)
        mul = 5.2 * in1 * (2 * par)
        div = 5.2 / in1 / (3 / par)
        pow = (0.2 ** in1) + (2 ** par)
        tot11 = add + sub + mul + div + pow

        add4 = in4 + par4 + 5.2
        sub4 = in4 - par4 - 5.2
        mul4 = in4 * par4 * 5.2
        div4 = in4 / par4 / 5.2
        pow4 = in4 ** par4 ** 2
        sin4 = Sin(par4) + Sin(5.2)
        cos4 = Cos(par4) + Cos(5.2)
        tan4 = Tan(par4) + Tan(5.2)
        relu4 = Relu(par4) + Relu(5.2)
        tanh4 = Tanh(par4) + Tanh(5.2)
        tot4 = add4 + sub4 + mul4 + div4 + pow4 + sin4 + cos4 + tan4 + relu4 + tanh4

        add4 = 5.2 + in4 + (3 + par4)
        sub4 = - 5.2 - par4 + (3 - in4)
        mul4 = 5.2 * in4 * (2 * par4)
        div4 = 5.2 / in4 / (3 / par4)
        pow4 = (0.2 ** in4) + (2 ** par4)
        tot41 = add4 + sub4 + mul4 + div4 + pow4

        out1 = Output('out1', tot1)
        out11 = Output('out11', tot11)
        out4 = Output('out4', tot4)
        out41 = Output('out41', tot41)

        linW = Parameter('linW', dimensions=(4,1),values=[[1],[1],[1],[1]])
        outtot = Output('outtot', tot1 + Linear(W=linW)(tot4))
        test = Modely(visualizer=None, seed=1)
        test.addModel('out',[out1,out4,outtot,out11,out41])
        test.neuralizeModel()

        results = test({'in1': [1, 2, -2],'in4': [[6, 2, 2, 4], [7, 2, 2, 4], [-6, -5, 5, 4]]})
        self.assertEqual((3,), np.array(results['out1']).shape)
        self.assertEqual((3,1,4), np.array(results['out4']).shape)
        self.assertEqual((3,), np.array(results['outtot']).shape)

        self.TestAlmostEqual([34.8819529, 33554496.0,  -33554480.0], results['out1'] )
        self.TestAlmostEqual([[[58.9539756, 46.1638031, 554.231201171875, 4294967296.0]], [[67.3462829589843, 46.16380310058594, 554.231201171875, 4294967296.0]], [[ -41.75371170043945, 567.6907348632812, 1953220.375, 4294967296.0]]], results['out4'])
        self.TestAlmostEqual([4294967808.0, 4328522240.0,  4263366656.0], results['outtot'])

        self.assertEqual((3,), np.array(results['out11']).shape)
        self.assertEqual((3,1,4), np.array(results['out41']).shape)

        self.TestAlmostEqual([98.86666667, 146.37333333, -45.33333333], results['out11'])
        self.TestAlmostEqual([[[   70.68895289,    53.37333333,    79.04      ,   190.13493333]],
                                   [[   81.04763185,    53.37333333,    79.04      ,   190.13493333]],
                                   [[15570.31111111,  3030.30666667,   171.04032   ,   190.13493333]]], results['out41'],precision=2)

    def test_check_modify_stream(self):
        NeuObj.clearNames()
        in1 = Input('in1').last()
        par = Parameter('par', values=5)
        add1 = in1 + par # 1 + 5 = 6
        add2 = add1 + 5.2 # 6 + 5.2 = 11.2
        tot1 = add1 + add2 # 6 + 11.2 = 17.2
        out1 = Output('out1', tot1) # = 17.2
        tot2 = add1 + in1 # 6 + 1 = 7
        out12 = Output('out12', tot1 + tot2) # = 24.2
        out2 = Output('out2', tot2) #= 7
        test = Modely(visualizer=None,seed=1)
        test.addModel('out',[out1,out12,out2])
        test.neuralizeModel()

        results = test({'in1': [1]})
        self.assertEqual((1,), np.array(results['out1']).shape)
        self.TestAlmostEqual([17.2], results['out1'] )
        self.TestAlmostEqual([24.2], results['out12'])
        self.TestAlmostEqual([7], results['out2'])


    def test_parameter_and_linear(self):
        NeuObj.clearNames()
        input = Input('in1').last()
        W15 = Parameter('W15', dimensions=(1, 5), values=[[1,2,3,4,5]])
        b15 = Parameter('b15', dimensions=5, values=[1,2,3,4,5])
        input4 = Input('in4', dimensions=4).last()
        W45 = Parameter('W45', dimensions=(4, 5), values=[[1,2,3,4,5],[5,3,3,4,5],[1,2,3,4,7],[-8,2,3,4,5]])
        b45 = Parameter('b45', dimensions=5, values=[5,2,3,4,5])

        o = Output('out' , Linear(input) + Linear(input4))
        o3 = Output('out3' , Linear(3)(input) + Linear(3)(input4))
        oW = Output('outW' , Linear(W = W15)(input) + Linear(W = W45)(input4))
        oWb = Output('outWb' , Linear(W = W15,b = b15)(input) + Linear(W = W45, b = b45)(input4))

        n = Modely(visualizer=None, seed=1)
        n.addModel('out',[o,o3,oW,oWb])
        #n.addModel('out', [oW])
        n.neuralizeModel()
        results = n({'in1': [1, 2], 'in4': [[6, 2, 2, 4], [7, 2, 2, 4]]})
        #self.assertEqual((2,), np.array(results['out']).shape)
        #self.TestAlmostEqual([9.274794578552246,10.3853759765625], results['out'])
        #self.assertEqual((2,1,3), np.array(results['out3']).shape)
        #self.TestAlmostEqual([[[9.247159004211426, 6.103044033050537,7.719359397888184]],[[10.68740463256836, 6.687504291534424,8.585973739624023]]], results['out3'])
        #W15 = torch.tensor([[1,2,3,4,5]])
        #in1 = torch.tensor([[1, 2]])
        #W45 = torch.tensor([[1, 2, 3, 4, 5], [5, 3, 3, 4, 5], [1, 2, 3, 4, 7], [-8, 2, 3, 4, 5]])
        #in4 = torch.tensor([[6, 2, 2, 4], [7, 2, 2, 4]])
        #torch.matmul(W45.t(),in4.t())+torch.matmul(W15.t(),in1)
        self.assertEqual((2, 1, 5), np.array(results['outW']).shape)
        self.TestAlmostEqual([[[-13.0,32.0,45.0,60.0,79.0]],[[-11.0,36.0,51.,68.,89.]]], results['outW'])
        # W15 = torch.tensor([[1,2,3,4,5]])
        # b15 = torch.tensor([[5, 2, 3, 4, 5]])
        # in1 = torch.tensor([[1, 2]])
        # W45 = torch.tensor([[1, 2, 3, 4, 5], [5, 3, 3, 4, 5], [1, 2, 3, 4, 7], [-8, 2, 3, 4, 5]])
        # b45 = torch.tensor([[1, 2, 3, 4, 5]])
        # in4 = torch.tensor([[6, 2, 2, 4], [7, 2, 2, 4]])
        # oo = torch.matmul(W45.t(),in4.t())+b45.t()+torch.matmul(W15.t(),in1)+b15.t()
        self.assertEqual((2, 1, 5), np.array(results['outWb']).shape)
        self.TestAlmostEqual([[[-7, 36, 51, 68, 89]],[[-5, 40, 57, 76, 99]]], results['outWb'])

        NeuObj.clearNames()
        input2 = Input('in1').sw([-1,1])
        input42 = Input('in4', dimensions=4).sw([-1,1])

        o = Output('out' , Linear(input2) + Linear(input42))
        o3 = Output('out3' , Linear(3)(input2) + Linear(3)(input42))
        oW = Output('outW' , Linear(W = W15)(input2) + Linear(W = W45)(input42))
        oWb = Output('outWb' , Linear(W = W15,b = b15)(input2) + Linear(W = W45, b = b45)(input42))
        n = Modely(visualizer=None)
        n.addModel('out',[o,o3,oW,oWb])
        n.neuralizeModel()
        results = n({'in1': [1, 2], 'in4': [[6, 2, 2, 4], [7, 2, 2, 4]]})
        self.assertEqual((1, 2), np.array(results['out']).shape)
        self.TestAlmostEqual([[7.3881096839904785, 7.91458797454834]], results['out'])
        self.assertEqual((1, 2, 3), np.array(results['out3']).shape)
        self.TestAlmostEqual([[[8.117439270019531, 6.014362812042236, 7.489190578460693],
                             [9.261265754699707, 6.2568135261535645, 7.929978370666504]]], results['out3'])
        self.assertEqual((1, 2, 5), np.array(results['outW']).shape)
        self.TestAlmostEqual([[[-13.0, 32.0, 45.0, 60.0, 79.0], [-11.0, 36.0, 51., 68., 89.]]], results['outW'])
        self.assertEqual((1, 2, 5), np.array(results['outWb']).shape)
        self.TestAlmostEqual([[[-7, 36, 51, 68, 89], [-5, 40, 57, 76, 99]]], results['outWb'])

    def test_initialization(self):
        NeuObj.clearNames()
        input = Input('in1')
        W = Parameter('W', dimensions=(1,1), init='init_constant')
        b = Parameter('b', dimensions=1, init='init_constant')
        o = Output('out', Linear(W=W,b=b)(input.last()))

        W5 = Parameter('W5', dimensions=(1,1), init='init_constant', init_params={'value':5})
        b2 = Parameter('b2', dimensions=1, init='init_constant', init_params={'value':2})
        o52 = Output('out52', Linear(W=W5,b=b2)(input.last()))

        par = Parameter('par', dimensions=3, sw=2, init='init_constant')
        opar = Output('outpar', Fir(W=par)(input.sw(2)))

        par2 = Parameter('par2', dimensions=3, sw=2, init='init_constant', init_params={'value':2})
        opar2 = Output('outpar2', Fir(W=par2, b=False)(input.sw(2)))

        ol = Output('outl', Linear(output_dimension=1,b=True,W_init='init_constant',b_init='init_constant')(input.last()))
        ol52 = Output('outl52', Linear(output_dimension=1,b=True,W_init='init_constant',b_init='init_constant',W_init_params={'value':5},b_init_params={'value':2})(input.last()))
        ofpar = Output('outfpar', Fir(output_dimension=3,W_init='init_constant')(input.sw(2)))
        ofpar2 = Output('outfpar2', Fir(output_dimension=3,W_init='init_constant',W_init_params={'value':2})(input.sw(2)))

        outnegexp = Output('outnegexp', Fir(output_dimension=3,W_init='init_negexp')(input.sw(2)))
        outnegexp2 = Output('outnegexp2', Fir(output_dimension=3,W_init='init_negexp',W_init_params={'size_index':1, 'first_value':3, 'lambda':1})(input.sw(2)))

        outexp = Output('outexp', Fir(output_dimension=3,W_init='init_exp')(input.sw(2)))
        outexp2 = Output('outexp2', Fir(output_dimension=3,W_init='init_exp',W_init_params={'size_index':1, 'max_value':2, 'lambda':2, 'monotonicity':'increasing'})(input.sw(2)))
        outexp2D = Output('outexp2D', Fir(output_dimension=3,W_init='init_exp',W_init_params={'size_index':1, 'max_value':2, 'lambda':2, 'monotonicity':'decreasing'})(input.sw(2)))

        outlin = Output('outlin', Fir(output_dimension=3,W_init='init_lin')(input.sw(2)))
        outlin2 = Output('outlin2', Fir(output_dimension=3,W_init='init_lin',W_init_params={'size_index':1, 'first_value':4, 'last_value':5})(input.sw(2)))

        n = Modely(visualizer=None,seed=1)
        n.addModel('model',[o,o52,opar,opar2,ol,ol52,ofpar,ofpar2,outnegexp,outnegexp2,outexp,outexp2,outexp2D,outlin,outlin2])
        n.neuralizeModel()
        results = n({'in1': [1, 1, 2]})
        self.assertEqual((2,), np.array(results['out']).shape)
        self.TestAlmostEqual([2,3], results['out'])
        self.assertEqual((2,), np.array(results['out52']).shape)
        self.TestAlmostEqual([7,12], results['out52'])

        self.assertEqual((2,1,3), np.array(results['outpar']).shape)
        self.TestAlmostEqual([[[2,2,2]],[[3,3,3]]], results['outpar'])
        self.assertEqual((2,1,3), np.array(results['outpar2']).shape)
        self.TestAlmostEqual([[[4,4,4]],[[6,6,6]]], results['outpar2'])

        self.assertEqual((2,), np.array(results['outl']).shape)
        self.TestAlmostEqual([2,3], results['outl'])
        self.assertEqual((2,), np.array(results['outl52']).shape)
        self.TestAlmostEqual([7.0,12.0], results['outl52'])

        self.assertEqual((2,1,3), np.array(results['outfpar']).shape)
        self.TestAlmostEqual([[[2,2,2]],[[3,3,3]]], results['outfpar'])
        self.assertEqual((2,1,3), np.array(results['outfpar2']).shape)
        self.TestAlmostEqual([[[4,4,4]],[[6,6,6]]], results['outfpar2'])

        self.assertEqual((2,1,3), np.array(results['outnegexp']).shape)
        self.TestAlmostEqual([[[1.0497870445251465,1.0497870445251465,1.0497870445251465]],[[2.0497870445251465,2.0497870445251465,2.0497870445251465]]], results['outnegexp'])
        self.assertEqual((2,1,3), np.array(results['outnegexp2']).shape)
        self.TestAlmostEqual([[[2.2072765827178955,3.63918399810791,6.0]],[[3.310914993286133,5.458775997161865,9.0]]], results['outnegexp2'])

        self.assertEqual((2,1,3), np.array(results['outexp']).shape)
        self.TestAlmostEqual([[[1.0497870445251465,1.0497870445251465,1.0497870445251465]],[[1.099574089050293,1.099574089050293,1.099574089050293]]], results['outexp'])
        self.assertEqual((2,1,3), np.array(results['outexp2']).shape)
        self.TestAlmostEqual([[[0.5413411259651184,1.47151780128479,4]],[[0.81201171875,2.2072768211364746,6]]], results['outexp2'])
        self.assertEqual((2,1,3), np.array(results['outexp2D']).shape)
        self.TestAlmostEqual([[[4.0,1.47151780128479,0.5413411259651184]],[[6,2.2072768211364746,0.81201171875]]], results['outexp2D'])

        self.assertEqual((2,1,3), np.array(results['outlin']).shape)
        self.TestAlmostEqual([[[1,1,1]],[[1,1,1]]], results['outlin'])
        self.assertEqual((2,1,3), np.array(results['outlin2']).shape)
        self.TestAlmostEqual([[[8,9,10]],[[12,13.5,15.0]]], results['outlin2'])

    def test_sample_part_and_select(self):
        NeuObj.clearNames()
        in1 = Input('in1')
        # Offset before the sample window
        with self.assertRaises(IndexError):
            in1.sw([-5, -2], offset=-6)
        # Offset after the sample window
        with self.assertRaises(IndexError):
            in1.sw([-5, -2], offset=-2)
        # Offset before the sample window
        with self.assertRaises(IndexError):
            in1.sw([0, 3], offset=-1)
        # Offset after the sample window
        with self.assertRaises(IndexError):
            in1.sw([0, 3], offset=3)

        sw3,sw32 = in1.sw([-5, -2], offset=-4), in1.sw([0, 3], offset=0)
        out_sw3 = Output('in_sw3', sw3)
        out_sw32 = Output('in_sw32', sw32)
        #Get after the window
        with self.assertRaises(ValueError):
            SamplePart(sw3, 0, 4)
        #Empty sample window
        with self.assertRaises(ValueError):
            SamplePart(sw3, 0, 0)
        #Get before the sample window
        with self.assertRaises(ValueError):
            SamplePart(sw3, -1, 0)
        # Get after the window
        with self.assertRaises(ValueError):
            SamplePart(sw32, 0, 4)
        #Empty sample window
        with self.assertRaises(ValueError):
            SamplePart(sw32, 0, 0)
        #Get before the sample window
        with self.assertRaises(ValueError):
            SamplePart(sw32, -1, 0)

        # Offset before the sample window
        with self.assertRaises(IndexError):
            SamplePart(sw3, 0, 3, offset=-1)
        # Offset after the sample window
        with self.assertRaises(IndexError):
            SamplePart(sw3, 0, 3, offset=3)
        # Offset before the sample window
        with self.assertRaises(IndexError):
            SamplePart(sw32, 0, 3, offset=-1)
        # Offset after the sample window
        with self.assertRaises(IndexError):
            SamplePart(sw32, 0, 3, offset=3)
        in_SP1first = Output('in_SP1first', SamplePart(sw3, 0, 1))
        in_SP1mid = Output('in_SP1mid', SamplePart(sw3, 1, 2))
        in_SP1last = Output('in_SP1last', SamplePart(sw3, 2, 3))
        in_SP1all = Output('in_SP1all', SamplePart(sw3, 0, 3))
        in_SP1off1 = Output('in_SP1off1', SamplePart(sw3, 0, 3, offset=0))
        in_SP1off2 = Output('in_SP1off2', SamplePart(sw3, 0, 3, offset=1))
        in_SP1off3 = Output('in_SP1off3', SamplePart(sw3, 0, 3, offset=2))
        with self.assertRaises(ValueError):
            SampleSelect(sw3, -1)
        with self.assertRaises(ValueError):
            SampleSelect(sw3, 3)
        with self.assertRaises(ValueError):
            SampleSelect(sw32, -1)
        with self.assertRaises(ValueError):
            SampleSelect(sw32, 3)
        in_SS1 = Output('in_SS1', SampleSelect(sw3, 0))
        in_SS2 = Output('in_SS2', SampleSelect(sw3, 1))
        in_SS3 = Output('in_SS3', SampleSelect(sw3, 2))
        test = Modely(visualizer=None)
        test.addModel('out',[out_sw3, out_sw32,
                       in_SP1first, in_SP1mid, in_SP1last, in_SP1all, in_SP1off1, in_SP1off2, in_SP1off3,
                       in_SS1, in_SS2, in_SS3])
        test.neuralizeModel()
        results = test({'in1': [0, 1, 2, 3, 4, 5, 6, 7]})

        self.assertEqual((1, 3), np.array(results['in_sw3']).shape)
        self.TestAlmostEqual([[-1,0,1]], results['in_sw3'])
        self.assertEqual((1, 3), np.array(results['in_sw32']).shape)
        self.TestAlmostEqual([[0,1,2]], results['in_sw32'])

        self.assertEqual((1,), np.array(results['in_SP1first']).shape)
        self.TestAlmostEqual([-1], results['in_SP1first'])
        self.assertEqual((1,), np.array(results['in_SP1mid']).shape)
        self.TestAlmostEqual([0], results['in_SP1mid'])
        self.assertEqual((1,), np.array(results['in_SP1last']).shape)
        self.TestAlmostEqual([1], results['in_SP1last'])
        self.assertEqual((1,3), np.array(results['in_SP1all']).shape)
        self.TestAlmostEqual([[-1,0,1]], results['in_SP1all'])
        self.assertEqual((1,3), np.array(results['in_SP1off1']).shape)
        self.TestAlmostEqual([[0,1,2]], results['in_SP1off1'])
        self.assertEqual((1,3), np.array(results['in_SP1off2']).shape)
        self.TestAlmostEqual([[-1,0,1]], results['in_SP1off2'])
        self.assertEqual((1,3), np.array(results['in_SP1off3']).shape)
        self.TestAlmostEqual([[-2,-1,0]], results['in_SP1off3'])

        self.assertEqual((1,), np.array(results['in_SS1']).shape)
        self.TestAlmostEqual([-1], results['in_SS1'])
        self.assertEqual((1,), np.array(results['in_SS2']).shape)
        self.TestAlmostEqual([0], results['in_SS2'])
        self.assertEqual((1,), np.array(results['in_SS3']).shape)
        self.TestAlmostEqual([1], results['in_SS3'])

        results = test({'in1': [0, 1, 2, 3, 4, 5, 6, 7, 10]})

        self.assertEqual((2, 3), np.array(results['in_sw3']).shape)
        self.TestAlmostEqual([[-1, 0, 1],[-1, 0, 1]], results['in_sw3'])
        self.assertEqual((2, 3), np.array(results['in_sw32']).shape)
        self.TestAlmostEqual([[0, 1, 2],[0, 1, 4]], results['in_sw32'])

        self.assertEqual((2,), np.array(results['in_SP1first']).shape)
        self.TestAlmostEqual([-1,-1], results['in_SP1first'])
        self.assertEqual((2,), np.array(results['in_SP1mid']).shape)
        self.TestAlmostEqual([0,0], results['in_SP1mid'])
        self.assertEqual((2,), np.array(results['in_SP1last']).shape)
        self.TestAlmostEqual([1,1], results['in_SP1last'])
        self.assertEqual((2, 3), np.array(results['in_SP1all']).shape)
        self.TestAlmostEqual([[-1, 0, 1],[-1, 0, 1]], results['in_SP1all'])
        self.assertEqual((2, 3), np.array(results['in_SP1off1']).shape)
        self.TestAlmostEqual([[0, 1, 2],[0, 1, 2]], results['in_SP1off1'])
        self.assertEqual((2, 3), np.array(results['in_SP1off2']).shape)
        self.TestAlmostEqual([[-1, 0, 1],[-1, 0, 1]], results['in_SP1off2'])
        self.assertEqual((2, 3), np.array(results['in_SP1off3']).shape)
        self.TestAlmostEqual([[-2, -1, 0],[-2, -1, 0]], results['in_SP1off3'])

        self.assertEqual((2,), np.array(results['in_SS1']).shape)
        self.TestAlmostEqual([-1,-1], results['in_SS1'])
        self.assertEqual((2,), np.array(results['in_SS2']).shape)
        self.TestAlmostEqual([0,0], results['in_SS2'])
        self.assertEqual((2,), np.array(results['in_SS3']).shape)
        self.TestAlmostEqual([1,1], results['in_SS3'])

    def test_time_part(self):
        NeuObj.clearNames()
        in1 = Input('in1')
        # Offset before the time window
        with self.assertRaises(IndexError):
            in1.tw([-5, -2], offset=-6)
        # Offset after the time window
        with self.assertRaises(IndexError):
            in1.tw([-5, -2], offset=-2)
        # Offset before the time window
        with self.assertRaises(IndexError):
            in1.tw([0, 3], offset=-1)
        # Offset after the time window
        with self.assertRaises(IndexError):
            in1.tw([0, 3], offset=3)

        tw3, tw32 = in1.tw([-5, -2], offset=-4), in1.tw([0, 3], offset=0)
        out_tw3 = Output('in_tw3', tw3)
        out_tw32 = Output('in_tw32', tw32)
        # Get after the window
        with self.assertRaises(ValueError):
            TimePart(tw3, 0, 4)
        # Empty time window
        with self.assertRaises(ValueError):
            TimePart(tw3, 0, 0)
        # Get before the time window
        with self.assertRaises(ValueError):
            TimePart(tw3, -1, 0)
        # Get after the window
        with self.assertRaises(ValueError):
            TimePart(tw32, 0, 4)
        # Empty sample window
        with self.assertRaises(ValueError):
            TimePart(tw32, 0, 0)
        # Get before the time window
        with self.assertRaises(ValueError):
            TimePart(tw32, -1, 0)

        # Offset before the time window
        with self.assertRaises(IndexError):
            TimePart(tw3, 0, 3, offset=-1)
        # Offset after the time window
        with self.assertRaises(IndexError):
            TimePart(tw3, 0, 3, offset=3)
        # Offset before the time window
        with self.assertRaises(IndexError):
            TimePart(tw32, 0, 3, offset=-1)
        # Offset after the time window
        with self.assertRaises(IndexError):
            TimePart(tw32, 0, 3, offset=3)

        in_TP1first = Output('in_TP1first', TimePart(tw32, 0, 1))
        in_TP1mid = Output('in_TP1mid', TimePart(tw32, 1, 2))
        in_TP1last = Output('in_TP1last', TimePart(tw32, 2, 3))
        in_TP1all = Output('in_TP1all', TimePart(tw32, 0, 3))
        in_TP1off1 = Output('in_TP1off1', TimePart(tw32, 0, 3, offset=0))
        in_TP1off2 = Output('in_TP1off2', TimePart(tw32, 0, 3, offset=1))
        in_TP1off3 = Output('in_TP1off3', TimePart(tw32, 0, 3, offset=2))

        test = Modely(visualizer=None)
        test.addModel('out',[out_tw3, out_tw32,
                       in_TP1first, in_TP1mid, in_TP1last, in_TP1all, in_TP1off1, in_TP1off2, in_TP1off3])
        test.neuralizeModel(1)
        results = test({'in1': [0, 1, 2, 3, 4, 5, 6, 7]})

        self.assertEqual((1, 3), np.array(results['in_tw3']).shape)
        self.TestAlmostEqual([[-1, 0, 1]], results['in_tw3'])
        self.assertEqual((1, 3), np.array(results['in_tw32']).shape)
        self.TestAlmostEqual([[0, 1, 2]], results['in_tw32'])

        self.assertEqual((1,), np.array(results['in_TP1first']).shape)
        self.TestAlmostEqual([0], results['in_TP1first'])
        self.assertEqual((1,), np.array(results['in_TP1mid']).shape)
        self.TestAlmostEqual([1], results['in_TP1mid'])
        self.assertEqual((1,), np.array(results['in_TP1last']).shape)
        self.TestAlmostEqual([2], results['in_TP1last'])
        self.assertEqual((1, 3), np.array(results['in_TP1all']).shape)
        self.TestAlmostEqual([[0, 1, 2]], results['in_TP1all'])
        self.assertEqual((1, 3), np.array(results['in_TP1off1']).shape)
        self.TestAlmostEqual([[0, 1, 2]], results['in_TP1off1'])
        self.assertEqual((1, 3), np.array(results['in_TP1off2']).shape)
        self.TestAlmostEqual([[-1, 0, 1]], results['in_TP1off2'])
        self.assertEqual((1, 3), np.array(results['in_TP1off3']).shape)
        self.TestAlmostEqual([[-2, -1, 0]], results['in_TP1off3'])

        results = test({'in1': [0, 1, 2, 3, 4, 5, 6, 7, 10]})

        self.assertEqual((2, 3), np.array(results['in_tw3']).shape)
        self.TestAlmostEqual([[-1, 0, 1], [-1, 0, 1]], results['in_tw3'])
        self.assertEqual((2, 3), np.array(results['in_tw32']).shape)
        self.TestAlmostEqual([[0, 1, 2], [0, 1, 4]], results['in_tw32'])

        self.assertEqual((2,), np.array(results['in_TP1first']).shape)
        self.TestAlmostEqual([0, 0], results['in_TP1first'])
        self.assertEqual((2,), np.array(results['in_TP1mid']).shape)
        self.TestAlmostEqual([1, 1], results['in_TP1mid'])
        self.assertEqual((2,), np.array(results['in_TP1last']).shape)
        self.TestAlmostEqual([2, 4], results['in_TP1last'])
        self.assertEqual((2, 3), np.array(results['in_TP1all']).shape)
        self.TestAlmostEqual([[0, 1, 2], [0, 1, 4]], results['in_TP1all'])
        self.assertEqual((2, 3), np.array(results['in_TP1off1']).shape)
        self.TestAlmostEqual([[0, 1, 2], [0, 1, 4]], results['in_TP1off1'])
        self.assertEqual((2, 3), np.array(results['in_TP1off2']).shape)
        self.TestAlmostEqual([[-1, 0, 1], [-1, 0, 3]], results['in_TP1off2'])
        self.assertEqual((2, 3), np.array(results['in_TP1off3']).shape)
        self.TestAlmostEqual([[-2, -1, 0], [-4, -3, 0]], results['in_TP1off3'])
    
    def test_part_and_select(self):
        NeuObj.clearNames()
        in1 = Input('in1',dimensions=4)

        tw3, tw32 = in1.tw([-5, -2], offset=-4), in1.tw([0, 3], offset=0)
        out_tw3 = Output('in_tw3', tw3)
        out_tw32 = Output('in_tw32', tw32)
        # Get after the window
        with self.assertRaises(IndexError):
            Part(tw3, 0, 5)
        # Empty time window
        with self.assertRaises(IndexError):
            Part(tw3, 0, 0)
        # Get before the time window
        with self.assertRaises(IndexError):
            Part(tw3, -1, 0)
        # Get after the window
        with self.assertRaises(IndexError):
            Part(tw32, 0, 5)
        # Empty sample window
        with self.assertRaises(IndexError):
            Part(tw32, 0, 0)
        # Get before the time window
        with self.assertRaises(IndexError):
            Part(tw32, -1, 0)

        in_P1first = Output('in_P1first', Part(tw32, 0, 1))
        in_P1mid = Output('in_P1mid', Part(tw32, 1, 2))
        in_P1last = Output('in_P1last', Part(tw32, 2, 4))
        in_P1all = Output('in_P1all', Part(tw32, 0, 4))

        with self.assertRaises(IndexError):
            Select(tw3, -1)
        with self.assertRaises(IndexError):
            Select(tw3, 4)
        with self.assertRaises(IndexError):
            Select(tw32, -1)
        with self.assertRaises(IndexError):
            Select(tw32, 4)
        in_S1 = Output('in_S1', Select(tw3, 0))
        in_S2 = Output('in_S2', Select(tw3, 1))
        in_S3 = Output('in_S3', Select(tw3, 2))
        in_S4 = Output('in_S4', Select(tw3, 3))

        test = Modely(visualizer=None)
        test.addModel('out',[out_tw3, out_tw32,
                       in_P1first, in_P1mid, in_P1last, in_P1all,
                       in_S1, in_S2, in_S3, in_S4])
        test.neuralizeModel(1)
        results = test({'in1': [[0,1,2,4], [1,3,4,5], [2,5,6,7], [3,3,4,1], [4,4,6,7], [5,6,7,8], [6,7,5,4],[7,2,3,1]]})

        self.assertEqual((1, 3, 4), np.array(results['in_tw3']).shape)
        self.TestAlmostEqual([[[-1,-2,-2,-1], [0,0,0,0], [1,2,2,2]]], results['in_tw3'])
        self.assertEqual((1, 3, 4), np.array(results['in_tw32']).shape)
        self.TestAlmostEqual([[[0,0,0,0], [1,1,-2,-4],[2,-4,-4,-7]]], results['in_tw32'])

        self.assertEqual((1,3), np.array(results['in_P1first']).shape)
        self.TestAlmostEqual([[0,1,2]], results['in_P1first'])
        self.assertEqual((1,3), np.array(results['in_P1mid']).shape)
        self.TestAlmostEqual([[0,1,-4]], results['in_P1mid'])
        self.assertEqual((1,3,2), np.array(results['in_P1last']).shape)
        self.TestAlmostEqual([[[0,0],[-2,-4],[-4,-7]]], results['in_P1last'])
        self.assertEqual((1,3,4), np.array(results['in_P1all']).shape)
        self.TestAlmostEqual([[[0,0,0,0], [1,1,-2,-4],[2,-4,-4,-7]]], results['in_P1all'])

        self.assertEqual((1,3), np.array(results['in_S1']).shape)
        self.TestAlmostEqual([[-1,0,1]], results['in_S1'])
        self.assertEqual((1,3), np.array(results['in_S2']).shape)
        self.TestAlmostEqual([[-2,0,2]], results['in_S2'])
        self.assertEqual((1,3), np.array(results['in_S3']).shape)
        self.TestAlmostEqual([[-2,0,2]], results['in_S3'])
        self.assertEqual((1,3), np.array(results['in_S4']).shape)
        self.TestAlmostEqual([[-1,0,2]], results['in_S4'])

        results = test({'in1': [[0,1,2,4], [1,3,4,5], [2,5,6,7], [3,3,4,1], [4,4,6,7], [5,6,7,8], [6,7,5,4],[7,2,3,1],[0,7,0,0]]})

        self.assertEqual((2, 3, 4), np.array(results['in_tw3']).shape)
        self.TestAlmostEqual([[[-1, -2, -2, -1], [0, 0, 0, 0], [1, 2, 2, 2]],
                                    [[-1, -2, -2, -2], [0, 0, 0, 0], [1, -2, -2, -6]]], results['in_tw3'])
        self.assertEqual((2, 3, 4), np.array(results['in_tw32']).shape)
        self.TestAlmostEqual([[[0, 0, 0, 0], [1, 1, -2, -4], [2, -4, -4, -7]],
                                    [[0, 0, 0, 0], [1, -5, -2, -3], [-6, 0, -5, -4]]], results['in_tw32'])

        self.assertEqual((2, 3), np.array(results['in_P1first']).shape)
        self.TestAlmostEqual([[0, 1, 2],[0, 1, -6]], results['in_P1first'])
        self.assertEqual((2, 3), np.array(results['in_P1mid']).shape)
        self.TestAlmostEqual([[0, 1, -4],[0, -5, 0]], results['in_P1mid'])
        self.assertEqual((2, 3, 2), np.array(results['in_P1last']).shape)
        self.TestAlmostEqual([[[0, 0], [-2, -4], [-4, -7]],
                                    [[0, 0], [-2, -3], [-5, -4]]], results['in_P1last'])
        self.assertEqual((2, 3, 4), np.array(results['in_P1all']).shape)
        self.TestAlmostEqual([[[0, 0, 0, 0], [1, 1, -2, -4], [2, -4, -4, -7]],
                                    [[0, 0, 0, 0], [1, -5, -2, -3], [-6, 0, -5, -4]]], results['in_P1all'])

        self.assertEqual((2, 3), np.array(results['in_S1']).shape)
        self.TestAlmostEqual([[-1, 0, 1],[-1, 0, 1]], results['in_S1'])
        self.assertEqual((2, 3), np.array(results['in_S2']).shape)
        self.TestAlmostEqual([[-2, 0, 2],[-2, 0, -2]], results['in_S2'])
        self.assertEqual((2, 3), np.array(results['in_S3']).shape)
        self.TestAlmostEqual([[-2, 0, 2],[-2, 0, -2]], results['in_S3'])
        self.assertEqual((2, 3), np.array(results['in_S4']).shape)
        self.TestAlmostEqual([[-1, 0, 2],[-2, 0, -6]], results['in_S4'])

    def test_predict_paramfun_param_const(self):
        NeuObj.clearNames()
        input2 = Input('in2')
        pp = Parameter('pp', values=[[7],[8],[9]])
        ll = Constant('ll', values=[[12],[13],[14]])
        oo = Constant('oo', values=[[1],[2],[3]])
        pp, oo, input2.tw(0.03), ll
        def fun_test(x, y, z, k):
            return (x + y) * (z - k)

        NeuObj.clearNames()
        out = Output('out',ParamFun(fun_test,parameters_and_constants=[ll,oo,pp])(input2.tw(0.03)))
        test = Modely(visualizer=None)
        test.addModel('out',[out])
        test.neuralizeModel(0.01)
        results = test({'in2': [0, 1, 2]})
        self.assertEqual((1, 3), np.array(results['out']).shape)
        self.assertEqual([[-72.0, -84.0, -96.0]], results['out'])

        NeuObj.clearNames()
        out = Output('out',ParamFun(fun_test,parameters_and_constants={'z':pp,'y':ll,'k':oo})(input2.tw(0.03)))
        test = Modely(visualizer=None)
        test.addModel('out',[out])
        test.neuralizeModel(0.01)
        results = test({'in2': [0, 1, 2]})
        self.assertEqual((1, 3), np.array(results['out']).shape)
        self.assertEqual([[72.0, 84.0, 96.0]], results['out'])

        NeuObj.clearNames()
        parfun = ParamFun(fun_test)
        out1 = Output('out1', parfun(input2.tw(0.03), ll, pp, oo))
        out2 = Output('out2', parfun(input2.tw(0.03), ll, oo, pp))
        out3 = Output('out3', parfun(pp, oo, input2.tw(0.03), ll))
        test = Modely(visualizer=None)
        test.addModel('out',[out1,out2,out3])
        test.neuralizeModel(0.01)
        results = test({'in2': [0, 1, 2]})
        self.assertEqual((1, 3), np.array(results['out1']).shape)
        self.assertEqual((1, 3), np.array(results['out2']).shape)
        self.assertEqual((1, 3), np.array(results['out3']).shape)
        self.assertEqual([[72.0, 84.0, 96.0]], results['out1'])
        self.assertEqual([[-72.0, -84.0, -96.0]], results['out2'])
        self.assertEqual([[-96.0, -120.0, -144.0]], results['out3'])

    def test_predict_paramfun_map_over_batch(self):
        NeuObj.clearNames()
        input2 = Input('in2')
        pp = Parameter('pp', sw=3, values=[[7],[8],[9]])
        ll = Constant('ll', sw=3, values=[[12],[13],[14]])
        oo = Constant('oo', sw=3, values=[[1],[2],[3]])

        def fun_test(x, y, z, k):
            return (x + y) * (z - k)

        fun_map = ParamFun(fun_test,parameters_and_constants=[ll,oo, pp])
        fun = ParamFun(fun_test, parameters_and_constants=[ll,oo, pp])
        fun_map_2 = ParamFun(fun_test, map_over_batch=True)

        out1 = Output('out1',fun_map(input2.tw(0.03)))
        out2 = Output('out2', fun(input2.tw(0.03)))
        test = Modely(visualizer=None)
        test.addModel('out',[out1,out2])
        test.neuralizeModel(0.01)
        results = test({'in2': [0, 1, 2]})
        self.assertEqual((1, 3), np.array(results['out1']).shape)
        self.assertEqual([[-72.0, -84.0, -96.0]], results['out1'])
        self.assertEqual((1, 3), np.array(results['out2']).shape)
        self.assertEqual([[-72.0, -84.0, -96.0]], results['out2'])

        out3 = Output('out3', fun_map_2(input2.tw(0.03), 4.0, pp, ll))
        out4 = Output('out4', fun_map_2(input2.tw(0.01), 2.0, pp, oo))
        with self.assertRaises(ValueError):
            fun_map_2(4.0, 1, pp, ll)
        test.addModel('out-new', [out3,out4])
        with self.assertRaises(NameError):
            test.addModel('out',[out1,out2])
        test.neuralizeModel(0.01)

        results = test({'in2': [0, 1, 2]})
        self.assertEqual((1, 3), np.array(results['out3']).shape)
        self.assertEqual((1, 3), np.array(results['out4']).shape)
        # ([0,1,2]+4)*([7,8,9]-[12,13,14]) -> [4,5,6]*[-5,-5,-5]
        self.assertEqual([[-20.0, -25.0, -30.0]], results['out3'])
        self.assertEqual([[24.0, 24.0, 24.0]], results['out4'])

    def test_predict_fuzzify(self):
        NeuObj.clearNames()
        input = Input('in1')
        fuzzi = Fuzzify(6, range=[0, 5], functions='Rectangular')(input.last())
        out = Output('out', fuzzi)
        test = Modely(visualizer=None)
        test.addModel('out',[out])
        test.neuralizeModel()
        results = test({'in1': [0, 1, 2]})
        self.assertEqual((3, 1, 6), np.array(results['out']).shape)
        self.assertEqual([[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],[[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]],[[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]], results['out'])

        def fun(x):
            import torch
            return torch.sign(x)

        fuz = Fuzzify(output_dimension=11, range=[-5, 5], functions=[fun,fun])(input.last())
        out = Output('out2', fuz)
        test.addModel('out2',[out])
        test.neuralizeModel()
        results = test({'in1': [0, 1, 2]})
        self.assertEqual((3, 1, 11), np.array(results['out2']).shape)
        self.assertEqual([[[1.0, 1.0, 1.0, 1.0, 1.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0]],
                                [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, -1.0, -1.0, -1.0, -1.0]],
                                [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, -1.0, -1.0, -1.0]]], results['out2'])

    def test_sw_on_stream_sw_by_heand(self):
        NeuObj.clearNames()
        input = Input('in1')
        sw_from_input = input.sw(7)

        state = Input('state')
        sw_from_output = Connect(sw_from_input, state)
        out_aux = Output('out_aux', sw_from_output)
        out1 = Output('out1', state.sw(3))

        test = Modely(visualizer=None)
        test.addModel('out_A',  [out_aux,out1])
        test.neuralizeModel()
        results = test({'in1': [14, 1, 2, 3, 4, 5, 6, 7, 8, 9]})
        self.assertEqual((4, 3), np.array(results['out1']).shape)
        self.assertEqual([[4.0, 5.0, 6.0], [5.0, 6.0, 7.0], [6.0, 7.0, 8.0], [7.0, 8.0, 9.0]], results['out1'])

        NeuObj.clearNames()
        state = Input('state')
        out_aux = Output('out_aux', sw_from_input)
        out1 = Output('out1', state.sw(3))

        test = Modely(visualizer=None)
        test.addModel('out_A',  [out_aux,out1])
        test.neuralizeModel()
        results = test({'in1': [14, 1, 2, 3, 4, 5, 6, 7, 8, 9]}, connect={'state': 'out_aux'})
        with self.assertRaises(ValueError):
            test({'in1': [14, 1, 2, 3, 4, 5, 6, 7, 8, 9]}, connect={'out_aux': 'state'})
        self.assertEqual((4, 3), np.array(results['out1']).shape)
        self.assertEqual([[4.0, 5.0, 6.0], [5.0, 6.0, 7.0], [6.0, 7.0, 8.0], [7.0, 8.0, 9.0]], results['out1'])

    def test_sw_on_stream_sw(self):
        NeuObj.clearNames()
        input = Input('in1')
        sw_from_input = input.sw(7)

        out1 = Output('out1', sw_from_input.sw(3))

        test = Modely(visualizer=None)
        test.addModel('out_A',  out1)
        test.neuralizeModel()
        results = test({'in1': [14, 1, 2, 3, 4, 5, 6, 7, 8, 9]})
        self.assertEqual((4, 3), np.array(results['out1']).shape)
        self.assertEqual([[4.0, 5.0, 6.0], [5.0, 6.0, 7.0], [6.0, 7.0, 8.0], [7.0, 8.0, 9.0]], results['out1'])

        NeuObj.clearNames(['out1','out2'])
        out1 = Output('out1', sw_from_input.sw(3))
        out2 = Output('out2', sw_from_input.sw(8))
        with self.assertRaises(ValueError):
            Output('out3', sw_from_input.sw(-1))

        test = Modely(visualizer=None)
        test.addModel('out_A', [out1,out2])
        test.neuralizeModel()
        results = test({'in1': [14, 1, 2, 3, 4, 5, 6, 7, 8, 9]})
        self.assertEqual((4, 3), np.array(results['out1']).shape)
        self.assertEqual([[4.0, 5.0, 6.0], [5.0, 6.0, 7.0], [6.0, 7.0, 8.0], [7.0, 8.0, 9.0]], results['out1'])
        self.assertEqual((4, 8), np.array(results['out2']).shape)
        self.assertEqual([[0.0, 14.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                          [14.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                          [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]], results['out2'])

    def test_tw_on_stream_tw(self):
        NeuObj.clearNames()
        input = Input('in1')
        tw_from_input = input.tw(3.5)

        out1 = Output('out1', tw_from_input.tw(1.5))

        test = Modely(visualizer=None)
        test.addModel('out_A',  out1)
        test.neuralizeModel(0.5)
        results = test({'in1': [14, 1, 2, 3, 4, 5, 6, 7, 8, 9]})
        self.assertEqual((4, 3), np.array(results['out1']).shape)
        self.assertEqual([[4.0, 5.0, 6.0], [5.0, 6.0, 7.0], [6.0, 7.0, 8.0], [7.0, 8.0, 9.0]], results['out1'])

        out1 = Output('out11', tw_from_input.tw(1.5))
        out2 = Output('out21', tw_from_input.tw(4))
        with self.assertRaises(ValueError):
            Output('out3', tw_from_input.tw(-1))

        test = Modely(visualizer=None)
        test.addModel('out_A', [out1,out2])
        test.neuralizeModel(0.5)
        results = test({'in1': [14, 1, 2, 3, 4, 5, 6, 7, 8, 9]})
        self.assertEqual((4, 3), np.array(results['out11']).shape)
        self.assertEqual([[4.0, 5.0, 6.0], [5.0, 6.0, 7.0], [6.0, 7.0, 8.0], [7.0, 8.0, 9.0]], results['out11'])
        self.assertEqual((4, 8), np.array(results['out21']).shape)
        self.assertEqual([[0.0, 14.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                          [14.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                          [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]], results['out21'])

    def test_sw_on_stream_sw_complex(self):
        NeuObj.clearNames()
        input = Input('in1')
        state = Input('state')

        sw_3 = input.sw(3)
        sw_7 = input.sw(7)

        state4 = state.sw(4)
        state8 = state.sw(8)

        out21 = Output('out21', sw_3.sw(2))
        out61 = Output('out61', sw_7.sw(6))
        out22 = Output('out22', SamplePart(sw_3,1,3))
        out62 = Output('out62', SamplePart(sw_7,1,7))
        test = Modely(visualizer=None)
        test.addModel('out_A', [out21,out61,out22,out62])
        test.neuralizeModel()
        results = test({'in1': [14, 1, 2, 3, 4, 5, 6, 7, 8, 9]})
        self.assertEqual(results['out21'], results['out22'])
        self.assertEqual(results['out61'], results['out62'])

        out31 = Output('out31', sw_3)
        out71 = Output('out71', sw_7)
        out32 = Output('out32', SamplePart(state4,1,4))
        out72 = Output('out72', SamplePart(state8,1,8))
        out33 = Output('out33', sw_3.sw(3))
        out73 = Output('out73', sw_7.sw(7))
        test = Modely(visualizer=None)
        test.addModel('out_B', [out31,out71,out32,out72,out33,out73])
        test.addConnect(sw_7, state)
        test.neuralizeModel()
        results = test({'in1': [14, 1, 2, 3, 4, 5, 6, 7, 8, 9]})
        self.assertEqual(results['out31'], results['out32'])
        self.assertEqual(results['out32'], results['out33'])
        self.assertEqual(results['out71'], results['out72'])
        self.assertEqual(results['out72'], results['out73'])

        out41 = Output('out41', state4)
        out42 = Output('out42', sw_3.sw(4))
        test = Modely(visualizer=None)
        test.addModel('out_C', [out41,out42])
        test.addConnect(sw_3, state)
        test.neuralizeModel()
        results = test({'in1': [14, 1, 2, 3, 4, 5, 6, 7, 8, 9]})
        self.assertEqual(results['out41'], results['out42'])

    def test_sw_on_stream_tw_and_opposite(self):
        NeuObj.clearNames()
        input = Input('in1')

        sw_3 = input.sw(3)
        tw_4 = input.tw(1)

        out3tw1 = Output('out3tw1', sw_3.tw(0.2))
        out3tw10 = Output('out3tw10', sw_3.tw(2))
        out4sw2 = Output('out4sw2', tw_4.sw(2))
        out4sw6 = Output('out4sw6', tw_4.sw(6))
        test = Modely(visualizer=None)
        test.addModel('out_A', [out3tw1, out3tw10, out4sw2, out4sw6])
        test.neuralizeModel(0.2)

        results = test({'in1': [14, 1, 2, 3, 4, 5, 6, 7, 8, 9, -3]})
        self.assertEqual(results['out3tw1'], [4, 5, 6, 7, 8, 9, -3])
        self.assertEqual(results['out3tw10'], [[0, 0, 0, 0, 0, 0, 0, 2, 3, 4],
                                               [0, 0, 0, 0, 0,0, 2, 3, 4, 5],
                                               [0, 0, 0, 0, 0, 2, 3, 4 ,5, 6],
                                               [0, 0, 0, 0, 2, 3, 4, 5, 6, 7],
                                                [0, 0, 0, 2, 3, 4, 5, 6, 7, 8],
                                                [0, 0, 2, 3, 4, 5, 6, 7, 8, 9],
                                                [0, 2, 3, 4, 5, 6, 7, 8, 9, -3]])
        self.assertEqual(results['out4sw2'], [[3, 4],[4,5],[5,6],[6,7],[7,8],[8,9], [9,-3]])
        self.assertEqual(results['out4sw6'], [[0, 14, 1, 2, 3, 4],
                                               [14, 1, 2, 3, 4, 5],
                                               [1, 2, 3, 4 ,5, 6],
                                               [2, 3, 4, 5, 6, 7],
                                                [3, 4, 5, 6, 7, 8],
                                                [4, 5, 6, 7, 8, 9],
                                                [5, 6, 7, 8, 9, -3]])

    def test_sw_on_stream_sw_delay(self):
        NeuObj.clearNames()
        input = Input('in1')

        sw_3 = input.sw(3)

        out21 = Output('out21', sw_3.sw(2))
        out41 = Output('out41', sw_3.sw(4))
        out22 = Output('out22', sw_3.sw([-2,0]))
        out42 = Output('out42', sw_3.sw([-4,0]))

        out231 = Output('out231', sw_3.sw([-3,-1]))
        out451 = Output('out451', sw_3.sw([-5,-1]))

        out242 = Output('out242', sw_3.sw([-4,-2]))
        out462 = Output('out462', sw_3.sw([-6,-2]))

        test = Modely(visualizer=None)
        test.addModel('out_A', [out21,out41,out22,out42,out231,out451,out242,out462])
        test.neuralizeModel()
        results = test({'in1': [14, 1, 2, 3, 4, 5, 6, 7, 8, 9]})
        self.assertEqual(results['out21'], results['out22'])
        self.assertEqual(results['out21'], [[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9]])

        self.assertEqual(results['out41'], results['out42'])
        self.assertEqual(results['out41'], [[0, 14, 1, 2], [14, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9]])

        self.assertEqual(results['out231'], [[14,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8]])
        self.assertEqual(results['out451'], [[0, 0, 14, 1], [0, 14, 1, 2], [14, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8]])

        self.assertEqual(results['out242'], [[0,14],[14,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7]])
        self.assertEqual(results['out462'], [[0, 0, 0, 14], [0, 0, 14, 1], [0, 14, 1, 2], [14, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]])

    def test_tw_on_stream_tw_delay(self):
        NeuObj.clearNames()
        input = Input('in1')

        tw_3 = input.tw(1.5)

        out21 = Output('out21', tw_3.tw(1))
        out41 = Output('out41', tw_3.tw(2))
        out22 = Output('out22', tw_3.tw([-1,0]))
        out42 = Output('out42', tw_3.tw([-2,0]))

        out231 = Output('out231', tw_3.tw([-1.5,-0.5]))
        out451 = Output('out451', tw_3.tw([-2.5,-0.5]))

        out242 = Output('out242', tw_3.tw([-2,-1]))
        out462 = Output('out462', tw_3.tw([-3,-1]))

        test = Modely(visualizer=None)
        test.addModel('out_A', [out21,out41,out22,out42,out231,out451,out242,out462])
        test.neuralizeModel(0.5)
        results = test({'in1': [14, 1, 2, 3, 4, 5, 6, 7, 8, 9]})
        self.assertEqual(results['out21'], results['out22'])
        self.assertEqual(results['out21'], [[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9]])

        self.assertEqual(results['out41'], results['out42'])
        self.assertEqual(results['out41'], [[0, 14, 1, 2], [14, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9]])

        self.assertEqual(results['out231'], [[14,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8]])
        self.assertEqual(results['out451'], [[0, 0, 14, 1], [0, 14, 1, 2], [14, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8]])

        self.assertEqual(results['out242'], [[0,14],[14,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7]])
        self.assertEqual(results['out462'], [[0, 0, 0, 14], [0, 0, 14, 1], [0, 14, 1, 2], [14, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]])

    def test_z_on_stream_sw(self):
        NeuObj.clearNames()
        input = Input('inin')
        sw_from_input = input.sw(5)

        out2 = Output('out2', sw_from_input.z(1))
        with self.assertRaises(ValueError):
            Output('out3', sw_from_input.z(-1))
        with self.assertRaises(TypeError):
            Output('out3', sw_from_input.delay(1))

        test = Modely(visualizer=None)
        test.addModel('out_A', [out2])
        test.neuralizeModel(0.5)
        results = test({'inin': [14, 1, 2, 3, 4, 5, 6, 7, 8, 9]})
        self.assertEqual((6, 5), np.array(results['out2']).shape)
        self.assertEqual( [[0.0, 14.0, 1.0, 2.0, 3.0],
                                 [14.0, 1.0, 2.0, 3.0, 4.0],
                                 [1.0, 2.0, 3.0, 4.0, 5.0],
                                 [2.0, 3.0, 4.0, 5.0, 6.0],
                                 [3.0, 4.0, 5.0, 6.0, 7.0],
                                 [4.0, 5.0, 6.0, 7.0, 8.0]], results['out2'])

    def test_delay_on_stream_tw(self):
        NeuObj.clearNames()
        input = Input('inin')
        tw_from_input = input.tw(3.5)

        out1 = Output('out1', tw_from_input.delay(1))
        with self.assertRaises(ValueError):
            Output('out3', tw_from_input.delay(-1))
        with self.assertRaises(TypeError):
            Output('out3', tw_from_input.z(1))

        test = Modely(visualizer=None)
        test.addModel('out_A', [out1])
        test.neuralizeModel(0.5)
        results = test({'inin': [14, 1, 2, 3, 4, 5, 6, 7, 8, 9]})
        self.assertEqual((4,7), np.array(results['out1']).shape)
        self.assertEqual([[0.0, 0.0, 14.0, 1.0, 2.0, 3.0, 4.0],
                                 [0.0, 14.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                                 [14.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                                 [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]],  results['out1'])
        #TODO add test with initialization of state variable

    def test_localmodel(self):
        NeuObj.clearNames()
        x = Input('x')
        F = Input('F')
        activationA = Fuzzify(2, [0, 1], functions='Triangular')(x.tw(1))
        activationB = Fuzzify(2, [0, 1], functions='Triangular')(F.tw(1))

        def myFun(in1, p1, p2):
            return p1 * in1 + p2

        p1_0 = Parameter('p1_0', values=[[1]])
        p1_1 = Parameter('p1_1', values=[[2]])
        p2_0 = Parameter('p2_0', values=[[2]])
        p2_1 = Parameter('p2_1', values=[[3]])

        def input_function_gen(idx_list):
            if idx_list == [0, 0]:
                p1, p2 = p1_0, p2_0
            if idx_list == [0, 1]:
                p1, p2 = p1_0, p2_1
            if idx_list == [1, 0]:
                p1, p2 = p1_1, p2_0
            if idx_list == [1, 1]:
                p1, p2 = p1_1, p2_1
            return ParamFun(myFun, parameters_and_constants=[p1, p2])

        def output_function_gen(idx_list):
            pfir = Parameter('pfir_' + str(idx_list), tw=1, dimensions=2,
                             values=[[1 + idx_list[0], 2 + idx_list[1]], [3 + idx_list[0], 4 + idx_list[1]]])
            return Fir(2, W=pfir)

        loc = LocalModel(input_function=input_function_gen, output_function=output_function_gen, pass_indexes=True)(x.tw(1), (activationA, activationB))
        # Example of the structure of the local model
        pfir00 = Parameter('N_pfir_[0, 0]', tw=1, dimensions=2, values=[[1, 2], [3, 4]])
        pfir01 = Parameter('N_pfir_[0, 1]', tw=1, dimensions=2, values=[[1, 3], [3, 5]])
        pfir10 = Parameter('N_pfir_[1, 0]', tw=1, dimensions=2, values=[[2, 2], [4, 4]])
        pfir11 = Parameter('N_pfir_[1, 1]', tw=1, dimensions=2, values=[[2, 3], [4, 5]])
        parfun_00 = ParamFun(myFun, parameters_and_constants=[p1_0, p2_0])(x.tw(1))
        parfun_01 = ParamFun(myFun, parameters_and_constants=[p1_0, p2_1])(x.tw(1))
        parfun_10 = ParamFun(myFun, parameters_and_constants=[p1_1, p2_0])(x.tw(1))
        parfun_11 = ParamFun(myFun, parameters_and_constants=[p1_1, p2_1])(x.tw(1))
        out_in_00 = Output('parfun00', parfun_00)
        out_in_01 = Output('parfun01', parfun_01)
        out_in_10 = Output('parfun10', parfun_10)
        out_in_11 = Output('parfun11', parfun_11)
        actA = Output('fuzzyA', activationA)
        actB = Output('fuzzyB', activationB)
        act_selA0 = Select(activationA, 0)
        act_selA1 = Select(activationA, 1)
        act_selB0 = Select(activationB, 0)
        act_selB1 = Select(activationB, 1)
        out_act_selA0 = Output('fuzzy_selA0', act_selA0)
        out_act_selA1 = Output('fuzzy_selA1', act_selA1)
        out_act_selB0 = Output('fuzzy_selB0', act_selB0)
        out_act_selB1 = Output('fuzzy_selB1', act_selB1)
        mul00 = parfun_00 * act_selA0 * act_selB0
        mul01 = parfun_01 * act_selA0 * act_selB1
        mul10 = parfun_10 * act_selA1 * act_selB0
        mul11 = parfun_11 * act_selA1 * act_selB1
        out_mul00 = Output('mul00', mul00)
        out_mul01 = Output('mul01', mul01)
        out_mul10 = Output('mul10', mul10)
        out_mul11 = Output('mul11', mul11)
        fir00 = Fir(2, W=pfir00)(mul00)
        fir01 = Fir(2, W=pfir01)(mul01)
        fir10 = Fir(2, W=pfir10)(mul10)
        fir11 = Fir(2, W=pfir11)(mul11)
        out_fir00 = Output('fir00', fir00)
        out_fir01 = Output('fir01', fir01)
        out_fir10 = Output('fir10', fir10)
        out_fir11 = Output('fir11', fir11)
        sum = fir00 + fir01 + fir10 + fir11
        out_sum = Output('out_sum', sum)
        out = Output('out', loc)
        test = Modely(visualizer=None)
        test.addModel('all_out', [out_in_00, out_in_01, out_in_10, out_in_11,
                                     out_act_selA0, out_act_selA1, out_act_selB0, out_act_selB1,
                                     out_mul00, out_mul01, out_mul10, out_mul11,
                                     out_fir00, out_fir01, out_fir10, out_fir11,
                                     out_sum])
        test.addModel('out', out)
        test.neuralizeModel(0.5)
        # Three semples with a dimensions 2
        result = test({'x': [0, 1, -2, 3], 'F': [-2, 2, 1, 5]})
        self.assertEqual(result['out_sum'],result['out'])

    def test_integrate_derivate(self):
        NeuObj.clearNames()
        input = Input('in1')

        in1_s = Output('in1_s', input.s(1))
        in1_s2 = Output('in1_s2', input.s(2))
        in1_s2_2 = Output('in1_s2_2', Differentiate(input.s(1)))
        in1_s2_3 = Output('in1_s2_3', input.s(1).s(1))
        in1_s_2 = Output('in1_s_2', input.s(2).s(-1))

        in1_sm = Output('in1_sm', input.s(-1))
        in1_sm2 = Output('in1_sm2', input.s(-2))
        in1_sm2_2 = Output('in1_sm2_2', Integrate(input.s(-1)))
        in1_sm2_3 = Output('in1_sm2_3', input.s(-1).s(-1))
        in1_sm_2 = Output('in1_sm_2', input.s(-2).s(1))

        in1_1 = Output('in1_1', Integrate(input.s(1)))
        in1_2 = Output('in1_2', Integrate(Integrate(input.s(2))))
        in1_3 = Output('in1_3', Integrate(Integrate(Differentiate(input.s(1)))))

        in1_1_2 = Output('in1_1_2', Differentiate(input.s(-1)))
        in1_2_2 = Output('in1_2_2', Differentiate(Differentiate(input.s(-2))))
        in1_3_2 = Output('in1_3_2', Differentiate(Differentiate(Integrate(input.s(-1)))))

        test = Modely(visualizer=None)
        test.addModel('out_A', [in1_s, in1_s2, in1_s2_2, in1_s2_3, in1_s_2, in1_sm, in1_sm2, in1_sm2_2, in1_sm2_3, in1_sm_2, in1_1, in1_2, in1_3, in1_1_2, in1_2_2, in1_3_2])
        test.neuralizeModel(1)
        inin = {'in1': [14, 1, 2, 3, 4, 5, 6, 7, 8, 9]}
        results = test(inin)
        self.assertEqual(results['in1_1'], inin['in1'])
        self.assertEqual(results['in1_2'], inin['in1'])
        self.assertEqual(results['in1_3'], inin['in1'])
        self.assertEqual(results['in1_1_2'], inin['in1'])
        self.assertEqual(results['in1_2_2'], inin['in1'])
        self.assertEqual(results['in1_3_2'], inin['in1'])

        inin_s = [14, -13, 1, 1, 1, 1, 1, 1, 1, 1]
        inin_s2 = [14, -27, 14, 0, 0, 0, 0, 0, 0, 0]
        self.assertEqual(results['in1_s'], inin_s)
        self.assertEqual(results['in1_s_2'], inin_s)
        self.assertEqual(results['in1_s2'], inin_s2)
        self.assertEqual(results['in1_s2_2'], inin_s2)
        self.assertEqual(results['in1_s2_3'], inin_s2)

        inin_sm = [14, 15, 17, 20, 24, 29, 35, 42, 50, 59]
        inin_sm2 = [14, 29, 46, 66, 90, 119, 154, 196, 246, 305]
        self.assertEqual(results['in1_sm'], inin_sm)
        self.assertEqual(results['in1_sm_2'], inin_sm)
        self.assertEqual(results['in1_sm2'], inin_sm2)
        self.assertEqual(results['in1_sm2_2'], inin_sm2)
        self.assertEqual(results['in1_sm2_3'], inin_sm2)

        test = Modely(visualizer=None)
        test.addModel('out_A',
                      [in1_s, in1_s2, in1_s2_2, in1_s2_3, in1_s_2, in1_sm, in1_sm2, in1_sm2_2, in1_sm2_3, in1_sm_2, in1_1, in1_2, in1_3, in1_1_2, in1_2_2, in1_3_2])
        test.neuralizeModel(0.01)
        inin = {'in1': [14, 1, 2, 3, 4, 5, 6, 7, 8, 9]}
        results = test(inin)
        self.TestAlmostEqual(results['in1_1'], inin['in1'])
        self.TestAlmostEqual(results['in1_2'], inin['in1'])
        self.TestAlmostEqual(results['in1_3'], inin['in1'])
        self.TestAlmostEqual(results['in1_1_2'], inin['in1'])
        self.TestAlmostEqual(results['in1_2_2'], inin['in1'])
        self.TestAlmostEqual(results['in1_3_2'], inin['in1'])

        inin_s = [1400, -1300, 100, 100, 100, 100, 100, 100, 100, 100]
        inin_s2 = [140000, -270000, 140000, 0, 0, 0, 0, 0, 0, 0]
        self.TestAlmostEqual(results['in1_s'], inin_s)
        self.TestAlmostEqual(results['in1_s_2'], inin_s)
        self.TestAlmostEqual(results['in1_s2'], inin_s2)
        self.TestAlmostEqual(results['in1_s2_2'], inin_s2)
        self.TestAlmostEqual(results['in1_s2_3'], inin_s2)

        inin_sm = [0.14, 0.15, 0.17, 0.20, 0.24, 0.29, 0.35, 0.42, 0.50, 0.59]
        inin_sm2 = [0.0014, 0.0029, 0.0046, 0.0066, 0.0090, 0.0119, 0.0154, 0.0196, 0.0246, 0.0305]
        self.TestAlmostEqual(results['in1_sm'], inin_sm)
        self.TestAlmostEqual(results['in1_sm_2'], inin_sm)
        self.TestAlmostEqual(results['in1_sm2'], inin_sm2)
        self.TestAlmostEqual(results['in1_sm2_2'], inin_sm2)
        self.TestAlmostEqual(results['in1_sm2_3'], inin_sm2)

    def test_integrate_derivate_trapezoidal(self):
        NeuObj.clearNames()
        input = Input('in1')

        in1_s = Output('in1_s', input.s(1,method='trapezoidal'))
        in1_s2 = Output('in1_s2', input.s(2,method='trapezoidal'))
        in1_s2_2 = Output('in1_s2_2', Differentiate(input.s(1,method='trapezoidal'),method='trapezoidal'))
        in1_s2_3 = Output('in1_s2_3', input.s(1,method='trapezoidal').s(1,method='trapezoidal'))
        in1_s_2 = Output('in1_s_2', input.s(2,method='trapezoidal').s(-1,method='trapezoidal'))

        in1_sm = Output('in1_sm', input.s(-1,method='trapezoidal'))
        in1_sm2 = Output('in1_sm2', input.s(-2,method='trapezoidal'))
        in1_sm2_2 = Output('in1_sm2_2', Integrate(input.s(-1,method='trapezoidal'),method='trapezoidal'))
        in1_sm2_3 = Output('in1_sm2_3', input.s(-1,method='trapezoidal').s(-1,method='trapezoidal'))
        in1_sm_2 = Output('in1_sm_2', input.s(-2,method='trapezoidal').s(1,method='trapezoidal'))

        in1_1 = Output('in1_1', Integrate(input.s(1,method='trapezoidal'),method='trapezoidal'))
        in1_2 = Output('in1_2', Integrate(Integrate(input.s(2,method='trapezoidal'),method='trapezoidal'),method='trapezoidal'))
        in1_3 = Output('in1_3', Integrate(Integrate(Differentiate(input.s(1,method='trapezoidal'),method='trapezoidal'),method='trapezoidal'),method='trapezoidal'))

        in1_1_2 = Output('in1_1_2', Differentiate(input.s(-1,method='trapezoidal'),method='trapezoidal'))
        in1_2_2 = Output('in1_2_2', Differentiate(Differentiate(input.s(-2,method='trapezoidal'),method='trapezoidal'),method='trapezoidal'))
        in1_3_2 = Output('in1_3_2', Differentiate(Differentiate(Integrate(input.s(-1,method='trapezoidal'),method='trapezoidal'),method='trapezoidal'),method='trapezoidal'))

        test = Modely(visualizer=None)
        test.addModel('out_A', [in1_s, in1_s2, in1_s2_2, in1_s2_3, in1_s_2, in1_sm, in1_sm2, in1_sm2_2, in1_sm2_3, in1_sm_2, in1_1, in1_2, in1_3, in1_1_2, in1_2_2, in1_3_2])
        test.neuralizeModel(1)
        inin = {'in1': [14, 1, 2, 3, 4, 5, 6, 7, 8, 9]}
        results = test(inin)
        self.assertEqual(results['in1_1'], inin['in1'])
        self.assertEqual(results['in1_2'], inin['in1'])
        self.assertEqual(results['in1_3'], inin['in1'])
        self.assertEqual(results['in1_1_2'], inin['in1'])
        self.assertEqual(results['in1_2_2'], inin['in1'])
        self.assertEqual(results['in1_3_2'], inin['in1'])

        inin_sm = [7., 14.5, 16., 18.5, 22., 26.5, 32., 38.5, 46., 54.5]
        inin_sm2 = [  3.5 ,  14.25,  29.5 ,  46.75,  67.  ,  91.25, 120.5 , 155.75, 198.  , 248.25]
        self.assertEqual(results['in1_sm'], inin_sm)
        self.assertEqual(results['in1_sm_2'], inin_sm)
        self.assertEqual(results['in1_sm2'], inin_sm2)
        self.assertEqual(results['in1_sm2_2'], inin_sm2)
        self.assertEqual(results['in1_sm2_3'], inin_sm2)

        test = Modely(visualizer=None)
        test.addModel('out_A',
                      [in1_s, in1_s2, in1_s2_2, in1_s2_3, in1_s_2, in1_sm, in1_sm2, in1_sm2_2, in1_sm2_3, in1_sm_2, in1_1, in1_2, in1_3, in1_1_2, in1_2_2, in1_3_2])
        test.neuralizeModel(0.01)
        inin = {'in1': [14, 1, 2, 3, 4, 5, 6, 7, 8, 9]}
        results = test(inin)
        self.TestAlmostEqual(results['in1_1'], inin['in1'], precision=4)
        self.TestAlmostEqual(results['in1_2'], inin['in1'], precision=4)
        self.TestAlmostEqual(results['in1_3'], inin['in1'], precision=4)
        self.TestAlmostEqual(results['in1_1_2'], inin['in1'], precision=4)
        self.TestAlmostEqual(results['in1_2_2'], inin['in1'], precision=3)
        self.TestAlmostEqual(results['in1_3_2'], inin['in1'], precision=3)

        inin_sm = [ 0.07 , 0.145, 0.16 , 0.185, 0.22 , 0.265, 0.32 , 0.385, 0.46 , 0.545]
        inin_sm2 = [0.00035 , 0.001425, 0.00295 , 0.004675, 0.0067  , 0.009125, 0.01205 , 0.015575, 0.0198  , 0.024825]
        self.TestAlmostEqual(results['in1_sm'], inin_sm)
        self.TestAlmostEqual(results['in1_sm_2'], inin_sm)
        self.TestAlmostEqual(results['in1_sm2'], inin_sm2)
        self.TestAlmostEqual(results['in1_sm2_2'], inin_sm2)
        self.TestAlmostEqual(results['in1_sm2_3'], inin_sm2)

    def test_derivate_wrt_input(self):
        NeuObj.clearNames()
        x = Input('x')

        x_last = x.last()

        def parametric_fun(x, a, b, c, d):
            import torch
            return x ** 3 * a + x ** 2 * b + torch.sin(x) * c + d

        def dx_parametric_fun(x, a, b, c, d):
            import torch
            return (3 * x ** 2 * a) + (2 * x * b) + c * torch.cos(x)

        fun = ParamFun(parametric_fun,['a','b','c','d'])(x_last)
        approx_y = Output('out', fun)
        approx_dy_dx = Output('d_out', Differentiate(fun, x_last))

        test = Modely(visualizer=None, seed=12)

        test.addModel('model', [approx_dy_dx, approx_y])
        test.neuralizeModel()
        results = test({'x':[1,2]})
        self.assertAlmostEqual(results['out'][0], parametric_fun(torch.tensor(1), torch.tensor(test.parameters['a']),
                                                            torch.tensor(test.parameters['b']),
                                                            torch.tensor(test.parameters['c']),
                                                            torch.tensor(test.parameters['d'])).detach().numpy().tolist()[0],places=5)
        self.assertAlmostEqual(results['d_out'][0], dx_parametric_fun(torch.tensor(1), torch.tensor(test.parameters['a']),
                                                            torch.tensor(test.parameters['b']),
                                                            torch.tensor(test.parameters['c']),
                                                            torch.tensor(test.parameters['d'])).detach().numpy().tolist()[0],places=5)
        self.assertAlmostEqual(results['out'][1], parametric_fun(torch.tensor(2), torch.tensor(test.parameters['a']),
                                                            torch.tensor(test.parameters['b']),
                                                            torch.tensor(test.parameters['c']),
                                                            torch.tensor(test.parameters['d'])).detach().numpy().tolist()[0],places=5)
        self.assertAlmostEqual(results['d_out'][1], dx_parametric_fun(torch.tensor(2), torch.tensor(test.parameters['a']),
                                                            torch.tensor(test.parameters['b']),
                                                            torch.tensor(test.parameters['c']),
                                                            torch.tensor(test.parameters['d'])).detach().numpy().tolist()[0],places=5)

