import sys, os, unittest, copy

import numpy as np

from nnodely import *
from nnodely.basic.relation import NeuObj, Stream
from nnodely.support.logger import logging, nnLogger
from nnodely.support.jsonutils import subjson_from_model

log = nnLogger(__name__, logging.CRITICAL)
log.setAllLevel(logging.CRITICAL)

sys.path.append(os.getcwd())

# 9 Tests
# This test file tests the json, in particular
# the dimensions that are propagated through the relations
# and the structure of the json itself

def myFun(K1,K2,p1,p2):
    import torch
    return p1*K1+p2*torch.sin(K2)

def myFun_out5(K1,p1):
    import torch
    return torch.stack([K1,K1,K1,K1,K1],dim=2).squeeze(-1)*p1

def myFunPar(x, p1):
    import torch
    if len(p1.shape) == 0:
        out = torch.tensor([[[1]]]).repeat((x.shape[0], 1, 1))
    else:
        out = torch.tensor([[p1.shape]]).repeat((x.shape[0],1,1))
    return out

NeuObj.count = 0

class ModelyJsonTest(unittest.TestCase):
    def TestAlmostEqual(self, data1, data2, precision=4):
        assert np.asarray(data1, dtype=np.float32).ndim == np.asarray(data2, dtype=np.float32).ndim, f'Inputs must have the same dimension! Received {type(data1)} and {type(data2)}'
        if type(data1) == type(data2) == list:
            self.assertEqual(len(data1),len(data2))
            for pred, label in zip(data1, data2):
                self.TestAlmostEqual(pred, label, precision=precision)
        else:
            self.assertAlmostEqual(data1, data2, places=precision)

    def test_input(self):
        input = Input('in1')
        self.assertEqual({'Info':{},'Constants': {},'Inputs': {'in1': {'dim': 1}}, 'Functions' : {}, 'Parameters' : {}, 'Outputs': {}, 'Relations': {}},input.json)

        #Discrete input removed
        #input = Input('in', values=[2,3,4])
        #self.assertEqual({'Inputs': {'in': {'dim': 1, 'discrete': [2,3,4], 'tw': [0,0], 'sw': [0, 0]}},'Functions' : {}, 'Parameters' : {}, 'Outputs': {}, 'Relations': {}},input.json)

    def test_aritmetic(self):
        Stream.resetCount()
        NeuObj.clearNames()
        input = Input('in1')
        inlast = input.last()
        out = inlast+inlast
        self.assertEqual({'Info':{},'Constants': {},'Inputs': {'in1': {'dim': 1, 'sw': [-1, 0]}},'Functions' : {}, 'Parameters' : {}, 'Outputs': {}, 'Relations': {'Add2': ['Add', ['SamplePart1', 'SamplePart1']],
               'SamplePart1': ['SamplePart', ['in1'], -1, [-1, 0]]}},out.json)
        out = input.tw(1) + input.tw(1)
        self.assertEqual({'Info':{},'Constants': {},'Inputs': {'in1': {'dim': 1, 'tw': [-1,0]}},'Functions' : {}, 'Parameters' : {}, 'Outputs': {}, 'Relations': {'Add7': ['Add', ['TimePart4', 'TimePart6']],
               'TimePart4': ['TimePart', ['in1'], -1, [-1, 0]],
               'TimePart6': ['TimePart', ['in1'], -1, [-1, 0]]}},out.json)
        out = input.tw(1) * input.tw(1)
        self.assertEqual({'Info':{},'Constants': {},'Inputs': {'in1': {'dim': 1, 'tw': [-1,0]}},'Functions' : {}, 'Parameters' : {}, 'Outputs': {},  'Relations': {'Mul12': ['Mul', ['TimePart9', 'TimePart11']],
               'TimePart9': ['TimePart', ['in1'], -1, [-1, 0]],
               'TimePart11': ['TimePart', ['in1'], -1, [-1, 0]]}},out.json)
        out = input.tw(1) - input.tw(1)
        self.assertEqual({'Info':{},'Constants': {},'Inputs': {'in1': {'dim': 1, 'tw': [-1,0]}},'Functions' : {}, 'Parameters' : {}, 'Outputs': {},  'Relations': {'Sub17': ['Sub', ['TimePart14', 'TimePart16']],
               'TimePart14': ['TimePart', ['in1'], -1, [-1, 0]],
               'TimePart16': ['TimePart', ['in1'], -1, [-1, 0]]}},out.json)
        input = Input('in2', dimensions = 5)
        inlast = input.last()
        out = inlast + inlast
        self.assertEqual({'Info':{},'Constants': {},'Inputs': {'in2': {'dim': 5, 'sw': [-1, 0]}},'Functions' : {}, 'Parameters' : {}, 'Outputs': {},  'Relations': {'Add20': ['Add', ['SamplePart19', 'SamplePart19']],
               'SamplePart19': ['SamplePart', ['in2'], -1, [-1, 0]]}},out.json)
        out = input.tw(1) + input.tw(1)
        self.assertEqual({'Info':{},'Constants': {},'Inputs': {'in2': {'dim': 5, 'tw': [-1, 0]}}, 'Functions': {}, 'Parameters': {},'Outputs': {},  'Relations': {'Add25': ['Add', ['TimePart22', 'TimePart24']],
               'TimePart22': ['TimePart', ['in2'], -1, [-1, 0]],
               'TimePart24': ['TimePart', ['in2'], -1, [-1, 0]]}}, out.json)
        out = input.tw([2,5]) + input.tw([3,6])
        self.assertEqual({'Info':{},'Constants': {},'Inputs': {'in2': {'dim': 5, 'tw': [2, 6]}}, 'Functions': {}, 'Parameters': {},'Outputs': {},  'Relations': {'Add30': ['Add', ['TimePart27', 'TimePart29']],
               'TimePart27': ['TimePart', ['in2'], -1, [2, 5]],
               'TimePart29': ['TimePart', ['in2'], -1, [3, 6]]}}, out.json)
        out = input.tw([-5,-2]) + input.tw([-6,-3])
        self.assertEqual({'Info':{},'Constants': {},'Inputs': {'in2': {'dim': 5, 'tw': [-6, -2]}}, 'Functions': {}, 'Parameters': {},'Outputs': {},  'Relations': {'Add35': ['Add', ['TimePart32', 'TimePart34']],
               'TimePart32': ['TimePart', ['in2'], -1, [-5, -2]],
               'TimePart34': ['TimePart', ['in2'], -1, [-6, -3]]}}, out.json)

    def test_scalar_input_dimensions(self):
        NeuObj.clearNames()
        input = Input('in1').last()
        out = input+input
        self.assertEqual({'dim': 1,'sw': 1}, out.dim)
        out = Fir(input)
        self.assertEqual({'dim': 1,'sw': 1}, out.dim)
        out = Fir(7)(input)
        self.assertEqual({'dim': 7,'sw': 1}, out.dim)
        out = Fuzzify(5, [-1,1])(input)
        self.assertEqual({'dim': 5,'sw': 1}, out.dim)
        out = ParamFun(myFun)(input)
        self.assertEqual({'dim': 1,'sw': 1}, out.dim)
        out = ParamFun(myFun_out5)(input)
        self.assertEqual({'dim': 5, 'sw': 1}, out.dim)
        with self.assertRaises(ValueError):
            out = Fir(Fir(7)(input))
        #
        with self.assertRaises(IndexError):
            out = Part(input,0,4)
        inpart = ParamFun(myFun_out5)(input)
        out = Part(inpart,0,4)
        self.assertEqual({'dim': 4, 'sw': 1}, out.dim)
        out = Part(inpart,0,1)
        self.assertEqual({'dim': 1, 'sw': 1}, out.dim)
        out = Part(inpart,1,3)
        self.assertEqual({'dim': 2, 'sw': 1}, out.dim)
        out = Select(inpart,0)
        self.assertEqual({'dim': 1, 'sw': 1}, out.dim)
        with self.assertRaises(IndexError):
            out = Select(inpart,5)
        with self.assertRaises(IndexError):
            out = Select(inpart,-1)
        with self.assertRaises(KeyError):
            out = TimePart(inpart,-1,0)

    def test_scalar_input_tw_dimensions(self):
        NeuObj.clearNames()
        input = Input('in1')
        out = input.tw(1) + input.tw(1)
        self.assertEqual({'dim': 1, 'tw': 1}, out.dim)
        out = Fir(input.tw(1))
        self.assertEqual({'dim': 1, 'sw': 1}, out.dim)
        out = Fir(5)(input.tw(1))
        self.assertEqual({'dim': 5, 'sw': 1}, out.dim)
        out = Fuzzify(5, [0,5])(input.tw(2))
        self.assertEqual({'dim': 5, 'tw': 2}, out.dim)
        out = Fuzzify(5,range=[-1,5])(input.tw(2))
        self.assertEqual({'dim': 5, 'tw': 2}, out.dim)
        out = Fuzzify(centers=[-1,5])(input.tw(2))
        self.assertEqual({'dim': 2, 'tw': 2}, out.dim)
        out = ParamFun(myFun)(input.tw(1))
        self.assertEqual({'dim': 1, 'tw' : 1}, out.dim)
        out = ParamFun(myFun_out5)(input.tw(2))
        self.assertEqual({'dim': 5, 'tw': 2}, out.dim)
        out = ParamFun(myFun_out5)(input.tw(2),input.tw(1))
        self.assertEqual({'dim': 5, 'tw': 2}, out.dim)
        inpart = ParamFun(myFun_out5)(input.tw(2))
        out = Part(inpart,0,4)
        self.assertEqual({'dim': 4,'tw': 2}, out.dim)
        out = Part(inpart,0,1)
        self.assertEqual({'dim': 1,'tw': 2}, out.dim)
        out = Part(inpart,1,3)
        self.assertEqual({'dim': 2,'tw': 2}, out.dim)
        out = Select(inpart,0)
        self.assertEqual({'dim': 1,'tw': 2}, out.dim)
        with self.assertRaises(IndexError):
            out = Select(inpart,5)
        with self.assertRaises(IndexError):
            out = Select(inpart,-1)
        out = TimePart(inpart, 0,1)
        self.assertEqual({'dim': 5, 'tw': 1}, out.dim)
        #out = TimeSelect(inpart,0)
        #self.assertEqual({'dim': 5}, out.dim)
        #with self.assertRaises(ValueError):
        #   out = TimeSelect(inpart,-3)
        twinput = input.tw([-2,4])
        out = TimePart(twinput, 0, 1)
        self.assertEqual({'dim': 1, 'tw': 1}, out.dim)

    def test_scalar_input_tw2_dimensions(self):
        NeuObj.clearNames()
        input = Input('in1')
        out = input.tw([-1,1])+input.tw([-2,0])
        self.assertEqual({'dim': 1, 'tw': 2}, out.dim)
        out = input.tw(1)+input.tw([-1,0])
        self.assertEqual({'dim': 1, 'tw': 1}, out.dim)
        out = Fir(input.tw(1) + input.tw([-1, 0]))
        self.assertEqual({'dim': 1, 'sw': 1}, out.dim)
        out = input.tw([-1,0])+input.tw([-4,-3])+input.tw(1)
        self.assertEqual({'dim': 1,'tw': 1}, out.dim)
        with self.assertRaises(ValueError):
             out = input.tw([-2,0])-input.tw([-1,0])
        with self.assertRaises(ValueError):
             out = input.tw([-2,0])+input.tw([-1,0])

    def test_scalar_input_sw_dimensions(self):
        NeuObj.clearNames()
        input = Input('in1')
        out = input.sw([-1,1])+input.sw([-2,0])
        self.assertEqual({'dim': 1, 'sw': 2}, out.dim)
        out = input.sw(1)+input.sw([-1,0])
        self.assertEqual({'dim': 1, 'sw': 1}, out.dim)
        out = Fir(input.sw(1) + input.sw([-1, 0]))
        self.assertEqual({'dim': 1, 'sw': 1}, out.dim)
        out = input.sw([-1,0])+input.sw([-4,-3])+input.sw(1)
        self.assertEqual({'dim': 1,'sw': 1}, out.dim)
        with self.assertRaises(ValueError):
            out = input.sw([-2,0])-input.sw([-1,0])
        with self.assertRaises(ValueError):
            out = input.sw([-2,0])+input.sw([-1,0])
        with self.assertRaises(TypeError):
            out = input.sw(1) + input.tw([-1, 0])
        with self.assertRaises(TypeError):
            out = input.sw(1.2)
        with self.assertRaises(TypeError):
            out = input.sw([-1.2,0.05])

    def test_vector_input_dimensions(self):
        NeuObj.clearNames()
        input = Input('in1', dimensions = 5)
        self.assertEqual({'dim': 5}, input.dim)
        self.assertEqual({'dim': 5, 'tw' : 2}, input.tw(2).dim)
        out = input.tw(1) + input.tw(1)
        self.assertEqual({'dim': 5, 'tw': 1}, out.dim)
        out = Relu(input.tw(1))
        self.assertEqual({'dim': 5, 'tw': 1}, out.dim)
        with self.assertRaises(TypeError):
            Fir(7)(input)
        with self.assertRaises(TypeError):
            Fuzzify(7,[1,7])(input)
        out = ParamFun(myFun)(input.tw(1))
        self.assertEqual({'dim': 5, 'tw' : 1}, out.dim)
        out = ParamFun(myFun)(input.tw(2))
        self.assertEqual({'dim': 5, 'tw': 2}, out.dim)
        out = ParamFun(myFun)(input.tw(2),input.tw(1))
        self.assertEqual({'dim': 5, 'tw': 2}, out.dim)

    def test_parameter_and_linear(self):
        NeuObj.clearNames()
        input = Input('in1').last()
        W15 = Parameter('W15', dimensions=(1, 5))
        b15 = Parameter('b15', dimensions=5)
        input4 = Input('in4', dimensions=4).last()
        W45 = Parameter('W45', dimensions=(4, 5))
        b45 = Parameter('b45', dimensions=5)

        out = Linear(input) + Linear(input4)
        out3 = Linear(3)(input) + Linear(3)(input4)
        outW = Linear(W = W15)(input) + Linear(W = W45)(input4)
        outWb = Linear(W = W15,b = b15)(input) + Linear(W = W45, b = b45)(input4)
        self.assertEqual({'dim': 1, 'sw': 1}, out.dim)
        self.assertEqual({'dim': 3, 'sw': 1}, out3.dim)
        self.assertEqual({'dim': 5, 'sw': 1}, outW.dim)
        self.assertEqual({'dim': 5, 'sw': 1}, outWb.dim)

        NeuObj.clearNames()
        input2 = Input('in1').sw([-1,1])
        W15 = Parameter('W15', dimensions=(1, 5))
        b15 = Parameter('b15', dimensions=5)
        input42 = Input('in4', dimensions=4).sw([-1,1])
        W45 = Parameter('W45', dimensions=(4, 5))
        b45 = Parameter('b45', dimensions=5)

        out = Linear(input2) + Linear(input42)
        out3 = Linear(3)(input2) + Linear(3)(input42)
        outW = Linear(W = W15)(input2) + Linear(W = W45)(input42)
        outWb = Linear(W = W15,b = b15)(input2) + Linear(W = W45, b = b45)(input42)
        self.assertEqual({'dim': 1, 'sw': 2}, out.dim)
        self.assertEqual({'dim': 3, 'sw': 2}, out3.dim)
        self.assertEqual({'dim': 5, 'sw': 2}, outW.dim)
        self.assertEqual({'dim': 5, 'sw': 2}, outWb.dim)

        with self.assertRaises(ValueError):
            Linear(input) + Linear(input42)
        with self.assertRaises(ValueError):
            Linear(3)(input2) + Linear(3)(input4)
        with self.assertRaises(ValueError):
            Linear(W = W15)(input) + Linear(W = W45)(input42)
        with self.assertRaises(ValueError):
            Linear(W = W15,b = b15)(input2) + Linear(W = W45, b = b45)(input4)

    def test_input_paramfun_param_const(self):
        NeuObj.clearNames()
        input2 = Input('in2')
        def fun_test(x,y,z,k):
            return x*y*z*k

        NeuObj.clearNames()
        out = ParamFun(fun_test)(input2.tw(0.01))
        self.assertEqual({'dim': 1, 'tw': 0.01}, out.dim)
        self.assertEqual({'FParamFun0k': {'dim': 1},'FParamFun0y': {'dim': 1},'FParamFun0z': {'dim': 1}}, out.json['Parameters'])

        NeuObj.clearNames()
        out = ParamFun(fun_test)(input2.tw(0.01),input2.tw(0.01))
        self.assertEqual({'dim': 1, 'tw': 0.01}, out.dim)
        self.assertEqual({'FParamFun0k': {'dim': 1}, 'FParamFun0z': {'dim': 1}}, out.json['Parameters'])

        NeuObj.clearNames()
        out = ParamFun(fun_test,parameters_and_constants=['t'])(input2.tw(0.01),input2.tw(0.01))
        self.assertEqual({'dim': 1, 'tw': 0.01}, out.dim)
        self.assertEqual({'FParamFun0z': {'dim': 1}, 't': {'dim': 1}}, out.json['Parameters'])

        NeuObj.clearNames()
        out = ParamFun(fun_test,parameters_and_constants=['t','r'])(input2.tw(0.01),input2.tw(0.01))
        self.assertEqual({'dim': 1, 'tw': 0.01}, out.dim)
        self.assertEqual({'r': {'dim': 1}, 't': {'dim': 1}}, out.json['Parameters'])

        NeuObj.clearNames()
        out = ParamFun(fun_test,parameters_and_constants={'k':'t'})(input2.tw(0.01),input2.tw(0.01))
        self.assertEqual({'dim': 1, 'tw': 0.01}, out.dim)
        self.assertEqual({'FParamFun0z': {'dim': 1}, 't': {'dim': 1}}, out.json['Parameters'])

        NeuObj.clearNames()
        out = ParamFun(fun_test,parameters_and_constants={'k':(1,2)})(input2.tw(0.01),input2.tw(0.01))
        self.assertEqual({'dim': 2, 'tw': 0.01}, out.dim)
        self.assertEqual({'FParamFun0k': {'dim': [1,2]}, 'FParamFun0z': {'dim': 1}}, out.json['Parameters'])

        with self.assertRaises(ValueError):
            ParamFun(fun_test,parameters_and_constants={'k':(1,2),'y':'r'})(input2.tw(0.01),input2.tw(0.01))
        with self.assertRaises(ValueError):
           ParamFun(fun_test,parameters_and_constants=[1.0,(1,2),'gg'])(input2.tw(0.01),input2.tw(0.01))
        with self.assertRaises(ValueError):
            ParamFun(fun_test,parameters_and_constants=[(1,2),'pp','c',[[1.0]]])(input2.tw(0.01))

        NeuObj.clearNames()
        out = ParamFun(fun_test,parameters_and_constants={'k':(1,2)})(input2.tw(0.01),input2.tw(0.01),input2.tw(0.01))
        self.assertEqual({'dim': 2, 'tw': 0.01}, out.dim)
        self.assertEqual({'FParamFun0k': {'dim': [1,2]}}, out.json['Parameters'])

        with self.assertRaises(ValueError):
            ParamFun(fun_test,parameters_and_constants={'z':(1,2)})(input2.tw(0.01),input2.tw(0.01),input2.tw(0.01))

        with self.assertRaises(ValueError):
            ParamFun(fun_test,parameters_and_constants={'z':'g'})(input2.tw(0.01),input2.tw(0.01),input2.tw(0.01))

        with self.assertRaises(ValueError):
            ParamFun(fun_test,parameters_and_constants={'z':'o'})(input2.tw(0.01),input2.tw(0.01),input2.tw(0.01))

        NeuObj.clearNames()
        out = ParamFun(fun_test,parameters_and_constants=['pp','tt'])(input2.tw(0.01))
        self.assertEqual({'dim': 1, 'tw': 0.01}, out.dim)
        self.assertEqual({'FParamFun0y': {'dim': 1}, 'pp': {'dim': 1},'tt': {'dim': 1}}, out.json['Parameters'])

        NeuObj.clearNames()
        out = ParamFun(fun_test,parameters_and_constants={'y':'pp','k':'el'})(input2.tw(0.01))
        self.assertEqual({'dim': 1, 'tw': 0.01}, out.dim)
        self.assertEqual({'FParamFun0z': {'dim': 1}, 'pp': {'dim': 1},'el': {'dim': 1}}, out.json['Parameters'])

        NeuObj.clearNames()
        out = ParamFun(fun_test,parameters_and_constants=['pp','oo',Constant('el',values=2.0)])(input2.tw(0.01))
        self.assertEqual({'dim': 1, 'tw': 0.01}, out.dim)
        self.assertEqual({'oo': {'dim': 1}, 'pp': {'dim': 1}}, out.json['Parameters'])
        self.assertEqual({'el': {'dim': 1,'values':[2.0]}}, out.json['Constants'])

        with self.assertRaises(NameError):
            ParamFun(fun_test,parameters_and_constants=['pp','oo','el'])(input2.tw(0.01))

        with self.assertRaises(NameError):
            ParamFun(fun_test,parameters_and_constants=['pp','oo','el'])(input2.tw(0.01))

        NeuObj.clearNames()
        out = ParamFun(fun_test,parameters_and_constants=['pp',Constant('oo',values=[[2.0]]),Constant('ll',sw=1,values=[[7.0]])])(input2.tw(0.01))
        self.assertEqual({'dim': 1, 'tw': 0.01}, out.dim)
        self.assertEqual({'pp': {'dim': 1}}, out.json['Parameters'])
        self.assertEqual({'oo': {'dim': [1,1],'values':[[2.0]]}, 'll': {'dim': 1,'sw': 1,'values':[[7.0]]}}, out.json['Constants'])

        NeuObj.clearNames()
        pp = Parameter('pp')
        ll = Constant('ll', values=[[1,2,3],[1,2,3]])
        oo = Constant('oo', values=[1,2,3])
        out = ParamFun(fun_test,parameters_and_constants=[pp,ll,oo])(input2.tw(0.01))
        self.assertEqual({'dim': 3, 'sw': 2}, out.dim)
        self.assertEqual({'pp': {'dim': 1}}, out.json['Parameters'])
        self.assertEqual({'oo': {'dim': 3, 'values': [1,2,3]}, 'll': {'dim': [2,3], 'values':[[1,2,3],[1,2,3]]}}, out.json['Constants'])

        out = ParamFun(fun_test,parameters_and_constants={'z':pp,'y':ll,'k':oo})(input2.tw(0.01))
        self.assertEqual({'dim': 3, 'sw': 2}, out.dim)
        self.assertEqual({'pp': {'dim': 1}}, out.json['Parameters'])
        self.assertEqual(['ll', 'pp', 'oo'],out.json['Functions']['FParamFun4']['params_and_consts'])
        self.assertEqual({'oo': {'dim': 3, 'values': [1,2,3]}, 'll': {'dim': [2,3], 'values':[[1,2,3],[1,2,3]]}}, out.json['Constants'])

        NeuObj.clearNames()
        Stream.resetCount()
        pp = Parameter('pp')
        ll = Constant('ll', values=[1,2,3])
        oo = Constant('oo', tw=0.01, values=[[1]])
        out = ParamFun(fun_test)(input2.tw(0.01),ll,oo,pp)
        self.assertEqual({'dim': 3, 'tw': 0.01}, out.dim)
        self.assertEqual({'pp': {'dim': 1}}, out.json['Parameters'])
        self.assertEqual({'oo': {'dim': 1, 'tw':0.01, 'values': [[1]]}, 'll': {'dim': 3,  'values': [1,2,3]}}, out.json['Constants'])
        self.assertEqual(['TimePart1', 'll', 'oo', 'pp'], out.json['Relations']['ParamFun2'][1])

    def test_check_multiple_streams_compatibility_paramfun(self):
        NeuObj.clearNames()
        log.setAllLevel(logging.WARNING)
        x = Input('x')
        F = Input('F')

        def myFun(p1, p2, k1, k2):
            import torch
            return k1 * torch.sin(p1) + k2 * torch.cos(p2)

        K1 = Parameter('k1', dimensions=1, sw=1, values=[[2.0]])
        K2 = Parameter('k2', dimensions=1, sw=1, values=[[3.0]])
        parfun = ParamFun(myFun, parameters_and_constants=[K1,K2])

        rel1 = parfun(x.last(), F.last())
        rel2 = parfun(Tanh(F.sw(2)+F.sw([-2,-0])+F.sw([-3,-1])+F.sw([-4,-2])), Tanh(F.sw([0,2])))
        rel3 = parfun(Tanh(F.sw([-2,1])))
        rel4 = parfun(Tanh(F.sw([-2,1])), K1)
        rel5 = parfun(K1, Tanh(F.sw(1)))
        with self.assertRaises(TypeError):
            parfun(Fir(3)(parfun(x.tw(0.4), x.tw(0.4))))

        out1 = Output('out1', rel1)
        out2 = Output('out2', rel2)
        out3 = Output('out3', rel3)
        out4 = Output('out4', rel4)
        out5 = Output('out5', rel5)

        # m = MPLVisualizer(5)
        # m.showFunctions(list(example.json['Functions'].keys()), xlim=[[-5, 5], [-1, 1]])
        exampleA = Modely(visualizer=None, seed=2)
        with self.assertRaises(TypeError):
            exampleA.addModel('model', [out1, out2, out3])
        exampleA.addModel('model_A', [out1, out2])
        with self.assertRaises(TypeError):
            exampleA.addModel('model_B', [out3])
        exampleA.addModel('model_A2', [out1, out2, out4, out5])
        exampleA.neuralizeModel(0.25)

        exampleB = Modely(visualizer=None, seed=2)
        exampleB.addModel('model_B', [out3])
        exampleB.neuralizeModel(1)

        resultsA = exampleA({'x': [1, 3, 3]})
        self.TestAlmostEqual([4.682941913604736, 3.2822399139404297, 3.2822399139404297], resultsA['out1'])
        self.TestAlmostEqual([[3.0, 3.0],[3.0 , 3.0],[3.0 , 3.0]], resultsA['out2'])
        self.TestAlmostEqual([[-1.2484405040740967, -1.2484405040740967, -1.2484405040740967],
                             [-1.2484405040740967, -1.2484405040740967, -1.2484405040740967],
                             [-1.2484405040740967, -1.2484405040740967, -1.2484405040740967]], resultsA['out4'])
        self.TestAlmostEqual([4.818594932556152, 4.818594932556152, 4.818594932556152], resultsA['out5'])

        resultsB = exampleB({'F': [1, 3, 4]})
        self.TestAlmostEqual([[3.831000328063965, 4.128425598144531, 4.133065223693848]], resultsB['out3'])

        log.setAllLevel(logging.CRITICAL)

    def test_check_multiple_streams_compatibility_linear(self):
        NeuObj.clearNames()
        log.setAllLevel(logging.WARNING)
        x = Input('x',dimensions=3)
        f = Input('f')

        lin = Linear()

        l1out = lin(x.last()) + Fir(lin(x.tw(2.0))) + Fir(lin(x.sw(3)))
        l2out = lin(f.last()) + Fir(lin(f.tw(2.0))) + Fir(lin(f.sw(3)))

        out1 = Output('out1', l1out)
        out2 = Output('out2', l2out)

        exampleA = Modely(visualizer=None, seed=2)
        with self.assertRaises(TypeError):
            exampleA.addModel('model', [out1, out2])
        exampleA.addModel('model_A', [out1])
        with self.assertRaises(TypeError):
            exampleA.addModel('model_B', [out2])
        exampleA.neuralizeModel(1)

        exampleB = Modely(visualizer=None, seed=2)
        exampleB.addModel('model_B', [out2])
        exampleB.neuralizeModel(1)

        resultsA = exampleA({'x': [[1, 3, 3], [1, 2, 1], [2, 3, 4]]})
        self.TestAlmostEqual([12.507442474365234], resultsA['out1'])

        resultsB = exampleB({'f': [1, 3, 3, 1, 2, 1]})
        self.TestAlmostEqual([6.585615158081055, 4.480303764343262, 4.106618881225586, 3.18161678314209], resultsB['out2'])

        log.setAllLevel(logging.CRITICAL)

    def test_check_multiple_streams_compatibility_fir(self):
        NeuObj.clearNames()
        log.setAllLevel(logging.WARNING)
        x = Input('x')

        fir = Fir()
        with self.assertRaises(TypeError):
            fir(x.last()) + fir(x.tw(2.0)) + fir(x.sw(3))

        out1 = Output('out1', fir(x.last()))
        out2 = Output('out2', fir(x.tw(2.0)))

        exampleA = Modely(visualizer=None, seed=2)
        with self.assertRaises(TypeError):
            exampleA.addModel('model', [out1, out2])
        exampleA.addModel('model_A', [out1])
        with self.assertRaises(TypeError):
            exampleA.addModel('model_B', [out2])
        exampleA.neuralizeModel(1)

        exampleB = Modely(visualizer=None, seed=2)
        exampleB.addModel('model_B', [out2])
        exampleB.neuralizeModel(1)

        resultsA = exampleA({'x': [1, 3]})
        self.TestAlmostEqual([0.6146950721740723, 1.8440852165222168], resultsA['out1'])

        resultsB = exampleB({'x': [1, 4, 5]})
        self.TestAlmostEqual([2.138746500015259, 4.363844871520996], resultsB['out2'])

        log.setAllLevel(logging.CRITICAL)

    def test_constant_and_parameter(self):
        NeuObj.clearNames()
        c1 = Constant('c1', values=5.0)  # {'dim': 1} -> shape (1,)
        c11 = Constant('c11', values=5.0)  # {'dim': 1} -> shape (1,)
        c2 = Constant('c2', values=[5.0, 2.0, 1.0])  # {'dim': 3} -> shape (3,)
        c3 = Constant('c3', values=[[5.0, 2.0, 1.0], [3.0, 4.0, 5.0]])  # {'dim': (2,3)} -> shape (2,3)
        c4 = Constant('c4', sw=2, values=[[2, 3, 4], [1, 2, 3]])  # {'sw':2, 'dim': 3} -> shape (2,3)
        c5 = Constant('c5', tw=4, values=[[2, 3, 4], [1, 2, 3]])  # {'tw':4, 'dim': 3} -> shape (2,3)
        c6 = Constant('c6', sw=2, values=[[[5.0, 2.0, 1.0], [3.0, 4.0, 5.0]], [[5.0, 2.0, 1.0], [3.0, 4.0,5.0]]])  # {'sw':2, 'dim': (2,3)} -> shape (2,2,3)
        self.assertEqual({'dim': 1}, c1.dim)
        self.assertEqual({'dim': 1}, c11.dim)
        self.assertEqual({'dim': 3}, c2.dim)
        self.assertEqual({'dim': [2,3]}, c3.dim)
        self.assertEqual({'dim': 3, 'sw': 2}, c4.dim)
        self.assertEqual({'dim': 3, 'tw': 4}, c5.dim)
        self.assertEqual({'dim': [2,3], 'sw':2}, c6.dim)

        p1 = Parameter('p1', values=5.0)  # {'dim': 1} -> shape (1,)
        p11 = Parameter('p11', values=[5.0]) # {'dim': 1} -> shape (1,)
        p111 = Parameter('p111', values=[[5.0]])  # {'dim': 1} -> shape (1,)
        p2 = Parameter('p2', values=[5.0, 2.0, 1.0])  # {'dim': 3}
        p22 = Parameter('p22', sw=1, values=[[2, 3, 4]]) # {'dim': 3, 'sw': 1} -> shape (1,3)
        p3 = Parameter('p3', values=[[5.0, 2.0, 1.0], [3.0, 4.0, 5.0],[3.0, 4.0, 5.0],[3.0, 4.0, 5.0],[3.0, 4.0, 5.0]])  # {'dim': [5,3]}
        p4 = Parameter('p4', sw=2, values=[[2, 3, 4], [1, 2, 3]])  # {'sw':2, 'dim': 3}
        p5 = Parameter('p5', tw=4, values=[[2, 3, 4], [1, 2, 3]])  # {'tw':4, 'dim': 3}
        p6 = Parameter('p6', sw=2, values=[[[5.0, 2.0, 1.0], [3.0, 4.0, 5.0]], [[5.0, 2.0, 1.0], [3.0, 4.0, 5.0]]])  # {'sw':2, 'dim': (2,3)} -> shape (2,2,3)
        self.assertEqual({'dim': 1}, p1.dim)
        self.assertEqual({'dim': 1}, p11.dim)
        self.assertEqual({'dim': [1,1]}, p111.dim)
        self.assertEqual({'dim': 3}, p2.dim)
        self.assertEqual({'dim': 3, 'sw':1}, p22.dim)
        self.assertEqual({'dim': [5,3]}, p3.dim)
        self.assertEqual({'dim': 3, 'sw': 2}, p4.dim)
        self.assertEqual({'dim': 3, 'tw': 4}, p5.dim)
        self.assertEqual({'dim': [2,3], 'sw':2}, p6.dim)

        x = Input('x',dimensions=5)
        out1 = Output('out1', ParamFun(myFunPar, parameters_and_constants=[p1])(x.last()))
        out11 = Output('out11', ParamFun(myFunPar, parameters_and_constants=[p11])(x.last()))
        out111 = Output('out111', ParamFun(myFunPar, parameters_and_constants=[p111])(x.last()))
        out2 = Output('out2', ParamFun(myFunPar, parameters_and_constants=[p2])(x.last()))
        out22 = Output('out22', ParamFun(myFunPar, parameters_and_constants=[p22])(x.last()))
        out3 = Output('out3', ParamFun(myFunPar, parameters_and_constants=[p3])(x.last()))
        out4 = Output('out4', ParamFun(myFunPar, parameters_and_constants=[p4])(x.last()))
        out5 = Output('out5', ParamFun(myFunPar, parameters_and_constants=[p5])(x.last()))
        out6 = Output('out6', ParamFun(myFunPar, parameters_and_constants=[p6])(x.last()))

        nn = Modely(visualizer=None)
        nn.addModel('model', [out1,out11,out111,out2,out22,out3,out4,out5,out6])
        nn.neuralizeModel(2.0)
        results = nn({'x':[[1,2,3,4,5]]})
        self.assertEqual(results['out1'][0], p1.dim['dim'])
        self.assertEqual(results['out11'][0], p11.dim['dim'])
        self.assertEqual(results['out111'][0][0], p111.dim['dim'])
        self.assertEqual(results['out2'][0], p2.dim['dim'])
        self.assertEqual(results['out22'][0][0][1], p22.dim['dim'])
        self.assertEqual(results['out22'][0][0][0], p22.dim['sw'])
        self.assertEqual(results['out3'][0][0], p3.dim['dim'])
        self.assertEqual(results['out3'][0][0][1], p4.dim['dim'])
        self.assertEqual(results['out4'][0][0][0], p4.dim['sw'])
        self.assertEqual(results['out5'][0][0][1], p5.dim['dim'])
        self.assertEqual(results['out5'][0][0][0], p5.dim['tw']/2.0)
        self.assertEqual(results['out6'][0][0][1:3], p6.dim['dim'])
        self.assertEqual(results['out6'][0][0][0], p6.dim['sw'])

        NeuObj.clearNames(['x','out1','out11','out2','out22','out3','out4','out5','out6'])
        x = Input('x')
        with self.assertRaises(TypeError):
            Output('out1', Linear(W=p1, b=p1)(x.last()))
        with self.assertRaises(TypeError):
            Output('out1', Linear(W=p11, b=p11)(x.last()))
        out1 = Output('out1', Linear(W=p111, b=p1)(x.last()))
        out11 = Output('out11', Linear(W=p111, b=p11)(x.last()))
        with self.assertRaises(TypeError):
            Output('out111', Linear(W=p111, b=p111)(x.last()))

        x5 = Input('x5',dimensions=5)
        with self.assertRaises(TypeError):
            Output('out2', Linear(W=p1, b=p1)(x5.last()))
        with self.assertRaises(TypeError):
            Output('out2', Linear(output_dimension=3, W=p1, b=p1)(x5.last()))
        out2 = Output('out2', Linear(W=p3, b=p2)(x5.last()))
        out3 = Output('out3', Linear(output_dimension=3, W=p3, b=p2)(x5.last()))
        with self.assertRaises(TypeError):
            Output('out3', Linear(output_dimension=3, W=p3, b=p22)(x5.last()))
        with self.assertRaises(TypeError):
            Output('out6', Linear(W=p6)(x.sw(2)))

        x2 = Input('x2', dimensions=2)
        with self.assertRaises(TypeError):
            Output('out4', Linear(output_dimension=3, W=p4, b=p2)(x2.last()))

        out4 = Output('out4', Fir(W=p4, b=p2)(x.sw(2)))
        out41 = Output('out41', Fir(output_dimension=3, W=p4, b=p2)(x.sw(2)))
        with self.assertRaises(TypeError):
            Output('out41', Fir(output_dimension=3, W=p5, b=p2)(x.sw(2)))
        with self.assertRaises(ValueError):
            Output('out41', Fir(output_dimension=3, W=p4, b=p2)(x.sw(4)))
        with self.assertRaises(TypeError):
            Output('out41', Fir(output_dimension=3, W=p4, b=p4)(x.sw(2)))
        out5 = Output('out5', Fir(output_dimension=3, W=p5, b=p2)(x.tw(4)))
        out51 = Output('out51', Fir(W=p5, b=p2)(x.tw(4)))
        with self.assertRaises(ValueError):
            Output('out6', Fir(output_dimension=3, W=p6)(x.sw(2)))
        with self.assertRaises(TypeError):
            Output('out6', Fir(W=p6)(x.sw(2)))

        nn = Modely(visualizer=None)
        nn.addModel('model', [out1, out11, out111, out2, out22, out3, out4, out41, out5, out51, out6])
        nn.neuralizeModel(2.0)
        results = nn({'x': [1, 2, 3, 4, 5, 6, 7, 8, 9], 'x5': [[1, 2, 3, 4, 5]]})
        self.assertEqual((1,),np.array(results['out1']).shape)
        self.assertEqual((1,),np.array(results['out11']).shape)
        self.assertEqual((1, 1, 3),np.array(results['out2']).shape)
        self.assertEqual((1, 1, 3), np.array(results['out3']).shape)
        self.assertEqual((1, 1, 3), np.array(results['out4']).shape)
        self.assertEqual((1, 1, 3), np.array(results['out41']).shape)
        self.assertEqual((1, 1, 3), np.array(results['out5']).shape)
        self.assertEqual((1, 1, 3), np.array(results['out51']).shape)

    def test_multi_model_json_and_subjson(self):
        Stream.resetCount()
        NeuObj.clearNames()
        x = Input('x')
        y = Input('y')

        c1 = Constant('c1', values=5.0)
        c3 = Constant('c3', values=5.0)

        rel2 = Linear(W=Parameter('W2', values=[[2.0]]), b=False)(y.last())
        rel4 = Linear(W=Parameter('W4', values=[[4.0]]), b=False)(y.last())
        
        def fun2(x, a):
            return x * a
        
        def fun4(x, b):
            return x + b

        out1 = Output('out1', c1 + Linear(W=Parameter('W1', values=[[1.0]]), b=False)(x.last()))
        out2 = Output('out2', rel2 + ParamFun(fun2, parameters_and_constants=[c1])(y.last()))
        out3 = Output('out3', c3 + Linear(W=Parameter('W3', values=[[3.0]]), b=False)(x.last()))
        out4 = Output('out4', rel4 + ParamFun(fun4, parameters_and_constants=[c3])(y.last()))

        nn = Modely(visualizer=None)
        nn.addModel('model_A', [out1, out2])
        nn.addModel('model_B', [out3, out4])
        nn.addClosedLoop(rel2,y)
        
        subjson_A = subjson_from_model(nn.json, 'model_A')
        subjson_B = subjson_from_model(nn.json, 'model_B')
        self.assertEqual(subjson_A['Constants']['c1'],nn.json['Constants']['c1'])
        self.assertEqual(subjson_B['Constants']['c3'],nn.json['Constants']['c3'])
        self.assertEqual(subjson_A['Functions']['FParamFun11'], nn.json['Functions']['FParamFun11'])
        self.assertEqual(subjson_B['Functions']['FParamFun16'], nn.json['Functions']['FParamFun16'])
        self.assertEqual(subjson_A['Inputs']['x'], nn.json['Inputs']['x'])
        self.assertEqual(subjson_B['Inputs']['x'], nn.json['Inputs']['x'])
        self.assertEqual(subjson_A['Models'], 'model_A')
        self.assertEqual(subjson_B['Models'], 'model_B')
        self.assertEqual(sorted(list(subjson_A['Relations'].keys())), sorted(nn.json['Models']['model_A']['Relations']))
        self.assertEqual(sorted(list(subjson_B['Relations'].keys())), sorted(nn.json['Models']['model_B']['Relations']))
        self.assertEqual(sorted(list(subjson_A['Parameters'].keys())), sorted(['W2', 'W1']))
        self.assertEqual(sorted(list(subjson_B['Parameters'].keys())), sorted(['W3', 'W4']))
        self.assertEqual(sorted(list(subjson_A['Outputs'].keys())), sorted(['out1', 'out2']))
        self.assertEqual(sorted(list(subjson_B['Outputs'].keys())), sorted(['out3', 'out4']))
        yval = copy.deepcopy(nn.json['Inputs']['y'])
        del yval['closedLoop']
        del yval['local']
        self.assertEqual(subjson_A['Inputs']['y'], nn.json['Inputs']['y'])
        self.assertEqual(subjson_B['Inputs']['y'], yval)

        aa = Modely(visualizer=None)
        aa.addModel('model_A', [out1, out2])
        aa.addClosedLoop(rel2, y)
        self.assertEqual(subjson_A, aa.json)

        bb = Modely(visualizer=None)
        bb.addModel('model_B', [out3, out4])
        self.assertEqual(subjson_B, bb.json)

    def test_add_remove_models(self):
        Stream.resetCount()
        NeuObj.clearNames()
        x = Input('x')
        y = Input('y')

        c1 = Constant('c1', values=5.0)
        c3 = Constant('c3', values=5.0)

        rel2 = Linear(W=Parameter('W2', values=[[2.0]]), b=False)(y.last())
        rel4 = Linear(W=Parameter('W4', values=[[4.0]]), b=False)(y.last())

        def fun2(x, a):
            return x * a

        def fun4(x, b):
            return x + b

        out1 = Output('out1', c1 + Linear(W=Parameter('W1', values=[[1.0]]), b=False)(x.last()))
        out2 = Output('out2', rel2 + ParamFun(fun2, parameters_and_constants=[c1])(y.last()))
        out3 = Output('out3', c3 + Linear(W=Parameter('W3', values=[[3.0]]), b=False)(x.last()))
        out4 = Output('out4', rel4 + ParamFun(fun4, parameters_and_constants=[c3])(y.last()))

        nn = Modely(visualizer=None)
        nn.addModel('model_A', [out1, out2])
        model_A_json_1 = nn.json
        nn.addModel('model_B', [out3, out4])
        nn.removeModel('model_B')
        model_A_json_2 = nn.json
        self.assertEqual(model_A_json_1, model_A_json_2)

    def test_add_remove_minimize(self):
        clearNames()
        input1 = Input('in1').last()
        input2 = Input('in2').last()
        input3 = Input('in3').last()
        output1 = Output('out1', input1)
        output2 = Output('out2', input1)
        output3 = Output('out3', input1)

        test = Modely(visualizer=TextVisualizer(), seed=42)
        test.addModel('model', [output1, output2, output3])
        test.addMinimize('error1', input1, output1)
        test_json_1 = test.json
        test.addMinimize('error2', input2, output2)
        test_json_2 = test.json
        test.addMinimize('error3', input3, output3)
        test_json_3 = test.json
        test.removeMinimize('error3')
        self.assertEqual(test_json_2, test.json)
        test.addMinimize('error3', input3, output3)
        self.assertEqual(test_json_3, test.json)
        test.removeMinimize(['error3','error2'])
        self.assertEqual(test_json_1, test.json)