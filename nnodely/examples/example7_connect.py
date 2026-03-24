#!/usr/bin/env python3
"""
"""
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
from nnodely import Input, Fir, Output, Model
# from tensorflow.keras.utils import plot_model

# Optional: set Keras backend (torch, tensorflow, jax)
# import keras
# keras.config.set_backend("torch")

# def main():
    # x = Input('x', dim=1)
    # x0 = Input('x0', dim=1)
    # out = Output('y', Fir(out_features=2)(x.sw(5)+x0.sw(5)))
    # m1 = Model('m1', x, out).build()
    # m1._keras_model.summary()
    # m1({'x': np.random.randn(3, 5, 1).astype(np.float32)})

    # x = Input('x', dim=1)
    # out = Output('y', Fir(out_features=2)(x.sw(5)))
    # #out.closedLoop(x)
    # m1 = Model('m1', x, out).build()
    # plot_model(m1._keras_model, to_file='m1.png', show_shapes=True, expand_nested=True, show_layer_names=True)
    
    # x1 = Input('x1', dim=1)
    # out1 = Output('y1', Fir(out_features=2)(x1.sw(5)))
    # m2 = Model('m2', x1, out1)

    # x2 = Input('x2', dim=1).connect(out1)
    # out2 = Output('y2', Fir(out_features=2)(x2.sw(5)))
    # out2.closedLoop('cl',x1)
    # m3 = Model('m3', [x2,x1], [out2,out1])


    # m4 = Model().load('m3')
    # m5 = Model().load('m5')

    # m4.inputs['x2'].connect(m5.outputs['y1'])

    # m6 = Model('m6', [m4.inputs['y2'],m5.inputs['x1']], [m4.outputs['y2'],m5.outputs['y1']])

    # m6 = Model('m7', [m4.inputs['y2'],m5.inputs['x1']], [m4.outputs['y2'],m5.outputs['y1']])


    # m6 = Model('m6', {'f':m4.inputs['y2'],'k':m5.inputs['x1']}, [m4.outputs['y2'],m5.outputs['y1']])

    # m6.addMinimize('error', m6.outputs['y2'], m6.outputs['y1'], 'mse')
    # m6.addMinimize('less', m6.outputs['y2'], Input('target', dim=2).sw(3), 'less')
    # k = Model('k', [m6.inputs['y2'],m6.inputs['x1']], m6.outputs['y2'])
    # m6.addMinimize('less', m6.outputs['y2'], k(Input('target', dim=2).sw(3)), 'less')
    # m6()

    # d = Data(m6, ... )
    # m6.train(models=['m6'],train = [d1,d2,d4], validation = [d3,d5], test = [d6,d7])

    # loss = Minimize('y2', Fir(out_features=2)(x2.sw(5)),x3.sw(3))



    # m9 = Model('m9', [m6.inputs['y2'],m6.inputs['x1']], [m6.outputs['y2'],m6.outputs['y1']], [loss,less])

    # m6.train(   )

    # print(m1({'x': np.random.randn(3, 5, 1).astype(np.float32)}))
    # print(m2({'x': np.random.randn(3, 5, 1).astype(np.float32)}))

    

    # o = Input('x', dim=1)
    # o1 = Input('x', dim=1)
    # oo = Output('oo', m1(o.sw(5))+m2(o1.sw(5)))
    # m3 = Model('m3', [o,o1], oo).build()
    # m3._keras_model.summary()
    # print(m3(np.random.randn(3, 5, 1).astype(np.float32)))
    # exit()

    # Connect('prova',m1.outputs['y'], m2.inputs['x1'])
    

# main()