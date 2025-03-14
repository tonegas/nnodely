import sys
import os

from nnodely.relation import NeuObj

# append a new directory to sys.path
sys.path.append(os.getcwd())

from nnodely import *

example = 10
data_struct = ['time', ('x', 'x_state'), 'x_s', 'F']
data_folder = './data/'

print("------------------------EXAMPLE 1------------------------")
# NON Recurrent Training
x = Input('x')
F = Input('F')
x_state = State('x_state')
x_out = Fir(x_state.tw(0.5))+F.last()
x_out.closedLoop(x_state)
out = Output('out',x_out)

mass_spring_damper = Modely(seed=42)
mass_spring_damper.addModel('out', out)
mass_spring_damper.addMinimize('error', out, x.next())

mass_spring_damper.neuralizeModel(0.1)

mass_spring_damper.loadData(name='dataset', source=data_folder, skiplines=1, format=data_struct)

print('F (first):', mass_spring_damper.data['dataset']['F'][0])
print('F (last):', mass_spring_damper.data['dataset']['F'][-1])
print('x (first):', mass_spring_damper.data['dataset']['x'][0])
print('x (last):', mass_spring_damper.data['dataset']['x'][-1])
print('x_state (first):', mass_spring_damper.data['dataset']['x_state'][0])
print('x_state (last):', mass_spring_damper.data['dataset']['x_state'][-1])

# Training non ricorrente
params = {'num_of_epochs': 1,
      'train_batch_size': 4,
      'val_batch_size':4,
      'test_batch_size':1,
      'lr':0.001}
mass_spring_damper.trainModel(splits=[70,20,10], shuffle_data=False, training_params=params)


print("------------------------EXAMPLE 2------------------------")
NeuObj.clearNames()
# Recurrent Training
x = Input('x')
F = Input('F')
x_state = State('x_state')
x_out = Fir(x_state.tw(0.5))+F.last()
x_out.closedLoop(x_state)
out = Output('out',x_out)

mass_spring_damper = Modely(seed=42)
mass_spring_damper.addModel('model', out)
mass_spring_damper.addMinimize('error', out, x.next())

mass_spring_damper.neuralizeModel(0.1)

mass_spring_damper.loadData(name='dataset', source=data_folder, skiplines=1, format=data_struct)

print('F (first):', mass_spring_damper.data['dataset']['F'][0])
print('F (last):', mass_spring_damper.data['dataset']['F'][-1])
print('x (first):', mass_spring_damper.data['dataset']['x'][0])
print('x (last):', mass_spring_damper.data['dataset']['x'][-1])
print('x_state (first):', mass_spring_damper.data['dataset']['x_state'][0])
print('x_state (last):', mass_spring_damper.data['dataset']['x_state'][-1])

# Training ricorrente
params = {'num_of_epochs': 3,
      'train_batch_size': 4,
      'val_batch_size':4,
      'test_batch_size':1,
      'lr':0.001}
mass_spring_damper.trainModel(splits=[50,30,20], prediction_samples=2, shuffle_data=False, training_params=params)

print('finale state: ', mass_spring_damper.model.states)
mass_spring_damper.resetStates()
print('state clear: ', mass_spring_damper.model.states)


print("------------------------EXAMPLE 3------------------------")
NeuObj.clearNames()
# NON Recurrent Training (2 state variables)
x = Input('x')
x_state = State('x_state')
y_state = State('y_state')
x_out = Fir(x_state.tw(0.5))
y_out = Fir(y_state.tw(0.5))
x_out = ClosedLoop(x_out, x_state)
y_out = ClosedLoop(y_out, y_state)
out = Output('out',x_out+y_out)

mass_spring_damper = Modely(seed=42)
mass_spring_damper.addModel('model', out)
mass_spring_damper.addMinimize('error', out, x.next())

mass_spring_damper.neuralizeModel(0.1)

mass_spring_damper.loadData(name='dataset', source=data_folder, skiplines=1, format=data_struct)

print('x (first):', mass_spring_damper.data['dataset']['x'][0])
print('x (last):', mass_spring_damper.data['dataset']['x'][-1])
print('x_state (first):', mass_spring_damper.data['dataset']['x_state'][0])
print('x_state (last):', mass_spring_damper.data['dataset']['x_state'][-1])

# Training non ricorrente
params = {'num_of_epochs': 3,
      'train_batch_size': 4,
      'val_batch_size':4,
      'test_batch_size':1,
      'lr':0.001}
mass_spring_damper.trainModel(splits=[100,0,0], shuffle_data=False, training_params=params)
print('finale state: ', mass_spring_damper.model.states)


print("------------------------EXAMPLE 4------------------------")
NeuObj.clearNames()
# Recurrent Training (2 state variables)
x = Input('x')
F = Input('F')
x_state = State('x_state')
y_state = State('y_state')
x_out = Fir(x_state.tw(0.5))
y_out = Fir(y_state.tw(0.5))
x_out.closedLoop(x_state)
y_out.closedLoop(y_state)
out = Output('out',x_out+y_out)

mass_spring_damper = Modely(seed=42)
mass_spring_damper.addModel('model', out)
mass_spring_damper.addMinimize('error', out, x.next())

mass_spring_damper.neuralizeModel(0.1)

mass_spring_damper.loadData(name='dataset', source=data_folder, skiplines=1, format=data_struct)

print('x (first):', mass_spring_damper.data['dataset']['x'][0])
print('x (last):', mass_spring_damper.data['dataset']['x'][-1])
print('x_state (first):', mass_spring_damper.data['dataset']['x_state'][0])
print('x_state (last):', mass_spring_damper.data['dataset']['x_state'][-1])

# Training non ricorrente
params = {'num_of_epochs': 3,
      'train_batch_size': 4,
      'val_batch_size':4,
      'test_batch_size':1,
      'lr':0.01}
mass_spring_damper.trainModel(splits=[50,30,20], prediction_samples=3, shuffle_data=False, training_params=params)
print('finale state: ', mass_spring_damper.model.states)


print("------------------------EXAMPLE 5------------------------")
NeuObj.clearNames()
# Recurrent Training with multi-dimensional output and multi-window
x = Input('x', dimensions=3)
F = Input('F')
x_state = State('x_state', dimensions=3)
y_state = State('y_state', dimensions=3)
x_out = Linear(output_dimension=3)(x_state.tw(0.5))
y_out = Linear(output_dimension=3)(y_state.tw(0.5))
x_out.closedLoop(x_state)
y_out.closedLoop(y_state)
out = Output('out',x_out+y_out)

mass_spring_damper = Modely(seed=42)
mass_spring_damper.addModel('model', out)
mass_spring_damper.addMinimize('error', out, x.tw(0.5))

mass_spring_damper.neuralizeModel(0.1)

mass_spring_damper.loadData(name='dataset', source=data_folder, skiplines=1, format=data_struct)

print('x (first):', mass_spring_damper.data['dataset']['x'][0])
print('x (last):', mass_spring_damper.data['dataset']['x'][-1])
print('x_state (first):', mass_spring_damper.data['dataset']['x_state'][0])
print('x_state (last):', mass_spring_damper.data['dataset']['x_state'][-1])

# Training non ricorrente
params = {'num_of_epochs': 3,
      'train_batch_size': 4,
      'val_batch_size':4,
      'test_batch_size':1,
      'lr':0.01}
mass_spring_damper.trainModel(splits=[50,30,20], prediction_samples=3, shuffle_data=False, training_params=params)
print('finale state: ', mass_spring_damper.model.states)


print("------------------------EXAMPLE 6------------------------")
NeuObj.clearNames()
# Recurrent Training with state variables and close_loop
x = Input('x')
F = Input('F')
x_state = State('x_state')
y_state = State('y_state')
x_out = Fir(x_state.tw(0.3))
y_out = Fir(y_state.tw(0.3))
x_out.closedLoop(x_state)
y_out.closedLoop(y_state)
out = Output('out',x_out+y_out+F.last())

mass_spring_damper = Modely(seed=42)
mass_spring_damper.addModel('out', out)
mass_spring_damper.addMinimize('error', out, x.next())

mass_spring_damper.neuralizeModel(0.1)

mass_spring_damper.loadData(name='dataset', source=data_folder, skiplines=1, format=data_struct)

print('x (first):', mass_spring_damper.data['dataset']['x'][0])
print('x (last):', mass_spring_damper.data['dataset']['x'][-1])
print('x_state (first):', mass_spring_damper.data['dataset']['x_state'][0])
print('x_state (last):', mass_spring_damper.data['dataset']['x_state'][-1])

# Training non ricorrente
params = {'num_of_epochs': 3,
      'train_batch_size': 4,
      'val_batch_size':4,
      'test_batch_size':1,
      'lr':0.01}
mass_spring_damper.trainModel(splits=[50,30,20], closed_loop={'F':'out'}, prediction_samples=3, shuffle_data=False, training_params=params)
print('finale state: ', mass_spring_damper.model.states)


print("------------------------EXAMPLE 7------------------------")
NeuObj.clearNames()
# Simple integral operation
x = Input('x')
out = Output('int',Integrate(x.last()))
integrator = Modely(seed=42)
integrator.addModel('out', out)
integrator.neuralizeModel(0.1)
print(integrator({'x':[1,1,1,1,1,1,1]}))


print("------------------------EXAMPLE 8------------------------")
NeuObj.clearNames()
# Simple one pole convergence
x = Input('x')
xs = State('x_state')
int = Integrate((x.last()-xs.last()))
int.closedLoop(xs)
out = Output('pole',int)
integrator = Modely(seed=42)
integrator.addModel('out', out)
integrator.neuralizeModel(0.1)
print(integrator({'x':[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]}))


print("------------------------EXAMPLE 9------------------------")
NeuObj.clearNames()
# Simple derivative operation
x = Input('x')
der = Derivate(x.last())
out = Output('der',der)
integrator = Modely(seed=42)
integrator.addModel('out', out)
integrator.neuralizeModel(1)
print(integrator({'x':[1,2,3,2,1,1,1,1]}))


print("------------------------EXAMPLE 10------------------------")
NeuObj.clearNames()
# Simple derivative operation
x = Input('x').s(1)
out = Output('der',x)
integrator = Modely(seed=42)
integrator.addModel('out', out)
integrator.neuralizeModel(1)
print(integrator({'x':[1,2,3,2,1,1,1,1]}))