import sys, os
# append a new directory to sys.path
sys.path.append(os.getcwd())

from nnodely import *

x = Input('x')
F = Input('F')

print("------------------------EXAMPLE 1------------------------")
# Example 1 Parametric Function Basic
# This function has two parameters p1 and p2 of size 1 and two inputs K1 and K2
# The output size is user defined
# if it is not specified the output is expected to be 1
def myFun(K1,K2,p1,p2):
    import torch
    return p1*K1+p2*torch.sin(K2)

parfun = ParamFun(myFun)
out = Output('out',parfun(x.last(),F.last()))
example = Modely(visualizer=MPLVisualizer())
example.addModel('out',out)
example.neuralizeModel()
print(example({'x':[1],'F':[1]}))
print(example({'x':[1,2],'F':[1,2]}))
example.visualizer.showFunctions(list(example.model_def['Functions'].keys()),xlim=[[-5,5],[-1,1]])
#

print("------------------------EXAMPLE 2------------------------")
# Example 2
# I define the output of the function which is now 4 and then if the function does not output with size 4 I give an error
# also in this case I have two parameters p1 and p2
# in the function there is a product between a vector and a scalar and then a sum with a scalar
# the size of parameters p1 and p2 is 1
def myFun(K1,K2,p1,p2):
    import torch
    return torch.tensor([p1,p1,p1,p1])*K1+p2*torch.sin(K2)
parfun = ParamFun(myFun) # definisco una funzione scalare basata su myFun
out = Output('out-2',parfun(x.last(),F.last()))
example = Modely()
example.addModel('out',out)
example.neuralizeModel()
print(example({'x':[1],'F':[1]}))
print(example({'x':[1,2],'F':[1,2]}))
#

print("------------------------EXAMPLE 3------------------------")
# Example 2
# I define the output of the function which is now 4 and then if the function does not output with size 4 I give an error
# also in this case I have two parameters p1 and p2
# in the function there is a product between a vector and a scalar and then a sum with a scalar
# the size of parameters p1 and p2 is 1
def myFun(K1,K2,p1,p2):
    import torch
    return p1*K1+p2*torch.sin(K2)
parfun = ParamFun(myFun) # definisco una funzione scalare basata su myFun
out = Output('out-3',parfun(x.tw(2),F.tw(2)))
example = Modely()
example.addModel('out',out)
example.neuralizeModel(1)
print(example({'x':[1,1],'F':[1,1]}))
print(example({'x':[1,2,3],'F':[1,2,3]}))
#

print("------------------------EXAMPLE 4------------------------")
# Example 4
# This case I define the specific size of the parameters
# the first p1 is a 4 row column vector
# The output size of the function is 1
# in this case I make a dot product between the vector and p1 which is a vector [4,1]
# The time dimension of the output is not defined but depends on the input
# In the first call parfun(x.tw(1),F.tw(1)) the time output is a 1 sec window
# In the second call parfun(x,F) is an instant output
def myFun(K1,K2,p1):
    import torch
    return torch.stack([K1,2*K1,3*K1,4*K1],dim=2).squeeze(-1)*p1+K2
parfun = ParamFun(myFun, parameters_and_constants = {'p1':(1,4)})
out = Output('out-4',parfun(x.last(),F.last()))
example = Modely()
example.addModel('out',out)
example.neuralizeModel()
print(example({'x':[1],'F':[1]}))
#

print("------------------------EXAMPLE 5------------------------")
# Example 5
# This case I create a parameter that I pass to the parametric function
# The parametric function takes a parameter of size 1 and tw = 1
# The function has two inputs, the first two are inputs and the second is a K parameter
# The function creates a tensor performs a dot product between input 1 and p1 (which is effectively K Parameter)
def myFun(K1,p1):
    return K1*p1
K = Parameter('k', dimensions =  1, sw = 1,values=[[2.0]])
parfun = ParamFun(myFun, parameters_and_constants = [K] )
out = Output('out-5',parfun(x.sw(1)))
example = Modely(visualizer=MPLVisualizer())
example.addModel('out',out)
example.neuralizeModel(0.25)
print(example({'x':[1,1,1,1],'F':[1,1,1,1]}))
example.visualizer.showFunctions(list(example.model_def['Functions'].keys()), xlim = [-5,5])
#

print("------------------------EXAMPLE 6------------------------")
# Example 6
P1 = 7.0
def myFun(K1,p1):
    return K1*p1
parfun = ParamFun(myFun)
out = Output('out-6',parfun(x.sw(1),Constant('const',values=P1)))
example = Modely(visualizer=MPLVisualizer())
example.addModel('out',out)
example.neuralizeModel(1)
print(example({'x':[1,1,1,1]}))
example.visualizer.showFunctions(list(example.model_def['Functions'].keys()))
#

print("------------------------EXAMPLE 7------------------------")
# Example 7
def myFun(K1,p1):
    return K1*p1
K = Parameter('k1', dimensions =  1, tw = 1, values=[[2.0],[3.0],[4.0],[5.0]])
R = Parameter('r1', dimensions =  1, tw = 1, values=[[5.0],[4.0],[3.0],[2.0]])
parfun = ParamFun(myFun)
out = Output('out-7',parfun(x.tw(1),K)+parfun(x.tw(1),R))
example = Modely()
example.addModel('out',out)
example.neuralizeModel(0.25)
print(example({'x':[1,1,1,1],'F':[1,1,1,1]}))
#

print("------------------------EXAMPLE 8------------------------")
# Example 8
P1 = [[5.0],[4.0],[3.0],[2.0]]
def myFun(K1,p1):
    return K1*p1
parfun = ParamFun(myFun)
out = Output('out-8',parfun(x.sw(4),Constant('ccc',values=P1)))
example = Modely()
example.addModel('out',out)
example.neuralizeModel(0.25)
print(example({'x':[1,1,1,1]}))
#

print("------------------------EXAMPLE 9------------------------")
# Example 9
P1 = 7.0
def myFun(K1,p1):
    return K1*p1
parfun = ParamFun(myFun,parameters_and_constants=[Constant('r',values=P1)])
out = Output('out-9',parfun(x.sw(4)))
example = Modely()
example.addModel('out',out)
example.neuralizeModel(0.25)
print(example({'x':[1,1,1,1]}))
#

print("------------------------EXAMPLE 10------------------------")
# Example 10
P1 = 12.0
def myFun(K1,p1):
    return K1*p1
parfun = ParamFun(myFun,parameters_and_constants=[Constant('rr',values=P1)])
out = Output('out-10',parfun(x.sw(4)))
example = Modely()
example.addModel('out',out)
example.neuralizeModel(0.25)
print(example({'x':[1,1,1,1]}))
#

print("------------------------EXAMPLE 11------------------------")
# Example 11
def myFun(inin, p1):
    print(f'inin:{inin.shape}')
    print(f'p1:{p1.shape}')
    return inin * p1
parfun = ParamFun(myFun, map_over_batch=True)
out = Output('out-11',parfun(x.sw(4)))
example = Modely()
example.addModel('out',out)
example.neuralizeModel(0.25)
print(example({'x':[1,3,3,1]}))
#

print("------------------------EXAMPLE 12------------------------")
# Example 12
def myFun(inin, p1):
    print(f'inin:{inin.shape}')
    print(f'p1:{p1.shape}')
    return inin * p1
parfun = ParamFun(myFun, map_over_batch=True)
p = Constant('co',values=[[2]])
#out = Output('out-121',parfun(x.sw(4),p))
out = Output('out-12',parfun(p,x.sw(4)))
example = Modely()
example.addModel('out',out)
example.neuralizeModel(0.25)
print(example({'x':[1,3,3,1]}))
#

print("------------------------EXAMPLE 13------------------------")
# Example 13
def myFun(inin, p1, p2, p3):
    return inin * p1 + p2
parfun = ParamFun(myFun,parameters_and_constants=[1,(1,1),3])
out = Output('out-13',parfun(x.sw(4)))
example = Modely()
example.addModel('out',out)
example.neuralizeModel(0.25)
print(example({'x':[1,3,3,1]}))
#