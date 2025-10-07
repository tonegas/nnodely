import torch

import torch.nn as nn

from nnodely.basic.relation import NeuObj, Stream
from nnodely.basic.model import Model
from nnodely.support.utils import check, enforce_types
from nnodely.support.jsonutils import merge

interpolation_relation_name = 'Interpolation'
class Interpolation(NeuObj):
    """
    Represents an Interpolation relation in the neural network model.
    This class performs linear interpolation of an input tensor `x` given two vectors of points.

    Parameters
    ----------
    x_points : list[int]|list[float]|list[torch.Tensor]
        A tensor containing the x-coordinates of the data points.
    y_points : list[int]|list[float]|list[torch.Tensor]
        A tensor containing the y-coordinates of the data points.
    mode : str, optional
        The type of interpolation to perform. Possible modalities are: ['linear', ].
        Default is 'linear'.

    Examples
    --------
    .. image:: https://colab.research.google.com/assets/colab-badge.svg
        :target: https://colab.research.google.com/github/tonegas/nnodely/blob/main/examples/interpolation.ipynb
        :alt: Open in Colab

    Example - basic usage:
        >>> x_points = [1.0, 2.0, 3.0, 4.0]
        >>> y_points = [1.0, 4.0, 9.0, 16.0]
        >>> x = Input('x')

        >>> rel1 = Interpolation(x_points=x_points,y_points=y_points, mode='linear')(x.last())
        
        >>> out = Output('out',rel1)
    """

    @enforce_types
    def __init__(self, x_points:list,
                 y_points:list, *,
                 mode:str|None = 'linear'):

        self.relation_name = interpolation_relation_name
        self.x_points = x_points
        self.y_points = y_points
        self.mode = mode

        self.available_modes = ['linear', 'polynomial']

        super().__init__('P' + interpolation_relation_name + str(NeuObj.count))
        check(len(x_points) == len(y_points), ValueError, 'The x_points and y_points must have the same length.')
        check(mode in self.available_modes, ValueError, f'The mode must be one of {self.available_modes}.')
        check(len(torch.tensor(x_points).shape) == 1, ValueError, 'The x_points must be a 1D tensor.')
        check(len(torch.tensor(y_points).shape) == 1, ValueError, 'The y_points must be a 1D tensor.')

    @enforce_types
    def __call__(self, obj:Stream) -> Stream:
        stream_name = interpolation_relation_name + str(Stream.count)
        check(type(obj) is Stream, TypeError, f"The type of {obj} is {type(obj)} and is not supported for Interpolation operation.")

        stream_json = merge(self.json,obj.json)
        stream_json['Relations'][stream_name] = [interpolation_relation_name, [obj.name], self.x_points, self.y_points, self.mode]
        return Stream(stream_name, stream_json, obj.dim)


class Interpolation_Layer(nn.Module):
    def __init__(self, x_points, y_points, mode='linear'):
        super(Interpolation_Layer, self).__init__()
        self.mode = mode
        ## Sort the points
        if type(x_points) is not torch.Tensor:
            x_points = torch.tensor(x_points)
        if type(y_points) is not torch.Tensor:
            y_points = torch.tensor(y_points)
        self.x_points, indices = torch.sort(x_points)
        self.y_points = y_points[indices]

        self.x_points = self.x_points.unsqueeze(-1)
        self.y_points = self.y_points.unsqueeze(-1)

    def forward(self, x):
        if self.mode == 'linear':
            return self.linear_interpolation(x)
        else:
            raise NotImplementedError
    
    def linear_interpolation(self, x):
        # Inputs: 
        # x: query point, a tensor of shape torch.Size([N, 1, 1])
        # x_data: map of x values, sorted in ascending order, a tensor of shape torch.Size([Q, 1])
        # y_data: map of y values, a tensor of shape torch.Size([Q, 1])
        # Output:
        # y: interpolated value at x, a tensor of shape torch.Size([N, 1, 1])

        # Saturate x to the range of x_data
        x = torch.min(torch.max(x,self.x_points[0]),self.x_points[-1])
        # Find the index of the closest value in x_data
        idx = torch.argmin(torch.abs(self.x_points[:-1] - x),dim=1)
        # Linear interpolation
        y = self.y_points[idx] + (self.y_points[idx+1] - self.y_points[idx])/(self.x_points[idx+1] - self.x_points[idx])*(x - self.x_points[idx])
        return y

def createInterpolation(self, *inputs):
    return Interpolation_Layer(x_points=inputs[0], y_points=inputs[1], mode=inputs[2])

setattr(Model, interpolation_relation_name, createInterpolation)