import torch


def argmax_max(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])


def argmin_min(iterable):
    return min(enumerate(iterable), key=lambda x: x[1])


def argmax_dict(iterable: dict):
    return max(iterable.items(), key=lambda x: x[1])


def argmin_dict(iterable: dict):
    return min(iterable.items(), key=lambda x: x[1])


# Linear interpolation function, operating on batches of input data and returning batches of output data
def linear_interp(x, x_data, y_data):
    # Inputs:
    # x: query point, a tensor of shape torch.Size([N, 1, 1])
    # x_data: map of x values, sorted in ascending order, a tensor of shape torch.Size([Q, 1])
    # y_data: map of y values, a tensor of shape torch.Size([Q, 1])
    # Output:
    # y: interpolated value at x, a tensor of shape torch.Size([N, 1, 1])

    # Saturate x to the range of x_data
    x = torch.min(torch.max(x, x_data[0]), x_data[-1])

    # Find the index of the closest value in x_data
    idx = torch.argmin(torch.abs(x_data[:-1] - x), dim=1)

    # Linear interpolation
    y = y_data[idx] + (y_data[idx + 1] - y_data[idx]) / (
        x_data[idx + 1] - x_data[idx]
    ) * (x - x_data[idx])
    return y
