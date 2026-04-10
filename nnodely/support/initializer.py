def init_constant(indexes, params_size, dict_param={"value": 1}):
    """
    Initializes parameters to a constant value.

    Parameters
    ----------
    dict_param : dict, optional
        Dictionary containing the initialization parameters. Default is {'value': 1}.
            value : int or float
                The constant value to initialize the parameters with.
    """
    return dict_param["value"]


def init_negexp(
    indexes, params_size, dict_param={"size_index": 0, "first_value": 1, "lambda": 3}
):
    """
    Initializes parameters using a negative decay exponential function.

    Parameters
    ----------
    indexes : list
        List of indexes for the parameters.
    params_size : list
        List of sizes for each dimension of the parameters.
    dict_param : dict, optional
        Dictionary containing the initialization parameters. Default is {'size_index': 0, 'first_value': 1, 'lambda': 3}.
            size_index : int
                The index of the dimension to apply the exponential function.
            first_value : int or float
                The value at the start of the range.
            lambda : int or float
                The decay rate parameter of the exponential function.
    """
    import numpy as np

    size_index = dict_param["size_index"]
    # check if the size of the list of parameters is 1, to avoid a division by zero
    x = (
        1
        if params_size[size_index] - 1 == 0
        else indexes[size_index] / (params_size[size_index] - 1)
    )
    return dict_param["first_value"] * np.exp(-dict_param["lambda"] * (1 - x))


def init_exp(
    indexes,
    params_size,
    dict_param={
        "size_index": 0,
        "max_value": 1,
        "lambda": 3,
        "monotonicity": "decreasing",
    },
):
    """
    Initializes parameters using an increasing or decreasing exponential function.

    Parameters
    ----------
    indexes : list
        List of indexes for the parameters.
    params_size : list
        List of sizes for each dimension of the parameters.
    dict_param : dict, optional
        Dictionary containing the initialization parameters. Default is {'size_index': 0, 'max_value': 1, 'lambda': 3, 'monotonicity': 'decreasing'}.
            size_index : int
                The index of the dimension to apply the exponential function.
            max_value : int or float
                The maximum value of the exponential function.
            lambda : int or float
                The rate parameter of the exponential function.
            monotonicity : str
                The monotonicity of the exponential function. Can be 'increasing' or 'decreasing'.

    Raises
    ------
    ValueError
        If the monotonicity is not 'increasing' or 'decreasing'.
    """
    import numpy as np

    size_index = dict_param["size_index"]
    monotonicity = dict_param["monotonicity"]
    if monotonicity == "increasing":
        # increasing exponential, the 'max_value' is the value at x=1, i.e, at the end of the range
        x = (
            1
            if params_size[size_index] - 1 == 0
            else indexes[size_index] / (params_size[size_index] - 1)
        )
        out = dict_param["max_value"] * np.exp(dict_param["lambda"] * (x - 1))
    elif monotonicity == "decreasing":
        # decreasing exponential, the 'max_value' is the value at x=0, i.e, at the beginning of the range
        x = (
            0
            if params_size[size_index] - 1 == 0
            else indexes[size_index] / (params_size[size_index] - 1)
        )
        out = dict_param["max_value"] * np.exp(-dict_param["lambda"] * x)
    else:
        raise ValueError(
            "The parameter monotonicity must be either increasing or decreasing."
        )
    return out


def init_lin(
    indexes,
    params_size,
    dict_param={"size_index": 0, "first_value": 1, "last_value": 0},
):
    """
    Initializes parameters using a linear function.

    Parameters
    ----------
    indexes : list
        List of indexes for the parameters.
    params_size : list
        List of sizes for each dimension of the parameters.
    dict_param : dict, optional
        Dictionary containing the initialization parameters. Default is {'size_index': 0, 'first_value': 1, 'last_value': 0}.
            size_index : int
                The index of the dimension to apply the linear function.
            first_value : int or float
                The value at the start of the range.
            last_value : int or float
                The value at the end of the range.
    """
    size_index = dict_param["size_index"]
    x = (
        0
        if params_size[size_index] - 1 == 0
        else indexes[size_index] / (params_size[size_index] - 1)
    )
    return (dict_param["last_value"] - dict_param["first_value"]) * x + dict_param[
        "first_value"
    ]
