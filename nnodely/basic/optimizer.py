import copy
import torch

from nnodely.support.utils import check


class Optimizer:
    """
    Represents an optimizer for training neural network models.

    Parameters
    ----------
    name : str
        The name of the optimizer.
    optimizer_defaults : dict, optional
        A dictionary of default optimizer settings.
    optimizer_params : list, optional
        A list of parameter groups for the optimizer.

    Attributes
    ----------
    name : str
        The name of the optimizer.
    optimizer_defaults : dict
        A dictionary of default optimizer settings.
    optimizer_params : list
        A list of parameter groups for the optimizer.
    all_params : dict or None
        A dictionary of all parameters in the model.
    params_to_train : list or None
        A list of parameters to be trained.
    """

    def __init__(self, name, optimizer_defaults={}, optimizer_params=[]):
        """
        Initializes the Optimizer object.

        Parameters
        ----------
        name : str
            The name of the optimizer.
        optimizer_defaults : dict, optional
            A dictionary of default optimizer settings. Default is an empty dictionary.
        optimizer_params : list, optional
            A list of parameter groups for the optimizer. Default is an empty list.
        """
        self.name = name
        self.optimizer_defaults = copy.deepcopy(optimizer_defaults)
        self.optimizer_params = self.unfold(copy.deepcopy(optimizer_params))
        self.all_params = None
        self.params_to_train = None

    def set_params_to_train(self, all_params, params_to_train):
        """
        Sets the parameters to be trained by the optimizer.

        Parameters
        ----------
        all_params : dict
            A dictionary of all parameters in the model.
        params_to_train : list
            A list of parameters to be trained.
        """
        self.all_params = all_params
        self.params_to_train = params_to_train
        if self.optimizer_params == []:
            for param_name in self.all_params.keys():
                if param_name in self.params_to_train:
                    self.optimizer_params.append({"params": param_name})
                else:
                    self.optimizer_params.append({"params": param_name, "lr": 0.0})

    def set_defaults(self, optimizer_defaults):
        """
        Sets the default optimizer settings.

        Parameters
        ----------
        optimizer_defaults : dict
            A dictionary of default optimizer settings.
        """
        self.optimizer_defaults = copy.deepcopy(optimizer_defaults)

    def set_params(self, optimizer_params):
        """
        Sets the parameter groups for the optimizer.

        Parameters
        ----------
        optimizer_params : list
            A list of parameter groups for the optimizer.
        """
        self.optimizer_params = self.unfold(optimizer_params)

    def unfold(self, params):
        """
        Unfolds the parameter groups into a flat list.

        Parameters
        ----------
        params : list
            A list of parameter groups.

        Returns
        -------
        list
            A flat list of parameter groups.

        Raises
        ------
        KeyError
            If the params argument is not a list.
        """
        optimizer_params = []
        check(type(params) is list, KeyError, f"The params {params} must be a list")
        for param in params:
            if type(param["params"]) is list:
                par_copy = copy.deepcopy(param)
                del par_copy["params"]
                for par in param["params"]:
                    optimizer_params.append({"params": par} | par_copy)
            else:
                optimizer_params.append(param)
        return optimizer_params

    def add_defaults(self, option_name, params, overwrite=True):
        """
        Adds default settings to the optimizer.

        Parameters
        ----------
        option_name : str
            The name of the option to add.
        params : any
            The parameters for the option.
        overwrite : bool, optional
            Whether to overwrite existing settings. Default is True.
        """
        if params is not None:
            if overwrite:
                self.optimizer_defaults[option_name] = params
            elif option_name not in self.optimizer_defaults:
                self.optimizer_defaults[option_name] = params

    def add_option_to_params(self, option_name, params, overwrite=True):
        if params is None:
            return
        for key, value in params.items():
            check(
                self.all_params is not None,
                RuntimeError,
                "Call set_params before add_option_to_params",
            )
            old_key = False
            for param in self.optimizer_params:
                if param["params"] == key:
                    old_key = True
                    if overwrite:
                        param[option_name] = value
                    elif option_name not in param:
                        param[option_name] = value
            if old_key == False:
                self.optimizer_params.append({"params": key, option_name: value})

    def replace_key_with_params(self):
        params = copy.deepcopy(self.optimizer_params)
        for param in params:
            if type(param["params"]) is list:
                for ind, par in enumerate(param["params"]):
                    param["params"][ind] = self.all_params[par]
            else:
                param["params"] = self.all_params[param["params"]]
        return params

    def get_torch_optimizer(self):
        raise NotImplemented("The function get_torch_optimizer must be implemented.")


class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer.

    See also:
            Official PyTorch SGD documentation:
            `torch.optim.SGD <https://pytorch.org/docs/stable/generated/torch.optim.SGD.html>`_

    Parameters
    ----------
    name : str
        The name of the optimizer.
    optimizer_defaults : dict
        A dictionary of default optimizer settings.
    optimizer_params : list
        A list of parameter groups for the optimizer.

    Attributes
    ----------
    name : str
        The name of the optimizer.
    lr : float, optional
        Learning rate. Default is 0.01.
    momentum : float, optional
        Momentum factor. Default is 0.0.
    dampening : float, optional
        Dampening for momentum. Default is 0.0.
    weight_decay : float, optional
        Weight decay (L2 penalty). Default is 0.0.
    nesterov : bool, optional
        Enables Nesterov momentum. Default is False.
    """

    def __init__(self, optimizer_defaults={}, optimizer_params=[]):
        super(SGD, self).__init__("SGD", optimizer_defaults, optimizer_params)

    def get_torch_optimizer(self):
        return torch.optim.SGD(
            self.replace_key_with_params(), **self.optimizer_defaults
        )


class Adam(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer.

    See also:
            Official PyTorch Adam documentation:
            `torch.optim.Adam <https://pytorch.org/docs/stable/generated/torch.optim.SGD.html>`_

    Parameters
    ----------
    name : str
        The name of the optimizer.
    optimizer_defaults : dict
        A dictionary of default optimizer settings.
    optimizer_params : list
        A list of parameter groups for the optimizer.

    Attributes
    ----------
    name : str
        The name of the optimizer.
    lr : float, optional
        Learning rate. Default is 0.001.
    betas : tuple of (float, float), optional
        Coefficients used for computing running averages of gradient and its square. Default is (0.9, 0.999).
    eps : float, optional
        Term added to the denominator to improve numerical stability. Default is 1e-8.
    weight_decay : float, optional
        Weight decay (L2 penalty). Default is 0.0.
    amsgrad : bool, optional
        Whether to use the AMSGrad variant of this algorithm. Default is False.
    """

    def __init__(self, optimizer_defaults={}, optimizer_params=[]):
        super(Adam, self).__init__("Adam", optimizer_defaults, optimizer_params)

    def get_torch_optimizer(self):
        return torch.optim.Adam(
            self.replace_key_with_params(), **self.optimizer_defaults
        )
