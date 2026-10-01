import torch


def nnodely_basic_model_update_state(data_in, rel):
    data_out = data_in.clone()
    max_dim = min(rel.size(1), data_in.size(1))
    data_out[:, -max_dim:, :] = rel[:, -max_dim:, :]
    return data_out


def nnodely_basic_model_timeshift(data_in):
    return torch.cat((data_in[:, 1:, :], data_in[:, :1, :]), dim=1)


class TracerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.all_parameters = {}
        self.all_constants = {}
        self.all_constants["SampleTime"] = torch.tensor(
            0.009999999776482582, requires_grad=False
        )
        self.all_parameters["D"] = torch.nn.Parameter(
            torch.tensor([8.177507400512695]), requires_grad=True
        )
        self.all_parameters["I"] = torch.nn.Parameter(
            torch.tensor([18.563440322875977]), requires_grad=True
        )
        self.all_parameters["P"] = torch.nn.Parameter(
            torch.tensor([16.827768325805664]), requires_grad=True
        )
        self.all_parameters["PFir3W"] = torch.nn.Parameter(
            torch.tensor(
                [
                    [-0.18469615280628204],
                    [-0.12608873844146729],
                    [-0.06660982221364975],
                    [-0.005940185859799385],
                    [0.05625353008508682],
                    [0.1205686405301094],
                    [0.18780383467674255],
                    [0.2590898871421814],
                    [0.3359234631061554],
                    [0.4205183684825897],
                ]
            ),
            requires_grad=True,
        )
        self.all_parameters["PFir5W"] = torch.nn.Parameter(
            torch.tensor(
                [
                    [-1.662617978581693e-05],
                    [4.8189936933340505e-05],
                    [7.177559018600732e-05],
                    [5.9527203120524064e-05],
                    [0.00011585278843995184],
                    [0.00019496057939250022],
                    [0.00014330405974760652],
                    [0.00018421869026497006],
                    [0.00017740165640134364],
                    [8.003542461665347e-05],
                ]
            ),
            requires_grad=True,
        )
        self.all_constants["SamplePart10"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart12"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart17"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["SamplePart26"] = torch.tensor(
            [[0.0, 1.0]], requires_grad=True
        )
        self.all_constants["SamplePart28"] = torch.tensor(
            [[1.0, 0.0]], requires_grad=True
        )
        self.all_constants["SamplePart8"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["TimePart1"] = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            requires_grad=True,
        )
        self.all_constants["TimePart4"] = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            requires_grad=True,
        )
        self.all_parameters = torch.nn.ParameterDict(self.all_parameters)
        self.all_constants = torch.nn.ParameterDict(self.all_constants)

    def update(self, closed_loop={}, connect={}, disconnect=False):
        pass

    def forward(self, kwargs):
        getitem = kwargs["x_m"]
        relation_forward_sample_part12_w = self.all_constants.SamplePart12
        einsum = torch.functional.einsum(
            "bij,ki->bkj", getitem, relation_forward_sample_part12_w
        )
        getitem = relation_forward_sample_part12_w = None
        getitem_1 = kwargs["x_t"]
        relation_forward_sample_part10_w = self.all_constants.SamplePart10
        einsum_1 = torch.functional.einsum(
            "bij,ki->bkj", getitem_1, relation_forward_sample_part10_w
        )
        getitem_1 = relation_forward_sample_part10_w = None
        sub = einsum_1 - einsum
        einsum_1 = einsum = None
        getitem_2 = kwargs["Sub13_int14"]
        update_state = nnodely_basic_model_update_state(getitem_2, sub)
        getitem_2 = None
        relation_forward_sample_part28_w = self.all_constants.SamplePart28
        einsum_2 = torch.functional.einsum(
            "bij,ki->bkj", update_state, relation_forward_sample_part28_w
        )
        relation_forward_sample_part28_w = None
        relation_forward_sample_part26_w = self.all_constants.SamplePart26
        einsum_3 = torch.functional.einsum(
            "bij,ki->bkj", update_state, relation_forward_sample_part26_w
        )
        relation_forward_sample_part26_w = None
        sub_1 = einsum_3 - einsum_2
        einsum_3 = einsum_2 = None
        all_constants_sample_time = self.all_constants.SampleTime
        truediv = sub_1 / all_constants_sample_time
        sub_1 = None
        all_parameters_d = self.all_parameters.D
        mul = truediv * all_parameters_d
        truediv = all_parameters_d = None
        mul_1 = sub * all_constants_sample_time
        all_constants_sample_time = None
        getitem_3 = kwargs["Sub13_int13"]
        relation_forward_sample_part17_w = self.all_constants.SamplePart17
        einsum_4 = torch.functional.einsum(
            "bij,ki->bkj", getitem_3, relation_forward_sample_part17_w
        )
        getitem_3 = relation_forward_sample_part17_w = None
        add = einsum_4 + mul_1
        einsum_4 = mul_1 = None
        all_parameters_i = self.all_parameters.I
        mul_2 = add * all_parameters_i
        all_parameters_i = None
        all_parameters_p = self.all_parameters.P
        mul_3 = sub * all_parameters_p
        sub = all_parameters_p = None
        add_1 = mul_3 + mul_2
        mul_3 = mul_2 = None
        add_2 = add_1 + mul
        add_1 = mul = None
        getitem_4 = kwargs["F"]
        relation_forward_time_part4_w = self.all_constants.TimePart4
        einsum_5 = torch.functional.einsum(
            "bij,ki->bkj", getitem_4, relation_forward_time_part4_w
        )
        getitem_4 = relation_forward_time_part4_w = None
        size = einsum_5.size(0)
        relation_forward_fir5_weights = self.all_parameters.PFir5W
        size_1 = relation_forward_fir5_weights.size(1)
        squeeze = einsum_5.squeeze(-1)
        einsum_5 = None
        matmul = torch.matmul(squeeze, relation_forward_fir5_weights)
        squeeze = relation_forward_fir5_weights = None
        to = matmul.to(dtype=torch.float32)
        matmul = None
        view = to.view(size, 1, size_1)
        to = size = size_1 = None
        getitem_5 = kwargs["x"]
        relation_forward_time_part1_w = self.all_constants.TimePart1
        einsum_6 = torch.functional.einsum(
            "bij,ki->bkj", getitem_5, relation_forward_time_part1_w
        )
        getitem_5 = relation_forward_time_part1_w = None
        size_2 = einsum_6.size(0)
        relation_forward_fir2_weights = self.all_parameters.PFir3W
        size_3 = relation_forward_fir2_weights.size(1)
        squeeze_1 = einsum_6.squeeze(-1)
        einsum_6 = None
        matmul_1 = torch.matmul(squeeze_1, relation_forward_fir2_weights)
        squeeze_1 = relation_forward_fir2_weights = None
        to_1 = matmul_1.to(dtype=torch.float32)
        matmul_1 = None
        view_1 = to_1.view(size_2, 1, size_3)
        to_1 = size_2 = size_3 = None
        add_3 = view_1 + view
        view_1 = view = None
        getitem_6 = kwargs["x_t"]
        kwargs = None
        relation_forward_sample_part8_w = self.all_constants.SamplePart8
        einsum_7 = torch.functional.einsum(
            "bij,ki->bkj", getitem_6, relation_forward_sample_part8_w
        )
        getitem_6 = relation_forward_sample_part8_w = None
        return (
            {"F_PID": add_2, "x_n": add_3},
            {"SamplePart8": einsum_7, "Add6": add_3},
            {"Sub13_int13": add},
            {"Sub13_int14": update_state},
        )


class RecurrentModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Cell = TracerModel()
        self.inputs = [
            "x_m",
            "x_t",
            "F",
            "x",
        ]
        self.states = dict()

    def forward(self, kwargs):
        n_samples = min([kwargs[key].size(0) for key in self.inputs])
        self.states["Sub13_int14"] = kwargs["Sub13_int14"]
        self.states["Sub13_int13"] = kwargs["Sub13_int13"]
        results = {
            "F_PID": [],
            "x_n": [],
        }
        X = dict()
        for idx in range(n_samples):
            for key in self.inputs:
                X[key] = kwargs[key][idx]
            for key, value in self.states.items():
                X[key] = value
            out, _, closed_loop, connect = self.Cell(X)
            for key, value in results.items():
                results[key].append(out[key])
            for key, val in closed_loop.items():
                self.states[key] = nnodely_basic_model_timeshift(self.states[key])
                self.states[key] = nnodely_basic_model_update_state(
                    self.states[key], val
                )
            for key, val in connect.items():
                self.states[key] = nnodely_basic_model_timeshift(val)
        return results
