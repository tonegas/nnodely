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
        self.all_parameters["weight_fir"] = torch.nn.Parameter(torch.tensor([[[-5.033620357513428, -4.02910852432251, -3.042726755142212, -1.913421869277954, -0.7563386559486389, 0.523976743221283, 1.7564716339111328, 3.0856471061706543, 4.078126907348633, 5.212916374206543]], [[0.0002466370933689177, 0.0008778220508247614, 0.0017402776284143329, 0.002738319570198655, 0.004075315315276384, 0.00543161341920495, 0.007128583267331123, 0.008107611909508705, 0.009506390430033207, 0.008429705165326595]]]), requires_grad=True)
        self.all_constants["SamplePart10"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["Select7"] = torch.tensor([1.0, 0.0], requires_grad=True)
        self.all_constants["TimePart1"] = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], requires_grad=True)
        self.all_constants["TimePart3"] = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], requires_grad=True)
        self.all_constants["TimePart8"] = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], requires_grad=True)
        self.all_parameters = torch.nn.ParameterDict(self.all_parameters)
        self.all_constants = torch.nn.ParameterDict(self.all_constants)

    def update(self, closed_loop={}, connect={}, disconnect=False):
        pass
    
    def forward(self, kwargs):
        getitem = kwargs['x_t']
        relation_forward_sample_part10_w = self.all_constants.SamplePart10
        einsum = torch.functional.einsum('bij,ki->bkj', getitem, relation_forward_sample_part10_w);  getitem = relation_forward_sample_part10_w = None
        getitem_1 = kwargs['x']
        relation_forward_time_part1_w = self.all_constants.TimePart1
        einsum_1 = torch.functional.einsum('bij,ki->bkj', getitem_1, relation_forward_time_part1_w);  getitem_1 = relation_forward_time_part1_w = None
        getitem_2 = kwargs['F'];  kwargs = None
        relation_forward_time_part3_w = self.all_constants.TimePart3
        einsum_2 = torch.functional.einsum('bij,ki->bkj', getitem_2, relation_forward_time_part3_w);  getitem_2 = relation_forward_time_part3_w = None
        cat = torch.cat((einsum_1, einsum_2), dim = 2);  einsum_1 = einsum_2 = None
        all_parameters_weight_fir = self.all_parameters.weight_fir
        fneural_ode5 = nnodely_layers_neuralODE_FNeuralODE5(cat, all_parameters_weight_fir);  cat = all_parameters_weight_fir = None
        relation_forward_select7_w = self.all_constants.Select7
        einsum_3 = torch.functional.einsum('ijk,k->ij', fneural_ode5, relation_forward_select7_w);  fneural_ode5 = relation_forward_select7_w = None
        unsqueeze = einsum_3.unsqueeze(2);  einsum_3 = None
        relation_forward_time_part8_w = self.all_constants.TimePart8
        einsum_4 = torch.functional.einsum('bij,ki->bkj', unsqueeze, relation_forward_time_part8_w);  unsqueeze = relation_forward_time_part8_w = None
        return ({'x_n': einsum_4}, {'SamplePart10': einsum, 'TimePart8': einsum_4}, {'x': einsum_4}, {})
        
class RecurrentModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Cell = TracerModel()
        self.inputs = ['F', 'x_t', ]
        self.states = dict()

    def forward(self, kwargs, n_samples = None):
        n_samples = n_samples if n_samples else min([kwargs[key].size(0) for key in self.inputs])
        self.states['x'] = kwargs['x']
        results = {'x_n':[], }
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
                self.states[key] = nnodely_basic_model_update_state(self.states[key], val)
            for key, val in connect.items():
                self.states[key] = nnodely_basic_model_timeshift(val)
        return results
