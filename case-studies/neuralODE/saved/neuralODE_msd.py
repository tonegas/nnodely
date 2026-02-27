import torch

def nnodely_basic_model_update_state(data_in, rel):
    data_out = data_in.clone()
    max_dim = min(rel.size(1), data_in.size(1))
    data_out[:, -max_dim:, :] = rel[:, -max_dim:, :]
    return data_out

def nnodely_basic_model_timeshift(data_in):
    return torch.cat((data_in[:, 1:, :], data_in[:, :1, :]), dim=1)

def nnodely_layers_neuralODE_FNeuralODE5(state, *weights):
    from nnodely.support.odeint.adjoint import odeint_adjoint as odeint
    def ode_func_torch(t, state, weight_fir):
        import torch
        import torch.nn.functional as F
        # state: (B, W, 2)
        #   B = batch size
        #   W = window length
        #   2 channels = [position_window, force_window]
        #
        # weight_fir: FIR weights applied independently to each channel
    
        # Apply two independent FIR filters (grouped convolution):
        # this produces one-step predictions for each channel
        # (B, W, 2) -> (B, 2, W) -> conv -> (B, 2, 1)
        out = F.conv1d(state.transpose(1, 2), weight_fir, groups=2)
    
        # Restore time dimension ordering
        # (B, 2, 1) -> (B, 1, 2)
        out = out.transpose(1, 2)
    
        # Combine the two FIR contributions:
        # next_velocity = free_response + forced_response
        out[:, :, 0] = out[:, :, 0] + out[:, :, -1]
    
        # Set the force prediction to zero, for the intermediate time steps done in the integration process (if a variable step integrator, such as Dopri5, is used)
        out[:, :, -1] = 0.0
    
        # Shift the sliding window forward by one step
        # drop the oldest sample and append the new prediction
        # resulting shape: (B, W, 2)
        out = torch.cat((state[:, 1:, :], out), dim=1)
    
        return out
    
    ans = odeint(lambda t, y: ode_func_torch(t, y, *weights), state, t=torch.tensor([0.0, 0.01]), rtol=1e-07, atol=1e-09, method='dopri5', adjoint_params=list(weights))
    return ans[-1]

class TracerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.all_parameters = {}
        self.all_constants = {}
        self.all_parameters["weight_fir"] = torch.nn.Parameter(torch.tensor([[[-5.0214056968688965, -4.025257587432861, -3.0458261966705322, -1.9209206104278564, -0.7650308012962341, 0.5171442031860352, 1.7535150051116943, 3.087076187133789, 4.083190441131592, 5.219380855560303]], [[0.00026014805189333856, 0.0007117472123354673, 0.0013745456235483289, 0.002709923079237342, 0.003882204182446003, 0.005919742863625288, 0.0070823198184370995, 0.008177743293344975, 0.009529106318950653, 0.008549299091100693]]]), requires_grad=True)
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
