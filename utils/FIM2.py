import copy
from utils.fmodule import FModule
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader

def get_module_from_model(model, res = None):
    if res==None: res = []
    ch_names = [item[0] for item in model.named_children()]
    if ch_names==[]:
        if model._parameters:
            res.append(model)
    else:
        for name in ch_names:
            get_module_from_model(model.__getattr__(name), res)
    return res


def flatten_tensors(tensors):
    """
    Reference: https://github.com/facebookresearch/stochastic_gradient_push
    Flatten dense tensors into a contiguous 1D buffer. Assume tensors are of
    same dense type.
    Since inputs are dense, the resulting tensor will be a concatenated 1D
    buffer. Element-wise operation on this buffer will be equivalent to
    operating individually.
    Arguments:
        tensors (Iterable[Tensor]): dense tensors to flatten.
    Returns:
        A 1D buffer containing input tensors.
    """
    if len(tensors) == 1:
        return tensors[0].view(-1).clone()
    flat = torch.cat([t.view(-1) for t in tensors], dim=0)
    return flat


def flatten_model(model):
    ten = torch.cat([flatten_tensors(i) for i in model.parameters()])
    return ten


hidden = 4096 

total_A0 = None
total_A1 = None

total_G1 = None
total_G2 = None

total_dw1 = None
total_dw2 = None


class MLPv2(FModule):
    def __init__(self, bias=False):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, hidden, bias=bias)
        self.fc2 = nn.Linear(hidden, 10, bias=bias)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    

class MLP2(MLPv2):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.a0 = x.view(x.shape[0], -1)
        self.s1 = self.fc1(self.a0)
        self.a1 = F.relu(self.s1)
        self.s2 = self.fc2(self.a1)
        self.FIM_params = [self.s1, self.s2]
        return self.s2
    

def inverse(input: torch.Tensor):
    EYE = torch.eye(input.shape[0]).to(input.device)
    try:
        res = torch.linalg.inv(input + EYE * 1e-2)
    except:
        try:
            res = torch.linalg.inv(input + EYE * 1e-1)
        except:
            try:
                res = torch.linalg.inv(input + EYE * 1e-0)
            except:
                raise Exception("Can not invert")
    return res


def compute_grads(model:MLP2, X:torch.Tensor, Y:torch.Tensor, loss_fn=torch.nn.KLDivLoss(reduction='batchmean'), device='cuda'):
    global total_A0, total_A1, total_G1, total_G2, total_dw1, total_dw2
    
    X, Y = X.to(device), Y.to(device)
    
    pred = model(X)
    loss = loss_fn(F.log_softmax(pred, dim=1), F.softmax(F.one_hot(Y, 10) * 1.0, dim=1))    
    
    grads = torch.autograd.grad(loss, [*model.FIM_params, *model.parameters()], create_graph=False, retain_graph=False)
    g1, g2, dw1, dw2 = grads
    a0, a1 = model.a0, model.a1
    
    A0 = torch.sum(a0.unsqueeze(2) @ a0.unsqueeze(2).transpose(1,2), dim=0)
    total_A0 = A0 if total_A0 is None else total_A0 + A0
    
    A1 = torch.sum(a1.unsqueeze(2) @ a1.unsqueeze(2).transpose(1,2), dim=0)
    total_A1 = A1 if total_A1 is None else total_A1 + A1
    
    G1 = torch.sum(g1.unsqueeze(2) @ g1.unsqueeze(2).transpose(1,2), dim=0)
    total_G1 = G1 if total_G1 is None else total_G1 + G1
    
    G2 = torch.sum(g2.unsqueeze(2) @ g2.unsqueeze(2).transpose(1,2), dim=0)
    total_G2 = G2 if total_G2 is None else total_G2 + G2
    
    total_dw1 = dw1 if total_dw1 is None else total_dw1 + dw1
    total_dw2 = dw2 if total_dw2 is None else total_dw2 + dw2
    
    

def compute_natural_grads(total_data:int):
    global total_A0, total_A1, total_G1, total_G2, total_dw1, total_dw2
    
    natural_grads = [
        inverse(total_G1/total_data) @ (total_dw1/total_data) @ inverse(total_A0/total_data),
        inverse(total_G2/total_data) @ (total_dw2/total_data) @ inverse(total_A1/total_data),
    ]
    
    return natural_grads


def step(centroid, nat_grads, lr = 1.0, device='cuda'):
    res = MLP2()
    res_modules = get_module_from_model(res.to(device))
    cen_modules = get_module_from_model(centroid.to(device))
    with torch.no_grad():
        for r_module, c_module, nat_grad in zip(res_modules, cen_modules, nat_grads):
            r_module._parameters['weight'].copy_(c_module._parameters['weight'] - lr * nat_grad)
    return res


def FIM2_step(centroid, clients_training_dataset, client_id_list, eta=1.0, device='cuda'):
    print("FIM2: Perform one step natural gradient with Fisher Matrix... ", end="")
    # Each client compute grads using their own dataset but the centroid's model
    total_data = 0
    for client_id in client_id_list:
        # Training process
        my_training_dataset = clients_training_dataset[client_id]
        train_dataloader = DataLoader(my_training_dataset, batch_size=len(my_training_dataset), shuffle=True, drop_last=False)
        
        X, y = next(iter(train_dataloader))
        X, y = X.to(device), y.to(device)
        compute_grads(centroid, X, y)
        total_data += len(my_training_dataset)
    
    # Server compute natural gradients
    nat_grads = compute_natural_grads(total_data)
    
    # Server update the model
    global_model = step(centroid, nat_grads, lr=eta)
    print("Done!")   
    return global_model