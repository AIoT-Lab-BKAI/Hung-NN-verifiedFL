from utils.fmodule import FModule
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from utils.FIM2 import inverse, get_module_from_model
from torch.utils.data import DataLoader


hidden = 4096  

total_A0 = None
total_A1 = None
total_A2 = None

total_G1 = None
total_G2 = None
total_G3 = None

total_dw1 = None
total_dw2 = None
total_dw3 = None

class MLPv3(FModule):
    def __init__(self, bias=False):
        super().__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, hidden, bias=bias)
        self.fc2 = nn.Linear(hidden, hidden, bias=bias)
        self.fc3 = nn.Linear(hidden, 10, bias=bias)
    
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    

class MLP3(MLPv3):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.a0 = x.view(x.shape[0], -1)
        self.s1 = self.fc1(self.a0)
        self.a1 = F.relu(self.s1)
        self.s2 = self.fc2(self.a1)
        self.a2 = F.relu(self.s2)
        self.s3 = self.fc3(self.a2)
        self.FIM_params = [self.s1, self.s2, self.s3]
        return self.s3
        

def compute_grads(model:MLP3, X:torch.Tensor, Y:torch.Tensor, loss_fn=torch.nn.KLDivLoss(reduction='batchmean')):
    global total_A0, total_A1, total_A2, total_G1, total_G2, total_G3, total_dw1, total_dw2, total_dw3
    
    pred = model(X)
    loss = loss_fn(F.log_softmax(pred, dim=1), F.softmax(F.one_hot(Y, 10) * 1.0, dim=1))
    
    grads = torch.autograd.grad(loss, [*model.FIM_params, *model.parameters()], create_graph=False, retain_graph=False)
    g1, g2, g3, dw1, dw2, dw3 = grads
    a0, a1, a2 = model.a0, model.a1, model.a2
    
    A0 = torch.sum(a0.unsqueeze(2) @ a0.unsqueeze(2).transpose(1,2), dim=0)
    total_A0 = A0 if total_A0 is None else total_A0 + A0
    
    A1 = torch.sum(a1.unsqueeze(2) @ a1.unsqueeze(2).transpose(1,2), dim=0)
    total_A1 = A1 if total_A1 is None else total_A1 + A1
    
    A2 = torch.sum(a2.unsqueeze(2) @ a2.unsqueeze(2).transpose(1,2), dim=0)
    total_A2 = A2 if total_A2 is None else total_A2 + A2
    
    G1 = torch.sum(g1.unsqueeze(2) @ g1.unsqueeze(2).transpose(1,2), dim=0)
    total_G1 = G1 if total_G1 is None else total_G1 + G1
    
    G2 = torch.sum(g2.unsqueeze(2) @ g2.unsqueeze(2).transpose(1,2), dim=0)
    total_G2 = G2 if total_G2 is None else total_G2 + G2
    
    G3 = torch.sum(g3.unsqueeze(2) @ g3.unsqueeze(2).transpose(1,2), dim=0)
    total_G3 = G3 if total_G3 is None else total_G3 + G3
    
    total_dw1 = dw1 if total_dw1 is None else total_dw1 + dw1
    total_dw2 = dw2 if total_dw2 is None else total_dw2 + dw2
    total_dw3 = dw3 if total_dw3 is None else total_dw3 + dw3
    

def compute_natural_grads(total_data:int):
    global total_A0, total_A1, total_A2, total_G1, total_G2, total_G3, total_dw1, total_dw2, total_dw3
    
    natural_grads = [
        inverse(total_G1/total_data) @ (total_dw1/total_data) @ inverse(total_A0/total_data),
        inverse(total_G2/total_data) @ (total_dw2/total_data) @ inverse(total_A1/total_data),
        inverse(total_G3/total_data) @ (total_dw3/total_data) @ inverse(total_A2/total_data),
    ]
    
    return natural_grads


def step(centroid, nat_grads, lr = 1.0):
    res = MLP3()
    res_modules = get_module_from_model(res)
    cen_modules = get_module_from_model(centroid)
    with torch.no_grad():
        for r_module, c_module, nat_grad in zip(res_modules, cen_modules, nat_grads):
            r_module._parameters['weight'].copy_(c_module._parameters['weight'] - lr * nat_grad)
    return res


def FIM3_step(centroid, clients_training_dataset, client_id_list, eta=0.1, device='cuda'):
    print("FIM3: Perform one step natural gradient with Fisher Matrix... ", end="")
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

