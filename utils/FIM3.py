from utils.fmodule import FModule
import torch.nn as nn
import torch.nn.functional as F
import torch
from utils.FIM2 import inverse, get_module_from_model
from torch.utils.data import DataLoader


class MLP3(FModule):
    def __init__(self, bias=False):
        super().__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 4096, bias=bias)
        self.fc2 = nn.Linear(4096, 4096, bias=bias)
        self.fc3 = nn.Linear(4096, 10, bias=bias)

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
    pred = model(X)
    loss = loss_fn(F.softmax(pred, dim=1), F.softmax(F.one_hot(Y, 10) * 1.0, dim=1))
    
    grads = torch.autograd.grad(loss, [*model.FIM_params, *model.parameters()], create_graph=False, retain_graph=False)
    g1, g2, g3, dw1, dw2, dw3 = grads
    a0, a1, a2 = model.a0, model.a1, model.a2
    
    A0 = torch.sum(a0.unsqueeze(2).cpu() @ a0.unsqueeze(2).cpu().transpose(1,2), dim=0)
    A1 = torch.sum(a1.unsqueeze(2).cpu() @ a1.unsqueeze(2).cpu().transpose(1,2), dim=0)
    A2 = torch.sum(a2.unsqueeze(2).cpu() @ a2.unsqueeze(2).cpu().transpose(1,2), dim=0)
    
    G1 = torch.sum(g1.unsqueeze(2).cpu() @ g1.unsqueeze(2).cpu().transpose(1,2), dim=0)
    G2 = torch.sum(g2.unsqueeze(2).cpu() @ g2.unsqueeze(2).cpu().transpose(1,2), dim=0)
    G3 = torch.sum(g3.unsqueeze(2).cpu() @ g3.unsqueeze(2).cpu().transpose(1,2), dim=0)
    
    return A0, A1, A2, G1, G2, G3, dw1.cpu(), dw2.cpu(), dw3.cpu()


def compute_natural_grads(grads):
    A0, A1, A2, G1, G2, G3, dw1, dw2, dw3 = grads
    
    natural_grads = [
        inverse(G1) @ dw1 @ inverse(A0),
        inverse(G2) @ dw2 @ inverse(A1),
        inverse(G3) @ dw3 @ inverse(A2),
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
    print("Perform one step natural gradient with Fisher Matrix... ", end="")
    # Each client compute grads using their own dataset but the centroid's model
    grad_list = []
    total_data = 0
    for client_id in client_id_list:
        # Training process
        my_training_dataset = clients_training_dataset[client_id]
        train_dataloader = DataLoader(my_training_dataset, batch_size=len(my_training_dataset), shuffle=True, drop_last=False)
        
        X, y = next(iter(train_dataloader))
        X, y = X.to(device), y.to(device)
        grad_list.append(compute_grads(centroid, X, y))
        total_data += len(my_training_dataset)
        
    # Server average the gradients
    vgrad = [None for i in range(len(grad_list[0]))]
    for grad in grad_list:
        # Example: grad = (A0, A1, G1, G2, dw1, dw2)
        for i in range(len(grad)):
            vgrad[i] = grad[i] if vgrad[i] is None else vgrad[i] + grad[i]
    
    vgrad = [vg/total_data for vg in vgrad]
    
    # Server compute natural gradients
    nat_grads = compute_natural_grads(vgrad)
    
    # Server update the model
    global_model = step(centroid.cpu(), nat_grads, lr=eta)
    print("Done!")   
    return global_model

