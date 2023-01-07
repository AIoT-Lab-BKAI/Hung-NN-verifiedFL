from utils.fmodule import FModule
import torch.nn as nn
import torch.nn.functional as F
import torch
from FIM2 import get_module_from_model, flatten_model, inverse


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
        self.s3 = self.fc2(self.a2)
        self.FIM_params = [self.s1, self.s2, self.s3]
        return self.s3
    

def compute_grads(model:MLP3, X:torch.Tensor, Y:torch.Tensor, loss_fn=torch.nn.KLDivLoss(reduction='batchmean')):
    pred = model(X)
    loss = loss_fn(F.softmax(pred, dim=1), F.softmax(F.one_hot(Y, 10) * 1.0, dim=1))
    
    grads = torch.autograd.grad(loss, [*model.FIM_params, *model.parameters()], create_graph=False, retain_graph=False)
    g1, g2, g3, dw1, dw2, dw3 = grads
    a0, a1, a2 = model.a0, model.a1, model.a2
    
    A0 = torch.mean(a0.unsqueeze(2).cpu() @ a0.unsqueeze(2).cpu().transpose(1,2), dim=0)
    A1 = torch.mean(a1.unsqueeze(2).cpu() @ a1.unsqueeze(2).cpu().transpose(1,2), dim=0)
    A2 = torch.mean(a2.unsqueeze(2).cpu() @ a2.unsqueeze(2).cpu().transpose(1,2), dim=0)
    
    G1 = torch.mean(g1.unsqueeze(2).cpu() @ g1.unsqueeze(2).cpu().transpose(1,2), dim=0)
    G2 = torch.mean(g2.unsqueeze(2).cpu() @ g2.unsqueeze(2).cpu().transpose(1,2), dim=0)
    G3 = torch.mean(g3.unsqueeze(2).cpu() @ g3.unsqueeze(2).cpu().transpose(1,2), dim=0)
    
    return A0, A1, A2, G1, G2, G3, dw1.cpu(), dw2.cpu(), dw3.cpu()


def compute_natural_grads(grads):
    A0, A1, A2, G1, G2, G3, dw1, dw2, dw3 = grads
    
    natural_grads = [
        inverse(G1) @ dw1 @ inverse(A0),
        inverse(G2) @ dw2 @ inverse(A1),
        inverse(G3) @ dw3 @ inverse(A2),
    ]
    
    return natural_grads