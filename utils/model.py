import torch.nn.functional as F
from torch import nn
import torch
from utils.fmodule import FModule

def batch_similarity(a, b):
    """
    Args:
        a of shape (x, y)
        b of shape (z, y)
    return:
        c = sim (a, b) of shape (x, z)
    """
    psi_x =  (a @ b.T)/ (torch.norm(a, dim=1, keepdim=True) @ torch.norm(b, dim=1, keepdim=True).T)
    psi_x = (psi_x - torch.mean(psi_x, dim=1, keepdim=True)) / torch.std(psi_x, dim=1, keepdim=True)
    psi_x = torch.softmax(psi_x, dim=1)
    return psi_x


def batch_euclidean(a, b):
    """
    Args:
        a of shape (x, y)
        b of shape (z, y)
    return:
        c = sim (a, b) of shape (x, z)
    """
    z = a.unsqueeze(1) - b.unsqueeze(1).transpose(0,1)
    z = torch.pow(torch.norm(z, dim=2),2)/(z.shape[0] * z.shape[1]) + 0.01
    return z
    

class NeuralNetwork(FModule):
    def __init__(self, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, 10, bias=bias)
        self.lowrank_mtx = None
        self.Phi = None
        self.Psi = None
        return
    
    def forward(self, x, true_psi=None):
        r_x = self.encoder(x)
        return self.masking(r_x, true_psi=true_psi)
    
    def pred_and_rep(self, x):
        r_x = self.encoder(x)
        output = self.masking(r_x)
        return output, r_x
    
    def encoder(self, x):
        x = x.view((x.shape[0],28,28))
        x = x.unsqueeze(1)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        return x
    
    def masking(self, r_x, true_psi=None):
        logits = self.fc2(r_x).unsqueeze(2)
        mask = None
        if self.lowrank_mtx is not None:
            mask = (self.lowrank_mtx @ self.lowrank_mtx).unsqueeze(0)
        else:
            b = r_x.shape[0]
            if true_psi is None:
                psi_x = batch_similarity(r_x.detach(), self.Psi)
                min = torch.min(psi_x, dim=1, keepdim=True).values
                max = torch.max(psi_x, dim=1, keepdim=True).values
                psi_x = (psi_x - min)/(max - min)
                psi_x = psi_x/torch.sum(psi_x, dim=1, keepdim=True)
            else:
                psi_x = true_psi
            
            mask = (self.Phi.view(self.Phi.shape[0], -1).T @ psi_x.unsqueeze(2)).view(b, 10, 10)
        return (mask.to("cuda" if logits.is_cuda else "cpu").to(torch.float32) @ logits).squeeze(2)

    def update_mask(self, mask):
        """
        Note: For client only
        """
        self.lowrank_mtx = mask
        return
    
    def prepare(self, Phi, Psi):
        self.Psi = Psi
        self.Phi = Phi
        self.lowrank_mtx = None
        return
    

class DNN(FModule):
    def __init__(self, input_dim = 784, mid_dim = 100, output_dim = 10, bias=False):
        super(DNN, self).__init__()
        # define network layers
        self.fc1 = nn.Linear(input_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim, output_dim, bias=bias)
        self.lowrank_mtx = None
        self.Phi = None
        self.Psi = None
        
    def forward(self, x, true_psi=None):
        r_x = self.encoder(x)
        output = self.masking(r_x, true_psi=true_psi)
        output = F.log_softmax(output, dim=1)
        return output
    
    def encoder(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return x
    
    def decoder(self, r_x):
        x = self.fc2(r_x)
        x = F.log_softmax(x, dim=1)
        return x
    
    def pred_and_rep(self, x):
        r_x = self.encoder(x)
        output = self.masking(r_x)
        output = F.log_softmax(output, dim=1)
        return output, r_x
    
    def masking(self, r_x, true_psi=None):
        logits = self.fc2(r_x).unsqueeze(2)
        mask = None
        if self.lowrank_mtx is not None:
            mask = (self.lowrank_mtx @ self.lowrank_mtx).unsqueeze(0)
        else:
            b = r_x.shape[0]
            if true_psi is None:
                psi_x = batch_similarity(r_x.detach(), self.Psi)
                min = torch.min(psi_x, dim=1, keepdim=True).values
                max = torch.max(psi_x, dim=1, keepdim=True).values
                psi_x = (psi_x - min)/(max - min)
                psi_x = psi_x/torch.sum(psi_x, dim=1, keepdim=True)
            else:
                psi_x = true_psi
            
            mask = (self.Phi.view(self.Phi.shape[0], -1).T @ psi_x.unsqueeze(2)).view(b, 10, 10)
        return (mask.to("cuda" if logits.is_cuda else "cpu").to(torch.float32) @ logits).squeeze(2)

    def update_mask(self, mask):
        """
        Note: For client only
        """
        self.lowrank_mtx = mask
        return
    
    def prepare(self, Phi, Psi):
        self.Psi = Psi
        self.Phi = Phi
        self.lowrank_mtx = None
        return
    
class DNN2(DNN):
    def __init__(self, input_dim=784, mid_dim=100, output_dim=10, bias=False):
        super().__init__(input_dim, mid_dim, output_dim, bias)
        
    def masking(self, r_x, true_psi=None):
        logits = self.fc2(r_x).unsqueeze(2)
        mask = None
        if self.lowrank_mtx is not None:
            mask = (self.lowrank_mtx @ self.lowrank_mtx).unsqueeze(0)
        else:
            b = r_x.shape[0]
            if true_psi is None:
                distance = batch_euclidean(r_x.detach(), self.Psi)
                psi_x = 1/distance
                psi_x = psi_x/torch.sum(psi_x, dim=1, keepdim=True)
                psi_x = torch.argmax(psi_x, dim=1, keepdim=True)
                psi_x = F.one_hot(psi_x.squeeze(), num_classes=self.Psi.shape[0]) * 1.0
            else:
                psi_x = true_psi
            
            mask = (self.Phi.view(self.Phi.shape[0], -1).T @ psi_x.unsqueeze(2)).view(b, 10, 10)
        return (mask.to("cuda" if logits.is_cuda else "cpu").to(torch.float32) @ logits).squeeze(2)