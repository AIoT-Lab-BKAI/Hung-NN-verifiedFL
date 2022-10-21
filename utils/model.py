import torch
import torch.nn.functional as F
from torch import nn
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
    z = torch.pow(torch.norm(z, dim=2),2)/(z.shape[0] * z.shape[1])
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
                # min = torch.min(psi_x, dim=1, keepdim=True).values
                # max = torch.max(psi_x, dim=1, keepdim=True).values
                # psi_x = (psi_x - min)/(max - min)
                # psi_x = psi_x/torch.sum(psi_x, dim=1, keepdim=True)
                psi_x = torch.argmax(psi_x, dim=1, keepdim=True)
                psi_x = F.one_hot(psi_x.squeeze(), num_classes=self.Psi.shape[0]) * 1.0
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
    
    def encoder(self, x):
        x = torch.flatten(x, 1)
        x = F.sigmoid(self.fc1(x))
        return x
    
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


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
        
class DNN_proposal(FModule):
    def __init__(self, input_dim = 784, mid_dim = 100, output_dim = 10):
        super().__init__()
        # define network layers
        self.fc1 = nn.Linear(input_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim, output_dim)
        # mask regenerator
        self.mg_fc1 = nn.Linear(mid_dim, 256)
        self.mg_fc2 = nn.Linear(256, output_dim ** 2)
        self.apply(init_weights)
    
    def __call__(self, x, mask=None):
        return self.forward(x, mask=mask)
    
    def forward(self, x, mask=None):
        r_x = self.encoder(x)
        l_x = self.decoder(r_x)
        m_x = self.mask_regenerator(r_x.detach())
        if mask is None:
            suro_l_x = l_x
        else:
            suro_l_x = self.surogate_logits(l_x, mask)
        mirr_suro_l_x = self.mirror_surogate_logits(suro_l_x, m_x.detach())
        output = F.log_softmax(mirr_suro_l_x, dim=1)
        return output
    
    def mask(self, x):
        """
        This function returns the mask of x
        """
        r_x = self.encoder(x)
        m_x = self.mask_regenerator(r_x.detach())
        return m_x 
    
    def encoder(self, x):
        """
        This function returns the representation of x
        """
        r_x = torch.flatten(x, 1)
        r_x = F.relu(self.fc1(r_x))
        return r_x
    
    def decoder(self, r_x):
        """
        This function returns the logits of r_x
        """
        l_x = self.fc2(r_x)
        return l_x
    
    def mask_regenerator(self, r_x):
        """
        This function generate a mask for each element in r_x,
        returning shape of b x 10 x 10
        """
        m_x = F.leaky_relu(self.mg_fc1(r_x))
        m_x = torch.sigmoid(self.mg_fc2(m_x))
        m_x = m_x.view(r_x.shape[0], 10, 10)
        return m_x
    
    def surogate_logits(self, l_x, mask):
        """
        Args:
            l_x     : b x 10
            mask    : 10 x 10
        
        This function return the logits that are masked,
        the returning shape b x 10
        """
        l_x = l_x.unsqueeze(2)
        l_x_suro = (mask * 1.0) @ l_x
        return l_x_suro.squeeze(2)
    
    def mirror_surogate_logits(self, l_x_suro, m_x):
        """
        Args:
            l_x_suro: b x 10
            m_x     : b x 10 x 10
        
        This function perform dot multiplication of m_x and l_x_suro,
        returning the matrix of shape b x 10
        """
        l_x_mirr_suro = m_x @ l_x_suro.unsqueeze(2)
        return l_x_mirr_suro.squeeze(2)
    
    def freeze_feature_extractor(self):
        self.fc1.requires_grad = False
        return