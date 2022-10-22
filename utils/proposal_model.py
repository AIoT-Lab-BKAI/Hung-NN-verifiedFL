import torch
import torch.nn.functional as F
from torch import nn
from utils.fmodule import FModule

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
        self.mg_fc1 = nn.Linear(mid_dim, 100)
        self.mg_fc2 = nn.Linear(100, output_dim)
        self.apply(init_weights)
    
    def forward(self, x):
        r_x = self.encoder(x).detach()
        l_x = self.decoder(r_x)
        dm_x = self.mask_diagonal_regenerator(r_x).detach()
        m_x = torch.diag_embed(dm_x)
        suro_l_x = self.surogate_logits(l_x, m_x)
        output = F.log_softmax(suro_l_x, dim=1)
        return output
    
    def mask_diagonal(self, x):
        """
        This function returns the mask's diagonal vector of x
        """
        r_x = self.encoder(x).detach()
        dm_x = self.mask_diagonal_regenerator(r_x)
        return dm_x 
    
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
    
    def mask_diagonal_regenerator(self, r_x):
        """
        This function generate a mask's diagonal vector for each element in r_x,
        returning shape of b x 10
        """
        dm_x = F.leaky_relu(self.mg_fc1(r_x))
        dm_x = torch.relu(self.mg_fc2(dm_x))
        dm_x = dm_x.view(r_x.shape[0], 10)
        return dm_x
        
    def surogate_logits(self, l_x, m_x):
        """
        Args:
            l_x     : b x 10
            m_x     : b x 10 x 10
        
        This function perform dot multiplication of m_x @ m_x and l_x,
        returning the matrix of shape b x 10
        """
        m_x = m_x @ m_x.transpose(1,2)
        l_x_suro = m_x @ l_x.unsqueeze(2)
        return l_x_suro.squeeze(2)