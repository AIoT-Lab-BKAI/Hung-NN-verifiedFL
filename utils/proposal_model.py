import torch
import torch.nn.functional as F
from torch import nn
from utils.fmodule import FModule, get_module_from_model
import utils.fmodule as fmodule

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
        self.mg_fc1 = nn.Linear(mid_dim, 128)
        self.mg_fc2 = nn.Linear(128, output_dim)
        self.apply(init_weights)
        
    def  __call__(self, x, original_mask_diagonal=None):            
        return self.forward(x, original_mask_diagonal)
    
    def forward(self, x, original_mask_diagonal=None):
        r_x = self.encoder(x).detach()
        l_x = self.decoder(r_x)
        dm_x = self.mask_diagonal_regenerator(r_x).detach()
        dm_x = (dm_x > 1/10) * 1.
        m_x = torch.diag_embed(dm_x)
        
        if original_mask_diagonal is None:
            """ When inference """
            suro_l_x = l_x
        else:
            """ When training """
            suro_l_x = self.surogate_logits(l_x, torch.diag_embed(original_mask_diagonal))
        
        mirr_suro_l_x = self.mirror_surogate_logits(suro_l_x, m_x)
        # output = F.log_softmax(mirr_suro_l_x, dim=1)
        output = mirr_suro_l_x
        return output
    
    def encoder(self, x):
        """
        This function returns the representation of x
        """
        r_x = torch.flatten(x, 1)
        r_x = torch.sigmoid(self.fc1(r_x))
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
        dm_x = F.relu(self.mg_fc1(r_x))
        dm_x = torch.softmax(self.mg_fc2(dm_x), dim=1)
        dm_x = dm_x.view(r_x.shape[0], 10)
        return dm_x
           
    def surogate_logits(self, l_x, original_mask):
        """
        Args:
            l_x             : b x 10
            original_mask   : 10 x 10
        
        This function return the logits that are masked,
        the returning shape b x 10
        """
        l_x = l_x.unsqueeze(2)
        suro_l_x = (original_mask * 1.0) @ l_x
        return suro_l_x.squeeze(2)
    
    def mirror_surogate_logits(self, suro_l_x, m_x):
        """
        Args:
            suro_l_x: b x 10
            m_x     : b x 10 x 10
        
        This function perform dot multiplication of m_x and suro_l_x,
        returning the matrix of shape b x 10
        """
        mirr_suro_l_x = m_x @ suro_l_x.unsqueeze(2)
        return mirr_suro_l_x.squeeze(2)
    
    
def get_ultimate_layer(model: nn.Module):
    penul = get_module_from_model(model)[-1]._parameters['weight']
    return penul


@torch.no_grad()
def augment_model(model: DNN_proposal, original_mask: torch.Tensor, scale: float, device="cuda:1"):
    original_mask = original_mask.to(device)
    model = model.to(device)
    
    classifier = get_ultimate_layer(model)
    classifier.copy_(original_mask @ classifier)
    classifier.mul_(1/scale)
    return model