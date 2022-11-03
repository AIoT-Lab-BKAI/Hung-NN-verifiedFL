import torch
import torch.nn.functional as F
from torch import nn
from utils.fmodule import FModule, get_module_from_model
import utils.fmodule as fmodule
import numpy as np

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
        
def get_ultimate_layer(model: nn.Module):
    penul = get_module_from_model(model)[-1]._parameters['weight']
    return penul

class FeatureExtractor(FModule):
    def __init__(self, input_dim = 784, output_dim = 100):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.apply(init_weights)
        
    def forward(self, x):
        """
        This function returns the representation of x
        """
        r_x = torch.flatten(x, 1)
        r_x = torch.sigmoid(self.fc1(r_x))
        return r_x
    
class Classifier(FModule): 
    def __init__(self, input_dim = 100, output_dim = 10):
        super().__init__()
        self.fc2 = nn.Linear(input_dim, output_dim)
        self.apply(init_weights)
        
    def forward(self, r_x):
        """
        This function returns the logits of r_x
        """
        l_x = self.fc2(r_x)
        return l_x
    
class MaskGenerator(FModule):
    def __init__(self, input_dim = 100, mid_dim = 128, output_dim = 10):
        super().__init__()
        self.fc3 = nn.Linear(input_dim, mid_dim)
        self.fc4 = nn.Linear(mid_dim, output_dim)
        self.apply(init_weights)
    
    def forward(self, r_x):
        """
        This function generate a mask's diagonal vector for each element in r_x,
        returning shape of b x 10
        """
        dm_x = F.relu(self.fc3(r_x))
        dm_x = torch.softmax(self.fc4(dm_x), dim=1)
        dm_x = dm_x.view(r_x.shape[0], 10)
        return dm_x
    