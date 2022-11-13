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
    def __init__(self, output_dim = 512):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(3136, output_dim)
        self.apply(init_weights)
        
    def forward(self, x):
        """
        This function returns the representation of x
        """
        x = x.view((x.shape[0],28,28))
        x = x.unsqueeze(1)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        r_x = F.relu(self.fc1(x))
        # r_x = r_x / torch.norm(r_x, dim=1, keepdim=True)
        return r_x
    
class Classifier(FModule): 
    def __init__(self, input_dim = 512, output_dim = 10):
        super().__init__()
        self.fc2 = nn.Linear(input_dim, output_dim)
        self.apply(init_weights)
        
    def forward(self, r_x):
        """
        This function returns the logits of r_x
        """
        l_x = self.fc2(r_x)
        return l_x
    