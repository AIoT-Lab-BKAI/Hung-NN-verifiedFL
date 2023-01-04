from torch import nn
import torch
import torch.nn.functional as F
from utils.fmodule import FModule


class NeuralNetwork(FModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encoder(self, x):
        x = x.view((x.shape[0],28,28))
        x = x.unsqueeze(1)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        return x

    def decoder(self, x):
        x = self.fc2(x)
        return x
    
    def pred_and_rep(self, x):
        r_x = self.encoder(x)
        output = self.decoder(r_x)
        return output, r_x
    
    def get_representation(self, x):
        return self.encoder(x)
    
    
class MLP(FModule):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return x