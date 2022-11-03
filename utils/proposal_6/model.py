from utils.elements import FeatureExtractor, Classifier, init_weights

import torch
import torch.nn.functional as F
from torch import nn
from utils.fmodule import FModule

class Model(FModule):
    def __init__(self):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.classifier = Classifier()
        self.apply(init_weights)
        
    def forward(self, x):
        r_x = self.feature_extractor(x)
        l_x = self.classifier(r_x)
        return l_x
