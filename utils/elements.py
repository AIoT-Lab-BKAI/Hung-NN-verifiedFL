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
    
class ProposedNet(FModule):
    def __init__(self, feature_extractor=None, classifier=None, mask_generator=None):
        super().__init__()
        self.feature_extractor = FeatureExtractor() if feature_extractor is None else feature_extractor
        self.classifier = Classifier() if classifier is None else classifier
        self.mask_generator = MaskGenerator() if mask_generator is None else mask_generator
        
    def  __call__(self, x, original_mask_diagonal=None):
        return self.forward(x, original_mask_diagonal)
    
    def forward(self, x, original_mask_diagonal=None):
        r_x = self.feature_extractor(x).detach()
        dm_x = self.mask_generator(r_x).detach()
        dm_x = (dm_x > 1/10) * 1.
        m_x = torch.diag_embed(dm_x)
        l_x = self.classifier(r_x)
        
        if original_mask_diagonal is None:
            """ When inference """
            suro_l_x = l_x
        else:
            """ When training """
            suro_l_x = self.surogate_logits(l_x, torch.diag_embed(original_mask_diagonal))
        
        mirr_suro_l_x = self.mirror_surogate_logits(suro_l_x, m_x)
        output = mirr_suro_l_x
        return output
    
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
    
    @staticmethod
    def aggregate_fe(feature_extractors, impact_factors, device):
        feature_extractors = [fe.to(device) for fe in feature_extractors]
        final_fe = fmodule._model_sum([fe * p_k for fe, p_k in zip(feature_extractors, impact_factors)])
        return final_fe
    
    @staticmethod
    def aggregate_cl(augmented_classifiers, impact_factors, device):
        """
        This method aggregates augmented classifiers
        Meaning, these classifiers have been augmented by: 
            aug_cl = mask @ original_cl
        """
        augmented_classifiers = [cl.to(device) for cl in augmented_classifiers]
        final_cl = fmodule._model_sum([cl * p_k for cl, p_k in zip(augmented_classifiers, impact_factors)])
        return final_cl
    
    @staticmethod
    def aggregate_mg(global_mask_generator: MaskGenerator, mask_generators, local_representation_storage, epochs=8, device="cuda"):
        """
        This method distills knowledge from each mg in mask_generators
        to the aggregated model
        Args:
            mask_generators: list of Mask generator
            local_representation_storage: dictionary
                {
                    client_id: [(rep_batch1, mask1), (rep_batch2, mask2), ...]
                }
        Note:
            The mask generators must map to the corresponding storage
        """
        mask_generators = [mg.to(device) for mg in mask_generators]
        global_mask_generator = global_mask_generator.to(device)
        
        optimizer = torch.optim.Adam(global_mask_generator.parameters(), lr=1e-3)
        
        total_transfer_data = []
        for client_id in local_representation_storage.keys():
            transfer_data = local_representation_storage[client_id]
            total_transfer_data += transfer_data
                
        for e in range(epochs):
            losses = []
            for rep, mask in total_transfer_data:
                rep, mask = rep.to(device), mask.to(device)
                student_mask = global_mask_generator(rep)
                loss = torch.sum(torch.pow(student_mask - mask, 2))/student_mask.shape[0]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                student_mask = (student_mask > 1/10) * 1.0
                loss = torch.sum(torch.abs(student_mask - mask))/student_mask.shape[0]
                losses.append(loss.item())
                
            print(f"\t   Server mask transferring epoch {e}, Avg. loss {np.mean(losses):>.3f}")
    
        return global_mask_generator
    

@torch.no_grad()
def augmented_classifier(classifier: Classifier, original_mask: torch.Tensor, scale: float, device="cuda:1"):
    original_mask = original_mask.to(device)
    classifier = classifier.to(device)
    
    classifier.fc2.weight.copy_(original_mask @ classifier.fc2.weight)
    classifier.fc2.weight.mul_(1/scale)
    return classifier