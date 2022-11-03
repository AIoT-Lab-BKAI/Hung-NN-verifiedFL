from utils.elements import  *

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