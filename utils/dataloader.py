from torch.utils.data import Dataset
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.targets =self.__num_labels()
        
    def __num_labels(self):
        labels = []
        for i in self.idxs:
            label = self.dataset.targets[i]
            labels.append(label)
        return labels

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label