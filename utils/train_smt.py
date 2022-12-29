import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix


def test(model, testing_data, device="cuda"):
    test_loader = DataLoader(testing_data, batch_size=32, shuffle=True, drop_last=False)
    model = model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()

    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    model.eval()
    test_loss, correct = 0, 0
    confmat = ConfusionMatrix(num_classes=10).to(device)
    cmtx = 0

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            cmtx += confmat(pred, y)

    test_loss /= num_batches
    correct /= size

    acc, cfmtx =  correct, cmtx.cpu().numpy()
    down = np.sum(cfmtx, axis=1, keepdims=True)
    down[down == 0] = 1
    cfmtx = cfmtx/down
    return acc, cfmtx


@torch.no_grad()
def check_global_contrastive(model, dataset, device):
    model = model.to(device)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=False)
    
    num_classes = len(np.unique(dataset.targets))
    sim_mtx = torch.zeros([num_classes, num_classes])
    frq_mtx = torch.zeros([num_classes, num_classes])
    
    same_class_dis = []
    different_class_dis = []
    
    for e in range(10):
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            rep = model.get_representation(X)
            for i in range(0, rep.shape[0] - 1):
                similarity = rep[i] @ rep[i+1] / (torch.norm(rep[i]) * torch.norm(rep[i+1]))
                
                label_i = y[i].detach().item()
                label_j = y[i+1].detach().item()
                
                sim_mtx[label_i][label_j] += similarity.detach().item()
                sim_mtx[label_j][label_i] += similarity.detach().item()
                
                frq_mtx[label_i][label_j] += 1
                frq_mtx[label_j][label_i] += 1
                
                if label_i == label_j:
                    same_class_dis.append(similarity.detach().item())
                else:
                    different_class_dis.append(similarity.detach().item())

    frq_mtx[frq_mtx == 0] = 1
    sim_mtx = sim_mtx / frq_mtx
    sim_mtx = sim_mtx.numpy()
    
    return np.mean(same_class_dis) if len(same_class_dis) else 0, \
            np.mean(different_class_dis) if len(different_class_dis) else 0, \
                sim_mtx
                
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
    
def print_cfmtx(mtx):
    for i in range(mtx.shape[0]):
        for j in range(mtx.shape[1]):
            if i == j:
                print(f"\033[48;5;225m{mtx[i,j]:>.3f}\033[0;0m", end="  ")
            else:
                print(f"{mtx[i,j]:>.3f}", end="  ")
        print()
    return