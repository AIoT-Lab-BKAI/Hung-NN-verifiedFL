import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix
from torchvision import datasets, transforms
from utils.dataloader import CustomDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

def batch_similarity(a, b):
    """
    Args:
        a of shape (x, y)
        b of shape (z, y)
    return:
        c = sim (a, b) of shape (x, z)
    """
    return (a @ b.T)/ (torch.norm(a, dim=1, keepdim=True) @ torch.norm(b, dim=1, keepdim=True).T)


def contrastive_training(contrastive_loader, model, loss_fn, optimizer, device, contrastive):
    model = model.to(device)
    model.train()
    
    same_class_dis = []
    different_class_dis = []
    classification_loss = []
            
    for batch, (X, y) in enumerate(contrastive_loader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        rep = model.encoder(X)
        rep = rep / torch.norm(rep, dim=1, keepdim=True)
        pred = model.decoder(rep)
        
        simi_loss = 0
        for i in range(0, rep.shape[0] - 1, 2):
            similarity = rep[i] @ rep[i+1]
            if y[i].detach().item() == y[i+1].detach().item():
                simi_loss += 1 - similarity
                same_class_dis.append(similarity.detach().item())
            else:
                simi_loss += similarity
                different_class_dis.append(similarity.detach().item())
        
        loss = loss_fn(pred, y)
        classification_loss.append(loss.detach().item())
        
        if contrastive:
            loss += 0.01 * simi_loss * 2 / rep.shape[0]
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return np.mean(classification_loss), \
        np.mean(same_class_dis) if len(same_class_dis) else 0, \
        np.mean(different_class_dis) if len(different_class_dis) else 0, \
            

def train_representation(dataloader, model, condense_representation):
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred, rep = model.pred_and_rep(X)
        condense_representation.retain_grad()
        sim = batch_similarity(rep.detach(), condense_representation)
        condense_represent_loss = torch.sum(1 - sim)
        condense_represent_loss.backward()
        condense_representation = condense_representation - 5 * condense_representation.grad
            
    return condense_representation


def test(model, testing_data, device="cuda"):
    test_loader = DataLoader(testing_data, batch_size=32, shuffle=True, drop_last=False)
    model.to(device)

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
    return cfmtx


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