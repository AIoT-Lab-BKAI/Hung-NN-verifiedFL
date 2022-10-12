from copy import deepcopy
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from utils.dataloader import CustomDataset
from torchvision import datasets, transforms
from torchmetrics import ConfusionMatrix
import json
import os
import argparse
import numpy as np

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

def gen_representation(dataloader, model, condense_representation):
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred, rep = model.pred_and_rep(X)
        condense_representation.retain_grad()
        sim = batch_similarity(rep.detach(), condense_representation)
        condense_represent_loss = torch.sum(1 - sim)
        condense_represent_loss.backward()
        condense_representation = condense_representation - 5 * condense_representation.grad
            
    return condense_representation

def train(dataloader, model, loss_fn, optimizer, fct=0.01):
    base_model = deepcopy(model).cuda()
    base_model.freeze_grad()
    
    model = model.cuda()
    model.train()
    losses = []
    
    # mse = torch.nn.MSELoss()
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred, rep = model.pred_and_rep(X)
        _, rep_base = base_model.pred_and_rep(X)
        
        loss = loss_fn(pred, y) - fct * torch.sum(torch.diag((batch_similarity(rep, rep_base))))
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        
    return losses

def test(model, testing_data):
    test_loader = DataLoader(testing_data, batch_size=32, shuffle=True, drop_last=False)
    model.cuda()

    loss_fn = torch.nn.CrossEntropyLoss()
    device = "cuda"

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

def test_ideal(model, testing_data):
    test_loader = DataLoader(testing_data, batch_size=32, shuffle=True, drop_last=False)
    model.cuda()

    loss_fn = torch.nn.CrossEntropyLoss()
    device = "cuda"

    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    model.eval()
    test_loss, correct = 0, 0
    confmat = ConfusionMatrix(num_classes=10).to(device)
    cmtx = 0
    
    label_list = [[1,2],[3,4],[5,6],[7,8],[9,0]]
    def recoord(labels):
        coord = []
        for label in labels:
            for idx in range(len(label_list)):
                if label in label_list[idx]:
                    coord.append(idx)
        true_psi = torch.zeros([len(labels), len(label_list)])
        """
        coord = [0,1,2,1,1,3]
        """
        for i in range(len(coord)):
            true_psi[i][coord[i]] = 1
        return true_psi

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            true_psi = recoord(y).to(device)
            pred = model(X, true_psi)
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