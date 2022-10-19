import argparse
import json
import torch
import os
from pathlib import Path
from utils.model import NeuralNetwork, batch_similarity
from utils.fmodule import get_module_from_model
from utils import fmodule
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchmetrics import ConfusionMatrix
import numpy as np


def create_mask(dim, labels):
    mask = torch.zeros([dim, dim])
    for label in labels:
        mask[label, label] = 1
    return mask

def get_ultimate_layer(model):
    penul = get_module_from_model(model)[-1]._parameters['weight']
    return penul

@torch.no_grad()
def classifier_aggregation(final_model, models, masks, indexes, device0):
    """
    Args:
        final_model : the model with the feature extractor already aggregated
        models = [model0, model5, model7, model10]
        masks = [U0, U5, U7, U10]
        indexes = [0, 5, 7, 10]
    """
    base_classifier = get_ultimate_layer(final_model).mul_(0)
    for i in range(len(indexes)):
        client_id = indexes[i]
        client_model = models[i]
        client_mask = masks[client_id].to(device0, dtype=torch.float32)
        base_classifier.add_(client_mask @ client_mask @ get_ultimate_layer(client_model))
    return final_model

def aggregate(client_models, representations, indexes, device="cuda"):
    client_models = [model.to(device) for model in client_models]
    
    impact_factors = 1/len(client_models)
    body_model = fmodule._model_sum([model_k * impact_factors for model_k in client_models])
    
    masks = [model.lowrank_mtx for model in client_models]
    final_model = classifier_aggregation(body_model, client_models, masks, indexes, device)
    
    Phi = torch.stack(masks, dim=0).cuda()
    Psi = representations.cuda()
    
    final_model.prepare(Phi, Psi)
    return final_model

def test(global_model, testing_data):
    test_loader = DataLoader(testing_data, batch_size=32, shuffle=True, drop_last=False)
    global_model.cuda()

    loss_fn = torch.nn.CrossEntropyLoss()
    device = "cuda"

    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    global_model.eval()
    test_loss, correct = 0, 0
    confmat = ConfusionMatrix(num_classes=10).to(device)
    cmtx = 0

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = global_model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            cmtx += confmat(pred, y)

    test_loss /= num_batches
    correct /= size

    acc, cfmtx =  correct, cmtx.cpu().numpy()
    cfmtx = cfmtx/np.sum(cfmtx, axis=1, keepdims=True)
    return cfmtx

def check_representations(model, representations, testing_data, device):
    test_loader = DataLoader(testing_data, batch_size=32, shuffle=True, drop_last=False)
    model.to(device)
    representations = representations.to(device)
    
    label_list = [[1,2],[3,4],[5,6],[7,8],[9,0]]
    confmat = ConfusionMatrix(num_classes=len(label_list)).to(device)
    
    def recoord(labels):
        coord = []
        for label in labels:
            for idx in range(len(label_list)):
                if label in label_list[idx]:
                    coord.append(idx)
        return torch.tensor(coord).cuda()
    
    cmtx = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred, rep = model.pred_and_rep(X)
            
            psi_x = batch_similarity(rep, representations)
            min = torch.min(psi_x, dim=1, keepdim=True).values
            max = torch.max(psi_x, dim=1, keepdim=True).values
            psi_x = (psi_x - min)/(max - min)
            cmtx += confmat(psi_x, recoord(y))
            
        cfmtx = cmtx.cpu().numpy()
        down = np.sum(cfmtx, axis=1, keepdims=True)
        down[down == 0] = 1
        cfmtx = cfmtx/down
    
    return cfmtx
