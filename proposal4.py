import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils import fmodule
from utils.dataloader import CustomDataset
from utils.proposal_model import DNN_proposal
from utils.train_smt import NumpyEncoder, print_cfmtx
from torchmetrics import ConfusionMatrix

device = "cuda" if torch.cuda.is_available() else "cpu"

def create_mask(dim, dataset):
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
    mask = torch.zeros([dim, dim])
    for X, y in train_dataloader:
        label = y.item()
        mask[label, label] = 1
    return mask

def classification_training(dataloader, model, loss_fn, optimizer, device="cuda:1"):
    model = model.to(device)
    model.train()
    losses = []
        
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        classification_loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        classification_loss.backward()
        optimizer.step()
        losses.append(classification_loss.item())
        
    return losses

def mask_training(dataloader, model, optimizer, original_mask, device="cuda:1"):
    """
    This method trains to make the model generate a mask that is
    close to the original mask
    """
    model = model.to(device)
    model.train()
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        mirr_mask = model.mask(X)
        mask_loss = torch.sum(torch.pow(mirr_mask - original_mask, 2))/(mirr_mask.shape[0] * mirr_mask.shape[1] * mirr_mask.shape[2])

        # Backpropagation
        optimizer.zero_grad()
        mask_loss.backward()
        optimizer.step()
        
    return

def representation_training(dataloader, model, loss_fn, optimizer, device="cuda:1"):
    """
    This method trains for a discriminative representation space,
    using constrastive learning
    
    Args:
        dataloader: batch_size of 2, drop_last = True
        loss_fn:    mean square error
    """
    model = model.to(device)
    model.train()
        
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        representations = model.encoder(X)
        
        alpha = 1.0 if y[0].item() == y[1].item() else -1.0
        distance = loss_fn(representations[0], representations[1])
        loss = alpha * distance
                
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return

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
            pred, m_x = model(X)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--warmup_round", type=int, default=1)
    args = parser.parse_args()
    batch_size = args.batch_size
    epochs = args.epochs
    
    training_data = datasets.MNIST(
        root="./data",
        train=True,
        download=False,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
    )
    
    testing_data = datasets.MNIST(
        root="data",
        train=False,
        download=False,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
    )
    
    client_id_list = [0,1,2,3,4]
    clients_dataset = [CustomDataset(training_data, json.load(open(f"./jsons/client{client_id}.json", 'r'))) for client_id in client_id_list]
    impact_factors = [1/len(dataset) for dataset in clients_dataset]
    
    global_model = DNN_proposal().to(device)
    local_loss_record = {client_id:[] for client_id in client_id_list}
    local_cfmtx_bfag_record = {client_id:[] for client_id in client_id_list}
    local_cfmtx_afag_record = {client_id:[] for client_id in client_id_list}
    global_cfmtx_record = []
    U_cfmtx_record = []
    clients_mask = [None for client_id in client_id_list]
    
    warmup = True
    
    for cur_round in range(args.round):
        print("============ Round {} ==============".format(cur_round))
        client_models = []
        
        if cur_round > args.warmup_round:
            warmup = False 
        
        # Local training
        for client_id in client_id_list:
            print("    Client {} training... ".format(client_id), end="")
            # Training process
            mydataset = clients_dataset[client_id]
            
            if clients_mask[client_id] is None:
                clients_mask[client_id] = create_mask(dim=10, dataset=mydataset)
                
            local_model = copy.deepcopy(global_model)
            optimizer = torch.optim.SGD(local_model.parameters(), lr=1e-3)
            
            if warmup:
                # Train the representation space
                train_dataloader = DataLoader(mydataset, batch_size=2, shuffle=True, drop_last=True)
                loss_fn = torch.nn.MSELoss()
                for t in range(epochs):
                    representation_training(train_dataloader, local_model, loss_fn, optimizer, device)
                    
            else:
                train_dataloader = DataLoader(mydataset, batch_size=batch_size, shuffle=True, drop_last=False)
                # Train the mask first then train the classifier
                for t in range(epochs):
                    mask_training(train_dataloader, local_model, optimizer, clients_mask[client_id], device)
                    
                epoch_loss = []
                for t in range(epochs):
                    epoch_loss.append(np.mean(classification_training(train_dataloader, local_model, loss_fn, optimizer, device)))
                local_loss_record[client_id].append(np.mean(epoch_loss))

                # Testing the local_model to its own data
                cfmtx = test(local_model, mydataset)
                local_cfmtx_bfag_record[client_id].append(cfmtx)
                print(f"Done! Aver. round loss: {np.mean(epoch_loss):>.3f}")
            
            client_models.append(local_model)
            
        print("    # Server aggregating... ", end="")
        # Aggregation
        global_model = fmodule._model_sum([model * pk for model, pk in zip(client_models, impact_factors)])
        print("Done!")
        
        print("    # Server testing... ", end="")
        cfmtx = test(global_model, testing_data, device)
        global_cfmtx_record.append(cfmtx)
        print("Done!")
        np.set_printoptions(precision=2, suppress=True)
        print_cfmtx(cfmtx)
        
        if not Path("records/proposal4").exists():
            os.makedirs("records/proposal4")
        
        json.dump(local_loss_record,        open("records/proposal4/local_loss_record.json", "w"),         cls=NumpyEncoder)
        json.dump(local_cfmtx_bfag_record,  open("records/proposal4/local_cfmtx_bfag_record.json", "w"),   cls=NumpyEncoder)
        json.dump(global_cfmtx_record,      open("records/proposal4/global_cfmtx_record.json", "w"),       cls=NumpyEncoder)