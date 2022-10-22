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

def create_mask_diagonal(dim, dataset):
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
    mask = torch.zeros([dim])
    for X, y in train_dataloader:
        label = y.item()
        mask[label] = 1
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
    
    # print("classification losses", losses)
    return losses

def mask_training(dataloader, model, optimizer, original_mask_diagonal, device="cuda:1"):
    """
    This method trains to make the model generate a mask that is
    close to the original mask
    """
    original_mask_diagonal = original_mask_diagonal.to(device)
    model = model.to(device)
    model.train()
    
    losses = []
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        mirr_mask_diagonal = model.mask_diagonal(X)
        mask_loss = torch.sum(torch.pow(mirr_mask_diagonal - original_mask_diagonal, 2))/mirr_mask_diagonal.shape[0]

        # Backpropagation
        optimizer.zero_grad()
        mask_loss.backward()
        optimizer.step()
        losses.append(mask_loss.item())
        
    # print("Masking losses", losses)
    return losses

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

def test(model, testing_data, device="cuda:1"):
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

def check_masking_loss(global_model, testing_data, loss_fn, device="cuda:1"):
    test_loader = DataLoader(testing_data, batch_size=32, shuffle=True, drop_last=False)
    global_model.to(device)
    
    label_list = [[1,2],[3,4],[5,6],[7,8],[9,0]]
    
    def masking(labels, dim=10):
        mask = torch.zeros([len(labels), dim, dim])
        for label_idx in range(len(labels)):
            label = labels[label_idx]
            mask_units = None
            for client_label in label_list:
                if label in client_label:
                    mask_units = client_label
                    break
            
            for unit in mask_units:
                mask[label_idx, unit, unit] = 1
                
        return mask
    
    total_loss = 0
    num_batch = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            mirr_mask_diagonal = global_model.mask_diagonal(X)
            mirr_mask = torch.diag_embed(mirr_mask_diagonal)
            ground_mask = masking(y.tolist()).to(device)
            total_loss += loss_fn(mirr_mask, ground_mask).item()
            num_batch += 1
            
    return total_loss/num_batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--round", type=int, default=100)
    parser.add_argument("--warmup_round", type=int, default=20)
    args = parser.parse_args()
    batch_size = args.batch_size
    epochs = args.epochs
    
    training_data = datasets.MNIST(
        root="../data",
        train=True,
        download=False,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
    )
    
    testing_data = datasets.MNIST(
        root="../data",
        train=False,
        download=False,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
    )
    
    client_id_list = [0,1,2,3,4]
    clients_dataset = [CustomDataset(training_data, json.load(open(f"./jsons/client{client_id}.json", 'r'))) for client_id in client_id_list]
    impact_factors = [1/len(dataset) for dataset in clients_dataset]
    
    global_model = DNN_proposal().to(device)
    local_loss_record = {client_id:[] for client_id in client_id_list}
    local_mask_record = {client_id:[] for client_id in client_id_list}
    local_mask_loss_record = {client_id:[] for client_id in client_id_list}
    local_cfmtx_bfag_record = {client_id:[] for client_id in client_id_list}
    local_cfmtx_afag_record = {client_id:[] for client_id in client_id_list}
    global_cfmtx_record = []
    global_masking_loss = []
    clients_mask_diagonal = [None for client_id in client_id_list]
    
    warmup = True
    
    for cur_round in range(args.round + 1):
        print("============ Round {} ==============".format(cur_round))
        client_models = []
        
        if cur_round > args.warmup_round:
            warmup = False 
        
        # Local training
        for client_id in client_id_list:
            print("    Client {} training... ".format(client_id), end="")
            # Training process
            mydataset = clients_dataset[client_id]
            
            if clients_mask_diagonal[client_id] is None:
                clients_mask_diagonal[client_id] = create_mask_diagonal(dim=10, dataset=mydataset)
                
            local_model = copy.deepcopy(global_model)
            optimizer = torch.optim.SGD(local_model.parameters(), lr=1e-3)
            mask_optimizer = torch.optim.Adam(local_model.parameters(), lr=1e-2)
            
            if warmup:
                # Train the representation space
                print("warming up...", end="")
                train_dataloader = DataLoader(mydataset, batch_size=2, shuffle=True, drop_last=True)
                
                loss_fn = torch.nn.MSELoss()
                for t in range(epochs):
                    representation_training(train_dataloader, local_model, loss_fn, optimizer, device)
                print(f"Done!")
                
            else:
                train_dataloader = DataLoader(mydataset, batch_size=batch_size, shuffle=True, drop_last=False)
                
                # Train the mask first
                epoch_loss = []
                for t in range(epochs):
                    mask_loss = mask_training(train_dataloader, local_model, mask_optimizer, clients_mask_diagonal[client_id], device)
                    # local_mask_record[client_id].append(final_mask)
                    epoch_loss.append(mask_loss)
                local_mask_loss_record[client_id].append(np.mean(epoch_loss))
                print(f"\n\tMasking done! Aver. round loss: {np.mean(epoch_loss):>.3f}")
                
                # Then train the classifier
                loss_fn = torch.nn.CrossEntropyLoss()
                epoch_loss = []
                for t in range(epochs):
                    epoch_loss.append(np.mean(classification_training(train_dataloader, local_model, loss_fn, optimizer, device)))
                local_loss_record[client_id].append(np.mean(epoch_loss))

                # Testing the local_model to its own data
                cfmtx = test(local_model, mydataset)
                local_cfmtx_bfag_record[client_id].append(cfmtx)
                print(f"\tClassification done! Aver. round loss: {np.mean(epoch_loss):>.3f}")
            
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
        if not warmup:
            print_cfmtx(cfmtx)

        masking_loss = check_masking_loss(global_model, testing_data, torch.nn.MSELoss(), device)
        global_masking_loss.append(masking_loss)
        
    
    if not Path("records/proposal4").exists():
        os.makedirs("records/proposal4")
    
    json.dump(local_loss_record,        open("records/proposal4/local_loss_record.json", "w"),         cls=NumpyEncoder)
    json.dump(local_mask_loss_record,   open("records/proposal4/local_mask_loss_record.json", "w"),    cls=NumpyEncoder)
    # json.dump(local_mask_record,        open("records/proposal4/local_mask_record.json", "w"),         cls=NumpyEncoder)
    json.dump(local_cfmtx_bfag_record,  open("records/proposal4/local_cfmtx_bfag_record.json", "w"),   cls=NumpyEncoder)
    json.dump(global_cfmtx_record,      open("records/proposal4/global_cfmtx_record.json", "w"),       cls=NumpyEncoder)
    json.dump(global_masking_loss,      open("records/proposal4/global_masking_loss.json", "w"),       cls=NumpyEncoder)
    